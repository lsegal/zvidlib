//! Runtime-dispatched SIMD kernels for the HEVC §8.6.3 dequantization
//! scaling loop and the §8.6.4 inverse-transform butterfly passes.
//!
//! Both kernels are pure fixed-point integer reductions that the decoder
//! runs on the order of millions of times per frame, so they are the
//! natural place to spend explicit vectorization effort:
//!
//! * [`transform_1d`] is one pass of the separable 1-D inverse DCT/DST.
//!   Driving the butterfly as one broadcast-multiply-add per *non-zero*
//!   input — `out[ .. ] += x[ j ] * basisRow( j )` — turns the
//!   `nTbS`-term dot product into `nTbS` independent lanes and lets the
//!   (typically very sparse) coefficient vector skip most of the work
//!   outright. The whole pass lives inside one `#[target_feature]`
//!   function so the per-row work is never charged a non-inlinable call.
//! * [`dequant_block`] is §8.6.3 equation 8-309 applied to every
//!   position of a transform block.
//!
//! Every backend is bit-exact with [`Backend::Scalar`]: the vector code
//! performs the same operations, in the same widths, in the same order.
//! Selection happens at run time via [`detected`], which probes the host
//! with `is_x86_feature_detected!` on x86_64 and relies on NEON being
//! architecturally guaranteed on aarch64. Every other target — and every
//! input outside a backend's exactness preconditions — falls back to the
//! scalar path.

use core::sync::atomic::{AtomicU8, Ordering};

/// The kernel implementation selected for a call.
///
/// Callers normally take whatever [`detected`] reports; the variants are
/// nameable so tests and benchmarks can exercise each backend the host
/// supports without touching global state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Portable scalar reference. Always available, and the definition
    /// of "bit-exact" every other backend is checked against.
    Scalar,
    /// x86_64 SSE4.1 (`pmulld` / `pcmpgtq` come from SSE4.1 and SSE4.2
    /// respectively, so the dequantization kernel additionally requires
    /// SSE4.2 — see [`Backend::supported`]).
    Sse41,
    /// x86_64 SSE4.2, i.e. SSE4.1 plus the 64-bit `pcmpgtq` compare the
    /// dequantization clip needs.
    Sse42,
    /// x86_64 AVX2.
    Avx2,
    /// aarch64 Advanced SIMD (NEON).
    Neon,
}

impl Backend {
    /// Whether the running host can execute this backend at all.
    #[must_use]
    pub fn supported(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Self::Sse41 => is_x86_feature_detected!("sse4.1"),
            #[cfg(target_arch = "x86_64")]
            Self::Sse42 => is_x86_feature_detected!("sse4.2"),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "aarch64")]
            Self::Neon => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

/// Cache for [`detected`]: `0` = not probed yet, otherwise `1 +` the
/// backend's index in [`PRIORITY`].
static DETECTED: AtomicU8 = AtomicU8::new(0);

/// Candidate backends in descending preference order. The first entry
/// whose [`Backend::supported`] holds is what [`detected`] returns, so
/// SSE4.2 outranks SSE4.1 (it is a strict superset, and only it can run
/// the vector dequantization clip).
const PRIORITY: [Backend; 4] = [Backend::Avx2, Backend::Neon, Backend::Sse42, Backend::Sse41];

/// The best kernel backend this host supports.
///
/// The CPU feature probe runs once and is cached; the result never
/// changes for the lifetime of the process.
#[must_use]
pub fn detected() -> Backend {
    let cached = DETECTED.load(Ordering::Relaxed);
    if cached != 0 {
        return PRIORITY
            .get(cached as usize - 1)
            .copied()
            .unwrap_or(Backend::Scalar);
    }
    let mut chosen = Backend::Scalar;
    let mut index = PRIORITY.len();
    for (i, candidate) in PRIORITY.iter().enumerate() {
        if candidate.supported() {
            chosen = *candidate;
            index = i;
            break;
        }
    }
    DETECTED.store(index as u8 + 1, Ordering::Relaxed);
    chosen
}

/// Every backend the host can actually run, scalar first. Used by the
/// bit-exactness tests and the benchmark so both cover each code path
/// the machine is capable of executing rather than only the preferred
/// one.
#[must_use]
pub fn supported_backends() -> Vec<Backend> {
    let mut backends = vec![Backend::Scalar];
    backends.extend(PRIORITY.iter().copied().filter(|b| b.supported()));
    backends
}

/// One pass of the §8.6.4.2 separable inverse transform:
/// `out[ i ] = Σ_j basis[ j * row_step ][ i ] * input[ j ]`.
///
/// `basis` is a row-major `transMatrix` table of `basis_stride`-wide
/// rows ([`crate::hevc::engine::transform`] passes the flattened `DCT32`
/// or `DST4`), and `row_step` is equation 8-317's `1 << ( 5 − log2( nTbS
/// ) )` basis-row stride (always 1 for the DST). The sum is evaluated as
/// one broadcast-multiply-add per non-zero `input[ j ]` across a tile of
/// output lanes, which is the same set of integer products added in a
/// different order. The vector kernels hold each tile's accumulators in
/// registers for the whole `j` sweep, so a basis row costs one load and
/// one multiply-add instead of a read-modify-write of `out`.
///
/// Callers must guarantee that no partial sum leaves the `i32` range;
/// see [`crate::hevc::engine::transform::inverse_transform`], which
/// checks the worst-case bound for `nTbS` and the block's
/// `coeffMin`/`coeffMax` before choosing this path.
///
/// # Panics
/// Panics if `input` and `out` differ in length, or if `basis` is too
/// short for the addressed rows.
pub fn transform_1d(
    backend: Backend,
    input: &[i32],
    out: &mut [i32],
    basis: &[i32],
    basis_stride: usize,
    row_step: usize,
) {
    assert_eq!(input.len(), out.len(), "butterfly operand length mismatch");
    assert!(
        out.len() <= basis_stride,
        "butterfly output wider than the basis"
    );
    assert!(
        basis.len() >= (input.len() - 1) * row_step * basis_stride + out.len(),
        "butterfly basis table too short"
    );
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => {
            // SAFETY: `Backend::Avx2` is only produced by `detected` /
            // `supported_backends` after `is_x86_feature_detected!` has
            // confirmed AVX2 on this host. The length preconditions are
            // asserted above.
            unsafe { x86::transform_1d_avx2(input, out, basis, basis_stride, row_step) }
        }
        #[cfg(target_arch = "x86_64")]
        Backend::Sse41 | Backend::Sse42 => {
            // SAFETY: as above, for SSE4.1 (SSE4.2 implies SSE4.1).
            unsafe { x86::transform_1d_sse41(input, out, basis, basis_stride, row_step) }
        }
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {
            // SAFETY: NEON is architecturally guaranteed on aarch64.
            unsafe { aarch64::transform_1d_neon(input, out, basis, basis_stride, row_step) }
        }
        _ => transform_1d_scalar(input, out, basis, basis_stride, row_step),
    }
}

/// Portable reference for [`transform_1d`].
fn transform_1d_scalar(
    input: &[i32],
    out: &mut [i32],
    basis: &[i32],
    basis_stride: usize,
    row_step: usize,
) {
    out.fill(0);
    for (j, &xj) in input.iter().enumerate() {
        if xj == 0 {
            continue;
        }
        let row = &basis[j * row_step * basis_stride..][..out.len()];
        for (o, &c) in out.iter_mut().zip(row.iter()) {
            *o += xj * c;
        }
    }
}

/// Fixed inputs of §8.6.3 equation 8-309 that are constant across a
/// transform block, gathered so the kernels take one argument instead of
/// six.
#[derive(Debug, Clone, Copy)]
pub struct DequantParams {
    /// `levelScale[ qP % 6 ]`, the per-`qP` multiplier.
    pub level_scale: i32,
    /// `qP / 6`, the left shift applied to the product.
    pub qp_div6: u32,
    /// `bdShift`, the right shift applied after rounding. Always in
    /// `1..=62` for the 8..=16-bit depths the decoder accepts.
    pub bd_shift: u32,
    /// `coeffMin` from §7.4.5 equations 7-27..7-30.
    pub coeff_min: i32,
    /// `coeffMax` from §7.4.5 equations 7-27..7-30.
    pub coeff_max: i32,
}

/// §8.6.3 equation 8-309 over a whole transform block:
/// `d[ i ] = Clip3( coeffMin, coeffMax, ( ( levels[ i ] * m[ i ] *
/// levelScale << ( qP / 6 ) ) + ( 1 << ( bdShift − 1 ) ) ) >> bdShift )`.
///
/// `m` supplies the per-position scaling factor `m[ x ][ y ]` in the same
/// row-major order as `levels`, or is `None` for the flat-16 default.
/// The intermediate product is formed in 64 bits on every backend, so no
/// input needs a magnitude precondition.
///
/// # Panics
/// Panics if `out`, `levels`, or a supplied `m` disagree in length.
pub fn dequant_block(
    backend: Backend,
    out: &mut [i32],
    levels: &[i32],
    m: Option<&[u16]>,
    params: DequantParams,
) {
    assert_eq!(out.len(), levels.len(), "dequant operand length mismatch");
    if let Some(m) = m {
        assert_eq!(
            m.len(),
            levels.len(),
            "dequant scaling-matrix length mismatch"
        );
    }
    // The vector paths emulate a 64-bit arithmetic right shift, which is
    // only defined for a shift inside the word; `bdShift == 0` would also
    // make the equation's rounding offset ill-formed. Neither happens for
    // the dimensioned 8..=16-bit range, but stay exact if it ever does.
    if !(1..=62).contains(&params.bd_shift) {
        dequant_block_scalar(out, levels, m, params);
        return;
    }
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => {
            // SAFETY: `Backend::Avx2` is only produced after
            // `is_x86_feature_detected!("avx2")` succeeded on this host.
            unsafe { x86::dequant_block_avx2(out, levels, m, params) }
        }
        #[cfg(target_arch = "x86_64")]
        Backend::Sse42 => {
            // SAFETY: as above, for SSE4.2.
            unsafe { x86::dequant_block_sse42(out, levels, m, params) }
        }
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {
            // SAFETY: NEON is architecturally guaranteed on aarch64.
            unsafe { aarch64::dequant_block_neon(out, levels, m, params) }
        }
        // SSE4.1 without SSE4.2 lacks the 64-bit `pcmpgtq` the clip
        // needs, so it keeps the scalar dequantization while still using
        // the vector butterfly.
        _ => dequant_block_scalar(out, levels, m, params),
    }
}

/// Portable reference for [`dequant_block`].
fn dequant_block_scalar(out: &mut [i32], levels: &[i32], m: Option<&[u16]>, params: DequantParams) {
    let round = 1i64 << (params.bd_shift - 1);
    for (i, (o, &level)) in out.iter_mut().zip(levels.iter()).enumerate() {
        let factor = m.map_or(16i32, |m| i32::from(m[i])) * params.level_scale;
        let prod = i64::from(level) * i64::from(factor);
        let shifted = (prod << params.qp_div6) + round;
        *o = (shifted >> params.bd_shift)
            .clamp(i64::from(params.coeff_min), i64::from(params.coeff_max)) as i32;
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::DequantParams;
    use core::arch::x86_64::*;

    /// AVX2 [`super::transform_1d`]: sixteen `i32` output lanes per
    /// tile, `vpmulld` + `vpaddd` into register accumulators, narrowing
    /// to 8- and 4-wide tiles and finally a scalar one for widths the
    /// widest tile cannot cover.
    ///
    /// # Safety
    /// The host must support AVX2. `basis` must cover every addressed row.
    #[target_feature(enable = "avx2")]
    pub unsafe fn transform_1d_avx2(
        input: &[i32],
        out: &mut [i32],
        basis: &[i32],
        basis_stride: usize,
        row_step: usize,
    ) {
        unsafe {
            let n = out.len();
            let row_stride = row_step * basis_stride;
            let mut base = 0;
            while base + 16 <= n {
                let mut acc0 = _mm256_setzero_si256();
                let mut acc1 = _mm256_setzero_si256();
                for (j, &xj) in input.iter().enumerate() {
                    if xj == 0 {
                        continue;
                    }
                    let row = basis.as_ptr().add(j * row_stride + base);
                    let s = _mm256_set1_epi32(xj);
                    acc0 = _mm256_add_epi32(
                        acc0,
                        _mm256_mullo_epi32(_mm256_loadu_si256(row.cast()), s),
                    );
                    acc1 = _mm256_add_epi32(
                        acc1,
                        _mm256_mullo_epi32(_mm256_loadu_si256(row.add(8).cast()), s),
                    );
                }
                _mm256_storeu_si256(out.as_mut_ptr().add(base).cast(), acc0);
                _mm256_storeu_si256(out.as_mut_ptr().add(base + 8).cast(), acc1);
                base += 16;
            }
            while base + 8 <= n {
                let mut acc = _mm256_setzero_si256();
                for (j, &xj) in input.iter().enumerate() {
                    if xj != 0 {
                        let row = basis.as_ptr().add(j * row_stride + base);
                        acc = _mm256_add_epi32(
                            acc,
                            _mm256_mullo_epi32(
                                _mm256_loadu_si256(row.cast()),
                                _mm256_set1_epi32(xj),
                            ),
                        );
                    }
                }
                _mm256_storeu_si256(out.as_mut_ptr().add(base).cast(), acc);
                base += 8;
            }
            while base + 4 <= n {
                let mut acc = _mm_setzero_si128();
                for (j, &xj) in input.iter().enumerate() {
                    if xj != 0 {
                        let row = basis.as_ptr().add(j * row_stride + base);
                        acc = _mm_add_epi32(
                            acc,
                            _mm_mullo_epi32(_mm_loadu_si128(row.cast()), _mm_set1_epi32(xj)),
                        );
                    }
                }
                _mm_storeu_si128(out.as_mut_ptr().add(base).cast(), acc);
                base += 4;
            }
            while base < n {
                let mut acc = 0i32;
                for (j, &xj) in input.iter().enumerate() {
                    acc += xj * *basis.as_ptr().add(j * row_stride + base);
                }
                out[base] = acc;
                base += 1;
            }
        }
    }

    /// SSE4.1 [`super::transform_1d`]: eight `i32` output lanes per
    /// tile, held in two register accumulators.
    ///
    /// # Safety
    /// The host must support SSE4.1. `basis` must cover every addressed row.
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn transform_1d_sse41(
        input: &[i32],
        out: &mut [i32],
        basis: &[i32],
        basis_stride: usize,
        row_step: usize,
    ) {
        unsafe {
            let n = out.len();
            let row_stride = row_step * basis_stride;
            let mut base = 0;
            while base + 8 <= n {
                let mut acc0 = _mm_setzero_si128();
                let mut acc1 = _mm_setzero_si128();
                for (j, &xj) in input.iter().enumerate() {
                    if xj == 0 {
                        continue;
                    }
                    let row = basis.as_ptr().add(j * row_stride + base);
                    let s = _mm_set1_epi32(xj);
                    acc0 = _mm_add_epi32(acc0, _mm_mullo_epi32(_mm_loadu_si128(row.cast()), s));
                    acc1 =
                        _mm_add_epi32(acc1, _mm_mullo_epi32(_mm_loadu_si128(row.add(4).cast()), s));
                }
                _mm_storeu_si128(out.as_mut_ptr().add(base).cast(), acc0);
                _mm_storeu_si128(out.as_mut_ptr().add(base + 4).cast(), acc1);
                base += 8;
            }
            while base + 4 <= n {
                let mut acc = _mm_setzero_si128();
                for (j, &xj) in input.iter().enumerate() {
                    if xj != 0 {
                        let row = basis.as_ptr().add(j * row_stride + base);
                        acc = _mm_add_epi32(
                            acc,
                            _mm_mullo_epi32(_mm_loadu_si128(row.cast()), _mm_set1_epi32(xj)),
                        );
                    }
                }
                _mm_storeu_si128(out.as_mut_ptr().add(base).cast(), acc);
                base += 4;
            }
            while base < n {
                let mut acc = 0i32;
                for (j, &xj) in input.iter().enumerate() {
                    acc += xj * *basis.as_ptr().add(j * row_stride + base);
                }
                out[base] = acc;
                base += 1;
            }
        }
    }

    /// AVX2 [`super::dequant_block`]: four 64-bit lanes per step.
    ///
    /// `vpmuldq` gives the exact 32x32 -> 64 product equation 8-309 needs;
    /// the 64-bit arithmetic right shift AVX2 lacks is synthesized by
    /// biasing into the unsigned domain (`x ^ 2^63`), shifting logically,
    /// and subtracting the shifted bias, which is exact for every `i64`.
    ///
    /// # Safety
    /// The host must support AVX2. `params.bd_shift` must be in `1..=62`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dequant_block_avx2(
        out: &mut [i32],
        levels: &[i32],
        m: Option<&[u16]>,
        params: DequantParams,
    ) {
        unsafe {
            let shl = _mm_cvtsi32_si128(params.qp_div6 as i32);
            let shr = _mm_cvtsi32_si128(params.bd_shift as i32);
            let round = _mm256_set1_epi64x(1i64 << (params.bd_shift - 1));
            let sign = _mm256_set1_epi64x(i64::MIN);
            let bias = _mm256_set1_epi64x(((1u64 << 63) >> params.bd_shift) as i64);
            let lo = _mm256_set1_epi64x(i64::from(params.coeff_min));
            let hi = _mm256_set1_epi64x(i64::from(params.coeff_max));
            let scale = _mm_set1_epi32(params.level_scale);
            let flat = _mm_set1_epi32(16);
            let gather = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);

            let mut i = 0;
            while i + 4 <= out.len() {
                let level = _mm256_cvtepi32_epi64(_mm_loadu_si128(levels.as_ptr().add(i).cast()));
                // m[ x ][ y ] * levelScale fits in i32 (m <= 255,
                // levelScale <= 72), so the widened product below is the
                // whole of equation 8-309's multiplication.
                let factor32 = match m {
                    // `pmovzxwd` widens the four u16 scaling factors to i32.
                    Some(m) => _mm_mullo_epi32(
                        _mm_cvtepu16_epi32(_mm_loadl_epi64(m.as_ptr().add(i).cast())),
                        scale,
                    ),
                    None => _mm_mullo_epi32(flat, scale),
                };
                let factor = _mm256_cvtepi32_epi64(factor32);
                let prod = _mm256_mul_epi32(level, factor);
                let shifted = _mm256_add_epi64(_mm256_sll_epi64(prod, shl), round);
                // Arithmetic >> bd_shift, emulated.
                let biased = _mm256_xor_si256(shifted, sign);
                let scaled = _mm256_sub_epi64(_mm256_srl_epi64(biased, shr), bias);
                let clipped = _mm256_blendv_epi8(
                    _mm256_blendv_epi8(scaled, lo, _mm256_cmpgt_epi64(lo, scaled)),
                    hi,
                    _mm256_cmpgt_epi64(scaled, hi),
                );
                // The clipped values fit in i32; keep each 64-bit lane's
                // low half and store the four results contiguously.
                let packed = _mm256_permutevar8x32_epi32(clipped, gather);
                _mm_storeu_si128(
                    out.as_mut_ptr().add(i).cast(),
                    _mm256_castsi256_si128(packed),
                );
                i += 4;
            }
            if i < out.len() {
                let tail_m = m.map(|m| &m[i..]);
                super::dequant_block_scalar(&mut out[i..], &levels[i..], tail_m, params);
            }
        }
    }

    /// SSE4.2 [`super::dequant_block`]: two 64-bit lanes per step, with
    /// the same emulated arithmetic shift as the AVX2 kernel. SSE4.2 (not
    /// merely SSE4.1) is required for `pcmpgtq`.
    ///
    /// # Safety
    /// The host must support SSE4.2. `params.bd_shift` must be in `1..=62`.
    #[target_feature(enable = "sse4.2")]
    pub unsafe fn dequant_block_sse42(
        out: &mut [i32],
        levels: &[i32],
        m: Option<&[u16]>,
        params: DequantParams,
    ) {
        unsafe {
            let shl = _mm_cvtsi32_si128(params.qp_div6 as i32);
            let shr = _mm_cvtsi32_si128(params.bd_shift as i32);
            let round = _mm_set1_epi64x(1i64 << (params.bd_shift - 1));
            let sign = _mm_set1_epi64x(i64::MIN);
            let bias = _mm_set1_epi64x(((1u64 << 63) >> params.bd_shift) as i64);
            let lo = _mm_set1_epi64x(i64::from(params.coeff_min));
            let hi = _mm_set1_epi64x(i64::from(params.coeff_max));
            let scale = _mm_set1_epi32(params.level_scale);
            let flat = _mm_set1_epi32(16);

            let mut i = 0;
            while i + 2 <= out.len() {
                let level = _mm_cvtepi32_epi64(_mm_loadl_epi64(levels.as_ptr().add(i).cast()));
                let factor32 = match m {
                    Some(m) => {
                        // `pmovzxwd` widens the two u16 scaling factors
                        // packed into one i32 up to a lane each.
                        let pair = i32::from(m[i]) | (i32::from(m[i + 1]) << 16);
                        _mm_mullo_epi32(_mm_cvtepu16_epi32(_mm_cvtsi32_si128(pair)), scale)
                    }
                    None => _mm_mullo_epi32(flat, scale),
                };
                let factor = _mm_cvtepi32_epi64(factor32);
                let prod = _mm_mul_epi32(level, factor);
                let shifted = _mm_add_epi64(_mm_sll_epi64(prod, shl), round);
                let biased = _mm_xor_si128(shifted, sign);
                let scaled = _mm_sub_epi64(_mm_srl_epi64(biased, shr), bias);
                let clipped = _mm_blendv_epi8(
                    _mm_blendv_epi8(scaled, lo, _mm_cmpgt_epi64(lo, scaled)),
                    hi,
                    _mm_cmpgt_epi64(scaled, hi),
                );
                // Keep the low half of each 64-bit lane: lanes 0 and 2 of
                // the 32-bit view.
                let packed = _mm_shuffle_epi32::<0b0000_1000>(clipped);
                _mm_storel_epi64(out.as_mut_ptr().add(i).cast(), packed);
                i += 2;
            }
            if i < out.len() {
                let tail_m = m.map(|m| &m[i..]);
                super::dequant_block_scalar(&mut out[i..], &levels[i..], tail_m, params);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::DequantParams;
    use core::arch::aarch64::*;

    /// NEON [`super::transform_1d`]: eight `i32` output lanes per tile,
    /// held in two `vmlaq_s32` register accumulators.
    ///
    /// # Safety
    /// The host must support NEON (guaranteed on aarch64). `basis` must
    /// cover every addressed row.
    #[target_feature(enable = "neon")]
    pub unsafe fn transform_1d_neon(
        input: &[i32],
        out: &mut [i32],
        basis: &[i32],
        basis_stride: usize,
        row_step: usize,
    ) {
        unsafe {
            let n = out.len();
            let row_stride = row_step * basis_stride;
            let mut base = 0;
            while base + 8 <= n {
                let mut acc0 = vdupq_n_s32(0);
                let mut acc1 = vdupq_n_s32(0);
                for (j, &xj) in input.iter().enumerate() {
                    if xj == 0 {
                        continue;
                    }
                    let row = basis.as_ptr().add(j * row_stride + base);
                    let s = vdupq_n_s32(xj);
                    acc0 = vmlaq_s32(acc0, vld1q_s32(row), s);
                    acc1 = vmlaq_s32(acc1, vld1q_s32(row.add(4)), s);
                }
                vst1q_s32(out.as_mut_ptr().add(base), acc0);
                vst1q_s32(out.as_mut_ptr().add(base + 4), acc1);
                base += 8;
            }
            while base + 4 <= n {
                let mut acc = vdupq_n_s32(0);
                for (j, &xj) in input.iter().enumerate() {
                    if xj != 0 {
                        acc = vmlaq_s32(
                            acc,
                            vld1q_s32(basis.as_ptr().add(j * row_stride + base)),
                            vdupq_n_s32(xj),
                        );
                    }
                }
                vst1q_s32(out.as_mut_ptr().add(base), acc);
                base += 4;
            }
            while base < n {
                let mut acc = 0i32;
                for (j, &xj) in input.iter().enumerate() {
                    acc += xj * *basis.as_ptr().add(j * row_stride + base);
                }
                out[base] = acc;
                base += 1;
            }
        }
    }

    /// NEON [`super::dequant_block`]: four coefficients per step, carried
    /// through two 64-bit lane pairs.
    ///
    /// `vmull_s32` is the exact 32x32 -> 64 widening multiply equation
    /// 8-309 needs, and `vshlq_s64` with a negative count is an
    /// arithmetic (sign-preserving, toward negative infinity) right
    /// shift, matching Rust's `i64 >>`.
    ///
    /// # Safety
    /// The host must support NEON. `params.bd_shift` must be in `1..=62`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dequant_block_neon(
        out: &mut [i32],
        levels: &[i32],
        m: Option<&[u16]>,
        params: DequantParams,
    ) {
        unsafe {
            let shl = vdupq_n_s64(i64::from(params.qp_div6));
            let shr = vdupq_n_s64(-i64::from(params.bd_shift));
            let round = vdupq_n_s64(1i64 << (params.bd_shift - 1));
            let lo = vdupq_n_s64(i64::from(params.coeff_min));
            let hi = vdupq_n_s64(i64::from(params.coeff_max));
            let scale = vdupq_n_s32(params.level_scale);
            let flat = vdupq_n_s32(16);

            let clip = |v: int64x2_t| {
                let v = vbslq_s64(vcgtq_s64(lo, v), lo, v);
                vbslq_s64(vcgtq_s64(v, hi), hi, v)
            };
            let apply = |level: int32x2_t, factor: int32x2_t| {
                let prod = vmull_s32(level, factor);
                let shifted = vaddq_s64(vshlq_s64(prod, shl), round);
                vmovn_s64(clip(vshlq_s64(shifted, shr)))
            };

            let mut i = 0;
            while i + 4 <= out.len() {
                let level = vld1q_s32(levels.as_ptr().add(i));
                // m[ x ][ y ] * levelScale fits in i32 (m <= 255,
                // levelScale <= 72).
                let factor = match m {
                    Some(m) => vmulq_s32(
                        vreinterpretq_s32_u32(vmovl_u16(vld1_u16(m.as_ptr().add(i)))),
                        scale,
                    ),
                    None => vmulq_s32(flat, scale),
                };
                let low = apply(vget_low_s32(level), vget_low_s32(factor));
                let high = apply(vget_high_s32(level), vget_high_s32(factor));
                vst1q_s32(out.as_mut_ptr().add(i), vcombine_s32(low, high));
                i += 4;
            }
            if i < out.len() {
                let tail_m = m.map(|m| &m[i..]);
                super::dequant_block_scalar(&mut out[i..], &levels[i..], tail_m, params);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::engine::scaling_list::ScalingFactorMatrix;
    use crate::hevc::engine::transform::{
        Component, PredMode, coeff_range, inverse_transform_reference,
        inverse_transform_with_backend, scale_coefficients,
    };

    /// Small deterministic LCG so tests are reproducible without `rand`.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        /// A signed value spanning `[ -bound, bound ]`.
        fn signed(&mut self, bound: i32) -> i32 {
            (self.next_u32() % (2 * bound as u32 + 1)) as i32 - bound
        }
    }

    /// The host always offers the scalar reference, and never reports a
    /// backend for another architecture.
    #[test]
    fn detected_backend_is_supported() {
        assert!(detected().supported());
        let backends = supported_backends();
        assert_eq!(backends[0], Backend::Scalar);
        assert!(backends.contains(&detected()));
        assert!(backends.iter().all(|b| b.supported()));
    }

    /// Every vector butterfly kernel reproduces the scalar pass
    /// exactly, including the sub-vector widths and basis-row strides a
    /// caller could pass (block sides are always 4/8/16/32, but the
    /// kernel must not silently drop a remainder).
    #[test]
    fn transform_1d_matches_scalar_for_every_backend() {
        let mut rng = Lcg(0x51ed_2701);
        for (n, basis_stride, row_step) in [
            (1usize, 32usize, 1usize),
            (3, 32, 1),
            (4, 4, 1),
            (4, 32, 8),
            (5, 32, 1),
            (7, 32, 1),
            (8, 32, 4),
            (9, 32, 1),
            (15, 32, 1),
            (16, 32, 2),
            (31, 32, 1),
            (32, 32, 1),
        ] {
            let basis: Vec<i32> = (0..(n * row_step * basis_stride + basis_stride))
                .map(|_| rng.signed(90))
                .collect();
            for round in 0..64 {
                // Alternate dense and sparse inputs so the kernels' zero
                // skip is covered alongside the full accumulation.
                let input: Vec<i32> = (0..n)
                    .map(|_| {
                        if round % 2 == 0 && rng.next_u32() % 4 != 0 {
                            0
                        } else {
                            rng.signed(1 << 15)
                        }
                    })
                    .collect();

                let mut expected = vec![0i32; n];
                transform_1d(
                    Backend::Scalar,
                    &input,
                    &mut expected,
                    &basis,
                    basis_stride,
                    row_step,
                );
                for backend in supported_backends() {
                    let mut got = vec![i32::MIN; n];
                    transform_1d(backend, &input, &mut got, &basis, basis_stride, row_step);
                    assert_eq!(got, expected, "backend {backend:?}, n {n}");
                }
            }
        }
    }

    /// Every vector dequantization kernel reproduces §8.6.3 exactly for
    /// every block size, both scaling-matrix modes, the whole HEVC `qP`
    /// range, and both coefficient ranges.
    #[test]
    fn dequant_block_matches_scalar_for_every_backend() {
        let mut rng = Lcg(0x0bad_c0de);
        for n_tbs in [4usize, 8, 16, 32] {
            let count = n_tbs * n_tbs;
            for extended in [false, true] {
                for bit_depth in [8u8, 10, 12, 16] {
                    let (coeff_min, coeff_max) = coeff_range(bit_depth, extended);
                    let log2_range = if extended {
                        core::cmp::max(15, i32::from(bit_depth) + 6)
                    } else {
                        15
                    };
                    let log2_tbs = n_tbs.trailing_zeros() as i32;
                    let bd_shift = i32::from(bit_depth) + log2_tbs + 10 - log2_range;
                    // The full §7.4.3.1 qP range for the bit depth,
                    // 0..=51 + 6 * (bitDepth - 8).
                    for q_p in (0..=51 + 6 * (u32::from(bit_depth) - 8)).step_by(3) {
                        let levels: Vec<i32> = (0..count).map(|_| rng.signed(coeff_max)).collect();
                        // Explicit matrices carry the §7.4.5 1..=255 range.
                        let m: Vec<u16> = (0..count)
                            .map(|_| 1 + (rng.next_u32() % 255) as u16)
                            .collect();
                        let params = DequantParams {
                            level_scale: super::super::transform::LEVEL_SCALE[(q_p % 6) as usize],
                            qp_div6: q_p / 6,
                            bd_shift: bd_shift as u32,
                            coeff_min,
                            coeff_max,
                        };
                        for matrix in [None, Some(m.as_slice())] {
                            let mut expected = vec![0i32; count];
                            dequant_block(Backend::Scalar, &mut expected, &levels, matrix, params);
                            for backend in supported_backends() {
                                let mut got = vec![0i32; count];
                                dequant_block(backend, &mut got, &levels, matrix, params);
                                assert_eq!(
                                    got,
                                    expected,
                                    "backend {backend:?}, nTbS {n_tbs}, bitDepth \
                                     {bit_depth}, extended {extended}, qP {q_p}, \
                                     matrix {}",
                                    matrix.is_some()
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The dispatched §8.6.3 scaling process is bit-exact with a direct
    /// transcription of equation 8-309, end to end through
    /// [`scale_coefficients`] (i.e. including the row-major
    /// `ScalingFactor` handoff).
    #[test]
    fn scale_coefficients_matches_equation_8_309() {
        let mut rng = Lcg(0x8609_0309);
        for n_tbs in [4usize, 8, 16, 32] {
            let count = n_tbs * n_tbs;
            for bit_depth in [8u8, 10, 12] {
                let (coeff_min, coeff_max) = coeff_range(bit_depth, false);
                for q_p in 0..=51 + 6 * (u32::from(bit_depth) - 8) {
                    let levels: Vec<i32> = (0..count).map(|_| rng.signed(coeff_max)).collect();
                    let matrix = ScalingFactorMatrix {
                        dim: n_tbs as u8,
                        coef: (0..count)
                            .map(|_| 1 + (rng.next_u32() % 255) as u16)
                            .collect(),
                    };
                    for scaling in [None, Some(&matrix)] {
                        let got =
                            scale_coefficients(&levels, n_tbs, q_p, bit_depth, false, scaling)
                                .expect("valid scaling inputs");
                        let log2_tbs = n_tbs.trailing_zeros() as i32;
                        let bd_shift = i32::from(bit_depth) + log2_tbs + 10 - 15;
                        let round = 1i64 << (bd_shift - 1);
                        let level_scale =
                            i64::from(super::super::transform::LEVEL_SCALE[(q_p % 6) as usize]);
                        for y in 0..n_tbs {
                            for x in 0..n_tbs {
                                let idx = y * n_tbs + x;
                                let m = scaling.map_or(16i64, |sf| i64::from(sf.at(x, y)));
                                let prod = i64::from(levels[idx]) * m * level_scale;
                                let want = (((prod << (q_p / 6)) + round) >> bd_shift)
                                    .clamp(i64::from(coeff_min), i64::from(coeff_max))
                                    as i32;
                                assert_eq!(
                                    got[idx], want,
                                    "nTbS {n_tbs}, bitDepth {bit_depth}, qP {q_p}, \
                                     ({x},{y})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The vectorized §8.6.4 inverse transform is bit-exact with the
    /// `i64` reference for every transform size, both `trType` values,
    /// and every backend the host supports — including saturated inputs
    /// at the edges of the coefficient range.
    #[test]
    fn inverse_transform_matches_reference_for_every_backend() {
        let mut rng = Lcg(0xfeed_face);
        for n_tbs in [4usize, 8, 16, 32] {
            let count = n_tbs * n_tbs;
            for bit_depth in [8u8, 10, 12, 16] {
                let (coeff_min, coeff_max) = coeff_range(bit_depth, false);
                for (pred_mode, component) in [
                    (PredMode::Intra, Component::Luma),
                    (PredMode::Intra, Component::Cb),
                    (PredMode::Inter, Component::Luma),
                ] {
                    for round in 0..24 {
                        let d: Vec<i32> = (0..count)
                            .map(|_| match round % 4 {
                                // Sparse blocks (the common decode case),
                                // dense blocks, and worst-case saturation.
                                0 if rng.next_u32() % 8 != 0 => 0,
                                1 => coeff_max,
                                2 => coeff_min,
                                _ => rng.signed(coeff_max),
                            })
                            .collect();
                        let expected = inverse_transform_reference(
                            &d, n_tbs, pred_mode, component, bit_depth, false,
                        )
                        .expect("valid transform inputs");
                        for backend in supported_backends() {
                            let got = inverse_transform_with_backend(
                                backend, &d, n_tbs, pred_mode, component, bit_depth, false,
                            )
                            .expect("valid transform inputs");
                            assert_eq!(
                                got, expected,
                                "backend {backend:?}, nTbS {n_tbs}, bitDepth {bit_depth}, \
                                 {pred_mode:?}/{component:?}, round {round}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Extended precision widens the coefficient range past what an
    /// `i32` accumulator can hold, so it must keep taking the exact
    /// `i64` path rather than overflowing a vector lane.
    #[test]
    fn extended_precision_falls_back_to_the_i64_reference() {
        let n_tbs = 32;
        let count = n_tbs * n_tbs;
        let (_, coeff_max) = coeff_range(16, true);
        let d = vec![coeff_max; count];
        let expected =
            inverse_transform_reference(&d, n_tbs, PredMode::Inter, Component::Luma, 16, true)
                .expect("valid transform inputs");
        for backend in supported_backends() {
            let got = inverse_transform_with_backend(
                backend,
                &d,
                n_tbs,
                PredMode::Inter,
                Component::Luma,
                16,
                true,
            )
            .expect("valid transform inputs");
            assert_eq!(got, expected, "backend {backend:?}");
        }
    }

    /// Benchmark for the two vectorized kernels across the four HEVC
    /// transform sizes, reported per backend against the scalar path.
    ///
    /// Ignored by default because it only measures; run it with
    /// `cargo test --release --features native --lib \
    ///  transform_simd::tests::bench -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn bench_inverse_transform_and_dequant() {
        use std::time::Instant;

        let bit_depth = 8u8;
        let (coeff_min, coeff_max) = coeff_range(bit_depth, false);
        println!(
            "host backend: {:?} (available: {:?})",
            detected(),
            supported_backends()
        );
        for n_tbs in [4usize, 8, 16, 32] {
            let count = n_tbs * n_tbs;
            let mut rng = Lcg(0x1234_5678);
            // A realistically sparse dequantized block: a low-frequency
            // cluster plus scattered high-frequency levels.
            let d: Vec<i32> = (0..count)
                .map(|i| {
                    let (x, y) = (i % n_tbs, i / n_tbs);
                    if x + y < 4 || rng.next_u32() % 16 == 0 {
                        rng.signed(coeff_max)
                    } else {
                        0
                    }
                })
                .collect();
            let levels = d.clone();
            // Enough repetitions that each timed section runs for
            // ~100ms, plus an untimed warmup, so the numbers are not
            // dominated by first-touch and loop-entry noise.
            let iterations = 20_000_000 / count;
            let warmup = iterations / 8;

            // Baseline: the unreassociated i64 reference this change
            // replaced, so the table shows the end-to-end speedup and not
            // just backend-to-backend deltas.
            let mut sink = 0i64;
            for _ in 0..warmup {
                sink += i64::from(
                    inverse_transform_reference(
                        &d,
                        n_tbs,
                        PredMode::Inter,
                        Component::Luma,
                        bit_depth,
                        false,
                    )
                    .expect("valid transform inputs")[0],
                );
            }
            let start = Instant::now();
            for _ in 0..iterations {
                let r = inverse_transform_reference(
                    &d,
                    n_tbs,
                    PredMode::Inter,
                    Component::Luma,
                    bit_depth,
                    false,
                )
                .expect("valid transform inputs");
                sink += i64::from(r[0]);
            }
            let baseline = start.elapsed();
            println!(
                "{n_tbs:>2}x{n_tbs:<2} i64 reference: inverse transform {:>9.2} ns/block",
                baseline.as_nanos() as f64 / iterations as f64,
            );

            for backend in supported_backends() {
                for _ in 0..warmup {
                    sink += i64::from(
                        inverse_transform_with_backend(
                            backend,
                            &d,
                            n_tbs,
                            PredMode::Inter,
                            Component::Luma,
                            bit_depth,
                            false,
                        )
                        .expect("valid transform inputs")[0],
                    );
                }
                let start = Instant::now();
                for _ in 0..iterations {
                    let r = inverse_transform_with_backend(
                        backend,
                        &d,
                        n_tbs,
                        PredMode::Inter,
                        Component::Luma,
                        bit_depth,
                        false,
                    )
                    .expect("valid transform inputs");
                    sink += i64::from(r[0]);
                }
                let transform = start.elapsed();

                let params = DequantParams {
                    level_scale: 51,
                    qp_div6: 5,
                    bd_shift: 7,
                    coeff_min,
                    coeff_max,
                };
                let mut out = vec![0i32; count];
                for _ in 0..warmup {
                    dequant_block(backend, &mut out, &levels, None, params);
                    sink += i64::from(out[0]);
                }
                let start = Instant::now();
                for _ in 0..iterations {
                    dequant_block(backend, &mut out, &levels, None, params);
                    sink += i64::from(out[0]);
                }
                let dequant = start.elapsed();

                println!(
                    "{n_tbs:>2}x{n_tbs:<2} {backend:?}: inverse transform {:>9.2} ns/block \
                     ({:.2}x baseline), dequant {:>9.2} ns/block (checksum {sink})",
                    transform.as_nanos() as f64 / iterations as f64,
                    baseline.as_secs_f64() / transform.as_secs_f64(),
                    dequant.as_nanos() as f64 / iterations as f64,
                );
            }
        }
    }
}
