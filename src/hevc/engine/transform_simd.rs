//! Runtime-dispatched SIMD kernels for the HEVC §8.6.3 dequantization
//! scaling loop and the §8.6.4 inverse-transform butterfly passes.
//!
//! Both kernels are pure fixed-point integer reductions that the decoder
//! runs on the order of millions of times per frame, so they are the
//! natural place to spend explicit vectorization effort:
//!
//! * [`accumulate_scaled`] is the inner operation of the separable 1-D
//!   inverse DCT/DST: `out[ i ] += scale * coeffs[ i ]` over a whole
//!   basis row. Driving the butterfly as one broadcast-multiply-add per
//!   *non-zero* input turns the `nTbS`-term dot product into `nTbS`
//!   independent lanes and lets the (typically very sparse) coefficient
//!   vector skip most of the work outright.
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

/// `out[ i ] += scale * coeffs[ i ]` for the whole slice — one term of
/// the §8.6.4.2 equation-8-315/8-317 butterfly, accumulated across all
/// output positions at once.
///
/// `coeffs` is a row of the transform basis (`transMatrix`) and `scale`
/// a single input coefficient. Callers must guarantee that no partial
/// sum leaves the `i32` range; see
/// [`crate::hevc::engine::transform::inverse_transform`], which checks
/// the worst-case bound for `nTbS` and the block's `coeffMin`/`coeffMax`
/// before choosing this path.
///
/// # Panics
/// Panics if `out` and `coeffs` have different lengths.
pub fn accumulate_scaled(backend: Backend, out: &mut [i32], coeffs: &[i32], scale: i32) {
    assert_eq!(out.len(), coeffs.len(), "butterfly operand length mismatch");
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => {
            // SAFETY: `Backend::Avx2` is only produced by `detected` /
            // `supported_backends` after `is_x86_feature_detected!` has
            // confirmed AVX2 on this host.
            unsafe { x86::accumulate_scaled_avx2(out, coeffs, scale) }
        }
        #[cfg(target_arch = "x86_64")]
        Backend::Sse41 | Backend::Sse42 => {
            // SAFETY: as above, for SSE4.1 (SSE4.2 implies SSE4.1).
            unsafe { x86::accumulate_scaled_sse41(out, coeffs, scale) }
        }
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {
            // SAFETY: NEON is architecturally guaranteed on aarch64.
            unsafe { aarch64::accumulate_scaled_neon(out, coeffs, scale) }
        }
        _ => accumulate_scaled_scalar(out, coeffs, scale),
    }
}

/// Portable reference for [`accumulate_scaled`].
fn accumulate_scaled_scalar(out: &mut [i32], coeffs: &[i32], scale: i32) {
    for (o, &c) in out.iter_mut().zip(coeffs.iter()) {
        *o += scale * c;
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

    /// AVX2 [`super::accumulate_scaled`]: eight `i32` lanes per step,
    /// `vpmulld` + `vpaddd`, with a scalar tail for the (never taken for
    /// 4/8/16/32-wide blocks) remainder.
    ///
    /// # Safety
    /// The host must support AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn accumulate_scaled_avx2(out: &mut [i32], coeffs: &[i32], scale: i32) {
        unsafe {
            let s = _mm256_set1_epi32(scale);
            let mut i = 0;
            while i + 8 <= out.len() {
                let c = _mm256_loadu_si256(coeffs.as_ptr().add(i).cast());
                let o = _mm256_loadu_si256(out.as_ptr().add(i).cast());
                let acc = _mm256_add_epi32(o, _mm256_mullo_epi32(c, s));
                _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), acc);
                i += 8;
            }
            super::accumulate_scaled_scalar(&mut out[i..], &coeffs[i..], scale);
        }
    }

    /// SSE4.1 [`super::accumulate_scaled`]: four `i32` lanes per step.
    ///
    /// # Safety
    /// The host must support SSE4.1.
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn accumulate_scaled_sse41(out: &mut [i32], coeffs: &[i32], scale: i32) {
        unsafe {
            let s = _mm_set1_epi32(scale);
            let mut i = 0;
            while i + 4 <= out.len() {
                let c = _mm_loadu_si128(coeffs.as_ptr().add(i).cast());
                let o = _mm_loadu_si128(out.as_ptr().add(i).cast());
                let acc = _mm_add_epi32(o, _mm_mullo_epi32(c, s));
                _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), acc);
                i += 4;
            }
            super::accumulate_scaled_scalar(&mut out[i..], &coeffs[i..], scale);
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

    /// NEON [`super::accumulate_scaled`]: four `i32` lanes per step.
    ///
    /// # Safety
    /// The host must support NEON (guaranteed on aarch64).
    #[target_feature(enable = "neon")]
    pub unsafe fn accumulate_scaled_neon(out: &mut [i32], coeffs: &[i32], scale: i32) {
        unsafe {
            let s = vdupq_n_s32(scale);
            let mut i = 0;
            while i + 4 <= out.len() {
                let c = vld1q_s32(coeffs.as_ptr().add(i));
                let o = vld1q_s32(out.as_ptr().add(i));
                vst1q_s32(out.as_mut_ptr().add(i), vmlaq_s32(o, c, s));
                i += 4;
            }
            super::accumulate_scaled_scalar(&mut out[i..], &coeffs[i..], scale);
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
