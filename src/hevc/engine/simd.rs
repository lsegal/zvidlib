//! Vectorized kernels for the §8.7.2 deblocking and §8.7.3 SAO in-loop
//! filters, with runtime CPU feature detection and a scalar fallback.
//!
//! The in-loop filters are the two stages of the HEVC decode pipeline that
//! touch *every* reconstructed sample of *every* picture, so they dominate
//! the software decoder's per-pixel cost once prediction and the inverse
//! transform have been optimized. This module keeps the spec-shaped scalar
//! implementations in [`deblock`] / [`sao`] as the normative reference and
//! adds bit-exact SIMD kernels underneath them:
//!
//! * **SAO band offset** (§8.7.3.2 equations 8-414..8-415) and **SAO edge
//!   offset** (equations 8-409..8-413) run over long runs of contiguous
//!   samples, so they use the widest available vector: AVX2 (8 x `i32`)
//!   when the CPU has it, otherwise SSE4.1 (4 x `i32`) on `x86_64` or NEON
//!   (4 x `i32`) on `aarch64`.
//! * **Deblocking luma strong / weak filtering** (§8.7.2.5.7 equations
//!   8-389..8-402) and **chroma filtering** (equations 8-403..8-405) are
//!   defined on a four-row edge *segment* with one shared decision, so the
//!   natural vectorization maps the segment's four rows onto four `i32`
//!   lanes. That is exactly one SSE4.1 / NEON register; AVX2's extra width
//!   has nothing to fill it with here, so the AVX2-capable path
//!   deliberately uses the same 128-bit kernels for deblocking and spends
//!   its width on SAO instead.
//!
//! Every kernel is written once as a generic function over the [`Ops`]
//! vector trait and instantiated per instruction set, so the SSE4.1, AVX2
//! and NEON paths cannot drift apart. Targets without a supported vector
//! ISA (notably `wasm32`) compile to the scalar path only; the dispatcher
//! is a cached runtime feature probe, so a binary built for a baseline
//! `x86_64` still uses AVX2 on a machine that has it.
//!
//! Bit-exactness against the scalar reference is asserted by the module
//! tests over exhaustive boundary-strength / QP / bit-depth / SAO
//! type / edge-class sweeps, and `bench_in_loop_filters` (an `#[ignore]`d
//! timing test) reports the measured speedup on a representative
//! reconstructed frame.

use core::sync::atomic::{AtomicU8, Ordering};

/// The selected kernel family. Cached in [`ISA`] after the first probe.
const ISA_UNKNOWN: u8 = 0;
const ISA_SCALAR: u8 = 1;
#[cfg(target_arch = "x86_64")]
const ISA_SSE41: u8 = 2;
#[cfg(target_arch = "x86_64")]
const ISA_AVX2: u8 = 3;
#[cfg(target_arch = "aarch64")]
const ISA_NEON: u8 = 4;

static ISA: AtomicU8 = AtomicU8::new(ISA_UNKNOWN);

/// Test-only switch that forces the scalar path, so the benchmark can
/// time both families in one process and the bit-exactness tests can pin
/// the reference side.
#[cfg(test)]
pub(crate) static FORCE_SCALAR: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Probe (once) and return the kernel family to use.
#[inline]
fn isa() -> u8 {
    #[cfg(test)]
    if FORCE_SCALAR.load(Ordering::Relaxed) {
        return ISA_SCALAR;
    }
    let cached = ISA.load(Ordering::Relaxed);
    if cached != ISA_UNKNOWN {
        return cached;
    }
    let detected = detect();
    ISA.store(detected, Ordering::Relaxed);
    detected
}

#[cfg(target_arch = "x86_64")]
fn detect() -> u8 {
    if std::is_x86_feature_detected!("avx2") {
        ISA_AVX2
    } else if std::is_x86_feature_detected!("sse4.1") {
        ISA_SSE41
    } else {
        ISA_SCALAR
    }
}

#[cfg(target_arch = "aarch64")]
fn detect() -> u8 {
    // NEON (`asimd`) is architecturally guaranteed on aarch64.
    ISA_NEON
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect() -> u8 {
    ISA_SCALAR
}

// ---------------------------------------------------------------------------
// Vector abstraction
// ---------------------------------------------------------------------------

/// The lane operations every kernel in this module needs, over a vector of
/// signed 32-bit lanes (HEVC sample arrays are stored as `i32`).
///
/// Implementations are thin wrappers over one instruction set's
/// intrinsics; each carries the `#[target_feature]` that makes calling it
/// sound once the corresponding [`detect`] probe has succeeded.
trait Ops: Copy {
    /// Lanes per vector.
    const LANES: usize;
    /// # Safety
    /// The caller must have verified the implementation's CPU feature.
    unsafe fn splat(v: i32) -> Self;
    /// # Safety
    /// `src` must be readable for `LANES` `i32`s, and the feature present.
    unsafe fn load(src: *const i32) -> Self;
    /// # Safety
    /// `dst` must be writable for `LANES` `i32`s, and the feature present.
    unsafe fn store(self, dst: *mut i32);
    /// # Safety
    /// The caller must have verified the implementation's CPU feature.
    unsafe fn add(self, o: Self) -> Self;
    /// # Safety
    /// The caller must have verified the implementation's CPU feature.
    unsafe fn sub(self, o: Self) -> Self;
    /// # Safety
    /// The caller must have verified the implementation's CPU feature.
    unsafe fn min(self, o: Self) -> Self;
    /// # Safety
    /// The caller must have verified the implementation's CPU feature.
    unsafe fn max(self, o: Self) -> Self;
    /// # Safety
    /// The caller must have verified the implementation's CPU feature.
    unsafe fn and(self, o: Self) -> Self;
    /// # Safety
    /// The caller must have verified the implementation's CPU feature.
    unsafe fn or(self, o: Self) -> Self;
    /// `(!self) & o`.
    ///
    /// # Safety
    /// The caller must have verified the implementation's CPU feature.
    unsafe fn andnot(self, o: Self) -> Self;
    /// Lanewise `self > o` as an all-ones / all-zeros mask.
    ///
    /// # Safety
    /// The caller must have verified the implementation's CPU feature.
    unsafe fn cmpgt(self, o: Self) -> Self;
    /// Lanewise `self == o` as an all-ones / all-zeros mask.
    ///
    /// # Safety
    /// The caller must have verified the implementation's CPU feature.
    unsafe fn cmpeq(self, o: Self) -> Self;
    /// Arithmetic right shift by a runtime count in `0..32`.
    ///
    /// # Safety
    /// The caller must have verified the implementation's CPU feature.
    unsafe fn sra(self, n: i32) -> Self;
    /// Left shift by a runtime count in `0..32`.
    ///
    /// # Safety
    /// The caller must have verified the implementation's CPU feature.
    unsafe fn sll(self, n: i32) -> Self;
}

/// `mask ? a : b` lanewise, for an all-ones / all-zeros `mask`.
///
/// # Safety
/// The caller must have verified `V`'s CPU feature.
#[inline(always)]
unsafe fn blend<V: Ops>(mask: V, a: V, b: V) -> V {
    unsafe { mask.and(a).or(mask.andnot(b)) }
}

/// Lanewise `|x|`.
///
/// # Safety
/// The caller must have verified `V`'s CPU feature.
#[inline(always)]
unsafe fn vabs<V: Ops>(x: V) -> V {
    unsafe { x.max(V::splat(0).sub(x)) }
}

// ---------------------------------------------------------------------------
// SSE4.1
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::Ops;
    use core::arch::x86_64::*;

    /// 4 x `i32` over SSE4.1.
    #[derive(Clone, Copy)]
    pub(super) struct V4(pub __m128i);

    impl Ops for V4 {
        const LANES: usize = 4;
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn splat(v: i32) -> Self {
            V4(_mm_set1_epi32(v))
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn load(src: *const i32) -> Self {
            unsafe { V4(_mm_loadu_si128(src.cast())) }
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn store(self, dst: *mut i32) {
            unsafe { _mm_storeu_si128(dst.cast(), self.0) }
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn add(self, o: Self) -> Self {
            V4(_mm_add_epi32(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn sub(self, o: Self) -> Self {
            V4(_mm_sub_epi32(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn min(self, o: Self) -> Self {
            V4(_mm_min_epi32(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn max(self, o: Self) -> Self {
            V4(_mm_max_epi32(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn and(self, o: Self) -> Self {
            V4(_mm_and_si128(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn or(self, o: Self) -> Self {
            V4(_mm_or_si128(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn andnot(self, o: Self) -> Self {
            V4(_mm_andnot_si128(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn cmpgt(self, o: Self) -> Self {
            V4(_mm_cmpgt_epi32(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn cmpeq(self, o: Self) -> Self {
            V4(_mm_cmpeq_epi32(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn sra(self, n: i32) -> Self {
            V4(_mm_sra_epi32(self.0, _mm_cvtsi32_si128(n)))
        }
        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn sll(self, n: i32) -> Self {
            V4(_mm_sll_epi32(self.0, _mm_cvtsi32_si128(n)))
        }
    }

    /// 8 x `i32` over AVX2.
    #[derive(Clone, Copy)]
    pub(super) struct V8(pub __m256i);

    impl Ops for V8 {
        const LANES: usize = 8;
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn splat(v: i32) -> Self {
            V8(_mm256_set1_epi32(v))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn load(src: *const i32) -> Self {
            unsafe { V8(_mm256_loadu_si256(src.cast())) }
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn store(self, dst: *mut i32) {
            unsafe { _mm256_storeu_si256(dst.cast(), self.0) }
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn add(self, o: Self) -> Self {
            V8(_mm256_add_epi32(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn sub(self, o: Self) -> Self {
            V8(_mm256_sub_epi32(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn min(self, o: Self) -> Self {
            V8(_mm256_min_epi32(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn max(self, o: Self) -> Self {
            V8(_mm256_max_epi32(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn and(self, o: Self) -> Self {
            V8(_mm256_and_si256(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn or(self, o: Self) -> Self {
            V8(_mm256_or_si256(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn andnot(self, o: Self) -> Self {
            V8(_mm256_andnot_si256(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn cmpgt(self, o: Self) -> Self {
            V8(_mm256_cmpgt_epi32(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn cmpeq(self, o: Self) -> Self {
            V8(_mm256_cmpeq_epi32(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn sra(self, n: i32) -> Self {
            V8(_mm256_sra_epi32(self.0, _mm_cvtsi32_si128(n)))
        }
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn sll(self, n: i32) -> Self {
            V8(_mm256_sll_epi32(self.0, _mm_cvtsi32_si128(n)))
        }
    }
}

// ---------------------------------------------------------------------------
// NEON
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::Ops;
    use core::arch::aarch64::*;

    /// 4 x `i32` over NEON (architecturally guaranteed on `aarch64`).
    #[derive(Clone, Copy)]
    pub(super) struct V4(pub int32x4_t);

    impl Ops for V4 {
        const LANES: usize = 4;
        #[inline]
        unsafe fn splat(v: i32) -> Self {
            unsafe { V4(vdupq_n_s32(v)) }
        }
        #[inline]
        unsafe fn load(src: *const i32) -> Self {
            unsafe { V4(vld1q_s32(src)) }
        }
        #[inline]
        unsafe fn store(self, dst: *mut i32) {
            unsafe { vst1q_s32(dst, self.0) }
        }
        #[inline]
        unsafe fn add(self, o: Self) -> Self {
            unsafe { V4(vaddq_s32(self.0, o.0)) }
        }
        #[inline]
        unsafe fn sub(self, o: Self) -> Self {
            unsafe { V4(vsubq_s32(self.0, o.0)) }
        }
        #[inline]
        unsafe fn min(self, o: Self) -> Self {
            unsafe { V4(vminq_s32(self.0, o.0)) }
        }
        #[inline]
        unsafe fn max(self, o: Self) -> Self {
            unsafe { V4(vmaxq_s32(self.0, o.0)) }
        }
        #[inline]
        unsafe fn and(self, o: Self) -> Self {
            unsafe { V4(vandq_s32(self.0, o.0)) }
        }
        #[inline]
        unsafe fn or(self, o: Self) -> Self {
            unsafe { V4(vorrq_s32(self.0, o.0)) }
        }
        #[inline]
        unsafe fn andnot(self, o: Self) -> Self {
            // NEON's `vbicq` is `a & !b`, so the operands swap.
            unsafe { V4(vbicq_s32(o.0, self.0)) }
        }
        #[inline]
        unsafe fn cmpgt(self, o: Self) -> Self {
            unsafe { V4(vreinterpretq_s32_u32(vcgtq_s32(self.0, o.0))) }
        }
        #[inline]
        unsafe fn cmpeq(self, o: Self) -> Self {
            unsafe { V4(vreinterpretq_s32_u32(vceqq_s32(self.0, o.0))) }
        }
        #[inline]
        unsafe fn sra(self, n: i32) -> Self {
            unsafe { V4(vshlq_s32(self.0, vdupq_n_s32(-n))) }
        }
        #[inline]
        unsafe fn sll(self, n: i32) -> Self {
            unsafe { V4(vshlq_s32(self.0, vdupq_n_s32(n))) }
        }
    }
}

// ---------------------------------------------------------------------------
// SAO band offset (§8.7.3.2 equations 8-414..8-415)
// ---------------------------------------------------------------------------

/// Scalar reference for one row run of SAO band offset.
///
/// `left` is `sao_band_position`, `band_shift` is `BitDepth - 5`, and
/// `off` is `SaoOffsetVal[..]` (`off[0]` is always 0). Equation 8-414's
/// `bandTable` is inverted here: band `b` is one of the four selected
/// bands exactly when `(b - left) & 31 < 4`, and then takes
/// `off[((b - left) & 31) + 1]`.
#[inline]
fn sao_band_row_scalar(
    src: &[i32],
    dst: &mut [i32],
    off: &[i32; 5],
    left: i32,
    band_shift: i32,
    max: i32,
) {
    for (d, &cur) in dst.iter_mut().zip(src.iter()) {
        let k = ((cur >> band_shift) - left) & 31;
        let o = if k < 4 { off[(k + 1) as usize] } else { 0 };
        *d = (cur + o).clamp(0, max);
    }
}

/// Vector body of [`sao_band_row_scalar`], with a scalar tail.
///
/// # Safety
/// The caller must have verified `V`'s CPU feature.
#[inline(always)]
unsafe fn sao_band_row_simd<V: Ops>(
    src: &[i32],
    dst: &mut [i32],
    off: &[i32; 5],
    left: i32,
    band_shift: i32,
    max: i32,
) {
    unsafe {
        let n = src.len().min(dst.len());
        let zero = V::splat(0);
        let vmax = V::splat(max);
        let vleft = V::splat(left);
        let m31 = V::splat(31);
        let ks = [V::splat(0), V::splat(1), V::splat(2), V::splat(3)];
        let os = [
            V::splat(off[1]),
            V::splat(off[2]),
            V::splat(off[3]),
            V::splat(off[4]),
        ];
        let mut i = 0usize;
        while i + V::LANES <= n {
            let cur = V::load(src.as_ptr().add(i));
            let k = cur.sra(band_shift).sub(vleft).and(m31);
            let mut o = zero;
            o = blend(k.cmpeq(ks[0]), os[0], o);
            o = blend(k.cmpeq(ks[1]), os[1], o);
            o = blend(k.cmpeq(ks[2]), os[2], o);
            o = blend(k.cmpeq(ks[3]), os[3], o);
            cur.add(o)
                .max(zero)
                .min(vmax)
                .store(dst.as_mut_ptr().add(i));
            i += V::LANES;
        }
        sao_band_row_scalar(&src[i..n], &mut dst[i..n], off, left, band_shift, max);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sao_band_row_sse41(
    src: &[i32],
    dst: &mut [i32],
    off: &[i32; 5],
    left: i32,
    band_shift: i32,
    max: i32,
) {
    unsafe { sao_band_row_simd::<x86::V4>(src, dst, off, left, band_shift, max) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sao_band_row_avx2(
    src: &[i32],
    dst: &mut [i32],
    off: &[i32; 5],
    left: i32,
    band_shift: i32,
    max: i32,
) {
    unsafe { sao_band_row_simd::<x86::V8>(src, dst, off, left, band_shift, max) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn sao_band_row_neon(
    src: &[i32],
    dst: &mut [i32],
    off: &[i32; 5],
    left: i32,
    band_shift: i32,
    max: i32,
) {
    unsafe { sao_band_row_simd::<arm::V4>(src, dst, off, left, band_shift, max) }
}

/// Apply SAO band offset to one contiguous run of a plane row.
///
/// `src` and `dst` are the pre-SAO and output runs (equal length);
/// see [`sao_band_row_scalar`] for the parameter meanings.
pub(crate) fn sao_band_row(
    src: &[i32],
    dst: &mut [i32],
    off: &[i32; 5],
    left: i32,
    band_shift: i32,
    max: i32,
) {
    match isa() {
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 => unsafe { sao_band_row_avx2(src, dst, off, left, band_shift, max) },
        #[cfg(target_arch = "x86_64")]
        ISA_SSE41 => unsafe { sao_band_row_sse41(src, dst, off, left, band_shift, max) },
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => unsafe { sao_band_row_neon(src, dst, off, left, band_shift, max) },
        _ => sao_band_row_scalar(src, dst, off, left, band_shift, max),
    }
}

// ---------------------------------------------------------------------------
// SAO edge offset (§8.7.3.2 equations 8-409..8-413)
// ---------------------------------------------------------------------------

/// Scalar reference for one row run of SAO edge offset.
///
/// `cur` is the pre-SAO run; `n0` / `n1` are the matching runs of the two
/// Table 8-13 neighbours (already offset by `hPos` / `vPos`), which the
/// caller has guaranteed to lie inside the picture.
#[inline]
fn sao_edge_row_scalar(
    cur: &[i32],
    n0: &[i32],
    n1: &[i32],
    dst: &mut [i32],
    off: &[i32; 5],
    max: i32,
) {
    for i in 0..dst.len() {
        let c = cur[i];
        // equation 8-411.
        let mut idx = 2 + (c - n0[i]).signum() + (c - n1[i]).signum();
        // equation 8-412.
        if idx <= 2 {
            idx = if idx == 2 { 0 } else { idx + 1 };
        }
        // equation 8-413.
        dst[i] = (c + off[idx as usize]).clamp(0, max);
    }
}

/// Vector body of [`sao_edge_row_scalar`], with a scalar tail.
///
/// # Safety
/// The caller must have verified `V`'s CPU feature.
#[inline(always)]
unsafe fn sao_edge_row_simd<V: Ops>(
    cur: &[i32],
    n0: &[i32],
    n1: &[i32],
    dst: &mut [i32],
    off: &[i32; 5],
    max: i32,
) {
    unsafe {
        let n = dst.len();
        let zero = V::splat(0);
        let one = V::splat(1);
        let two = V::splat(2);
        let vmax = V::splat(max);
        let idxs = [
            V::splat(0),
            V::splat(1),
            V::splat(2),
            V::splat(3),
            V::splat(4),
        ];
        let os = [
            V::splat(off[1]),
            V::splat(off[2]),
            V::splat(off[3]),
            V::splat(off[4]),
        ];
        let mut i = 0usize;
        while i + V::LANES <= n {
            let c = V::load(cur.as_ptr().add(i));
            let a = V::load(n0.as_ptr().add(i));
            let b = V::load(n1.as_ptr().add(i));
            // Sign( x ) as `(x < 0) - (x > 0)` over all-ones masks.
            let d0 = c.sub(a);
            let d1 = c.sub(b);
            let s0 = zero.cmpgt(d0).sub(d0.cmpgt(zero));
            let s1 = zero.cmpgt(d1).sub(d1.cmpgt(zero));
            // equation 8-411.
            let raw = two.add(s0).add(s1);
            // equation 8-412: 0 -> 1, 1 -> 2, 2 -> 0, 3 and 4 unchanged.
            let idx = blend(
                raw.cmpeq(idxs[2]),
                zero,
                blend(two.cmpgt(raw), raw.add(one), raw),
            );
            // equation 8-413 (`off[0]` is 0, so index 0 needs no blend).
            let mut o = zero;
            o = blend(idx.cmpeq(idxs[1]), os[0], o);
            o = blend(idx.cmpeq(idxs[2]), os[1], o);
            o = blend(idx.cmpeq(idxs[3]), os[2], o);
            o = blend(idx.cmpeq(idxs[4]), os[3], o);
            c.add(o).max(zero).min(vmax).store(dst.as_mut_ptr().add(i));
            i += V::LANES;
        }
        sao_edge_row_scalar(&cur[i..n], &n0[i..n], &n1[i..n], &mut dst[i..n], off, max);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn sao_edge_row_sse41(
    cur: &[i32],
    n0: &[i32],
    n1: &[i32],
    dst: &mut [i32],
    off: &[i32; 5],
    max: i32,
) {
    unsafe { sao_edge_row_simd::<x86::V4>(cur, n0, n1, dst, off, max) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sao_edge_row_avx2(
    cur: &[i32],
    n0: &[i32],
    n1: &[i32],
    dst: &mut [i32],
    off: &[i32; 5],
    max: i32,
) {
    unsafe { sao_edge_row_simd::<x86::V8>(cur, n0, n1, dst, off, max) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn sao_edge_row_neon(
    cur: &[i32],
    n0: &[i32],
    n1: &[i32],
    dst: &mut [i32],
    off: &[i32; 5],
    max: i32,
) {
    unsafe { sao_edge_row_simd::<arm::V4>(cur, n0, n1, dst, off, max) }
}

/// Apply SAO edge offset to one contiguous run of a plane row whose two
/// Table 8-13 neighbour runs (`n0` / `n1`) are entirely inside the
/// picture and unmasked by slice / tile / PCM guards.
pub(crate) fn sao_edge_row(
    cur: &[i32],
    n0: &[i32],
    n1: &[i32],
    dst: &mut [i32],
    off: &[i32; 5],
    max: i32,
) {
    debug_assert_eq!(cur.len(), dst.len());
    debug_assert_eq!(n0.len(), dst.len());
    debug_assert_eq!(n1.len(), dst.len());
    match isa() {
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 => unsafe { sao_edge_row_avx2(cur, n0, n1, dst, off, max) },
        #[cfg(target_arch = "x86_64")]
        ISA_SSE41 => unsafe { sao_edge_row_sse41(cur, n0, n1, dst, off, max) },
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => unsafe { sao_edge_row_neon(cur, n0, n1, dst, off, max) },
        _ => sao_edge_row_scalar(cur, n0, n1, dst, off, max),
    }
}

// ---------------------------------------------------------------------------
// Deblocking luma filtering (§8.7.2.5.7 equations 8-389..8-402)
// ---------------------------------------------------------------------------

/// The four rows of one luma edge segment, as `p[i][k]` = `pi,k`.
pub(crate) type LumaSeg = [[i32; 4]; 4];
/// The filtered `p0'..p2'` / `q0'..q2'` of one luma edge segment.
pub(crate) type LumaSegOut = [[i32; 4]; 3];

/// Vectorized §8.7.2.5.7 luma filtering of a whole four-row edge segment.
///
/// The segment's four rows share one `dE` / `dEp` / `dEq` / `tC`
/// decision, so they map onto the four lanes of a 128-bit vector.
///
/// Rows for which the weak filter's equation 8-395 `|delta| < tC * 10`
/// test fails keep their input values in `out_p` / `out_q`, which makes
/// writing them back a no-op — the scalar path expresses the same thing
/// as `nDp = nDq = 0`.
///
/// # Safety
/// The caller must have verified `V`'s CPU feature; `V` must be 4-lane.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn filter_luma_rows_simd<V: Ops>(
    p: &LumaSeg,
    q: &LumaSeg,
    de: u8,
    dep: u8,
    deq: u8,
    tc: i32,
    bit_depth: u8,
    out_p: &mut LumaSegOut,
    out_q: &mut LumaSegOut,
) {
    unsafe {
        let p0 = V::load(p[0].as_ptr());
        let p1 = V::load(p[1].as_ptr());
        let p2 = V::load(p[2].as_ptr());
        let p3 = V::load(p[3].as_ptr());
        let q0 = V::load(q[0].as_ptr());
        let q1 = V::load(q[1].as_ptr());
        let q2 = V::load(q[2].as_ptr());
        let q3 = V::load(q[3].as_ptr());
        let two_tc = V::splat(2 * tc);

        if de == 2 {
            // Strong filter (equations 8-389..8-394); the ±2*tC clip is
            // the whole clipping, there is no Clip1 here.
            let four = V::splat(4);
            let two = V::splat(2);
            let clip3 = |x: V, v: V| x.sub(two_tc).max(x.add(two_tc).min(v));
            // p0' = ( p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4 ) >> 3
            let v = p2
                .add(p1.sll(1))
                .add(p0.sll(1))
                .add(q0.sll(1))
                .add(q1)
                .add(four)
                .sra(3);
            clip3(p0, v).store(out_p[0].as_mut_ptr());
            // p1' = ( p2 + p1 + p0 + q0 + 2 ) >> 2
            let v = p2.add(p1).add(p0).add(q0).add(two).sra(2);
            clip3(p1, v).store(out_p[1].as_mut_ptr());
            // p2' = ( 2*p3 + 3*p2 + p1 + p0 + q0 + 4 ) >> 3
            let v = p3
                .sll(1)
                .add(p2.sll(1))
                .add(p2)
                .add(p1)
                .add(p0)
                .add(q0)
                .add(four)
                .sra(3);
            clip3(p2, v).store(out_p[2].as_mut_ptr());
            // q0' = ( p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4 ) >> 3
            let v = p1
                .add(p0.sll(1))
                .add(q0.sll(1))
                .add(q1.sll(1))
                .add(q2)
                .add(four)
                .sra(3);
            clip3(q0, v).store(out_q[0].as_mut_ptr());
            // q1' = ( p0 + q0 + q1 + q2 + 2 ) >> 2
            let v = p0.add(q0).add(q1).add(q2).add(two).sra(2);
            clip3(q1, v).store(out_q[1].as_mut_ptr());
            // q2' = ( p0 + q0 + q1 + 3*q2 + 2*q3 + 4 ) >> 3
            let v = p0
                .add(q0)
                .add(q1)
                .add(q2.sll(1))
                .add(q2)
                .add(q3.sll(1))
                .add(four)
                .sra(3);
            clip3(q2, v).store(out_q[2].as_mut_ptr());
            return;
        }

        // Weak filter (equations 8-395..8-402).
        let zero = V::splat(0);
        let one = V::splat(1);
        let vhigh = V::splat((1i32 << bit_depth) - 1);
        let clip1 = |x: V| x.max(zero).min(vhigh);
        let a = q0.sub(p0);
        let b = q1.sub(p1);
        // delta = ( 9*(q0 - p0) - 3*(q1 - p1) + 8 ) >> 4
        let delta = a.sll(3).add(a).sub(b.sll(1).add(b)).add(V::splat(8)).sra(4);
        // The rows that pass |delta| < tC * 10 are the filtered ones.
        let keep = V::splat(tc * 10).cmpgt(vabs(delta));
        let d = delta.max(V::splat(-tc)).min(V::splat(tc)); // equation 8-396
        blend(keep, clip1(p0.add(d)), p0).store(out_p[0].as_mut_ptr()); // eq. 8-397
        blend(keep, clip1(q0.sub(d)), q0).store(out_q[0].as_mut_ptr()); // eq. 8-398
        let half_lo = V::splat(-(tc >> 1));
        let half_hi = V::splat(tc >> 1);
        if dep == 1 {
            // equations 8-399 / 8-400.
            let dp = p2
                .add(p0)
                .add(one)
                .sra(1)
                .sub(p1)
                .add(d)
                .sra(1)
                .max(half_lo)
                .min(half_hi);
            blend(keep, clip1(p1.add(dp)), p1).store(out_p[1].as_mut_ptr());
        } else {
            p1.store(out_p[1].as_mut_ptr());
        }
        if deq == 1 {
            // equations 8-401 / 8-402.
            let dq = q2
                .add(q0)
                .add(one)
                .sra(1)
                .sub(q1)
                .sub(d)
                .sra(1)
                .max(half_lo)
                .min(half_hi);
            blend(keep, clip1(q1.add(dq)), q1).store(out_q[1].as_mut_ptr());
        } else {
            q1.store(out_q[1].as_mut_ptr());
        }
        p2.store(out_p[2].as_mut_ptr());
        q2.store(out_q[2].as_mut_ptr());
    }
}

/// Scalar reference: [`super::deblock::filter_luma_sample`] per row.
#[allow(clippy::too_many_arguments)]
fn filter_luma_rows_scalar(
    p: &LumaSeg,
    q: &LumaSeg,
    de: u8,
    dep: u8,
    deq: u8,
    tc: i32,
    bit_depth: u8,
    out_p: &mut LumaSegOut,
    out_q: &mut LumaSegOut,
) {
    for k in 0..4 {
        let row_p = [p[0][k], p[1][k], p[2][k], p[3][k]];
        let row_q = [q[0][k], q[1][k], q[2][k], q[3][k]];
        let out = super::deblock::filter_luma_sample(row_p, row_q, de, dep, deq, tc, bit_depth);
        // `out.p` / `out.q` hold the input samples wherever the filter
        // did not apply, so copying all three is a no-op there.
        for i in 0..3 {
            out_p[i][k] = out.p[i];
            out_q[i][k] = out.q[i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
unsafe fn filter_luma_rows_sse41(
    p: &LumaSeg,
    q: &LumaSeg,
    de: u8,
    dep: u8,
    deq: u8,
    tc: i32,
    bit_depth: u8,
    out_p: &mut LumaSegOut,
    out_q: &mut LumaSegOut,
) {
    unsafe {
        filter_luma_rows_simd::<x86::V4>(p, q, de, dep, deq, tc, bit_depth, out_p, out_q);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn filter_luma_rows_neon(
    p: &LumaSeg,
    q: &LumaSeg,
    de: u8,
    dep: u8,
    deq: u8,
    tc: i32,
    bit_depth: u8,
    out_p: &mut LumaSegOut,
    out_q: &mut LumaSegOut,
) {
    unsafe {
        filter_luma_rows_simd::<arm::V4>(p, q, de, dep, deq, tc, bit_depth, out_p, out_q);
    }
}

/// §8.7.2.5.7 luma filtering of one four-row edge segment.
///
/// `p[i][k]` / `q[i][k]` are the segment's samples (`i` = distance from
/// the edge, `k` = row along it); `out_p` / `out_q` receive `p0'..p2'` /
/// `q0'..q2'`. Only `0..nDp` / `0..nDq` of them are replacements, exactly
/// as for [`super::deblock::filter_luma_sample`]; the remaining entries
/// hold the unmodified input samples.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_luma_rows(
    p: &LumaSeg,
    q: &LumaSeg,
    de: u8,
    dep: u8,
    deq: u8,
    tc: i32,
    bit_depth: u8,
    out_p: &mut LumaSegOut,
    out_q: &mut LumaSegOut,
) {
    match isa() {
        // AVX2 has no extra width to spend on a four-row segment.
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 | ISA_SSE41 => unsafe {
            filter_luma_rows_sse41(p, q, de, dep, deq, tc, bit_depth, out_p, out_q);
        },
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => unsafe {
            filter_luma_rows_neon(p, q, de, dep, deq, tc, bit_depth, out_p, out_q);
        },
        _ => filter_luma_rows_scalar(p, q, de, dep, deq, tc, bit_depth, out_p, out_q),
    }
}

// ---------------------------------------------------------------------------
// Deblocking chroma filtering (§8.7.2.5.7 equations 8-403..8-405)
// ---------------------------------------------------------------------------

/// The four rows of one chroma edge segment, as `p[i][k]` = `pi,k`.
pub(crate) type ChromaSeg = [[i32; 4]; 2];

/// Vectorized chroma filtering of a whole four-row edge segment.
///
/// # Safety
/// The caller must have verified `V`'s CPU feature; `V` must be 4-lane.
#[inline(always)]
unsafe fn filter_chroma_rows_simd<V: Ops>(
    p: &ChromaSeg,
    q: &ChromaSeg,
    tc: i32,
    bit_depth: u8,
    out_p0: &mut [i32; 4],
    out_q0: &mut [i32; 4],
) {
    unsafe {
        let p0 = V::load(p[0].as_ptr());
        let p1 = V::load(p[1].as_ptr());
        let q0 = V::load(q[0].as_ptr());
        let q1 = V::load(q[1].as_ptr());
        let zero = V::splat(0);
        let vhigh = V::splat((1i32 << bit_depth) - 1);
        // delta = Clip3( -tC, tC, ( ( ( q0 - p0 ) << 2 ) + p1 - q1 + 4 ) >> 3 )
        let d = q0
            .sub(p0)
            .sll(2)
            .add(p1)
            .sub(q1)
            .add(V::splat(4))
            .sra(3)
            .max(V::splat(-tc))
            .min(V::splat(tc));
        p0.add(d).max(zero).min(vhigh).store(out_p0.as_mut_ptr()); // eq. 8-404
        q0.sub(d).max(zero).min(vhigh).store(out_q0.as_mut_ptr()); // eq. 8-405
    }
}

/// Scalar reference: [`super::deblock::filter_chroma_sample`] per row.
fn filter_chroma_rows_scalar(
    p: &ChromaSeg,
    q: &ChromaSeg,
    tc: i32,
    bit_depth: u8,
    out_p0: &mut [i32; 4],
    out_q0: &mut [i32; 4],
) {
    for k in 0..4 {
        let (a, b) = super::deblock::filter_chroma_sample(
            [p[0][k], p[1][k]],
            [q[0][k], q[1][k]],
            tc,
            bit_depth,
        );
        out_p0[k] = a;
        out_q0[k] = b;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn filter_chroma_rows_sse41(
    p: &ChromaSeg,
    q: &ChromaSeg,
    tc: i32,
    bit_depth: u8,
    out_p0: &mut [i32; 4],
    out_q0: &mut [i32; 4],
) {
    unsafe { filter_chroma_rows_simd::<x86::V4>(p, q, tc, bit_depth, out_p0, out_q0) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn filter_chroma_rows_neon(
    p: &ChromaSeg,
    q: &ChromaSeg,
    tc: i32,
    bit_depth: u8,
    out_p0: &mut [i32; 4],
    out_q0: &mut [i32; 4],
) {
    unsafe { filter_chroma_rows_simd::<arm::V4>(p, q, tc, bit_depth, out_p0, out_q0) }
}

/// §8.7.2.5.7 chroma filtering of one four-row edge segment, producing
/// `p0'` / `q0'` for each row.
pub(crate) fn filter_chroma_rows(
    p: &ChromaSeg,
    q: &ChromaSeg,
    tc: i32,
    bit_depth: u8,
    out_p0: &mut [i32; 4],
    out_q0: &mut [i32; 4],
) {
    match isa() {
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 | ISA_SSE41 => unsafe {
            filter_chroma_rows_sse41(p, q, tc, bit_depth, out_p0, out_q0);
        },
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => unsafe {
            filter_chroma_rows_neon(p, q, tc, bit_depth, out_p0, out_q0);
        },
        _ => filter_chroma_rows_scalar(p, q, tc, bit_depth, out_p0, out_q0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::engine::deblock::{
        EdgePos, EdgeQp, EdgeType, SamplePlane, filter_chroma_block_edge, filter_luma_block_edge,
    };
    use crate::hevc::engine::picture::{Picture, Plane};
    use crate::hevc::engine::sao::{ResolvedSaoComponent, SaoBoundaries, apply_sao_ctb_full};
    use std::sync::{Mutex, MutexGuard};

    /// Serializes the tests that pin [`FORCE_SCALAR`], which is process
    /// global. Only these tests need to exclude each other: the scalar
    /// and vector kernels are bit-exact, so an unrelated test that runs
    /// while the switch is on still sees correct output.
    static PIN: Mutex<()> = Mutex::new(());

    /// A pinned reference run: everything inside `f` uses the scalar
    /// kernels.
    fn with_scalar<T>(f: impl FnOnce() -> T) -> (T, MutexGuard<'static, ()>) {
        let guard = PIN.lock().unwrap_or_else(|e| e.into_inner());
        FORCE_SCALAR.store(true, Ordering::SeqCst);
        let out = f();
        FORCE_SCALAR.store(false, Ordering::SeqCst);
        (out, guard)
    }

    /// Deterministic xorshift so failures reproduce exactly.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        fn next(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn sample(&mut self, bit_depth: u8) -> i32 {
            (self.next() % (1u32 << bit_depth)) as i32
        }
    }

    fn offsets(rng: &mut Rng, scale: i32) -> [i32; 5] {
        let mut o = [0i32; 5];
        for v in o.iter_mut().skip(1) {
            *v = (rng.next() % (2 * scale as u32 + 1)) as i32 - scale;
        }
        o
    }

    #[test]
    fn sao_band_row_kernel_is_bit_exact_across_bands_and_bit_depths() {
        // Hold PIN so no concurrently running test can pin the
        // scalar path underneath us; this must exercise the vector one.
        let _pin = PIN.lock().unwrap_or_else(|e| e.into_inner());
        let mut rng = Rng::new(0x5A0B);
        for &bit_depth in &[8u8, 10, 12] {
            let max = (1i32 << bit_depth) - 1;
            let band_shift = i32::from(bit_depth) - 5;
            for band_position in 0..32i32 {
                let off = offsets(&mut rng, 7 << (bit_depth - 8));
                // Lengths straddle every vector width so the scalar tail
                // of each kernel is exercised too.
                for len in 0..40usize {
                    let src: Vec<i32> = (0..len).map(|_| rng.sample(bit_depth)).collect();
                    let mut want = vec![0i32; len];
                    let mut got = vec![0i32; len];
                    sao_band_row_scalar(&src, &mut want, &off, band_position, band_shift, max);
                    sao_band_row(&src, &mut got, &off, band_position, band_shift, max);
                    assert_eq!(got, want, "bd={bit_depth} band={band_position} len={len}");
                }
            }
        }
    }

    #[test]
    fn sao_edge_row_kernel_is_bit_exact_for_every_sign_pattern() {
        // Hold PIN so no concurrently running test can pin the
        // scalar path underneath us; this must exercise the vector one.
        let _pin = PIN.lock().unwrap_or_else(|e| e.into_inner());
        let mut rng = Rng::new(0xED9E);
        for &bit_depth in &[8u8, 10, 12] {
            let max = (1i32 << bit_depth) - 1;
            let off = offsets(&mut rng, 7 << (bit_depth - 8));
            for len in 0..40usize {
                // Neighbours drawn from a tiny alphabet so all nine
                // (sign, sign) combinations of equation 8-411 appear.
                let src: Vec<i32> = (0..len).map(|_| (rng.next() % 3) as i32 + 1).collect();
                let n0: Vec<i32> = (0..len).map(|_| (rng.next() % 3) as i32 + 1).collect();
                let n1: Vec<i32> = (0..len).map(|_| (rng.next() % 3) as i32 + 1).collect();
                let mut want = vec![0i32; len];
                let mut got = vec![0i32; len];
                sao_edge_row_scalar(&src, &n0, &n1, &mut want, &off, max);
                sao_edge_row(&src, &n0, &n1, &mut got, &off, max);
                assert_eq!(got, want, "sign sweep bd={bit_depth} len={len}");
                // ... and again over the full sample range, which also
                // exercises the equation 8-413 clip at both ends.
                let src: Vec<i32> = (0..len).map(|_| rng.sample(bit_depth)).collect();
                let n0: Vec<i32> = (0..len).map(|_| rng.sample(bit_depth)).collect();
                let n1: Vec<i32> = (0..len).map(|_| rng.sample(bit_depth)).collect();
                let mut want = vec![0i32; len];
                let mut got = vec![0i32; len];
                sao_edge_row_scalar(&src, &n0, &n1, &mut want, &off, max);
                sao_edge_row(&src, &n0, &n1, &mut got, &off, max);
                assert_eq!(got, want, "range sweep bd={bit_depth} len={len}");
            }
        }
    }

    #[test]
    fn deblock_luma_rows_are_bit_exact_across_decisions_and_tc() {
        // Hold PIN so no concurrently running test can pin the
        // scalar path underneath us; this must exercise the vector one.
        let _pin = PIN.lock().unwrap_or_else(|e| e.into_inner());
        let mut rng = Rng::new(0xDEB1);
        for &bit_depth in &[8u8, 10] {
            // Every tC the §8.7.2.5.3 table can produce at this depth.
            for q_tc in 0..=53i32 {
                let tc = super::super::deblock::tc_prime(q_tc) * (1 << (bit_depth - 8));
                for de in 1..=2u8 {
                    for dep in 0..=1u8 {
                        for deq in 0..=1u8 {
                            for _ in 0..8 {
                                let mut p: LumaSeg = [[0; 4]; 4];
                                let mut q: LumaSeg = [[0; 4]; 4];
                                for i in 0..4 {
                                    for k in 0..4 {
                                        p[i][k] = rng.sample(bit_depth);
                                        q[i][k] = rng.sample(bit_depth);
                                    }
                                }
                                let mut want_p: LumaSegOut = [[0; 4]; 3];
                                let mut want_q: LumaSegOut = [[0; 4]; 3];
                                let mut got_p: LumaSegOut = [[0; 4]; 3];
                                let mut got_q: LumaSegOut = [[0; 4]; 3];
                                filter_luma_rows_scalar(
                                    &p,
                                    &q,
                                    de,
                                    dep,
                                    deq,
                                    tc,
                                    bit_depth,
                                    &mut want_p,
                                    &mut want_q,
                                );
                                filter_luma_rows(
                                    &p, &q, de, dep, deq, tc, bit_depth, &mut got_p, &mut got_q,
                                );
                                let ndp = if de == 2 { 3 } else { (dep + 1) as usize };
                                let ndq = if de == 2 { 3 } else { (deq + 1) as usize };
                                assert_eq!(
                                    got_p[..ndp],
                                    want_p[..ndp],
                                    "p side bd={bit_depth} tc={tc} dE={de} dEp={dep}"
                                );
                                assert_eq!(
                                    got_q[..ndq],
                                    want_q[..ndq],
                                    "q side bd={bit_depth} tc={tc} dE={de} dEq={deq}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn deblock_chroma_rows_are_bit_exact_across_tc() {
        // Hold PIN so no concurrently running test can pin the
        // scalar path underneath us; this must exercise the vector one.
        let _pin = PIN.lock().unwrap_or_else(|e| e.into_inner());
        let mut rng = Rng::new(0xC780);
        for &bit_depth in &[8u8, 10] {
            for q_tc in 0..=53i32 {
                let tc = super::super::deblock::tc_prime(q_tc) * (1 << (bit_depth - 8));
                for _ in 0..16 {
                    let mut p: ChromaSeg = [[0; 4]; 2];
                    let mut q: ChromaSeg = [[0; 4]; 2];
                    for i in 0..2 {
                        for k in 0..4 {
                            p[i][k] = rng.sample(bit_depth);
                            q[i][k] = rng.sample(bit_depth);
                        }
                    }
                    let (mut want_p, mut want_q) = ([0i32; 4], [0i32; 4]);
                    let (mut got_p, mut got_q) = ([0i32; 4], [0i32; 4]);
                    filter_chroma_rows_scalar(&p, &q, tc, bit_depth, &mut want_p, &mut want_q);
                    filter_chroma_rows(&p, &q, tc, bit_depth, &mut got_p, &mut got_q);
                    assert_eq!((got_p, got_q), (want_p, want_q), "bd={bit_depth} tc={tc}");
                }
            }
        }
    }

    /// A `SaoBoundaries` whose CTBs all share one slice and one tile, so
    /// `neighbour_allowed` is always true. Passing it keeps
    /// `apply_sao_ctb_full` on its normative scalar loop with exactly the
    /// semantics of the `None` (vectorized) path.
    fn permissive_boundaries(pic: &Picture, ctb_log2: u32) -> SaoBoundaries {
        let w_ctbs = pic.width_luma().div_ceil(1 << ctb_log2);
        let h_ctbs = pic.height_luma().div_ceil(1 << ctb_log2);
        SaoBoundaries {
            slice_addr_of_ctb: vec![0; w_ctbs * h_ctbs],
            tile_id_of_ctb: vec![0; w_ctbs * h_ctbs],
            pic_w_ctbs: w_ctbs,
            ctb_log2_size_y: ctb_log2,
            across_slices: true,
            across_tiles: true,
            filter_across_of_ctb: None,
            ctb_ts_of_rs: None,
        }
    }

    fn filled_picture(w: usize, h: usize, bit_depth: u8, seed: u64) -> Picture {
        let mut pic = Picture::new(w, h, 1, bit_depth, bit_depth);
        let mut rng = Rng::new(seed);
        for y in 0..h {
            for x in 0..w {
                pic.set_sample(Plane::Luma, x, y, rng.sample(bit_depth));
            }
        }
        let (cw, ch) = pic.plane_dims(Plane::Cb);
        for y in 0..ch {
            for x in 0..cw {
                let v = rng.sample(bit_depth);
                pic.set_sample(Plane::Cb, x, y, v);
                let v = rng.sample(bit_depth);
                pic.set_sample(Plane::Cr, x, y, v);
            }
        }
        pic
    }

    #[test]
    fn sao_ctb_vector_path_matches_the_normative_scalar_loop() {
        // Hold PIN so no concurrently running test can pin the
        // scalar path underneath us; this must exercise the vector one.
        let _pin = PIN.lock().unwrap_or_else(|e| e.into_inner());
        // 43 x 37 is deliberately not a multiple of any vector width or
        // of the CTB size, so partial CTBs and scalar tails are covered.
        let mut rng = Rng::new(0x5A0C7B);
        for &bit_depth in &[8u8, 10] {
            let rec = filled_picture(48, 40, bit_depth, 0x1234 + u64::from(bit_depth));
            let bounds = permissive_boundaries(&rec, 4);
            for sao_type_idx in 1..=2u8 {
                // All four Table 8-13 classes / all 32 band positions.
                for param in 0..32u8 {
                    let comp = ResolvedSaoComponent {
                        sao_type_idx,
                        offset_val: offsets(&mut rng, 7 << (bit_depth - 8)),
                        band_position: param,
                        eo_class: param & 3,
                    };
                    for plane in [Plane::Luma, Plane::Cb, Plane::Cr] {
                        let (pw, ph) = rec.plane_dims(plane);
                        for y_ctb in (0..ph).step_by(16) {
                            for x_ctb in (0..pw).step_by(16) {
                                let mut want = rec.clone();
                                let mut got = rec.clone();
                                apply_sao_ctb_full(
                                    &rec,
                                    &mut want,
                                    plane,
                                    &comp,
                                    x_ctb,
                                    y_ctb,
                                    16,
                                    16,
                                    Some(&bounds),
                                    None,
                                );
                                apply_sao_ctb_full(
                                    &rec, &mut got, plane, &comp, x_ctb, y_ctb, 16, 16, None, None,
                                );
                                assert_eq!(
                                    got.plane(plane),
                                    want.plane(plane),
                                    "type={sao_type_idx} param={param} ctb=({x_ctb},{y_ctb}) bd={bit_depth}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn deblock_block_edges_are_bit_exact_across_bs_qp_and_orientation() {
        let (w, h) = (32usize, 32usize);
        for &bit_depth in &[8u8, 10] {
            for edge in [EdgeType::Vertical, EdgeType::Horizontal] {
                for bs in 1..=2u8 {
                    for qp in (0..=51i32).step_by(3) {
                        for &off in &[-6i32, 0, 6] {
                            let base = filled_picture(w, h, bit_depth, 0xABCD + qp as u64);
                            let qpx = EdgeQp {
                                qp_q: qp,
                                qp_p: (qp + 4).min(51),
                                beta_offset_div2: off,
                                tc_offset_div2: off,
                                bit_depth,
                            };
                            let pos = EdgePos { ex: 8, ey: 8, edge };
                            let run = |pic: &mut Picture| {
                                let (buf, stride) = pic.plane_mut(Plane::Luma);
                                let mut sp = SamplePlane {
                                    samples: buf,
                                    width: w,
                                    stride,
                                };
                                let dec = filter_luma_block_edge(&mut sp, pos, bs, qpx);
                                let tc = filter_chroma_block_edge(&mut sp, pos, qpx, 0, 1);
                                (dec, tc)
                            };
                            let mut want = base.clone();
                            let (want_dec, guard) = with_scalar(|| run(&mut want));
                            let mut got = base.clone();
                            let got_dec = run(&mut got);
                            drop(guard);
                            assert_eq!(got_dec, want_dec);
                            assert_eq!(
                                got.plane(Plane::Luma),
                                want.plane(Plane::Luma),
                                "bd={bit_depth} bs={bs} qp={qp} off={off} {edge:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Reports the measured in-loop-filter speedup on a representative
    /// reconstructed frame. Ignored by default (it is a timing
    /// measurement, not an assertion):
    ///
    /// ```text
    /// cargo test --release --features native --lib \
    ///     hevc::engine::simd::tests::bench_in_loop_filters -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn bench_in_loop_filters() {
        use std::time::Instant;
        const W: usize = 1920;
        const H: usize = 1080;
        const CTB: usize = 64;
        let rec = filled_picture(W, H, 8, 0xB0B);
        let mut rng = Rng::new(0xBEE5);
        // One representative SAO parameter set per CTB, cycling the
        // band / edge types the way a real slice does.
        let comps: Vec<ResolvedSaoComponent> = (0..(W / CTB + 1) * (H / CTB + 1))
            .map(|i| ResolvedSaoComponent {
                sao_type_idx: [1u8, 2, 2, 1][i % 4],
                offset_val: offsets(&mut rng, 7),
                band_position: (i % 32) as u8,
                eo_class: (i % 4) as u8,
            })
            .collect();

        let sao_pass = || {
            let mut out = rec.clone();
            let mut c = 0usize;
            for y in (0..H).step_by(CTB) {
                for x in (0..W).step_by(CTB) {
                    apply_sao_ctb_full(
                        &rec,
                        &mut out,
                        Plane::Luma,
                        &comps[c % comps.len()],
                        x,
                        y,
                        CTB,
                        CTB,
                        None,
                        None,
                    );
                    c += 1;
                }
            }
            out
        };

        let deblock_pass = || {
            let mut pic = rec.clone();
            let (buf, stride) = pic.plane_mut(Plane::Luma);
            let mut sp = SamplePlane {
                samples: buf,
                width: W,
                stride,
            };
            let qp = EdgeQp {
                qp_q: 32,
                qp_p: 30,
                beta_offset_div2: 0,
                tc_offset_div2: 0,
                bit_depth: 8,
            };
            // The §8.7.2.5.1 sampling grid: every 8 samples across the
            // edge, every 4 along it, both orientations.
            for y in (4..H - 8).step_by(4) {
                for x in (8..W - 8).step_by(8) {
                    let pos = EdgePos {
                        ex: x,
                        ey: y,
                        edge: EdgeType::Vertical,
                    };
                    filter_luma_block_edge(&mut sp, pos, 2, qp);
                }
            }
            for y in (8..H - 8).step_by(8) {
                for x in (4..W - 8).step_by(4) {
                    let pos = EdgePos {
                        ex: x,
                        ey: y,
                        edge: EdgeType::Horizontal,
                    };
                    filter_luma_block_edge(&mut sp, pos, 2, qp);
                }
            }
        };

        let reps = 3;
        let guard = PIN.lock().unwrap_or_else(|e| e.into_inner());
        FORCE_SCALAR.store(true, Ordering::SeqCst);
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(sao_pass());
        }
        let sao_scalar = t.elapsed();
        let t = Instant::now();
        for _ in 0..reps {
            deblock_pass();
        }
        let deblock_scalar = t.elapsed();
        FORCE_SCALAR.store(false, Ordering::SeqCst);
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(sao_pass());
        }
        let sao_simd = t.elapsed();
        let t = Instant::now();
        for _ in 0..reps {
            deblock_pass();
        }
        let deblock_simd = t.elapsed();
        drop(guard);

        let ratio = |a: std::time::Duration, b: std::time::Duration| {
            a.as_secs_f64() / b.as_secs_f64().max(f64::EPSILON)
        };
        println!(
            "in-loop filter benchmark, {W}x{H} luma, {reps} frames, isa={}",
            isa()
        );
        println!(
            "  SAO      scalar {:>9.3?} / vector {:>9.3?}  => {:.2}x",
            sao_scalar / reps,
            sao_simd / reps,
            ratio(sao_scalar, sao_simd)
        );
        println!(
            "  deblock  scalar {:>9.3?} / vector {:>9.3?}  => {:.2}x",
            deblock_scalar / reps,
            deblock_simd / reps,
            ratio(deblock_scalar, deblock_simd)
        );
    }
}
