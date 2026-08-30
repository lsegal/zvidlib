//! Runtime-dispatched SIMD kernels for the §8.5.3.3 inter-prediction hot
//! loops (`crate::hevc::engine::inter_pred`).
//!
//! Two primitives cover every vectorizable loop in the fractional-sample
//! interpolation and weighted-sample-prediction processes:
//!
//! * [`filter_taps`] — the separable filter accumulation
//!   `out[i] = ( Σ coeff[t] · tap[t][i] ) >> shift`. The §8.5.3.3.3.2
//!   8-tap luma and §8.5.3.3.3.3 4-tap chroma filters both reduce to it,
//!   for the horizontal pass (the taps are overlapping windows of one
//!   source row) and for the vertical pass (the taps are consecutive rows
//!   of the intermediate buffer) alike.
//! * [`combine_weighted`] — the sample combine
//!   `out[i] = Clip3( 0, max, ( ( Σ w[t] · tap[t][i] ) + round ) >> shift
//!   + post )`. The §8.5.3.3.4.2 default uni-/bi-predictive average and
//!   the §8.5.3.3.4.3 explicit weighted combine are both instances of it.
//!
//! Each primitive has an SSE4.1 and an AVX2 implementation on `x86_64`, a
//! NEON implementation on `aarch64`, and a scalar implementation used as
//! the fallback everywhere else (including `wasm32`). The backend is
//! selected once per process by [`detected_isa`] from the runtime CPU
//! feature flags; passing an [`Isa`] the running CPU does not support to
//! any kernel silently falls back to the scalar path, so the `_with`
//! entry points are safe to call with any value.
//!
//! Every backend is **bit-exact** with the scalar one: the accumulation
//! is plain wrapping-free `i32` arithmetic in the same order, and the
//! shifts are arithmetic right shifts. The interpolation intermediates
//! stay far inside `i32` for all supported bit depths (the widest case,
//! 16-bit two-dimensional luma, peaks near 2^20), and the explicit
//! weighted combine is only vectorized when a bound check proves the
//! `i32` product cannot overflow — otherwise it stays on the `i64`
//! scalar path (see `inter_pred`).

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use std::sync::OnceLock;

/// The instruction-set backend a kernel runs on.
///
/// Only the variants that can exist on the target architecture are
/// compiled in; [`Isa::Scalar`] is always available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Isa {
    /// Portable scalar fallback — the bit-exactness reference.
    Scalar,
    /// x86_64 SSE4.1 (`pmulld` / `pminsd` / `pmaxsd`), 4 lanes.
    #[cfg(target_arch = "x86_64")]
    Sse41,
    /// x86_64 AVX2, 8 lanes.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AArch64 NEON, 4 lanes.
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// Detects the widest backend the running CPU supports, once per process.
///
/// NEON is architecturally mandatory on AArch64, so `aarch64` reports
/// [`Isa::Neon`] unconditionally; `x86_64` probes AVX2 then SSE4.1 at
/// runtime; every other target (including `wasm32`) reports
/// [`Isa::Scalar`].
#[must_use]
pub fn detected_isa() -> Isa {
    static ISA: OnceLock<Isa> = OnceLock::new();
    *ISA.get_or_init(detect)
}

fn detect() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            Isa::Avx2
        } else if is_x86_feature_detected!("sse4.1") {
            Isa::Sse41
        } else {
            Isa::Scalar
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        Isa::Neon
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Isa::Scalar
    }
}

/// Every backend usable on this CPU, narrowest first, always starting
/// with [`Isa::Scalar`].
///
/// Intended for tests and benchmarks that want to exercise or time each
/// available backend against the scalar reference.
#[must_use]
pub fn available_isas() -> Vec<Isa> {
    let mut isas = vec![Isa::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            isas.push(Isa::Sse41);
        }
        if is_x86_feature_detected!("avx2") {
            isas.push(Isa::Avx2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        isas.push(Isa::Neon);
    }
    isas
}

/// Whether the running CPU can execute `isa`'s kernels.
#[inline]
fn supported(isa: Isa) -> bool {
    match isa {
        Isa::Scalar => true,
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => matches!(detected_isa(), Isa::Sse41 | Isa::Avx2),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => detected_isa() == Isa::Avx2,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => true,
    }
}

// ---------------------------------------------------------------------------
// Separable filter accumulation
// ---------------------------------------------------------------------------

/// `out[i] = ( Σ coeffs[t] · taps[t][i] ) >> shift` for every `i` in
/// `out`, on the requested backend.
///
/// `N` is the tap count (8 for the §8.5.3.3.3.2 luma filter, 4 for the
/// §8.5.3.3.3.3 chroma filter). Every `taps[t]` must be at least
/// `out.len()` long. `shift` must be in `0..32`. An `isa` the running
/// CPU does not support falls back to [`Isa::Scalar`].
#[inline]
pub fn filter_taps<const N: usize>(
    isa: Isa,
    taps: &[&[i32]; N],
    coeffs: &[i32; N],
    shift: i32,
    out: &mut [i32],
) {
    debug_assert!((0..32).contains(&shift));
    debug_assert!(taps.iter().all(|t| t.len() >= out.len()));
    if !supported(isa) {
        return filter_taps_scalar(taps, coeffs, shift, out);
    }
    match isa {
        Isa::Scalar => filter_taps_scalar(taps, coeffs, shift, out),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `supported` confirmed SSE4.1 above, and the tap slices
        // are at least `out.len()` long so every load stays in bounds.
        Isa::Sse41 => unsafe { filter_taps_sse41(taps, coeffs, shift, out) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `supported` confirmed AVX2 above; same bounds argument.
        Isa::Avx2 => unsafe { filter_taps_avx2(taps, coeffs, shift, out) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is mandatory on AArch64; same bounds argument.
        Isa::Neon => unsafe { filter_taps_neon(taps, coeffs, shift, out) },
    }
}

fn filter_taps_scalar<const N: usize>(
    taps: &[&[i32]; N],
    coeffs: &[i32; N],
    shift: i32,
    out: &mut [i32],
) {
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0i32;
        for (&c, tap) in coeffs.iter().zip(taps.iter()) {
            acc += c * tap[i];
        }
        *o = acc >> shift;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn filter_taps_sse41<const N: usize>(
    taps: &[&[i32]; N],
    coeffs: &[i32; N],
    shift: i32,
    out: &mut [i32],
) {
    unsafe {
        let count = out.len();
        let sh = _mm_cvtsi32_si128(shift);
        let mut i = 0usize;
        while i + 4 <= count {
            let mut acc = _mm_setzero_si128();
            for (&c, tap) in coeffs.iter().zip(taps.iter()) {
                let v = _mm_loadu_si128(tap.as_ptr().add(i).cast());
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(v, _mm_set1_epi32(c)));
            }
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm_sra_epi32(acc, sh));
            i += 4;
        }
        filter_taps_tail(taps, coeffs, shift, out, i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_taps_avx2<const N: usize>(
    taps: &[&[i32]; N],
    coeffs: &[i32; N],
    shift: i32,
    out: &mut [i32],
) {
    unsafe {
        let count = out.len();
        let sh = _mm_cvtsi32_si128(shift);
        let mut i = 0usize;
        while i + 8 <= count {
            let mut acc = _mm256_setzero_si256();
            for (&c, tap) in coeffs.iter().zip(taps.iter()) {
                let v = _mm256_loadu_si256(tap.as_ptr().add(i).cast());
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(v, _mm256_set1_epi32(c)));
            }
            _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), _mm256_sra_epi32(acc, sh));
            i += 8;
        }
        while i + 4 <= count {
            let mut acc = _mm_setzero_si128();
            for (&c, tap) in coeffs.iter().zip(taps.iter()) {
                let v = _mm_loadu_si128(tap.as_ptr().add(i).cast());
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(v, _mm_set1_epi32(c)));
            }
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm_sra_epi32(acc, sh));
            i += 4;
        }
        filter_taps_tail(taps, coeffs, shift, out, i);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_taps_neon<const N: usize>(
    taps: &[&[i32]; N],
    coeffs: &[i32; N],
    shift: i32,
    out: &mut [i32],
) {
    unsafe {
        let count = out.len();
        // A negative `vshlq_s32` amount is an arithmetic right shift.
        let sh = vdupq_n_s32(-shift);
        let mut i = 0usize;
        while i + 4 <= count {
            let mut acc = vdupq_n_s32(0);
            for (&c, tap) in coeffs.iter().zip(taps.iter()) {
                acc = vmlaq_n_s32(acc, vld1q_s32(tap.as_ptr().add(i)), c);
            }
            vst1q_s32(out.as_mut_ptr().add(i), vshlq_s32(acc, sh));
            i += 4;
        }
        filter_taps_tail(taps, coeffs, shift, out, i);
    }
}

/// The scalar remainder of a vector kernel, for the `out.len() % lanes`
/// samples the vector loops could not cover.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn filter_taps_tail<const N: usize>(
    taps: &[&[i32]; N],
    coeffs: &[i32; N],
    shift: i32,
    out: &mut [i32],
    from: usize,
) {
    for i in from..out.len() {
        let mut acc = 0i32;
        for (&c, tap) in coeffs.iter().zip(taps.iter()) {
            acc += c * tap[i];
        }
        out[i] = acc >> shift;
    }
}

// ---------------------------------------------------------------------------
// Weighted sample combine
// ---------------------------------------------------------------------------

/// `out[i] = Clip3( 0, max_val, ( ( ( Σ weights[t] · taps[t][i] ) + round )
/// >> shift ) + post )` for every `i` in `out`, on the requested backend.
///
/// `N` is 1 for the uni-predictive combines (§8.5.3.3.4.2 equations
/// 8-262 / 8-263, §8.5.3.3.4.3 equations 8-275 / 8-276) and 2 for the
/// bi-predictive ones (equations 8-264 / 8-277). Every `taps[t]` must be
/// at least `out.len()` long and `shift` must be in `0..32`. The caller
/// is responsible for having established that the accumulation cannot
/// overflow `i32`. An `isa` the running CPU does not support falls back
/// to [`Isa::Scalar`].
// The round / shift / post / max quartet are the four distinct spec
// quantities of equations 8-262..8-277; grouping them into the private
// `CombineParams` would force that type into the public signature.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn combine_weighted<const N: usize>(
    isa: Isa,
    taps: &[&[i32]; N],
    weights: &[i32; N],
    round: i32,
    shift: i32,
    post: i32,
    max_val: i32,
    out: &mut [i32],
) {
    debug_assert!((0..32).contains(&shift));
    debug_assert!(taps.iter().all(|t| t.len() >= out.len()));
    let p = CombineParams {
        round,
        shift,
        post,
        max_val,
    };
    if !supported(isa) {
        return combine_weighted_scalar(taps, weights, p, out);
    }
    match isa {
        Isa::Scalar => combine_weighted_scalar(taps, weights, p, out),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `supported` confirmed SSE4.1 above, and the tap slices
        // are at least `out.len()` long so every load stays in bounds.
        Isa::Sse41 => unsafe { combine_weighted_sse41(taps, weights, p, out) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `supported` confirmed AVX2 above; same bounds argument.
        Isa::Avx2 => unsafe { combine_weighted_avx2(taps, weights, p, out) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is mandatory on AArch64; same bounds argument.
        Isa::Neon => unsafe { combine_weighted_neon(taps, weights, p, out) },
    }
}

/// The scalar parameters of [`combine_weighted`], grouped so the backend
/// helpers stay under the argument-count lint.
#[derive(Debug, Clone, Copy)]
struct CombineParams {
    round: i32,
    shift: i32,
    post: i32,
    max_val: i32,
}

fn combine_weighted_scalar<const N: usize>(
    taps: &[&[i32]; N],
    weights: &[i32; N],
    p: CombineParams,
    out: &mut [i32],
) {
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = p.round;
        for (&w, tap) in weights.iter().zip(taps.iter()) {
            acc += w * tap[i];
        }
        *o = ((acc >> p.shift) + p.post).clamp(0, p.max_val);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn combine_weighted_sse41<const N: usize>(
    taps: &[&[i32]; N],
    weights: &[i32; N],
    p: CombineParams,
    out: &mut [i32],
) {
    unsafe {
        let count = out.len();
        let sh = _mm_cvtsi32_si128(p.shift);
        let round = _mm_set1_epi32(p.round);
        let post = _mm_set1_epi32(p.post);
        let lo = _mm_setzero_si128();
        let hi = _mm_set1_epi32(p.max_val);
        let mut i = 0usize;
        while i + 4 <= count {
            let mut acc = round;
            for (&w, tap) in weights.iter().zip(taps.iter()) {
                let v = _mm_loadu_si128(tap.as_ptr().add(i).cast());
                acc = _mm_add_epi32(acc, _mm_mullo_epi32(v, _mm_set1_epi32(w)));
            }
            let v = _mm_add_epi32(_mm_sra_epi32(acc, sh), post);
            let v = _mm_min_epi32(_mm_max_epi32(v, lo), hi);
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), v);
            i += 4;
        }
        combine_weighted_tail(taps, weights, p, out, i);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn combine_weighted_avx2<const N: usize>(
    taps: &[&[i32]; N],
    weights: &[i32; N],
    p: CombineParams,
    out: &mut [i32],
) {
    unsafe {
        let count = out.len();
        let sh = _mm_cvtsi32_si128(p.shift);
        let round = _mm256_set1_epi32(p.round);
        let post = _mm256_set1_epi32(p.post);
        let lo = _mm256_setzero_si256();
        let hi = _mm256_set1_epi32(p.max_val);
        let mut i = 0usize;
        while i + 8 <= count {
            let mut acc = round;
            for (&w, tap) in weights.iter().zip(taps.iter()) {
                let v = _mm256_loadu_si256(tap.as_ptr().add(i).cast());
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(v, _mm256_set1_epi32(w)));
            }
            let v = _mm256_add_epi32(_mm256_sra_epi32(acc, sh), post);
            let v = _mm256_min_epi32(_mm256_max_epi32(v, lo), hi);
            _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), v);
            i += 8;
        }
        combine_weighted_tail(taps, weights, p, out, i);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn combine_weighted_neon<const N: usize>(
    taps: &[&[i32]; N],
    weights: &[i32; N],
    p: CombineParams,
    out: &mut [i32],
) {
    unsafe {
        let count = out.len();
        let sh = vdupq_n_s32(-p.shift);
        let round = vdupq_n_s32(p.round);
        let post = vdupq_n_s32(p.post);
        let lo = vdupq_n_s32(0);
        let hi = vdupq_n_s32(p.max_val);
        let mut i = 0usize;
        while i + 4 <= count {
            let mut acc = round;
            for (&w, tap) in weights.iter().zip(taps.iter()) {
                acc = vmlaq_n_s32(acc, vld1q_s32(tap.as_ptr().add(i)), w);
            }
            let v = vaddq_s32(vshlq_s32(acc, sh), post);
            let v = vminq_s32(vmaxq_s32(v, lo), hi);
            vst1q_s32(out.as_mut_ptr().add(i), v);
            i += 4;
        }
        combine_weighted_tail(taps, weights, p, out, i);
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn combine_weighted_tail<const N: usize>(
    taps: &[&[i32]; N],
    weights: &[i32; N],
    p: CombineParams,
    out: &mut [i32],
    from: usize,
) {
    for i in from..out.len() {
        let mut acc = p.round;
        for (&w, tap) in weights.iter().zip(taps.iter()) {
            acc += w * tap[i];
        }
        out[i] = ((acc >> p.shift) + p.post).clamp(0, p.max_val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random `i32` samples in `[-limit, limit]`.
    fn samples(seed: u64, len: usize, limit: i32) -> Vec<i32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let v = (state >> 33) as i64 % (2 * i64::from(limit) + 1);
                (v - i64::from(limit)) as i32
            })
            .collect()
    }

    #[test]
    fn every_backend_matches_scalar_filter_taps() {
        let src: Vec<Vec<i32>> = (0..8).map(|t| samples(t + 7, 200, 40_000)).collect();
        let taps8: [&[i32]; 8] = std::array::from_fn(|t| src[t].as_slice());
        let taps4: [&[i32]; 4] = std::array::from_fn(|t| src[t].as_slice());
        let c8 = [-1, 4, -11, 40, 40, -11, 4, -1];
        let c4 = [-2, 58, 10, -2];
        // Widths that exercise the 8-, 4- and 1-wide code paths.
        for len in [1usize, 2, 3, 4, 5, 7, 8, 12, 16, 31, 64, 200] {
            for shift in [0i32, 2, 4, 6] {
                let mut reference = vec![0i32; len];
                filter_taps(Isa::Scalar, &taps8, &c8, shift, &mut reference);
                let mut reference4 = vec![0i32; len];
                filter_taps(Isa::Scalar, &taps4, &c4, shift, &mut reference4);
                for isa in available_isas() {
                    let mut got = vec![0i32; len];
                    filter_taps(isa, &taps8, &c8, shift, &mut got);
                    assert_eq!(got, reference, "{isa:?} 8-tap len={len} shift={shift}");
                    let mut got4 = vec![0i32; len];
                    filter_taps(isa, &taps4, &c4, shift, &mut got4);
                    assert_eq!(got4, reference4, "{isa:?} 4-tap len={len} shift={shift}");
                }
            }
        }
    }

    #[test]
    fn every_backend_matches_scalar_combine_weighted() {
        let a = samples(3, 200, 30_000);
        let b = samples(11, 200, 30_000);
        for len in [1usize, 3, 4, 5, 8, 13, 16, 64, 200] {
            for (weights, round, shift, post) in [
                ([1, 1], 32, 6, 0),
                ([1, 1], 64, 7, 0),
                ([64, -12], 1 << 12, 13, 0),
                ([-5, 3], 0, 5, 17),
            ] {
                let taps2: [&[i32]; 2] = [&a, &b];
                let taps1: [&[i32]; 1] = [&a];
                let w1 = [weights[0]];
                let mut reference = vec![0i32; len];
                combine_weighted(
                    Isa::Scalar,
                    &taps2,
                    &weights,
                    round,
                    shift,
                    post,
                    255,
                    &mut reference,
                );
                let mut reference1 = vec![0i32; len];
                combine_weighted(
                    Isa::Scalar,
                    &taps1,
                    &w1,
                    round,
                    shift,
                    post,
                    1023,
                    &mut reference1,
                );
                for isa in available_isas() {
                    let mut got = vec![0i32; len];
                    combine_weighted(isa, &taps2, &weights, round, shift, post, 255, &mut got);
                    assert_eq!(got, reference, "{isa:?} bi len={len}");
                    let mut got1 = vec![0i32; len];
                    combine_weighted(isa, &taps1, &w1, round, shift, post, 1023, &mut got1);
                    assert_eq!(got1, reference1, "{isa:?} uni len={len}");
                }
            }
        }
    }

    /// An unsupported backend must not be executed; it degrades to the
    /// scalar path rather than issuing an illegal instruction.
    #[test]
    fn unsupported_backend_falls_back_to_scalar() {
        let a = vec![7i32; 16];
        let taps: [&[i32]; 1] = [&a];
        let mut out = vec![0i32; 16];
        for isa in [
            Isa::Scalar,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2,
            #[cfg(target_arch = "x86_64")]
            Isa::Sse41,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon,
        ] {
            filter_taps(isa, &taps, &[64], 6, &mut out);
            assert!(out.iter().all(|&v| v == 7));
        }
    }
}
