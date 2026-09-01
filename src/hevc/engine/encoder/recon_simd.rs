//! SIMD kernels for the two scalar stages of encoder-side reconstruction.
//!
//! [`crate::hevc::engine::encoder::recon`] spends nearly all of its time in two
//! per-sample loops, neither of which is one of the decoder's already-vectorized
//! in-loop filter kernels:
//!
//! - the §8.6.6 reconstruction loop `recSamples = Clip1( predSamples +
//!   resSamples )`, run over every plane of every prediction partition, and
//! - the encoder-side §8.7.3 SAO parameter search, which classifies every
//!   sample of every CTB into a §8.7.3.2 edge category once per candidate
//!   edge-offset class — four full passes over the picture before the filter
//!   itself runs.
//!
//! The search's band-offset half ([`band_offset_row`]) is a dispatch site here
//! too, but every arm of it resolves to the scalar reference: its 32-way
//! scatter is not something SSE4.1, AVX2 or NEON can express, and both vector
//! shapes that leaves were written and measured below parity. That kernel's
//! documentation carries the numbers.
//!
//! Both are dispatched here through cached runtime CPU feature detection
//! ([`isa`]) to an SSE4.1 or AVX2 implementation on `x86_64`, a NEON
//! implementation on `aarch64`, or the portable scalar reference everywhere
//! else, exactly as [`crate::hevc::engine::encoder::rdcost`] does for the
//! distortion metrics. The crate-wide [`crate::simd::set_override`] is
//! consulted ahead of the cached probe, so this module appears in
//! [`crate::simd::active_by_site`] as `hevc_recon` and the benchmark harness's
//! override guard covers it.
//!
//! Every vectorized path is bit-identical to the scalar one: the reconstruction
//! kernel's intermediates all fit in 16 bits at 8-bit depth, and the SAO search
//! accumulates in 32-bit lanes over runs no longer than one CTB row, so no
//! vector path can round or saturate where the scalar one does not. Enabling
//! SIMD therefore changes only how fast a picture reconstructs, never the
//! reconstruction itself or which SAO parameters the encoder signals.

use std::sync::atomic::{AtomicU8, Ordering};

/// The bit depth the reconstruction kernel clips to, matching the encoder's
/// fixed 8-bit geometry.
const BIT_DEPTH_MAX: i32 = 255;

/// The instruction set the reconstruction kernels in this module are running on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Isa {
    /// Portable fallback used on targets without a vectorized implementation.
    Scalar,
    /// x86_64 SSE4.1 (128-bit).
    #[cfg(target_arch = "x86_64")]
    Sse41,
    /// x86_64 AVX2 (256-bit).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// aarch64 Advanced SIMD (128-bit).
    #[cfg(target_arch = "aarch64")]
    Neon,
}

const ISA_UNDETECTED: u8 = 0;
const ISA_SCALAR: u8 = 1;
#[cfg(target_arch = "x86_64")]
const ISA_SSE41: u8 = 2;
#[cfg(target_arch = "x86_64")]
const ISA_AVX2: u8 = 3;
#[cfg(target_arch = "aarch64")]
const ISA_NEON: u8 = 4;

static DETECTED_ISA: AtomicU8 = AtomicU8::new(ISA_UNDETECTED);

fn detect_isa() -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return ISA_AVX2;
        }
        if is_x86_feature_detected!("sse4.1") {
            return ISA_SSE41;
        }
    }
    // NEON is part of the aarch64 baseline, so no runtime probe is needed there.
    #[cfg(target_arch = "aarch64")]
    {
        return ISA_NEON;
    }
    #[allow(unreachable_code)]
    ISA_SCALAR
}

/// Maps the crate-wide SIMD override, if any, onto this module's ISA codes.
#[inline]
fn overridden_isa_code() -> Option<u8> {
    use crate::simd::SimdIsa;
    Some(match crate::simd::override_isa()? {
        SimdIsa::Scalar => ISA_SCALAR,
        #[cfg(target_arch = "x86_64")]
        SimdIsa::Sse41 => ISA_SSE41,
        #[cfg(target_arch = "x86_64")]
        SimdIsa::Avx2 => ISA_AVX2,
        #[cfg(target_arch = "aarch64")]
        SimdIsa::Neon => ISA_NEON,
        #[allow(unreachable_patterns)]
        _ => ISA_SCALAR,
    })
}

/// The cached ISA code, with any [`crate::simd::set_override`] override taking
/// precedence so it applies even after detection has resolved.
#[inline]
fn isa_code() -> u8 {
    if let Some(code) = overridden_isa_code() {
        return code;
    }
    let cached = DETECTED_ISA.load(Ordering::Relaxed);
    if cached != ISA_UNDETECTED {
        return cached;
    }
    let detected = detect_isa();
    DETECTED_ISA.store(detected, Ordering::Relaxed);
    detected
}

/// Returns the instruction set the reconstruction kernels will use on this machine.
pub(crate) fn isa() -> Isa {
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 => Isa::Avx2,
        #[cfg(target_arch = "x86_64")]
        ISA_SSE41 => Isa::Sse41,
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => Isa::Neon,
        _ => Isa::Scalar,
    }
}

/// One row of §8.6.6 reconstruction: `dst = Clip1( pred + (src − pred) )`.
///
/// `pred` is the prediction the encoder formed for this run of samples and
/// `src` the co-located source run, so `src − pred` is the residual this
/// lossless writer codes. The subtraction and the add back are kept as written
/// rather than folded away, because a quantizing writer would round-trip the
/// residual between them and only the middle of this expression would change.
///
/// # Panics
/// Panics unless all three slices have the same length.
pub(crate) fn reconstruct_row(dst: &mut [i32], src: &[u8], pred: &[u8]) {
    assert_eq!(dst.len(), src.len(), "destination and source rows differ");
    assert_eq!(
        dst.len(),
        pred.len(),
        "destination and prediction rows differ"
    );
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 => {
            // SAFETY: the lengths agree, and this arm is only reachable after
            // `is_x86_feature_detected!("avx2")` or an override clamped to an
            // available instruction set.
            unsafe { x86::reconstruct_row_avx2(dst, src, pred) }
        }
        #[cfg(target_arch = "x86_64")]
        ISA_SSE41 => {
            // SAFETY: as above, with SSE4.1 detected.
            unsafe { x86::reconstruct_row_sse41(dst, src, pred) }
        }
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => {
            // SAFETY: the lengths agree, and NEON is part of the aarch64
            // baseline this code is compiled for.
            unsafe { neon::reconstruct_row(dst, src, pred) }
        }
        _ => reconstruct_row_scalar(dst, src, pred),
    }
}

/// The portable reference for [`reconstruct_row`].
pub(crate) fn reconstruct_row_scalar(dst: &mut [i32], src: &[u8], pred: &[u8]) {
    for ((d, &s), &p) in dst.iter_mut().zip(src.iter()).zip(pred.iter()) {
        let predicted = i32::from(p);
        let residual = i32::from(s) - predicted;
        *d = (predicted + residual).clamp(0, BIT_DEPTH_MAX);
    }
}

/// Per-category error sums and sample counts for one run of the §8.7.3.2
/// edge-offset classification, indexed by `SaoOffsetVal` category `1..=4`.
///
/// Index 0 is unused and always zero, matching the `SaoOffsetVal` layout the
/// caller signals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EdgeStats {
    /// Summed `source − reconstruction` error per category.
    pub sums: [i64; 5],
    /// Number of samples per category.
    pub counts: [i64; 5],
}

/// Classifies one horizontal run of samples into §8.7.3.2 edge categories and
/// accumulates the source-versus-reconstruction error of each into `stats`.
///
/// `here` is the run of reconstructed samples, `a` and `b` the two co-located
/// neighbour runs the edge-offset class selects, and `src` the co-located
/// source samples. The caller is responsible for trimming the run to the
/// samples whose neighbours lie inside the plane, which is why no bounds
/// handling appears here: `edgeIdx == 2` (category 0) contributes nothing and
/// is the only sample class this kernel drops.
///
/// # Panics
/// Panics unless all four slices have the same length.
pub(crate) fn edge_offset_row(
    here: &[i32],
    a: &[i32],
    b: &[i32],
    src: &[u8],
    stats: &mut EdgeStats,
) {
    assert_eq!(here.len(), a.len(), "run and first neighbour differ");
    assert_eq!(here.len(), b.len(), "run and second neighbour differ");
    assert_eq!(here.len(), src.len(), "run and source differ");
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 => {
            // SAFETY: the lengths agree, and this arm is only reachable with
            // AVX2 available.
            unsafe { x86::edge_offset_row_avx2(here, a, b, src, stats) }
        }
        #[cfg(target_arch = "x86_64")]
        ISA_SSE41 => {
            // SAFETY: as above, with SSE4.1 detected.
            unsafe { x86::edge_offset_row_sse41(here, a, b, src, stats) }
        }
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => {
            // SAFETY: the lengths agree, and NEON is part of the aarch64
            // baseline this code is compiled for.
            unsafe { neon::edge_offset_row(here, a, b, src, stats) }
        }
        _ => edge_offset_row_scalar(here, a, b, src, stats),
    }
}

/// §8.7.3.2 `edgeIdx` for one sample, remapped to its `SaoOffsetVal` category
/// index (1..=4), or `None` for category 0.
#[inline]
fn category(here: i32, a: i32, b: i32) -> Option<usize> {
    let edge_idx = 2 + (here - a).signum() + (here - b).signum();
    // §8.7.3.2: edgeIdx 0/1/2/3/4 maps to categories 1, 2, 0, 3, 4.
    match edge_idx {
        0 => Some(1),
        1 => Some(2),
        3 => Some(3),
        4 => Some(4),
        _ => None,
    }
}

/// The portable reference for [`edge_offset_row`].
pub(crate) fn edge_offset_row_scalar(
    here: &[i32],
    a: &[i32],
    b: &[i32],
    src: &[u8],
    stats: &mut EdgeStats,
) {
    for i in 0..here.len() {
        let Some(category) = category(here[i], a[i], b[i]) else {
            continue;
        };
        stats.sums[category] += i64::from(src[i]) - i64::from(here[i]);
        stats.counts[category] += 1;
    }
}

/// Per-band error sums and sample counts for one run of the §8.7.3.2
/// band-offset classification, indexed by the band index
/// `sample >> (bitDepth − 5)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BandStats {
    /// Summed `source − reconstruction` error per band.
    pub sums: [i64; 32],
    /// Number of samples per band.
    pub counts: [i64; 32],
}

/// §8.7.3.2 `bandShift`: the reconstruction's top five bits index the 32 bands,
/// which at this module's fixed 8-bit geometry is a shift of three.
const BAND_SHIFT: i32 = 3;

/// Classifies one horizontal run of reconstructed samples into §8.7.3.2 bands
/// and accumulates the source-versus-reconstruction error of each into `stats`.
///
/// `here` is the run of reconstructed samples and `src` the co-located source
/// samples. Every sample lands in exactly one of the 32 bands, so unlike
/// [`edge_offset_row`] nothing is dropped and the counts sum to the run length.
///
/// The band index clamps the reconstruction into the 8-bit range before
/// shifting, but the error is taken against the unclamped sample.
///
/// # This is a dispatch site with no vector arm, and that is the measured result
///
/// Unlike [`edge_offset_row`], every arm below resolves to the scalar
/// reference. The band search is not a 5-way classification like the edge
/// search is — it is a 32-way *scatter*, and none of the instruction sets this
/// module targets can express one. SSE4.1, AVX2 and NEON have no scatter, and
/// masking 32 accumulators per vector costs far more than the four to eight
/// dependent read-modify-writes it would replace. What is left to vectorize is
/// only the classification in front of the scatter — the clamp, the shift and
/// the widened subtraction — which is a minority of the work.
///
/// Both shapes that leaves were written and measured on an Apple Silicon host,
/// against this scalar reference over L1-resident runs of 16, 64, 256 and 1024
/// samples, best of interleaved rounds, and measured a second time from a
/// standalone harness to check the first:
///
/// | NEON shape | ratio to scalar |
/// |---|---|
/// | classify into staging buffers, then scatter the buffers | 0.42-1.30x |
/// | classify and scatter straight out of the vector lanes | 0.44-1.24x |
///
/// Neither shape separates from the scalar reference. Both straddle 1.00x by
/// less than the spread between repeats of the same measurement on this
/// contended host — the ranges above are run-to-run noise around parity, not a
/// speedup at one run length and a slowdown at another, and repeating the
/// 64-sample point alone moved each shape across most of its range. That is
/// the same answer the whole-picture `hevc_encode_640x352_reconstruct` group
/// gave, where the NEON arm did not improve.
///
/// The staging round trip in particular costs more than the classification it
/// vectorizes, and extracting the lanes only replaces four stores with four
/// lane reads in front of the same four dependent read-modify-writes. Neither
/// was worth landing, so this dispatches to scalar the way
/// [`crate::hevc::engine::simd::combine_weighted`] does on the instruction sets
/// where its kernel measured below parity.
///
/// The site is kept rather than the call inlined into
/// [`crate::hevc::engine::encoder::recon`] so the band search stays a named
/// `hevc_recon` dispatch point: the reference is exercised under every
/// `crate::simd::set_override` pin by the tests below, so a future kernel — an
/// AVX-512 one, where `vpconflictd` and a real scatter change the arithmetic
/// above — has a place to go and a bit-exactness harness already pointed at it.
/// x86_64 is untimed here and tracked separately.
pub(crate) fn band_offset_row(here: &[i32], src: &[u8], stats: &mut BandStats) {
    assert_eq!(here.len(), src.len(), "run and source differ");
    band_offset_row_scalar(here, src, stats)
}

/// The portable reference for [`band_offset_row`], and on every instruction set
/// this module targets, the implementation it dispatches to.
pub(crate) fn band_offset_row_scalar(here: &[i32], src: &[u8], stats: &mut BandStats) {
    for i in 0..here.len() {
        let recon = here[i];
        let band = (recon.clamp(0, BIT_DEPTH_MAX) >> BAND_SHIFT) as usize;
        stats.sums[band] += i64::from(src[i]) - i64::from(recon);
        stats.counts[band] += 1;
    }
}

/// The `edgeIdx` values that map to categories 1, 2, 3 and 4, in that order.
const EDGE_IDX_BY_CATEGORY: [i32; 4] = [0, 1, 3, 4];

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{BIT_DEPTH_MAX, EDGE_IDX_BY_CATEGORY, EdgeStats};
    use std::arch::x86_64::*;

    /// Sum of the four `i32` lanes.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn hsum_epi32(v: __m128i) -> i32 {
        let hi = _mm_unpackhi_epi64(v, v);
        let s = _mm_add_epi32(v, hi);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_01_01_01));
        _mm_cvtsi128_si32(s)
    }

    /// `Clip1(pred + (src − pred))` over eight samples held as `i16` lanes.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn reconstruct8_sse41(dst: *mut i32, src: *const u8, pred: *const u8) {
        unsafe {
            let p = _mm_cvtepu8_epi16(_mm_loadl_epi64(pred.cast()));
            let s = _mm_cvtepu8_epi16(_mm_loadl_epi64(src.cast()));
            let residual = _mm_sub_epi16(s, p);
            let sum = _mm_add_epi16(p, residual);
            let clipped = _mm_min_epi16(
                _mm_max_epi16(sum, _mm_setzero_si128()),
                _mm_set1_epi16(BIT_DEPTH_MAX as i16),
            );
            _mm_storeu_si128(dst.cast(), _mm_cvtepi16_epi32(clipped));
            _mm_storeu_si128(
                dst.add(4).cast(),
                _mm_cvtepi16_epi32(_mm_srli_si128(clipped, 8)),
            );
        }
    }

    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn reconstruct_row_sse41(dst: &mut [i32], src: &[u8], pred: &[u8]) {
        unsafe {
            let n = dst.len();
            let mut i = 0;
            while i + 8 <= n {
                reconstruct8_sse41(
                    dst.as_mut_ptr().add(i),
                    src.as_ptr().add(i),
                    pred.as_ptr().add(i),
                );
                i += 8;
            }
            super::reconstruct_row_scalar(&mut dst[i..], &src[i..], &pred[i..]);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn reconstruct_row_avx2(dst: &mut [i32], src: &[u8], pred: &[u8]) {
        unsafe {
            let n = dst.len();
            let zero = _mm256_setzero_si256();
            let max = _mm256_set1_epi16(BIT_DEPTH_MAX as i16);
            let mut i = 0;
            while i + 16 <= n {
                let p = _mm256_cvtepu8_epi16(_mm_loadu_si128(pred.as_ptr().add(i).cast()));
                let s = _mm256_cvtepu8_epi16(_mm_loadu_si128(src.as_ptr().add(i).cast()));
                let residual = _mm256_sub_epi16(s, p);
                let sum = _mm256_add_epi16(p, residual);
                let clipped = _mm256_min_epi16(_mm256_max_epi16(sum, zero), max);
                let out = dst.as_mut_ptr().add(i);
                _mm256_storeu_si256(
                    out.cast(),
                    _mm256_cvtepi16_epi32(_mm256_castsi256_si128(clipped)),
                );
                _mm256_storeu_si256(
                    out.add(8).cast(),
                    _mm256_cvtepi16_epi32(_mm256_extracti128_si256(clipped, 1)),
                );
                i += 16;
            }
            while i + 8 <= n {
                reconstruct8_sse41(
                    dst.as_mut_ptr().add(i),
                    src.as_ptr().add(i),
                    pred.as_ptr().add(i),
                );
                i += 8;
            }
            super::reconstruct_row_scalar(&mut dst[i..], &src[i..], &pred[i..]);
        }
    }

    /// `(here − other).signum()` over four `i32` lanes.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn signum_sse41(here: __m128i, other: __m128i) -> __m128i {
        // A comparison yields all-ones (−1) where it holds, so `lt − gt` is
        // −1, 0 or +1 exactly as `i32::signum` is.
        _mm_sub_epi32(_mm_cmplt_epi32(here, other), _mm_cmpgt_epi32(here, other))
    }

    /// Loads four `u8` samples as `i32` lanes.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn load4_u8_epi32(src: *const u8) -> __m128i {
        unsafe {
            let packed = src.cast::<u32>().read_unaligned();
            _mm_cvtepu8_epi32(_mm_cvtsi32_si128(packed as i32))
        }
    }

    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn edge_offset_row_sse41(
        here: &[i32],
        a: &[i32],
        b: &[i32],
        src: &[u8],
        stats: &mut EdgeStats,
    ) {
        unsafe {
            let n = here.len();
            let two = _mm_set1_epi32(2);
            let mut sums = [_mm_setzero_si128(); 4];
            let mut counts = [_mm_setzero_si128(); 4];
            let mut i = 0;
            while i + 4 <= n {
                let vh = _mm_loadu_si128(here.as_ptr().add(i).cast());
                let va = _mm_loadu_si128(a.as_ptr().add(i).cast());
                let vb = _mm_loadu_si128(b.as_ptr().add(i).cast());
                let edge = _mm_add_epi32(
                    two,
                    _mm_add_epi32(signum_sse41(vh, va), signum_sse41(vh, vb)),
                );
                let error = _mm_sub_epi32(load4_u8_epi32(src.as_ptr().add(i)), vh);
                for c in 0..4 {
                    let hit = _mm_cmpeq_epi32(edge, _mm_set1_epi32(EDGE_IDX_BY_CATEGORY[c]));
                    sums[c] = _mm_add_epi32(sums[c], _mm_and_si128(error, hit));
                    // `hit` is −1 where it matched, so subtracting counts it.
                    counts[c] = _mm_sub_epi32(counts[c], hit);
                }
                i += 4;
            }
            for c in 0..4 {
                stats.sums[c + 1] += i64::from(hsum_epi32(sums[c]));
                stats.counts[c + 1] += i64::from(hsum_epi32(counts[c]));
            }
            super::edge_offset_row_scalar(&here[i..], &a[i..], &b[i..], &src[i..], stats);
        }
    }

    /// `(here − other).signum()` over eight `i32` lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn signum_avx2(here: __m256i, other: __m256i) -> __m256i {
        _mm256_sub_epi32(
            _mm256_cmpgt_epi32(other, here),
            _mm256_cmpgt_epi32(here, other),
        )
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn edge_offset_row_avx2(
        here: &[i32],
        a: &[i32],
        b: &[i32],
        src: &[u8],
        stats: &mut EdgeStats,
    ) {
        unsafe {
            let n = here.len();
            let two = _mm256_set1_epi32(2);
            let mut sums = [_mm256_setzero_si256(); 4];
            let mut counts = [_mm256_setzero_si256(); 4];
            let mut i = 0;
            while i + 8 <= n {
                let vh = _mm256_loadu_si256(here.as_ptr().add(i).cast());
                let va = _mm256_loadu_si256(a.as_ptr().add(i).cast());
                let vb = _mm256_loadu_si256(b.as_ptr().add(i).cast());
                let edge = _mm256_add_epi32(
                    two,
                    _mm256_add_epi32(signum_avx2(vh, va), signum_avx2(vh, vb)),
                );
                let packed = src.as_ptr().add(i).cast::<u64>().read_unaligned();
                let vsrc = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(packed as i64));
                let error = _mm256_sub_epi32(vsrc, vh);
                for c in 0..4 {
                    let hit = _mm256_cmpeq_epi32(edge, _mm256_set1_epi32(EDGE_IDX_BY_CATEGORY[c]));
                    sums[c] = _mm256_add_epi32(sums[c], _mm256_and_si256(error, hit));
                    counts[c] = _mm256_sub_epi32(counts[c], hit);
                }
                i += 8;
            }
            for c in 0..4 {
                let sum = _mm_add_epi32(
                    _mm256_castsi256_si128(sums[c]),
                    _mm256_extracti128_si256(sums[c], 1),
                );
                let count = _mm_add_epi32(
                    _mm256_castsi256_si128(counts[c]),
                    _mm256_extracti128_si256(counts[c], 1),
                );
                stats.sums[c + 1] += i64::from(hsum_epi32(sum));
                stats.counts[c + 1] += i64::from(hsum_epi32(count));
            }
            super::edge_offset_row_scalar(&here[i..], &a[i..], &b[i..], &src[i..], stats);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{BIT_DEPTH_MAX, EDGE_IDX_BY_CATEGORY, EdgeStats};
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn reconstruct_row(dst: &mut [i32], src: &[u8], pred: &[u8]) {
        unsafe {
            let n = dst.len();
            let zero = vdupq_n_s16(0);
            let max = vdupq_n_s16(BIT_DEPTH_MAX as i16);
            let mut i = 0;
            while i + 16 <= n {
                let p = vld1q_u8(pred.as_ptr().add(i));
                let s = vld1q_u8(src.as_ptr().add(i));
                let out = dst.as_mut_ptr().add(i);
                let halves = [
                    (vget_low_u8(p), vget_low_u8(s)),
                    (vget_high_u8(p), vget_high_u8(s)),
                ];
                for (half, (p8, s8)) in halves.into_iter().enumerate() {
                    let pv = vreinterpretq_s16_u16(vmovl_u8(p8));
                    let sv = vreinterpretq_s16_u16(vmovl_u8(s8));
                    let residual = vsubq_s16(sv, pv);
                    let sum = vaddq_s16(pv, residual);
                    let clipped = vminq_s16(vmaxq_s16(sum, zero), max);
                    let base = out.add(half * 8);
                    vst1q_s32(base, vmovl_s16(vget_low_s16(clipped)));
                    vst1q_s32(base.add(4), vmovl_high_s16(clipped));
                }
                i += 16;
            }
            super::reconstruct_row_scalar(&mut dst[i..], &src[i..], &pred[i..]);
        }
    }

    /// `(here − other).signum()` over four `i32` lanes.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn signum(here: int32x4_t, other: int32x4_t) -> int32x4_t {
        // A comparison yields all-ones (−1) where it holds, so `lt − gt` is
        // −1, 0 or +1 exactly as `i32::signum` is.
        vsubq_s32(
            vreinterpretq_s32_u32(vcltq_s32(here, other)),
            vreinterpretq_s32_u32(vcgtq_s32(here, other)),
        )
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn edge_offset_row(
        here: &[i32],
        a: &[i32],
        b: &[i32],
        src: &[u8],
        stats: &mut EdgeStats,
    ) {
        unsafe {
            let n = here.len();
            let two = vdupq_n_s32(2);
            let mut sums = [vdupq_n_s32(0); 4];
            let mut counts = [vdupq_n_s32(0); 4];
            let mut i = 0;
            while i + 4 <= n {
                let vh = vld1q_s32(here.as_ptr().add(i));
                let va = vld1q_s32(a.as_ptr().add(i));
                let vb = vld1q_s32(b.as_ptr().add(i));
                let edge = vaddq_s32(two, vaddq_s32(signum(vh, va), signum(vh, vb)));
                let packed = src.as_ptr().add(i).cast::<u32>().read_unaligned();
                let bytes = vreinterpret_u8_u32(vdup_n_u32(packed));
                let vsrc = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(bytes))));
                let error = vsubq_s32(vsrc, vh);
                for c in 0..4 {
                    let hit = vceqq_s32(edge, vdupq_n_s32(EDGE_IDX_BY_CATEGORY[c]));
                    let mask = vreinterpretq_s32_u32(hit);
                    sums[c] = vaddq_s32(sums[c], vandq_s32(error, mask));
                    // `mask` is −1 where it matched, so subtracting counts it.
                    counts[c] = vsubq_s32(counts[c], mask);
                }
                i += 4;
            }
            for c in 0..4 {
                stats.sums[c + 1] += i64::from(vaddvq_s32(sums[c]));
                stats.counts[c + 1] += i64::from(vaddvq_s32(counts[c]));
            }
            super::edge_offset_row_scalar(&here[i..], &a[i..], &b[i..], &src[i..], stats);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{self, SimdIsa};

    /// A small xorshift so the fixtures are deterministic without a dependency.
    struct Rng(u64);

    impl Rng {
        fn next_u8(&mut self) -> u8 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 33) as u8
        }
    }

    fn bytes(seed: u64, n: usize) -> Vec<u8> {
        let mut rng = Rng(seed);
        (0..n).map(|_| rng.next_u8()).collect()
    }

    /// Reconstruction inputs are 8-bit samples, but the SAO search reads the
    /// reconstruction as `i32`, so the fixture keeps that width while staying
    /// in the range a reconstructed picture can actually hold.
    fn samples(seed: u64, n: usize) -> Vec<i32> {
        bytes(seed, n).into_iter().map(i32::from).collect()
    }

    /// Every run length that exercises a full vector body, a partial one, and
    /// the scalar tail of each backend, plus lengths shorter than one vector.
    const RUNS: &[usize] = &[0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65];

    #[test]
    fn reconstruction_matches_the_scalar_reference_on_every_instruction_set() {
        let _guard = simd::test_lock();
        for &n in RUNS {
            let src = bytes(0xdead_beef_cafe_f00d, n);
            let pred = bytes(0x0123_4567_89ab_cdef, n);
            let mut expected = vec![0i32; n];
            reconstruct_row_scalar(&mut expected, &src, &pred);
            for isa in simd::available() {
                simd::set_override(Some(isa));
                let mut got = vec![0i32; n];
                reconstruct_row(&mut got, &src, &pred);
                assert_eq!(
                    got,
                    expected,
                    "{} reconstruction of {n} samples",
                    isa.name()
                );
            }
        }
        simd::set_override(None);
    }

    #[test]
    fn the_reconstruction_of_an_8_bit_source_is_the_source() {
        // `pred + (src − pred)` is `src`, and `src` is already inside the
        // 8-bit range, so the clip never fires. This is the property the
        // lossless PCM writer's reconstruction depends on, and it is checked
        // here against the dispatched kernel rather than the reference.
        let _guard = simd::test_lock();
        let src = bytes(0x5151_2626_3737_4848, 67);
        let pred = bytes(0x9999_1111_2222_3333, 67);
        for isa in simd::available() {
            simd::set_override(Some(isa));
            let mut got = vec![0i32; src.len()];
            reconstruct_row(&mut got, &src, &pred);
            let expected: Vec<i32> = src.iter().map(|&s| i32::from(s)).collect();
            assert_eq!(got, expected, "{}", isa.name());
        }
        simd::set_override(None);
    }

    #[test]
    fn the_edge_offset_search_matches_the_scalar_reference_on_every_instruction_set() {
        let _guard = simd::test_lock();
        for &n in RUNS {
            let here = samples(0x1111_2222_3333_4444, n);
            // Neighbours drawn from an overlapping window of the same fixture
            // so that all five `edgeIdx` values, including the skipped
            // category 0, occur rather than only the extremes.
            let a = samples(0x1111_2222_3333_4445, n);
            let b = samples(0x1111_2222_3333_4446, n);
            let src = bytes(0xfeed_face_dead_10cc, n);
            let mut expected = EdgeStats::default();
            edge_offset_row_scalar(&here, &a, &b, &src, &mut expected);
            for isa in simd::available() {
                simd::set_override(Some(isa));
                let mut got = EdgeStats::default();
                edge_offset_row(&here, &a, &b, &src, &mut got);
                assert_eq!(got, expected, "{} edge search over {n} samples", isa.name());
            }
        }
        simd::set_override(None);
    }

    #[test]
    fn the_band_offset_search_matches_the_scalar_reference_on_every_instruction_set() {
        let _guard = simd::test_lock();
        for &n in RUNS {
            // A reconstruction spanning the full 8-bit range so every one of
            // the 32 bands is reachable, plus samples outside it so the
            // kernel's clamp is exercised alongside the unclamped error.
            let mut here = samples(0x7777_8888_9999_aaaa, n);
            for (i, h) in here.iter_mut().enumerate() {
                match i % 8 {
                    0 => *h = -7,
                    1 => *h = 262,
                    _ => {}
                }
            }
            let src = bytes(0xc0ff_ee00_1234_5678, n);
            let mut expected = BandStats::default();
            band_offset_row_scalar(&here, &src, &mut expected);
            assert_eq!(
                expected.counts.iter().sum::<i64>(),
                n as i64,
                "every sample lands in exactly one band"
            );
            for isa in simd::available() {
                simd::set_override(Some(isa));
                let mut got = BandStats::default();
                band_offset_row(&here, &src, &mut got);
                assert_eq!(got, expected, "{} band search over {n} samples", isa.name());
            }
        }
        simd::set_override(None);
    }

    #[test]
    fn the_band_search_accumulates_into_existing_stats() {
        // `band_stats` gathers a whole CTB by calling the kernel once per row,
        // so a second call has to add to the first rather than replace it.
        let _guard = simd::test_lock();
        let here = samples(0x3131_4141_5151_6161, 37);
        let src = bytes(0x2020_3030_4040_5050, 37);
        let mut expected = BandStats::default();
        band_offset_row_scalar(&here, &src, &mut expected);
        band_offset_row_scalar(&here, &src, &mut expected);
        for isa in simd::available() {
            simd::set_override(Some(isa));
            let mut got = BandStats::default();
            band_offset_row(&here, &src, &mut got);
            band_offset_row(&here, &src, &mut got);
            assert_eq!(got, expected, "{}", isa.name());
        }
        simd::set_override(None);
    }

    #[test]
    fn every_band_is_reachable_and_counted_once() {
        // One sample per band, at the low edge of each, so the §8.7.3.2
        // `sample >> (bitDepth - 5)` mapping is exercised end to end.
        let here: Vec<i32> = (0..32).map(|b| b * 8).collect();
        let src = vec![0u8; 32];
        let mut stats = BandStats::default();
        band_offset_row_scalar(&here, &src, &mut stats);
        assert_eq!(stats.counts, [1i64; 32]);
        let expected: Vec<i64> = (0..32).map(|b| -(b as i64) * 8).collect();
        assert_eq!(stats.sums.to_vec(), expected);
    }

    #[test]
    fn a_sample_outside_the_8_bit_range_still_lands_in_an_end_band() {
        // The reconstruction is clipped before it reaches the search, but the
        // kernel clamps anyway so an out-of-range sample cannot index past the
        // 32-band table. The error stays unclamped, matching the reference.
        let here = [-9i32, 300];
        let src = [4u8, 4];
        let mut stats = BandStats::default();
        band_offset_row_scalar(&here, &src, &mut stats);
        assert_eq!(stats.counts[0], 1);
        assert_eq!(stats.counts[31], 1);
        assert_eq!(stats.sums[0], 13);
        assert_eq!(stats.sums[31], -296);
    }

    #[test]
    fn every_edge_category_is_reachable_and_counted_once() {
        // One sample per `edgeIdx` 0..=4, in order, so every arm of the
        // §8.7.3.2 mapping is exercised including the dropped category 0.
        let here = [10, 10, 10, 10, 10];
        let a = [20, 20, 10, 10, 5];
        let b = [20, 10, 10, 5, 5];
        let src = [12u8, 13, 14, 15, 16];
        let mut stats = EdgeStats::default();
        edge_offset_row_scalar(&here, &a, &b, &src, &mut stats);
        // edgeIdx 0, 1, 2, 3, 4 => categories 1, 2, (dropped), 3, 4.
        assert_eq!(stats.counts, [0, 1, 1, 1, 1]);
        assert_eq!(stats.sums, [0, 2, 3, 5, 6]);
    }

    #[test]
    fn the_detected_instruction_set_is_the_best_one_this_machine_supports() {
        let _guard = simd::test_lock();
        simd::set_override(None);
        let expected = match simd::detected() {
            SimdIsa::Scalar => Isa::Scalar,
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Sse41 => Isa::Sse41,
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx2 => Isa::Avx2,
            #[cfg(target_arch = "aarch64")]
            SimdIsa::Neon => Isa::Neon,
            #[allow(unreachable_patterns)]
            _ => Isa::Scalar,
        };
        assert_eq!(isa(), expected);
    }

    /// Each vectorized implementation is checked directly rather than only
    /// through dispatch, so a machine that selects AVX2 still validates the
    /// SSE4.1 code as well.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_supported_x86_implementation_matches_the_scalar_reference() {
        for &n in RUNS {
            let src = bytes(0xabcd_ef01_2345_6789, n);
            let pred = bytes(0x1357_9bdf_0246_8ace, n);
            let here = samples(0x2222_3333_4444_5555, n);
            let a = samples(0x2222_3333_4444_5556, n);
            let b = samples(0x2222_3333_4444_5557, n);
            let mut recon_expected = vec![0i32; n];
            reconstruct_row_scalar(&mut recon_expected, &src, &pred);
            let mut sao_expected = EdgeStats::default();
            edge_offset_row_scalar(&here, &a, &b, &src, &mut sao_expected);
            if is_x86_feature_detected!("sse4.1") {
                let mut got = vec![0i32; n];
                let mut stats = EdgeStats::default();
                // SAFETY: the feature was just detected and every slice is `n` long.
                unsafe {
                    x86::reconstruct_row_sse41(&mut got, &src, &pred);
                    x86::edge_offset_row_sse41(&here, &a, &b, &src, &mut stats);
                }
                assert_eq!(got, recon_expected, "SSE4.1 reconstruction of {n}");
                assert_eq!(stats, sao_expected, "SSE4.1 edge search over {n}");
            }
            if is_x86_feature_detected!("avx2") {
                let mut got = vec![0i32; n];
                let mut stats = EdgeStats::default();
                // SAFETY: the feature was just detected and every slice is `n` long.
                unsafe {
                    x86::reconstruct_row_avx2(&mut got, &src, &pred);
                    x86::edge_offset_row_avx2(&here, &a, &b, &src, &mut stats);
                }
                assert_eq!(got, recon_expected, "AVX2 reconstruction of {n}");
                assert_eq!(stats, sao_expected, "AVX2 edge search over {n}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "destination and prediction rows differ")]
    fn a_prediction_row_of_the_wrong_length_is_rejected() {
        let mut dst = [0i32; 8];
        reconstruct_row(&mut dst, &[0u8; 8], &[0u8; 7]);
    }

    #[test]
    #[should_panic(expected = "run and source differ")]
    fn a_band_source_run_of_the_wrong_length_is_rejected() {
        let mut stats = BandStats::default();
        band_offset_row(&[0i32; 8], &[0u8; 7], &mut stats);
    }

    #[test]
    #[should_panic(expected = "run and second neighbour differ")]
    fn a_neighbour_run_of_the_wrong_length_is_rejected() {
        let mut stats = EdgeStats::default();
        edge_offset_row(&[0i32; 8], &[0i32; 8], &[0i32; 7], &[0u8; 8], &mut stats);
    }
}
