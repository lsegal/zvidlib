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
//! too, and the answer there depends on the instruction set: its 32-way scatter
//! is not something SSE4.1, AVX2 or NEON can express, so only the
//! classification in front of it vectorizes. That is enough on x86_64, where
//! the SSE4.1 and AVX2 kernels measured ahead of the reference, and not enough
//! on NEON, where both shapes measured below parity and the arm stays scalar.
//! That kernel's documentation carries the numbers for both.
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

/// One row of §8.6.6 reconstruction from an already-coded residual:
/// `dst = Clip1( pred + coded )`.
///
/// This is the quantizing writer's half of the same §8.6.6 loop
/// [`reconstruct_row`] runs for the lossless one. There the residual is the
/// exact `src − pred` and can be recovered from two byte runs; here it has
/// already been round-tripped through the §8.6.4 forward transform, §8.6.3
/// quantization and the decoder's own §8.6.2 reconstruction, so both operands
/// arrive as `i32` runs and the sum no longer fits the 16-bit lanes
/// [`reconstruct_row`] uses. The kernels below therefore stay in 32-bit lanes,
/// which is exact for any `coded` a decoder can produce.
///
/// # Panics
/// Panics unless all three slices have the same length.
pub(crate) fn add_clip_row(dst: &mut [i32], pred: &[i32], coded: &[i32]) {
    assert_eq!(
        dst.len(),
        pred.len(),
        "destination and prediction rows differ"
    );
    assert_eq!(
        dst.len(),
        coded.len(),
        "destination and residual rows differ"
    );
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 => {
            // SAFETY: the lengths agree, and this arm is only reachable after
            // `is_x86_feature_detected!("avx2")` or an override clamped to an
            // available instruction set.
            unsafe { x86::add_clip_row_avx2(dst, pred, coded) }
        }
        #[cfg(target_arch = "x86_64")]
        ISA_SSE41 => {
            // SAFETY: as above, with SSE4.1 detected.
            unsafe { x86::add_clip_row_sse41(dst, pred, coded) }
        }
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => {
            // SAFETY: the lengths agree, and NEON is part of the aarch64
            // baseline this code is compiled for.
            unsafe { neon::add_clip_row(dst, pred, coded) }
        }
        _ => add_clip_row_scalar(dst, pred, coded),
    }
}

/// The portable reference for [`add_clip_row`].
pub(crate) fn add_clip_row_scalar(dst: &mut [i32], pred: &[i32], coded: &[i32]) {
    for ((d, &p), &c) in dst.iter_mut().zip(pred.iter()).zip(coded.iter()) {
        *d = (p + c).clamp(0, BIT_DEPTH_MAX);
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
/// was worth landing on NEON, so `aarch64` still resolves to the scalar
/// reference here, the way
/// [`crate::hevc::engine::simd::combine_weighted`] does on the instruction sets
/// where its kernel measured below parity.
///
/// # x86_64 separates in isolation and not in the encoder
///
/// The site was kept rather than the call inlined into
/// [`crate::hevc::engine::encoder::recon`] so the band search stayed a named
/// `hevc_recon` dispatch point with a bit-exactness harness pointed at it, on
/// the argument that the NEON result was measured rather than universal. The
/// same two shapes were therefore timed on x86_64, and the answer depends on
/// which benchmark is asked.
///
/// **In isolation the classification does pay.** Timed by
/// `bench_band_offset_row` across five CPU models, nine `ubuntu-latest` draws
/// plus two `macos-15-intel` ones, best of nine interleaved rounds per draw and
/// grouped by CPU model - the pool draws several models and they disagree by
/// more than the effect, so an average across it answers nothing. Ratio of the
/// lane-scatter shape to this scalar reference, at the run lengths #305 used:
///
/// | CPU model | 16 | 64 | 256 | 1024 |
/// |---|---|---|---|---|
/// | Intel Xeon 6973P-C (AVX2) | 1.16x | 1.53x | 1.53x | 1.51x |
/// | Intel Xeon Platinum 8573C (AVX2) | 1.16x | 1.28x | 1.44x | 1.39x |
/// | Intel Core i7-8700B (AVX2) | 1.11x | 1.25x | 1.28x | 1.30x |
/// | AMD EPYC 7763 (AVX2) | 1.24x | 1.13x | 1.10x | 1.10x |
/// | AMD EPYC 9V74 (AVX2) | 1.23x | 0.97-1.02x | 1.01x | 0.97x |
///
/// Each row reproduces to about ±0.02 across that model's independent draws.
/// Four of the five models separate at every length, by 1.10x to 1.53x, and the
/// SSE4.1 shape is ahead of scalar on every model at every length (1.02-1.45x).
/// On that measurement alone the kernel is worth landing.
///
/// **In the encoder it does not.** The whole-picture
/// `hevc_encode_640x352_reconstruct` group was then run as a paired
/// branch-against-base comparison - both trees built and timed on the same
/// host, interleaved within a round, five rounds per draw, twelve draws across
/// five models - with the group's own scalar arm as the control, since that arm
/// resolves to this reference in both trees and so has to read 1.00x. It does,
/// to within ±0.02 everywhere. The kernel arms do not improve:
///
/// | CPU model | draws | `avx2` | `sse4.1` | `scalar` (control) |
/// |---|---|---|---|---|
/// | Intel Xeon Platinum 8573C | 1 | 1.00x | 1.01x | 1.01x |
/// | Intel Xeon Platinum 8370C | 2 | 1.01x | 1.00-1.02x | 1.00x |
/// | Intel Core i7-8700B | 1 | 0.99-1.00x | 1.01-1.02x | 1.00-1.02x |
/// | AMD EPYC 7763 | 5 | 0.97-0.99x | 0.95-0.98x | 0.99-1.00x |
/// | AMD EPYC 9V74 | 3 | 0.94-0.95x | 0.95x | 1.00x |
///
/// On the Intel parts the kernel is invisible against its own control; on both
/// AMD parts it is a 2% to 6% *regression*, reproducing across independent
/// draws with every round signed the same way, and well outside what the
/// control moves by. So the shape that reads 1.10-1.53x in a loop of its own
/// buys nothing where the encoder calls it, and on Zen costs more than it
/// saves.
///
/// The two benchmarks measure different things, and the encoder's is the one
/// that decides. `bench_band_offset_row` calls the kernel back-to-back over one
/// L1-resident run of up to 1024 samples with `stats` hot and the call fully
/// predicted; the encoder calls it once per CTB row, so 16 to 64 samples, in
/// between the rest of reconstruction. The leading explanation for the
/// remaining gap - that a `#[target_feature]` kernel cannot be inlined into the
/// caller the way this reference can, so the per-call overhead lands on a run
/// too short to amortize it - is consistent with the isolated win being
/// smallest at 16 samples, but it was not itself measured and is not the reason
/// the kernel is unlanded. The measured whole-picture result is.
///
/// **No x86_64 kernel is dispatched to**, the same call
/// [`crate::hevc::engine::simd::combine_weighted`] got at four lanes and the
/// same one #305 made for NEON. Both x86 shapes stay behind `#[cfg(test)]` as
/// the measurement apparatus, asserted bit-exact so the figures above are
/// figures for kernels that would be correct to land.
///
/// AVX-512 was timed too, since `ubuntu-latest` turned out to draw AVX-512CD
/// hosts, and it does not separate even in isolation. Resolving the scatter's
/// duplicate indices with `vpconflictd` reads **0.42-0.56x** on every host that
/// can run it: the conflict-resolving pointer chase costs more than the
/// read-modify-writes it removes, at 32 bands over 16 lanes where duplicates
/// are rare. Merely widening the classification to 512 bits reads 1.14-1.52x on
/// the Intel parts but **0.21-0.30x** on Zen 5, whose double-pumped AVX-512
/// makes the wider classification a large loss.
///
/// # Panics
/// Panics unless the run and the source are the same length.
pub(crate) fn band_offset_row(here: &[i32], src: &[u8], stats: &mut BandStats) {
    assert_eq!(here.len(), src.len(), "run and source differ");
    // Every arm resolves here: NEON measured below parity in isolation, and the
    // x86 kernels that do separate in isolation do not separate in the encoder.
    // See above for both measurements.
    band_offset_row_scalar(here, src, stats);
}

/// One CTB of the §8.7.3.2 band classification, dispatched once for the whole
/// rectangle rather than once per row.
///
/// # The call shape #382 measured
///
/// [`band_offset_row`] records that the lane-scatter kernel reads 1.10-1.53x
/// against the scalar reference in `bench_band_offset_row` and 1.00x (Intel) to
/// 0.94-0.99x (AMD) in `hevc_encode_640x352_reconstruct`, and names two
/// candidate causes it could not separate: a `#[target_feature]` call the
/// compiler cannot inline, landing on a 16-to-64-sample run too short to
/// amortize it, or a band search too small a share of a reconstruction for any
/// ratio on it to show.
///
/// #382 separated them by timing the reconstruction group with the band half of
/// the search skipped, which is what `hevc_encode_*_reconstruct_no_band_search`
/// exists for. The share is *not* small: see `benches/README.md`. That left the
/// call shape, and this entry is the direct test of it — the rows of a CTB are
/// walked inside one `#[target_feature]` entry, so a 16x16 luma CTB pays one
/// non-inlinable call instead of sixteen and a 8x8 chroma CTB one instead of
/// eight.
///
/// `here_stride` and `src_stride` are the two planes' own row pitches; `here`
/// and `src` start at the rectangle's first sample.
///
/// # Panics
/// Panics unless both planes hold `height` rows of at least `width` samples
/// from the given origins.
pub(crate) fn band_offset_rect(
    here: &[i32],
    here_stride: usize,
    src: &[u8],
    src_stride: usize,
    width: usize,
    height: usize,
    stats: &mut BandStats,
) {
    if width == 0 || height == 0 {
        return;
    }
    assert!(
        here.len() >= (height - 1) * here_stride + width,
        "reconstruction rectangle runs past the plane"
    );
    assert!(
        src.len() >= (height - 1) * src_stride + width,
        "source rectangle runs past the plane"
    );
    // Every instruction set resolves to the scalar reference: the once-per-CTB
    // shape was measured against it and is a regression, worse on Zen 5 than
    // the once-per-row shape it was meant to repair. See above.
    for y in 0..height {
        band_offset_row_scalar(
            &here[y * here_stride..y * here_stride + width],
            &src[y * src_stride..y * src_stride + width],
            stats,
        );
    }
}

/// The portable reference for [`band_offset_row`], and the implementation it
/// dispatches to on every instruction set without a kernel of its own.
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
    use super::{BAND_SHIFT, BIT_DEPTH_MAX, BandStats, EDGE_IDX_BY_CATEGORY, EdgeStats};
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

    /// `Clip1(pred + coded)` over four `i32` lanes.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn add_clip4_sse41(dst: *mut i32, pred: *const i32, coded: *const i32) {
        unsafe {
            let p = _mm_loadu_si128(pred.cast());
            let c = _mm_loadu_si128(coded.cast());
            let sum = _mm_add_epi32(p, c);
            let clipped = _mm_min_epi32(
                _mm_max_epi32(sum, _mm_setzero_si128()),
                _mm_set1_epi32(BIT_DEPTH_MAX),
            );
            _mm_storeu_si128(dst.cast(), clipped);
        }
    }

    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn add_clip_row_sse41(dst: &mut [i32], pred: &[i32], coded: &[i32]) {
        unsafe {
            let n = dst.len();
            let mut i = 0;
            while i + 4 <= n {
                add_clip4_sse41(
                    dst.as_mut_ptr().add(i),
                    pred.as_ptr().add(i),
                    coded.as_ptr().add(i),
                );
                i += 4;
            }
            super::add_clip_row_scalar(&mut dst[i..], &pred[i..], &coded[i..]);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn add_clip_row_avx2(dst: &mut [i32], pred: &[i32], coded: &[i32]) {
        unsafe {
            let n = dst.len();
            let zero = _mm256_setzero_si256();
            let max = _mm256_set1_epi32(BIT_DEPTH_MAX);
            let mut i = 0;
            while i + 8 <= n {
                let p = _mm256_loadu_si256(pred.as_ptr().add(i).cast());
                let c = _mm256_loadu_si256(coded.as_ptr().add(i).cast());
                let sum = _mm256_add_epi32(p, c);
                let clipped = _mm256_min_epi32(_mm256_max_epi32(sum, zero), max);
                _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), clipped);
                i += 8;
            }
            while i + 4 <= n {
                add_clip4_sse41(
                    dst.as_mut_ptr().add(i),
                    pred.as_ptr().add(i),
                    coded.as_ptr().add(i),
                );
                i += 4;
            }
            super::add_clip_row_scalar(&mut dst[i..], &pred[i..], &coded[i..]);
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

    /// §8.7.3.2 band classification for eight samples: the eight band indices
    /// `Clip3(0, 255, recon) >> bandShift`, and the eight `src − recon` errors
    /// taken against the *unclamped* reconstruction the way the scalar
    /// reference does.
    ///
    /// This is the whole of what a vector unit without a scatter can do for the
    /// band search, and it is shared by both candidate shapes below so that the
    /// measurement separates the scatter from the classification rather than
    /// from two different classifications.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn band_classify8_avx2(here: *const i32, src: *const u8) -> (__m256i, __m256i) {
        unsafe {
            let recon = _mm256_loadu_si256(here.cast());
            let clamped = _mm256_min_epi32(
                _mm256_max_epi32(recon, _mm256_setzero_si256()),
                _mm256_set1_epi32(BIT_DEPTH_MAX),
            );
            let band = _mm256_srli_epi32::<BAND_SHIFT>(clamped);
            let s = _mm256_cvtepu8_epi32(_mm_loadl_epi64(src.cast()));
            (band, _mm256_sub_epi32(s, recon))
        }
    }

    #[cfg(test)]
    /// Candidate A on AVX2: classify eight samples in the vector unit, store
    /// the indices and errors to staging buffers, then scatter the buffers with
    /// a scalar loop.
    ///
    /// Not dispatched to: it is the control the dispatched
    /// [`band_offset_row_avx2_lanes`] was chosen against, kept because the
    /// choice between the two shapes is what the measurement below decides. It
    /// is compiled and asserted bit-exact against the scalar reference so the
    /// number it produces is a number for a kernel that would be correct to
    /// land.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn band_offset_row_avx2_staged(
        here: &[i32],
        src: &[u8],
        stats: &mut BandStats,
    ) {
        unsafe {
            let n = here.len();
            let mut bands = [0i32; 8];
            let mut errs = [0i32; 8];
            let mut i = 0;
            while i + 8 <= n {
                let (band, err) = band_classify8_avx2(here.as_ptr().add(i), src.as_ptr().add(i));
                _mm256_storeu_si256(bands.as_mut_ptr().cast(), band);
                _mm256_storeu_si256(errs.as_mut_ptr().cast(), err);
                for k in 0..8 {
                    let b = bands[k] as usize;
                    stats.sums[b] += i64::from(errs[k]);
                    stats.counts[b] += 1;
                }
                i += 8;
            }
            super::band_offset_row_scalar(&here[i..], &src[i..], stats);
        }
    }

    /// Candidate B on AVX2: classify eight samples in the vector unit, then
    /// scatter them straight out of the lanes with `vpextrd` rather than
    /// through the staging buffers [`band_offset_row_avx2_staged`] uses.
    ///
    /// The better of the two x86 shapes - at or ahead of the staged one at
    /// every run length on every CPU model timed - and still not dispatched to:
    /// its isolated 1.10-1.53x does not reach the whole-picture group, which
    /// reads 1.00x on Intel and 0.94-0.99x on AMD. See
    /// [`super::band_offset_row`].
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn band_offset_row_avx2_lanes(
        here: &[i32],
        src: &[u8],
        stats: &mut BandStats,
    ) {
        unsafe {
            let n = here.len();
            let mut i = 0;
            while i + 8 <= n {
                let (band, err) = band_classify8_avx2(here.as_ptr().add(i), src.as_ptr().add(i));
                let lo_b = _mm256_castsi256_si128(band);
                let hi_b = _mm256_extracti128_si256(band, 1);
                let lo_e = _mm256_castsi256_si128(err);
                let hi_e = _mm256_extracti128_si256(err, 1);
                macro_rules! scatter {
                    ($b:expr, $e:expr, $k:literal) => {{
                        let idx = _mm_extract_epi32::<$k>($b) as usize;
                        stats.sums[idx] += i64::from(_mm_extract_epi32::<$k>($e));
                        stats.counts[idx] += 1;
                    }};
                }
                scatter!(lo_b, lo_e, 0);
                scatter!(lo_b, lo_e, 1);
                scatter!(lo_b, lo_e, 2);
                scatter!(lo_b, lo_e, 3);
                scatter!(hi_b, hi_e, 0);
                scatter!(hi_b, hi_e, 1);
                scatter!(hi_b, hi_e, 2);
                scatter!(hi_b, hi_e, 3);
                i += 8;
            }
            super::band_offset_row_scalar(&here[i..], &src[i..], stats);
        }
    }

    #[cfg(test)]
    /// One CTB of [`band_offset_row_avx2_lanes`] under a single
    /// `#[target_feature]` entry.
    ///
    /// This is the once-per-CTB call shape #382 tested the call-overhead
    /// explanation with: the rows are walked inside the vector entry, so a
    /// 16x16 luma CTB pays one non-inlinable call instead of sixteen. See
    /// [`super::band_offset_rect`] for what that bought.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn band_offset_rect_avx2(
        here: &[i32],
        here_stride: usize,
        src: &[u8],
        src_stride: usize,
        width: usize,
        height: usize,
        stats: &mut BandStats,
    ) {
        unsafe {
            for y in 0..height {
                let recon = &here[y * here_stride..y * here_stride + width];
                let source = &src[y * src_stride..y * src_stride + width];
                band_offset_row_avx2_lanes(recon, source, stats);
            }
        }
    }

    #[cfg(test)]
    /// [`band_offset_rect_avx2`] at four lanes.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn band_offset_rect_sse41(
        here: &[i32],
        here_stride: usize,
        src: &[u8],
        src_stride: usize,
        width: usize,
        height: usize,
        stats: &mut BandStats,
    ) {
        unsafe {
            for y in 0..height {
                let recon = &here[y * here_stride..y * here_stride + width];
                let source = &src[y * src_stride..y * src_stride + width];
                band_offset_row_sse41_lanes(recon, source, stats);
            }
        }
    }

    #[cfg(test)]
    /// §8.7.3.2 band classification for sixteen samples, the AVX-512 form of
    /// [`band_classify8_avx2`].
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn band_classify16_avx512(here: *const i32, src: *const u8) -> (__m512i, __m512i) {
        unsafe {
            let recon = _mm512_loadu_si512(here.cast());
            let clamped = _mm512_min_epi32(
                _mm512_max_epi32(recon, _mm512_setzero_si512()),
                _mm512_set1_epi32(BIT_DEPTH_MAX),
            );
            let band = _mm512_srli_epi32::<{ BAND_SHIFT as u32 }>(clamped);
            let s = _mm512_cvtepu8_epi32(_mm_loadu_si128(src.cast()));
            (band, _mm512_sub_epi32(s, recon))
        }
    }

    #[cfg(test)]
    /// Candidate A at sixteen lanes: AVX-512 classification into staging
    /// buffers, then the same scalar scatter the narrower staged candidates
    /// use. Not dispatched to.
    ///
    /// This is the control for [`band_offset_row_avx512_conflict`]: it widens
    /// the classification without touching the scatter, so the difference
    /// between the two is the conflict resolution rather than the vector
    /// width.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn band_offset_row_avx512_staged(
        here: &[i32],
        src: &[u8],
        stats: &mut BandStats,
    ) {
        unsafe {
            let n = here.len();
            let mut bands = [0i32; 16];
            let mut errs = [0i32; 16];
            let mut i = 0;
            while i + 16 <= n {
                let (band, err) = band_classify16_avx512(here.as_ptr().add(i), src.as_ptr().add(i));
                _mm512_storeu_si512(bands.as_mut_ptr().cast(), band);
                _mm512_storeu_si512(errs.as_mut_ptr().cast(), err);
                for k in 0..16 {
                    let b = bands[k] as usize;
                    stats.sums[b] += i64::from(errs[k]);
                    stats.counts[b] += 1;
                }
                i += 16;
            }
            super::band_offset_row_scalar(&here[i..], &src[i..], stats);
        }
    }

    #[cfg(test)]
    /// Candidate C, the shape only AVX-512 can express: resolve the scatter's
    /// duplicate indices *inside* the vector unit with `vpconflictd`, so the
    /// scalar loop that follows performs one read-modify-write per **distinct**
    /// band in the sixteen samples rather than one per sample.
    ///
    /// `vpconflictd` gives each lane the bitmask of earlier lanes that classify
    /// into the same band. Pointer-jumping over that mask — four rounds, since
    /// a conflict group is at most sixteen lanes — accumulates each group's
    /// error sum and sample count into the group's *last* lane, and a lane is
    /// last exactly when no other lane's conflict mask names it. Compressing
    /// those lanes leaves one `(band, sum, count)` triple per distinct band,
    /// which is what the scalar tail scatters.
    ///
    /// Not dispatched to; see [`band_offset_row_avx2_staged`]. Requires
    /// AVX-512CD for `vpconflictd`/`vplzcntd`; the classification itself is
    /// AVX-512F.
    #[target_feature(enable = "avx512f,avx512cd")]
    pub(super) unsafe fn band_offset_row_avx512_conflict(
        here: &[i32],
        src: &[u8],
        stats: &mut BandStats,
    ) {
        unsafe {
            let n = here.len();
            let mut bands = [0i32; 16];
            let mut group_sums = [0i32; 16];
            let mut group_counts = [0i32; 16];
            let mut i = 0;
            while i + 16 <= n {
                let (band, err) = band_classify16_avx512(here.as_ptr().add(i), src.as_ptr().add(i));

                // Lane `j` of `conflicts` holds the bitmask of lanes before it
                // that landed in the same band.
                let conflicts = _mm512_conflict_epi32(band);
                // A lane is the last of its group when no lane names it, so the
                // union of every mask is exactly the set of non-last lanes.
                let named = _mm512_reduce_or_epi32(conflicts) as u16;
                let last = !named;

                // `ptr` starts at the closest earlier lane in the same group —
                // the highest set bit of that lane's mask — and doubles its
                // reach each round; `valid` says whether such a lane exists.
                let mut ptr = _mm512_and_si512(
                    _mm512_sub_epi32(_mm512_set1_epi32(31), _mm512_lzcnt_epi32(conflicts)),
                    _mm512_set1_epi32(15),
                );
                let mut valid =
                    _mm512_maskz_set1_epi32(_mm512_test_epi32_mask(conflicts, conflicts), -1);
                let mut sum = err;
                let mut count = _mm512_set1_epi32(1);
                // Four rounds reach sixteen lanes, the largest a group can be.
                for _ in 0..4 {
                    let live = _mm512_test_epi32_mask(valid, valid);
                    if live == 0 {
                        break;
                    }
                    sum = _mm512_mask_add_epi32(sum, live, sum, _mm512_permutexvar_epi32(ptr, sum));
                    count = _mm512_mask_add_epi32(
                        count,
                        live,
                        count,
                        _mm512_permutexvar_epi32(ptr, count),
                    );
                    valid = _mm512_and_si512(valid, _mm512_permutexvar_epi32(ptr, valid));
                    ptr = _mm512_permutexvar_epi32(ptr, ptr);
                }

                _mm512_storeu_si512(
                    bands.as_mut_ptr().cast(),
                    _mm512_maskz_compress_epi32(last, band),
                );
                _mm512_storeu_si512(
                    group_sums.as_mut_ptr().cast(),
                    _mm512_maskz_compress_epi32(last, sum),
                );
                _mm512_storeu_si512(
                    group_counts.as_mut_ptr().cast(),
                    _mm512_maskz_compress_epi32(last, count),
                );
                for k in 0..last.count_ones() as usize {
                    let b = bands[k] as usize;
                    stats.sums[b] += i64::from(group_sums[k]);
                    stats.counts[b] += i64::from(group_counts[k]);
                }
                i += 16;
            }
            super::band_offset_row_scalar(&here[i..], &src[i..], stats);
        }
    }

    /// The four-lane classification behind the two SSE4.1 candidates.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn band_classify4_sse41(here: *const i32, src: *const u8) -> (__m128i, __m128i) {
        unsafe {
            let recon = _mm_loadu_si128(here.cast());
            let clamped = _mm_min_epi32(
                _mm_max_epi32(recon, _mm_setzero_si128()),
                _mm_set1_epi32(BIT_DEPTH_MAX),
            );
            let band = _mm_srli_epi32::<BAND_SHIFT>(clamped);
            let s = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(src.cast::<u32>().read_unaligned() as i32));
            (band, _mm_sub_epi32(s, recon))
        }
    }

    #[cfg(test)]
    /// Candidate A at four lanes: SSE4.1 classification into staging buffers,
    /// then a scalar scatter. Not dispatched to.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn band_offset_row_sse41_staged(
        here: &[i32],
        src: &[u8],
        stats: &mut BandStats,
    ) {
        unsafe {
            let n = here.len();
            let mut bands = [0i32; 4];
            let mut errs = [0i32; 4];
            let mut i = 0;
            while i + 4 <= n {
                let (band, err) = band_classify4_sse41(here.as_ptr().add(i), src.as_ptr().add(i));
                _mm_storeu_si128(bands.as_mut_ptr().cast(), band);
                _mm_storeu_si128(errs.as_mut_ptr().cast(), err);
                for k in 0..4 {
                    let b = bands[k] as usize;
                    stats.sums[b] += i64::from(errs[k]);
                    stats.counts[b] += 1;
                }
                i += 4;
            }
            super::band_offset_row_scalar(&here[i..], &src[i..], stats);
        }
    }

    /// Candidate B at four lanes: the SSE4.1 classification scattered straight
    /// out of the lanes, the narrower form of [`band_offset_row_avx2_lanes`].
    /// Not dispatched to, for the same measured reason.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn band_offset_row_sse41_lanes(
        here: &[i32],
        src: &[u8],
        stats: &mut BandStats,
    ) {
        unsafe {
            let n = here.len();
            let mut i = 0;
            while i + 4 <= n {
                let (band, err) = band_classify4_sse41(here.as_ptr().add(i), src.as_ptr().add(i));
                macro_rules! scatter {
                    ($k:literal) => {{
                        let idx = _mm_extract_epi32::<$k>(band) as usize;
                        stats.sums[idx] += i64::from(_mm_extract_epi32::<$k>(err));
                        stats.counts[idx] += 1;
                    }};
                }
                scatter!(0);
                scatter!(1);
                scatter!(2);
                scatter!(3);
                i += 4;
            }
            super::band_offset_row_scalar(&here[i..], &src[i..], stats);
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

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn add_clip_row(dst: &mut [i32], pred: &[i32], coded: &[i32]) {
        unsafe {
            let n = dst.len();
            let zero = vdupq_n_s32(0);
            let max = vdupq_n_s32(BIT_DEPTH_MAX);
            let mut i = 0;
            while i + 4 <= n {
                let p = vld1q_s32(pred.as_ptr().add(i));
                let c = vld1q_s32(coded.as_ptr().add(i));
                let sum = vaddq_s32(p, c);
                let clipped = vminq_s32(vmaxq_s32(sum, zero), max);
                vst1q_s32(dst.as_mut_ptr().add(i), clipped);
                i += 4;
            }
            super::add_clip_row_scalar(&mut dst[i..], &pred[i..], &coded[i..]);
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

    /// Coded-residual fixtures for the quantized path: signed values wide
    /// enough that `pred + coded` leaves the 8-bit range in both directions,
    /// so every run exercises the clip at 0 and at 255 as well as the samples
    /// that pass through untouched.
    fn residuals(seed: u64, n: usize) -> Vec<i32> {
        bytes(seed, n)
            .into_iter()
            .map(|b| i32::from(b) * 4 - 480)
            .collect()
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

    /// The strided rectangles the #382 call-shape measurement was taken over,
    /// and the row-by-row scalar answer every arm has to reproduce.
    ///
    /// Strides wider than the rectangle are the case that matters: a CTB is a
    /// window into a plane, so the rows a rect call walks are not contiguous,
    /// and a kernel that ignored the pitch would still pass on a full-width
    /// rectangle. The widths straddle both vector steps and the heights cover
    /// luma and chroma CTBs.
    fn band_rect_fixture(width: usize, height: usize) -> (Vec<i32>, Vec<u8>, BandStats) {
        const STRIDE: usize = BAND_RECT_STRIDE;
        let mut here = samples(0x7777_8888_9999_aaaa, STRIDE * height);
        for (i, value) in here.iter_mut().enumerate() {
            *value = (i as i32 * 7) % 300 - 12;
        }
        let src = bytes(0x0123_4567_89ab_cdef, STRIDE * height);
        let mut expected = BandStats::default();
        for y in 0..height {
            band_offset_row_scalar(
                &here[y * STRIDE..y * STRIDE + width],
                &src[y * STRIDE..y * STRIDE + width],
                &mut expected,
            );
        }
        (here, src, expected)
    }

    /// The pitch [`band_rect_fixture`] lays its rows out at, wider than every
    /// rectangle measured against it.
    const BAND_RECT_STRIDE: usize = 71;

    /// The rectangles both the dispatch test and the x86 kernel test cover.
    const BAND_RECTS: &[(usize, usize)] = &[
        (1, 1),
        (3, 2),
        (4, 4),
        (7, 3),
        (8, 8),
        (16, 16),
        (17, 5),
        (64, 4),
    ];

    /// The once-per-CTB entry has to give the row-by-row answer under every
    /// pin, the same way the once-per-row entry does.
    #[test]
    fn the_band_offset_rect_matches_the_row_reference_on_every_instruction_set() {
        let _guard = simd::test_lock();
        const STRIDE: usize = BAND_RECT_STRIDE;
        for &(width, height) in BAND_RECTS {
            let (here, src, expected) = band_rect_fixture(width, height);
            for isa in simd::available() {
                simd::set_override(Some(isa));
                let mut got = BandStats::default();
                band_offset_rect(&here, STRIDE, &src, STRIDE, width, height, &mut got);
                assert_eq!(
                    got,
                    expected,
                    "{} band rect {width}x{height} at stride {STRIDE}",
                    isa.name()
                );
            }
        }
        simd::set_override(None);
    }

    /// The once-per-CTB x86 kernels are not dispatched to - #382 measured them
    /// as a regression - so nothing else reaches them. They are still asserted
    /// bit-exact here, for the same reason the once-per-row candidates are:
    /// the figures recorded for them have to be figures for kernels that would
    /// have been correct to land.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_x86_band_offset_rect_kernels_match_the_row_reference() {
        const STRIDE: usize = BAND_RECT_STRIDE;
        for &(width, height) in BAND_RECTS {
            let (here, src, expected) = band_rect_fixture(width, height);
            if is_x86_feature_detected!("sse4.1") {
                let mut got = BandStats::default();
                // SAFETY: SSE4.1 is detected, and the rectangle lies inside
                // both planes.
                unsafe {
                    x86::band_offset_rect_sse41(
                        &here, STRIDE, &src, STRIDE, width, height, &mut got,
                    );
                }
                assert_eq!(got, expected, "sse4.1 rect {width}x{height}");
            }
            if is_x86_feature_detected!("avx2") {
                let mut got = BandStats::default();
                // SAFETY: AVX2 is detected, and the rectangle lies inside both
                // planes.
                unsafe {
                    x86::band_offset_rect_avx2(
                        &here, STRIDE, &src, STRIDE, width, height, &mut got,
                    );
                }
                assert_eq!(got, expected, "avx2 rect {width}x{height}");
            }
        }
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
    fn the_coded_add_and_clip_matches_the_scalar_reference_on_every_instruction_set() {
        let _guard = simd::test_lock();
        for &n in RUNS {
            let pred = samples(0x1a2b_3c4d_5e6f_7081, n);
            let coded = residuals(0x7fff_0001_2345_6789, n);
            let mut expected = vec![0i32; n];
            add_clip_row_scalar(&mut expected, &pred, &coded);
            for isa in simd::available() {
                simd::set_override(Some(isa));
                let mut got = vec![0i32; n];
                add_clip_row(&mut got, &pred, &coded);
                assert_eq!(got, expected, "{} add-and-clip of {n} samples", isa.name());
            }
        }
        simd::set_override(None);
    }

    #[test]
    fn the_coded_add_and_clip_saturates_at_both_ends_of_the_8_bit_range() {
        // The fixture above is only a bit-exactness check against the scalar
        // reference; this pins the §8.6.6 clip itself, including a residual
        // far outside the range any single sample could reach unclipped.
        let _guard = simd::test_lock();
        let pred = [0i32, 128, 255, 200, 40, 128, 128];
        let coded = [-1i32, -200, 1, 100, -100, 32_767, -32_768];
        let expected = [0i32, 0, 255, 255, 0, 255, 0];
        for isa in simd::available() {
            simd::set_override(Some(isa));
            let mut got = [0i32; 7];
            add_clip_row(&mut got, &pred, &coded);
            assert_eq!(got, expected, "{}", isa.name());
        }
        simd::set_override(None);
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
            let coded = residuals(0x0f1e_2d3c_4b5a_6978, n);
            let mut add_clip_expected = vec![0i32; n];
            add_clip_row_scalar(&mut add_clip_expected, &here, &coded);
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
                let mut got_add_clip = vec![0i32; n];
                // SAFETY: as above.
                unsafe { x86::add_clip_row_sse41(&mut got_add_clip, &here, &coded) }
                assert_eq!(got, recon_expected, "SSE4.1 reconstruction of {n}");
                assert_eq!(
                    got_add_clip, add_clip_expected,
                    "SSE4.1 add-and-clip of {n}"
                );
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
                let mut got_add_clip = vec![0i32; n];
                // SAFETY: as above.
                unsafe { x86::add_clip_row_avx2(&mut got_add_clip, &here, &coded) }
                assert_eq!(got, recon_expected, "AVX2 reconstruction of {n}");
                assert_eq!(got_add_clip, add_clip_expected, "AVX2 add-and-clip of {n}");
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
    #[should_panic(expected = "destination and residual rows differ")]
    fn a_coded_residual_row_of_the_wrong_length_is_rejected() {
        let mut dst = [0i32; 8];
        add_clip_row(&mut dst, &[0i32; 8], &[0i32; 7]);
    }

    #[test]
    #[should_panic(expected = "run and second neighbour differ")]
    fn a_neighbour_run_of_the_wrong_length_is_rejected() {
        let mut stats = EdgeStats::default();
        edge_offset_row(&[0i32; 8], &[0i32; 8], &[0i32; 7], &[0u8; 8], &mut stats);
    }

    /// Both x86_64 candidate shapes for the band search, at both vector
    /// widths, against the scalar reference. They are not dispatched to, so
    /// this is what says the measurement below is a measurement of a kernel
    /// that would be correct to land rather than of a faster wrong answer.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_x86_band_offset_candidates_match_the_scalar_reference() {
        for &n in RUNS {
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

            if is_x86_feature_detected!("sse4.1") {
                for (name, f) in x86_band_candidates_sse41() {
                    let mut got = BandStats::default();
                    // SAFETY: SSE4.1 is detected, and the runs are equal length.
                    unsafe { f(&here, &src, &mut got) };
                    assert_eq!(got, expected, "{name} over {n} samples");
                }
            }
            if is_x86_feature_detected!("avx2") {
                for (name, f) in x86_band_candidates_avx2() {
                    let mut got = BandStats::default();
                    // SAFETY: AVX2 is detected, and the runs are equal length.
                    unsafe { f(&here, &src, &mut got) };
                    assert_eq!(got, expected, "{name} over {n} samples");
                }
            }
            if host_has_avx512_conflict() {
                for (name, f) in x86_band_candidates_avx512() {
                    let mut got = BandStats::default();
                    // SAFETY: AVX-512F and AVX-512CD are detected, and the runs
                    // are equal length.
                    unsafe { f(&here, &src, &mut got) };
                    assert_eq!(got, expected, "{name} over {n} samples");
                }
            }
        }
    }

    /// The type every band-offset arm the measurement times shares.
    #[cfg(target_arch = "x86_64")]
    type BandKernel = unsafe fn(&[i32], &[u8], &mut BandStats);

    #[cfg(target_arch = "x86_64")]
    fn x86_band_candidates_sse41() -> [(&'static str, BandKernel); 2] {
        [
            ("sse41 staged", x86::band_offset_row_sse41_staged),
            ("sse41 lanes", x86::band_offset_row_sse41_lanes),
        ]
    }

    #[cfg(target_arch = "x86_64")]
    fn x86_band_candidates_avx2() -> [(&'static str, BandKernel); 2] {
        [
            ("avx2 staged", x86::band_offset_row_avx2_staged),
            ("avx2 lanes", x86::band_offset_row_avx2_lanes),
        ]
    }

    /// The AVX-512 candidates, which need AVX-512CD for `vpconflictd` as well
    /// as AVX-512F for the classification.
    #[cfg(target_arch = "x86_64")]
    fn x86_band_candidates_avx512() -> [(&'static str, BandKernel); 2] {
        [
            ("avx512 staged", x86::band_offset_row_avx512_staged),
            ("avx512 conflict", x86::band_offset_row_avx512_conflict),
        ]
    }

    /// Whether this host can run the AVX-512 candidates at all.
    #[cfg(target_arch = "x86_64")]
    fn host_has_avx512_conflict() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512cd")
    }

    /// What the §8.7.3 band-offset search costs on x86_64, scalar reference
    /// against every candidate shape, at the run lengths #305's NEON
    /// measurement used.
    ///
    /// Ignored by default because it measures rather than asserts; run it with
    /// `cargo test --features native --release --lib
    /// recon_simd::tests::bench_band_offset_row -- --ignored --nocapture`.
    ///
    /// The runs are L1-resident and the arms are interleaved within each round
    /// — every arm is timed once per round, and the reported time is the
    /// minimum across rounds — so a scheduling artefact has to hit the same arm
    /// in every round to survive. The spread column is what separates a result
    /// from noise: it is how far the *worst* round of an arm sat above its own
    /// best, and a ratio nearer 1.00x than that spread has not separated from
    /// the reference.
    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn bench_band_offset_row() {
        use std::hint::black_box;

        /// The run lengths #305 timed on NEON.
        const LENGTHS: &[usize] = &[16, 64, 256, 1024];
        const ROUNDS: usize = 9;
        /// Samples classified per timed call, held constant across the run
        /// lengths so a short run is not measured over less work than a long
        /// one.
        const SAMPLES_PER_CALL: usize = 1 << 20;

        println!("# host instruction sets (band-offset measurement)");
        for (name, present) in [
            ("sse4.1", is_x86_feature_detected!("sse4.1")),
            ("avx2", is_x86_feature_detected!("avx2")),
            ("avx512f", is_x86_feature_detected!("avx512f")),
            ("avx512cd", is_x86_feature_detected!("avx512cd")),
            ("avx512bw", is_x86_feature_detected!("avx512bw")),
            ("avx512vl", is_x86_feature_detected!("avx512vl")),
        ] {
            println!("#   {name}: {present}");
        }
        println!("# dispatch site hevc_recon detected: {:?}", isa());

        for &n in LENGTHS {
            let here = {
                let mut v = samples(0x7777_8888_9999_aaaa, n);
                for (i, h) in v.iter_mut().enumerate() {
                    if i % 37 == 0 {
                        *h = 262;
                    }
                }
                v
            };
            let src = bytes(0xc0ff_ee00_1234_5678, n);
            let calls = SAMPLES_PER_CALL / n;

            let mut arms: Vec<(&'static str, BandKernel)> =
                vec![("scalar", band_offset_row_scalar as BandKernel)];
            if is_x86_feature_detected!("sse4.1") {
                arms.extend(x86_band_candidates_sse41());
            }
            if is_x86_feature_detected!("avx2") {
                arms.extend(x86_band_candidates_avx2());
            }
            if host_has_avx512_conflict() {
                arms.extend(x86_band_candidates_avx512());
            }

            let mut best = vec![f64::INFINITY; arms.len()];
            let mut worst = vec![0f64; arms.len()];
            let reference = {
                let mut stats = BandStats::default();
                band_offset_row_scalar(&here, &src, &mut stats);
                stats
            };
            for _ in 0..ROUNDS {
                for (arm, (name, f)) in arms.iter().enumerate() {
                    let mut stats = BandStats::default();
                    let start = std::time::Instant::now();
                    for _ in 0..calls {
                        // SAFETY: every candidate's feature was detected above.
                        unsafe { f(black_box(&here), black_box(&src), &mut stats) };
                    }
                    let secs = start.elapsed().as_secs_f64();
                    black_box(&stats);
                    assert_eq!(
                        stats.counts.iter().sum::<i64>(),
                        (n * calls) as i64,
                        "{name} lost samples"
                    );
                    assert_eq!(stats.sums[0] / calls as i64, reference.sums[0], "{name}");
                    best[arm] = best[arm].min(secs);
                    worst[arm] = worst[arm].max(secs);
                }
            }

            println!();
            println!("run of {n} samples, {calls} calls per round, best of {ROUNDS} rounds");
            println!(
                "{:<14}{:>12}{:>12}{:>12}",
                "arm", "Msamp/s", "ratio", "spread"
            );
            let samples_done = (n * calls) as f64;
            for (arm, (name, _)) in arms.iter().enumerate() {
                println!(
                    "{:<14}{:>12.1}{:>11.2}x{:>11.1}%",
                    name,
                    samples_done / best[arm] / 1e6,
                    best[0] / best[arm],
                    100.0 * (worst[arm] / best[arm] - 1.0),
                );
            }
        }
    }
}
