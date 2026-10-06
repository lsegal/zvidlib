//! Runtime-dispatched vector kernels for the Vorbis encoder's analysis stages
//! (issue #573), registered as the `vorbis_encode` site of
//! [`crate::simd::active_by_site`].
//!
//! # What is vectorized, and why only that
//!
//! A profile of the encoder (30 s of 44.1 kHz stereo at quality 3, 48 kHz
//! stereo at quality 6 and 44.1 kHz mono at quality 0) spends its time in
//! tone masking (~21%), coupling/quantization (11-18%), noise masking (~15%),
//! floor fitting (~12%), the real FFT (~10%), envelope detection (~8%) and the
//! forward MDCT (~6%). The kernels here cover the parts of those stages whose
//! iterations are independent, which is the only shape a kernel can take
//! while staying bit-exact:
//!
//! | Kernel | Scalar reference |
//! | --- | --- |
//! | forward MDCT: pre-rotation, butterflies, bit-reversal, post-rotation | `mdct::MdctLookup::forward_scalar` (also the envelope's 128-point MDCT) |
//! | real FFT: the `ido == 1` radix-4 pass and the radix-2/4 twiddle loops | `smallft::dradf4_first_rows`, `dradf4_twiddle_step`, `dradf2_twiddle_step` |
//! | noise-mask least-squares fits and companding | `psy::noise_bark_scalar`, `noise_fixed_scalar`, `noise_extrapolate_scalar`, `noise_compand_scalar` |
//! | floor-fit accumulation | `floor1::accumulate_fit_scalar` |
//! | log power spectra | `mapping0::log_mdct_scalar`, `log_fft_scalar` |
//!
//! The rest stays scalar on purpose. Tone masking is a scatter-max of masking
//! curves into a seed array followed by a monotonic-stack sweep
//! (`seed_chase`) and a running minimum (`max_seeds`); every step depends on
//! the one before it. Coupling and quantization work in 16-bin partitions
//! whose noise normalization sorts each partition and walks it with a running
//! energy budget. The noise mask's running sums (`bark_noise_hybridmp`'s
//! prefix pass), the envelope's band accumulation and the 32-point MDCT tail
//! are each a single floating-point recurrence that a vector could only
//! compute by reassociating it, which would change the bits.
//!
//! # Bit-exactness
//!
//! The encoder is a bit-exact port of libvorbis, and every kernel here is
//! bit-exact with its scalar reference: the same IEEE-754 single- and
//! double-precision operations, in the same order, on the same operands, with
//! no fused multiply-add (the scalar code never contracts). Encoded packets
//! are therefore identical with SIMD on and off, which `tests` asserts both
//! kernel by kernel and on whole encodes.

// Only x86_64 and aarch64 have entry points; elsewhere (`wasm32` included)
// every dispatcher runs its scalar reference.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod kernels;
#[cfg(test)]
mod tests;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod vector;

use crate::simd::SimdIsa;
use std::sync::OnceLock;

use super::floor1::FitSums;
use super::mdct::MdctLookup;
use super::psy::NoiseSums;

/// The instruction set the Vorbis encoder kernels will actually run.
///
/// Registered as the `vorbis_encode` site of [`crate::simd::active_by_site`].
/// It consults [`crate::simd::set_override`] first and falls back to its own
/// cached CPU probe, so pinning an instruction set reaches these kernels and a
/// benchmark can prove that it did.
#[must_use]
pub(crate) fn active_isa() -> SimdIsa {
    static DETECTED: OnceLock<SimdIsa> = OnceLock::new();
    crate::simd::override_isa().unwrap_or_else(|| *DETECTED.get_or_init(crate::simd::detected))
}

// ---------------------------------------------------------------------
// Per-instruction-set entry points
//
// Each kernel gets one `#[target_feature]` wrapper per instruction set, and
// the wrapper is the only place its instruction set is enabled. The generic
// kernel bodies and every `vector` method are `#[inline(always)]`: a
// `#[target_feature]` attribute applies to the function it is written on and
// not to what that function calls, so a kernel body the inliner declined
// would be compiled at the baseline instruction set, with every intrinsic an
// out-of-line call (issue #336). The wrappers themselves are
// `#[inline(never)]` so each stays one feature-enabled symbol rather than
// being folded into a dispatcher that cannot carry the feature.
// `.github/scripts/check_simd_target_features.py` checks the emitted x86_64
// assembly for any out-of-line `core::core_arch` call, crate-wide, so a
// kernel here that loses its instruction set fails CI (issue #341).
//
// The AVX2 entry points run the same four-lane kernels as SSE4.1,
// VEX-encoded. Every kernel shape here is four lanes or narrower: complex
// pairs two to a register, stride-4 MDCT walks, and 4x4 transposes.
// ---------------------------------------------------------------------

macro_rules! entry_points {
    (
        $(#[$meta:meta])*
        fn [$sse:ident, $avx:ident, $neon:ident]($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)?
            = $kernel:ident;
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse4.1")]
        #[inline(never)]
        $(#[$meta])*
        unsafe fn $sse($($arg: $ty),*) $(-> $ret)? {
            unsafe { kernels::$kernel::<vector::Sse>($($arg),*) }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[inline(never)]
        $(#[$meta])*
        unsafe fn $avx($($arg: $ty),*) $(-> $ret)? {
            unsafe { kernels::$kernel::<vector::Sse>($($arg),*) }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline(never)]
        $(#[$meta])*
        unsafe fn $neon($($arg: $ty),*) $(-> $ret)? {
            unsafe { kernels::$kernel::<vector::Neon>($($arg),*) }
        }
    };
}

/// Runs the vector entry point for `isa`, or `$fallback` (the scalar
/// reference) for [`SimdIsa::Scalar`] and any instruction set this target
/// does not have.
macro_rules! dispatch {
    ($isa:expr, [$sse:ident, $avx:ident, $neon:ident]($($arg:expr),* $(,)?), $fallback:expr) => {
        match $isa {
            // SAFETY: `active_isa` and `crate::simd::set_override` only ever
            // yield an instruction set this host executes.
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Sse41 => unsafe { $sse($($arg),*) },
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx2 => unsafe { $avx($($arg),*) },
            #[cfg(target_arch = "aarch64")]
            SimdIsa::Neon => unsafe { $neon($($arg),*) },
            _ => $fallback,
        }
    };
}

entry_points! {
    fn [mdct_sse41, mdct_avx2, mdct_neon](
        m: &MdctLookup, input: &[f32], out: &mut [f32], w: &mut [f32]
    ) = mdct_forward;
}
entry_points! {
    fn [dradf4_ido1_sse41, dradf4_ido1_avx2, dradf4_ido1_neon](
        l1: usize, cc: &[f32], ch: &mut [f32]
    ) = dradf4_ido1;
}
entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [dradf4_twiddle_sse41, dradf4_twiddle_avx2, dradf4_twiddle_neon](
        ido: usize, l1: usize, cc: &[f32], ch: &mut [f32],
        wa1: &[f32], wa2: &[f32], wa3: &[f32]
    ) = dradf4_twiddle;
}
entry_points! {
    fn [dradf2_twiddle_sse41, dradf2_twiddle_avx2, dradf2_twiddle_neon](
        ido: usize, l1: usize, cc: &[f32], ch: &mut [f32], wa1: &[f32]
    ) = dradf2_twiddle;
}
entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [noise_bark_sse41, noise_bark_avx2, noise_bark_neon](
        sums: &NoiseSums, b: &[i32], start: usize, end: usize,
        reflect: bool, offset: f32, noise: &mut [f32]
    ) = noise_bark;
}
entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [noise_fixed_sse41, noise_fixed_avx2, noise_fixed_neon](
        sums: &NoiseSums, fixed: i32, start: usize, end: usize,
        reflect: bool, offset: f32, noise: &mut [f32]
    ) = noise_fixed;
}
entry_points! {
    fn [noise_extrapolate_sse41, noise_extrapolate_avx2, noise_extrapolate_neon](
        fit: (f32, f32, f32), start: usize, end: usize, offset: f32,
        lower_only: bool, noise: &mut [f32]
    ) = noise_extrapolate;
}
entry_points! {
    fn [noise_compand_sse41, noise_compand_avx2, noise_compand_neon](
        logmdct: &[f32], work: &[f32], logmask: &mut [f32], compand: &[f32]
    ) = noise_compand;
}
entry_points! {
    fn [accumulate_fit_sse41, accumulate_fit_avx2, accumulate_fit_neon](
        flr: &[f32], mdct: &[f32], x0: i32, x1: i32, twofitatten: f32
    ) -> FitSums = accumulate_fit;
}
entry_points! {
    fn [log_mdct_sse41, log_mdct_avx2, log_mdct_neon](
        mdct: &[f32], logmdct: &mut [f32]
    ) = log_mdct;
}
entry_points! {
    fn [log_fft_sse41, log_fft_avx2, log_fft_neon](
        pcm: &mut [f32], n: usize, scale_db: f32
    ) = log_fft;
}

// ---------------------------------------------------------------------
// Dispatchers. Each takes the instruction set explicitly, so a caller reads
// `active_isa` once per block and the tests can run every arm side by side.
// ---------------------------------------------------------------------

/// [`MdctLookup::forward_scalar`].
pub(super) fn mdct_forward(
    isa: SimdIsa,
    m: &MdctLookup,
    input: &[f32],
    out: &mut [f32],
    w: &mut [f32],
) {
    if m.n < 64 {
        return m.forward_scalar(input, out, w);
    }
    dispatch!(
        isa,
        [mdct_sse41, mdct_avx2, mdct_neon](m, input, out, w),
        m.forward_scalar(input, out, w)
    );
}

/// The `ido == 1` case of [`super::smallft::dradf4_first_rows`].
pub(super) fn dradf4_ido1(isa: SimdIsa, l1: usize, cc: &[f32], ch: &mut [f32]) {
    dispatch!(
        isa,
        [dradf4_ido1_sse41, dradf4_ido1_avx2, dradf4_ido1_neon](l1, cc, ch),
        super::smallft::dradf4_first_rows(1, l1, cc, ch, 0)
    );
}

/// [`super::smallft::dradf4_twiddle_scalar`].
#[allow(clippy::too_many_arguments)]
pub(super) fn dradf4_twiddle(
    isa: SimdIsa,
    ido: usize,
    l1: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
    wa2: &[f32],
    wa3: &[f32],
) {
    dispatch!(
        isa,
        [
            dradf4_twiddle_sse41,
            dradf4_twiddle_avx2,
            dradf4_twiddle_neon
        ](ido, l1, cc, ch, wa1, wa2, wa3),
        super::smallft::dradf4_twiddle_scalar(ido, l1, cc, ch, wa1, wa2, wa3)
    );
}

/// [`super::smallft::dradf2_twiddle_scalar`].
pub(super) fn dradf2_twiddle(
    isa: SimdIsa,
    ido: usize,
    l1: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
) {
    dispatch!(
        isa,
        [
            dradf2_twiddle_sse41,
            dradf2_twiddle_avx2,
            dradf2_twiddle_neon
        ](ido, l1, cc, ch, wa1),
        super::smallft::dradf2_twiddle_scalar(ido, l1, cc, ch, wa1)
    );
}

/// [`super::psy::noise_bark_scalar`].
#[allow(clippy::too_many_arguments)]
pub(super) fn noise_bark(
    isa: SimdIsa,
    sums: &NoiseSums,
    b: &[i32],
    start: usize,
    end: usize,
    reflect: bool,
    offset: f32,
    noise: &mut [f32],
) {
    dispatch!(
        isa,
        [noise_bark_sse41, noise_bark_avx2, noise_bark_neon](
            sums, b, start, end, reflect, offset, noise
        ),
        super::psy::noise_bark_scalar(sums, b, start, end, reflect, offset, noise)
    );
}

/// [`super::psy::noise_fixed_scalar`].
#[allow(clippy::too_many_arguments)]
pub(super) fn noise_fixed(
    isa: SimdIsa,
    sums: &NoiseSums,
    fixed: i32,
    start: usize,
    end: usize,
    reflect: bool,
    offset: f32,
    noise: &mut [f32],
) {
    dispatch!(
        isa,
        [noise_fixed_sse41, noise_fixed_avx2, noise_fixed_neon](
            sums, fixed, start, end, reflect, offset, noise
        ),
        super::psy::noise_fixed_scalar(sums, fixed, start, end, reflect, offset, noise)
    );
}

/// [`super::psy::noise_extrapolate_scalar`].
pub(super) fn noise_extrapolate(
    isa: SimdIsa,
    fit: (f32, f32, f32),
    start: usize,
    end: usize,
    offset: f32,
    lower_only: bool,
    noise: &mut [f32],
) {
    dispatch!(
        isa,
        [
            noise_extrapolate_sse41,
            noise_extrapolate_avx2,
            noise_extrapolate_neon
        ](fit, start, end, offset, lower_only, noise),
        super::psy::noise_extrapolate_scalar(fit, start, end, offset, lower_only, noise)
    );
}

/// [`super::psy::noise_compand_scalar`].
pub(super) fn noise_compand(
    isa: SimdIsa,
    logmdct: &[f32],
    work: &[f32],
    logmask: &mut [f32],
    compand: &[f32],
) {
    dispatch!(
        isa,
        [noise_compand_sse41, noise_compand_avx2, noise_compand_neon](
            logmdct, work, logmask, compand
        ),
        super::psy::noise_compand_scalar(logmdct, work, logmask, compand, 0)
    );
}

/// [`super::floor1::accumulate_fit_scalar`].
pub(super) fn accumulate_fit(
    isa: SimdIsa,
    flr: &[f32],
    mdct: &[f32],
    x0: i32,
    x1: i32,
    twofitatten: f32,
) -> FitSums {
    dispatch!(
        isa,
        [
            accumulate_fit_sse41,
            accumulate_fit_avx2,
            accumulate_fit_neon
        ](flr, mdct, x0, x1, twofitatten),
        super::floor1::accumulate_fit_scalar(flr, mdct, x0, x1, twofitatten)
    )
}

/// [`super::mapping0::log_mdct_scalar`].
pub(super) fn log_mdct(isa: SimdIsa, mdct: &[f32], logmdct: &mut [f32]) {
    dispatch!(
        isa,
        [log_mdct_sse41, log_mdct_avx2, log_mdct_neon](mdct, logmdct),
        super::mapping0::log_mdct_scalar(mdct, logmdct, 0)
    );
}

/// [`super::mapping0::log_fft_scalar`].
pub(super) fn log_fft(isa: SimdIsa, pcm: &mut [f32], n: usize, scale_db: f32) {
    dispatch!(
        isa,
        [log_fft_sse41, log_fft_avx2, log_fft_neon](pcm, n, scale_db),
        super::mapping0::log_fft_scalar(pcm, n, scale_db, 1)
    );
}
