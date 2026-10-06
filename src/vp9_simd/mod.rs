//! Runtime-dispatched SIMD kernels for the native VP9 profile 0 decoder.
//!
//! The decoder in [`crate::vp9_dec`] spends nearly all of its frame time in
//! four families of per-pixel loops: the inverse transforms and their
//! add-to-prediction (`vp9_dec::recon::inverse_transform_add`), the
//! sub-pixel convolution of inter prediction (`vp9_dec::recon::convolve`),
//! the intra predictors (`vp9_dec::recon::predict_intra`) and the loop filter
//! (`vp9_dec::loopfilter`). This module vectorizes all four for SSE4.1 and
//! AVX2 on `x86_64` and NEON on `aarch64`, selected at runtime, with the
//! scalar code kept as the reference and as the fallback on every other
//! target (including `wasm32`).
//!
//! The kernels are written once, generic over the 32-bit lane abstraction
//! [`crate::av1_simd::vector::I32x`] the AV1 kernels already use, and
//! instantiated per instruction set behind `#[target_feature]` wrappers. The
//! 4-wide shapes (4x4 transforms, 4-pixel-wide predictions) have no useful
//! 256-bit form, so AVX2 runs them through the SSE4.1 instantiation.
//!
//! # Bit-exactness
//!
//! Every kernel reproduces its scalar reference exactly, not approximately.
//! The scalar transforms are a transliteration of libvpx's C, computed in
//! `i64` and wrapped to 16 and 32 bits where libvpx's storage types wrap;
//! [`idct1d`] is the same text with each value a vector of `i32` lanes. Every
//! stage of the DCTs truncates to 16 bits, so their 32-bit lanes are exact
//! for any input. The ADSTs carry wider values, so each is range-checked
//! against the largest input for which no intermediate can leave `i32`
//! ([`transforms::ADST4_INPUT_LIMIT`] and its siblings), and a block that
//! exceeds it stays on the scalar path. Positions the vector loop filter
//! cannot reach without reading outside the plane, scaled horizontal
//! convolution, and anything else a kernel does not cover likewise stay
//! scalar. `tests` asserts equality against the scalar path on randomized
//! and edge-case input for every instruction set the host supports.

// Targets with no vector implementation (`wasm32` in particular) never
// instantiate the generic kernels and never reach the dispatchers' vector
// arms, so the resulting unused-code warnings are silenced there and only
// there.
#![cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code, unused_variables, unreachable_code, unused_imports)
)]

pub(crate) mod convolve;
pub(crate) mod idct1d;
pub(crate) mod intra;
pub(crate) mod loopfilter;
pub(crate) mod transforms;
pub(crate) mod wide;

pub use crate::av1_simd::SimdIsa;
use crate::av1_simd::vector;

/// The instruction set the VP9 kernels will actually use: the crate-wide
/// [`crate::simd::active`] value, so [`crate::simd::set_override`] reaches
/// them like every other dispatch site.
#[must_use]
pub fn active_isa() -> SimdIsa {
    crate::simd::active()
}

/// The narrower instruction set a kernel runs on when its shape is only four
/// pixels wide. AVX2's eight lanes would leave half of every vector idle, so
/// those shapes take the SSE4.1 instantiation, which every AVX2 host also
/// supports.
fn narrow(isa: SimdIsa, width: usize) -> SimdIsa {
    if isa == SimdIsa::Avx2 && width < 8 {
        SimdIsa::Sse41
    } else {
        isa
    }
}

// ---------------------------------------------------------------------
// Per-instruction-set entry points
//
// As in `av1_simd`, every generic kernel a wrapper names is
// `#[inline(always)]`. A `#[target_feature]` attribute applies to the
// function it is written on and not to what it calls, so a kernel the
// inliner declined would be compiled at the baseline instruction set and
// every intrinsic in it would become an out-of-line call (issues #336 and
// #341). `.github/scripts/check_simd_target_features.py` checks the emitted
// assembly for exactly that, for this module as well as `av1_simd`.
// ---------------------------------------------------------------------

macro_rules! simd_entry_points {
    (
        $(#[$meta:meta])*
        fn [$sse:ident, $avx:ident, $neon:ident]($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)?
            = $module:ident::$kernel:ident;
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse4.1")]
        $(#[$meta])*
        unsafe fn $sse($($arg: $ty),*) $(-> $ret)? {
            unsafe { $module::$kernel::<vector::Sse4>($($arg),*) }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        $(#[$meta])*
        unsafe fn $avx($($arg: $ty),*) $(-> $ret)? {
            unsafe { $module::$kernel::<vector::Avx2>($($arg),*) }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        $(#[$meta])*
        unsafe fn $neon($($arg: $ty),*) $(-> $ret)? {
            unsafe { $module::$kernel::<vector::Neon>($($arg),*) }
        }
    };
}

/// Calls the entry point for `isa`, or evaluates `$fallback` when `isa` is
/// scalar or not one this build can reach.
macro_rules! dispatch {
    ($isa:expr, [$sse:ident, $avx:ident, $neon:ident]($($arg:expr),* $(,)?), $fallback:expr) => {
        match $isa {
            // SAFETY: `isa` came from `crate::simd`, which only reports an
            // instruction set this host supports.
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

simd_entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [itx_sse41, itx_avx2, itx_neon](
        coefficients: &[i32], dest: &mut [u8], stride: usize,
        tx_size: u8, tx_type: u8, eob: usize, lossless: bool
    ) -> bool = transforms::inverse_transform_add;
}
simd_entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [convolve_sse41, convolve_avx2, convolve_neon](
        src: &[u8], origin: usize, src_stride: usize, dest: &mut [u8], stride: usize,
        w: usize, h: usize, kernel: &convolve::Kernel, x_frac: i32, x_step: i32,
        y_frac: i32, y_step: i32, average: bool, temp: &mut [u8; 64 * 135]
    ) -> bool = convolve::convolve;
}
simd_entry_points! {
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    fn [intra_sse41, intra_avx2, intra_neon](
        dest: &mut [u8], stride: usize, bs: usize, mode: u8,
        above: &[u8; 65], left: &[u8; 32], have_left: bool, have_above: bool
    ) -> bool = intra::predict_intra;
}
simd_entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [lpf_sse41, lpf_avx2, lpf_neon](
        data: &mut [u8], start: isize, step: isize, along: isize, count: usize,
        thresholds: loopfilter::Thresholds, taps: loopfilter::Taps
    ) -> bool = loopfilter::filter_edge;
}

// ---------------------------------------------------------------------
// Safe dispatchers used by `vp9_dec`
//
// Each returns `false`, having changed nothing, when the caller has to run
// the scalar reference instead: `isa` is scalar or unreachable here, or the
// input is outside what the vector kernel covers.
// ---------------------------------------------------------------------

/// Inverse transforms one transform block and adds it to the prediction in
/// `dest`, as `vp9_dec::recon::inverse_transform_add` does.
#[allow(clippy::too_many_arguments)]
pub(crate) fn inverse_transform_add(
    isa: SimdIsa,
    coefficients: &[i32],
    dest: &mut [u8],
    stride: usize,
    tx_size: u8,
    tx_type: u8,
    eob: usize,
    lossless: bool,
) -> bool {
    let n = if lossless {
        4
    } else {
        4usize << tx_size.min(3)
    };
    if !transforms::covers(coefficients, dest, stride, n) {
        return false;
    }
    dispatch!(
        narrow(isa, n),
        [itx_sse41, itx_avx2, itx_neon](
            coefficients,
            dest,
            stride,
            tx_size,
            tx_type,
            eob,
            lossless
        ),
        false
    )
}

/// The sub-pixel convolution of inter prediction, as
/// `vp9_dec::recon::convolve` computes it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn convolve(
    isa: SimdIsa,
    src: &[u8],
    origin: usize,
    src_stride: usize,
    dest: &mut [u8],
    stride: usize,
    w: usize,
    h: usize,
    kernel: &convolve::Kernel,
    x_frac: i32,
    x_step: i32,
    y_frac: i32,
    y_step: i32,
    average: bool,
    temp: &mut [u8; 64 * 135],
) -> bool {
    if w % 4 != 0 || w > 64 {
        return false;
    }
    dispatch!(
        narrow(isa, w),
        [convolve_sse41, convolve_avx2, convolve_neon](
            src, origin, src_stride, dest, stride, w, h, kernel, x_frac, x_step, y_frac, y_step,
            average, temp
        ),
        false
    )
}

/// One intra prediction, as `vp9_dec::recon::predict_intra` computes it.
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub(crate) fn predict_intra(
    isa: SimdIsa,
    dest: &mut [u8],
    stride: usize,
    bs: usize,
    mode: u8,
    above: &[u8; 65],
    left: &[u8; 32],
    have_left: bool,
    have_above: bool,
) -> bool {
    if !matches!(bs, 4 | 8 | 16 | 32) || (bs - 1) * stride + bs > dest.len() {
        return false;
    }
    dispatch!(
        narrow(isa, bs),
        [intra_sse41, intra_avx2, intra_neon](
            dest, stride, bs, mode, above, left, have_left, have_above
        ),
        false
    )
}

/// One loop filter application along `count` pixels of an edge: `step`
/// crosses the edge and `along` moves along it, as in
/// `vp9_dec::loopfilter`'s `lpf4`, `lpf8` and `lpf16`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_edge(
    isa: SimdIsa,
    data: &mut [u8],
    start: isize,
    step: isize,
    along: isize,
    count: usize,
    thresholds: loopfilter::Thresholds,
    taps: loopfilter::Taps,
) -> bool {
    if !loopfilter::covers(data.len(), start, step, along, count, taps) {
        return false;
    }
    dispatch!(
        isa,
        [lpf_sse41, lpf_avx2, lpf_neon](data, start, step, along, count, thresholds, taps),
        false
    )
}

#[cfg(test)]
mod tests;
