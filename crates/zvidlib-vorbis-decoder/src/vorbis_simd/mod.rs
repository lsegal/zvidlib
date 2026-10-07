//! Runtime-dispatched SIMD kernels for the Vorbis decoder's synthesis.
//!
//! The decoder in `crate::vorbis_decoder` spends its arithmetic in a handful
//! of per-sample loops over `f32` spectra and PCM: the inverse MDCT (Vorbis I
//! section 4.3.7), windowed overlap-add and the output clamp (4.3.8), inverse
//! channel coupling (4.3.5) and the floor-times-residue product (4.3.6). This
//! module has an SSE4.1 and an AVX2 kernel for each on `x86_64` and a NEON one
//! on `aarch64`, with the scalar reference kept as the fallback on every other
//! target, `wasm32` included.
//!
//! The vendored decoder is `#![forbid(unsafe_code)]`, so the kernels live here
//! and it calls the safe dispatchers below.
//!
//! # Bit-exactness
//!
//! Every vector kernel is a lane-by-lane transliteration of its scalar
//! reference: the same IEEE 754 operations on the same operands in the same
//! order, and no fused multiply-add, so each lane rounds exactly as the scalar
//! code rounds that element. The tests in `tests.rs` compare the two bit for
//! bit on randomized and edge-case input for every instruction set the host
//! supports.
//!
//! The inverse MDCT is this crate's own ([`Imdct`]) rather than
//! `symphonia_core`'s, which is the open question issue #572 settled: the only
//! vector FFT Symphonia offers is `rustfft`'s, which chooses its own
//! instruction set where [`crate::simd::set_override`] cannot reach it and
//! does not round like its scalar FFT does.
//!
//! # Selecting an instruction set
//!
//! The kernels follow the crate-wide [`crate::simd`] override, reported as the
//! `vorbis_decode` site of [`crate::simd::active_by_site`].

// Targets with no vector implementation (`wasm32` in particular) never
// instantiate the generic kernels, so the resulting unused-code warnings are
// silenced there and only there.
#![cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code, unused_imports)
)]

pub mod bench;
mod imdct;
mod kernels;
mod vector;

#[cfg(test)]
mod tests;

pub(crate) use imdct::Imdct;

use crate::simd::SimdIsa;
use imdct::Plan;
use kernels::Scratch;

/// The instruction set the Vorbis kernels will use: the crate-wide
/// [`crate::simd::active`] value, so an override reaches them immediately.
#[must_use]
#[doc(hidden)]
pub fn active_isa() -> SimdIsa {
    crate::simd::active()
}

// ---------------------------------------------------------------------------
// Per-instruction-set entry points
//
// Each kernel gets one `#[target_feature]` wrapper per instruction set, and
// every generic kernel a wrapper names is `#[inline(always)]`. As in
// `crate::av1_simd`, that is a codegen requirement rather than a speed hint:
// `#[target_feature]` widens only the function it is written on, so a generic
// kernel the inliner declined would be compiled at the target's baseline,
// with every intrinsic an out-of-line call (issues #336 and #341). Only an
// x86_64 benchmark would notice, so
// `.github/scripts/check_simd_target_features.py` reads it off the emitted
// assembly in CI instead, for `vorbis_simd` as for `av1_simd`.
//
// The wrappers themselves are `#[inline(never)]`: a `#[target_feature]`
// function cannot be inlined into the dispatchers, which lack the feature, and
// saying so keeps each arm one symbol that the check above can name.
// ---------------------------------------------------------------------------

macro_rules! entry_points {
    (
        fn [$sse:ident, $avx:ident, $neon:ident]($($arg:ident : $ty:ty),* $(,)?)
            = [$sse_kernel:expr, $avx_kernel:expr, $neon_kernel:expr];
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse4.1")]
        #[inline(never)]
        unsafe fn $sse($($arg: $ty),*) {
            unsafe { $sse_kernel($($arg),*) }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[inline(never)]
        unsafe fn $avx($($arg: $ty),*) {
            unsafe { $avx_kernel($($arg),*) }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline(never)]
        unsafe fn $neon($($arg: $ty),*) {
            unsafe { $neon_kernel($($arg),*) }
        }
    };
}

entry_points! {
    fn [imdct_sse41, imdct_avx2, imdct_neon](
        plan: &Plan, spec: &[f32], out: &mut [f32], scratch: Scratch<'_>
    ) = [
        kernels::imdct::<vector::Sse4, vector::Sse4>,
        kernels::imdct::<vector::Avx2, vector::Sse4>,
        kernels::imdct::<vector::Neon, vector::Neon>
    ];
}
entry_points! {
    fn [overlap_add_sse41, overlap_add_avx2, overlap_add_neon](
        out: &mut [f32], left: &[f32], right: &[f32], win: &[f32], win_rev: &[f32]
    ) = [
        kernels::overlap_add::<vector::Sse4>,
        kernels::overlap_add::<vector::Avx2>,
        kernels::overlap_add::<vector::Neon>
    ];
}
entry_points! {
    fn [clamp_unit_sse41, clamp_unit_avx2, clamp_unit_neon](buf: &mut [f32]) = [
        kernels::clamp_unit::<vector::Sse4>,
        kernels::clamp_unit::<vector::Avx2>,
        kernels::clamp_unit::<vector::Neon>
    ];
}
entry_points! {
    fn [coupling_sse41, coupling_avx2, coupling_neon](magnitude: &mut [f32], angle: &mut [f32]) = [
        kernels::inverse_coupling::<vector::Sse4>,
        kernels::inverse_coupling::<vector::Avx2>,
        kernels::inverse_coupling::<vector::Neon>
    ];
}
entry_points! {
    fn [apply_floor_sse41, apply_floor_avx2, apply_floor_neon](floor: &mut [f32], residue: &[f32]) = [
        kernels::apply_floor::<vector::Sse4>,
        kernels::apply_floor::<vector::Avx2>,
        kernels::apply_floor::<vector::Neon>
    ];
}

/// Expands to a `match` on `$isa` that calls the matching entry point, or the
/// scalar reference on a host or target without one.
///
/// Every caller passes either [`active_isa`] or an instruction set taken from
/// [`crate::simd::available`], both of which this host can execute, which is
/// what makes the `unsafe` calls sound.
macro_rules! dispatch {
    ($isa:expr, [$sse:ident, $avx:ident, $neon:ident], $scalar:path, ($($arg:expr),*)) => {
        match $isa {
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx2 => unsafe { $avx($($arg),*) },
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Sse41 => unsafe { $sse($($arg),*) },
            #[cfg(target_arch = "aarch64")]
            SimdIsa::Neon => unsafe { $neon($($arg),*) },
            _ => $scalar($($arg),*),
        }
    };
}

/// Windowed overlap-add: `out[i] = left[i] * win_rev[i] + right[i] * win[i]`,
/// with `win_rev` being `win` reversed. All five slices are the same length.
pub(crate) fn overlap_add(
    out: &mut [f32],
    left: &[f32],
    right: &[f32],
    win: &[f32],
    win_rev: &[f32],
) {
    overlap_add_with(active_isa(), out, left, right, win, win_rev);
}

pub(crate) fn overlap_add_with(
    isa: SimdIsa,
    out: &mut [f32],
    left: &[f32],
    right: &[f32],
    win: &[f32],
    win_rev: &[f32],
) {
    let len = out.len();
    assert!(left.len() == len && right.len() == len && win.len() == len && win_rev.len() == len);
    dispatch!(
        isa,
        [overlap_add_sse41, overlap_add_avx2, overlap_add_neon],
        kernels::overlap_add_scalar,
        (out, left, right, win, win_rev)
    );
}

/// Clamps every sample to `-1.0..=1.0` as [`f32::clamp`] does.
pub(crate) fn clamp_unit(buf: &mut [f32]) {
    clamp_unit_with(active_isa(), buf);
}

pub(crate) fn clamp_unit_with(isa: SimdIsa, buf: &mut [f32]) {
    dispatch!(
        isa,
        [clamp_unit_sse41, clamp_unit_avx2, clamp_unit_neon],
        kernels::clamp_unit_scalar,
        (buf)
    );
}

/// Undoes one square-polar coupling step (Vorbis I section 4.3.5) in place.
/// Both slices are the same length.
pub(crate) fn inverse_coupling(magnitude: &mut [f32], angle: &mut [f32]) {
    inverse_coupling_with(active_isa(), magnitude, angle);
}

pub(crate) fn inverse_coupling_with(isa: SimdIsa, magnitude: &mut [f32], angle: &mut [f32]) {
    assert_eq!(magnitude.len(), angle.len());
    dispatch!(
        isa,
        [coupling_sse41, coupling_avx2, coupling_neon],
        kernels::inverse_coupling_scalar,
        (magnitude, angle)
    );
}

/// Multiplies the floor curve by the residue in place (Vorbis I section
/// 4.3.6). Both slices are the same length.
pub(crate) fn apply_floor(floor: &mut [f32], residue: &[f32]) {
    apply_floor_with(active_isa(), floor, residue);
}

pub(crate) fn apply_floor_with(isa: SimdIsa, floor: &mut [f32], residue: &[f32]) {
    assert_eq!(floor.len(), residue.len());
    dispatch!(
        isa,
        [apply_floor_sse41, apply_floor_avx2, apply_floor_neon],
        kernels::apply_floor_scalar,
        (floor, residue)
    );
}
