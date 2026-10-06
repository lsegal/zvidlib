//! Runtime-dispatched SIMD kernels for the VP8 encoder and the reconstruction
//! it shares with the decoder.
//!
//! VP8 coding is pure Rust here, and its per-sample loops are where its time
//! goes: motion search and mode decision measure every candidate with SAD and
//! SATD, every coded block goes through the forward DCT and quantization, and
//! every macroblock is reconstructed and loop-filtered exactly as a decoder
//! would. This module vectorizes those kernels for SSE4.1 and AVX2 on `x86_64`
//! and NEON on `aarch64`, and leaves the scalar code in `frame_encoder.rs`,
//! `predict.rs` and `loop_filter.rs` as the reference and as the fallback on
//! every other target, `wasm32` included.
//!
//! The kernels fall into two dispatch sites, registered with
//! [`crate::simd::active_by_site`]:
//!
//! - `vp8_encode` ([`encode_isa`]): the encoder-only kernels - [`sad16`],
//!   [`satd4`] and [`satd`], the residual and forward DCT ([`residual_dct`]),
//!   the forward Walsh-Hadamard transform ([`forward_walsh`]) and
//!   [`quantize`].
//! - `vp8_recon` ([`recon_isa`]): reconstruction, which the encoder runs
//!   through the decoder's own code so its reference frames are the decoder's -
//!   the inverse DCT ([`idct_add`]) and Walsh-Hadamard transform
//!   ([`inverse_walsh`]), six-tap and bilinear inter prediction ([`sixtap`]),
//!   `TM_PRED` ([`tm_predict`]) and the loop filter
//!   ([`filter_horizontal_edge`], [`filter_vertical_edge`]). The decoder takes
//!   the same kernels.
//!
//! The 4x4 subblock intra predictors stay scalar: each of the ten modes is its
//! own pattern of averages over at most thirteen edge samples, with no shape a
//! vector covers, and they write sixteen bytes.
//!
//! # Bit-exactness
//!
//! Every kernel computes exactly what its scalar reference computes, so the
//! encoder writes byte-identical bitstreams and the decoder reconstructs
//! byte-identical pictures under every instruction set. `tests.rs` holds each
//! kernel to its reference on random and extreme inputs, and the encoder's
//! tests encode the same frames under every available instruction set and
//! compare the bytes.
//!
//! # Dispatch
//!
//! Each dispatcher here returns `None` (or `false`) when the active
//! instruction set has no vector path, and the caller then runs its scalar
//! code. The instruction set is [`crate::simd::active`], so
//! [`crate::simd::set_override`] reaches these kernels like every other site.
//! AVX2 hosts run the 4x4 kernels - the transforms and SATD - through the
//! 128-bit body, as `av1_simd` does its 4-point transforms (#342): a 4x4 block
//! is four 4-lane rows with no 256-bit shape, and the transposes between their
//! passes are 128-bit operations. The pixel kernels use all eight AVX2 lanes.

// Targets with no vector implementation never reach the vector arms, so the
// resulting unused-code warnings are silenced there and only there.
#![cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code, unused_variables, unreachable_code)
)]

mod kernels;
mod sad;
#[cfg(test)]
mod tests;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::av1_simd::vector;
use crate::simd::SimdIsa;

pub(crate) use kernels::EdgeLimits;

/// The instruction set the `vp8_encode` kernels use.
#[must_use]
pub(crate) fn encode_isa() -> SimdIsa {
    crate::simd::active()
}

/// The instruction set the `vp8_recon` kernels use.
#[must_use]
pub(crate) fn recon_isa() -> SimdIsa {
    crate::simd::active()
}

/// The instruction set a kernel narrower than eight lanes runs under: AVX2
/// hosts take the SSE4.1 body.
fn narrow(isa: SimdIsa) -> SimdIsa {
    #[cfg(target_arch = "x86_64")]
    if isa == SimdIsa::Avx2 {
        // Every AVX2 CPU has SSE4.1, but an override can name AVX2 without
        // detection having seen it, so the redirect is checked.
        return if std::is_x86_feature_detected!("sse4.1") {
            SimdIsa::Sse41
        } else {
            SimdIsa::Scalar
        };
    }
    isa
}

// ---------------------------------------------------------------------
// Per-instruction-set entry points
//
// One `#[target_feature]` wrapper per kernel and instruction set, exactly as
// in `av1_simd`: the generic kernels in `kernels.rs` are `#[inline(always)]`,
// and only once inlined into a wrapper are they compiled with its feature. A
// kernel the inliner declined would run at the baseline instruction set,
// several times slower than the scalar code it replaces, while staying
// bit-exact (#336), so `.github/scripts/check_simd_target_features.py` checks
// the emitted assembly for exactly that (#341).
// ---------------------------------------------------------------------

macro_rules! entry_points {
    (
        $(#[$meta:meta])*
        fn [$sse:ident, $avx:ident, $neon:ident]($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)?
            = $kernel:ident, avx2 = $avx_vector:ident;
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse4.1")]
        $(#[$meta])*
        unsafe fn $sse($($arg: $ty),*) $(-> $ret)? {
            unsafe { kernels::$kernel::<vector::Sse4>($($arg),*) }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        $(#[$meta])*
        unsafe fn $avx($($arg: $ty),*) $(-> $ret)? {
            unsafe { kernels::$kernel::<vector::$avx_vector>($($arg),*) }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        $(#[$meta])*
        unsafe fn $neon($($arg: $ty),*) $(-> $ret)? {
            unsafe { kernels::$kernel::<vector::Neon>($($arg),*) }
        }
    };
}

entry_points! {
    fn [satd4_sse41, satd4_avx2, satd4_neon](block: &[i16; 16]) -> u32
        = satd4, avx2 = Sse4;
}
entry_points! {
    fn [satd_sse41, satd_avx2, satd_neon](
        source: &[u8], prediction: &[u8], origin: usize, stride: usize, size: usize
    ) -> u32 = satd, avx2 = Sse4;
}
entry_points! {
    fn [residual_dct_sse41, residual_dct_avx2, residual_dct_neon](
        source: &[u8], prediction: &[u8], offset: usize, stride: usize
    ) -> [i16; 16] = residual_dct, avx2 = Sse4;
}
entry_points! {
    fn [fwht_sse41, fwht_avx2, fwht_neon](input: &[i16; 16]) -> [i16; 16]
        = forward_walsh, avx2 = Sse4;
}
entry_points! {
    fn [quantize_sse41, quantize_avx2, quantize_neon](
        coefficients: &[i16; 16], factors: [i32; 2], first: usize
    ) -> ([i16; 16], [i16; 16]) = quantize, avx2 = Avx2;
}
entry_points! {
    fn [idct_add_sse41, idct_add_avx2, idct_add_neon](
        coefficients: &[i16; 16], plane: &mut [u8], offset: usize, stride: usize
    ) = idct_add, avx2 = Sse4;
}
entry_points! {
    fn [iwht_sse41, iwht_avx2, iwht_neon](input: &[i16; 16]) -> [i16; 16]
        = inverse_walsh, avx2 = Sse4;
}
entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [sixtap_sse41, sixtap_avx2, sixtap_neon](
        window: &[u8], width: usize, height: usize, horizontal: &[i32; 6],
        vertical: &[i32; 6], output: &mut [u8], destination: usize, stride: usize
    ) = sixtap, avx2 = Avx2;
}
entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [tm_sse41, tm_avx2, tm_neon](
        above: &[u8], left: &[u8], corner: i32, size: usize, plane: &mut [u8],
        offset: usize, stride: usize
    ) = tm_predict, avx2 = Avx2;
}
entry_points! {
    fn [filter_h_sse41, filter_h_avx2, filter_h_neon](
        data: &mut [u8], at: usize, stride: usize, count: usize, limits: EdgeLimits
    ) = filter_horizontal_edge, avx2 = Avx2;
}
entry_points! {
    fn [filter_v_sse41, filter_v_avx2, filter_v_neon](
        data: &mut [u8], at: usize, stride: usize, count: usize, limits: EdgeLimits
    ) = filter_vertical_edge, avx2 = Avx2;
}

#[cfg(target_arch = "aarch64")]
use sad::arm::sad16_neon;
#[cfg(target_arch = "x86_64")]
use sad::x86::{sad16_avx2, sad16_sse41};

/// Expands to a `match` over `$isa` that calls the matching wrapper and
/// evaluates to its result in `Some`, or to `None` on scalar or on an
/// instruction set this build has no wrapper for.
macro_rules! dispatch {
    ($isa:expr, [$sse:ident, $avx:ident, $neon:ident]($($arg:expr),* $(,)?)) => {
        match $isa {
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Sse41 => Some(unsafe { $sse($($arg),*) }),
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx2 => Some(unsafe { $avx($($arg),*) }),
            #[cfg(target_arch = "aarch64")]
            SimdIsa::Neon => Some(unsafe { $neon($($arg),*) }),
            _ => None,
        }
    };
}

// ---------------------------------------------------------------------
// Safe dispatchers
//
// The instruction set each match arm names has been established by
// `crate::simd::active`, which only ever reports one this host can execute,
// and every kernel reads and writes its slices through bounds-checked
// indexing, so a dispatcher is safe to call with any arguments.
// ---------------------------------------------------------------------

/// The SAD of two 16x16 blocks at the front of `source` and `prediction`.
pub(crate) fn sad16(
    source: &[u8],
    source_stride: usize,
    prediction: &[u8],
    prediction_stride: usize,
) -> Option<u32> {
    dispatch!(
        encode_isa(),
        [sad16_sse41, sad16_avx2, sad16_neon](source, source_stride, prediction, prediction_stride)
    )
}

/// `satd4` of a residual block.
pub(crate) fn satd4(block: &[i16; 16]) -> Option<u32> {
    dispatch!(encode_isa(), [satd4_sse41, satd4_avx2, satd4_neon](block))
}

/// The SATD of the `size`x`size` block at `origin` of two planes of one
/// stride.
pub(crate) fn satd(
    source: &[u8],
    prediction: &[u8],
    origin: usize,
    stride: usize,
    size: usize,
) -> Option<u32> {
    dispatch!(
        encode_isa(),
        [satd_sse41, satd_avx2, satd_neon](source, prediction, origin, stride, size)
    )
}

/// The forward DCT of the residual of the 4x4 block at `offset`.
pub(crate) fn residual_dct(
    source: &[u8],
    prediction: &[u8],
    offset: usize,
    stride: usize,
) -> Option<[i16; 16]> {
    dispatch!(
        encode_isa(),
        [residual_dct_sse41, residual_dct_avx2, residual_dct_neon](
            source, prediction, offset, stride
        )
    )
}

/// The forward Walsh-Hadamard transform of the luma DC coefficients.
pub(crate) fn forward_walsh(input: &[i16; 16]) -> Option<[i16; 16]> {
    dispatch!(encode_isa(), [fwht_sse41, fwht_avx2, fwht_neon](input))
}

/// The levels and dequantized values of a block's coefficients.
pub(crate) fn quantize(
    coefficients: &[i16; 16],
    factors: [i32; 2],
    first: usize,
) -> Option<([i16; 16], [i16; 16])> {
    dispatch!(
        encode_isa(),
        [quantize_sse41, quantize_avx2, quantize_neon](coefficients, factors, first)
    )
}

/// Adds the inverse DCT of `coefficients` to the 4x4 block at `offset`,
/// returning whether a vector kernel did.
pub(crate) fn idct_add(
    coefficients: &[i16; 16],
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) -> bool {
    matches!(
        dispatch!(
            recon_isa(),
            [idct_add_sse41, idct_add_avx2, idct_add_neon](coefficients, plane, offset, stride)
        ),
        Some(())
    )
}

/// The inverse Walsh-Hadamard transform of the Y2 block.
pub(crate) fn inverse_walsh(input: &[i16; 16]) -> Option<[i16; 16]> {
    dispatch!(recon_isa(), [iwht_sse41, iwht_avx2, iwht_neon](input))
}

/// Filters `window` into the `width`x`height` block at `destination`,
/// returning whether a vector kernel did. `width` is 4, 8 or 16.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sixtap(
    window: &[u8],
    width: usize,
    height: usize,
    horizontal: &[i32; 6],
    vertical: &[i32; 6],
    output: &mut [u8],
    destination: usize,
    stride: usize,
) -> bool {
    let isa = match recon_isa() {
        isa if width % 8 != 0 => narrow(isa),
        isa => isa,
    };
    matches!(
        dispatch!(
            isa,
            [sixtap_sse41, sixtap_avx2, sixtap_neon](
                window,
                width,
                height,
                horizontal,
                vertical,
                output,
                destination,
                stride
            )
        ),
        Some(())
    )
}

/// `TM_PRED` of a `size`x`size` block (8 or 16), returning whether a vector
/// kernel did it.
pub(crate) fn tm_predict(
    above: &[u8],
    left: &[u8],
    corner: i32,
    size: usize,
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) -> bool {
    matches!(
        dispatch!(
            recon_isa(),
            [tm_sse41, tm_avx2, tm_neon](above, left, corner, size, plane, offset, stride)
        ),
        Some(())
    )
}

/// Loop-filters `count` (8 or 16) segments across a horizontal edge,
/// returning whether a vector kernel did.
pub(crate) fn filter_horizontal_edge(
    data: &mut [u8],
    at: usize,
    stride: usize,
    count: usize,
    limits: EdgeLimits,
) -> bool {
    matches!(
        dispatch!(
            recon_isa(),
            [filter_h_sse41, filter_h_avx2, filter_h_neon](data, at, stride, count, limits)
        ),
        Some(())
    )
}

/// Loop-filters `count` (8 or 16) segments across a vertical edge, returning
/// whether a vector kernel did.
pub(crate) fn filter_vertical_edge(
    data: &mut [u8],
    at: usize,
    stride: usize,
    count: usize,
    limits: EdgeLimits,
) -> bool {
    matches!(
        dispatch!(
            recon_isa(),
            [filter_v_sse41, filter_v_avx2, filter_v_neon](data, at, stride, count, limits)
        ),
        Some(())
    )
}
