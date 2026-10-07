//! Color conversion shared by zvidlib's video codecs and its platform hardware
//! backends: the reconstructed 8-bit YUV frame and its `Rgba8` output
//! conversion used by the AV1, VP8 and VP9 decoders, the HEVC decoder's output
//! conversion, and the `Rgba8` to YUV 4:2:0 input conversion the HEVC and VP8
//! encoders share.
//!
//! This is an internal crate of [zvidlib](https://crates.io/crates/zvidlib).
//! Depend on `zvidlib` rather than on this crate directly.

#[doc(hidden)]
pub mod color_convert;
#[doc(hidden)]
pub mod colorconv;
pub mod frame;
#[doc(hidden)]
pub mod yuv_to_rgba;

#[allow(unused_imports)]
use zvidlib_core::*;

pub use frame::{FilterFrame, FilterPlane, MatrixCoefficients, convert_to_rgba8};
