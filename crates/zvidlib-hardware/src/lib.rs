//! The platform hardware video backends zvidlib's HEVC, VP8 and VP9 factories
//! select between: NVIDIA NVDEC on 64-bit Windows and Linux, Media Foundation
//! on Windows, and VideoToolbox on macOS, with the readback and reframing they
//! share.
//!
//! This is an internal crate of [zvidlib](https://crates.io/crates/zvidlib).
//! Depend on `zvidlib` rather than on this crate directly.

// Annex B and length-prefixed reframing for the platform encoders: Media Foundation on Windows
// and VideoToolbox on macOS.
#[cfg(any(windows, target_os = "macos", all(test, not(target_arch = "wasm32"))))]
#[doc(hidden)]
pub mod annexb;
#[cfg(all(any(windows, target_os = "linux"), target_pointer_width = "64"))]
#[doc(hidden)]
pub mod nvdec;
#[cfg(any(windows, all(target_os = "linux", target_pointer_width = "64")))]
#[doc(hidden)]
pub mod planar;
// internal — exposed for the hardware benchmark suite; not part of the stable API
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub mod readback;
#[cfg(target_os = "macos")]
#[doc(hidden)]
pub mod videotoolbox;
#[cfg(target_os = "macos")]
#[doc(hidden)]
pub mod videotoolbox_encoder;
#[cfg(target_os = "macos")]
#[doc(hidden)]
pub mod videotoolbox_vp9;
#[cfg(windows)]
#[doc(hidden)]
pub mod windows_mf;
#[cfg(windows)]
#[doc(hidden)]
pub mod windows_mf_encoder;

#[allow(unused_imports)]
use zvidlib_color::color_convert;
#[allow(unused_imports)]
use zvidlib_core::*;
#[allow(unused_imports)]
use zvidlib_hevc_syntax as engine;

// The codec crates these backends serve, for the tests that round-trip a
// backend's output through zvidlib's own encoders and decoders.
#[cfg(test)]
#[allow(unused_imports)]
use zvidlib_container::derive_codec_string;
#[cfg(test)]
#[allow(unused_imports)]
use zvidlib_hevc_decoder::native_hevc_video_decoder_factory;
#[cfg(test)]
#[allow(unused_imports)]
use zvidlib_hevc_encoder::native_hevc_video_encoder_factory;
#[cfg(test)]
#[allow(unused_imports)]
use zvidlib_vp9_decoder::native_vp9_video_decoder_factory;
