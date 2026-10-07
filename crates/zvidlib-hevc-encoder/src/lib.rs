//! Native HEVC/H.265 encoding: a dependency-free software encoder, and the
//! platform hardware encoders (VideoToolbox on macOS, Media Foundation on
//! Windows) where the host has one.
//!
//! The encoder is derived in part from `oxideav-h265`; see `NOTICE.md` and
//! `LICENSE`. This is an internal crate of
//! [zvidlib](https://crates.io/crates/zvidlib), which re-exports
//! `native_hevc_video_encoder_factory`. Depend on `zvidlib` rather than on this
//! crate directly.

#![cfg(not(target_arch = "wasm32"))]

// internal — exposed for the criterion benchmark suite; not part of the stable API
#[doc(hidden)]
pub mod bench;
#[doc(hidden)]
pub mod encoder;
// internal — the HEVC engine with its encoder half; not part of the stable API
#[doc(hidden)]
pub mod engine {
    pub use zvidlib_hevc_decoder::engine::*;
    pub mod encoder;
}

pub use encoder::native_hevc_video_encoder_factory;

#[allow(unused_imports)]
use zvidlib_core::*;
#[cfg(test)]
#[allow(unused_imports)]
use zvidlib_hevc_decoder::native_hevc_video_decoder_factory;
#[cfg(target_os = "macos")]
use zvidlib_hardware::videotoolbox_encoder;
#[cfg(windows)]
use zvidlib_hardware::windows_mf_encoder;
// The containers and conformance harness the tests read their fixtures with.
#[cfg(test)]
#[allow(unused_imports)]
use zvidlib_container::*;
