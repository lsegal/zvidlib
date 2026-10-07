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
    pub use zvidlib_hevc::engine::*;
    pub mod encoder;
}

pub use encoder::native_hevc_video_encoder_factory;

#[allow(unused_imports)]
use zvidlib_core::*;
#[cfg(all(feature = "hardware", target_os = "macos"))]
use zvidlib_hardware::videotoolbox_encoder;
#[cfg(all(feature = "hardware", windows))]
use zvidlib_hardware::windows_mf_encoder;
#[cfg(test)]
#[allow(unused_imports)]
use zvidlib_hevc_decoder::native_hevc_video_decoder_factory;
// The containers and conformance harness the tests read their fixtures with.
#[cfg(test)]
#[allow(unused_imports)]
use zvidlib_container::*;

/// The SIMD dispatch sites in this crate, each with the instruction set it
/// resolves to right now. `zvidlib::simd::active_by_site` reports every crate's
/// sites together and documents what each one covers.
#[doc(hidden)]
#[must_use]
pub fn simd_sites() -> Vec<(&'static str, zvidlib_core::SimdIsa)> {
    use engine::encoder::{quant_simd, rdcost, recon_simd};
    vec![
        ("hevc_rdcost", from_rdcost_isa(rdcost::isa())),
        ("hevc_recon", from_recon_isa(recon_simd::isa())),
        (
            "hevc_fwd_transform_quant",
            quant_simd::detected().as_simd_isa(),
        ),
    ]
}

fn from_recon_isa(isa: engine::encoder::recon_simd::Isa) -> zvidlib_core::SimdIsa {
    use engine::encoder::recon_simd::Isa;
    use zvidlib_core::SimdIsa;
    match isa {
        Isa::Scalar => SimdIsa::Scalar,
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => SimdIsa::Sse41,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => SimdIsa::Avx2,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => SimdIsa::Neon,
    }
}

fn from_rdcost_isa(isa: engine::encoder::rdcost::Isa) -> zvidlib_core::SimdIsa {
    use engine::encoder::rdcost::Isa;
    use zvidlib_core::SimdIsa;
    match isa {
        Isa::Scalar => SimdIsa::Scalar,
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => SimdIsa::Sse41,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => SimdIsa::Avx2,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => SimdIsa::Neon,
    }
}
