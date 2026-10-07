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

/// The SIMD dispatch sites in this crate, each with the instruction set it
/// resolves to right now. `zvidlib::simd::active_by_site` reports every crate's
/// sites together and documents what each one covers.
#[doc(hidden)]
#[must_use]
pub fn simd_sites() -> Vec<(&'static str, SimdIsa)> {
    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    let mut sites = Vec::new();
    // The HEVC conversions have no vector backend on `wasm32`.
    #[cfg(not(target_arch = "wasm32"))]
    {
        sites.push(("hevc_colorconv", colorconv::isa().as_simd_isa()));
        sites.push((
            "hevc_color_convert",
            from_color_convert_isa(color_convert::detected_isa()),
        ));
    }
    sites.push((
        "yuv_to_rgba",
        from_yuv_to_rgba_isa(yuv_to_rgba::detected_isa()),
    ));
    sites
}

#[cfg(not(target_arch = "wasm32"))]
fn from_color_convert_isa(isa: color_convert::Isa) -> SimdIsa {
    use color_convert::Isa;
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

fn from_yuv_to_rgba_isa(isa: yuv_to_rgba::Isa) -> SimdIsa {
    use yuv_to_rgba::Isa;
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
