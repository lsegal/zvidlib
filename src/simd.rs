//! The process-wide SIMD instruction-set override shared by every codec kernel.
//!
//! zvidlib's pure-Rust HEVC, AV1, VP8, VP9 and Vorbis codecs dispatch their hot
//! loops to runtime-detected vector kernels in several independent places: the
//! AV1 transforms and in-loop filters ([`zvidlib_av1::av1_simd`]), AV1 inter
//! prediction ([`zvidlib_av1::av1_mc`]), AV1 intra prediction
//! ([`zvidlib_av1::av1_intra_pred`]), the Vorbis encoder's analysis stages, the VP8
//! encoder's distortion metrics, forward transforms and quantization and the
//! reconstruction and loop filter it shares with the VP8 decoder, the VP8
//! decoder's subblock intra prediction and DC-only inverse DCT, the VP9
//! decoder's inverse transforms, prediction and loop filter
//! (`zvidlib_vp9_decoder::vp9_simd`), the HEVC engine's inter/intra prediction, in-loop
//! filters, inverse transforms, encoder-side distortion metrics, and
//! encoder-side color conversion, the AV1, VP8 and VP9 decoders' shared output
//! color conversion ([`crate::av1_filters::convert_to_rgba8`]), the Vorbis
//! decoder's synthesis (`zvidlib_vorbis_decoder::vorbis_simd`), and the native VP9 encoder's
//! pixel kernels. Each of those sites caches its own CPU feature probe, which
//! is what you want in production but makes "run this workload with SIMD off"
//! impossible to express from outside the crate.
//!
//! This module is that single switch. [`set_override`] pins **every** kernel in
//! the crate to one [`SimdIsa`] (or restores per-site automatic detection with
//! `None`), [`active`] reports what the kernels will actually use, and
//! [`available`] lists every instruction set this host can execute. The
//! override is consulted ahead of each site's cached detection rather than
//! baked into it, so it takes effect immediately and can be changed any number
//! of times in one process — which is exactly what the criterion benchmarks in
//! `benches/` need to time a scalar arm and a vector arm back to back.
//!
//! Every vector backend in the crate is documented and tested as bit-exact with
//! its scalar reference, so the override only ever changes performance, never
//! decoded or encoded output.
//!
//! ```
//! use zvidlib::simd::{self, SimdIsa};
//!
//! // Force the portable scalar path everywhere.
//! simd::set_override(Some(SimdIsa::Scalar));
//! assert_eq!(simd::active(), SimdIsa::Scalar);
//!
//! // Back to per-host detection.
//! simd::set_override(None);
//! assert_eq!(simd::active(), simd::detected());
//! ```

#[cfg(test)]
use zvidlib_core::simd::test_lock;
pub use zvidlib_core::simd::{SimdIsa, active, available, detected, set_override};

/// The instruction set every individual dispatch site resolves to right now,
/// paired with a stable name for that site.
///
/// [`active`] reports what the override *asks* for; this reports what each
/// family of kernels will *actually* run, read back from that family's own
/// selector. The two agreeing is the property that makes a scalar-vs-SIMD
/// benchmark meaningful, and on a host where the scalar reference happens to
/// auto-vectorize well it is the only way to confirm the switch landed —
/// timings alone cannot distinguish "the override did not reach this kernel"
/// from "this kernel's vector path is not faster here".
///
/// The site names are stable and safe to assert on:
///
/// | Site | Kernels |
/// | --- | --- |
/// | `av1_simd` | AV1 transforms and in-loop filters |
/// | `av1_mc` | AV1 motion compensation (the level [`zvidlib_av1::av1_mc::McContext::new`] picks) |
/// | `av1_intra_pred` | AV1 intra prediction and residual reconstruction |
/// | `av1_coeff_ctx` | AV1 encoder-side coefficient context derivation (§8.3.2) |
/// | `vorbis_encode` | Vorbis encoder forward MDCT, real FFT, noise-mask fits, floor fitting and log spectra |
/// | `vp8_encode` | VP8 encoder-side SAD and SATD, residual and forward DCT, forward WHT and quantization |
/// | `vp8_recon` | VP8 inverse transforms, inter and `TM_PRED` prediction and loop filter, shared by the encoder and decoder |
/// | `vp8_decode` | VP8 4x4 subblock intra prediction and DC-only inverse DCT |
/// | `vp9_decode` | VP9 decoder inverse transforms, inter and intra prediction, and loop filter |
/// | `vp9_encode` | VP9 encoder forward transforms, quantization, distortion metrics, intra/inter prediction and RGBA8 to YUV420 input conversion |
/// | `hevc_prediction_filters` | HEVC inter/intra prediction and in-loop filters |
/// | `hevc_transforms` | HEVC inverse transforms and dequantization |
/// | `hevc_rdcost` | HEVC encoder-side distortion metrics |
/// | `hevc_recon` | HEVC encoder-side reconstruction and SAO parameter search |
/// | `hevc_fwd_transform_quant` | HEVC encoder-side forward transform and quantization |
/// | `hevc_colorconv` | HEVC encoder-side RGBA8 to YUV420 input conversion |
/// | `hevc_color_convert` | HEVC decoder output YUV420-to-RGBA conversion |
/// | `yuv_to_rgba` | AV1, VP8 and VP9 decoder output YUV-to-RGBA conversion |
/// | `vorbis_decode` | Vorbis inverse MDCT, overlap-add, inverse coupling and floor product |
///
/// The `hevc_*` sites are absent on `wasm32`, where the HEVC kernels have no
/// vector backend and always run the scalar path.
#[must_use]
pub fn active_by_site() -> Vec<(&'static str, SimdIsa)> {
    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    let mut sites = vec![
        ("av1_simd", zvidlib_av1::av1_simd::active_isa()),
        (
            "av1_mc",
            from_mc_level(zvidlib_av1::av1_mc::default_level()),
        ),
        (
            "av1_intra_pred",
            from_intra_simd(zvidlib_av1::av1_intra_pred::av1_intra_simd()),
        ),
        ("av1_coeff_ctx", zvidlib_av1::av1_simd::coeff::active_isa()),
        (
            "vorbis_encode",
            zvidlib_vorbis_encoder::vorbis_encoder::simd::active_isa(),
        ),
        ("vp8_encode", zvidlib_vp8::simd::encode_isa()),
        ("vp8_recon", zvidlib_vp8::simd::recon_isa()),
        ("vp8_decode", zvidlib_vp8::simd::decode_isa()),
        ("vp9_decode", zvidlib_vp9_decoder::vp9_simd::active_isa()),
        (
            "vp9_encode",
            from_vp9_encode_isa(zvidlib_vp9_encoder::simd::isa()),
        ),
    ];
    #[cfg(not(target_arch = "wasm32"))]
    {
        use zvidlib_color::color_convert;
        use zvidlib_color::colorconv;
        use zvidlib_hevc_decoder::engine::{simd as hevc_simd, transform_simd};
        use zvidlib_hevc_encoder::engine::encoder::{rdcost, recon_simd};
        sites.push((
            "hevc_prediction_filters",
            from_hevc_isa(hevc_simd::detected_isa()),
        ));
        sites.push((
            "hevc_transforms",
            from_hevc_backend(transform_simd::detected()),
        ));
        sites.push(("hevc_rdcost", from_rdcost_isa(rdcost::isa())));
        sites.push(("hevc_recon", from_recon_isa(recon_simd::isa())));
        sites.push((
            "hevc_fwd_transform_quant",
            from_hevc_backend(zvidlib_hevc_encoder::engine::encoder::quant_simd::detected()),
        ));
        sites.push(("hevc_colorconv", from_colorconv_isa(colorconv::isa())));
        sites.push((
            "hevc_color_convert",
            from_color_convert_isa(color_convert::detected_isa()),
        ));
    }
    sites.push((
        "yuv_to_rgba",
        from_yuv_to_rgba_isa(zvidlib_color::yuv_to_rgba::detected_isa()),
    ));
    sites.push((
        "vorbis_decode",
        zvidlib_vorbis_decoder::vorbis_simd::active_isa(),
    ));
    sites
}

fn from_yuv_to_rgba_isa(isa: zvidlib_color::yuv_to_rgba::Isa) -> SimdIsa {
    use zvidlib_color::yuv_to_rgba::Isa;
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

fn from_mc_level(level: zvidlib_av1::av1_mc::SimdLevel) -> SimdIsa {
    use zvidlib_av1::av1_mc::SimdLevel;
    match level {
        SimdLevel::Scalar => SimdIsa::Scalar,
        SimdLevel::Sse41 => SimdIsa::Sse41,
        SimdLevel::Avx2 => SimdIsa::Avx2,
        SimdLevel::Neon => SimdIsa::Neon,
    }
}

fn from_intra_simd(simd: zvidlib_av1::av1_intra_pred::Av1IntraSimd) -> SimdIsa {
    use zvidlib_av1::av1_intra_pred::Av1IntraSimd;
    match simd {
        Av1IntraSimd::Scalar => SimdIsa::Scalar,
        Av1IntraSimd::Sse41 => SimdIsa::Sse41,
        Av1IntraSimd::Avx2 => SimdIsa::Avx2,
        Av1IntraSimd::Neon => SimdIsa::Neon,
    }
}

fn from_vp9_encode_isa(isa: zvidlib_vp9_encoder::simd::Isa) -> SimdIsa {
    use zvidlib_vp9_encoder::simd::Isa;
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

#[cfg(not(target_arch = "wasm32"))]
fn from_hevc_isa(isa: zvidlib_hevc_decoder::engine::simd::Isa) -> SimdIsa {
    use zvidlib_hevc_decoder::engine::simd::Isa;
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

/// SSE4.2 is SSE4.1 plus the 64-bit compare the dequantization clip needs; the
/// crate-wide vocabulary has no separate name for it, so both report as
/// [`SimdIsa::Sse41`].
#[cfg(not(target_arch = "wasm32"))]
fn from_hevc_backend(backend: zvidlib_hevc_decoder::engine::transform_simd::Backend) -> SimdIsa {
    use zvidlib_hevc_decoder::engine::transform_simd::Backend;
    match backend {
        Backend::Scalar => SimdIsa::Scalar,
        Backend::Sse41 | Backend::Sse42 => SimdIsa::Sse41,
        Backend::Avx2 => SimdIsa::Avx2,
        Backend::Neon => SimdIsa::Neon,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn from_color_convert_isa(isa: zvidlib_color::color_convert::Isa) -> SimdIsa {
    use zvidlib_color::color_convert::Isa;
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

#[cfg(not(target_arch = "wasm32"))]
fn from_recon_isa(isa: zvidlib_hevc_encoder::engine::encoder::recon_simd::Isa) -> SimdIsa {
    use zvidlib_hevc_encoder::engine::encoder::recon_simd::Isa;
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

#[cfg(not(target_arch = "wasm32"))]
fn from_rdcost_isa(isa: zvidlib_hevc_encoder::engine::encoder::rdcost::Isa) -> SimdIsa {
    use zvidlib_hevc_encoder::engine::encoder::rdcost::Isa;
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

#[cfg(not(target_arch = "wasm32"))]
fn from_colorconv_isa(isa: zvidlib_color::colorconv::Isa) -> SimdIsa {
    use zvidlib_color::colorconv::Isa;
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

#[cfg(test)]
mod tests {
    use super::*;

    use super::test_lock as lock;

    /// The override is only useful if it actually reaches the kernels, and
    /// each dispatch family resolves its instruction set through a different
    /// selector. This pins every one of them at once.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn pinning_scalar_reaches_every_dispatch_site() {
        use zvidlib_av1::av1_intra_pred::{Av1IntraSimd, av1_intra_simd};
        use zvidlib_av1::av1_mc::{McContext, SimdLevel, default_level};
        use zvidlib_color::color_convert;
        use zvidlib_color::colorconv;
        use zvidlib_hevc_decoder::engine::{simd as hevc_simd, transform_simd};
        use zvidlib_hevc_encoder::engine::encoder::{quant_simd, rdcost, recon_simd};
        use zvidlib_vp9_encoder::simd as vp9_encode;

        let _guard = lock();
        set_override(Some(SimdIsa::Scalar));

        // AV1 transforms and in-loop filters.
        assert_eq!(zvidlib_av1::av1_simd::active_isa(), SimdIsa::Scalar);
        // AV1 intra prediction, whose `OnceLock` detection may already have
        // resolved to a vector backend in an earlier test.
        assert_eq!(av1_intra_simd(), Av1IntraSimd::Scalar);
        // AV1 encoder-side coefficient context derivation, whose `OnceLock`
        // detection may likewise have already resolved to a vector backend.
        assert_eq!(zvidlib_av1::av1_simd::coeff::active_isa(), SimdIsa::Scalar);
        // The Vorbis encoder's analysis kernels, whose `OnceLock` detection
        // may likewise have already resolved to a vector backend.
        assert_eq!(
            zvidlib_vorbis_encoder::vorbis_encoder::simd::active_isa(),
            SimdIsa::Scalar
        );
        // AV1 motion compensation, through the level `McContext::new` picks.
        assert_eq!(default_level(), SimdLevel::Scalar);
        assert_eq!(McContext::new().level(), SimdLevel::Scalar);
        // VP8 encoder-side distortion metrics, forward transforms and
        // quantization.
        assert_eq!(zvidlib_vp8::simd::encode_isa(), SimdIsa::Scalar);
        // VP8 reconstruction and loop filter, shared with the decoder.
        assert_eq!(zvidlib_vp8::simd::recon_isa(), SimdIsa::Scalar);
        // VP8 subblock intra prediction and the DC-only inverse DCT.
        assert_eq!(zvidlib_vp8::simd::decode_isa(), SimdIsa::Scalar);
        // VP9 decoder transforms, prediction and loop filter.
        assert_eq!(zvidlib_vp9_decoder::vp9_simd::active_isa(), SimdIsa::Scalar);
        // The VP9 encoder's kernels.
        assert_eq!(vp9_encode::isa(), vp9_encode::Isa::Scalar);
        // HEVC inter/intra prediction and in-loop filters.
        assert_eq!(hevc_simd::detected_isa(), hevc_simd::Isa::Scalar);
        // HEVC inverse transforms and dequantization.
        assert_eq!(transform_simd::detected(), transform_simd::Backend::Scalar);
        // HEVC encoder-side distortion metrics.
        assert_eq!(rdcost::isa(), rdcost::Isa::Scalar);
        // HEVC encoder-side reconstruction and SAO parameter search.
        assert_eq!(recon_simd::isa(), recon_simd::Isa::Scalar);
        // HEVC encoder-side forward transform and quantization.
        assert_eq!(quant_simd::detected(), transform_simd::Backend::Scalar);
        // HEVC encoder-side RGBA8 to YUV420 input conversion.
        assert_eq!(colorconv::isa(), colorconv::Isa::Scalar);
        // The HEVC decoder's YUV420-to-RGBA output conversion.
        assert_eq!(color_convert::detected_isa(), color_convert::Isa::Scalar);
        // The AV1, VP8 and VP9 decoders' YUV-to-RGBA output conversion.
        assert_eq!(
            zvidlib_color::yuv_to_rgba::detected_isa(),
            zvidlib_color::yuv_to_rgba::Isa::Scalar
        );
        // The Vorbis decoder's synthesis kernels.
        assert_eq!(
            zvidlib_vorbis_decoder::vorbis_simd::active_isa(),
            SimdIsa::Scalar
        );

        // The list above is written out by hand, one selector per site, so it
        // only stays exhaustive as long as it matches `active_by_site`. A new
        // site added there has to fail here rather than quietly go unchecked.
        let checked = [
            "av1_simd",
            "av1_mc",
            "av1_intra_pred",
            "av1_coeff_ctx",
            "vorbis_encode",
            "vp8_encode",
            "vp8_recon",
            "vp8_decode",
            "vp9_decode",
            "vp9_encode",
            "hevc_prediction_filters",
            "hevc_transforms",
            "hevc_rdcost",
            "hevc_recon",
            "hevc_fwd_transform_quant",
            "hevc_colorconv",
            "hevc_color_convert",
            "yuv_to_rgba",
            "vorbis_decode",
        ];
        let sites: Vec<&str> = active_by_site().into_iter().map(|(site, _)| site).collect();
        assert_eq!(sites, checked);

        set_override(None);
    }

    /// The site table in the `active_by_site` rustdoc promises the names are
    /// "stable and safe to assert on", which is only true if it lists them
    /// all. Read the table back out of this file and compare.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn the_documented_site_table_lists_every_dispatch_site() {
        let source = include_str!("simd.rs");
        let table = source
            .split_once("/// | Site | Kernels |")
            .expect("site table")
            .1;
        let documented: Vec<&str> = table
            .lines()
            .skip(1)
            .map(str::trim_start)
            .take_while(|line| line.starts_with("///"))
            .filter_map(|line| line.strip_prefix("/// | `"))
            .filter_map(|row| row.split_once('`'))
            .map(|(site, _)| site)
            .collect();
        let sites: Vec<&str> = active_by_site().into_iter().map(|(site, _)| site).collect();
        assert_eq!(documented, sites);
    }

    /// Clearing the override has to hand every site back to its own detection,
    /// not leave it pinned to whatever the last test asked for.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn clearing_the_override_restores_per_site_detection() {
        use zvidlib_av1::av1_intra_pred::{Av1IntraSimd, av1_intra_simd};
        use zvidlib_av1::av1_mc::default_level;
        use zvidlib_color::color_convert;
        use zvidlib_color::colorconv;
        use zvidlib_hevc_decoder::engine::{simd as hevc_simd, transform_simd};
        use zvidlib_hevc_encoder::engine::encoder::{quant_simd, rdcost, recon_simd};
        use zvidlib_vp9_encoder::simd as vp9_encode;

        let _guard = lock();
        set_override(Some(SimdIsa::Scalar));
        set_override(None);

        let vectorized = detected() != SimdIsa::Scalar;
        // AV1 transforms and in-loop filters.
        assert_eq!(zvidlib_av1::av1_simd::active_isa(), detected());
        // AV1 intra prediction.
        assert_eq!(av1_intra_simd() != Av1IntraSimd::Scalar, vectorized);
        // AV1 encoder-side coefficient context derivation.
        assert_eq!(
            zvidlib_av1::av1_simd::coeff::active_isa() != SimdIsa::Scalar,
            vectorized
        );
        // The Vorbis encoder's analysis kernels.
        assert_eq!(
            zvidlib_vorbis_encoder::vorbis_encoder::simd::active_isa(),
            detected()
        );
        // AV1 motion compensation.
        assert_eq!(
            default_level() != zvidlib_av1::av1_mc::SimdLevel::Scalar,
            vectorized
        );
        // VP8 encoder-side distortion metrics, forward transforms and
        // quantization.
        assert_eq!(zvidlib_vp8::simd::encode_isa(), detected());
        // VP8 reconstruction and loop filter, shared with the decoder.
        assert_eq!(zvidlib_vp8::simd::recon_isa(), detected());
        // VP8 subblock intra prediction and the DC-only inverse DCT.
        assert_eq!(zvidlib_vp8::simd::decode_isa(), detected());
        // VP9 decoder transforms, prediction and loop filter.
        assert_eq!(zvidlib_vp9_decoder::vp9_simd::active_isa(), detected());
        // The VP9 encoder's kernels.
        assert_eq!(vp9_encode::isa() != vp9_encode::Isa::Scalar, vectorized);
        // HEVC inter/intra prediction and in-loop filters.
        assert_eq!(
            hevc_simd::detected_isa() != hevc_simd::Isa::Scalar,
            vectorized
        );
        // HEVC inverse transforms and dequantization.
        assert_eq!(
            transform_simd::detected() != transform_simd::Backend::Scalar,
            vectorized
        );
        // HEVC encoder-side distortion metrics.
        assert_eq!(rdcost::isa() != rdcost::Isa::Scalar, vectorized);
        // HEVC encoder-side reconstruction and SAO parameter search.
        assert_eq!(recon_simd::isa() != recon_simd::Isa::Scalar, vectorized);
        // HEVC encoder-side forward transform and quantization.
        assert_eq!(
            quant_simd::detected() != transform_simd::Backend::Scalar,
            vectorized
        );
        // HEVC encoder-side RGBA8 to YUV420 input conversion.
        assert_eq!(colorconv::isa() != colorconv::Isa::Scalar, vectorized);
        // The HEVC decoder's YUV420-to-RGBA output conversion.
        assert_eq!(
            color_convert::detected_isa() != color_convert::Isa::Scalar,
            vectorized
        );
        // The AV1, VP8 and VP9 decoders' YUV-to-RGBA output conversion.
        assert_eq!(
            zvidlib_color::yuv_to_rgba::detected_isa() != zvidlib_color::yuv_to_rgba::Isa::Scalar,
            vectorized
        );
        // The Vorbis decoder's synthesis kernels.
        assert_eq!(
            zvidlib_vorbis_decoder::vorbis_simd::active_isa(),
            detected()
        );

        // As in `pinning_scalar_reaches_every_dispatch_site`, the list above is
        // written out by hand, one selector per site, so it only stays
        // exhaustive as long as it matches `active_by_site`. A new site added
        // there has to fail here rather than quietly go unchecked.
        let checked = [
            "av1_simd",
            "av1_mc",
            "av1_intra_pred",
            "av1_coeff_ctx",
            "vorbis_encode",
            "vp8_encode",
            "vp8_recon",
            "vp8_decode",
            "vp9_decode",
            "vp9_encode",
            "hevc_prediction_filters",
            "hevc_transforms",
            "hevc_rdcost",
            "hevc_recon",
            "hevc_fwd_transform_quant",
            "hevc_colorconv",
            "hevc_color_convert",
            "yuv_to_rgba",
            "vorbis_decode",
        ];
        let sites: Vec<&str> = active_by_site().into_iter().map(|(site, _)| site).collect();
        assert_eq!(sites, checked);
    }

    #[test]
    fn every_site_reports_the_pinned_instruction_set() {
        let _guard = lock();
        for isa in available() {
            set_override(Some(isa));
            for (site, site_isa) in active_by_site() {
                assert_eq!(site_isa, isa, "site {site} did not follow the override");
            }
        }
        set_override(None);
        for (site, site_isa) in active_by_site() {
            assert_eq!(
                site_isa,
                detected(),
                "site {site} did not fall back to detection"
            );
        }
    }

    #[test]
    fn the_legacy_av1_entry_point_delegates_to_the_shared_override() {
        let _guard = lock();
        zvidlib_av1::av1_simd::set_active_isa(Some(SimdIsa::Scalar));
        assert_eq!(active(), SimdIsa::Scalar);
        assert_eq!(zvidlib_av1::av1_simd::active_isa(), SimdIsa::Scalar);
        zvidlib_av1::av1_simd::set_active_isa(None);
        assert_eq!(active(), detected());
        set_override(None);
    }

    /// `benches/README.md`'s `The HEVC per-stage groups` table carries a
    /// `Vectorized` column, and unlike a committed baseline table it asserts
    /// something about `HEAD` rather than about a stamped commit: it says
    /// whether the group's kernels run vectorized *now*. So it can be answered
    /// from a build, and it fails rather than reports - a row claiming `yes`
    /// after its kernel was removed, or still claiming `no` after one landed,
    /// is otherwise invisible until somebody reads a ratio that disagrees with
    /// it (issue #391).
    ///
    /// The group-to-site attribution is read out of the `SITE_GROUP_PREFIXES`
    /// table `.github/scripts/criterion_baseline.py` already encodes, rather
    /// than restated here, so the two cannot drift apart.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn the_per_stage_vectorized_column_matches_the_kernels_that_exist() {
        /// Groups with no dispatch site at all, and the exact cell they are
        /// expected to carry. `hevc_cabac` is the one the table itself calls
        /// out: §9.3.4 bin decoding is serial by construction, so it has no
        /// kernel to check and its row is an intentional `no`, not a stale
        /// one. Naming it here rather than inferring "unmapped means exempt"
        /// keeps a group whose site was *deleted* from going quiet.
        const EXEMPT: &[(&str, &str)] = &[("hevc_cabac", "no, by design")];

        const HEADER: &str = "| Group | Stage | Vectorized |";
        let readme = include_str!("../benches/README.md");
        assert_eq!(
            readme.matches(HEADER).count(),
            1,
            "`{HEADER}` no longer identifies exactly one table"
        );
        let rows: Vec<(&str, &str)> = readme
            .split_once(HEADER)
            .expect("per-stage group table")
            .1
            .lines()
            .skip_while(|line| !line.starts_with("| ---"))
            .skip(1)
            .take_while(|line| line.starts_with('|'))
            .map(|line| {
                let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
                assert_eq!(cells.len(), 3, "unexpected row shape: {line}");
                (cells[0].trim_matches('`'), cells[2])
            })
            .collect();
        assert!(!rows.is_empty(), "per-stage group table has no rows");

        let attribution = site_group_prefixes();
        let active: std::collections::HashMap<&str, SimdIsa> = {
            let _guard = lock();
            set_override(None);
            active_by_site().into_iter().collect()
        };
        // On a host with no vector instruction set every site resolves to
        // scalar whether or not it has a kernel, so the column is unreadable
        // rather than wrong. The parsing and attribution above still ran.
        let vectorized_host = detected() != SimdIsa::Scalar;

        for (group, claim) in rows {
            if let Some((_, expected)) = EXEMPT.iter().find(|(name, _)| *name == group) {
                assert_eq!(
                    claim, *expected,
                    "`{group}` is exempt from the check and its row has to keep saying so"
                );
                continue;
            }
            let sites: Vec<&str> = attribution
                .iter()
                .filter(|(_, prefixes)| prefixes.iter().any(|p| group.starts_with(p.as_str())))
                .map(|(site, _)| site.as_str())
                .collect();
            assert_eq!(
                sites.len(),
                1,
                "`{group}` maps to {sites:?} in `SITE_GROUP_PREFIXES`; a benchmark \
                 group needs exactly one dispatch site, or an `EXEMPT` entry saying \
                 why it has none"
            );
            let site = sites[0];
            let isa = *active
                .get(site)
                .unwrap_or_else(|| panic!("site `{site}` is not registered in `active_by_site`"));
            let claimed = match claim {
                "yes" => true,
                "no" => false,
                other => panic!("`{group}` has an unreadable `Vectorized` cell: {other:?}"),
            };
            if !vectorized_host {
                continue;
            }
            assert_eq!(
                isa != SimdIsa::Scalar,
                claimed,
                "`benches/README.md` says `{group}` is `{claim}`, but its `{site}` \
                 dispatch site resolves to {} on this host",
                isa.name()
            );
        }
    }

    /// The `SITE_GROUP_PREFIXES` mapping in
    /// `.github/scripts/criterion_baseline.py`, read back as site name to the
    /// benchmark-group name prefixes attributed to it. Parsed rather than
    /// duplicated: the Python side is the one the staleness check uses, and a
    /// second copy here would be a second thing to keep true.
    #[cfg(not(target_arch = "wasm32"))]
    fn site_group_prefixes() -> Vec<(String, Vec<String>)> {
        let script = include_str!("../.github/scripts/criterion_baseline.py");
        let body = script
            .split_once("SITE_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {")
            .expect("`SITE_GROUP_PREFIXES` literal")
            .1
            .split_once("\n}")
            .expect("end of `SITE_GROUP_PREFIXES` literal")
            .0;

        let mut table: Vec<(String, Vec<String>)> = Vec::new();
        for line in body.lines() {
            // No escapes appear in these literals, so the quoted strings are
            // just the odd-indexed pieces of a split on `"`.
            let pieces: Vec<&str> = line.split('"').collect();
            let mut strings = pieces.iter().skip(1).step_by(2).copied().peekable();
            if strings.peek().is_none() {
                continue;
            }
            // A key line is `    "site": (`; a continuation line is an
            // indented prefix. Only the former has a `:` right after its
            // first string.
            let is_key = pieces
                .get(2)
                .is_some_and(|rest| rest.trim_start().starts_with(':'));
            if is_key {
                let site = strings.next().expect("site name").to_string();
                table.push((site, strings.map(str::to_string).collect()));
            } else {
                let prefixes = &mut table.last_mut().expect("a site to attribute to").1;
                prefixes.extend(strings.map(str::to_string));
            }
        }
        assert!(!table.is_empty(), "`SITE_GROUP_PREFIXES` parsed as empty");
        for (site, prefixes) in &table {
            assert!(
                !prefixes.is_empty(),
                "site `{site}` parsed with no prefixes"
            );
        }
        table
    }
}
