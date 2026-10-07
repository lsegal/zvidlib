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

#[cfg(all(test, feature = "all"))]
use zvidlib_core::simd::test_lock;
#[doc(inline)]
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
/// vector backend and always run the scalar path. A codec's sites are listed
/// only when the Cargo feature that builds it is enabled.
#[must_use]
pub fn active_by_site() -> Vec<(&'static str, SimdIsa)> {
    #[allow(unused_mut)]
    let mut sites = Vec::new();
    #[cfg(any(feature = "av1-decoder", feature = "av1-encoder"))]
    sites.extend(zvidlib_av1::simd_sites());
    #[cfg(feature = "vorbis-encoder")]
    sites.extend(zvidlib_vorbis_encoder::simd_sites());
    #[cfg(any(feature = "vp8-decoder", feature = "vp8-encoder"))]
    sites.extend(zvidlib_vp8::simd_sites());
    #[cfg(feature = "vp9-decoder")]
    sites.extend(zvidlib_vp9_decoder::simd_sites());
    #[cfg(feature = "vp9-encoder")]
    sites.extend(zvidlib_vp9_encoder::simd_sites());
    #[cfg(feature = "hevc-decoder")]
    sites.extend(zvidlib_hevc_decoder::simd_sites());
    #[cfg(all(feature = "hevc-encoder", not(target_arch = "wasm32")))]
    sites.extend(zvidlib_hevc_encoder::simd_sites());
    sites.extend(zvidlib_color::simd_sites());
    #[cfg(feature = "vorbis-decoder")]
    sites.extend(zvidlib_vorbis_decoder::simd_sites());
    sites
}

// The tests pin and compare every codec's dispatch site at once.
#[cfg(all(test, feature = "all"))]
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
}
