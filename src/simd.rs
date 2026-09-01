//! The process-wide SIMD instruction-set override shared by every codec kernel.
//!
//! zvidlib's pure-Rust HEVC and AV1 codecs dispatch their hot loops to
//! runtime-detected vector kernels in several independent places: the AV1
//! transforms and in-loop filters ([`crate::av1_simd`]), AV1 inter prediction
//! ([`crate::av1_mc`]), AV1 intra prediction ([`crate::av1_intra_pred`]), and
//! the HEVC engine's inter/intra prediction, in-loop filters, inverse
//! transforms, encoder-side distortion metrics, and encoder-side color
//! conversion. Each of those sites caches
//! its own CPU feature probe, which is what you want in production but makes
//! "run this workload with SIMD off" impossible to express from outside the
//! crate.
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

use core::sync::atomic::{AtomicU8, Ordering};

pub use crate::av1_simd::SimdIsa;

/// `0` means "no override"; every other value is a [`SimdIsa::code`].
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Forces every SIMD-dispatched kernel in the crate onto `isa`, or restores
/// per-site automatic detection with `None`.
///
/// The override reaches all four dispatch families: the AV1 transform and
/// in-loop filter kernels, AV1 motion compensation (through the default level
/// [`crate::av1_mc::McContext::new`] picks up), AV1 intra prediction, and every
/// HEVC engine kernel. [`SimdIsa::Scalar`] therefore genuinely reaches the
/// scalar code path rather than merely the widest scalar-ish one.
///
/// An instruction set this host cannot execute is clamped to
/// [`SimdIsa::Scalar`] rather than silently ignored, so a caller that asks for
/// AVX2 on an aarch64 machine gets a defined, reproducible arm instead of the
/// host's best vector kernels.
///
/// This is safe to call at any time and from any thread: the kernels agree
/// bit-for-bit, so a switch that lands between two blocks of the same frame
/// still produces the same output.
pub fn set_override(isa: Option<SimdIsa>) {
    let code = match isa {
        Some(isa) if available().contains(&isa) => isa.code(),
        Some(_) => SimdIsa::Scalar.code(),
        None => 0,
    };
    OVERRIDE.store(code, Ordering::Relaxed);
}

/// The instruction set currently in force: the [`set_override`] value when one
/// is set, otherwise [`detected`].
#[must_use]
pub fn active() -> SimdIsa {
    override_isa().unwrap_or_else(detected)
}

/// The widest instruction set this host supports, ignoring any
/// [`set_override`].
#[must_use]
pub fn detected() -> SimdIsa {
    crate::av1_simd::detected_isa()
}

/// Every instruction set this host can execute, always including
/// [`SimdIsa::Scalar`].
///
/// Benchmarks iterate this to build one measurement arm per available
/// instruction set, so scalar and vector timings sit side by side.
#[must_use]
pub fn available() -> Vec<SimdIsa> {
    crate::av1_simd::available_isas()
}

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
/// | `av1_mc` | AV1 motion compensation (the level [`crate::av1_mc::McContext::new`] picks) |
/// | `av1_intra_pred` | AV1 intra prediction and residual reconstruction |
/// | `hevc_prediction_filters` | HEVC inter/intra prediction and in-loop filters |
/// | `hevc_transforms` | HEVC inverse transforms and dequantization |
/// | `hevc_rdcost` | HEVC encoder-side distortion metrics |
/// | `hevc_fwd_transform_quant` | HEVC encoder-side forward transform and quantization |
/// | `hevc_colorconv` | HEVC encoder-side RGBA8 to YUV420 input conversion |
/// | `hevc_color_convert` | HEVC decoder output YUV420-to-RGBA conversion |
///
/// The `hevc_*` sites are absent on `wasm32`, which does not build the HEVC
/// engine.
#[must_use]
pub fn active_by_site() -> Vec<(&'static str, SimdIsa)> {
    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    let mut sites = vec![
        ("av1_simd", crate::av1_simd::active_isa()),
        ("av1_mc", from_mc_level(crate::av1_mc::default_level())),
        (
            "av1_intra_pred",
            from_intra_simd(crate::av1_intra_pred::av1_intra_simd()),
        ),
    ];
    #[cfg(not(target_arch = "wasm32"))]
    {
        use crate::hevc::color_convert;
        use crate::hevc::engine::encoder::{colorconv, rdcost};
        use crate::hevc::engine::{simd as hevc_simd, transform_simd};
        sites.push((
            "hevc_prediction_filters",
            from_hevc_isa(hevc_simd::detected_isa()),
        ));
        sites.push((
            "hevc_transforms",
            from_hevc_backend(transform_simd::detected()),
        ));
        sites.push(("hevc_rdcost", from_rdcost_isa(rdcost::isa())));
        sites.push((
            "hevc_fwd_transform_quant",
            from_hevc_backend(crate::hevc::engine::encoder::quant_simd::detected()),
        ));
        sites.push(("hevc_colorconv", from_colorconv_isa(colorconv::isa())));
        sites.push((
            "hevc_color_convert",
            from_color_convert_isa(color_convert::detected_isa()),
        ));
    }
    sites
}

fn from_mc_level(level: crate::av1_mc::SimdLevel) -> SimdIsa {
    use crate::av1_mc::SimdLevel;
    match level {
        SimdLevel::Scalar => SimdIsa::Scalar,
        SimdLevel::Sse41 => SimdIsa::Sse41,
        SimdLevel::Avx2 => SimdIsa::Avx2,
        SimdLevel::Neon => SimdIsa::Neon,
    }
}

fn from_intra_simd(simd: crate::av1_intra_pred::Av1IntraSimd) -> SimdIsa {
    use crate::av1_intra_pred::Av1IntraSimd;
    match simd {
        Av1IntraSimd::Scalar => SimdIsa::Scalar,
        Av1IntraSimd::Sse41 => SimdIsa::Sse41,
        Av1IntraSimd::Avx2 => SimdIsa::Avx2,
        Av1IntraSimd::Neon => SimdIsa::Neon,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn from_hevc_isa(isa: crate::hevc::engine::simd::Isa) -> SimdIsa {
    use crate::hevc::engine::simd::Isa;
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
fn from_hevc_backend(backend: crate::hevc::engine::transform_simd::Backend) -> SimdIsa {
    use crate::hevc::engine::transform_simd::Backend;
    match backend {
        Backend::Scalar => SimdIsa::Scalar,
        Backend::Sse41 | Backend::Sse42 => SimdIsa::Sse41,
        Backend::Avx2 => SimdIsa::Avx2,
        Backend::Neon => SimdIsa::Neon,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn from_color_convert_isa(isa: crate::hevc::color_convert::Isa) -> SimdIsa {
    use crate::hevc::color_convert::Isa;
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
fn from_rdcost_isa(isa: crate::hevc::engine::encoder::rdcost::Isa) -> SimdIsa {
    use crate::hevc::engine::encoder::rdcost::Isa;
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
fn from_colorconv_isa(isa: crate::hevc::engine::encoder::colorconv::Isa) -> SimdIsa {
    use crate::hevc::engine::encoder::colorconv::Isa;
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

/// The active override, or `None` when detection is in charge.
///
/// Every dispatch site in the crate consults this *before* its own cached
/// probe, which is what lets the override win over a `OnceLock` that has
/// already resolved.
#[inline]
#[must_use]
pub(crate) fn override_isa() -> Option<SimdIsa> {
    SimdIsa::from_code(OVERRIDE.load(Ordering::Relaxed))
}

/// Serializes every test that pins the process-wide override.
///
/// The override is one global now, so tests that used to pin four independent
/// switches (in `av1_simd`, the HEVC in-loop filter dispatcher, and here) can
/// no longer each hold their own mutex — they would swap the instruction set
/// out from under each other. They all take this one instead.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::test_lock as lock;

    #[test]
    fn available_always_includes_scalar_and_contains_the_detected_set() {
        let isas = available();
        assert!(isas.contains(&SimdIsa::Scalar));
        assert!(isas.contains(&detected()));
    }

    #[test]
    fn override_takes_precedence_over_detection_and_clears() {
        let _guard = lock();
        set_override(Some(SimdIsa::Scalar));
        assert_eq!(active(), SimdIsa::Scalar);
        set_override(None);
        assert_eq!(active(), detected());
    }

    #[test]
    fn every_available_instruction_set_can_be_pinned() {
        let _guard = lock();
        for isa in available() {
            set_override(Some(isa));
            assert_eq!(active(), isa, "{}", isa.name());
        }
        set_override(None);
    }

    #[test]
    fn an_unsupported_instruction_set_clamps_to_scalar() {
        let _guard = lock();
        let unsupported = [
            SimdIsa::Scalar,
            SimdIsa::Sse41,
            SimdIsa::Avx2,
            SimdIsa::Neon,
        ]
        .into_iter()
        .find(|isa| !available().contains(isa));
        if let Some(isa) = unsupported {
            set_override(Some(isa));
            assert_eq!(active(), SimdIsa::Scalar, "{}", isa.name());
        }
        set_override(None);
    }

    /// The override is only useful if it actually reaches the kernels, and
    /// each dispatch family resolves its instruction set through a different
    /// selector. This pins every one of them at once.
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn pinning_scalar_reaches_every_dispatch_site() {
        use crate::av1_intra_pred::{Av1IntraSimd, av1_intra_simd};
        use crate::av1_mc::{McContext, SimdLevel, default_level};
        use crate::hevc::color_convert;
        use crate::hevc::engine::encoder::{colorconv, quant_simd, rdcost};
        use crate::hevc::engine::{simd as hevc_simd, transform_simd};

        let _guard = lock();
        set_override(Some(SimdIsa::Scalar));

        // AV1 transforms and in-loop filters.
        assert_eq!(crate::av1_simd::active_isa(), SimdIsa::Scalar);
        // AV1 intra prediction, whose `OnceLock` detection may already have
        // resolved to a vector backend in an earlier test.
        assert_eq!(av1_intra_simd(), Av1IntraSimd::Scalar);
        // AV1 motion compensation, through the level `McContext::new` picks.
        assert_eq!(default_level(), SimdLevel::Scalar);
        assert_eq!(McContext::new().level(), SimdLevel::Scalar);
        // HEVC inter/intra prediction and in-loop filters.
        assert_eq!(hevc_simd::detected_isa(), hevc_simd::Isa::Scalar);
        // HEVC inverse transforms and dequantization.
        assert_eq!(transform_simd::detected(), transform_simd::Backend::Scalar);
        // HEVC encoder-side distortion metrics.
        assert_eq!(rdcost::isa(), rdcost::Isa::Scalar);
        // HEVC encoder-side forward transform and quantization.
        assert_eq!(quant_simd::detected(), transform_simd::Backend::Scalar);
        // HEVC encoder-side RGBA8 to YUV420 input conversion.
        assert_eq!(colorconv::isa(), colorconv::Isa::Scalar);
        // The HEVC decoder's YUV420-to-RGBA output conversion.
        assert_eq!(color_convert::detected_isa(), color_convert::Isa::Scalar);

        // The list above is written out by hand, one selector per site, so it
        // only stays exhaustive as long as it matches `active_by_site`. A new
        // site added there has to fail here rather than quietly go unchecked.
        let checked = [
            "av1_simd",
            "av1_mc",
            "av1_intra_pred",
            "hevc_prediction_filters",
            "hevc_transforms",
            "hevc_rdcost",
            "hevc_fwd_transform_quant",
            "hevc_colorconv",
            "hevc_color_convert",
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
        use crate::av1_intra_pred::{Av1IntraSimd, av1_intra_simd};
        use crate::av1_mc::default_level;
        use crate::hevc::engine::encoder::{colorconv, quant_simd, rdcost};
        use crate::hevc::engine::{simd as hevc_simd, transform_simd};

        let _guard = lock();
        set_override(Some(SimdIsa::Scalar));
        set_override(None);

        let vectorized = detected() != SimdIsa::Scalar;
        // AV1 transforms and in-loop filters.
        assert_eq!(crate::av1_simd::active_isa(), detected());
        // AV1 intra prediction.
        assert_eq!(av1_intra_simd() != Av1IntraSimd::Scalar, vectorized);
        // AV1 motion compensation.
        assert_eq!(
            default_level() != crate::av1_mc::SimdLevel::Scalar,
            vectorized
        );
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
        // HEVC encoder-side forward transform and quantization.
        assert_eq!(
            quant_simd::detected() != transform_simd::Backend::Scalar,
            vectorized
        );
        // HEVC encoder-side RGBA8 to YUV420 input conversion.
        assert_eq!(colorconv::isa() != colorconv::Isa::Scalar, vectorized);

        // As in `pinning_scalar_reaches_every_dispatch_site`, the list above is
        // written out by hand, one selector per site, so it only stays
        // exhaustive as long as it matches `active_by_site`. A new site added
        // there has to fail here rather than quietly go unchecked.
        let checked = [
            "av1_simd",
            "av1_mc",
            "av1_intra_pred",
            "hevc_prediction_filters",
            "hevc_transforms",
            "hevc_rdcost",
            "hevc_fwd_transform_quant",
            "hevc_colorconv",
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
        crate::av1_simd::set_active_isa(Some(SimdIsa::Scalar));
        assert_eq!(active(), SimdIsa::Scalar);
        assert_eq!(crate::av1_simd::active_isa(), SimdIsa::Scalar);
        crate::av1_simd::set_active_isa(None);
        assert_eq!(active(), detected());
        set_override(None);
    }
}
