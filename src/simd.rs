//! The process-wide SIMD instruction-set override shared by every codec kernel.
//!
//! zvidlib's pure-Rust HEVC and AV1 codecs dispatch their hot loops to
//! runtime-detected vector kernels in several independent places: the AV1
//! transforms and in-loop filters ([`crate::av1_simd`]), AV1 inter prediction
//! ([`crate::av1_mc`]), AV1 intra prediction ([`crate::av1_intra_pred`]), and
//! the HEVC engine's inter/intra prediction, in-loop filters, inverse
//! transforms, and encoder-side distortion metrics. Each of those sites caches
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the tests that pin the process-wide override.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

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
