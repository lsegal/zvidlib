//! The process-wide SIMD instruction-set override shared by every codec kernel.
//!
//! zvidlib's codecs dispatch their hot loops to runtime-detected vector kernels
//! in many independent places, each caching its own CPU feature probe. This
//! module is the single switch every one of them consults first; see
//! `zvidlib::simd` for the user-facing description and the per-site report.

use core::sync::atomic::{AtomicU8, Ordering};

#[doc(hidden)]
pub mod vector;

/// `0` means "no override"; every other value is a [`SimdIsa::code`].
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Forces every SIMD-dispatched kernel in the crate onto `isa`, or restores
/// per-site automatic detection with `None`.
///
/// The override reaches every dispatch family: the AV1 transform and in-loop
/// filter kernels, AV1 motion compensation (through the default level
/// `crate::av1_mc::McContext::new` picks up), AV1 intra prediction, the
/// Vorbis encoder's analysis kernels, the VP8 encoder, reconstruction and
/// decoder kernels, every VP9 decoder kernel, every HEVC engine kernel, the
/// AV1, VP8 and VP9 output color conversion, the Vorbis decoder's synthesis
/// kernels, and the VP9 encoder's kernels. [`SimdIsa::Scalar`] therefore
/// genuinely reaches the scalar code path rather than merely the widest
/// scalar-ish one.
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
    detected_isa()
}

/// Every instruction set this host can execute, always including
/// [`SimdIsa::Scalar`].
///
/// Benchmarks iterate this to build one measurement arm per available
/// instruction set, so scalar and vector timings sit side by side.
#[must_use]
pub fn available() -> Vec<SimdIsa> {
    available_isas()
}

/// An instruction set the SIMD kernels can run on.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SimdIsa {
    /// The portable scalar reference implementation.
    Scalar,
    /// x86_64 SSE4.1 (128-bit, four 32-bit lanes).
    Sse41,
    /// x86_64 AVX2 (256-bit, eight 32-bit lanes).
    Avx2,
    /// aarch64 NEON (128-bit, four 32-bit lanes).
    Neon,
}

impl SimdIsa {
    /// A short stable name, useful in benchmark and diagnostic output.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            SimdIsa::Scalar => "scalar",
            SimdIsa::Sse41 => "sse4.1",
            SimdIsa::Avx2 => "avx2",
            SimdIsa::Neon => "neon",
        }
    }

    #[doc(hidden)]
    pub fn code(self) -> u8 {
        match self {
            SimdIsa::Scalar => 1,
            SimdIsa::Sse41 => 2,
            SimdIsa::Avx2 => 3,
            SimdIsa::Neon => 4,
        }
    }

    #[doc(hidden)]
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(SimdIsa::Scalar),
            2 => Some(SimdIsa::Sse41),
            3 => Some(SimdIsa::Avx2),
            4 => Some(SimdIsa::Neon),
            _ => None,
        }
    }
}

/// Number of samples a single vector operation covers on `isa`, or `0` when
/// `isa` has no vector path and callers should stay scalar.
#[must_use]
pub fn lanes(isa: SimdIsa) -> usize {
    match isa {
        SimdIsa::Scalar => 0,
        SimdIsa::Sse41 | SimdIsa::Neon => 4,
        SimdIsa::Avx2 => 8,
    }
}

/// The best instruction set this CPU supports, ignoring any [`set_override`].
#[must_use]
pub fn detected_isa() -> SimdIsa {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return SimdIsa::Avx2;
        }
        if std::is_x86_feature_detected!("sse4.1") {
            return SimdIsa::Sse41;
        }
        SimdIsa::Scalar
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory in the aarch64 base architecture, so no runtime
        // probe is needed (and `is_aarch64_feature_detected!` is still
        // unstable).
        SimdIsa::Neon
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdIsa::Scalar
    }
}

/// Every instruction set that can be exercised on this host, always including
/// [`SimdIsa::Scalar`]. Used by the bit-exactness tests and the benchmark.
#[must_use]
pub fn available_isas() -> Vec<SimdIsa> {
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(unused_mut)
    )]
    let mut isas = vec![SimdIsa::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            isas.push(SimdIsa::Sse41);
        }
        if std::is_x86_feature_detected!("avx2") {
            isas.push(SimdIsa::Avx2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        isas.push(SimdIsa::Neon);
    }
    isas
}

/// The active override, or `None` when detection is in charge.
///
/// Every dispatch site in the crate consults this *before* its own cached
/// probe, which is what lets the override win over a `OnceLock` that has
/// already resolved.
#[doc(hidden)]
#[inline]
#[must_use]
pub fn override_isa() -> Option<SimdIsa> {
    SimdIsa::from_code(OVERRIDE.load(Ordering::Relaxed))
}

/// Serializes every test that pins the process-wide override.
///
/// The override is one global now, so tests that used to pin four independent
/// switches (in `av1_simd`, the HEVC in-loop filter dispatcher, and here) can
/// no longer each hold their own mutex — they would swap the instruction set
/// out from under each other. They all take this one instead.
#[doc(hidden)]
pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
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
}
