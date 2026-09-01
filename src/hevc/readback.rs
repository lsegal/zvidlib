//! Where a hardware-decoded frame's time goes between the fixed-function block
//! and the host-side `VideoFrame` a caller receives.
//!
//! Issue #151 asked for the surface-copy cost as its own benchmark arm "where
//! the backend exposes one", and #170 recorded that none of the three did:
//! [`crate::VideoDecoder`] hands back a host-side [`crate::VideoFrame`], the
//! HEVC decoder configuration only accepts [`crate::PixelFormat::Rgba8`], and
//! each backend maps its own surface and converts to RGBA inside `submit`. The
//! two costs arrive as one number, and a playback pipeline is often bounded by
//! the second one.
//!
//! This module is the seam that separates them, and it is deliberately a
//! *measurement* seam rather than a zero-copy output path.
//!
//! # Why not a zero-copy output path
//!
//! The other way to expose the same boundary is to let a caller take the
//! decoded surface before readback — a `CVPixelBuffer`, a `CUdeviceptr` and its
//! context, or an `ID3D11Texture2D` and the device that owns it. That is three
//! platform handle types, three lifetime and threading contracts, and a second
//! `PixelFormat` family (NV12) in the public API, none of which the crate can
//! keep stable across the backends it does not control, and none of which the
//! benchmark that motivated the issue needs: a benchmark wants the cost of the
//! copy that runs, not a way to skip it. The zero-copy path stays unbuilt until
//! a caller needs it (a texture upload or a wgpu consumer would be the case for
//! it); the attribution below is what the measurement needs, and it measures
//! the code that actually runs rather than a reimplemented stand-in.
//!
//! # The phases
//!
//! - [`Phase::SurfaceCopy`] — getting the decoded surface where the CPU can
//!   read it: `cuvidMapVideoFrame` plus the `cuMemcpyDtoH` into host memory
//!   (NVDEC), the staging-texture `CopySubresourceRegion` and `Map` (Media
//!   Foundation), or `CVPixelBufferLockBaseAddress` (VideoToolbox). This is the
//!   part that varies most by host: a discrete GPU pays a PCIe transfer here,
//!   unified memory pays little more than a lock.
//! - [`Phase::ColorConvert`] — the NV12-to-RGBA pass every backend then runs
//!   over those bytes, plus the RGBA allocation it fills. This is host CPU work
//!   in every case.
//!
//! # Threads, and why this is not [`crate::hevc::decode_profile`]
//!
//! NVDEC and VideoToolbox deliver frames from a driver or framework callback
//! that need not run on the thread that called `submit`, so the thread-local
//! accumulators the software decoder's stage profiler uses would miss them
//! entirely. State here is process-wide atomics for that reason, which also
//! means a [`report`] covers every hardware HEVC decoder alive in the process.
//! The benchmark runs one at a time; a caller that does not should treat the
//! numbers as a process total rather than a per-decoder one.
//!
//! # Cost
//!
//! Accumulation is unconditional — no start/stop switch and no cargo feature,
//! for the same reason [`crate::hevc::decode_profile`] leaves its scopes on the
//! ordinary path: a gated profiler measures a build nobody ships. It costs two
//! `Instant::now()` reads and one relaxed `fetch_add` per phase per *frame*,
//! tens of nanoseconds against a whole-frame surface copy and colour
//! conversion, which is why the phases are placed per frame and never per row
//! or per sample.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Nanoseconds attributed to [`Phase::SurfaceCopy`] since the last [`reset`].
static SURFACE_COPY_NANOS: AtomicU64 = AtomicU64::new(0);
/// Nanoseconds attributed to [`Phase::ColorConvert`] since the last [`reset`].
static COLOR_CONVERT_NANOS: AtomicU64 = AtomicU64::new(0);
/// Frames a backend delivered since the last [`reset`].
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// One half of the readback a hardware backend performs per frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Making the decoded surface's bytes readable by the CPU.
    SurfaceCopy,
    /// Converting those bytes to the RGBA8 frame the caller receives.
    ColorConvert,
}

impl Phase {
    /// A short, stable name for tables and benchmark ids.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Phase::SurfaceCopy => "surface_copy",
            Phase::ColorConvert => "color_convert",
        }
    }

    fn counter(self) -> &'static AtomicU64 {
        match self {
            Phase::SurfaceCopy => &SURFACE_COPY_NANOS,
            Phase::ColorConvert => &COLOR_CONVERT_NANOS,
        }
    }
}

/// What the hardware backends spent on readback since the last [`reset`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Frames delivered, so the durations can be read per frame.
    pub frames: u64,
    /// Time in [`Phase::SurfaceCopy`].
    pub surface_copy: Duration,
    /// Time in [`Phase::ColorConvert`].
    pub color_convert: Duration,
}

impl Report {
    /// The whole readback: surface copy plus colour conversion.
    #[must_use]
    pub fn total(self) -> Duration {
        self.surface_copy.saturating_add(self.color_convert)
    }

    /// [`Report::total`] divided over the frames it covers.
    ///
    /// Zero frames report [`Duration::ZERO`] rather than dividing by zero: a
    /// window that delivered nothing has no per-frame cost to state.
    #[must_use]
    pub fn total_per_frame(self) -> Duration {
        if self.frames == 0 {
            Duration::ZERO
        } else {
            self.total() / u32::try_from(self.frames).unwrap_or(u32::MAX)
        }
    }
}

/// Clears the accumulators, so the next [`report`] covers only what follows.
pub fn reset() {
    SURFACE_COPY_NANOS.store(0, Ordering::Relaxed);
    COLOR_CONVERT_NANOS.store(0, Ordering::Relaxed);
    FRAMES.store(0, Ordering::Relaxed);
}

/// Reads the accumulators without clearing them.
#[must_use]
pub fn report() -> Report {
    Report {
        frames: FRAMES.load(Ordering::Relaxed),
        surface_copy: Duration::from_nanos(SURFACE_COPY_NANOS.load(Ordering::Relaxed)),
        color_convert: Duration::from_nanos(COLOR_CONVERT_NANOS.load(Ordering::Relaxed)),
    }
}

/// An open measurement, closed by [`Timer::record`].
///
/// A guard rather than a closure because two of the three instrumented regions
/// are `unsafe` FFI sequences with their own early returns, which a closure
/// would have to be restructured around. An error path drops the timer without
/// recording, so a failed readback contributes nothing rather than contributing
/// a partial one.
///
/// `dead_code` is allowed because a target with no fixed-function backend
/// compiled in — the `unsafe` FFI modules are all `cfg`-gated — has nothing
/// that constructs one.
#[allow(dead_code)]
pub(crate) struct Timer(Instant);

#[allow(dead_code)]
impl Timer {
    /// Starts measuring.
    pub(crate) fn start() -> Self {
        Timer(Instant::now())
    }

    /// Charges everything since [`Timer::start`] to `phase`.
    pub(crate) fn record(self, phase: Phase) {
        let nanos = u64::try_from(self.0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        phase.counter().fetch_add(nanos, Ordering::Relaxed);
    }
}

/// Counts one delivered frame.
///
/// Called once per frame from the colour-conversion site, which every backend
/// runs exactly once per delivered frame; see [`Timer`] for the `dead_code`
/// allowance.
#[allow(dead_code)]
pub(crate) fn count_frame() {
    FRAMES.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The accumulators are process-wide, so the tests that touch them run
    /// under one lock rather than in parallel.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn records_each_phase_against_its_own_counter() {
        let _guard = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
        reset();
        let timer = Timer::start();
        std::thread::sleep(Duration::from_millis(2));
        timer.record(Phase::SurfaceCopy);
        count_frame();

        let report = report();
        assert_eq!(report.frames, 1);
        assert!(report.surface_copy >= Duration::from_millis(2));
        assert_eq!(report.color_convert, Duration::ZERO);
        assert_eq!(report.total(), report.surface_copy);
        assert_eq!(report.total_per_frame(), report.total());
    }

    #[test]
    fn accumulates_across_threads_and_clears_on_reset() {
        let _guard = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
        reset();
        // The frame-delivering callback need not be the submitting thread,
        // which is the reason this state is not thread-local.
        std::thread::spawn(|| {
            let timer = Timer::start();
            std::thread::sleep(Duration::from_millis(2));
            timer.record(Phase::ColorConvert);
            count_frame();
        })
        .join()
        .unwrap();

        let report = report();
        assert_eq!(report.frames, 1);
        assert!(report.color_convert >= Duration::from_millis(2));

        reset();
        assert_eq!(super::report(), Report::default());
    }

    #[test]
    fn a_window_without_frames_has_no_per_frame_cost() {
        let empty = Report {
            frames: 0,
            surface_copy: Duration::from_millis(5),
            color_convert: Duration::from_millis(5),
        };
        assert_eq!(empty.total(), Duration::from_millis(10));
        assert_eq!(empty.total_per_frame(), Duration::ZERO);
    }
}
