//! Shared helpers for zvidlib's criterion benchmarks: throughput reporting,
//! synthetic inputs, a minimal executor, and the scalar-vs-SIMD harness in
//! [`isa`].
//!
//! Each package's bench targets reach this through a local `support` module
//! that [`bench_support!`] declares. The fixtures a package's benchmarks decode
//! - the bundled samples and the checked-in streams - stay in that package's
//! own `benches/support/`, next to the codec they need, so a benchmark depends
//! on this crate, its own package, and nothing it does not measure.
//!
//! Development-only: this crate is not published.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use criterion::Throughput;
use zvidlib_core::{ColorRange, Limits, PixelFormat, Plane, VideoDimensions, VideoFrame};

pub mod isa;

/// The dispatch sites a package's benchmarks exercise, each with the
/// instruction set it resolves to right now: the concatenation of the
/// `simd_sites` functions of the crates it measures.
pub type Sites = fn() -> Vec<(&'static str, zvidlib_core::SimdIsa)>;

/// The `simd=on` / `simd=off` suffix every criterion group name carries,
/// for a package built with or without its additive `simd` feature.
pub fn simd_tag(simd_enabled: bool) -> &'static str {
    if simd_enabled { "simd=on" } else { "simd=off" }
}

/// Declares a bench target's local `support` items: the package's `simd`
/// feature switch and group naming, its dispatch sites, and the [`isa`]
/// harness bound to those sites. Everything else in this crate is re-exported
/// alongside them.
///
/// The `simd` feature is read where this expands, so it is the bench
/// package's own feature. The crate's vector kernels are selected by runtime
/// CPU feature detection, so the feature gates no code today; it exists so a
/// benchmark run records which arm it measured.
#[macro_export]
macro_rules! bench_support {
    ($($sites:path),+ $(,)?) => {
        pub use $crate::*;

        /// Whether the package was built with its additive `simd` cargo feature.
        pub fn simd_enabled() -> bool {
            cfg!(feature = "simd")
        }

        /// The `simd=on` / `simd=off` suffix every criterion group name carries.
        pub fn simd_tag() -> &'static str {
            $crate::simd_tag(simd_enabled())
        }

        /// Builds a criterion group name of the form `hevc_decode/simd=off`.
        pub fn group_name(stage: &str) -> String {
            format!("{stage}/{}", simd_tag())
        }

        /// The dispatch sites this package's benchmarks exercise.
        pub fn sites() -> Vec<(&'static str, $crate::SimdIsa)> {
            let mut sites = Vec::new();
            $(sites.extend($sites());)+
            sites
        }

        /// [`$crate::isa`], bound to this package's dispatch sites.
        pub mod isa {
            pub use $crate::isa::{AudioIsaWorkload, IsaWorkload, checksum};

            pub fn assert_bit_exact_across_isas<F>(label: &str, run: &F)
            where
                F: Fn() -> Vec<u8>,
            {
                $crate::isa::assert_bit_exact_across_isas(label, super::sites, run);
            }

            pub fn log_host_isas(criterion: &mut criterion::Criterion) {
                $crate::isa::log_host_isas(criterion, super::sites);
            }

            pub fn bench_across_isas<F>(
                criterion: &mut criterion::Criterion,
                workload: &IsaWorkload<'_>,
                run: F,
            ) where
                F: Fn() -> Vec<u8>,
            {
                $crate::isa::bench_across_isas(criterion, workload, super::sites, run);
            }

            pub fn bench_audio_across_isas<F>(
                criterion: &mut criterion::Criterion,
                workload: &AudioIsaWorkload<'_>,
                run: F,
            ) where
                F: FnMut() -> Vec<u8>,
            {
                $crate::isa::bench_audio_across_isas(criterion, workload, super::sites, run);
            }

            pub fn assert_reached_every_site(codec: &str, isa: $crate::SimdIsa) {
                $crate::isa::assert_reached_every_site(codec, isa, super::sites);
            }
        }
    };
}

pub use zvidlib_core::SimdIsa;

/// The amount of pixel work one benchmark iteration performs.
///
/// Wall time alone is not comparable across fixtures of different resolutions,
/// so benchmarks report `Throughput::Elements(frames)` (criterion prints
/// `elem/s`, i.e. frames per second) together with the megapixels each frame
/// carries, which converts that rate to megapixels per second.
#[derive(Clone, Copy, Debug)]
pub struct FrameWork {
    pub frames: u64,
    pub width: u64,
    pub height: u64,
}

impl FrameWork {
    pub fn new(frames: u64, width: u64, height: u64) -> Self {
        Self {
            frames,
            width,
            height,
        }
    }

    /// Frames per iteration, which criterion turns into a frames/sec rate.
    pub fn elements(&self) -> Throughput {
        Throughput::Elements(self.frames)
    }

    /// Megapixels touched per iteration.
    pub fn megapixels(&self) -> f64 {
        (self.frames * self.width * self.height) as f64 / 1e6
    }

    /// Megapixels per second for a measured per-iteration duration.
    pub fn megapixels_per_second(&self, per_iteration: std::time::Duration) -> f64 {
        let seconds = per_iteration.as_secs_f64();
        if seconds <= 0.0 {
            return 0.0;
        }
        self.megapixels() / seconds
    }
}

/// Registers a benchmark's throughput and prints the megapixel scale criterion's
/// own `elem/s` line does not carry.
///
/// `frames/s * megapixels-per-frame = megapixels/s`, so printing the factor once
/// per benchmark makes every reported rate convertible without re-deriving each
/// fixture's resolution.
pub fn report_throughput<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    id: &str,
    work: FrameWork,
) {
    group.throughput(work.elements());
    println!(
        "# {id}: {} frame(s)/iter at {}x{} = {:.4} Mpx/iter (frames/s x {:.4} = Mpx/s)",
        work.frames,
        work.width,
        work.height,
        work.megapixels(),
        work.megapixels() / work.frames.max(1) as f64,
    );
}

/// Minimal executor for the crate's `async` I/O entry points.
pub fn block_on<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = Box::pin(future);
    loop {
        match Pin::new(&mut future).poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

/// Decodes a hex dump, ignoring surrounding whitespace.
pub fn from_hex(hex: &str) -> Vec<u8> {
    let hex = hex.trim();
    assert_eq!(hex.len() & 1, 0, "hex fixture must have an even length");
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16)
                .expect("hex fixture must be valid hex")
        })
        .collect()
}

/// Builds a deterministic synthetic YUV420 frame sequence for encoder inputs.
///
/// Encoder benchmarks need frames without paying for a decode first, and the
/// content has a moving gradient plus low-amplitude noise so neither prediction
/// nor entropy coding degenerates into an unrepresentative best case.
pub fn synthetic_yuv420_sequence(width: u32, height: u32, frames: usize) -> Vec<VideoFrame> {
    let limits = Limits::default();
    let dimensions =
        VideoDimensions::new(width, height, &limits).expect("synthetic dimensions are valid");
    let (luma_w, luma_h) = (width as usize, height as usize);
    let (chroma_w, chroma_h) = (luma_w.div_ceil(2), luma_h.div_ceil(2));
    (0..frames)
        .map(|frame| {
            let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ frame as u64;
            let mut next_noise = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 58) as i32
            };
            let shift = (frame * 3) as i32;
            let luma = (0..luma_h)
                .flat_map(|y| (0..luma_w).map(move |x| (x, y)))
                .map(|(x, y)| {
                    let gradient = (x as i32 + y as i32 + shift) / 2;
                    ((gradient + next_noise()) & 0xff) as u8
                })
                .collect::<Vec<_>>();
            let chroma = |offset: i32| {
                (0..chroma_h)
                    .flat_map(|y| (0..chroma_w).map(move |x| (x, y)))
                    .map(|(x, y)| (128 + ((x as i32 - y as i32 + shift + offset) % 24) - 12) as u8)
                    .collect::<Vec<_>>()
            };
            VideoFrame::new(
                dimensions,
                PixelFormat::Yuv420p8,
                ColorRange::Limited,
                vec![
                    Plane {
                        data: luma,
                        stride: luma_w,
                    },
                    Plane {
                        data: chroma(0),
                        stride: chroma_w,
                    },
                    Plane {
                        data: chroma(7),
                        stride: chroma_w,
                    },
                ],
                &limits,
            )
            .expect("synthetic YUV420 frames are valid")
        })
        .collect()
}

/// Deterministic 8-bit monochrome planes for the AV1 encoder.
///
/// Borrows [`synthetic_yuv420_sequence`]'s luma so the encoded stream carries
/// the same moving gradient plus low-amplitude noise every other synthetic
/// fixture does, rather than content that degenerates into a best case.
pub fn av1_gray8_planes(width: u32, height: u32, frames: usize) -> Vec<Vec<u8>> {
    synthetic_yuv420_sequence(width, height, frames)
        .into_iter()
        .map(|frame| frame.planes.into_iter().next().expect("luma plane").data)
        .collect()
}

/// The amount of audio work one benchmark iteration performs.
///
/// Audio has no pixels, so [`FrameWork`]'s megapixel scale says nothing about
/// it. The comparable pair for a sample-clock workload is the one the AAC decode
/// groups report: `Throughput::Elements(samples)`, which criterion prints as a
/// per-channel samples/sec rate, and the x-realtime factor that rate divides
/// into — a track covering 32 s of audio muxed in 8 ms is 4000x realtime.
/// Both sides of the write path and both sides of the read path report on this
/// same scale, so mux, demux, and decode numbers are directly comparable.
#[derive(Clone, Copy, Debug)]
pub struct AudioWork {
    /// Samples per channel, i.e. the length of the covered sample interval.
    pub samples: u64,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioWork {
    pub fn new(samples: u64, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    /// Samples per iteration, which criterion turns into a samples/sec rate.
    pub fn elements(&self) -> Throughput {
        Throughput::Elements(self.samples)
    }

    /// Seconds of audio one iteration covers.
    pub fn seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples as f64 / f64::from(self.sample_rate)
    }

    /// How many times faster than realtime a measured iteration ran.
    ///
    /// This is the playback-relevant figure: anything at or below 1.0 cannot
    /// sustain real-time output, and the margin above it is the headroom the
    /// path under test leaves for everything else on the thread.
    pub fn realtime_factor(&self, per_iteration: std::time::Duration) -> f64 {
        let elapsed = per_iteration.as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }
        self.seconds() / elapsed
    }

    /// Samples per second for a measured per-iteration duration.
    pub fn samples_per_second(&self, per_iteration: std::time::Duration) -> f64 {
        let elapsed = per_iteration.as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }
        self.samples as f64 / elapsed
    }
}

/// Registers an audio benchmark's throughput and prints the realtime scale
/// criterion's own `elem/s` line does not carry.
///
/// `samples/s / sample_rate = x-realtime`, so printing the sample rate and the
/// covered duration once per benchmark makes every reported rate convertible
/// without re-deriving the fixture's timing.
pub fn report_audio_throughput<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    id: &str,
    work: AudioWork,
) {
    group.throughput(work.elements());
    println!(
        "# {id}: {} sample(s)/iter at {} Hz x{}ch = {:.4} s of audio/iter (samples/s / {} = x-realtime)",
        work.samples,
        work.sample_rate,
        work.channels,
        work.seconds(),
        work.sample_rate,
    );
}

/// Builds a deterministic synthetic RGBA8 frame sequence for encoder inputs.
///
/// [`synthetic_yuv420_sequence`] is the right input for the encoder's *later*
/// stages, which consume YUV420 planes. The public HEVC encoder's own input
/// format is RGBA8, so a whole-frame encode benchmark needs the same content in
/// that format: the same moving gradient plus low-amplitude noise, so neither
/// prediction nor entropy coding degenerates into an unrepresentative best case,
/// and still no decode cost folded into the measurement.
pub fn synthetic_rgba8_sequence(width: u32, height: u32, frames: usize) -> Vec<VideoFrame> {
    let limits = Limits::default();
    let dimensions =
        VideoDimensions::new(width, height, &limits).expect("synthetic dimensions are valid");
    let (w, h) = (width as usize, height as usize);
    (0..frames)
        .map(|frame| {
            let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ frame as u64;
            let mut next_noise = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 58) as i32
            };
            let shift = (frame * 3) as i32;
            let mut data = Vec::with_capacity(w * h * 4);
            for y in 0..h {
                for x in 0..w {
                    let gradient = (x as i32 + y as i32 + shift) / 2;
                    let noise = next_noise();
                    data.push(((gradient + noise) & 0xff) as u8);
                    data.push(((gradient / 2 + (x as i32 - y as i32 + shift)) & 0xff) as u8);
                    data.push(((gradient / 3 + noise * 2) & 0xff) as u8);
                    data.push(0xff);
                }
            }
            VideoFrame::new(
                dimensions,
                PixelFormat::Rgba8,
                ColorRange::Limited,
                vec![Plane {
                    data,
                    stride: w * 4,
                }],
                &limits,
            )
            .expect("synthetic RGBA8 frames are valid")
        })
        .collect()
}
