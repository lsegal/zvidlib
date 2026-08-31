//! The platform hardware HEVC decoders measured against the software decoder.
//!
//! zvidlib ships three fixed-function HEVC backends behind
//! `native_hevc_video_decoder_factory` — NVDEC (`src/hevc/nvdec.rs`), Windows
//! Media Foundation (`src/hevc/windows_mf.rs`), and VideoToolbox
//! (`src/hevc/videotoolbox.rs`) — with the pure-Rust software decoder as the
//! fallback. This group quantifies what that fallback costs on whichever
//! backend the host actually provides.
//!
//! # No scalar-vs-SIMD axis
//!
//! Unlike every other group in this suite, the arms here are *not* built
//! through [`support::isa`]. These decoders are opaque drivers and OS
//! frameworks; `zvidlib::simd`'s process-wide override does not reach a single
//! instruction they execute. Emitting `scalar` and `neon` arms for them would
//! produce a pair of numbers that differ only by measurement noise and read as
//! a result. The meaningful comparison is hardware against software on the same
//! input, and that is the only comparison this module makes.
//!
//! The group name likewise carries no `simd=on`/`simd=off` build tag. That tag
//! exists to keep two *builds* of the crate's own kernels separately recorded;
//! the hardware arms are identical in both builds, so tagging them would split
//! one measurement across two names for no reason. The software comparison arm
//! is the crate's own decoder and does vary by build, so it keeps the tag.
//!
//! # Setup latency versus steady state
//!
//! A hardware backend pays a real one-time cost before it decodes anything: a
//! CUDA context and parser, an MFT plus its D3D11 device, or a VideoToolbox
//! decompression session. Folding that into a throughput average misrepresents
//! both numbers — it makes the throughput look worse than a playback pipeline
//! ever sees, and hides a latency a seek-heavy caller pays on every reset. So
//! the two are measured as separate benchmarks, both through
//! `Bencher::iter_custom`: [`session_setup`] times construction through the
//! first delivered frame, and [`steady_state`] starts its clock only after that
//! first frame is already out.
//!
//! # Frame readback
//!
//! The issue this module closes asks for the surface-copy cost as its own arm
//! "where the backend exposes one". None of the three do today: `VideoDecoder`
//! hands back a host-side [`zvidlib::VideoFrame`], the decoder configuration
//! only accepts `PixelFormat::Rgba8` (`src/hevc/mod.rs`), and each backend maps
//! its own surface and converts to RGBA inside `submit`. There is no public
//! seam between "the fixed-function block finished" and "the pixels are in a
//! `Vec<u8>`", so a separate readback number would have to be a reimplemented
//! stand-in rather than a measurement of the code that runs. The group reports
//! that the two are inseparable instead of reporting a proxy.

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion};
use zvidlib::{
    CancellationToken, CodecSupport, EncodedVideoSample, HardwarePreference, Limits,
    VideoDecoderConfig, VideoDecoderFactory, native_hevc_video_decoder_factory,
};

use super::support::{self, FrameWork, group_name};

/// Frames one steady-state hardware iteration decodes past the first.
///
/// Large enough that the per-iteration `iter_custom` bookkeeping and the first
/// frame excluded from the clock are both negligible, small enough that one
/// iteration stays well under a second on a backend decoding 1080p faster than
/// real time.
const HARDWARE_FRAMES: u64 = 32;

/// Frames one steady-state software iteration decodes past the first.
///
/// The software decoder runs single-digit fps on the bundled 1080p sample, so
/// it measures a shorter window than the hardware arm. Both arms report
/// megapixels per second, which is resolution- and count-independent, so the
/// ratio stays meaningful across the two window sizes.
const SOFTWARE_FRAMES: u64 = 4;

/// Environment variable that opts into the slow software comparison arm.
///
/// Shared with the other groups that decode the bundled 1080p sample through
/// the software decoder; see `benches/codec.rs`.
const LARGE_GROUP_ENV: &str = "ZVIDLIB_BENCH_LARGE";

/// The backends compiled in for this target, for the group's log lines.
///
/// `native_hevc_video_decoder_factory` tries NVDEC, then Media Foundation, then
/// VideoToolbox, and does not report which one it settled on, so on Windows
/// this names both candidates rather than guessing.
const fn compiled_backends() -> &'static str {
    if cfg!(target_os = "macos") {
        "VideoToolbox"
    } else if cfg!(windows) {
        "NVDEC or Media Foundation, whichever initialized first"
    } else if cfg!(target_os = "linux") {
        "NVDEC"
    } else {
        "none for this target"
    }
}

/// The bundled sample's decoder configuration at a given hardware preference.
fn configuration(hardware: HardwarePreference) -> VideoDecoderConfig {
    let mut configuration = support::bundled_hevc_sample().configuration.clone();
    configuration.hardware = hardware;
    configuration
}

/// Decodes until `frames` have been delivered, returning the wall time spent
/// after the first one.
///
/// The split is the whole point of this module: `setup` covers decoder
/// construction and everything up to the first delivered frame, `steady` covers
/// only the frames after it. Both are measured in one pass so the two numbers
/// describe the same session rather than two differently warmed ones.
fn timed_decode(
    factory: &dyn VideoDecoderFactory,
    configuration: &VideoDecoderConfig,
    samples: &[EncodedVideoSample],
    frames: u64,
) -> (Duration, Duration) {
    let cancellation = CancellationToken::new();
    let started = Instant::now();
    let mut decoder = factory
        .create(configuration, &Limits::default())
        .expect("the decoder is constructible after its capability was checked");
    let mut delivered = 0_u64;
    let mut first_frame_at = None;
    for sample in samples {
        for decoded in decoder
            .submit(sample, &cancellation)
            .expect("the bundled sample decodes")
        {
            black_box(&decoded.frame);
            delivered += 1;
            if first_frame_at.is_none() {
                first_frame_at = Some(started.elapsed());
            }
            if delivered > frames {
                let setup = first_frame_at.unwrap_or_default();
                return (setup, started.elapsed() - setup);
            }
        }
    }
    panic!("the bundled sample yields at least {frames} frames past the first");
}

/// Registers one arm and prints its setup and steady-state numbers.
///
/// Criterion measures the two as separate benchmarks; the single untimed pass
/// here is what puts both halves, and the megapixel scale they convert to, on
/// one line in the run's output so a reader does not have to correlate two
/// criterion reports to get the ratio the issue asks for.
fn bench_arm(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    configuration: &VideoDecoderConfig,
    frames: u64,
) -> f64 {
    let factory = native_hevc_video_decoder_factory();
    let sample = support::bundled_hevc_sample();
    let samples = &sample.samples;
    let work = FrameWork::new(frames, sample.width, sample.height);
    let setup_work = FrameWork::new(1, sample.width, sample.height);

    let (setup, steady) = timed_decode(&factory, configuration, samples, frames);
    let rate = work.megapixels_per_second(steady);
    println!(
        "# {label}: session setup to first frame {:.1} ms; {frames} further frame(s) in {:.4}s \
         => {:.1} fps, {rate:.1} Mpx/s",
        setup.as_secs_f64() * 1e3,
        steady.as_secs_f64(),
        frames as f64 / steady.as_secs_f64(),
    );

    group.throughput(setup_work.elements());
    group.bench_function(format!("{label}/session_setup_to_first_frame"), |bencher| {
        bencher.iter_custom(|iterations| {
            (0..iterations)
                .map(|_| timed_decode(&factory, configuration, samples, frames).0)
                .sum()
        });
    });

    group.throughput(work.elements());
    group.bench_function(format!("{label}/steady_state"), |bencher| {
        bencher.iter_custom(|iterations| {
            (0..iterations)
                .map(|_| timed_decode(&factory, configuration, samples, frames).1)
                .sum()
        });
    });

    rate
}

/// Hardware HEVC decode against the software decoder on the bundled 1080p
/// sample.
///
/// Skips with a message rather than failing when the host has no hardware
/// decoder, the way `tests/native_hevc_hardware.rs` does: a dev box without
/// NVDEC must still be able to run `cargo bench`.
pub fn hevc_hardware(criterion: &mut Criterion) {
    let factory = native_hevc_video_decoder_factory();
    let hardware = configuration(HardwarePreference::Require);
    if factory.capability(&hardware)
        != (CodecSupport::Supported {
            implementation: zvidlib::CodecImplementation::Hardware,
        })
    {
        let reason = factory
            .create(&hardware, &Limits::default())
            .err()
            .map_or_else(|| "unknown reason".into(), |error| error.to_string());
        println!(
            "# skipping the hardware HEVC group: no hardware decoder on this host ({reason}); \
             compiled backends: {}",
            compiled_backends()
        );
        return;
    }

    println!(
        "# hardware HEVC group: {} decoding examples/media/BigBuckBunny.mp4",
        compiled_backends()
    );
    println!(
        "# frame readback is not a separate arm: every backend maps its own surface and converts \
         to RGBA inside `submit`, so the copy is inside the decode number by construction"
    );

    let mut group = criterion.benchmark_group("hevc_hardware");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));
    let hardware_rate = bench_arm(&mut group, "hardware", &hardware, HARDWARE_FRAMES);
    group.finish();

    if std::env::var_os(LARGE_GROUP_ENV).is_none() {
        println!(
            "# skipping the software comparison arm; set {LARGE_GROUP_ENV}=1 to run it and get \
             the hardware-vs-software ratio"
        );
        return;
    }

    let software = configuration(HardwarePreference::Avoid);
    let mut group = criterion.benchmark_group(group_name("hevc_hardware_software_baseline"));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(20));
    let software_rate = bench_arm(&mut group, "software", &software, SOFTWARE_FRAMES);
    group.finish();

    if software_rate > 0.0 {
        println!(
            "# hardware-vs-software: {:.1} Mpx/s vs {:.1} Mpx/s = {:.1}x on {}",
            hardware_rate,
            software_rate,
            hardware_rate / software_rate,
            compiled_backends()
        );
    }
}
