//! Where HEVC decode time actually goes, on the bundled 1080p sample.
//!
//! The per-stage criterion groups in `benches/hevc_decode.rs` time each kernel
//! on a synthetic workload of its own: they say how fast §8.7.3 SAO is, not how
//! much SAO a 1080p frame runs. Issue #189 is the gap that leaves — the
//! whole-frame `hevc_decode/<isa>` arms move only ~1.06x while the individual
//! kernels measure 1.3x-2.4x, and neither number explains the other without a
//! share-of-total breakdown.
//!
//! This example produces that breakdown. It decodes frames of
//! `examples/media/BigBuckBunny.mp4` through the ordinary public decoder with
//! `zvidlib::hevc_decode_profile` running, then prints each stage's exclusive
//! share, the fraction of the decode the crate's vector kernels cover, and the
//! Amdahl ceiling that fraction implies.
//!
//! It is an example rather than a benchmark on purpose: criterion reports the
//! distribution of one workload's wall time, and the question here is how a
//! single decode divides across stages, which is a composition rather than a
//! measurement to regress against.
//!
//! ```sh
//! cargo run --release --example hevc_decode_profile
//! cargo run --release --example hevc_decode_profile -- 120     # frame count
//! cargo run --release --example hevc_decode_profile -- 60 scalar
//! ```
//!
//! The second argument pins an instruction set through `zvidlib::simd`, so the
//! same breakdown can be read with and without the vector kernels. Run under
//! `--release`: the decoder is a pure-Rust software codec and a debug build's
//! stage mix is not the shipped build's.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use zvidlib::hevc_decode_profile as profile;
use zvidlib::io::MemorySource;
use zvidlib::simd::{self, SimdIsa};
use zvidlib::{
    CancellationToken, Codec, CodecProfile, ColorRange, EncodedVideoSample, HardwarePreference,
    Limits, Mp4Demuxer, Mp4DemuxerOptions, PixelFormat, VideoDecoderConfig, VideoDecoderFactory,
    native_hevc_video_decoder_factory,
};

/// Frames to decode when the caller names no count.
///
/// Enough to cross several inter-coded pictures — an I-frame-only profile would
/// read as an intra decoder — and few enough to finish in seconds.
const DEFAULT_FRAMES: usize = 48;

fn main() {
    let mut args = std::env::args().skip(1);
    let frames: usize = args
        .next()
        .map_or(DEFAULT_FRAMES, |arg| arg.parse().expect("frame count"));
    let isa = args.next().map(|name| parse_isa(&name));

    if let Some(isa) = isa {
        simd::set_override(Some(isa));
    }
    let active = simd::active();

    let (configuration, samples) = bundled_sample();
    let factory = native_hevc_video_decoder_factory();
    let mut decoder = factory
        .create(&configuration, &Limits::default())
        .expect("the software HEVC decoder is constructible");
    let cancellation = CancellationToken::new();

    // Decode a few access units before profiling so the report covers steady
    // state rather than the first picture's parameter-set activation and
    // allocation.
    let mut warmup = 0usize;
    let mut iter = samples.iter();
    for encoded in iter.by_ref() {
        warmup += decoder
            .submit(encoded, &cancellation)
            .expect("the bundled sample decodes")
            .len();
        if warmup >= 2 {
            break;
        }
    }

    assert!(
        profile::start(),
        "stage profiling needs a monotonic clock; this target has none"
    );
    let mut decoded = 0usize;
    for encoded in iter {
        decoded += decoder
            .submit(encoded, &cancellation)
            .expect("the bundled sample decodes")
            .len();
        if decoded >= frames {
            break;
        }
    }
    let report = profile::finish().expect("profiling was started");

    if isa.is_some() {
        simd::set_override(None);
    }

    println!("# HEVC decode stage attribution");
    println!();
    println!(
        "{decoded} frames of examples/media/BigBuckBunny.mp4, instruction set `{active:?}`, \
         {:.1} ms total ({:.2} ms/frame)",
        report.total().as_secs_f64() * 1e3,
        report.total().as_secs_f64() * 1e3 / decoded.max(1) as f64,
    );
    println!();
    print!("{}", report.markdown_table(decoded));
    println!();

    println!(
        "Vectorized stages cover {:.1}% of the measured total, and {:.1}% of decode \
         proper (total minus `color_convert`, which is output conversion, not decoding).",
        report.vectorized_share() * 100.0,
        report.vectorized_decode_share() * 100.0,
    );
    println!(
        "Amdahl ceiling on the measured total: infinitely fast vector kernels would run \
         {:.2}x faster; a uniform 2x on those stages gives {:.2}x, a uniform 4x gives {:.2}x.",
        report.max_whole_frame_speedup(),
        report.speedup_at(2.0),
        report.speedup_at(4.0),
    );
    println!(
        "`color_convert` alone is {:.1}% of the measured total and has no vector kernel.",
        report.share(profile::Stage::ColorConvert) * 100.0,
    );
    println!(
        "Serial entropy decoding (`slice_data_cabac` + `residual_cabac`) is {:.1}% of the total.",
        (report.share(profile::Stage::SliceData) + report.share(profile::Stage::Residual)) * 100.0,
    );
    println!(
        "Profiler overhead is at most {:.1} ms of that ({:.1}%, {} scopes).",
        report.overhead().as_secs_f64() * 1e3,
        report.overhead().as_secs_f64() / report.total().as_secs_f64() * 100.0,
        report.scopes(),
    );
}

/// Maps a command-line instruction-set name onto [`SimdIsa`].
fn parse_isa(name: &str) -> SimdIsa {
    simd::available()
        .iter()
        .copied()
        .find(|isa| format!("{isa:?}").eq_ignore_ascii_case(name))
        .unwrap_or_else(|| {
            panic!(
                "unknown instruction set {name:?}; this host offers {:?}",
                simd::available()
            )
        })
}

/// Demuxes the bundled sample into a decoder configuration and its access units.
fn bundled_sample() -> (VideoDecoderConfig, Vec<EncodedVideoSample>) {
    let limits = Limits::default();
    let source = MemorySource::new(include_bytes!("media/BigBuckBunny.mp4").to_vec());
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))
        .expect("the bundled sample is a readable MP4");
    let track = movie.track(1).expect("the bundled sample has track 1");
    let dimensions = track
        .dimensions
        .expect("the bundled sample's video track is dimensioned");
    let samples = block_on(track.to_encoded_video_samples(&source, &limits))
        .expect("the bundled sample's video samples are readable");
    (
        VideoDecoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: dimensions,
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            // The question is where *this crate's* decode time goes, not the
            // host's fixed-function block's.
            hardware: HardwarePreference::Avoid,
            configuration: track.decoder_config.clone(),
        },
        samples,
    )
}

/// Minimal executor for the crate's `async` I/O entry points.
fn block_on<T>(future: impl Future<Output = T>) -> T {
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
