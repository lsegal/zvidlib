//! What the drag preview cadence costs the frame under the pointer.
//!
//! Issue #363 gave a drag its picture back by publishing on a cadence during the walk to the
//! frame under the pointer, and issue #379 is the measurement that was never taken: the 150 ms
//! `PREVIEW_INTERVAL` in `examples/native_gl/scrub.rs` was arithmetic over #355's per-frame
//! numbers (2.9 ms to decode a frame, 6.6 ms to convert a published one to RGBA) against #354's
//! 1.7 s walk, and nothing had timed the two together.
//!
//! This example times them. It drives the same [`scrub::FrameService`] the window drives, over
//! the same bundled 1080p sample through the same hardware decoder, with no window and no
//! renderer in the way, and reports for each cadence:
//!
//! * how long the frame under the pointer takes to arrive, against a baseline that publishes
//!   nothing on the way to it,
//! * how many pictures the walk publishes, and
//! * what each published picture costs, which is the difference between those two divided by the
//!   count rather than a per-picture figure measured somewhere else.
//!
//! The baseline is [`scrub::FrameServiceSource`], playback's path: an exact request decodes the
//! frames before its target for reference only and converts none of them, which is exactly what
//! the drag did between #355 and #363. So the overhead each arm reports is the cadence's own,
//! measured against the same decoder in the same process.
//!
//! It is an example rather than a criterion benchmark for the reason
//! `examples/hevc_decode_profile.rs` is: this is one composition of a walk - arrival, publish
//! count, cost per publish - and not a distribution to regress a kernel against. Each arm builds
//! its own [`scrub::FrameService`], so every walk starts from a cold decoder at a random-access
//! point, as a drag from a standing start does.
//!
//! ```sh
//! cargo run --release --features native --example scrub_preview_profile
//! cargo run --release --features native --example scrub_preview_profile -- 5
//! cargo run --release --features native --example scrub_preview_profile -- 3 80 150 400
//! ```
//!
//! The first argument is how many times each arm runs; the rest are the cadences in
//! milliseconds. Every arm reports the *fastest* of its runs: the decoder is shared hardware and
//! anything else on the host only ever adds time, so the minimum is the arm's own cost and the
//! mean is the host's. Run under `--release`; a debug build's conversion pass is not the shipped
//! build's, and the conversion is what is being charged for.

use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use zvidlib::io::MemorySource;
use zvidlib::{
    CancellationToken, Codec, CodecProfile, ColorRange, EncodedVideoSample, ErrorKind, FrameIndex,
    HardwarePreference, Limits, Mp4Demuxer, Mp4DemuxerOptions, PixelFormat, PlaybackVideoSource,
    VideoDecoderConfig, VideoDecoderFactory, native_hevc_video_decoder_factory,
};

#[path = "native_gl/scrub.rs"]
mod scrub;

use scrub::{FrameService, target_frame};

/// Runs of each arm when the caller names no count.
///
/// Three is enough for the minimum to settle: the arms differ by tens of percent and the run to
/// run spread on an idle host is a couple.
const DEFAULT_RUNS: usize = 3;

/// The cadences swept when the caller names none, in milliseconds.
///
/// The issue asks for 80 ms to 400 ms, which brackets the shipped 150 ms by roughly a factor of
/// two either way - far enough out that a cost proportional to the publish count has to show.
const DEFAULT_INTERVALS_MS: &[u64] = &[80, 100, 150, 200, 300, 400];

/// How often the harness collects, standing in for the render thread's poll.
///
/// The window collects on a redraw, so it would quantise every arrival to a vsync interval it
/// shares with every other arm. Polling faster than that measures the service rather than the
/// display, and the vsync wait is added back by the window either way.
const POLL_INTERVAL: Duration = Duration::from_millis(2);

/// What one walk to the frame under the pointer cost.
struct Walk {
    /// Time from the request to the target frame being collected.
    arrival: Duration,
    /// Pictures the walk published, the target itself included. Counted by the service rather
    /// than by what this harness collected: a picture the render thread never draws still cost
    /// its conversion, and a poll that finds two only draws the newer.
    published: u64,
    /// Pictures the walk converted to RGBA. A published picture converts the frames behind it
    /// that fit in the reader's cache as well as itself, so this is the larger number and it is
    /// the one the time is actually going into.
    converted: u64,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let runs: usize = args
        .next()
        .map_or(DEFAULT_RUNS, |arg| arg.parse().expect("run count"));
    let intervals: Vec<u64> = {
        let named: Vec<u64> = args
            .map(|arg| arg.parse().expect("interval in ms"))
            .collect();
        if named.is_empty() {
            DEFAULT_INTERVALS_MS.to_vec()
        } else {
            named
        }
    };

    let (configuration, samples) = bundled_sample();
    let factory = native_hevc_video_decoder_factory();
    let frame_count = samples.len() as u64;
    // The far end of the bar: the frame a drag released against the right edge asks for, and the
    // one #354 and #363 both quote their seconds for.
    let target = target_frame(1.0, frame_count);
    println!(
        "{} frames, {:?} through {:?}; walking to frame {target} from cold, {runs} run(s) each",
        frame_count,
        configuration.codec,
        factory.capability(&configuration),
    );

    let baseline = (0..runs)
        .map(|_| exact_walk(&factory, &configuration, &samples, target))
        .min()
        .expect("at least one run");
    println!(
        "\nno previews (playback's exact request, which is what #355 left the drag doing)\n  \
         frame {target} arrives in {:>7.0} ms, publishing 1 picture",
        baseline.as_secs_f64() * 1000.0,
    );

    // `spacing` is what issue #363 actually asked for - how often the picture under the pointer
    // moves - and `per convert` is what the time is going into. The interval is a budget for a
    // stride, not the spacing itself: a stride is capped, and a published picture converts the
    // cache tail behind it as well as itself, so neither column follows the interval directly.
    println!(
        "\n{:>9}  {:>10}  {:>9}  {:>9}  {:>9}  {:>9}  {:>11}",
        "interval", "arrival", "published", "spacing", "converted", "overhead", "per convert"
    );
    for milliseconds in intervals {
        let interval = Duration::from_millis(milliseconds);
        let walk = (0..runs)
            .map(|_| preview_walk(&factory, &configuration, &samples, target, interval))
            .min_by_key(|walk| walk.arrival)
            .expect("at least one run");
        let overhead = walk.arrival.saturating_sub(baseline);
        // The cadence's whole cost divided by the pictures it converted, which is the unit that
        // stays put across the sweep: what a published picture costs depends on how many frames
        // behind it the reader converts with it, and that varies with the stride.
        let per_convert = overhead
            .checked_div(u32::try_from(walk.converted.max(1)).unwrap_or(u32::MAX))
            .unwrap_or_default();
        let spacing = walk
            .arrival
            .checked_div(u32::try_from(walk.published.max(1)).unwrap_or(u32::MAX))
            .unwrap_or_default();
        println!(
            "{milliseconds:>6} ms  {:>7.0} ms  {:>9}  {:>6.0} ms  {:>9}  {:>8.0}%  {:>8.1} ms",
            walk.arrival.as_secs_f64() * 1000.0,
            walk.published,
            spacing.as_secs_f64() * 1000.0,
            walk.converted,
            overhead.as_secs_f64() / baseline.as_secs_f64() * 100.0,
            per_convert.as_secs_f64() * 1000.0,
        );
    }
}

/// Times a drag's walk to `target` at `interval`, from a cold decoder.
fn preview_walk(
    factory: &dyn VideoDecoderFactory,
    configuration: &VideoDecoderConfig,
    samples: &[EncodedVideoSample],
    target: u64,
    interval: Duration,
) -> Walk {
    let mut frames = FrameService::with_preview_interval(
        factory,
        configuration.clone(),
        samples.to_vec(),
        Limits::default(),
        interval,
    )
    .expect("the frame service opens over the bundled sample");
    let started = Instant::now();
    frames.request(target);
    loop {
        // Collecting is what the render thread does with a picture, and it is what frees the
        // delivery queue, so a harness that never collected would not be timing a drag.
        match frames.take_latest() {
            Some((index, _frame)) if index == target => {
                let arrival = started.elapsed();
                let (published, statistics) = frames.published();
                return Walk {
                    arrival,
                    published,
                    converted: statistics
                        .samples_submitted
                        .saturating_sub(statistics.samples_skipped),
                };
            }
            Some(_) => {}
            None => thread::sleep(POLL_INTERVAL),
        }
    }
}

/// Times playback's exact request for `target`, which publishes nothing on the way to it.
fn exact_walk(
    factory: &dyn VideoDecoderFactory,
    configuration: &VideoDecoderConfig,
    samples: &[EncodedVideoSample],
    target: u64,
) -> Duration {
    let frames = FrameService::new(
        factory,
        configuration.clone(),
        samples.to_vec(),
        Limits::default(),
    )
    .expect("the frame service opens over the bundled sample");
    let mut source = frames.source();
    let cancellation = CancellationToken::new();
    let started = Instant::now();
    loop {
        match source.get_exact(FrameIndex(target), &cancellation) {
            Ok(_) => return started.elapsed(),
            Err(error) if error.kind() == ErrorKind::WouldBlock => thread::sleep(POLL_INTERVAL),
            Err(error) => panic!("the walk failed: {}", error.message()),
        }
    }
}

/// The bundled 1080p HEVC sample's decoder configuration and samples in presentation order.
fn bundled_sample() -> (VideoDecoderConfig, Vec<EncodedVideoSample>) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/media/BigBuckBunny.mp4");
    let bytes = std::fs::read(&path).expect("the bundled sample is checked in");
    let source = MemorySource::new(bytes);
    let demuxer = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))
        .expect("the bundled sample demuxes");
    let track = demuxer
        .tracks
        .iter()
        .find(|track| track.kind == zvidlib::TrackKind::Video)
        .expect("the bundled sample has a video track");
    let limits = Limits::default();
    let samples = block_on(track.to_encoded_video_samples(&source, &limits))
        .expect("the video track's samples are readable");
    let configuration = VideoDecoderConfig {
        codec: Codec::Hevc,
        profile: CodecProfile::HevcMain,
        coded_dimensions: track
            .dimensions
            .expect("the video track reports dimensions"),
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Prefer,
        configuration: track.decoder_config.clone(),
    };
    (configuration, samples)
}

/// Drives a future to completion on this thread; the demuxer's reads are all in memory here.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    let mut future = Box::pin(future);
    let waker = std::task::Waker::noop();
    let mut context = std::task::Context::from_waker(waker);
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
    }
}
