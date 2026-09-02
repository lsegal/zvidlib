//! What an exact frame at an arbitrary point of a track costs, by backend and
//! by how often the track lets a decode restart.
//!
//! Issue #374 asked for "<50 ms to any part of a video" and #383 answered the
//! *scrub* half of it with a background index of shrunk pictures. It did not
//! change what reaching the **exact** frame costs, and this target is what
//! measures that remaining cost so a decision about it rests on figures rather
//! than on the one host and one clip #383 happened to instrument.
//!
//! # The two axes
//!
//! **Backend.** [`zvidlib::HardwarePreference::Avoid`] measures the crate's own
//! software decoder, `Prefer` measures whichever fixed-function block the host
//! provides (VideoToolbox, NVDEC, or Media Foundation). A host with no hardware
//! backend skips those arms rather than silently reporting a second software
//! number under a hardware name.
//!
//! **Random-access cadence.** This is the axis the issue exists for. A seek to
//! presentation frame *n* must decode every sample from the nearest preceding
//! random-access point, so the cadence — not the frame rate, resolution or
//! codec — is what sets the length of that walk. The bundled 1080p sample codes
//! its 768 frames as a single group of pictures, so on its own it can only ever
//! describe the worst case. `support::gop_cadence_tracks()` supplies a pair
//! encoded from the same source at the same size and quality, differing only in
//! `keyint`: one random-access point against twenty-four. Measuring both is the
//! whole point — the two cases were previously assumed to behave alike, and the
//! numbers this target prints are what separates them.
//!
//! # What is on the clock
//!
//! [`ExactFrameReader`] construction is *not*: `benches/hevc_hardware.rs`
//! already measures session setup as its own arm, and folding it in here would
//! mix a one-time driver cost into a per-seek one. Each iteration builds a fresh
//! reader off the clock — so no cache, no decoder position, nothing warm — and
//! times a single [`ExactFrameReader::get`] to the target frame. That is exactly
//! the cold seek a caller pays when a user drops the playhead somewhere new.
//!
//! # The preview arm
//!
//! [`preview_lookup`] times [`zvidlib::PreviewIndex::nearest`] over the same
//! track, once its background pass has covered it. It decodes nothing: it is a
//! lock, a search and a clone of a picture that already exists. It is here
//! because a decision between "make the exact seek faster" and "answer a
//! different question immediately" needs both numbers in the same units, from
//! the same run, on the same host.

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::{
    CancellationToken, ExactFrameReader, FrameIndex, HardwarePreference, Limits, PreviewIndex,
    PreviewOptions, VideoDecoderConfig, VideoDecoderFactory, native_hevc_video_decoder_factory,
};

mod support;

use support::{GopCadenceTrack, group_name};

/// Where along the track the measured seek lands.
///
/// Four fifths of the way is the position #374 and #383 both used, and it is
/// deliberately not the last frame: the last frame is the single worst case and
/// would read as the answer for "an arbitrary point", which it is not. On the
/// 768-frame fixtures this is frame 613 — 613 samples of reference decoding from
/// the only random-access point on the `raps=1` arm, and 5 from the nearest one
/// on `raps=24`.
const TARGET_NUMERATOR: u64 = 4;
const TARGET_DENOMINATOR: u64 = 5;

/// Environment variable that opts into the slow software arms.
///
/// Shared with the other groups that decode a whole group of pictures through
/// the software decoder; see `benches/hevc_decode.rs`. A single `raps=1`
/// software seek decodes 613 pictures, which is minutes rather than seconds, so
/// it is not part of a default `cargo bench` run.
const LARGE_GROUP_ENV: &str = "ZVIDLIB_BENCH_LARGE";

/// The frame an arm seeks to on a track of `frames` frames.
fn target_frame(frames: usize) -> FrameIndex {
    FrameIndex(frames as u64 * TARGET_NUMERATOR / TARGET_DENOMINATOR)
}

/// A track's configuration at a given hardware preference.
fn configuration(track: &GopCadenceTrack, hardware: HardwarePreference) -> VideoDecoderConfig {
    let mut configuration = track.configuration.clone();
    configuration.hardware = hardware;
    configuration
}

/// Whether the host can actually create a decoder for `configuration`.
///
/// `HardwarePreference::Prefer` falls back to software when no fixed-function
/// backend is present, so a "hardware" arm on such a host would be a second
/// software measurement wearing the wrong name. This asks the factory instead of
/// assuming, and the caller skips the arm when it says no.
fn supported(configuration: &VideoDecoderConfig) -> bool {
    native_hevc_video_decoder_factory()
        .capability(configuration)
        .is_supported()
}

/// Times `count` cold seeks to `target`, building a fresh reader for each one
/// off the clock.
fn timed_seeks(
    configuration: &VideoDecoderConfig,
    track: &GopCadenceTrack,
    target: FrameIndex,
    count: u64,
) -> Duration {
    let factory = native_hevc_video_decoder_factory();
    let cancellation = CancellationToken::new();
    let mut total = Duration::ZERO;
    for _ in 0..count {
        let mut reader = ExactFrameReader::new(
            &factory,
            configuration.clone(),
            track.samples.clone(),
            Limits::default(),
        )
        .expect("the fixture's decoder is constructible after its capability was checked");
        let started = Instant::now();
        let frame = reader
            .get(target, &cancellation)
            .expect("the fixture's target frame decodes");
        total += started.elapsed();
        black_box(&frame);
    }
    total
}

/// The exact-seek group: every (cadence, backend) pair the host can measure.
fn exact_seek(criterion: &mut Criterion) {
    let large = std::env::var_os(LARGE_GROUP_ENV).is_some();
    let mut group = criterion.benchmark_group(group_name("exact_seek"));
    // One cold `raps=1` seek is a whole group of pictures on either backend, so
    // criterion's default hundred samples would be hours. The arms are timed by
    // `iter_custom` with an explicit sample count instead.
    group.sample_size(10);
    for track in support::gop_cadence_tracks() {
        let target = target_frame(track.samples.len());
        println!(
            "# {}: {} frames, {} random-access point(s), seeking to frame {}",
            track.label,
            track.samples.len(),
            track.random_access_points,
            target.0,
        );
        for (backend, hardware) in [
            ("hardware", HardwarePreference::Prefer),
            ("software", HardwarePreference::Avoid),
        ] {
            let configuration = configuration(track, hardware);
            if hardware == HardwarePreference::Prefer && !cfg!(any(target_os = "macos", windows)) {
                println!("# skipping {}/{backend}: no hardware backend", track.label);
                continue;
            }
            if !supported(&configuration) {
                println!("# skipping {}/{backend}: unsupported here", track.label);
                continue;
            }
            if backend == "software" && !large {
                println!(
                    "# skipping {}/{backend}: set {LARGE_GROUP_ENV}=1 to measure it",
                    track.label,
                );
                continue;
            }
            group.bench_function(format!("{}/{backend}", track.label), |bencher| {
                bencher.iter_custom(|iterations| {
                    timed_seeks(&configuration, track, target, iterations)
                });
            });
        }
    }
    group.finish();
}

/// The preview arm: what answering "what is at this point" costs instead.
///
/// Deliberately in the same group name family as [`exact_seek`] so the two
/// numbers are read together. The index's background pass is waited out before
/// the clock starts, so this measures the steady-state lookup a drag performs
/// and not the pass that fills it — the pass's own cost is one decode of the
/// track, which the `raps=1` software arm above already prices.
fn preview_lookup(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group(group_name("preview_lookup"));
    let track = &support::gop_cadence_tracks()[0];
    let factory = native_hevc_video_decoder_factory();
    let configuration = configuration(track, HardwarePreference::Prefer);
    // Building the index is one forward decode of the whole track. That is
    // seconds on a fixed-function backend and minutes on the software one, so a
    // host without hardware only measures this when it has opted into the slow
    // arms - the lookup itself is a lock and a clone and does not vary by
    // backend anyway.
    let hardware = cfg!(any(target_os = "macos", windows)) && supported(&configuration);
    if !hardware && std::env::var_os(LARGE_GROUP_ENV).is_none() {
        println!("# skipping preview_lookup: no hardware backend, and {LARGE_GROUP_ENV} is unset");
        group.finish();
        return;
    }
    if !supported(&configuration) {
        println!("# skipping preview_lookup: no usable decoder");
        group.finish();
        return;
    }
    let index = PreviewIndex::new(
        &factory,
        configuration,
        track.samples.clone(),
        Limits::default(),
        PreviewOptions::for_frame_rate(24),
    )
    .expect("the preview index starts");
    index.wait_for_coverage();
    let target = target_frame(track.samples.len());
    group.bench_function("nearest", |bencher| {
        bencher.iter(|| black_box(index.nearest(black_box(target.0))));
    });
    group.finish();
}

criterion_group!(benches, exact_seek, preview_lookup);
criterion_main!(benches);
