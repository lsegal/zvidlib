//! zvidlib's criterion benchmark suite.
//!
//! This target is the harness the per-codec benchmark tickets extend: it wires
//! up criterion, the shared fixtures in [`support`], and the `simd` feature tag
//! that every group name carries. See `benches/README.md` for how to run and
//! filter it.

mod support;

use std::time::Instant;

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use zvidlib::{
    Av1InterDecoder, CancellationToken, ExactFrameReader, FrameIndex, Limits,
    decode_av1_lossless_intra, native_hevc_video_decoder_factory,
};

use support::{FrameWork, group_name, report_throughput};

/// Environment variable that opts into the long-running 1080p group.
///
/// The bundled sample is 768 frames of 1920x1080 through a pure-Rust decoder, so
/// a default `cargo bench` would take minutes. Keeping it opt-in leaves the
/// default run fast enough to be part of an ordinary edit loop.
const LARGE_GROUP_ENV: &str = "ZVIDLIB_BENCH_LARGE";

/// The number of 1080p frames the long-running group decodes per iteration.
const LARGE_GROUP_FRAMES: u64 = 4;

/// Smoke benchmark: end-to-end proof that fixture loading, decoding, and
/// throughput reporting all work.
fn smoke(criterion: &mut Criterion) {
    let stream = support::av1_lossless_intra_stream();
    let frame = support::av1_lossless_intra_frame();
    let width = u64::from(frame.dimensions.width);
    let height = u64::from(frame.dimensions.height);
    let work = FrameWork::new(1, width, height);

    // A single timed decode before criterion starts, so a run reports a
    // megapixels/sec figure even when it is filtered down to nothing.
    let started = Instant::now();
    let decoded = decode_av1_lossless_intra(stream, &Limits::default())
        .expect("the checked-in AV1 intra vector decodes");
    let elapsed = started.elapsed();
    assert_eq!(
        decoded.planes, frame.planes,
        "cached fixture matches a fresh decode"
    );
    println!(
        "# zvidlib benches: simd feature {}, AV1 {}x{} intra smoke decode {:.2} Mpx/s",
        if support::simd_enabled() { "on" } else { "off" },
        width,
        height,
        work.megapixels_per_second(elapsed),
    );

    let name = group_name("smoke");
    let mut group = criterion.benchmark_group(&name);
    report_throughput(&mut group, "av1_intra_17x9", work);
    group.bench_function("av1_intra_17x9", |bencher| {
        bencher.iter(|| {
            black_box(decode_av1_lossless_intra(black_box(stream), &Limits::default()).unwrap())
        });
    });
    group.finish();
}

/// AV1 fixtures decoded from their checked-in elementary streams.
fn av1_decode(criterion: &mut Criterion) {
    let stream = support::av1_inter_stream();
    let units = support::av1_inter_temporal_units();
    let frames = units.len() as u64;

    let name = group_name("av1_decode");
    let mut group = criterion.benchmark_group(&name);
    report_throughput(
        &mut group,
        "inter_show_existing_16x16",
        FrameWork::new(frames, 16, 16),
    );
    group.bench_function("inter_show_existing_16x16", |bencher| {
        bencher.iter(|| {
            let mut decoder = Av1InterDecoder::new(Limits::default()).unwrap();
            for unit in units {
                black_box(decoder.decode_temporal_unit(&stream[unit.clone()]).unwrap());
            }
        });
    });
    group.finish();
}

/// Synthetic encoder inputs, built without decoding anything first.
fn encoder_input(criterion: &mut Criterion) {
    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 180;
    const FRAMES: usize = 8;

    let name = group_name("encoder_input");
    let mut group = criterion.benchmark_group(&name);
    report_throughput(
        &mut group,
        "synthetic_yuv420_320x180",
        FrameWork::new(FRAMES as u64, u64::from(WIDTH), u64::from(HEIGHT)),
    );
    group.bench_function("synthetic_yuv420_320x180", |bencher| {
        bencher.iter(|| black_box(support::synthetic_yuv420_sequence(WIDTH, HEIGHT, FRAMES)));
    });
    group.finish();
}

/// The bundled 1080p HEVC sample. Opt-in; see [`LARGE_GROUP_ENV`].
fn hevc_decode_1080p(criterion: &mut Criterion) {
    if std::env::var_os(LARGE_GROUP_ENV).is_none() {
        println!("# skipping the 1080p HEVC group; set {LARGE_GROUP_ENV}=1 to run it",);
        return;
    }

    let sample = support::bundled_hevc_sample();
    let name = group_name("hevc_decode_1080p");
    let mut group = criterion.benchmark_group(&name);
    group.sample_size(10);
    report_throughput(
        &mut group,
        "sequential_from_keyframe",
        FrameWork::new(LARGE_GROUP_FRAMES, sample.width, sample.height),
    );
    group.bench_function("sequential_from_keyframe", |bencher| {
        bencher.iter(|| {
            let mut reader = ExactFrameReader::new(
                &native_hevc_video_decoder_factory(),
                sample.configuration.clone(),
                sample.samples.clone(),
                Limits::default(),
            )
            .unwrap();
            let cancellation = CancellationToken::new();
            for index in 0..LARGE_GROUP_FRAMES {
                black_box(reader.get(FrameIndex(index), &cancellation).unwrap());
            }
        });
    });
    group.finish();
}

criterion_group!(benches, smoke, av1_decode, encoder_input, hevc_decode_1080p);
criterion_main!(benches);
