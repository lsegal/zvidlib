//! zvidlib's criterion benchmark suite.
//!
//! This target is the harness the per-codec benchmark tickets extend: it wires
//! up criterion, the shared fixtures in [`support`], the `simd` feature tag that
//! every group name carries, and the scalar-vs-SIMD groups built on
//! `zvidlib::simd`'s process-wide instruction-set override. See
//! `benches/README.md` for how to run and filter it.

mod support;

use std::time::Instant;

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use zvidlib::av1_filters::{FilterFrame, FilterPlane, LoopFilterParams, deblock_frame};
use zvidlib::av1_mc::{InterpFilter, McContext, RefPlane};
use zvidlib::hevc_stage_bench::HevcStageInputs;
use zvidlib::{
    Av1InterDecoder, CancellationToken, ExactFrameReader, FrameDigest, FrameIndex, Limits,
    VideoDecoderFactory, decode_av1_lossless_intra, native_hevc_video_decoder_factory,
};

use support::isa::{IsaWorkload, bench_across_isas};
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

/// Luma dimensions for the synthetic scalar-vs-SIMD workloads. One 1080p plane
/// is large enough that per-call dispatch overhead is negligible next to the
/// vectorized inner loops.
const ISA_WIDTH: usize = 1920;
const ISA_HEIGHT: usize = 1080;

/// Frames of the bundled sample one HEVC scalar-vs-SIMD iteration decodes.
///
/// Enough to cover a key frame plus a run of inter frames, so inter prediction,
/// the in-loop filters, and the inverse transforms all run, while keeping one
/// criterion sample under a second on the software decoder.
const ISA_HEVC_FRAMES: u64 = 8;

/// A deterministic synthetic luma plane for the kernel-level groups.
///
/// [`support::synthetic_yuv420_sequence`] builds whole validated
/// [`zvidlib::VideoFrame`]s for encoder inputs; the in-loop filter and motion
/// compensation kernels want one bare plane, so this borrows its first frame's
/// luma.
fn isa_luma_plane() -> &'static [u8] {
    static PLANE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    PLANE.get_or_init(|| {
        support::synthetic_yuv420_sequence(ISA_WIDTH as u32, ISA_HEIGHT as u32, 1)
            .remove(0)
            .planes
            .remove(0)
            .data
    })
}

/// AV1 deblocking over a synthetic 1080p luma plane: the arm that exercises
/// `av1_simd`, which the in-loop filters dispatch to.
fn av1_deblock_by_isa(criterion: &mut Criterion) {
    let plane = isa_luma_plane();
    let params = LoopFilterParams {
        y_vertical_level: 24,
        y_horizontal_level: 24,
        u_level: 0,
        v_level: 0,
        sharpness: 0,
    };
    let workload = IsaWorkload::new(
        "av1_deblock",
        FrameWork::new(1, ISA_WIDTH as u64, ISA_HEIGHT as u64),
    );
    bench_across_isas(criterion, &workload, || {
        let mut y = FilterPlane::new(ISA_WIDTH, ISA_HEIGHT, &Limits::default())
            .expect("the synthetic plane fits the default limits");
        y.data.copy_from_slice(plane);
        let mut frame = FilterFrame::new_monochrome(y);
        deblock_frame(&mut frame, &params, None).expect("deblocking succeeds");
        frame.y.data
    });
}

/// AV1 sub-pel motion compensation: the arm that exercises `av1_mc`, reached
/// through `McContext::new`, which honours the crate-wide override.
fn av1_motion_compensation_by_isa(criterion: &mut Criterion) {
    const BLOCK: usize = 16;
    let plane = isa_luma_plane();
    let blocks_x = (ISA_WIDTH / BLOCK) - 1;
    let blocks_y = (ISA_HEIGHT / BLOCK) - 1;
    let workload = IsaWorkload::new(
        "av1_motion_compensation",
        FrameWork::new(1, (blocks_x * BLOCK) as u64, (blocks_y * BLOCK) as u64),
    );
    bench_across_isas(criterion, &workload, || {
        let mut context = McContext::new();
        let reference = RefPlane::new(plane, ISA_WIDTH, ISA_HEIGHT);
        let mut dst = vec![0u8; blocks_x * blocks_y * BLOCK * BLOCK];
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let offset = (by * blocks_x + bx) * BLOCK * BLOCK;
                context.predict_single(
                    reference,
                    (bx * BLOCK) as i32,
                    (by * BLOCK) as i32,
                    BLOCK,
                    BLOCK,
                    (bx % 16).max(1),
                    (by % 16).max(1),
                    InterpFilter::Regular,
                    &mut dst[offset..offset + BLOCK * BLOCK],
                    BLOCK,
                );
            }
        }
        dst
    });
}

/// The bundled 1080p HEVC sample decoded through zvidlib's own software
/// decoder, once per instruction set.
///
/// This is the group the issue's "the switch actually reaches the HEVC kernels"
/// requirement rides on. Whether it shows a *timing* difference is host- and
/// kernel-dependent — it comes out near parity on Apple Silicon, where LLVM
/// auto-vectorizes the scalar reference well under `lto = "fat"` — so
/// `bench_across_isas` asserts the override landed rather than inferring it
/// from the clock. Opt-in behind the same environment variable as the other
/// 1080p group.
fn hevc_decode_by_isa(criterion: &mut Criterion) {
    if std::env::var_os(LARGE_GROUP_ENV).is_none() {
        println!("# skipping the per-ISA 1080p HEVC group; set {LARGE_GROUP_ENV}=1 to run it");
        return;
    }
    let sample = support::bundled_hevc_sample();
    let factory = native_hevc_video_decoder_factory();
    let workload = IsaWorkload {
        measurement_time: std::time::Duration::from_secs(10),
        ..IsaWorkload::new(
            "hevc_decode",
            FrameWork::new(ISA_HEVC_FRAMES, sample.width, sample.height),
        )
    };
    bench_across_isas(criterion, &workload, || {
        let mut decoder = factory
            .create(&sample.configuration, &Limits::default())
            .expect("the software HEVC decoder is constructible");
        let cancellation = CancellationToken::new();
        let mut digests = Vec::new();
        for encoded in &sample.samples {
            for decoded in decoder
                .submit(encoded, &cancellation)
                .expect("the bundled sample decodes")
            {
                digests.extend_from_slice(
                    FrameDigest::from_frame(&decoded.frame)
                        .expect("a decoded frame digests")
                        .to_hex()
                        .as_bytes(),
                );
            }
            if digests.len() as u64 >= ISA_HEVC_FRAMES * 64 {
                break;
            }
        }
        assert!(
            digests.len() as u64 >= ISA_HEVC_FRAMES * 64,
            "the bundled sample yields at least {ISA_HEVC_FRAMES} decoded frames"
        );
        digests
    });
}


/// The prepared per-stage HEVC inputs, built once per process.
///
/// Construction allocates a 1080p luma plane, a full 4:2:0 picture, several
/// thousand intra reference arrays and coefficient blocks, and the CABAC
/// buffer. That is setup, not measurement, so it is cached here rather than
/// repeated per group — and it is deliberately outside every timed loop.
fn hevc_stage_inputs() -> &'static HevcStageInputs {
    static INPUTS: std::sync::OnceLock<HevcStageInputs> = std::sync::OnceLock::new();
    INPUTS.get_or_init(|| HevcStageInputs::new(ISA_WIDTH, ISA_HEIGHT))
}

/// A per-stage HEVC group: one arm per available instruction set, sized by the
/// samples that stage actually touches.
///
/// The stages are timed separately from the whole-frame group so a regression
/// can be attributed to a kernel rather than only observed at the frame level,
/// and so the ratio between a stage's vector speedup and the whole-frame
/// speedup is readable directly off one report.
fn hevc_stage_group<F>(criterion: &mut Criterion, name: &str, samples: u64, run: F)
where
    F: Fn(&'static HevcStageInputs) -> Vec<u8>,
{
    let inputs = hevc_stage_inputs();
    let workload = IsaWorkload {
        sample_size: 20,
        measurement_time: std::time::Duration::from_secs(5),
        ..IsaWorkload::new(name, FrameWork::new(1, samples, 1))
    };
    bench_across_isas(criterion, &workload, || run(inputs));
}

/// §8.5.3.3 inter prediction: the 8-tap luma interpolation and the weighted
/// combine, the two `engine::simd` primitives.
fn hevc_inter_pred_by_isa(criterion: &mut Criterion) {
    let samples = hevc_stage_inputs().inter_pred_samples();
    hevc_stage_group(criterion, "hevc_inter_pred", samples, |inputs| {
        inputs.run_inter_pred()
    });
}

/// §8.4.4.2 intra prediction: reference smoothing plus the planar, DC and
/// angular predictors, over all 35 modes at 4x4 through 32x32.
fn hevc_intra_pred_by_isa(criterion: &mut Criterion) {
    let samples = hevc_stage_inputs().intra_pred_samples();
    hevc_stage_group(criterion, "hevc_intra_pred", samples, |inputs| {
        inputs.run_intra_pred()
    });
}

/// §8.7.2 deblocking: every vertical luma edge segment of a 1080p frame.
///
/// The plane alternates textured and flat bands, because the wide filter is
/// gated on the §8.7.2.5.3 flatness check — a purely textured input would time
/// only the narrow path and silently under-report the kernel.
fn hevc_deblock_by_isa(criterion: &mut Criterion) {
    let samples = hevc_stage_inputs().deblock_samples();
    hevc_stage_group(criterion, "hevc_deblock", samples, |inputs| {
        inputs.run_deblock()
    });
}

/// §8.7.3 SAO over a full 4:2:0 picture, with both the band and the edge
/// classifier (all four edge classes) present in the CTB grid.
fn hevc_sao_by_isa(criterion: &mut Criterion) {
    let samples = hevc_stage_inputs().sao_samples();
    hevc_stage_group(criterion, "hevc_sao", samples, |inputs| inputs.run_sao());
}

/// §8.6 dequantization and the inverse transform at every block size.
fn hevc_inverse_transform_by_isa(criterion: &mut Criterion) {
    let samples = hevc_stage_inputs().inverse_transform_samples();
    hevc_stage_group(criterion, "hevc_inverse_transform", samples, |inputs| {
        inputs.run_inverse_transform()
    });
}

/// §9.3.4 CABAC bin decoding — the serial stage, measured for the ceiling it
/// puts on everything else.
///
/// It has no vector path and is not expected to grow one, so its arms should
/// come out equal. That is the point: whatever fraction of a whole-frame decode
/// this stage owns is the fraction no amount of SIMD elsewhere can remove, and
/// the number is only meaningful next to the other stages in the same report.
/// The "samples" axis here is bins, so its Mpx/s line reads as megabins/sec.
fn hevc_cabac_by_isa(criterion: &mut Criterion) {
    let bins = hevc_stage_inputs().cabac_bins();
    hevc_stage_group(criterion, "hevc_cabac", bins, |inputs| inputs.run_cabac());
}

criterion_group!(
    benches,
    smoke,
    av1_decode,
    encoder_input,
    hevc_decode_1080p,
    av1_deblock_by_isa,
    av1_motion_compensation_by_isa,
    hevc_decode_by_isa,
    hevc_inter_pred_by_isa,
    hevc_intra_pred_by_isa,
    hevc_deblock_by_isa,
    hevc_sao_by_isa,
    hevc_inverse_transform_by_isa,
    hevc_cabac_by_isa
);
criterion_main!(benches);
