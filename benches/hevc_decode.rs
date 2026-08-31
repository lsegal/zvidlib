//! Scalar-versus-SIMD benchmarks for zvidlib's pure-Rust HEVC software decoder.
//!
//! The companion to `benches/av1_decode.rs`, for the other software decoder.
//! It answers two questions separately: how fast a whole 1080p HEVC frame
//! decodes end to end, and where that time goes. Every group runs once per
//! instruction set `zvidlib::simd::available()` reports, through the crate-wide
//! override in [`zvidlib::simd`], and `benches/support/isa.rs` asserts both that
//! each arm is bit-exact with scalar before timing it and that the override
//! really landed in every dispatch family — so a reported speedup cannot come
//! from a kernel that quietly diverged or a switch that never took effect.
//!
//! # Groups
//!
//! | Group | Stage | Vectorized |
//! | --- | --- | --- |
//! | `hevc_decode_1080p` | whole-frame decode through `ExactFrameReader` | n/a |
//! | `hevc_decode` | whole-frame decode, one arm per instruction set | n/a |
//! | `hevc_inter_pred` | §8.5.3.3 8-tap luma interpolation + weighted combine | yes |
//! | `hevc_intra_pred` | §8.4.4.2 reference smoothing, planar / DC / angular | yes |
//! | `hevc_deblock` | §8.7.2 luma block-edge deblocking | yes |
//! | `hevc_sao` | §8.7.3 sample adaptive offset, band and edge | yes |
//! | `hevc_inverse_transform` | §8.6 dequantization + inverse DCT/DST | yes |
//! | `hevc_cabac` | §9.3.4 arithmetic bin decoding | no, by design |
//!
//! The two whole-frame groups need the bundled sample and so sit behind
//! `ZVIDLIB_BENCH_LARGE=1`; the per-stage groups do not and always run.
//!
//! A whole-frame group makes a regression *observable*; the per-stage groups
//! make it *attributable*. `hevc_cabac` is here precisely because it has no
//! vector path and is not expected to grow one — the arithmetic decoder is
//! inherently serial, so whatever fraction of a decode it owns is the ceiling on
//! what vectorizing everything else can buy, and that is only readable next to
//! the other stages in the same report.
//!
//! The per-stage inputs come from `zvidlib::hevc_decoder_bench`, a narrow public
//! surface over the otherwise crate-private HEVC engine, which is what lets an
//! external benchmark crate reach the individual stages at all.
//!
//! See `benches/README.md` for how to run and filter the suite.

mod support;

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use zvidlib::hevc_decoder_bench::HevcStageInputs;
use zvidlib::{
    CancellationToken, ExactFrameReader, FrameDigest, FrameIndex, Limits, VideoDecoderFactory,
    native_hevc_video_decoder_factory,
};

use support::isa::{IsaWorkload, bench_across_isas};
use support::{FrameWork, group_name, report_throughput};

/// Environment variable that opts into the long-running 1080p groups.
///
/// The bundled sample is 768 frames of 1920x1080 through a pure-Rust decoder, so
/// a default `cargo bench` would take minutes. Keeping the whole-frame groups
/// opt-in leaves the per-stage ones fast enough for an ordinary edit loop.
const LARGE_GROUP_ENV: &str = "ZVIDLIB_BENCH_LARGE";

/// The number of 1080p frames the long-running group decodes per iteration.
const LARGE_GROUP_FRAMES: u64 = 4;

/// Luma dimensions the per-stage groups run over.
const ISA_WIDTH: usize = 1920;
const ISA_HEIGHT: usize = 1080;

/// Frames of the bundled sample one whole-frame scalar-vs-SIMD iteration
/// decodes.
///
/// Enough to cover a key frame plus a run of inter frames, so inter prediction,
/// the in-loop filters, and the inverse transforms all run.
const ISA_HEVC_FRAMES: u64 = 8;

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
    hevc_decode_1080p,
    hevc_decode_by_isa,
    hevc_inter_pred_by_isa,
    hevc_intra_pred_by_isa,
    hevc_deblock_by_isa,
    hevc_sao_by_isa,
    hevc_inverse_transform_by_isa,
    hevc_cabac_by_isa
);
criterion_main!(benches);
