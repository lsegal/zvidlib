//! Scalar-versus-SIMD benchmarks for zvidlib's pure-Rust VP9 software decoder.
//!
//! The VP9 counterpart of `benches/av1_decode.rs` and `benches/hevc_decode.rs`:
//! how fast whole VP9 frames decode, and how fast each stage the `vp9_decode`
//! dispatch site vectorizes runs on its own. Every group runs once per
//! instruction set `zvidlib_core::simd::available()` reports, through the
//! crate-wide override in [`zvidlib_core::simd`], and `benches/support/isa.rs`
//! asserts that every arm is bit-exact with scalar before timing it and that
//! the override really landed in every dispatch site.
//!
//! # Groups
//!
//! | Group | Stage |
//! | --- | --- |
//! | `vp9_decode_to_picture` | whole-frame decode of the bundled 256x144 libvpx stream, stopping at the YUV picture |
//! | `vp9_inverse_dct_{4x4,8x8,16x16,32x32}` | inverse DCT and add-to-prediction, `src/vp9_simd/transforms.rs` |
//! | `vp9_inverse_adst_{4x4,8x8,16x16}` | inverse ADST and add-to-prediction |
//! | `vp9_inverse_wht_4x4` | the lossless Walsh-Hadamard transform |
//! | `vp9_mc_{regular,smooth,sharp,bilinear}` | 16x16 sub-pixel inter prediction per filter, `src/vp9_simd/convolve.rs` |
//! | `vp9_mc_4x4` | 4x4 inter prediction, the narrowest block |
//! | `vp9_mc_compound` | two predictions averaged, as a compound block is |
//! | `vp9_intra_dc`, `vp9_intra_tm`, `vp9_intra_directional` | intra prediction, 4x4 to 32x32, `src/vp9_simd/intra.rs` |
//! | `vp9_loop_filter_{4,8,16}` | `filter4`, `filter8` and the 16-wide filter on every edge of a plane, `src/vp9_simd/loopfilter.rs` |
//!
//! The per-stage groups run over one 1080p luma plane each, through
//! `zvidlib_vp9_decoder::vp9_simd::bench`, the narrow benchmark-only surface over the
//! otherwise crate-private decoder. The whole-frame group stops at the decoded
//! YUV picture, as `hevc_decode_to_picture` does: the public decoder's RGBA
//! output conversion is the separate `yuv_to_rgba` site, and
//! `benches/vpx_decode.rs`'s `vp9_decode_frame` is the `submit`-to-RGBA round
//! trip that includes it.
//!
//! See `benches/README.md` for how to run and filter the suite.

mod support;

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib_container::{Mp4Demuxer, Mp4DemuxerOptions};
use zvidlib_core::io::MemorySource;
use zvidlib_core::{Codec, Limits};
use zvidlib_vp9_decoder::vp9_simd::bench::{
    InterpFilter, IntraKind, LoopFilterTaps, Stage, TransformKind,
};

use support::isa::{IsaWorkload, bench_across_isas, log_host_isas};
use support::{FrameWork, block_on};

/// Luma dimensions the per-stage groups run over.
const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;

/// Criterion windows for the per-stage groups, which are many and each
/// measured once per instruction set. Every group name is a literal argument
/// here, which is what lets `tests/bench_group_names_are_unique.rs` see it.
fn kernel_workload<'a>(codec: &'a str, work: FrameWork) -> IsaWorkload<'a> {
    IsaWorkload {
        measurement_time: Duration::from_secs(2),
        warm_up_time: Duration::from_millis(300),
        ..IsaWorkload::new(codec, work)
    }
}

/// One 1080p plane of work.
fn frame_work() -> FrameWork {
    FrameWork::new(1, WIDTH as u64, HEIGHT as u64)
}

fn bench_stages(criterion: &mut Criterion, stages: Vec<(IsaWorkload<'_>, Stage)>) {
    for (workload, stage) in stages {
        bench_across_isas(criterion, &workload, || stage.run());
    }
}

/// The bundled 256x144 stream, libvpx's two-pass encode of the sample with
/// hidden alternate reference frames, decoded end to end.
fn vp9_decode_to_picture(criterion: &mut Criterion) {
    let source =
        MemorySource::new(include_bytes!("../tests/fixtures/vp9_bbb_256x144.mp4").to_vec());
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))
        .expect("the VP9 fixture is a readable MP4");
    let track = movie.track(1).expect("the VP9 fixture has track 1");
    assert_eq!(track.codec, Codec::Vp9);
    let samples = block_on(track.to_encoded_video_samples(&source, &Limits::default()))
        .expect("the VP9 fixture's samples are readable");
    let frames = samples.len() as u64;
    let stage = Stage::decode(samples.into_iter().map(|sample| sample.data).collect());
    let workload = IsaWorkload {
        measurement_time: Duration::from_secs(5),
        ..IsaWorkload::new("vp9_decode_to_picture", FrameWork::new(frames, 256, 144))
    };
    bench_across_isas(criterion, &workload, || stage.run());
}

fn vp9_inverse_transforms(criterion: &mut Criterion) {
    let transform = |kind, size| Stage::inverse_transform(kind, size, WIDTH, HEIGHT);
    bench_stages(
        criterion,
        vec![
            (
                kernel_workload("vp9_inverse_dct_4x4", frame_work()),
                transform(TransformKind::Dct, 4),
            ),
            (
                kernel_workload("vp9_inverse_dct_8x8", frame_work()),
                transform(TransformKind::Dct, 8),
            ),
            (
                kernel_workload("vp9_inverse_dct_16x16", frame_work()),
                transform(TransformKind::Dct, 16),
            ),
            (
                kernel_workload("vp9_inverse_dct_32x32", frame_work()),
                transform(TransformKind::Dct, 32),
            ),
            (
                kernel_workload("vp9_inverse_adst_4x4", frame_work()),
                transform(TransformKind::Adst, 4),
            ),
            (
                kernel_workload("vp9_inverse_adst_8x8", frame_work()),
                transform(TransformKind::Adst, 8),
            ),
            (
                kernel_workload("vp9_inverse_adst_16x16", frame_work()),
                transform(TransformKind::Adst, 16),
            ),
            (
                kernel_workload("vp9_inverse_wht_4x4", frame_work()),
                transform(TransformKind::Wht, 4),
            ),
        ],
    );
}

fn vp9_inter_prediction(criterion: &mut Criterion) {
    let inter =
        |filter, compound, block| Stage::inter_prediction(filter, compound, block, WIDTH, HEIGHT);
    bench_stages(
        criterion,
        vec![
            (
                kernel_workload("vp9_mc_regular", frame_work()),
                inter(InterpFilter::Regular, false, 16),
            ),
            (
                kernel_workload("vp9_mc_smooth", frame_work()),
                inter(InterpFilter::Smooth, false, 16),
            ),
            (
                kernel_workload("vp9_mc_sharp", frame_work()),
                inter(InterpFilter::Sharp, false, 16),
            ),
            (
                kernel_workload("vp9_mc_bilinear", frame_work()),
                inter(InterpFilter::Bilinear, false, 16),
            ),
            (
                kernel_workload("vp9_mc_4x4", frame_work()),
                inter(InterpFilter::Regular, false, 4),
            ),
            (
                kernel_workload("vp9_mc_compound", frame_work()),
                inter(InterpFilter::Regular, true, 16),
            ),
        ],
    );
}

fn vp9_intra_prediction(criterion: &mut Criterion) {
    let intra = |kind| Stage::intra_prediction(kind, WIDTH, HEIGHT);
    bench_stages(
        criterion,
        vec![
            (
                kernel_workload("vp9_intra_dc", frame_work()),
                intra(IntraKind::Dc),
            ),
            (
                kernel_workload("vp9_intra_tm", frame_work()),
                intra(IntraKind::Tm),
            ),
            (
                kernel_workload("vp9_intra_directional", frame_work()),
                intra(IntraKind::Directional),
            ),
        ],
    );
}

fn vp9_loop_filter(criterion: &mut Criterion) {
    let filter = |taps| Stage::loop_filter(taps, WIDTH, HEIGHT);
    bench_stages(
        criterion,
        vec![
            (
                kernel_workload("vp9_loop_filter_4", frame_work()),
                filter(LoopFilterTaps::Four),
            ),
            (
                kernel_workload("vp9_loop_filter_8", frame_work()),
                filter(LoopFilterTaps::Eight),
            ),
            (
                kernel_workload("vp9_loop_filter_16", frame_work()),
                filter(LoopFilterTaps::Sixteen),
            ),
        ],
    );
}

criterion_group!(
    benches,
    log_host_isas,
    vp9_decode_to_picture,
    vp9_inverse_transforms,
    vp9_inter_prediction,
    vp9_intra_prediction,
    vp9_loop_filter
);
criterion_main!(benches);
