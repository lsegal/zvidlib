//! Scalar-versus-SIMD benchmarks for zvidlib's pure-Rust VP9 software decoder.
//!
//! The VP9 counterpart of `benches/av1_decode.rs` and `benches/hevc_decode.rs`:
//! how fast whole VP9 frames decode, and how fast each stage the `vp9_decode`
//! dispatch site vectorizes runs on its own. Every group runs once per
//! instruction set `zvidlib::simd::available()` reports, through the
//! crate-wide override in [`zvidlib::simd`], and `benches/support/isa.rs`
//! asserts that every arm is bit-exact with scalar before timing it and that
//! the override really landed in every dispatch site.
//!
//! # Groups
//!
//! | Group | Stage |
//! | --- | --- |
//! | `vp9_decode_frame` | whole-frame decode of the bundled 256x144 libvpx stream, to YUV |
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
//! `zvidlib::vp9_decoder_bench`, the narrow benchmark-only surface over the
//! otherwise crate-private decoder. The whole-frame group stops at the decoded
//! YUV picture: the public decoder's RGBA output conversion is shared, scalar
//! code tracked separately, and would only dilute the ratio.
//!
//! See `benches/README.md` for how to run and filter the suite.

mod support;

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::io::MemorySource;
use zvidlib::vp9_decoder_bench::{InterpFilter, IntraKind, LoopFilterTaps, Stage, TransformKind};
use zvidlib::{Codec, Limits, Mp4Demuxer, Mp4DemuxerOptions};

use support::isa::{IsaWorkload, bench_across_isas, log_host_isas};
use support::{FrameWork, block_on};

/// Luma dimensions the per-stage groups run over.
const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;

/// Criterion windows for the per-stage groups, which are many and each
/// measured once per instruction set.
fn stage_workload<'a>(codec: &'a str, work: FrameWork) -> IsaWorkload<'a> {
    IsaWorkload {
        measurement_time: Duration::from_secs(2),
        warm_up_time: Duration::from_millis(300),
        ..IsaWorkload::new(codec, work)
    }
}

fn frame_work() -> FrameWork {
    FrameWork::new(1, WIDTH as u64, HEIGHT as u64)
}

fn bench_stage(criterion: &mut Criterion, name: &str, stage: &Stage) {
    bench_across_isas(criterion, &stage_workload(name, frame_work()), || {
        stage.run()
    });
}

/// The bundled 256x144 stream, libvpx's two-pass encode of the sample with
/// hidden alternate reference frames, decoded end to end.
fn vp9_decode_frame(criterion: &mut Criterion) {
    let source =
        MemorySource::new(include_bytes!("../tests/fixtures/codec/vp9_bbb_256x144.mp4").to_vec());
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
        ..IsaWorkload::new("vp9_decode_frame", FrameWork::new(frames, 256, 144))
    };
    bench_across_isas(criterion, &workload, || stage.run());
}

fn vp9_inverse_transforms(criterion: &mut Criterion) {
    for (kind, name, sizes) in [
        (TransformKind::Dct, "dct", &[4usize, 8, 16, 32][..]),
        (TransformKind::Adst, "adst", &[4, 8, 16]),
        (TransformKind::Wht, "wht", &[4]),
    ] {
        for &size in sizes {
            let stage = Stage::inverse_transform(kind, size, WIDTH, HEIGHT);
            bench_stage(
                criterion,
                &format!("vp9_inverse_{name}_{size}x{size}"),
                &stage,
            );
        }
    }
}

fn vp9_inter_prediction(criterion: &mut Criterion) {
    for (filter, name) in [
        (InterpFilter::Regular, "regular"),
        (InterpFilter::Smooth, "smooth"),
        (InterpFilter::Sharp, "sharp"),
        (InterpFilter::Bilinear, "bilinear"),
    ] {
        let stage = Stage::inter_prediction(filter, false, 16, WIDTH, HEIGHT);
        bench_stage(criterion, &format!("vp9_mc_{name}"), &stage);
    }
    let stage = Stage::inter_prediction(InterpFilter::Regular, false, 4, WIDTH, HEIGHT);
    bench_stage(criterion, "vp9_mc_4x4", &stage);
    let stage = Stage::inter_prediction(InterpFilter::Regular, true, 16, WIDTH, HEIGHT);
    bench_stage(criterion, "vp9_mc_compound", &stage);
}

fn vp9_intra_prediction(criterion: &mut Criterion) {
    for (kind, name) in [
        (IntraKind::Dc, "dc"),
        (IntraKind::Tm, "tm"),
        (IntraKind::Directional, "directional"),
    ] {
        let stage = Stage::intra_prediction(kind, WIDTH, HEIGHT);
        bench_stage(criterion, &format!("vp9_intra_{name}"), &stage);
    }
}

fn vp9_loop_filter(criterion: &mut Criterion) {
    for (taps, name) in [
        (LoopFilterTaps::Four, "4"),
        (LoopFilterTaps::Eight, "8"),
        (LoopFilterTaps::Sixteen, "16"),
    ] {
        let stage = Stage::loop_filter(taps, WIDTH, HEIGHT);
        bench_stage(criterion, &format!("vp9_loop_filter_{name}"), &stage);
    }
}

criterion_group!(
    benches,
    log_host_isas,
    vp9_decode_frame,
    vp9_inverse_transforms,
    vp9_inter_prediction,
    vp9_intra_prediction,
    vp9_loop_filter
);
criterion_main!(benches);
