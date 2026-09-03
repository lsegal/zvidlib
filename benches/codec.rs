//! zvidlib's criterion benchmark suite.
//!
//! This target is the harness the per-codec benchmark tickets extend: it wires
//! up criterion, the shared fixtures in [`support`], the `simd` feature tag that
//! every group name carries, and the scalar-vs-SIMD groups built on
//! `zvidlib::simd`'s process-wide instruction-set override. See
//! `benches/README.md` for how to run and filter it.
//!
//! The per-codec targets own their own measurements: `benches/hevc_decode.rs`,
//! `benches/av1_decode.rs`, `benches/hevc_encode.rs`, `benches/audio_decode.rs`
//! and `benches/audio_mux.rs`.

mod support;

use std::time::Instant;

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use zvidlib::av1_filters::{FilterFrame, FilterPlane, LoopFilterParams, deblock_frame};
use zvidlib::av1_mc::{InterpFilter, McContext, RefPlane};
use zvidlib::{Av1InterDecoder, Limits, TxSizeGrid, decode_av1_lossless_intra};

use support::isa::{IsaWorkload, bench_across_isas, log_host_isas};
use support::{FrameWork, group_name, report_throughput};

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

/// Luma dimensions for the synthetic scalar-vs-SIMD workloads. One 1080p plane
/// is large enough that per-call dispatch overhead is negligible next to the
/// vectorized inner loops.
const ISA_WIDTH: usize = 1920;
const ISA_HEIGHT: usize = 1080;

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
///
/// The group is `av1_deblock_luma` rather than `av1_deblock`, which is the name
/// it carried until issue #414. `benches/av1_decode.rs` has a group of its own
/// called `av1_deblock` - the narrow-filter member of its deblocking trio - and
/// criterion keys a group by its name alone, so two targets claiming one name
/// write the same `target/criterion/av1_deblock/<isa>` directory and the one
/// that runs second is the only one a collected baseline can still see. The two
/// measure different things: this one is a synthetic 1080p plane at level 24,
/// the other a structured plane at level 32, and their `scalar` arms are 27%
/// apart because the scalar path branches per position on the filter mask
/// while the vector kernels do fixed masked work per lane and land within 0.1%
/// of each other. That is the whole of what #414 reports as a 25% regression:
/// the two committed tables collected opposite sides of the collision, because
/// the draws ran the targets in opposite orders. The name pairs with
/// [`av1_deblock_chroma_by_isa`] below, which is this group's chroma half.
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
        "av1_deblock_luma",
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

/// The synthetic 4:2:0 chroma planes, decoded once and reused.
fn isa_chroma_planes() -> &'static (Vec<u8>, Vec<u8>) {
    static PLANES: std::sync::OnceLock<(Vec<u8>, Vec<u8>)> = std::sync::OnceLock::new();
    PLANES.get_or_init(|| {
        let mut planes = support::synthetic_yuv420_sequence(ISA_WIDTH as u32, ISA_HEIGHT as u32, 1)
            .remove(0)
            .planes;
        let v = planes.remove(2).data;
        let u = planes.remove(1).data;
        (u, v)
    })
}

/// AV1 chroma deblocking with transform-size metadata present, which is what
/// makes §7.14.5 select the 6-tap chroma filter.
///
/// The luma levels are zero and only the chroma planes are filtered, so this
/// isolates the chroma edge kernels from [`av1_deblock_by_isa`]'s luma work.
/// The 16x16 luma grid subsamples to 8x8 chroma transforms, so every interior
/// chroma edge takes the 6-tap path rather than the narrow one.
fn av1_deblock_chroma_by_isa(criterion: &mut Criterion) {
    const CHROMA_WIDTH: usize = ISA_WIDTH / 2;
    const CHROMA_HEIGHT: usize = ISA_HEIGHT / 2;

    let luma = isa_luma_plane();
    let (u_data, v_data) = isa_chroma_planes();
    let params = LoopFilterParams {
        y_vertical_level: 0,
        y_horizontal_level: 0,
        u_level: 24,
        v_level: 24,
        sharpness: 0,
    };
    let mut grid = TxSizeGrid::new(ISA_WIDTH, ISA_HEIGHT);
    for y in (0..ISA_HEIGHT).step_by(16) {
        for x in (0..ISA_WIDTH).step_by(16) {
            grid.set_block(x, y, 16, 16);
        }
    }
    let workload = IsaWorkload::new(
        "av1_deblock_chroma",
        FrameWork::new(1, CHROMA_WIDTH as u64, 2 * CHROMA_HEIGHT as u64),
    );
    bench_across_isas(criterion, &workload, || {
        let limits = Limits::default();
        let mut y = FilterPlane::new(ISA_WIDTH, ISA_HEIGHT, &limits)
            .expect("the synthetic plane fits the default limits");
        y.data.copy_from_slice(luma);
        let mut u = FilterPlane::new(CHROMA_WIDTH, CHROMA_HEIGHT, &limits)
            .expect("the synthetic chroma plane fits the default limits");
        u.data.copy_from_slice(u_data);
        let mut v = FilterPlane::new(CHROMA_WIDTH, CHROMA_HEIGHT, &limits)
            .expect("the synthetic chroma plane fits the default limits");
        v.data.copy_from_slice(v_data);
        let mut frame =
            FilterFrame::new_yuv(y, u, v, true, true).expect("the synthetic frame is 4:2:0");
        deblock_frame(&mut frame, &params, Some(&grid)).expect("deblocking succeeds");
        let mut out = frame.u.expect("the frame has a U plane").data;
        out.extend_from_slice(&frame.v.expect("the frame has a V plane").data);
        out
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

criterion_group!(
    benches,
    log_host_isas,
    smoke,
    av1_decode,
    encoder_input,
    av1_deblock_by_isa,
    av1_deblock_chroma_by_isa,
    av1_motion_compensation_by_isa
);
criterion_main!(benches);
