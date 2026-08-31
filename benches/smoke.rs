//! Smoke benchmarks for the benchmarking foundation itself.
//!
//! This target is deliberately not a codec measurement suite. It exists to
//! prove two things about the harness the rest of the benchmark series builds
//! on:
//!
//! 1. **Fixtures load.** The bundled 1920x1080 MP4 demuxes and decodes, and the
//!    checked-in AV1 elementary streams parse, from an external `benches/`
//!    crate — with the parsing done once and cached, so what criterion times is
//!    codec work.
//! 2. **The SIMD switch reaches the kernels.** `zvidlib::simd::set_override`
//!    visibly changes timings for HEVC *and* for all three AV1 dispatch
//!    families (transforms/in-loop filters, motion compensation, and intra
//!    prediction), which is exactly the property a scalar-vs-SIMD comparison
//!    depends on and which no single pre-existing toggle had.
//!
//! Every group also runs `support::harness::assert_bit_exact_across_isas`
//! before timing, so a divergent kernel fails the benchmark instead of
//! reporting a speedup.
//!
//! Run with `cargo bench --bench smoke`; see `benches/README.md`.

// Benches are native-only: `wasm32` has no criterion harness and no vector
// kernels to compare against, so the whole target compiles away there.
#![cfg(not(target_arch = "wasm32"))]

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use zvidlib::av1_filters::{FilterFrame, FilterPlane, LoopFilterParams, deblock_frame};
use zvidlib::av1_intra_pred::{add_residual_row, paeth_row, sum_samples};
use zvidlib::av1_mc::{InterpFilter, McContext, RefPlane};
use zvidlib::{
    Av1InterDecoder, CancellationToken, FrameDigest, Limits, VideoDecoderFactory,
    native_hevc_video_decoder_factory,
};

mod support;

use support::harness::{Workload, bench_across_isas};
use support::{fixtures, synth};

/// Frames of the bundled sample one HEVC iteration decodes.
///
/// Two is enough to cover a key frame plus an inter frame (so inter
/// prediction, the in-loop filters, and the inverse transforms all run) while
/// keeping a criterion sample inside a few hundred milliseconds on the
/// software decoder.
const HEVC_FRAMES: u64 = 2;

/// Luma dimensions for the synthetic AV1 kernel workloads. One 1080p plane is
/// large enough that per-call dispatch overhead is negligible next to the
/// vectorized inner loops.
const SYNTH_WIDTH: usize = 1920;
const SYNTH_HEIGHT: usize = 1080;

/// Decodes the first frames of the bundled HEVC sample through zvidlib's own
/// software decoder, returning the frames' digests as the comparison payload.
fn hevc_decode(c: &mut Criterion) {
    let fixture = fixtures::hevc_bundled();
    let factory = native_hevc_video_decoder_factory();
    let limits = Limits::default();

    let workload = Workload {
        measurement_time: Duration::from_secs(10),
        ..Workload::new("hevc_decode", HEVC_FRAMES, fixture.pixels_per_frame())
    };
    bench_across_isas(c, &workload, || {
        let mut decoder = factory
            .create(&fixture.configuration, &limits)
            .expect("the software HEVC decoder must be constructible");
        let cancellation = CancellationToken::new();
        let mut digests = Vec::new();
        for sample in &fixture.samples {
            for decoded in decoder
                .submit(sample, &cancellation)
                .expect("the bundled sample must decode")
            {
                digests.extend_from_slice(
                    FrameDigest::from_frame(&decoded.frame)
                        .expect("a decoded frame must digest")
                        .to_hex()
                        .as_bytes(),
                );
            }
            if digests.len() as u64 >= HEVC_FRAMES * 64 {
                break;
            }
        }
        assert!(
            digests.len() as u64 >= HEVC_FRAMES * 64,
            "the bundled sample must yield at least {HEVC_FRAMES} decoded frames"
        );
        digests
    });
}

/// Decodes the checked-in AV1 key/inter elementary stream. Tiny frames, so
/// this measures fixture loading and decoder plumbing rather than kernel
/// throughput — the AV1 kernel arms below are what the SIMD switch shows up in.
fn av1_decode(c: &mut Criterion) {
    let units = fixtures::av1_inter_temporal_units();
    let workload = Workload {
        measurement_time: Duration::from_secs(3),
        sample_size: 20,
        ..Workload::new("av1_decode", units.len() as u64, 16 * 16)
    };
    bench_across_isas(c, &workload, || {
        let mut decoder =
            Av1InterDecoder::new(Limits::default()).expect("the AV1 inter decoder must construct");
        let mut output = Vec::new();
        for unit in units {
            let frame = decoder
                .decode_temporal_unit(unit)
                .expect("the AV1 fixture must decode");
            for plane in &frame.planes {
                output.extend_from_slice(&plane.data);
            }
        }
        output
    });
}

/// AV1 deblocking over a synthetic 1080p luma plane. This is the arm that
/// exercises `av1_simd`, whose kernels the in-loop filters dispatch to.
fn av1_deblock(c: &mut Criterion) {
    let plane = synth::luma_plane(SYNTH_WIDTH, SYNTH_HEIGHT);
    let params = LoopFilterParams {
        y_vertical_level: 24,
        y_horizontal_level: 24,
        u_level: 0,
        v_level: 0,
        sharpness: 0,
    };
    let workload = Workload {
        measurement_time: Duration::from_secs(5),
        ..Workload::new("av1_deblock", 1, (SYNTH_WIDTH * SYNTH_HEIGHT) as u64)
    };
    bench_across_isas(c, &workload, || {
        let mut y = FilterPlane::new(SYNTH_WIDTH, SYNTH_HEIGHT, &Limits::default())
            .expect("the synthetic plane must fit the default limits");
        y.data.copy_from_slice(&plane);
        let mut frame = FilterFrame::new_monochrome(y);
        deblock_frame(&mut frame, &params, None).expect("deblocking must succeed");
        frame.y.data
    });
}

/// AV1 sub-pel motion compensation over the synthetic plane. This is the arm
/// that exercises `av1_mc`, reached through `McContext::new`, which now honours
/// the crate-wide override.
fn av1_motion_compensation(c: &mut Criterion) {
    let plane = synth::luma_plane(SYNTH_WIDTH, SYNTH_HEIGHT);
    const BLOCK: usize = 16;
    let blocks_x = (SYNTH_WIDTH / BLOCK) - 1;
    let blocks_y = (SYNTH_HEIGHT / BLOCK) - 1;
    let workload = Workload {
        measurement_time: Duration::from_secs(5),
        ..Workload::new(
            "av1_motion_compensation",
            1,
            (blocks_x * blocks_y * BLOCK * BLOCK) as u64,
        )
    };
    bench_across_isas(c, &workload, || {
        let mut context = McContext::new();
        let reference = RefPlane::new(&plane, SYNTH_WIDTH, SYNTH_HEIGHT);
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

/// AV1 intra prediction and residual reconstruction over the synthetic plane.
/// This is the arm that exercises `av1_intra_pred`, whose `OnceLock` detection
/// previously had no override at all.
fn av1_intra_prediction(c: &mut Criterion) {
    let plane = synth::luma_plane(SYNTH_WIDTH, SYNTH_HEIGHT);
    let residuals: Vec<i16> = (0..SYNTH_WIDTH)
        .map(|x| ((x % 61) as i16) - 30)
        .collect::<Vec<_>>();
    let workload = Workload {
        measurement_time: Duration::from_secs(5),
        ..Workload::new("av1_intra_pred", 1, (SYNTH_WIDTH * SYNTH_HEIGHT) as u64)
    };
    bench_across_isas(c, &workload, || {
        let mut out = vec![0u8; SYNTH_WIDTH * SYNTH_HEIGHT];
        let mut dc = 0u64;
        for row in 0..SYNTH_HEIGHT {
            let top = &plane[(row % (SYNTH_HEIGHT - 1)) * SYNTH_WIDTH..][..SYNTH_WIDTH];
            let target = &mut out[row * SYNTH_WIDTH..][..SYNTH_WIDTH];
            paeth_row(top[0], top, plane[row * SYNTH_WIDTH], target);
            add_residual_row(&residuals, target);
            dc += u64::from(sum_samples(target));
        }
        out.extend_from_slice(&dc.to_le_bytes());
        out
    });
}

criterion_group!(
    benches,
    hevc_decode,
    av1_decode,
    av1_deblock,
    av1_motion_compensation,
    av1_intra_prediction
);
criterion_main!(benches);
