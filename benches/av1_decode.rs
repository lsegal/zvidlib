//! Scalar-versus-SIMD benchmarks for zvidlib's pure-Rust AV1 software decoder.
//!
//! This target answers two questions the crate's other benchmarks do not: how
//! fast a whole AV1 frame decodes end to end, and where that time goes. Every
//! group runs once per instruction set `zvidlib::simd::available()` reports,
//! through the crate-wide override in [`zvidlib::simd`], which is what makes a
//! single "scalar" arm meaningful here: AV1's kernels are reached through three
//! independent dispatch sites (`av1_simd` for the transforms and in-loop
//! filters, `av1_mc` for inter prediction, `av1_intra_pred` for intra
//! prediction) and pinning only one of them would leave the others vectorized.
//!
//! `benches/support/isa.rs` additionally asserts that every arm is bit-exact
//! with scalar before timing it and that the override really landed in each
//! dispatch family, so a reported speedup cannot come from a kernel that
//! quietly diverged or from a switch that never took effect.
//!
//! # Groups
//!
//! | Group | Stage |
//! | --- | --- |
//! | `av1_decode_frame` | whole-frame decode through `native_av1_video_decoder_factory` |
//! | `av1_inverse_dct_*`, `av1_inverse_adst_*` | inverse transforms, `src/av1_simd/transforms.rs` |
//! | `av1_deblock*`, `av1_cdef`, `av1_wiener`, `av1_self_guided` | in-loop filters, `src/av1_simd/filters.rs` and `src/av1_filters.rs` |
//! | `av1_mc_*` | inter prediction, `src/av1_mc.rs` |
//! | `av1_intra_*` | intra prediction, `src/av1_intra_pred.rs` |
//! | `av1_entropy_symbol` | arithmetic symbol decode, `src/av1_entropy.rs` |
//!
//! This target replaces the ad-hoc, `#[ignore]`d `tests/av1_simd_bench.rs`
//! (issue #120): its input generators moved to `benches/support/mod.rs` and its
//! hand-rolled timing loops became the criterion groups below, so the same
//! measurements now produce stored baselines and can be tracked for
//! regressions. The bit-exactness tests it sat next to — `tests/av1_simd_intra.rs`
//! and `src/av1_simd/tests.rs` — are correctness checks and are unaffected.
//!
//! See `benches/README.md` for how to run and filter the suite.

mod support;

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::av1_filters::{
    CdefStrength, FilterFrame, LoopFilterParams, RestorationUnit, apply_restoration_unit,
    cdef_frame, deblock_frame,
};
use zvidlib::av1_intra_pred::{SmoothMode, directional_row, paeth_row, smooth_row};
use zvidlib::av1_mc::{
    InterpFilter, McContext, RefPlane, blend_average, blend_mask, build_difference_mask,
    default_level,
};
use zvidlib::{
    AV1_CDF_MAX, Av1SymbolDecoder, Av1TxType, CancellationToken, FrameDigest, Limits,
    VideoDecoderFactory, inverse_transform, native_av1_video_decoder_factory,
};

use support::FrameWork;
use support::isa::{IsaWorkload, bench_across_isas, checksum};

/// Luma dimensions the kernel-level groups run over. One 1080p plane is large
/// enough that per-call dispatch overhead is negligible next to the vectorized
/// inner loops, and it is the size the measurements this target replaces used,
/// so their numbers stay comparable.
const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;

/// Criterion windows for the kernel groups.
///
/// There are a dozen of them and each is measured once per available
/// instruction set, so the default five-second window would put a plain
/// `cargo bench --bench av1_decode` into the tens of minutes. Two seconds over
/// a 1080p plane is still hundreds of iterations of work per sample.
fn kernel_workload<'a>(codec: &'a str, work: FrameWork) -> IsaWorkload<'a> {
    IsaWorkload {
        measurement_time: Duration::from_secs(2),
        warm_up_time: Duration::from_millis(300),
        ..IsaWorkload::new(codec, work)
    }
}

/// A full-frame [`FrameWork`] for the 1920x1080 kernel groups.
fn frame_work() -> FrameWork {
    FrameWork::new(1, WIDTH as u64, HEIGHT as u64)
}

// ---------------------------------------------------------------------------
// Whole-frame decode
// ---------------------------------------------------------------------------

/// The synthetic AV1 stream decoded end to end, once per instruction set.
///
/// Encoding the fixture happens once per process inside
/// [`support::synthetic_av1_stream`], so an iteration is decoder work only:
/// OBU parse, entropy decode, reconstruction, and the RGBA output conversion.
/// The harness reports it both as frames per second and as megapixels per
/// second.
fn av1_decode_frame(criterion: &mut Criterion) {
    let stream = support::synthetic_av1_stream();
    let factory = native_av1_video_decoder_factory();
    let workload = IsaWorkload {
        measurement_time: Duration::from_secs(10),
        ..IsaWorkload::new(
            "av1_decode_frame",
            FrameWork::new(
                support::AV1_STREAM_FRAMES as u64,
                stream.width,
                stream.height,
            ),
        )
    };
    bench_across_isas(criterion, &workload, || {
        let mut decoder = factory
            .create(&stream.configuration, &Limits::default())
            .expect("the native AV1 decoder is constructible");
        let cancellation = CancellationToken::new();
        let mut digests = Vec::new();
        for sample in &stream.samples {
            for decoded in decoder
                .submit(sample, &cancellation)
                .expect("the synthetic stream decodes")
            {
                digests.extend_from_slice(
                    FrameDigest::from_frame(&decoded.frame)
                        .expect("a decoded frame digests")
                        .to_hex()
                        .as_bytes(),
                );
            }
        }
        for decoded in decoder.drain(&cancellation).expect("the decoder drains") {
            digests.extend_from_slice(
                FrameDigest::from_frame(&decoded.frame)
                    .expect("a decoded frame digests")
                    .to_hex()
                    .as_bytes(),
            );
        }
        assert!(
            !digests.is_empty(),
            "the synthetic stream yields decoded frames"
        );
        digests
    });
}

// ---------------------------------------------------------------------------
// Inverse transforms (src/av1_simd/transforms.rs)
// ---------------------------------------------------------------------------

/// Every inverse transform size and family the vector kernels cover, applied
/// over a whole frame's worth of blocks.
///
/// The 16- and 32-point inverse DCTs and the ADST family are the kernels issue
/// #138 vectorized; 4, 8 and 64 are included because the frame-scale block
/// counts differ by three orders of magnitude between them, and a per-block
/// speedup that only holds at one size is worth seeing.
fn av1_inverse_transforms(criterion: &mut Criterion) {
    for (name, size, tx_type) in [
        ("av1_inverse_dct_4x4", 4usize, Av1TxType::DctDct),
        ("av1_inverse_dct_8x8", 8, Av1TxType::DctDct),
        ("av1_inverse_dct_16x16", 16, Av1TxType::DctDct),
        ("av1_inverse_dct_32x32", 32, Av1TxType::DctDct),
        ("av1_inverse_dct_64x64", 64, Av1TxType::DctDct),
        ("av1_inverse_adst_8x8", 8, Av1TxType::AdstAdst),
        ("av1_inverse_flipadst_16x16", 16, Av1TxType::FlipadstAdst),
    ] {
        // Deterministic coefficients that are neither sparse nor saturating, so
        // no size degenerates into an all-zero early out.
        let coefficients: Vec<i32> = (0..size * size)
            .map(|index| (index as i32 * 37) % 121 - 60)
            .collect();
        let blocks = (WIDTH / size) * (HEIGHT / size);
        let workload = kernel_workload(
            name,
            FrameWork::new(1, (WIDTH / size * size) as u64, (HEIGHT / size * size) as u64),
        );
        bench_across_isas(criterion, &workload, || {
            let mut digest = 0u64;
            for _ in 0..blocks {
                let residual = inverse_transform(&coefficients, size, tx_type, 20, 14);
                digest ^= checksum(&residual[0].to_le_bytes());
            }
            digest.to_le_bytes().to_vec()
        });
    }
}

// ---------------------------------------------------------------------------
// In-loop filters (src/av1_simd/filters.rs, src/av1_filters.rs)
// ---------------------------------------------------------------------------

/// Deblocking level shared by the three deblocking groups, high enough that
/// every edge is filtered rather than skipped by the level check.
const DEBLOCK_PARAMS: LoopFilterParams = LoopFilterParams {
    y_vertical_level: 32,
    y_horizontal_level: 32,
    u_level: 0,
    v_level: 0,
    sharpness: 0,
};

/// Deblocking over structured 1080p content: the narrow filters and the
/// data-dependent branches that select them.
fn av1_deblock(criterion: &mut Criterion) {
    let source = support::av1_structured_plane(WIDTH, HEIGHT);
    let workload = kernel_workload("av1_deblock", frame_work());
    bench_across_isas(criterion, &workload, || {
        let mut frame = FilterFrame::new_monochrome(source.clone());
        deblock_frame(&mut frame, &DEBLOCK_PARAMS, None).expect("deblocking succeeds");
        frame.y.data
    });
}

/// Deblocking with a frame-wide 32x32 transform grid over near-flat content:
/// the combination that actually reaches the wide 8-tap and 14-tap filters
/// (issue #137), which the flatness check otherwise skips.
fn av1_deblock_wide(criterion: &mut Criterion) {
    let flat = support::av1_flat_blocks_plane(WIDTH, HEIGHT);
    let grid = support::av1_wide_tx_grid(WIDTH, HEIGHT);
    let workload = kernel_workload("av1_deblock_wide", frame_work());
    bench_across_isas(criterion, &workload, || {
        let mut frame = FilterFrame::new_monochrome(flat.clone());
        deblock_frame(&mut frame, &DEBLOCK_PARAMS, Some(&grid)).expect("deblocking succeeds");
        frame.y.data
    });
}

/// Deblocking many tiny planes, where boundary positions dominate.
///
/// A 33x17 plane is almost entirely frame border and partial rows and columns,
/// so this is the arm that shows what the edge handling costs once the vector
/// kernels no longer have full-width runs to work on.
fn av1_deblock_boundary(criterion: &mut Criterion) {
    const SMALL_WIDTH: usize = 33;
    const SMALL_HEIGHT: usize = 17;
    const PLANES: usize = 64;

    let small = support::av1_flat_blocks_plane(SMALL_WIDTH, SMALL_HEIGHT);
    let workload = kernel_workload(
        "av1_deblock_boundary",
        FrameWork::new(
            PLANES as u64,
            SMALL_WIDTH as u64,
            SMALL_HEIGHT as u64,
        ),
    );
    bench_across_isas(criterion, &workload, || {
        let mut out = Vec::with_capacity(PLANES * SMALL_WIDTH * SMALL_HEIGHT);
        for _ in 0..PLANES {
            let mut frame = FilterFrame::new_monochrome(small.clone());
            deblock_frame(&mut frame, &DEBLOCK_PARAMS, None).expect("deblocking succeeds");
            out.extend_from_slice(&frame.y.data);
        }
        out
    });
}

/// CDEF over a structured 1080p frame.
fn av1_cdef(criterion: &mut Criterion) {
    let source = support::av1_structured_plane(WIDTH, HEIGHT);
    let strength = CdefStrength {
        y_primary: 4,
        y_secondary: 2,
        uv_primary: 0,
        uv_secondary: 0,
        damping: 3,
    };
    let workload = kernel_workload("av1_cdef", frame_work());
    bench_across_isas(criterion, &workload, || {
        let mut frame = FilterFrame::new_monochrome(source.clone());
        cdef_frame(&mut frame, &strength, &Limits::default()).expect("cdef succeeds");
        frame.y.data
    });
}

/// Wiener loop restoration over a whole 1080p frame.
fn av1_wiener(criterion: &mut Criterion) {
    let source = support::av1_structured_plane(WIDTH, HEIGHT);
    let unit = RestorationUnit::Wiener {
        horizontal: [3, -7, 15],
        vertical: [-2, 5, 11],
    };
    let workload = kernel_workload("av1_wiener", frame_work());
    bench_across_isas(criterion, &workload, || {
        let mut plane = source.clone();
        apply_restoration_unit(&mut plane, &unit, 0, 0, WIDTH, HEIGHT)
            .expect("wiener restoration succeeds");
        plane.data
    });
}

/// Self-guided loop restoration over one 256x256 restoration unit.
///
/// Self-guided restoration is signaled per restoration unit rather than per
/// frame, so a unit — not a frame — is the honest unit of work here.
fn av1_self_guided(criterion: &mut Criterion) {
    const UNIT: usize = 256;

    let source = support::av1_structured_plane(WIDTH, HEIGHT);
    let unit = RestorationUnit::SelfGuided {
        radius: [2, 3],
        eps: [12, 30],
        weight: [40, 24],
    };
    let workload = kernel_workload(
        "av1_self_guided",
        FrameWork::new(1, UNIT as u64, UNIT as u64),
    );
    bench_across_isas(criterion, &workload, || {
        let mut plane = source.clone();
        apply_restoration_unit(&mut plane, &unit, 0, 0, UNIT, UNIT)
            .expect("self-guided restoration succeeds");
        plane.data
    });
}

// ---------------------------------------------------------------------------
// Inter prediction (src/av1_mc.rs)
// ---------------------------------------------------------------------------

/// Block size the motion-compensation groups predict in.
const MC_BLOCK: usize = 16;

/// Blocks per row and column of the frame-scale motion-compensation sweeps.
///
/// One short of the full count in each direction so every block's 8-tap window
/// stays inside the reference plane without relying on edge extension for the
/// whole measurement.
fn mc_blocks() -> (usize, usize) {
    ((WIDTH / MC_BLOCK) - 1, (HEIGHT / MC_BLOCK) - 1)
}

/// A frame's worth of 8-tap sub-pel single-reference prediction.
fn av1_mc_single(criterion: &mut Criterion) {
    let plane = support::av1_structured_plane(WIDTH, HEIGHT);
    let (blocks_x, blocks_y) = mc_blocks();
    let workload = kernel_workload(
        "av1_mc_single",
        FrameWork::new(1, (blocks_x * MC_BLOCK) as u64, (blocks_y * MC_BLOCK) as u64),
    );
    bench_across_isas(criterion, &workload, || {
        let mut context = McContext::new();
        let reference = RefPlane::new(&plane.data, WIDTH, HEIGHT);
        let mut dst = vec![0u8; blocks_x * blocks_y * MC_BLOCK * MC_BLOCK];
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let offset = (by * blocks_x + bx) * MC_BLOCK * MC_BLOCK;
                context.predict_single(
                    reference,
                    (bx * MC_BLOCK) as i32,
                    (by * MC_BLOCK) as i32,
                    MC_BLOCK,
                    MC_BLOCK,
                    (bx % 16).max(1),
                    (by % 16).max(1),
                    InterpFilter::Regular,
                    &mut dst[offset..offset + MC_BLOCK * MC_BLOCK],
                    MC_BLOCK,
                );
            }
        }
        dst
    });
}

/// A frame's worth of compound prediction: two sub-pel predictions per block
/// plus the average blend, which is the compound path the decoder signals.
fn av1_mc_compound_average(criterion: &mut Criterion) {
    let plane = support::av1_structured_plane(WIDTH, HEIGHT);
    let (blocks_x, blocks_y) = mc_blocks();
    let workload = kernel_workload(
        "av1_mc_compound_average",
        FrameWork::new(1, (blocks_x * MC_BLOCK) as u64, (blocks_y * MC_BLOCK) as u64),
    );
    bench_across_isas(criterion, &workload, || {
        let level = default_level();
        let mut context = McContext::new();
        let reference = RefPlane::new(&plane.data, WIDTH, HEIGHT);
        let mut pred0 = vec![0i16; MC_BLOCK * MC_BLOCK];
        let mut pred1 = vec![0i16; MC_BLOCK * MC_BLOCK];
        let mut dst = vec![0u8; blocks_x * blocks_y * MC_BLOCK * MC_BLOCK];
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let offset = (by * blocks_x + bx) * MC_BLOCK * MC_BLOCK;
                context.predict_compound(
                    reference,
                    (bx * MC_BLOCK) as i32,
                    (by * MC_BLOCK) as i32,
                    MC_BLOCK,
                    MC_BLOCK,
                    (bx % 16).max(1),
                    (by % 16).max(1),
                    InterpFilter::Regular,
                    &mut pred0,
                    MC_BLOCK,
                );
                context.predict_compound(
                    reference,
                    (bx * MC_BLOCK) as i32 + 1,
                    (by * MC_BLOCK) as i32 + 1,
                    MC_BLOCK,
                    MC_BLOCK,
                    (by % 16).max(1),
                    (bx % 16).max(1),
                    InterpFilter::Smooth,
                    &mut pred1,
                    MC_BLOCK,
                );
                blend_average(
                    level,
                    &pred0,
                    MC_BLOCK,
                    &pred1,
                    MC_BLOCK,
                    MC_BLOCK,
                    MC_BLOCK,
                    &mut dst[offset..offset + MC_BLOCK * MC_BLOCK],
                    MC_BLOCK,
                );
            }
        }
        dst
    });
}

/// The masked compound blend, over a difference mask built from the same two
/// predictions. Separated from the average blend because it is a different
/// kernel with a per-sample mask read.
fn av1_mc_blend_mask(criterion: &mut Criterion) {
    let plane = support::av1_structured_plane(WIDTH, HEIGHT);
    let (blocks_x, blocks_y) = mc_blocks();
    let workload = kernel_workload(
        "av1_mc_blend_mask",
        FrameWork::new(1, (blocks_x * MC_BLOCK) as u64, (blocks_y * MC_BLOCK) as u64),
    );
    bench_across_isas(criterion, &workload, || {
        let level = default_level();
        let mut context = McContext::new();
        let reference = RefPlane::new(&plane.data, WIDTH, HEIGHT);
        let mut pred0 = vec![0i16; MC_BLOCK * MC_BLOCK];
        let mut pred1 = vec![0i16; MC_BLOCK * MC_BLOCK];
        let mut mask = vec![0u8; MC_BLOCK * MC_BLOCK];
        let mut dst = vec![0u8; blocks_x * blocks_y * MC_BLOCK * MC_BLOCK];
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let offset = (by * blocks_x + bx) * MC_BLOCK * MC_BLOCK;
                context.predict_compound(
                    reference,
                    (bx * MC_BLOCK) as i32,
                    (by * MC_BLOCK) as i32,
                    MC_BLOCK,
                    MC_BLOCK,
                    (bx % 16).max(1),
                    (by % 16).max(1),
                    InterpFilter::Regular,
                    &mut pred0,
                    MC_BLOCK,
                );
                context.predict_compound(
                    reference,
                    (bx * MC_BLOCK) as i32 + 1,
                    (by * MC_BLOCK) as i32 + 1,
                    MC_BLOCK,
                    MC_BLOCK,
                    (by % 16).max(1),
                    (bx % 16).max(1),
                    InterpFilter::Smooth,
                    &mut pred1,
                    MC_BLOCK,
                );
                build_difference_mask(
                    &pred0,
                    MC_BLOCK,
                    &pred1,
                    MC_BLOCK,
                    MC_BLOCK,
                    MC_BLOCK,
                    false,
                    &mut mask,
                    MC_BLOCK,
                );
                blend_mask(
                    level,
                    &pred0,
                    MC_BLOCK,
                    &pred1,
                    MC_BLOCK,
                    &mask,
                    MC_BLOCK,
                    MC_BLOCK,
                    MC_BLOCK,
                    &mut dst[offset..offset + MC_BLOCK * MC_BLOCK],
                    MC_BLOCK,
                );
            }
        }
        dst
    });
}

// ---------------------------------------------------------------------------
// Intra prediction (src/av1_intra_pred.rs)
// ---------------------------------------------------------------------------

/// Intra block size the prediction groups predict in.
const INTRA_BLOCK: usize = 32;

/// Neighbour samples for one intra block, deterministic and non-degenerate.
fn intra_neighbours() -> (Vec<u8>, Vec<u8>) {
    let top = (0..INTRA_BLOCK)
        .map(|index| (index * 7 + 13) as u8)
        .collect();
    let left = (0..INTRA_BLOCK)
        .map(|index| (index * 11 + 29) as u8)
        .collect();
    (top, left)
}

/// Blocks one intra sweep predicts, chosen so the sweep covers a 1080p frame.
fn intra_blocks() -> usize {
    (WIDTH / INTRA_BLOCK) * (HEIGHT / INTRA_BLOCK)
}

/// Paeth, smooth, and directional intra prediction over a frame's worth of
/// 32x32 blocks (issue #133).
///
/// `directional_row` has no vector path today, so its arms are expected to read
/// alike; it is measured anyway because it is the third of the three intra
/// predictors the decoder reaches, and a flat profile is itself the finding.
fn av1_intra_prediction(criterion: &mut Criterion) {
    let (top, left) = intra_neighbours();
    let blocks = intra_blocks();
    let work = FrameWork::new(
        1,
        (WIDTH / INTRA_BLOCK * INTRA_BLOCK) as u64,
        (HEIGHT / INTRA_BLOCK * INTRA_BLOCK) as u64,
    );

    let workload = kernel_workload("av1_intra_paeth", work);
    bench_across_isas(criterion, &workload, || {
        let mut out = vec![0u8; INTRA_BLOCK * INTRA_BLOCK];
        let mut digest = 0u64;
        for _ in 0..blocks {
            for row in 0..INTRA_BLOCK {
                let (predicted, _) = out.split_at_mut(INTRA_BLOCK);
                paeth_row(top[0], &top, left[row], predicted);
            }
            digest ^= checksum(&out);
        }
        digest.to_le_bytes().to_vec()
    });

    let workload = kernel_workload("av1_intra_smooth", work);
    bench_across_isas(criterion, &workload, || {
        let mut out = vec![0u8; INTRA_BLOCK * INTRA_BLOCK];
        let mut digest = 0u64;
        for _ in 0..blocks {
            for row in 0..INTRA_BLOCK {
                let (predicted, _) = out.split_at_mut(INTRA_BLOCK);
                smooth_row(SmoothMode::Smooth, &top, &left, row, predicted);
            }
            digest ^= checksum(&out);
        }
        digest.to_le_bytes().to_vec()
    });

    let workload = kernel_workload("av1_intra_directional", work);
    bench_across_isas(criterion, &workload, || {
        let mut out = vec![0u8; INTRA_BLOCK * INTRA_BLOCK];
        let mut digest = 0u64;
        for _ in 0..blocks {
            for row in 0..INTRA_BLOCK {
                let (predicted, _) = out.split_at_mut(INTRA_BLOCK);
                directional_row(67, &top, &left, row, true, predicted);
            }
            digest ^= checksum(&out);
        }
        digest.to_le_bytes().to_vec()
    });
}

// ---------------------------------------------------------------------------
// Entropy decode (src/av1_entropy.rs)
// ---------------------------------------------------------------------------

/// Symbols one entropy iteration decodes. Large enough that decoder
/// construction is noise, small enough to keep a sample sub-millisecond.
const ENTROPY_SYMBOLS: usize = 200_000;

/// AV1 arithmetic symbol decoding, the serial stage that bounds everything
/// else.
///
/// This is deliberately a scalar, inherently serial measurement: it establishes
/// the Amdahl ceiling on the whole-frame SIMD win, since no amount of
/// vectorizing the reconstruction kernels can speed up the bit-by-bit range
/// decoder that feeds them. Its arms are expected to read alike — the number to
/// take away is its share of `av1_decode_frame`, not a speedup.
///
/// There is no CDF-adaptation arm to measure alongside it: both AV1 decoders in
/// this crate require `disable_cdf_update = 1` (see `src/av1_inter_decoder.rs`),
/// so `src/av1_cdf.rs`'s tables are read but never adapted, and reading them is
/// exactly what the CDF argument below costs.
fn av1_entropy_symbol(criterion: &mut Criterion) {
    // Deterministic pseudo-random payload. The range decoder never reads past
    // its input — it pads once `remaining_bits` runs out — so a buffer this
    // size feeds the whole sweep without a restart discontinuity.
    let mut state = 0x853c_49e6_748f_ea9b_u64;
    let payload: Vec<u8> = (0..1 << 16)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        })
        .collect();

    // A boolean CDF, a three-symbol CDF, and a four-symbol coefficient-shaped
    // CDF: the shapes the coefficient and mode syntax elements actually use.
    const BOOL_CDF: [u16; 2] = [16_384, AV1_CDF_MAX];
    const EOB_CDF: [u16; 3] = [10_000, 26_000, AV1_CDF_MAX];
    const COEFF_CDF: [u16; 4] = [4_016, 15_324, 25_112, AV1_CDF_MAX];

    let workload = IsaWorkload {
        sample_size: 20,
        ..kernel_workload("av1_entropy_symbol", FrameWork::new(1, WIDTH as u64, HEIGHT as u64))
    };
    bench_across_isas(criterion, &workload, || {
        let mut decoder =
            Av1SymbolDecoder::new(&payload).expect("the synthetic payload initializes");
        let mut digest = 0u64;
        for index in 0..ENTROPY_SYMBOLS {
            let cdf: &[u16] = match index % 3 {
                0 => &BOOL_CDF,
                1 => &EOB_CDF,
                _ => &COEFF_CDF,
            };
            digest = digest
                .wrapping_mul(31)
                .wrapping_add(decoder.symbol(cdf).expect("a valid CDF always decodes") as u64);
        }
        digest.to_le_bytes().to_vec()
    });
}

criterion_group!(
    benches,
    av1_decode_frame,
    av1_inverse_transforms,
    av1_deblock,
    av1_deblock_wide,
    av1_deblock_boundary,
    av1_cdef,
    av1_wiener,
    av1_self_guided,
    av1_mc_single,
    av1_mc_compound_average,
    av1_mc_blend_mask,
    av1_intra_prediction,
    av1_entropy_symbol
);
criterion_main!(benches);
