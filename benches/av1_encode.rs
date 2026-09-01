//! Scalar-versus-SIMD benchmarks for zvidlib's AV1 encoder-side kernels.
//!
//! The AV1 encoder's forward transforms are the counterpart to the inverse
//! transforms `benches/av1_decode.rs` measures: the same block sizes, the same
//! `Av1TxType` families, and the same `av1_simd` dispatch site, run in the
//! encoding direction. They are measured here rather than beside the inverse
//! sweep because they are encoder work, and a decoder target that reports
//! encoder numbers is a target whose scope cannot be read off its name.
//!
//! Every group runs once per instruction set `zvidlib::simd::available()`
//! reports, through the crate-wide override in [`zvidlib::simd`], and
//! `benches/support/isa.rs` asserts that each arm is bit-exact with scalar
//! before timing it and that the override really landed in each dispatch
//! family — so a reported speedup cannot come from a kernel that quietly
//! diverged or from a switch that never took effect.
//!
//! # Groups
//!
//! | Group | Stage |
//! | --- | --- |
//! | `av1_forward_dct_{4x4,8x8,16x16,32x32}` | forward DCT, `src/av1_encoder/transform.rs` through `zvidlib::forward_transform` |
//! | `av1_forward_adst_8x8`, `av1_forward_flipadst_16x16` | the forward ADST family, including a flipped type |
//! | `av1_encode_frame_q{0,32,160}` | one whole frame through the public encoder, `src/av1_encoder/tile.rs` |
//!
//! The block counts and the coefficient generator are the ones the groups were
//! introduced with (issue #140, in `tests/av1_simd_bench.rs`, and then in
//! `benches/av1_decode.rs`), so the numbers stay directly comparable with the
//! inverse-transform groups and with everything reported for them before the
//! move.
//!
//! See `benches/README.md` for how to run and filter the suite.

mod support;

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::{
    Av1TxType, Codec, CodecProfile, ColorRange, CpuFrameSource, FrameIndex, FrameSource,
    HardwarePreference, Limits, Orientation, PixelFormat, Plane, VideoDimensions,
    VideoEncoderConfig, VideoEncoderFactory, VideoFrame, forward_transform,
    native_av1_video_encoder_factory,
};

use support::FrameWork;
use support::isa::{IsaWorkload, bench_across_isas, checksum};

/// Luma dimensions the kernel-level groups run over, matching
/// `benches/av1_decode.rs`. One 1080p plane is large enough that per-call
/// dispatch overhead is negligible next to the vectorized inner loops, and it
/// is the size these measurements have always used, so their numbers stay
/// comparable across the move.
const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;

/// Criterion windows for the kernel groups.
///
/// Each group is measured once per available instruction set, so the default
/// five-second window would stretch a plain `cargo bench --bench av1_encode`
/// out for no extra resolution. Two seconds over a 1080p plane is still
/// hundreds of iterations of work per sample.
fn kernel_workload<'a>(codec: &'a str, work: FrameWork) -> IsaWorkload<'a> {
    IsaWorkload {
        measurement_time: Duration::from_secs(2),
        warm_up_time: Duration::from_millis(300),
        ..IsaWorkload::new(codec, work)
    }
}

// ---------------------------------------------------------------------------
// Forward transforms (src/av1_encoder/transform.rs, src/av1_simd/transforms.rs)
// ---------------------------------------------------------------------------

/// Every forward transform size and family the vector kernels cover, applied
/// over a whole frame's worth of blocks.
fn av1_forward_transforms(criterion: &mut Criterion) {
    for (name, size, tx_type) in [
        ("av1_forward_dct_4x4", 4usize, Av1TxType::DctDct),
        ("av1_forward_dct_8x8", 8, Av1TxType::DctDct),
        ("av1_forward_dct_16x16", 16, Av1TxType::DctDct),
        ("av1_forward_dct_32x32", 32, Av1TxType::DctDct),
        ("av1_forward_adst_8x8", 8, Av1TxType::AdstAdst),
        ("av1_forward_flipadst_16x16", 16, Av1TxType::FlipadstAdst),
    ] {
        let residual: Vec<i32> = (0..size * size)
            .map(|index| (index as i32 * 53) % 511 - 255)
            .collect();
        let blocks = (WIDTH / size) * (HEIGHT / size);
        let covered_width = (WIDTH / size * size) as u64;
        let covered_height = (HEIGHT / size * size) as u64;
        let work = FrameWork::new(1, covered_width, covered_height);
        let workload = kernel_workload(name, work);
        bench_across_isas(criterion, &workload, || {
            let mut digest = 0u64;
            for _ in 0..blocks {
                let coefficients = forward_transform(&residual, size, tx_type);
                digest ^= checksum(&coefficients[0].to_le_bytes());
            }
            digest.to_le_bytes().to_vec()
        });
    }
}

// ---------------------------------------------------------------------------
// Whole-frame encode (src/av1_encoder/tile.rs)
// ---------------------------------------------------------------------------

/// Environment variable that opts into the 1080p-scale whole-frame group.
///
/// One 1080p non-lossless frame is most of a second of search work, so a
/// default `cargo bench` would stretch out for minutes. Keeping the large size
/// opt-in leaves the default run usable in an ordinary edit loop, the same way
/// `benches/hevc_encode.rs` gates its 1080p groups.
const LARGE_GROUP_ENV: &str = "ZVIDLIB_BENCH_LARGE";

/// The size the whole-frame groups always run.
const FRAME_SMALL: (u32, u32) = (640, 352);

/// The 1080p-class size, behind [`LARGE_GROUP_ENV`].
const FRAME_LARGE: (u32, u32) = (1920, 1080);

/// `base_q_idx` values the whole-frame groups measure.
///
/// `0` is the lossless WHT path, which searches nothing; the others run the
/// partition, transform-size and transform-type searches, which is where this
/// encoder's time goes. Reading the two side by side is the point of the group:
/// it is the ratio between them, not either number alone, that says what the
/// search costs.
const FRAME_QINDEXES: [u8; 3] = [0, 32, 160];

/// Encodes one synthetic monochrome frame through the public AV1 encoder at
/// each `base_q_idx`, once per instruction set.
fn av1_encode_frames(criterion: &mut Criterion, (width, height): (u32, u32), suffix: &str) {
    let limits = Limits::default();
    let dimensions =
        VideoDimensions::new(width, height, &limits).expect("benchmark dimensions are valid");
    let luma = support::av1_gray8_planes(width, height, 1)
        .pop()
        .expect("one synthetic monochrome plane");
    let frame = VideoFrame::new(
        dimensions,
        PixelFormat::Gray8,
        ColorRange::Full,
        vec![Plane {
            data: luma,
            stride: width as usize,
        }],
        &limits,
    )
    .expect("synthetic monochrome frames are valid");

    for qindex in FRAME_QINDEXES {
        let configuration = VideoEncoderConfig {
            codec: Codec::Av1,
            profile: CodecProfile::Av1Main,
            coded_dimensions: dimensions,
            input_format: PixelFormat::Gray8,
            color_range: ColorRange::Full,
            // Measure the crate's own encode work, not whichever fixed-function
            // block the host happens to ship.
            hardware: HardwarePreference::Avoid,
            timescale: 30,
            frame_duration: 1,
            configuration: if qindex == 0 {
                Vec::new()
            } else {
                vec![qindex]
            },
        };
        let name = format!("av1_encode_frame_q{qindex}{suffix}");
        let workload = IsaWorkload {
            measurement_time: Duration::from_secs(5),
            warm_up_time: Duration::from_millis(500),
            ..IsaWorkload::new(
                &name,
                FrameWork::new(1, u64::from(width), u64::from(height)),
            )
        };
        bench_across_isas(criterion, &workload, || {
            let mut encoder = native_av1_video_encoder_factory()
                .create(&configuration, &limits)
                .expect("the native AV1 encoder is constructible");
            let packets = support::block_on(encoder.encode(
                FrameIndex(0),
                FrameSource::Cpu(CpuFrameSource {
                    frame: &frame,
                    orientation: Orientation::TopLeft,
                }),
            ))
            .expect("the synthetic frame encodes");
            packets.into_iter().flat_map(|packet| packet.data).collect()
        });
    }
}

/// The whole-frame groups at [`FRAME_SMALL`], and at [`FRAME_LARGE`] when
/// [`LARGE_GROUP_ENV`] is set.
fn av1_encode_whole_frame(criterion: &mut Criterion) {
    av1_encode_frames(criterion, FRAME_SMALL, "");
    if std::env::var_os(LARGE_GROUP_ENV).is_some() {
        av1_encode_frames(criterion, FRAME_LARGE, "_1080p");
    } else {
        println!("# skipping av1_encode_frame_*_1080p; set {LARGE_GROUP_ENV}=1 to run it");
    }
}

criterion_group!(benches, av1_encode_whole_frame, av1_forward_transforms);
criterion_main!(benches);
