//! Scalar-versus-SIMD benchmarks for zvidlib's native VP8 encoder.
//!
//! The encoder's kernels dispatch through two sites in `src/vp8/simd/`:
//! `vp8_encode`, the encoder-only distortion metrics, forward transforms and
//! quantization, and `vp8_recon`, the reconstruction and loop filter the
//! encoder shares with the decoder so that its reference frames are the
//! decoder's. This target times whole frames through the public encoder and
//! then each of those kernels over a frame's worth of blocks.
//!
//! Every group runs once per instruction set `zvidlib::simd::available()`
//! reports, through the crate-wide override in [`zvidlib::simd`], and
//! `benches/support/isa.rs` asserts that each arm is bit-exact with scalar
//! before timing it and that the override really landed in each dispatch
//! family — so a reported speedup cannot come from a kernel that quietly
//! diverged or from a switch that never took effect. For the whole-frame
//! groups that guard is the encoded bitstream itself.
//!
//! # Groups
//!
//! | Group | Stage | Site |
//! | --- | --- | --- |
//! | `vp8_encode_frame_640x352_q{24,90}` | a key frame and three inter frames through the public encoder | all |
//! | `vp8_encode_stage_sad` | 16x16 whole-sample SAD, `sad16_full` | `vp8_encode` |
//! | `vp8_encode_stage_satd` | 16x16 SATD, `satd` | `vp8_encode` |
//! | `vp8_encode_stage_fdct` | residual and forward 4x4 DCT, `residual_dct` | `vp8_encode` |
//! | `vp8_encode_stage_fwht` | forward Y2 Walsh-Hadamard transform | `vp8_encode` |
//! | `vp8_encode_stage_quantize` | quantization and dequantization | `vp8_encode` |
//! | `vp8_encode_stage_idct` | inverse 4x4 DCT and add, `predict.rs` | `vp8_recon` |
//! | `vp8_encode_stage_iwht` | inverse Y2 Walsh-Hadamard transform, `predict.rs` | `vp8_recon` |
//! | `vp8_encode_stage_sixtap` | six-tap 16x16 inter prediction, `predict.rs` | `vp8_recon` |
//! | `vp8_encode_stage_tm_pred` | 16x16 `TM_PRED`, `predict.rs` | `vp8_recon` |
//! | `vp8_encode_stage_loop_filter` | the normal loop filter over a frame, `loop_filter.rs` | `vp8_recon` |
//!
//! The per-stage groups reach the kernels through
//! [`zvidlib::vp8_encoder_bench`], the `#[doc(hidden)]` per-stage access that is
//! the VP8 counterpart to `zvidlib::av1_encoder_bench`. Each covers one
//! 640x352 frame, the whole-frame size, so a stage's time reads directly
//! against a frame's. `ZVIDLIB_BENCH_LARGE=1` adds the 1080p whole-frame
//! groups.
//!
//! See `benches/README.md` for how to run and filter the suite, and for the
//! speedups measured when the kernels landed.

mod support;

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::vp8_encoder_bench::{self as encoder_bench, BenchPlane};
use zvidlib::{
    Codec, CodecProfile, ColorRange, CpuFrameSource, FrameIndex, FrameSource, HardwarePreference,
    Limits, Orientation, PixelFormat, VideoDimensions, VideoEncoderConfig, VideoEncoderFactory,
    native_vp8_video_encoder_factory,
};

use support::FrameWork;
use support::isa::{IsaWorkload, bench_across_isas, log_host_isas};

/// Environment variable that opts into the 1080p whole-frame groups.
///
/// The encoder takes most of half a second for a 1080p frame on one core, so
/// four of them per arm would stretch a default `cargo bench` out for minutes,
/// the same reason `benches/av1_encode.rs` gates its 1080p groups.
const LARGE_GROUP_ENV: &str = "ZVIDLIB_BENCH_LARGE";

/// The size every group runs at.
const FRAME_SMALL: (u32, u32) = (640, 352);

/// The 1080p-class whole-frame size, behind [`LARGE_GROUP_ENV`].
const FRAME_LARGE: (u32, u32) = (1920, 1080);

/// The quantizer indexes the whole-frame groups encode at: the default, which
/// codes most blocks, and a coarse one, which skips many.
const FRAME_QS: [u8; 2] = [24, 90];

/// Frames per whole-frame iteration: a key frame, so intra mode decision is
/// measured, and three inter frames, so motion search is.
const FRAMES: usize = 4;

/// Encodes [`FRAMES`] frames of moving synthetic content through the public
/// VP8 encoder at each of [`FRAME_QS`], once per instruction set.
fn vp8_encode_frames(criterion: &mut Criterion, (width, height): (u32, u32), suffix: &str) {
    let limits = Limits::default();
    let frames = support::synthetic_rgba8_sequence(width, height, FRAMES);
    for q in FRAME_QS {
        let configuration = VideoEncoderConfig {
            codec: Codec::Vp8,
            profile: CodecProfile::Vp8,
            coded_dimensions: VideoDimensions::new(width, height, &limits)
                .expect("benchmark dimensions are valid"),
            input_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            timescale: 30,
            frame_duration: 1,
            configuration: vec![q],
        };
        let name = format!("vp8_encode_frame_{width}x{height}_q{q}{suffix}");
        let workload = IsaWorkload {
            measurement_time: Duration::from_secs(5),
            warm_up_time: Duration::from_millis(500),
            ..IsaWorkload::new(
                &name,
                FrameWork::new(FRAMES as u64, u64::from(width), u64::from(height)),
            )
        };
        bench_across_isas(criterion, &workload, || {
            let mut encoder = native_vp8_video_encoder_factory()
                .create(&configuration, &limits)
                .expect("the native VP8 encoder is constructible");
            let mut bytes = Vec::new();
            for (index, frame) in frames.iter().enumerate() {
                let samples = support::block_on(encoder.encode(
                    FrameIndex(index as u64),
                    FrameSource::Cpu(CpuFrameSource {
                        frame,
                        orientation: Orientation::TopLeft,
                    }),
                ))
                .expect("the synthetic frame encodes");
                bytes.extend(samples.into_iter().flat_map(|sample| sample.data));
            }
            bytes
        });
    }
}

/// The whole-frame groups at [`FRAME_SMALL`], and at [`FRAME_LARGE`] when
/// [`LARGE_GROUP_ENV`] is set.
fn vp8_encode_whole_frame(criterion: &mut Criterion) {
    vp8_encode_frames(criterion, FRAME_SMALL, "");
    if std::env::var_os(LARGE_GROUP_ENV).is_some() {
        vp8_encode_frames(criterion, FRAME_LARGE, "_1080p");
    } else {
        println!("# skipping vp8_encode_frame_*_1080p; set {LARGE_GROUP_ENV}=1 to run them");
    }
}

// ---------------------------------------------------------------------------
// Per-stage groups (src/vp8/, through zvidlib::vp8_encoder_bench)
// ---------------------------------------------------------------------------

/// A deterministic plane: a gradient with texture and noise, so neither a
/// distortion metric nor a filter sees a degenerate input.
fn plane(width: usize, height: usize, seed: u64) -> BenchPlane {
    let mut state = seed | 1;
    let data = (0..width * height)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let (x, y) = (index % width, index / width);
            let gradient = (x * 3 + y * 2) % 160 + ((x / 5 + y / 7) % 2) * 30;
            (gradient as u64 + 40 + (state >> 60)) as u8
        })
        .collect();
    BenchPlane::new(width, height, data)
}

/// Coefficient blocks with the spread a 4x4 DCT of a natural residual has:
/// a large DC, and AC terms that shrink with frequency.
fn coefficient_blocks(count: usize) -> Vec<[i16; 16]> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    (0..count)
        .map(|_| {
            let mut block = [0i16; 16];
            for (index, value) in block.iter_mut().enumerate() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let range = 2048 >> (index / 2).min(10);
                *value = ((state >> 33) % (2 * range as u64 + 1)) as i16 - range as i16;
            }
            block
        })
        .collect()
}

fn kernel_workload(name: &str) -> IsaWorkload<'_> {
    let (width, height) = FRAME_SMALL;
    IsaWorkload {
        measurement_time: Duration::from_secs(2),
        warm_up_time: Duration::from_millis(300),
        ..IsaWorkload::new(name, FrameWork::new(1, u64::from(width), u64::from(height)))
    }
}

fn vp8_encode_stages(criterion: &mut Criterion) {
    let (width, height) = (FRAME_SMALL.0 as usize, FRAME_SMALL.1 as usize);
    let source = plane(width, height, 1);
    let prediction = plane(width, height, 2);
    // One block of coefficients per 4x4 luma block, and one Y2 block per
    // macroblock.
    let blocks = coefficient_blocks(width * height / 16);
    let y2_blocks = coefficient_blocks(width * height / 256);

    bench_across_isas(criterion, &kernel_workload("vp8_encode_stage_sad"), || {
        encoder_bench::sad(&source, &prediction, 3, -2)
            .to_le_bytes()
            .to_vec()
    });
    bench_across_isas(criterion, &kernel_workload("vp8_encode_stage_satd"), || {
        encoder_bench::satd(&source, &prediction)
            .to_le_bytes()
            .to_vec()
    });
    bench_across_isas(criterion, &kernel_workload("vp8_encode_stage_fdct"), || {
        encoder_bench::forward_dct(&source, &prediction)
            .to_le_bytes()
            .to_vec()
    });
    bench_across_isas(criterion, &kernel_workload("vp8_encode_stage_fwht"), || {
        encoder_bench::forward_walsh(&y2_blocks)
            .to_le_bytes()
            .to_vec()
    });
    bench_across_isas(
        criterion,
        &kernel_workload("vp8_encode_stage_quantize"),
        || encoder_bench::quantize(&blocks, 24).to_le_bytes().to_vec(),
    );
    bench_across_isas(criterion, &kernel_workload("vp8_encode_stage_idct"), || {
        let mut plane = prediction.clone();
        encoder_bench::inverse_dct(&mut plane, &blocks);
        plane.data().to_vec()
    });
    bench_across_isas(criterion, &kernel_workload("vp8_encode_stage_iwht"), || {
        encoder_bench::inverse_walsh(&y2_blocks)
            .to_le_bytes()
            .to_vec()
    });
    bench_across_isas(
        criterion,
        &kernel_workload("vp8_encode_stage_sixtap"),
        || {
            let mut output = BenchPlane::new(width, height, vec![0; width * height]);
            // Fractional in both directions, so both filter passes run.
            encoder_bench::sixtap(&source, &mut output, 13, -6);
            output.data().to_vec()
        },
    );
    bench_across_isas(
        criterion,
        &kernel_workload("vp8_encode_stage_tm_pred"),
        || {
            let mut plane = source.clone();
            encoder_bench::tm_predict(&mut plane);
            plane.data().to_vec()
        },
    );
    let chroma = [
        source.clone(),
        plane(width / 2, height / 2, 3),
        plane(width / 2, height / 2, 4),
    ];
    bench_across_isas(
        criterion,
        &kernel_workload("vp8_encode_stage_loop_filter"),
        || {
            let mut planes = chroma.clone();
            encoder_bench::loop_filter(&mut planes, 32, false);
            planes
                .iter()
                .flat_map(|plane| plane.data().iter().copied())
                .collect()
        },
    );
}

criterion_group!(
    benches,
    log_host_isas,
    vp8_encode_whole_frame,
    vp8_encode_stages
);
criterion_main!(benches);
