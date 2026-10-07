//! Criterion benchmarks for zvidlib's native VP9 encoder, scalar versus SIMD.
//!
//! Two axes, as for the HEVC and AV1 encoder targets:
//!
//! * **Whole-frame versus per-stage.** `vp9_encode_key_frame` and
//!   `vp9_encode_sequence` encode synthetic RGBA8 frames through the public
//!   [`zvidlib_vp9_encoder::native_vp9_video_encoder_factory`]: one key frame, which
//!   searches intra prediction only, and a key frame followed by three inter
//!   frames, which adds the motion search. The `vp9_encode_stage_*` groups time
//!   each vectorized kernel on its own through [`zvidlib_vp9_encoder::bench`],
//!   so a whole-frame ratio can be attributed to the stage that moved it.
//! * **Instruction set.** Every group runs once per entry in
//!   `zvidlib_core::simd::available()` through `support::isa::bench_across_isas`,
//!   which pins the crate-wide override, asserts it reached every dispatch
//!   family (the encoder's kernels are the `vp9_encode` site), and checks each
//!   arm is bit-exact with scalar before timing it. For the whole-frame groups
//!   that check is the encoded bitstream itself.
//!
//! The whole-frame search also spends time the `vp9_encode` site does not
//! reach: the bool coder's bit costing, the mode and partition bookkeeping,
//! and the decoder's inverse transforms and loop filter, which are the VP9
//! decoder's own kernels and are vectorized with it (#570). Those arms of the
//! whole-frame ratio are expected to read flat.
//!
//! Inputs are synthetic frames from `benches/support`, never a decoded file.

mod support;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib_core::{
    Codec, CodecProfile, ColorRange, CpuFrameSource, FrameIndex, FrameSource, HardwarePreference,
    Limits, Orientation, PixelFormat, VideoDimensions, VideoEncoderConfig, VideoEncoderFactory,
    VideoFrame,
};
use zvidlib_vp9_encoder::bench as encoder_bench;
use zvidlib_vp9_encoder::native_vp9_video_encoder_factory;

use support::FrameWork;
use support::isa::{IsaWorkload, bench_across_isas, log_host_isas};

/// The resolution every group runs at.
const SIZE: (u32, u32) = (640, 360);

/// Frames in `vp9_encode_sequence`: one key frame and three inter frames.
const SEQUENCE_FRAMES: usize = 4;

fn encoder_config(width: u32, height: u32) -> VideoEncoderConfig {
    VideoEncoderConfig {
        codec: Codec::Vp9,
        profile: CodecProfile::Vp9Profile0,
        coded_dimensions: VideoDimensions::new(width, height, &Limits::default())
            .expect("benchmark dimensions are valid"),
        input_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        // Measure the crate's own encoder, not whichever fixed-function block
        // the host happens to ship.
        hardware: HardwarePreference::Avoid,
        timescale: 30_000,
        frame_duration: 1_001,
        configuration: Vec::new(),
    }
}

/// Encodes `frames` with a fresh encoder, returning the bitstream.
fn encode(configuration: &VideoEncoderConfig, frames: &[VideoFrame]) -> Vec<u8> {
    let mut encoder = native_vp9_video_encoder_factory()
        .create(configuration, &Limits::default())
        .expect("the native VP9 encoder is constructible");
    let mut bitstream = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        let source = FrameSource::Cpu(CpuFrameSource {
            frame,
            orientation: Orientation::TopLeft,
        });
        for sample in support::block_on(encoder.encode(FrameIndex(index as u64), source))
            .expect("the synthetic frame encodes")
        {
            bitstream.extend_from_slice(&sample.data);
        }
    }
    bitstream
}

fn whole_frame(criterion: &mut Criterion) {
    let (width, height) = SIZE;
    let frames = support::synthetic_rgba8_sequence(width, height, SEQUENCE_FRAMES);
    let configuration = encoder_config(width, height);
    for (group, frames) in [
        ("vp9_encode_key_frame", &frames[..1]),
        ("vp9_encode_sequence", &frames[..]),
    ] {
        let workload = IsaWorkload::new(
            group,
            FrameWork::new(frames.len() as u64, u64::from(width), u64::from(height)),
        );
        bench_across_isas(criterion, &workload, || encode(&configuration, frames));
    }
}

/// The luma planes of two consecutive synthetic frames: the second is coded
/// against the first.
fn luma_pair() -> (Vec<u8>, Vec<u8>) {
    let frames = support::synthetic_yuv420_sequence(SIZE.0, SIZE.1, 2);
    (
        frames[1].planes[0].data.clone(),
        frames[0].planes[0].data.clone(),
    )
}

fn stages(criterion: &mut Criterion) {
    let (width, height) = (SIZE.0 as usize, SIZE.1 as usize);
    let work = FrameWork::new(1, u64::from(SIZE.0), u64::from(SIZE.1));
    let (current, reference) = luma_pair();

    for (tx_size, name) in ["4x4", "8x8", "16x16", "32x32"].into_iter().enumerate() {
        let group = format!("vp9_encode_stage_fdct_quant_{name}");
        bench_across_isas(criterion, &IsaWorkload::new(&group, work), || {
            encoder_bench::forward_transform_quantize(&current, &reference, width, height, tx_size)
        });
    }
    bench_across_isas(
        criterion,
        &IsaWorkload::new("vp9_encode_stage_sad", work),
        || encoder_bench::motion_search_sad(&current, &reference, width, height, 16),
    );
    bench_across_isas(
        criterion,
        &IsaWorkload::new("vp9_encode_stage_sse", work),
        || encoder_bench::block_and_plane_sse(&current, &reference, width, height),
    );
    bench_across_isas(
        criterion,
        &IsaWorkload::new("vp9_encode_stage_inter_pred", work),
        || encoder_bench::inter_prediction(&reference, width, height, 16),
    );
    bench_across_isas(
        criterion,
        &IsaWorkload::new("vp9_encode_stage_tm_pred", work),
        || encoder_bench::tm_prediction(&current, width, height, 16),
    );
    let rgba = support::synthetic_rgba8_sequence(SIZE.0, SIZE.1, 1);
    bench_across_isas(
        criterion,
        &IsaWorkload::new("vp9_encode_stage_rgba_to_yuv420", work),
        || encoder_bench::rgba_to_yuv420(&rgba[0]),
    );
}

criterion_group!(benches, log_host_isas, stages, whole_frame);
criterion_main!(benches);
