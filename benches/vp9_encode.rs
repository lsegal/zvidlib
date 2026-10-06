//! Criterion benchmarks for zvidlib's pure-Rust VP9 encoder.
//!
//! The encoder's cost is its rate-distortion search over partitions (64x64
//! down to 8x8), transform sizes (4x4 to 32x32), intra modes and motion
//! vectors, so these groups time whole frames through the public
//! [`zvidlib::native_vp9_video_encoder_factory`] rather than any one stage:
//!
//! * `vp9_encode_key_frame` encodes one key frame, which searches intra
//!   prediction only.
//! * `vp9_encode_sequence` encodes a key frame and three inter frames, the
//!   four frames of moving content the encoder's unit tests time. Its
//!   per-frame cost is mostly the inter frames' motion and mode search.
//!
//! The content is the encoder tests' moving scene at 640x360: a texture that
//! pans four samples a frame under a square that moves two, so the inter
//! frames find real motion and the search tries every partition depth. It is
//! generated once per process, outside the timed loop.
//!
//! The VP9 encoder has no SIMD dispatch site of its own, so these groups run a
//! single arm rather than one per instruction set.

mod support;

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::{
    Codec, CodecProfile, ColorRange, CpuFrameSource, FrameIndex, FrameSource, HardwarePreference,
    Limits, Orientation, PixelFormat, Plane, VideoDimensions, VideoEncoderConfig,
    VideoEncoderFactory, VideoFrame, native_vp9_video_encoder_factory,
};

use support::FrameWork;

/// The benchmark resolution: the 640x360 the search's cost was first
/// measured at (#567).
const SIZE: (u32, u32) = (640, 360);

/// Frames in the sequence group: one key frame and three inter frames.
const SEQUENCE_FRAMES: u32 = 4;

/// One frame of the encoder tests' moving content (`moving_yuv_frame` in
/// `src/vp9_encoder/tests.rs`).
fn moving_frame(width: u32, height: u32, index: u32, limits: &Limits) -> VideoFrame {
    let (width, height) = (width as usize, height as usize);
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let shift = index as usize * 4;
    let mut luma = vec![0_u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let u = x + shift;
            let texture = ((u * 7) ^ (y * 13)) & 0x3f;
            let mut value = 40 + (u * 160 / (width + 30)) + texture;
            let square = 10 + index as usize * 2;
            if (square..square + 12).contains(&x) && (8..20).contains(&y) {
                value = 230 - ((x + y) & 7) * 4;
            }
            luma[y * width + x] = value.min(255) as u8;
        }
    }
    let cb = (0..chroma_width * chroma_height)
        .map(|i| (100 + (i % chroma_width + shift / 2) * 50 / (chroma_width + 30)) as u8)
        .collect();
    let cr = (0..chroma_width * chroma_height)
        .map(|i| (150 - (i / chroma_width) % 40) as u8)
        .collect();
    VideoFrame::new(
        VideoDimensions {
            width: width as u32,
            height: height as u32,
        },
        PixelFormat::Yuv420p8,
        ColorRange::Limited,
        vec![
            Plane {
                data: luma,
                stride: width,
            },
            Plane {
                data: cb,
                stride: chroma_width,
            },
            Plane {
                data: cr,
                stride: chroma_width,
            },
        ],
        limits,
    )
    .expect("synthetic frames are valid")
}

fn configuration(width: u32, height: u32) -> VideoEncoderConfig {
    VideoEncoderConfig {
        codec: Codec::Vp9,
        profile: CodecProfile::Vp9Profile0,
        coded_dimensions: VideoDimensions { width, height },
        input_format: PixelFormat::Yuv420p8,
        color_range: ColorRange::Limited,
        // Measure the crate's own encoder, not whichever fixed-function block
        // the host happens to ship.
        hardware: HardwarePreference::Avoid,
        timescale: 30_000,
        frame_duration: 1_001,
        configuration: Vec::new(),
    }
}

/// Encodes `frames` with a fresh encoder and returns the total bytes, so the
/// work cannot be optimized away.
fn encode(configuration: &VideoEncoderConfig, limits: &Limits, frames: &[VideoFrame]) -> usize {
    let mut encoder = native_vp9_video_encoder_factory()
        .create(configuration, limits)
        .expect("the native VP9 encoder is constructible");
    let mut bytes = 0;
    for (index, frame) in frames.iter().enumerate() {
        let samples = support::block_on(encoder.encode(
            FrameIndex(index as u64),
            FrameSource::Cpu(CpuFrameSource {
                frame,
                orientation: Orientation::TopLeft,
            }),
        ))
        .expect("the synthetic frame encodes");
        bytes += samples
            .iter()
            .map(|sample| sample.data.len())
            .sum::<usize>();
    }
    bytes
}

fn vp9_encode_frames(criterion: &mut Criterion) {
    let (width, height) = SIZE;
    let limits = Limits::default();
    let configuration = configuration(width, height);
    let frames: Vec<VideoFrame> = (0..SEQUENCE_FRAMES)
        .map(|index| moving_frame(width, height, index, &limits))
        .collect();

    for (id, frames) in [
        ("vp9_encode_key_frame", &frames[..1]),
        ("vp9_encode_sequence", &frames[..]),
    ] {
        let mut group = criterion.benchmark_group(support::group_name(id));
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(10));
        let work = FrameWork::new(frames.len() as u64, u64::from(width), u64::from(height));
        support::report_throughput(&mut group, id, work);
        group.bench_function(format!("{width}x{height}"), |bencher| {
            bencher.iter(|| encode(&configuration, &limits, frames));
        });
        group.finish();
    }
}

criterion_group!(benches, vp9_encode_frames);
criterion_main!(benches);
