//! Scalar-versus-SIMD benchmarks for the VP8 and VP9 software decoders and the
//! YUV-to-RGBA output conversion they share with the AV1 decoder (issue #574).
//!
//! Every group runs once per instruction set `zvidlib::simd::available()`
//! reports, through the crate-wide override in [`zvidlib::simd`];
//! `benches/support/isa.rs` asserts each arm is bit-exact with scalar before
//! timing it.
//!
//! # Groups
//!
//! | Group | Stage |
//! | --- | --- |
//! | `yuv_to_rgba_1080p`, `yuv_to_rgba_4k` | `convert_to_rgba8` over one 4:2:0 picture, `crates/zvidlib-color/src/yuv_to_rgba.rs` |
//! | `vp8_decode_frame` | whole-frame decode through `native_vp8_video_decoder_factory` |
//! | `vp9_decode_frame` | whole-frame decode through `native_vp9_video_decoder_factory` |
//!
//! The conversion groups time the kernel on its own. The whole-frame groups
//! decode a stream the crate's own encoders produce once per process, so an
//! iteration is decoder work only, conversion included. Both decoders' own
//! decoding kernels are vectorized too (issues #568 and #570), and
//! `crates/zvidlib-vp8/benches/vp8_decode.rs` and
//! `crates/zvidlib-vp9-decoder/benches/vp9_decode.rs` time them on their own
//! and in a decode that stops before the conversion.
//!
//! See `benches/README.md` for how to run and filter the suite.

mod support;

use std::sync::OnceLock;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::av1_filters::{FilterFrame, MatrixCoefficients, convert_to_rgba8};
use zvidlib::{
    CancellationToken, Codec, CodecProfile, ColorRange, CpuFrameSource, EncodedVideoSample,
    FilterPlane, FrameDigest, FrameIndex, FrameSource, HardwarePreference, Limits, Orientation,
    PixelFormat, Plane, VideoDecoderConfig, VideoDecoderFactory, VideoEncoderConfig,
    VideoEncoderFactory, VideoFrame, native_vp8_video_decoder_factory,
    native_vp8_video_encoder_factory, native_vp9_video_decoder_factory,
    native_vp9_video_encoder_factory,
};

use support::isa::{IsaWorkload, bench_across_isas, log_host_isas};
use support::{FrameWork, block_on, synthetic_yuv420_sequence};

// ---------------------------------------------------------------------------
// Output conversion (crates/zvidlib-color/src/yuv_to_rgba.rs)
// ---------------------------------------------------------------------------

/// One synthetic 4:2:0 picture as the decoders hand it to `convert_to_rgba8`.
fn filter_frame(width: u32, height: u32) -> FilterFrame {
    let limits = Limits::default();
    let frame = synthetic_yuv420_sequence(width, height, 1)
        .pop()
        .expect("one synthetic frame");
    let (chroma_width, chroma_height) =
        ((width as usize).div_ceil(2), (height as usize).div_ceil(2));
    let plane = |index: usize, plane_width: usize, plane_height: usize| {
        let source: &Plane = &frame.planes[index];
        let samples = (0..plane_height)
            .flat_map(|row| &source.data[row * source.stride..row * source.stride + plane_width])
            .copied()
            .collect();
        FilterPlane::from_samples(plane_width, plane_height, samples, &limits)
            .expect("synthetic plane is valid")
    };
    FilterFrame::new_yuv(
        plane(0, width as usize, height as usize),
        plane(1, chroma_width, chroma_height),
        plane(2, chroma_width, chroma_height),
        true,
        true,
    )
    .expect("synthetic frame is valid")
}

/// `convert_to_rgba8` over one picture, with the limited-range BT.709 matrix
/// HD VP9 and AV1 streams signal.
fn yuv_to_rgba_group(criterion: &mut Criterion, name: &str, width: u32, height: u32) {
    let frame = filter_frame(width, height);
    let limits = Limits {
        max_width: width,
        max_height: height,
        ..Limits::default()
    };
    let workload = IsaWorkload {
        measurement_time: Duration::from_secs(3),
        warm_up_time: Duration::from_millis(300),
        ..IsaWorkload::new(name, FrameWork::new(1, u64::from(width), u64::from(height)))
    };
    bench_across_isas(criterion, &workload, || {
        convert_to_rgba8(
            &frame,
            ColorRange::Limited,
            MatrixCoefficients::Bt709,
            &limits,
        )
        .expect("the picture converts")
    });
}

fn yuv_to_rgba_1080p(criterion: &mut Criterion) {
    yuv_to_rgba_group(criterion, "yuv_to_rgba_1080p", 1920, 1080);
}

fn yuv_to_rgba_4k(criterion: &mut Criterion) {
    yuv_to_rgba_group(criterion, "yuv_to_rgba_4k", 3840, 2160);
}

// ---------------------------------------------------------------------------
// Whole-frame decode
// ---------------------------------------------------------------------------

/// Luma width of the synthetic VP8 and VP9 streams.
const STREAM_WIDTH: u32 = 640;
/// Luma height of the synthetic VP8 and VP9 streams.
const STREAM_HEIGHT: u32 = 360;
/// Frames in each synthetic stream: a key frame and inter frames after it.
const STREAM_FRAMES: usize = 6;

/// An elementary stream and the decoder configuration that decodes it.
struct SyntheticStream {
    configuration: VideoDecoderConfig,
    samples: Vec<EncodedVideoSample>,
}

/// Encodes `frames` with `factory`, once per call site's `OnceLock`.
fn encode_stream(
    factory: &dyn VideoEncoderFactory,
    codec: Codec,
    profile: CodecProfile,
    frames: &[VideoFrame],
) -> SyntheticStream {
    let limits = Limits::default();
    let first = &frames[0];
    let mut encoder = factory
        .create(
            &VideoEncoderConfig {
                codec,
                profile,
                coded_dimensions: first.dimensions,
                input_format: first.pixel_format,
                color_range: ColorRange::Limited,
                hardware: HardwarePreference::Avoid,
                timescale: 30,
                frame_duration: 1,
                configuration: Vec::new(),
            },
            &limits,
        )
        .expect("the native encoder is constructible");
    let mut packets = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        packets.extend(
            block_on(encoder.encode(
                FrameIndex(index as u64),
                FrameSource::Cpu(CpuFrameSource {
                    frame,
                    orientation: Orientation::TopLeft,
                }),
            ))
            .expect("the synthetic frame encodes"),
        );
    }
    packets.extend(block_on(encoder.finish()).expect("the encoder finishes"));
    SyntheticStream {
        configuration: VideoDecoderConfig {
            codec,
            profile,
            coded_dimensions: first.dimensions,
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: encoder.config().decoder_config.clone(),
        },
        samples: packets
            .into_iter()
            .enumerate()
            .map(|(index, packet)| EncodedVideoSample {
                presentation_index: FrameIndex(index as u64),
                random_access: packet.is_sync,
                data: packet.data,
            })
            .collect(),
    }
}

/// The synthetic 4:2:0 sequence, as the VP9 encoder takes it.
fn yuv_frames() -> Vec<VideoFrame> {
    synthetic_yuv420_sequence(STREAM_WIDTH, STREAM_HEIGHT, STREAM_FRAMES)
}

/// The same sequence as limited-range RGBA, which is what the VP8 encoder
/// takes: luma as grey, tinted by the chroma planes.
fn rgba_frames() -> Vec<VideoFrame> {
    let limits = Limits::default();
    let (width, height) = (STREAM_WIDTH as usize, STREAM_HEIGHT as usize);
    yuv_frames()
        .into_iter()
        .map(|frame| {
            let [y, u, v] = [&frame.planes[0], &frame.planes[1], &frame.planes[2]];
            let mut rgba = Vec::with_capacity(width * height * 4);
            for row in 0..height {
                for column in 0..width {
                    let luma = i32::from(y.data[row * y.stride + column]);
                    let chroma = (row / 2) * u.stride + column / 2;
                    let cb = i32::from(u.data[chroma]) - 128;
                    let cr = i32::from(v.data[chroma]) - 128;
                    let clip = |value: i32| value.clamp(16, 235) as u8;
                    rgba.extend_from_slice(&[
                        clip(luma + cr),
                        clip(luma - (cb + cr) / 2),
                        clip(luma + cb),
                        255,
                    ]);
                }
            }
            VideoFrame::new(
                frame.dimensions,
                PixelFormat::Rgba8,
                ColorRange::Limited,
                vec![Plane {
                    data: rgba,
                    stride: width * 4,
                }],
                &limits,
            )
            .expect("synthetic RGBA frames are valid")
        })
        .collect()
}

fn vp8_stream() -> &'static SyntheticStream {
    static STREAM: OnceLock<SyntheticStream> = OnceLock::new();
    STREAM.get_or_init(|| {
        encode_stream(
            &native_vp8_video_encoder_factory(),
            Codec::Vp8,
            CodecProfile::Vp8,
            &rgba_frames(),
        )
    })
}

fn vp9_stream() -> &'static SyntheticStream {
    static STREAM: OnceLock<SyntheticStream> = OnceLock::new();
    STREAM.get_or_init(|| {
        encode_stream(
            &native_vp9_video_encoder_factory(),
            Codec::Vp9,
            CodecProfile::Vp9Profile0,
            &yuv_frames(),
        )
    })
}

/// Decodes `stream` end to end, once per instruction set.
fn decode_group(
    criterion: &mut Criterion,
    name: &str,
    factory: &dyn VideoDecoderFactory,
    stream: &SyntheticStream,
) {
    let work = FrameWork::new(
        stream.samples.len() as u64,
        u64::from(STREAM_WIDTH),
        u64::from(STREAM_HEIGHT),
    );
    let workload = IsaWorkload {
        measurement_time: Duration::from_secs(5),
        ..IsaWorkload::new(name, work)
    };
    bench_across_isas(criterion, &workload, || {
        let mut decoder = factory
            .create(&stream.configuration, &Limits::default())
            .expect("the native decoder is constructible");
        let cancellation = CancellationToken::new();
        let mut digests = Vec::new();
        let mut digest = |frame: &VideoFrame| {
            digests.extend_from_slice(
                FrameDigest::from_frame(frame)
                    .expect("a decoded frame digests")
                    .to_hex()
                    .as_bytes(),
            );
        };
        for sample in &stream.samples {
            for decoded in decoder
                .submit(sample, &cancellation)
                .expect("the synthetic stream decodes")
            {
                digest(&decoded.frame);
            }
        }
        for decoded in decoder.drain(&cancellation).expect("the decoder drains") {
            digest(&decoded.frame);
        }
        assert!(
            !digests.is_empty(),
            "the synthetic stream yields decoded frames"
        );
        digests
    });
}

fn vp8_decode_frame(criterion: &mut Criterion) {
    decode_group(
        criterion,
        "vp8_decode_frame",
        &native_vp8_video_decoder_factory(),
        vp8_stream(),
    );
}

fn vp9_decode_frame(criterion: &mut Criterion) {
    decode_group(
        criterion,
        "vp9_decode_frame",
        &native_vp9_video_decoder_factory(),
        vp9_stream(),
    );
}

criterion_group!(
    benches,
    log_host_isas,
    yuv_to_rgba_1080p,
    yuv_to_rgba_4k,
    vp8_decode_frame,
    vp9_decode_frame
);
criterion_main!(benches);
