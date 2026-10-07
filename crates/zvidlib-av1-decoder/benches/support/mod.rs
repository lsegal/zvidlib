//! The `av1_decode` target's support: an AV1 stream produced by the crate's
//! own encoder once per process, and the structured synthetic planes the
//! kernel groups run on, on top of `zvidlib-bench-support`'s shared helpers.
// Each bench target of this package compiles this module and uses only what
// its own groups need, so unused-here is not dead.
#![allow(dead_code)]

use std::sync::OnceLock;

use zvidlib_av1::{FilterPlane, TxSizeGrid};
use zvidlib_core::{
    Codec, CodecProfile, ColorRange, EncodedVideoSample, Limits, PixelFormat, Plane,
    VideoDecoderConfig, VideoFrame,
};

zvidlib_bench_support::bench_support!(zvidlib_av1::simd_sites, zvidlib_color::simd_sites);

// ---------------------------------------------------------------------------
// AV1 software decoder fixtures.
//
// The AV1 groups need two kinds of input the fixtures above do not provide: a
// whole AV1 stream at a resolution where per-frame overhead is negligible, and
// the structured synthetic planes the kernel-level measurements run on. The
// checked-in AV1 vectors are 17x9 and 16x16 conformance streams — correct, but
// far too small to time a decoder with — so the stream below is produced by the
// crate's own AV1 encoder once per process and the planes are generated.
// ---------------------------------------------------------------------------

/// Luma width of [`synthetic_av1_stream`].
pub const AV1_STREAM_WIDTH: u32 = 320;
/// Luma height of [`synthetic_av1_stream`].
pub const AV1_STREAM_HEIGHT: u32 = 180;
/// Frames in [`synthetic_av1_stream`].
///
/// Enough that per-call decoder setup is a small share of one iteration, while
/// keeping a criterion sample well under a second on the software decoder.
pub const AV1_STREAM_FRAMES: usize = 8;

/// An AV1 elementary stream and the decoder configuration that decodes it.
pub struct SyntheticAv1Stream {
    pub configuration: VideoDecoderConfig,
    pub samples: Vec<EncodedVideoSample>,
    pub width: u64,
    pub height: u64,
}

/// Encodes a deterministic monochrome sequence with the crate's native AV1
/// encoder, once per process.
///
/// The native AV1 decoder implements the bounded Main-profile 8-bit lossless
/// monochrome subset its module documents, so a "representative stream" for it
/// is one this crate's own encoder produces. Encoding is hoisted here — a
/// `OnceLock` outside every timed loop — so a decode benchmark measures decode
/// and nothing else.
pub fn synthetic_av1_stream() -> &'static SyntheticAv1Stream {
    use zvidlib_av1_encoder::native_av1_video_encoder_factory;
    use zvidlib_core::{
        CpuFrameSource, FrameIndex, FrameSource, HardwarePreference, Orientation, VideoDimensions,
        VideoEncoderConfig, VideoEncoderFactory,
    };

    static STREAM: OnceLock<SyntheticAv1Stream> = OnceLock::new();
    STREAM.get_or_init(|| {
        let limits = Limits::default();
        let dimensions = VideoDimensions::new(AV1_STREAM_WIDTH, AV1_STREAM_HEIGHT, &limits)
            .expect("the synthetic AV1 dimensions are valid");
        let mut encoder = native_av1_video_encoder_factory()
            .create(
                &VideoEncoderConfig {
                    codec: Codec::Av1,
                    profile: CodecProfile::Av1Main,
                    coded_dimensions: dimensions,
                    input_format: PixelFormat::Gray8,
                    color_range: ColorRange::Full,
                    hardware: HardwarePreference::Avoid,
                    timescale: 30,
                    frame_duration: 1,
                    configuration: Vec::new(),
                },
                &limits,
            )
            .expect("the native AV1 encoder is constructible");

        let mut packets = Vec::new();
        for (index, luma) in
            av1_gray8_planes(AV1_STREAM_WIDTH, AV1_STREAM_HEIGHT, AV1_STREAM_FRAMES)
                .into_iter()
                .enumerate()
        {
            let frame = VideoFrame::new(
                dimensions,
                PixelFormat::Gray8,
                ColorRange::Full,
                vec![Plane {
                    data: luma,
                    stride: AV1_STREAM_WIDTH as usize,
                }],
                &limits,
            )
            .expect("synthetic monochrome frames are valid");
            packets.extend(
                block_on(encoder.encode(
                    FrameIndex(index as u64),
                    FrameSource::Cpu(CpuFrameSource {
                        frame: &frame,
                        orientation: Orientation::TopLeft,
                    }),
                ))
                .expect("the synthetic frame encodes"),
            );
        }
        packets.extend(block_on(encoder.finish()).expect("the encoder finishes"));
        assert_eq!(
            packets.len(),
            AV1_STREAM_FRAMES,
            "the encoder emits one packet per submitted frame"
        );

        SyntheticAv1Stream {
            configuration: VideoDecoderConfig {
                codec: Codec::Av1,
                profile: CodecProfile::Av1Main,
                coded_dimensions: dimensions,
                output_format: PixelFormat::Rgba8,
                color_range: ColorRange::Full,
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
            width: u64::from(AV1_STREAM_WIDTH),
            height: u64::from(AV1_STREAM_HEIGHT),
        }
    })
}

/// Deterministic frame content with enough local structure that the in-loop
/// filters' data-dependent branches are actually taken.
///
/// Ported from the ad-hoc `tests/av1_simd_bench.rs` this suite replaces; the
/// generators there were the useful part of that file.
pub fn av1_structured_plane(width: usize, height: usize) -> FilterPlane {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut data = Vec::with_capacity(width * height);
    for index in 0..width * height {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = (state >> 56) as i32 & 0x1f;
        let gradient = ((index % width) / 8 + (index / width) / 8) as i32;
        data.push(((gradient + noise) & 0xff) as u8);
    }
    FilterPlane::from_samples(width, height, data, &Limits::default())
        .expect("the structured plane fits the default limits")
}

/// Near-flat block content.
///
/// The wide (8-tap and 14-tap) deblocking filters are gated on a flatness
/// check, so they only do work on content like this — which is exactly why
/// this generator, not [`av1_structured_plane`], is the input to the wide-filter
/// measurement.
pub fn av1_flat_blocks_plane(width: usize, height: usize) -> FilterPlane {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut data = Vec::with_capacity(width * height);
    for index in 0..width * height {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let (x, y) = (index % width, index / width);
        let block = ((x / 32 + y / 32) % 5) as i32;
        data.push((100 + block * 6 + ((state >> 60) as i32 & 1)) as u8);
    }
    FilterPlane::from_samples(width, height, data, &Limits::default())
        .expect("the flat-block plane fits the default limits")
}

/// A frame-wide grid of 32x32 transform blocks, which is what makes every luma
/// edge select the 14-tap deblocking filter (AV1 spec §7.14.5).
pub fn av1_wide_tx_grid(width: usize, height: usize) -> TxSizeGrid {
    let mut grid = TxSizeGrid::new(width, height);
    for y in (0..height).step_by(32) {
        for x in (0..width).step_by(32) {
            grid.set_block(x, y, 32, 32);
        }
    }
    grid
}
