//! Dependency-free native AV1 encoding.
//!
//! The backend is deliberately narrow and honest: all-intra, 8-bit monochrome
//! AV1 Main profile. Every sample is an independent temporal unit containing a
//! sequence header and key frame, so MP4 random access does not depend on
//! encoder-private state.
//!
//! Two quantization profiles are available, selected by
//! [`VideoEncoderConfig::configuration`]: an empty configuration (the default)
//! encodes losslessly with the 4x4 WHT, and a single-byte configuration
//! carrying a nonzero `base_q_idx` encodes non-lossless, which is what gives
//! [`transform::forward_transform`] a caller. See [`parse_base_q_idx`].

pub mod bench;
#[allow(dead_code)]
mod bitwriter;
#[allow(dead_code)]
mod cdf;
mod headers;
#[allow(dead_code)]
mod leb128;
#[allow(dead_code)]
mod symbol;
mod tile;
pub(crate) mod transform;
#[allow(dead_code)]
pub(crate) mod wht;

use crate::{
    Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange, EncodedSample,
    EncoderConfig, EncoderFuture, Error, ErrorKind, FrameIndex, FrameSource, HardwarePreference,
    Limits, Orientation, PixelFormat, Result, SampleDependency, VideoDimensions, VideoEncoder,
    VideoEncoderConfig, VideoEncoderFactory, VideoEncoderFormat,
};
use headers::Av1StillConfig;
use tile::FrameEncoder;

/// Returns the native software AV1 encoder backend.
///
/// The current backend supports lossless `Gray8` AV1 Main-profile input. Color
/// and inter-frame tools can be added without changing the factory contract.
pub fn native_av1_video_encoder_factory() -> impl VideoEncoderFactory {
    NativeAv1EncoderFactory
}

struct NativeAv1EncoderFactory;

impl VideoEncoderFactory for NativeAv1EncoderFactory {
    fn capability(&self, configuration: &VideoEncoderConfig) -> CodecSupport {
        validate_configuration(configuration)
    }

    fn create(
        &self,
        configuration: &VideoEncoderConfig,
        limits: &Limits,
    ) -> Result<Box<dyn VideoEncoder>> {
        let support = self.capability(configuration);
        if !support.is_supported() {
            return Err(capability_error(support));
        }
        validate_limits(configuration.coded_dimensions, limits)?;

        let level = headers::pick_level(
            configuration.coded_dimensions.width,
            configuration.coded_dimensions.height,
            configuration.timescale,
            configuration.frame_duration,
        )?;
        let stream = stream_configuration(configuration.color_range, level);
        let decoder_config = make_av1c(configuration.coded_dimensions, &stream)?;
        let base_q_idx = parse_base_q_idx(&configuration.configuration).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                "the native AV1 encoder's configuration is either empty (lossless) or a single \
                 base_q_idx byte",
            )
        })?;
        Ok(Box::new(NativeAv1Encoder {
            declared: EncoderConfig {
                codec: Codec::Av1,
                timescale: configuration.timescale,
                decoder_config,
            },
            format: VideoEncoderFormat {
                dimensions: configuration.coded_dimensions,
                pixel_format: PixelFormat::Gray8,
            },
            color_range: configuration.color_range,
            frame_duration: configuration.frame_duration,
            limits: *limits,
            next_index: 0,
            finished: false,
            stream,
            base_q_idx,
        }))
    }
}

fn validate_configuration(configuration: &VideoEncoderConfig) -> CodecSupport {
    if configuration.codec != Codec::Av1 {
        return CodecSupport::UnsupportedCodec;
    }
    if configuration.profile != CodecProfile::Av1Main {
        return CodecSupport::UnsupportedProfile;
    }
    if configuration.hardware == HardwarePreference::Require {
        return CodecSupport::HardwareUnavailable;
    }
    if configuration.coded_dimensions.width == 0 || configuration.coded_dimensions.height == 0 {
        return invalid_support("AV1 coded dimensions must be nonzero");
    }
    if configuration.input_format != PixelFormat::Gray8 {
        return invalid_support("the native AV1 backend currently requires Gray8 input");
    }
    if configuration.timescale == 0 || configuration.frame_duration == 0 {
        return invalid_support("AV1 timescale and frame duration must be nonzero");
    }
    if parse_base_q_idx(&configuration.configuration).is_none() {
        return invalid_support(
            "the native AV1 encoder's configuration is either empty (lossless) or a single \
             base_q_idx byte",
        );
    }
    if headers::pick_level(
        configuration.coded_dimensions.width,
        configuration.coded_dimensions.height,
        configuration.timescale,
        configuration.frame_duration,
    )
    .is_err()
    {
        return invalid_support("AV1 dimensions and frame rate exceed level 6.0 limits");
    }
    CodecSupport::Supported {
        implementation: CodecImplementation::Software,
    }
}

/// The backend-private configuration this encoder accepts, as the frame header's `base_q_idx`
/// (AV1 §5.9.12), or `None` when the blob is not one this backend understands.
///
/// An empty configuration is `base_q_idx = 0`, the lossless profile every earlier release
/// emitted. A one-byte configuration is that byte, so `vec![32]` asks for a non-lossless frame at
/// quantizer index 32; the quantizer index is the only encoder knob the AV1 bitstream needs, and
/// keeping it a single byte means the whole surface is `configuration.is_empty()` or not.
///
/// Non-lossless streams round-trip through [`crate::decode_av1_lossless_intra`], the crate's own
/// decoder. They are *not* yet interchange-grade: the symbols only a non-lossless frame reads
/// (`eob_pt` above 16 coefficients, `tx_depth`, `ext_tx`) use this crate's placeholder CDFs
/// rather than the specification's default tables, and the encoder writes no `delta_q_present`
/// bit, matching the decoder's parse. Lossless output is unchanged and stays the default.
fn parse_base_q_idx(configuration: &[u8]) -> Option<u8> {
    match configuration {
        [] => Some(0),
        [base_q_idx] => Some(*base_q_idx),
        _ => None,
    }
}

fn invalid_support(reason: &str) -> CodecSupport {
    CodecSupport::InvalidConfiguration {
        reason: reason.into(),
    }
}

fn capability_error(support: CodecSupport) -> Error {
    let (kind, message) = match support {
        CodecSupport::UnsupportedCodec => (ErrorKind::Unsupported, "unsupported encoder codec"),
        CodecSupport::UnsupportedProfile => {
            (ErrorKind::Unsupported, "unsupported AV1 encoder profile")
        }
        CodecSupport::InvalidConfiguration { reason } => {
            return Error::new(ErrorKind::InvalidInput, reason);
        }
        CodecSupport::HardwareUnavailable => (
            ErrorKind::Unsupported,
            "the native AV1 encoder is software-only",
        ),
        CodecSupport::Supported { .. } => (
            ErrorKind::Internal,
            "encoder capability changed unexpectedly",
        ),
    };
    Error::new(kind, message)
}

fn validate_limits(dimensions: VideoDimensions, limits: &Limits) -> Result<()> {
    if dimensions.width > limits.max_width || dimensions.height > limits.max_height {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "AV1 dimensions exceed the configured limits",
        ));
    }
    let pixels = u64::from(dimensions.width)
        .checked_mul(u64::from(dimensions.height))
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "AV1 frame size overflow"))?;
    let bounded_working_allocation = pixels
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(64 * 1024))
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "AV1 allocation size overflow"))?;
    if bounded_working_allocation > limits.max_allocation_bytes {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "AV1 frame exceeds the configured allocation limit",
        ));
    }
    Ok(())
}

fn stream_configuration(color_range: ColorRange, level: u8) -> Av1StillConfig {
    Av1StillConfig {
        seq_profile: 0,
        seq_level_idx_0: level,
        seq_tier_0: 0,
        high_bitdepth: false,
        twelve_bit: false,
        monochrome: true,
        chroma_subsampling_x: 1,
        chroma_subsampling_y: 1,
        chroma_sample_position: 0,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
        full_range: color_range == ColorRange::Full,
    }
}

fn sequence_obu(dimensions: VideoDimensions, stream: &Av1StillConfig) -> Vec<u8> {
    let payload = headers::sequence_header_payload(stream, dimensions.width, dimensions.height);
    let mut obu = Vec::new();
    headers::write_obu(&mut obu, 1, &payload);
    obu
}

fn make_av1c(dimensions: VideoDimensions, stream: &Av1StillConfig) -> Result<Vec<u8>> {
    let config_obus = sequence_obu(dimensions, stream);
    let payload_len = 4_usize
        .checked_add(config_obus.len())
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "av1C size overflow"))?;
    let box_len = 8_usize
        .checked_add(payload_len)
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "av1C size overflow"))?;
    let box_len = u32::try_from(box_len)
        .map_err(|_| Error::new(ErrorKind::ResourceLimit, "av1C box is too large"))?;

    let mut output = Vec::with_capacity(box_len as usize);
    output.extend_from_slice(&box_len.to_be_bytes());
    output.extend_from_slice(b"av1C");
    output.push(0x81);
    output.push((stream.seq_profile << 5) | (stream.seq_level_idx_0 & 0x1f));
    output.push(
        (stream.seq_tier_0 << 7)
            | (u8::from(stream.high_bitdepth) << 6)
            | (u8::from(stream.twelve_bit) << 5)
            | (u8::from(stream.monochrome) << 4)
            | (stream.chroma_subsampling_x << 3)
            | (stream.chroma_subsampling_y << 2)
            | (stream.chroma_sample_position & 3),
    );
    output.push(0); // no initial presentation delay
    output.extend_from_slice(&config_obus);
    Ok(output)
}

struct NativeAv1Encoder {
    declared: EncoderConfig,
    format: VideoEncoderFormat,
    color_range: ColorRange,
    frame_duration: u32,
    limits: Limits,
    next_index: u64,
    finished: bool,
    stream: Av1StillConfig,
    /// The frame header's `base_q_idx`; `0` selects the lossless WHT profile.
    base_q_idx: u8,
}

impl VideoEncoder for NativeAv1Encoder {
    fn config(&self) -> &EncoderConfig {
        &self.declared
    }

    fn format(&self) -> VideoEncoderFormat {
        self.format
    }

    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        frame: FrameSource<'a>,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move { self.encode_frame(index, frame) })
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move {
            self.finished = true;
            Ok(Vec::new())
        })
    }
}

impl NativeAv1Encoder {
    fn encode_frame(
        &mut self,
        index: FrameIndex,
        source: FrameSource<'_>,
    ) -> Result<Vec<EncodedSample>> {
        if self.finished {
            return Err(Error::new(
                ErrorKind::InvalidState,
                "cannot encode AV1 frames after finish",
            ));
        }
        if index.0 != self.next_index {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "AV1 input frame indexes must be zero-based and consecutive",
            ));
        }
        let FrameSource::Cpu(source) = source else {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "the native AV1 backend currently requires CPU frames",
            ));
        };
        let frame = source.frame;
        if frame.dimensions != self.format.dimensions
            || frame.pixel_format != PixelFormat::Gray8
            || frame.color_range != self.color_range
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "AV1 input frame does not match the configured dimensions, format, and range",
            ));
        }
        let plane = frame.planes.first().ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                "AV1 Gray8 input requires one plane",
            )
        })?;
        let width = usize::try_from(self.format.dimensions.width)
            .map_err(|_| Error::new(ErrorKind::ResourceLimit, "AV1 width overflow"))?;
        let height = usize::try_from(self.format.dimensions.height)
            .map_err(|_| Error::new(ErrorKind::ResourceLimit, "AV1 height overflow"))?;
        if plane.stride < width {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "AV1 Gray8 plane stride is smaller than the coded width",
            ));
        }
        let required = plane
            .stride
            .checked_mul(height)
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "AV1 plane size overflow"))?;
        if plane.data.len() < required {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "AV1 Gray8 plane is shorter than its stride and height",
            ));
        }

        let pixels = width
            .checked_mul(height)
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "AV1 pixel count overflow"))?;
        let mut packed = Vec::with_capacity(pixels);
        for logical_row in 0..height {
            let stored_row = match source.orientation {
                Orientation::TopLeft => logical_row,
                Orientation::BottomLeft => height - logical_row - 1,
            };
            let start = stored_row * plane.stride;
            packed.extend_from_slice(&plane.data[start..start + width]);
        }

        let mi_cols = 2 * ((width + 7) >> 3);
        let mi_rows = 2 * ((height + 7) >> 3);
        let sequence = headers::sequence_header_payload(
            &self.stream,
            self.format.dimensions.width,
            self.format.dimensions.height,
        );
        let order_hint = (index.0 % u64::from(1u32 << headers::ORDER_HINT_BITS)) as u32;
        let mut frame_payload = headers::frame_header_payload(
            self.format.dimensions.width,
            self.format.dimensions.height,
            u32::try_from(mi_cols)
                .map_err(|_| Error::new(ErrorKind::ResourceLimit, "AV1 MI width overflow"))?,
            u32::try_from(mi_rows)
                .map_err(|_| Error::new(ErrorKind::ResourceLimit, "AV1 MI height overflow"))?,
            order_hint,
            self.base_q_idx,
        );
        frame_payload.extend_from_slice(
            &FrameEncoder::new(&packed, width, height, self.base_q_idx).encode(),
        );
        let data = headers::assemble_temporal_unit(&sequence, &frame_payload);
        if u64::try_from(data.len()).unwrap_or(u64::MAX) > self.limits.max_allocation_bytes {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "encoded AV1 access unit exceeds the configured allocation limit",
            ));
        }

        let timestamp = i64::try_from(index.0)
            .ok()
            .and_then(|value| value.checked_mul(i64::from(self.frame_duration)))
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "AV1 timestamp overflow"))?;
        self.next_index = self
            .next_index
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "AV1 frame index overflow"))?;
        Ok(vec![EncodedSample {
            data,
            dts: timestamp,
            pts: timestamp,
            duration: self.frame_duration,
            is_sync: true,
            dependency: SampleDependency::INDEPENDENT,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuFrameSource, Plane, VideoFrame};
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let mut context = Context::from_waker(Waker::noop());
        let mut future = pin!(future);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
        }
    }

    pub(super) fn configuration() -> VideoEncoderConfig {
        VideoEncoderConfig {
            codec: Codec::Av1,
            profile: CodecProfile::Av1Main,
            coded_dimensions: VideoDimensions {
                width: 17,
                height: 9,
            },
            input_format: PixelFormat::Gray8,
            color_range: ColorRange::Full,
            hardware: HardwarePreference::Avoid,
            timescale: 30_000,
            frame_duration: 1_001,
            configuration: Vec::new(),
        }
    }

    fn frame(value: u8) -> VideoFrame {
        let config = configuration();
        let width = config.coded_dimensions.width as usize;
        let height = config.coded_dimensions.height as usize;
        VideoFrame::new(
            config.coded_dimensions,
            PixelFormat::Gray8,
            ColorRange::Full,
            vec![Plane {
                data: vec![value; width * height],
                stride: width,
            }],
            &Limits::default(),
        )
        .unwrap()
    }

    #[test]
    fn factory_advertises_only_the_implemented_surface() {
        let factory = native_av1_video_encoder_factory();
        assert!(factory.capability(&configuration()).is_supported());

        let mut invalid = configuration();
        invalid.input_format = PixelFormat::Rgba8;
        assert!(matches!(
            factory.capability(&invalid),
            CodecSupport::InvalidConfiguration { .. }
        ));

        invalid = configuration();
        invalid.hardware = HardwarePreference::Require;
        assert_eq!(
            factory.capability(&invalid),
            CodecSupport::HardwareUnavailable
        );
    }

    #[test]
    fn av1c_matches_the_monochrome_main_sequence_header() {
        let factory = native_av1_video_encoder_factory();
        let encoder = factory
            .create(&configuration(), &Limits::default())
            .unwrap();
        let av1c = &encoder.config().decoder_config;
        assert_eq!(&av1c[4..8], b"av1C");
        assert_eq!(av1c[8], 0x81);
        assert_eq!(av1c[9] >> 5, 0);
        assert_ne!(av1c[10] & 0x10, 0);
        assert_ne!(av1c[10] & 0x0c, 0);
        assert_eq!((av1c[12] >> 3) & 0x0f, 1);
    }

    #[test]
    fn emits_independent_standard_obu_access_units_with_exact_timing() {
        let factory = native_av1_video_encoder_factory();
        let mut encoder = factory
            .create(&configuration(), &Limits::default())
            .unwrap();
        for index in 0..2 {
            let source = frame(80 + index as u8);
            let packets = block_on(encoder.encode(
                FrameIndex(index),
                FrameSource::Cpu(CpuFrameSource {
                    frame: &source,
                    orientation: Orientation::TopLeft,
                }),
            ))
            .unwrap();
            assert_eq!(packets.len(), 1);
            let packet = &packets[0];
            assert_eq!(packet.dts, index as i64 * 1_001);
            assert_eq!(packet.pts, packet.dts);
            assert_eq!(packet.duration, 1_001);
            assert!(packet.is_sync);
            assert_eq!(packet.dependency, SampleDependency::INDEPENDENT);
            assert_eq!((packet.data[0] >> 3) & 0x0f, 1);
            let frame_header = packet
                .data
                .iter()
                .position(|byte| (byte >> 3) & 0x0f == 6)
                .unwrap();
            assert!(frame_header > 0);
        }
    }

    #[test]
    fn enforces_limits_and_consecutive_indexes() {
        let factory = native_av1_video_encoder_factory();
        let limits = Limits {
            max_width: 16,
            ..Limits::default()
        };
        let error = match factory.create(&configuration(), &limits) {
            Ok(_) => panic!("oversized AV1 configuration unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);

        let mut encoder = factory
            .create(&configuration(), &Limits::default())
            .unwrap();
        let source = frame(10);
        let error = block_on(encoder.encode(
            FrameIndex(1),
            FrameSource::Cpu(CpuFrameSource {
                frame: &source,
                orientation: Orientation::TopLeft,
            }),
        ))
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}

// Gated off wasm32: every test here round-trips through the in-tree AV1 decoder, whose factory
// is itself native-only.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod nonlossless_tests {
    use super::*;
    use crate::av1_intra::Av1TxType;
    use crate::{
        CancellationToken, CpuFrameSource, EncodedVideoSample, Plane, VideoDecoderConfig,
        VideoDecoderFactory, VideoFrame, native_av1_video_decoder_factory,
    };
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let mut context = Context::from_waker(Waker::noop());
        let mut future = pin!(future);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
        }
    }

    fn test_pattern(width: u32, height: u32) -> Vec<u8> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    // A flat left half and a busy right half, so one frame exercises both the
                    // large-transform and the split/small-transform sides of the search.
                    if x < width / 2 {
                        (60 + y / 8) as u8
                    } else {
                        ((x * 7 + y * 29) ^ (x * y)) as u8
                    }
                })
            })
            .collect()
    }

    /// Deterministic 32-bit LCG, so a generated frame is the same on every host and run.
    fn lcg(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *state
    }

    /// Frames whose content statistics differ sharply from [`test_pattern`], which is the single
    /// frame the transform-gain sampling interval was originally calibrated on.
    ///
    /// The point of the set is *variation*, not difficulty: the sampled estimator reuses one
    /// frame's accumulated per-size gain ratio across the blocks it does not probe, so it is
    /// safe exactly while that ratio is stable over a frame. `quadrants` and `scene_edge` are
    /// the frames where it is not - their halves have deliberately unrelated statistics - and
    /// `noise`, `smooth` and `diagonals` pin the stationary extremes on either side of
    /// `test_pattern`.
    fn content_frames(width: u32, height: u32) -> Vec<(&'static str, Vec<u8>)> {
        let pixel = |f: &dyn Fn(u32, u32) -> u8| -> Vec<u8> {
            (0..height)
                .flat_map(|y| (0..width).map(move |x| f(x, y)).collect::<Vec<_>>())
                .collect()
        };
        // Full-range noise: no correlation for any transform to compact, so every size's type
        // gain is near zero and the ratio is measured off very small differences.
        let mut state = 0x1234_5678_u32;
        let noise: Vec<u8> = (0..width * height)
            .map(|_| (lcg(&mut state) >> 24) as u8)
            .collect();
        // Smooth: a slowly varying surface the largest transforms code almost for free.
        let smooth = pixel(&|x, y| (110 + (x * 3 + y * 2) / 5 % 40) as u8);
        // Diagonals: strongly directional edges, which is where the non-DCT types in the set
        // earn their keep and the gain ratio is at its largest.
        let diagonals = pixel(&|x, y| if (x + y) % 12 < 6 { 30 } else { 220 });
        // Quadrants: four unrelated statistics in one frame, so the frame-wide accumulated ratio
        // is never representative of the quadrant a block is in.
        let mut quad_state = 0x9E37_79B9_u32;
        let quadrants: Vec<u8> = (0..height)
            .flat_map(|y| {
                let mut row = Vec::with_capacity(width as usize);
                for x in 0..width {
                    let left = x < width / 2;
                    let top = y < height / 2;
                    row.push(match (top, left) {
                        (true, true) => 128, // flat
                        (true, false) => {
                            if x % 8 < 4 {
                                40
                            } else {
                                200
                            }
                        } // vertical stripes
                        (false, true) => (16 + (x + y) % 224) as u8, // ramp
                        (false, false) => (lcg(&mut quad_state) >> 24) as u8, // noise
                    });
                }
                row
            })
            .collect();
        // Scene edge: a low-contrast textured region meeting a high-frequency synthetic one at a
        // hard horizontal boundary, the case a frame-wide ratio is least able to represent.
        let scene_edge = pixel(&|x, y| {
            if y < height * 3 / 5 {
                (100 + ((x * 5 + y * 3) % 17)) as u8
            } else {
                (((x / 2) ^ (y / 2)) % 2 * 255) as u8
            }
        });
        vec![
            ("noise", noise),
            ("smooth", smooth),
            ("diagonals", diagonals),
            ("quadrants", quadrants),
            ("scene_edge", scene_edge),
        ]
    }

    fn encode(width: u32, height: u32, qindex: u8, pixels: &[u8]) -> Vec<u8> {
        let limits = Limits::default();
        let dimensions = VideoDimensions::new(width, height, &limits).unwrap();
        let configuration = VideoEncoderConfig {
            coded_dimensions: dimensions,
            configuration: if qindex == 0 {
                Vec::new()
            } else {
                vec![qindex]
            },
            ..super::tests::configuration()
        };
        let factory = native_av1_video_encoder_factory();
        let mut encoder = factory.create(&configuration, &limits).unwrap();
        let frame = VideoFrame::new(
            dimensions,
            PixelFormat::Gray8,
            ColorRange::Full,
            vec![Plane {
                data: pixels.to_vec(),
                stride: width as usize,
            }],
            &limits,
        )
        .unwrap();
        block_on(encoder.encode(
            FrameIndex(0),
            FrameSource::Cpu(CpuFrameSource {
                frame: &frame,
                orientation: Orientation::TopLeft,
            }),
        ))
        .unwrap()
        .remove(0)
        .data
    }

    /// Decodes one access unit with the crate's own AV1 decoder and returns its luma plane.
    /// The decoder emits RGBA and the stream is monochrome with identity matrix coefficients, so
    /// every luma sample is its pixel's R byte.
    fn decode_luma(data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let limits = Limits::default();
        let dimensions = VideoDimensions::new(width, height, &limits).unwrap();
        let encoder = native_av1_video_encoder_factory()
            .create(
                &VideoEncoderConfig {
                    coded_dimensions: dimensions,
                    ..super::tests::configuration()
                },
                &limits,
            )
            .unwrap();
        let mut decoder = native_av1_video_decoder_factory()
            .create(
                &VideoDecoderConfig {
                    codec: Codec::Av1,
                    profile: CodecProfile::Av1Main,
                    coded_dimensions: dimensions,
                    output_format: PixelFormat::Rgba8,
                    color_range: ColorRange::Full,
                    hardware: HardwarePreference::Avoid,
                    configuration: encoder.config().decoder_config.clone(),
                },
                &limits,
            )
            .unwrap();
        let cancellation = CancellationToken::new();
        let mut frames = decoder
            .submit(
                &EncodedVideoSample {
                    presentation_index: FrameIndex(0),
                    random_access: true,
                    data: data.to_vec(),
                },
                &cancellation,
            )
            .unwrap();
        frames.extend(decoder.drain(&cancellation).unwrap());
        assert_eq!(frames.len(), 1);
        let frame = &frames[0].frame;
        let plane = &frame.planes[0];
        (0..height as usize)
            .flat_map(|row| {
                let start = row * plane.stride;
                plane.data[start..start + width as usize * 4]
                    .iter()
                    .step_by(4)
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn psnr(source: &[u8], decoded: &[u8]) -> f64 {
        let sse: f64 = source
            .iter()
            .zip(decoded)
            .map(|(&a, &b)| {
                let error = f64::from(i32::from(a) - i32::from(b));
                error * error
            })
            .sum();
        if sse == 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0 * 255.0 * source.len() as f64 / sse).log10()
    }

    /// The `(size, tx_type)` pairs the tile encoder wrote for `pixels`, checked against the
    /// access unit the public encoder produced for the same frame so the trace describes the
    /// stream that actually round-trips rather than a re-encode of it.
    fn traced_transform_blocks(
        width: u32,
        height: u32,
        qindex: u8,
        pixels: &[u8],
        access_unit: &[u8],
    ) -> Vec<(usize, Av1TxType)> {
        let (tile, emitted) =
            tile::FrameEncoder::new(pixels, width as usize, height as usize, qindex)
                .encode_with_trace();
        assert!(
            access_unit.ends_with(&tile),
            "the traced tile is not the tile the encoder emitted"
        );
        emitted
    }

    /// Every `(size, tx_type)` combination the encoder is able to write, derived from the same
    /// §5.11.47 set derivation the encoder and [`crate::av1_intra_decoder`] both use, under the
    /// `reduced_tx_set = 1` the frame header signals. `TX_64X64` has no forward kernel, so the
    /// sizes stop at 32.
    fn emittable() -> std::collections::BTreeSet<(usize, String)> {
        [4_usize, 8, 16, 32]
            .into_iter()
            .flat_map(|size| {
                let set = crate::av1_cdf::get_tx_set(size, false, true);
                crate::av1_cdf::tx_type_inverse_set(set)
                    .iter()
                    .filter_map(move |&(_, tx_type)| Some((size, format!("{:?}", tx_type?))))
            })
            .collect()
    }

    /// The quantizers the determinism tests sweep. The same six the round-trip bound uses, so
    /// the reproducibility assertions cover exactly the operating points that test measures.
    const DETERMINISM_QINDEXES: [u8; 6] = [1, 8, 32, 80, 160, 200];

    /// FNV-1a over an access unit, so a divergence is reported as one digest rather than as a
    /// diff of several thousand bytes.
    fn digest(data: &[u8]) -> u64 {
        data.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Everything one encode of `pixels` at `qindex` decides, as one comparable value: the
    /// access unit bytes, and the `(size, tx_type)` trace of both the shipped search and the
    /// exhaustive one it stands in for. The traces are where the transform-type divergence was
    /// observed, and the bytes are where a divergence anywhere else in the encoder would show up.
    #[derive(PartialEq, Eq)]
    struct EncodeFingerprint {
        access_unit: Vec<u8>,
        shipped: Vec<(usize, Av1TxType)>,
        exhaustive: Vec<(usize, Av1TxType)>,
    }

    fn encode_fingerprint(width: u32, height: u32, qindex: u8, pixels: &[u8]) -> EncodeFingerprint {
        let access_unit = encode(width, height, qindex, pixels);
        let shipped = traced_transform_blocks(width, height, qindex, pixels, &access_unit);
        let exhaustive = tile::FrameEncoder::new(pixels, width as usize, height as usize, qindex)
            .without_search_shortcuts()
            .encode_with_report()
            .trace;
        EncodeFingerprint {
            access_unit,
            shipped,
            exhaustive,
        }
    }

    /// An encoder has to produce one bitstream for one input, and the rate-distortion search that
    /// picks each block's transform size and type is decided by sums of squared errors that a
    /// single-coefficient difference can reorder. Every one of those coefficients comes from
    /// [`transform::forward_transform`] or [`crate::av1_intra::inverse_transform`], both of which
    /// dispatch on the host's instruction set, so "the vector kernels are bit-exact with scalar"
    /// is what makes the encoder reproducible - and until now nothing asserted it end to end.
    ///
    /// The crate-wide [`crate::simd`] override makes every instruction set the host offers
    /// runnable in one process, so this pins each in turn and asserts the whole encode is
    /// identical. It covers the dispatch as well as the kernels: an instruction set that fell
    /// back to a different path, or a size the vector driver declined on one host and not
    /// another, would show up here too.
    #[test]
    fn one_frame_encodes_identically_under_every_instruction_set() {
        let _guard = crate::simd::test_lock();
        let (width, height) = (96_u32, 80_u32);
        let pixels = test_pattern(width, height);
        for qindex in DETERMINISM_QINDEXES {
            let mut reference: Option<(crate::simd::SimdIsa, EncodeFingerprint)> = None;
            for isa in crate::simd::available() {
                crate::simd::set_override(Some(isa));
                assert_eq!(
                    crate::simd::active(),
                    isa,
                    "the override should pin {isa:?}"
                );
                let fingerprint = encode_fingerprint(width, height, qindex, &pixels);
                match &reference {
                    None => reference = Some((isa, fingerprint)),
                    Some((first, expected)) => {
                        assert!(
                            fingerprint.access_unit == expected.access_unit,
                            "qindex {qindex}: {isa:?} encoded {} bytes (digest {:016x}) against \
                             {first:?}'s {} bytes (digest {:016x})",
                            fingerprint.access_unit.len(),
                            digest(&fingerprint.access_unit),
                            expected.access_unit.len(),
                            digest(&expected.access_unit)
                        );
                        assert_eq!(
                            fingerprint.shipped, expected.shipped,
                            "qindex {qindex}: the shipped search picked different transform \
                             blocks under {isa:?} than under {first:?}"
                        );
                        assert_eq!(
                            fingerprint.exhaustive, expected.exhaustive,
                            "qindex {qindex}: the exhaustive search picked different transform \
                             blocks under {isa:?} than under {first:?}"
                        );
                    }
                }
            }
            crate::simd::set_override(None);
        }
    }

    /// The digests [`a_fixed_frame_encodes_to_the_same_bytes_on_every_host`] pins, one per entry
    /// of [`DETERMINISM_QINDEXES`], for the 96x80 [`test_pattern`] frame.
    ///
    /// These are constants of the format, not of the machine that produced them: regenerate them
    /// only alongside a deliberate change to what the encoder emits, never to make a host pass.
    const FIXED_FRAME_DIGESTS: [(usize, u64); 6] = [
        (5417, 0x7b83_6e95_6bad_aae1),
        (4298, 0xa4c4_6a5e_8ef1_57c0),
        (3022, 0xba87_1a3d_44c1_4f33),
        (2238, 0xc29d_7d70_a472_3158),
        (1264, 0xc8cd_fcf5_8882_a86c),
        (752, 0x73f1_c047_f3cf_211f),
    ];

    /// [`one_frame_encodes_identically_under_every_instruction_set`] can only compare the
    /// instruction sets *one* host offers, and the divergence this pair of tests exists for was
    /// between hosts: the same commit and the same frame selected `IDTX` on one machine and not
    /// on three CI runners. A per-host loop cannot see that. A committed digest can, because
    /// every runner has to reproduce the same bytes.
    ///
    /// This is the assertion that makes "reproducible bitstream" a property of the encoder rather
    /// than of the machine it was built on.
    #[test]
    fn a_fixed_frame_encodes_to_the_same_bytes_on_every_host() {
        let (width, height) = (96_u32, 80_u32);
        let pixels = test_pattern(width, height);
        for (qindex, (length, expected)) in
            DETERMINISM_QINDEXES.into_iter().zip(FIXED_FRAME_DIGESTS)
        {
            let data = encode(width, height, qindex, &pixels);
            assert_eq!(
                (data.len(), digest(&data)),
                (length, expected),
                "qindex {qindex} encoded to {} bytes with digest {:016x}, which is not what this \
                 frame encodes to on other hosts",
                data.len(),
                digest(&data)
            );
        }
    }

    #[test]
    fn non_lossless_frames_round_trip_within_a_distortion_bound() {
        let (width, height) = (96_u32, 80_u32);
        let pixels = test_pattern(width, height);
        let mut covered = std::collections::BTreeSet::new();
        let mut previous: Option<(f64, usize)> = None;
        // Measured floors with margin: the point is that a coarser quantizer keeps costing fewer
        // bits and reconstructing worse, never that a particular decibel is hit exactly.
        for (qindex, floor) in [
            (1_u8, 48.0),
            (8, 46.0),
            (32, 40.0),
            (80, 33.0),
            (160, 23.0),
            (200, 17.0),
        ] {
            let data = encode(width, height, qindex, &pixels);
            let decoded = decode_luma(&data, width, height);
            assert_eq!(decoded.len(), pixels.len());
            let measured = psnr(&pixels, &decoded);
            assert!(
                measured >= floor,
                "qindex {qindex} reconstructed at {measured:.2} dB, below the {floor} dB bound"
            );
            if let Some((previous_psnr, previous_bytes)) = previous {
                assert!(
                    measured < previous_psnr && data.len() < previous_bytes,
                    "qindex {qindex} ({measured:.2} dB, {} bytes) did not trade quality for size \
                     against the finer quantizer before it ({previous_psnr:.2} dB, \
                     {previous_bytes} bytes)",
                    data.len()
                );
            }
            previous = Some((measured, data.len()));
            covered.extend(
                traced_transform_blocks(width, height, qindex, &pixels, &data)
                    .into_iter()
                    .map(|(size, tx_type)| (size, format!("{tx_type:?}"))),
            );
        }
        // Which *pair* the rate-distortion search picks is a property of the test pattern, not of
        // the encoder, so the assertion is that nothing outside the signallable set is ever
        // written and that every size and every type the set names is exercised by some block.
        let emittable = emittable();
        assert!(
            covered.is_subset(&emittable),
            "the encoder wrote a transform the decoder cannot read back: {:?}",
            &covered - &emittable
        );
        let sizes = |set: &std::collections::BTreeSet<(usize, String)>| {
            set.iter()
                .map(|(size, _)| *size)
                .collect::<std::collections::BTreeSet<_>>()
        };
        let types = |set: &std::collections::BTreeSet<(usize, String)>| {
            set.iter()
                .map(|(_, tx_type)| tx_type.clone())
                .collect::<std::collections::BTreeSet<_>>()
        };
        // Every size the decoder can read back is selected by the shipped search on this
        // pattern, including `TX_4X4`: the size trials correct for what the transform-type
        // search is worth at each size (see `tile.rs`), so the smallest size keeps the advantage
        // its sixteen-chances-to-one type search gives it. Which *type* each block ends up with
        // is still not asserted here - the DCT-only ranking leaves the wins that decide IDTX
        // close enough together that they can fall either way - so type coverage runs against
        // the exhaustive search the shortcut stands in for, and the shipped one keeps the
        // assertion that it writes nothing outside the set.
        assert_eq!(
            sizes(&covered),
            sizes(&emittable),
            "the shipped search did not select every signallable transform size"
        );
        let exhaustively_covered: std::collections::BTreeSet<(usize, String)> =
            [1_u8, 8, 32, 80, 160, 200]
                .into_iter()
                .flat_map(|qindex| {
                    tile::FrameEncoder::new(&pixels, width as usize, height as usize, qindex)
                        .without_search_shortcuts()
                        .encode_with_report()
                        .trace
                        .into_iter()
                        .map(|(size, tx_type)| (size, format!("{tx_type:?}")))
                })
                .collect();
        assert_eq!(sizes(&exhaustively_covered), sizes(&emittable));
        assert_eq!(types(&exhaustively_covered), types(&emittable));
    }

    #[test]
    #[ignore = "measurement sweep, not an assertion"]
    fn measure_type_gain_sampling_intervals() {
        let (width, height) = (192_usize, 160_usize);
        let mut frames = content_frames(width as u32, height as u32);
        frames.push(("test_pattern", test_pattern(width as u32, height as u32)));
        let quality = |report: &tile::SearchReport, pixels: &[u8]| {
            let reconstruction: Vec<u8> = (0..height)
                .flat_map(|row| report.reconstruction[row * report.coded_width..][..width].to_vec())
                .collect();
            psnr(pixels, &reconstruction)
        };
        let sse = |report: &tile::SearchReport, pixels: &[u8]| -> i64 {
            (0..height)
                .flat_map(|row| {
                    report.reconstruction[row * report.coded_width..][..width]
                        .iter()
                        .enumerate()
                        .map(move |(column, &value)| (row * width + column, value))
                })
                .map(|(index, value)| {
                    let error = i64::from(i32::from(pixels[index]) - i32::from(value));
                    error * error
                })
                .sum()
        };
        println!(
            "frame,qindex,interval,fast_psnr,exhaustive_psnr,fast_bytes,exh_bytes,candidates,lambda,fast_rd,exh_rd"
        );
        for (name, pixels) in &frames {
            for qindex in [1_u8, 8, 32, 80, 160, 200] {
                let ac = i64::from(crate::av1_intra::get_ac_quant(qindex));
                let lambda = (ac * ac / 256).max(1);
                let exhaustive = tile::FrameEncoder::new(pixels, width, height, qindex)
                    .without_search_shortcuts()
                    .encode_with_report();
                let exh_psnr = quality(&exhaustive, pixels);
                let exh_rd = sse(&exhaustive, pixels) + lambda * exhaustive.tile.len() as i64 * 8;
                for interval in [1_usize, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64] {
                    let fast = tile::FrameEncoder::new(pixels, width, height, qindex)
                        .with_type_gain_interval(interval)
                        .encode_with_report();
                    let fast_rd = sse(&fast, pixels) + lambda * fast.tile.len() as i64 * 8;
                    println!(
                        "{name},{qindex},{interval},{:.4},{exh_psnr:.4},{},{},{},{lambda},{fast_rd},{exh_rd}",
                        quality(&fast, pixels),
                        fast.tile.len(),
                        exhaustive.tile.len(),
                        fast.candidates_evaluated,
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "measurement sweep, not an assertion"]
    fn measure_type_gain_sampling_cost() {
        use std::time::Instant;
        let (width, height) = (192_usize, 160_usize);
        let mut frames = content_frames(width as u32, height as u32);
        frames.push(("test_pattern", test_pattern(width as u32, height as u32)));
        let intervals = [1_usize, 2, 4, 8, 16];
        let mut best = std::collections::BTreeMap::new();
        let mut candidates = std::collections::BTreeMap::new();
        let mut exhaustive_candidates = 0_u64;
        let mut exhaustive_best = f64::MAX;
        // Interleaved rounds with the minimum taken per arm: a single pass would attribute this
        // host's own load to whichever arm happened to run under it.
        for _ in 0..5 {
            for interval in intervals {
                let start = Instant::now();
                let mut total = 0_u64;
                for (_, pixels) in &frames {
                    for qindex in [1_u8, 8, 32, 80, 160, 200] {
                        let report = tile::FrameEncoder::new(pixels, width, height, qindex)
                            .with_type_gain_interval(interval)
                            .encode_with_report();
                        total += report.candidates_evaluated;
                    }
                }
                let elapsed = start.elapsed().as_secs_f64();
                let slot = best.entry(interval).or_insert(f64::MAX);
                *slot = slot.min(elapsed);
                candidates.insert(interval, total);
            }
            let start = Instant::now();
            let mut total = 0_u64;
            for (_, pixels) in &frames {
                for qindex in [1_u8, 8, 32, 80, 160, 200] {
                    let report = tile::FrameEncoder::new(pixels, width, height, qindex)
                        .without_search_shortcuts()
                        .encode_with_report();
                    total += report.candidates_evaluated;
                }
            }
            exhaustive_best = exhaustive_best.min(start.elapsed().as_secs_f64());
            exhaustive_candidates = total;
        }
        println!("interval,seconds,candidates");
        for interval in intervals {
            println!(
                "{interval},{:.4},{}",
                best[&interval], candidates[&interval]
            );
        }
        println!("exhaustive,{exhaustive_best:.4},{exhaustive_candidates}");
    }

    /// Summed squared error of a report's reconstruction against the source frame, over the
    /// visible `width x height` region rather than the coded one.
    fn sse_against(
        report: &tile::SearchReport,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> i64 {
        (0..height)
            .flat_map(|row| {
                report.reconstruction[row * report.coded_width..][..width]
                    .iter()
                    .enumerate()
                    .map(move |(column, &value)| (row * width + column, value))
            })
            .map(|(index, value)| {
                let error = i64::from(i32::from(pixels[index]) - i32::from(value));
                error * error
            })
            .sum()
    }

    /// Sweeps the recency window the per-size gain ratio is accumulated over.
    ///
    /// Prints, per frame and quantizer, the sampled estimator's `sse + lambda * bits` at each
    /// window against the same estimator probing every size search, which is what
    /// `TYPE_GAIN_MEMORY` is chosen from. `usize::MAX` is the frame-wide accumulation.
    #[test]
    #[ignore = "measurement sweep, not an assertion"]
    fn measure_type_gain_memory_windows() {
        for (width, height) in [(128_usize, 96_usize), (192, 160)] {
            measure_type_gain_memory_windows_at(width, height);
        }
    }

    fn measure_type_gain_memory_windows_at(width: usize, height: usize) {
        let mut frames = content_frames(width as u32, height as u32);
        frames.push(("test_pattern", test_pattern(width as u32, height as u32)));
        let sse = |report: &tile::SearchReport, pixels: &[u8]| -> i64 {
            (0..height)
                .flat_map(|row| {
                    report.reconstruction[row * report.coded_width..][..width]
                        .iter()
                        .enumerate()
                        .map(move |(column, &value)| (row * width + column, value))
                })
                .map(|(index, value)| {
                    let error = i64::from(i32::from(pixels[index]) - i32::from(value));
                    error * error
                })
                .sum()
        };
        println!("size,{width}x{height}");
        println!("frame,qindex,memory,penalty_percent,bytes,candidates");
        for (name, pixels) in &frames {
            for qindex in [1_u8, 8, 32, 80, 160, 200] {
                let ac = i64::from(crate::av1_intra::get_ac_quant(qindex));
                let lambda = (ac * ac / 256).max(1);
                let cost = |interval: usize, memory: usize| {
                    let report = tile::FrameEncoder::new(pixels, width, height, qindex)
                        .with_type_gain_interval(interval)
                        .with_type_gain_memory(memory)
                        .encode_with_report();
                    (
                        sse(&report, pixels) + lambda * report.tile.len() as i64 * 8,
                        report.tile.len(),
                        report.candidates_evaluated,
                    )
                };
                let (unsampled, _, _) = cost(1, usize::MAX);
                for memory in [1_usize, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, usize::MAX] {
                    let (sampled, bytes, candidates) =
                        cost(tile::TYPE_GAIN_SAMPLE_INTERVAL, memory);
                    let penalty = sampled as f64 / unsampled as f64 * 100.0 - 100.0;
                    println!("{name},{qindex},{memory},{penalty:+.2},{bytes},{candidates}");
                }
            }
        }
    }

    /// Sweeps where a trial that did not probe reads its gain ratio back from.
    ///
    /// [`tile::TYPE_GAIN_MEMORY`] ages the running accumulator, but the ageing is in *probe
    /// order*: the size searches are visited in superblock raster order, so a window of four
    /// probes spans a horizontal run of coding blocks and carries nothing about the block
    /// directly above. This prints, per frame and quantizer, the sampled estimator's
    /// `sse + lambda * bits` against the same estimator probing every size search, for the
    /// running accumulator alone (what #272 shipped), the per-superblock-column accumulator
    /// alone, and the two summed, which is what `GainLocality::Blended` ships.
    #[test]
    #[ignore = "measurement sweep, not an assertion"]
    fn measure_type_gain_locality() {
        for (width, height) in [(128_usize, 96_usize), (192, 160)] {
            measure_type_gain_locality_at(width, height);
        }
    }

    fn measure_type_gain_locality_at(width: usize, height: usize) {
        let mut frames = content_frames(width as u32, height as u32);
        frames.push(("test_pattern", test_pattern(width as u32, height as u32)));
        let arms = [
            ("running/weighted", tile::GainLocality::Running, tile::GainRatio::Weighted),
            ("column/weighted", tile::GainLocality::Column, tile::GainRatio::Weighted),
            ("blended/weighted", tile::GainLocality::Blended, tile::GainRatio::Weighted),
            ("running/mean", tile::GainLocality::Running, tile::GainRatio::Mean),
            ("column/mean", tile::GainLocality::Column, tile::GainRatio::Mean),
            ("blended/mean", tile::GainLocality::Blended, tile::GainRatio::Mean),
        ];
        println!("size,{width}x{height}");
        println!("frame,qindex,arm,penalty_percent,bytes,candidates");
        for (name, pixels) in &frames {
            for qindex in [1_u8, 8, 32, 80, 160, 200] {
                let ac = i64::from(crate::av1_intra::get_ac_quant(qindex));
                let lambda = (ac * ac / 256).max(1);
                let cost = |interval: usize,
                            locality: tile::GainLocality,
                            ratio: tile::GainRatio| {
                    let report = tile::FrameEncoder::new(pixels, width, height, qindex)
                        .with_type_gain_interval(interval)
                        .with_type_gain_locality(locality)
                        .with_type_gain_ratio(ratio)
                        .encode_with_report();
                    (
                        sse_against(&report, pixels, width, height)
                            + lambda * report.tile.len() as i64 * 8,
                        report.tile.len(),
                        report.candidates_evaluated,
                    )
                };
                let (unsampled, _, _) =
                    cost(1, tile::GainLocality::Running, tile::GainRatio::Weighted);
                for (label, locality, ratio) in arms {
                    let (sampled, bytes, candidates) =
                        cost(tile::TYPE_GAIN_SAMPLE_INTERVAL, locality, ratio);
                    let penalty = sampled as f64 / unsampled as f64 * 100.0 - 100.0;
                    println!("{name},{qindex},{label},{penalty:+.2},{bytes},{candidates}");
                }
            }
        }
    }

    /// Sweeps how much of a remembered gain a trial that did not probe is corrected by.
    ///
    /// Neither locality nor a steadier probe moves the residual `scene_edge` penalty at 192x160,
    /// and the frame's flipped decisions are one-directional - the sampled estimator codes the
    /// frame in smaller transforms than the unsampled one. That is what an over-large correction
    /// looks like: it scales with the trial's block count, so inflating it favours the size with
    /// more blocks. This prints the penalty against the unsampled estimator as the remembered
    /// correction is shrunk from full (`16`) to none (`0`).
    #[test]
    #[ignore = "measurement sweep, not an assertion"]
    fn measure_type_gain_trust() {
        for (width, height) in [(128_usize, 96_usize), (192, 160)] {
            let mut frames = content_frames(width as u32, height as u32);
            frames.push(("test_pattern", test_pattern(width as u32, height as u32)));
            println!("size,{width}x{height}");
            println!("frame,qindex,trust,penalty_percent,bytes,candidates");
            for (name, pixels) in &frames {
                for qindex in [1_u8, 8, 32, 80, 160, 200] {
                    let ac = i64::from(crate::av1_intra::get_ac_quant(qindex));
                    let lambda = (ac * ac / 256).max(1);
                    let cost = |interval: usize, trust: i64| {
                        let report = tile::FrameEncoder::new(pixels, width, height, qindex)
                            .with_type_gain_interval(interval)
                            .with_type_gain_trust(trust)
                            .encode_with_report();
                        (
                            sse_against(&report, pixels, width, height)
                                + lambda * report.tile.len() as i64 * 8,
                            report.tile.len(),
                            report.candidates_evaluated,
                        )
                    };
                    let (unsampled, _, _) = cost(1, 16);
                    for trust in [0_i64, 1, 2, 3, 4, 5, 6, 8, 12, 16] {
                        let (sampled, bytes, candidates) =
                            cost(tile::TYPE_GAIN_SAMPLE_INTERVAL, trust);
                        let penalty = sampled as f64 / unsampled as f64 * 100.0 - 100.0;
                        println!("{name},{qindex},{trust},{penalty:+.2},{bytes},{candidates}");
                    }
                }
            }
        }
    }

    /// What the locality blend costs in encode time, against each accumulator on its own.
    ///
    /// Interleaved rounds with the minimum taken per arm: a single pass would attribute this
    /// host's own load to whichever arm happened to run under it.
    #[test]
    #[ignore = "measurement sweep, not an assertion"]
    fn measure_type_gain_locality_cost() {
        use std::time::Instant;
        let (width, height) = (192_usize, 160_usize);
        let mut frames = content_frames(width as u32, height as u32);
        frames.push(("test_pattern", test_pattern(width as u32, height as u32)));
        let arms = [
            ("running", tile::GainLocality::Running),
            ("column", tile::GainLocality::Column),
            ("blended", tile::GainLocality::Blended),
        ];
        let mut best = std::collections::BTreeMap::new();
        let mut candidates = std::collections::BTreeMap::new();
        for _ in 0..5 {
            for (label, locality) in arms {
                let start = Instant::now();
                let mut total = 0_u64;
                for (_, pixels) in &frames {
                    for qindex in [1_u8, 8, 32, 80, 160, 200] {
                        let report = tile::FrameEncoder::new(pixels, width, height, qindex)
                            .with_type_gain_locality(locality)
                            .encode_with_report();
                        total += report.candidates_evaluated;
                    }
                }
                let elapsed = start.elapsed().as_secs_f64();
                let slot = best.entry(label).or_insert(f64::MAX);
                *slot = slot.min(elapsed);
                candidates.insert(label, total);
            }
        }
        println!("locality,seconds,candidates");
        for (label, _) in arms {
            println!("{label},{:.4},{}", best[label], candidates[label]);
        }
    }

    /// What the recency weighting costs in encode time.
    ///
    /// The correction exists to be cheap, so this is the check that ageing the accumulator did
    /// not make it expensive. Interleaved rounds with the minimum taken per arm: a single pass
    /// would attribute this host's own load to whichever arm happened to run under it.
    #[test]
    #[ignore = "measurement sweep, not an assertion"]
    fn measure_type_gain_memory_cost() {
        use std::time::Instant;
        let (width, height) = (192_usize, 160_usize);
        let mut frames = content_frames(width as u32, height as u32);
        frames.push(("test_pattern", test_pattern(width as u32, height as u32)));
        let windows = [1_usize, 2, 4, 8, 32, usize::MAX];
        let mut best = std::collections::BTreeMap::new();
        let mut candidates = std::collections::BTreeMap::new();
        for _ in 0..5 {
            for memory in windows {
                let start = Instant::now();
                let mut total = 0_u64;
                for (_, pixels) in &frames {
                    for qindex in [1_u8, 8, 32, 80, 160, 200] {
                        let report = tile::FrameEncoder::new(pixels, width, height, qindex)
                            .with_type_gain_memory(memory)
                            .encode_with_report();
                        total += report.candidates_evaluated;
                    }
                }
                let elapsed = start.elapsed().as_secs_f64();
                let slot = best.entry(memory).or_insert(f64::MAX);
                *slot = slot.min(elapsed);
                candidates.insert(memory, total);
            }
        }
        println!("memory,seconds,candidates");
        for memory in windows {
            println!("{memory},{:.4},{}", best[&memory], candidates[&memory]);
        }
    }

    /// What the sampled transform-gain estimator costs on content it was not tuned on.
    ///
    /// `TYPE_GAIN_SAMPLE_INTERVAL` probes one coding block's size search in every `n` and
    /// corrects the rest from the frame's accumulated per-size ratio, which is only representative
    /// while the frame's content is. `test_pattern` at 96x80 - the frame the interval was
    /// originally chosen on - cannot see the difference: every interval from 1 to 16 lands within
    /// 0.03% of the unsampled estimator there, which is why this runs on a set of frames with
    /// deliberately unlike statistics at a size large enough for the accumulated ratio to drift.
    ///
    /// The comparison is against the *same* estimator probing every size search, not against the
    /// exhaustive search: that isolates what the sampling costs from what the other shortcuts do.
    /// Cost is the encoder's own `sse + lambda * bits` at equal quantizer, and the ceilings are
    /// the measured penalties with margin. They are not aspirations - each one is under what the
    /// previous interval of 8 measured on the same frame (`scene_edge` +55.5%, `smooth` +4.4%,
    /// `test_pattern` +2.1%), so a regression of the constant fails here on three frames rather
    /// than passing unnoticed as it did on 96x80 alone.
    ///
    /// `scene_edge`'s ceiling was 40% while the per-size ratio was accumulated over the whole
    /// frame: every interval from 2 to 4 measured the same penalty on it, because what it was
    /// paying for was the estimator mixing two regions' statistics rather than the sampling rate.
    /// `TYPE_GAIN_MEMORY` ages that accumulation so a block reads back its own neighbourhood's
    /// ratio, which brings the frame to +2.13% here and takes the ceiling to 4%.
    #[test]
    fn the_type_gain_sampling_interval_holds_on_content_it_was_not_tuned_on() {
        let (width, height) = (128_usize, 96_usize);
        let mut frames = content_frames(width as u32, height as u32);
        frames.push(("test_pattern", test_pattern(width as u32, height as u32)));
        let ceilings = std::collections::BTreeMap::from([
            ("noise", 1.0),
            ("smooth", 2.5),
            ("diagonals", 1.0),
            ("quadrants", 1.0),
            ("scene_edge", 4.0),
            ("test_pattern", 1.0),
        ]);
        let sse = |report: &tile::SearchReport, pixels: &[u8]| -> i64 {
            (0..height)
                .flat_map(|row| {
                    report.reconstruction[row * report.coded_width..][..width]
                        .iter()
                        .enumerate()
                        .map(move |(column, &value)| (row * width + column, value))
                })
                .map(|(index, value)| {
                    let error = i64::from(i32::from(pixels[index]) - i32::from(value));
                    error * error
                })
                .sum()
        };
        for (name, pixels) in &frames {
            let ceiling = ceilings[name];
            for qindex in [1_u8, 8, 32, 80, 160, 200] {
                let ac = i64::from(crate::av1_intra::get_ac_quant(qindex));
                let lambda = (ac * ac / 256).max(1);
                let cost = |interval: usize| {
                    let report = tile::FrameEncoder::new(pixels, width, height, qindex)
                        .with_type_gain_interval(interval)
                        .encode_with_report();
                    sse(&report, pixels) + lambda * report.tile.len() as i64 * 8
                };
                let sampled = cost(tile::TYPE_GAIN_SAMPLE_INTERVAL);
                let unsampled = cost(1);
                let penalty = sampled as f64 / unsampled as f64 * 100.0 - 100.0;
                assert!(
                    penalty <= ceiling,
                    "{name} at qindex {qindex} cost {sampled} against the unsampled estimator's \
                     {unsampled} ({penalty:+.2}%), past the {ceiling}% this frame is allowed"
                );
            }
        }
    }

    /// The stated bound on what the search shortcuts cost.
    ///
    /// Two of the three are exact - a block whose residual cannot pay for one coefficient, and a
    /// partition whose unsplit cost is already below the split's header charge, are decided by
    /// the cost function itself rather than by a trial. The third ranks transform sizes and
    /// partitions on the set's DCT alone, corrected by what the type search measures out to be
    /// worth on a probe block of each size, instead of on all five types everywhere - which is an
    /// approximation, so this is the assertion that says how large an approximation it is allowed
    /// to be. Both bounds were loosened for the DCT-only ranking and tightened again once the
    /// per-size correction landed: the fast search now reconstructs no *worse* than the
    /// exhaustive one at every quantizer here.
    #[test]
    fn the_search_shortcuts_stay_within_their_rate_and_distortion_bound() {
        let (width, height) = (96_usize, 80_usize);
        let pixels = test_pattern(width as u32, height as u32);
        for qindex in [1_u8, 8, 32, 80, 160, 200] {
            let fast = tile::FrameEncoder::new(&pixels, width, height, qindex).encode_with_report();
            let exhaustive = tile::FrameEncoder::new(&pixels, width, height, qindex)
                .without_search_shortcuts()
                .encode_with_report();
            let quality = |report: &tile::SearchReport| {
                let reconstruction: Vec<u8> = (0..height)
                    .flat_map(|row| {
                        report.reconstruction[row * report.coded_width..][..width].to_vec()
                    })
                    .collect();
                psnr(&pixels, &reconstruction)
            };
            let (fast_psnr, exhaustive_psnr) = (quality(&fast), quality(&exhaustive));
            assert!(
                fast_psnr >= exhaustive_psnr - 0.05,
                "qindex {qindex} reconstructed at {fast_psnr:.3} dB against the exhaustive \
                 search's {exhaustive_psnr:.3} dB"
            );
            let growth = fast.tile.len() as f64 / exhaustive.tile.len() as f64 - 1.0;
            assert!(
                growth <= 0.015,
                "qindex {qindex} spent {} bytes against the exhaustive search's {} ({:+.2}%)",
                fast.tile.len(),
                exhaustive.tile.len(),
                growth * 100.0
            );
            assert!(
                fast.candidates_evaluated * 4 < exhaustive.candidates_evaluated,
                "qindex {qindex} evaluated {} transform-type candidates against the exhaustive \
                 search's {}, which is not the reduction the shortcuts exist for",
                fast.candidates_evaluated,
                exhaustive.candidates_evaluated
            );
        }
    }

    /// `TX_4X4` is a size the shipped search *selects*, not just one the emitting path could
    /// write.
    ///
    /// Ranking size trials on the set's DCT alone made the smallest size unreachable: coding a
    /// block as sixteen 4x4 transforms is worth its extra header bits because the type search
    /// gets sixteen chances to beat DCT instead of one, and a DCT-only ranking cannot see that.
    /// The per-size correction in `tile.rs` measures the gain on a probe block and extrapolates
    /// it over the trial, so the advantage scales with block count again.
    #[test]
    fn the_size_search_selects_the_smallest_transform() {
        let (width, height) = (96_usize, 80_usize);
        let pixels = test_pattern(width as u32, height as u32);
        let mut selected_at = Vec::new();
        for qindex in [1_u8, 8, 32, 80, 160, 200] {
            let report =
                tile::FrameEncoder::new(&pixels, width, height, qindex).encode_with_report();
            if report.trace.iter().any(|&(size, _)| size == 4) {
                selected_at.push(qindex);
            }
        }
        assert!(
            !selected_at.is_empty(),
            "no quantizer selected TX_4X4, so the smallest transform's emit path gets no \
             coverage from a normal encode"
        );
    }

    /// A frame flat enough that no transform can pay for a single coefficient is coded without
    /// running any transform at all: the residual is dropped, every block is skipped, and the
    /// decoder reconstructs the DC prediction it predicted from.
    #[test]
    fn a_flat_frame_skips_the_transform_search_entirely() {
        let (width, height) = (64_usize, 64_usize);
        let pixels = vec![128_u8; width * height];
        let report = tile::FrameEncoder::new(&pixels, width, height, 160).encode_with_report();
        assert_eq!(
            report.candidates_evaluated, 0,
            "a flat frame still evaluated {} transform-type candidates",
            report.candidates_evaluated
        );
        let data = encode(width as u32, height as u32, 160, &pixels);
        assert_eq!(decode_luma(&data, width as u32, height as u32), pixels);
    }

    #[test]
    fn lossless_output_is_unchanged_by_the_quantizer_plumbing() {
        let (width, height) = (33_u32, 17_u32);
        let pixels = test_pattern(width, height);
        let data = encode(width, height, 0, &pixels);
        assert_eq!(decode_luma(&data, width, height), pixels);
    }

    #[test]
    fn the_configuration_surface_is_empty_or_one_byte() {
        assert_eq!(parse_base_q_idx(&[]), Some(0));
        assert_eq!(parse_base_q_idx(&[40]), Some(40));
        assert_eq!(parse_base_q_idx(&[1, 2]), None);
        let factory = native_av1_video_encoder_factory();
        let mut invalid = super::tests::configuration();
        invalid.configuration = vec![1, 2, 3];
        assert!(matches!(
            factory.capability(&invalid),
            CodecSupport::InvalidConfiguration { .. }
        ));
    }
}
