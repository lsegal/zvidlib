//! Dependency-free native AV1 encoding.
//!
//! The first backend is deliberately narrow and honest: lossless, all-intra,
//! 8-bit monochrome AV1 Main profile. Every sample is an independent temporal
//! unit containing a sequence header and key frame, so MP4 random access does
//! not depend on encoder-private state.

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
#[allow(dead_code)]
mod wht;

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
    if !configuration.configuration.is_empty() {
        return invalid_support(
            "the native AV1 encoder does not accept backend-private configuration",
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
        let mut frame_payload = headers::frame_header_payload(
            self.format.dimensions.width,
            self.format.dimensions.height,
            u32::try_from(mi_cols)
                .map_err(|_| Error::new(ErrorKind::ResourceLimit, "AV1 MI width overflow"))?,
            u32::try_from(mi_rows)
                .map_err(|_| Error::new(ErrorKind::ResourceLimit, "AV1 MI height overflow"))?,
        );
        frame_payload.extend_from_slice(&FrameEncoder::new(&packed, width, height).encode());
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

    fn configuration() -> VideoEncoderConfig {
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
