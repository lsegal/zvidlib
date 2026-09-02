//! Native, dependency-free AV1 software decoding, registered as a
//! [`VideoDecoderFactory`].
//!
//! This wraps [`Av1InterDecoder`], which itself only decodes the bounded
//! Main-profile 8-bit lossless monochrome subset documented on that type: a
//! single tile, static (non-adapted) CDFs, order-hint-based reference
//! selection, whole-pel `NEARESTMV`/`GLOBALMV`/bounded `NEWMV` inter
//! prediction plus average LAST/LAST2 compound prediction, and validated
//! `show_existing_frame` output. `CodedLossless` streams never signal
//! deblocking, CDEF, loop restoration, super-resolution, or film grain (AV1
//! spec §5.9.11), so this decoder never needs to invoke the in-loop filter
//! stages in [`crate::av1_filters`] beyond the final, always-applicable
//! output color conversion.
//!
//! Every reconstructed frame is monochrome (neutral, mid-gray chroma), so
//! [`convert_to_rgba8`] is called with an arbitrary non-identity matrix
//! (`Bt601`): with neutral chroma the matrix choice is a no-op on the
//! resulting R=G=B luma value for every supported matrix except `Identity`.

use crate::{
    Av1CodecConfigurationRecord, Av1InterDecoder, Av1SyntaxSupport, CancellationToken, Codec,
    CodecImplementation, CodecProfile, CodecSupport, DecodedVideoFrame, EncodedVideoSample, Error,
    ErrorKind, FilterFrame, FilterPlane, HardwarePreference, Limits, MatrixCoefficients,
    PixelFormat, Result, VideoDecoder, VideoDecoderConfig, VideoDecoderFactory, VideoFrame,
    convert_to_rgba8,
};

/// Returns the dependency-free native AV1 Main lossless monochrome software
/// decoder backend.
pub fn native_av1_video_decoder_factory() -> impl VideoDecoderFactory {
    Av1DecoderFactory
}

#[derive(Clone, Copy, Debug)]
struct Av1DecoderFactory;

impl VideoDecoderFactory for Av1DecoderFactory {
    fn capability(&self, configuration: &VideoDecoderConfig) -> CodecSupport {
        match self.capability_without_parsing(configuration) {
            CodecSupport::Supported { .. } => {}
            other => return other,
        }
        if let Err(error) = ParsedConfiguration::parse(configuration, &Limits::default()) {
            return invalid_configuration(error.message());
        }
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        }
    }

    fn create(
        &self,
        configuration: &VideoDecoderConfig,
        limits: &Limits,
    ) -> Result<Box<dyn VideoDecoder>> {
        match self.capability_without_parsing(configuration) {
            CodecSupport::Supported { .. } => {}
            CodecSupport::UnsupportedCodec => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native AV1 decoder requires AV1",
                ));
            }
            CodecSupport::UnsupportedProfile => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native AV1 decoder supports the Main profile",
                ));
            }
            CodecSupport::HardwareUnavailable => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native AV1 decoder is software-only",
                ));
            }
            CodecSupport::InvalidConfiguration { reason } => {
                return Err(Error::new(ErrorKind::InvalidInput, reason));
            }
        }
        ParsedConfiguration::parse(configuration, limits)?;
        Ok(Box::new(Av1Decoder {
            configuration: configuration.clone(),
            limits: *limits,
            inner: Av1InterDecoder::new(*limits)?,
            output_wanted: true,
        }))
    }
}

impl Av1DecoderFactory {
    fn capability_without_parsing(&self, configuration: &VideoDecoderConfig) -> CodecSupport {
        if configuration.codec != Codec::Av1 {
            return CodecSupport::UnsupportedCodec;
        }
        if configuration.profile != CodecProfile::Av1Main {
            return CodecSupport::UnsupportedProfile;
        }
        if configuration.hardware == HardwarePreference::Require {
            return CodecSupport::HardwareUnavailable;
        }
        if configuration.output_format != PixelFormat::Rgba8 {
            return invalid_configuration("native AV1 Main decoding currently outputs RGBA8");
        }
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        }
    }
}

fn invalid_configuration(reason: impl Into<String>) -> CodecSupport {
    CodecSupport::InvalidConfiguration {
        reason: reason.into(),
    }
}

/// A validated `av1C` configuration record. Unlike the HEVC decoder's hvcC
/// SPS, `av1C`'s `configOBUs` are not required to carry a sequence header
/// (AV1 spec §5.9.16 note), so the decoded frame's actual coded dimensions
/// are validated against `configuration.coded_dimensions` when each frame is
/// produced instead of upfront here.
struct ParsedConfiguration;

impl ParsedConfiguration {
    fn parse(configuration: &VideoDecoderConfig, limits: &Limits) -> Result<Self> {
        if configuration.configuration.len() as u64 > limits.max_allocation_bytes {
            return Err(limit("AV1 configuration exceeds the allocation limit"));
        }
        let record = Av1CodecConfigurationRecord::parse(&configuration.configuration, limits)
            .map_err(|error| malformed(format!("invalid av1C configuration: {error}")))?;
        if record.support != Av1SyntaxSupport::MainProfile
            || record.high_bitdepth
            || record.twelve_bit
            || !record.monochrome
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "native AV1 decoder requires an 8-bit monochrome Main-profile av1C configuration",
            ));
        }
        Ok(Self)
    }
}

struct Av1Decoder {
    configuration: VideoDecoderConfig,
    limits: Limits,
    inner: Av1InterDecoder,
    /// Whether a caller wants the pictures being decoded, or is only walking past them.
    output_wanted: bool,
}

impl VideoDecoder for Av1Decoder {
    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        check_cancelled(cancellation)?;
        if sample.data.len() as u64 > self.limits.max_allocation_bytes {
            return Err(limit("AV1 temporal unit exceeds the allocation limit"));
        }
        let decoded = self.inner.decode_temporal_unit(&sample.data)?;
        if !self.output_wanted {
            // The frame is decoded and stays a reference for the ones after it; only the
            // YUV-to-RGBA pass over it is skipped, which is what a caller walking to a later
            // frame is paying for and never looks at.
            return Ok(Vec::new());
        }
        let frame = temporal_unit_to_rgba(&decoded, &self.configuration, &self.limits)?;
        Ok(vec![DecodedVideoFrame {
            presentation_index: sample.presentation_index,
            frame,
        }])
    }

    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
        check_cancelled(cancellation)?;
        // Every temporal unit this decoder accepts makes exactly one frame
        // visible immediately (`show_frame` or `show_existing_frame`); there
        // is no delayed/reordered output to release on drain.
        Ok(Vec::new())
    }

    fn reset(&mut self) -> Result<()> {
        self.inner.reset();
        Ok(())
    }

    fn set_output_wanted(&mut self, wanted: bool) {
        self.output_wanted = wanted;
    }
}

fn temporal_unit_to_rgba(
    decoded: &VideoFrame,
    configuration: &VideoDecoderConfig,
    limits: &Limits,
) -> Result<VideoFrame> {
    if decoded.pixel_format != PixelFormat::Yuv420p8 || decoded.planes.len() != 3 {
        return Err(malformed(
            "AV1 decoder produced an unexpected internal pixel format",
        ));
    }
    if decoded.dimensions != configuration.coded_dimensions {
        return Err(malformed(
            "decoded AV1 frame does not match its display dimensions",
        ));
    }
    let width = usize::try_from(decoded.dimensions.width)
        .map_err(|_| limit("AV1 decoded width is not representable"))?;
    let height = usize::try_from(decoded.dimensions.height)
        .map_err(|_| limit("AV1 decoded height is not representable"))?;
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let y = plane_to_filter_plane(&decoded.planes[0], width, height, limits)?;
    let u = plane_to_filter_plane(&decoded.planes[1], chroma_width, chroma_height, limits)?;
    let v = plane_to_filter_plane(&decoded.planes[2], chroma_width, chroma_height, limits)?;
    let filter_frame = FilterFrame::new_yuv(y, u, v, true, true)
        .map_err(|error| malformed(format!("invalid AV1 decoded plane layout: {error}")))?;
    let rgba = convert_to_rgba8(
        &filter_frame,
        decoded.color_range,
        MatrixCoefficients::Bt601,
        limits,
    )?;
    let stride = width
        .checked_mul(4)
        .ok_or_else(|| limit("AV1 RGBA stride overflows"))?;
    VideoFrame::new(
        configuration.coded_dimensions,
        PixelFormat::Rgba8,
        decoded.color_range,
        vec![crate::Plane { data: rgba, stride }],
        limits,
    )
}

fn plane_to_filter_plane(
    plane: &crate::Plane,
    width: usize,
    height: usize,
    limits: &Limits,
) -> Result<FilterPlane> {
    let mut packed = Vec::new();
    packed
        .try_reserve(width.checked_mul(height).unwrap_or(0))
        .map_err(|_| limit("AV1 plane extraction allocation failed"))?;
    for row in 0..height {
        let start = row
            .checked_mul(plane.stride)
            .ok_or_else(|| limit("AV1 plane row offset overflows"))?;
        let end = start
            .checked_add(width)
            .ok_or_else(|| limit("AV1 plane row length overflows"))?;
        let bytes = plane
            .data
            .get(start..end)
            .ok_or_else(|| malformed("AV1 decoded plane is shorter than its active rows"))?;
        packed.extend_from_slice(bytes);
    }
    FilterPlane::from_samples(width, height, packed, limits)
        .map_err(|error| malformed(format!("invalid AV1 decoded plane: {error}")))
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(Error::new(
            ErrorKind::Cancelled,
            "codec operation cancelled",
        ))
    } else {
        Ok(())
    }
}

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

fn limit(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColorRange, VideoDimensions};

    fn av1c(payload: &[u8]) -> Vec<u8> {
        let mut bytes = (8u32 + payload.len() as u32).to_be_bytes().to_vec();
        bytes.extend_from_slice(b"av1C");
        bytes.extend_from_slice(payload);
        bytes
    }

    fn monochrome_main_av1c() -> Vec<u8> {
        av1c(&[0x81, 0x00, 0x1C, 0x00])
    }

    fn config() -> VideoDecoderConfig {
        VideoDecoderConfig {
            codec: Codec::Av1,
            profile: CodecProfile::Av1Main,
            coded_dimensions: VideoDimensions::new(16, 16, &Limits::default()).unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Full,
            hardware: HardwarePreference::Avoid,
            configuration: monochrome_main_av1c(),
        }
    }

    #[test]
    fn capability_distinguishes_codec_profile_configuration_and_hardware() {
        let factory = native_av1_video_decoder_factory();
        assert!(factory.capability(&config()).is_supported());

        let mut candidate = config();
        candidate.codec = Codec::Hevc;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::UnsupportedCodec
        );

        candidate = config();
        candidate.profile = CodecProfile::HevcMain;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::UnsupportedProfile
        );

        candidate = config();
        candidate.hardware = HardwarePreference::Require;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::HardwareUnavailable
        );

        candidate = config();
        candidate.output_format = PixelFormat::Yuv420p8;
        assert!(matches!(
            factory.capability(&candidate),
            CodecSupport::InvalidConfiguration { .. }
        ));

        candidate = config();
        candidate.configuration = vec![1, 2, 3];
        assert!(matches!(
            factory.capability(&candidate),
            CodecSupport::InvalidConfiguration { .. }
        ));

        // Non-monochrome av1C: monochrome bit cleared.
        candidate = config();
        candidate.configuration = av1c(&[0x81, 0x00, 0x0C, 0x00]);
        assert!(matches!(
            factory.capability(&candidate),
            CodecSupport::InvalidConfiguration { .. }
        ));
    }

    #[test]
    fn create_rejects_truncated_configuration_without_panicking() {
        let mut candidate = config();
        candidate.configuration = vec![1, 2, 3];
        let error = native_av1_video_decoder_factory()
            .create(&candidate, &Limits::default())
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::MalformedMedia);
    }

    #[test]
    fn create_enforces_allocation_limit() {
        let restrictive = Limits {
            max_allocation_bytes: 0,
            ..Limits::default()
        };
        let error = native_av1_video_decoder_factory()
            .create(&config(), &restrictive)
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);
    }
}
