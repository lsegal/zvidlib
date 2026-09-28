//! Native, dependency-free AV1 software decoding, registered as a
//! [`VideoDecoderFactory`].
//!
//! This wraps the crate's AV1 Main-profile decoder (`crate::av1_dec`), which
//! implements the complete decoding process of the AV1 specification for
//! 8-bit 4:2:0 colour and monochrome streams: every intra and inter
//! prediction tool, CDF adaptation, tiles, and the deblocking, CDEF,
//! super-resolution, loop restoration and film grain stages.
//!
//! Decoded pictures are converted to `Rgba8` with [`convert_to_rgba8`]. Colour
//! pictures use the matrix their sequence header signals (BT.601 when it is
//! unspecified or one the conversion does not implement); monochrome pictures
//! carry neutral chroma, for which the matrix choice is a no-op on the
//! resulting R=G=B luma value for every supported matrix except `Identity`, so
//! they are converted with `Bt601` whatever they signal.

use crate::av1_dec::{DecodedPicture, Decoder};
use crate::{
    Av1CodecConfigurationRecord, Av1SyntaxSupport, CancellationToken, Codec, CodecImplementation,
    CodecProfile, CodecSupport, ColorRange, DecodedVideoFrame, EncodedVideoSample, Error,
    ErrorKind, FilterFrame, FilterPlane, HardwarePreference, Limits, MatrixCoefficients,
    PixelFormat, Result, VideoDecoder, VideoDecoderConfig, VideoDecoderFactory, VideoFrame,
    convert_to_rgba8,
};

/// Returns the dependency-free native AV1 Main-profile (8-bit 4:2:0 and
/// monochrome) software decoder backend.
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
        if limits.max_av1_blocks_per_frame == 0 {
            return Err(limit("AV1 reconstruction block limit must be nonzero"));
        }
        Ok(Box::new(Av1Decoder {
            configuration: configuration.clone(),
            limits: *limits,
            inner: Decoder::new(*limits),
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
        let four_two_zero = record.chroma_subsampling_x && record.chroma_subsampling_y;
        if record.support != Av1SyntaxSupport::MainProfile
            || record.high_bitdepth
            || record.twelve_bit
            || !(record.monochrome || four_two_zero)
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "native AV1 decoder requires an 8-bit 4:2:0 or monochrome Main-profile av1C configuration",
            ));
        }
        Ok(Self)
    }
}

struct Av1Decoder {
    configuration: VideoDecoderConfig,
    limits: Limits,
    inner: Decoder,
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
        // A temporal unit shows exactly one frame; with spatial layers the
        // last picture shown is the highest layer, the one to present.
        let Some(picture) = self.inner.decode_temporal_unit(&sample.data)?.pop() else {
            return Ok(Vec::new());
        };
        let frame = picture_to_rgba(&picture, &self.configuration, &self.limits)?;
        Ok(vec![DecodedVideoFrame {
            presentation_index: sample.presentation_index,
            frame,
        }])
    }

    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
        check_cancelled(cancellation)?;
        // Every temporal unit this decoder accepts makes its frame visible
        // immediately (`show_frame` or `show_existing_frame`); there is no
        // delayed/reordered output to release on drain.
        Ok(Vec::new())
    }

    fn reset(&mut self) -> Result<()> {
        self.inner.reset();
        Ok(())
    }

    fn set_output_wanted(&mut self, wanted: bool) {
        // The frame is decoded and stays a reference for the ones after it;
        // only the output (film grain and the YUV-to-RGBA pass) is skipped,
        // which is what a caller walking to a later frame is paying for and
        // never looks at.
        self.inner.set_output_wanted(wanted);
    }
}

fn picture_to_rgba(
    picture: &DecodedPicture,
    configuration: &VideoDecoderConfig,
    limits: &Limits,
) -> Result<VideoFrame> {
    let width = picture.width;
    let height = picture.height;
    if width as u64 != u64::from(configuration.coded_dimensions.width)
        || height as u64 != u64::from(configuration.coded_dimensions.height)
    {
        return Err(malformed(
            "decoded AV1 frame does not match its display dimensions",
        ));
    }
    let y = FilterPlane::from_samples(width, height, picture.planes[0].clone(), limits)
        .map_err(|error| malformed(format!("invalid AV1 decoded plane: {error}")))?;
    let (filter_frame, matrix) = if picture.monochrome {
        (FilterFrame::new_monochrome(y), MatrixCoefficients::Bt601)
    } else {
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let u = FilterPlane::from_samples(
            chroma_width,
            chroma_height,
            picture.planes[1].clone(),
            limits,
        )
        .map_err(|error| malformed(format!("invalid AV1 decoded plane: {error}")))?;
        let v = FilterPlane::from_samples(
            chroma_width,
            chroma_height,
            picture.planes[2].clone(),
            limits,
        )
        .map_err(|error| malformed(format!("invalid AV1 decoded plane: {error}")))?;
        let frame = FilterFrame::new_yuv(y, u, v, picture.subsampling_x, picture.subsampling_y)
            .map_err(|error| malformed(format!("invalid AV1 decoded plane layout: {error}")))?;
        let matrix = match MatrixCoefficients::from_raw(picture.matrix_coefficients) {
            Ok(MatrixCoefficients::Identity) | Err(_) => MatrixCoefficients::Bt601,
            Ok(matrix) => matrix,
        };
        (frame, matrix)
    };
    let color_range = if picture.full_range {
        ColorRange::Full
    } else {
        ColorRange::Limited
    };
    let rgba = convert_to_rgba8(&filter_frame, color_range, matrix, limits)?;
    let stride = width
        .checked_mul(4)
        .ok_or_else(|| limit("AV1 RGBA stride overflows"))?;
    VideoFrame::new(
        configuration.coded_dimensions,
        PixelFormat::Rgba8,
        color_range,
        vec![crate::Plane { data: rgba, stride }],
        limits,
    )
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
    use crate::VideoDimensions;

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

        // Colour 4:2:0 av1C: monochrome bit cleared, both subsampling bits set.
        candidate = config();
        candidate.configuration = av1c(&[0x81, 0x00, 0x0C, 0x00]);
        assert!(factory.capability(&candidate).is_supported());

        // 10-bit (high_bitdepth) Main-profile av1C.
        candidate = config();
        candidate.configuration = av1c(&[0x81, 0x00, 0x4C, 0x00]);
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
