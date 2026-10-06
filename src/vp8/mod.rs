//! Native, dependency-free VP8 software decoding, registered as a
//! [`VideoDecoderFactory`].
//!
//! This is a complete implementation of the VP8 decoding process of RFC 6386
//! for every bitstream version (0 to 3): the boolean entropy decoder, key-frame
//! and inter-frame mode and motion-vector parsing, the DCT and Walsh-Hadamard
//! transforms, intra prediction, six-tap and bilinear motion compensation,
//! segmentation, the normal and simple loop filters, and last, golden and
//! alternate reference frames with their sign bias, copies and probability
//! persistence. Its output matches libvpx bit for bit on the VP8 test vectors
//! in `tests/fixtures/codec/vp8/`.
//!
//! Each [`EncodedVideoSample`] holds exactly one VP8 frame, as an IVF frame or
//! a WebM block does. A hidden frame (`show_frame` = 0, typically an
//! alternate reference the encoder codes ahead of the frames that use it) is
//! decoded and updates the references, but produces no picture, so its
//! sample returns no [`DecodedVideoFrame`].
//!
//! Decoded pictures are 4:2:0 BT.601 in the limited range, which is the only
//! colour space VP8 defines, and are converted to `Rgba8` with
//! [`convert_to_rgba8`]. Like the other software decoders, decoding runs on
//! the calling thread.

mod bool_decoder;
mod decoder;
mod loop_filter;
mod predict;
mod tables;
#[cfg(test)]
mod tests;

use decoder::{Decoder, Picture};

use crate::{
    CancellationToken, Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange,
    DecodedVideoFrame, EncodedVideoSample, Error, ErrorKind, FilterFrame, FilterPlane,
    HardwarePreference, Limits, MatrixCoefficients, PixelFormat, Result, VideoDecoder,
    VideoDecoderConfig, VideoDecoderFactory, VideoFrame, convert_to_rgba8,
};

/// Returns the dependency-free native VP8 software decoder backend.
///
/// The decoder accepts [`Codec::Vp8`] with [`CodecProfile::Vp8`] and outputs
/// [`PixelFormat::Rgba8`]. VP8 carries no codec configuration record, so
/// `configuration` is ignored. It is software-only, so a configuration that
/// requires hardware reports [`CodecSupport::HardwareUnavailable`].
pub fn native_vp8_video_decoder_factory() -> impl VideoDecoderFactory {
    Vp8DecoderFactory
}

#[derive(Clone, Copy, Debug)]
struct Vp8DecoderFactory;

impl VideoDecoderFactory for Vp8DecoderFactory {
    fn capability(&self, configuration: &VideoDecoderConfig) -> CodecSupport {
        if configuration.codec != Codec::Vp8 {
            return CodecSupport::UnsupportedCodec;
        }
        if configuration.profile != CodecProfile::Vp8 {
            return CodecSupport::UnsupportedProfile;
        }
        if configuration.hardware == HardwarePreference::Require {
            return CodecSupport::HardwareUnavailable;
        }
        if configuration.output_format != PixelFormat::Rgba8 {
            return CodecSupport::InvalidConfiguration {
                reason: "native VP8 decoding currently outputs RGBA8".into(),
            };
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
        match self.capability(configuration) {
            CodecSupport::Supported { .. } => {}
            CodecSupport::UnsupportedCodec => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native VP8 decoder requires VP8",
                ));
            }
            CodecSupport::UnsupportedProfile => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native VP8 decoder requires the VP8 profile",
                ));
            }
            CodecSupport::HardwareUnavailable => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native VP8 decoder is software-only",
                ));
            }
            CodecSupport::InvalidConfiguration { reason } => {
                return Err(Error::new(ErrorKind::InvalidInput, reason));
            }
        }
        Ok(Box::new(Vp8Decoder {
            configuration: configuration.clone(),
            limits: *limits,
            inner: Decoder::new(*limits),
        }))
    }
}

struct Vp8Decoder {
    configuration: VideoDecoderConfig,
    limits: Limits,
    inner: Decoder,
}

impl VideoDecoder for Vp8Decoder {
    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        cancellation.check()?;
        if sample.data.len() as u64 > self.limits.max_allocation_bytes {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "VP8 frame exceeds the allocation limit",
            ));
        }
        let Some(picture) = self.inner.decode(&sample.data)? else {
            // A hidden frame, or the caller asked not to see this one.
            return Ok(Vec::new());
        };
        let frame = picture_to_rgba(&picture, &self.configuration, &self.limits)?;
        Ok(vec![DecodedVideoFrame {
            presentation_index: sample.presentation_index,
            frame,
        }])
    }

    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
        cancellation.check()?;
        // VP8 has no output reordering: a shown frame is output by the
        // sample that codes it.
        Ok(Vec::new())
    }

    fn reset(&mut self) -> Result<()> {
        self.inner.reset();
        Ok(())
    }

    fn set_output_wanted(&mut self, wanted: bool) {
        // The frame is still decoded and kept as a reference; only the crop
        // and the YUV-to-RGBA conversion are skipped.
        self.inner.set_output_wanted(wanted);
    }
}

fn picture_to_rgba(
    picture: &Picture,
    configuration: &VideoDecoderConfig,
    limits: &Limits,
) -> Result<VideoFrame> {
    if picture.width as u64 != u64::from(configuration.coded_dimensions.width)
        || picture.height as u64 != u64::from(configuration.coded_dimensions.height)
    {
        return Err(Error::new(
            ErrorKind::MalformedMedia,
            "decoded VP8 frame does not match its configured dimensions",
        ));
    }
    let plane = |width: usize, height: usize, samples: &[u8]| {
        FilterPlane::from_samples(width, height, samples.to_vec(), limits).map_err(|error| {
            Error::new(
                ErrorKind::MalformedMedia,
                format!("invalid VP8 decoded plane: {error}"),
            )
        })
    };
    let chroma_width = picture.width.div_ceil(2);
    let chroma_height = picture.height.div_ceil(2);
    let frame = FilterFrame::new_yuv(
        plane(picture.width, picture.height, &picture.planes[0])?,
        plane(chroma_width, chroma_height, &picture.planes[1])?,
        plane(chroma_width, chroma_height, &picture.planes[2])?,
        true,
        true,
    )?;
    let rgba = convert_to_rgba8(
        &frame,
        ColorRange::Limited,
        MatrixCoefficients::Bt601,
        limits,
    )?;
    let stride = configuration
        .coded_dimensions
        .width
        .checked_mul(4)
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "VP8 RGBA stride overflows"))?;
    VideoFrame::new(
        configuration.coded_dimensions,
        PixelFormat::Rgba8,
        ColorRange::Limited,
        vec![crate::Plane {
            data: rgba,
            stride: stride as usize,
        }],
        limits,
    )
}
