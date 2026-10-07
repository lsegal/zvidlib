//! Native VP8 decoding, registered as a [`VideoDecoderFactory`]: a
//! dependency-free software decoder, and NVIDIA NVDEC or Media Foundation
//! where the host has them.
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
//!
//! NVDEC decodes VP8 on 64-bit Windows and Linux hosts with an NVIDIA adapter
//! whose driver reports VP8 support. Where NVDEC is unavailable, Windows
//! decodes VP8 through Media Foundation on adapters that expose the D3D11 VP8
//! decoder profile (`D3D11_DECODER_PROFILE_VP8_VLD`), as Intel and some AMD
//! drivers do, with an installed D3D11-aware VP8 decoder transform such as the
//! VP9 Video Extensions'. NVIDIA drivers do not expose that profile; they
//! decode VP8 through NVDEC alone. Either backend's pictures are cropped and
//! converted to RGBA by the same code as the software decoder's, and VP8
//! decoding is exact, so all three produce the same pixels. VideoToolbox
//! exposes no VP8 decoder, so macOS decodes in software.

pub mod bench;
mod bool_decoder;
mod bool_encoder;
pub mod decode_bench;
mod decoder;
mod encoder;
mod frame_encoder;
mod loop_filter;
mod predict;
#[doc(hidden)]
pub mod simd;
mod tables;
#[cfg(test)]
mod tests;

pub use encoder::native_vp8_video_encoder_factory;

use decoder::{Decoder, Picture};

#[allow(unused_imports)]
use zvidlib_color::*;
#[allow(unused_imports)]
use zvidlib_core::*;
// The containers and conformance harness the tests read their fixtures with.
#[cfg(test)]
#[allow(unused_imports)]
use zvidlib_container::*;

use crate::{
    CancellationToken, Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange,
    DecodedVideoFrame, EncodedVideoSample, Error, ErrorKind, FilterFrame, FilterPlane,
    HardwarePreference, Limits, MatrixCoefficients, PixelFormat, Result, VideoDecoder,
    VideoDecoderConfig, VideoDecoderFactory, VideoFrame, convert_to_rgba8,
};

/// Returns the native VP8 decoder backend.
///
/// The decoder accepts [`Codec::Vp8`] with [`CodecProfile::Vp8`] and outputs
/// [`PixelFormat::Rgba8`]. VP8 carries no codec configuration record, so
/// `configuration` is ignored.
///
/// `Prefer` and `Require` select NVIDIA NVDEC on supported 64-bit Windows and
/// Linux hosts, and otherwise Media Foundation on Windows hosts whose adapter
/// exposes the D3D11 VP8 decoder profile. `Prefer` falls back to the
/// dependency-free software decoder when neither is available, and `Require`
/// reports [`CodecSupport::HardwareUnavailable`]. `Avoid` always selects
/// software. All produce the same pixels.
pub fn native_vp8_video_decoder_factory() -> impl VideoDecoderFactory {
    Vp8DecoderFactory
}

#[derive(Clone, Copy, Debug)]
struct Vp8DecoderFactory;

impl VideoDecoderFactory for Vp8DecoderFactory {
    fn capability(&self, configuration: &VideoDecoderConfig) -> CodecSupport {
        let support = self.capability_without_hardware(configuration);
        if !support.is_supported() {
            return support;
        }
        if configuration.hardware != HardwarePreference::Avoid && hardware_available(configuration)
        {
            return CodecSupport::Supported {
                implementation: CodecImplementation::Hardware,
            };
        }
        if configuration.hardware == HardwarePreference::Require {
            return CodecSupport::HardwareUnavailable;
        }
        support
    }

    fn create(
        &self,
        configuration: &VideoDecoderConfig,
        limits: &Limits,
    ) -> Result<Box<dyn VideoDecoder>> {
        match self.capability_without_hardware(configuration) {
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
            CodecSupport::HardwareUnavailable => unreachable!("hardware is not checked here"),
            CodecSupport::InvalidConfiguration { reason } => {
                return Err(Error::new(ErrorKind::InvalidInput, reason));
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native VP8 decoder does not support this configuration",
                ));
            }
        }
        if configuration.hardware != HardwarePreference::Avoid {
            #[cfg_attr(
                not(any(windows, all(target_os = "linux", target_pointer_width = "64"))),
                allow(unused_mut)
            )]
            let mut hardware_errors = Vec::<String>::new();
            #[cfg(all(any(windows, target_os = "linux"), target_pointer_width = "64"))]
            match zvidlib_hardware::nvdec::create_vp8(configuration, limits, planes_to_rgba) {
                Ok(decoder) => return Ok(decoder),
                Err(error) => hardware_errors.push(format!("NVDEC: {}", error.message())),
            }
            #[cfg(windows)]
            match zvidlib_hardware::windows_mf::create_vp8(configuration, limits, planes_to_rgba) {
                Ok(decoder) => return Ok(decoder),
                Err(error) => {
                    hardware_errors.push(format!("Media Foundation: {}", error.message()));
                }
            }
            if configuration.hardware == HardwarePreference::Require {
                let detail = if hardware_errors.is_empty() {
                    "no accelerated backend exists for this target".to_owned()
                } else {
                    hardware_errors.join("; ")
                };
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    format!("hardware VP8 decoding is unavailable ({detail})"),
                ));
            }
        }
        Ok(Box::new(Vp8Decoder {
            configuration: configuration.clone(),
            limits: *limits,
            inner: Decoder::new(*limits),
        }))
    }
}

impl Vp8DecoderFactory {
    fn capability_without_hardware(&self, configuration: &VideoDecoderConfig) -> CodecSupport {
        if configuration.codec != Codec::Vp8 {
            return CodecSupport::UnsupportedCodec;
        }
        if configuration.profile != CodecProfile::Vp8 {
            return CodecSupport::UnsupportedProfile;
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
}

fn hardware_available(_configuration: &VideoDecoderConfig) -> bool {
    #[cfg(all(any(windows, target_os = "linux"), target_pointer_width = "64"))]
    if zvidlib_hardware::nvdec::is_vp8_available(_configuration.coded_dimensions) {
        return true;
    }
    #[cfg(windows)]
    if zvidlib_hardware::windows_mf::is_vp8_available(_configuration.coded_dimensions) {
        return true;
    }
    false
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

/// Converts the cropped 4:2:0 planes a hardware decoder read back, so its pictures take exactly
/// the software decoder's conversion.
#[cfg(any(windows, all(target_os = "linux", target_pointer_width = "64")))]
pub(crate) fn planes_to_rgba(
    planes: [Vec<u8>; 3],
    configuration: &VideoDecoderConfig,
    limits: &Limits,
) -> Result<VideoFrame> {
    let dimensions = configuration.coded_dimensions;
    let picture = Picture {
        width: dimensions.width as usize,
        height: dimensions.height as usize,
        planes,
    };
    picture_to_rgba(&picture, configuration, limits)
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
