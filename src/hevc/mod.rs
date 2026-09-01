//! Native HEVC/H.265 decoding with platform acceleration and a dependency-free software fallback.

// internal — exposed for the criterion benchmark suite; not part of the stable API
#[doc(hidden)]
pub mod bench;
pub(crate) mod color_convert;
pub mod decode_bench;
// internal — exposed for the stage-attribution example; not part of the stable API
#[doc(hidden)]
pub use engine::profile as decode_profile;
mod encoder;
pub(crate) mod engine;
#[cfg(all(any(windows, target_os = "linux"), target_pointer_width = "64"))]
mod nvdec;
// internal — exposed for the hardware benchmark suite; not part of the stable API
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub mod readback;
#[cfg(target_os = "macos")]
mod videotoolbox;
#[cfg(windows)]
mod windows_mf;

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::{
    CancellationToken, Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange,
    DecodedVideoFrame, EncodedVideoSample, Error, ErrorKind, FrameIndex, HardwarePreference,
    Limits, PixelFormat, Plane, Result, VideoDecoder, VideoDecoderConfig, VideoDecoderFactory,
    VideoFrame,
};
pub use encoder::native_hevc_video_encoder_factory;
use engine::hvcc::{HvccRecord, parse_hvcc, split_length_prefixed};
use engine::picture::{Picture, Plane as HevcPlane, sub_wh_c};
use engine::sequence::{DecodedFrame, SequenceDecoder, SequenceError};
use engine::sps::SeqParameterSet;

const DEFAULT_REORDER_DEPTH: usize = 8;

/// Returns the native HEVC Main decoder backend.
///
/// `Prefer` and `Require` select NVIDIA NVDEC on supported 64-bit Windows/Linux hosts, D3D11-aware
/// Media Foundation on Windows, or VideoToolbox on macOS. `Prefer` falls back to the dependency-free
/// software decoder when acceleration is unavailable; `Avoid` always selects software.
pub fn native_hevc_video_decoder_factory() -> impl VideoDecoderFactory {
    HevcDecoderFactory
}

#[derive(Clone, Copy, Debug)]
struct HevcDecoderFactory;

impl VideoDecoderFactory for HevcDecoderFactory {
    fn capability(&self, configuration: &VideoDecoderConfig) -> CodecSupport {
        if let unsupported @ (CodecSupport::UnsupportedCodec
        | CodecSupport::UnsupportedProfile
        | CodecSupport::InvalidConfiguration { .. }) =
            self.capability_without_hardware(configuration)
        {
            return unsupported;
        }
        if let Err(error) = ParsedConfiguration::parse(configuration, &Limits::default()) {
            return invalid_configuration(error.message());
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
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        }
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
                    "native HEVC decoder requires HEVC",
                ));
            }
            CodecSupport::UnsupportedProfile => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native HEVC decoder supports the Main profile",
                ));
            }
            CodecSupport::HardwareUnavailable => unreachable!("hardware is not checked here"),
            CodecSupport::InvalidConfiguration { reason } => {
                return Err(Error::new(ErrorKind::InvalidInput, reason));
            }
        }
        let parsed = ParsedConfiguration::parse(configuration, limits)?;
        if configuration.hardware != HardwarePreference::Avoid {
            let mut hardware_errors = Vec::new();
            #[cfg(all(any(windows, target_os = "linux"), target_pointer_width = "64"))]
            match nvdec::create(configuration, limits, &parsed.record) {
                Ok(decoder) => return Ok(decoder),
                Err(error) => hardware_errors.push(format!("NVDEC: {}", error.message())),
            }
            #[cfg(windows)]
            match windows_mf::create(configuration, limits, &parsed.record) {
                Ok(decoder) => return Ok(decoder),
                Err(error) => {
                    hardware_errors.push(format!("Media Foundation: {}", error.message()));
                }
            }
            #[cfg(target_os = "macos")]
            match videotoolbox::create(configuration, limits, &parsed.record) {
                Ok(decoder) => return Ok(decoder),
                Err(error) => {
                    hardware_errors.push(format!("VideoToolbox: {}", error.message()));
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
                    format!("hardware HEVC decoding is unavailable ({detail})"),
                ));
            }
        }
        Ok(Box::new(HevcDecoder::new(
            configuration.clone(),
            *limits,
            parsed,
        )?))
    }
}

impl HevcDecoderFactory {
    fn capability_without_hardware(&self, configuration: &VideoDecoderConfig) -> CodecSupport {
        if configuration.codec != Codec::Hevc {
            return CodecSupport::UnsupportedCodec;
        }
        if configuration.profile != CodecProfile::HevcMain {
            return CodecSupport::UnsupportedProfile;
        }
        if configuration.output_format != PixelFormat::Rgba8 {
            return invalid_configuration("native HEVC Main decoding currently outputs RGBA8");
        }
        if configuration.color_range != ColorRange::Limited {
            return invalid_configuration(
                "native HEVC Main RGBA output requires limited-range input",
            );
        }
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        }
    }
}

fn hardware_available(_configuration: &VideoDecoderConfig) -> bool {
    #[cfg(all(any(windows, target_os = "linux"), target_pointer_width = "64"))]
    if nvdec::is_available(_configuration.coded_dimensions) {
        return true;
    }
    #[cfg(windows)]
    if windows_mf::is_available(_configuration.coded_dimensions) {
        return true;
    }
    #[cfg(target_os = "macos")]
    if videotoolbox::is_available(_configuration.coded_dimensions) {
        return true;
    }
    false
}

fn invalid_configuration(reason: impl Into<String>) -> CodecSupport {
    CodecSupport::InvalidConfiguration {
        reason: reason.into(),
    }
}

#[derive(Debug)]
struct ParsedConfiguration {
    record: HvccRecord,
}

impl ParsedConfiguration {
    fn parse(configuration: &VideoDecoderConfig, limits: &Limits) -> Result<Self> {
        if configuration.configuration.len() as u64 > limits.max_allocation_bytes {
            return Err(limit("HEVC configuration exceeds the allocation limit"));
        }
        let payload = hvcc_payload(&configuration.configuration)?;
        let record = parse_hvcc(payload)
            .map_err(|error| malformed(format!("invalid hvcC configuration: {error}")))?;
        if record.general_profile_idc != 1
            || record.bit_depth_luma_minus8 != 0
            || record.bit_depth_chroma_minus8 != 0
            || record.chroma_format_idc != 1
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "native HEVC decoder requires an 8-bit 4:2:0 Main-profile hvcC configuration",
            ));
        }
        let sps_unit = record
            .nal_units
            .iter()
            .find(|unit| unit.header.nal_unit_type == 33)
            .ok_or_else(|| malformed("hvcC configuration does not contain an SPS"))?;
        let sps = SeqParameterSet::parse(&sps_unit.rbsp)
            .map_err(|error| malformed(format!("invalid HEVC SPS: {error}")))?;
        validate_sps(&sps, configuration, limits)?;
        Ok(Self { record })
    }
}

fn hvcc_payload(configuration: &[u8]) -> Result<&[u8]> {
    if configuration.len() >= 8 && &configuration[4..8] == b"hvcC" {
        let declared = u32::from_be_bytes(configuration[..4].try_into().expect("four bytes"));
        if usize::try_from(declared).ok() != Some(configuration.len()) {
            return Err(malformed(
                "hvcC box size does not match its configuration bytes",
            ));
        }
        return Ok(&configuration[8..]);
    }
    if configuration.first() == Some(&1) {
        return Ok(configuration);
    }
    Err(malformed("HEVC configuration is not an hvcC record"))
}

fn validate_sps(
    sps: &SeqParameterSet,
    configuration: &VideoDecoderConfig,
    limits: &Limits,
) -> Result<()> {
    if sps.chroma_format_idc != 1
        || sps.separate_colour_plane_flag
        || sps.bit_depth_luma_minus8 != 0
        || sps.bit_depth_chroma_minus8 != 0
    {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "native HEVC decoder requires SPS Main-profile 8-bit 4:2:0 syntax",
        ));
    }
    if sps.pic_width_in_luma_samples > limits.max_width
        || sps.pic_height_in_luma_samples > limits.max_height
    {
        return Err(limit("HEVC SPS coded dimensions exceed configured limits"));
    }
    let (sub_width, sub_height) = sub_wh_c(sps.chroma_format_idc);
    let cropped_width = usize::try_from(sps.pic_width_in_luma_samples)
        .ok()
        .and_then(|width| {
            let crop = usize::try_from(
                sps.conformance_window.left_offset + sps.conformance_window.right_offset,
            )
            .ok()?
            .checked_mul(sub_width)?;
            width.checked_sub(crop)
        })
        .ok_or_else(|| malformed("HEVC SPS horizontal conformance window is invalid"))?;
    let cropped_height = usize::try_from(sps.pic_height_in_luma_samples)
        .ok()
        .and_then(|height| {
            let crop = usize::try_from(
                sps.conformance_window.top_offset + sps.conformance_window.bottom_offset,
            )
            .ok()?
            .checked_mul(sub_height)?;
            height.checked_sub(crop)
        })
        .ok_or_else(|| malformed("HEVC SPS vertical conformance window is invalid"))?;
    if cropped_width != configuration.coded_dimensions.width as usize
        || cropped_height != configuration.coded_dimensions.height as usize
    {
        return Err(malformed(
            "HEVC SPS display dimensions do not match decoder configuration",
        ));
    }
    let pixels = u64::from(sps.pic_width_in_luma_samples)
        .checked_mul(u64::from(sps.pic_height_in_luma_samples))
        .ok_or_else(|| limit("HEVC picture dimensions overflow"))?;
    let picture_bytes = pixels
        .checked_mul(6)
        .ok_or_else(|| limit("HEVC picture allocation overflows"))?;
    let output_bytes = u64::from(configuration.coded_dimensions.width)
        .checked_mul(u64::from(configuration.coded_dimensions.height))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| limit("HEVC RGBA output allocation overflows"))?;
    let ordering = sps.sub_layer_ordering_info[usize::from(sps.max_sub_layers_minus1)];
    let retained_pictures = u64::from(ordering.max_dec_pic_buffering_minus1)
        .checked_add(2)
        .ok_or_else(|| limit("HEVC decoded-picture-buffer size overflows"))?;
    let decoder_bytes = picture_bytes
        .checked_mul(retained_pictures)
        .and_then(|value| value.checked_add(output_bytes))
        .ok_or_else(|| limit("HEVC decoder allocation budget overflows"))?;
    if decoder_bytes > limits.max_allocation_bytes {
        return Err(limit(
            "HEVC decoded-picture buffer exceeds the allocation limit",
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct HevcDecoder {
    configuration: VideoDecoderConfig,
    limits: Limits,
    configuration_nals: Vec<engine::nal::NalUnit>,
    nal_length_size: usize,
    sequence: SequenceDecoder,
    reorder: Vec<DecodedFrame>,
    presentation_indexes: BinaryHeap<Reverse<FrameIndex>>,
}

impl HevcDecoder {
    fn new(
        configuration: VideoDecoderConfig,
        limits: Limits,
        parsed: ParsedConfiguration,
    ) -> Result<Self> {
        let configuration_nals = parsed.record.nal_units;
        let nal_length_size = parsed.record.length_size;
        let mut decoder = Self {
            configuration,
            limits,
            configuration_nals,
            nal_length_size,
            sequence: SequenceDecoder::new(),
            reorder: Vec::new(),
            presentation_indexes: BinaryHeap::new(),
        };
        decoder.reinitialize()?;
        Ok(decoder)
    }

    fn reinitialize(&mut self) -> Result<()> {
        self.sequence = SequenceDecoder::new();
        for unit in self.configuration_nals.clone() {
            self.sequence.push_nal_unit(unit).map_err(sequence_error)?;
        }
        self.reorder.clear();
        self.presentation_indexes.clear();
        Ok(())
    }

    fn collect(&mut self, flush: bool) -> Result<Vec<DecodedVideoFrame>> {
        let pictures = self.collect_pictures(flush)?;
        let mut output = Vec::with_capacity(pictures.len());
        for (presentation_index, picture) in pictures {
            output.push(DecodedVideoFrame {
                presentation_index,
                frame: picture_to_rgba(&picture, &self.configuration, &self.limits)?,
            });
        }
        Ok(output)
    }

    /// The output pictures ready at this point, in presentation order, before
    /// colour conversion.
    ///
    /// Issue #220: this is the decoder's own product. [`collect`] converts each
    /// one to RGBA, which is the round trip an application pays but is not
    /// decoding, and `hevc_decoder_bench` times the two intervals separately so
    /// a whole-frame SIMD ratio is not diluted by a stage no HEVC kernel
    /// touches.
    ///
    /// [`collect`]: Self::collect
    fn collect_pictures(&mut self, flush: bool) -> Result<Vec<(FrameIndex, Picture)>> {
        self.reorder.extend(self.sequence.take_decoded());
        self.reorder
            .sort_by_key(|frame| (frame.cvs_index, frame.poc));
        let depth = if flush {
            0
        } else {
            self.sequence
                .max_num_reorder_pics()
                .map_or(DEFAULT_REORDER_DEPTH, |depth| depth as usize)
        };
        let mut output = Vec::new();
        while self.reorder.len() > depth {
            let decoded = self.reorder.remove(0);
            if !decoded.output {
                continue;
            }
            let presentation_index = self
                .presentation_indexes
                .pop()
                .ok_or_else(|| {
                    malformed("HEVC decoder produced a picture without a submitted identity")
                })?
                .0;
            output.push((presentation_index, decoded.picture));
        }
        Ok(output)
    }

    /// Parses one access unit into the sequence decoder, without collecting
    /// anything it made ready.
    ///
    /// The push and the collect are separate so [`VideoDecoder::submit`] and
    /// the benchmark surface's picture-only path share exactly the same
    /// bitstream handling and differ only in what they do with the result.
    fn push_sample(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        check_cancelled(cancellation)?;
        if sample.data.len() as u64 > self.limits.max_allocation_bytes {
            return Err(limit("HEVC access unit exceeds the allocation limit"));
        }
        let units = split_length_prefixed(&sample.data, self.nal_length_size)
            .map_err(|error| malformed(format!("invalid HEVC access unit: {error}")))?;
        self.presentation_indexes
            .push(Reverse(sample.presentation_index));
        let result = catch_unwind(AssertUnwindSafe(|| {
            for unit in units {
                self.sequence.push_nal_unit(unit)?;
            }
            Ok::<(), SequenceError>(())
        }));
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(sequence_error(error)),
            Err(_) => Err(malformed(
                "HEVC bitstream triggered an invalid decoder state",
            )),
        }
    }

    /// Decodes one access unit and stops at the decoded pictures.
    ///
    /// The picture-only half of the issue #220 split, used by
    /// [`crate::hevc_decoder_bench`].
    fn submit_pictures(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<Picture>> {
        self.push_sample(sample, cancellation)?;
        Ok(self
            .collect_pictures(false)?
            .into_iter()
            .map(|(_, picture)| picture)
            .collect())
    }
}

impl VideoDecoder for HevcDecoder {
    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        self.push_sample(sample, cancellation)?;
        self.collect(false)
    }

    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
        check_cancelled(cancellation)?;
        match catch_unwind(AssertUnwindSafe(|| self.sequence.flush())) {
            Ok(Ok(())) => self.collect(true),
            Ok(Err(error)) => Err(sequence_error(error)),
            Err(_) => Err(malformed(
                "HEVC bitstream triggered an invalid decoder state while draining",
            )),
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.reinitialize()
    }
}

/// Issue #189 stage attribution: colour conversion is not decoding, but it is
/// on the path every whole-frame measurement takes, so it is reported as its
/// own stage rather than left in the unattributed remainder.
fn picture_to_rgba(
    picture: &Picture,
    configuration: &VideoDecoderConfig,
    limits: &Limits,
) -> Result<VideoFrame> {
    let _profile = engine::profile::scope(engine::profile::Stage::ColorConvert);
    if picture.chroma_array_type() != 1
        || picture.bit_depth_luma() != 8
        || picture.bit_depth_chroma() != 8
    {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "decoded HEVC picture is not Main-profile 8-bit 4:2:0",
        ));
    }
    let width = configuration.coded_dimensions.width as usize;
    let height = configuration.coded_dimensions.height as usize;
    if picture.width_luma() < width || picture.height_luma() < height {
        return Err(malformed(
            "decoded HEVC picture is smaller than its display dimensions",
        ));
    }
    let stride = width
        .checked_mul(4)
        .ok_or_else(|| limit("HEVC RGBA stride overflows"))?;
    let length = stride
        .checked_mul(height)
        .ok_or_else(|| limit("HEVC RGBA allocation overflows"))?;
    if length as u64 > limits.max_allocation_bytes {
        return Err(limit("HEVC RGBA output exceeds the allocation limit"));
    }
    let luma = picture.plane(HevcPlane::Luma);
    let cb = picture.plane(HevcPlane::Cb);
    let cr = picture.plane(HevcPlane::Cr);
    let chroma_width = picture.plane_dims(HevcPlane::Cb).0;
    let mut rgba = vec![0_u8; length];
    color_convert::convert_yuv420_to_rgba(
        luma,
        picture.width_luma(),
        cb,
        cr,
        chroma_width,
        width,
        height,
        &mut rgba,
        stride,
    );
    VideoFrame::new(
        configuration.coded_dimensions,
        PixelFormat::Rgba8,
        configuration.color_range,
        vec![Plane { data: rgba, stride }],
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

fn sequence_error(error: SequenceError) -> Error {
    match error {
        SequenceError::Unsupported(message) => Error::new(ErrorKind::Unsupported, message),
        other => malformed(format!("invalid HEVC bitstream: {other}")),
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

    #[test]
    fn rejects_truncated_configuration_without_panicking() {
        let limits = Limits::default();
        let configuration = VideoDecoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: crate::VideoDimensions::new(16, 16, &limits).unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: vec![1, 2, 3],
        };
        let support = native_hevc_video_decoder_factory().capability(&configuration);
        assert!(matches!(support, CodecSupport::InvalidConfiguration { .. }));
    }

    #[test]
    fn canonical_conversion_uses_decoder_integer_rounding() {
        use color_convert::{clip_u8, multiply_high};

        assert_eq!(multiply_high(-8, -3_209), 0);
        assert_eq!(clip_u8(multiply_high(0, 9_539)), 0);
    }
}
