//! Native HEVC/H.265 decoding with platform acceleration and a dependency-free software fallback.
//!
//! The software decoder is portable and also builds for `wasm32`, where it is the browser's
//! fallback when WebCodecs cannot decode a track (issue #504); the platform backends and the
//! encoder's public factory stay native-only.

// Annex B and length-prefixed reframing for the platform encoders: Media Foundation on Windows
// and VideoToolbox on macOS.
#[cfg(any(windows, target_os = "macos", all(test, not(target_arch = "wasm32"))))]
mod annexb;
// internal — exposed for the criterion benchmark suite; not part of the stable API
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub mod bench;
pub(crate) mod color_convert;
#[cfg(not(target_arch = "wasm32"))]
pub mod decode_bench;
// internal — exposed for the stage-attribution example; not part of the stable API
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use engine::inter_pred::narrow_interp;
// internal — exposed for the stage-attribution example; not part of the stable API
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use engine::profile as decode_profile;
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(target_os = "macos")]
mod videotoolbox_encoder;
#[cfg(windows)]
mod windows_mf;
#[cfg(windows)]
mod windows_mf_encoder;

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::{
    CancellationToken, Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange,
    DecodedVideoFrame, EncodedVideoSample, Error, ErrorKind, FrameIndex, HardwarePreference,
    Limits, PixelFormat, Plane, Result, VideoDecoder, VideoDecoderConfig, VideoDecoderFactory,
    VideoFrame,
};
#[cfg(not(target_arch = "wasm32"))]
pub use encoder::native_hevc_video_encoder_factory;
use engine::hvcc::{HvccRecord, parse_hvcc, split_length_prefixed};
use engine::picture::{Picture, Plane as HevcPlane, sub_wh_c};
use engine::sequence::{DecodedFrame, SequenceDecoder, SequenceError};
use engine::sps::SeqParameterSet;

const DEFAULT_REORDER_DEPTH: usize = 8;

/// Returns the native HEVC Main and Main 10 decoder backend.
///
/// For 8-bit Main, `Prefer` and `Require` select NVIDIA NVDEC on supported 64-bit Windows/Linux
/// hosts, D3D11-aware Media Foundation on Windows, or VideoToolbox on macOS. `Prefer` falls back to
/// the dependency-free software decoder when acceleration is unavailable; `Avoid` always selects
/// software. Main 10 tracks with more than 8 bits per sample are always decoded in software, since
/// none of the accelerated backends is wired for them, so `Require` reports them unavailable.
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
        let parsed = match ParsedConfiguration::parse(configuration, &Limits::default()) {
            Ok(parsed) => parsed,
            Err(error) => return invalid_configuration(error.message()),
        };
        if configuration.hardware != HardwarePreference::Avoid
            && !parsed.high_bit_depth
            && hardware_available(configuration)
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
                    "native HEVC decoder supports the Main and Main 10 profiles",
                ));
            }
            CodecSupport::HardwareUnavailable => unreachable!("hardware is not checked here"),
            CodecSupport::InvalidConfiguration { reason } => {
                return Err(Error::new(ErrorKind::InvalidInput, reason));
            }
        }
        let parsed = ParsedConfiguration::parse(configuration, limits)?;
        if configuration.hardware == HardwarePreference::Require && parsed.high_bit_depth {
            // Every accelerated backend below is 8-bit Main only, and would only notice a 10-bit
            // sequence once decoding had started.
            return Err(Error::new(
                ErrorKind::Unsupported,
                "hardware HEVC decoding is unavailable (no accelerated backend decodes more than 8 bits per sample)",
            ));
        }
        if configuration.hardware != HardwarePreference::Avoid && !parsed.high_bit_depth {
            #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
            let mut hardware_errors = Vec::<String>::new();
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
        if !matches!(
            configuration.profile,
            CodecProfile::HevcMain | CodecProfile::HevcMain10
        ) {
            return CodecSupport::UnsupportedProfile;
        }
        if configuration.output_format != PixelFormat::Rgba8 {
            return invalid_configuration("native HEVC decoding currently outputs RGBA8");
        }
        if configuration.color_range != ColorRange::Limited {
            return invalid_configuration("native HEVC RGBA output requires limited-range input");
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
    /// Whether either component carries more than 8 bits per sample, which only the software
    /// decoder handles.
    high_bit_depth: bool,
}

impl ParsedConfiguration {
    fn parse(configuration: &VideoDecoderConfig, limits: &Limits) -> Result<Self> {
        if configuration.configuration.len() as u64 > limits.max_allocation_bytes {
            return Err(limit("HEVC configuration exceeds the allocation limit"));
        }
        let payload = hvcc_payload(&configuration.configuration)?;
        let record = parse_hvcc(payload)
            .map_err(|error| malformed(format!("invalid hvcC configuration: {error}")))?;
        let max_bit_depth_minus8 = max_bit_depth_minus8(configuration.profile);
        let profile_idc_allowed = match configuration.profile {
            // A Main 10 decoder also decodes Main.
            CodecProfile::HevcMain10 => matches!(record.general_profile_idc, 1 | 2),
            _ => record.general_profile_idc == 1,
        };
        if !profile_idc_allowed
            || record.bit_depth_luma_minus8 > max_bit_depth_minus8
            || record.bit_depth_chroma_minus8 > max_bit_depth_minus8
            || record.chroma_format_idc != 1
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                match configuration.profile {
                    CodecProfile::HevcMain10 => {
                        "native HEVC decoder requires an 8- to 10-bit 4:2:0 Main or Main 10 hvcC configuration"
                    }
                    _ => {
                        "native HEVC decoder requires an 8-bit 4:2:0 Main-profile hvcC configuration"
                    }
                },
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
        let high_bit_depth = record.bit_depth_luma_minus8 != 0
            || record.bit_depth_chroma_minus8 != 0
            || sps.bit_depth_luma_minus8 != 0
            || sps.bit_depth_chroma_minus8 != 0;
        Ok(Self {
            record,
            high_bit_depth,
        })
    }
}

/// The deepest samples `profile` allows, as `bit_depth_*_minus8`: 8 bits for Main, 10 for
/// Main 10.
fn max_bit_depth_minus8(profile: CodecProfile) -> u8 {
    match profile {
        CodecProfile::HevcMain10 => 2,
        _ => 0,
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
    let max_bit_depth_minus8 = max_bit_depth_minus8(configuration.profile);
    if sps.chroma_format_idc != 1
        || sps.separate_colour_plane_flag
        || sps.bit_depth_luma_minus8 > max_bit_depth_minus8
        || sps.bit_depth_chroma_minus8 > max_bit_depth_minus8
    {
        return Err(Error::new(
            ErrorKind::Unsupported,
            match configuration.profile {
                CodecProfile::HevcMain10 => {
                    "native HEVC decoder requires SPS 8- to 10-bit 4:2:0 syntax"
                }
                _ => "native HEVC decoder requires SPS Main-profile 8-bit 4:2:0 syntax",
            },
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
    /// Whether a caller wants the pictures being decoded, or is only walking past them.
    output_wanted: bool,
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
            output_wanted: true,
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
        if !self.output_wanted {
            // The pictures are still taken from the reorder queue above, which is what keeps the
            // decoder's own bookkeeping straight; only their conversion is skipped.
            return Ok(Vec::new());
        }
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
    #[cfg(not(target_arch = "wasm32"))]
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

    fn set_output_wanted(&mut self, wanted: bool) {
        self.output_wanted = wanted;
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
    let max_bit_depth = 8 + max_bit_depth_minus8(configuration.profile);
    if picture.chroma_array_type() != 1
        || !(8..=max_bit_depth).contains(&picture.bit_depth_luma())
        || !(8..=max_bit_depth).contains(&picture.bit_depth_chroma())
    {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "decoded HEVC picture is not 4:2:0 at a bit depth its profile allows",
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
    if picture.bit_depth_luma() == 8 && picture.bit_depth_chroma() == 8 {
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
    } else {
        color_convert::convert_high_bit_depth_yuv420_to_rgba(
            luma,
            picture.width_luma(),
            picture.bit_depth_luma(),
            cb,
            cr,
            chroma_width,
            picture.bit_depth_chroma(),
            width,
            height,
            &mut rgba,
            stride,
        );
    }
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

    fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        use std::task::{Context, Poll, Waker};
        let mut future = Box::pin(future);
        let mut context = Context::from_waker(Waker::noop());
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
        }
    }

    /// The 12-frame HEVC Main 10 fixture's decoder configuration and decode-order samples.
    fn main10_fixture(
        profile: CodecProfile,
        hardware: HardwarePreference,
    ) -> (VideoDecoderConfig, Vec<EncodedVideoSample>) {
        use crate::Mp4DemuxerOptions;
        use crate::io::MemorySource;
        use crate::mp4_demux::Mp4Demuxer;

        let limits = Limits::default();
        let source = MemorySource::new(
            include_bytes!("../../tests/fixtures/codec/bbb_hevc_main10_128x72.mp4").to_vec(),
        );
        let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
        let track = &movie.tracks[0];
        let samples = block_on(track.to_encoded_video_samples(&source, &limits)).unwrap();
        let configuration = VideoDecoderConfig {
            codec: Codec::Hevc,
            profile,
            coded_dimensions: track.dimensions.unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware,
            configuration: track.decoder_config.clone(),
        };
        (configuration, samples)
    }

    /// Issue #508: the 10-bit samples themselves are FFmpeg's `yuv420p10le` decode of the same
    /// fixture exactly, and their RGBA conversion is the fixture's, which applies the documented
    /// matrix to that FFmpeg decode.
    #[test]
    fn main10_pictures_match_an_independent_reference_decode() {
        let digests = |fixture: &str| {
            fixture
                .lines()
                .map(|line| line.split_once(' ').unwrap().1.to_owned())
                .collect::<Vec<_>>()
        };
        let expected = digests(include_str!(
            "../../tests/fixtures/codec/bbb_hevc_main10_128x72_yuv420p10le.sha256"
        ));
        let expected_rgba = digests(include_str!(
            "../../tests/fixtures/codec/bbb_hevc_main10_128x72_rgba.sha256"
        ));
        let limits = Limits::default();
        let (configuration, samples) =
            main10_fixture(CodecProfile::HevcMain10, HardwarePreference::Avoid);
        let parsed = ParsedConfiguration::parse(&configuration, &limits).unwrap();
        assert!(parsed.high_bit_depth);
        let mut decoder = HevcDecoder::new(configuration.clone(), limits, parsed).unwrap();
        let cancellation = CancellationToken::new();
        let mut pictures = Vec::new();
        for sample in &samples {
            decoder.push_sample(sample, &cancellation).unwrap();
            pictures.extend(decoder.collect_pictures(false).unwrap());
        }
        decoder.sequence.flush().unwrap();
        pictures.extend(decoder.collect_pictures(true).unwrap());
        pictures.sort_by_key(|(index, _)| *index);

        assert_eq!(pictures.len(), expected.len());
        for (((index, picture), expected), expected_rgba) in
            pictures.iter().zip(&expected).zip(&expected_rgba)
        {
            assert_eq!(picture.bit_depth_luma(), 10);
            assert_eq!(picture.bit_depth_chroma(), 10);
            let digest = crate::conformance::FrameDigest(crate::conformance::sha256(
                &picture.to_planar_le16(),
            ));
            assert_eq!(&digest.to_hex(), expected, "frame {}", index.0);
            let rgba = picture_to_rgba(picture, &configuration, &limits).unwrap();
            let digest = crate::conformance::FrameDigest::from_frame(&rgba).unwrap();
            assert_eq!(&digest.to_hex(), expected_rgba, "frame {} RGBA", index.0);
        }
    }

    #[test]
    fn main10_is_decoded_in_software_whatever_the_hardware_preference() {
        let factory = native_hevc_video_decoder_factory();
        for hardware in [HardwarePreference::Avoid, HardwarePreference::Prefer] {
            let (configuration, _) = main10_fixture(CodecProfile::HevcMain10, hardware);
            assert_eq!(
                factory.capability(&configuration),
                CodecSupport::Supported {
                    implementation: CodecImplementation::Software
                },
                "{hardware:?}"
            );
        }
        let (configuration, _) =
            main10_fixture(CodecProfile::HevcMain10, HardwarePreference::Require);
        assert_eq!(
            factory.capability(&configuration),
            CodecSupport::HardwareUnavailable
        );
        let error = factory
            .create(&configuration, &Limits::default())
            .err()
            .expect("no accelerated backend decodes Main 10");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn a_main_profile_configuration_still_refuses_a_10_bit_track() {
        let (configuration, _) = main10_fixture(CodecProfile::HevcMain, HardwarePreference::Avoid);
        let factory = native_hevc_video_decoder_factory();
        assert!(matches!(
            factory.capability(&configuration),
            CodecSupport::InvalidConfiguration { .. }
        ));
        let error = factory
            .create(&configuration, &Limits::default())
            .err()
            .expect("HEVC Main is 8-bit");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[test]
    fn canonical_conversion_uses_decoder_integer_rounding() {
        use color_convert::{clip_u8, multiply_high};

        assert_eq!(multiply_high(-8, -3_209), 0);
        assert_eq!(clip_u8(multiply_high(0, 9_539)), 0);
    }
}
