//! Native VP9 encoding: a dependency-free software encoder, and the
//! platform's hardware encoder where the host has one.
//!
//! The software backend encodes VP9 profile 0 (8-bit 4:2:0) with key frames and
//! inter frames. A key frame starts every group of pictures, and each inter
//! frame predicts from the frame before it, so every sample after a key frame
//! up to the next one depends on the samples before it. See [`frame`] for the
//! coding tools.
//!
//! [`VideoEncoderConfig::configuration`] is either empty, which encodes at
//! [`DEFAULT_BASE_Q_IDX`] with a key frame every [`DEFAULT_KEYFRAME_INTERVAL`]
//! frames, a single nonzero `base_q_idx` byte, that byte followed by the key
//! frame interval in frames as a big-endian `u16`, or those three bytes
//! followed by a flags byte whose bit 0 ([`FLAG_ERROR_RESILIENT`]) makes every
//! frame error resilient. See [`parse_configuration`]. A hardware encoder
//! takes the same configuration without that flag: it is rate controlled by
//! quality, which `base_q_idx` maps onto, instead of to a bitrate.

#[cfg(not(target_arch = "wasm32"))]
pub mod bench;
mod bitwriter;
mod context;
mod dsp;
mod frame;
#[doc(hidden)]
pub mod simd;
mod tables;

#[allow(unused_imports)]
use zvidlib_core::*;
// The containers and conformance harness the tests read their fixtures with.
#[cfg(test)]
#[allow(unused_imports)]
use zvidlib_container::*;
#[cfg(test)]
#[allow(unused_imports)]
use zvidlib_vp9_syntax::{key_frame_vpcc, vpcc_from_key_frame};
pub(crate) use zvidlib_vp9_syntax::{pick_level, vpcc_box};

use crate::{
    Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange, EncodedSample,
    EncoderConfig, EncoderFuture, Error, ErrorKind, FrameIndex, FrameSource, HardwarePreference,
    Limits, Orientation, PixelFormat, Result, SampleDependency, VideoDimensions, VideoEncoder,
    VideoEncoderConfig, VideoEncoderFactory, VideoEncoderFormat, VideoFrame,
};
use context::FrameContext;
use frame::{CodingTools, FrameEncoder, Geometry, ModeInfo, Picture};

/// The quantizer index an empty configuration encodes at.
pub const DEFAULT_BASE_Q_IDX: u8 = 80;
/// The key frame interval, in frames, an empty or one-byte configuration uses.
pub const DEFAULT_KEYFRAME_INTERVAL: u16 = 60;
/// The configuration flag that codes every frame error resilient: with the
/// default probabilities, and without the probabilities or motion vectors
/// earlier frames adapted to, so a frame decodes from its reference picture
/// alone.
pub const FLAG_ERROR_RESILIENT: u8 = 1;
/// The widest frame the single-tile-column encoder accepts. VP9 requires more
/// than one tile column above this width.
const MAX_WIDTH: u32 = 4096;

/// Returns the native VP9 profile 0 encoder backend.
///
/// `Avoid` always selects the software encoder, which encodes VP9 profile 0
/// from `Yuv420p8`, `Rgba8`, `Bgra8` or `Gray8` frames up to 4096 pixels wide,
/// entirely in Rust. RGB input is converted to 4:2:0 with the BT.601 matrix in
/// the configured range.
///
/// `Prefer` and `Require` ask for a hardware encoder: on Windows the GPU
/// vendor's Media Foundation VP9 encoder (for example Intel Quick Sync), which
/// takes `Rgba8`, `Bgra8` or `Yuv420p8`, and on macOS VideoToolbox's, which
/// takes `Rgba8` or `Bgra8`, in both cases limited-range input at even
/// dimensions. Its output is a VP9 profile 0 stream like the software
/// encoder's, with a `vpcC` built from its first key frame. `Prefer` falls
/// back to the software encoder when no hardware encoder is usable, and
/// `Require` reports [`CodecSupport::HardwareUnavailable`] instead. Linux has
/// no hardware encoder yet. [`VideoEncoder::implementation`] and
/// [`VideoEncoder::backend_name`] report which one a created encoder is.
pub fn native_vp9_video_encoder_factory() -> impl VideoEncoderFactory {
    NativeVp9EncoderFactory
}

struct NativeVp9EncoderFactory;

impl VideoEncoderFactory for NativeVp9EncoderFactory {
    fn capability(&self, configuration: &VideoEncoderConfig) -> CodecSupport {
        if configuration.codec != Codec::Vp9 {
            return CodecSupport::UnsupportedCodec;
        }
        if configuration.profile != CodecProfile::Vp9Profile0 {
            return CodecSupport::UnsupportedProfile;
        }
        if configuration.hardware != HardwarePreference::Avoid {
            if platform::capability(configuration) {
                return CodecSupport::Supported {
                    implementation: CodecImplementation::Hardware,
                };
            }
            if configuration.hardware == HardwarePreference::Require {
                return platform::unavailable(configuration);
            }
        }
        validate_configuration(configuration)
    }

    fn create(
        &self,
        configuration: &VideoEncoderConfig,
        limits: &Limits,
    ) -> Result<Box<dyn VideoEncoder>> {
        if configuration.codec != Codec::Vp9 || configuration.profile != CodecProfile::Vp9Profile0 {
            return Err(capability_error(self.capability(configuration)));
        }
        if configuration.hardware != HardwarePreference::Avoid {
            match platform::create(configuration, limits) {
                Ok(encoder) => return Ok(encoder),
                Err(error) if configuration.hardware == HardwarePreference::Require => {
                    return Err(error);
                }
                Err(_) => {}
            }
        }
        Ok(Box::new(NativeVp9Encoder::new(configuration, limits)?))
    }
}

/// What a hardware encoder is asked for, resolved from a configuration on
/// every platform alike.
#[cfg_attr(not(any(windows, target_os = "macos")), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct HardwareRequest {
    /// `base_q_idx` mapped onto `0.0..=1.0`, best quality highest.
    quality: f64,
    /// A nominal bitrate for that quality, for encoders that want one declared
    /// even when they are rate controlled by quality.
    nominal_bits_per_second: u32,
    keyframe_interval: u32,
}

/// The hardware request `configuration` resolves to, or why no hardware
/// encoder can take it. Platform restrictions, such as the input formats a
/// backend converts, are the backend's to add.
#[cfg_attr(not(any(windows, target_os = "macos")), allow(dead_code))]
fn hardware_request(
    configuration: &VideoEncoderConfig,
) -> std::result::Result<HardwareRequest, String> {
    let dimensions = configuration.coded_dimensions;
    if dimensions.width == 0 || dimensions.height == 0 {
        return Err("VP9 coded dimensions must be nonzero".into());
    }
    if dimensions.width % 2 != 0 || dimensions.height % 2 != 0 {
        return Err("hardware VP9 encoding requires even dimensions".into());
    }
    if configuration.color_range != ColorRange::Limited {
        return Err("hardware VP9 encoding requires limited-range input".into());
    }
    if configuration.timescale == 0 || configuration.frame_duration == 0 {
        return Err("VP9 timescale and frame duration must be nonzero".into());
    }
    let settings = parse_configuration(&configuration.configuration).ok_or(CONFIGURATION_SHAPE)?;
    if settings.error_resilient {
        return Err("hardware VP9 encoders cannot code every frame error resilient".into());
    }
    pick_level(
        dimensions,
        configuration.timescale,
        configuration.frame_duration,
    )
    .ok_or("VP9 dimensions and frame rate exceed level 6.2 limits")?;
    // `base_q_idx` 1 is the finest quantizer and 255 the coarsest.
    let quality = f64::from(255 - settings.base_q_idx) / 254.0;
    // About 0.17 bits a pixel at the default quantizer, rising steeply toward
    // the finest one: 1080p30 at the default declares roughly 10 Mbit/s.
    let bits_per_pixel = 0.05 + 0.25 * quality * quality;
    let pixels_per_second = f64::from(dimensions.width)
        * f64::from(dimensions.height)
        * f64::from(configuration.timescale)
        / f64::from(configuration.frame_duration);
    Ok(HardwareRequest {
        quality,
        nominal_bits_per_second: (pixels_per_second * bits_per_pixel)
            .clamp(100_000.0, f64::from(i32::MAX)) as u32,
        keyframe_interval: u32::from(settings.keyframe_interval),
    })
}

/// The hardware encoders, behind one interface so the factory reads the same
/// on every target.
#[cfg(windows)]
mod platform {
    use crate::{CodecSupport, Error, ErrorKind, Limits, Result, VideoEncoder, VideoEncoderConfig};
    use zvidlib_hardware::windows_mf_encoder::{self, MftClass, OutputFormat, Settings};

    /// The Media Foundation request `c` resolves to, or why it cannot be one.
    fn settings(c: &VideoEncoderConfig) -> std::result::Result<Settings, String> {
        let request = super::hardware_request(c)?;
        let settings = Settings {
            format: OutputFormat::Vp9,
            width: c.coded_dimensions.width,
            height: c.coded_dimensions.height,
            input_format: c.input_format,
            timescale: c.timescale,
            frame_duration: c.frame_duration,
            bits_per_second: request.nominal_bits_per_second,
            quality: Some((request.quality * 100.0).round() as u32),
            keyframe_interval: request.keyframe_interval,
        };
        match settings.unsupported_reason() {
            Some(reason) => Err(reason),
            None => Ok(settings),
        }
    }

    /// Whether a hardware VP9 MFT accepts `c`. Microsoft ships no software
    /// VP9 encoder, and the native one is the fallback in any case.
    pub(super) fn capability(c: &VideoEncoderConfig) -> bool {
        settings(c)
            .is_ok_and(|settings| windows_mf_encoder::probe(settings, MftClass::Hardware).is_ok())
    }

    /// Why `Require` cannot be met: a configuration no hardware encoder takes
    /// is invalid, and one it would take on a host without one is unavailable.
    pub(super) fn unavailable(c: &VideoEncoderConfig) -> CodecSupport {
        match settings(c) {
            Err(reason) => CodecSupport::InvalidConfiguration { reason },
            Ok(_) => CodecSupport::HardwareUnavailable,
        }
    }

    pub(super) fn create(c: &VideoEncoderConfig, limits: &Limits) -> Result<Box<dyn VideoEncoder>> {
        let settings = settings(c).map_err(|reason| Error::new(ErrorKind::InvalidInput, reason))?;
        windows_mf_encoder::create(settings, MftClass::Hardware, limits).map_err(|error| {
            Error::new(
                ErrorKind::Unsupported,
                format!("hardware VP9 encoding is unavailable ({})", error.message()),
            )
        })
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use crate::{
        Codec, CodecSupport, Error, ErrorKind, Limits, PixelFormat, Result, VideoEncoder,
        VideoEncoderConfig,
    };
    use zvidlib_hardware::videotoolbox_encoder::{self, Settings};

    /// The VideoToolbox request `c` resolves to, or why it cannot be one.
    fn settings(c: &VideoEncoderConfig) -> std::result::Result<Settings, String> {
        let request = super::hardware_request(c)?;
        if !matches!(c.input_format, PixelFormat::Rgba8 | PixelFormat::Bgra8) {
            return Err("VideoToolbox VP9 encoding requires Rgba8 or Bgra8 input".into());
        }
        if i32::try_from(c.timescale).is_err() {
            return Err("VideoToolbox VP9 encoding requires a timescale below 2^31".into());
        }
        Ok(Settings {
            bits_per_second: request.nominal_bits_per_second,
            quality: Some(request.quality),
            keyframe_interval: request.keyframe_interval,
        })
    }

    pub(super) fn capability(c: &VideoEncoderConfig) -> bool {
        settings(c).is_ok() && videotoolbox_encoder::is_available(Codec::Vp9, c.coded_dimensions)
    }

    /// Why `Require` cannot be met: a configuration VideoToolbox cannot take
    /// is invalid, and one it would take on a host without the hardware is
    /// unavailable.
    pub(super) fn unavailable(c: &VideoEncoderConfig) -> CodecSupport {
        match settings(c) {
            Err(reason) => CodecSupport::InvalidConfiguration { reason },
            Ok(_) => CodecSupport::HardwareUnavailable,
        }
    }

    pub(super) fn create(c: &VideoEncoderConfig, limits: &Limits) -> Result<Box<dyn VideoEncoder>> {
        let settings = settings(c).map_err(|reason| Error::new(ErrorKind::InvalidInput, reason))?;
        if !videotoolbox_encoder::is_available(Codec::Vp9, c.coded_dimensions) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "hardware VP9 encoding is unavailable (VideoToolbox has no hardware VP9 \
                 encoder for this size)",
            ));
        }
        videotoolbox_encoder::create(c, limits, settings)
    }
}

#[cfg(not(any(windows, target_os = "macos")))]
mod platform {
    use crate::{CodecSupport, Error, ErrorKind, Limits, Result, VideoEncoder, VideoEncoderConfig};

    pub(super) fn capability(_: &VideoEncoderConfig) -> bool {
        false
    }

    pub(super) fn unavailable(_: &VideoEncoderConfig) -> CodecSupport {
        CodecSupport::HardwareUnavailable
    }

    pub(super) fn create(_: &VideoEncoderConfig, _: &Limits) -> Result<Box<dyn VideoEncoder>> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "hardware VP9 encoding is unavailable (no platform encoder exists for this target)",
        ))
    }
}

const CONFIGURATION_SHAPE: &str = "the native VP9 encoder's configuration is empty, a nonzero \
                                   base_q_idx byte, that byte and a nonzero big-endian u16 key \
                                   frame interval, or those and a flags byte with only bit 0 \
                                   (error resilient) defined";

fn validate_configuration(configuration: &VideoEncoderConfig) -> CodecSupport {
    if configuration.codec != Codec::Vp9 {
        return CodecSupport::UnsupportedCodec;
    }
    if configuration.profile != CodecProfile::Vp9Profile0 {
        return CodecSupport::UnsupportedProfile;
    }
    let dimensions = configuration.coded_dimensions;
    if dimensions.width == 0 || dimensions.height == 0 {
        return invalid_support("VP9 coded dimensions must be nonzero");
    }
    if dimensions.width > MAX_WIDTH || dimensions.height > 65_536 {
        return invalid_support("the native VP9 encoder supports frames up to 4096 pixels wide");
    }
    if !matches!(
        configuration.input_format,
        PixelFormat::Yuv420p8 | PixelFormat::Rgba8 | PixelFormat::Bgra8 | PixelFormat::Gray8
    ) {
        return invalid_support(
            "the native VP9 encoder requires Yuv420p8, Rgba8, Bgra8 or Gray8 input",
        );
    }
    if configuration.timescale == 0 || configuration.frame_duration == 0 {
        return invalid_support("VP9 timescale and frame duration must be nonzero");
    }
    if parse_configuration(&configuration.configuration).is_none() {
        return invalid_support(CONFIGURATION_SHAPE);
    }
    if pick_level(
        dimensions,
        configuration.timescale,
        configuration.frame_duration,
    )
    .is_none()
    {
        return invalid_support("VP9 dimensions and frame rate exceed level 6.2 limits");
    }
    CodecSupport::Supported {
        implementation: CodecImplementation::Software,
    }
}

/// The settings a backend-private configuration selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Settings {
    /// The frame header's quantizer index (VP9 section 7.2.9); higher is
    /// smaller and blurrier. Zero, which would select VP9's lossless mode, is
    /// not supported.
    base_q_idx: u8,
    /// A key frame every this many frames; 1 makes every frame a key frame.
    keyframe_interval: u16,
    /// Code every frame error resilient instead of adapting probabilities
    /// from frame to frame.
    error_resilient: bool,
}

/// The backend-private configuration's settings, or `None` when the blob is
/// not one this backend understands.
fn parse_configuration(configuration: &[u8]) -> Option<Settings> {
    let (base_q_idx, keyframe_interval, flags) = match *configuration {
        [] => (DEFAULT_BASE_Q_IDX, DEFAULT_KEYFRAME_INTERVAL, 0),
        [base_q_idx] => (base_q_idx, DEFAULT_KEYFRAME_INTERVAL, 0),
        [base_q_idx, high, low] => (base_q_idx, u16::from_be_bytes([high, low]), 0),
        [base_q_idx, high, low, flags] => (base_q_idx, u16::from_be_bytes([high, low]), flags),
        _ => return None,
    };
    (base_q_idx != 0 && keyframe_interval != 0 && flags & !FLAG_ERROR_RESILIENT == 0).then_some(
        Settings {
            base_q_idx,
            keyframe_interval,
            error_resilient: flags & FLAG_ERROR_RESILIENT != 0,
        },
    )
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
            (ErrorKind::Unsupported, "unsupported VP9 encoder profile")
        }
        CodecSupport::InvalidConfiguration { reason } => {
            return Error::new(ErrorKind::InvalidInput, reason);
        }
        CodecSupport::HardwareUnavailable => (
            ErrorKind::Unsupported,
            "no hardware VP9 encoder is available",
        ),
        CodecSupport::Supported { .. } => (
            ErrorKind::Internal,
            "encoder capability changed unexpectedly",
        ),
        _ => (
            ErrorKind::Unsupported,
            "unsupported VP9 encoder configuration",
        ),
    };
    Error::new(kind, message)
}

fn validate_limits(dimensions: VideoDimensions, limits: &Limits) -> Result<()> {
    if dimensions.width > limits.max_width || dimensions.height > limits.max_height {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "VP9 dimensions exceed the configured limits",
        ));
    }
    let aligned =
        u64::from(dimensions.width.div_ceil(8) * 8) * u64::from(dimensions.height.div_ceil(8) * 8);
    // The source, reconstruction and reference pictures at 1.5 bytes a pixel,
    // the filtered copy and superblock-padded copy the loop filter search
    // holds alongside them, plus per-block mode information.
    let working = aligned
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(64 * 1024))
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "VP9 allocation size overflow"))?;
    if working > limits.max_allocation_bytes {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "VP9 frame exceeds the configured allocation limit",
        ));
    }
    Ok(())
}

struct NativeVp9Encoder {
    declared: EncoderConfig,
    format: VideoEncoderFormat,
    color_range: ColorRange,
    frame_duration: u32,
    limits: Limits,
    geometry: Geometry,
    base_q_idx: u8,
    keyframe_interval: u64,
    error_resilient: bool,
    /// The partition and transform sizes the frame encoder searches.
    tools: CodingTools,
    /// The previous frame's reconstruction, which the next inter frame
    /// predicts from.
    reference: Option<Picture>,
    /// The probabilities the next frame codes with: the decoder's frame
    /// context 0, which every frame that is not error resilient adapts and
    /// saves.
    context: FrameContext,
    /// The previous frame's block modes, whose motion vectors the next inter
    /// frame takes as candidates.
    previous_mode_info: Vec<ModeInfo>,
    previous_was_key: bool,
    next_index: u64,
    finished: bool,
}

impl VideoEncoder for NativeVp9Encoder {
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
            self.reference = None;
            self.previous_mode_info = Vec::new();
            Ok(Vec::new())
        })
    }
}

impl NativeVp9Encoder {
    fn new(configuration: &VideoEncoderConfig, limits: &Limits) -> Result<Self> {
        let support = validate_configuration(configuration);
        if !support.is_supported() {
            return Err(capability_error(support));
        }
        validate_limits(configuration.coded_dimensions, limits)?;
        let settings = parse_configuration(&configuration.configuration)
            .ok_or_else(|| Error::new(ErrorKind::InvalidInput, CONFIGURATION_SHAPE))?;
        let level = pick_level(
            configuration.coded_dimensions,
            configuration.timescale,
            configuration.frame_duration,
        )
        .ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                "VP9 dimensions and frame rate exceed level 6.2 limits",
            )
        })?;
        let full_range = configuration.color_range == ColorRange::Full;
        Ok(Self {
            declared: EncoderConfig {
                codec: Codec::Vp9,
                timescale: configuration.timescale,
                // The bitstream signals CS_BT_601.
                decoder_config: vpcc_box(level, 1, full_range),
            },
            format: VideoEncoderFormat {
                dimensions: configuration.coded_dimensions,
                pixel_format: configuration.input_format,
            },
            color_range: configuration.color_range,
            frame_duration: configuration.frame_duration,
            limits: *limits,
            geometry: Geometry::new(
                configuration.coded_dimensions.width as usize,
                configuration.coded_dimensions.height as usize,
            ),
            base_q_idx: settings.base_q_idx,
            keyframe_interval: u64::from(settings.keyframe_interval),
            error_resilient: settings.error_resilient,
            tools: CodingTools::ALL,
            reference: None,
            context: FrameContext::default(),
            previous_mode_info: Vec::new(),
            previous_was_key: false,
            next_index: 0,
            finished: false,
        })
    }

    fn encode_frame(
        &mut self,
        index: FrameIndex,
        source: FrameSource<'_>,
    ) -> Result<Vec<EncodedSample>> {
        if self.finished {
            return Err(Error::new(
                ErrorKind::InvalidState,
                "cannot encode VP9 frames after finish",
            ));
        }
        if index.0 != self.next_index {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "VP9 input frame indexes must be zero-based and consecutive",
            ));
        }
        let FrameSource::Cpu(source) = source else {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "the native VP9 backend requires CPU frames",
            ));
        };
        let frame = source.frame;
        if frame.dimensions != self.format.dimensions
            || frame.pixel_format != self.format.pixel_format
            || frame.color_range != self.color_range
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "VP9 input frame does not match the configured dimensions, format, and range",
            ));
        }
        let picture = source_picture(&self.geometry, frame, source.orientation)?;

        let key = index.0 % self.keyframe_interval == 0;
        let reference = if key { None } else { self.reference.as_ref() };
        // Key frames and error-resilient frames reset every frame context to
        // the defaults (`setup_past_independence`).
        if key || self.error_resilient {
            self.context = FrameContext::default();
        }
        // Each frame is shown and the same size as the one before, so a frame
        // that is not error resilient uses its motion vectors.
        let previous_mode_info = (!self.error_resilient && !self.previous_mode_info.is_empty())
            .then_some(self.previous_mode_info.as_slice());
        let encoded = FrameEncoder::new(
            self.geometry,
            &picture,
            reference,
            self.base_q_idx,
            self.tools,
            self.error_resilient,
            &self.context,
            previous_mode_info,
        )
        .encode(self.color_range == ColorRange::Full);
        let data = encoded.data;
        if u64::try_from(data.len()).unwrap_or(u64::MAX) > self.limits.max_allocation_bytes {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "encoded VP9 frame exceeds the configured allocation limit",
            ));
        }
        if !self.error_resilient {
            // refresh_frame_context = 1, frame_parallel_decoding_mode = 0.
            self.context = self.context.adapted(
                &encoded.counts,
                key,
                self.previous_was_key,
                self.tools.larger_transforms,
            );
        }
        self.reference = Some(encoded.reconstruction);
        self.previous_mode_info = encoded.mode_info;
        self.previous_was_key = key;

        let timestamp = i64::try_from(index.0)
            .ok()
            .and_then(|value| value.checked_mul(i64::from(self.frame_duration)))
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "VP9 timestamp overflow"))?;
        self.next_index = self
            .next_index
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "VP9 frame index overflow"))?;
        Ok(vec![EncodedSample {
            data,
            dts: timestamp,
            pts: timestamp,
            duration: self.frame_duration,
            is_sync: key,
            dependency: if key {
                SampleDependency::INDEPENDENT
            } else {
                SampleDependency::DEPENDENT
            },
        }])
    }
}

/// Converts a frame to an 8-aligned 4:2:0 picture, replicating the last row
/// and column into the alignment padding.
fn source_picture(
    geometry: &Geometry,
    frame: &VideoFrame,
    orientation: Orientation,
) -> Result<Picture> {
    let (width, height) = (geometry.width, geometry.height);
    let (chroma_width, chroma_height) = (geometry.chroma_width(), geometry.chroma_height());
    let layouts = crate::media::required_plane_layouts(frame.dimensions, frame.pixel_format)?;
    if frame.planes.len() != layouts.len() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "VP9 input frame has the wrong number of planes",
        ));
    }
    for (plane, &(row_bytes, rows)) in frame.planes.iter().zip(&layouts) {
        let required = rows
            .checked_sub(1)
            .and_then(|last| last.checked_mul(plane.stride))
            .and_then(|bytes| bytes.checked_add(row_bytes))
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "VP9 plane size overflow"))?;
        if plane.stride < row_bytes || plane.data.len() < required {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "a VP9 input plane is shorter than its stride and height",
            ));
        }
    }
    // The stored row of logical row `row` in a plane `rows` tall.
    let stored = |row: usize, rows: usize| match orientation {
        Orientation::TopLeft => row,
        Orientation::BottomLeft => rows - 1 - row,
    };

    let mut picture = Picture::new(geometry);
    let full = frame.color_range == ColorRange::Full;
    match frame.pixel_format {
        PixelFormat::Yuv420p8 => {
            for (index, (plane_width, plane_height)) in [
                (width, height),
                (chroma_width, chroma_height),
                (chroma_width, chroma_height),
            ]
            .into_iter()
            .enumerate()
            {
                let input = &frame.planes[index];
                let stride = picture.strides[index];
                for row in 0..plane_height {
                    let start = stored(row, plane_height) * input.stride;
                    picture.planes[index][row * stride..row * stride + plane_width]
                        .copy_from_slice(&input.data[start..start + plane_width]);
                }
            }
        }
        PixelFormat::Gray8 => {
            let input = &frame.planes[0];
            let stride = picture.strides[0];
            for row in 0..height {
                let start = stored(row, height) * input.stride;
                picture.planes[0][row * stride..row * stride + width]
                    .copy_from_slice(&input.data[start..start + width]);
            }
            picture.planes[1].fill(128);
            picture.planes[2].fill(128);
        }
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => {
            let input = &frame.planes[0];
            let bgra = frame.pixel_format == PixelFormat::Bgra8;
            // The BT.601 rows, reordered to the pixels' byte order.
            let order = |mut coefficients: [i32; 3]| {
                if bgra {
                    coefficients.swap(0, 2);
                }
                coefficients
            };
            let (luma, luma_offset, cb, cr) = if full {
                ([77, 150, 29], 0, [-43, -85, 128], [128, -107, -21])
            } else {
                ([66, 129, 25], 16, [-38, -74, 112], [112, -94, -18])
            };
            let (luma, cb, cr) = (order(luma), order(cb), order(cr));
            let row = |y: usize| {
                let start = stored(y, height) * input.stride;
                &input.data[start..start + width * 4]
            };
            let stride = picture.strides[0];
            for y in 0..height {
                simd::luma_row(
                    row(y),
                    luma,
                    luma_offset,
                    &mut picture.planes[0][y * stride..y * stride + width],
                );
            }
            // An odd width's last chroma sample averages the last column with
            // itself, and an odd height's last row averages the last row with
            // itself.
            let mut padded = [Vec::new(), Vec::new()];
            let chroma_stride = picture.strides[1];
            let [_, cb_plane, cr_plane] = &mut picture.planes;
            for y in 0..chroma_height {
                let mut rows = [row(y * 2), row((y * 2 + 1).min(height - 1))];
                if width % 2 == 1 {
                    for (padded, row) in padded.iter_mut().zip(&mut rows) {
                        padded.clear();
                        padded.extend_from_slice(row);
                        padded.extend_from_slice(&row[row.len() - 4..]);
                        *row = padded;
                    }
                }
                let range = y * chroma_stride..y * chroma_stride + chroma_width;
                simd::chroma_row(
                    rows[0],
                    rows[1],
                    cb,
                    cr,
                    &mut cb_plane[range.clone()],
                    &mut cr_plane[range],
                );
            }
        }
        PixelFormat::Rgb8 => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "the native VP9 backend does not accept Rgb8 input",
            ));
        }
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "the native VP9 backend does not accept this pixel format",
            ));
        }
    }

    // Replicate edges into the 8-aligned padding so it codes cheaply.
    for (index, (plane_width, plane_height)) in [
        (width, height),
        (chroma_width, chroma_height),
        (chroma_width, chroma_height),
    ]
    .into_iter()
    .enumerate()
    {
        let stride = picture.strides[index];
        let rows = picture.planes[index].len() / stride;
        let plane = &mut picture.planes[index];
        for row in 0..plane_height {
            let edge = plane[row * stride + plane_width - 1];
            plane[row * stride + plane_width..(row + 1) * stride].fill(edge);
        }
        for row in plane_height..rows {
            plane.copy_within(
                (plane_height - 1) * stride..plane_height * stride,
                row * stride,
            );
        }
    }
    Ok(picture)
}

#[cfg(test)]
mod tests;

/// The SIMD dispatch sites in this crate, each with the instruction set it
/// resolves to right now. `zvidlib::simd::active_by_site` reports every crate's
/// sites together and documents what each one covers.
#[doc(hidden)]
#[must_use]
pub fn simd_sites() -> Vec<(&'static str, zvidlib_core::SimdIsa)> {
    vec![("vp9_encode", from_vp9_encode_isa(simd::isa()))]
}

fn from_vp9_encode_isa(isa: simd::Isa) -> zvidlib_core::SimdIsa {
    use simd::Isa;
    use zvidlib_core::SimdIsa;
    match isa {
        Isa::Scalar => SimdIsa::Scalar,
        #[cfg(target_arch = "x86_64")]
        Isa::Sse41 => SimdIsa::Sse41,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => SimdIsa::Avx2,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => SimdIsa::Neon,
    }
}
