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
//! frames, a single nonzero `base_q_idx` byte, or that byte followed by the key
//! frame interval in frames as a big-endian `u16`. See [`parse_configuration`].
//! A hardware encoder takes the same configuration: it is rate controlled by
//! quality, which `base_q_idx` maps onto, instead of to a bitrate.

mod bitwriter;
mod dsp;
mod frame;
mod tables;

use crate::{
    Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange, EncodedSample,
    EncoderConfig, EncoderFuture, Error, ErrorKind, FrameIndex, FrameSource, HardwarePreference,
    Limits, Orientation, PixelFormat, Result, SampleDependency, VideoDimensions, VideoEncoder,
    VideoEncoderConfig, VideoEncoderFactory, VideoEncoderFormat, VideoFrame,
};
use frame::{CodingTools, FrameEncoder, Geometry, Picture};

/// The quantizer index an empty configuration encodes at.
pub const DEFAULT_BASE_Q_IDX: u8 = 80;
/// The key frame interval, in frames, an empty or one-byte configuration uses.
pub const DEFAULT_KEYFRAME_INTERVAL: u16 = 60;
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
    let (base_q_idx, keyframe_interval) =
        parse_configuration(&configuration.configuration).ok_or(CONFIGURATION_SHAPE)?;
    pick_level(
        dimensions,
        configuration.timescale,
        configuration.frame_duration,
    )
    .ok_or("VP9 dimensions and frame rate exceed level 6.2 limits")?;
    // `base_q_idx` 1 is the finest quantizer and 255 the coarsest.
    let quality = f64::from(255 - base_q_idx) / 254.0;
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
        keyframe_interval: u32::from(keyframe_interval),
    })
}

/// The `vpcC` a chunk's first frame signals, or `None` when that frame is not
/// a profile 0 key frame. A hardware encoder's sample is a sync sample exactly
/// when this is `Some`, and every key frame has to signal what the track
/// declares.
#[cfg(any(windows, target_os = "macos"))]
pub(crate) fn key_frame_vpcc(chunk: &[u8], level: u8) -> Option<Vec<u8>> {
    let frames = crate::vp9_dec::chunk_frames(chunk).ok()?;
    vpcc_from_key_frame(frames.first()?, level)
}

/// The hardware encoders, behind one interface so the factory reads the same
/// on every target.
#[cfg(windows)]
mod platform {
    use crate::hevc::windows_mf_encoder::{self, MftClass, OutputFormat, Settings};
    use crate::{CodecSupport, Error, ErrorKind, Limits, Result, VideoEncoder, VideoEncoderConfig};

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
    use crate::hevc::videotoolbox_encoder::{self, Settings};
    use crate::{
        Codec, CodecSupport, Error, ErrorKind, Limits, PixelFormat, Result, VideoEncoder,
        VideoEncoderConfig,
    };

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
                                   base_q_idx byte, or that byte and a nonzero big-endian u16 key \
                                   frame interval";

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

/// The backend-private configuration as `(base_q_idx, keyframe_interval)`, or
/// `None` when the blob is not one this backend understands.
///
/// `base_q_idx` is the frame header's quantizer index (VP9 section 7.2.9);
/// higher is smaller and blurrier. Zero, which would select VP9's lossless
/// mode, is not supported. A key frame interval of 1 makes every frame a key
/// frame.
fn parse_configuration(configuration: &[u8]) -> Option<(u8, u16)> {
    match *configuration {
        [] => Some((DEFAULT_BASE_Q_IDX, DEFAULT_KEYFRAME_INTERVAL)),
        [base_q_idx] if base_q_idx != 0 => Some((base_q_idx, DEFAULT_KEYFRAME_INTERVAL)),
        [base_q_idx, high, low] if base_q_idx != 0 => {
            let interval = u16::from_be_bytes([high, low]);
            (interval != 0).then_some((base_q_idx, interval))
        }
        _ => None,
    }
}

/// The lowest VP9 level (as `level_idc`, e.g. 31 for 3.1) whose picture size and
/// luma sample rate limits admit the stream (VP9 Annex A).
pub(crate) fn pick_level(
    dimensions: VideoDimensions,
    timescale: u32,
    frame_duration: u32,
) -> Option<u8> {
    const LEVELS: [(u8, u64, u64); 14] = [
        (10, 36_864, 829_440),
        (11, 73_728, 2_764_800),
        (20, 122_880, 4_608_000),
        (21, 245_760, 9_216_000),
        (30, 552_960, 20_736_000),
        (31, 983_040, 36_864_000),
        (40, 2_228_224, 83_558_400),
        (41, 2_228_224, 160_432_128),
        (50, 8_912_896, 311_951_360),
        (51, 8_912_896, 588_251_136),
        (52, 8_912_896, 1_176_502_272),
        (60, 35_651_584, 1_176_502_272),
        (61, 35_651_584, 2_353_004_544),
        (62, 35_651_584, 4_706_009_088),
    ];
    if timescale == 0 || frame_duration == 0 {
        return None;
    }
    let picture = u64::from(dimensions.width) * u64::from(dimensions.height);
    let sample_rate = (picture * u64::from(timescale)).div_ceil(u64::from(frame_duration));
    LEVELS
        .iter()
        .find(|&&(_, max_picture, max_rate)| picture <= max_picture && sample_rate <= max_rate)
        .map(|&(level, _, _)| level)
}

/// The complete `vpcC` box (`VPCodecConfigurationBox`, version 1) for an
/// 8-bit 4:2:0 profile 0 stream whose bitstream colour space is `color_space`
/// (VP9 section 7.2.2).
pub(crate) fn vpcc_box(level: u8, color_space: u8, full_range: bool) -> Vec<u8> {
    // ISO/IEC 23091-2 colour primaries, transfer characteristics and matrix
    // coefficients for each VP9 colour space; unknown and reserved values
    // map to "unspecified".
    let (primaries, transfer, matrix) = match color_space {
        1 | 3 => (6, 6, 6), // BT.601, SMPTE 170M
        2 => (1, 1, 1),     // BT.709
        4 => (7, 7, 7),     // SMPTE 240M
        5 => (9, 14, 9),    // BT.2020 non-constant luminance
        7 => (1, 13, 0),    // sRGB
        _ => (2, 2, 2),
    };
    let mut output = Vec::with_capacity(20);
    output.extend_from_slice(&20_u32.to_be_bytes());
    output.extend_from_slice(b"vpcC");
    output.extend_from_slice(&[1, 0, 0, 0]); // version 1, flags 0
    output.push(0); // profile
    output.push(level);
    // bitDepth 8, chromaSubsampling 1 (4:2:0 co-located with luma), range.
    output.push((8 << 4) | (1 << 1) | u8::from(full_range));
    output.extend_from_slice(&[primaries, transfer, matrix]);
    output.extend_from_slice(&0_u16.to_be_bytes()); // codecInitializationDataSize
    output
}

/// Builds the `vpcC` for a stream from its first key frame, reading the colour
/// space and range the encoder signalled, or `None` when `frame` is not a
/// profile 0 key frame.
///
/// The browser encoder needs this because `WebCodecs` reports no decoder
/// configuration for VP9: like AV1, everything travels in band.
#[cfg_attr(not(all(feature = "web", target_arch = "wasm32")), allow(dead_code))]
pub(crate) fn vpcc_from_key_frame(frame: &[u8], level: u8) -> Option<Vec<u8>> {
    let bit = |index: usize| -> Option<u8> {
        frame
            .get(index / 8)
            .map(|byte| (byte >> (7 - index % 8)) & 1)
    };
    let literal = |start: usize, bits: usize| -> Option<u32> {
        (start..start + bits).try_fold(0_u32, |value, index| {
            Some((value << 1) | u32::from(bit(index)?))
        })
    };
    // frame_marker, profile_low_bit, profile_high_bit, show_existing_frame,
    // frame_type, show_frame, error_resilient_mode, then the sync code.
    if literal(0, 2)? != 2 || literal(2, 2)? != 0 || bit(4)? != 0 || bit(5)? != 0 {
        return None;
    }
    if literal(8, 24)? != 0x49_83_42 {
        return None;
    }
    let color_space = literal(32, 3)? as u8;
    // sRGB is always full range and has no range bit; profile 0 has no
    // subsampling bits.
    let full_range = color_space == 7 || bit(35)? == 1;
    Some(vpcc_box(level, color_space, full_range))
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
    /// The partition and transform sizes the frame encoder searches.
    tools: CodingTools,
    /// The previous frame's reconstruction, which the next inter frame
    /// predicts from.
    reference: Option<Picture>,
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
        let (base_q_idx, keyframe_interval) = parse_configuration(&configuration.configuration)
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
            base_q_idx,
            keyframe_interval: u64::from(keyframe_interval),
            tools: CodingTools::ALL,
            reference: None,
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
        let (data, reconstruction) = FrameEncoder::new(
            self.geometry,
            &picture,
            reference,
            self.base_q_idx,
            self.tools,
        )
        .encode(self.color_range == ColorRange::Full);
        if u64::try_from(data.len()).unwrap_or(u64::MAX) > self.limits.max_allocation_bytes {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "encoded VP9 frame exceeds the configured allocation limit",
            ));
        }
        self.reference = Some(reconstruction);

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
            let (red, blue) = if frame.pixel_format == PixelFormat::Rgba8 {
                (0, 2)
            } else {
                (2, 0)
            };
            let rgb = |x: usize, y: usize| {
                let offset = stored(y, height) * input.stride + x * 4;
                [
                    i32::from(input.data[offset + red]),
                    i32::from(input.data[offset + 1]),
                    i32::from(input.data[offset + blue]),
                ]
            };
            let stride = picture.strides[0];
            for y in 0..height {
                for x in 0..width {
                    picture.planes[0][y * stride + x] = luma(rgb(x, y), full);
                }
            }
            let chroma_stride = picture.strides[1];
            for y in 0..chroma_height {
                for x in 0..chroma_width {
                    let mut sum = [0_i32; 3];
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let sample = rgb((x * 2 + dx).min(width - 1), (y * 2 + dy).min(height - 1));
                        for channel in 0..3 {
                            sum[channel] += sample[channel];
                        }
                    }
                    let [cb, cr] = chroma(sum.map(|value| (value + 2) >> 2), full);
                    picture.planes[1][y * chroma_stride + x] = cb;
                    picture.planes[2][y * chroma_stride + x] = cr;
                }
            }
        }
        PixelFormat::Rgb8 => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "the native VP9 backend does not accept Rgb8 input",
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

/// BT.601 luma from 8-bit RGB.
fn luma([r, g, b]: [i32; 3], full: bool) -> u8 {
    if full {
        ((77 * r + 150 * g + 29 * b + 128) >> 8).clamp(0, 255) as u8
    } else {
        (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16).clamp(0, 255) as u8
    }
}

/// BT.601 Cb and Cr from 8-bit RGB.
fn chroma([r, g, b]: [i32; 3], full: bool) -> [u8; 2] {
    let (cb, cr) = if full {
        (
            (-43 * r - 85 * g + 128 * b + 128) >> 8,
            (128 * r - 107 * g - 21 * b + 128) >> 8,
        )
    } else {
        (
            (-38 * r - 74 * g + 112 * b + 128) >> 8,
            (112 * r - 94 * g - 18 * b + 128) >> 8,
        )
    };
    [
        (cb + 128).clamp(0, 255) as u8,
        (cr + 128).clamp(0, 255) as u8,
    ]
}

#[cfg(test)]
mod tests;
