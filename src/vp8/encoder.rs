//! The native VP8 encoder backend, registered as a [`VideoEncoderFactory`].

use super::frame_encoder::{FrameEncoder, Source};
use super::predict::Plane;
use crate::hevc::engine::encoder::colorconv;
use crate::{
    Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange, EncodedSample,
    EncoderConfig, EncoderFuture, Error, ErrorKind, FrameIndex, FrameSource, HardwarePreference,
    Limits, Orientation, PixelFormat, Result, SampleDependency, VideoEncoder, VideoEncoderConfig,
    VideoEncoderFactory, VideoEncoderFormat, VideoFrame,
};

/// The quantizer index an empty configuration encodes at: visually close to
/// the source at a moderate bit rate.
const DEFAULT_QUANTIZER: u8 = 24;

/// The largest width or height a VP8 key frame header can carry.
const MAX_DIMENSION: u32 = 16_383;

const CONFIGURATION_HELP: &str = "the native VP8 encoder's configuration is empty (the default \
     quality), one quantizer-index byte from 0 to 127, or four big-endian bytes of target bits a \
     second optionally followed by four big-endian bytes of keyframe interval in frames";

/// Returns the native VP8 encoder backend.
///
/// The encoder accepts [`Codec::Vp8`] with [`CodecProfile::Vp8`] and
/// limited-range [`PixelFormat::Rgba8`] or [`PixelFormat::Bgra8`] CPU
/// frames of any size up to 16383 pixels a side, top-down or bottom-up, and
/// emits one VP8 frame per sample, ready for [`crate::WebmMuxer`]. VP8 has no
/// codec configuration record, so [`EncoderConfig::decoder_config`] is empty.
///
/// It is a dependency-free, pure-Rust implementation that codes key frames
/// and inter frames predicted from the previous frame, with whole-macroblock
/// and subblock intra prediction, quarter-sample motion compensation and the
/// loop filter, and its reconstruction is the decoder's, so
/// [`crate::native_vp8_video_decoder_factory`] and libvpx decode exactly the
/// pictures it predicts from. Frames are converted to YCbCr with the BT.601
/// matrix VP8 defines.
///
/// [`VideoEncoderConfig::configuration`] selects the operating point:
///
/// * empty: a fixed quantizer giving good quality at a moderate bit rate;
/// * one byte: a fixed quantizer index from 0 (finest) to 127 (coarsest);
/// * four big-endian bytes: a target bit rate in bits a second, which the
///   encoder meets on average by adjusting the quantizer from frame to frame;
/// * eight big-endian bytes: a target bit rate followed by a keyframe interval
///   in frames.
///
/// Without an explicit interval a key frame starts every second, rounded up
/// to a whole frame.
///
/// No platform encoder exposes VP8 to zvidlib's hardware backends, so the
/// encoder always runs in software: `Prefer` and `Avoid` select it, and
/// `Require` reports [`CodecSupport::HardwareUnavailable`]. Like the other
/// software encoders it runs on the calling thread.
pub fn native_vp8_video_encoder_factory() -> impl VideoEncoderFactory {
    Vp8EncoderFactory
}

struct Vp8EncoderFactory;

/// How the encoder chooses each frame's quantizer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OperatingPoint {
    FixedQuantizer(u8),
    TargetBitrate {
        bits_per_second: u32,
        /// Zero for the default interval.
        keyframe_interval: u32,
    },
}

fn parse_operating_point(configuration: &[u8]) -> Option<OperatingPoint> {
    let bitrate = |bytes: [u8; 4], keyframe_interval: u32| match u32::from_be_bytes(bytes) {
        0 => None,
        bits_per_second => Some(OperatingPoint::TargetBitrate {
            bits_per_second,
            keyframe_interval,
        }),
    };
    match *configuration {
        [] => Some(OperatingPoint::FixedQuantizer(DEFAULT_QUANTIZER)),
        [q] if q <= 127 => Some(OperatingPoint::FixedQuantizer(q)),
        [a, b, c, d] => bitrate([a, b, c, d], 0),
        [a, b, c, d, e, f, g, h] => bitrate([a, b, c, d], u32::from_be_bytes([e, f, g, h])),
        _ => None,
    }
}

fn invalid(reason: &str) -> CodecSupport {
    CodecSupport::InvalidConfiguration {
        reason: reason.into(),
    }
}

impl VideoEncoderFactory for Vp8EncoderFactory {
    fn capability(&self, c: &VideoEncoderConfig) -> CodecSupport {
        if c.codec != Codec::Vp8 {
            return CodecSupport::UnsupportedCodec;
        }
        if c.profile != CodecProfile::Vp8 {
            return CodecSupport::UnsupportedProfile;
        }
        if c.hardware == HardwarePreference::Require {
            return CodecSupport::HardwareUnavailable;
        }
        if !matches!(c.input_format, PixelFormat::Rgba8 | PixelFormat::Bgra8) {
            return invalid("native VP8 encoding requires RGBA8 or BGRA8 input");
        }
        if c.color_range != ColorRange::Limited {
            return invalid("native VP8 encoding requires limited-range input");
        }
        let dimensions = c.coded_dimensions;
        if dimensions.width == 0 || dimensions.height == 0 {
            return invalid("VP8 coded dimensions must be nonzero");
        }
        if dimensions.width > MAX_DIMENSION || dimensions.height > MAX_DIMENSION {
            return invalid("VP8 frames are at most 16383 pixels wide and high");
        }
        if c.timescale == 0 || c.frame_duration == 0 {
            return invalid("VP8 timescale and frame duration must be nonzero");
        }
        if parse_operating_point(&c.configuration).is_none() {
            return invalid(CONFIGURATION_HELP);
        }
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        }
    }

    fn create(&self, c: &VideoEncoderConfig, limits: &Limits) -> Result<Box<dyn VideoEncoder>> {
        match self.capability(c) {
            CodecSupport::Supported { .. } => {}
            CodecSupport::UnsupportedCodec => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native VP8 encoder requires VP8",
                ));
            }
            CodecSupport::UnsupportedProfile => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "native VP8 encoder requires the VP8 profile",
                ));
            }
            CodecSupport::HardwareUnavailable => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "hardware VP8 encoding is unavailable; the native VP8 encoder is software-only",
                ));
            }
            CodecSupport::InvalidConfiguration { reason } => {
                return Err(Error::new(ErrorKind::InvalidInput, reason));
            }
        }
        let dimensions = c.coded_dimensions;
        if dimensions.width > limits.max_width || dimensions.height > limits.max_height {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "VP8 dimensions exceed the configured limits",
            ));
        }
        let frames = FrameEncoder::new(dimensions.width as usize, dimensions.height as usize);
        let (padded_width, padded_height) = frames.padded_dimensions();
        // The source, the frame being coded and two references, each 4:2:0.
        let frame_bytes = (padded_width * padded_height * 3 / 2) as u64;
        if frame_bytes.saturating_mul(4) > limits.max_allocation_bytes {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "VP8 frame exceeds the allocation limit",
            ));
        }
        let operating_point =
            parse_operating_point(&c.configuration).expect("capability checked the configuration");
        let (rate, keyframe_interval) = match operating_point {
            OperatingPoint::FixedQuantizer(q) => (RateControl::Fixed(q), 0),
            OperatingPoint::TargetBitrate {
                bits_per_second,
                keyframe_interval,
            } => (
                RateControl::target(
                    bits_per_second,
                    c.timescale,
                    c.frame_duration,
                    u64::from(dimensions.width) * u64::from(dimensions.height),
                ),
                keyframe_interval,
            ),
        };
        let keyframe_interval = match keyframe_interval {
            // One second, rounded up to a whole frame.
            0 => c.timescale.div_ceil(c.frame_duration).max(1),
            frames => frames,
        };
        Ok(Box::new(Vp8Encoder {
            declared: EncoderConfig {
                codec: Codec::Vp8,
                timescale: c.timescale,
                decoder_config: Vec::new(),
            },
            format: VideoEncoderFormat {
                dimensions,
                pixel_format: c.input_format,
            },
            color_range: c.color_range,
            frame_duration: c.frame_duration,
            limits: *limits,
            next_index: 0,
            finished: false,
            frames,
            rate,
            keyframe_interval: u64::from(keyframe_interval),
        }))
    }
}

/// The quantizer each frame is coded at.
#[derive(Clone, Copy, Debug)]
enum RateControl {
    Fixed(u8),
    /// Steers the quantizer towards `target_bits` a frame, carrying the
    /// accumulated surplus or deficit in `buffer` so the average converges.
    Target {
        target_bits: f64,
        q: f64,
        buffer: f64,
    },
}

impl RateControl {
    fn target(bits_per_second: u32, timescale: u32, frame_duration: u32, pixels: u64) -> Self {
        let target_bits =
            f64::from(bits_per_second) * f64::from(frame_duration) / f64::from(timescale);
        let bits_per_pixel = target_bits / pixels.max(1) as f64;
        // A starting quantizer from the budget, which the feedback then
        // corrects within a few frames.
        let q = (40.0 - 15.0 * (bits_per_pixel / 0.1).log2()).clamp(4.0, 120.0);
        Self::Target {
            target_bits: target_bits.max(1.0),
            q,
            buffer: 0.0,
        }
    }

    fn quantizer(&self) -> u8 {
        match *self {
            Self::Fixed(q) => q,
            Self::Target { q, .. } => q.round().clamp(0.0, 127.0) as u8,
        }
    }

    fn update(&mut self, bytes: usize, key_frame: bool, frames_per_second: f64) {
        let Self::Target {
            target_bits,
            q,
            buffer,
        } = self
        else {
            return;
        };
        let bits = (bytes * 8) as f64;
        // Hold at most two seconds of surplus or deficit.
        let limit = *target_bits * frames_per_second.max(1.0) * 2.0;
        *buffer = (*buffer + bits - *target_bits).clamp(-limit, limit);
        // A key frame is expected to cost several inter frames; judging the
        // quantizer by its size alone would starve the frames after it.
        let observed = if key_frame { bits / 4.0 } else { bits };
        let ratio = ((observed + *buffer * 0.1) / *target_bits).max(1.0 / 64.0);
        *q = (*q + (6.0 * ratio.log2()).clamp(-8.0, 8.0)).clamp(2.0, 127.0);
    }
}

struct Vp8Encoder {
    declared: EncoderConfig,
    format: VideoEncoderFormat,
    color_range: ColorRange,
    frame_duration: u32,
    limits: Limits,
    next_index: u64,
    finished: bool,
    frames: FrameEncoder,
    rate: RateControl,
    keyframe_interval: u64,
}

impl VideoEncoder for Vp8Encoder {
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
            // Every frame is emitted as it is encoded.
            self.finished = true;
            Ok(Vec::new())
        })
    }
}

impl Vp8Encoder {
    fn encode_frame(
        &mut self,
        index: FrameIndex,
        source: FrameSource<'_>,
    ) -> Result<Vec<EncodedSample>> {
        if self.finished {
            return Err(Error::new(
                ErrorKind::InvalidState,
                "cannot encode VP8 frames after finish",
            ));
        }
        if index.0 != self.next_index {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "VP8 input frame indexes must be zero-based and consecutive",
            ));
        }
        let FrameSource::Cpu(source) = source else {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "the native VP8 encoder requires CPU frames",
            ));
        };
        let frame = source.frame;
        if frame.dimensions != self.format.dimensions
            || frame.pixel_format != self.format.pixel_format
            || frame.color_range != self.color_range
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "VP8 input frame does not match the configured dimensions, format, and range",
            ));
        }
        let planes = source_planes(frame, source.orientation, self.frames.padded_dimensions())?;

        let key_frame = index.0 % self.keyframe_interval == 0;
        let data = self
            .frames
            .encode(&planes, key_frame, self.rate.quantizer())?;
        if data.len() as u64 > self.limits.max_allocation_bytes {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "encoded VP8 frame exceeds the configured allocation limit",
            ));
        }
        let frames_per_second = f64::from(self.declared.timescale) / f64::from(self.frame_duration);
        self.rate.update(data.len(), key_frame, frames_per_second);

        let timestamp = i64::try_from(index.0)
            .ok()
            .and_then(|value| value.checked_mul(i64::from(self.frame_duration)))
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "VP8 timestamp overflow"))?;
        self.next_index += 1;
        Ok(vec![EncodedSample {
            data,
            dts: timestamp,
            pts: timestamp,
            duration: self.frame_duration,
            is_sync: key_frame,
            dependency: if key_frame {
                SampleDependency::INDEPENDENT
            } else {
                SampleDependency::DEPENDENT
            },
        }])
    }
}

/// Converts an `Rgba8` or `Bgra8` frame to BT.601 limited-range 4:2:0
/// planes padded to `(width, height)` by repeating the last column and row.
fn source_planes(
    frame: &VideoFrame,
    orientation: Orientation,
    (padded_width, padded_height): (usize, usize),
) -> Result<Source> {
    let width = frame.dimensions.width as usize;
    let height = frame.dimensions.height as usize;
    let plane = frame
        .planes
        .first()
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "VP8 input requires one plane"))?;
    let row_bytes = width * 4;
    if plane.stride < row_bytes || plane.data.len() < plane.stride * (height - 1) + row_bytes {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "VP8 input plane is smaller than its stride and dimensions",
        ));
    }
    let bgra = frame.pixel_format == PixelFormat::Bgra8;
    // One padded RGBA row of the source, for row `y` of the padded picture.
    let fill_row = |y: usize, row: &mut [u8]| {
        let y = y.min(height - 1);
        let stored = match orientation {
            Orientation::TopLeft => y,
            Orientation::BottomLeft => height - 1 - y,
        };
        let start = stored * plane.stride;
        row[..row_bytes].copy_from_slice(&plane.data[start..start + row_bytes]);
        if bgra {
            for pixel in row[..row_bytes].chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        let (filled, rest) = row.split_at_mut(row_bytes);
        let last = &filled[row_bytes - 4..];
        for pixel in rest.chunks_exact_mut(4) {
            pixel.copy_from_slice(last);
        }
    };
    let mut luma = Plane::new(padded_width, padded_height);
    let mut cb = Plane::new(padded_width / 2, padded_height / 2);
    let mut cr = Plane::new(padded_width / 2, padded_height / 2);
    let mut top = vec![0u8; padded_width * 4];
    let mut bottom = vec![0u8; padded_width * 4];
    let chroma_width = padded_width / 2;
    for pair in 0..padded_height / 2 {
        fill_row(pair * 2, &mut top);
        fill_row(pair * 2 + 1, &mut bottom);
        let y = pair * 2 * padded_width;
        colorconv::luma_row(&top, &mut luma.data[y..y + padded_width]);
        colorconv::luma_row(
            &bottom,
            &mut luma.data[y + padded_width..y + 2 * padded_width],
        );
        let c = pair * chroma_width;
        colorconv::chroma_row_pair(
            &top,
            &bottom,
            &mut cb.data[c..c + chroma_width],
            &mut cr.data[c..c + chroma_width],
        );
    }
    Ok([luma, cb, cr])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuFrameSource, Plane as FramePlane, VideoDimensions};
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

    fn configuration(width: u32, height: u32, configuration: Vec<u8>) -> VideoEncoderConfig {
        VideoEncoderConfig {
            codec: Codec::Vp8,
            profile: CodecProfile::Vp8,
            coded_dimensions: VideoDimensions { width, height },
            input_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Prefer,
            timescale: 30,
            frame_duration: 1,
            configuration,
        }
    }

    /// A moving gradient with a bright square, so frames have both texture
    /// and motion.
    pub(crate) fn test_frame(width: u32, height: u32, index: u32) -> VideoFrame {
        let mut data = vec![0u8; (width * height * 4) as usize];
        for y in 0..height {
            for x in 0..width {
                let at = ((y * width + x) * 4) as usize;
                let shifted = x + index * 3;
                let in_square =
                    (x + 40).wrapping_sub(index * 2) % width < 24 && (y + 8 + index) % height < 20;
                let (r, g, b) = if in_square {
                    (240, 220, 40)
                } else {
                    (
                        (shifted * 255 / (width + 90)) as u8,
                        (y * 255 / height) as u8,
                        ((shifted / 4 + y / 4) % 2 * 60 + 80) as u8,
                    )
                };
                data[at..at + 4].copy_from_slice(&[r, g, b, 255]);
            }
        }
        VideoFrame::new(
            VideoDimensions { width, height },
            PixelFormat::Rgba8,
            ColorRange::Limited,
            vec![FramePlane {
                data,
                stride: (width * 4) as usize,
            }],
            &Limits::default(),
        )
        .unwrap()
    }

    pub(crate) fn encode(config: &VideoEncoderConfig, frames: &[VideoFrame]) -> Vec<EncodedSample> {
        let mut encoder = native_vp8_video_encoder_factory()
            .create(config, &Limits::default())
            .unwrap();
        let mut samples = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            samples.extend(
                block_on(encoder.encode(
                    FrameIndex(index as u64),
                    FrameSource::Cpu(CpuFrameSource {
                        frame,
                        orientation: Orientation::TopLeft,
                    }),
                ))
                .unwrap(),
            );
        }
        samples.extend(block_on(encoder.finish()).unwrap());
        samples
    }

    #[test]
    fn factory_reports_a_software_encoder_for_vp8_only() {
        let factory = native_vp8_video_encoder_factory();
        let supported = configuration(64, 48, Vec::new());
        assert_eq!(
            factory.capability(&supported),
            CodecSupport::Supported {
                implementation: CodecImplementation::Software
            }
        );
        let mut other = supported.clone();
        other.codec = Codec::Av1;
        assert_eq!(factory.capability(&other), CodecSupport::UnsupportedCodec);
        other = supported.clone();
        other.hardware = HardwarePreference::Require;
        assert_eq!(
            factory.capability(&other),
            CodecSupport::HardwareUnavailable
        );
        other = supported.clone();
        other.input_format = PixelFormat::Gray8;
        assert!(matches!(
            factory.capability(&other),
            CodecSupport::InvalidConfiguration { .. }
        ));
        for configuration in [vec![128], vec![0, 0, 0, 0], vec![1, 2]] {
            other = supported.clone();
            other.configuration = configuration;
            assert!(matches!(
                factory.capability(&other),
                CodecSupport::InvalidConfiguration { .. }
            ));
        }
        let encoder = factory.create(&supported, &Limits::default()).unwrap();
        assert_eq!(encoder.config().codec, Codec::Vp8);
        assert!(encoder.config().decoder_config.is_empty());
    }

    #[test]
    fn key_frames_follow_the_interval_and_timing_is_exact() {
        let config = configuration(
            48,
            32,
            [100_000u32.to_be_bytes(), 3u32.to_be_bytes()].concat(),
        );
        let frames: Vec<_> = (0..7).map(|index| test_frame(48, 32, index)).collect();
        let samples = encode(&config, &frames);
        assert_eq!(samples.len(), 7);
        for (index, sample) in samples.iter().enumerate() {
            assert_eq!(sample.is_sync, index % 3 == 0, "frame {index}");
            assert_eq!(sample.data[0] & 1 == 0, sample.is_sync);
            assert_eq!(sample.pts, index as i64);
            assert_eq!(sample.dts, index as i64);
            assert_eq!(sample.duration, 1);
        }
    }

    #[test]
    fn a_target_bit_rate_is_met_on_average() {
        let (width, height) = (96, 64);
        let frames: Vec<_> = (0..60)
            .map(|index| test_frame(width, height, index))
            .collect();
        for bits_per_second in [60_000u32, 240_000] {
            let config = configuration(width, height, bits_per_second.to_be_bytes().to_vec());
            let samples = encode(&config, &frames);
            let bits: usize = samples.iter().map(|sample| sample.data.len() * 8).sum();
            // Two seconds at 30 frames a second.
            let average = bits as f64 / 2.0;
            let ratio = average / f64::from(bits_per_second);
            assert!(
                (0.6..1.4).contains(&ratio),
                "{bits_per_second} b/s target gave {average} b/s"
            );
        }
    }
}
