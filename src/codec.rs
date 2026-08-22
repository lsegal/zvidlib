use crate::{
    AudioBuffer, Codec, ColorRange, Error, ErrorKind, FrameIndex, FrameSource, Limits, PixelFormat,
    Plane, Result, VideoDimensions, VideoFrame,
};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A non-`Send` encoder future that works on native and single-threaded WASM.
pub type EncoderFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + 'a>>;

/// The kind of media carried by an encoded track.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrackKind {
    Video,
    Audio,
}

/// Dependency information written to the MP4 sample dependency table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampleDependency {
    pub is_leading: u8,
    pub depends_on: u8,
    pub is_depended_on: u8,
    pub has_redundancy: u8,
}

impl SampleDependency {
    pub const INDEPENDENT: Self = Self {
        is_leading: 0,
        depends_on: 2,
        is_depended_on: 0,
        has_redundancy: 0,
    };

    pub const DEPENDENT: Self = Self {
        is_leading: 0,
        depends_on: 1,
        is_depended_on: 0,
        has_redundancy: 0,
    };

    pub(crate) fn to_sdtp(self) -> u8 {
        ((self.is_leading & 3) << 6)
            | ((self.depends_on & 3) << 4)
            | ((self.is_depended_on & 3) << 2)
            | (self.has_redundancy & 3)
    }
}

/// One encoded access unit with exact decode and presentation timing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedSample {
    pub data: Vec<u8>,
    pub dts: i64,
    pub pts: i64,
    pub duration: u32,
    pub is_sync: bool,
    pub dependency: SampleDependency,
}

/// Codec and sample-entry information needed to declare an MP4 track.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncoderConfig {
    pub codec: Codec,
    pub timescale: u32,
    /// Complete codec configuration box, including its size and fourcc.
    pub decoder_config: Vec<u8>,
}

/// Video format accepted by a video encoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VideoEncoderFormat {
    pub dimensions: VideoDimensions,
    pub pixel_format: PixelFormat,
}

/// Audio format accepted by an audio encoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioEncoderFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

/// Gapless audio metadata produced when an encoder is drained.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AudioGapless {
    pub priming: u32,
    pub padding: u32,
}

/// Packets and metadata emitted while draining an audio encoder.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AudioDrain {
    pub samples: Vec<EncodedSample>,
    pub gapless: AudioGapless,
}

/// Backend-neutral video encoder contract.
pub trait VideoEncoder {
    fn config(&self) -> &EncoderConfig;
    fn format(&self) -> VideoEncoderFormat;
    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        frame: FrameSource<'a>,
    ) -> EncoderFuture<'a, Vec<EncodedSample>>;
    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, Vec<EncodedSample>>;
}

/// Backend-neutral audio encoder contract.
pub trait AudioEncoder {
    fn config(&self) -> &EncoderConfig;
    fn format(&self) -> AudioEncoderFormat;
    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        buffer: AudioBuffer,
    ) -> EncoderFuture<'a, Vec<EncodedSample>>;
    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, AudioDrain>;
}

/// A normalized codec profile that can be compared across backend implementations.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CodecProfile {
    UncompressedGray8,
    HevcMain,
    HevcMain10,
    Av1Main,
    Av1High,
    Av1Professional,
    AacLowComplexity,
}

/// Whether a caller permits or requires a hardware codec implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum HardwarePreference {
    Require,
    Prefer,
    Avoid,
}

/// The implementation class selected by a codec factory.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CodecImplementation {
    Software,
    Hardware,
}

/// The result of querying a factory with a complete normalized configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CodecSupport {
    Supported { implementation: CodecImplementation },
    UnsupportedCodec,
    UnsupportedProfile,
    InvalidConfiguration { reason: String },
    HardwareUnavailable,
}

impl CodecSupport {
    pub const fn is_supported(&self) -> bool {
        matches!(self, Self::Supported { .. })
    }
}

/// Backend-neutral video decoder configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoDecoderConfig {
    pub codec: Codec,
    pub profile: CodecProfile,
    pub coded_dimensions: VideoDimensions,
    pub output_format: PixelFormat,
    pub color_range: ColorRange,
    pub hardware: HardwarePreference,
    /// Container configuration bytes in the codec's standardized format.
    pub configuration: Vec<u8>,
}

/// Backend-neutral video encoder configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoEncoderConfig {
    pub codec: Codec,
    pub profile: CodecProfile,
    pub coded_dimensions: VideoDimensions,
    pub input_format: PixelFormat,
    pub color_range: ColorRange,
    pub hardware: HardwarePreference,
    /// Exact media-clock timescale used for emitted DTS and PTS values.
    pub timescale: u32,
    /// Exact duration, in `timescale` ticks, of every submitted frame.
    pub frame_duration: u32,
    pub configuration: Vec<u8>,
}

/// One owned compressed sample in decode order with its presentation identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedVideoSample {
    pub presentation_index: FrameIndex,
    pub random_access: bool,
    pub data: Vec<u8>,
}

/// One normalized decoder output, independent of backend-private image types.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedVideoFrame {
    pub presentation_index: FrameIndex,
    pub frame: VideoFrame,
}

/// A cheap cloneable cancellation signal shared with a running codec operation.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::new(
                ErrorKind::Cancelled,
                "codec operation cancelled",
            ))
        } else {
            Ok(())
        }
    }
}

/// A stateful video decoder that may retain reference and reorder state.
pub trait VideoDecoder {
    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>>;
    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>>;
    fn reset(&mut self) -> Result<()>;
}

/// Discovers and creates video decoders without exposing backend-specific types.
pub trait VideoDecoderFactory {
    fn capability(&self, configuration: &VideoDecoderConfig) -> CodecSupport;
    fn create(
        &self,
        configuration: &VideoDecoderConfig,
        limits: &Limits,
    ) -> Result<Box<dyn VideoDecoder>>;
}

/// Discovers and creates video encoders using the normalized capability model.
pub trait VideoEncoderFactory {
    fn capability(&self, configuration: &VideoEncoderConfig) -> CodecSupport;
    fn create(
        &self,
        configuration: &VideoEncoderConfig,
        limits: &Limits,
    ) -> Result<Box<dyn VideoEncoder>>;
}

/// Observable counters for validating cache and decoder reuse behavior.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DecodeStatistics {
    pub samples_submitted: u64,
    pub cache_hits: u64,
    pub resets: u64,
    pub drains: u64,
}

/// Presentation-indexed exact-frame access over decode-order compressed samples.
pub struct ExactFrameReader {
    configuration: VideoDecoderConfig,
    decoder: Box<dyn VideoDecoder>,
    samples: Vec<EncodedVideoSample>,
    decode_position_by_presentation: HashMap<FrameIndex, usize>,
    cache: BTreeMap<FrameIndex, VideoFrame>,
    lru: VecDeque<FrameIndex>,
    published_since_reset: HashSet<FrameIndex>,
    next_decode_position: Option<usize>,
    limits: Limits,
    statistics: DecodeStatistics,
}

impl ExactFrameReader {
    pub fn new(
        factory: &dyn VideoDecoderFactory,
        configuration: VideoDecoderConfig,
        samples: Vec<EncodedVideoSample>,
        limits: Limits,
    ) -> Result<Self> {
        if limits.max_cached_frames == 0 || limits.max_decode_samples_per_seek == 0 {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "exact-frame limits must permit cached frames and decode work",
            ));
        }
        let capability = factory.capability(&configuration);
        if !capability.is_supported() {
            return Err(capability_error(capability));
        }
        if samples.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "an exact-frame reader requires at least one sample",
            ));
        }
        if !samples[0].random_access {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "the first decode-order sample must be a random-access point",
            ));
        }
        let mut positions = HashMap::with_capacity(samples.len());
        for (position, sample) in samples.iter().enumerate() {
            if positions
                .insert(sample.presentation_index, position)
                .is_some()
            {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "presentation frame identities must be unique",
                ));
            }
        }
        let decoder = factory.create(&configuration, &limits)?;
        Ok(Self {
            configuration,
            decoder,
            samples,
            decode_position_by_presentation: positions,
            cache: BTreeMap::new(),
            lru: VecDeque::new(),
            published_since_reset: HashSet::new(),
            next_decode_position: None,
            limits,
            statistics: DecodeStatistics::default(),
        })
    }

    /// Returns exactly the requested presentation frame.
    pub fn get(
        &mut self,
        presentation_index: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<VideoFrame> {
        cancellation.check()?;
        if let Some(frame) = self.cache.get(&presentation_index).cloned() {
            self.statistics.cache_hits = self.statistics.cache_hits.saturating_add(1);
            self.touch(presentation_index);
            return Ok(frame);
        }
        let target_position = *self
            .decode_position_by_presentation
            .get(&presentation_index)
            .ok_or_else(|| {
                Error::new(ErrorKind::InvalidInput, "presentation frame is not indexed")
            })?;
        let random_access_position = self.nearest_random_access(target_position);
        let can_reuse = self.next_decode_position.is_some_and(|position| {
            position >= random_access_position && position <= target_position
        });
        if !can_reuse {
            self.decoder.reset()?;
            self.statistics.resets = self.statistics.resets.saturating_add(1);
            self.published_since_reset.clear();
            self.next_decode_position = Some(random_access_position);
        }

        let mut work = 0_u32;
        while let Some(position) = self.next_decode_position {
            cancellation.check()?;
            if work >= self.limits.max_decode_samples_per_seek {
                return Err(Error::new(
                    ErrorKind::ResourceLimit,
                    "exact-frame request exceeded the configured decode-work limit",
                ));
            }
            if position == self.samples.len() {
                self.drain_internal(cancellation)?;
                break;
            }
            let outputs = self.decoder.submit(&self.samples[position], cancellation)?;
            self.statistics.samples_submitted = self.statistics.samples_submitted.saturating_add(1);
            self.next_decode_position = Some(position + 1);
            work += 1;
            self.publish(outputs)?;
            if let Some(frame) = self.cache.get(&presentation_index).cloned() {
                self.touch(presentation_index);
                return Ok(frame);
            }
        }
        if let Some(frame) = self.cache.get(&presentation_index).cloned() {
            self.touch(presentation_index);
            Ok(frame)
        } else {
            Err(Error::new(
                ErrorKind::MalformedMedia,
                "decoder did not produce the requested presentation frame",
            ))
        }
    }

    /// Drains delayed output into the bounded presentation cache.
    pub fn drain(&mut self, cancellation: &CancellationToken) -> Result<usize> {
        cancellation.check()?;
        self.drain_internal(cancellation)
    }

    /// Clears decoder, reorder, and frame-cache state.
    pub fn reset(&mut self) -> Result<()> {
        self.decoder.reset()?;
        self.statistics.resets = self.statistics.resets.saturating_add(1);
        self.next_decode_position = None;
        self.cache.clear();
        self.lru.clear();
        self.published_since_reset.clear();
        Ok(())
    }

    pub const fn statistics(&self) -> DecodeStatistics {
        self.statistics
    }

    pub fn cached_frames(&self) -> usize {
        self.cache.len()
    }

    fn nearest_random_access(&self, target_position: usize) -> usize {
        (0..=target_position)
            .rev()
            .find(|position| self.samples[*position].random_access)
            .unwrap_or(0)
    }

    fn drain_internal(&mut self, cancellation: &CancellationToken) -> Result<usize> {
        let outputs = self.decoder.drain(cancellation)?;
        let count = outputs.len();
        self.statistics.drains = self.statistics.drains.saturating_add(1);
        self.next_decode_position = Some(self.samples.len());
        self.publish(outputs)?;
        Ok(count)
    }

    fn publish(&mut self, outputs: Vec<DecodedVideoFrame>) -> Result<()> {
        for output in outputs {
            if !self.published_since_reset.insert(output.presentation_index) {
                return Err(Error::new(
                    ErrorKind::MalformedMedia,
                    "decoder produced the same presentation frame more than once without a reset",
                ));
            }
            if !self
                .decode_position_by_presentation
                .contains_key(&output.presentation_index)
            {
                return Err(Error::new(
                    ErrorKind::MalformedMedia,
                    "decoder produced an unindexed presentation frame",
                ));
            }
            if output.frame.dimensions != self.configuration.coded_dimensions
                || output.frame.pixel_format != self.configuration.output_format
                || output.frame.color_range != self.configuration.color_range
            {
                return Err(Error::new(
                    ErrorKind::MalformedMedia,
                    "decoder output does not match its normalized configuration",
                ));
            }
            self.insert_cache(output.presentation_index, output.frame);
        }
        Ok(())
    }

    fn insert_cache(&mut self, index: FrameIndex, frame: VideoFrame) {
        self.cache.insert(index, frame);
        self.touch(index);
        while self.cache.len() > self.limits.max_cached_frames as usize {
            if let Some(evicted) = self.lru.pop_front() {
                self.cache.remove(&evicted);
            }
        }
    }

    fn touch(&mut self, index: FrameIndex) {
        self.lru.retain(|candidate| *candidate != index);
        self.lru.push_back(index);
    }
}

fn capability_error(capability: CodecSupport) -> Error {
    let (kind, message) = match capability {
        CodecSupport::UnsupportedCodec => (
            ErrorKind::Unsupported,
            "decoder does not support the requested codec".into(),
        ),
        CodecSupport::UnsupportedProfile => (
            ErrorKind::Unsupported,
            "decoder does not support the requested profile".into(),
        ),
        CodecSupport::InvalidConfiguration { reason } => (ErrorKind::InvalidInput, reason),
        CodecSupport::HardwareUnavailable => (
            ErrorKind::Unsupported,
            "requested hardware decoder is unavailable".into(),
        ),
        CodecSupport::Supported { .. } => (
            ErrorKind::Internal,
            "decoder capability changed unexpectedly".into(),
        ),
    };
    Error::new(kind, message)
}

/// Returns the portable, software-only uncompressed Gray8 decoder backend.
///
/// Each packet contains one tightly packed Gray8 image. The concrete backend
/// remains private while this factory provides an end-to-end reference path.
pub fn uncompressed_video_decoder_factory() -> impl VideoDecoderFactory {
    UncompressedVideoDecoderFactory
}

struct UncompressedVideoDecoderFactory;

impl VideoDecoderFactory for UncompressedVideoDecoderFactory {
    fn capability(&self, configuration: &VideoDecoderConfig) -> CodecSupport {
        if configuration.codec != Codec::UncompressedVideo {
            return CodecSupport::UnsupportedCodec;
        }
        if configuration.profile != CodecProfile::UncompressedGray8 {
            return CodecSupport::UnsupportedProfile;
        }
        if configuration.output_format != PixelFormat::Gray8
            || !configuration.configuration.is_empty()
        {
            return CodecSupport::InvalidConfiguration {
                reason: "uncompressed Gray8 requires Gray8 output and no configuration bytes"
                    .into(),
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
        let capability = self.capability(configuration);
        if !capability.is_supported() {
            return Err(capability_error(capability));
        }
        if configuration.coded_dimensions.width > limits.max_width
            || configuration.coded_dimensions.height > limits.max_height
        {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "decoded dimensions exceed the configured limits",
            ));
        }
        let required = u64::from(configuration.coded_dimensions.width)
            .checked_mul(u64::from(configuration.coded_dimensions.height))
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "decoded frame size overflow"))?;
        if required > limits.max_allocation_bytes {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "decoded frame exceeds the configured allocation limit",
            ));
        }
        Ok(Box::new(UncompressedVideoDecoder {
            configuration: configuration.clone(),
            limits: *limits,
        }))
    }
}

struct UncompressedVideoDecoder {
    configuration: VideoDecoderConfig,
    limits: Limits,
}

impl VideoDecoder for UncompressedVideoDecoder {
    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        cancellation.check()?;
        let width = usize::try_from(self.configuration.coded_dimensions.width)
            .map_err(|_| Error::new(ErrorKind::ResourceLimit, "decoded width is too large"))?;
        let height = usize::try_from(self.configuration.coded_dimensions.height)
            .map_err(|_| Error::new(ErrorKind::ResourceLimit, "decoded height is too large"))?;
        let length = width
            .checked_mul(height)
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "decoded frame size overflow"))?;
        if sample.data.len() != length {
            return Err(Error::new(
                ErrorKind::MalformedMedia,
                "uncompressed Gray8 packet size does not match coded dimensions",
            ));
        }
        let frame = VideoFrame::new(
            self.configuration.coded_dimensions,
            PixelFormat::Gray8,
            self.configuration.color_range,
            vec![Plane {
                data: sample.data.clone(),
                stride: width,
            }],
            &self.limits,
        )?;
        Ok(vec![DecodedVideoFrame {
            presentation_index: sample.presentation_index,
            frame,
        }])
    }

    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
        cancellation.check()?;
        Ok(Vec::new())
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VideoDecoderConfig {
        VideoDecoderConfig {
            codec: Codec::UncompressedVideo,
            profile: CodecProfile::UncompressedGray8,
            coded_dimensions: VideoDimensions::new(1, 1, &Limits::default()).unwrap(),
            output_format: PixelFormat::Gray8,
            color_range: ColorRange::Full,
            hardware: HardwarePreference::Avoid,
            configuration: Vec::new(),
        }
    }

    fn sample(index: u64, value: u8, random_access: bool) -> EncodedVideoSample {
        EncodedVideoSample {
            presentation_index: FrameIndex(index),
            random_access,
            data: vec![value],
        }
    }

    fn value(frame: &VideoFrame) -> u8 {
        frame.planes[0].data[0]
    }

    #[test]
    fn capability_distinguishes_codec_profile_configuration_and_hardware() {
        let factory = uncompressed_video_decoder_factory();
        assert!(factory.capability(&config()).is_supported());
        let mut candidate = config();
        candidate.codec = Codec::Av1;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::UnsupportedCodec
        );
        candidate = config();
        candidate.profile = CodecProfile::Av1Main;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::UnsupportedProfile
        );
        candidate = config();
        candidate.output_format = PixelFormat::Rgba8;
        assert!(matches!(
            factory.capability(&candidate),
            CodecSupport::InvalidConfiguration { .. }
        ));
        candidate = config();
        candidate.hardware = HardwarePreference::Require;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::HardwareUnavailable
        );

        candidate = config();
        candidate.output_format = PixelFormat::Rgba8;
        assert_eq!(
            factory
                .create(&candidate, &Limits::default())
                .err()
                .unwrap()
                .kind(),
            ErrorKind::InvalidInput
        );

        let restrictive = Limits {
            max_width: 0,
            ..Limits::default()
        };
        assert_eq!(
            factory
                .create(&config(), &restrictive)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::ResourceLimit
        );
    }

    #[test]
    fn conformance_vector_reorders_exact_frames_and_reuses_bounded_state() {
        let samples = vec![
            sample(0, 10, true),
            sample(3, 40, false),
            sample(1, 20, false),
            sample(2, 30, false),
            sample(4, 50, true),
            sample(6, 70, false),
            sample(5, 60, false),
        ];
        let factory = uncompressed_video_decoder_factory();
        let mut reader = ExactFrameReader::new(
            &factory,
            config(),
            samples,
            Limits {
                max_cached_frames: 3,
                ..Limits::default()
            },
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        assert_eq!(
            value(&reader.get(FrameIndex(1), &cancellation).unwrap()),
            20
        );
        let first = reader.statistics();
        assert_eq!(
            value(&reader.get(FrameIndex(2), &cancellation).unwrap()),
            30
        );
        assert_eq!(reader.statistics().resets, first.resets);
        assert_eq!(
            value(&reader.get(FrameIndex(3), &cancellation).unwrap()),
            40
        );
        assert!(reader.statistics().cache_hits > 0);
        assert_eq!(
            value(&reader.get(FrameIndex(5), &cancellation).unwrap()),
            60
        );
        assert_eq!(
            value(&reader.get(FrameIndex(0), &cancellation).unwrap()),
            10
        );
        assert!(reader.statistics().resets > first.resets);
        assert!(reader.cached_frames() <= 3);
    }

    #[test]
    fn malformed_packets_and_cancellation_are_reported() {
        let factory = uncompressed_video_decoder_factory();
        let mut reader = ExactFrameReader::new(
            &factory,
            config(),
            vec![EncodedVideoSample {
                presentation_index: FrameIndex(0),
                random_access: true,
                data: Vec::new(),
            }],
            Limits::default(),
        )
        .unwrap();
        assert_eq!(
            reader
                .get(FrameIndex(0), &CancellationToken::new())
                .unwrap_err()
                .kind(),
            ErrorKind::MalformedMedia
        );
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert_eq!(
            reader.get(FrameIndex(0), &cancelled).unwrap_err().kind(),
            ErrorKind::Cancelled
        );
    }

    #[test]
    fn random_access_decode_work_obeys_the_configured_bound() {
        let factory = uncompressed_video_decoder_factory();
        let mut reader = ExactFrameReader::new(
            &factory,
            config(),
            vec![sample(0, 1, true), sample(1, 2, false), sample(2, 3, false)],
            Limits {
                max_decode_samples_per_seek: 2,
                ..Limits::default()
            },
        )
        .unwrap();
        let error = reader
            .get(FrameIndex(2), &CancellationToken::new())
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);
        assert_eq!(reader.statistics().samples_submitted, 2);
    }

    struct DelayedFactory;
    impl VideoDecoderFactory for DelayedFactory {
        fn capability(&self, _: &VideoDecoderConfig) -> CodecSupport {
            CodecSupport::Supported {
                implementation: CodecImplementation::Software,
            }
        }
        fn create(
            &self,
            configuration: &VideoDecoderConfig,
            limits: &Limits,
        ) -> Result<Box<dyn VideoDecoder>> {
            Ok(Box::new(DelayedDecoder {
                configuration: configuration.clone(),
                limits: *limits,
                delayed: None,
            }))
        }
    }
    struct DelayedDecoder {
        configuration: VideoDecoderConfig,
        limits: Limits,
        delayed: Option<EncodedVideoSample>,
    }
    impl DelayedDecoder {
        fn decode(&self, sample: EncodedVideoSample) -> Result<DecodedVideoFrame> {
            Ok(DecodedVideoFrame {
                presentation_index: sample.presentation_index,
                frame: VideoFrame::new(
                    self.configuration.coded_dimensions,
                    PixelFormat::Gray8,
                    ColorRange::Full,
                    vec![Plane {
                        data: sample.data,
                        stride: 1,
                    }],
                    &self.limits,
                )?,
            })
        }
    }
    impl VideoDecoder for DelayedDecoder {
        fn submit(
            &mut self,
            sample: &EncodedVideoSample,
            cancellation: &CancellationToken,
        ) -> Result<Vec<DecodedVideoFrame>> {
            cancellation.check()?;
            let previous = self.delayed.replace(sample.clone());
            previous
                .map(|sample| self.decode(sample).map(|frame| vec![frame]))
                .unwrap_or(Ok(Vec::new()))
        }
        fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
            cancellation.check()?;
            self.delayed
                .take()
                .map(|sample| self.decode(sample).map(|frame| vec![frame]))
                .unwrap_or(Ok(Vec::new()))
        }
        fn reset(&mut self) -> Result<()> {
            self.delayed = None;
            Ok(())
        }
    }

    #[test]
    fn delayed_output_drains_and_reset_clears_state() {
        let mut reader = ExactFrameReader::new(
            &DelayedFactory,
            config(),
            vec![sample(0, 11, true), sample(1, 22, false)],
            Limits::default(),
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        assert_eq!(
            value(&reader.get(FrameIndex(1), &cancellation).unwrap()),
            22
        );
        assert_eq!(reader.statistics().drains, 1);
        reader.reset().unwrap();
        assert_eq!(reader.cached_frames(), 0);
        assert_eq!(
            value(&reader.get(FrameIndex(0), &cancellation).unwrap()),
            11
        );
    }

    struct MalformedOutputFactory;

    impl VideoDecoderFactory for MalformedOutputFactory {
        fn capability(&self, _: &VideoDecoderConfig) -> CodecSupport {
            CodecSupport::Supported {
                implementation: CodecImplementation::Software,
            }
        }

        fn create(
            &self,
            configuration: &VideoDecoderConfig,
            limits: &Limits,
        ) -> Result<Box<dyn VideoDecoder>> {
            let factory = UncompressedVideoDecoderFactory;
            factory.create(configuration, limits).map(|decoder| {
                Box::new(MalformedOutputDecoder {
                    decoder,
                    previous: None,
                }) as Box<dyn VideoDecoder>
            })
        }
    }

    struct MalformedOutputDecoder {
        decoder: Box<dyn VideoDecoder>,
        previous: Option<FrameIndex>,
    }

    impl VideoDecoder for MalformedOutputDecoder {
        fn submit(
            &mut self,
            sample: &EncodedVideoSample,
            cancellation: &CancellationToken,
        ) -> Result<Vec<DecodedVideoFrame>> {
            let mut output = self.decoder.submit(sample, cancellation)?;
            if let (Some(previous), Some(frame)) = (self.previous, output.first_mut()) {
                frame.presentation_index = previous;
            }
            self.previous = Some(sample.presentation_index);
            Ok(output)
        }

        fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
            self.decoder.drain(cancellation)
        }

        fn reset(&mut self) -> Result<()> {
            self.previous = None;
            self.decoder.reset()
        }
    }

    #[test]
    fn malformed_backend_output_is_rejected() {
        let mut reader = ExactFrameReader::new(
            &MalformedOutputFactory,
            config(),
            vec![sample(0, 1, true), sample(1, 2, false)],
            Limits::default(),
        )
        .unwrap();
        let error = reader
            .get(FrameIndex(1), &CancellationToken::new())
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MalformedMedia);
    }
}
