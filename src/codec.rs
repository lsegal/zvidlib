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
///
/// # zvidlib does not implement this trait
///
/// This crate ships **no audio encoder**, and that is a deliberate, recorded
/// decision (issue #174), not an oversight or an unfinished corner. `AudioEncoder`
/// exists as the seam that platform and browser backends fill; the only
/// implementations in the tree are the pass-through PCM doubles used by
/// `tests/indexed_mp4_output.rs` and `benches/audio_mux.rs`, which package sample
/// ranges into [`EncodedSample`]s without compressing anything.
///
/// The reasoning:
///
/// * A pure-Rust AAC-LC encoder is a large subsystem in its own right -- filter
///   bank, psychoacoustic model, quantization and rate control, bitstream writer
///   -- and its output quality, not its throughput, is what would have to be
///   defended. A mediocre encoder shipped under this crate's name is worse for
///   callers than no encoder at all, because it is harder to route around.
/// * Every target zvidlib runs on already has a good AAC encoder. Browsers expose
///   `WebCodecs` `AudioEncoder`; macOS has AudioToolbox and Windows has Media
///   Foundation. Delegating to any one of them is per-platform work that still
///   leaves other platforms uncovered, so it answers no portability question that
///   this trait does not already answer better.
/// * The crate's subject is frame-accurate video and synchronized audio *I/O*.
///   [`crate::MediaOutput`] can already mux an audio track given any
///   `AudioEncoder`, so the write path is complete up to the codec, and the codec
///   is the part callers are best positioned to choose.
///
/// If that calculus changes, adding an implementation is additive and breaks no
/// caller: everything downstream of this trait is already written against it.
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

    pub(crate) fn check(&self) -> Result<()> {
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
///
/// Requires `Send` so an [`ExactFrameReader`] (and the decoder it owns) can be moved to a
/// background thread, which native examples rely on to decode without stalling rendering.
pub trait VideoDecoder: Send {
    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>>;
    fn drain(&mut self, cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>>;
    fn reset(&mut self) -> Result<()>;

    /// Whether the caller wants the pictures the next samples decode to, or is only decoding
    /// them because a later frame references them.
    ///
    /// Reaching a frame in the middle of a long group of pictures means decoding every sample
    /// before it, and on the way there nothing looks at those pictures. What they cost is not
    /// the decoding: a hardware backend hands back NV12 and every one of these decoders then
    /// converts a whole picture to RGBA on the CPU and allocates the frame that holds it. On
    /// the bundled 1080p sample that conversion is 6.6 ms of the 9.6 ms a frame takes through
    /// VideoToolbox, so a seek that skips it for the frames it passes costs a third of what it
    /// did.
    ///
    /// A decoder must still *decode* a suppressed sample - the frames after it reference the
    /// picture - and must still account for its presentation identity; what it may skip is
    /// producing the [`DecodedVideoFrame`], which it reports by returning fewer frames than it
    /// was given samples.
    ///
    /// The default implementation ignores the hint, which is always correct: a decoder that
    /// keeps producing every frame is a decoder that is merely no faster. [`ExactFrameReader`]
    /// tracks what it was actually handed rather than what it asked for, so a backend that
    /// implements this and one that does not answer the same frames.
    fn set_output_wanted(&mut self, _wanted: bool) {}
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
    /// Of those, the ones submitted only so a later frame could reference them, with the
    /// decoder told not to produce their pictures.
    pub samples_skipped: u64,
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
    /// Frames whose sample was submitted while the decoder was told nothing wanted its picture,
    /// so the decoder passed through them without producing one. Reaching one again needs the
    /// same reset an evicted published frame does.
    suppressed_since_reset: HashSet<FrameIndex>,
    /// Frames whose sample has been submitted and whose picture has not come back yet, because a
    /// reordering decoder holds one until the samples after it arrive. They are the frames a
    /// decoder may drop the moment output stops being wanted, so turning it off writes them off
    /// with the rest.
    in_flight_since_reset: HashSet<FrameIndex>,
    /// What the decoder was last told, so a walk toggles it once at the target rather than
    /// on every sample.
    output_wanted: bool,
    next_decode_position: Option<usize>,
    limits: Limits,
    statistics: DecodeStatistics,
}

/// What a caller wants out of a request, which is what decides whether the reader keeps the
/// frames behind the one it was asked for.
///
/// The frames a walk passes are decoded either way - a picture cannot be decoded without its
/// references - so this only decides which of them are converted to RGBA, which is the part of
/// a walk that costs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Request {
    /// Somewhere to be: keep the frames behind it that the cache can hold, so coming back to
    /// one of them is free.
    Destination,
    /// Somewhere to pass through: keep only the frame asked for, and the reordered frames at or
    /// after it that the next step needs anyway.
    Step,
}

impl Request {
    /// How far behind the target this request keeps pictures, in presentation frames.
    fn cache_tail(self, limits: &Limits) -> u64 {
        match self {
            Self::Destination => u64::from(limits.max_cached_frames),
            Self::Step => 0,
        }
    }
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
            suppressed_since_reset: HashSet::new(),
            in_flight_since_reset: HashSet::new(),
            output_wanted: true,
            next_decode_position: None,
            limits,
            statistics: DecodeStatistics::default(),
        })
    }

    /// Returns exactly the requested presentation frame, and the frames behind it that the
    /// cache can hold, so a request that comes back for one of those is answered for nothing.
    ///
    /// This is the destination of a seek. A walk that is going to keep walking forwards should
    /// ask for its intermediate frames with [`get_step`] instead, which converts the frame it
    /// was asked for and nothing else.
    ///
    /// [`get_step`]: Self::get_step
    pub fn get(
        &mut self,
        presentation_index: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<VideoFrame> {
        self.get_request(presentation_index, Request::Destination, cancellation)
    }

    /// Returns exactly the requested presentation frame and converts nothing else.
    ///
    /// A step is a picture on the way to somewhere, not somewhere: the caller is going to ask
    /// for a frame further forward next, so the frames behind this one that [`get`] would keep
    /// are converted for a request that never comes. That tail is bounded per call and so is
    /// affordable for one seek, but a walk that publishes as it goes pays it once per published
    /// picture, and once the stride is shorter than the tail the tails overlap and the walk
    /// converts every picture it passes rather than the few it publishes (issue #402).
    ///
    /// Frames at or after the target in presentation order are still kept, exactly as [`get`]
    /// keeps them: on a stream with B-pictures those are the reordered frames the next request
    /// asks for, and dropping them would reset the decoder and re-walk the group of pictures.
    ///
    /// [`get`]: Self::get
    pub fn get_step(
        &mut self,
        presentation_index: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<VideoFrame> {
        self.get_request(presentation_index, Request::Step, cancellation)
    }

    fn get_request(
        &mut self,
        presentation_index: FrameIndex,
        request: Request,
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
        // A decoder with output reordering (e.g. hierarchical B-frames) may need to be fed
        // samples *past* `target_position` before the reordered frame at `target_position` is
        // actually emitted. So `next_decode_position > target_position` does not by itself mean
        // the frame is unreachable without a reset: it may simply still be buffered inside the
        // decoder, pending release as more samples are submitted. The only case that truly
        // requires a reset is when the frame was already published once and evicted from the
        // cache, since a decoder must never be asked to emit the same presentation frame twice
        // without an intervening reset (see `publish`).
        //
        // A frame the decoder walked past without producing is in the same position as one that
        // was published and evicted: the decoder will not emit it again, so only a reset can
        // reach it.
        let can_reuse = self
            .next_decode_position
            .is_some_and(|position| position >= random_access_position)
            && !self.published_since_reset.contains(&presentation_index)
            && !self.suppressed_since_reset.contains(&presentation_index);
        if !can_reuse {
            self.decoder.reset()?;
            self.statistics.resets = self.statistics.resets.saturating_add(1);
            self.published_since_reset.clear();
            self.suppressed_since_reset.clear();
            self.in_flight_since_reset.clear();
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
                self.set_output_wanted(true);
                self.drain_internal(cancellation)?;
                break;
            }
            // Nothing looks at a picture decoded on the way to the target, and skipping the
            // colour conversion it would otherwise pay is most of what a long walk costs. Two
            // kinds of frame are kept anyway, both because the request after this one is very
            // likely to be for them.
            //
            // The first is anything at or after the target in *presentation* order. A walk
            // arrives at its target from behind and carries on forwards, and deciding this by
            // decode position instead discards exactly the reordered frames a stream with
            // B-pictures asks for next: on the bundled sample every fourth request would then
            // reset the decoder and walk the whole group of pictures again.
            //
            // The second is the frames immediately behind the target, as many as the cache can
            // hold, and that one belongs to the request rather than to the reader. Walking here
            // fills the cache with them as a side effect, and that is what makes stepping
            // backwards a frame at a time cost nothing after a seek; skipping them would leave
            // the cache empty behind the target and turn every backward step into another walk
            // from the random-access point. For a destination they are a bounded charge - the
            // last `max_cached_frames` conversions of a walk however long - against a saving
            // that grows with the distance. For a step they are a charge with no saving behind
            // it at all: the caller is walking forwards and will never ask for them, and paying
            // the tail once per step is what made a 150 ms preview cadence convert 765 of the
            // bundled sample's 768 pictures (issue #402).
            let cache_tail = request.cache_tail(&self.limits);
            let wanted = position >= target_position
                || self.samples[position].presentation_index.0
                    >= presentation_index.0.saturating_sub(cache_tail);
            self.set_output_wanted(wanted);
            let suppressed = !self.output_wanted;
            let outputs = self.decoder.submit(&self.samples[position], cancellation)?;
            self.statistics.samples_submitted = self.statistics.samples_submitted.saturating_add(1);
            if suppressed {
                self.statistics.samples_skipped = self.statistics.samples_skipped.saturating_add(1);
                self.suppressed_since_reset
                    .insert(self.samples[position].presentation_index);
            } else {
                self.in_flight_since_reset
                    .insert(self.samples[position].presentation_index);
            }
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
        self.set_output_wanted(true);
        self.drain_internal(cancellation)
    }

    /// Tells the decoder whether the next samples' pictures are wanted, when that has changed.
    ///
    /// Turning output off writes off the frames the decoder is still holding as well as the ones
    /// it is about to be handed. A reordering decoder releases a picture several samples after
    /// the one that carried it, so a frame submitted while output was wanted can come out - and
    /// be dropped - during the suppressed stretch that follows. Not writing those off is a
    /// decoder that will never produce them and a reader that still believes it can ask, which
    /// ends in `decoder did not produce the requested presentation frame`. A picture that does
    /// arrive after all takes itself back off the list in [`publish`].
    ///
    /// [`publish`]: Self::publish
    fn set_output_wanted(&mut self, wanted: bool) {
        if self.output_wanted == wanted {
            return;
        }
        if !wanted {
            for frame in self.in_flight_since_reset.drain() {
                self.suppressed_since_reset.insert(frame);
            }
        }
        self.decoder.set_output_wanted(wanted);
        self.output_wanted = wanted;
    }

    /// Clears decoder, reorder, and frame-cache state.
    pub fn reset(&mut self) -> Result<()> {
        self.set_output_wanted(true);
        self.decoder.reset()?;
        self.statistics.resets = self.statistics.resets.saturating_add(1);
        self.next_decode_position = None;
        self.cache.clear();
        self.lru.clear();
        self.published_since_reset.clear();
        self.suppressed_since_reset.clear();
        self.in_flight_since_reset.clear();
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
            // A decoder that ignores the output hint, or one that only emits a suppressed
            // sample's picture later, hands back a frame this reader had written off. What it
            // was handed is what counts, so it is a cached frame again rather than a reset.
            self.suppressed_since_reset
                .remove(&output.presentation_index);
            self.in_flight_since_reset
                .remove(&output.presentation_index);
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
    use std::sync::Mutex;

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

    /// Records what each sample was asked to produce, so a test can see the hint the reader
    /// actually gave rather than only its effect.
    struct HintFactory {
        wanted: Arc<Mutex<Vec<(u64, bool)>>>,
    }

    struct HintDecoder {
        configuration: VideoDecoderConfig,
        limits: Limits,
        wanted: Arc<Mutex<Vec<(u64, bool)>>>,
        output_wanted: bool,
    }

    impl VideoDecoderFactory for HintFactory {
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
            Ok(Box::new(HintDecoder {
                configuration: configuration.clone(),
                limits: *limits,
                wanted: Arc::clone(&self.wanted),
                output_wanted: true,
            }))
        }
    }

    impl VideoDecoder for HintDecoder {
        fn submit(
            &mut self,
            sample: &EncodedVideoSample,
            cancellation: &CancellationToken,
        ) -> Result<Vec<DecodedVideoFrame>> {
            cancellation.check()?;
            self.wanted
                .lock()
                .expect("hint log poisoned")
                .push((sample.presentation_index.0, self.output_wanted));
            if !self.output_wanted {
                return Ok(Vec::new());
            }
            Ok(vec![DecodedVideoFrame {
                presentation_index: sample.presentation_index,
                frame: VideoFrame::new(
                    self.configuration.coded_dimensions,
                    PixelFormat::Gray8,
                    ColorRange::Full,
                    vec![Plane {
                        data: sample.data.clone(),
                        stride: 1,
                    }],
                    &self.limits,
                )?,
            }])
        }

        fn drain(&mut self, _: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
            Ok(Vec::new())
        }

        fn reset(&mut self) -> Result<()> {
            Ok(())
        }

        fn set_output_wanted(&mut self, wanted: bool) {
            self.output_wanted = wanted;
        }
    }

    /// Ten frames behind one random-access point, as the bundled sample's single group of
    /// pictures is, with `value == index * 10`.
    fn single_group_of_pictures() -> Vec<EncodedVideoSample> {
        (0..10)
            .map(|index| sample(index, (index * 10) as u8, index == 0))
            .collect()
    }

    /// A cache small enough that a walk over ten frames is longer than the tail the reader
    /// keeps behind its target; the default cache would hold the whole group of pictures.
    fn small_cache() -> Limits {
        Limits {
            max_cached_frames: 2,
            ..Limits::default()
        }
    }

    #[test]
    fn walking_to_a_distant_frame_does_not_ask_for_the_pictures_it_passes() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let factory = HintFactory {
            wanted: Arc::clone(&log),
        };
        let mut reader = ExactFrameReader::new(
            &factory,
            config(),
            single_group_of_pictures(),
            small_cache(),
        )
        .unwrap();
        let cancellation = CancellationToken::new();

        assert_eq!(
            value(&reader.get(FrameIndex(8), &cancellation).unwrap()),
            80
        );
        assert_eq!(
            *log.lock().unwrap(),
            vec![
                (0, false),
                (1, false),
                (2, false),
                (3, false),
                (4, false),
                (5, false),
                (6, true),
                (7, true),
                (8, true),
            ],
            "only the target and the tail the cache can hold are produced"
        );
        let statistics = reader.statistics();
        assert_eq!(statistics.samples_submitted, 9);
        assert_eq!(statistics.samples_skipped, 6);
        assert_eq!(statistics.resets, 1, "the walk itself is one decode");
    }

    #[test]
    fn a_frame_the_walk_went_past_is_decoded_again_rather_than_reported_missing() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let factory = HintFactory {
            wanted: Arc::clone(&log),
        };
        let mut reader = ExactFrameReader::new(
            &factory,
            config(),
            single_group_of_pictures(),
            small_cache(),
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        reader.get(FrameIndex(8), &cancellation).unwrap();
        let resets = reader.statistics().resets;

        // Frame 4 was submitted with nothing wanting its picture, so this decoder will never
        // emit it again. The reader has to notice that and start over, exactly as it does for a
        // frame it published and then evicted.
        assert_eq!(
            value(&reader.get(FrameIndex(4), &cancellation).unwrap()),
            40
        );
        assert_eq!(reader.statistics().resets, resets + 1);
        // And having decoded it, it is a cached frame again rather than a second reset.
        assert_eq!(
            value(&reader.get(FrameIndex(4), &cancellation).unwrap()),
            40
        );
        assert_eq!(reader.statistics().resets, resets + 1);
    }

    #[test]
    fn a_reordered_frame_at_or_past_the_target_is_kept_for_the_request_that_follows() {
        // Decode order 0, 4, 2, 1, 3: reaching frame 4 means submitting samples whose own
        // frames are 2, 1 and 3, which is a walk *backwards* in presentation order. Skipping by
        // decode position would discard them and make every following request a fresh walk.
        let samples = vec![
            sample(0, 0, true),
            sample(4, 40, false),
            sample(2, 20, false),
            sample(1, 10, false),
            sample(3, 30, false),
            sample(8, 80, false),
            sample(6, 60, false),
            sample(5, 50, false),
            sample(7, 70, false),
        ];
        let log = Arc::new(Mutex::new(Vec::new()));
        let factory = HintFactory {
            wanted: Arc::clone(&log),
        };
        let mut reader = ExactFrameReader::new(&factory, config(), samples, small_cache()).unwrap();
        let cancellation = CancellationToken::new();

        assert_eq!(
            value(&reader.get(FrameIndex(4), &cancellation).unwrap()),
            40
        );
        let resets = reader.statistics().resets;
        for (index, expected) in [(1_u64, 10_u8), (2, 20), (3, 30)] {
            assert_eq!(
                value(&reader.get(FrameIndex(index), &cancellation).unwrap()),
                expected,
                "frame {index} is still available"
            );
        }
        assert_eq!(
            reader.statistics().resets,
            resets,
            "a frame after the target in decode order is not skipped"
        );
        assert_eq!(
            reader.statistics().samples_skipped,
            1,
            "only frame 0, the random-access point the walk starts from, is behind frame 4"
        );
    }

    #[test]
    fn a_walk_leaves_the_frames_behind_its_target_cached_so_stepping_back_is_free() {
        // Walking fills the cache with the frames just before the target as a side effect, and
        // that is what makes the example's previous-frame key cheap. Skipping those too would
        // send every backward step all the way back to the random-access point.
        let log = Arc::new(Mutex::new(Vec::new()));
        let factory = HintFactory {
            wanted: Arc::clone(&log),
        };
        let samples: Vec<_> = (0..40)
            .map(|index| sample(index, index as u8, index == 0))
            .collect();
        let mut reader = ExactFrameReader::new(
            &factory,
            config(),
            samples,
            Limits {
                max_cached_frames: 4,
                ..Limits::default()
            },
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        reader.get(FrameIndex(32), &cancellation).unwrap();
        let after_walk = reader.statistics();
        assert_eq!(after_walk.samples_skipped, 28, "frames 0 to 27 are skipped");

        for index in [31_u64, 30, 29] {
            assert_eq!(
                value(&reader.get(FrameIndex(index), &cancellation).unwrap()),
                index as u8
            );
        }
        assert_eq!(
            reader.statistics().resets,
            after_walk.resets,
            "the tail the walk kept is in the cache, so stepping back decodes nothing"
        );
        assert_eq!(
            reader.statistics().samples_submitted,
            after_walk.samples_submitted
        );
    }

    #[test]
    fn a_step_converts_its_own_picture_and_no_tail_behind_it() {
        // The charge issue #402 is about. A destination keeps the frames behind it because the
        // request after it is very likely to be for one of them; a step is passed through on
        // the way somewhere else, so those frames are converted for a request that never comes,
        // and a walk that steps more often than the tail is long converts everything it passes.
        let log = Arc::new(Mutex::new(Vec::new()));
        let factory = HintFactory {
            wanted: Arc::clone(&log),
        };
        let samples: Vec<_> = (0..40)
            .map(|index| sample(index, index as u8, index == 0))
            .collect();
        let mut reader = ExactFrameReader::new(
            &factory,
            config(),
            samples,
            Limits {
                max_cached_frames: 4,
                ..Limits::default()
            },
        )
        .unwrap();
        let cancellation = CancellationToken::new();

        assert_eq!(
            value(&reader.get_step(FrameIndex(32), &cancellation).unwrap()),
            32
        );
        let after_step = reader.statistics();
        assert_eq!(
            after_step.samples_skipped, 32,
            "frames 0 to 31 are all passed through, tail included"
        );
        assert_eq!(
            reader.cached_frames(),
            1,
            "only the picture the step asked for was converted"
        );
    }

    #[test]
    fn a_walk_of_steps_converts_what_it_asks_for_rather_than_what_it_passes() {
        // The shape of a drag's preview walk: repeated forward steps towards a far target. Each
        // step used to pay its own cache tail, and with a stride shorter than the tail the tails
        // overlapped and covered the whole span. As steps, the walk converts one picture per
        // step however long the span is.
        let log = Arc::new(Mutex::new(Vec::new()));
        let factory = HintFactory {
            wanted: Arc::clone(&log),
        };
        let samples: Vec<_> = (0..64)
            .map(|index| sample(index, index as u8, index == 0))
            .collect();
        let mut reader = ExactFrameReader::new(
            &factory,
            config(),
            samples,
            Limits {
                max_cached_frames: 8,
                ..Limits::default()
            },
        )
        .unwrap();
        let cancellation = CancellationToken::new();

        // A stride of 4 against a tail of 8: every frame in the span is within some step's tail.
        let steps: Vec<u64> = (4..60).step_by(4).collect();
        for step in &steps {
            reader.get_step(FrameIndex(*step), &cancellation).unwrap();
        }
        let converted =
            reader.statistics().samples_submitted - reader.statistics().samples_skipped;
        assert_eq!(
            converted,
            steps.len() as u64,
            "one conversion per published step, not one per frame passed"
        );
        assert_eq!(
            reader.statistics().resets,
            1,
            "only the cold start resets; a forward walk of steps never starts over again"
        );
    }

    #[test]
    fn a_step_walk_that_ends_in_a_destination_still_leaves_a_cache_tail_to_step_back_through() {
        // What the request split must not cost: a committed scrub is a walk of steps that ends
        // at the frame under the pointer, and stepping backwards from there is still expected
        // to come out of the cache rather than walking from the random-access point again.
        let log = Arc::new(Mutex::new(Vec::new()));
        let factory = HintFactory {
            wanted: Arc::clone(&log),
        };
        let samples: Vec<_> = (0..40)
            .map(|index| sample(index, index as u8, index == 0))
            .collect();
        let mut reader = ExactFrameReader::new(
            &factory,
            config(),
            samples,
            Limits {
                max_cached_frames: 4,
                ..Limits::default()
            },
        )
        .unwrap();
        let cancellation = CancellationToken::new();

        for step in [8_u64, 16, 24] {
            reader.get_step(FrameIndex(step), &cancellation).unwrap();
        }
        reader.get(FrameIndex(32), &cancellation).unwrap();
        let after_walk = reader.statistics();

        for index in [31_u64, 30, 29] {
            assert_eq!(
                value(&reader.get(FrameIndex(index), &cancellation).unwrap()),
                index as u8
            );
        }
        assert_eq!(
            reader.statistics().resets,
            after_walk.resets,
            "the destination kept its tail, so stepping back decodes nothing"
        );
        assert_eq!(
            reader.statistics().samples_submitted,
            after_walk.samples_submitted
        );
    }

    /// A decoder that releases each picture one sample late, and drops whatever it is holding
    /// when nothing wants its output - which is what a reordering decoder does.
    struct LateFactory;

    struct LateDecoder {
        configuration: VideoDecoderConfig,
        limits: Limits,
        held: Option<EncodedVideoSample>,
        output_wanted: bool,
    }

    impl VideoDecoderFactory for LateFactory {
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
            Ok(Box::new(LateDecoder {
                configuration: configuration.clone(),
                limits: *limits,
                held: None,
                output_wanted: true,
            }))
        }
    }

    impl LateDecoder {
        fn release(&self, sample: EncodedVideoSample) -> Result<Vec<DecodedVideoFrame>> {
            if !self.output_wanted {
                return Ok(Vec::new());
            }
            Ok(vec![DecodedVideoFrame {
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
            }])
        }
    }

    impl VideoDecoder for LateDecoder {
        fn submit(
            &mut self,
            sample: &EncodedVideoSample,
            cancellation: &CancellationToken,
        ) -> Result<Vec<DecodedVideoFrame>> {
            cancellation.check()?;
            match self.held.replace(sample.clone()) {
                Some(previous) => self.release(previous),
                None => Ok(Vec::new()),
            }
        }

        fn drain(&mut self, _: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
            match self.held.take() {
                Some(held) => self.release(held),
                None => Ok(Vec::new()),
            }
        }

        fn reset(&mut self) -> Result<()> {
            self.held = None;
            Ok(())
        }

        fn set_output_wanted(&mut self, wanted: bool) {
            self.output_wanted = wanted;
        }
    }

    #[test]
    fn a_frame_still_inside_a_reordering_decoder_when_output_stops_is_written_off_too() {
        // The frame a decoder is holding when output stops being wanted comes out during the
        // suppressed stretch and is dropped, even though its own sample was submitted while
        // output was still wanted. A reader that does not write it off believes it can still be
        // asked for and reports the decoder as malformed when it never arrives.
        let mut reader = ExactFrameReader::new(
            &LateFactory,
            config(),
            single_group_of_pictures(),
            small_cache(),
        )
        .unwrap();
        let cancellation = CancellationToken::new();

        // Reaching frame 4 means submitting frame 5's sample to release it, so the decoder is
        // left holding frame 5 - which it releases, and drops, on the first suppressed submit of
        // the request that follows.
        assert_eq!(
            value(&reader.get(FrameIndex(4), &cancellation).unwrap()),
            40
        );
        assert_eq!(
            value(&reader.get(FrameIndex(9), &cancellation).unwrap()),
            90
        );
        assert_eq!(
            value(&reader.get(FrameIndex(5), &cancellation).unwrap()),
            50,
            "the frame that was dropped in flight is decoded again, not reported missing"
        );
    }

    #[test]
    fn a_decoder_that_ignores_the_hint_still_answers_every_frame() {
        // `set_output_wanted` is advisory: `uncompressed_video_decoder_factory` does not
        // implement it, and the reader has to be no less correct for that.
        let factory = uncompressed_video_decoder_factory();
        let mut reader = ExactFrameReader::new(
            &factory,
            config(),
            single_group_of_pictures(),
            small_cache(),
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        reader.get(FrameIndex(8), &cancellation).unwrap();
        assert_eq!(reader.statistics().samples_skipped, 6);
        for index in 0..9_u64 {
            assert_eq!(
                value(&reader.get(FrameIndex(index), &cancellation).unwrap()),
                (index * 10) as u8,
                "frame {index} came back from the decoder that ignored the hint"
            );
        }
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

    #[test]
    fn sequential_playback_through_a_pipelined_decoder_does_not_reset_every_frame() {
        // `DelayedDecoder` only emits a frame's output on the *next* `submit` call (or on
        // `drain`), modeling the one-frame pipeline latency a real reordering decoder exhibits.
        // Even though `next_decode_position` therefore always runs one step ahead of the frame
        // a straightforward sequential `get(0), get(1), get(2), ...` walk is actually waiting
        // on, that lag must not force a full reset-and-redecode-from-keyframe on every call.
        let mut reader = ExactFrameReader::new(
            &DelayedFactory,
            config(),
            vec![
                sample(0, 11, true),
                sample(1, 22, false),
                sample(2, 33, false),
                sample(3, 44, false),
                sample(4, 55, false),
            ],
            Limits::default(),
        )
        .unwrap();
        let cancellation = CancellationToken::new();

        assert_eq!(
            value(&reader.get(FrameIndex(0), &cancellation).unwrap()),
            11
        );
        assert_eq!(reader.statistics().resets, 1);

        for (index, expected) in [(1, 22_u8), (2, 33), (3, 44), (4, 55)] {
            let frame = reader.get(FrameIndex(index), &cancellation).unwrap();
            assert_eq!(value(&frame), expected);
            assert_eq!(
                reader.statistics().resets,
                1,
                "sequential forward playback must not reset once decoding has started"
            );
        }
        assert_eq!(reader.statistics().samples_submitted, 5);
        assert_eq!(reader.statistics().drains, 1);
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
