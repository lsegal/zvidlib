//! Audio-clock-driven playback shared by native and browser adapters.
//!
//! Natively, a [`PlaybackController`] reads its sources synchronously: an
//! [`crate::ExactFrameReader`] and [`crate::AudioSampleReader`] over owned samples, or over an
//! on-demand [`crate::TrackSampleProvider`] whose source answers immediately. The browser's main
//! thread cannot wait for a `fetch`, so there the sources are an [`OnDemandVideoSource`] and an
//! [`OnDemandAudioSource`], which read compressed samples only from what their
//! [`TrackSampleLoader`] has already loaded and report [`ErrorKind::WouldBlock`] for anything else.
//! The controller passes that error up without losing its place, and the caller awaits
//! [`PlaybackController::prefetch`] - which loads the run the current frame and the scheduling
//! window need, plus a readahead - before trying again (issue #672):
//!
//! ```ignore
//! match controller.present() {
//!     Err(error) if error.kind() == ErrorKind::WouldBlock => controller.prefetch().await?,
//!     result => draw(result?),
//! }
//! ```

use crate::io::{ByteSource, IoFuture};
use crate::{
    AudioBuffer, CancellationToken, Error, ErrorKind, FrameIndex, Result, SampleRange, Timeline,
    TrackSampleLoader, VideoFrame,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioOutputKind {
    Native,
    WebAudio,
}

/// Scheduling operations supplied by a native device or browser binding.
pub trait AudioOutputBackend {
    fn clock_samples(&self) -> u64;
    fn start(&mut self, media_sample: u64) -> Result<()>;
    fn schedule(&mut self, buffer: AudioBuffer, generation: u64) -> Result<()>;
    fn cancel_queued(&mut self, generation: u64) -> Result<()>;
    fn stop(&mut self) -> Result<()>;
}

/// Adapts a native audio-device backend to the shared playback contract.
pub struct NativeAudioOutput<B>(pub B);

/// Adapts a browser `AudioContext` backend to the shared playback contract.
pub struct WebAudioOutput<B>(pub B);

/// Native device and Web Audio implementations expose the same monotonic clock.
pub trait PlaybackAudioOutput {
    fn kind(&self) -> AudioOutputKind;
    fn clock_samples(&self) -> u64;
    fn start(&mut self, media_sample: u64) -> Result<()>;
    fn schedule(&mut self, buffer: AudioBuffer, generation: u64) -> Result<()>;
    fn cancel_queued(&mut self, generation: u64) -> Result<()>;
    fn stop(&mut self) -> Result<()>;
}

macro_rules! output_adapter {
    ($adapter:ident, $kind:expr) => {
        impl<B: AudioOutputBackend> PlaybackAudioOutput for $adapter<B> {
            fn kind(&self) -> AudioOutputKind {
                $kind
            }
            fn clock_samples(&self) -> u64 {
                self.0.clock_samples()
            }
            fn start(&mut self, media_sample: u64) -> Result<()> {
                self.0.start(media_sample)
            }
            fn schedule(&mut self, buffer: AudioBuffer, generation: u64) -> Result<()> {
                self.0.schedule(buffer, generation)
            }
            fn cancel_queued(&mut self, generation: u64) -> Result<()> {
                self.0.cancel_queued(generation)
            }
            fn stop(&mut self) -> Result<()> {
                self.0.stop()
            }
        }
    };
}

output_adapter!(NativeAudioOutput, AudioOutputKind::Native);
output_adapter!(WebAudioOutput, AudioOutputKind::WebAudio);

/// A boxed output, so a controller's output can be chosen at run time: a
/// device for an input with audio, or only a clock for one without.
impl<O: PlaybackAudioOutput + ?Sized> PlaybackAudioOutput for Box<O> {
    fn kind(&self) -> AudioOutputKind {
        (**self).kind()
    }
    fn clock_samples(&self) -> u64 {
        (**self).clock_samples()
    }
    fn start(&mut self, media_sample: u64) -> Result<()> {
        (**self).start(media_sample)
    }
    fn schedule(&mut self, buffer: AudioBuffer, generation: u64) -> Result<()> {
        (**self).schedule(buffer, generation)
    }
    fn cancel_queued(&mut self, generation: u64) -> Result<()> {
        (**self).cancel_queued(generation)
    }
    fn stop(&mut self) -> Result<()> {
        (**self).stop()
    }
}

pub trait PlaybackAudioSource {
    fn sample_rate(&self) -> u32;
    fn presentation_length(&self) -> u64;
    fn read(
        &mut self,
        range: crate::SampleRange,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer>;
    fn reset(&mut self) -> Result<()>;
}

pub trait PlaybackVideoSource {
    fn get_exact(
        &mut self,
        frame: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<VideoFrame>;
    fn reset(&mut self) -> Result<()>;
}

/// A [`PlaybackVideoSource`] whose reads report [`ErrorKind::WouldBlock`] until what they need
/// has been loaded, and which can load it without blocking.
pub trait PrefetchVideoSource: PlaybackVideoSource {
    /// Loads what [`PlaybackVideoSource::get_exact`] of `frame` reads, along with whatever an
    /// earlier read reported missing and a readahead for playing on from `frame`.
    fn prefetch(&mut self, frame: FrameIndex) -> IoFuture<'_, ()>;
}

/// A [`PlaybackAudioSource`] whose reads report [`ErrorKind::WouldBlock`] until what they need
/// has been loaded, and which can load it without blocking.
pub trait PrefetchAudioSource: PlaybackAudioSource {
    /// Loads what [`PlaybackAudioSource::read`] of `range` reads, along with whatever an earlier
    /// read reported missing and a readahead for playing on past `range`.
    fn prefetch(&mut self, range: SampleRange) -> IoFuture<'_, ()>;
}

/// Exact video frames decoded from compressed samples a [`TrackSampleLoader`] loads on demand,
/// for a byte source whose reads suspend, such as a browser `fetch`.
///
/// [`PlaybackVideoSource::get_exact`] never waits on the source: it reports
/// [`ErrorKind::WouldBlock`] when a sample it needs is not loaded yet, and
/// [`PrefetchVideoSource::prefetch`] loads it.
pub struct OnDemandVideoSource<S> {
    reader: crate::ExactFrameReader,
    loader: TrackSampleLoader<S>,
    readahead_samples: usize,
}

impl<S: ByteSource> OnDemandVideoSource<S> {
    /// `reader` must have been built with [`crate::ExactFrameReader::from_provider`] over
    /// `loader`'s [`TrackSampleLoader::sample_provider`]. A prefetch loads up to
    /// `readahead_samples` decode-order samples past the run the requested frame needs, as far
    /// as the loader's budget allows.
    pub fn new(
        reader: crate::ExactFrameReader,
        loader: TrackSampleLoader<S>,
        readahead_samples: usize,
    ) -> Self {
        Self {
            reader,
            loader,
            readahead_samples,
        }
    }

    pub fn reader(&self) -> &crate::ExactFrameReader {
        &self.reader
    }

    pub fn reader_mut(&mut self) -> &mut crate::ExactFrameReader {
        &mut self.reader
    }

    pub fn loader(&self) -> &TrackSampleLoader<S> {
        &self.loader
    }
}

impl<S: ByteSource> PlaybackVideoSource for OnDemandVideoSource<S> {
    fn get_exact(
        &mut self,
        frame: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<VideoFrame> {
        self.reader.get(frame, cancellation)
    }

    fn reset(&mut self) -> Result<()> {
        self.loader.clear_missing();
        self.reader.reset()
    }
}

impl<S: ByteSource> PrefetchVideoSource for OnDemandVideoSource<S> {
    fn prefetch(&mut self, frame: FrameIndex) -> IoFuture<'_, ()> {
        Box::pin(async move {
            self.loader.load_missing().await?;
            let positions = self.reader.decode_positions_for(frame)?;
            self.loader.load(positions, self.readahead_samples).await
        })
    }
}

/// Exact audio ranges decoded from compressed packets a [`TrackSampleLoader`] loads on demand,
/// for a byte source whose reads suspend, such as a browser `fetch`.
///
/// [`PlaybackAudioSource::read`] never waits on the source: it reports
/// [`ErrorKind::WouldBlock`] when a packet it needs is not loaded yet, and
/// [`PrefetchAudioSource::prefetch`] loads it.
pub struct OnDemandAudioSource<D, S> {
    reader: crate::AudioSampleReader<D>,
    loader: TrackSampleLoader<S>,
    readahead_packets: usize,
}

impl<D: crate::AudioDecoder, S: ByteSource> OnDemandAudioSource<D, S> {
    /// `reader` must have been built with [`crate::AudioSampleReader::from_provider`] over
    /// `loader`'s [`TrackSampleLoader::audio_packet_provider`]. A prefetch loads up to
    /// `readahead_packets` packets past the run the requested range needs, as far as the
    /// loader's budget allows. The budget must pass [`TrackSampleLoader::check_audio_budget`]
    /// for the reader's provider and preroll count, or a read that resets the decoder may
    /// never find its packets loaded together.
    pub fn new(
        reader: crate::AudioSampleReader<D>,
        loader: TrackSampleLoader<S>,
        readahead_packets: usize,
    ) -> Self {
        Self {
            reader,
            loader,
            readahead_packets,
        }
    }

    pub fn reader(&self) -> &crate::AudioSampleReader<D> {
        &self.reader
    }

    pub fn reader_mut(&mut self) -> &mut crate::AudioSampleReader<D> {
        &mut self.reader
    }

    pub fn loader(&self) -> &TrackSampleLoader<S> {
        &self.loader
    }
}

impl<D: crate::AudioDecoder, S: ByteSource> PlaybackAudioSource for OnDemandAudioSource<D, S> {
    fn sample_rate(&self) -> u32 {
        self.reader.sample_rate()
    }

    fn presentation_length(&self) -> u64 {
        self.reader.presentation_length()
    }

    fn read(
        &mut self,
        range: SampleRange,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        self.reader.get_range(range, cancellation)
    }

    fn reset(&mut self) -> Result<()> {
        self.loader.clear_missing();
        self.reader.reset()
    }
}

impl<D: crate::AudioDecoder, S: ByteSource> PrefetchAudioSource for OnDemandAudioSource<D, S> {
    fn prefetch(&mut self, range: SampleRange) -> IoFuture<'_, ()> {
        Box::pin(async move {
            self.loader.load_missing().await?;
            for run in self.reader.packets_for_range(range)? {
                self.loader.load(run, self.readahead_packets).await?;
            }
            Ok(())
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlaybackOptions {
    pub schedule_ahead_samples: u64,
    pub preroll_samples: u64,
}

impl PlaybackOptions {
    pub fn for_sample_rate(sample_rate: u32) -> Self {
        Self {
            schedule_ahead_samples: u64::from(sample_rate) / 5,
            preroll_samples: u64::from(sample_rate) / 20,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Presentation {
    pub requested_frame: FrameIndex,
    pub frame: Option<FrameIndex>,
    pub finished: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexedPresentationTimeline {
    frame_audio_ranges: Vec<SampleRange>,
}

impl IndexedPresentationTimeline {
    pub fn new(frame_audio_ranges: Vec<SampleRange>) -> Result<Self> {
        if frame_audio_ranges.is_empty() {
            return Err(invalid(
                "playback timeline requires at least one video frame",
            ));
        }
        let mut previous_end = 0;
        for range in &frame_audio_ranges {
            if range.is_empty() || range.start < previous_end {
                return Err(invalid(
                    "playback frame audio ranges must be nonempty and ordered",
                ));
            }
            previous_end = range.end;
        }
        Ok(Self { frame_audio_ranges })
    }

    pub fn from_track(
        track: &crate::Track,
        audio_sample_rate: u32,
        limits: &crate::Limits,
    ) -> Result<Self> {
        if track.kind != crate::TrackKind::Video {
            return Err(invalid("indexed playback timeline requires a video track"));
        }
        if audio_sample_rate == 0 || audio_sample_rate > limits.max_sample_rate {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "playback audio sample rate is outside configured limits",
            ));
        }
        let mut ranges = Vec::with_capacity(track.presentation_order.len());
        for sample in track
            .presentation_order
            .iter()
            .map(|&index| &track.samples[index])
        {
            if sample.pts < 0 {
                return Err(Error::new(
                    ErrorKind::MalformedMedia,
                    "negative video PTS is unsupported for indexed playback",
                ));
            }
            let start =
                scale_to_audio_samples(sample.pts as u64, track.timescale, audio_sample_rate)?;
            let end_ticks = (sample.pts as u64)
                .checked_add(u64::from(sample.duration))
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::ResourceLimit,
                        "video presentation timing overflow",
                    )
                })?;
            let end = scale_to_audio_samples(end_ticks, track.timescale, audio_sample_rate)?;
            ranges.push(SampleRange::new(start, end)?);
        }
        Self::new(ranges)
    }

    pub fn audio_interval_for_frame(&self, frame: FrameIndex) -> Result<SampleRange> {
        let index = usize::try_from(frame.0)
            .map_err(|_| Error::new(ErrorKind::InvalidInput, "frame index is out of range"))?;
        self.frame_audio_ranges
            .get(index)
            .copied()
            .ok_or_else(|| invalid("presentation frame is not indexed"))
    }

    /// The audio sample the last frame's interval ends at, where the presentation ends.
    pub fn end_sample(&self) -> u64 {
        self.frame_audio_ranges.last().map_or(0, |range| range.end)
    }

    pub fn frame_for_audio_sample(&self, sample: u64) -> Result<FrameIndex> {
        let index = self
            .frame_audio_ranges
            .partition_point(|range| range.end <= sample);
        let index = index.min(self.frame_audio_ranges.len() - 1);
        Ok(FrameIndex(index as u64))
    }
}

/// A presentation timeline whose frames are indexed as playback reaches them
/// rather than all before it starts, such as a cued WebM's (issue #692).
///
/// Its frames need not be numbered densely: a [`FrameIndex`] is whatever
/// identity the timeline and the video source it is paired with agree on, such
/// as a frame's presentation time.
pub trait LazyPresentationTimeline {
    /// The frame presented at `sample`, as
    /// [`IndexedPresentationTimeline::frame_for_audio_sample`] answers it, or
    /// [`ErrorKind::WouldBlock`] while the part of the timeline holding it is
    /// not indexed, which [`Self::prepare`] indexes.
    fn frame_for_audio_sample(&self, sample: u64) -> Result<FrameIndex>;

    /// The audio samples `frame` is presented for, or
    /// [`ErrorKind::WouldBlock`] while its part of the timeline is not indexed.
    fn audio_interval_for_frame(&self, frame: FrameIndex) -> Result<SampleRange>;

    /// Indexes the part of the timeline holding `sample`.
    fn prepare(&mut self, sample: u64) -> IoFuture<'_, ()>;
}

/// How a [`PlaybackController`] maps the audio clock to video frames.
pub enum PresentationTimeline {
    /// A constant frame rate.
    Constant(Timeline),
    /// Every frame's interval, indexed up front.
    Indexed(IndexedPresentationTimeline),
    /// Frames indexed as playback reaches them.
    Lazy(Box<dyn LazyPresentationTimeline>),
}

impl From<Timeline> for PresentationTimeline {
    fn from(timeline: Timeline) -> Self {
        Self::Constant(timeline)
    }
}

impl From<IndexedPresentationTimeline> for PresentationTimeline {
    fn from(timeline: IndexedPresentationTimeline) -> Self {
        Self::Indexed(timeline)
    }
}

impl From<Box<dyn LazyPresentationTimeline>> for PresentationTimeline {
    fn from(timeline: Box<dyn LazyPresentationTimeline>) -> Self {
        Self::Lazy(timeline)
    }
}

impl PresentationTimeline {
    pub fn audio_interval_for_frame(&self, frame: FrameIndex) -> Result<SampleRange> {
        match self {
            Self::Constant(timeline) => timeline.audio_interval_for_frame(frame),
            Self::Indexed(timeline) => timeline.audio_interval_for_frame(frame),
            Self::Lazy(timeline) => timeline.audio_interval_for_frame(frame),
        }
    }

    pub fn frame_for_audio_sample(&self, sample: u64) -> Result<FrameIndex> {
        match self {
            Self::Constant(timeline) => timeline.frame_for_audio_sample(sample),
            Self::Indexed(timeline) => timeline.frame_for_audio_sample(sample),
            Self::Lazy(timeline) => timeline.frame_for_audio_sample(sample),
        }
    }

    /// Indexes the part of a lazy timeline holding `sample`; the others are
    /// indexed already.
    async fn prepare(&mut self, sample: u64) -> Result<()> {
        match self {
            Self::Lazy(timeline) => timeline.prepare(sample).await,
            Self::Constant(_) | Self::Indexed(_) => Ok(()),
        }
    }
}

/// Schedules audio ahead and selects exact video frames from its master clock.
pub struct PlaybackController<V, A, O> {
    video: V,
    audio: A,
    output: O,
    timeline: PresentationTimeline,
    options: PlaybackOptions,
    cancellation: CancellationToken,
    generation: u64,
    media_anchor: u64,
    clock_anchor: u64,
    queued_until: u64,
    last_presented: Option<FrameIndex>,
    playing: bool,
}

impl<V: PlaybackVideoSource, A: PlaybackAudioSource, O: PlaybackAudioOutput>
    PlaybackController<V, A, O>
{
    pub fn new(
        video: V,
        audio: A,
        output: O,
        timeline: Timeline,
        options: PlaybackOptions,
    ) -> Result<Self> {
        if audio.sample_rate() != timeline.audio_sample_rate() {
            return Err(invalid(
                "playback timeline and audio source sample rates do not match",
            ));
        }
        Self::new_with_timeline(video, audio, output, timeline, options)
    }

    pub fn new_with_indexed_timeline(
        video: V,
        audio: A,
        output: O,
        timeline: IndexedPresentationTimeline,
        options: PlaybackOptions,
    ) -> Result<Self> {
        Self::new_with_timeline(video, audio, output, timeline, options)
    }

    /// A controller on any [`PresentationTimeline`], such as a
    /// [`LazyPresentationTimeline`] whose frames are indexed as playback
    /// reaches them.
    pub fn new_with_timeline(
        video: V,
        audio: A,
        output: O,
        timeline: impl Into<PresentationTimeline>,
        options: PlaybackOptions,
    ) -> Result<Self> {
        let timeline = timeline.into();
        if options.schedule_ahead_samples == 0 {
            return Err(invalid("playback scheduling window must be nonzero"));
        }
        Ok(Self {
            video,
            audio,
            output,
            timeline,
            options,
            cancellation: CancellationToken::new(),
            generation: 0,
            media_anchor: 0,
            clock_anchor: 0,
            queued_until: 0,
            last_presented: None,
            playing: false,
        })
    }

    pub fn play(&mut self) -> Result<()> {
        if self.playing {
            return Ok(());
        }
        let queued_from = self.queued_until;
        self.clock_anchor = self.output.clock_samples();
        self.output.start(self.media_anchor)?;
        self.playing = true;
        match self.fill_audio() {
            // Leave the controller paused where it was, so a retry once the source has loaded
            // the audio schedules it from the play position rather than from wherever the
            // clock has run to by then (issue #695).
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                if self.queued_until != queued_from {
                    self.generation = self
                        .generation
                        .checked_add(1)
                        .ok_or_else(|| invalid("playback generation overflow"))?;
                    self.output.cancel_queued(self.generation)?;
                }
                self.output.stop()?;
                self.playing = false;
                self.queued_until = queued_from;
                Err(error)
            }
            result => result,
        }
    }

    pub fn pause(&mut self) -> Result<()> {
        if !self.playing {
            return Ok(());
        }
        self.media_anchor = self.current_sample();
        self.clock_anchor = self.output.clock_samples();
        self.queued_until = self.media_anchor;
        self.output.stop()?;
        self.cancellation.cancel();
        self.cancellation = CancellationToken::new();
        self.playing = false;
        Ok(())
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    pub fn current_frame(&mut self) -> Result<VideoFrame> {
        let frame = self.current_frame_index()?;
        let decoded = self.video.get_exact(frame, &self.cancellation)?;
        self.last_presented = Some(frame);
        Ok(decoded)
    }

    pub fn current_frame_index(&self) -> Result<FrameIndex> {
        self.timeline.frame_for_audio_sample(self.current_sample())
    }

    /// The audio sample playback is on: where it was paused, or where the
    /// audio clock has run to while it plays.
    pub fn current_audio_sample(&self) -> u64 {
        self.current_sample()
    }

    /// The audio samples `frame` is presented for, on the controller's
    /// timeline.
    pub fn audio_interval_for_frame(&self, frame: FrameIndex) -> Result<SampleRange> {
        self.timeline.audio_interval_for_frame(frame)
    }

    /// Cancels old decode/scheduling work, resets both streams, prerolls, then resumes.
    pub fn seek(&mut self, frame: FrameIndex) -> Result<()> {
        let target = self.timeline.audio_interval_for_frame(frame)?.start;
        self.seek_to_sample(target)
    }

    /// [`Self::seek`] to an audio sample rather than a frame: playback goes on
    /// from `target`, presenting whichever frame is on screen there. A seek
    /// by time, which a timeline whose frames are indexed as playback reaches
    /// them can answer before it knows which frame that is (issue #692).
    pub fn seek_to_sample(&mut self, target: u64) -> Result<()> {
        if target > self.audio.presentation_length() {
            return Err(invalid(
                "playback seek exceeds the audio presentation duration",
            ));
        }
        self.cancellation.cancel();
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("playback generation overflow"))?;
        self.output.cancel_queued(self.generation)?;
        self.video.reset()?;
        self.audio.reset()?;
        self.cancellation = CancellationToken::new();
        self.media_anchor = target;
        self.clock_anchor = self.output.clock_samples();
        let preroll_start = target.saturating_sub(self.options.preroll_samples);
        if preroll_start < target {
            let preroll = crate::SampleRange::new(preroll_start, target)?;
            match self.audio.read(preroll, &self.cancellation) {
                // The preroll only warms the decoder up, and an audio reader decodes its own
                // preroll packets after a reset anyway, so an on-demand source that has not
                // loaded it yet does not hold the seek up.
                Ok(_) => {}
                Err(error) if error.kind() == ErrorKind::WouldBlock => {}
                Err(error) => return Err(error),
            }
        }
        self.queued_until = target;
        self.last_presented = None;
        if self.playing {
            self.output.start(target)?;
            match self.fill_audio() {
                // The seek itself is complete, and the next `present` or `pump_audio` tops the
                // queue up from where this stopped once the source has loaded it.
                Err(error) if error.kind() == ErrorKind::WouldBlock => {}
                result => result?,
            }
        }
        Ok(())
    }

    /// Swaps the audio source and its output for others, such as another
    /// language's track, keeping the frame playback is on and whether it is
    /// playing (issue #689). Returns the old source and output, stopped.
    ///
    /// `timeline` maps the video's frames onto the new source's samples, which
    /// may come at a different rate than the old one's. The video source is not
    /// reset, so the frame on screen stays valid and playback goes on from it
    /// without decoding its way back. While playing, audio resumes at the start
    /// of the current frame's interval.
    pub fn replace_audio(
        &mut self,
        audio: A,
        output: O,
        timeline: impl Into<PresentationTimeline>,
    ) -> Result<(A, O)> {
        let timeline = timeline.into();
        let frame = self.current_frame_index()?;
        let target = timeline.audio_interval_for_frame(frame)?.start;
        if target > audio.presentation_length() {
            return Err(invalid(
                "the replacement audio ends before the current frame",
            ));
        }
        let playing = self.playing;
        if playing {
            self.output.stop()?;
        }
        self.cancellation.cancel();
        self.cancellation = CancellationToken::new();
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("playback generation overflow"))?;
        let old_audio = std::mem::replace(&mut self.audio, audio);
        let old_output = std::mem::replace(&mut self.output, output);
        self.timeline = timeline;
        self.media_anchor = target;
        self.clock_anchor = self.output.clock_samples();
        self.queued_until = target;
        // Paused until the new output has started, should it fail to.
        self.playing = false;
        if playing {
            self.output.start(target)?;
            self.playing = true;
            match self.fill_audio() {
                // As after a seek: the next `present` or `pump_audio` tops the
                // queue up once the source has loaded what it needs.
                Err(error) if error.kind() == ErrorKind::WouldBlock => {}
                result => result?,
            }
        }
        Ok((old_audio, old_output))
    }

    pub fn present(&mut self) -> Result<(Presentation, Option<VideoFrame>)> {
        if !self.playing {
            return Err(Error::new(
                ErrorKind::InvalidState,
                "playback is not running",
            ));
        }
        self.fill_audio()?;
        let elapsed = self
            .output
            .clock_samples()
            .saturating_sub(self.clock_anchor);
        let media_sample = self.media_anchor.saturating_add(elapsed);
        if media_sample >= self.audio.presentation_length() {
            return Ok((
                Presentation {
                    requested_frame: self.timeline.frame_for_audio_sample(media_sample)?,
                    frame: None,
                    finished: true,
                },
                None,
            ));
        }
        let requested = self.timeline.frame_for_audio_sample(media_sample)?;
        if self.last_presented == Some(requested) {
            return Ok((
                Presentation {
                    requested_frame: requested,
                    frame: None,
                    finished: false,
                },
                None,
            ));
        }
        let frame = self.video.get_exact(requested, &self.cancellation)?;
        self.last_presented = Some(requested);
        Ok((
            Presentation {
                requested_frame: requested,
                frame: Some(requested),
                finished: false,
            },
            Some(frame),
        ))
    }

    /// Keeps the audio queue scheduled ahead without selecting or decoding a video frame.
    ///
    /// [`Self::present`] decodes the frame the audio clock currently calls for, which is exactly
    /// what a caller that is displaying something else - a scrub preview decoded elsewhere - does
    /// not want to pay for on its render thread. This tops up the same scheduling window
    /// `present` would have, so audio keeps playing across a scrub without the video decode.
    /// It is a no-op while paused.
    pub fn pump_audio(&mut self) -> Result<()> {
        if !self.playing {
            return Ok(());
        }
        self.fill_audio()
    }

    pub fn stop(&mut self) -> Result<()> {
        if self.playing {
            self.output.stop()?;
        }
        self.playing = false;
        self.cancellation.cancel();
        Ok(())
    }

    pub fn output_kind(&self) -> AudioOutputKind {
        self.output.kind()
    }

    pub fn video(&self) -> &V {
        &self.video
    }

    pub fn audio(&self) -> &A {
        &self.audio
    }

    fn current_sample(&self) -> u64 {
        if !self.playing {
            return self.media_anchor;
        }
        self.media_anchor
            .saturating_add(
                self.output
                    .clock_samples()
                    .saturating_sub(self.clock_anchor),
            )
            .min(self.audio.presentation_length())
    }

    fn fill_audio(&mut self) -> Result<()> {
        let elapsed = self
            .output
            .clock_samples()
            .saturating_sub(self.clock_anchor);
        let now = self.media_anchor.saturating_add(elapsed);
        let target = now
            .saturating_add(self.options.schedule_ahead_samples)
            .min(self.audio.presentation_length());
        if self.queued_until < now {
            // Do not enqueue audio whose clock deadline has already passed.
            self.queued_until = now;
        }
        if self.queued_until
            < self
                .media_anchor
                .saturating_sub(self.options.preroll_samples)
        {
            self.queued_until = self
                .media_anchor
                .saturating_sub(self.options.preroll_samples);
        }
        while self.queued_until < target {
            let end = self
                .queued_until
                .saturating_add(self.options.schedule_ahead_samples)
                .min(target);
            let range = crate::SampleRange::new(self.queued_until, end)?;
            let buffer = self.audio.read(range, &self.cancellation)?;
            self.output.schedule(buffer, self.generation)?;
            self.queued_until = end;
        }
        Ok(())
    }
}

impl<V: PrefetchVideoSource, A: PrefetchAudioSource, O: PlaybackAudioOutput>
    PlaybackController<V, A, O>
{
    /// Loads what the next [`Self::present`] reads - the frame the audio clock calls for now and
    /// the audio that tops the scheduling window up - or, while paused, what
    /// [`Self::current_frame`] and [`Self::play`] read, along with each source's readahead.
    ///
    /// This is the browser's way through playback: with on-demand sources, `present`, `play`
    /// and `current_frame` report [`ErrorKind::WouldBlock`] instead of waiting for a sample to
    /// arrive, leave the controller where it was, and succeed once this has loaded what they
    /// were missing. Calling it ahead of time, between frames, keeps them from reporting it at
    /// all. [`Self::seek`] itself never needs it: the only thing it reads is a preroll it can
    /// skip.
    pub async fn prefetch(&mut self) -> Result<()> {
        let now = self.current_sample();
        self.timeline.prepare(now).await?;
        let frame = self.timeline.frame_for_audio_sample(now)?;
        self.video.prefetch(frame).await?;
        let length = self.audio.presentation_length();
        let start = self.queued_until.max(now).min(length);
        let end = now
            .saturating_add(self.options.schedule_ahead_samples)
            .min(length);
        if start < end {
            self.audio.prefetch(SampleRange::new(start, end)?).await?;
        }
        Ok(())
    }
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

pub(crate) fn scale_to_audio_samples(
    value: u64,
    source_timescale: u32,
    audio_sample_rate: u32,
) -> Result<u64> {
    if source_timescale == 0 {
        return Err(Error::new(
            ErrorKind::MalformedMedia,
            "video track timescale must be nonzero",
        ));
    }
    let scaled = u128::from(value)
        .checked_mul(u128::from(audio_sample_rate))
        .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "video timing scale overflow"))?
        / u128::from(source_timescale);
    u64::try_from(scaled)
        .map_err(|_| Error::new(ErrorKind::ResourceLimit, "scaled video time overflows"))
}

impl PlaybackVideoSource for crate::ExactFrameReader {
    fn get_exact(
        &mut self,
        frame: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<VideoFrame> {
        self.get(frame, cancellation)
    }

    fn reset(&mut self) -> Result<()> {
        self.reset()
    }
}

impl<D: crate::AudioDecoder> PlaybackAudioSource for crate::AudioSampleReader<D> {
    fn sample_rate(&self) -> u32 {
        self.sample_rate()
    }

    fn presentation_length(&self) -> u64 {
        self.presentation_length()
    }

    fn read(
        &mut self,
        range: crate::SampleRange,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        self.get_range(range, cancellation)
    }

    fn reset(&mut self) -> Result<()> {
        self.reset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColorRange, Limits, PixelFormat, Plane, SampleRange, VideoDimensions};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct FixtureVideo {
        requested: Arc<Mutex<Vec<FrameIndex>>>,
        resets: Arc<Mutex<usize>>,
    }

    impl PlaybackVideoSource for FixtureVideo {
        fn get_exact(&mut self, frame: FrameIndex, _: &CancellationToken) -> Result<VideoFrame> {
            self.requested.lock().unwrap().push(frame);
            VideoFrame::new(
                VideoDimensions::new(1, 1, &Limits::default())?,
                PixelFormat::Rgba8,
                ColorRange::Full,
                vec![Plane {
                    data: vec![frame.0 as u8, 0, 0, 255],
                    stride: 4,
                }],
                &Limits::default(),
            )
        }

        fn reset(&mut self) -> Result<()> {
            *self.resets.lock().unwrap() += 1;
            Ok(())
        }
    }

    struct FixtureAudio {
        reads: Arc<Mutex<Vec<SampleRange>>>,
    }

    impl PlaybackAudioSource for FixtureAudio {
        fn sample_rate(&self) -> u32 {
            48_000
        }
        fn presentation_length(&self) -> u64 {
            48_000
        }
        fn read(&mut self, range: SampleRange, _: &CancellationToken) -> Result<AudioBuffer> {
            self.reads.lock().unwrap().push(range);
            AudioBuffer::new(
                range,
                48_000,
                1,
                vec![0.0; range.len() as usize],
                &Limits::default(),
            )
        }
        fn reset(&mut self) -> Result<()> {
            Ok(())
        }
    }

    struct FixtureBackend {
        clock: Arc<Mutex<u64>>,
        scheduled: Arc<Mutex<Vec<(SampleRange, u64)>>>,
        canceled: Arc<Mutex<Vec<u64>>>,
    }

    impl AudioOutputBackend for FixtureBackend {
        fn clock_samples(&self) -> u64 {
            *self.clock.lock().unwrap()
        }
        fn start(&mut self, _: u64) -> Result<()> {
            Ok(())
        }
        fn schedule(&mut self, buffer: AudioBuffer, generation: u64) -> Result<()> {
            self.scheduled
                .lock()
                .unwrap()
                .push((buffer.range, generation));
            Ok(())
        }
        fn cancel_queued(&mut self, generation: u64) -> Result<()> {
            self.canceled.lock().unwrap().push(generation);
            Ok(())
        }
        fn stop(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn exercise(kind: AudioOutputKind) {
        let requested = Arc::new(Mutex::new(Vec::new()));
        let resets = Arc::new(Mutex::new(0));
        let reads = Arc::new(Mutex::new(Vec::new()));
        let clock = Arc::new(Mutex::new(0));
        let scheduled = Arc::new(Mutex::new(Vec::new()));
        let canceled = Arc::new(Mutex::new(Vec::new()));
        let backend = FixtureBackend {
            clock: clock.clone(),
            scheduled: scheduled.clone(),
            canceled: canceled.clone(),
        };
        match kind {
            AudioOutputKind::Native => run_playback(
                NativeAudioOutput(backend),
                requested.clone(),
                resets.clone(),
                reads.clone(),
                clock.clone(),
            ),
            AudioOutputKind::WebAudio => run_playback(
                WebAudioOutput(backend),
                requested.clone(),
                resets.clone(),
                reads.clone(),
                clock.clone(),
            ),
        }
        assert_eq!(&*canceled.lock().unwrap(), &[1]);
        assert!(
            scheduled
                .lock()
                .unwrap()
                .iter()
                .all(|(_, generation)| *generation <= 1)
        );
    }

    fn run_playback<O: PlaybackAudioOutput>(
        output: O,
        requested: Arc<Mutex<Vec<FrameIndex>>>,
        resets: Arc<Mutex<usize>>,
        reads: Arc<Mutex<Vec<SampleRange>>>,
        clock: Arc<Mutex<u64>>,
    ) {
        let video = FixtureVideo {
            requested: requested.clone(),
            resets: resets.clone(),
        };
        let audio = FixtureAudio {
            reads: reads.clone(),
        };
        let timeline = Timeline::new(crate::FrameRate::new(30, 1).unwrap(), 48_000).unwrap();
        let mut playback = PlaybackController::new(
            video,
            audio,
            output,
            timeline,
            PlaybackOptions {
                schedule_ahead_samples: 3_200,
                preroll_samples: 800,
            },
        )
        .unwrap();

        playback.play().unwrap();
        *clock.lock().unwrap() = 3_300;
        let (presentation, frame) = playback.present().unwrap();
        assert_eq!(presentation.requested_frame, FrameIndex(2));
        assert_eq!(presentation.frame, Some(FrameIndex(2)));
        assert!(frame.is_some());
        assert_eq!(&*requested.lock().unwrap(), &[FrameIndex(2)]);
        assert!(reads.lock().unwrap().contains(&SampleRange {
            start: 3_300,
            end: 6_500
        }));

        playback.seek(FrameIndex(10)).unwrap();
        assert_eq!(*resets.lock().unwrap(), 1);
        assert!(reads.lock().unwrap().contains(&SampleRange {
            start: 15_200,
            end: 16_000
        }));
    }

    #[test]
    fn native_output_uses_audio_clock_and_exact_video_identity() {
        exercise(AudioOutputKind::Native);
    }

    #[test]
    fn web_audio_output_uses_the_same_seek_and_sync_contract() {
        exercise(AudioOutputKind::WebAudio);
    }

    #[test]
    fn pumping_audio_schedules_ahead_without_requesting_a_video_frame() {
        let requested = Arc::new(Mutex::new(Vec::new()));
        let reads = Arc::new(Mutex::new(Vec::new()));
        let clock = Arc::new(Mutex::new(0));
        let scheduled = Arc::new(Mutex::new(Vec::new()));
        let backend = FixtureBackend {
            clock: clock.clone(),
            scheduled: scheduled.clone(),
            canceled: Arc::new(Mutex::new(Vec::new())),
        };
        let timeline = Timeline::new(crate::FrameRate::new(30, 1).unwrap(), 48_000).unwrap();
        let mut playback = PlaybackController::new(
            FixtureVideo {
                requested: requested.clone(),
                resets: Arc::new(Mutex::new(0)),
            },
            FixtureAudio {
                reads: reads.clone(),
            },
            NativeAudioOutput(backend),
            timeline,
            PlaybackOptions {
                schedule_ahead_samples: 3_200,
                preroll_samples: 800,
            },
        )
        .unwrap();

        playback.play().unwrap();
        scheduled.lock().unwrap().clear();
        *clock.lock().unwrap() = 3_300;
        playback.pump_audio().unwrap();

        // The same window `present` would have queued, and not one decoded frame with it.
        assert!(reads.lock().unwrap().contains(&SampleRange {
            start: 3_300,
            end: 6_500
        }));
        assert!(!scheduled.lock().unwrap().is_empty());
        assert!(requested.lock().unwrap().is_empty());

        // Still nothing to schedule and nothing to decode once paused.
        playback.pause().unwrap();
        scheduled.lock().unwrap().clear();
        playback.pump_audio().unwrap();
        assert!(scheduled.lock().unwrap().is_empty());
        assert!(requested.lock().unwrap().is_empty());
    }

    #[test]
    fn pause_preserves_the_current_playback_position() {
        let requested = Arc::new(Mutex::new(Vec::new()));
        let reads = Arc::new(Mutex::new(Vec::new()));
        let clock = Arc::new(Mutex::new(0));
        let backend = FixtureBackend {
            clock: clock.clone(),
            scheduled: Arc::new(Mutex::new(Vec::new())),
            canceled: Arc::new(Mutex::new(Vec::new())),
        };
        let timeline = Timeline::new(crate::FrameRate::new(30, 1).unwrap(), 48_000).unwrap();
        let mut playback = PlaybackController::new(
            FixtureVideo {
                requested: requested.clone(),
                resets: Arc::new(Mutex::new(0)),
            },
            FixtureAudio { reads },
            NativeAudioOutput(backend),
            timeline,
            PlaybackOptions {
                schedule_ahead_samples: 3_200,
                preroll_samples: 800,
            },
        )
        .unwrap();

        playback.play().unwrap();
        *clock.lock().unwrap() = 3_300;
        playback.pause().unwrap();

        assert!(!playback.is_playing());
        assert_eq!(playback.current_frame_index().unwrap(), FrameIndex(2));
        let frame = playback.current_frame().unwrap();
        assert_eq!(frame.planes[0].data[0], 2);
        assert_eq!(&*requested.lock().unwrap(), &[FrameIndex(2)]);
    }

    #[test]
    fn seeking_while_paused_moves_the_position_without_resuming_output() {
        let requested = Arc::new(Mutex::new(Vec::new()));
        let reads = Arc::new(Mutex::new(Vec::new()));
        let clock = Arc::new(Mutex::new(0));
        let scheduled = Arc::new(Mutex::new(Vec::new()));
        let backend = FixtureBackend {
            clock: clock.clone(),
            scheduled: scheduled.clone(),
            canceled: Arc::new(Mutex::new(Vec::new())),
        };
        let timeline = Timeline::new(crate::FrameRate::new(30, 1).unwrap(), 48_000).unwrap();
        let mut playback = PlaybackController::new(
            FixtureVideo {
                requested: requested.clone(),
                resets: Arc::new(Mutex::new(0)),
            },
            FixtureAudio { reads },
            NativeAudioOutput(backend),
            timeline,
            PlaybackOptions {
                schedule_ahead_samples: 3_200,
                preroll_samples: 800,
            },
        )
        .unwrap();

        playback.play().unwrap();
        *clock.lock().unwrap() = 3_300;
        playback.pause().unwrap();
        scheduled.lock().unwrap().clear();
        requested.lock().unwrap().clear();

        // Frame stepping and timeline scrubbing while paused: the position moves and the exact
        // frame is decodable, but nothing is queued to the audio device until playback resumes.
        playback.seek(FrameIndex(7)).unwrap();
        assert!(!playback.is_playing());
        assert_eq!(playback.current_frame_index().unwrap(), FrameIndex(7));
        assert_eq!(playback.current_frame().unwrap().planes[0].data[0], 7);
        assert_eq!(&*requested.lock().unwrap(), &[FrameIndex(7)]);
        assert!(scheduled.lock().unwrap().is_empty());

        // Resuming picks up from the scrubbed position rather than where pause happened.
        playback.play().unwrap();
        assert!(playback.is_playing());
        assert_eq!(playback.current_frame_index().unwrap(), FrameIndex(7));
    }

    #[test]
    fn indexed_timeline_selects_video_by_audio_sample_ranges() {
        let timeline = IndexedPresentationTimeline::new(vec![
            SampleRange::new(0, 100).unwrap(),
            SampleRange::new(100, 250).unwrap(),
            SampleRange::new(250, 300).unwrap(),
        ])
        .unwrap();

        assert_eq!(
            timeline.audio_interval_for_frame(FrameIndex(1)).unwrap(),
            SampleRange {
                start: 100,
                end: 250
            }
        );
        assert_eq!(timeline.frame_for_audio_sample(0).unwrap(), FrameIndex(0));
        assert_eq!(timeline.frame_for_audio_sample(99).unwrap(), FrameIndex(0));
        assert_eq!(timeline.frame_for_audio_sample(100).unwrap(), FrameIndex(1));
        assert_eq!(timeline.frame_for_audio_sample(249).unwrap(), FrameIndex(1));
        assert_eq!(timeline.frame_for_audio_sample(250).unwrap(), FrameIndex(2));
        assert_eq!(timeline.frame_for_audio_sample(400).unwrap(), FrameIndex(2));
    }

    #[test]
    fn indexed_timeline_drives_playback_without_constant_frame_rate() {
        let requested = Arc::new(Mutex::new(Vec::new()));
        let resets = Arc::new(Mutex::new(0));
        let reads = Arc::new(Mutex::new(Vec::new()));
        let clock = Arc::new(Mutex::new(0));
        let scheduled = Arc::new(Mutex::new(Vec::new()));
        let backend = FixtureBackend {
            clock: clock.clone(),
            scheduled,
            canceled: Arc::new(Mutex::new(Vec::new())),
        };
        let timeline = IndexedPresentationTimeline::new(vec![
            SampleRange::new(0, 100).unwrap(),
            SampleRange::new(100, 250).unwrap(),
            SampleRange::new(250, 300).unwrap(),
        ])
        .unwrap();
        let mut playback = PlaybackController::new_with_indexed_timeline(
            FixtureVideo {
                requested: requested.clone(),
                resets,
            },
            FixtureAudio { reads },
            NativeAudioOutput(backend),
            timeline,
            PlaybackOptions {
                schedule_ahead_samples: 50,
                preroll_samples: 10,
            },
        )
        .unwrap();

        playback.play().unwrap();
        *clock.lock().unwrap() = 110;
        let (presentation, _) = playback.present().unwrap();

        assert_eq!(presentation.requested_frame, FrameIndex(1));
        assert_eq!(&*requested.lock().unwrap(), &[FrameIndex(1)]);
    }

    /// Issue #672: `PlaybackController` plays and seeks through on-demand
    /// sources whose byte source suspends on every read, and never waits on it.
    /// The same scenario runs natively and in the browser, where it is driven by
    /// the browser's own event loop on its single thread.
    mod on_demand {
        use super::*;
        use crate::io::MemorySource;
        use crate::{
            AudioSampleReader, AudioTrackTiming, Codec, CodecProfile, ColorRange,
            EncodedAudioSample, ExactFrameReader, HardwarePreference, PixelFormat,
            SampleDependency, Track, TrackKind, TrackSample, VideoDecoderConfig,
            uncompressed_video_decoder_factory,
        };
        use std::cell::Cell;
        use std::future::Future;
        use std::pin::Pin;
        use std::rc::Rc;
        use std::task::{Context, Poll};

        const FRAMES: usize = 24;
        const GROUP: usize = 8;
        const SAMPLE_RATE: u32 = 48_000;
        const SAMPLES_PER_FRAME: u64 = 2_000;
        const SAMPLES_PER_PACKET: u64 = 1_000;

        /// A source whose every read suspends once before completing, the way
        /// a `fetch` does.
        struct SuspendingSource {
            inner: MemorySource,
            reads: Rc<Cell<usize>>,
        }

        impl ByteSource for SuspendingSource {
            fn len(&self) -> Option<u64> {
                self.inner.len()
            }

            fn read_at<'a>(
                &'a self,
                offset: u64,
                destination: &'a mut [u8],
            ) -> IoFuture<'a, usize> {
                self.reads.set(self.reads.get() + 1);
                Box::pin(async move {
                    YieldOnce(false).await;
                    self.inner.read_at(offset, destination).await
                })
            }
        }

        struct YieldOnce(bool);

        impl Future for YieldOnce {
            type Output = ();

            fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
                if self.0 {
                    Poll::Ready(())
                } else {
                    self.0 = true;
                    context.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        }

        /// Each packet decodes to its first byte, its packet index, held for
        /// its whole interval.
        struct PacketIndexAudio;

        impl crate::AudioDecoder for PacketIndexAudio {
            fn decode(
                &mut self,
                sample: &EncodedAudioSample,
                _: &CancellationToken,
            ) -> Result<AudioBuffer> {
                AudioBuffer::new(
                    sample.decoded_range,
                    SAMPLE_RATE,
                    1,
                    vec![f32::from(sample.data[0]); sample.decoded_range.len() as usize],
                    &Limits::default(),
                )
            }

            fn reset(&mut self) -> Result<()> {
                Ok(())
            }
        }

        fn track(kind: TrackKind, codec: Codec, samples: Vec<TrackSample>) -> Track {
            Track {
                id: 1,
                kind,
                codec,
                timescale: SAMPLE_RATE,
                duration: FRAMES as u64 * SAMPLES_PER_FRAME,
                dimensions: None,
                channels: Some(1),
                sample_rate: Some(SAMPLE_RATE),
                language: None,
                decoder_config: Vec::new(),
                edits: Vec::new(),
                presentation_order: (0..samples.len()).collect(),
                samples,
                vorbis_packet_heads: Vec::new(),
            }
        }

        fn sample(offset: usize, size: u32, index: usize, is_sync: bool) -> TrackSample {
            TrackSample {
                offset: offset as u64,
                size,
                dts: index as u64,
                pts: index as i64,
                duration: 1,
                dependency: SampleDependency::INDEPENDENT,
                is_sync,
            }
        }

        /// A file interleaving 1x1 Gray8 video frames, each holding its own
        /// index, with the two audio packets that play under each.
        fn media() -> (Vec<u8>, Track, Track) {
            let mut bytes = Vec::new();
            let mut video = Vec::new();
            let mut audio = Vec::new();
            for frame in 0..FRAMES {
                video.push(sample(bytes.len(), 1, frame, frame % GROUP == 0));
                bytes.push(frame as u8);
                for _ in 0..SAMPLES_PER_FRAME / SAMPLES_PER_PACKET {
                    let packet = audio.len();
                    audio.push(sample(bytes.len(), 2, packet, true));
                    bytes.extend([packet as u8, 0]);
                }
            }
            (
                bytes,
                track(TrackKind::Video, Codec::UncompressedVideo, video),
                track(TrackKind::Audio, Codec::Aac, audio),
            )
        }

        async fn until_loaded<T, V, A, O>(
            playback: &mut PlaybackController<V, A, O>,
            mut step: impl FnMut(&mut PlaybackController<V, A, O>) -> Result<T>,
        ) -> T
        where
            V: PrefetchVideoSource,
            A: PrefetchAudioSource,
            O: PlaybackAudioOutput,
        {
            for _ in 0..64 {
                match step(playback) {
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        playback.prefetch().await.unwrap();
                    }
                    result => return result.unwrap(),
                }
            }
            panic!("playback still had not loaded what it needed");
        }

        type OnDemandPlayback = PlaybackController<
            OnDemandVideoSource<SuspendingSource>,
            OnDemandAudioSource<PacketIndexAudio, SuspendingSource>,
            WebAudioOutput<FixtureBackend>,
        >;

        /// A controller over [`media`] whose reads go through a [`SuspendingSource`]
        /// counting into `reads`, timed by `clock` and scheduling into `scheduled`.
        fn playback(
            reads: &Rc<Cell<usize>>,
            clock: &Arc<Mutex<u64>>,
            scheduled: &Arc<Mutex<Vec<(SampleRange, u64)>>>,
        ) -> OnDemandPlayback {
            let (bytes, video_track, audio_track) = media();
            let source = || SuspendingSource {
                inner: MemorySource::new(bytes.clone()),
                reads: reads.clone(),
            };
            // Budgets smaller than one group of pictures and one scheduling
            // window, so both are streamed through over several prefetches.
            let video_loader = TrackSampleLoader::new(video_track, source(), 4).unwrap();
            let video_reader = ExactFrameReader::from_provider(
                &uncompressed_video_decoder_factory(),
                VideoDecoderConfig {
                    codec: Codec::UncompressedVideo,
                    profile: CodecProfile::UncompressedGray8,
                    coded_dimensions: VideoDimensions::new(1, 1, &Limits::default()).unwrap(),
                    output_format: PixelFormat::Gray8,
                    color_range: ColorRange::Full,
                    hardware: HardwarePreference::Avoid,
                    configuration: Vec::new(),
                },
                Box::new(video_loader.sample_provider().unwrap()),
                Limits::default(),
            )
            .unwrap();
            let audio_loader = TrackSampleLoader::new(audio_track, source(), 8).unwrap();
            let decoded_ranges = (0..FRAMES as u64 * 2)
                .map(|packet| {
                    SampleRange::new(
                        packet * SAMPLES_PER_PACKET,
                        (packet + 1) * SAMPLES_PER_PACKET,
                    )
                    .unwrap()
                })
                .collect();
            let audio_reader = AudioSampleReader::from_provider(
                PacketIndexAudio,
                Box::new(audio_loader.audio_packet_provider(decoded_ranges).unwrap()),
                SAMPLE_RATE,
                1,
                AudioTrackTiming::default(),
                1,
                Limits::default(),
            )
            .unwrap();
            let backend = FixtureBackend {
                clock: clock.clone(),
                scheduled: scheduled.clone(),
                canceled: Arc::new(Mutex::new(Vec::new())),
            };
            let timeline =
                Timeline::new(crate::FrameRate::new(24, 1).unwrap(), SAMPLE_RATE).unwrap();
            PlaybackController::new(
                OnDemandVideoSource::new(video_reader, video_loader, 2),
                OnDemandAudioSource::new(audio_reader, audio_loader, 2),
                WebAudioOutput(backend),
                timeline,
                PlaybackOptions {
                    schedule_ahead_samples: 4_000,
                    preroll_samples: 1_000,
                },
            )
            .unwrap()
        }

        async fn plays_and_seeks_without_blocking() {
            let reads = Rc::new(Cell::new(0));
            let clock = Arc::new(Mutex::new(0));
            let scheduled = Arc::new(Mutex::new(Vec::new()));
            let mut playback = playback(&reads, &clock, &scheduled);

            let error = playback.play().unwrap_err();
            assert_eq!(error.kind(), ErrorKind::WouldBlock);
            assert_eq!(reads.get(), 0, "a synchronous call read the source");
            until_loaded(&mut playback, |playback| playback.play()).await;

            // The clock reading and media sample at the last seek, which the
            // controller measures media time from.
            let mut anchor = (0, 0);
            for frame in [0, 1, 2, 5, 9, 20, 3, 4] {
                if frame == 20 || frame == 3 {
                    playback.seek(FrameIndex(frame)).unwrap();
                    anchor = (*clock.lock().unwrap(), frame * SAMPLES_PER_FRAME);
                }
                *clock.lock().unwrap() = anchor.0 + frame * SAMPLES_PER_FRAME - anchor.1;
                let (presentation, picture) =
                    until_loaded(&mut playback, |playback| playback.present()).await;
                assert_eq!(presentation.frame, Some(FrameIndex(frame)));
                assert_eq!(picture.unwrap().planes[0].data, [frame as u8]);
            }
            assert!(reads.get() > 0);
            assert!(!scheduled.lock().unwrap().is_empty());
        }

        /// Issue #695: a `play` that reports `WouldBlock` leaves the controller
        /// paused where it was, so its retry schedules audio from the play
        /// position however far the output's clock has run in the meantime.
        async fn retried_play_schedules_from_the_play_position() {
            let reads = Rc::new(Cell::new(0));
            let clock = Arc::new(Mutex::new(0));
            let scheduled = Arc::new(Mutex::new(Vec::new()));
            let mut playback = playback(&reads, &clock, &scheduled);
            for (start, waited) in [(0, 2_400), (5 * SAMPLES_PER_FRAME, 7_000)] {
                if start > 0 {
                    playback.pause().unwrap();
                    playback
                        .seek(FrameIndex(start / SAMPLES_PER_FRAME))
                        .unwrap();
                }
                scheduled.lock().unwrap().clear();

                let error = playback.play().unwrap_err();
                assert_eq!(error.kind(), ErrorKind::WouldBlock);
                assert!(!playback.is_playing());
                assert!(scheduled.lock().unwrap().is_empty());

                // The page awaits a prefetch and tries again, while the
                // output's clock runs on.
                until_loaded(&mut playback, |playback| {
                    *clock.lock().unwrap() += waited;
                    assert_eq!(
                        playback.current_frame_index().unwrap(),
                        FrameIndex(start / SAMPLES_PER_FRAME)
                    );
                    playback.play()
                })
                .await;
                assert!(playback.is_playing());

                let scheduled = scheduled.lock().unwrap();
                assert_eq!(scheduled.first().map(|(range, _)| range.start), Some(start));
                assert!(
                    scheduled
                        .windows(2)
                        .all(|pair| pair[0].0.end == pair[1].0.start)
                );
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        fn run_natively<F: Future<Output = ()>>(future: F) {
            let mut future = std::pin::pin!(future);
            let mut context = Context::from_waker(std::task::Waker::noop());
            while future.as_mut().poll(&mut context).is_pending() {}
        }

        #[cfg(not(target_arch = "wasm32"))]
        #[test]
        fn natively() {
            run_natively(plays_and_seeks_without_blocking());
        }

        #[cfg(not(target_arch = "wasm32"))]
        #[test]
        fn retried_play_natively() {
            run_natively(retried_play_schedules_from_the_play_position());
        }

        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

        #[cfg(target_arch = "wasm32")]
        #[wasm_bindgen_test::wasm_bindgen_test(async)]
        async fn in_the_browser() {
            plays_and_seeks_without_blocking().await;
        }

        #[cfg(target_arch = "wasm32")]
        #[wasm_bindgen_test::wasm_bindgen_test(async)]
        async fn retried_play_in_the_browser() {
            retried_play_schedules_from_the_play_position().await;
        }
    }
}
