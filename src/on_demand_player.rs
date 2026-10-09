//! Native on-demand playback of an MP4 or WebM that is never loaded whole
//! (issues #689, #685 and #692).
//!
//! [`OnDemandPlayer`] is the native counterpart of the browser's
//! `OnDemandPlayback`: it reads the container's header and index when it opens
//! and then only the compressed samples playback reaches, into one cache for
//! the video track and one for the selected audio track, each bounded by a byte
//! budget. It chooses the video decoder and the AAC, Opus or Vorbis audio
//! decoder from the tracks, and plays on the default audio device:
//!
//! ```no_run
//! use std::time::Duration;
//! use zvidlib::{OnDemandOptions, OnDemandPlayer};
//!
//! # fn main() -> zvidlib::Result<()> {
//! let mut player = OnDemandPlayer::open("movie.mp4", OnDemandOptions {
//!     video_budget_bytes: 3_500_000,
//!     audio_budget_bytes: 512 * 1024,
//!     audio_track: 0,
//!     ..OnDemandOptions::default()
//! })?;
//! player.play()?;
//! player.seek(Duration::from_secs(50))?;
//! player.select_audio_language("fra")?;
//! let presentation = player.present()?;
//! # Ok(())
//! # }
//! ```
//!
//! Playback is addressed by time. A WebM with `Cues` is opened from its header
//! elements and `Cues` alone, and its blocks are indexed a cue span at a time
//! as playback or a seek reaches them (issue #692), so how many frames it has
//! is not known when it opens, and [`OnDemandPlayer::frame_count`] is `None`
//! for it.
//!
//! The budgets bound compressed data only. Decoded pictures are held apart
//! from them, up to [`OnDemandOptions::max_cached_frames`] at a time, and at
//! high resolutions they are what dominates memory: the decoders output
//! [`crate::PixelFormat::Rgba8`], so one 7680x4320 picture is about 133 MB.
//!
//! Every call that needs a sample not loaded yet loads it before returning,
//! so a caller never sees [`ErrorKind::WouldBlock`]. A [`FileSource`] answers
//! at once; a source whose reads suspend is waited on, and an async caller can
//! [`OnDemandPlayer::prefetch`] ahead of time instead so that no call has to.

use crate::codec::HardwarePreference;
use crate::io::{ByteSource, CachingByteSource, FileSource, IoFuture};
use crate::media::{VideoDimensions, VideoFrame};
use crate::on_demand::{
    AUDIO_READAHEAD_PACKETS, DEFAULT_AUDIO_BUDGET_BYTES, DEFAULT_VIDEO_BUDGET_BYTES,
    INDEX_CACHE_BYTES, INDEX_PAGE_BYTES, SilentAudioSource, VIDEO_ONLY_CLOCK_RATE, VideoDecoding,
    audio_packets, crate_video_source,
};
use crate::on_demand_cues::{
    AudioPackets, AudioWindow, CuedSilence, CuedSpans, CuedTimeline, CuedVideoSource,
    PacketAudioSource,
};
use crate::playback::{
    AudioOutputBackend, IndexedPresentationTimeline, LazyPresentationTimeline, NativeAudioOutput,
    OnDemandVideoSource, PlaybackAudioOutput, PlaybackAudioSource, PlaybackController,
    PlaybackOptions, PlaybackVideoSource, PrefetchAudioSource, PrefetchVideoSource,
    PresentationTimeline,
};
use crate::timeline::{FrameIndex, SampleRange};
use crate::track::Track;
use crate::{
    AudioBuffer, AudioDecoder, AudioSampleReader, AudioTrackTiming, CancellationToken, Error,
    ErrorKind, Limits, Result, TrackKind, TrackSampleLoader,
};
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

/// How many times one call loads what it is missing before giving up: each
/// load takes in the frame and audio the call needs with a readahead, so only
/// budgets too small to hold one frame's worth of samples run out of passes.
const MAX_LOAD_PASSES: usize = 16;

/// What [`OnDemandPlayer`] opens an input with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OnDemandOptions {
    /// The most compressed video the player holds at once, in bytes. It must
    /// hold the track's largest sample.
    pub video_budget_bytes: u64,
    /// The most compressed audio the player holds at once, in bytes, counting
    /// only the selected track's packets. It must hold the largest one
    /// together with the packets decoded ahead of it after a seek, or opening
    /// the track fails with [`crate::ErrorKind::ResourceLimit`] (see
    /// [`crate::TrackSampleLoader::audio_budget_floor`]).
    pub audio_budget_bytes: u64,
    /// The audio track to play first, by its position among the input's
    /// audio tracks. Ignored for an input with no audio track.
    pub audio_track: usize,
    /// The most decoded pictures the player keeps, which the byte budgets do
    /// not cover (see [`crate::Limits::max_cached_frames`]).
    pub max_cached_frames: u32,
}

impl Default for OnDemandOptions {
    fn default() -> Self {
        Self {
            video_budget_bytes: DEFAULT_VIDEO_BUDGET_BYTES,
            audio_budget_bytes: DEFAULT_AUDIO_BUDGET_BYTES,
            audio_track: 0,
            max_cached_frames: Limits::default().max_cached_frames,
        }
    }
}

/// What [`OnDemandPlayer::present`] found the clock calling for.
#[derive(Clone, Debug, PartialEq)]
pub struct OnDemandPresentation {
    /// When the frame the clock calls for starts.
    pub time: Duration,
    /// That frame, decoded, when it is not the one presented last.
    pub frame: Option<VideoFrame>,
    /// Whether the clock has passed the end of the presentation.
    pub finished: bool,
}

/// Opens the output an audio track plays on, given its sample rate and
/// channel count. [`OnDemandPlayer`] calls it at open and again whenever a
/// newly selected track needs it.
pub type AudioOutputOpener = Box<dyn FnMut(u32, u16) -> Result<Box<dyn PlaybackAudioOutput>>>;

/// What a cued WebM's spans are indexed through: a page cache over the input.
type IndexSource<S> = CachingByteSource<S>;

type TrackAudio<S> = PacketAudioSource<Box<dyn AudioDecoder>, IndexSource<S>, S>;

/// What the player's video comes from: the whole track's index, or a cued
/// WebM's spans.
enum PlayerVideo<S> {
    Whole(OnDemandVideoSource<S>),
    Cued(CuedVideoSource<IndexSource<S>, OnDemandVideoSource<S>>),
}

impl<S: ByteSource> PlayerVideo<S> {
    fn resident_bytes(&self) -> u64 {
        match self {
            Self::Whole(video) => video.loader().resident_bytes(),
            Self::Cued(video) => video.resident_bytes(),
        }
    }
}

impl<S: ByteSource> PlaybackVideoSource for PlayerVideo<S> {
    fn get_exact(
        &mut self,
        frame: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<VideoFrame> {
        match self {
            Self::Whole(video) => video.get_exact(frame, cancellation),
            Self::Cued(video) => video.get_exact(frame, cancellation),
        }
    }

    fn reset(&mut self) -> Result<()> {
        match self {
            Self::Whole(video) => video.reset(),
            Self::Cued(video) => video.reset(),
        }
    }
}

impl<S: ByteSource> PrefetchVideoSource for PlayerVideo<S> {
    fn prefetch(&mut self, frame: FrameIndex) -> IoFuture<'_, ()> {
        match self {
            Self::Whole(video) => video.prefetch(frame),
            Self::Cued(video) => video.prefetch(frame),
        }
    }
}

/// What the player's audio comes from: the selected audio track, or silence
/// when the input has none.
enum PlayerAudio<S> {
    Track(Box<TrackAudio<S>>),
    Silent(SilentAudioSource),
    CuedSilent(CuedSilence<IndexSource<S>>),
}

impl<S: ByteSource + Clone> PlayerAudio<S> {
    fn resident_bytes(&self) -> u64 {
        match self {
            Self::Track(audio) => audio.loader().resident_bytes(),
            Self::Silent(_) | Self::CuedSilent(_) => 0,
        }
    }
}

impl<S: ByteSource + Clone> PlaybackAudioSource for PlayerAudio<S> {
    fn sample_rate(&self) -> u32 {
        match self {
            Self::Track(audio) => audio.sample_rate(),
            Self::Silent(audio) => audio.sample_rate(),
            Self::CuedSilent(audio) => audio.sample_rate(),
        }
    }

    fn presentation_length(&self) -> u64 {
        match self {
            Self::Track(audio) => audio.presentation_length(),
            Self::Silent(audio) => audio.presentation_length(),
            Self::CuedSilent(audio) => audio.presentation_length(),
        }
    }

    fn read(
        &mut self,
        range: SampleRange,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        match self {
            Self::Track(audio) => audio.read(range, cancellation),
            Self::Silent(audio) => audio.read(range, cancellation),
            Self::CuedSilent(audio) => audio.read(range, cancellation),
        }
    }

    fn reset(&mut self) -> Result<()> {
        match self {
            Self::Track(audio) => audio.reset(),
            Self::Silent(audio) => audio.reset(),
            Self::CuedSilent(audio) => audio.reset(),
        }
    }
}

impl<S: ByteSource + Clone> PrefetchAudioSource for PlayerAudio<S> {
    fn prefetch(&mut self, range: SampleRange) -> IoFuture<'_, ()> {
        match self {
            Self::Track(audio) => audio.prefetch(range),
            Self::Silent(audio) => audio.prefetch(range),
            Self::CuedSilent(audio) => audio.prefetch(range),
        }
    }
}

/// Times playback of an input with no audio track on the system's monotonic
/// clock, and plays nothing.
struct WallClock {
    started: Instant,
    rate: u32,
}

impl AudioOutputBackend for WallClock {
    fn clock_samples(&self) -> u64 {
        let elapsed = self.started.elapsed().as_nanos();
        u64::try_from(elapsed * u128::from(self.rate) / 1_000_000_000).unwrap_or(u64::MAX)
    }

    fn start(&mut self, _: u64) -> Result<()> {
        Ok(())
    }

    fn schedule(&mut self, _: AudioBuffer, _: u64) -> Result<()> {
        Ok(())
    }

    fn cancel_queued(&mut self, _: u64) -> Result<()> {
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        Ok(())
    }
}

type Controller<S> =
    PlaybackController<PlayerVideo<S>, PlayerAudio<S>, Box<dyn PlaybackAudioOutput>>;

/// Plays an MP4 or WebM natively, reading only the compressed samples playback
/// reaches; see the [module documentation](self).
pub struct OnDemandPlayer<S = FileSource> {
    controller: Controller<S>,
    input: Input<S>,
    audio_track: Option<usize>,
    open_output: AudioOutputOpener,
}

/// How the input's tracks are indexed.
enum Index<S> {
    /// Every track's samples, read when the input opened.
    Whole {
        video: Track,
        /// Each audio track's timing on the decoded sample clock, as its
        /// container gives it, or why it has none; a track's failure is
        /// reported when it is selected, not when the input opens.
        audio_timings: Vec<Result<AudioTrackTiming>>,
    },
    /// A cued WebM's spans, indexed as playback reaches them.
    Cued(CuedSpans<IndexSource<S>>),
}

/// What the player opened, from which it builds the audio of any track.
struct Input<S> {
    source: S,
    index: Index<S>,
    audio_tracks: Vec<Track>,
    options: OnDemandOptions,
    limits: Limits,
    dimensions: VideoDimensions,
}

type PlayerAudioParts<S> = (
    PlayerAudio<S>,
    Box<dyn PlaybackAudioOutput>,
    PresentationTimeline,
);

impl<S: ByteSource + Clone + 'static> Input<S> {
    /// The timeline that maps a clock of `sample_rate` to the video's frames.
    fn timeline(&self, sample_rate: u32) -> Result<PresentationTimeline> {
        Ok(match &self.index {
            Index::Whole { video, .. } => {
                IndexedPresentationTimeline::from_track(video, sample_rate, &self.limits)?.into()
            }
            Index::Cued(spans) => {
                let timeline: Box<dyn LazyPresentationTimeline> = Box::new(CuedTimeline::new(
                    spans.clone(),
                    sample_rate,
                    &self.limits,
                )?);
                timeline.into()
            }
        })
    }

    /// The source, output and timeline of the audio track at `index` among
    /// the input's audio tracks, or of silence for `None`. Reads no packet but
    /// an Opus track's last, and for a cued WebM nothing at all but what the
    /// spans its first packets are in need.
    fn audio(
        &self,
        index: Option<usize>,
        open_output: &mut AudioOutputOpener,
    ) -> Result<PlayerAudioParts<S>> {
        let Some(index) = index else {
            let timeline = self.timeline(VIDEO_ONLY_CLOCK_RATE)?;
            let audio = match &self.index {
                Index::Whole { video, .. } => PlayerAudio::Silent(SilentAudioSource {
                    sample_rate: VIDEO_ONLY_CLOCK_RATE,
                    length: IndexedPresentationTimeline::from_track(
                        video,
                        VIDEO_ONLY_CLOCK_RATE,
                        &self.limits,
                    )?
                    .end_sample(),
                    limits: self.limits,
                }),
                Index::Cued(spans) => PlayerAudio::CuedSilent(CuedSilence {
                    spans: spans.clone(),
                    sample_rate: VIDEO_ONLY_CLOCK_RATE,
                    limits: self.limits,
                }),
            };
            let output: Box<dyn PlaybackAudioOutput> = Box::new(NativeAudioOutput(WallClock {
                started: Instant::now(),
                rate: VIDEO_ONLY_CLOCK_RATE,
            }));
            return Ok((audio, output, timeline));
        };
        let track = self
            .audio_tracks
            .get(index)
            .ok_or_else(|| no_audio_track(index, self.audio_tracks.len()))?
            .clone();
        let (decoder, channels) = audio_decoder(&track, self.limits)?;
        let sample_rate = track.audio_sample_rate()?;
        let (packets, provider, timing, preroll) = match &self.index {
            Index::Whole { audio_timings, .. } => {
                let timing = audio_timings[index].clone()?;
                let loader = TrackSampleLoader::new(
                    track,
                    self.source.clone(),
                    self.options.audio_budget_bytes,
                )?;
                let (provider, preroll) = block_on(audio_packets(&loader))?;
                (AudioPackets::Whole(loader), provider, timing, preroll)
            }
            Index::Cued(spans) => {
                let (window, provider, timing) = block_on(AudioWindow::open(
                    spans.clone(),
                    &track,
                    self.source.clone(),
                    self.options.audio_budget_bytes,
                ))?;
                let preroll = window.preroll();
                (
                    AudioPackets::Cued(Box::new(window)),
                    provider,
                    timing,
                    preroll,
                )
            }
        };
        let reader = AudioSampleReader::from_provider(
            decoder,
            Box::new(provider),
            sample_rate,
            channels,
            timing,
            preroll,
            self.limits,
        )?;
        let audio = PlayerAudio::Track(Box::new(PacketAudioSource::new(
            reader,
            packets,
            AUDIO_READAHEAD_PACKETS,
        )));
        let timeline = self.timeline(sample_rate)?;
        let output = open_output(sample_rate, channels)?;
        Ok((audio, output, timeline))
    }
}

impl OnDemandPlayer<FileSource> {
    /// Opens the MP4 or WebM at `path`, playing its audio on the default audio
    /// device.
    pub fn open(path: impl AsRef<std::path::Path>, options: OnDemandOptions) -> Result<Self> {
        Self::from_source(FileSource::open(path)?, options)
    }
}

impl<S: ByteSource + Clone + 'static> OnDemandPlayer<S> {
    /// Opens the MP4 or WebM `source` reads, playing its audio on the default
    /// audio device.
    pub fn from_source(source: S, options: OnDemandOptions) -> Result<Self> {
        Self::with_output(
            source,
            options,
            Box::new(|sample_rate, channels| {
                Ok(Box::new(NativeAudioOutput(crate::DefaultAudioOutput::open(
                    sample_rate,
                    channels,
                )?)) as Box<dyn PlaybackAudioOutput>)
            }),
        )
    }

    /// Opens the MP4 or WebM `source` reads, playing its audio on the outputs
    /// `open_output` opens. An input with no audio track opens none, and plays
    /// on the system's monotonic clock.
    ///
    /// Reads the container's header and index, and of the samples only an AV1
    /// or VP9 track's first and an Opus track's last. A WebM with `Cues` is
    /// read from its header elements and `Cues`, and the block headers of its
    /// first cue span, however long it is (issue #692); its other spans'
    /// block headers are read as playback or a seek reaches them.
    pub fn with_output(
        source: S,
        options: OnDemandOptions,
        mut open_output: AudioOutputOpener,
    ) -> Result<Self> {
        let limits = Limits {
            max_cached_frames: options.max_cached_frames,
            ..Limits::default()
        };
        let index_source =
            CachingByteSource::new(source.clone(), INDEX_PAGE_BYTES, INDEX_CACHE_BYTES)?;
        let (index, audio_tracks, video_source, dimensions) =
            match block_on(CuedSpans::open(index_source, &limits))? {
                Some(spans) => Self::open_cued(spans, &source, options, limits)?,
                None => Self::open_whole(&source, options, limits)?,
            };
        let audio_track = if audio_tracks.is_empty() {
            None
        } else if options.audio_track < audio_tracks.len() {
            Some(options.audio_track)
        } else {
            return Err(no_audio_track(options.audio_track, audio_tracks.len()));
        };
        let input = Input {
            dimensions,
            source,
            index,
            audio_tracks,
            options,
            limits,
        };
        let (audio, output, timeline) = input.audio(audio_track, &mut open_output)?;
        let sample_rate = audio.sample_rate();
        let controller = PlaybackController::new_with_timeline(
            video_source,
            audio,
            output,
            timeline,
            PlaybackOptions::for_sample_rate(sample_rate),
        )?;
        Ok(Self {
            controller,
            input,
            audio_track,
            open_output,
        })
    }

    /// The tracks of an input indexed whole when it opens.
    #[allow(clippy::type_complexity)]
    fn open_whole(
        source: &S,
        options: OnDemandOptions,
        limits: Limits,
    ) -> Result<(Index<S>, Vec<Track>, PlayerVideo<S>, VideoDimensions)> {
        let media = block_on(crate::container::open_media(source, &limits))?;
        let video = media
            .first_track(TrackKind::Video)
            .ok_or_else(|| Error::new(ErrorKind::Unsupported, "the input has no video track"))?
            .clone();
        let audio_tracks: Vec<Track> = media
            .tracks()
            .iter()
            .filter(|track| track.kind == TrackKind::Audio)
            .cloned()
            .collect();
        let audio_timings = audio_tracks
            .iter()
            .map(|track| media.audio_timing(track))
            .collect();
        let dimensions = video_dimensions(&video)?;
        let video_source = block_on(crate_video_source(
            &video,
            source.clone(),
            options.video_budget_bytes,
            HardwarePreference::Prefer,
            limits,
        ))?;
        Ok((
            Index::Whole {
                video,
                audio_timings,
            },
            audio_tracks,
            PlayerVideo::Whole(video_source),
            dimensions,
        ))
    }

    /// The tracks of a cued WebM, whose first span is indexed.
    #[allow(clippy::type_complexity)]
    fn open_cued(
        spans: CuedSpans<IndexSource<S>>,
        source: &S,
        options: OnDemandOptions,
        limits: Limits,
    ) -> Result<(Index<S>, Vec<Track>, PlayerVideo<S>, VideoDimensions)> {
        let index = spans.index();
        let audio_tracks: Vec<Track> = index
            .tracks
            .iter()
            .filter(|track| track.kind == TrackKind::Audio)
            .cloned()
            .collect();
        let first = spans
            .indexed(0)
            .ok_or_else(|| Error::new(ErrorKind::Internal, "the first cue span is not indexed"))?;
        let video = first
            .track(index.video_track)
            .ok_or_else(|| Error::new(ErrorKind::Internal, "a cue span lost its video track"))?;
        let dimensions = video_dimensions(video)?;
        // The decoder is chosen from the stream's first sample, as it is for a
        // whole track, and every span decodes on one like it.
        let decoding = block_on(VideoDecoding::open(
            video,
            source,
            HardwarePreference::Prefer,
            limits,
        ))?;
        let span_source = source.clone();
        let video_source = CuedVideoSource::new(
            spans.clone(),
            options.video_budget_bytes,
            Box::new(move |track, cache, base| {
                decoding.source(TrackSampleLoader::with_cache(
                    track,
                    span_source.clone(),
                    cache,
                    base,
                )?)
            }),
        );
        Ok((
            Index::Cued(spans),
            audio_tracks,
            PlayerVideo::Cued(video_source),
            dimensions,
        ))
    }

    /// Runs `step`, loading whatever it reports missing and trying again.
    fn with_loads<T>(
        &mut self,
        mut step: impl FnMut(&mut Controller<S>) -> Result<T>,
    ) -> Result<T> {
        for _ in 0..MAX_LOAD_PASSES {
            match step(&mut self.controller) {
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    block_on(self.controller.prefetch())?;
                }
                result => return result,
            }
        }
        Err(Error::new(
            ErrorKind::ResourceLimit,
            "the byte budgets cannot hold the samples one frame of playback reads",
        ))
    }

    /// The playback clock's rate: the selected audio track's sample rate, or
    /// the rate of the clock an input with no audio plays on.
    fn clock_rate(&self) -> u32 {
        self.controller.audio().sample_rate()
    }

    fn time_of(&self, sample: u64) -> Duration {
        samples_to_duration(sample, self.clock_rate())
    }

    pub fn play(&mut self) -> Result<()> {
        self.with_loads(PlaybackController::play)
    }

    pub fn pause(&mut self) -> Result<()> {
        self.controller.pause()
    }

    pub fn is_playing(&self) -> bool {
        self.controller.is_playing()
    }

    /// Moves playback to `time` from the start, keeping whether it plays. The
    /// frame on screen there is the one presented next. Fails with
    /// [`ErrorKind::InvalidInput`] past [`Self::duration`].
    pub fn seek(&mut self, time: Duration) -> Result<()> {
        let sample = duration_to_samples(time, self.clock_rate());
        self.controller.seek_to_sample(sample)
    }

    /// The frame the audio clock calls for, decoded, when it is not the one
    /// presented last; see [`PlaybackController::present`]. Fails with
    /// [`ErrorKind::InvalidState`] while paused, when
    /// [`Self::current_frame`] gives the frame to show.
    pub fn present(&mut self) -> Result<OnDemandPresentation> {
        let (presentation, frame) = self.with_loads(PlaybackController::present)?;
        let start = self.with_loads(|controller| {
            controller.audio_interval_for_frame(presentation.requested_frame)
        })?;
        Ok(OnDemandPresentation {
            time: self.time_of(start.start),
            frame,
            finished: presentation.finished,
        })
    }

    /// The frame playback is on, decoded.
    pub fn current_frame(&mut self) -> Result<VideoFrame> {
        self.with_loads(PlaybackController::current_frame)
    }

    /// Where playback is, from the start: where it was paused or sought to,
    /// or where the audio clock has run to while it plays.
    pub fn current_time(&self) -> Duration {
        self.time_of(self.controller.current_audio_sample())
    }

    /// Loads what the next call reads, so that it need not; see
    /// [`PlaybackController::prefetch`]. For an async caller whose source
    /// suspends, which the other calls would otherwise wait on.
    pub async fn prefetch(&mut self) -> Result<()> {
        self.controller.prefetch().await
    }

    /// The input's audio tracks, in the order [`Self::select_audio_track`]
    /// and [`OnDemandOptions::audio_track`] number them. Each one's
    /// [`Track::language`] is its ISO 639-2 code. A cued WebM's tracks are
    /// listed without samples, which are indexed as playback reaches them.
    pub fn audio_tracks(&self) -> &[Track] {
        &self.input.audio_tracks
    }

    /// The position of the audio track playing among [`Self::audio_tracks`],
    /// or `None` for an input with none.
    pub fn audio_track(&self) -> Option<usize> {
        self.audio_track
    }

    /// Plays the audio track at `index` among [`Self::audio_tracks`] from now
    /// on, keeping the frame playback is on and whether it plays.
    ///
    /// The old track's loaded packets are released, so only the selected
    /// track's ever count against the audio budget.
    pub fn select_audio_track(&mut self, index: usize) -> Result<()> {
        if index >= self.input.audio_tracks.len() {
            return Err(no_audio_track(index, self.input.audio_tracks.len()));
        }
        if self.audio_track == Some(index) {
            return Ok(());
        }
        let (audio, output, timeline) = self.input.audio(Some(index), &mut self.open_output)?;
        // The old source, and the packets its loader holds, drop here.
        self.controller.replace_audio(audio, output, timeline)?;
        self.audio_track = Some(index);
        Ok(())
    }

    /// Plays the first audio track whose [`Track::language`] is
    /// `language`, an ISO 639-2 code such as `"eng"`, as
    /// [`Self::select_audio_track`] does. Fails with
    /// [`ErrorKind::InvalidInput`] if no track has that language.
    pub fn select_audio_language(&mut self, language: &str) -> Result<()> {
        let index = self
            .input
            .audio_tracks
            .iter()
            .position(|track| track.language.as_deref() == Some(language))
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidInput,
                    format!("the input has no audio track in language {language:?}"),
                )
            })?;
        self.select_audio_track(index)
    }

    /// How many frames the video has, or `None` for a cued WebM, whose blocks
    /// are indexed as playback reaches them.
    pub fn frame_count(&self) -> Option<u64> {
        match &self.input.index {
            Index::Whole { video, .. } => Some(video.presentation_order.len() as u64),
            Index::Cued(_) => None,
        }
    }

    /// How long the presentation plays: the selected audio track's
    /// presentation, or the video's for an input with no audio. For a cued
    /// WebM it is estimated from the segment's `Duration` until playback or a
    /// seek reaches the file's last cue span, and exact from then on.
    pub fn duration(&self) -> Duration {
        self.time_of(self.controller.audio().presentation_length())
    }

    pub fn dimensions(&self) -> VideoDimensions {
        self.input.dimensions
    }

    /// The selected audio track's sample rate, or `None` for an input with no
    /// audio track.
    pub fn sample_rate(&self) -> Option<u32> {
        self.audio_track
            .map(|_| self.controller.audio().sample_rate())
    }

    pub fn options(&self) -> OnDemandOptions {
        self.input.options
    }

    /// Compressed video bytes loaded now, never more than
    /// [`OnDemandOptions::video_budget_bytes`].
    pub fn video_resident_bytes(&self) -> u64 {
        self.controller.video().resident_bytes()
    }

    /// Compressed bytes of the selected audio track loaded now, never more
    /// than [`OnDemandOptions::audio_budget_bytes`].
    pub fn audio_resident_bytes(&self) -> u64 {
        self.controller.audio().resident_bytes()
    }
}

fn video_dimensions(video: &Track) -> Result<VideoDimensions> {
    video
        .dimensions
        .ok_or_else(|| Error::new(ErrorKind::MalformedMedia, "video track has no dimensions"))
}

/// `sample` of a clock running at `rate`, as a time from the start.
pub(crate) fn samples_to_duration(sample: u64, rate: u32) -> Duration {
    let nanoseconds = u128::from(sample) * 1_000_000_000 / u128::from(rate.max(1));
    Duration::from_nanos(u64::try_from(nanoseconds).unwrap_or(u64::MAX))
}

/// The sample of a clock running at `rate` nearest `time` from the start.
pub(crate) fn duration_to_samples(time: Duration, rate: u32) -> u64 {
    let samples = (time.as_nanos() * u128::from(rate) + 500_000_000) / 1_000_000_000;
    u64::try_from(samples).unwrap_or(u64::MAX)
}

/// The crate's decoder for an audio track, and the channels it decodes to.
fn audio_decoder(track: &Track, limits: Limits) -> Result<(Box<dyn AudioDecoder>, u16)> {
    match track.codec {
        #[cfg(feature = "aac-decoder")]
        crate::Codec::Aac => {
            let config = track.aac_config()?;
            Ok((
                Box::new(crate::NativeAacDecoder::new(&config, limits)?),
                config.channels,
            ))
        }
        #[cfg(feature = "opus-decoder")]
        crate::Codec::Opus => {
            let head = track.opus_config()?;
            Ok((
                Box::new(crate::NativeOpusDecoder::new(&head, limits)?),
                u16::from(head.channels),
            ))
        }
        #[cfg(feature = "vorbis-decoder")]
        crate::Codec::Vorbis => {
            let config = track.vorbis_config()?;
            Ok((
                Box::new(crate::NativeVorbisDecoder::new(&config, limits)?),
                u16::from(config.channels),
            ))
        }
        _ => {
            let _ = limits;
            Err(Error::new(
                ErrorKind::Unsupported,
                "on-demand playback decodes AAC audio with the aac-decoder feature, Opus audio \
                 with the opus-decoder feature and Vorbis audio with the vorbis-decoder feature",
            ))
        }
    }
}

fn no_audio_track(index: usize, count: usize) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        format!("audio track {index} does not exist; the input has {count}"),
    )
}

/// Drives `future` to completion on this thread, parking it while the future
/// waits to be woken. A [`FileSource`]'s reads never wait.
fn block_on<F: Future>(future: F) -> F::Output {
    struct Unpark(std::thread::Thread);

    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::park();
    }
}
