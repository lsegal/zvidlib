//! Native on-demand playback of an MP4 or WebM that is never loaded whole
//! (issues #689 and #685).
//!
//! [`OnDemandPlayer`] is the native counterpart of the browser's
//! `OnDemandPlayback`: it reads the container's header and index when it opens
//! and then only
//! the compressed samples playback reaches, into one cache for the video track
//! and one for the selected audio track, each bounded by a byte budget. It
//! chooses the video decoder and the AAC or Opus audio decoder from the tracks,
//! and plays on the default audio device:
//!
//! ```no_run
//! use zvidlib::{FrameIndex, OnDemandOptions, OnDemandPlayer};
//!
//! # fn main() -> zvidlib::Result<()> {
//! let mut player = OnDemandPlayer::open("movie.mp4", OnDemandOptions {
//!     video_budget_bytes: 3_500_000,
//!     audio_budget_bytes: 512 * 1024,
//!     audio_track: 0,
//!     ..OnDemandOptions::default()
//! })?;
//! player.play()?;
//! player.seek(FrameIndex(1200))?;
//! player.select_audio_language("fra")?;
//! let (presentation, frame) = player.present()?;
//! # Ok(())
//! # }
//! ```
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
use crate::io::{ByteSource, FileSource, IoFuture};
use crate::media::{VideoDimensions, VideoFrame};
use crate::on_demand::{
    AUDIO_READAHEAD_PACKETS, DEFAULT_AUDIO_BUDGET_BYTES, DEFAULT_VIDEO_BUDGET_BYTES,
    SilentAudioSource, VIDEO_ONLY_CLOCK_RATE, audio_packets, crate_video_source,
};
use crate::playback::{
    AudioOutputBackend, IndexedPresentationTimeline, NativeAudioOutput, OnDemandAudioSource,
    OnDemandVideoSource, PlaybackAudioOutput, PlaybackAudioSource, PlaybackController,
    PlaybackOptions, PrefetchAudioSource, Presentation,
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
use std::time::Instant;

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
    /// only the selected track's packets. It must hold the largest one.
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

/// Opens the output an audio track plays on, given its sample rate and
/// channel count. [`OnDemandPlayer`] calls it at open and again whenever a
/// newly selected track needs it.
pub type AudioOutputOpener = Box<dyn FnMut(u32, u16) -> Result<Box<dyn PlaybackAudioOutput>>>;

type TrackAudio<S> = OnDemandAudioSource<Box<dyn AudioDecoder>, S>;

/// What the player's audio comes from: the selected audio track, or silence
/// when the input has none.
enum PlayerAudio<S> {
    Track(Box<TrackAudio<S>>),
    Silent(SilentAudioSource),
}

impl<S: ByteSource> PlayerAudio<S> {
    fn resident_bytes(&self) -> u64 {
        match self {
            Self::Track(audio) => audio.loader().resident_bytes(),
            Self::Silent(_) => 0,
        }
    }
}

impl<S: ByteSource> PlaybackAudioSource for PlayerAudio<S> {
    fn sample_rate(&self) -> u32 {
        match self {
            Self::Track(audio) => audio.sample_rate(),
            Self::Silent(audio) => audio.sample_rate(),
        }
    }

    fn presentation_length(&self) -> u64 {
        match self {
            Self::Track(audio) => audio.presentation_length(),
            Self::Silent(audio) => audio.presentation_length(),
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
        }
    }

    fn reset(&mut self) -> Result<()> {
        match self {
            Self::Track(audio) => audio.reset(),
            Self::Silent(audio) => audio.reset(),
        }
    }
}

impl<S: ByteSource> PrefetchAudioSource for PlayerAudio<S> {
    fn prefetch(&mut self, range: SampleRange) -> IoFuture<'_, ()> {
        match self {
            Self::Track(audio) => audio.prefetch(range),
            Self::Silent(audio) => audio.prefetch(range),
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
    PlaybackController<OnDemandVideoSource<S>, PlayerAudio<S>, Box<dyn PlaybackAudioOutput>>;

/// Plays an MP4 or WebM natively, reading only the compressed samples playback
/// reaches; see the [module documentation](self).
pub struct OnDemandPlayer<S = FileSource> {
    controller: Controller<S>,
    input: Input<S>,
    audio_track: Option<usize>,
    open_output: AudioOutputOpener,
}

/// What the player opened, from which it builds the audio of any track.
struct Input<S> {
    source: S,
    video: Track,
    audio_tracks: Vec<Track>,
    /// Each audio track's timing on the decoded sample clock, as its container
    /// gives it, or why it has none; a track's failure is reported when it is
    /// selected, not when the input opens.
    audio_timings: Vec<Result<AudioTrackTiming>>,
    options: OnDemandOptions,
    limits: Limits,
    dimensions: VideoDimensions,
}

type PlayerAudioParts<S> = (
    PlayerAudio<S>,
    Box<dyn PlaybackAudioOutput>,
    IndexedPresentationTimeline,
);

impl<S: ByteSource + Clone> Input<S> {
    /// The source, output and timeline of the audio track at `index` among
    /// the input's audio tracks, or of silence for `None`. Reads no packet but
    /// an Opus track's last.
    fn audio(
        &self,
        index: Option<usize>,
        open_output: &mut AudioOutputOpener,
    ) -> Result<PlayerAudioParts<S>> {
        let Some(index) = index else {
            let timeline = IndexedPresentationTimeline::from_track(
                &self.video,
                VIDEO_ONLY_CLOCK_RATE,
                &self.limits,
            )?;
            let audio = PlayerAudio::Silent(SilentAudioSource {
                sample_rate: VIDEO_ONLY_CLOCK_RATE,
                length: timeline.end_sample(),
                limits: self.limits,
            });
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
        let timing = self.audio_timings[index].clone()?;
        let loader =
            TrackSampleLoader::new(track, self.source.clone(), self.options.audio_budget_bytes)?;
        let (packets, preroll) = block_on(audio_packets(&loader))?;
        let reader = AudioSampleReader::from_provider(
            decoder,
            Box::new(packets),
            sample_rate,
            channels,
            timing,
            preroll,
            self.limits,
        )?;
        let audio = PlayerAudio::Track(Box::new(OnDemandAudioSource::new(
            reader,
            loader,
            AUDIO_READAHEAD_PACKETS,
        )));
        let timeline =
            IndexedPresentationTimeline::from_track(&self.video, sample_rate, &self.limits)?;
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

impl<S: ByteSource + Clone> OnDemandPlayer<S> {
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
    /// Reads the container's header and index, and of the samples only an AV1 or VP9 track's
    /// first and an Opus track's last.
    pub fn with_output(
        source: S,
        options: OnDemandOptions,
        mut open_output: AudioOutputOpener,
    ) -> Result<Self> {
        let limits = Limits {
            max_cached_frames: options.max_cached_frames,
            ..Limits::default()
        };
        let media = block_on(crate::container::open_media(&source, &limits))?;
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
        let audio_track = if audio_tracks.is_empty() {
            None
        } else if options.audio_track < audio_tracks.len() {
            Some(options.audio_track)
        } else {
            return Err(no_audio_track(options.audio_track, audio_tracks.len()));
        };
        let dimensions = video.dimensions.ok_or_else(|| {
            Error::new(ErrorKind::MalformedMedia, "video track has no dimensions")
        })?;
        let video_source = block_on(crate_video_source(
            &video,
            source.clone(),
            options.video_budget_bytes,
            HardwarePreference::Prefer,
            limits,
        ))?;
        let input = Input {
            dimensions,
            source,
            video,
            audio_tracks,
            audio_timings,
            options,
            limits,
        };
        let (audio, output, timeline) = input.audio(audio_track, &mut open_output)?;
        let sample_rate = audio.sample_rate();
        let controller = PlaybackController::new_with_indexed_timeline(
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

    pub fn play(&mut self) -> Result<()> {
        self.with_loads(PlaybackController::play)
    }

    pub fn pause(&mut self) -> Result<()> {
        self.controller.pause()
    }

    pub fn is_playing(&self) -> bool {
        self.controller.is_playing()
    }

    /// Moves playback to `frame`, keeping whether it plays.
    pub fn seek(&mut self, frame: FrameIndex) -> Result<()> {
        self.with_loads(|controller| controller.seek(frame))
    }

    /// The frame the audio clock calls for, decoded, when it is not the one
    /// presented last; see [`PlaybackController::present`]. Fails with
    /// [`ErrorKind::InvalidState`] while paused, when
    /// [`Self::current_frame`] gives the frame to show.
    pub fn present(&mut self) -> Result<(Presentation, Option<VideoFrame>)> {
        self.with_loads(PlaybackController::present)
    }

    /// The frame playback is on, decoded.
    pub fn current_frame(&mut self) -> Result<VideoFrame> {
        self.with_loads(PlaybackController::current_frame)
    }

    pub fn current_frame_index(&self) -> Result<FrameIndex> {
        self.controller.current_frame_index()
    }

    /// Loads what the next call reads, so that it need not; see
    /// [`PlaybackController::prefetch`]. For an async caller whose source
    /// suspends, which the other calls would otherwise wait on.
    pub async fn prefetch(&mut self) -> Result<()> {
        self.controller.prefetch().await
    }

    /// The input's audio tracks, in the order [`Self::select_audio_track`]
    /// and [`OnDemandOptions::audio_track`] number them. Each one's
    /// [`Track::language`] is its ISO 639-2 code.
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

    pub fn frame_count(&self) -> u64 {
        self.input.video.presentation_order.len() as u64
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
        self.controller.video().loader().resident_bytes()
    }

    /// Compressed bytes of the selected audio track loaded now, never more
    /// than [`OnDemandOptions::audio_budget_bytes`].
    pub fn audio_resident_bytes(&self) -> u64 {
        self.controller.audio().resident_bytes()
    }
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
        _ => {
            let _ = limits;
            Err(Error::new(
                ErrorKind::Unsupported,
                "on-demand playback decodes AAC audio with the aac-decoder feature and Opus \
                 audio with the opus-decoder feature",
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
