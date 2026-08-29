//! Audio-clock-driven playback shared by native and browser adapters.

use crate::{
    AudioBuffer, CancellationToken, Error, ErrorKind, FrameIndex, Result, SampleRange, Timeline,
    VideoFrame,
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

    pub fn from_mp4_track(
        track: &crate::Mp4Track,
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

    pub fn frame_for_audio_sample(&self, sample: u64) -> Result<FrameIndex> {
        let index = self
            .frame_audio_ranges
            .partition_point(|range| range.end <= sample);
        let index = index.min(self.frame_audio_ranges.len() - 1);
        Ok(FrameIndex(index as u64))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PlaybackTimeline {
    Constant(Timeline),
    Indexed(IndexedPresentationTimeline),
}

impl PlaybackTimeline {
    fn audio_interval_for_frame(&self, frame: FrameIndex) -> Result<SampleRange> {
        match self {
            Self::Constant(timeline) => timeline.audio_interval_for_frame(frame),
            Self::Indexed(timeline) => timeline.audio_interval_for_frame(frame),
        }
    }

    fn frame_for_audio_sample(&self, sample: u64) -> Result<FrameIndex> {
        match self {
            Self::Constant(timeline) => timeline.frame_for_audio_sample(sample),
            Self::Indexed(timeline) => timeline.frame_for_audio_sample(sample),
        }
    }
}

/// Schedules audio ahead and selects exact video frames from its master clock.
pub struct PlaybackController<V, A, O> {
    video: V,
    audio: A,
    output: O,
    timeline: PlaybackTimeline,
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
        Self::new_with_playback_timeline(
            video,
            audio,
            output,
            PlaybackTimeline::Constant(timeline),
            options,
        )
    }

    pub fn new_with_indexed_timeline(
        video: V,
        audio: A,
        output: O,
        timeline: IndexedPresentationTimeline,
        options: PlaybackOptions,
    ) -> Result<Self> {
        Self::new_with_playback_timeline(
            video,
            audio,
            output,
            PlaybackTimeline::Indexed(timeline),
            options,
        )
    }

    fn new_with_playback_timeline(
        video: V,
        audio: A,
        output: O,
        timeline: PlaybackTimeline,
        options: PlaybackOptions,
    ) -> Result<Self> {
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
        self.clock_anchor = self.output.clock_samples();
        self.output.start(self.media_anchor)?;
        self.playing = true;
        self.fill_audio()
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

    /// Cancels old decode/scheduling work, resets both streams, prerolls, then resumes.
    pub fn seek(&mut self, frame: FrameIndex) -> Result<()> {
        let target = self.timeline.audio_interval_for_frame(frame)?.start;
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
            let _ = self.audio.read(preroll, &self.cancellation)?;
        }
        self.queued_until = target;
        self.last_presented = None;
        if self.playing {
            self.output.start(target)?;
            self.fill_audio()?;
        }
        Ok(())
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

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn scale_to_audio_samples(
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
        cancelled: Arc<Mutex<Vec<u64>>>,
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
            self.cancelled.lock().unwrap().push(generation);
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
        let cancelled = Arc::new(Mutex::new(Vec::new()));
        let backend = FixtureBackend {
            clock: clock.clone(),
            scheduled: scheduled.clone(),
            cancelled: cancelled.clone(),
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
        assert_eq!(&*cancelled.lock().unwrap(), &[1]);
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
    fn pause_preserves_the_current_playback_position() {
        let requested = Arc::new(Mutex::new(Vec::new()));
        let reads = Arc::new(Mutex::new(Vec::new()));
        let clock = Arc::new(Mutex::new(0));
        let backend = FixtureBackend {
            clock: clock.clone(),
            scheduled: Arc::new(Mutex::new(Vec::new())),
            cancelled: Arc::new(Mutex::new(Vec::new())),
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
            cancelled: Arc::new(Mutex::new(Vec::new())),
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
}
