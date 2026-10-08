//! `PlaybackController` plays and seeks through on-demand video and audio
//! sources: synchronously over a source that answers immediately, as native
//! playback does, and over a source whose every read suspends, as a browser
//! `fetch` does, without ever waiting on it (issue #672). Each test runs over
//! the bundled MP4 and over a WebM, opened through the same container-agnostic
//! entry point (issue #685).

#![cfg(not(target_arch = "wasm32"))]

mod common;

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use common::{block_on, vp9_opus_webm};
use zvidlib::io::{ByteSource, CachingByteSource, IoFuture, MemorySource};
use zvidlib::{
    AudioBuffer, AudioDecoder, AudioOutputBackend, AudioPacketProvider, AudioSampleReader,
    AudioTrackTiming, CancellationToken, Codec, CodecProfile, ColorRange, EncodedAudioSample,
    ErrorKind, ExactFrameReader, FrameIndex, HardwarePreference, IndexedPresentationTimeline,
    Limits, Mp4SampleLoader, Mp4SampleProvider, Mp4Track, NativeAudioOutput, OnDemandAudioSource,
    OnDemandVideoSource, PixelFormat, PlaybackController, PlaybackOptions, Result, TrackKind,
    VideoDecoderConfig, VideoDecoderFactory, VideoFrame, WebAudioOutput,
    native_hevc_video_decoder_factory, native_vp9_video_decoder_factory,
};

/// A source whose every read suspends once before completing, the way a
/// browser `fetch` does, so a synchronous caller polling it once never gets
/// an answer.
struct SuspendingSource {
    inner: MemorySource,
    reads: Cell<usize>,
}

impl SuspendingSource {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            inner: MemorySource::new(bytes),
            reads: Cell::new(0),
        }
    }
}

impl ByteSource for SuspendingSource {
    fn len(&self) -> Option<u64> {
        self.inner.len()
    }

    fn read_at<'a>(&'a self, offset: u64, destination: &'a mut [u8]) -> IoFuture<'a, usize> {
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

/// Stands in for an audio codec: each packet decodes to its own bytes, spread
/// over its interval, so two readers agree only if they were handed the same
/// packet bytes for the same interval.
struct BytesAsAudio {
    sample_rate: u32,
    channels: u16,
}

impl AudioDecoder for BytesAsAudio {
    fn decode(
        &mut self,
        sample: &EncodedAudioSample,
        _: &CancellationToken,
    ) -> Result<AudioBuffer> {
        let count = sample.decoded_range.len() as usize * usize::from(self.channels);
        let samples = (0..count)
            .map(|index| f32::from(sample.data[index % sample.data.len()]))
            .collect();
        AudioBuffer::new(
            sample.decoded_range,
            self.sample_rate,
            self.channels,
            samples,
            &Limits::default(),
        )
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Default)]
struct Backend {
    clock: Arc<Mutex<u64>>,
    scheduled: Arc<Mutex<Vec<AudioBuffer>>>,
}

impl AudioOutputBackend for Backend {
    fn clock_samples(&self) -> u64 {
        *self.clock.lock().unwrap()
    }
    fn start(&mut self, _: u64) -> Result<()> {
        Ok(())
    }
    fn schedule(&mut self, buffer: AudioBuffer, _: u64) -> Result<()> {
        self.scheduled.lock().unwrap().push(buffer);
        Ok(())
    }
    fn cancel_queued(&mut self, _: u64) -> Result<()> {
        Ok(())
    }
    fn stop(&mut self) -> Result<()> {
        Ok(())
    }
}

/// An input's bytes and tracks, with what decoding its video takes.
struct Bundled {
    bytes: Vec<u8>,
    video: Mp4Track,
    audio: Mp4Track,
    /// The audio track's timing on the decoded sample clock, as its
    /// container gives it.
    timing: AudioTrackTiming,
    decoder: Box<dyn VideoDecoderFactory>,
    profile: CodecProfile,
}

/// Opens `bytes` through the container-agnostic entry point on-demand
/// playback uses.
fn open(bytes: Vec<u8>, decoder: Box<dyn VideoDecoderFactory>, profile: CodecProfile) -> Bundled {
    let source = MemorySource::new(bytes.clone());
    let media = block_on(zvidlib::container::open_media(&source, &Limits::default())).unwrap();
    let track = |kind| {
        media
            .first_track(kind)
            .cloned()
            .unwrap_or_else(|| panic!("the input has no {kind:?} track"))
    };
    let audio = track(TrackKind::Audio);
    Bundled {
        timing: media.audio_timing(&audio).unwrap(),
        video: track(TrackKind::Video),
        audio,
        bytes,
        decoder,
        profile,
    }
}

/// The bundled HEVC and AAC MP4.
fn bundled() -> Bundled {
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/media/BigBuckBunny.mp4"
    ))
    .to_vec();
    open(
        bytes,
        Box::new(native_hevc_video_decoder_factory()),
        CodecProfile::HevcMain,
    )
}

/// A VP9 and Opus WebM, whose index comes from its block headers and whose
/// audio is timed by its `CodecDelay` rather than an edit list.
fn webm() -> Bundled {
    let bundled = open(
        vp9_opus_webm(true),
        Box::new(native_vp9_video_decoder_factory()),
        CodecProfile::Vp9Profile0,
    );
    assert_eq!(bundled.video.codec, Codec::Vp9);
    assert_eq!(bundled.audio.codec, Codec::Opus);
    bundled
}

/// A small frame cache: the bundled sample is 1080p and CI decodes it in
/// software, so every cached frame is a conversion paid on each walk.
fn video_limits() -> Limits {
    Limits {
        max_cached_frames: 4,
        ..Limits::default()
    }
}

impl Bundled {
    fn video_configuration(&self) -> VideoDecoderConfig {
        VideoDecoderConfig {
            codec: self.video.codec,
            profile: self.profile,
            coded_dimensions: self.video.dimensions.unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: self.video.decoder_config.clone(),
        }
    }

    /// The audio track's packet provider over `loader`, built from the
    /// track's index alone.
    fn audio_packets<S: ByteSource + 'static>(
        &self,
        loader: &Mp4SampleLoader<S>,
    ) -> Box<dyn AudioPacketProvider> {
        Box::new(match self.audio.codec {
            Codec::Aac => loader.aac_packet_provider().unwrap(),
            Codec::Opus => block_on(loader.opus_packet_provider()).unwrap(),
            codec => panic!("no packet provider for {codec:?}"),
        })
    }

    fn eager_audio_packets(&self) -> Vec<EncodedAudioSample> {
        let source = MemorySource::new(self.bytes.clone());
        block_on(
            self.audio
                .to_encoded_audio_samples(&source, &Limits::default()),
        )
        .unwrap()
    }

    fn audio_reader(
        &self,
        packets: Box<dyn zvidlib::AudioPacketProvider>,
    ) -> AudioSampleReader<BytesAsAudio> {
        let sample_rate = self.audio.audio_sample_rate().unwrap();
        let channels = self.audio.channels.unwrap();
        AudioSampleReader::from_provider(
            BytesAsAudio {
                sample_rate,
                channels,
            },
            packets,
            sample_rate,
            channels,
            self.timing.clone(),
            2,
            Limits::default(),
        )
        .unwrap()
    }

    fn eager_video_reader(&self) -> ExactFrameReader {
        let source = MemorySource::new(self.bytes.clone());
        let samples = block_on(
            self.video
                .to_encoded_video_samples(&source, &Limits::default()),
        )
        .unwrap();
        ExactFrameReader::new(
            self.decoder.as_ref(),
            self.video_configuration(),
            samples,
            video_limits(),
        )
        .unwrap()
    }

    fn timeline(&self) -> IndexedPresentationTimeline {
        IndexedPresentationTimeline::from_mp4_track(
            &self.video,
            self.audio.audio_sample_rate().unwrap(),
            &Limits::default(),
        )
        .unwrap()
    }

    fn options(&self) -> PlaybackOptions {
        PlaybackOptions::for_sample_rate(self.audio.audio_sample_rate().unwrap())
    }
}

/// Every scheduled buffer holds exactly what an eagerly read reader returns
/// for the same range.
fn assert_scheduled_audio_matches(bundled: &Bundled, scheduled: &[AudioBuffer]) {
    assert!(!scheduled.is_empty(), "no audio was scheduled");
    let mut eager = bundled.audio_reader(Box::new(bundled.eager_audio_packets()));
    let cancellation = CancellationToken::new();
    for buffer in scheduled {
        let expected = eager.get_range(buffer.range, &cancellation).unwrap();
        assert_eq!(
            buffer.samples, expected.samples,
            "scheduled audio {:?} differs from the eager reader's",
            buffer.range
        );
    }
}

fn presentation_start(timeline: &IndexedPresentationTimeline, frame: u64) -> u64 {
    timeline
        .audio_interval_for_frame(FrameIndex(frame))
        .unwrap()
        .start
}

/// Native playback: the synchronous on-demand video provider of issue #669
/// drives the controller directly, never reporting anything not loaded.
#[test]
fn playback_plays_and_seeks_through_a_synchronous_on_demand_video_provider() {
    plays_and_seeks_through_a_synchronous_on_demand_video_provider(&bundled());
}

#[test]
fn playback_plays_and_seeks_a_webm_through_a_synchronous_on_demand_video_provider() {
    plays_and_seeks_through_a_synchronous_on_demand_video_provider(&webm());
}

fn plays_and_seeks_through_a_synchronous_on_demand_video_provider(bundled: &Bundled) {
    let total_video_bytes: u64 = bundled
        .video
        .samples
        .iter()
        .map(|sample| u64::from(sample.size))
        .sum();
    let cache = CachingByteSource::new(
        MemorySource::new(bundled.bytes.clone()),
        64 * 1024,
        total_video_bytes / 10,
    )
    .unwrap();
    let video = ExactFrameReader::from_provider(
        bundled.decoder.as_ref(),
        bundled.video_configuration(),
        Box::new(Mp4SampleProvider::new(bundled.video.clone(), cache).unwrap()),
        video_limits(),
    )
    .unwrap();
    let audio = bundled.audio_reader(Box::new(bundled.eager_audio_packets()));
    let backend = Backend::default();
    let timeline = bundled.timeline();
    let mut playback = PlaybackController::new_with_indexed_timeline(
        video,
        audio,
        NativeAudioOutput(backend.clone()),
        timeline.clone(),
        bundled.options(),
    )
    .unwrap();
    let mut eager = bundled.eager_video_reader();
    let cancellation = CancellationToken::new();

    playback.play().unwrap();
    for frame in [0, 1, 2, 5] {
        *backend.clock.lock().unwrap() = presentation_start(&timeline, frame);
        let (presentation, picture) = playback.present().unwrap();
        assert_eq!(presentation.frame, Some(FrameIndex(frame)));
        assert_eq!(
            picture.unwrap().planes[0].data,
            eager.get(FrameIndex(frame), &cancellation).unwrap().planes[0].data
        );
    }
    let target = 48;
    playback.seek(FrameIndex(target)).unwrap();
    let (presentation, picture) = playback.present().unwrap();
    assert_eq!(presentation.frame, Some(FrameIndex(target)));
    assert_eq!(
        picture.unwrap().planes[0].data,
        eager.get(FrameIndex(target), &cancellation).unwrap().planes[0].data
    );
    assert_scheduled_audio_matches(bundled, &backend.scheduled.lock().unwrap());
}

/// Retries `step` until it stops reporting `WouldBlock`, awaiting the
/// controller's prefetch in between: the browser's render loop.
fn until_loaded<T, V, A, O>(
    playback: &mut PlaybackController<V, A, O>,
    mut step: impl FnMut(&mut PlaybackController<V, A, O>) -> Result<T>,
) -> (T, usize)
where
    V: zvidlib::PrefetchVideoSource,
    A: zvidlib::PrefetchAudioSource,
    O: zvidlib::PlaybackAudioOutput,
{
    for prefetches in 0..16 {
        match step(playback) {
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                block_on(playback.prefetch()).unwrap();
            }
            result => return (result.unwrap(), prefetches),
        }
    }
    panic!("playback still had not loaded what it needed after 16 prefetches");
}

/// Browser playback: every read of the source suspends, so the controller's
/// synchronous calls must never be the ones to read it. They report
/// `WouldBlock`, the prefetch loads what they were missing, and the frames and
/// audio are exactly the eager path's. The video budget holds only the first
/// 24 samples of the track's single group of pictures, so the walk to frame
/// 60 streams through it over several prefetches.
#[test]
fn playback_plays_and_seeks_through_on_demand_sources_over_a_suspending_source() {
    let bundled = bundled();
    let total_video_bytes: u64 = bundled
        .video
        .samples
        .iter()
        .map(|sample| u64::from(sample.size))
        .sum();
    assert!(first_samples_bytes(&bundled, 24) < total_video_bytes / 10);
    plays_and_seeks_through_on_demand_sources_over_a_suspending_source(&bundled);
}

/// The same over a WebM, whose 24-sample budget holds less than one of its
/// two groups of pictures.
#[test]
fn playback_plays_and_seeks_a_webm_through_on_demand_sources_over_a_suspending_source() {
    plays_and_seeks_through_on_demand_sources_over_a_suspending_source(&webm());
}

/// The compressed bytes of the track's first `count` video samples.
fn first_samples_bytes(bundled: &Bundled, count: usize) -> u64 {
    bundled.video.samples[..count]
        .iter()
        .map(|sample| u64::from(sample.size))
        .sum()
}

fn plays_and_seeks_through_on_demand_sources_over_a_suspending_source(bundled: &Bundled) {
    let video_budget = first_samples_bytes(bundled, 24);
    let video_loader = Mp4SampleLoader::new(
        bundled.video.clone(),
        SuspendingSource::new(bundled.bytes.clone()),
        video_budget,
    )
    .unwrap();
    let video_reader = ExactFrameReader::from_provider(
        bundled.decoder.as_ref(),
        bundled.video_configuration(),
        Box::new(video_loader.sample_provider().unwrap()),
        video_limits(),
    )
    .unwrap();
    let video = OnDemandVideoSource::new(video_reader, video_loader, 16);

    let audio_loader = Mp4SampleLoader::new(
        bundled.audio.clone(),
        SuspendingSource::new(bundled.bytes.clone()),
        256 * 1024,
    )
    .unwrap();
    let audio_reader = bundled.audio_reader(bundled.audio_packets(&audio_loader));
    let audio = OnDemandAudioSource::new(audio_reader, audio_loader, 32);

    let backend = Backend::default();
    let timeline = bundled.timeline();
    let mut playback = PlaybackController::new_with_indexed_timeline(
        video,
        audio,
        WebAudioOutput(backend.clone()),
        timeline.clone(),
        bundled.options(),
    )
    .unwrap();
    let mut eager = bundled.eager_video_reader();
    let cancellation = CancellationToken::new();

    // Nothing is loaded yet: starting playback asks for audio it does not
    // have, and says so instead of reading the source itself.
    let error = playback.play().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::WouldBlock);
    until_loaded(&mut playback, |playback| playback.play());

    // The controller measures media time from the clock reading at the last
    // seek, so the clock reading that presents a frame is relative to it.
    let anchor = Cell::new((0_u64, 0_u64));
    let mut present = |playback: &mut PlaybackController<_, _, _>, frame: u64| {
        let (clock, media) = anchor.get();
        *backend.clock.lock().unwrap() = clock + presentation_start(&timeline, frame) - media;
        let ((presentation, picture), _) = until_loaded(playback, |playback| playback.present());
        assert_eq!(presentation.frame, Some(FrameIndex(frame)));
        let picture: VideoFrame = picture.unwrap();
        assert_eq!(
            picture.planes[0].data,
            eager.get(FrameIndex(frame), &cancellation).unwrap().planes[0].data,
            "frame {frame} differs from the eager reader's"
        );
    };
    // A seek reads nothing it has to wait for, so it never reports anything
    // missing itself; the frame after it is loaded by the prefetch.
    let seek = |playback: &mut PlaybackController<_, _, _>, frame: u64| {
        playback.seek(FrameIndex(frame)).unwrap();
        anchor.set((
            *backend.clock.lock().unwrap(),
            presentation_start(&timeline, frame),
        ));
    };
    for frame in [0, 1, 2, 3, 8] {
        present(&mut playback, frame);
    }
    seek(&mut playback, 60);
    present(&mut playback, 60);
    seek(&mut playback, 20);
    present(&mut playback, 20);
    present(&mut playback, 21);

    // Paused, the current frame comes through the same loop.
    playback.pause().unwrap();
    seek(&mut playback, 30);
    let (picture, _) = until_loaded(&mut playback, |playback| playback.current_frame());
    assert_eq!(
        picture.planes[0].data,
        eager.get(FrameIndex(30), &cancellation).unwrap().planes[0].data
    );

    assert_scheduled_audio_matches(bundled, &backend.scheduled.lock().unwrap());
}

/// Prefetching ahead of time - between frames, as a render loop does - means
/// the next frames never report anything missing at all.
#[test]
fn prefetching_ahead_keeps_playback_from_reporting_missing_samples() {
    prefetching_ahead_keeps_playback_from_reporting_missing_samples_of(&bundled());
}

#[test]
fn prefetching_ahead_keeps_webm_playback_from_reporting_missing_samples() {
    prefetching_ahead_keeps_playback_from_reporting_missing_samples_of(&webm());
}

fn prefetching_ahead_keeps_playback_from_reporting_missing_samples_of(bundled: &Bundled) {
    let video_loader = Mp4SampleLoader::new(
        bundled.video.clone(),
        SuspendingSource::new(bundled.bytes.clone()),
        8 * 1024 * 1024,
    )
    .unwrap();
    let video_reader = ExactFrameReader::from_provider(
        bundled.decoder.as_ref(),
        bundled.video_configuration(),
        Box::new(video_loader.sample_provider().unwrap()),
        video_limits(),
    )
    .unwrap();
    let audio_loader = Mp4SampleLoader::new(
        bundled.audio.clone(),
        SuspendingSource::new(bundled.bytes.clone()),
        256 * 1024,
    )
    .unwrap();
    let audio_reader = bundled.audio_reader(bundled.audio_packets(&audio_loader));
    let backend = Backend::default();
    let timeline = bundled.timeline();
    let mut playback = PlaybackController::new_with_indexed_timeline(
        OnDemandVideoSource::new(video_reader, video_loader, 32),
        OnDemandAudioSource::new(audio_reader, audio_loader, 64),
        WebAudioOutput(backend.clone()),
        timeline.clone(),
        bundled.options(),
    )
    .unwrap();

    block_on(playback.prefetch()).unwrap();
    playback.play().unwrap();
    for frame in 0..12 {
        *backend.clock.lock().unwrap() = presentation_start(&timeline, frame);
        let (presentation, _) = playback.present().unwrap();
        assert_eq!(presentation.frame, Some(FrameIndex(frame)));
        block_on(playback.prefetch()).unwrap();
    }
}
