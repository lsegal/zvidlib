//! `OnDemandPlayer` opens an MP4 in one call, plays and seeks it within its
//! byte budgets without the caller loading anything, and switches audio tracks
//! by position or language mid-playback (issue #689).

#![cfg(all(any(unix, windows), not(target_arch = "wasm32")))]

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use zvidlib::io::{ByteSource, FileSource, IoFuture, MemorySink, MemorySource};
use zvidlib::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::transfer::{CpuFrameSource, FrameSource, Orientation};
use zvidlib::webm::WebmMuxer;
use zvidlib::{
    AudioBuffer, AudioEncoderConfig, AudioEncoderFactory, AudioOutputBackend, AudioOutputOpener,
    Codec, CodecProfile, ColorRange, ErrorKind, FrameIndex, HardwarePreference, Limits, Mp4Demuxer,
    Mp4DemuxerOptions, NativeAudioOutput, OnDemandOptions, OnDemandPlayer, PixelFormat, Plane,
    PlaybackAudioOutput, SampleRange, TrackKind, VideoDimensions, VideoEncoderConfig,
    VideoEncoderFactory, VideoFrame, native_av1_video_encoder_factory,
    native_opus_audio_encoder_factory,
};

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
    }
}

const RATE: u64 = 30;
const FRAMES: u64 = 60;
const LANGUAGES: [&str; 3] = ["eng", "fra", "deu"];

/// The gray levels of frame `index` of [`movie`], a moving gradient.
fn gray(index: u64) -> Vec<u8> {
    (0..18_u32)
        .flat_map(|y| (0..32_u32).map(move |x| ((x * 7 + y * 3 + index as u32 * 5) % 256) as u8))
        .collect()
}

/// `codec` packets, Opus or Vorbis, of `frames` stereo samples of a tone at
/// `frequency`.
fn audio_track(
    codec: Codec,
    frames: u64,
    frequency: f32,
) -> (
    Mp4TrackConfig,
    Vec<zvidlib::EncodedSample>,
    zvidlib::AudioGapless,
) {
    let samples: Vec<f32> = (0..frames)
        .flat_map(|i| {
            let level = 0.3 * (2.0 * std::f32::consts::PI * frequency * i as f32 / 48_000.0).sin();
            [level, level]
        })
        .collect();
    let (factory, profile, configuration): (Box<dyn AudioEncoderFactory>, _, _) = match codec {
        Codec::Opus => (
            Box::new(native_opus_audio_encoder_factory()),
            CodecProfile::Opus,
            96_000_u32.to_be_bytes().to_vec(),
        ),
        #[cfg(feature = "vorbis-encoder")]
        Codec::Vorbis => (
            Box::new(zvidlib::native_vorbis_audio_encoder_factory()),
            CodecProfile::Vorbis,
            Vec::new(),
        ),
        _ => unreachable!("the test movies' audio is Opus or Vorbis"),
    };
    let mut encoder = factory
        .create(
            &AudioEncoderConfig {
                codec,
                profile,
                sample_rate: 48_000,
                channels: 2,
                timescale: 48_000,
                configuration,
            },
            &Limits::default(),
        )
        .unwrap();
    let buffer = AudioBuffer::new(
        SampleRange::new(0, frames).unwrap(),
        48_000,
        2,
        samples,
        &Limits::default(),
    )
    .unwrap();
    let mut packets = block_on(encoder.encode(FrameIndex(0), buffer)).unwrap();
    let drain = block_on(encoder.finish()).unwrap();
    packets.extend(drain.samples);
    let track = Mp4TrackConfig {
        encoder: encoder.config().clone(),
        format: Mp4TrackFormat::Audio { channels: 2 },
    };
    (track, packets, drain.gapless)
}

/// Two seconds of lossless monochrome 32x18 AV1 at 30 fps, and `audio_tracks`
/// Opus tracks beside it, each a different tone in the language
/// [`LANGUAGES`] gives it.
fn movie(audio_tracks: usize) -> Vec<u8> {
    let (configs, samples, gapless) = tracks(audio_tracks);
    let mut muxer = block_on(Mp4Muxer::new(MemorySink::new(), configs, 100_000)).unwrap();
    for (track, samples) in samples.into_iter().enumerate() {
        for sample in samples {
            block_on(muxer.write_sample(track, sample)).unwrap();
        }
    }
    for (index, gapless) in gapless.into_iter().enumerate() {
        muxer.set_audio_gapless(index + 1, gapless).unwrap();
    }
    let mut bytes = block_on(muxer.finish()).unwrap().into_inner();
    label_languages(&mut bytes);
    bytes
}

/// [`movie`]'s tracks as a WebM, whose audio is timed by its `CodecDelay`
/// rather than an edit list and whose tracks have no language (issue #685).
fn webm(audio_tracks: usize) -> Vec<u8> {
    let (configs, samples, gapless) = tracks(audio_tracks);
    webm_of(configs, samples, gapless)
}

/// A WebM of the given tracks, video first, and each audio track's gapless
/// trim.
fn webm_of(
    configs: Vec<Mp4TrackConfig>,
    samples: Vec<Vec<zvidlib::EncodedSample>>,
    gapless: Vec<zvidlib::AudioGapless>,
) -> Vec<u8> {
    let timescales: Vec<u32> = configs
        .iter()
        .map(|config| config.encoder.timescale)
        .collect();
    // A WebM cluster interleaves its tracks, so the samples are written in
    // presentation-time order across all of them.
    let mut ordered: Vec<_> = samples
        .into_iter()
        .enumerate()
        .flat_map(|(track, samples)| samples.into_iter().map(move |sample| (track, sample)))
        .collect();
    ordered.sort_by(|(a, a_sample), (b, b_sample)| {
        let seconds = |track: usize, pts: i64| pts as f64 / f64::from(timescales[track]);
        seconds(*a, a_sample.pts).total_cmp(&seconds(*b, b_sample.pts))
    });
    let mut remaining = vec![0_usize; timescales.len()];
    for (track, _) in &ordered {
        remaining[*track] += 1;
    }
    let mut muxer = block_on(WebmMuxer::new(MemorySink::new(), configs, 100_000)).unwrap();
    for (track, sample) in ordered {
        block_on(muxer.write_sample(track, sample)).unwrap();
        remaining[track] -= 1;
        // An audio track's end trim is written with its last block, before
        // the other tracks' later samples.
        if track > 0 && remaining[track] == 0 {
            muxer.set_audio_gapless(track, gapless[track - 1]).unwrap();
        }
    }
    block_on(muxer.finish()).unwrap().into_inner()
}

/// The configurations of [`movie`]'s video track and `audio_tracks` Opus
/// tracks, each track's samples, and each audio track's gapless trim.
fn tracks(
    audio_tracks: usize,
) -> (
    Vec<Mp4TrackConfig>,
    Vec<Vec<zvidlib::EncodedSample>>,
    Vec<zvidlib::AudioGapless>,
) {
    let limits = Limits::default();
    let dimensions = VideoDimensions::new(32, 18, &limits).unwrap();
    let mut encoder = native_av1_video_encoder_factory()
        .create(
            &VideoEncoderConfig {
                codec: Codec::Av1,
                profile: CodecProfile::Av1Main,
                coded_dimensions: dimensions,
                input_format: PixelFormat::Gray8,
                color_range: ColorRange::Full,
                hardware: HardwarePreference::Avoid,
                timescale: RATE as u32,
                frame_duration: 1,
                configuration: Vec::new(),
            },
            &limits,
        )
        .unwrap();
    let audio: Vec<_> = (0..audio_tracks)
        .map(|index| {
            audio_track(
                Codec::Opus,
                48_000 * FRAMES / RATE,
                330.0 * (index + 1) as f32,
            )
        })
        .collect();
    let mut video = Vec::new();
    for index in 0..FRAMES {
        let frame = VideoFrame::new(
            dimensions,
            PixelFormat::Gray8,
            ColorRange::Full,
            vec![Plane {
                data: gray(index),
                stride: dimensions.width as usize,
            }],
            &limits,
        )
        .unwrap();
        video.extend(
            block_on(encoder.encode(
                FrameIndex(index),
                FrameSource::Cpu(CpuFrameSource {
                    frame: &frame,
                    orientation: Orientation::TopLeft,
                }),
            ))
            .unwrap(),
        );
    }
    video.extend(block_on(encoder.finish()).unwrap());
    let mut configs = vec![Mp4TrackConfig {
        encoder: encoder.config().clone(),
        format: Mp4TrackFormat::Video(dimensions),
    }];
    let mut samples = vec![video];
    let mut gapless = Vec::new();
    for (config, packets, trim) in audio {
        configs.push(config);
        samples.push(packets);
        gapless.push(trim);
    }
    (configs, samples, gapless)
}

/// Gives each audio track of [`movie`] its language from [`LANGUAGES`]: the
/// muxer marks every track undetermined. The `mdhd` boxes come in track
/// order, video first, and each packs its code 20 bytes into the box's
/// version-0 body.
fn label_languages(bytes: &mut [u8]) {
    let boxes: Vec<usize> = bytes
        .windows(4)
        .enumerate()
        .filter(|(_, kind)| kind == b"mdhd")
        .map(|(at, _)| at)
        .collect();
    for (&at, language) in boxes.iter().skip(1).zip(LANGUAGES) {
        assert_eq!(bytes[at + 4], 0, "the muxer writes version-0 mdhd boxes");
        let packed = language.bytes().fold(0_u16, |packed, letter| {
            (packed << 5) | u16::from(letter - 0x60)
        });
        bytes[at + 24..at + 26].copy_from_slice(&packed.to_be_bytes());
    }
}

/// A shared clock the test moves by hand, in samples of whatever rate the
/// output was opened at.
#[derive(Clone, Default)]
struct Clock(Arc<Mutex<u64>>);

impl Clock {
    fn advance(&self, samples: u64) {
        *self.0.lock().unwrap() += samples;
    }
}

/// What one output the player opened was asked to play.
#[derive(Default)]
struct Opened {
    sample_rate: u32,
    channels: u16,
    scheduled: Vec<SampleRange>,
}

struct Backend {
    clock: Clock,
    opened: Arc<Mutex<Opened>>,
}

impl AudioOutputBackend for Backend {
    fn clock_samples(&self) -> u64 {
        *self.clock.0.lock().unwrap()
    }
    fn start(&mut self, _: u64) -> zvidlib::Result<()> {
        Ok(())
    }
    fn schedule(&mut self, buffer: AudioBuffer, _: u64) -> zvidlib::Result<()> {
        self.opened.lock().unwrap().scheduled.push(buffer.range);
        Ok(())
    }
    fn cancel_queued(&mut self, _: u64) -> zvidlib::Result<()> {
        Ok(())
    }
    fn stop(&mut self) -> zvidlib::Result<()> {
        Ok(())
    }
}

/// Outputs on one hand-moved clock, recording each one the player opens.
/// Every output an opener has opened, in order.
type OpenedOutputs = Arc<Mutex<Vec<Arc<Mutex<Opened>>>>>;

fn outputs(clock: &Clock) -> (AudioOutputOpener, OpenedOutputs) {
    let all = Arc::new(Mutex::new(Vec::new()));
    let (clock, recorded) = (clock.clone(), Arc::clone(&all));
    let opener: AudioOutputOpener = Box::new(move |sample_rate, channels| {
        let opened = Arc::new(Mutex::new(Opened {
            sample_rate,
            channels,
            scheduled: Vec::new(),
        }));
        recorded.lock().unwrap().push(Arc::clone(&opened));
        Ok(Box::new(NativeAudioOutput(Backend {
            clock: clock.clone(),
            opened,
        })) as Box<dyn PlaybackAudioOutput>)
    });
    (opener, all)
}

fn assert_gray(frame: &VideoFrame, index: u64) {
    let plane = &frame.planes[0];
    let row = frame.dimensions.width as usize * 4;
    let levels: Vec<u8> = plane
        .data
        .chunks(plane.stride)
        .take(frame.dimensions.height as usize)
        .flat_map(|line| line[..row].chunks(4).map(|pixel| pixel[0]))
        .collect();
    assert_eq!(levels, gray(index), "frame {index}");
}

#[test]
fn reads_each_audio_tracks_language_from_its_media_header() {
    let source = MemorySource::new(movie(3));
    let demuxer = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
    let languages: Vec<_> = demuxer
        .tracks
        .iter()
        .map(|track| (track.kind, track.language.as_deref()))
        .collect();
    assert_eq!(
        languages,
        [
            (TrackKind::Video, Some("und")),
            (TrackKind::Audio, Some("eng")),
            (TrackKind::Audio, Some("fra")),
            (TrackKind::Audio, Some("deu")),
        ]
    );
}

#[test]
fn opens_a_file_and_plays_it_without_the_caller_loading_anything() {
    let path = std::env::temp_dir().join(format!(
        "zvidlib-on-demand-player-{}.mp4",
        std::process::id()
    ));
    std::fs::write(&path, movie(1)).unwrap();
    let clock = Clock::default();
    let (opener, opened) = outputs(&clock);
    let mut player = OnDemandPlayer::with_output(
        FileSource::open(&path).unwrap(),
        OnDemandOptions::default(),
        opener,
    )
    .unwrap();
    assert_eq!(player.frame_count(), FRAMES);
    assert_eq!(player.sample_rate(), Some(48_000));
    assert_eq!(player.audio_track(), Some(0));
    {
        let opened = opened.lock().unwrap();
        assert_eq!(opened.len(), 1);
        let first = opened[0].lock().unwrap();
        assert_eq!((first.sample_rate, first.channels), (48_000, 2));
    }

    assert_gray(&player.current_frame().unwrap(), 0);
    player.play().unwrap();
    let mut presented = Vec::new();
    for _ in 0..20 {
        clock.advance(48_000 / RATE);
        if let (_, Some(frame)) = player.present().unwrap() {
            let index = player.current_frame_index().unwrap();
            assert_gray(&frame, index.0);
            presented.push(index.0);
        }
    }
    assert_eq!(presented, (1..=20).collect::<Vec<_>>());
    assert!(
        !opened.lock().unwrap()[0]
            .lock()
            .unwrap()
            .scheduled
            .is_empty()
    );

    player.seek(FrameIndex(45)).unwrap();
    assert!(player.is_playing());
    let (presentation, frame) = player.present().unwrap();
    assert_eq!(presentation.frame, Some(FrameIndex(45)));
    assert_gray(&frame.unwrap(), 45);
    drop(player);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn switching_audio_tracks_keeps_the_frame_and_releases_the_old_tracks_packets() {
    let clock = Clock::default();
    let (opener, opened) = outputs(&clock);
    let options = OnDemandOptions {
        audio_budget_bytes: 16 * 1024,
        ..OnDemandOptions::default()
    };
    let mut player =
        OnDemandPlayer::with_output(MemorySource::new(movie(3)), options, opener).unwrap();
    let languages: Vec<_> = player
        .audio_tracks()
        .iter()
        .map(|track| track.language.clone().unwrap())
        .collect();
    assert_eq!(languages, LANGUAGES);

    player.seek(FrameIndex(10)).unwrap();
    player.play().unwrap();
    clock.advance(48_000 / RATE);
    let (presentation, _) = player.present().unwrap();
    assert_eq!(presentation.frame, Some(FrameIndex(11)));
    assert!(player.audio_resident_bytes() > 0);

    player.select_audio_language("fra").unwrap();
    assert_eq!(player.audio_track(), Some(1));
    assert!(player.is_playing());
    assert_eq!(player.current_frame_index().unwrap(), FrameIndex(11));
    // Nothing of the old track is held, and nothing of the new one yet.
    assert_eq!(player.audio_resident_bytes(), 0);

    // The new track plays on its own output from the frame's interval.
    let (presentation, _) = player.present().unwrap();
    assert_eq!(presentation.requested_frame, FrameIndex(11));
    assert!(player.audio_resident_bytes() > 0);
    assert!(player.audio_resident_bytes() <= options.audio_budget_bytes);
    {
        let opened = opened.lock().unwrap();
        assert_eq!(opened.len(), 2);
        let scheduled = &opened[1].lock().unwrap().scheduled;
        assert_eq!(scheduled.first().unwrap().start, 11 * 48_000 / RATE);
    }
    clock.advance(48_000 / RATE);
    let (_, frame) = player.present().unwrap();
    assert_gray(&frame.unwrap(), 12);

    // Paused, a switch stays paused on the same frame.
    player.pause().unwrap();
    let paused_on = player.current_frame_index().unwrap();
    player.select_audio_track(2).unwrap();
    assert!(!player.is_playing());
    assert_eq!(player.current_frame_index().unwrap(), paused_on);
    assert_gray(&player.current_frame().unwrap(), paused_on.0);
}

/// Issue #685: a WebM opens through the same path an MP4 does, plays and
/// seeks with its audio scheduled from the frame's interval, and switches
/// between its audio tracks.
#[test]
fn plays_seeks_and_switches_the_audio_of_a_webm() {
    let clock = Clock::default();
    let (opener, opened) = outputs(&clock);
    let mut player = OnDemandPlayer::with_output(
        MemorySource::new(webm(2)),
        OnDemandOptions::default(),
        opener,
    )
    .unwrap();
    assert_eq!(player.frame_count(), FRAMES);
    assert_eq!(player.sample_rate(), Some(48_000));
    assert_eq!(player.audio_tracks().len(), 2);
    assert!(
        player
            .audio_tracks()
            .iter()
            .all(|track| track.codec == Codec::Opus && track.language.is_none())
    );

    assert_gray(&player.current_frame().unwrap(), 0);
    player.play().unwrap();
    // A WebM times its blocks in whole milliseconds, so the clock is read in
    // the middle of each frame rather than at its rounded start.
    clock.advance(48_000 / RATE / 2);
    for frame in 1..=5 {
        clock.advance(48_000 / RATE);
        let (presentation, picture) = player.present().unwrap();
        assert_eq!(presentation.frame, Some(FrameIndex(frame)));
        assert_gray(&picture.unwrap(), frame);
    }
    assert_eq!(
        opened.lock().unwrap()[0]
            .lock()
            .unwrap()
            .scheduled
            .first()
            .unwrap()
            .start,
        0
    );

    player.seek(FrameIndex(45)).unwrap();
    let (presentation, picture) = player.present().unwrap();
    assert_eq!(presentation.frame, Some(FrameIndex(45)));
    assert_gray(&picture.unwrap(), 45);

    player.select_audio_track(1).unwrap();
    assert_eq!(player.audio_track(), Some(1));
    assert_eq!(player.current_frame_index().unwrap(), FrameIndex(45));
    player.present().unwrap();
    let opened = opened.lock().unwrap();
    assert_eq!(opened.len(), 2);
    let scheduled = &opened[1].lock().unwrap().scheduled;
    assert_eq!(scheduled.first().unwrap().start, 45 * 48_000 / RATE);
}

/// Issue #686: a WebM whose audio is Vorbis plays and seeks, backward too,
/// with its audio scheduled from each frame's interval: its packets'
/// intervals come from the first bytes the demuxer recorded, so nothing but
/// the packets playback reaches is loaded, within the audio budget.
#[cfg(all(feature = "vorbis-decoder", feature = "vorbis-encoder"))]
#[test]
fn plays_and_seeks_a_webm_with_vorbis_audio() {
    let (mut configs, mut samples, _) = tracks(0);
    let (config, packets, gapless) = audio_track(Codec::Vorbis, 48_000 * FRAMES / RATE, 440.0);
    configs.push(config);
    samples.push(packets);
    let clock = Clock::default();
    let (opener, opened) = outputs(&clock);
    let options = OnDemandOptions::default();
    let audio_budget = options.audio_budget_bytes;
    let mut player = OnDemandPlayer::with_output(
        MemorySource::new(webm_of(configs, samples, vec![gapless])),
        options,
        opener,
    )
    .unwrap();
    assert_eq!(player.sample_rate(), Some(48_000));
    assert_eq!(player.audio_tracks().len(), 1);
    assert_eq!(player.audio_tracks()[0].codec, Codec::Vorbis);

    assert_gray(&player.current_frame().unwrap(), 0);
    player.play().unwrap();
    // A WebM times its blocks in whole milliseconds, so the clock is read in
    // the middle of each frame rather than at its rounded start.
    clock.advance(48_000 / RATE / 2);
    for frame in 1..=5 {
        clock.advance(48_000 / RATE);
        let (presentation, picture) = player.present().unwrap();
        assert_eq!(presentation.frame, Some(FrameIndex(frame)));
        assert_gray(&picture.unwrap(), frame);
    }
    for frame in [45, 10] {
        player.seek(FrameIndex(frame)).unwrap();
        let (presentation, picture) = player.present().unwrap();
        assert_eq!(presentation.frame, Some(FrameIndex(frame)));
        assert_gray(&picture.unwrap(), frame);
    }
    assert!(player.audio_resident_bytes() <= audio_budget);

    let opened = opened.lock().unwrap();
    let output = opened[0].lock().unwrap();
    assert_eq!(output.scheduled.first().unwrap().start, 0);
    assert_eq!(output.scheduled.first().unwrap().start, 0);
    // Each seek schedules from its frame's start, which the WebM gives in
    // whole milliseconds.
    for frame in [45, 10] {
        assert!(
            output
                .scheduled
                .iter()
                .any(|range| range.start == frame * 1_000 / RATE * 48),
            "nothing was scheduled from frame {frame}"
        );
    }
}

#[test]
fn refuses_an_audio_track_or_language_the_input_lacks() {
    let clock = Clock::default();
    let (opener, _) = outputs(&clock);
    let bytes = movie(2);
    let error = OnDemandPlayer::with_output(
        MemorySource::new(bytes.clone()),
        OnDemandOptions {
            audio_track: 2,
            ..OnDemandOptions::default()
        },
        opener,
    )
    .err()
    .unwrap();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);

    let (opener, _) = outputs(&clock);
    let mut player =
        OnDemandPlayer::with_output(MemorySource::new(bytes), OnDemandOptions::default(), opener)
            .unwrap();
    assert_eq!(
        player.select_audio_track(2).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(
        player.select_audio_language("deu").unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    assert_eq!(player.audio_track(), Some(0));
}

#[test]
fn plays_a_video_only_input_on_its_own_clock() {
    let clock = Clock::default();
    let (opener, opened) = outputs(&clock);
    let mut player = OnDemandPlayer::with_output(
        MemorySource::new(movie(0)),
        OnDemandOptions::default(),
        opener,
    )
    .unwrap();
    assert!(opened.lock().unwrap().is_empty());
    assert_eq!(player.sample_rate(), None);
    assert_eq!(player.audio_track(), None);
    player.seek(FrameIndex(30)).unwrap();
    assert_gray(&player.current_frame().unwrap(), 30);
    player.play().unwrap();
    let (presentation, _) = player.present().unwrap();
    assert!(presentation.requested_frame >= FrameIndex(30));
}

/// A source whose every read suspends once before completing, the way a
/// network read does, so a caller that polls it once gets no answer.
#[derive(Clone)]
struct SuspendingSource {
    inner: Rc<MemorySource>,
    reads: Rc<Cell<usize>>,
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

#[test]
fn waits_on_a_source_whose_reads_suspend() {
    let clock = Clock::default();
    let (opener, _) = outputs(&clock);
    let source = SuspendingSource {
        inner: Rc::new(MemorySource::new(movie(1))),
        reads: Rc::new(Cell::new(0)),
    };
    let mut player =
        OnDemandPlayer::with_output(source.clone(), OnDemandOptions::default(), opener).unwrap();
    player.play().unwrap();
    for index in 1..=5 {
        clock.advance(48_000 / RATE);
        let (_, frame) = player.present().unwrap();
        assert_gray(&frame.unwrap(), index);
    }
    assert!(source.reads.get() > 0);
}

/// The bundled AV1 and AAC sample, read through budgets too small for the
/// frames and audio these seeks reach - three of its largest video samples,
/// and about two seconds of its audio - never holds more compressed data than
/// they allow while it plays and seeks back and forth.
#[cfg(feature = "aac-decoder")]
#[test]
fn plays_and_seeks_the_bundled_sample_within_its_byte_budgets() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/media/BigBuckBunny.av1.mp4");
    let source = FileSource::open(&path).unwrap();
    let demuxer = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
    let largest = |kind| {
        let track = demuxer
            .tracks
            .iter()
            .find(|track| track.kind == kind)
            .unwrap();
        let largest = track
            .samples
            .iter()
            .map(|sample| u64::from(sample.size))
            .max()
            .unwrap();
        let total: u64 = track
            .samples
            .iter()
            .map(|sample| u64::from(sample.size))
            .sum();
        (largest, total)
    };
    let (largest_video, total_video) = largest(TrackKind::Video);
    let (largest_audio, total_audio) = largest(TrackKind::Audio);
    let options = OnDemandOptions {
        video_budget_bytes: largest_video * 3,
        audio_budget_bytes: largest_audio * 64,
        max_cached_frames: 4,
        ..OnDemandOptions::default()
    };
    assert!(options.video_budget_bytes < total_video / 4);
    assert!(options.audio_budget_bytes < total_audio / 4);

    let clock = Clock::default();
    let (opener, opened) = outputs(&clock);
    let mut player = OnDemandPlayer::with_output(source, options, opener).unwrap();
    let rate = u64::from(player.sample_rate().unwrap());
    let within_budgets = |player: &OnDemandPlayer| {
        assert!(player.video_resident_bytes() <= options.video_budget_bytes);
        assert!(player.audio_resident_bytes() <= options.audio_budget_bytes);
    };
    player.play().unwrap();
    // The sample is one group of pictures, so a seek decodes every frame
    // before its target: these stay near the start to keep the test quick,
    // and still move back and forth across what the budgets can hold.
    for target in [0, 48, 16, 72, 24] {
        player.seek(FrameIndex(target)).unwrap();
        within_budgets(&player);
        for _ in 0..6 {
            clock.advance(rate / 24);
            player.present().unwrap();
            within_budgets(&player);
        }
    }
    let scheduled = opened.lock().unwrap()[0].lock().unwrap().scheduled.len();
    assert!(scheduled > 0);
}
