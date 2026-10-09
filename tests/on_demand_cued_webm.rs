//! A WebM with `Cues` opens for on-demand playback from its header elements
//! and `Cues` alone, however long it is, and indexes its blocks a cue span at
//! a time as playback and seeks reach them (issue #692). It plays and seeks to
//! exactly the frames and audio the whole file's index gives, which the same
//! file with its `Cues` hidden still plays from.

#![cfg(all(any(unix, windows), not(target_arch = "wasm32")))]

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use zvidlib::io::{ByteSource, IoFuture, MemorySink, MemorySource};
use zvidlib::mp4::{Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::transfer::{CpuFrameSource, FrameSource, Orientation};
use zvidlib::webm::WebmMuxer;
use zvidlib::{
    AudioBuffer, AudioEncoderConfig, AudioEncoderFactory, AudioGapless, AudioOutputBackend,
    AudioOutputOpener, Codec, CodecProfile, ColorRange, EncodedSample, FrameIndex,
    HardwarePreference, Limits, NativeAudioOutput, OnDemandOptions, OnDemandPlayer, PixelFormat,
    Plane, PlaybackAudioOutput, SampleRange, VideoDimensions, VideoEncoderConfig,
    VideoEncoderFactory, VideoFrame, native_opus_audio_encoder_factory,
    native_vp8_video_encoder_factory,
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

/// Frames a second, and so in each group of pictures: the VP8 encoder starts
/// a key frame every second, and the muxer opens a cued Cluster on each.
const RATE: u64 = 30;
const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;

/// `seconds` of 64x48 VP8 at 30 frames a second, a key frame opening each
/// second, every frame a different picture.
fn video(seconds: u64) -> (Mp4TrackConfig, Vec<EncodedSample>) {
    let limits = Limits::default();
    let dimensions = VideoDimensions::new(WIDTH, HEIGHT, &limits).unwrap();
    let mut encoder = native_vp8_video_encoder_factory()
        .create(
            &VideoEncoderConfig {
                codec: Codec::Vp8,
                profile: CodecProfile::Vp8,
                coded_dimensions: dimensions,
                input_format: PixelFormat::Rgba8,
                color_range: ColorRange::Limited,
                hardware: HardwarePreference::Avoid,
                timescale: RATE as u32,
                frame_duration: 1,
                configuration: Vec::new(),
            },
            &limits,
        )
        .unwrap();
    let mut samples = Vec::new();
    for index in 0..seconds * RATE {
        let data = (0..HEIGHT)
            .flat_map(|y| {
                (0..WIDTH).flat_map(move |x| {
                    let level = ((x * 3 + y * 5 + index as u32 * 7) % 200 + 20) as u8;
                    [level, 255 - level, (index * 4 % 256) as u8, 255]
                })
            })
            .collect();
        let frame = VideoFrame::new(
            dimensions,
            PixelFormat::Rgba8,
            ColorRange::Limited,
            vec![Plane {
                data,
                stride: WIDTH as usize * 4,
            }],
            &limits,
        )
        .unwrap();
        samples.extend(
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
    samples.extend(block_on(encoder.finish()).unwrap());
    let config = Mp4TrackConfig {
        encoder: encoder.config().clone(),
        format: Mp4TrackFormat::Video(dimensions),
    };
    (config, samples)
}

/// `seconds` of a stereo Opus tone that sweeps up, so no two packets decode
/// alike.
fn audio(seconds: u64) -> (Mp4TrackConfig, Vec<EncodedSample>, AudioGapless) {
    let frames = seconds * 48_000;
    let samples: Vec<f32> = (0..frames)
        .flat_map(|i| {
            let time = i as f32 / 48_000.0;
            let level = 0.3 * (2.0 * std::f32::consts::PI * (220.0 + 40.0 * time) * time).sin();
            [level, -level]
        })
        .collect();
    let mut encoder = native_opus_audio_encoder_factory()
        .create(
            &AudioEncoderConfig {
                codec: Codec::Opus,
                profile: CodecProfile::Opus,
                sample_rate: 48_000,
                channels: 2,
                timescale: 48_000,
                configuration: 64_000_u32.to_be_bytes().to_vec(),
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
    let config = Mp4TrackConfig {
        encoder: encoder.config().clone(),
        format: Mp4TrackFormat::Audio { channels: 2 },
    };
    (config, packets, drain.gapless)
}

/// A WebM of `video`, and `audio` beside it when given, interleaved in
/// presentation order as a WebM Cluster holds them. The muxer cues every key
/// frame.
fn webm(
    video: (Mp4TrackConfig, Vec<EncodedSample>),
    audio: Option<(Mp4TrackConfig, Vec<EncodedSample>, AudioGapless)>,
) -> Vec<u8> {
    let mut configs = vec![video.0];
    let mut ordered: Vec<(f64, usize, EncodedSample)> = video
        .1
        .into_iter()
        .map(|sample| (sample.pts as f64 / RATE as f64, 0, sample))
        .collect();
    let mut gapless = None;
    if let Some((config, packets, trim)) = audio {
        configs.push(config);
        ordered.extend(
            packets
                .into_iter()
                .map(|packet| (packet.pts as f64 / 48_000.0, 1, packet)),
        );
        gapless = Some(trim);
    }
    ordered.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut remaining = ordered.iter().filter(|(_, track, _)| *track == 1).count();
    let mut muxer = block_on(WebmMuxer::new(MemorySink::new(), configs, 1_000_000)).unwrap();
    for (_, track, sample) in ordered {
        block_on(muxer.write_sample(track, sample)).unwrap();
        if track == 1 {
            remaining -= 1;
            // The end trim is written with the track's last block.
            if remaining == 0 {
                muxer.set_audio_gapless(1, gapless.unwrap()).unwrap();
            }
        }
    }
    block_on(muxer.finish()).unwrap().into_inner()
}

/// `bytes` with its `Cues` hidden, renamed to `Tags` where the `SeekHead`
/// says they are, so it opens by scanning every block as it did before issue
/// #692.
fn without_cues(mut bytes: Vec<u8>) -> Vec<u8> {
    const CUES: [u8; 4] = [0x1C, 0x53, 0xBB, 0x6B];
    const TAGS: [u8; 4] = [0x12, 0x54, 0xC3, 0x67];
    let at = bytes.windows(4).rposition(|window| window == CUES).unwrap();
    bytes[at..at + 4].copy_from_slice(&TAGS);
    bytes
}

/// A source whose every read suspends once before completing, the way a
/// network read does, and which counts the reads.
#[derive(Clone)]
struct SuspendingSource {
    inner: Rc<MemorySource>,
    reads: Rc<Cell<usize>>,
}

impl SuspendingSource {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            inner: Rc::new(MemorySource::new(bytes)),
            reads: Rc::new(Cell::new(0)),
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

/// Each buffer an output was asked to play: its samples and where they go.
type Scheduled = Vec<(SampleRange, Vec<f32>)>;

/// A clock the test moves by hand, and every buffer scheduled against it.
#[derive(Clone, Default)]
struct Output {
    clock: Arc<Mutex<u64>>,
    scheduled: Arc<Mutex<Scheduled>>,
}

impl AudioOutputBackend for Output {
    fn clock_samples(&self) -> u64 {
        *self.clock.lock().unwrap()
    }
    fn start(&mut self, _: u64) -> zvidlib::Result<()> {
        Ok(())
    }
    fn schedule(&mut self, buffer: AudioBuffer, _: u64) -> zvidlib::Result<()> {
        self.scheduled
            .lock()
            .unwrap()
            .push((buffer.range, buffer.samples));
        Ok(())
    }
    fn cancel_queued(&mut self, _: u64) -> zvidlib::Result<()> {
        Ok(())
    }
    fn stop(&mut self) -> zvidlib::Result<()> {
        Ok(())
    }
}

fn opener(output: &Output) -> AudioOutputOpener {
    let output = output.clone();
    Box::new(move |_, _| {
        Ok(Box::new(NativeAudioOutput(output.clone())) as Box<dyn PlaybackAudioOutput>)
    })
}

/// Something a playback did that two playbacks of one file must agree on.
#[derive(Debug, PartialEq)]
enum Event {
    Presented {
        time: Duration,
        picture: Option<Vec<u8>>,
        finished: bool,
    },
    Paused(Vec<u8>),
}

/// Plays what `source` reads through a fixed script of clock steps, seeks and
/// pauses, and returns what it presented, the audio it scheduled, and its
/// duration once it played to the end.
fn play(source: SuspendingSource) -> (Vec<Event>, Scheduled, Duration) {
    let output = Output::default();
    let mut player = OnDemandPlayer::with_output(
        source.clone(),
        OnDemandOptions {
            video_budget_bytes: 64 * 1024,
            audio_budget_bytes: 16 * 1024,
            ..OnDemandOptions::default()
        },
        opener(&output),
    )
    .unwrap();
    let mut events = Vec::new();
    let present = |player: &mut OnDemandPlayer<SuspendingSource>, events: &mut Vec<Event>| {
        let presentation = player.present().unwrap();
        events.push(Event::Presented {
            time: presentation.time,
            picture: presentation.frame.map(|frame| frame.planes[0].data.clone()),
            finished: presentation.finished,
        });
        presentation.finished
    };
    let step = |output: &Output, samples: u64| *output.clock.lock().unwrap() += samples;

    events.push(Event::Paused(
        player.current_frame().unwrap().planes[0].data.clone(),
    ));
    player.play().unwrap();
    // Across the first span boundary, a little more than a frame at a time.
    for _ in 0..40 {
        step(&output, 1_700);
        present(&mut player, &mut events);
    }
    // Into the middle of a span nothing has reached yet, and on through the
    // next boundary.
    player.seek(Duration::from_millis(2_500)).unwrap();
    for _ in 0..25 {
        step(&output, 1_600);
        present(&mut player, &mut events);
    }
    // Paused, back onto a key frame, then a frame before it.
    player.pause().unwrap();
    player.seek(Duration::from_secs(1)).unwrap();
    events.push(Event::Paused(
        player.current_frame().unwrap().planes[0].data.clone(),
    ));
    player.seek(Duration::from_millis(990)).unwrap();
    events.push(Event::Paused(
        player.current_frame().unwrap().planes[0].data.clone(),
    ));
    player.play().unwrap();
    for _ in 0..10 {
        step(&output, 1_600);
        present(&mut player, &mut events);
    }
    // Near the end, and on until it finishes.
    player.seek(Duration::from_millis(3_700)).unwrap();
    let mut finished = false;
    for _ in 0..40 {
        step(&output, 1_600);
        if present(&mut player, &mut events) {
            finished = true;
            break;
        }
    }
    assert!(finished, "playback never reached the end");
    let duration = player.duration();
    let scheduled = std::mem::take(&mut *output.scheduled.lock().unwrap());
    (events, scheduled, duration)
}

/// Issue #692: a cued WebM plays and seeks, over a source whose reads
/// suspend, to the same frames and audio, scheduled at the same samples, as
/// the same file opened by scanning every block, and finds the same end.
#[test]
fn plays_and_seeks_a_cued_webm_as_its_whole_index_does() {
    let bytes = webm(video(4), Some(audio(4)));
    let cued = SuspendingSource::new(bytes.clone());
    let scanned = SuspendingSource::new(without_cues(bytes));
    let (cued_events, cued_audio, cued_duration) = play(cued);
    let (events, audio, duration) = play(scanned);
    assert_eq!(cued_events.len(), events.len());
    for (index, (cued, whole)) in cued_events.iter().zip(&events).enumerate() {
        assert_eq!(cued, whole, "event {index} differs");
    }
    let pictures = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                Event::Presented {
                    picture: Some(_),
                    ..
                }
            )
        })
        .count();
    assert!(pictures > 60, "only {pictures} pictures were presented");
    assert_eq!(cued_audio.len(), audio.len());
    for (index, (cued, whole)) in cued_audio.iter().zip(&audio).enumerate() {
        assert_eq!(cued.0, whole.0, "scheduled range {index} differs");
        assert!(cued.1 == whole.1, "scheduled audio {index} differs");
    }
    assert_eq!(cued_duration, duration);
}

/// A video-only cued WebM shows, wherever a seek lands, the frame the whole
/// file's index shows there, and finds its end where that index puts it. With
/// no audio track it plays on the system's clock, so the seeks are made while
/// paused, where nothing depends on how fast the test runs.
#[test]
fn seeks_a_cued_webm_with_no_audio_as_its_whole_index_does() {
    let bytes = webm(video(4), None);
    let seek_through = |bytes: Vec<u8>| {
        let output = Output::default();
        let mut player = OnDemandPlayer::with_output(
            SuspendingSource::new(bytes),
            OnDemandOptions::default(),
            opener(&output),
        )
        .unwrap();
        let mut pictures = Vec::new();
        for millis in [0, 500, 1_000, 999, 2_970, 3_400, 1_500, 3_990] {
            player.seek(Duration::from_millis(millis)).unwrap();
            pictures.push(player.current_frame().unwrap().planes[0].data.clone());
        }
        (pictures, player.duration())
    };
    let (cued, cued_duration) = seek_through(bytes.clone());
    let (whole, duration) = seek_through(without_cues(bytes));
    assert!(cued == whole, "a seek landed on a different frame");
    assert_eq!(cued_duration, duration);
    assert_eq!(duration, Duration::from_secs(4));
}

// The VP8 frames and Opus packets of one second, repeated `seconds` times
/// one after another: a long file made without encoding all of it, whose
/// packets need not decode for the file to open.
fn repeated(seconds: u64) -> Vec<u8> {
    let (video_config, frames) = video(1);
    let (audio_config, packets, gapless) = audio(1);
    let frame_count = frames.len() as i64;
    let mut long_video = Vec::new();
    for second in 0..seconds as i64 {
        long_video.extend(frames.iter().cloned().map(|mut frame| {
            frame.pts += second * frame_count;
            frame.dts += second * frame_count;
            frame
        }));
    }
    let per_second: Vec<_> = packets
        .iter()
        .filter(|packet| packet.pts < 48_000)
        .cloned()
        .collect();
    let mut long_audio = Vec::new();
    for second in 0..seconds as i64 {
        long_audio.extend(per_second.iter().cloned().map(|mut packet| {
            packet.pts += second * 48_000;
            packet.dts += second * 48_000;
            packet
        }));
    }
    webm(
        (video_config, long_video),
        Some((audio_config, long_audio, gapless)),
    )
}

/// Issue #692: opening a cued WebM reads its header elements, its `Cues` and
/// its first span's block headers: the same few requests whether it lasts
/// five minutes or fifteen, and no more for a few seconds. A `Cues` element
/// grows with the file, but the page cache fetches the part of it past its
/// first page in one request however long it is. Scanning every block, as
/// opening a WebM without `Cues` does, takes requests in proportion to its
/// length.
#[test]
fn opening_a_long_cued_webm_takes_as_few_reads_as_a_short_one() {
    let opening_reads = |bytes: Vec<u8>| {
        let source = SuspendingSource::new(bytes);
        let output = Output::default();
        let player = OnDemandPlayer::with_output(
            source.clone(),
            OnDemandOptions::default(),
            opener(&output),
        )
        .unwrap();
        assert_eq!(player.frame_count(), None);
        source.reads.get()
    };
    let short = opening_reads(repeated(3));
    let long = opening_reads(repeated(300));
    let longer = opening_reads(repeated(900));
    assert_eq!(long, longer, "opening read more of the longer file");
    assert!(short <= long);
    assert!(long < 10, "opening took {long} reads");

    let source = SuspendingSource::new(without_cues(repeated(300)));
    let output = Output::default();
    let player =
        OnDemandPlayer::with_output(source.clone(), OnDemandOptions::default(), opener(&output))
            .unwrap();
    assert_eq!(player.frame_count(), Some(300 * RATE));
    assert!(
        source.reads.get() > 10 * long,
        "scanning every block took only {} reads",
        source.reads.get()
    );
}
