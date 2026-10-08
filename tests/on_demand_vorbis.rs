//! A WebM's Vorbis track plays on demand (issue #686): its packets' decoded
//! intervals come from the track's index, so an `AudioSampleReader` over an
//! on-demand provider reads no packet when it is built, and it decodes the
//! samples the eager `Vec`-backed reader does, after backward seeks too, both
//! on its own and as the audio of an on-demand `PlaybackController`.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use zvidlib::io::{ByteSource, IoFuture, MemorySink, MemorySource};
use zvidlib::mp4::{Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::{
    AudioBuffer, AudioGapless, AudioOutputBackend, AudioSampleReader, CancellationToken, Codec,
    ColorRange, EncodedSample, EncoderConfig, ErrorKind, FrameIndex, IndexedPresentationTimeline,
    Limits, Mp4AudioPacketProvider, Mp4SampleLoader, Mp4Track, NativeVorbisDecoder,
    OnDemandAudioSource, PixelFormat, Plane, PlaybackAudioSource, PlaybackController,
    PlaybackOptions, PlaybackVideoSource, PrefetchAudioSource, PrefetchVideoSource, Result,
    SampleDependency, SampleRange, TrackKind, VORBIS_PREROLL_PACKETS, VideoDimensions, VideoFrame,
    VorbisConfig, WebAudioOutput, WebmDemuxer, WebmDemuxerOptions,
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

/// libvorbis's own streams: stereo music, a mono one whose transients code
/// short blocks among the long ones, and six channels at a low rate.
const FIXTURES: [(&str, &[u8]); 3] = [
    (
        "vorbis_stereo_44k",
        include_bytes!("../crates/zvidlib-vorbis-decoder/tests/fixtures/vorbis_stereo_44k.ogg"),
    ),
    (
        "vorbis_transient_mono",
        include_bytes!("../crates/zvidlib-vorbis-decoder/tests/fixtures/vorbis_transient_mono.ogg"),
    ),
    (
        "vorbis_6ch_16k",
        include_bytes!("../crates/zvidlib-vorbis-decoder/tests/fixtures/vorbis_6ch_16k.ogg"),
    ),
];

/// The packets of a single-stream Ogg file (RFC 3533), and the granule
/// position of its last page.
fn ogg_packets(bytes: &[u8]) -> (Vec<Vec<u8>>, u64) {
    let mut packets = Vec::new();
    let mut partial = Vec::new();
    let mut granule = 0;
    let mut at = 0;
    while at < bytes.len() {
        assert_eq!(&bytes[at..at + 4], b"OggS", "an Ogg page at {at}");
        granule = u64::from_le_bytes(bytes[at + 6..at + 14].try_into().unwrap());
        let segments = usize::from(bytes[at + 26]);
        let lacing = &bytes[at + 27..at + 27 + segments];
        let mut body = at + 27 + segments;
        for &length in lacing {
            partial.extend_from_slice(&bytes[body..body + usize::from(length)]);
            body += usize::from(length);
            if length < 255 {
                packets.push(std::mem::take(&mut partial));
            }
        }
        at = body;
    }
    (packets, granule)
}

/// An Ogg Vorbis stream remuxed into a WebM, its end trimmed to its last
/// granule position with the last block's `DiscardPadding`.
fn webm_from_ogg(ogg: &[u8]) -> Vec<u8> {
    let (mut packets, granule) = ogg_packets(ogg);
    let audio = packets.split_off(3);
    let [identification, comment, setup]: [Vec<u8>; 3] = packets.try_into().unwrap();
    let config = VorbisConfig::from_headers(identification, comment, setup).unwrap();
    let samples = config.encoded_samples(audio).unwrap();
    let decoded = samples.last().unwrap().decoded_range.end;
    let mut muxer = block_on(zvidlib::WebmMuxer::new(
        MemorySink::new(),
        vec![Mp4TrackConfig {
            encoder: EncoderConfig {
                codec: Codec::Vorbis,
                timescale: config.sample_rate,
                decoder_config: config.to_codec_private(),
            },
            format: Mp4TrackFormat::Audio {
                channels: u16::from(config.channels),
            },
        }],
        1_000_000,
    ))
    .unwrap();
    for sample in samples {
        let start = i64::try_from(sample.decoded_range.start).unwrap();
        block_on(muxer.write_sample(
            0,
            EncodedSample {
                data: sample.data,
                dts: start,
                pts: start,
                duration: u32::try_from(sample.decoded_range.len()).unwrap(),
                is_sync: true,
                dependency: SampleDependency::INDEPENDENT,
            },
        ))
        .unwrap();
    }
    muxer
        .set_audio_gapless(
            0,
            AudioGapless {
                priming: 0,
                padding: u32::try_from(decoded - granule).unwrap(),
            },
        )
        .unwrap();
    block_on(muxer.finish()).unwrap().into_inner()
}

/// One fixture remuxed into a WebM, indexed, with its expected length.
struct Fixture {
    name: &'static str,
    bytes: Vec<u8>,
    demuxer: WebmDemuxer,
    track: Mp4Track,
    /// The stream's last granule position: its length once trimmed.
    granule: u64,
}

fn fixtures() -> Vec<Fixture> {
    FIXTURES
        .iter()
        .map(|&(name, ogg)| {
            let bytes = webm_from_ogg(ogg);
            let source = MemorySource::new(bytes.clone());
            let demuxer =
                block_on(WebmDemuxer::open(&source, WebmDemuxerOptions::default())).unwrap();
            let track = demuxer
                .tracks
                .iter()
                .find(|track| track.kind == TrackKind::Audio)
                .cloned()
                .unwrap();
            assert_eq!(track.codec, Codec::Vorbis);
            Fixture {
                name,
                bytes,
                demuxer,
                track,
                granule: ogg_packets(ogg).1,
            }
        })
        .collect()
}

impl Fixture {
    fn eager_packets(&self) -> Vec<zvidlib::EncodedAudioSample> {
        let source = MemorySource::new(self.bytes.clone());
        block_on(
            self.track
                .to_encoded_audio_samples(&source, &Limits::default()),
        )
        .unwrap()
    }

    fn reader(
        &self,
        packets: Box<dyn zvidlib::AudioPacketProvider>,
    ) -> AudioSampleReader<NativeVorbisDecoder> {
        let config = self.track.vorbis_config().unwrap();
        AudioSampleReader::from_provider(
            NativeVorbisDecoder::new(&config, Limits::default()).unwrap(),
            packets,
            config.sample_rate,
            u16::from(config.channels),
            self.demuxer.audio_timing(self.track.id).unwrap(),
            VORBIS_PREROLL_PACKETS,
            Limits::default(),
        )
        .unwrap()
    }

    fn eager_reader(&self) -> AudioSampleReader<NativeVorbisDecoder> {
        self.reader(Box::new(self.eager_packets()))
    }

    /// Ranges across the presentation, out of order: from the start, near the
    /// end, back to the start, a single sample, then the middle.
    fn ranges(&self, length: u64) -> Vec<SampleRange> {
        let range = |start: u64, end: u64| SampleRange::new(start, end.min(length)).unwrap();
        let window = length / 8;
        vec![
            range(0, window),
            range(window, 2 * window),
            range(length - window, length),
            range(window / 2, window / 2 + window),
            range(length / 3, length / 3 + 1),
            range(length / 2 - window, length / 2 + window),
        ]
    }
}

/// Counts every byte read from it.
struct CountingSource {
    inner: MemorySource,
    bytes_read: Arc<Mutex<u64>>,
}

impl ByteSource for CountingSource {
    fn len(&self) -> Option<u64> {
        self.inner.len()
    }

    fn read_at<'a>(&'a self, offset: u64, destination: &'a mut [u8]) -> IoFuture<'a, usize> {
        Box::pin(async move {
            let read = self.inner.read_at(offset, destination).await?;
            *self.bytes_read.lock().unwrap() += read as u64;
            Ok(read)
        })
    }
}

/// A synchronous on-demand provider indexes the track without reading a
/// packet, reads only the packets a request decodes, and decodes exactly what
/// the eager reader does, trimmed to the stream's granule position.
#[test]
fn a_synchronous_provider_reads_no_packet_until_decoding_and_matches_the_eager_path() {
    for fixture in fixtures() {
        let name = fixture.name;
        let eager_packets = fixture.eager_packets();
        let bytes_read = Arc::new(Mutex::new(0));
        let provider = Mp4AudioPacketProvider::new(
            fixture.track.clone(),
            CountingSource {
                inner: MemorySource::new(fixture.bytes.clone()),
                bytes_read: Arc::clone(&bytes_read),
            },
        )
        .unwrap();
        use zvidlib::AudioPacketProvider;
        assert_eq!(provider.len(), eager_packets.len());
        for (index, packet) in eager_packets.iter().enumerate() {
            assert_eq!(
                provider.decoded_range(index),
                packet.decoded_range,
                "{name}: packet {index}"
            );
        }
        let mut on_demand = fixture.reader(Box::new(provider));
        assert_eq!(
            *bytes_read.lock().unwrap(),
            0,
            "{name}: building the reader read packet data"
        );

        let mut eager = fixture.eager_reader();
        let length = eager.presentation_length();
        assert_eq!(length, fixture.granule, "{name}: the end was not trimmed");
        assert_eq!(on_demand.presentation_length(), length);
        let total: u64 = eager_packets
            .iter()
            .map(|packet| packet.data.len() as u64)
            .sum();
        let cancellation = CancellationToken::new();
        for (index, range) in fixture.ranges(length).into_iter().enumerate() {
            let expected = eager.get_range(range, &cancellation).unwrap();
            let got = on_demand.get_range(range, &cancellation).unwrap();
            assert_eq!(got.range, range);
            assert_eq!(got.samples, expected.samples, "{name}: {range:?}");
            if index == 0 {
                let read = *bytes_read.lock().unwrap();
                assert!(
                    0 < read && read < total,
                    "{name}: the first eighth read {read} of the track's {total} packet bytes"
                );
            }
        }
    }
}

/// A source whose every read suspends once before completing, the way a
/// browser `fetch` does.
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

/// The budget a test's loader may hold: a quarter of the track's packets,
/// and at least four of the largest, so it holds a packet with its preroll.
fn budget(track: &Mp4Track) -> u64 {
    let total: u64 = track
        .samples
        .iter()
        .map(|sample| u64::from(sample.size))
        .sum();
    let largest = track
        .samples
        .iter()
        .map(|sample| u64::from(sample.size))
        .max()
        .unwrap();
    (total / 4).max(4 * largest)
}

/// Over a source that suspends, the loader's provider is built from the
/// index alone, reads report `WouldBlock` until a prefetch loads what they
/// need, the samples are the eager reader's, and the loader never holds more
/// than its budget.
#[test]
fn an_on_demand_audio_source_loads_packets_as_reads_reach_them() {
    for fixture in fixtures() {
        let name = fixture.name;
        let budget = budget(&fixture.track);
        let loader = Mp4SampleLoader::new(
            fixture.track.clone(),
            SuspendingSource::new(fixture.bytes.clone()),
            budget,
        )
        .unwrap();
        let provider = loader.vorbis_packet_provider().unwrap();
        assert_eq!(
            loader.source().reads.get(),
            0,
            "{name}: building the provider read the source"
        );
        let mut audio = OnDemandAudioSource::new(fixture.reader(Box::new(provider)), loader, 8);
        let mut eager = fixture.eager_reader();
        let length = audio.presentation_length();
        assert_eq!(length, fixture.granule);
        let cancellation = CancellationToken::new();
        for range in fixture.ranges(length) {
            let got = loop {
                match audio.read(range, &cancellation) {
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        block_on(audio.prefetch(range)).unwrap();
                    }
                    result => break result.unwrap(),
                }
            };
            let expected = eager.get_range(range, &cancellation).unwrap();
            assert_eq!(got.samples, expected.samples, "{name}: {range:?}");
            assert!(audio.loader().resident_bytes() <= budget);
        }
        assert!(audio.loader().source().reads.get() > 0);
    }
}

/// Stands in for a video track: every frame is the same small gray picture,
/// so the controller's audio is all that is under test.
struct GrayVideo;

impl PlaybackVideoSource for GrayVideo {
    fn get_exact(&mut self, _: FrameIndex, _: &CancellationToken) -> Result<VideoFrame> {
        let limits = Limits::default();
        VideoFrame::new(
            VideoDimensions::new(2, 2, &limits)?,
            PixelFormat::Gray8,
            ColorRange::Full,
            vec![Plane {
                data: vec![128; 4],
                stride: 2,
            }],
            &limits,
        )
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

impl PrefetchVideoSource for GrayVideo {
    fn prefetch(&mut self, _: FrameIndex) -> IoFuture<'_, ()> {
        Box::pin(async { Ok(()) })
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

/// Retries `step` until it stops reporting `WouldBlock`, awaiting the
/// controller's prefetch in between: the browser's render loop.
fn until_loaded<T, V, A, O>(
    playback: &mut PlaybackController<V, A, O>,
    mut step: impl FnMut(&mut PlaybackController<V, A, O>) -> Result<T>,
) -> T
where
    V: PrefetchVideoSource,
    A: PrefetchAudioSource,
    O: zvidlib::PlaybackAudioOutput,
{
    for _ in 0..16 {
        match step(playback) {
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                block_on(playback.prefetch()).unwrap();
            }
            result => return result.unwrap(),
        }
    }
    panic!("playback still had not loaded what it needed after 16 prefetches");
}

/// The on-demand `PlaybackController` plays and seeks, backward too, with a
/// Vorbis track as its audio, over a source whose every read suspends, and
/// what it schedules is exactly what the eager reader decodes.
#[test]
fn playback_plays_and_seeks_with_vorbis_audio_over_a_suspending_source() {
    for fixture in fixtures() {
        let name = fixture.name;
        let loader = Mp4SampleLoader::new(
            fixture.track.clone(),
            SuspendingSource::new(fixture.bytes.clone()),
            budget(&fixture.track),
        )
        .unwrap();
        let reader = fixture.reader(Box::new(loader.vorbis_packet_provider().unwrap()));
        let sample_rate = reader.sample_rate();
        let length = reader.presentation_length();
        let audio = OnDemandAudioSource::new(reader, loader, 8);

        // Twenty frames a second over the whole track.
        let per_frame = u64::from(sample_rate) / 20;
        let frames = length / per_frame;
        let timeline = IndexedPresentationTimeline::new(
            (0..frames)
                .map(|frame| SampleRange::new(frame * per_frame, (frame + 1) * per_frame).unwrap())
                .collect(),
        )
        .unwrap();
        let backend = Backend::default();
        let mut playback = PlaybackController::new_with_indexed_timeline(
            GrayVideo,
            audio,
            WebAudioOutput(backend.clone()),
            timeline.clone(),
            PlaybackOptions::for_sample_rate(sample_rate),
        )
        .unwrap();

        assert_eq!(
            playback.play().unwrap_err().kind(),
            ErrorKind::WouldBlock,
            "{name}: playback read audio that was not loaded"
        );
        until_loaded(&mut playback, |playback| playback.play());
        let anchor = Cell::new((0_u64, 0_u64));
        let present = |playback: &mut PlaybackController<_, _, _>, frame: u64| {
            let (clock, media) = anchor.get();
            let start = timeline.audio_interval_for_frame(FrameIndex(frame)).unwrap();
            *backend.clock.lock().unwrap() = clock + start.start - media;
            let (presentation, _) = until_loaded(playback, |playback| playback.present());
            assert_eq!(presentation.frame, Some(FrameIndex(frame)), "{name}");
        };
        let seek = |playback: &mut PlaybackController<_, _, _>, frame: u64| {
            playback.seek(FrameIndex(frame)).unwrap();
            let start = timeline.audio_interval_for_frame(FrameIndex(frame)).unwrap();
            anchor.set((*backend.clock.lock().unwrap(), start.start));
        };
        for frame in [0, 1, 2, 4] {
            present(&mut playback, frame);
        }
        seek(&mut playback, frames - 4);
        present(&mut playback, frames - 4);
        seek(&mut playback, frames / 3);
        present(&mut playback, frames / 3);
        present(&mut playback, frames / 3 + 1);
        seek(&mut playback, 1);
        present(&mut playback, 1);

        let scheduled = backend.scheduled.lock().unwrap();
        assert!(!scheduled.is_empty(), "{name}: no audio was scheduled");
        let mut eager = fixture.eager_reader();
        let cancellation = CancellationToken::new();
        for buffer in scheduled.iter() {
            let expected = eager.get_range(buffer.range, &cancellation).unwrap();
            assert_eq!(
                buffer.samples, expected.samples,
                "{name}: scheduled audio {:?} differs from the eager reader's",
                buffer.range
            );
        }
    }
}
