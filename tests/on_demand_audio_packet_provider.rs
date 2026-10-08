//! `AudioSampleReader` built from an on-demand [`TrackAudioPacketProvider`]
//! decodes the same PCM, through the native AAC decoder, that it does from an
//! eagerly read `Vec<EncodedAudioSample>` (issue #671).

#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll, Waker};

use zvidlib::io::{ByteSource, IoFuture, MemorySource};
use zvidlib::{
    AudioSampleReader, CancellationToken, Limits, Mp4Demuxer, Mp4DemuxerOptions, NativeAacDecoder,
    SampleRange, Track, TrackAudioPacketProvider, TrackKind,
};

fn block_on<F: Future>(future: F) -> F::Output {
    let mut boxed = Box::pin(future);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    match boxed.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("unexpected pending future"),
    }
}

const FIXTURES: [&[u8]; 2] = [
    include_bytes!("../examples/media/BigBuckBunny.mp4"),
    include_bytes!("fixtures/codec/aac_lc_mono_48k.m4a"),
];

/// Counts every byte read from it, so a test can tell whether building a
/// reader read any packet data.
struct CountingSource {
    inner: MemorySource,
    bytes_read: Arc<AtomicU64>,
}

impl ByteSource for CountingSource {
    fn len(&self) -> Option<u64> {
        self.inner.len()
    }

    fn read_at<'a>(&'a self, offset: u64, destination: &'a mut [u8]) -> IoFuture<'a, usize> {
        Box::pin(async move {
            let read = self.inner.read_at(offset, destination).await?;
            self.bytes_read.fetch_add(read as u64, Ordering::Relaxed);
            Ok(read)
        })
    }
}

fn audio_track(bytes: &[u8]) -> (Track, u32) {
    let source = MemorySource::new(bytes.to_vec());
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
    let track = movie
        .tracks
        .into_iter()
        .find(|track| track.kind == TrackKind::Audio)
        .expect("the fixture has an audio track");
    (track, movie.movie_timescale)
}

/// Builds a reader over `provider` with `track`'s configuration and timing.
fn reader_from_provider(
    track: &Track,
    movie_timescale: u32,
    provider: Box<dyn zvidlib::AudioPacketProvider>,
) -> AudioSampleReader<NativeAacDecoder> {
    let limits = Limits::default();
    let config = track.aac_config().unwrap();
    AudioSampleReader::from_provider(
        NativeAacDecoder::new(&config, limits).unwrap(),
        provider,
        config.sample_rate,
        config.channels,
        track.audio_timing(movie_timescale).unwrap(),
        2,
        limits,
    )
    .unwrap()
}

/// Constructing the reader reads no packet data, and the PCM it then decodes
/// over the whole presentation is identical to the eager `Vec`-backed
/// reader's.
#[test]
fn reader_over_the_on_demand_provider_matches_the_eager_reader() {
    let cancellation = CancellationToken::new();
    for bytes in FIXTURES {
        let (track, movie_timescale) = audio_track(bytes);

        let source = MemorySource::new(bytes.to_vec());
        let packets =
            block_on(track.to_encoded_audio_samples(&source, &Limits::default())).unwrap();
        let mut eager = reader_from_provider(&track, movie_timescale, Box::new(packets));

        let bytes_read = Arc::new(AtomicU64::new(0));
        let counting = CountingSource {
            inner: MemorySource::new(bytes.to_vec()),
            bytes_read: Arc::clone(&bytes_read),
        };
        let provider = TrackAudioPacketProvider::new(track.clone(), counting).unwrap();
        let mut on_demand = reader_from_provider(&track, movie_timescale, Box::new(provider));
        assert_eq!(
            bytes_read.load(Ordering::Relaxed),
            0,
            "building the reader read packet data"
        );

        let length = eager.presentation_length();
        assert_eq!(on_demand.presentation_length(), length);
        let whole = SampleRange::new(0, length).unwrap();
        let expected = eager.get_range(whole, &cancellation).unwrap();
        let actual = on_demand.get_range(whole, &cancellation).unwrap();
        assert_eq!(actual.range, expected.range);
        assert_eq!(actual.channels, expected.channels);
        assert_eq!(actual.samples, expected.samples);
        assert!(bytes_read.load(Ordering::Relaxed) > 0);
    }
}
