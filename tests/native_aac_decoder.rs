//! `NativeAacDecoder` against the platform AAC decoders (#561).
//!
//! On macOS and Windows the backend is the operating system's own decoder, so
//! these hold it to the sample-exact contract `AacSampleReader` is built on:
//! the same presentation length and boundaries the reader produced when the
//! crate carried Symphonia's AAC decoder, and PCM within a small tolerance of
//! that decoder's output, recorded in `tests/fixtures/codec/aac_reference.bin`.
//! Decoders are not bit-exact with each other, but a decoder that added or
//! dropped even one sample of delay would be off by far more than the
//! tolerance. Elsewhere there is no platform decoder and the backend must say
//! so.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

use zvidlib::io::MemorySource;
use zvidlib::{
    AacSampleReader, CancellationToken, ErrorKind, Limits, Mp4Demuxer, Mp4DemuxerOptions,
    NativeAacDecoder, SampleRange, TrackKind,
};

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = Box::pin(future);
    loop {
        match Pin::new(&mut future).poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

const BUNDLED_STEREO: &[u8] = include_bytes!("../examples/media/BigBuckBunny.mp4");
const FIXTURE_MONO: &[u8] = include_bytes!("fixtures/codec/aac_lc_mono_48k.m4a");

/// Frames in each reference window.
const WINDOW: u64 = 1024;

/// Largest difference allowed between a platform decoder's sample and the
/// Symphonia reference. AAC-LC decoders differ in rounding, not in content:
/// a one-sample misalignment of these signals is off by an order of magnitude
/// more than this.
#[cfg(any(target_os = "macos", windows))]
const TOLERANCE: f32 = 1.0e-3;

/// The presentation length each fixture's reader reported with Symphonia, and
/// where its reference windows start: the first frame, a point off every
/// access-unit boundary partway through, and the last full window.
struct Fixture {
    name: &'static str,
    bytes: &'static [u8],
    presentation_length: u64,
    channels: u16,
}

const FIXTURES: [Fixture; 2] = [
    Fixture {
        name: "BigBuckBunny.mp4",
        bytes: BUNDLED_STEREO,
        presentation_length: 0,
        channels: 2,
    },
    Fixture {
        name: "aac_lc_mono_48k.m4a",
        bytes: FIXTURE_MONO,
        presentation_length: 0,
        channels: 1,
    },
];

fn window_starts(length: u64) -> [u64; 3] {
    [0, length / 2 + 333, length - WINDOW]
}

fn open_reader(
    bytes: &'static [u8],
    preroll: usize,
) -> zvidlib::Result<AacSampleReader<NativeAacDecoder>> {
    let limits = Limits::default();
    let source = MemorySource::new(bytes.to_vec());
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))?;
    let track = movie
        .tracks
        .iter()
        .find(|track| track.kind == TrackKind::Audio)
        .expect("the fixture has an audio track");
    let config = track.aac_config()?;
    let timing = track.audio_timing(movie.movie_timescale)?;
    let packets = block_on(track.to_encoded_audio_samples(&source, &limits))?;
    let decoder = NativeAacDecoder::new(&config, limits)?;
    AacSampleReader::new(
        decoder,
        packets,
        config.sample_rate,
        config.channels,
        timing,
        preroll,
        limits,
    )
}

#[test]
#[ignore = "regenerates the reference with the decoder under test"]
fn dump_reference() {
    let mut out = Vec::new();
    for fixture in &FIXTURES {
        let mut reader = open_reader(fixture.bytes, 2).unwrap();
        let length = reader.presentation_length();
        println!("{} presentation length {length}", fixture.name);
        for start in window_starts(length) {
            let buffer = reader
                .get_range(
                    SampleRange::new(start, start + WINDOW).unwrap(),
                    &CancellationToken::new(),
                )
                .unwrap();
            for sample in buffer.samples {
                out.extend_from_slice(&sample.to_le_bytes());
            }
        }
    }
    std::fs::write("tests/fixtures/codec/aac_reference.bin", out).unwrap();
}
