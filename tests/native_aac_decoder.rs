//! `NativeAacDecoder` against the platform AAC decoders (#561).
//!
//! On macOS and Windows the backend is the operating system's own decoder, so
//! these hold it to the sample-exact contract `AacSampleReader` is built on:
//! the same presentation length and boundaries the reader produced when the
//! crate carried Symphonia's AAC decoder, and PCM close to that decoder's
//! output, recorded in `tests/fixtures/codec/aac_reference.bin`. Decoders are
//! not bit-exact with each other, so the PCM is compared by signal-to-noise
//! ratio, and each window must match the reference clearly better than it does
//! one sample early or late. Elsewhere there is no platform decoder and the
//! backend must say so.

#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

#[cfg(not(any(target_os = "macos", windows)))]
use zvidlib::ErrorKind;
use zvidlib::io::MemorySource;
use zvidlib::{
    AacSampleReader, Limits, Mp4Demuxer, Mp4DemuxerOptions, NativeAacDecoder, TrackKind,
};
#[cfg(any(target_os = "macos", windows))]
use zvidlib::{CancellationToken, SampleRange};

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
#[cfg(any(target_os = "macos", windows))]
const WINDOW: u64 = 1024;

/// The signal-to-noise ratio, in dB, a platform decoder's PCM must reach
/// against the Symphonia reference. Decoders are not bit-exact with each other:
/// AudioToolbox and Media Foundation both agree with Symphonia to 48 dB or
/// better on the mono fixture but only to about 15 dB on the bundled stereo
/// track's first window, which is nearly silent, so this is set low enough for
/// that window and [`MIN_ALIGNMENT_MARGIN_DB`] is what pins the alignment.
#[cfg(any(target_os = "macos", windows))]
const MIN_SNR_DB: f32 = 12.0;

/// How much closer to the reference a window must be than the same window one
/// sample early or late. Measured at 6 dB or more on every window with both
/// platform decoders.
#[cfg(any(target_os = "macos", windows))]
const MIN_ALIGNMENT_MARGIN_DB: f32 = 5.0;

/// One fixture's presentation length as `AacSampleReader` reported it over
/// Symphonia's decoder, which the platform decoders must reproduce exactly.
#[cfg_attr(not(any(target_os = "macos", windows)), allow(dead_code))]
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
        presentation_length: 1_536_000,
        channels: 2,
    },
    // Carries a real `elst` whose media time is the encoder priming.
    Fixture {
        name: "aac_lc_mono_48k.m4a",
        bytes: FIXTURE_MONO,
        presentation_length: 98_304,
        channels: 1,
    },
];

/// Where each fixture's reference windows start: the first frame, a point off
/// every access-unit boundary partway through, and the last full window.
#[cfg(any(target_os = "macos", windows))]
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

/// The reference windows for every fixture, in [`FIXTURES`] order and each
/// fixture's windows in [`window_starts`] order: interleaved little-endian
/// `f32`, decoded by Symphonia 0.5.5 through `AacSampleReader` with a preroll
/// of two access units before `symphonia-codec-aac` was dropped.
#[cfg(any(target_os = "macos", windows))]
fn reference_windows() -> Vec<Vec<f32>> {
    let bytes = include_bytes!("fixtures/codec/aac_reference.bin");
    let mut samples = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()));
    let windows = FIXTURES
        .iter()
        .flat_map(|fixture| {
            let len = (WINDOW * u64::from(fixture.channels)) as usize;
            (0..3)
                .map(|_| samples.by_ref().take(len).collect())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(samples.next(), None, "the reference holds only the windows");
    windows
}

/// `reference`'s energy over that of its difference from `decoded`, in dB.
#[cfg(any(target_os = "macos", windows))]
fn snr_db(decoded: &[f32], reference: &[f32]) -> f32 {
    let signal: f32 = reference.iter().map(|sample| sample * sample).sum();
    let noise: f32 = decoded
        .iter()
        .zip(reference)
        .map(|(decoded, reference)| (decoded - reference) * (decoded - reference))
        .sum();
    10.0 * (signal / noise.max(f32::MIN_POSITIVE)).log10()
}

/// Holds `decoded` to [`MIN_SNR_DB`] against `reference`, and to agreeing with
/// it [`MIN_ALIGNMENT_MARGIN_DB`] better than either neighbouring alignment
/// does, which is what pins the window boundaries to the sample.
#[cfg(any(target_os = "macos", windows))]
fn assert_matches_reference(fixture: &Fixture, start: u64, decoded: &[f32], reference: &[f32]) {
    assert_eq!(
        decoded.len(),
        reference.len(),
        "{} at {start}",
        fixture.name
    );
    let step = usize::from(fixture.channels);
    let aligned = snr_db(decoded, reference);
    let early = snr_db(&decoded[step..], &reference[..reference.len() - step]);
    let late = snr_db(&decoded[..decoded.len() - step], &reference[step..]);
    println!(
        "{} window at {start}: {aligned:.1} dB aligned, {early:.1} dB a sample early, \
         {late:.1} dB a sample late",
        fixture.name
    );
    assert!(
        aligned >= MIN_SNR_DB && aligned - early.max(late) >= MIN_ALIGNMENT_MARGIN_DB,
        "{} window at {start} is {aligned:.1} dB from the Symphonia reference \
         ({early:.1} dB a sample early, {late:.1} dB a sample late)",
        fixture.name
    );
}

/// The issue's acceptance check: the same presentation length and window
/// boundaries as Symphonia, and the same PCM to within [`MIN_SNR_DB`]. Every window
/// is a random access, so each read resets the decoder and decodes the preroll
/// again; reading them forwards and then backwards holds `reset` to returning
/// the decoder to a clean state whichever way the reader seeks.
#[cfg(any(target_os = "macos", windows))]
#[test]
fn platform_decoder_matches_the_symphonia_reference_through_the_exact_range_reader() {
    let reference = reference_windows();
    let cancellation = CancellationToken::new();
    for (fixture_index, fixture) in FIXTURES.iter().enumerate() {
        let mut reader = open_reader(fixture.bytes, 2).unwrap();
        assert_eq!(
            reader.presentation_length(),
            fixture.presentation_length,
            "{}",
            fixture.name
        );
        let starts = window_starts(fixture.presentation_length);
        let order = [0, 1, 2, 2, 1, 0];
        for window in order {
            let start = starts[window];
            let buffer = reader
                .get_range(
                    SampleRange::new(start, start + WINDOW).unwrap(),
                    &cancellation,
                )
                .unwrap();
            assert_eq!(
                buffer.range,
                SampleRange::new(start, start + WINDOW).unwrap()
            );
            assert_eq!(buffer.channels, fixture.channels);
            assert_matches_reference(
                fixture,
                start,
                &buffer.samples,
                &reference[fixture_index * 3 + window],
            );
        }
    }
}

/// The whole presentation in one read, so every access unit in both fixtures
/// passes through the platform decoder and comes back exactly as long as its
/// indexed interval.
#[cfg(any(target_os = "macos", windows))]
#[test]
fn platform_decoder_reads_each_fixture_end_to_end() {
    for fixture in &FIXTURES {
        let mut reader = open_reader(fixture.bytes, 2).unwrap();
        let length = reader.presentation_length();
        let buffer = reader
            .get_range(
                SampleRange::new(0, length).unwrap(),
                &CancellationToken::new(),
            )
            .unwrap();
        assert_eq!(
            buffer.samples.len() as u64,
            length * u64::from(fixture.channels),
            "{}",
            fixture.name
        );
        let peak = buffer
            .samples
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
        println!("{} peaks at {peak}", fixture.name);
        assert!(peak > 0.01, "{} decoded to silence", fixture.name);
    }
}

/// With no platform AAC decoder, `NativeAacDecoder` says so rather than
/// falling back to a software decoder this crate does not carry.
#[cfg(not(any(target_os = "macos", windows)))]
#[test]
fn native_aac_decoding_is_unsupported_without_a_platform_decoder() {
    for fixture in &FIXTURES {
        let error = match open_reader(fixture.bytes, 2) {
            Ok(_) => panic!("{} decoded without a platform AAC decoder", fixture.name),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{}", fixture.name);
        assert!(
            error.message().contains("AudioToolbox")
                && error.message().contains("Media Foundation"),
            "the error names the platform decoders: {}",
            error.message()
        );
    }
}
