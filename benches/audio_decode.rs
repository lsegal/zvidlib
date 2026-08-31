//! Criterion benchmarks for the AAC decode path and the exact-range reader.
//!
//! Two things are measured here, and keeping them apart is the point of the
//! group layout:
//!
//! 1. [`NativeAacDecoder::decode`] over a fixed run of AAC-LC access units, in
//!    both channel layouts the backend accepts. Demuxing happens once in the
//!    shared fixture cache, never inside a timed loop, so an iteration is
//!    Symphonia's decode work and the interleave into `AudioBuffer` and
//!    nothing else. Reported as decoded samples per second and, in the printed
//!    scale lines, as a realtime factor - the number that decides whether
//!    playback can keep up.
//! 2. [`AacSampleReader::get_range`], where the non-trivial work lives. Its
//!    `decoded: BTreeMap` cache makes the same call cost two very different
//!    things depending on whether the requested media range is already
//!    resident, so the cache-hit path, the sequential playback walk, and the
//!    random-access path that forces a decoder reset plus a preroll re-decode
//!    are three separate groups. Averaging them together would hide the seek
//!    cost entirely, which is the one that shows up as an audible stall.
//!
//! # No scalar-versus-SIMD axis
//!
//! Unlike the groups in `benches/codec.rs`, nothing here runs once per
//! instruction set and the group names carry no `simd=` tag. AAC decoding is
//! delegated to the third-party `symphonia-codec-aac` crate, the process-wide
//! override in `zvidlib::simd` does not reach it, and this crate has no audio
//! SIMD kernels of its own - so a scalar arm and a vector arm would be the same
//! code producing two identical numbers. If Symphonia's own performance turns
//! out to bound playback, that is a dependency-level finding for its own
//! ticket rather than an axis to add here.

mod support;

use std::hint::black_box;
use std::time::Instant;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main};
use zvidlib::{
    AacDecoder, AacSampleReader, AudioEdit, AudioTrackTiming, CancellationToken,
    EncodedAudioSample, Limits, NativeAacDecoder, SampleRange,
};

use support::{AudioWork, BundledAacTrack, report_audio_throughput};

/// Access units decoded per iteration of the raw-decode group.
///
/// At 1024 samples per AAC-LC access unit and 48 kHz this is about 2.7 seconds
/// of audio, long enough for a stable measurement and short enough that the
/// group stays part of an ordinary `cargo bench` run. The mono fixture is
/// shorter than this and contributes every packet it has.
const DECODE_PACKETS: usize = 128;

/// Preroll depth the reader is built with, matching the `native_gl` example's
/// playback configuration so the seek group measures the real cost.
const PREROLL_PACKETS: usize = 2;

/// Presentation samples per reader request, a little under one access unit.
const READ_SAMPLES: u64 = 1024;

/// Requests per iteration of the sequential and cached reader groups.
const SEQUENTIAL_READS: u64 = 32;

/// Requests per iteration of the seek group. Each one re-decodes, so this is
/// smaller than the sequential count - but not so small that a single slow
/// iteration dominates the sample, which at eight reads it did.
const SEEK_READS: u64 = 16;

/// Where the sequential walk starts, ten seconds into the bundled sample, so
/// it measures steady-state playback rather than the first-packet case.
const SEQUENTIAL_START: u64 = 480_000;

fn main_criterion() -> Criterion {
    Criterion::default()
}

/// Raw access-unit decode, mono and stereo.
fn aac_decode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("aac_decode");
    group.sample_size(20);
    decode_arm(
        &mut group,
        "access_units_stereo_48k",
        support::bundled_aac_track(),
    );
    decode_arm(
        &mut group,
        "access_units_mono_48k",
        support::aac_mono_track(),
    );
    group.finish();
}

/// One channel layout's raw-decode arm.
fn decode_arm(group: &mut BenchmarkGroup<'_, WallTime>, id: &str, track: &BundledAacTrack) {
    let packets = track.prefix(DECODE_PACKETS);
    let work = AudioWork::new(covered_samples(packets), track.sample_rate, track.channels);
    let cancellation = CancellationToken::new();
    let mut decoder = NativeAacDecoder::new(&track.config, Limits::default())
        .expect("the fixture is AAC-LC mono or stereo, which the native backend accepts");
    // Every iteration starts from the same decoder state, so the measurement
    // does not depend on how many iterations ran before it.
    let mut run = || {
        decoder.reset().expect("resetting an AAC decoder succeeds");
        let mut produced = 0_usize;
        for packet in packets {
            let buffer = decoder
                .decode(black_box(packet), &cancellation)
                .expect("the fixture's access units decode");
            produced += buffer.samples.len();
        }
        produced
    };
    report_realtime(id, work, &mut run);
    report_audio_throughput(group, id, work);
    group.bench_function(id, |bencher| bencher.iter(|| black_box(run())));
}

/// Reader requests that stay inside already-decoded packets, and the forward
/// walk that crosses into new ones.
fn aac_reader_sequential(criterion: &mut Criterion) {
    let track = support::bundled_aac_track();
    let cancellation = CancellationToken::new();
    let mut group = criterion.benchmark_group("aac_reader_sequential");
    group.sample_size(20);

    // The cache-hit path: one range, already resident, read repeatedly. No
    // decode at all, so this is the `decoded` BTreeMap lookup plus the copy
    // out of the resident buffers and nothing else.
    let cached_id = "cached_repeat_stereo_48k";
    let cached_work = AudioWork::new(
        SEQUENTIAL_READS * READ_SAMPLES,
        track.sample_rate,
        track.channels,
    );
    let cached_range = range_at(SEQUENTIAL_START, READ_SAMPLES);
    let mut cached_reader = reader_for(track, track.timing.clone());
    cached_reader
        .get_range(cached_range, &cancellation)
        .expect("priming the cache with the range about to be re-read succeeds");
    let mut cached_run = || {
        for _ in 0..SEQUENTIAL_READS {
            black_box(
                cached_reader
                    .get_range(cached_range, &cancellation)
                    .expect("a resident range reads back"),
            );
        }
    };
    report_realtime(cached_id, cached_work, &mut cached_run);
    report_audio_throughput(&mut group, cached_id, cached_work);
    group.bench_function(cached_id, |bencher| bencher.iter(&mut cached_run));

    // The playback path: consecutive forward ranges. `ensure_decoded` clears
    // the whole cache whenever the requested packets are not all resident, so
    // at this read size a request that advances past the previous one does not
    // extend the cache - it discards it and re-decodes with preroll. That is
    // why this arm measures close to the seek group rather than to the cached
    // one, and it is a property of the reader, not of the fixture.
    let forward_id = "forward_walk_stereo_48k";
    let forward_ranges = (0..SEQUENTIAL_READS)
        .map(|step| range_at(SEQUENTIAL_START + step * READ_SAMPLES, READ_SAMPLES))
        .collect::<Vec<_>>();
    let forward_work = AudioWork::new(
        SEQUENTIAL_READS * READ_SAMPLES,
        track.sample_rate,
        track.channels,
    );
    let mut forward_reader = reader_for(track, track.timing.clone());
    let mut forward_run = || {
        // Rewinding is what makes the walk repeatable; the reset itself is one
        // decoder reset per iteration against 32 requests, not per request.
        forward_reader
            .reset()
            .expect("resetting the reader between iterations succeeds");
        for range in &forward_ranges {
            black_box(
                forward_reader
                    .get_range(*range, &cancellation)
                    .expect("a forward playback range reads"),
            );
        }
    };
    report_realtime(forward_id, forward_work, &mut forward_run);
    report_audio_throughput(&mut group, forward_id, forward_work);
    group.bench_function(forward_id, |bencher| bencher.iter(&mut forward_run));

    group.finish();
}

/// Random-access reads, each of which forces a seek and a preroll re-decode.
fn aac_reader_seek(criterion: &mut Criterion) {
    let track = support::bundled_aac_track();
    let cancellation = CancellationToken::new();
    let mut reader = reader_for(track, track.timing.clone());

    // Deterministic, widely spaced, non-monotonic offsets. Consecutive targets
    // are nowhere near each other, so `ensure_decoded` misses, resets the
    // decoder, and re-decodes `PREROLL_PACKETS` access units ahead of the one
    // actually wanted - the cost this group exists to expose.
    let span = reader
        .presentation_length()
        .saturating_sub(READ_SAMPLES)
        .max(1);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let ranges = (0..SEEK_READS)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            range_at((state >> 11) % span, READ_SAMPLES)
        })
        .collect::<Vec<_>>();

    let id = "random_seek_preroll_stereo_48k";
    let work = AudioWork::new(SEEK_READS * READ_SAMPLES, track.sample_rate, track.channels);
    let mut run = || {
        for range in &ranges {
            black_box(
                reader
                    .get_range(*range, &cancellation)
                    .expect("a random-access range reads"),
            );
        }
    };
    let mut group = criterion.benchmark_group("aac_reader_seek");
    group.sample_size(20);
    report_realtime(id, work, &mut run);
    report_audio_throughput(&mut group, id, work);
    group.bench_function(id, |bencher| bencher.iter(&mut run));
    group.finish();
}

/// Reads that cross edit-list boundaries and gapless priming/padding trims, so
/// the timeline mapping in `AudioTrackTiming` is visible rather than amortized
/// away inside a decode.
fn aac_reader_edits(criterion: &mut Criterion) {
    let cancellation = CancellationToken::new();
    let mut group = criterion.benchmark_group("aac_reader_edits");
    group.sample_size(20);

    // The mono fixture carries a real `elst` whose media time is the decoder
    // priming, so its first and last presentation samples are the trimmed ones.
    let mono = support::aac_mono_track();
    let mono_id = "priming_padding_trim_mono_48k";
    let mut mono_reader = reader_for(mono, mono.timing.clone());
    let mono_length = mono_reader.presentation_length();
    let mono_ranges = vec![
        range_at(0, READ_SAMPLES),
        range_at(mono_length - READ_SAMPLES, READ_SAMPLES),
    ];
    let mono_work = AudioWork::new(
        mono_ranges.len() as u64 * READ_SAMPLES,
        mono.sample_rate,
        mono.channels,
    );
    let mut mono_run = || {
        for range in &mono_ranges {
            black_box(
                mono_reader
                    .get_range(*range, &cancellation)
                    .expect("the trimmed head and tail of the mono fixture read"),
            );
        }
    };
    report_realtime(mono_id, mono_work, &mut mono_run);
    report_audio_throughput(&mut group, mono_id, mono_work);
    group.bench_function(mono_id, |bencher| bencher.iter(&mut mono_run));

    // The bundled stereo sample has no edit list at all, so the multi-edit case
    // is built over its packets: an empty (silent) edit, two media edits that
    // are not adjacent in media time, and a whole-track presentation offset.
    let stereo = support::bundled_aac_track();
    let stereo_id = "edit_boundary_crossings_stereo_48k";
    let timing = synthetic_edits(stereo.decoded_samples);
    let boundaries = timing
        .edits
        .iter()
        .skip(1)
        .map(|edit| {
            let start = (edit.presentation.start as i64 + timing.track_offset) as u64;
            range_at(start - READ_SAMPLES / 2, READ_SAMPLES)
        })
        .collect::<Vec<_>>();
    let mut stereo_reader = reader_for(stereo, timing);
    let stereo_work = AudioWork::new(
        boundaries.len() as u64 * READ_SAMPLES,
        stereo.sample_rate,
        stereo.channels,
    );
    let mut stereo_run = || {
        for range in &boundaries {
            black_box(
                stereo_reader
                    .get_range(*range, &cancellation)
                    .expect("a range straddling an edit boundary reads"),
            );
        }
    };
    report_realtime(stereo_id, stereo_work, &mut stereo_run);
    report_audio_throughput(&mut group, stereo_id, stereo_work);
    group.bench_function(stereo_id, |bencher| bencher.iter(&mut stereo_run));

    group.finish();
}

/// Builds an [`AacSampleReader`] over a demuxed fixture.
///
/// The packets are cloned because the reader owns them; that happens once per
/// benchmark during setup, never inside a timed loop.
fn reader_for(
    track: &BundledAacTrack,
    timing: AudioTrackTiming,
) -> AacSampleReader<NativeAacDecoder> {
    let decoder = NativeAacDecoder::new(&track.config, Limits::default())
        .expect("the fixture is AAC-LC mono or stereo, which the native backend accepts");
    AacSampleReader::new(
        decoder,
        track.packets.clone(),
        track.sample_rate,
        track.channels,
        timing,
        PREROLL_PACKETS,
        Limits::default(),
    )
    .expect("the fixture's packets are contiguous and its timing is valid")
}

/// A multi-edit timeline over a track that has none of its own.
///
/// The two media edits are deliberately far apart in media time, so a request
/// that straddles their presentation boundary maps to two disjoint media ranges
/// rather than one contiguous run.
fn synthetic_edits(decoded_length: u64) -> AudioTrackTiming {
    const PRIMING: u64 = 1024;
    const PADDING: u64 = 1024;
    /// One second of presentation per media edit.
    const SEGMENT: u64 = 48_000;
    /// A tenth of a second of leading silence.
    const SILENCE: u64 = 4_800;

    let media_span = decoded_length
        .checked_sub(PRIMING + PADDING)
        .expect("the fixture is longer than its priming and padding");
    assert!(
        SEGMENT * 2 <= media_span,
        "the fixture must be long enough for two one-second media edits"
    );
    AacTrackEdits {
        priming: PRIMING,
        padding: PADDING,
        // A whole-track delay, so the presentation clock is not the edit clock.
        track_offset: 480,
        edits: vec![
            (SampleRange::new(0, SILENCE).unwrap(), None),
            (
                SampleRange::new(SILENCE, SILENCE + SEGMENT).unwrap(),
                Some(PRIMING),
            ),
            (
                SampleRange::new(SILENCE + SEGMENT, SILENCE + 2 * SEGMENT).unwrap(),
                Some(PRIMING + media_span - SEGMENT),
            ),
        ],
    }
    .into()
}

/// Intermediate shape for [`synthetic_edits`], kept separate so the numeric
/// widths are converted in exactly one place.
struct AacTrackEdits {
    priming: u64,
    padding: u64,
    track_offset: i64,
    edits: Vec<(SampleRange, Option<u64>)>,
}

impl From<AacTrackEdits> for AudioTrackTiming {
    fn from(value: AacTrackEdits) -> Self {
        AudioTrackTiming {
            priming: u32::try_from(value.priming).expect("priming fits a u32"),
            padding: u32::try_from(value.padding).expect("padding fits a u32"),
            track_offset: value.track_offset,
            edits: value
                .edits
                .into_iter()
                .map(|(presentation, media_start)| AudioEdit {
                    presentation,
                    media_start,
                })
                .collect(),
        }
    }
}

/// A half-open range of `length` samples starting at `start`.
fn range_at(start: u64, length: u64) -> SampleRange {
    SampleRange::new(start, start + length).expect("benchmark ranges are nonempty and in order")
}

/// Decoded samples covered by a run of contiguous access units.
fn covered_samples(packets: &[EncodedAudioSample]) -> u64 {
    match (packets.first(), packets.last()) {
        (Some(first), Some(last)) => last.decoded_range.end - first.decoded_range.start,
        _ => 0,
    }
}

/// Times one pass of `run` and prints its samples/sec and realtime factor.
///
/// Criterion's `Throughput::Elements` line already reports decoded samples per
/// second, but x-realtime is the figure a playback path is judged by and
/// criterion has no unit for it, so one untimed-by-criterion pass supplies it.
fn report_realtime<T>(id: &str, work: AudioWork, run: &mut impl FnMut() -> T) {
    let started = Instant::now();
    black_box(run());
    let elapsed = started.elapsed();
    println!(
        "# {id}: {:.4}s of audio in {:.4}s => {:.0} samples/s = {:.1}x realtime",
        work.seconds(),
        elapsed.as_secs_f64(),
        work.samples_per_second(elapsed),
        work.realtime_factor(elapsed),
    );
}

criterion_group! {
    name = audio;
    config = main_criterion();
    targets = aac_decode, aac_reader_sequential, aac_reader_seek, aac_reader_edits
}
criterion_main!(audio);
