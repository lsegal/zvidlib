//! Audio write-path and read-path benchmarks: MP4 muxing, sample-table growth,
//! packet extraction, and gapless timing reconstruction.
//!
//! # There is no audio encoder in this crate to benchmark
//!
//! [`zvidlib::AudioEncoder`] (`src/codec.rs`) is a trait with no implementation
//! anywhere in the tree. Its only implementor is `PcmFixtureEncoder` in
//! `tests/indexed_mp4_output.rs`, a test double that packages PCM sample ranges
//! into `EncodedSample`s without compressing anything. So "benchmark the audio
//! encoder" has no subject.
//!
//! That is now settled rather than open: the crate ships no audio encoder on
//! purpose, and `AudioEncoder` stays the seam that platform and browser backends
//! fill. The rationale is recorded on the trait itself in `src/codec.rs` and in
//! the README. The practical consequence for this suite is that there will be no
//! audio-encode target to add later unless that decision is revisited, so nothing
//! here is a placeholder waiting on one.
//!
//! What *does* exist on the audio write path is the container work, and that is
//! what is measured here. [`PcmBenchEncoder`] below is the same kind of pass-
//! through double, kept bench-local on purpose: with codec work held at
//! effectively zero, every measured nanosecond belongs to
//! [`zvidlib::output::MediaOutput`] and [`zvidlib::mp4::Mp4Muxer`].
//!
//! # What each group measures
//!
//! * `audio_mux` — the write path. `media_output_*` drives the full synchronized
//!   session (index checking, timeline interval validation, encoder dispatch,
//!   muxer writes, gapless drain, finalization). `sample_table_*` drives the
//!   muxer alone over a long audio-only track, which is the part that scales:
//!   the muxer writes one chunk per sample, so `stsz` and `co64` each grow by a
//!   fixed width per sample while `stts`/`stsc` stay run-length constant. The
//!   two sizes an order of magnitude apart are the regression guard — the ratio
//!   between them should stay linear.
//! * `audio_demux` — the read path over the bundled sample's real AAC track:
//!   `Mp4Demuxer::open` (which parses those same sample tables back),
//!   `to_encoded_audio_samples` (packet extraction over the decoded sample
//!   clock), and `audio_timing` (priming, padding, and edit-list mapping).
//!
//! # No SIMD axis
//!
//! Muxing and demuxing are bit-shuffling and table building, not arithmetic over
//! sample arrays, so there is nothing here for a vector kernel to do. These
//! groups run on the detected instruction set only and do not use
//! `support::isa::bench_across_isas`. They still carry the `simd=off`/`simd=on`
//! build tag every group name in the suite carries, because that tag records
//! which *build* produced a number, not which kernel ran.
//!
//! See `benches/README.md` for how to run and filter this target.

mod support;

use support::isa::log_host_isas;

use std::hint::black_box;
use std::time::Instant;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::io::{ByteSink, MemorySink, MemorySource};
use zvidlib::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::transfer::{CpuFrameSource, FrameSource, Orientation};
use zvidlib::{
    AudioBuffer, AudioDrain, AudioEncoder, AudioEncoderFormat, AudioGapless, Codec, ColorRange,
    EncodedSample, EncoderConfig, EncoderFuture, FrameIndex, FrameRate, Limits, MediaOutput,
    Mp4Demuxer, Mp4DemuxerOptions, OutputOptions, PixelFormat, Plane, SampleDependency, Timeline,
    VideoDimensions, VideoEncoder, VideoEncoderFormat, VideoFrame,
};

use support::{AudioWork, block_on, group_name, report_audio_throughput};

/// The output sample rate every write-path fixture uses.
const SAMPLE_RATE: u32 = 48_000;

/// Stereo, matching the bundled sample's real AAC track.
const CHANNELS: u16 = 2;

/// The video track's MP4 timescale. 30000/1000 keeps one frame exactly 1000
/// ticks, so decode timestamps stay contiguous without rounding.
const VIDEO_TIMESCALE: u32 = 30_000;

/// Ticks per frame in [`VIDEO_TIMESCALE`].
const VIDEO_FRAME_DURATION: u32 = 1_000;

/// Frames the synchronized session writes per iteration: one second at 30 fps,
/// which is 48,000 audio samples.
const SESSION_FRAMES: u64 = 30;

/// Samples an AAC-LC access unit covers. The sample-table groups use it so their
/// packet counts translate directly to seconds of audio.
const AAC_FRAME_SAMPLES: u64 = 1_024;

/// Wraps a payload in a minimal MP4 box, which is all the muxer validates a
/// codec configuration to be.
fn codec_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut output = u32::try_from(payload.len() + 8)
        .expect("bench codec configuration fits in a 32-bit box")
        .to_be_bytes()
        .to_vec();
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    output
}

/// A pass-through "audio encoder" that packages one packet per write.
///
/// This is not a codec and is not pretending to be one; see the module docs. It
/// exists so the muxer has a source of well-formed `EncodedSample`s whose cost
/// is a `Vec` allocation, leaving the measurement to the container path.
struct PcmBenchEncoder {
    config: EncoderConfig,
    format: AudioEncoderFormat,
    gapless: AudioGapless,
    /// Bytes emitted per packet, so packet sizes resemble AAC-LC access units
    /// rather than being degenerate.
    packet_bytes: usize,
}

impl PcmBenchEncoder {
    fn new() -> Self {
        Self {
            config: EncoderConfig {
                codec: Codec::Aac,
                timescale: SAMPLE_RATE,
                decoder_config: codec_box(b"esds", &[0, 0, 0, 0]),
            },
            format: AudioEncoderFormat {
                sample_rate: SAMPLE_RATE,
                channels: CHANNELS,
            },
            // Real AAC-LC decoder priming, so the drain exercises the muxer's
            // gapless validation and `edts` construction rather than skipping it.
            gapless: AudioGapless {
                priming: 1_024,
                padding: 512,
            },
            packet_bytes: 384,
        }
    }
}

impl AudioEncoder for PcmBenchEncoder {
    fn config(&self) -> &EncoderConfig {
        &self.config
    }

    fn format(&self) -> AudioEncoderFormat {
        self.format
    }

    fn encode<'a>(
        &'a mut self,
        _index: FrameIndex,
        buffer: AudioBuffer,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move {
            let dts = i64::try_from(buffer.range.start).expect("bench sample clock fits in i64");
            Ok(vec![EncodedSample {
                data: vec![0xa5; self.packet_bytes],
                dts,
                pts: dts,
                duration: u32::try_from(buffer.range.len()).expect("bench packet duration fits"),
                is_sync: true,
                dependency: SampleDependency::INDEPENDENT,
            }])
        })
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, AudioDrain> {
        Box::pin(async {
            Ok(AudioDrain {
                samples: Vec::new(),
                gapless: self.gapless,
            })
        })
    }
}

/// A pass-through video encoder, present only because a synchronized session
/// requires both tracks. It emits one in-order sample per frame so the video
/// side contributes no reordering or buffering to the measurement.
struct PassthroughVideoEncoder {
    config: EncoderConfig,
    format: VideoEncoderFormat,
}

impl PassthroughVideoEncoder {
    fn new(dimensions: VideoDimensions) -> Self {
        Self {
            config: EncoderConfig {
                codec: Codec::Av1,
                timescale: VIDEO_TIMESCALE,
                decoder_config: codec_box(b"av1C", &[0x81, 0, 0, 0]),
            },
            format: VideoEncoderFormat {
                dimensions,
                pixel_format: PixelFormat::Gray8,
            },
        }
    }
}

impl VideoEncoder for PassthroughVideoEncoder {
    fn config(&self) -> &EncoderConfig {
        &self.config
    }

    fn format(&self) -> VideoEncoderFormat {
        self.format
    }

    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        _frame: FrameSource<'a>,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move {
            let dts = i64::try_from(index.0 * u64::from(VIDEO_FRAME_DURATION))
                .expect("bench video clock fits in i64");
            Ok(vec![EncodedSample {
                data: vec![0x5a; 64],
                dts,
                pts: dts,
                duration: VIDEO_FRAME_DURATION,
                is_sync: index.0 == 0,
                dependency: SampleDependency::INDEPENDENT,
            }])
        })
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}

/// One 2x2 gray frame, reused for every video write. Video content is not the
/// subject here and a larger frame would only add a memcpy the encoder ignores.
fn bench_video_frame(dimensions: VideoDimensions) -> VideoFrame {
    VideoFrame::new(
        dimensions,
        PixelFormat::Gray8,
        ColorRange::Full,
        vec![Plane {
            data: vec![0x40; 4],
            stride: 2,
        }],
        &Limits::default(),
    )
    .expect("the bench video frame is valid")
}

/// The audio buffers a whole session writes, built once so the timed loop does
/// not measure PCM synthesis.
fn session_audio_buffers(timeline: Timeline, frames: u64) -> Vec<AudioBuffer> {
    let limits = Limits::default();
    (0..frames)
        .map(|frame| {
            let range = timeline
                .audio_interval_for_frame(FrameIndex(frame))
                .expect("the bench timeline has an interval for every frame");
            let count = usize::try_from(range.len() * u64::from(CHANNELS))
                .expect("the bench audio interval fits in memory");
            AudioBuffer::new(range, SAMPLE_RATE, CHANNELS, vec![0.0; count], &limits)
                .expect("bench audio buffers are valid")
        })
        .collect()
}

/// Runs one synchronized session end to end and returns the written bytes.
fn run_session(
    buffers: Vec<AudioBuffer>,
    dimensions: VideoDimensions,
    frame: &VideoFrame,
) -> usize {
    block_on(async {
        let timeline = Timeline::new(
            FrameRate::new(SESSION_FRAMES as u32, 1).expect("the bench frame rate is valid"),
            SAMPLE_RATE,
        )
        .expect("the bench timeline is valid");
        let mut output = MediaOutput::new(
            MemorySink::new(),
            PassthroughVideoEncoder::new(dimensions),
            PcmBenchEncoder::new(),
            timeline,
            OutputOptions::default(),
        )
        .await
        .expect("the bench output session opens");
        for (index, buffer) in buffers.into_iter().enumerate() {
            let index = FrameIndex(index as u64);
            output
                .put_video(
                    index,
                    FrameSource::Cpu(CpuFrameSource {
                        frame,
                        orientation: Orientation::TopLeft,
                    }),
                )
                .await
                .expect("the bench video write succeeds");
            output
                .put_audio(index, buffer)
                .await
                .expect("the bench audio write succeeds");
        }
        output
            .finish()
            .await
            .expect("the bench output session finalizes")
            .position() as usize
    })
}

/// Writes `packets` audio samples through an audio-only muxer and finalizes it.
///
/// This is the sample-table workload: `write_sample` appends one record and one
/// chunk offset per packet, and `finish` turns the whole accumulated table into
/// `stts`/`stsc`/`stsz`/`co64`.
fn run_sample_table(packets: u64, packet_bytes: usize) -> usize {
    block_on(async {
        let config = Mp4TrackConfig {
            encoder: EncoderConfig {
                codec: Codec::Aac,
                timescale: SAMPLE_RATE,
                decoder_config: codec_box(b"esds", &[0, 0, 0, 0]),
            },
            format: Mp4TrackFormat::Audio { channels: CHANNELS },
        };
        let mut muxer = Mp4Muxer::new(
            MemorySink::new(),
            vec![config],
            usize::try_from(packets).expect("the bench packet count fits"),
        )
        .await
        .expect("the bench audio muxer opens");
        for packet in 0..packets {
            let dts = i64::try_from(packet * AAC_FRAME_SAMPLES).expect("bench dts fits in i64");
            muxer
                .write_sample(
                    0,
                    EncodedSample {
                        data: vec![0xa5; packet_bytes],
                        dts,
                        pts: dts,
                        duration: AAC_FRAME_SAMPLES as u32,
                        is_sync: true,
                        dependency: SampleDependency::INDEPENDENT,
                    },
                )
                .await
                .expect("the bench audio sample is writable");
        }
        muxer
            .set_audio_gapless(
                0,
                AudioGapless {
                    priming: 1_024,
                    padding: 512,
                },
            )
            .expect("the bench track accepts gapless metadata");
        muxer
            .finish()
            .await
            .expect("the bench audio muxer finalizes")
            .position() as usize
    })
}

/// The audio write path: a synchronized session, and sample-table growth.
fn audio_mux(criterion: &mut Criterion) {
    let dimensions = VideoDimensions::new(2, 2, &Limits::default())
        .expect("the bench video dimensions are valid");
    let frame = bench_video_frame(dimensions);
    let timeline = Timeline::new(
        FrameRate::new(SESSION_FRAMES as u32, 1).expect("the bench frame rate is valid"),
        SAMPLE_RATE,
    )
    .expect("the bench timeline is valid");
    let buffers = session_audio_buffers(timeline, SESSION_FRAMES);
    let session_samples: u64 = buffers.iter().map(|buffer| buffer.range.len()).sum();
    let session_work = AudioWork::new(session_samples, SAMPLE_RATE, CHANNELS);

    // One timed pass before criterion starts, so a filtered run still reports
    // the x-realtime figure criterion's own `elem/s` line does not carry.
    let started = Instant::now();
    let written = run_session(buffers.clone(), dimensions, &frame);
    println!(
        "# zvidlib audio benches: simd feature {}, {} samples muxed into {written} bytes at {:.0}x realtime",
        if support::simd_enabled() { "on" } else { "off" },
        session_samples,
        session_work.realtime_factor(started.elapsed()),
    );

    let name = group_name("audio_mux");
    let mut group = criterion.benchmark_group(&name);

    report_audio_throughput(&mut group, "media_output_1s_30fps", session_work);
    group.bench_function("media_output_1s_30fps", |bencher| {
        bencher.iter_batched(
            || buffers.clone(),
            |buffers| black_box(run_session(buffers, dimensions, &frame)),
            criterion::BatchSize::SmallInput,
        );
    });

    // An order of magnitude apart, so the growth is visible as a ratio rather
    // than a single number: the muxer emits one chunk per sample, so `stsz` and
    // `co64` are both O(samples) while `stts`/`stsc` stay run-length constant.
    for packets in [1_500_u64, 15_000] {
        let id = format!("sample_table_{packets}_packets");
        let work = AudioWork::new(packets * AAC_FRAME_SAMPLES, SAMPLE_RATE, CHANNELS);
        report_audio_throughput(&mut group, &id, work);
        group.bench_function(&id, |bencher| {
            bencher.iter(|| black_box(run_sample_table(black_box(packets), 384)));
        });
    }
    group.finish();
}

/// The audio read path over the bundled sample's real AAC track.
fn audio_demux(criterion: &mut Criterion) {
    let fixture = support::bundled_aac_track();
    let work = fixture.work();
    let limits = Limits::default();

    let started = Instant::now();
    let packets = block_on(
        fixture
            .track()
            .to_encoded_audio_samples(&fixture.source, &limits),
    )
    .expect("the bundled AAC packets are readable");
    println!(
        "# bundled AAC track: {} packets, {} samples at {} Hz ({:.2} s), extracted at {:.0}x realtime",
        packets.len(),
        work.samples,
        work.sample_rate,
        work.seconds(),
        work.realtime_factor(started.elapsed()),
    );

    let name = group_name("audio_demux");
    let mut group = criterion.benchmark_group(&name);

    // Deliberately first: `audio_timing` reconstructs priming, padding, and the
    // edit list, which is O(edits) (one, here) and touches no samples at all, so
    // a samples/sec rate for it would be a fabricated number. Criterion's group
    // throughput is sticky once set, so this benchmark has to run before the two
    // that do register one.
    group.bench_function("aac_timing_bundled", |bencher| {
        bencher.iter(|| {
            black_box(
                fixture
                    .track()
                    .audio_timing(black_box(fixture.movie.movie_timescale))
                    .expect("the bundled AAC track has readable gapless timing"),
            )
        });
    });

    // `open` parses the whole movie, including the video track's much larger
    // sample tables, so it is the read-side counterpart to `finish` writing them.
    report_audio_throughput(&mut group, "open_bundled_mp4", work);
    group.bench_function("open_bundled_mp4", |bencher| {
        bencher.iter_batched(
            || MemorySource::new(support::bundled_mp4_bytes().to_vec()),
            |source| {
                black_box(
                    block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))
                        .expect("the bundled sample is a readable MP4"),
                )
            },
            criterion::BatchSize::LargeInput,
        );
    });

    report_audio_throughput(&mut group, "aac_packets_bundled", work);
    group.bench_function("aac_packets_bundled", |bencher| {
        bencher.iter(|| {
            black_box(
                block_on(
                    fixture
                        .track()
                        .to_encoded_audio_samples(&fixture.source, &limits),
                )
                .expect("the bundled AAC packets are readable"),
            )
        });
    });

    group.finish();
}

criterion_group!(benches, log_host_isas, audio_mux, audio_demux);
criterion_main!(benches);
