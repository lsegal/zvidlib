//! Scalar-versus-SIMD benchmarks for the native Vorbis encoder (issue #573).
//!
//! Each group encodes the same ten seconds of synthetic audio through the
//! public [`native_vorbis_audio_encoder_factory`], once per instruction set
//! `zvidlib_core::simd::available()` reports, through the crate-wide override in
//! [`zvidlib_core::simd`]. `benches/support/isa.rs` asserts that every arm's
//! packets are byte-identical with the scalar arm's before timing it, and that
//! the override really reached the `vorbis_encode` dispatch site, so a reported
//! speedup cannot come from a kernel that quietly diverged or from a switch
//! that never took effect.
//!
//! # Groups
//!
//! | Group | Configuration |
//! | --- | --- |
//! | `vorbis_encode_44100_stereo_q4` | 44.1 kHz stereo at the factory's default quality 4 (about 128 kb/s) |
//! | `vorbis_encode_48000_stereo_192k` | 48 kHz stereo at a 192 kb/s nominal rate |
//! | `vorbis_encode_44100_mono_64k` | 44.1 kHz mono at a 64 kb/s nominal rate |
//!
//! The whole encode is the unit on purpose. The vector kernels cover the
//! forward MDCT, the real FFT, the noise-mask fits, floor fitting and the log
//! spectra; tone masking, coupling/quantization and residue coding are serial
//! and stay scalar (see `src/vorbis_encoder/simd/mod.rs`). A whole-encode
//! ratio is therefore the speedup a caller actually gets, diluted by the
//! scalar stages, rather than a kernel-level ratio that overstates it.
//!
//! See `benches/README.md` for how to run and filter the suite.

mod support;

use std::f32::consts::TAU;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib_core::{
    AudioBuffer, AudioEncoderConfig, AudioEncoderFactory, Codec, CodecProfile, EncodedSample,
    FrameIndex, Limits, SampleRange,
};
use zvidlib_vorbis_encoder::native_vorbis_audio_encoder_factory;

use support::isa::{AudioIsaWorkload, bench_audio_across_isas, log_host_isas};
use support::{AudioWork, block_on};

/// Seconds of audio one iteration encodes: long enough that the stream start
/// (one long block of LPC extrapolation) is a small share of the work.
const SECONDS: u32 = 10;

/// Frames handed to the encoder per `encode` call, a typical capture buffer.
const CHUNK_FRAMES: usize = 1024;

/// Deterministic music-like input: two partials per channel, a slow vibrato,
/// periodic percussive bursts (which drive the encoder into short blocks) and
/// a low noise floor.
fn signal(sample_rate: u32, channels: u16) -> Vec<f32> {
    let frames = (sample_rate * SECONDS) as usize;
    let mut seed = 0x2545_f491_u32;
    let mut pcm = Vec::with_capacity(frames * usize::from(channels));
    for i in 0..frames {
        let t = i as f32 / sample_rate as f32;
        let vibrato = 1.0 + 0.003 * (TAU * 5.0 * t).sin();
        let beat = (i % (sample_rate as usize / 2)) as f32 / sample_rate as f32;
        let burst = (-beat * 60.0).exp();
        for c in 0..channels {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let f0 = 196.0 * (1.0 + 0.5 * f32::from(c)) * vibrato;
            let tone = 0.25 * (TAU * f0 * t).sin() + 0.08 * (TAU * 3.0 * f0 * t).sin();
            pcm.push(tone + 0.4 * burst * noise + 0.01 * noise);
        }
    }
    pcm
}

/// Encodes `pcm` and returns every packet's bytes followed by its timestamp
/// and duration, so the bit-exactness guard compares timing as well as
/// payload.
fn encode(config: &AudioEncoderConfig, pcm: &[f32]) -> Vec<u8> {
    let channels = usize::from(config.channels);
    let limits = Limits::default();
    let mut encoder = native_vorbis_audio_encoder_factory()
        .create(config, &limits)
        .expect("the native Vorbis encoder accepts the benchmark configuration");
    let mut out = Vec::new();
    let mut append = |samples: Vec<EncodedSample>| {
        for sample in samples {
            out.extend_from_slice(&sample.data);
            out.extend_from_slice(&sample.pts.to_le_bytes());
            out.extend_from_slice(&sample.duration.to_le_bytes());
        }
    };
    let frames = pcm.len() / channels;
    let mut start = 0;
    while start < frames {
        let end = (start + CHUNK_FRAMES).min(frames);
        let buffer = AudioBuffer::new(
            SampleRange::new(start as u64, end as u64).expect("a non-empty range"),
            config.sample_rate,
            config.channels,
            pcm[start * channels..end * channels].to_vec(),
            &limits,
        )
        .expect("a well-formed buffer");
        append(block_on(encoder.encode(FrameIndex(0), buffer)).expect("encode"));
        start = end;
    }
    append(block_on(encoder.finish()).expect("finish").samples);
    out
}

fn vorbis_encode(c: &mut Criterion) {
    let configs: [(u32, u16, &str, Option<u32>); 3] = [
        (44_100, 2, "q4", None),
        (48_000, 2, "192k", Some(192_000)),
        (44_100, 1, "64k", Some(64_000)),
    ];
    for (sample_rate, channels, label, bit_rate) in configs {
        let config = AudioEncoderConfig {
            codec: Codec::Vorbis,
            profile: CodecProfile::Vorbis,
            sample_rate,
            channels,
            timescale: sample_rate,
            configuration: bit_rate.map_or_else(Vec::new, |rate| rate.to_be_bytes().to_vec()),
        };
        let pcm = signal(sample_rate, channels);
        let layout = if channels == 1 { "mono" } else { "stereo" };
        let name = format!("vorbis_encode_{sample_rate}_{layout}_{label}");
        let work = AudioWork::new(u64::from(sample_rate * SECONDS), sample_rate, channels);
        // A whole encode takes a tenth of a second or more, so fewer, longer
        // samples than the stage-sized default.
        let workload = AudioIsaWorkload {
            sample_size: 10,
            measurement_time: Duration::from_secs(5),
            ..AudioIsaWorkload::new(&name, work)
        };
        bench_audio_across_isas(c, &workload, || encode(&config, &pcm));
    }
}

criterion_group!(benches, log_host_isas, vorbis_encode);
criterion_main!(benches);
