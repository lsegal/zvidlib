//! Opus decode against libopus, sample-accurate reads, and the native encoder.
//!
//! The fixtures under `tests/fixtures/codec/opus_*` were encoded by libopus
//! and decoded by libopus to the `*_libopus.s16` references (see that
//! directory's README), one per Opus coding mode. `opus_compare` below is the
//! RFC 6716/8251 conformance metric, ported from libopus.

use std::f32::consts::PI;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::task::{Context, Poll, Waker};

use zvidlib_container::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use zvidlib_container::{Mp4Demuxer, Mp4DemuxerOptions};
use zvidlib_container::{WebmDemuxer, WebmDemuxerOptions, WebmMuxer};
use zvidlib_core::io::{MemorySink, MemorySource};
use zvidlib_core::{
    AudioBuffer, AudioDecoder, AudioEncoderConfig, AudioEncoderFactory, AudioSampleReader,
    CancellationToken, Codec, CodecProfile, CodecSupport, EncodedAudioSample, FrameIndex, Limits,
    SampleRange, TrackKind,
};
use zvidlib_opus::{NativeOpusDecoder, OPUS_SAMPLE_RATE, OpusHead};
use zvidlib_opus::{native_opus_audio_encoder_factory, opus_packet_samples, opus_preroll_packets};

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

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn read_s16(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])))
        .collect()
}

/// Rounds decoded `f32` samples to 16-bit PCM values, as libopus's integer
/// API and `opus_demo` do.
fn to_s16(samples: &[f32]) -> Vec<f32> {
    samples
        .iter()
        .map(|value| (value * 32_768.0).round().clamp(-32_768.0, 32_767.0))
        .collect()
}

/// An exact-sample reader over an MP4's Opus track, as a caller builds one.
fn open_reader(bytes: Vec<u8>) -> (AudioSampleReader<NativeOpusDecoder>, OpusHead) {
    let source = MemorySource::new(bytes);
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
    let track = movie
        .tracks
        .iter()
        .find(|track| track.kind == TrackKind::Audio)
        .expect("an audio track");
    assert_eq!(track.codec, Codec::Opus);
    let head = track.opus_config().unwrap();
    let packets = block_on(track.to_encoded_audio_samples(&source, &Limits::default())).unwrap();
    let timing = track.audio_timing(movie.movie_timescale).unwrap();
    let preroll = opus_preroll_packets(&packets);
    let decoder = NativeOpusDecoder::new(&head, Limits::default()).unwrap();
    let reader = AudioSampleReader::new(
        decoder,
        packets,
        OPUS_SAMPLE_RATE,
        u16::from(head.channels),
        timing,
        preroll,
        Limits::default(),
    )
    .unwrap();
    (reader, head)
}

fn read_all(reader: &mut AudioSampleReader<NativeOpusDecoder>) -> Vec<f32> {
    let range = SampleRange::new(0, reader.presentation_length()).unwrap();
    reader
        .get_range(range, &CancellationToken::new())
        .unwrap()
        .samples
}

// --- opus_compare -----------------------------------------------------------
//
// A port of libopus's `src/opus_compare.c`: a pseudo-NMR over 21 Bark-derived
// bands with simple frequency and temporal masking, reduced to a quality
// figure. A decode passes when the figure is not negative.

const BANDS: [usize; 22] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 68, 80, 96, 120, 156, 200,
];
const NBANDS: usize = 21;
const NFREQS: usize = 240;
const TEST_WIN_SIZE: usize = 480;
const TEST_WIN_STEP: usize = 120;

#[allow(clippy::too_many_arguments)]
fn band_energy(
    mut out: Option<&mut [f32]>,
    ps: &mut [f32],
    input: &[f32],
    channels: usize,
    frames: usize,
) {
    let window_size = TEST_WIN_SIZE;
    let window: Vec<f32> = (0..window_size)
        .map(|j| 0.5 - 0.5 * ((2.0 * PI / (window_size - 1) as f32) * j as f32).cos())
        .collect();
    let cos: Vec<f32> = (0..window_size)
        .map(|j| ((2.0 * PI / window_size as f32) * j as f32).cos())
        .collect();
    let sin: Vec<f32> = (0..window_size)
        .map(|j| ((2.0 * PI / window_size as f32) * j as f32).sin())
        .collect();
    let ps_size = window_size / 2;
    let mut x = vec![0_f32; channels * window_size];
    for frame in 0..frames {
        for channel in 0..channels {
            for k in 0..window_size {
                x[channel * window_size + k] =
                    window[k] * input[(frame * TEST_WIN_STEP + k) * channels + channel];
            }
        }
        let mut bin = 0;
        for band in 0..NBANDS {
            let mut power = [0_f32; 2];
            while bin < BANDS[band + 1] {
                for channel in 0..channels {
                    let (mut re, mut im, mut ti) = (0_f32, 0_f32, 0_usize);
                    for k in 0..window_size {
                        re += cos[ti] * x[channel * window_size + k];
                        im -= sin[ti] * x[channel * window_size + k];
                        ti += bin;
                        if ti >= window_size {
                            ti -= window_size;
                        }
                    }
                    let value = re * re + im * im + 100_000.0;
                    ps[(frame * ps_size + bin) * channels + channel] = value;
                    power[channel] += value;
                }
                bin += 1;
            }
            if let Some(out) = out.as_deref_mut() {
                let width = (BANDS[band + 1] - BANDS[band]) as f32;
                for (channel, &value) in power.iter().enumerate().take(channels) {
                    out[(frame * NBANDS + band) * channels + channel] = value / width;
                }
            }
        }
    }
}

/// The `opus_compare` quality of decoded 48 kHz 16-bit samples `decoded`,
/// interleaved with `channels` channels, against a stereo reference. A mono
/// decode is compared against the reference's average of its two channels.
fn opus_compare(stereo_reference: &[f32], decoded: &[f32], channels: usize) -> f64 {
    let length = stereo_reference.len() / 2;
    let reference: Vec<f32> = if channels == 1 {
        (0..length)
            .map(|i| 0.5 * (stereo_reference[2 * i] + stereo_reference[2 * i + 1]))
            .collect()
    } else {
        stereo_reference.to_vec()
    };
    assert_eq!(length, decoded.len() / channels, "sample counts differ");
    let frames = (length - TEST_WIN_SIZE + TEST_WIN_STEP) / TEST_WIN_STEP;
    let mut xb = vec![0_f32; frames * NBANDS * channels];
    let mut xs = vec![0_f32; frames * NFREQS * channels];
    let mut ys = vec![0_f32; frames * NFREQS * channels];
    band_energy(Some(&mut xb), &mut xs, &reference, channels, frames);
    band_energy(None, &mut ys, decoded, channels, frames);
    let at =
        |frame: usize, band: usize, channel: usize| (frame * NBANDS + band) * channels + channel;
    for frame in 0..frames {
        for band in 1..NBANDS {
            for channel in 0..channels {
                xb[at(frame, band, channel)] += 0.1 * xb[at(frame, band - 1, channel)];
            }
        }
        for band in (0..NBANDS - 1).rev() {
            for channel in 0..channels {
                xb[at(frame, band, channel)] += 0.03 * xb[at(frame, band + 1, channel)];
            }
        }
        if frame > 0 {
            for band in 0..NBANDS {
                for channel in 0..channels {
                    xb[at(frame, band, channel)] += 0.5 * xb[at(frame - 1, band, channel)];
                }
            }
        }
        if channels == 2 {
            for band in 0..NBANDS {
                let (left, right) = (xb[at(frame, band, 0)], xb[at(frame, band, 1)]);
                xb[at(frame, band, 0)] += 0.01 * right;
                xb[at(frame, band, 1)] += 0.01 * left;
            }
        }
        for band in 0..NBANDS {
            for bin in BANDS[band]..BANDS[band + 1] {
                for channel in 0..channels {
                    let mask = 0.1 * xb[at(frame, band, channel)];
                    xs[(frame * NFREQS + bin) * channels + channel] += mask;
                    ys[(frame * NFREQS + bin) * channels + channel] += mask;
                }
            }
        }
    }
    for bin in 0..BANDS[NBANDS] {
        for channel in 0..channels {
            let mut previous_x = xs[bin * channels + channel];
            let mut previous_y = ys[bin * channels + channel];
            for frame in 1..frames {
                let index = (frame * NFREQS + bin) * channels + channel;
                let (x, y) = (xs[index], ys[index]);
                xs[index] += previous_x;
                ys[index] += previous_y;
                previous_x = x;
                previous_y = y;
            }
        }
    }
    let mut error = 0_f64;
    for frame in 0..frames {
        let mut frame_error = 0_f64;
        for band in 0..NBANDS {
            let mut band_error = 0_f64;
            for bin in BANDS[band]..BANDS[band + 1] {
                for channel in 0..channels {
                    let index = (frame * NFREQS + bin) * channels + channel;
                    let ratio = ys[index] / xs[index];
                    let mut distortion = ratio - ratio.ln() - 1.0;
                    if (79..=81).contains(&bin) {
                        distortion *= 0.1;
                    }
                    if bin == 80 {
                        distortion *= 0.1;
                    }
                    band_error += f64::from(distortion);
                }
            }
            band_error /= ((BANDS[band + 1] - BANDS[band]) * channels) as f64;
            frame_error += band_error * band_error;
        }
        frame_error /= NBANDS as f64;
        frame_error *= frame_error;
        error += frame_error * frame_error;
    }
    let error = (error / frames as f64).powf(1.0 / 16.0);
    100.0 * (1.0 - 0.5 * (1.0 + error).ln() / 1.13_f64.ln())
}

fn snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
    let signal: f64 = reference.iter().map(|&v| f64::from(v).powi(2)).sum();
    let noise: f64 = reference
        .iter()
        .zip(decoded)
        .map(|(&r, &d)| (f64::from(r) - f64::from(d)).powi(2))
        .sum();
    10.0 * (signal / noise.max(f64::MIN_POSITIVE)).log10()
}

// --- libopus fixtures --------------------------------------------------------

fn assert_matches_libopus(name: &str, expected_toc_config: u8) {
    let bytes = std::fs::read(fixture(&format!("{name}.mp4"))).unwrap();
    let (mut reader, head) = open_reader(bytes.clone());
    let channels = usize::from(head.channels);
    // The pre-skip and the edit list's end trim leave exactly the half second
    // that was encoded.
    assert_eq!(reader.presentation_length(), 24_000, "{name}");
    assert_eq!(head.pre_skip, 312, "{name}");

    let source = MemorySource::new(bytes);
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
    let packets =
        block_on(movie.tracks[0].to_encoded_audio_samples(&source, &Limits::default())).unwrap();
    for packet in &packets {
        assert_eq!(
            packet.data[0] >> 3,
            expected_toc_config,
            "{name} coding mode"
        );
        assert_eq!(
            u64::from(opus_packet_samples(&packet.data).unwrap()),
            packet.decoded_range.len()
        );
    }

    let decoded = to_s16(&read_all(&mut reader));
    let reference = read_s16(&fixture(&format!("{name}_libopus.s16")));
    assert_eq!(decoded.len(), reference.len(), "{name}");
    let largest_difference = decoded
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        largest_difference <= 2.0,
        "{name}: decoded samples differ from libopus's by up to {largest_difference}"
    );
    let stereo_reference: Vec<f32> = if channels == 1 {
        reference.iter().flat_map(|&v| [v, v]).collect()
    } else {
        reference.clone()
    };
    let quality = opus_compare(&stereo_reference, &decoded, channels);
    assert!(
        quality >= 99.0,
        "{name}: opus_compare quality {quality:.1}%"
    );
}

#[test]
fn celt_fullband_stereo_decodes_as_libopus_does() {
    assert_matches_libopus("opus_celt_stereo", 31);
}

#[test]
fn hybrid_fullband_mono_decodes_as_libopus_does() {
    assert_matches_libopus("opus_hybrid_mono", 15);
}

#[test]
fn silk_narrowband_mono_decodes_as_libopus_does() {
    assert_matches_libopus("opus_silk_mono", 1);
}

#[test]
fn reads_after_seeking_return_the_requested_samples() {
    for name in ["opus_celt_stereo", "opus_hybrid_mono", "opus_silk_mono"] {
        let bytes = std::fs::read(fixture(&format!("{name}.mp4"))).unwrap();
        let (mut continuous, head) = open_reader(bytes.clone());
        let channels = usize::from(head.channels);
        let full = read_all(&mut continuous);

        let (mut reader, _) = open_reader(bytes);
        let cancellation = CancellationToken::new();
        // Forwards, backwards, across packet boundaries, one sample, the very
        // start and the very end, each answered from a cold or a reset decoder.
        for (start, end) in [
            (17_003, 17_004),
            (12_345, 13_000),
            (500, 900),
            (0, 100),
            (23_000, 24_000),
            (9_599, 9_601),
            (4_000, 21_000),
        ] {
            let buffer = reader
                .get_range(SampleRange::new(start, end).unwrap(), &cancellation)
                .unwrap();
            assert_eq!(buffer.range, SampleRange { start, end });
            let expected = &full[start as usize * channels..end as usize * channels];
            assert_eq!(buffer.samples.len(), expected.len());
            // Decoding restarts a preroll ahead of the request, by when the
            // decoder has converged on the continuous decode's state: CELT to
            // within `f32` rounding, and SILK, whose filters run on 16-bit
            // integers, to within a few steps of 16-bit PCM.
            let largest_difference = buffer
                .samples
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(
                largest_difference <= 8.0 / 32_768.0,
                "{name} [{start}, {end}) differs from a continuous decode by {largest_difference}"
            );
        }
    }
}

#[test]
fn identification_header_and_dops_round_trip() {
    let mut head = OpusHead::new(2, 312, 44_100).unwrap();
    head.output_gain = -256;
    let dops = head.to_dops();
    assert_eq!(&dops[4..8], b"dOps");
    assert_eq!(OpusHead::from_dops(&dops).unwrap(), head);
    let packet = head.to_identification_header();
    assert_eq!(&packet[..8], b"OpusHead");
    assert_eq!(OpusHead::from_identification_header(&packet).unwrap(), head);
    assert!((head.output_gain_factor() - 10_f32.powf(-1.0 / 20.0)).abs() < 1e-6);
}

// --- the native encoder -------------------------------------------------------

/// One second of a stereo test signal: a chord with vibrato in one channel, a
/// rising sweep in the other, and a little noise in both.
fn test_signal(frames: usize) -> Vec<f32> {
    let mut seed = 0x1234_5678_u32;
    let mut samples = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        let t = i as f32 / OPUS_SAMPLE_RATE as f32;
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (seed >> 9) as f32 / (1 << 23) as f32 - 0.5;
        let vibrato = 3.0 * (2.0 * PI * 5.0 * t).sin();
        let left = 0.2 * (2.0 * PI * (330.0 + vibrato) * t).sin()
            + 0.15 * (2.0 * PI * 440.0 * t).sin()
            + 0.1 * (2.0 * PI * 1_320.0 * t).sin()
            + 0.02 * noise;
        let right = 0.3 * (2.0 * PI * (200.0 + 1_500.0 * t) * t).sin() + 0.02 * noise;
        samples.push(left);
        samples.push(right);
    }
    samples
}

/// Encodes interleaved stereo `samples` with the native encoder, in buffers of
/// `chunk` frames, and muxes the result into an MP4.
fn encode_to_mp4(samples: &[f32], channels: u16, chunk: usize, bit_rate: Option<u32>) -> Vec<u8> {
    let factory = native_opus_audio_encoder_factory();
    let configuration = AudioEncoderConfig {
        codec: Codec::Opus,
        profile: CodecProfile::Opus,
        sample_rate: OPUS_SAMPLE_RATE,
        channels,
        timescale: OPUS_SAMPLE_RATE,
        configuration: bit_rate.map_or_else(Vec::new, |rate| rate.to_be_bytes().to_vec()),
    };
    assert!(factory.capability(&configuration).is_supported());
    let mut encoder = factory.create(&configuration, &Limits::default()).unwrap();
    let mut encoded = Vec::new();
    let frames = samples.len() / usize::from(channels);
    let mut start = 0;
    while start < frames {
        let end = (start + chunk).min(frames);
        let buffer = AudioBuffer::new(
            SampleRange::new(start as u64, end as u64).unwrap(),
            OPUS_SAMPLE_RATE,
            channels,
            samples[start * usize::from(channels)..end * usize::from(channels)].to_vec(),
            &Limits::default(),
        )
        .unwrap();
        encoded.extend(block_on(encoder.encode(FrameIndex(0), buffer)).unwrap());
        start = end;
    }
    let drain = block_on(encoder.finish()).unwrap();
    encoded.extend(drain.samples);
    let mut muxer = block_on(Mp4Muxer::new(
        MemorySink::new(),
        vec![Mp4TrackConfig {
            encoder: encoder.config().clone(),
            format: Mp4TrackFormat::Audio { channels },
        }],
        1_000_000,
    ))
    .unwrap();
    for sample in &encoded {
        block_on(muxer.write_sample(0, sample.clone())).unwrap();
    }
    muxer.set_audio_gapless(0, drain.gapless).unwrap();
    block_on(muxer.finish()).unwrap().into_inner()
}

#[test]
fn encoded_opus_round_trips_through_mp4_with_its_exact_length() {
    let frames = 48_000;
    let input = test_signal(frames);
    // An odd buffer size, so frames straddle buffers and the drain codes a
    // partial frame.
    let bytes = encode_to_mp4(&input, 2, 1_001, Some(128_000));
    let (mut reader, head) = open_reader(bytes);
    assert_eq!(head.channels, 2);
    assert_eq!(reader.presentation_length(), frames as u64);
    let decoded = read_all(&mut reader);
    let snr = snr_db(&input, &decoded);
    assert!(snr > 15.0, "round-trip SNR is only {snr:.1} dB");
}

#[test]
fn an_impulse_decodes_at_the_sample_it_was_encoded_at() {
    let frames = 24_000;
    for at in [0, 311, 312, 960, 10_007, frames - 1] {
        let mut input = vec![0_f32; frames];
        input[at] = 0.9;
        let bytes = encode_to_mp4(&input, 1, 480, None);
        let (mut reader, _) = open_reader(bytes);
        assert_eq!(reader.presentation_length(), frames as u64);
        let decoded = read_all(&mut reader);
        let peak = decoded
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        assert!(
            peak.abs_diff(at) <= 1,
            "an impulse encoded at {at} decoded with its peak at {peak}"
        );
    }
}

#[test]
fn encoder_capability_rejects_what_it_cannot_encode() {
    let factory = native_opus_audio_encoder_factory();
    let base = AudioEncoderConfig {
        codec: Codec::Opus,
        profile: CodecProfile::Opus,
        sample_rate: OPUS_SAMPLE_RATE,
        channels: 2,
        timescale: OPUS_SAMPLE_RATE,
        configuration: Vec::new(),
    };
    assert_eq!(
        factory.capability(&base),
        CodecSupport::Supported {
            implementation: zvidlib_core::CodecImplementation::Software
        }
    );
    let cases = [
        AudioEncoderConfig {
            codec: Codec::Aac,
            ..base.clone()
        },
        AudioEncoderConfig {
            profile: CodecProfile::AacLowComplexity,
            ..base.clone()
        },
        AudioEncoderConfig {
            sample_rate: 44_100,
            timescale: 44_100,
            ..base.clone()
        },
        AudioEncoderConfig {
            channels: 3,
            ..base.clone()
        },
        AudioEncoderConfig {
            timescale: 90_000,
            ..base.clone()
        },
        AudioEncoderConfig {
            configuration: 1_000_u32.to_be_bytes().to_vec(),
            ..base.clone()
        },
        AudioEncoderConfig {
            configuration: vec![1, 2],
            ..base
        },
    ];
    for case in cases {
        assert!(
            !factory.capability(&case).is_supported(),
            "{case:?} should be refused"
        );
        assert!(factory.create(&case, &Limits::default()).is_err());
    }
}

fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

/// FFmpeg's libopus decode of a zvidlib MP4 lines up with the input. FFmpeg is
/// an optional oracle here, as in `mp4_cover_art`: without it the test skips.
#[test]
fn ffmpeg_decodes_the_native_encoders_mp4() {
    if !tool_available("ffmpeg") {
        eprintln!("ffmpeg is not installed; skipping");
        return;
    }
    let frames = 48_000;
    let input = test_signal(frames);
    let bytes = encode_to_mp4(&input, 2, 960, Some(128_000));
    let path = std::env::temp_dir().join(format!("zvidlib-opus-{}.mp4", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-f", "f32le", "-"])
        .output()
        .unwrap();
    std::fs::remove_file(&path).ok();
    assert!(
        output.status.success(),
        "ffmpeg rejected the MP4: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let decoded: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    // FFmpeg trims the pre-skip; whether it also trims the end padding
    // depends on its version, so only the encoded length is compared.
    assert!(
        decoded.len() >= input.len(),
        "ffmpeg decoded too few samples"
    );
    let snr = snr_db(&input, &decoded[..input.len()]);
    assert!(
        snr > 15.0,
        "ffmpeg's decode is only {snr:.1} dB from the input"
    );
}

// --- WebM ------------------------------------------------------------------------

/// An exact-sample reader over a WebM's Opus track, as a caller builds one.
fn open_webm_reader(bytes: Vec<u8>) -> (AudioSampleReader<NativeOpusDecoder>, OpusHead) {
    let source = MemorySource::new(bytes);
    let webm = block_on(WebmDemuxer::open(&source, WebmDemuxerOptions::default())).unwrap();
    let track = webm
        .tracks
        .iter()
        .find(|track| track.kind == TrackKind::Audio)
        .expect("an audio track");
    assert_eq!(track.codec, Codec::Opus);
    let head = track.opus_config().unwrap();
    let packets = block_on(track.to_encoded_audio_samples(&source, &Limits::default())).unwrap();
    let timing = webm.audio_timing(track.id).unwrap();
    let preroll = opus_preroll_packets(&packets);
    let decoder = NativeOpusDecoder::new(&head, Limits::default()).unwrap();
    let reader = AudioSampleReader::new(
        decoder,
        packets,
        OPUS_SAMPLE_RATE,
        u16::from(head.channels),
        timing,
        preroll,
        Limits::default(),
    )
    .unwrap();
    (reader, head)
}

/// FFmpeg's libopus WebM declares the pre-skip as `CodecDelay` and the end
/// trim as the last block's `DiscardPadding`; read through both, it decodes
/// to exactly libopus's output.
#[test]
fn a_libopus_webm_decodes_as_libopus_does() {
    let bytes = std::fs::read(fixture("opus_stereo.webm")).unwrap();
    let (mut reader, head) = open_webm_reader(bytes);
    assert_eq!(head.channels, 2);
    assert_eq!(reader.presentation_length(), 24_000);
    let decoded = to_s16(&read_all(&mut reader));
    let reference = read_s16(&fixture("opus_stereo_webm_libopus.s16"));
    assert_eq!(decoded.len(), reference.len());
    let largest_difference = decoded
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        largest_difference <= 2.0,
        "decoded samples differ from libopus's by up to {largest_difference}"
    );
}

/// Encodes interleaved `samples` with the native encoder into a WebM.
fn encode_to_webm(samples: &[f32], channels: u16, chunk: usize) -> Vec<u8> {
    let factory = native_opus_audio_encoder_factory();
    let configuration = AudioEncoderConfig {
        codec: Codec::Opus,
        profile: CodecProfile::Opus,
        sample_rate: OPUS_SAMPLE_RATE,
        channels,
        timescale: OPUS_SAMPLE_RATE,
        configuration: Vec::new(),
    };
    let mut encoder = factory.create(&configuration, &Limits::default()).unwrap();
    let frames = samples.len() / usize::from(channels);
    let mut muxer = block_on(WebmMuxer::new(
        MemorySink::new(),
        vec![Mp4TrackConfig {
            encoder: encoder.config().clone(),
            format: Mp4TrackFormat::Audio { channels },
        }],
        1_000_000,
    ))
    .unwrap();
    let mut start = 0;
    while start < frames {
        let end = (start + chunk).min(frames);
        let buffer = AudioBuffer::new(
            SampleRange::new(start as u64, end as u64).unwrap(),
            OPUS_SAMPLE_RATE,
            channels,
            samples[start * usize::from(channels)..end * usize::from(channels)].to_vec(),
            &Limits::default(),
        )
        .unwrap();
        for sample in block_on(encoder.encode(FrameIndex(0), buffer)).unwrap() {
            block_on(muxer.write_sample(0, sample)).unwrap();
        }
        start = end;
    }
    let drain = block_on(encoder.finish()).unwrap();
    for sample in drain.samples {
        block_on(muxer.write_sample(0, sample)).unwrap();
    }
    muxer.set_audio_gapless(0, drain.gapless).unwrap();
    block_on(muxer.finish()).unwrap().into_inner()
}

#[test]
fn encoded_opus_round_trips_through_webm_with_its_exact_length() {
    let frames = 48_000;
    let input = test_signal(frames);
    let bytes = encode_to_webm(&input, 2, 1_001);
    let (mut reader, head) = open_webm_reader(bytes);
    assert_eq!(head.channels, 2);
    assert_eq!(reader.presentation_length(), frames as u64);
    let decoded = read_all(&mut reader);
    let snr = snr_db(&input, &decoded);
    assert!(snr > 15.0, "round-trip SNR is only {snr:.1} dB");
}

/// FFmpeg reads zvidlib's Opus WebM with its trims applied. FFmpeg is an
/// optional oracle: without it the test skips.
#[test]
fn ffmpeg_decodes_the_native_encoders_webm_to_its_exact_length() {
    if !tool_available("ffmpeg") {
        eprintln!("ffmpeg is not installed; skipping");
        return;
    }
    let frames = 48_000;
    let input = test_signal(frames);
    let bytes = encode_to_webm(&input, 2, 960);
    let path = std::env::temp_dir().join(format!("zvidlib-opus-{}.webm", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-c:a", "libopus", "-i"])
        .arg(&path)
        .args(["-f", "f32le", "-"])
        .output()
        .unwrap();
    std::fs::remove_file(&path).ok();
    if !output.status.success() {
        // An FFmpeg built without libopus; its own Opus decoder honours
        // the same fields.
        eprintln!(
            "ffmpeg could not use libopus: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let decoded: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    assert_eq!(
        decoded.len(),
        input.len(),
        "ffmpeg did not apply both trims"
    );
    let snr = snr_db(&input, &decoded);
    assert!(
        snr > 15.0,
        "ffmpeg's decode is only {snr:.1} dB from the input"
    );
}

/// An Opus track interleaved with video: the audio blocks the muxer holds
/// back still land in presentation order, only the video is cued, and the
/// audio reads back with exactly its length.
#[test]
fn opus_interleaves_with_video_in_webm() {
    use zvidlib_core::{EncodedSample, EncoderConfig, SampleDependency, VideoDimensions};
    let frames = 48_000;
    let input = test_signal(frames);
    let factory = native_opus_audio_encoder_factory();
    let mut encoder = factory
        .create(
            &AudioEncoderConfig {
                codec: Codec::Opus,
                profile: CodecProfile::Opus,
                sample_rate: OPUS_SAMPLE_RATE,
                channels: 2,
                timescale: OPUS_SAMPLE_RATE,
                configuration: Vec::new(),
            },
            &Limits::default(),
        )
        .unwrap();
    let buffer = AudioBuffer::new(
        SampleRange::new(0, frames as u64).unwrap(),
        OPUS_SAMPLE_RATE,
        2,
        input.clone(),
        &Limits::default(),
    )
    .unwrap();
    let mut audio = block_on(encoder.encode(FrameIndex(0), buffer)).unwrap();
    let drain = block_on(encoder.finish()).unwrap();
    audio.extend(drain.samples);

    // A 30 fps video track of placeholder AV1 samples, a key frame a second.
    let mut av1c = 12_u32.to_be_bytes().to_vec();
    av1c.extend_from_slice(b"av1C");
    av1c.extend_from_slice(&[0x81, 0, 0, 0]);
    let video: Vec<EncodedSample> = (0..30)
        .map(|index| EncodedSample {
            data: vec![index as u8 + 1],
            dts: index,
            pts: index,
            duration: 1,
            is_sync: index % 30 == 0,
            dependency: if index % 30 == 0 {
                SampleDependency::INDEPENDENT
            } else {
                SampleDependency::DEPENDENT
            },
        })
        .collect();
    let mut muxer = block_on(WebmMuxer::new(
        MemorySink::new(),
        vec![
            Mp4TrackConfig {
                encoder: EncoderConfig {
                    codec: Codec::Av1,
                    timescale: 30,
                    decoder_config: av1c,
                },
                format: Mp4TrackFormat::Video(
                    VideoDimensions::new(16, 16, &Limits::default()).unwrap(),
                ),
            },
            Mp4TrackConfig {
                encoder: encoder.config().clone(),
                format: Mp4TrackFormat::Audio { channels: 2 },
            },
        ],
        1_000_000,
    ))
    .unwrap();
    // Write both tracks in presentation order, as a recorder does.
    let mut video = video.into_iter().peekable();
    let mut audio = audio.into_iter().peekable();
    loop {
        let video_ms = video.peek().map(|sample| sample.pts * 1_000 / 30);
        let audio_ms = audio.peek().map(|sample| sample.pts * 1_000 / 48_000);
        match (video_ms, audio_ms) {
            (Some(v), Some(a)) if v <= a => {
                block_on(muxer.write_sample(0, video.next().unwrap())).unwrap()
            }
            (_, Some(_)) => block_on(muxer.write_sample(1, audio.next().unwrap())).unwrap(),
            (Some(_), None) => block_on(muxer.write_sample(0, video.next().unwrap())).unwrap(),
            (None, None) => break,
        }
    }
    muxer.set_audio_gapless(1, drain.gapless).unwrap();
    let bytes = block_on(muxer.finish()).unwrap().into_inner();

    let source = MemorySource::new(bytes.clone());
    let webm = block_on(WebmDemuxer::open(&source, WebmDemuxerOptions::default())).unwrap();
    assert_eq!(webm.tracks.len(), 2);
    assert_eq!(webm.tracks[0].samples.len(), 30);
    assert!(webm.cues.iter().all(|cue| cue.track == webm.tracks[0].id));
    let (mut reader, _) = open_webm_reader(bytes);
    assert_eq!(reader.presentation_length(), frames as u64);
    let snr = snr_db(&input, &read_all(&mut reader));
    assert!(snr > 15.0, "round-trip SNR is only {snr:.1} dB");
}

// --- RFC 8251 test vectors ---------------------------------------------------

/// Runs the twelve official Opus decoder test vectors
/// (`opus_testvectors-rfc8251.tar.gz` from opus-codec.org) through
/// [`NativeOpusDecoder`] at 48 kHz in stereo and in mono, comparing each with
/// `opus_compare` as RFC 8251 does. The vectors are 75 MB, so they are not
/// committed: CI downloads them and points `ZVIDLIB_OPUS_TEST_VECTORS` at the
/// directory, and the test skips where that variable is unset.
#[test]
fn rfc8251_test_vectors_pass_opus_compare() {
    let Some(directory) = std::env::var_os("ZVIDLIB_OPUS_TEST_VECTORS") else {
        eprintln!("ZVIDLIB_OPUS_TEST_VECTORS is unset; skipping the RFC 8251 vectors");
        return;
    };
    let directory = PathBuf::from(directory);
    for vector in 1..=12 {
        let bitstream = std::fs::read(directory.join(format!("testvector{vector:02}.bit")))
            .unwrap_or_else(|error| panic!("reading test vector {vector}: {error}"));
        let mut packets = Vec::new();
        let mut at = 0;
        while at + 8 <= bitstream.len() {
            let length = u32::from_be_bytes(bitstream[at..at + 4].try_into().unwrap()) as usize;
            packets.push(bitstream[at + 8..at + 8 + length].to_vec());
            at += 8 + length;
        }
        let references = [
            read_s16(&directory.join(format!("testvector{vector:02}.dec"))),
            read_s16(&directory.join(format!("testvector{vector:02}m.dec"))),
        ];
        for channels in [2_u8, 1] {
            let head = OpusHead::new(channels, 0, OPUS_SAMPLE_RATE).unwrap();
            let mut decoder = NativeOpusDecoder::new(&head, Limits::default()).unwrap();
            let cancellation = CancellationToken::new();
            let mut decoded = Vec::new();
            let mut start = 0;
            let mut previous_length = 960;
            for data in &packets {
                // A lost packet is an empty one, concealed for as long as the
                // packet before it lasted.
                let length = if data.is_empty() {
                    previous_length
                } else {
                    u64::from(opus_packet_samples(data).unwrap())
                };
                previous_length = length;
                let packet = EncodedAudioSample {
                    decoded_range: SampleRange::new(start, start + length).unwrap(),
                    data: data.clone(),
                };
                decoded.extend(to_s16(
                    &decoder.decode(&packet, &cancellation).unwrap().samples,
                ));
                start += length;
            }
            // Either reference is acceptable (RFC 8251 section 6).
            let quality = references
                .iter()
                .map(|reference| opus_compare(reference, &decoded, usize::from(channels)))
                .fold(f64::NEG_INFINITY, f64::max);
            eprintln!("test vector {vector:02}, {channels} channel(s): quality {quality:.1}%");
            assert!(
                quality >= 0.0,
                "test vector {vector:02} fails decoded to {channels} channel(s)"
            );
        }
    }
}
