//! Vorbis decode against libvorbis, its configuration, and sample-accurate
//! reads.
//!
//! The fixtures under `tests/fixtures/codec/vorbis_*` were encoded by libvorbis
//! into Ogg, and the `*_libvorbis.s16` references are libvorbis's own decode of
//! them through `vorbisfile`, trimmed to the stream's granule positions (see
//! that directory's README). zvidlib has no Ogg container, so the test reads
//! the pages itself.
//!
//! The native encoder's packets are pinned bit for bit to libvorbis's by the
//! unit tests in `src/vorbis_encoder/`; here they are read back through the
//! decoder the way a demuxed track would be.

use std::path::{Path, PathBuf};

use zvidlib::io::MemorySink;
use zvidlib::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::{
    AudioBuffer, AudioEncoderConfig, AudioEncoderFactory, AudioSampleReader, AudioTrackTiming,
    CancellationToken, Codec, CodecProfile, CodecSupport, EncodedSample, FrameIndex, Limits,
    NativeVorbisDecoder, SampleRange, VORBIS_PREROLL_PACKETS, VorbisConfig,
    native_vorbis_audio_encoder_factory,
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/codec")
        .join(name)
}

fn read_s16(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])))
        .collect()
}

fn to_s16(samples: &[f32]) -> Vec<f32> {
    samples
        .iter()
        .map(|value| (value * 32_768.0).round().clamp(-32_768.0, 32_767.0))
        .collect()
}

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

/// A Vorbis stream's configuration and an exact-sample reader over its audio,
/// trimmed to its last granule position the way an Ogg demuxer would.
fn open(name: &str) -> (VorbisConfig, AudioSampleReader<NativeVorbisDecoder>) {
    let (mut packets, granule) = ogg_packets(&std::fs::read(fixture(name)).unwrap());
    let audio = packets.split_off(3);
    let [identification, comment, setup]: [Vec<u8>; 3] = packets.try_into().unwrap();
    let config = VorbisConfig::from_headers(identification, comment, setup).unwrap();
    let samples = config.encoded_samples(audio).unwrap();
    let decoded = samples.last().unwrap().decoded_range.end;
    let timing = AudioTrackTiming {
        padding: u32::try_from(decoded - granule).unwrap(),
        ..AudioTrackTiming::default()
    };
    let decoder = NativeVorbisDecoder::new(&config, Limits::default()).unwrap();
    let reader = AudioSampleReader::new(
        decoder,
        samples,
        config.sample_rate,
        u16::from(config.channels),
        timing,
        VORBIS_PREROLL_PACKETS,
        Limits::default(),
    )
    .unwrap();
    (config, reader)
}

fn read_all(reader: &mut AudioSampleReader<NativeVorbisDecoder>) -> Vec<f32> {
    let range = SampleRange::new(0, reader.presentation_length()).unwrap();
    reader
        .get_range(range, &CancellationToken::new())
        .unwrap()
        .samples
}

fn assert_matches_libvorbis(name: &str, channels: u8, sample_rate: u32, frames: u64) {
    let (config, mut reader) = open(&format!("{name}.ogg"));
    assert_eq!(config.channels, channels);
    assert_eq!(config.sample_rate, sample_rate);
    assert_eq!(reader.presentation_length(), frames);
    let decoded = to_s16(&read_all(&mut reader));
    let reference = read_s16(&fixture(&format!("{name}_libvorbis.s16")));
    assert_eq!(decoded.len(), reference.len());
    let largest_difference = decoded
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    // Both decoders compute in floating point, so they may round a sample
    // either side of a 16-bit step.
    assert!(
        largest_difference <= 1.0,
        "{name}: decoded samples differ from libvorbis's by up to {largest_difference}"
    );
}

#[test]
fn stereo_music_decodes_as_libvorbis_does() {
    assert_matches_libvorbis("vorbis_stereo_44k", 2, 44_100, 22_050);
}

#[test]
fn transients_in_short_blocks_decode_as_libvorbis_does() {
    let (config, _) = open("vorbis_transient_mono.ogg");
    let (packets, _) = ogg_packets(&std::fs::read(fixture("vorbis_transient_mono.ogg")).unwrap());
    let blocks: Vec<u16> = packets[3..]
        .iter()
        .map(|packet| config.packet_block_size(packet).unwrap())
        .collect();
    // The clicks make the encoder switch to short blocks, so the variable
    // packet durations are exercised.
    assert!(blocks.contains(&config.block_sizes.0), "{blocks:?}");
    assert!(blocks.contains(&config.block_sizes.1), "{blocks:?}");
    assert_matches_libvorbis("vorbis_transient_mono", 1, 48_000, 24_000);
}

#[test]
fn surround_configuration_parses_but_is_not_decoded() {
    // A 5.1 stream's setup header has several coupling steps and submaps, all
    // of which the configuration walk must get through to reach its modes.
    let (mut packets, _) = ogg_packets(&std::fs::read(fixture("vorbis_6ch_16k.ogg")).unwrap());
    let audio = packets.split_off(3);
    let [identification, comment, setup]: [Vec<u8>; 3] = packets.try_into().unwrap();
    let config = VorbisConfig::from_headers(identification, comment, setup).unwrap();
    assert_eq!(config.channels, 6);
    let samples = config.encoded_samples(audio).unwrap();
    assert!(samples.last().unwrap().decoded_range.end >= 4_000);
    // The decoder underneath decodes some surround streams wrongly, so it
    // refuses them rather than return the wrong audio.
    let error = NativeVorbisDecoder::new(&config, Limits::default())
        .err()
        .unwrap();
    assert_eq!(error.kind(), zvidlib::ErrorKind::Unsupported);
}

#[test]
fn reads_after_seeking_return_exactly_the_continuous_decode() {
    for name in ["vorbis_stereo_44k.ogg", "vorbis_transient_mono.ogg"] {
        let (config, mut continuous) = open(name);
        let channels = usize::from(config.channels);
        let full = read_all(&mut continuous);
        let length = continuous.presentation_length();
        let (_, mut reader) = open(name);
        let cancellation = CancellationToken::new();
        for (start, end) in [
            (13_001, 13_002),
            (9_000, 12_345),
            (100, 2_000),
            (0, 64),
            (length - 700, length),
            (2_047, 2_049),
            (1_000, length - 1_000),
        ] {
            let buffer = reader
                .get_range(SampleRange::new(start, end).unwrap(), &cancellation)
                .unwrap();
            // A Vorbis packet depends only on the one before it, which the
            // preroll decodes, so a read after a seek is bit-identical.
            assert_eq!(
                buffer.samples,
                full[start as usize * channels..end as usize * channels],
                "{name} [{start}, {end})"
            );
        }
    }
}

#[test]
fn codec_private_round_trips_the_three_headers() {
    let (config, _) = open("vorbis_stereo_44k.ogg");
    let codec_private = config.to_codec_private();
    assert_eq!(codec_private[0], 2);
    assert_eq!(
        VorbisConfig::from_codec_private(&codec_private).unwrap(),
        config
    );
    assert!(config.bitrate_nominal > 0);
    assert_eq!(config.block_sizes, (256, 2048));
}

#[test]
fn malformed_configuration_is_rejected() {
    let (config, _) = open("vorbis_stereo_44k.ogg");
    let codec_private = config.to_codec_private();
    for truncated in [0, 1, 3, codec_private.len() / 2, codec_private.len() - 1] {
        assert!(
            VorbisConfig::from_codec_private(&codec_private[..truncated]).is_err(),
            "{truncated} bytes"
        );
    }
    let mut wrong_type = config.identification_header.clone();
    wrong_type[0] = 3;
    assert!(
        VorbisConfig::from_headers(
            wrong_type,
            config.comment_header.clone(),
            config.setup_header.clone()
        )
        .is_err()
    );
    // An audio packet must start with a zero bit.
    assert!(config.packet_block_size(&[0x01]).is_err());
}

// --- the native encoder -------------------------------------------------------

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
    }
}

fn encoder_config(sample_rate: u32, channels: u16, configuration: Vec<u8>) -> AudioEncoderConfig {
    AudioEncoderConfig {
        codec: Codec::Vorbis,
        profile: CodecProfile::Vorbis,
        sample_rate,
        channels,
        timescale: sample_rate,
        configuration,
    }
}

/// Encodes interleaved `input` in buffers of `chunk` frames, then reads the
/// stream back through `NativeVorbisDecoder` exactly as a demuxed Matroska or
/// WebM track would be read: the `CodecPrivate`, each packet's interval, and
/// the drained end padding.
fn round_trip(
    input: &[f32],
    sample_rate: u32,
    channels: u16,
    chunk: usize,
    configuration: Vec<u8>,
) -> (Vec<f32>, u64, Vec<EncodedSample>) {
    let factory = native_vorbis_audio_encoder_factory();
    let config = encoder_config(sample_rate, channels, configuration);
    assert!(factory.capability(&config).is_supported());
    let mut encoder = factory.create(&config, &Limits::default()).unwrap();
    assert_eq!(encoder.config().codec, Codec::Vorbis);
    let frames = input.len() / usize::from(channels);
    let mut encoded = Vec::new();
    let mut start = 0;
    while start < frames {
        let end = (start + chunk).min(frames);
        let buffer = AudioBuffer::new(
            SampleRange::new(start as u64, end as u64).unwrap(),
            sample_rate,
            channels,
            input[start * usize::from(channels)..end * usize::from(channels)].to_vec(),
            &Limits::default(),
        )
        .unwrap();
        encoded.extend(block_on(encoder.encode(FrameIndex(0), buffer)).unwrap());
        start = end;
    }
    let drain = block_on(encoder.finish()).unwrap();
    encoded.extend(drain.samples);
    assert_eq!(drain.gapless.priming, 0);

    let stream = VorbisConfig::from_codec_private(&encoder.config().decoder_config).unwrap();
    assert_eq!(stream.sample_rate, sample_rate);
    assert_eq!(u16::from(stream.channels), channels);
    // The encoder's intervals are the ones the headers say the packets have.
    let packets = stream
        .encoded_samples(encoded.iter().map(|sample| sample.data.clone()).collect())
        .unwrap();
    for (packet, sample) in packets.iter().zip(&encoded) {
        assert_eq!(packet.decoded_range.start, sample.pts as u64);
        assert_eq!(packet.decoded_range.len(), u64::from(sample.duration));
    }
    let decoder = NativeVorbisDecoder::new(&stream, Limits::default()).unwrap();
    let mut reader = AudioSampleReader::new(
        decoder,
        packets,
        sample_rate,
        channels,
        AudioTrackTiming {
            padding: drain.gapless.padding,
            ..AudioTrackTiming::default()
        },
        VORBIS_PREROLL_PACKETS,
        Limits::default(),
    )
    .unwrap();
    let length = reader.presentation_length();
    let decoded = reader
        .get_range(
            SampleRange::new(0, length).unwrap(),
            &CancellationToken::new(),
        )
        .unwrap()
        .samples;
    (decoded, length, encoded)
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

fn music_like(frames: usize, sample_rate: u32) -> Vec<f32> {
    let mut seed = 0x2468_ace0_u32;
    (0..frames)
        .flat_map(|i| {
            let t = i as f32 / sample_rate as f32;
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 9) as f32 / (1 << 23) as f32 - 0.5;
            let tone = |f: f32| (2.0 * std::f32::consts::PI * f * t).sin();
            // A click every quarter second, so the encoder switches block
            // sizes as well.
            let click = if i % (sample_rate as usize / 4) < 40 {
                0.5 * noise
            } else {
                0.0
            };
            [
                0.25 * tone(220.0) + 0.15 * tone(1_100.0) + 0.02 * noise + click,
                0.25 * tone(330.0) + 0.1 * tone(2_500.0) + 0.02 * noise + click,
            ]
        })
        .collect()
}

#[test]
fn encoded_vorbis_round_trips_with_its_exact_length() {
    for (sample_rate, chunk) in [(44_100, 1_000), (48_000, 4_096), (22_050, 333)] {
        let frames = sample_rate as usize;
        let input = music_like(frames, sample_rate);
        let (decoded, length, encoded) = round_trip(&input, sample_rate, 2, chunk, Vec::new());
        assert_eq!(length, frames as u64, "{sample_rate} Hz");
        assert_eq!(decoded.len(), input.len());
        let snr = snr_db(&input, &decoded);
        assert!(
            snr > 20.0,
            "{sample_rate} Hz round trip is only {snr:.1} dB"
        );
        // The first packet primes the overlap and spans no samples.
        assert_eq!(encoded[0].duration, 0);
    }
}

#[test]
fn an_impulse_decodes_at_the_sample_it_was_encoded_at() {
    let frames = 20_000;
    for at in [0, 1, 1_023, 1_024, 7_777, frames - 1] {
        let mut input = vec![0_f32; frames];
        input[at] = 0.9;
        let (decoded, length, _) = round_trip(&input, 48_000, 1, 1_024, Vec::new());
        assert_eq!(length, frames as u64);
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
fn a_target_bit_rate_selects_the_quality_libvorbis_would() {
    let input = music_like(44_100, 44_100);
    let bits = |configuration: Vec<u8>| {
        let (_, _, encoded) = round_trip(&input, 44_100, 2, 4_096, configuration);
        encoded
            .iter()
            .map(|sample| sample.data.len() * 8)
            .sum::<usize>()
    };
    let low = bits(64_000_u32.to_be_bytes().to_vec());
    let high = bits(192_000_u32.to_be_bytes().to_vec());
    assert!(low < high, "64 kb/s used {low} bits, 192 kb/s {high}");
}

#[test]
fn encoder_capability_rejects_what_it_cannot_encode() {
    let factory = native_vorbis_audio_encoder_factory();
    let base = encoder_config(44_100, 2, Vec::new());
    assert_eq!(
        factory.capability(&base),
        CodecSupport::Supported {
            implementation: zvidlib::CodecImplementation::Software
        }
    );
    let cases = [
        AudioEncoderConfig {
            codec: Codec::Opus,
            ..base.clone()
        },
        AudioEncoderConfig {
            profile: CodecProfile::Opus,
            ..base.clone()
        },
        AudioEncoderConfig {
            channels: 6,
            ..base.clone()
        },
        AudioEncoderConfig {
            timescale: 48_000,
            ..base.clone()
        },
        AudioEncoderConfig {
            configuration: vec![1, 2, 3],
            ..base.clone()
        },
        AudioEncoderConfig {
            configuration: 1_u32.to_be_bytes().to_vec(),
            ..base.clone()
        },
        encoder_config(0, 2, Vec::new()),
    ];
    for case in cases {
        assert!(
            !factory.capability(&case).is_supported(),
            "{case:?} should be refused"
        );
        assert!(factory.create(&case, &Limits::default()).is_err());
    }
}

#[test]
fn mp4_refuses_a_vorbis_track() {
    let factory = native_vorbis_audio_encoder_factory();
    let encoder = factory
        .create(&encoder_config(44_100, 2, Vec::new()), &Limits::default())
        .unwrap();
    let result = block_on(Mp4Muxer::new(
        MemorySink::new(),
        vec![Mp4TrackConfig {
            encoder: encoder.config().clone(),
            format: Mp4TrackFormat::Audio { channels: 2 },
        }],
        1_000,
    ));
    let error = result.err().expect("Vorbis has no MP4 sample entry");
    assert_eq!(error.kind(), zvidlib::ErrorKind::Unsupported);
}
