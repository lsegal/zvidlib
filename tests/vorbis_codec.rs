//! Vorbis decode against libvorbis, its configuration, and sample-accurate
//! reads.
//!
//! The fixtures under `tests/fixtures/codec/vorbis_*` were encoded by libvorbis
//! into Ogg, and the `*_libvorbis.s16` references are libvorbis's own decode of
//! them through `vorbisfile`, trimmed to the stream's granule positions (see
//! that directory's README). zvidlib has no Ogg container, so the test reads
//! the pages itself.

use std::path::{Path, PathBuf};

use zvidlib::{
    AudioSampleReader, AudioTrackTiming, CancellationToken, Limits, NativeVorbisDecoder,
    SampleRange, VORBIS_PREROLL_PACKETS, VorbisConfig,
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
