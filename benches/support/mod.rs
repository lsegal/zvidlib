//! The root bench targets' support: the bundled sample's AAC track, the
//! Vorbis fixtures, and the paired HEVC cadence tracks, each demuxed once per
//! process, on top of `zvidlib-bench-support`'s shared helpers.
//!
//! These are the targets that measure zvidlib as a whole rather than one
//! codec: the audio read and write paths, exact seeking, and the VP8 and VP9
//! decoders side by side. Each codec's own targets live in its crate.

// Each bench target compiles this whole module, but uses only the fixtures its
// own measurements need. Unused-here is not dead: `cargo clippy --all-targets`
// would otherwise fail one target for helpers another target depends on.
#![allow(dead_code)]

use std::sync::OnceLock;

use zvidlib::io::MemorySource;
use zvidlib::{
    AacTrackConfig, AudioTrackTiming, Codec, CodecProfile, ColorRange, EncodedAudioSample,
    EncodedVideoSample, Limits, Mp4Demuxer, Mp4DemuxerOptions, Mp4Track, PixelFormat, TrackKind,
    VideoDecoderConfig, VorbisConfig,
};

zvidlib_bench_support::bench_support!(zvidlib::simd::active_by_site);

/// The bundled sample's AAC track, demuxed once per process.
///
/// `examples/media/BigBuckBunny.mp4` is the only real audio fixture checked into
/// the repository: 1,501 AAC-LC access units at 48 kHz stereo covering 1,536,000
/// decoded samples (32 s), with a one-edit edit list that produces 1,024 samples
/// of decoder priming. That makes it the read-side counterpart to the synthetic
/// write-path fixtures — real packet sizes, a real sample table, and real
/// gapless timing rather than a uniform synthetic track.
pub struct BundledAacTrack {
    pub source: MemorySource,
    pub movie: Mp4Demuxer,
    pub track_index: usize,
    /// The parsed `esds` configuration, which is what a decoder is built from.
    pub config: AacTrackConfig,
    pub sample_rate: u32,
    pub channels: u16,
    /// Decoded samples per channel the whole track covers.
    pub decoded_samples: u64,
    pub packets: Vec<EncodedAudioSample>,
    pub timing: AudioTrackTiming,
}

impl BundledAacTrack {
    pub fn track(&self) -> &Mp4Track {
        &self.movie.tracks[self.track_index]
    }

    /// The work the whole track represents, for throughput reporting.
    pub fn work(&self) -> AudioWork {
        AudioWork::new(self.decoded_samples, self.sample_rate, self.channels)
    }

    /// The first `count` access units, or all of them when the track is shorter.
    pub fn prefix(&self, count: usize) -> &[EncodedAudioSample] {
        &self.packets[..count.min(self.packets.len())]
    }

    /// Decoded samples covered by [`BundledAacTrack::prefix`].
    pub fn prefix_samples(&self, count: usize) -> u64 {
        self.prefix(count)
            .last()
            .map_or(0, |packet| packet.decoded_range.end)
    }
}

/// The bundled sample's bytes, held once so `Mp4Demuxer::open` can be timed
/// without re-reading the file.
pub fn bundled_mp4_bytes() -> &'static [u8] {
    include_bytes!("../../examples/media/BigBuckBunny.mp4")
}

/// Demuxes the single AAC track out of an in-memory MP4.
fn demux_aac_track(bytes: Vec<u8>, label: &str) -> BundledAacTrack {
    let limits = Limits::default();
    let source = MemorySource::new(bytes);
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))
        .unwrap_or_else(|error| panic!("{label} is a readable MP4: {error}"));
    let track_index = movie
        .tracks
        .iter()
        .position(|track| track.kind == TrackKind::Audio && track.codec == Codec::Aac)
        .unwrap_or_else(|| panic!("{label} has an AAC audio track"));
    let track = &movie.tracks[track_index];
    let config = track
        .aac_config()
        .unwrap_or_else(|error| panic!("{label} carries an AAC AudioSpecificConfig: {error}"));
    let packets = block_on(track.to_encoded_audio_samples(&source, &limits))
        .unwrap_or_else(|error| panic!("{label}'s AAC access units are readable: {error}"));
    let timing = track
        .audio_timing(movie.movie_timescale)
        .unwrap_or_else(|error| panic!("{label}'s edit list maps to the sample clock: {error}"));
    let decoded_samples = packets
        .last()
        .unwrap_or_else(|| panic!("{label}'s AAC track is not empty"))
        .decoded_range
        .end;
    BundledAacTrack {
        sample_rate: config.sample_rate,
        channels: config.channels,
        config,
        decoded_samples,
        packets,
        timing,
        track_index,
        movie,
        source,
    }
}

/// Demuxes the bundled sample's AAC track once per process.
pub fn bundled_aac_track() -> &'static BundledAacTrack {
    static TRACK: OnceLock<BundledAacTrack> = OnceLock::new();
    TRACK.get_or_init(|| {
        demux_aac_track(
            bundled_mp4_bytes().to_vec(),
            "the bundled BigBuckBunny sample",
        )
    })
}

/// A mono 48 kHz AAC-LC fixture with a real priming edit list.
///
/// The bundled sample is stereo, but `NativeAacDecoder` accepts AAC-LC mono as
/// well and rejects everything beyond stereo, so mono is the other half of the
/// backend's entire supported input space.
pub fn aac_mono_track() -> &'static BundledAacTrack {
    static TRACK: OnceLock<BundledAacTrack> = OnceLock::new();
    TRACK.get_or_init(|| {
        demux_aac_track(
            include_bytes!("../../tests/fixtures/codec/aac_lc_mono_48k.m4a").to_vec(),
            "the mono AAC-LC fixture",
        )
    })
}

/// A libvorbis-encoded Vorbis fixture, split into packets once per process.
///
/// The fixtures are the ones `tests/vorbis_codec.rs` checks the decoder against
/// libvorbis's own decode with, so a Vorbis benchmark times the same streams
/// the decoder is held correct on.
pub struct VorbisFixture {
    pub config: VorbisConfig,
    pub packets: Vec<EncodedAudioSample>,
    pub sample_rate: u32,
    pub channels: u16,
    /// Decoded samples per channel the whole stream covers.
    pub decoded_samples: u64,
}

impl VorbisFixture {
    /// The work the whole stream represents, for throughput reporting.
    pub fn work(&self) -> AudioWork {
        AudioWork::new(self.decoded_samples, self.sample_rate, self.channels)
    }

    fn parse(bytes: &[u8], label: &str) -> VorbisFixture {
        let mut packets = ogg_packets(bytes);
        assert!(packets.len() > 3, "{label} has audio packets");
        let audio = packets.split_off(3);
        let [identification, comment, setup]: [Vec<u8>; 3] = packets
            .try_into()
            .unwrap_or_else(|_| panic!("{label} starts with three header packets"));
        let config = VorbisConfig::from_headers(identification, comment, setup)
            .unwrap_or_else(|error| panic!("{label}'s headers parse: {error}"));
        let packets = config
            .encoded_samples(audio)
            .unwrap_or_else(|error| panic!("{label}'s packets index: {error}"));
        let decoded_samples = packets.last().map_or(0, |packet| packet.decoded_range.end);
        VorbisFixture {
            sample_rate: config.sample_rate,
            channels: u16::from(config.channels),
            config,
            packets,
            decoded_samples,
        }
    }
}

/// The packets of a single-stream Ogg file (RFC 3533), headers included.
fn ogg_packets(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut partial = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        assert_eq!(&bytes[at..at + 4], b"OggS", "an Ogg page at {at}");
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
    packets
}

/// Half a second of 44.1 kHz stereo music, which libvorbis codes in long
/// blocks with square-polar coupling: the common case.
pub fn vorbis_stereo_fixture() -> &'static VorbisFixture {
    static FIXTURE: OnceLock<VorbisFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        VorbisFixture::parse(
            include_bytes!(
                "../../crates/zvidlib-vorbis-decoder/tests/fixtures/vorbis_stereo_44k.ogg"
            ),
            "the stereo Vorbis fixture",
        )
    })
}

/// A quarter second of 48 kHz 5.1, whose mapping couples one channel in
/// several steps, so coupling and synthesis run over six channels a packet.
pub fn vorbis_surround_fixture() -> &'static VorbisFixture {
    static FIXTURE: OnceLock<VorbisFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        VorbisFixture::parse(
            include_bytes!("../../crates/zvidlib-vorbis-decoder/tests/fixtures/vorbis_6ch_48k.ogg"),
            "the 5.1 Vorbis fixture",
        )
    })
}

/// One of the paired 512x288 HEVC tracks that differ only in keyframe cadence.
pub struct GopCadenceTrack {
    /// `raps=1` or `raps=24`, the arm name a benchmark reports under.
    pub label: &'static str,
    pub configuration: VideoDecoderConfig,
    pub samples: Vec<EncodedVideoSample>,
    pub width: u64,
    pub height: u64,
    /// How many of `samples` are random-access points.
    pub random_access_points: usize,
}

/// The two tracks whose *only* difference is how often a decode may restart.
///
/// The bundled 1080p sample answers what an exact seek costs on real content,
/// but it codes its 768 frames as one group of pictures, so it cannot say
/// anything about a track with several random-access points. Two tracks encoded
/// from the same source at the same size, quality and preset, differing only in
/// `keyint`, can: any gap between them is the cadence and nothing else. See
/// `tests/fixtures/codec/README.md` for how they were produced.
pub fn gop_cadence_tracks() -> &'static [GopCadenceTrack; 2] {
    static TRACKS: OnceLock<[GopCadenceTrack; 2]> = OnceLock::new();
    TRACKS.get_or_init(|| {
        [
            gop_cadence_track(
                "raps=1",
                include_bytes!(
                    "../../crates/zvidlib-hevc-decoder/tests/fixtures/bbb_hevc_512x288_gop768.mp4"
                )
                .to_vec(),
            ),
            gop_cadence_track(
                "raps=24",
                include_bytes!(
                    "../../crates/zvidlib-hevc-decoder/tests/fixtures/bbb_hevc_512x288_gop32.mp4"
                )
                .to_vec(),
            ),
        ]
    })
}

fn gop_cadence_track(label: &'static str, bytes: Vec<u8>) -> GopCadenceTrack {
    let limits = Limits::default();
    let source = MemorySource::new(bytes);
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))
        .expect("the gop-cadence fixture is a readable MP4");
    let track = movie
        .tracks
        .iter()
        .find(|track| track.kind == TrackKind::Video)
        .expect("the gop-cadence fixture has a video track");
    let dimensions = track
        .dimensions
        .expect("the gop-cadence fixture's video track is dimensioned");
    let samples = block_on(track.to_encoded_video_samples(&source, &limits))
        .expect("the gop-cadence fixture's video samples are readable");
    let random_access_points = samples.iter().filter(|sample| sample.random_access).count();
    GopCadenceTrack {
        label,
        configuration: VideoDecoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: dimensions,
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: zvidlib::HardwarePreference::Avoid,
            configuration: track.decoder_config.clone(),
        },
        samples,
        width: u64::from(dimensions.width),
        height: u64::from(dimensions.height),
        random_access_points,
    }
}
