//! Bench inputs, parsed once per process and then handed out by reference.
//!
//! Criterion re-runs a routine hundreds of times. Demuxing the bundled MP4 or
//! re-parsing an elementary stream inside the measured closure would bury the
//! codec work these benches exist to measure, so every fixture here resolves
//! through a `OnceLock` and the benches borrow the result.

use std::sync::OnceLock;

use zvidlib::io::MemorySource;
use zvidlib::{
    Codec, CodecProfile, ColorRange, EncodedVideoSample, HardwarePreference, Limits, Mp4Demuxer,
    Mp4DemuxerOptions, PixelFormat, VideoDecoderConfig,
};

use super::block_on;

/// The bundled 1920x1080 HEVC Main sample the examples play.
const BUNDLED_MP4: &[u8] = include_bytes!("../../examples/media/BigBuckBunny.mp4");

/// A lossless 17x9 AV1 intra temporal unit (low-overhead bitstream format).
const AV1_LOSSLESS_INTRA_HEX: &str =
    include_str!("../../tests/fixtures/codec/av1_lossless_17x9.hex");

/// A 16x16 AV1 key/inter/compound/`show_existing_frame` sequence.
const AV1_INTER_HEX: &str =
    include_str!("../../tests/fixtures/codec/av1_inter_show_existing_16x16.hex");

/// A demuxed video track, ready to feed a decoder.
pub struct VideoFixture {
    /// Decoder configuration, including the track's codec-specific record.
    pub configuration: VideoDecoderConfig,
    /// Every sample of the track, keyed by presentation index.
    pub samples: Vec<EncodedVideoSample>,
    /// Coded luma dimensions, for megapixel throughput reporting.
    pub width: usize,
    pub height: usize,
}

impl VideoFixture {
    /// Luma samples in one frame, the unit megapixel throughput is derived
    /// from.
    #[must_use]
    pub fn pixels_per_frame(&self) -> u64 {
        (self.width * self.height) as u64
    }
}

/// The bundled HEVC sample, demuxed once.
///
/// The configuration pins [`HardwarePreference::Avoid`] so the benchmark
/// always exercises zvidlib's own pure-Rust decoder. A platform decoder would
/// ignore the SIMD override entirely and report timings that say nothing about
/// this crate's kernels.
pub fn hevc_bundled() -> &'static VideoFixture {
    static FIXTURE: OnceLock<VideoFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let limits = Limits::default();
        let source = MemorySource::new(BUNDLED_MP4.to_vec());
        let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))
            .expect("the bundled sample must demux");
        let track = movie
            .tracks
            .iter()
            .find(|track| track.codec == Codec::Hevc)
            .expect("the bundled sample must carry an HEVC track");
        let dimensions = track
            .dimensions
            .expect("the bundled HEVC track must declare dimensions");
        let samples = block_on(track.to_encoded_video_samples(&source, &limits))
            .expect("the bundled sample must yield encoded video samples");
        VideoFixture {
            configuration: VideoDecoderConfig {
                codec: Codec::Hevc,
                profile: CodecProfile::HevcMain,
                coded_dimensions: dimensions,
                output_format: PixelFormat::Rgba8,
                color_range: ColorRange::Limited,
                hardware: HardwarePreference::Avoid,
                configuration: track.decoder_config.clone(),
            },
            samples,
            width: dimensions.width as usize,
            height: dimensions.height as usize,
        }
    })
}

/// The lossless 17x9 AV1 intra temporal unit, decoded from its hex fixture.
pub fn av1_lossless_intra() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| decode_hex(AV1_LOSSLESS_INTRA_HEX))
}

/// The 16x16 AV1 key/inter stream, split into the temporal units an
/// [`zvidlib::Av1InterDecoder`] consumes one at a time.
pub fn av1_inter_temporal_units() -> &'static [Vec<u8>] {
    static UNITS: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    UNITS.get_or_init(|| {
        let stream = decode_hex(AV1_INTER_HEX);
        temporal_units(&stream)
            .into_iter()
            .map(<[u8]>::to_vec)
            .collect()
    })
}

/// Parses a whitespace-tolerant hex dump into bytes.
fn decode_hex(text: &str) -> Vec<u8> {
    let digits: Vec<u8> = text
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    assert!(digits.len() % 2 == 0, "hex fixture must be paired");
    digits
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).expect("hex fixture must be ASCII");
            u8::from_str_radix(text, 16).expect("hex fixture must be hexadecimal")
        })
        .collect()
}

/// Splits a low-overhead AV1 byte stream at each `TemporalDelimiter` OBU
/// (`obu_type == 2`), which is the unit `Av1InterDecoder::decode_temporal_unit`
/// accepts. Mirrors the parsing `tests/av1_inter_decoder.rs` uses for the same
/// fixture.
fn temporal_units(stream: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut cursor = 0usize;
    while cursor < stream.len() {
        let start = cursor;
        let header = stream[cursor];
        cursor += 1;
        let obu_type = (header >> 3) & 0x0f;
        assert_ne!(header & 0x02, 0, "fixture OBU must carry a size field");
        let mut payload_len = 0usize;
        let mut shift = 0usize;
        loop {
            let byte = stream[cursor];
            cursor += 1;
            payload_len |= usize::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        cursor += payload_len;
        assert!(cursor <= stream.len(), "fixture OBU length is in bounds");
        if obu_type == 2 {
            starts.push(start);
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(index, &start)| {
            let end = starts.get(index + 1).copied().unwrap_or(stream.len());
            &stream[start..end]
        })
        .collect()
}
