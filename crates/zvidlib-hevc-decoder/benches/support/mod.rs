//! The HEVC decode targets' support: the bundled 1080p sample, demuxed once
//! per process, on top of `zvidlib-bench-support`'s shared helpers.
// Each bench target of this package compiles this module and uses only what
// its own groups need, so unused-here is not dead.
#![allow(dead_code)]

use std::sync::OnceLock;

use zvidlib_container::{Mp4Demuxer, Mp4DemuxerOptions};
use zvidlib_core::io::MemorySource;
use zvidlib_core::{
    Codec, CodecProfile, ColorRange, EncodedVideoSample, HardwarePreference, Limits, PixelFormat,
    VideoDecoderConfig,
};

zvidlib_bench_support::bench_support!(
    zvidlib_hevc_decoder::simd_sites,
    zvidlib_hevc_encoder::simd_sites,
    zvidlib_color::simd_sites
);

/// The bundled 1920x1080 HEVC Main sample, demuxed once into decoder-ready
/// samples and a matching decoder configuration.
pub struct BundledHevcSample {
    pub configuration: VideoDecoderConfig,
    pub samples: Vec<EncodedVideoSample>,
    pub width: u64,
    pub height: u64,
}

/// Demuxes `examples/media/BigBuckBunny.mp4` once per process.
///
/// This is the only fixture large enough to matter: it is ~3 MB of bitstream and
/// 768 presentation frames, so it belongs to the long-running benchmark group
/// rather than the default fast path.
pub fn bundled_hevc_sample() -> &'static BundledHevcSample {
    static SAMPLE: OnceLock<BundledHevcSample> = OnceLock::new();
    SAMPLE.get_or_init(|| {
        let limits = Limits::default();
        let source = MemorySource::new(
            include_bytes!("../../../../examples/media/BigBuckBunny.mp4").to_vec(),
        );
        let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))
            .expect("the bundled sample is a readable MP4");
        let track = movie.track(1).expect("the bundled sample has track 1");
        let dimensions = track
            .dimensions
            .expect("the bundled sample's video track is dimensioned");
        let samples = block_on(track.to_encoded_video_samples(&source, &limits))
            .expect("the bundled sample's video samples are readable");
        BundledHevcSample {
            configuration: VideoDecoderConfig {
                codec: Codec::Hevc,
                profile: CodecProfile::HevcMain,
                coded_dimensions: dimensions,
                output_format: PixelFormat::Rgba8,
                color_range: ColorRange::Limited,
                // Benchmarks measure the crate's own decode work, not whichever
                // fixed-function block the host happens to ship.
                hardware: HardwarePreference::Avoid,
                configuration: track.decoder_config.clone(),
            },
            samples,
            width: u64::from(dimensions.width),
            height: u64::from(dimensions.height),
        }
    })
}
