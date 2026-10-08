//! `ExactFrameReader` built from an on-demand [`Mp4SampleProvider`] decodes
//! the same frames, through the real native HEVC and VP9 decoders, that it
//! does from an eagerly read `Vec<EncodedVideoSample>`, from an MP4 (issue
//! #669) and a WebM (issue #685) alike.

#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::{WEBM_FRAMES, block_on, vp9_opus_webm};
use zvidlib::io::{CachingByteSource, MemorySource};
use zvidlib::{
    CancellationToken, Codec, CodecProfile, ColorRange, ExactFrameReader, FrameIndex,
    HardwarePreference, Limits, Mp4SampleProvider, Mp4Track, PixelFormat, TrackKind,
    VideoDecoderConfig, VideoDecoderFactory, native_hevc_video_decoder_factory,
    native_vp9_video_decoder_factory,
};

/// The first video track of `bytes`, opened through the container-agnostic
/// entry point on-demand playback uses.
fn video_track(bytes: &[u8]) -> Mp4Track {
    let source = MemorySource::new(bytes.to_vec());
    block_on(zvidlib::container::open_media(&source, &Limits::default()))
        .unwrap()
        .first_track(TrackKind::Video)
        .expect("the input has a video track")
        .clone()
}

fn bundled_track_and_bytes() -> (Mp4Track, Vec<u8>) {
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/media/BigBuckBunny.mp4"
    ))
    .to_vec();
    (video_track(&bytes), bytes)
}

fn configuration(track: &Mp4Track, profile: CodecProfile) -> VideoDecoderConfig {
    VideoDecoderConfig {
        codec: track.codec,
        profile,
        coded_dimensions: track.dimensions.unwrap(),
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Prefer,
        configuration: track.decoder_config.clone(),
    }
}

/// An `ExactFrameReader` built over an on-demand provider, with no file read
/// up front beyond the container's own index, answers the same exact frames
/// as the eager `Vec`-backed path - forward, backward, and across the whole
/// track - while the compressed bytes behind it stay within a small cache
/// budget.
#[test]
fn on_demand_reader_matches_the_eager_reader_and_stays_within_its_cache_budget() {
    let (track, bytes) = bundled_track_and_bytes();
    assert_eq!(track.codec, Codec::Hevc);
    // A few nearby positions, including a backward seek back to the start -
    // the same positions answered by both readers must agree exactly. Kept
    // close together deliberately: the bundled sample is one group of
    // pictures with a single random-access point at frame 0 (see
    // `ARCHITECTURE.md` section 3.2), so walking far into it, or resetting
    // and walking again, decodes everything behind the target. This test's
    // job is proving the two readers agree on real decoder output, not
    // re-measuring a full-track walk, which the other on-demand-provider
    // tests already cover without paying for a decode.
    assert!(
        track.presentation_order.len() > 20,
        "the bundled sample is too short for this test"
    );
    assert_on_demand_matches_eager(
        track,
        bytes,
        &native_hevc_video_decoder_factory(),
        CodecProfile::HevcMain,
        &[0, 5, 12, 20, 5, 0],
    );
}

/// The same over a WebM's VP9 track, indexed from its block headers rather
/// than an MP4 sample table, walking across its second key frame and back.
#[test]
fn on_demand_reader_matches_the_eager_reader_over_a_webm() {
    let bytes = vp9_opus_webm(true);
    let track = video_track(&bytes);
    assert_eq!(track.codec, Codec::Vp9);
    assert_eq!(track.presentation_order.len() as u64, WEBM_FRAMES);
    assert_on_demand_matches_eager(
        track,
        bytes,
        &native_vp9_video_decoder_factory(),
        CodecProfile::Vp9Profile0,
        &[0, 5, 30, 59, 60, 75, 119, 12, 0],
    );
}

/// Decodes `positions` of `track` through an eager reader and through one
/// built over an on-demand provider whose compressed bytes stay within a
/// tenth of the track's, and requires the two to agree exactly.
fn assert_on_demand_matches_eager(
    track: Mp4Track,
    bytes: Vec<u8>,
    factory: &dyn VideoDecoderFactory,
    profile: CodecProfile,
    positions: &[u64],
) {
    let limits = Limits::default();
    let cancellation = CancellationToken::new();
    let configuration_template = configuration(&track, profile);

    let eager_source = MemorySource::new(bytes.clone());
    let eager_samples = block_on(track.to_encoded_video_samples(&eager_source, &limits)).unwrap();
    let mut eager_reader = ExactFrameReader::new(
        factory,
        configuration_template.clone(),
        eager_samples,
        limits,
    )
    .unwrap();

    let total_track_bytes: u64 = track
        .samples
        .iter()
        .map(|sample| u64::from(sample.size))
        .sum();
    let budget = total_track_bytes / 10;
    assert!(budget > 0, "the sample is too small for this test");
    let source = MemorySource::new(bytes);
    let cache = CachingByteSource::new(source, 64 * 1024, budget).unwrap();
    let provider = Mp4SampleProvider::new(track, cache).unwrap();
    let mut on_demand_reader = ExactFrameReader::from_provider(
        factory,
        configuration_template,
        Box::new(provider),
        limits,
    )
    .unwrap();

    for &frame in positions {
        let expected = eager_reader
            .get(FrameIndex(frame), &cancellation)
            .unwrap_or_else(|error| panic!("eager reader failed at frame {frame}: {error}"));
        let actual = on_demand_reader
            .get(FrameIndex(frame), &cancellation)
            .unwrap_or_else(|error| panic!("on-demand reader failed at frame {frame}: {error}"));
        assert_eq!(
            actual.planes[0].data, expected.planes[0].data,
            "frame {frame} differs between the eager and on-demand readers"
        );
    }
}
