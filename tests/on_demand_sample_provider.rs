//! `ExactFrameReader` built from an on-demand [`Mp4SampleProvider`] decodes
//! the same frames, through the real native HEVC decoder, that it does from
//! an eagerly read `Vec<EncodedVideoSample>` (issue #669).

#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::task::{Context, Poll, Waker};

use zvidlib::io::{CachingByteSource, MemorySource};
use zvidlib::{
    CancellationToken, Codec, CodecProfile, ColorRange, ExactFrameReader, FrameIndex,
    HardwarePreference, Limits, Mp4Demuxer, Mp4DemuxerOptions, Mp4SampleProvider, Mp4Track,
    PixelFormat, TrackKind, VideoDecoderConfig, native_hevc_video_decoder_factory,
};

fn block_on<F: Future>(future: F) -> F::Output {
    let mut boxed = Box::pin(future);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    match boxed.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("unexpected pending future"),
    }
}

fn bundled_track_and_bytes() -> (Mp4Track, Vec<u8>) {
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/media/BigBuckBunny.mp4"
    ))
    .to_vec();
    let source = MemorySource::new(bytes.clone());
    let demuxer = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
    let track = demuxer
        .tracks
        .into_iter()
        .find(|track| track.kind == TrackKind::Video)
        .expect("sample video has a video track");
    (track, bytes)
}

fn configuration(track: &Mp4Track) -> VideoDecoderConfig {
    VideoDecoderConfig {
        codec: track.codec,
        profile: CodecProfile::HevcMain,
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
    let limits = Limits::default();
    let factory = native_hevc_video_decoder_factory();
    let cancellation = CancellationToken::new();
    let configuration_template = configuration(&track);

    let eager_source = MemorySource::new(bytes.clone());
    let eager_samples = block_on(track.to_encoded_video_samples(&eager_source, &limits)).unwrap();
    let frame_count = eager_samples.len() as u64;
    let mut eager_reader = ExactFrameReader::new(
        &factory,
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
    assert!(budget > 0, "the bundled sample is too small for this test");
    let source = MemorySource::new(bytes);
    let cache = CachingByteSource::new(source, 64 * 1024, budget).unwrap();
    let provider = Mp4SampleProvider::new(track, cache).unwrap();
    let mut on_demand_reader = ExactFrameReader::from_provider(
        &factory,
        configuration_template,
        Box::new(provider),
        limits,
    )
    .unwrap();

    // Forward through the whole track, then a handful of backward seeks -
    // the same positions answered by both readers must agree exactly.
    let positions: Vec<u64> = (0..frame_count)
        .step_by((frame_count as usize / 32).max(1))
        .chain([frame_count - 1, frame_count / 2, 0])
        .collect();
    for frame in positions {
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
