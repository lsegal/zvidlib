//! What the library's preview tier answers with, and how fast.
//!
//! This moved here with [`zvidlib::PreviewIndex`] itself: issue #395 made the
//! preview tier part of the library rather than a copy inside
//! `examples/native_gl`, and the measurement that says the tier is worth having
//! belongs with the code it measures.

#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use zvidlib::io::MemorySource;
use zvidlib::{
    Codec, CodecProfile, ColorRange, HardwarePreference, Limits, Mp4Demuxer, Mp4DemuxerOptions,
    PixelFormat, PreviewIndex, PreviewOptions, TrackKind, VideoDecoderConfig,
    native_hevc_video_decoder_factory,
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

/// Issue #374: what a scrub anywhere on the bar costs once the index is built.
///
/// The point of the index is that this is not a decode, so the figure it prints should be
/// microseconds rather than the second the same position costs to decode exactly. Ignored
/// because it is a wall-clock reading on whatever host runs it, and because building the
/// index decodes the whole bundled track.
#[test]
#[ignore = "host-specific preview-latency measurement"]
fn a_preview_answers_any_position_without_decoding() {
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/media/BigBuckBunny.mp4"
    ));
    let source = MemorySource::new(bytes.to_vec());
    let demuxer = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
    let video = demuxer
        .tracks
        .iter()
        .find(|track| track.kind == TrackKind::Video)
        .unwrap();
    let limits = Limits::default();
    let samples = block_on(video.to_encoded_video_samples(&source, &limits)).unwrap();
    let frame_count = samples.len() as u64;
    let configuration = VideoDecoderConfig {
        codec: video.codec,
        profile: CodecProfile::HevcMain,
        coded_dimensions: video.dimensions.unwrap(),
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Prefer,
        configuration: video.decoder_config.clone(),
    };
    assert_eq!(configuration.codec, Codec::Hevc);
    let factory = native_hevc_video_decoder_factory();
    let started = Instant::now();
    let index = PreviewIndex::new(
        &factory,
        configuration,
        samples,
        limits,
        PreviewOptions::for_frame_rate(24),
    )
    .expect("preview index");

    // Wait for the pass to cover the whole track, which is the state a drag finds it in a
    // second or two after the window opens.
    index.wait_for_coverage();
    let (filled, total) = index.coverage();
    assert_eq!(filled, total, "the preview pass left a position undecoded");
    let built = started.elapsed();

    // Every position on the bar, sampled a hundred ways, answered from the index.
    let mut worst = Duration::ZERO;
    for step in 0..=100_u64 {
        let frame = step * (frame_count - 1) / 100;
        let started = Instant::now();
        let preview = index.nearest(frame).expect("a preview for every position");
        worst = worst.max(started.elapsed());
        assert_eq!(preview.pixel_format, PixelFormat::Rgba8);
    }
    eprintln!("preview index over {frame_count} frames built in {built:?}; worst lookup {worst:?}");
    assert!(
        worst < Duration::from_millis(50),
        "a preview lookup took {worst:?}"
    );
}
