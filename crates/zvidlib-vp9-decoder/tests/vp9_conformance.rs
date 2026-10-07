//! The registered VP9 decoder against libvpx's decode of the MP4 fixtures
//! (#527), through sequential, reverse and alternating seeks. Moved here from
//! the root package's `codec_conformance` (#612), so it runs with the crate it
//! covers.
#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use zvidlib_container::{
    FrameDigest, Mp4DemuxerOptions, VideoDecoderConformanceVector,
    verify_video_decoder_conformance,
};
use zvidlib_core::io::MemorySource;
use zvidlib_core::{
    Codec, CodecProfile, ColorRange, HardwarePreference, Limits, PixelFormat, VideoDecoderConfig,
    VideoDecoderFactory, VideoDimensions,
};
use zvidlib_vp9_decoder::native_vp9_video_decoder_factory;

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

/// The VP9 MP4 fixtures, decoded through the registered factory, match
/// libvpx's decode of the same tracks converted to RGBA by this crate's
/// BT.601 `convert_to_rgba8` (both tracks leave their colour space
/// unspecified), for sequential, reverse and alternating access. The
/// 256x144 track's superframes carry hidden alternate reference frames and
/// it has two random-access points; the 250x142 one has edges that are not
/// a multiple of 8. See `tests/fixtures/codec/README.md`.
#[test]
fn native_vp9_decoder_conforms_to_libvpx_for_sequential_reverse_and_alternating_seeks() {
    let limits = Limits::default();
    for (name, mp4, expected, (width, height), frames) in [
        (
            "VP9 256x144 with hidden frames",
            include_bytes!("fixtures/vp9_bbb_256x144.mp4")
                .as_slice(),
            include_str!(
                "fixtures/vp9_bbb_256x144_rgba.sha256"
            ),
            (256, 144),
            48,
        ),
        (
            "VP9 250x142",
            include_bytes!("fixtures/vp9_bbb_250x142.mp4")
                .as_slice(),
            include_str!(
                "fixtures/vp9_bbb_250x142_rgba.sha256"
            ),
            (250, 142),
            12,
        ),
    ] {
        let expected = expected
            .lines()
            .map(|line| {
                let (_, digest) = line.split_once(' ').unwrap();
                FrameDigest::from_hex(digest).unwrap()
            })
            .collect::<Vec<_>>();
        let source = MemorySource::new(mp4.to_vec());
        let vector = block_on(VideoDecoderConformanceVector::from_mp4(
            name,
            &source,
            Mp4DemuxerOptions::default(),
            1,
            VideoDecoderConfig {
                codec: Codec::Vp9,
                profile: CodecProfile::Vp9Profile0,
                coded_dimensions: VideoDimensions::new(width, height, &limits).unwrap(),
                output_format: PixelFormat::Rgba8,
                color_range: ColorRange::Limited,
                hardware: HardwarePreference::Prefer,
                configuration: Vec::new(),
            },
            &expected,
        ))
        .unwrap();
        assert_eq!(vector.samples.len(), frames);
        assert!(vector.samples[0].random_access);
        assert_eq!(&vector.configuration.configuration[4..8], b"vpcC");
        let factory = native_vp9_video_decoder_factory();
        assert!(factory.capability(&vector.configuration).is_supported());
        let report = verify_video_decoder_conformance(&factory, &vector, limits).unwrap();
        assert_eq!(report.frames_verified, 3 * frames as u64, "{name}");
        assert_eq!(report.access_patterns_verified, 3, "{name}");
    }
}
