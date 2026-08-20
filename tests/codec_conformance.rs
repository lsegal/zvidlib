use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use zvidlib::io::MemorySource;
use zvidlib::{
    Codec, CodecProfile, ColorRange, FrameDigest, HardwarePreference, Limits, Mp4DemuxerOptions,
    PixelFormat, VideoDecoderConfig, VideoDecoderConformanceVector, VideoDimensions,
};

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

#[test]
fn bundled_hevc_sample_is_a_complete_decoder_vector() {
    let expected = include_str!("fixtures/codec/big_buck_bunny_hevc_rgba.sha256")
        .lines()
        .map(|line| {
            let (_, digest) = line.split_once(' ').unwrap();
            FrameDigest::from_hex(digest).unwrap()
        })
        .collect::<Vec<_>>();
    let limits = Limits::default();
    let source = MemorySource::new(include_bytes!("../examples/media/BigBuckBunny.mp4").to_vec());
    let vector = block_on(VideoDecoderConformanceVector::from_mp4(
        "bundled HEVC Main sample",
        &source,
        Mp4DemuxerOptions::default(),
        1,
        VideoDecoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: VideoDimensions::new(320, 180, &limits).unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: Vec::new(),
        },
        &expected,
    ))
    .unwrap();
    assert_eq!(vector.samples.len(), 120);
    assert_eq!(vector.expected_frames.len(), 120);
    assert_eq!(vector.samples[0].presentation_index.0, 0);
    assert!(vector.samples[0].random_access);
    assert!(!vector.configuration.configuration.is_empty());
}
