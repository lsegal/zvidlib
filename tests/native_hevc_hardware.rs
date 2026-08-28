#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
#[cfg(windows)]
use std::time::Instant;

use zvidlib::io::MemorySource;
#[cfg(windows)]
use zvidlib::{CancellationToken, ExactFrameReader, FrameIndex};
use zvidlib::{
    Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange, FrameDigest,
    HardwarePreference, Limits, Mp4DemuxerOptions, PixelFormat, VideoDecoderConfig,
    VideoDecoderConformanceVector, VideoDecoderFactory, VideoDimensions,
    native_hevc_video_decoder_factory,
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

fn bundled_vector() -> VideoDecoderConformanceVector {
    let expected = include_str!("fixtures/codec/big_buck_bunny_hevc_rgba.sha256")
        .lines()
        .map(|line| {
            let (_, digest) = line.split_once(' ').unwrap();
            FrameDigest::from_hex(digest).unwrap()
        })
        .collect::<Vec<_>>();
    let limits = Limits::default();
    let source = MemorySource::new(include_bytes!("../examples/media/BigBuckBunny.mp4").to_vec());
    block_on(VideoDecoderConformanceVector::from_mp4(
        "bundled HEVC Main sample",
        &source,
        Mp4DemuxerOptions::default(),
        1,
        VideoDecoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: VideoDimensions::new(1920, 1080, &limits).unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: Vec::new(),
        },
        &expected,
    ))
    .unwrap()
}

#[test]
fn native_hevc_factory_honors_hardware_preference_and_fallback() {
    let vector = bundled_vector();
    let factory = native_hevc_video_decoder_factory();

    let mut avoid = vector.configuration.clone();
    avoid.hardware = HardwarePreference::Avoid;
    assert_eq!(
        factory.capability(&avoid),
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        }
    );

    let mut prefer = avoid.clone();
    prefer.hardware = HardwarePreference::Prefer;
    let preferred = factory.capability(&prefer);
    assert!(matches!(
        preferred,
        CodecSupport::Supported {
            implementation: CodecImplementation::Software | CodecImplementation::Hardware,
        }
    ));

    let mut require = avoid;
    require.hardware = HardwarePreference::Require;
    let required = factory.capability(&require);
    match preferred {
        CodecSupport::Supported {
            implementation: CodecImplementation::Hardware,
        } => assert_eq!(
            required,
            CodecSupport::Supported {
                implementation: CodecImplementation::Hardware,
            }
        ),
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        } => assert_eq!(required, CodecSupport::HardwareUnavailable),
        _ => unreachable!(),
    }
}

#[cfg(windows)]
#[test]
fn accelerated_hevc_preserves_exact_frame_identity_and_seek_behavior() {
    let mut vector = bundled_vector();
    vector.configuration.hardware = HardwarePreference::Require;
    let factory = native_hevc_video_decoder_factory();
    if factory.capability(&vector.configuration) == CodecSupport::HardwareUnavailable {
        let reason = factory
            .create(&vector.configuration, &Limits::default())
            .err()
            .map_or_else(|| "unknown reason".into(), |error| error.to_string());
        eprintln!("skipping: hardware HEVC unavailable: {reason}");
        return;
    }
    let sequential = (0..16).map(FrameIndex).collect::<Vec<_>>();
    let reverse = (0..16).rev().map(FrameIndex).collect::<Vec<_>>();
    let alternating = [0, 119, 1, 118, 2, 117, 3, 116]
        .into_iter()
        .map(FrameIndex)
        .collect::<Vec<_>>();
    let mut frames_verified = 0;
    for pattern in [&sequential, &reverse, &alternating] {
        let mut reader = ExactFrameReader::new(
            &factory,
            vector.configuration.clone(),
            vector.samples.clone(),
            Limits::default(),
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        for &index in pattern {
            let frame = reader.get(index, &cancellation).unwrap();
            let actual = FrameDigest::from_frame(&frame).unwrap();
            let expected = vector.expected_frames[index.0 as usize].digest;
            assert_eq!(actual, expected, "frame {} fingerprint mismatch", index.0);
            frames_verified += 1;
        }
    }
    assert_eq!(frames_verified, 40);
}

#[cfg(windows)]
#[test]
#[ignore = "host-specific real-time playback benchmark"]
fn accelerated_hevc_decodes_bundled_1080p_sample_at_source_rate() {
    let mut vector = bundled_vector();
    vector.configuration.hardware = HardwarePreference::Require;
    let factory = native_hevc_video_decoder_factory();
    if factory.capability(&vector.configuration) == CodecSupport::HardwareUnavailable {
        let reason = factory
            .create(&vector.configuration, &Limits::default())
            .err()
            .map_or_else(|| "unknown reason".into(), |error| error.to_string());
        eprintln!("skipping: hardware HEVC unavailable: {reason}");
        return;
    }
    let mut reader = ExactFrameReader::new(
        &factory,
        vector.configuration,
        vector.samples,
        Limits::default(),
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    let frames = 256_u64;
    let started = Instant::now();
    for index in 0..frames {
        reader.get(FrameIndex(index), &cancellation).unwrap();
    }
    let elapsed = started.elapsed();
    let fps = frames as f64 / elapsed.as_secs_f64();
    eprintln!("decoded {frames} 1920x1080 frames in {elapsed:?} ({fps:.2} FPS)");
    assert!(
        fps >= 24.0,
        "accelerated decoder achieved {fps:.2} FPS, below the 24 FPS source rate"
    );
}
