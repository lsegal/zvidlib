#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use zvidlib::hevc_hardware_readback as readback;
use zvidlib::io::MemorySource;
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

/// Issue #354: reaching the far end of the bundled sample's single group of pictures means
/// decoding every frame before it, and on a hardware backend the NV12-to-RGBA pass over each of
/// those pictures costs more than the decoding does. The reader now tells the decoder that the
/// pictures it is walking past are wanted for reference only, and VideoToolbox skips them with
/// `kVTDecodeFrame_DoNotOutputFrame`. The readback seam is what says the conversion really did
/// not run, and the fixture digest is what says skipping it changed no frame that was asked for.
#[test]
fn a_hardware_walk_converts_only_the_frames_it_is_asked_for() {
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
    let target = 120_u64;
    let mut reader = ExactFrameReader::new(
        &factory,
        vector.configuration.clone(),
        vector.samples.clone(),
        Limits::default(),
    )
    .unwrap();
    let cancellation = CancellationToken::new();

    readback::reset();
    let frame = reader.get(FrameIndex(target), &cancellation).unwrap();
    assert_eq!(
        FrameDigest::from_frame(&frame).unwrap(),
        vector.expected_frames[target as usize].digest,
        "the frame the walk stops on is the fixture's frame"
    );
    let statistics = reader.statistics();
    let converted = readback::report().frames;
    assert!(
        statistics.samples_skipped >= target / 2,
        "the walk decoded {} samples and skipped only {}",
        statistics.samples_submitted,
        statistics.samples_skipped
    );
    assert!(
        converted < statistics.samples_submitted,
        "{converted} frames were converted for {} samples, so nothing was skipped",
        statistics.samples_submitted
    );

    // The frames after it are still there to be decoded, and still exact: skipping a picture
    // must not disturb the reference decoding the frames after it depend on.
    for index in [target + 1, target + 8, target + 40] {
        let frame = reader.get(FrameIndex(index), &cancellation).unwrap();
        assert_eq!(
            FrameDigest::from_frame(&frame).unwrap(),
            vector.expected_frames[index as usize].digest,
            "frame {index} after a skipped walk does not match the fixture"
        );
    }
}

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

/// The readback seam (issue #283) has to attribute real time to both phases of
/// a hardware decode, or the benchmark group built on it silently reports
/// zeros. Only a hardware backend charges it; the software decoder's conversion
/// is attributed by `hevc_decode_profile` instead, so this asserts the seam is
/// quiet for software and populated for hardware.
#[test]
fn hardware_readback_seam_attributes_each_decoded_frame() {
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
    let frames = 8_u64;
    let mut reader = ExactFrameReader::new(
        &factory,
        vector.configuration,
        vector.samples,
        Limits::default(),
    )
    .unwrap();
    let cancellation = CancellationToken::new();

    readback::reset();
    assert_eq!(readback::report(), readback::Report::default());
    for index in 0..frames {
        reader.get(FrameIndex(index), &cancellation).unwrap();
    }
    let report = readback::report();

    // `ExactFrameReader` decodes from the preceding key frame, so it delivers at
    // least the frames it was asked for and possibly more.
    assert!(
        report.frames >= frames,
        "the seam attributed {} frames for {frames} requested",
        report.frames
    );
    assert!(
        report.color_convert > Duration::ZERO,
        "a 1920x1080 NV12-to-RGBA pass cannot cost zero"
    );
    assert!(
        report.total() >= report.color_convert,
        "the total has to include both phases"
    );
    assert!(
        report.total_per_frame() > Duration::ZERO && report.total_per_frame() <= report.total(),
        "per-frame readback {:?} is not a share of {:?}",
        report.total_per_frame(),
        report.total()
    );
    eprintln!(
        "readback over {} frames: surface copy {:?}, colour convert {:?}",
        report.frames, report.surface_copy, report.color_convert
    );
}
