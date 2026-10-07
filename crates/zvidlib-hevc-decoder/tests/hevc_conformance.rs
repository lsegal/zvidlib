//! The registered HEVC decoder against reference RGBA digests (#508): HEVC
//! Main on the bundled 1080p sample and HEVC Main 10, through sequential,
//! reverse and alternating seeks, and a seek that skips pictures still
//! returning the fixture's frames. Moved here from the root package's
//! `codec_conformance` (#612), so it runs with the crate it covers.
#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use zvidlib_container::{
    FrameDigest, Mp4DemuxerOptions, VideoDecoderConformanceVector, verify_video_decoder_conformance,
};
use zvidlib_core::io::MemorySource;
use zvidlib_core::{
    CancellationToken, Codec, CodecProfile, ColorRange, EncodedVideoSample, ErrorKind,
    ExactFrameReader, FrameIndex, HardwarePreference, Limits, PixelFormat, VideoDecoderConfig,
    VideoDecoderFactory, VideoDimensions,
};
use zvidlib_hevc_decoder::native_hevc_video_decoder_factory;

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
fn native_hevc_decoder_matches_an_independent_decode_of_the_bundled_sample() {
    // The bundled 1080p sample has a single random-access point, so
    // `verify_video_decoder_conformance`'s reverse and alternating patterns
    // would re-decode it from the start for nearly every frame - close to
    // 300,000 1080p pictures, which never finished on a CI runner once this
    // file was run (#599). As the AV1 colour sample below does, every frame is
    // checked in order instead, followed by backward and forward seeks. The
    // three patterns still run for HEVC Main on the 32-frame groups of
    // `bbb_hevc_512x288_gop32.mp4`, in `crates/zvidlib-hevc-decoder/src/lib.rs`.
    let expected = include_str!("fixtures/big_buck_bunny_hevc_rgba.sha256")
        .lines()
        .map(|line| {
            let (_, digest) = line.split_once(' ').unwrap();
            FrameDigest::from_hex(digest).unwrap()
        })
        .collect::<Vec<_>>();
    let limits = Limits::default();
    let source =
        MemorySource::new(include_bytes!("../../../examples/media/BigBuckBunny.mp4").to_vec());
    let vector = block_on(VideoDecoderConformanceVector::from_mp4(
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
    .unwrap();
    assert_eq!(vector.samples.len(), 768);
    assert_eq!(vector.expected_frames.len(), 768);
    assert_eq!(vector.samples[0].presentation_index.0, 0);
    assert!(vector.samples[0].random_access);
    assert!(!vector.configuration.configuration.is_empty());

    let factory = native_hevc_video_decoder_factory();
    let mut reader = ExactFrameReader::new(
        &factory,
        vector.configuration.clone(),
        vector.samples.clone(),
        limits,
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    let order = (0..768).chain([5, 400, 401, 3]);
    for index in order {
        let frame = reader.get(FrameIndex(index), &cancellation).unwrap();
        assert_eq!(
            FrameDigest::from_frame(&frame).unwrap(),
            expected[index as usize],
            "frame {index}"
        );
    }

    let mut decoder = factory.create(&vector.configuration, &limits).unwrap();
    let malformed = EncodedVideoSample {
        presentation_index: FrameIndex(0),
        random_access: true,
        data: vec![0, 0, 0, 10, 1],
    };
    let error = decoder
        .submit(&malformed, &CancellationToken::new())
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::MalformedMedia);

    let constrained = Limits {
        max_allocation_bytes: 1,
        ..limits
    };
    let error = factory
        .create(&vector.configuration, &constrained)
        .err()
        .expect("the HEVC decoder must enforce its allocation limit");
    assert_eq!(error.kind(), ErrorKind::ResourceLimit);
}

/// Issue #508: an HEVC Main 10 track decodes in software, and asking for hardware does not route it
/// to an 8-bit-only accelerated backend first.
#[test]
fn native_hevc_decoder_conforms_for_main10() {
    let expected = include_str!("fixtures/bbb_hevc_main10_128x72_rgba.sha256")
        .lines()
        .map(|line| {
            let (_, digest) = line.split_once(' ').unwrap();
            FrameDigest::from_hex(digest).unwrap()
        })
        .collect::<Vec<_>>();
    let limits = Limits::default();
    let source = MemorySource::new(include_bytes!("fixtures/bbb_hevc_main10_128x72.mp4").to_vec());
    let vector = block_on(VideoDecoderConformanceVector::from_mp4(
        "HEVC Main 10 sample",
        &source,
        Mp4DemuxerOptions::default(),
        1,
        VideoDecoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain10,
            coded_dimensions: VideoDimensions::new(128, 72, &limits).unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Prefer,
            configuration: Vec::new(),
        },
        &expected,
    ))
    .unwrap();
    assert_eq!(vector.samples.len(), 12);

    let report =
        verify_video_decoder_conformance(&native_hevc_video_decoder_factory(), &vector, limits)
            .unwrap();
    assert_eq!(report.frames_verified, 36);
    assert_eq!(report.access_patterns_verified, 3);
}

/// A frame in the middle of the bundled sample's single group of pictures can only be reached by
/// decoding everything before it, and issue #354 is what that used to cost: every one of those
/// pictures was converted to RGBA for nobody. The reader now tells the decoder they are wanted
/// for reference only, and this is what that must not change - the frame the seek asks for, the
/// tail it keeps behind it, and a frame it passed are all still the fixture's frames.
#[test]
fn a_seek_that_skips_the_pictures_it_passes_still_decodes_the_frames_it_returns() {
    let expected = include_str!("fixtures/big_buck_bunny_hevc_rgba.sha256")
        .lines()
        .map(|line| {
            let (_, digest) = line.split_once(' ').unwrap();
            FrameDigest::from_hex(digest).unwrap()
        })
        .collect::<Vec<_>>();
    let limits = Limits::default();
    let source =
        MemorySource::new(include_bytes!("../../../examples/media/BigBuckBunny.mp4").to_vec());
    let vector = block_on(VideoDecoderConformanceVector::from_mp4(
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
    .unwrap();

    // A cache smaller than the seek: the reader keeps the frames within `max_cached_frames` of
    // its target, so the default 32 would cover most of this one and skip almost nothing.
    let seek_limits = Limits {
        max_cached_frames: 4,
        ..limits
    };
    let mut reader = ExactFrameReader::new(
        &native_hevc_video_decoder_factory(),
        vector.configuration.clone(),
        vector.samples.clone(),
        seek_limits,
    )
    .unwrap();
    let cancellation = CancellationToken::new();

    // A click part-way along a timeline: one request, every frame before it decoded to reach it.
    let target = 48_u64;
    let frame = reader.get(FrameIndex(target), &cancellation).unwrap();
    assert_eq!(
        FrameDigest::from_frame(&frame).unwrap(),
        vector.expected_frames[target as usize].digest,
        "the frame the seek asked for is not the fixture's frame"
    );
    let statistics = reader.statistics();
    assert_eq!(
        statistics.resets, 1,
        "one decode, from the random-access point"
    );
    assert!(
        statistics.samples_skipped >= target / 2,
        "the frames on the way are decoded without being converted: {statistics:?}"
    );

    // The frames immediately behind it are kept, which is what makes stepping backwards from
    // where a seek lands a cache hit rather than another decode from the random-access point.
    for index in [target - 1, target - 2] {
        let frame = reader.get(FrameIndex(index), &cancellation).unwrap();
        assert_eq!(
            FrameDigest::from_frame(&frame).unwrap(),
            vector.expected_frames[index as usize].digest,
            "frame {index}, just behind the seek, does not match the fixture"
        );
    }
    assert_eq!(
        reader.statistics().resets,
        statistics.resets,
        "stepping back into the tail the seek kept decodes nothing again"
    );

    // A frame further back was skipped, so it costs a reset and a decode from the random-access
    // point - and comes back exactly, which is the part that matters.
    let passed = 12_u64;
    let frame = reader.get(FrameIndex(passed), &cancellation).unwrap();
    assert_eq!(
        FrameDigest::from_frame(&frame).unwrap(),
        vector.expected_frames[passed as usize].digest,
        "a frame the seek passed does not match the fixture when it is asked for"
    );
    assert_eq!(reader.statistics().resets, statistics.resets + 1);
}
