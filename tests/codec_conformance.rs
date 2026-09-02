#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use zvidlib::io::MemorySource;
use zvidlib::{
    CancellationToken, Codec, CodecProfile, ColorRange, EncodedVideoSample, ErrorKind,
    ExactFrameReader, ExpectedVideoFrame, FrameDigest, FrameIndex, HardwarePreference, Limits,
    Mp4DemuxerOptions, PixelFormat, VideoDecoderConfig, VideoDecoderConformanceVector,
    VideoDecoderFactory, VideoDimensions, native_av1_video_decoder_factory,
    native_hevc_video_decoder_factory, verify_video_decoder_conformance,
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
fn native_hevc_decoder_conforms_for_sequential_reverse_and_alternating_seeks() {
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

    let report =
        verify_video_decoder_conformance(&native_hevc_video_decoder_factory(), &vector, limits)
            .unwrap();
    assert_eq!(report.frames_verified, 2304);
    assert_eq!(report.access_patterns_verified, 3);

    let factory = native_hevc_video_decoder_factory();
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

/// A frame in the middle of the bundled sample's single group of pictures can only be reached by
/// decoding everything before it, and issue #354 is what that used to cost: every one of those
/// pictures was converted to RGBA for nobody. The reader now tells the decoder they are wanted
/// for reference only, and this is what that must not change - the frame it walks to, and the
/// ones it publishes after it, are still the fixture's frames.
#[test]
fn a_walk_that_skips_the_pictures_it_passes_still_decodes_the_frames_it_stops_on() {
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
            coded_dimensions: VideoDimensions::new(1920, 1080, &limits).unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: Vec::new(),
        },
        &expected,
    ))
    .unwrap();

    // A cache smaller than the walk: the reader keeps the frames within `max_cached_frames` of
    // its target, so the default 32 would cover this whole walk and skip nothing.
    let walk_limits = Limits {
        max_cached_frames: 4,
        ..limits
    };
    let mut reader = ExactFrameReader::new(
        &native_hevc_video_decoder_factory(),
        vector.configuration.clone(),
        vector.samples.clone(),
        walk_limits,
    )
    .unwrap();
    let cancellation = CancellationToken::new();

    // The walk a drag makes: stop every eight frames, as the native example does.
    for index in (0..=24_u64).step_by(8) {
        let frame = reader.get(FrameIndex(index), &cancellation).unwrap();
        assert_eq!(
            FrameDigest::from_frame(&frame).unwrap(),
            vector.expected_frames[index as usize].digest,
            "frame {index} does not match the fixture"
        );
    }
    let statistics = reader.statistics();
    assert_eq!(statistics.resets, 1, "one walk, not one per stop");
    assert!(
        statistics.samples_skipped >= 8,
        "the frames between the stops are decoded without being converted: {statistics:?}"
    );

    // The frames immediately before a stop are kept, which is what makes stepping backwards
    // from it a cache hit rather than another walk from the random-access point.
    let resets = statistics.resets;
    for index in [23_u64, 22] {
        let frame = reader.get(FrameIndex(index), &cancellation).unwrap();
        assert_eq!(
            FrameDigest::from_frame(&frame).unwrap(),
            vector.expected_frames[index as usize].digest,
            "frame {index}, just behind the last stop, does not match the fixture"
        );
    }
    assert_eq!(
        reader.statistics().resets,
        resets,
        "stepping back into the walk's own tail decodes nothing again"
    );

    // A frame the walk went past is not lost, only sometimes more expensive: whether it is still
    // cached, was kept because it follows a stop in presentation order, or has to be decoded
    // again from the random-access point, what comes back is the fixture's frame.
    for index in [13_u64, 5, 21, 24] {
        let frame = reader.get(FrameIndex(index), &cancellation).unwrap();
        assert_eq!(
            FrameDigest::from_frame(&frame).unwrap(),
            vector.expected_frames[index as usize].digest,
            "frame {index}, which the walk passed, does not match the fixture"
        );
    }
}

/// Splits a low-overhead AV1 byte stream into temporal units, delimited by
/// (and including) each `TemporalDelimiter` OBU (`obu_type == 2`). Mirrors
/// the parsing `tests/av1_inter_decoder.rs` already uses for this fixture.
fn av1_temporal_units(stream: &[u8]) -> Vec<&[u8]> {
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

/// A minimal `av1C` box declaring an 8-bit monochrome Main-profile stream
/// with no `configOBUs` (AV1 spec §5.9.16 does not require the sequence
/// header to be repeated there; this decoder validates coded dimensions
/// against each decoded frame instead, see `src/av1_decoder.rs`).
fn av1c_monochrome_main() -> Vec<u8> {
    let payload = [0x81_u8, 0x00, 0x1C, 0x00];
    let mut bytes = (8_u32 + payload.len() as u32).to_be_bytes().to_vec();
    bytes.extend_from_slice(b"av1C");
    bytes.extend_from_slice(&payload);
    bytes
}

#[test]
fn native_av1_decoder_conforms_for_sequential_reverse_and_alternating_seeks() {
    // This low-overhead OBU sequence (also exercised directly against
    // `Av1InterDecoder` in `tests/av1_inter_decoder.rs`) is generated from
    // the normative AV1 syntax tables and independently decoded by
    // FFmpeg/libdav1d; see `tests/fixtures/codec/README.md`. It contains a
    // key frame, two refreshed inter references, LAST/LAST2 average
    // compound prediction, and a show-existing-frame header for the
    // retained compound frame. The RGBA digests below are the canonical
    // output of this crate's own spec-documented, independently
    // unit-tested `convert_to_rgba8` BT.601 conversion (see
    // `src/av1_filters.rs`) applied to that hermetically decoded YUV420
    // output.
    let stream_hex = include_str!("fixtures/codec/av1_inter_show_existing_16x16.hex").trim();
    let stream: Vec<u8> = stream_hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    let units = av1_temporal_units(&stream);
    assert_eq!(units.len(), 5);

    let expected: Vec<FrameDigest> =
        include_str!("fixtures/codec/av1_inter_show_existing_16x16_rgba.sha256")
            .lines()
            .map(|line| {
                let (_, digest) = line.split_once(' ').unwrap();
                FrameDigest::from_hex(digest).unwrap()
            })
            .collect();
    assert_eq!(expected.len(), units.len());

    let limits = Limits::default();
    let configuration = VideoDecoderConfig {
        codec: Codec::Av1,
        profile: CodecProfile::Av1Main,
        coded_dimensions: VideoDimensions::new(16, 16, &limits).unwrap(),
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Avoid,
        configuration: av1c_monochrome_main(),
    };
    let samples: Vec<EncodedVideoSample> = units
        .iter()
        .enumerate()
        .map(|(index, unit)| EncodedVideoSample {
            presentation_index: FrameIndex(index as u64),
            random_access: index == 0,
            data: unit.to_vec(),
        })
        .collect();
    let expected_frames: Vec<ExpectedVideoFrame> = expected
        .iter()
        .enumerate()
        .map(|(index, &digest)| ExpectedVideoFrame {
            presentation_index: FrameIndex(index as u64),
            digest,
        })
        .collect();
    let vector = VideoDecoderConformanceVector {
        name: "AV1 Main lossless monochrome inter + show_existing_frame".into(),
        configuration,
        samples,
        expected_frames,
    };

    let report =
        verify_video_decoder_conformance(&native_av1_video_decoder_factory(), &vector, limits)
            .unwrap();
    assert_eq!(report.frames_verified, 15);
    assert_eq!(report.access_patterns_verified, 3);

    let factory = native_av1_video_decoder_factory();
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
        .expect("the AV1 decoder must enforce its allocation limit");
    assert_eq!(error.kind(), ErrorKind::ResourceLimit);
}
