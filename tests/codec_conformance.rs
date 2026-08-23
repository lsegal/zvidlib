#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use zvidlib::io::MemorySource;
use zvidlib::{
    CancellationToken, Codec, CodecProfile, ColorRange, EncodedVideoSample, ErrorKind,
    ExpectedVideoFrame, FrameDigest, FrameIndex, HardwarePreference, Limits, Mp4DemuxerOptions,
    PixelFormat, VideoDecoderConfig, VideoDecoderConformanceVector, VideoDecoderFactory,
    VideoDimensions, native_av1_video_decoder_factory, native_hevc_video_decoder_factory,
    verify_video_decoder_conformance,
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

    let report =
        verify_video_decoder_conformance(&native_hevc_video_decoder_factory(), &vector, limits)
            .unwrap();
    assert_eq!(report.frames_verified, 360);
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
