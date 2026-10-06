//! Hardware VP9 decoding through the registered factory.
//!
//! `codec_conformance.rs` holds the factory to libvpx's decode of the VP9
//! fixtures under `HardwarePreference::Prefer`, which is the hardware decoder
//! wherever the host has one. On such a host these tests also require it:
//! every frame must match the software decoder's, hidden frames and
//! `show_existing_frame` included, and `ExactFrameReader` must return the
//! requested frames in sequential, reverse and alternating order. Elsewhere
//! they skip, and only the hardware preference answers are checked.
#![cfg(not(target_arch = "wasm32"))]

use zvidlib::io::MemorySource;
use zvidlib::{
    CancellationToken, Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange,
    EncodedVideoSample, ErrorKind, ExactFrameReader, FrameDigest, FrameIndex, HardwarePreference,
    Limits, Mp4Demuxer, PixelFormat, VideoDecoderConfig, VideoDecoderFactory, VideoDimensions,
    WebmDemuxer, native_vp9_video_decoder_factory,
};

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    use std::task::{Context, Poll, Waker};
    let mut context = Context::from_waker(Waker::noop());
    let mut future = Box::pin(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
    }
}

struct Track {
    configuration: VideoDecoderConfig,
    samples: Vec<EncodedVideoSample>,
}

fn configuration(dimensions: VideoDimensions, record: Vec<u8>) -> VideoDecoderConfig {
    VideoDecoderConfig {
        codec: Codec::Vp9,
        profile: CodecProfile::Vp9Profile0,
        coded_dimensions: dimensions,
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Avoid,
        configuration: record,
    }
}

fn mp4_track(file: &[u8]) -> Track {
    let source = MemorySource::new(file.to_vec());
    let movie = block_on(Mp4Demuxer::open(&source, Default::default())).unwrap();
    let track = movie.track(1).unwrap();
    assert_eq!(track.codec, Codec::Vp9);
    Track {
        configuration: configuration(track.dimensions.unwrap(), track.decoder_config.clone()),
        samples: block_on(track.to_encoded_video_samples(&source, &Limits::default())).unwrap(),
    }
}

fn webm_track(file: &[u8]) -> Track {
    let source = MemorySource::new(file.to_vec());
    let demuxer = block_on(WebmDemuxer::open(&source, Default::default())).unwrap();
    let track = &demuxer.tracks[0];
    assert_eq!(track.codec, Codec::Vp9);
    Track {
        configuration: configuration(track.dimensions.unwrap(), track.decoder_config.clone()),
        samples: block_on(track.to_encoded_video_samples(&source, &Limits::default())).unwrap(),
    }
}

fn fixtures() -> [(&'static str, Track); 3] {
    [
        (
            "VP9 256x144 with hidden frames",
            mp4_track(include_bytes!("fixtures/codec/vp9_bbb_256x144.mp4")),
        ),
        (
            "VP9 250x142",
            mp4_track(include_bytes!("fixtures/codec/vp9_bbb_250x142.mp4")),
        ),
        (
            "VP9 in WebM",
            webm_track(include_bytes!("fixtures/codec/vp9_bbb_256x144.webm")),
        ),
    ]
}

/// The `Require` form of `configuration`, or `None`, after saying why, when
/// this host has no hardware VP9 decoder.
fn hardware_configuration(configuration: &VideoDecoderConfig) -> Option<VideoDecoderConfig> {
    let factory = native_vp9_video_decoder_factory();
    let configuration = VideoDecoderConfig {
        hardware: HardwarePreference::Require,
        ..configuration.clone()
    };
    if factory.capability(&configuration) == CodecSupport::HardwareUnavailable {
        let reason = factory
            .create(&configuration, &Limits::default())
            .err()
            .map_or_else(|| "unknown reason".into(), |error| error.to_string());
        eprintln!("skipping: hardware VP9 unavailable: {reason}");
        return None;
    }
    Some(configuration)
}

/// The digest of every frame from one uninterrupted decode, each checked to
/// be the one frame its sample shows.
fn sequential_digests(
    configuration: &VideoDecoderConfig,
    samples: &[EncodedVideoSample],
) -> Vec<FrameDigest> {
    let factory = native_vp9_video_decoder_factory();
    let mut decoder = factory.create(configuration, &Limits::default()).unwrap();
    let cancellation = CancellationToken::new();
    let mut digests = Vec::new();
    for sample in samples {
        let outputs = decoder.submit(sample, &cancellation).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].presentation_index, sample.presentation_index);
        digests.push(FrameDigest::from_frame(&outputs[0].frame).unwrap());
    }
    assert!(decoder.drain(&cancellation).unwrap().is_empty());
    digests
}

#[test]
fn capability_honors_the_hardware_preference() {
    let factory = native_vp9_video_decoder_factory();
    let limits = Limits::default();
    let mut candidate = configuration(VideoDimensions::new(256, 144, &limits).unwrap(), Vec::new());
    assert_eq!(
        factory.capability(&candidate),
        CodecSupport::Supported {
            implementation: CodecImplementation::Software
        }
    );
    candidate.hardware = HardwarePreference::Prefer;
    let preferred = factory.capability(&candidate);
    candidate.hardware = HardwarePreference::Require;
    let required = factory.capability(&candidate);
    match preferred {
        CodecSupport::Supported {
            implementation: CodecImplementation::Hardware,
        } => {
            assert_eq!(required, preferred);
            assert!(factory.create(&candidate, &limits).is_ok());
        }
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        } => {
            assert_eq!(required, CodecSupport::HardwareUnavailable);
            let error = factory.create(&candidate, &limits).err().unwrap();
            assert_eq!(error.kind(), ErrorKind::Unsupported);
        }
        other => panic!("unexpected Prefer capability {other:?}"),
    }
    // Prefer decodes either way, in hardware or in software.
    candidate.hardware = HardwarePreference::Prefer;
    let track = mp4_track(include_bytes!("fixtures/codec/vp9_bbb_256x144.mp4"));
    let mut decoder = factory.create(&candidate, &limits).unwrap();
    let outputs = decoder
        .submit(&track.samples[0], &CancellationToken::new())
        .unwrap();
    assert_eq!(outputs.len(), 1);
}

#[test]
fn hardware_vp9_matches_the_software_decoder_and_seeks_exactly() {
    for (name, track) in fixtures() {
        let Some(hardware) = hardware_configuration(&track.configuration) else {
            return;
        };
        let expected = sequential_digests(&track.configuration, &track.samples);
        assert_eq!(
            sequential_digests(&hardware, &track.samples),
            expected,
            "{name}"
        );

        let factory = native_vp9_video_decoder_factory();
        let shown = expected.len() as u64;
        let mut alternating = Vec::new();
        let (mut low, mut high) = (0, shown);
        while low < high {
            high -= 1;
            alternating.push(high);
            if low < high {
                alternating.push(low);
                low += 1;
            }
        }
        let cancellation = CancellationToken::new();
        for order in [
            (0..shown).collect::<Vec<_>>(),
            (0..shown).rev().collect(),
            alternating,
        ] {
            let mut reader = ExactFrameReader::new(
                &factory,
                hardware.clone(),
                track.samples.clone(),
                Limits::default(),
            )
            .unwrap();
            for index in order {
                let frame = reader
                    .get(FrameIndex(index), &cancellation)
                    .unwrap_or_else(|error| panic!("{name} frame {index}: {error}"));
                assert_eq!(
                    FrameDigest::from_frame(&frame).unwrap(),
                    expected[index as usize],
                    "{name} frame {index}"
                );
            }
        }
    }
}

#[test]
fn hardware_show_existing_frame_shows_the_same_reference_as_software() {
    let track = mp4_track(include_bytes!("fixtures/codec/vp9_bbb_256x144.mp4"));
    let Some(hardware) = hardware_configuration(&track.configuration) else {
        return;
    };
    // The first frames, then each reference slot shown again by a
    // one-byte `show_existing_frame` chunk: frame marker 2, profile 0,
    // show_existing_frame 1, then the slot.
    let mut samples = track.samples[..4].to_vec();
    for slot in 0..8u8 {
        samples.push(EncodedVideoSample {
            presentation_index: FrameIndex(samples.len() as u64),
            random_access: false,
            data: vec![0x88 | slot],
        });
    }
    let expected = sequential_digests(&track.configuration, &samples);
    assert_eq!(sequential_digests(&hardware, &samples), expected);
}

#[test]
fn hardware_refuses_what_the_software_decoder_refuses() {
    let track = mp4_track(include_bytes!("fixtures/codec/vp9_bbb_256x144.mp4"));
    let Some(hardware) = hardware_configuration(&track.configuration) else {
        return;
    };
    let factory = native_vp9_video_decoder_factory();
    let cancellation = CancellationToken::new();
    let mut decoder = factory.create(&hardware, &Limits::default()).unwrap();
    // An inter frame before any key frame, and a superframe that shows
    // nothing.
    let error = decoder
        .submit(&track.samples[2], &cancellation)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::MalformedMedia);
    let hidden = track
        .samples
        .iter()
        .find(|sample| {
            sample
                .data
                .last()
                .is_some_and(|marker| marker & 0xe7 == 0xc1)
        })
        .expect("a superframe of two frames");
    let marker = *hidden.data.last().unwrap();
    let magnitude = usize::from((marker >> 3) & 3) + 1;
    let index = &hidden.data[hidden.data.len() - 2 - 2 * magnitude..];
    let first = (0..magnitude).fold(0, |size, byte| {
        size | usize::from(index[1 + byte]) << (8 * byte)
    });
    assert_eq!(
        decoder
            .submit(&track.samples[0], &cancellation)
            .unwrap()
            .len(),
        1
    );
    let hidden_only = EncodedVideoSample {
        data: hidden.data[..first].to_vec(),
        ..hidden.clone()
    };
    let error = decoder.submit(&hidden_only, &cancellation).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::MalformedMedia);

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let error = decoder.submit(&track.samples[0], &cancelled).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
}
