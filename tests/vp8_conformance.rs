//! Exact-frame VP8 decoding through the registered factory.
//!
//! `src/vp8/tests.rs` holds the decoder to libvpx's own MD5 of every shown
//! frame of the `vp80-00-comprehensive` vectors in `fixtures/codec/vp8/`.
//! This checks the other half of the guarantee: that every frame comes back
//! the same through `ExactFrameReader` whatever order it is asked for in, so
//! a seek decodes from the right key frame and through every hidden frame
//! before it.
//!
//! On a host with a hardware VP8 decoder, the hardware tests below hold it to
//! the software decoder's output for every frame of the same vectors, and to
//! the same exact-frame seeks. Elsewhere they skip and only the hardware
//! preference answers are checked.
#![cfg(not(target_arch = "wasm32"))]

use zvidlib::{
    CancellationToken, Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange,
    EncodedVideoSample, ErrorKind, ExactFrameReader, ExpectedVideoFrame, FrameDigest, FrameIndex,
    HardwarePreference, Limits, PixelFormat, PreviewIndex, PreviewOptions, VideoDecoderConfig,
    VideoDecoderConformanceVector, VideoDecoderFactory, VideoDimensions, WebmDemuxer,
    native_vp8_video_decoder_factory, verify_video_decoder_conformance,
};

struct IvfFrame<'a> {
    data: &'a [u8],
    key_frame: bool,
    shown: bool,
}

fn ivf(file: &[u8]) -> (VideoDimensions, Vec<IvfFrame<'_>>) {
    assert_eq!(&file[0..4], b"DKIF");
    assert_eq!(&file[8..12], b"VP80");
    let width = u16::from_le_bytes([file[12], file[13]]);
    let height = u16::from_le_bytes([file[14], file[15]]);
    let dimensions =
        VideoDimensions::new(u32::from(width), u32::from(height), &Limits::default()).unwrap();
    let mut frames = Vec::new();
    let mut position = usize::from(u16::from_le_bytes([file[6], file[7]]));
    while position + 12 <= file.len() {
        let size = u32::from_le_bytes(file[position..position + 4].try_into().unwrap()) as usize;
        let data = &file[position + 12..position + 12 + size];
        frames.push(IvfFrame {
            data,
            key_frame: data[0] & 1 == 0,
            shown: data[0] & 0x10 != 0,
        });
        position += 12 + size;
    }
    (dimensions, frames)
}

fn configuration(dimensions: VideoDimensions) -> VideoDecoderConfig {
    VideoDecoderConfig {
        codec: Codec::Vp8,
        profile: CodecProfile::Vp8,
        coded_dimensions: dimensions,
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Avoid,
        configuration: Vec::new(),
    }
}

/// One sample per IVF frame. Shown frames are numbered in order; a hidden
/// frame has no presentation of its own, so it takes an identity past the
/// last shown frame that nothing asks for.
fn samples(frames: &[IvfFrame<'_>]) -> Vec<EncodedVideoSample> {
    let shown = frames.iter().filter(|frame| frame.shown).count() as u64;
    let mut next_shown = 0;
    let mut next_hidden = shown;
    frames
        .iter()
        .map(|frame| {
            let counter = if frame.shown {
                &mut next_shown
            } else {
                &mut next_hidden
            };
            let presentation_index = FrameIndex(*counter);
            *counter += 1;
            EncodedVideoSample {
                presentation_index,
                random_access: frame.key_frame,
                data: frame.data.to_vec(),
            }
        })
        .collect()
}

/// The digest of every shown frame from one uninterrupted decode.
fn sequential_digests(
    factory: &dyn VideoDecoderFactory,
    configuration: &VideoDecoderConfig,
    samples: &[EncodedVideoSample],
) -> Vec<FrameDigest> {
    let mut decoder = factory.create(configuration, &Limits::default()).unwrap();
    let cancellation = CancellationToken::new();
    let mut digests = Vec::new();
    for sample in samples {
        for output in decoder.submit(sample, &cancellation).unwrap() {
            assert_eq!(output.presentation_index, sample.presentation_index);
            digests.push(FrameDigest::from_frame(&output.frame).unwrap());
        }
    }
    digests
}

macro_rules! vectors {
    ($($number:literal),* $(,)?) => {
        [$((
            concat!("vp80-00-comprehensive-", $number),
            &include_bytes!(concat!(
                "fixtures/codec/vp8/vp80-00-comprehensive-",
                $number,
                ".ivf"
            ))[..],
        )),*]
    };
}

#[test]
fn native_vp8_decoder_conforms_for_sequential_reverse_and_alternating_seeks() {
    let factory = native_vp8_video_decoder_factory();
    let limits = Limits::default();
    let vectors = vectors!(
        "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013",
        "014", "015", "016", "017",
    );
    for (name, file) in vectors {
        let (dimensions, frames) = ivf(file);
        assert!(frames.iter().all(|frame| frame.shown), "{name}");
        let configuration = configuration(dimensions);
        let samples = samples(&frames);
        let expected = sequential_digests(&factory, &configuration, &samples);
        assert_eq!(expected.len(), samples.len(), "{name}");
        let vector = VideoDecoderConformanceVector {
            name: name.into(),
            configuration,
            expected_frames: expected
                .iter()
                .enumerate()
                .map(|(index, &digest)| ExpectedVideoFrame {
                    presentation_index: FrameIndex(index as u64),
                    digest,
                })
                .collect(),
            samples,
        };
        let report = verify_video_decoder_conformance(&factory, &vector, limits)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(report.access_patterns_verified, 3, "{name}");
        assert_eq!(report.frames_verified, 3 * frames.len() as u64, "{name}");
    }
}

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

#[test]
fn a_webm_vp8_track_returns_every_frame_exactly() {
    let source = zvidlib::io::MemorySource::new(
        include_bytes!("fixtures/codec/vp8/vp8_testsrc2_98x66.webm").to_vec(),
    );
    let limits = Limits::default();
    let demuxer = block_on(WebmDemuxer::open(&source, Default::default())).unwrap();
    let track = &demuxer.tracks[0];
    assert_eq!(track.codec, Codec::Vp8);
    let dimensions = track.dimensions.unwrap();
    assert_eq!((dimensions.width, dimensions.height), (98, 66));
    let samples = block_on(track.to_encoded_video_samples(&source, &limits)).unwrap();
    assert_eq!(
        samples.iter().filter(|sample| sample.random_access).count(),
        3
    );
    let factory = native_vp8_video_decoder_factory();
    let configuration = VideoDecoderConfig {
        configuration: track.decoder_config.clone(),
        ..configuration(dimensions)
    };
    let expected = sequential_digests(&factory, &configuration, &samples);
    let vector = VideoDecoderConformanceVector {
        name: "VP8 in WebM".into(),
        configuration,
        expected_frames: expected
            .iter()
            .enumerate()
            .map(|(index, &digest)| ExpectedVideoFrame {
                presentation_index: FrameIndex(index as u64),
                digest,
            })
            .collect(),
        samples,
    };
    let report = verify_video_decoder_conformance(&factory, &vector, limits).unwrap();
    assert_eq!(report.frames_verified, 3 * 30);
}

#[test]
fn a_seek_decodes_through_a_hidden_key_frame() {
    // Vector 018 opens with a key frame that is never shown; every frame
    // after it predicts from it.
    let (dimensions, frames) = ivf(include_bytes!(
        "fixtures/codec/vp8/vp80-00-comprehensive-018.ivf"
    ));
    assert!(frames[0].key_frame && !frames[0].shown);
    assert!(frames[1..].iter().all(|frame| frame.shown));
    let factory = native_vp8_video_decoder_factory();
    let configuration = configuration(dimensions);
    let samples = samples(&frames);
    let expected = sequential_digests(&factory, &configuration, &samples);
    assert_eq!(expected.len(), frames.len() - 1);

    assert_every_order_matches(&factory, &configuration, &samples, &expected);
}

/// Sequential, reverse and alternating presentation orders over `shown`
/// frames.
fn access_orders(shown: u64) -> [Vec<u64>; 3] {
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
    [
        (0..shown).collect(),
        (0..shown).rev().collect(),
        alternating,
    ]
}

/// Reads every shown frame through a fresh `ExactFrameReader` in each order
/// of [`access_orders`] and checks it against `expected`.
fn assert_every_order_matches(
    factory: &dyn VideoDecoderFactory,
    configuration: &VideoDecoderConfig,
    samples: &[EncodedVideoSample],
    expected: &[FrameDigest],
) {
    let cancellation = CancellationToken::new();
    for order in access_orders(expected.len() as u64) {
        let mut reader = ExactFrameReader::new(
            factory,
            configuration.clone(),
            samples.to_vec(),
            Limits::default(),
        )
        .unwrap();
        for index in order {
            let frame = reader.get(FrameIndex(index), &cancellation).unwrap();
            assert_eq!(
                FrameDigest::from_frame(&frame).unwrap(),
                expected[index as usize],
                "frame {index}"
            );
        }
    }
}

#[test]
fn a_webm_vp8_track_skips_its_hidden_alternate_references() {
    // Issue #537: three of this track's 43 blocks are hidden alternate
    // references. `src/vp8/tests.rs` holds the 40 shown frames to libvpx's
    // MD5s.
    let source = zvidlib::io::MemorySource::new(
        include_bytes!("fixtures/codec/vp8/vp8_altref_98x66.webm").to_vec(),
    );
    let demuxer = block_on(WebmDemuxer::open(&source, Default::default())).unwrap();
    let track = &demuxer.tracks[0];
    assert_eq!(track.presentation_order.len(), 40);
    assert_eq!(track.samples.len(), 43);
    let samples = block_on(track.to_encoded_video_samples(&source, &Limits::default())).unwrap();
    let mut identities: Vec<u64> = samples
        .iter()
        .map(|sample| sample.presentation_index.0)
        .collect();
    identities.sort_unstable();
    assert_eq!(identities, (0..43).collect::<Vec<_>>());

    let factory = native_vp8_video_decoder_factory();
    let configuration = configuration(track.dimensions.unwrap());
    let expected = sequential_digests(&factory, &configuration, &samples);
    assert_eq!(expected.len(), 40);
    assert_every_order_matches(&factory, &configuration, &samples, &expected);
}

#[test]
fn a_preview_index_covers_only_the_shown_frames() {
    // Issue #543: the three hidden alternate references are decode-only
    // samples, so the index plans its slots over the 40 shown frames, and
    // every slot it plans is one a decode fills.
    let source = zvidlib::io::MemorySource::new(
        include_bytes!("fixtures/codec/vp8/vp8_altref_98x66.webm").to_vec(),
    );
    let demuxer = block_on(WebmDemuxer::open(&source, Default::default())).unwrap();
    let track = &demuxer.tracks[0];
    let samples = block_on(track.to_encoded_video_samples(&source, &Limits::default())).unwrap();
    assert_eq!(samples.len(), 43);
    // One preview per frame, so a slot planned for a hidden sample's identity
    // would show up in the total rather than vanish into a wider stride.
    let options = PreviewOptions {
        previews_per_second: 30,
        ..PreviewOptions::for_frame_rate(30)
    };
    let index = PreviewIndex::with_frame_count(
        &native_vp8_video_decoder_factory(),
        configuration(track.dimensions.unwrap()),
        samples,
        track.presentation_order.len() as u64,
        Limits::default(),
        options,
    )
    .unwrap();
    assert_eq!(index.store().stride(), 1);
    index.wait_for_coverage();
    assert_eq!(index.coverage(), (40, 40));
}

#[test]
fn capability_honors_the_hardware_preference() {
    let factory = native_vp8_video_decoder_factory();
    let limits = Limits::default();
    let mut candidate = configuration(VideoDimensions::new(176, 144, &limits).unwrap());
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
    candidate.hardware = HardwarePreference::Prefer;
    assert!(factory.create(&candidate, &limits).is_ok());
    candidate.hardware = HardwarePreference::Avoid;
    candidate.output_format = PixelFormat::Yuv420p8;
    assert!(matches!(
        factory.capability(&candidate),
        CodecSupport::InvalidConfiguration { .. }
    ));
    candidate.output_format = PixelFormat::Rgba8;
    candidate.profile = CodecProfile::Av1Main;
    assert_eq!(
        factory.capability(&candidate),
        CodecSupport::UnsupportedProfile
    );
    candidate.codec = Codec::Av1;
    assert_eq!(
        factory.capability(&candidate),
        CodecSupport::UnsupportedCodec
    );
}

#[test]
fn a_frame_of_the_wrong_size_or_a_malformed_frame_is_an_error() {
    let (_, frames) = ivf(include_bytes!(
        "fixtures/codec/vp8/vp80-00-comprehensive-001.ivf"
    ));
    let factory = native_vp8_video_decoder_factory();
    let limits = Limits::default();
    let cancellation = CancellationToken::new();
    let sample = |data: &[u8]| EncodedVideoSample {
        presentation_index: FrameIndex(0),
        random_access: true,
        data: data.to_vec(),
    };

    let wrong_size = configuration(VideoDimensions::new(320, 240, &limits).unwrap());
    let mut decoder = factory.create(&wrong_size, &limits).unwrap();
    let error = decoder
        .submit(&sample(frames[0].data), &cancellation)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::MalformedMedia);

    let right_size = configuration(VideoDimensions::new(176, 144, &limits).unwrap());
    let mut decoder = factory.create(&right_size, &limits).unwrap();
    let error = decoder
        .submit(&sample(&frames[0].data[..12]), &cancellation)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::MalformedMedia);

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let error = decoder
        .submit(&sample(frames[0].data), &cancelled)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
}

/// The hardware configuration for `dimensions`, or `None`, after saying why,
/// when this host has no hardware VP8 decoder.
fn hardware_configuration(dimensions: VideoDimensions) -> Option<VideoDecoderConfig> {
    let factory = native_vp8_video_decoder_factory();
    let configuration = VideoDecoderConfig {
        hardware: HardwarePreference::Require,
        ..configuration(dimensions)
    };
    if factory.capability(&configuration) == CodecSupport::HardwareUnavailable {
        let reason = factory
            .create(&configuration, &Limits::default())
            .err()
            .map_or_else(|| "unknown reason".into(), |error| error.to_string());
        eprintln!("skipping: hardware VP8 unavailable: {reason}");
        return None;
    }
    Some(configuration)
}

/// Every shown frame of `samples` through `ExactFrameReader` in sequential,
/// reverse and alternating order, each compared with `expected`.
fn assert_exact_seeks(
    name: &str,
    factory: &dyn VideoDecoderFactory,
    configuration: &VideoDecoderConfig,
    samples: &[EncodedVideoSample],
    expected: &[FrameDigest],
) {
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
            factory,
            configuration.clone(),
            samples.to_vec(),
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

#[test]
fn hardware_vp8_matches_the_software_decoder_and_seeks_exactly() {
    let factory = native_vp8_video_decoder_factory();
    let mut vectors = vectors!(
        "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013",
        "014", "015", "016", "017", "018",
    )
    .to_vec();
    vectors.push((
        "vp8_altref_98x66",
        &include_bytes!("fixtures/codec/vp8/vp8_altref_98x66.ivf")[..],
    ));
    for (name, file) in vectors {
        let (dimensions, frames) = ivf(file);
        let Some(hardware) = hardware_configuration(dimensions) else {
            return;
        };
        let software = configuration(dimensions);
        let samples = samples(&frames);
        let expected = sequential_digests(&factory, &software, &samples);
        assert_eq!(
            expected.len(),
            frames.iter().filter(|frame| frame.shown).count(),
            "{name}"
        );

        // One uninterrupted hardware decode: a hidden frame's sample returns
        // no picture, and every shown frame matches the software decoder's.
        let mut decoder = factory.create(&hardware, &Limits::default()).unwrap();
        let cancellation = CancellationToken::new();
        let mut actual = Vec::new();
        for (sample, frame) in samples.iter().zip(&frames) {
            let outputs = decoder
                .submit(sample, &cancellation)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(outputs.len(), usize::from(frame.shown), "{name}");
            for output in outputs {
                assert_eq!(output.presentation_index, sample.presentation_index);
                actual.push(FrameDigest::from_frame(&output.frame).unwrap());
            }
        }
        assert!(decoder.drain(&cancellation).unwrap().is_empty(), "{name}");
        assert_eq!(actual, expected, "{name}");

        assert_exact_seeks(name, &factory, &hardware, &samples, &expected);
    }
}

#[test]
fn a_hardware_webm_vp8_track_matches_the_software_decoder() {
    let source = zvidlib::io::MemorySource::new(
        include_bytes!("fixtures/codec/vp8/vp8_testsrc2_98x66.webm").to_vec(),
    );
    let limits = Limits::default();
    let demuxer = block_on(WebmDemuxer::open(&source, Default::default())).unwrap();
    let track = &demuxer.tracks[0];
    let dimensions = track.dimensions.unwrap();
    let Some(hardware) = hardware_configuration(dimensions) else {
        return;
    };
    let samples = block_on(track.to_encoded_video_samples(&source, &limits)).unwrap();
    let factory = native_vp8_video_decoder_factory();
    let expected = sequential_digests(&factory, &configuration(dimensions), &samples);
    assert_eq!(sequential_digests(&factory, &hardware, &samples), expected);
    assert_exact_seeks("VP8 in WebM", &factory, &hardware, &samples, &expected);
}
