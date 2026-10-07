//! Hardware VP9 decoding through the registered factory.
//!
//! `codec_conformance.rs` holds the factory to libvpx's decode of the VP9
//! fixtures under `HardwarePreference::Prefer`, which is the hardware decoder
//! wherever the host has one. On such a host these tests also require it:
//! every frame must match the software decoder's, hidden frames and
//! `show_existing_frame` included, and `ExactFrameReader` must return the
//! requested frames in sequential, reverse and alternating order. Elsewhere
//! they skip, and only the hardware preference answers are checked.
//!
//! The encode direction is checked the same way: where the host has a
//! hardware VP9 encoder, its stream muxes into MP4 and WebM exactly as the
//! software encoder's does and decodes through the native VP9 decoder.
#![cfg(not(target_arch = "wasm32"))]

use std::sync::{Mutex, MutexGuard, PoisonError};

use zvidlib_core::io::MemorySource;
use zvidlib_container::{FrameDigest, Mp4Demuxer, WebmDemuxer};
use zvidlib_core::{CancellationToken, Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange, EncodedVideoSample, ErrorKind, ExactFrameReader, FrameIndex, HardwarePreference, Limits, PixelFormat, VideoDecoderConfig, VideoDecoderFactory, VideoDimensions};
use zvidlib_vp9_decoder::native_vp9_video_decoder_factory;

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

/// Holds the other tests in this file off while one runs. The `macos-latest`
/// runner is a virtual Mac (`Apple M1 (Virtual)`), and its VideoToolbox VP9
/// decoder fails a decode with OSStatus -12909 or -19092 while another session
/// in the process is decoding: run in parallel, this file failed 9 times in 12
/// there, and run one test at a time it failed none (#565).
fn serial() -> MutexGuard<'static, ()> {
    static HARDWARE: Mutex<()> = Mutex::new(());
    HARDWARE.lock().unwrap_or_else(PoisonError::into_inner)
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
            mp4_track(include_bytes!("../../zvidlib-vp9-decoder/tests/fixtures/vp9_bbb_256x144.mp4")),
        ),
        (
            "VP9 250x142",
            mp4_track(include_bytes!("../../zvidlib-vp9-decoder/tests/fixtures/vp9_bbb_250x142.mp4")),
        ),
        (
            "VP9 in WebM",
            webm_track(include_bytes!("../../zvidlib-vp9-decoder/tests/fixtures/vp9_bbb_256x144.webm")),
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
    let _serial = serial();
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
    let track = mp4_track(include_bytes!("../../zvidlib-vp9-decoder/tests/fixtures/vp9_bbb_256x144.mp4"));
    let mut decoder = factory.create(&candidate, &limits).unwrap();
    let outputs = decoder
        .submit(&track.samples[0], &CancellationToken::new())
        .unwrap();
    assert_eq!(outputs.len(), 1);
}

#[test]
fn hardware_vp9_matches_the_software_decoder_and_seeks_exactly() {
    let _serial = serial();
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
    let _serial = serial();
    let track = mp4_track(include_bytes!("../../zvidlib-vp9-decoder/tests/fixtures/vp9_bbb_256x144.mp4"));
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

/// Hardware sessions on several threads at once, each required to decode
/// every fixture exactly as the software decoder does. On a virtual Mac,
/// VideoToolbox fails a VP9 decode while another session decodes, so the
/// backend must keep its sessions from overlapping there (#580).
#[test]
fn concurrent_hardware_vp9_sessions_match_the_software_decoder() {
    const THREADS: usize = 4;
    const ROUNDS: usize = 3;
    let _serial = serial();
    let mut streams = Vec::new();
    for (name, track) in fixtures() {
        let Some(hardware) = hardware_configuration(&track.configuration) else {
            return;
        };
        let expected = sequential_digests(&track.configuration, &track.samples);
        streams.push((name, hardware, track.samples, expected));
    }
    let start = std::sync::Barrier::new(THREADS);
    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            let (streams, start) = (&streams, &start);
            scope.spawn(move || {
                start.wait();
                for round in 0..ROUNDS {
                    // Each thread starts on a different fixture, so sessions
                    // of every size decode side by side.
                    for offset in 0..streams.len() {
                        let (name, hardware, samples, expected) =
                            &streams[(thread + round + offset) % streams.len()];
                        assert_eq!(
                            &sequential_digests(hardware, samples),
                            expected,
                            "{name}, thread {thread}, round {round}"
                        );
                    }
                }
            });
        }
    });
}

#[test]
fn hardware_refuses_what_the_software_decoder_refuses() {
    let _serial = serial();
    let track = mp4_track(include_bytes!("../../zvidlib-vp9-decoder/tests/fixtures/vp9_bbb_256x144.mp4"));
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

/// A 320x240 RGBA test card that pans four pixels a frame under a fixed block.
fn encoder_frame(index: u32, limits: &Limits) -> zvidlib_core::VideoFrame {
    let (width, height) = (320_u32, 240_u32);
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let u = x + index * 4;
            if (80..160).contains(&x) && (60..120).contains(&y) {
                data.extend_from_slice(&[200, 60, 40, 255]);
            } else {
                data.extend_from_slice(&[
                    (u * 255 / (width + 64)) as u8,
                    (y * 255 / height) as u8,
                    ((u + y) * 255 / (width + height + 64)) as u8,
                    255,
                ]);
            }
        }
    }
    zvidlib_core::VideoFrame::new(
        VideoDimensions::new(width, height, limits).unwrap(),
        PixelFormat::Rgba8,
        ColorRange::Limited,
        vec![zvidlib_core::Plane {
            data,
            stride: (width * 4) as usize,
        }],
        limits,
    )
    .unwrap()
}

/// Issue #550: `Require` selects the host's hardware VP9 encoder where it has
/// one. Its stream opens on a key frame, its `vpcC` describes 8-bit 4:2:0
/// profile 0, it muxes as `vp09`/`vpcC` in MP4 and `V_VP9` in WebM exactly as
/// the software encoder's does, and it passes the conformance runner against
/// the native VP9 decoder. A host without one skips with the reason, after the
/// preference answers are checked.
#[test]
fn hardware_vp9_encoder_output_muxes_and_decodes_through_the_native_decoder() {
    let _serial = serial();
    use zvidlib_core::io::MemorySink;
    use zvidlib_container::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
    use zvidlib_container::{VideoEncoderConformanceVector, WebmMuxer, verify_video_encoder_conformance};
    use zvidlib_core::{CpuFrameSource, FrameSource, Orientation, VideoEncoderConfig, VideoEncoderFactory};
    use zvidlib_vp9_encoder::native_vp9_video_encoder_factory;

    const FRAMES: u32 = 12;
    let limits = Limits::default();
    let dimensions = VideoDimensions::new(320, 240, &limits).unwrap();
    let factory = native_vp9_video_encoder_factory();
    let encoder_configuration = |hardware| VideoEncoderConfig {
        codec: Codec::Vp9,
        profile: CodecProfile::Vp9Profile0,
        coded_dimensions: dimensions,
        input_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware,
        timescale: 30,
        frame_duration: 1,
        configuration: vec![60, 0, 5],
    };
    let required = encoder_configuration(HardwarePreference::Require);
    let hardware = CodecSupport::Supported {
        implementation: CodecImplementation::Hardware,
    };
    let preferred = factory.capability(&encoder_configuration(HardwarePreference::Prefer));
    if factory.capability(&required) != hardware {
        assert_eq!(
            factory.capability(&required),
            CodecSupport::HardwareUnavailable
        );
        assert_eq!(
            preferred,
            CodecSupport::Supported {
                implementation: CodecImplementation::Software
            }
        );
        let reason = factory
            .create(&required, &limits)
            .err()
            .map_or_else(|| "unknown reason".into(), |error| error.to_string());
        eprintln!("skipping: hardware VP9 encoding unavailable: {reason}");
        return;
    }
    assert_eq!(preferred, hardware);

    let frames: Vec<_> = (0..FRAMES)
        .map(|index| encoder_frame(index, &limits))
        .collect();
    let mut encoder = factory.create(&required, &limits).unwrap();
    assert_eq!(encoder.implementation(), CodecImplementation::Hardware);
    eprintln!("hardware VP9 encoder: {}", encoder.backend_name());
    let vpcc = encoder.config().decoder_config.clone();
    let derived = zvidlib_container::derive_codec_string(Codec::Vp9, &vpcc).unwrap();
    assert_eq!(derived.profile, CodecProfile::Vp9Profile0);
    assert!(
        derived.codec_string.starts_with("vp09.00."),
        "{}",
        derived.codec_string
    );
    assert!(
        derived.codec_string.ends_with(".08"),
        "{}",
        derived.codec_string
    );
    let mut samples = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        let source = FrameSource::Cpu(CpuFrameSource {
            frame,
            orientation: Orientation::TopLeft,
        });
        samples.extend(block_on(encoder.encode(FrameIndex(index as u64), source)).unwrap());
    }
    samples.extend(block_on(encoder.finish()).unwrap());
    assert!(!samples.is_empty());
    assert!(samples[0].is_sync, "the stream opens on a key frame");
    assert_eq!(samples[0].dts, 0);
    for pair in samples.windows(2) {
        assert_eq!(pair[0].dts + i64::from(pair[0].duration), pair[1].dts);
    }
    let last = samples.last().unwrap();
    assert_eq!(last.dts + i64::from(last.duration), i64::from(FRAMES));

    let track = Mp4TrackConfig {
        encoder: encoder.config().clone(),
        format: Mp4TrackFormat::Video(dimensions),
    };
    let mut mp4 = block_on(Mp4Muxer::new(MemorySink::new(), vec![track.clone()], 64)).unwrap();
    let mut webm = block_on(WebmMuxer::new(MemorySink::new(), vec![track], 64)).unwrap();
    for sample in &samples {
        block_on(mp4.write_sample(0, sample.clone())).unwrap();
        block_on(webm.write_sample(0, sample.clone())).unwrap();
    }
    let mp4 = block_on(mp4.finish()).unwrap().into_inner();
    let webm = block_on(webm.finish()).unwrap().into_inner();
    let movie = block_on(Mp4Demuxer::open(
        &MemorySource::new(mp4),
        Default::default(),
    ))
    .unwrap();
    let demuxed = &movie.tracks[0];
    assert_eq!(demuxed.codec, Codec::Vp9);
    assert_eq!(demuxed.decoder_config, vpcc);
    assert_eq!(demuxed.samples.len(), samples.len());
    let matroska = block_on(WebmDemuxer::open(
        &MemorySource::new(webm),
        Default::default(),
    ))
    .unwrap();
    let demuxed_webm = &matroska.tracks[0];
    assert_eq!(demuxed_webm.codec, Codec::Vp9);
    assert_eq!(demuxed_webm.samples.len(), samples.len());
    for (index, sample) in samples.iter().enumerate() {
        assert_eq!(
            demuxed.samples[index].is_sync, sample.is_sync,
            "MP4 sample {index}"
        );
        assert_eq!(
            demuxed_webm.samples[index].is_sync, sample.is_sync,
            "WebM sample {index}"
        );
    }

    let vector = VideoEncoderConformanceVector {
        name: "vp9-hardware-rgba-pan".into(),
        configuration: required,
        decoder_configuration: configuration(dimensions, Vec::new()),
        frames,
        minimum_psnr_db: 30.0,
    };
    let report = block_on(verify_video_encoder_conformance(
        &factory,
        &native_vp9_video_decoder_factory(),
        &vector,
        limits,
    ))
    .unwrap();
    assert_eq!(report.frames_encoded, u64::from(FRAMES));
}
