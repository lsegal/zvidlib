#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use zvidlib::hevc_hardware_readback as readback;
use zvidlib::io::{MemorySink, MemorySource};
use zvidlib::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::{CancellationToken, ExactFrameReader, FrameIndex};
use zvidlib::{
    Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange, CpuFrameSource,
    EncodedSample, EncodedVideoSample, FrameDigest, FrameSource, HardwarePreference, Limits,
    Mp4Demuxer, Mp4DemuxerOptions, Orientation, PixelFormat, Plane, VideoDecoderConfig,
    VideoDecoderConformanceVector, VideoDecoderFactory, VideoDimensions, VideoEncoder,
    VideoEncoderConfig, VideoEncoderConformanceVector, VideoEncoderFactory, VideoFrame,
    native_hevc_video_decoder_factory, native_hevc_video_encoder_factory,
    verify_video_encoder_conformance,
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

/// Issue #374: what a cold seek to an arbitrary frame of the bundled sample costs, on a hardware
/// backend, as a number rather than an impression.
///
/// The sample's `stss` names one sync sample for 768 frames, so every one of these is a walk from
/// frame zero. Ignored because it is a wall-clock reading on whatever host runs it.
#[test]
#[ignore = "host-specific seek-latency measurement"]
fn a_cold_hardware_seek_costs_this_much() {
    let mut vector = bundled_vector();
    vector.configuration.hardware = HardwarePreference::Require;
    let factory = native_hevc_video_decoder_factory();
    if factory.capability(&vector.configuration) == CodecSupport::HardwareUnavailable {
        eprintln!("skipping: hardware HEVC unavailable");
        return;
    }
    let cancellation = CancellationToken::new();
    for target in [76_u64, 384, 767] {
        let mut reader = ExactFrameReader::new(
            &factory,
            vector.configuration.clone(),
            vector.samples.clone(),
            Limits::default(),
        )
        .unwrap();
        let started = Instant::now();
        let frame = reader.get(FrameIndex(target), &cancellation).unwrap();
        let elapsed = started.elapsed();
        assert_eq!(
            FrameDigest::from_frame(&frame).unwrap(),
            vector.expected_frames[target as usize].digest
        );
        let statistics = reader.statistics();
        eprintln!(
            "cold seek to frame {target}: {elapsed:?} ({} submitted, {} skipped)",
            statistics.samples_submitted, statistics.samples_skipped
        );
    }
}

// ---- Hardware HEVC encoding (issue #486) ----
//
// What a stream from the hardware encoder `Require` selects has to survive: the encoder
// conformance runner, `Mp4Muxer` and `Mp4Demuxer`, BGRA and bottom-up input, and being abandoned
// mid-stream. Which encoder a preference selects, and the 1080p30 real-time bar, are
// `tests/native_hevc_encoder.rs`. Written against VideoToolbox, the backend a hosted runner can
// reach (`macos-latest`); every test skips itself on a host with no hardware encoder, the way
// the decoder tests above do.

/// A 30 fps clock.
const TIMESCALE: u32 = 90_000;
const FRAME_DURATION: u32 = 3_000;

fn bitrate(bits_per_second: u32, keyframe_interval: Option<u32>) -> Vec<u8> {
    let mut configuration = bits_per_second.to_be_bytes().to_vec();
    if let Some(interval) = keyframe_interval {
        configuration.extend_from_slice(&interval.to_be_bytes());
    }
    configuration
}

fn encoder_configuration(
    (width, height): (u32, u32),
    input_format: PixelFormat,
    hardware: HardwarePreference,
    configuration: Vec<u8>,
) -> VideoEncoderConfig {
    VideoEncoderConfig {
        codec: Codec::Hevc,
        profile: CodecProfile::HevcMain,
        coded_dimensions: VideoDimensions::new(width, height, &Limits::default()).unwrap(),
        input_format,
        color_range: ColorRange::Limited,
        hardware,
        timescale: TIMESCALE,
        frame_duration: FRAME_DURATION,
        configuration,
    }
}

/// Whether the host encodes `configuration` in hardware, printing why not when it does not.
fn hardware_encoder_available(configuration: &VideoEncoderConfig) -> bool {
    let support = native_hevc_video_encoder_factory().capability(configuration);
    if support
        == (CodecSupport::Supported {
            implementation: CodecImplementation::Hardware,
        })
    {
        return true;
    }
    eprintln!("skipping: hardware HEVC encoding unavailable: {support:?}");
    false
}

/// Smooth moving content: a diagonal gradient with a bright square travelling across it, so every
/// frame differs from the last and an encoder has real motion to code.
fn rgba_frame((width, height): (u32, u32), index: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let square = (h / 4).max(8);
    let left = (index as usize * 7) % (w - square);
    let top = (index as usize * 3) % (h - square);
    let mut pixels = vec![0_u8; w * h * 4];
    for (offset, pixel) in pixels.chunks_exact_mut(4).enumerate() {
        let (x, y) = (offset % w, offset / w);
        if (left..left + square).contains(&x) && (top..top + square).contains(&y) {
            pixel.copy_from_slice(&[230, 200, 60, 255]);
        } else {
            let r = 40 + (x * 150 / w) as u8;
            let g = 40 + (y * 150 / h) as u8;
            let b = 40 + ((x + y) * 100 / (w + h)) as u8;
            pixel.copy_from_slice(&[r, g, b, 255]);
        }
    }
    pixels
}

fn video_frame(size: (u32, u32), format: PixelFormat, data: Vec<u8>) -> VideoFrame {
    let limits = Limits::default();
    VideoFrame::new(
        VideoDimensions::new(size.0, size.1, &limits).unwrap(),
        format,
        ColorRange::Limited,
        vec![Plane {
            data,
            stride: size.0 as usize * 4,
        }],
        &limits,
    )
    .unwrap()
}

/// Encodes every frame through `encoder` and finishes it, returning the samples in order.
fn encode_all(
    encoder: &mut dyn VideoEncoder,
    frames: &[VideoFrame],
    orientation: Orientation,
) -> Vec<EncodedSample> {
    let mut samples = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        samples.extend(
            block_on(encoder.encode(
                FrameIndex(index as u64),
                FrameSource::Cpu(CpuFrameSource { frame, orientation }),
            ))
            .unwrap(),
        );
    }
    samples.extend(block_on(encoder.finish()).unwrap());
    samples
}

/// PSNR over the colour channels of an RGBA source and a decoded RGBA frame of the same size.
fn rgba_psnr(expected: &[u8], actual: &VideoFrame) -> f64 {
    let plane = &actual.planes[0];
    let width = actual.dimensions.width as usize * 4;
    let mut squared = 0.0;
    let mut count = 0.0;
    for row in 0..actual.dimensions.height as usize {
        let decoded = &plane.data[row * plane.stride..][..width];
        let source = &expected[row * width..][..width];
        for (index, (&a, &b)) in source.iter().zip(decoded).enumerate() {
            if index % 4 != 3 {
                let difference = f64::from(a) - f64::from(b);
                squared += difference * difference;
                count += 1.0;
            }
        }
    }
    if squared == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0 * 255.0 / (squared / count)).log10()
}

/// The `hvcC` a stream declares, checked field by field against ISO/IEC 14496-15 for 8-bit 4:2:0
/// Main with four-byte NAL lengths and one array each of VPS, SPS, and PPS.
fn assert_main_profile_hvcc(hvcc: &[u8]) {
    assert_eq!(&hvcc[4..8], b"hvcC");
    assert_eq!(
        u32::from_be_bytes(hvcc[..4].try_into().unwrap()) as usize,
        hvcc.len()
    );
    let body = &hvcc[8..];
    assert_eq!(body[0], 1, "configurationVersion");
    assert_eq!(body[1] & 0x1f, 1, "general_profile_idc is Main");
    assert_eq!(body[16] & 0x03, 1, "chroma_format_idc is 4:2:0");
    assert_eq!(body[17] & 0x07, 0, "8-bit luma");
    assert_eq!(body[18] & 0x07, 0, "8-bit chroma");
    assert_eq!(body[21] & 0x03, 3, "four-byte NAL unit lengths");
    let mut rest = &body[23..];
    let mut kinds = Vec::new();
    for _ in 0..body[22] {
        let kind = rest[0] & 0x3f;
        let count = u16::from_be_bytes([rest[1], rest[2]]);
        assert!(count > 0, "hvcC array of type {kind} is empty");
        rest = &rest[3..];
        for _ in 0..count {
            let length = u16::from_be_bytes([rest[0], rest[1]]) as usize;
            assert_eq!(
                (rest[2] >> 1) & 0x3f,
                kind,
                "NAL unit filed under its own type"
            );
            rest = &rest[2 + length..];
        }
        kinds.push(kind);
    }
    assert!(rest.is_empty(), "hvcC has trailing bytes");
    assert_eq!(kinds, [32, 33, 34]);
}

/// Samples an `hvc1` track can carry: contiguous from zero, in presentation order, covering
/// exactly `frames` frame durations, opening on a sync sample, with pictures only.
fn assert_muxable_samples(samples: &[EncodedSample], frames: u64) {
    assert!(!samples.is_empty());
    assert!(samples[0].is_sync, "the stream opens on a keyframe");
    let mut next = 0_i64;
    for sample in samples {
        assert_eq!(sample.dts, next, "decode timestamps are contiguous");
        assert_eq!(sample.pts, sample.dts, "no frame reordering");
        assert!(sample.duration > 0);
        let mut rest = sample.data.as_slice();
        while !rest.is_empty() {
            let length = u32::from_be_bytes(rest[..4].try_into().unwrap()) as usize;
            let kind = (rest[4] >> 1) & 0x3f;
            assert!(
                !(32..=35).contains(&kind),
                "parameter set or delimiter type {kind} left in a sample"
            );
            rest = &rest[4 + length..];
        }
        next += i64::from(sample.duration);
    }
    assert_eq!(next, frames as i64 * i64::from(FRAME_DURATION));
}

fn software_decoder_configuration(
    coded_dimensions: VideoDimensions,
    hvcc: Vec<u8>,
) -> VideoDecoderConfig {
    VideoDecoderConfig {
        codec: Codec::Hevc,
        profile: CodecProfile::HevcMain,
        coded_dimensions,
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Avoid,
        configuration: hvcc,
    }
}

/// The acceptance round trip: a hardware stream passes the crate's encoder conformance runner
/// against the software decoder, declares a standard Main-profile `hvcC`, keeps keyframes at the
/// requested interval, and survives `Mp4Muxer` and `Mp4Demuxer` unchanged.
#[test]
fn hardware_hevc_round_trips_through_the_muxer_and_software_decoder() {
    // Not a multiple of 16, so the encoder's conformance window is exercised as well.
    let size = (640, 360);
    let frame_count = 30_u32;
    let configuration = encoder_configuration(
        size,
        PixelFormat::Rgba8,
        HardwarePreference::Require,
        bitrate(4_000_000, Some(10)),
    );
    if !hardware_encoder_available(&configuration) {
        return;
    }
    let limits = Limits::default();
    let sources: Vec<Vec<u8>> = (0..frame_count)
        .map(|index| rgba_frame(size, index))
        .collect();
    let frames: Vec<VideoFrame> = sources
        .iter()
        .map(|data| video_frame(size, PixelFormat::Rgba8, data.clone()))
        .collect();
    let report = block_on(verify_video_encoder_conformance(
        &native_hevc_video_encoder_factory(),
        &native_hevc_video_decoder_factory(),
        &VideoEncoderConformanceVector {
            name: "VideoToolbox HEVC Main".into(),
            configuration: configuration.clone(),
            decoder_configuration: software_decoder_configuration(
                configuration.coded_dimensions,
                Vec::new(),
            ),
            frames: frames.clone(),
            minimum_psnr_db: 28.0,
        },
        limits,
    ))
    .unwrap();
    eprintln!(
        "VideoToolbox HEVC round trip: {} frames, {} packets, minimum PSNR {:.2} dB",
        report.frames_encoded, report.packets_emitted, report.minimum_observed_psnr_db
    );

    let mut encoder = native_hevc_video_encoder_factory()
        .create(&configuration, &limits)
        .unwrap();
    let encoder_config = encoder.config().clone();
    assert_eq!(encoder_config.codec, Codec::Hevc);
    assert_eq!(encoder_config.timescale, TIMESCALE);
    assert_main_profile_hvcc(&encoder_config.decoder_config);
    let samples = encode_all(encoder.as_mut(), &frames, Orientation::TopLeft);
    assert_muxable_samples(&samples, u64::from(frame_count));
    let keyframes: Vec<i64> = samples
        .iter()
        .filter(|sample| sample.is_sync)
        .map(|sample| sample.dts / i64::from(FRAME_DURATION))
        .collect();
    assert!(
        keyframes.windows(2).all(|pair| pair[1] - pair[0] <= 10)
            && i64::from(frame_count) - 1 - keyframes.last().unwrap() < 10,
        "a keyframe at least every 10 frames: {keyframes:?}"
    );

    let muxed = block_on(async {
        let mut muxer = Mp4Muxer::new(
            MemorySink::new(),
            vec![Mp4TrackConfig {
                encoder: encoder_config.clone(),
                format: Mp4TrackFormat::Video(configuration.coded_dimensions),
            }],
            1024,
        )
        .await
        .unwrap();
        for sample in samples.clone() {
            muxer.write_sample(0, sample).await.unwrap();
        }
        muxer.finish().await.unwrap().into_inner()
    });
    let source = MemorySource::new(muxed);
    let demuxer = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
    let track = &demuxer.tracks[0];
    assert_eq!(track.codec, Codec::Hevc);
    assert_eq!(track.decoder_config, encoder_config.decoder_config);
    assert_eq!(track.samples.len(), samples.len());
    let demuxed = block_on(track.to_encoded_video_samples(&source, &limits)).unwrap();
    for (sample, original) in demuxed.iter().zip(&samples) {
        assert_eq!(sample.data, original.data);
        assert_eq!(sample.random_access, original.is_sync);
    }

    let mut reader = ExactFrameReader::new(
        &native_hevc_video_decoder_factory(),
        software_decoder_configuration(
            configuration.coded_dimensions,
            track.decoder_config.clone(),
        ),
        demuxed,
        limits,
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    for sample in &samples {
        let index = (sample.dts / i64::from(FRAME_DURATION)) as usize;
        let decoded = reader.get(FrameIndex(index as u64), &cancellation).unwrap();
        let psnr = rgba_psnr(&sources[index], &decoded);
        assert!(
            psnr >= 28.0,
            "demuxed frame {index} decoded at {psnr:.2} dB"
        );
    }
}

/// `Bgra8` is copied into the encoder's pixel buffers as it is and `Rgba8` is swizzled, and a
/// bottom-up source is flipped on the way in, so all of them come back as the same picture.
#[test]
fn hardware_hevc_accepts_bgra_and_bottom_up_input() {
    let size = (320, 240);
    let frame_count = 8_u32;
    let configuration = encoder_configuration(
        size,
        PixelFormat::Bgra8,
        HardwarePreference::Require,
        bitrate(4_000_000, None),
    );
    if !hardware_encoder_available(&configuration) {
        return;
    }
    let limits = Limits::default();
    let row = size.0 as usize * 4;
    let sources: Vec<Vec<u8>> = (0..frame_count)
        .map(|index| rgba_frame(size, index))
        .collect();
    let bottom_up_bgra: Vec<VideoFrame> = sources
        .iter()
        .map(|rgba| {
            let mut bgra = Vec::with_capacity(rgba.len());
            for source_row in rgba.chunks_exact(row).rev() {
                for pixel in source_row.chunks_exact(4) {
                    bgra.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
                }
            }
            video_frame(size, PixelFormat::Bgra8, bgra)
        })
        .collect();
    let mut encoder = native_hevc_video_encoder_factory()
        .create(&configuration, &limits)
        .unwrap();
    assert_eq!(encoder.format().pixel_format, PixelFormat::Bgra8);
    let hvcc = encoder.config().decoder_config.clone();
    let samples = encode_all(encoder.as_mut(), &bottom_up_bgra, Orientation::BottomLeft);
    assert_muxable_samples(&samples, u64::from(frame_count));

    let mut reader = ExactFrameReader::new(
        &native_hevc_video_decoder_factory(),
        software_decoder_configuration(configuration.coded_dimensions, hvcc),
        samples
            .iter()
            .map(|sample| EncodedVideoSample {
                presentation_index: FrameIndex((sample.dts / i64::from(FRAME_DURATION)) as u64),
                random_access: sample.is_sync,
                data: sample.data.clone(),
            })
            .collect(),
        limits,
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    for sample in &samples {
        let index = (sample.dts / i64::from(FRAME_DURATION)) as usize;
        let decoded = reader.get(FrameIndex(index as u64), &cancellation).unwrap();
        let psnr = rgba_psnr(&sources[index], &decoded);
        assert!(
            psnr >= 28.0,
            "BGRA bottom-up frame {index} decoded at {psnr:.2} dB"
        );
    }
}

/// Dropping an encoder that was never finished is how a recording is abandoned. It has to tear
/// the session down without waiting for, or emitting, the frames still inside it and leave the
/// hardware usable; and an encoder that was finished refuses further frames.
#[test]
fn an_unfinished_hardware_encoder_is_cancelled_by_dropping_it() {
    let size = (1280, 720);
    let configuration = encoder_configuration(
        size,
        PixelFormat::Rgba8,
        HardwarePreference::Require,
        bitrate(6_000_000, None),
    );
    if !hardware_encoder_available(&configuration) {
        return;
    }
    let frame = video_frame(size, PixelFormat::Rgba8, rgba_frame(size, 0));
    let source = || {
        FrameSource::Cpu(CpuFrameSource {
            frame: &frame,
            orientation: Orientation::TopLeft,
        })
    };
    let factory = native_hevc_video_encoder_factory();
    let mut abandoned = factory.create(&configuration, &Limits::default()).unwrap();
    for index in 0..10 {
        block_on(abandoned.encode(FrameIndex(index), source())).unwrap();
    }
    drop(abandoned);

    let mut finished = factory.create(&configuration, &Limits::default()).unwrap();
    let mut samples = block_on(finished.encode(FrameIndex(0), source())).unwrap();
    samples.extend(block_on(finished.finish()).unwrap());
    assert_muxable_samples(&samples, 1);
    assert!(block_on(finished.encode(FrameIndex(1), source())).is_err());
}
