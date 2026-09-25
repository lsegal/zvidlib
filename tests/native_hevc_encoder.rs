//! Issue #487: HEVC encoding through the platform encoder `Prefer` and
//! `Require` select, measured from the public factory. On a host with no
//! hardware HEVC encoder - every CI runner, today - `Require` must report
//! that plainly and the hardware cases skip; the Media Foundation backend's
//! own unit tests still exercise Microsoft's software encoder wherever it is
//! installed.

#![cfg(not(target_arch = "wasm32"))]

use std::future::Future;
use std::path::PathBuf;
use std::pin::pin;
use std::process::{Command, Stdio};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use zvidlib::io::{MemorySink, MemorySource};
use zvidlib::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::{
    CancellationToken, Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange,
    CpuFrameSource, EncodedSample, ExactFrameReader, FrameIndex, FrameSource, HardwarePreference,
    Limits, Mp4Demuxer, Mp4DemuxerOptions, Orientation, PixelFormat, Plane, VideoDecoderConfig,
    VideoDimensions, VideoEncoder, VideoEncoderConfig, VideoEncoderFactory, VideoFrame,
    native_hevc_video_decoder_factory, native_hevc_video_encoder_factory,
};

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let mut context = Context::from_waker(Waker::noop());
    let mut future = pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::yield_now();
    }
}

/// A 30 fps target-bitrate configuration, with a one-second keyframe interval
/// written out rather than left to the default.
fn configuration(
    width: u32,
    height: u32,
    input_format: PixelFormat,
    hardware: HardwarePreference,
) -> VideoEncoderConfig {
    let limits = Limits::default();
    let mut rate = 8_000_000_u32.to_be_bytes().to_vec();
    rate.extend_from_slice(&30_u32.to_be_bytes());
    VideoEncoderConfig {
        codec: Codec::Hevc,
        profile: CodecProfile::HevcMain,
        coded_dimensions: VideoDimensions::new(width, height, &limits).unwrap(),
        input_format,
        color_range: ColorRange::Limited,
        hardware,
        timescale: 30,
        frame_duration: 1,
        configuration: rate,
    }
}

/// The hardware encoder `Require` selects, or `None` with the reason logged.
fn hardware_encoder(configuration: &VideoEncoderConfig) -> Option<Box<dyn VideoEncoder>> {
    let factory = native_hevc_video_encoder_factory();
    if factory.capability(configuration) == CodecSupport::HardwareUnavailable {
        let reason = factory
            .create(configuration, &Limits::default())
            .err()
            .map_or_else(|| "unknown reason".into(), |error| error.to_string());
        eprintln!("skipping: hardware HEVC encoding unavailable: {reason}");
        return None;
    }
    let encoder = factory
        .create(configuration, &Limits::default())
        .expect("capability reported hardware HEVC encoding");
    assert_eq!(encoder.implementation(), CodecImplementation::Hardware);
    eprintln!("hardware HEVC encoder: {}", encoder.backend_name());
    Some(encoder)
}

/// Frame `index` of a moving RGBA gradient.
fn rgba_frame(width: u32, height: u32, index: u64) -> VideoFrame {
    let limits = Limits::default();
    let (w, h) = (width as usize, height as usize);
    let shift = index as usize * 8;
    let mut data = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            data.extend([
                ((x + shift) * 255 / w) as u8,
                (y * 255 / h) as u8,
                ((x + y + shift) % 256) as u8,
                255,
            ]);
        }
    }
    VideoFrame::new(
        VideoDimensions::new(width, height, &limits).unwrap(),
        PixelFormat::Rgba8,
        ColorRange::Limited,
        vec![Plane {
            data,
            stride: w * 4,
        }],
        &limits,
    )
    .unwrap()
}

fn encode_all(encoder: &mut dyn VideoEncoder, frames: &[VideoFrame]) -> Vec<EncodedSample> {
    let mut samples = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        samples.extend(
            block_on(encoder.encode(
                FrameIndex(index as u64),
                FrameSource::Cpu(CpuFrameSource {
                    frame,
                    orientation: Orientation::TopLeft,
                }),
            ))
            .unwrap(),
        );
    }
    samples.extend(block_on(encoder.finish()).unwrap());
    samples
}

fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

#[test]
fn the_factory_honours_hardware_preference_and_reports_the_implementation() {
    let factory = native_hevc_video_encoder_factory();
    let limits = Limits::default();

    let avoid = configuration(64, 64, PixelFormat::Rgba8, HardwarePreference::Avoid);
    assert_eq!(
        factory.capability(&avoid),
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        }
    );
    let encoder = factory.create(&avoid, &limits).unwrap();
    assert_eq!(encoder.implementation(), CodecImplementation::Software);
    assert_eq!(encoder.backend_name(), "zvidlib");

    let require = configuration(64, 64, PixelFormat::Rgba8, HardwarePreference::Require);
    let required = factory.capability(&require);
    let prefer = configuration(64, 64, PixelFormat::Rgba8, HardwarePreference::Prefer);
    let preferred = factory.capability(&prefer);
    assert!(preferred.is_supported(), "{preferred:?}");
    match required {
        CodecSupport::Supported {
            implementation: CodecImplementation::Hardware,
        } => {
            assert_eq!(preferred, required, "Prefer takes hardware when it exists");
            let encoder = factory.create(&require, &limits).unwrap();
            assert_eq!(encoder.implementation(), CodecImplementation::Hardware);
            assert_ne!(encoder.backend_name(), "zvidlib");
        }
        CodecSupport::HardwareUnavailable => {
            let error = factory.create(&require, &limits).err().unwrap();
            assert_eq!(error.kind(), zvidlib::ErrorKind::Unsupported, "{error}");
            // Prefer still produces an encoder: Media Foundation's software
            // one where it is installed, else the native one.
            let encoder = factory.create(&prefer, &limits).unwrap();
            assert_eq!(encoder.implementation(), CodecImplementation::Software);
        }
        other => panic!("Require answered {other:?}"),
    }

    // A lossless PCM configuration has no hardware form. `Require` cannot be
    // met, and `Prefer` quietly keeps the native encoder that can.
    let mut pcm = require.clone();
    pcm.configuration.clear();
    assert!(!factory.capability(&pcm).is_supported());
    assert!(factory.create(&pcm, &limits).is_err());
    pcm.hardware = HardwarePreference::Prefer;
    let encoder = factory.create(&pcm, &limits).unwrap();
    assert_eq!(encoder.implementation(), CodecImplementation::Software);
    assert_eq!(encoder.backend_name(), "zvidlib");
}

/// The acceptance path end to end: 1080p hardware HEVC muxed by `Mp4Muxer`
/// into an MP4 that zvidlib demuxes and decodes, and that an independent
/// ffprobe and ffmpeg read as HEVC Main with every frame intact.
#[test]
fn hardware_output_muxes_to_an_mp4_that_zvidlib_and_ffmpeg_both_decode() {
    const FRAMES: u64 = 45;
    let configuration = configuration(1920, 1080, PixelFormat::Rgba8, HardwarePreference::Require);
    let Some(mut encoder) = hardware_encoder(&configuration) else {
        return;
    };
    let limits = Limits::default();
    let frames: Vec<_> = (0..FRAMES)
        .map(|index| rgba_frame(1920, 1080, index))
        .collect();
    let samples = encode_all(encoder.as_mut(), &frames);
    assert_eq!(samples.len() as u64, FRAMES);
    let sync: Vec<_> = samples
        .iter()
        .enumerate()
        .filter(|(_, sample)| sample.is_sync)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(sync, [0, 30], "a keyframe every 30 frames");

    // The `hvcC` describes an 8-bit 4:2:0 Main stream with four-byte lengths
    // and carries one VPS, SPS and PPS.
    let hvcc = &encoder.config().decoder_config;
    assert_eq!(&hvcc[4..8], b"hvcC");
    assert_eq!(hvcc[8], 1, "configurationVersion");
    assert_eq!(hvcc[9] & 0x1f, 1, "general_profile_idc Main");
    assert_eq!(hvcc[8 + 16] & 3, 1, "chroma_format_idc 4:2:0");
    assert_eq!(hvcc[8 + 17] & 7, 0, "8-bit luma");
    assert_eq!(hvcc[8 + 18] & 7, 0, "8-bit chroma");
    assert_eq!(hvcc[8 + 21] & 3, 3, "four-byte NAL lengths");
    assert_eq!(hvcc[8 + 22], 3, "VPS, SPS and PPS arrays");

    let dimensions = configuration.coded_dimensions;
    let bytes = block_on(async {
        let mut muxer = Mp4Muxer::new(
            MemorySink::new(),
            vec![Mp4TrackConfig {
                encoder: encoder.config().clone(),
                format: Mp4TrackFormat::Video(dimensions),
            }],
            1_000,
        )
        .await
        .unwrap();
        for sample in samples {
            muxer.write_sample(0, sample).await.unwrap();
        }
        muxer.finish().await.unwrap().into_inner()
    });

    // zvidlib reads its own file back: the demuxed track is the stream that
    // was written, and the decoder reproduces the source at either end of
    // the second group of pictures.
    let source = MemorySource::new(bytes.clone());
    let demuxer = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
    let track = &demuxer.tracks[0];
    assert_eq!(track.codec, Codec::Hevc);
    assert_eq!(track.samples.len() as u64, FRAMES);
    assert_eq!(&track.decoder_config, &encoder.config().decoder_config);
    let encoded = block_on(track.to_encoded_video_samples(&source, &limits)).unwrap();
    let mut reader = ExactFrameReader::new(
        &native_hevc_video_decoder_factory(),
        VideoDecoderConfig {
            codec: Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: dimensions,
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: track.decoder_config.clone(),
        },
        encoded,
        limits,
    )
    .unwrap();
    for index in [0, 31, FRAMES - 1] {
        let decoded = reader
            .get(FrameIndex(index), &CancellationToken::new())
            .unwrap();
        let reference = &frames[index as usize].planes[0].data;
        let decoded = &decoded.planes[0].data;
        let squared: f64 = reference
            .iter()
            .zip(decoded)
            .enumerate()
            .filter(|(byte, _)| byte % 4 != 3)
            .map(|(_, (&a, &b))| (f64::from(a) - f64::from(b)).powi(2))
            .sum();
        let mse = squared / (reference.len() as f64 * 0.75);
        let psnr = 10.0 * (255.0_f64.powi(2) / mse.max(1e-9)).log10();
        assert!(psnr > 30.0, "frame {index}: PSNR {psnr:.1} dB");
    }

    if !tool_available("ffmpeg") || !tool_available("ffprobe") {
        eprintln!("skipping the independent decode because ffmpeg/ffprobe is unavailable");
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "zvidlib-hevc-encoder-{}-{:?}.mp4",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::write(&path, &bytes).unwrap();
    let _cleanup = RemoveOnDrop(path.clone());
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,codec_tag_string,profile,width,height,pix_fmt,nb_read_frames",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&path)
        .output()
        .unwrap();
    let report = String::from_utf8_lossy(&probe.stdout);
    assert!(probe.status.success(), "ffprobe failed: {report}");
    for expected in [
        "codec_name=hevc",
        "codec_tag_string=hvc1",
        "profile=Main",
        "width=1920",
        "height=1080",
        "pix_fmt=yuv420p",
        "nb_read_frames=45",
    ] {
        assert!(
            report.lines().any(|line| line == expected),
            "{expected} not in:\n{report}"
        );
    }
    let decode = Command::new("ffmpeg")
        .args(["-v", "error", "-xerror", "-i"])
        .arg(&path)
        .args(["-f", "null", "-"])
        .output()
        .unwrap();
    let errors = String::from_utf8_lossy(&decode.stderr);
    assert!(
        decode.status.success() && errors.trim().is_empty(),
        "ffmpeg: {errors}"
    );
}

struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The acceptance criterion the platform encoders exist for: 1080p30 HEVC
/// Main encodes faster than real time through hardware. Three seconds of
/// video, frames prepared ahead so only the encode is timed.
#[test]
fn hardware_encodes_1080p30_faster_than_real_time() {
    const FRAMES: u64 = 90;
    let configuration = configuration(1920, 1080, PixelFormat::Rgba8, HardwarePreference::Require);
    let Some(mut encoder) = hardware_encoder(&configuration) else {
        return;
    };
    let pictures: Vec<_> = (0..4).map(|index| rgba_frame(1920, 1080, index)).collect();
    let frames: Vec<_> = (0..FRAMES)
        .map(|index| pictures[index as usize % pictures.len()].clone())
        .collect();
    let started = Instant::now();
    let samples = encode_all(encoder.as_mut(), &frames);
    let elapsed = started.elapsed();
    assert_eq!(samples.len() as u64, FRAMES);
    let fps = FRAMES as f64 / elapsed.as_secs_f64();
    eprintln!(
        "{}: {FRAMES} frames of 1080p in {elapsed:?} ({fps:.0} fps)",
        encoder.backend_name()
    );
    assert!(
        elapsed < Duration::from_secs(FRAMES / 30),
        "1080p30 took {elapsed:?} for {} s of video",
        FRAMES / 30
    );
}
