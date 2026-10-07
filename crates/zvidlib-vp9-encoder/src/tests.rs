use super::*;
use crate::{CpuFrameSource, Plane};
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let mut context = Context::from_waker(Waker::noop());
    let mut future = pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
    }
}

fn configuration(width: u32, height: u32, input_format: PixelFormat) -> VideoEncoderConfig {
    VideoEncoderConfig {
        codec: Codec::Vp9,
        profile: CodecProfile::Vp9Profile0,
        coded_dimensions: VideoDimensions { width, height },
        input_format,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Avoid,
        timescale: 30_000,
        frame_duration: 1_001,
        configuration: Vec::new(),
    }
}

/// A textured scene that pans four samples a frame (two in chroma) under a
/// square that moves two samples a frame, so inter frames find real motion.
fn moving_yuv_frame(width: u32, height: u32, index: u32) -> VideoFrame {
    let (width, height) = (width as usize, height as usize);
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let shift = index as usize * 4;
    let mut luma = vec![0_u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let u = x + shift;
            let texture = ((u * 7) ^ (y * 13)) & 0x3f;
            let mut value = 40 + (u * 160 / (width + 30)) + texture;
            let square = 10 + index as usize * 2;
            if (square..square + 12).contains(&x) && (8..20).contains(&y) {
                value = 230 - ((x + y) & 7) * 4;
            }
            luma[y * width + x] = value.min(255) as u8;
        }
    }
    let cb = (0..chroma_width * chroma_height)
        .map(|i| (100 + (i % chroma_width + shift / 2) * 50 / (chroma_width + 30)) as u8)
        .collect();
    let cr = (0..chroma_width * chroma_height)
        .map(|i| (150 - (i / chroma_width) % 40) as u8)
        .collect();
    VideoFrame::new(
        VideoDimensions {
            width: width as u32,
            height: height as u32,
        },
        PixelFormat::Yuv420p8,
        ColorRange::Limited,
        vec![
            Plane {
                data: luma,
                stride: width,
            },
            Plane {
                data: cb,
                stride: chroma_width,
            },
            Plane {
                data: cr,
                stride: chroma_width,
            },
        ],
        &Limits::default(),
    )
    .unwrap()
}

/// Smooth gradients under a soft highlight, drifting one sample a frame: the
/// flat, slowly varying content that larger blocks and transforms code cheaply.
fn smooth_yuv_frame(width: u32, height: u32, index: u32) -> VideoFrame {
    let (width, height) = (width as usize, height as usize);
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let shift = index as f64;
    let mut luma = vec![0_u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let (u, v) = (x as f64 + shift, y as f64);
            let highlight = (-((u - 70.0).powi(2) + (v - 40.0).powi(2)) / 900.0).exp();
            let value = 50.0 + u * 0.6 + v * 0.4 + 90.0 * highlight;
            luma[y * width + x] = value.round().clamp(0.0, 255.0) as u8;
        }
    }
    let chroma = |base: f64, slope: f64| -> Vec<u8> {
        (0..chroma_width * chroma_height)
            .map(|i| {
                let (x, y) = (
                    (i % chroma_width) as f64 + shift / 2.0,
                    (i / chroma_width) as f64,
                );
                (base + slope * x - 0.2 * y).round().clamp(0.0, 255.0) as u8
            })
            .collect()
    };
    VideoFrame::new(
        VideoDimensions {
            width: width as u32,
            height: height as u32,
        },
        PixelFormat::Yuv420p8,
        ColorRange::Limited,
        vec![
            Plane {
                data: luma,
                stride: width,
            },
            Plane {
                data: chroma(110.0, 0.3),
                stride: chroma_width,
            },
            Plane {
                data: chroma(150.0, -0.25),
                stride: chroma_width,
            },
        ],
        &Limits::default(),
    )
    .unwrap()
}

/// Smooth detail that pans 1.375 samples right and 0.625 down a frame, so
/// only sub-sample motion vectors predict it well.
fn sub_sample_yuv_frame(width: u32, height: u32, index: u32) -> VideoFrame {
    let (width, height) = (width as usize, height as usize);
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let (dx, dy) = (f64::from(index) * 1.375, f64::from(index) * 0.625);
    let scene = |x: f64, y: f64| {
        128.0
            + 45.0 * (0.41 * (x - dx)).sin() * (0.23 * (y - dy)).cos()
            + 30.0 * (0.17 * (x - dx) + 0.29 * (y - dy)).sin()
    };
    let sample = |value: f64| value.round().clamp(0.0, 255.0) as u8;
    let luma = (0..width * height)
        .map(|i| sample(scene((i % width) as f64, (i / width) as f64)))
        .collect();
    let chroma = |offset: f64| -> Vec<u8> {
        (0..chroma_width * chroma_height)
            .map(|i| {
                let (x, y) = ((i % chroma_width) as f64, (i / chroma_width) as f64);
                sample(128.0 + 0.4 * (scene(x * 2.0 + 0.5, y * 2.0 + 0.5) - 128.0) + offset)
            })
            .collect()
    };
    VideoFrame::new(
        VideoDimensions {
            width: width as u32,
            height: height as u32,
        },
        PixelFormat::Yuv420p8,
        ColorRange::Limited,
        vec![
            Plane {
                data: luma,
                stride: width,
            },
            Plane {
                data: chroma(-10.0),
                stride: chroma_width,
            },
            Plane {
                data: chroma(10.0),
                stride: chroma_width,
            },
        ],
        &Limits::default(),
    )
    .unwrap()
}

/// The visible `yuv420p` bytes of the encoder's reconstruction.
/// The visible bytes of the last frame `encoder` emitted.
fn reconstruction(encoder: &NativeVp9Encoder) -> Vec<u8> {
    visible(encoder.reconstructions.last().unwrap(), &encoder.geometry)
}

type Frames = fn(u32, u32, u32) -> VideoFrame;

fn encode_sequence(
    config: &VideoEncoderConfig,
    frames: u32,
) -> (Vec<EncodedSample>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    encode_with(config, frames, moving_yuv_frame, CodingTools::ALL)
}

/// Encodes `frames` frames of `content` with the given coding tools, returning
/// the samples, the encoder's reconstructions and the sources.
fn encode_with(
    config: &VideoEncoderConfig,
    frames: u32,
    content: Frames,
    tools: CodingTools,
) -> (Vec<EncodedSample>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut encoder = NativeVp9Encoder::new(config, &Limits::default()).unwrap();
    encoder.tools = tools;
    let mut samples = Vec::new();
    let mut sources = Vec::new();
    let (width, height) = (
        config.coded_dimensions.width,
        config.coded_dimensions.height,
    );
    for index in 0..frames {
        let frame = content(width, height, index);
        sources.push(
            frame
                .planes
                .iter()
                .flat_map(|plane| plane.data.clone())
                .collect(),
        );
        let source = FrameSource::Cpu(CpuFrameSource {
            frame: &frame,
            orientation: Orientation::TopLeft,
        });
        samples.extend(block_on(encoder.encode(FrameIndex(u64::from(index)), source)).unwrap());
    }
    samples.extend(block_on(encoder.finish()).unwrap());
    assert_eq!(samples.len(), frames as usize);
    let reconstructions = encoder
        .reconstructions
        .iter()
        .map(|picture| visible(picture, &encoder.geometry))
        .collect();
    (samples, reconstructions, sources)
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let error: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0 * 255.0 / error.max(1e-9)).log10()
}

#[test]
fn factory_advertises_only_the_implemented_surface() {
    let factory = native_vp9_video_encoder_factory();
    assert!(
        factory
            .capability(&configuration(64, 48, PixelFormat::Yuv420p8))
            .is_supported()
    );
    for format in [PixelFormat::Rgba8, PixelFormat::Bgra8, PixelFormat::Gray8] {
        assert!(
            factory
                .capability(&configuration(17, 9, format))
                .is_supported()
        );
    }
    let mut invalid = configuration(64, 48, PixelFormat::Rgb8);
    assert!(matches!(
        factory.capability(&invalid),
        CodecSupport::InvalidConfiguration { .. }
    ));
    invalid = configuration(64, 48, PixelFormat::Yuv420p8);
    invalid.profile = CodecProfile::Av1Main;
    assert_eq!(
        factory.capability(&invalid),
        CodecSupport::UnsupportedProfile
    );
    invalid = configuration(4097, 48, PixelFormat::Yuv420p8);
    assert!(matches!(
        factory.capability(&invalid),
        CodecSupport::InvalidConfiguration { .. }
    ));
    for blob in [vec![0], vec![40, 0, 0], vec![1, 2], vec![40, 0, 1, 0x80]] {
        invalid = configuration(64, 48, PixelFormat::Yuv420p8);
        invalid.configuration = blob;
        assert!(matches!(
            factory.capability(&invalid),
            CodecSupport::InvalidConfiguration { .. }
        ));
    }
}

/// `Avoid` is always the software encoder. `Prefer` and `Require` are the
/// hardware encoder where the host has one, and otherwise `Prefer` falls back
/// to software and `Require` is unavailable; either way the two agree, and a
/// created encoder reports what the capability promised.
#[test]
fn the_hardware_preference_selects_hardware_only_where_the_host_has_it() {
    let factory = native_vp9_video_encoder_factory();
    // RGBA is an input every platform's hardware encoder takes.
    let mut config = configuration(64, 48, PixelFormat::Rgba8);
    let software = CodecSupport::Supported {
        implementation: CodecImplementation::Software,
    };
    let hardware = CodecSupport::Supported {
        implementation: CodecImplementation::Hardware,
    };
    assert_eq!(factory.capability(&config), software);
    config.hardware = HardwarePreference::Require;
    let required = factory.capability(&config);
    config.hardware = HardwarePreference::Prefer;
    let preferred = factory.capability(&config);
    if required == hardware {
        assert_eq!(preferred, hardware);
    } else {
        assert_eq!(required, CodecSupport::HardwareUnavailable);
        assert_eq!(preferred, software);
        config.hardware = HardwarePreference::Require;
        let error = factory.create(&config, &Limits::default()).err().unwrap();
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{error}");
    }
    for preference in [HardwarePreference::Avoid, HardwarePreference::Prefer] {
        config.hardware = preference;
        let encoder = factory.create(&config, &Limits::default()).unwrap();
        let CodecSupport::Supported { implementation } = factory.capability(&config) else {
            unreachable!("checked above");
        };
        assert_eq!(encoder.implementation(), implementation, "{preference:?}");
    }

    // What no hardware encoder can take is an invalid configuration for
    // `Require` where the platform has hardware encoders, and `Prefer` still
    // falls back to software for it. Linux has none at all.
    for (width, height, range) in [(63, 48, ColorRange::Limited), (64, 48, ColorRange::Full)] {
        let mut config = configuration(width, height, PixelFormat::Rgba8);
        config.color_range = range;
        config.hardware = HardwarePreference::Require;
        let support = factory.capability(&config);
        if cfg!(all(feature = "hardware", any(windows, target_os = "macos"))) {
            assert!(
                matches!(support, CodecSupport::InvalidConfiguration { .. }),
                "{width}x{height} {range:?}: {support:?}"
            );
        } else {
            assert_eq!(support, CodecSupport::HardwareUnavailable);
        }
        config.hardware = HardwarePreference::Prefer;
        assert_eq!(factory.capability(&config), software);
    }
}

#[test]
fn hardware_requests_map_the_quantizer_onto_quality() {
    let request = |blob: Vec<u8>| {
        let mut config = configuration(1920, 1080, PixelFormat::Rgba8);
        config.configuration = blob;
        hardware_request(&config).unwrap()
    };
    assert_eq!(request(vec![1]).quality, 1.0);
    assert_eq!(request(vec![255]).quality, 0.0);
    let default = request(Vec::new());
    assert_eq!(default.quality, f64::from(255 - DEFAULT_BASE_Q_IDX) / 254.0);
    assert_eq!(
        default.keyframe_interval,
        u32::from(DEFAULT_KEYFRAME_INTERVAL)
    );
    assert_eq!(request(vec![60, 0, 5]).keyframe_interval, 5);
    // Finer quantizers declare more bits, and 1080p at about 30 frames a
    // second declares about 10 Mbit/s at the default.
    assert!(request(vec![10]).nominal_bits_per_second > default.nominal_bits_per_second);
    assert!(request(vec![250]).nominal_bits_per_second < default.nominal_bits_per_second);
    assert!((8_000_000..12_000_000).contains(&default.nominal_bits_per_second));

    let rejected = |config: VideoEncoderConfig| hardware_request(&config).unwrap_err();
    assert!(rejected(configuration(17, 10, PixelFormat::Rgba8)).contains("even"));
    let mut full = configuration(64, 48, PixelFormat::Rgba8);
    full.color_range = ColorRange::Full;
    assert!(rejected(full).contains("limited-range"));
    let mut blob = configuration(64, 48, PixelFormat::Rgba8);
    blob.configuration = vec![0];
    assert_eq!(rejected(blob.clone()), CONFIGURATION_SHAPE);
    // Error resilience is the software encoder's alone.
    blob.configuration = vec![60, 0, 5, FLAG_ERROR_RESILIENT];
    assert!(rejected(blob).contains("error resilient"));
}

/// A hardware encoder's samples are sync samples exactly when they open on a
/// key frame, and that key frame's colour fields are what the `vpcC` declares:
/// checked here against the software encoder's own key and inter frames.
#[cfg(all(feature = "hardware", any(windows, target_os = "macos")))]
#[test]
fn key_frame_vpcc_finds_key_frames_and_their_colour() {
    let mut config = configuration(64, 48, PixelFormat::Yuv420p8);
    config.configuration = vec![60, 0, 3];
    let level = pick_level(
        config.coded_dimensions,
        config.timescale,
        config.frame_duration,
    )
    .unwrap();
    let declared = NativeVp9Encoder::new(&config, &Limits::default())
        .unwrap()
        .declared
        .decoder_config;
    let (samples, _, _) = encode_sequence(&config, 6);
    for (index, sample) in samples.iter().enumerate() {
        let found = key_frame_vpcc(&sample.data, level);
        assert_eq!(found.is_some(), sample.is_sync, "frame {index}");
        if let Some(vpcc) = found {
            assert_eq!(vpcc, declared, "frame {index}");
        }
    }
    assert_eq!(key_frame_vpcc(&[], level), None);
}

#[test]
fn configuration_selects_quantizer_keyframe_interval_and_error_resilience() {
    let settings = |base_q_idx, keyframe_interval, error_resilient| {
        Some(Settings {
            base_q_idx,
            keyframe_interval,
            error_resilient,
        })
    };
    assert_eq!(
        parse_configuration(&[]),
        settings(DEFAULT_BASE_Q_IDX, DEFAULT_KEYFRAME_INTERVAL, false)
    );
    assert_eq!(
        parse_configuration(&[200]),
        settings(200, DEFAULT_KEYFRAME_INTERVAL, false)
    );
    assert_eq!(parse_configuration(&[30, 1, 2]), settings(30, 258, false));
    assert_eq!(
        parse_configuration(&[30, 1, 2, 0]),
        settings(30, 258, false)
    );
    assert_eq!(
        parse_configuration(&[30, 0, 1, FLAG_ERROR_RESILIENT]),
        settings(30, 1, true)
    );
    assert_eq!(parse_configuration(&[0]), None);
    assert_eq!(parse_configuration(&[30, 0, 1, 2]), None);
    assert_eq!(parse_configuration(&[0, 0, 1, 1]), None);
    assert_eq!(parse_configuration(&[30, 0, 0, 1]), None);
}

/// The error-resilient bit of a frame's uncompressed header.
fn error_resilient_mode(sample: &EncodedSample) -> bool {
    // frame_marker, profile_low_bit, profile_high_bit, show_existing_frame,
    // frame_type, show_frame, then error_resilient_mode.
    sample.data[0] & 1 == 1
}

#[test]
fn frames_adapt_probabilities_unless_error_resilience_is_requested() {
    let mut config = configuration(40, 24, PixelFormat::Yuv420p8);
    let (samples, _, _) = encode_sequence(&config, 3);
    assert!(samples.iter().all(|sample| !error_resilient_mode(sample)));
    config.configuration = vec![DEFAULT_BASE_Q_IDX, 0, 60, FLAG_ERROR_RESILIENT];
    let (samples, _, _) = encode_sequence(&config, 3);
    assert!(samples.iter().all(error_resilient_mode));
}

/// The total size and mean PSNR of a sequence of moving content.
fn size_and_quality(config: &VideoEncoderConfig, frames: u32) -> (usize, f64) {
    let (samples, reconstructions, sources) = encode_sequence(config, frames);
    let size = samples.iter().map(|sample| sample.data.len()).sum();
    let quality = reconstructions
        .iter()
        .zip(&sources)
        .map(|(reconstruction, source)| psnr(reconstruction, source))
        .sum::<f64>()
        / f64::from(frames);
    (size, quality)
}

/// Issue #555: adapting the probabilities from frame to frame gives a smaller
/// stream than coding every frame error resilient, as the encoder did before,
/// at equal or better quality. Adapted rates steer mode decisions toward
/// slightly different trade-offs, so the adaptive stream's quantizer is lowered
/// until its quality at least matches the error-resilient stream's.
#[test]
fn adaptive_probabilities_shrink_the_output_at_equal_quality() {
    let measure = |base_q_idx: u8, flags: u8| {
        let mut config = configuration(96, 64, PixelFormat::Yuv420p8);
        config.configuration = vec![base_q_idx, 0, 30, flags];
        size_and_quality(&config, 30)
    };
    for base_q_idx in [80, 160] {
        let (resilient_size, resilient_quality) = measure(base_q_idx, FLAG_ERROR_RESILIENT);
        let mut adaptive_q_idx = base_q_idx;
        let (adaptive_size, adaptive_quality) = loop {
            let (size, quality) = measure(adaptive_q_idx, 0);
            if quality >= resilient_quality {
                break (size, quality);
            }
            assert!(adaptive_q_idx > 4, "adaptive quality never caught up");
            adaptive_q_idx -= 4;
        };
        eprintln!(
            "q {base_q_idx}: error resilient {resilient_size} bytes at {resilient_quality:.2} dB,              adaptive (q {adaptive_q_idx}) {adaptive_size} bytes at {adaptive_quality:.2} dB"
        );
        assert!(
            adaptive_size < resilient_size,
            "adaptive {adaptive_size} bytes at {adaptive_quality:.2} dB (q {adaptive_q_idx})              against error-resilient {resilient_size} bytes at {resilient_quality:.2} dB              (q {base_q_idx})"
        );
    }
}

/// Issue #622: key frames used to code their tokens with VP9's default
/// coefficient probabilities, which fit textured intra content at coarse
/// quantizers badly. Updating them in the compressed header, and costing the
/// search with the updated ones, must shrink such key frames at the same
/// quality, judged at 6 dB per doubling of the size: by more than a quarter
/// for the detailed key frame the lookahead keeps where its detail pays off,
/// and at all for the plain one.
#[test]
fn coefficient_probability_updates_shrink_key_frames_at_equal_quality() {
    let geometry = Geometry::new(192, 128);
    let frame = moving_yuv_frame(192, 128, 0);
    let source = source_picture(&geometry, &frame, Orientation::TopLeft).unwrap();
    for base_q_idx in [230, 235, 240] {
        for (weighted, gain) in [(true, 0.75), (false, 1.0)] {
            let [(default_bytes, default_psnr), (updated_bytes, updated_psnr)] =
                [false, true].map(|coef_updates| {
                    let settings = StreamSettings {
                        geometry,
                        base_q_idx,
                        tools: CodingTools {
                            coef_updates,
                            ..CodingTools::ALL
                        },
                        error_resilient: true,
                        full_range: false,
                        loop_filter: true,
                    };
                    let coded = StreamState::default().code(&settings, &source, true, weighted);
                    (
                        coded.data.len() as f64,
                        psnr(
                            &visible(&coded.reconstruction, &geometry),
                            &visible(&source, &geometry),
                        ),
                    )
                });
            let equivalent_bytes = default_bytes * 2_f64.powf((updated_psnr - default_psnr) / 6.0);
            assert!(
                updated_bytes < equivalent_bytes * gain,
                "q {base_q_idx}, weighted {weighted}: {updated_bytes} bytes at \
                 {updated_psnr:.2} dB updated, {default_bytes} bytes at {default_psnr:.2} dB \
                 with the defaults"
            );
        }
    }
}

#[test]
fn levels_follow_picture_size_and_sample_rate() {
    let level = |width, height, timescale, duration| {
        pick_level(VideoDimensions { width, height }, timescale, duration)
    };
    assert_eq!(level(176, 144, 15, 1), Some(10));
    assert_eq!(level(1280, 720, 30, 1), Some(31));
    assert_eq!(level(1920, 1080, 30, 1), Some(40));
    assert_eq!(level(1920, 1080, 60, 1), Some(41));
    assert_eq!(level(3840, 2160, 30, 1), Some(50));
    assert_eq!(level(8192, 8192, 120, 1), None);
}

#[test]
fn vpcc_describes_8_bit_420_profile_0() {
    let encoder = native_vp9_video_encoder_factory()
        .create(
            &configuration(1920, 1080, PixelFormat::Rgba8),
            &Limits::default(),
        )
        .unwrap();
    let vpcc = &encoder.config().decoder_config;
    assert_eq!(encoder.config().codec, Codec::Vp9);
    assert_eq!(&vpcc[..8], &[0, 0, 0, 20, b'v', b'p', b'c', b'C']);
    assert_eq!(&vpcc[8..12], &[1, 0, 0, 0]);
    assert_eq!(vpcc[12], 0, "profile");
    assert_eq!(vpcc[13], 40, "level 4 for 1080p30");
    assert_eq!(vpcc[14], 0x82, "8-bit, 4:2:0 co-located, limited range");
    let derived = crate::derive_codec_string(Codec::Vp9, vpcc).unwrap();
    assert_eq!(derived.codec_string, "vp09.00.40.08");
    assert_eq!(derived.profile, CodecProfile::Vp9Profile0);
}

#[test]
fn vpcc_reads_colour_from_a_key_frame() {
    let mut config = configuration(32, 16, PixelFormat::Yuv420p8);
    let (samples, _, _) = encode_sequence(&config, 2);
    let vpcc = vpcc_from_key_frame(&samples[0].data, 10).unwrap();
    assert_eq!(&vpcc[12..17], &[0, 10, 0x82, 6, 6]);
    assert_eq!(vpcc_from_key_frame(&samples[1].data, 10), None);

    config.color_range = ColorRange::Full;
    let mut encoder = NativeVp9Encoder::new(&config, &Limits::default()).unwrap();
    let mut frame = moving_yuv_frame(32, 16, 0);
    frame.color_range = ColorRange::Full;
    let source = FrameSource::Cpu(CpuFrameSource {
        frame: &frame,
        orientation: Orientation::TopLeft,
    });
    block_on(encoder.encode(FrameIndex(0), source)).unwrap();
    let sample = block_on(encoder.finish()).unwrap().remove(0);
    let vpcc = vpcc_from_key_frame(&sample.data, 10).unwrap();
    assert_eq!(vpcc, encoder.config().decoder_config);
    assert_eq!(vpcc[14], 0x83);
}

#[test]
fn key_frames_start_each_group_and_timing_is_exact() {
    let mut config = configuration(40, 24, PixelFormat::Yuv420p8);
    config.configuration = vec![60, 0, 3];
    let (samples, _, _) = encode_sequence(&config, 7);
    let sync: Vec<bool> = samples.iter().map(|sample| sample.is_sync).collect();
    assert_eq!(sync, [true, false, false, true, false, false, true]);
    for (index, sample) in samples.iter().enumerate() {
        assert_eq!(sample.pts, index as i64 * 1001);
        assert_eq!(sample.dts, sample.pts);
        assert_eq!(sample.duration, 1001);
        // frame_marker 2, profile 0, show_existing_frame 0, then frame_type.
        assert_eq!(sample.data[0] >> 6, 2);
        assert_eq!((sample.data[0] >> 2) & 1, u8::from(!sample.is_sync));
        let expected = if sample.is_sync {
            SampleDependency::INDEPENDENT
        } else {
            SampleDependency::DEPENDENT
        };
        assert_eq!(sample.dependency, expected);
    }
}

#[test]
fn inter_frames_are_smaller_than_key_frames_for_moving_content() {
    let config = configuration(96, 64, PixelFormat::Yuv420p8);
    let (samples, reconstructions, sources) = encode_sequence(&config, 4);
    let key = samples[0].data.len();
    for sample in &samples[1..] {
        assert!(
            sample.data.len() * 2 < key,
            "inter frame of {} bytes against a {key}-byte key frame",
            sample.data.len()
        );
    }
    for (reconstruction, source) in reconstructions.iter().zip(&sources) {
        let quality = psnr(reconstruction, source);
        assert!(quality > 32.0, "reconstruction PSNR {quality:.1} dB");
    }
}

/// The coding tools the encoder had before larger partitions and transforms:
/// every block 8x8 with 4x4 transforms.
const SMALL_BLOCKS: CodingTools = CodingTools {
    largest_block: 0,
    larger_transforms: false,
    sub_sample_motion: true,
    coef_updates: true,
};

/// Every coding tool except sub-sample motion vectors.
const WHOLE_SAMPLE_MOTION: CodingTools = CodingTools {
    sub_sample_motion: false,
    ..CodingTools::ALL
};

/// The inter frames' bytes and mean PSNR.
fn inter_size_and_quality(
    config: &VideoEncoderConfig,
    frames: u32,
    content: Frames,
    tools: CodingTools,
) -> (usize, f64) {
    let (samples, reconstructions, sources) = encode_with(config, frames, content, tools);
    let quality = reconstructions[1..]
        .iter()
        .zip(&sources[1..])
        .map(|(reconstruction, source)| psnr(reconstruction, source))
        .sum::<f64>()
        / f64::from(frames - 1);
    (total_bytes(&samples[1..]), quality)
}

#[test]
fn sub_sample_motion_shrinks_output_at_equal_or_better_quality() {
    for (name, content, sub_sample) in [
        ("sub-sample", sub_sample_yuv_frame as Frames, true),
        ("moving", moving_yuv_frame, false),
        ("smooth", smooth_yuv_frame, false),
    ] {
        for base_q_idx in [40, DEFAULT_BASE_Q_IDX, 150] {
            let mut config = configuration(96, 64, PixelFormat::Yuv420p8);
            config.configuration = vec![base_q_idx];
            let (whole_bytes, whole_quality) =
                inter_size_and_quality(&config, 8, content, WHOLE_SAMPLE_MOTION);
            let (bytes, quality) = inter_size_and_quality(&config, 8, content, CodingTools::ALL);
            let message = format!(
                "{name} q{base_q_idx}: {bytes} bytes at {quality:.2} dB against {whole_bytes} \
                 bytes at {whole_quality:.2} dB with whole-sample motion"
            );
            eprintln!("{message}");
            if sub_sample {
                // Sub-sample motion cuts the inter frames by more than a fifth.
                assert!(bytes * 5 < whole_bytes * 4, "{message}");
                assert!(quality >= whole_quality, "{message}");
            } else {
                // Whole-sample motion gains nothing from refinement, and must
                // lose nothing to it either.
                assert!(bytes * 100 <= whole_bytes * 101, "{message}");
                assert!(quality >= whole_quality - 0.01, "{message}");
            }
        }
    }
}

fn total_bytes(samples: &[EncodedSample]) -> usize {
    samples.iter().map(|sample| sample.data.len()).sum()
}

#[test]
fn larger_partitions_and_transforms_shrink_the_stream_at_no_loss_of_quality() {
    for (name, content) in [
        ("moving", moving_yuv_frame as Frames),
        ("smooth", smooth_yuv_frame),
    ] {
        for base_q_idx in [40, DEFAULT_BASE_Q_IDX, 160] {
            let mut config = configuration(160, 96, PixelFormat::Yuv420p8);
            config.configuration = vec![base_q_idx, 0, 4];
            let (small, small_recon, sources) = encode_with(&config, 6, content, SMALL_BLOCKS);
            let (large, large_recon, _) = encode_with(&config, 6, content, CodingTools::ALL);
            let sources = sources.concat();
            let small_quality = psnr(&small_recon.concat(), &sources);
            let large_quality = psnr(&large_recon.concat(), &sources);
            let (small_bytes, large_bytes) = (total_bytes(&small), total_bytes(&large));
            eprintln!(
                "{name} q{base_q_idx}: {small_bytes} bytes at {small_quality:.2} dB in 8x8 \
                 blocks, {large_bytes} bytes at {large_quality:.2} dB in larger ones"
            );
            assert!(
                large_bytes < small_bytes,
                "{name} q{base_q_idx}: {large_bytes} bytes against {small_bytes}"
            );
            assert!(
                large_quality >= small_quality,
                "{name} q{base_q_idx}: {large_quality:.2} dB against {small_quality:.2} dB"
            );
        }
    }
}

/// The RGBA test card of
/// `crates/zvidlib-vp9-encoder/tests/native_vp9_encoder.rs`: gradients under a
/// checkerboard of sharp-edged squares, panning two samples a frame.
fn test_card_frame(width: u32, height: u32, index: u32) -> VideoFrame {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let u = x + index * 2;
            let checker = if ((u / 10) + (y / 10)) % 2 == 0 {
                60
            } else {
                0
            };
            pixels.extend_from_slice(&[
                (u * 255 / (width + 24)) as u8,
                (y * 255 / height) as u8 / 2 + checker,
                (255 - u * 200 / (width + 24)) as u8,
                255,
            ]);
        }
    }
    VideoFrame::new(
        VideoDimensions { width, height },
        PixelFormat::Rgba8,
        ColorRange::Limited,
        vec![Plane {
            data: pixels,
            stride: width as usize * 4,
        }],
        &Limits::default(),
    )
    .unwrap()
}

#[test]
fn smaller_and_sharper_than_the_8x8_only_encoder() {
    // Ten frames with a key frame every five, as the encoder before larger
    // partitions and transforms coded them, loop filtered: quantizer index,
    // bytes and PSNR over every plane.
    type Baselines = [(u8, usize, f64); 3];
    let cases: [(&str, Frames, PixelFormat, u32, Baselines); 2] = [
        (
            "moving",
            moving_yuv_frame,
            PixelFormat::Yuv420p8,
            96,
            [(40, 50060, 44.01), (80, 29587, 38.79), (160, 9518, 29.21)],
        ),
        (
            "test card",
            test_card_frame,
            PixelFormat::Rgba8,
            90,
            [(40, 10209, 47.95), (80, 7674, 43.16), (160, 4042, 33.92)],
        ),
    ];
    for (name, content, format, height, baselines) in cases {
        for (base_q_idx, baseline_bytes, baseline_quality) in baselines {
            let mut config = configuration(160, height, format);
            config.configuration = vec![base_q_idx, 0, 5];
            let mut encoder = NativeVp9Encoder::new(&config, &Limits::default()).unwrap();
            let (mut bytes, mut decoded, mut sources) = (0, Vec::new(), Vec::new());
            for index in 0..10 {
                let frame = content(160, height, index);
                let source = FrameSource::Cpu(CpuFrameSource {
                    frame: &frame,
                    orientation: Orientation::TopLeft,
                });
                let samples =
                    block_on(encoder.encode(FrameIndex(u64::from(index)), source)).unwrap();
                bytes += samples
                    .iter()
                    .map(|sample| sample.data.len())
                    .sum::<usize>();
                let picture =
                    source_picture(&encoder.geometry, &frame, Orientation::TopLeft).unwrap();
                for plane in 0..3 {
                    sources.extend_from_slice(&picture.planes[plane]);
                }
            }
            let samples = block_on(encoder.finish()).unwrap();
            bytes += samples
                .iter()
                .map(|sample| sample.data.len())
                .sum::<usize>();
            for reconstruction in &encoder.reconstructions {
                for plane in 0..3 {
                    decoded.extend_from_slice(&reconstruction.planes[plane]);
                }
            }
            let quality = psnr(&decoded, &sources);
            eprintln!(
                "{name} q{base_q_idx}: {bytes} bytes at {quality:.2} dB, was {baseline_bytes} \
                 bytes at {baseline_quality:.2} dB"
            );
            assert!(
                bytes < baseline_bytes && quality >= baseline_quality,
                "{name} q{base_q_idx}: {bytes} bytes at {quality:.2} dB, against \
                 {baseline_bytes} bytes at {baseline_quality:.2} dB"
            );
        }
    }
}

#[test]
fn flat_frames_code_in_whole_superblocks() {
    // 64x64 blocks with 32x32 transforms code a flat picture in a handful of
    // symbols, where 8x8 blocks need four 4x4 transforms each.
    fn flat_frame(width: u32, height: u32, _index: u32) -> VideoFrame {
        let (width, height) = (width as usize, height as usize);
        let chroma = width.div_ceil(2) * height.div_ceil(2);
        VideoFrame::new(
            VideoDimensions {
                width: width as u32,
                height: height as u32,
            },
            PixelFormat::Yuv420p8,
            ColorRange::Limited,
            vec![
                Plane {
                    data: vec![90; width * height],
                    stride: width,
                },
                Plane {
                    data: vec![120; chroma],
                    stride: width.div_ceil(2),
                },
                Plane {
                    data: vec![140; chroma],
                    stride: width.div_ceil(2),
                },
            ],
            &Limits::default(),
        )
        .unwrap()
    }
    let config = configuration(256, 256, PixelFormat::Yuv420p8);
    let (small, _, _) = encode_with(&config, 2, flat_frame, SMALL_BLOCKS);
    let (large, reconstructions, sources) = encode_with(&config, 2, flat_frame, CodingTools::ALL);
    for (frame, (small, large)) in small.iter().zip(&large).enumerate() {
        assert!(
            large.data.len() * 4 < small.data.len(),
            "frame {frame}: {} bytes against {} with 8x8 blocks",
            large.data.len(),
            small.data.len()
        );
    }
    assert_eq!(reconstructions, sources);
}

#[test]
fn rgba_input_converts_with_bt601() {
    let config = configuration(16, 16, PixelFormat::Rgba8);
    let mut encoder = NativeVp9Encoder::new(&config, &Limits::default()).unwrap();
    let pixels = [255_u8, 0, 0, 255].repeat(16 * 16);
    let frame = VideoFrame::new(
        config.coded_dimensions,
        PixelFormat::Rgba8,
        ColorRange::Limited,
        vec![Plane {
            data: pixels,
            stride: 64,
        }],
        &Limits::default(),
    )
    .unwrap();
    let source = FrameSource::Cpu(CpuFrameSource {
        frame: &frame,
        orientation: Orientation::BottomLeft,
    });
    block_on(encoder.encode(FrameIndex(0), source)).unwrap();
    block_on(encoder.finish()).unwrap();
    let decoded = reconstruction(&encoder);
    // Limited-range BT.601 red is Y 82, Cb 90, Cr 240.
    let near = |value: u8, expected: i32| (i32::from(value) - expected).abs() <= 3;
    assert!(decoded[..256].iter().all(|&y| near(y, 82)));
    assert!(decoded[256..320].iter().all(|&cb| near(cb, 90)));
    assert!(decoded[320..].iter().all(|&cr| near(cr, 240)));
}

#[test]
fn rejects_out_of_order_frames_and_encoding_after_finish() {
    let config = configuration(16, 16, PixelFormat::Yuv420p8);
    let mut encoder = NativeVp9Encoder::new(&config, &Limits::default()).unwrap();
    let frame = moving_yuv_frame(16, 16, 0);
    let source = || {
        FrameSource::Cpu(CpuFrameSource {
            frame: &frame,
            orientation: Orientation::TopLeft,
        })
    };
    let error = block_on(encoder.encode(FrameIndex(1), source())).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    // The key frame waits for the frames after it, so finishing emits it.
    assert!(
        block_on(encoder.encode(FrameIndex(0), source()))
            .unwrap()
            .is_empty()
    );
    assert_eq!(block_on(encoder.finish()).unwrap().len(), 1);
    let error = block_on(encoder.encode(FrameIndex(1), source())).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidState);
}

#[cfg(not(target_arch = "wasm32"))]
mod ffmpeg {
    use super::*;
    use std::io::Write;
    use std::process::{Command, Stdio};

    fn ivf(samples: &[EncodedSample], width: u32, height: u32) -> Vec<u8> {
        let mut output = Vec::new();
        output.extend_from_slice(b"DKIF");
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(&32_u16.to_le_bytes());
        output.extend_from_slice(b"VP90");
        output.extend_from_slice(&(width as u16).to_le_bytes());
        output.extend_from_slice(&(height as u16).to_le_bytes());
        output.extend_from_slice(&30_u32.to_le_bytes());
        output.extend_from_slice(&1_u32.to_le_bytes());
        output.extend_from_slice(&(samples.len() as u32).to_le_bytes());
        output.extend_from_slice(&0_u32.to_le_bytes());
        for (index, sample) in samples.iter().enumerate() {
            output.extend_from_slice(&(sample.data.len() as u32).to_le_bytes());
            output.extend_from_slice(&(index as u64).to_le_bytes());
            output.extend_from_slice(&sample.data);
        }
        output
    }

    fn decoder_available(name: &str) -> bool {
        Command::new("ffmpeg")
            .args(["-hide_banner", "-decoders"])
            .output()
            .is_ok_and(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .split_whitespace()
                    .any(|word| word == name)
            })
    }

    /// Decodes an IVF stream to concatenated `yuv420p` frames with ffmpeg's
    /// `decoder` (`vp9` or `libvpx-vp9`).
    fn decode(decoder: &str, stream: &[u8]) -> Vec<u8> {
        let mut child = Command::new("ffmpeg")
            .args([
                "-v", "error", "-c:v", decoder, "-f", "ivf", "-i", "pipe:0", "-f", "rawvideo",
                "-pix_fmt", "yuv420p", "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let input = stream.to_vec();
        let writer = std::thread::spawn(move || stdin.write_all(&input));
        let output = child.wait_with_output().unwrap();
        writer.join().unwrap().unwrap();
        assert!(
            output.status.success() && output.stderr.is_empty(),
            "{decoder} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    }

    /// Checks both the whole-sample test content and the sub-sample content,
    /// whose eighth-sample vectors exercise the 8-tap filters and the
    /// high-precision motion vector bits.
    fn assert_decoders_match_reconstruction(config: &VideoEncoderConfig, frames: u32) {
        for content in [moving_yuv_frame as Frames, sub_sample_yuv_frame] {
            assert_decoders_match(config, frames, content);
        }
    }

    fn assert_decoders_match(config: &VideoEncoderConfig, frames: u32, content: Frames) {
        let decoders: Vec<&str> = ["vp9", "libvpx-vp9"]
            .into_iter()
            .filter(|name| decoder_available(name))
            .collect();
        if decoders.is_empty() {
            eprintln!("skipping independent VP9 decode because ffmpeg has no VP9 decoder");
            return;
        }
        let (samples, reconstructions, _) = encode_with(config, frames, content, CodingTools::ALL);
        let (width, height) = (
            config.coded_dimensions.width,
            config.coded_dimensions.height,
        );
        let stream = ivf(&samples, width, height);
        let expected: Vec<u8> = reconstructions.concat();
        for decoder in decoders {
            let decoded = decode(decoder, &stream);
            assert_eq!(decoded.len(), expected.len(), "{decoder} frame count");
            let frame_size = expected.len() / frames as usize;
            for frame in 0..frames as usize {
                let range = frame * frame_size..(frame + 1) * frame_size;
                assert!(
                    decoded[range.clone()] == expected[range],
                    "{decoder} differs from the reconstruction at frame {frame} ({width}x{height})"
                );
            }
        }
    }

    #[test]
    fn independent_decoders_reproduce_the_reconstruction_exactly() {
        // Two groups of pictures, so the decoders also reset the adapted
        // probabilities at the second key frame.
        let mut config = configuration(96, 64, PixelFormat::Yuv420p8);
        config.configuration = vec![DEFAULT_BASE_Q_IDX, 0, 5];
        assert_decoders_match_reconstruction(&config, 12);
    }

    #[test]
    fn error_resilient_frames_decode_exactly() {
        let mut config = configuration(96, 64, PixelFormat::Yuv420p8);
        config.configuration = vec![DEFAULT_BASE_Q_IDX, 0, 5, FLAG_ERROR_RESILIENT];
        assert_decoders_match_reconstruction(&config, 8);
    }

    #[test]
    fn odd_sizes_and_partial_superblocks_decode_exactly() {
        for (width, height) in [(17, 9), (67, 45), (130, 72)] {
            let config = configuration(width, height, PixelFormat::Yuv420p8);
            assert_decoders_match_reconstruction(&config, 4);
        }
    }

    #[test]
    fn every_quantizer_extreme_decodes_exactly() {
        for base_q_idx in [1, 255] {
            let mut config = configuration(48, 40, PixelFormat::Yuv420p8);
            config.configuration = vec![base_q_idx];
            assert_decoders_match_reconstruction(&config, 3);
        }
    }

    #[test]
    fn whole_superblocks_and_large_transforms_decode_exactly() {
        // Smooth content codes in 32x32 and 64x64 blocks with transforms up to
        // 32x32, including superblocks cut by the bottom and right edges.
        for (width, height) in [(160, 96), (200, 136)] {
            for base_q_idx in [1, DEFAULT_BASE_Q_IDX, 255] {
                let mut config = configuration(width, height, PixelFormat::Yuv420p8);
                config.configuration = vec![base_q_idx, 0, 3];
                assert_decoders_match(&config, 4, smooth_yuv_frame);
            }
        }
    }
}

/// The `loop_filter_level` an encoded frame's uncompressed header signals.
fn signalled_filter_level(sample: &EncodedSample) -> u8 {
    // Key frames: marker, profile, flags, sync code, colour, size, render
    // size and frame_context_idx. Inter frames: marker, profile, flags,
    // refresh flags, references, sizes, motion vector precision,
    // interpolation filter and frame_context_idx. Frames that are not error
    // resilient add refresh_frame_context and frame_parallel_decoding_mode,
    // and inter frames reset_frame_context too.
    let adaptive = !error_resilient_mode(sample);
    let start = match (sample.is_sync, adaptive) {
        (true, false) => 71,
        (true, true) => 73,
        (false, false) => 36,
        (false, true) => 40,
    };
    (start..start + 6).fold(0, |level, index| {
        (level << 1) | ((sample.data[index / 8] >> (7 - index % 8)) & 1)
    })
}

/// The visible `yuv420p` bytes of a picture.
fn visible(picture: &Picture, geometry: &Geometry) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (plane, width, height) in [
        (0, geometry.width, geometry.height),
        (1, geometry.chroma_width(), geometry.chroma_height()),
        (2, geometry.chroma_width(), geometry.chroma_height()),
    ] {
        let stride = picture.strides[plane];
        for row in 0..height {
            bytes.extend_from_slice(&picture.planes[plane][row * stride..row * stride + width]);
        }
    }
    bytes
}

/// Encodes `frames` as one group of pictures, with or without the loop
/// filter, and returns the total size in bytes and the PSNR of the
/// reconstruction against the source.
fn encode_group(frames: &[VideoFrame], base_q_idx: u8, loop_filter: bool) -> (usize, f64) {
    let (sizes, psnr) = encode_group_frames(frames, base_q_idx, loop_filter);
    (sizes.iter().sum(), psnr)
}

/// The coding tools [`encode_group`] codes with: every tool, with the default
/// coefficient probabilities, so that its comparisons isolate the loop filter.
/// Fitted to each frame's own tokens (issue #622), the
/// probabilities move a group's size between neighbouring quantizers, and
/// between a weighted and an unweighted key frame, by more than either does.
const GROUP_TOOLS: CodingTools = CodingTools {
    coef_updates: false,
    ..CodingTools::ALL
};

/// [`encode_group`], with the size of every frame.
fn encode_group_frames(
    frames: &[VideoFrame],
    base_q_idx: u8,
    loop_filter: bool,
) -> (Vec<usize>, f64) {
    let dimensions = frames[0].dimensions;
    let geometry = Geometry::new(dimensions.width as usize, dimensions.height as usize);
    let pictures: Vec<Picture> = frames
        .iter()
        .map(|frame| source_picture(&geometry, frame, Orientation::TopLeft).unwrap())
        .collect();
    // Error resilient, so the comparison isolates the loop filter. The whole
    // sequence is one group, which the encoder's lookahead would see whole.
    let settings = StreamSettings {
        geometry,
        base_q_idx,
        tools: GROUP_TOOLS,
        error_resilient: true,
        full_range: false,
        loop_filter,
    };
    let coded = if loop_filter {
        StreamState::default().code_group_start(&settings, &pictures)
    } else {
        // The unfiltered encodes keep whichever key frame the lookahead keeps
        // with the filter on, so that the comparisons measure the filter and
        // not a near tie in the lookahead tipping one way unfiltered and the
        // other filtered (issue #626).
        let filtered = StreamSettings {
            loop_filter: true,
            ..settings
        };
        let kept = StreamState::default().code_group_start(&filtered, &pictures);
        let unweighted_key = StreamState::default().code(&filtered, &pictures[0], true, false);
        let weighted = kept[0].data != unweighted_key.data;
        let mut stream = StreamState::default();
        pictures
            .iter()
            .enumerate()
            .map(|(index, picture)| {
                stream.code(&settings, picture, index == 0, index == 0 && weighted)
            })
            .collect()
    };
    let (mut sizes, mut reconstructed, mut sources) = (Vec::new(), Vec::new(), Vec::new());
    for (frame, source) in coded.iter().zip(&pictures) {
        sizes.push(frame.data.len());
        reconstructed.extend(visible(&frame.reconstruction, &geometry));
        sources.extend(visible(source, &geometry));
    }
    (sizes, psnr(&reconstructed, &sources))
}

#[test]
fn every_frame_signals_a_loop_filter_level() {
    // Error resilient, so every frame codes the content afresh. With adapted
    // probabilities an inter frame can code it well enough that the search
    // rightly leaves the filter off.
    let mut config = configuration(96, 64, PixelFormat::Yuv420p8);
    config.configuration = vec![DEFAULT_BASE_Q_IDX, 0, 5, FLAG_ERROR_RESILIENT];
    let (samples, _, _) = encode_sequence(&config, 8);
    for (index, sample) in samples.iter().enumerate() {
        let level = signalled_filter_level(sample);
        assert!((1..=63).contains(&level), "frame {index} level {level}");
    }
}

/// How far the unfiltered encode of `frames` at `base_q_idx` lands off the
/// rate and distortion curve through its neighbouring quantizers, and how much
/// larger the filtered encode is than the unfiltered one would have to grow to
/// match its PSNR at the high-rate 6 dB per doubling of the rate, both in log2
/// of the size, which that rule is linear in. Also the filter's PSNR gain.
fn loop_filter_against_decision_noise(frames: &[VideoFrame], base_q_idx: u8) -> (f64, f64, f64) {
    let [below, (unfiltered_bytes, unfiltered_psnr), above] =
        [base_q_idx - 1, base_q_idx, base_q_idx + 1].map(|q| {
            let (bytes, psnr) = encode_group(frames, q, false);
            (bytes as f64, psnr)
        });
    let (filtered_bytes, filtered_psnr) = encode_group(frames, base_q_idx, true);
    let along = (unfiltered_psnr - below.1) / (above.1 - below.1);
    let curve = below.0.log2() + along * (above.0.log2() - below.0.log2());
    let equivalent = unfiltered_bytes.log2() + (filtered_psnr - unfiltered_psnr) / 6.0;
    (
        unfiltered_bytes.log2() - curve,
        (filtered_bytes as f64).log2() - equivalent,
        filtered_psnr - unfiltered_psnr,
    )
}

/// The RMS of `deviations`, in log2 of the size.
fn decision_noise(deviations: &[f64]) -> f64 {
    (deviations.iter().map(|d| d * d).sum::<f64>() / deviations.len() as f64).sqrt()
}

/// `log2` of a size ratio as a percentage change.
fn percent(log2: f64) -> f64 {
    (log2.exp2() - 1.0) * 100.0
}

#[test]
fn loop_filter_is_a_rate_distortion_gain() {
    // Larger blocks and transforms leave less blocking for the filter to
    // remove, so it mostly buys quality rather than bits: the filtered stream
    // must be sharper, and no larger than the unfiltered one would have to
    // grow to match it at the high-rate 6 dB per doubling of the rate. The
    // coarse quantizers are where the greedy per-frame level search used to
    // smooth the panning references until the whole sequence came out larger
    // and blurrier than with no filter (issue #563): 10% or more larger and
    // 0.3 dB or more blurrier.
    //
    // Held to that with no margin, the rule tracked the encoder's mode
    // decisions rather than the filter: scaling only the key frame's lambda
    // flipped which quantizer failed, by under 1.5% (issue #619). An
    // unfiltered encode lands off the curve through its neighbouring
    // quantizers by about 2% RMS here, so no point may lose more than that
    // measured noise, and the points together must still be a gain, as
    // `loop_filter_on_sharp_content_costs_less_than_decision_noise` does.
    let frames: Vec<VideoFrame> = (0..12)
        .map(|index| test_card_frame(160, 90, index))
        .collect();
    let (mut ratios, mut deviations) = (Vec::new(), Vec::new());
    for base_q_idx in [100, 150, 210, 220, 230] {
        let (deviation, ratio, gain) = loop_filter_against_decision_noise(&frames, base_q_idx);
        assert!(gain > 0.0, "q {base_q_idx}: filtered {gain:+.2} dB");
        deviations.push(deviation);
        ratios.push((base_q_idx, ratio));
    }
    let noise = decision_noise(&deviations);
    assert!(
        percent(noise) < 3.0,
        "unfiltered encodes are {:.2}% off their curve",
        percent(noise)
    );
    for &(base_q_idx, ratio) in &ratios {
        assert!(
            ratio <= noise,
            "q {base_q_idx}: filtered {:+.2}%, noise {:.2}%",
            percent(ratio),
            percent(noise)
        );
    }
    let mean = ratios.iter().map(|&(_, ratio)| ratio).sum::<f64>() / ratios.len() as f64;
    assert!(mean <= 0.0, "filtered {:+.2}% on average", percent(mean));
}

#[test]
fn panning_inter_frames_find_the_motion() {
    // The texture's fine detail leaves the motion search many local minima.
    // Started from the neighbours' vectors alone, a frame whose first blocks
    // missed the pan fell back to intra nearly everywhere and came out about
    // as large as the key frame, and the loop filter's small changes to the
    // reference decided which frames did: the filtered sequence was up to a
    // third larger than the unfiltered one would have to grow to match its
    // quality (issue #583). Content this sharp has little blocking left for
    // the filter to remove, so it must now cost next to nothing either way.
    //
    // At quantizers above about 230 a key frame weighed as a single frame
    // dropped nearly all of the texture, and an inter frame later bought it
    // back at up to three times the key frame's size (issue #597).
    for (width, height) in [(96, 64), (192, 128)] {
        let frames: Vec<VideoFrame> = (0..12)
            .map(|index| moving_yuv_frame(width, height, index))
            .collect();
        for base_q_idx in [30, 40, 130, 200, 210, 220, 230, 240] {
            let [
                (unfiltered_bytes, unfiltered_psnr),
                (filtered_bytes, filtered_psnr),
            ] = [false, true].map(|loop_filter| {
                let (sizes, psnr) = encode_group_frames(&frames, base_q_idx, loop_filter);
                for (index, &size) in sizes.iter().enumerate().skip(1) {
                    assert!(
                        size < sizes[0] / 2,
                        "{width}x{height} q {base_q_idx}, filter {loop_filter}: \
                         frame {index} is {size} bytes, the key frame {}",
                        sizes[0]
                    );
                }
                (sizes.iter().sum::<usize>(), psnr)
            });
            let equivalent_bytes =
                unfiltered_bytes as f64 * 2_f64.powf((filtered_psnr - unfiltered_psnr) / 6.0);
            assert!(
                filtered_bytes as f64 <= equivalent_bytes * 1.03,
                "{width}x{height} q {base_q_idx}: {filtered_bytes} bytes at \
                 {filtered_psnr:.2} dB filtered, {unfiltered_bytes} bytes at \
                 {unfiltered_psnr:.2} dB unfiltered"
            );
        }
    }
}

#[test]
fn long_pans_keep_key_frame_detail_only_where_it_pays() {
    // A key frame at a lower lambda keeps the texture that
    // `panning_inter_frames_find_the_motion` needs to avoid an inter frame
    // buying it back at several times the key frame's size. But this pan
    // replaces the whole picture within 48 frames, and an inter frame refining
    // a blurry key frame buys the texture more cheaply than intra coding does,
    // so over the longer group the weighted key frame left the sequence up to
    // 44% larger at equal quality (issue #618). The lookahead codes the
    // group's start both ways, so the long pan must code no worse than with
    // the key frame unweighted, beyond decision noise, with every coding tool
    // on. Fitted to each key frame's tokens (issue #622), the coefficient
    // probabilities made a weighted key frame cheaper, and at 96x64 and q 235
    // the lookahead kept one on a near tie that left the stream 5% larger
    // (issue #626).
    for (width, height) in [(96, 64), (192, 128)] {
        let frames: Vec<VideoFrame> = (0..48)
            .map(|index| moving_yuv_frame(width, height, index))
            .collect();
        let geometry = Geometry::new(width as usize, height as usize);
        for base_q_idx in [230, 235, 240] {
            let settings = StreamSettings {
                geometry,
                base_q_idx,
                tools: CodingTools::ALL,
                error_resilient: true,
                full_range: false,
                loop_filter: true,
            };
            let pictures: Vec<Picture> = frames
                .iter()
                .map(|frame| source_picture(&geometry, frame, Orientation::TopLeft).unwrap())
                .collect();
            let sizes_and_quality = |coded: &[CodedFrame]| {
                let (mut reconstructed, mut sources) = (Vec::new(), Vec::new());
                for (frame, source) in coded.iter().zip(&pictures) {
                    reconstructed.extend(visible(&frame.reconstruction, &geometry));
                    sources.extend(visible(source, &geometry));
                }
                (
                    coded.iter().map(|frame| frame.data.len()).sum::<usize>(),
                    psnr(&reconstructed, &sources),
                )
            };
            let mut stream = StreamState::default();
            let unweighted: Vec<CodedFrame> = pictures
                .iter()
                .enumerate()
                .map(|(index, picture)| stream.code(&settings, picture, index == 0, false))
                .collect();
            let (unweighted_bytes, unweighted_psnr) = sizes_and_quality(&unweighted);
            let (bytes, quality) =
                sizes_and_quality(&StreamState::default().code_group_start(&settings, &pictures));
            let equivalent_bytes =
                unweighted_bytes as f64 * 2_f64.powf((quality - unweighted_psnr) / 6.0);
            assert!(
                bytes as f64 <= equivalent_bytes * 1.01,
                "{width}x{height} q {base_q_idx}: {bytes} bytes at {quality:.2} dB, \
                 {unweighted_bytes} bytes at {unweighted_psnr:.2} dB unweighted"
            );
        }
    }
}

#[test]
fn the_lookahead_codes_each_key_frame_as_it_would_code_alone() {
    // The weighted and unweighted codings of a group's start run side by side
    // (issue #628). Each must be exactly the coding it would be alone, and
    // the stream must continue from the one kept: here the weighted key frame
    // is kept at q 240 and not at q 220, where it codes differently.
    let (width, height) = (96, 64);
    let geometry = Geometry::new(width as usize, height as usize);
    let pictures: Vec<Picture> = (0..13)
        .map(|index| {
            let frame = moving_yuv_frame(width, height, index);
            source_picture(&geometry, &frame, Orientation::TopLeft).unwrap()
        })
        .collect();
    let (start, next) = pictures.split_at(12);
    for (base_q_idx, weighted) in [(240, true), (220, false)] {
        let settings = StreamSettings {
            geometry,
            base_q_idx,
            tools: CodingTools::ALL,
            error_resilient: true,
            full_range: false,
            loop_filter: true,
        };
        let mut stream = StreamState::default();
        let mut kept = stream.code_group_start(&settings, start);
        kept.push(stream.code(&settings, &next[0], false, false));
        let mut alone = StreamState::default();
        let expected: Vec<Vec<u8>> = pictures
            .iter()
            .enumerate()
            .map(|(index, picture)| {
                alone
                    .code(&settings, picture, index == 0, index == 0 && weighted)
                    .data
            })
            .collect();
        let kept: Vec<Vec<u8>> = kept.into_iter().map(|frame| frame.data).collect();
        assert!(kept == expected, "q {base_q_idx}: weighted {weighted}");
    }
}

#[test]
fn small_key_frame_weights_code_the_group_start_once() {
    // Below `base_q_idx` 160 the weight lowers the key frame's lambda by a
    // few percent at most, so the lookahead codes the group's start
    // unweighted only (issue #630). Before, it kept the weighted key frame
    // here on decision noise, which left the test card's 5 frames 3.5%
    // smaller at q 140 and 3% smaller at q 150.
    let (width, height) = (160, 90);
    let geometry = Geometry::new(width as usize, height as usize);
    let pictures: Vec<Picture> = (0..5)
        .map(|index| {
            let frame = test_card_frame(width, height, index);
            source_picture(&geometry, &frame, Orientation::TopLeft).unwrap()
        })
        .collect();
    for base_q_idx in [140, 150] {
        let settings = StreamSettings {
            geometry,
            base_q_idx,
            tools: CodingTools::ALL,
            error_resilient: false,
            full_range: false,
            loop_filter: true,
        };
        let kept: Vec<Vec<u8>> = StreamState::default()
            .code_group_start(&settings, &pictures)
            .into_iter()
            .map(|frame| frame.data)
            .collect();
        let mut alone = StreamState::default();
        let unweighted: Vec<Vec<u8>> = pictures
            .iter()
            .enumerate()
            .map(|(index, picture)| alone.code(&settings, picture, index == 0, false).data)
            .collect();
        assert!(kept == unweighted, "q {base_q_idx}");
    }
}

#[test]
fn group_starts_are_emitted_once_the_lookahead_fills() {
    // The key frame waits for the rest of its group, up to the lookahead,
    // and frames past the lookahead code as they arrive.
    let mut config = configuration(16, 16, PixelFormat::Yuv420p8);
    config.configuration = vec![60, 0, 3];
    let mut encoder = NativeVp9Encoder::new(&config, &Limits::default()).unwrap();
    let frame_number = |sample: &EncodedSample| sample.pts / i64::from(sample.duration);
    let mut emitted = Vec::new();
    for index in 0..7 {
        let frame = moving_yuv_frame(16, 16, index);
        let source = FrameSource::Cpu(CpuFrameSource {
            frame: &frame,
            orientation: Orientation::TopLeft,
        });
        let samples = block_on(encoder.encode(FrameIndex(u64::from(index)), source)).unwrap();
        emitted.push(samples.iter().map(frame_number).collect::<Vec<_>>());
    }
    let drained: Vec<i64> = block_on(encoder.finish())
        .unwrap()
        .iter()
        .map(frame_number)
        .collect();
    let empty = Vec::new();
    assert_eq!(
        emitted,
        [
            empty.clone(),
            empty.clone(),
            vec![0, 1, 2],
            empty.clone(),
            empty,
            vec![3, 4, 5],
            vec![]
        ]
    );
    assert_eq!(drained, [6]);

    config.configuration = vec![60, 1, 0];
    let encoder = NativeVp9Encoder::new(&config, &Limits::default()).unwrap();
    assert_eq!(encoder.lookahead, LOOKAHEAD_FRAMES);
}

/// A frame of interleaved 4-byte pixels whose channels vary independently,
/// with a pattern that pans by `index` samples a frame.
fn moving_rgb_frame(
    width: u32,
    height: u32,
    index: u32,
    format: PixelFormat,
    range: ColorRange,
) -> VideoFrame {
    let (width, height) = (width as usize, height as usize);
    // A stride wider than the row, so the conversion must honour it.
    let stride = width * 4 + 12;
    let mut data = vec![0_u8; stride * height];
    for y in 0..height {
        for x in 0..width {
            let u = x + index as usize * 3;
            let pixel = &mut data[y * stride + x * 4..][..4];
            pixel[0] = ((u * 5 + y * 3) % 256) as u8;
            pixel[1] = (((u ^ y) * 7) % 256) as u8;
            pixel[2] = (255 - (u * 2 + y * 9) % 256) as u8;
            pixel[3] = (x * y % 256) as u8;
        }
    }
    VideoFrame::new(
        VideoDimensions {
            width: width as u32,
            height: height as u32,
        },
        format,
        range,
        vec![Plane { data, stride }],
        &Limits::default(),
    )
    .unwrap()
}

/// The RGBA/BGRA conversion as it was written per pixel before it was
/// vectorized: BT.601 luma per pixel, and chroma from each channel's rounded
/// average over a 2x2 block whose last row and column repeat at odd sizes.
fn reference_conversion(
    frame: &VideoFrame,
    orientation: Orientation,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (width, height) = (
        frame.dimensions.width as usize,
        frame.dimensions.height as usize,
    );
    let plane = &frame.planes[0];
    let full = frame.color_range == ColorRange::Full;
    let (red, blue) = if frame.pixel_format == PixelFormat::Rgba8 {
        (0, 2)
    } else {
        (2, 0)
    };
    let rgb = |x: usize, y: usize| {
        let y = match orientation {
            Orientation::TopLeft => y,
            Orientation::BottomLeft => height - 1 - y,
        };
        let offset = y * plane.stride + x * 4;
        [red, 1, blue].map(|channel| i32::from(plane.data[offset + channel]))
    };
    let mut luma = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let [r, g, b] = rgb(x, y);
            luma.push(if full {
                ((77 * r + 150 * g + 29 * b + 128) >> 8).clamp(0, 255) as u8
            } else {
                (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16).clamp(0, 255) as u8
            });
        }
    }
    let (mut cb, mut cr) = (Vec::new(), Vec::new());
    for y in 0..height.div_ceil(2) {
        for x in 0..width.div_ceil(2) {
            let mut sum = [0_i32; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let sample = rgb((x * 2 + dx).min(width - 1), (y * 2 + dy).min(height - 1));
                for channel in 0..3 {
                    sum[channel] += sample[channel];
                }
            }
            let [r, g, b] = sum.map(|value| (value + 2) >> 2);
            let (u, v) = if full {
                (
                    (-43 * r - 85 * g + 128 * b + 128) >> 8,
                    (128 * r - 107 * g - 21 * b + 128) >> 8,
                )
            } else {
                (
                    (-38 * r - 74 * g + 112 * b + 128) >> 8,
                    (112 * r - 94 * g - 18 * b + 128) >> 8,
                )
            };
            cb.push((u + 128).clamp(0, 255) as u8);
            cr.push((v + 128).clamp(0, 255) as u8);
        }
    }
    (luma, cb, cr)
}

#[test]
fn rgb_conversion_matches_the_per_pixel_bt601_reference_on_every_isa() {
    let _guard = zvidlib_core::simd::test_lock();
    for isa in zvidlib_core::simd::available() {
        zvidlib_core::simd::set_override(Some(isa));
        for (width, height) in [(16, 16), (17, 9), (33, 2), (1, 1), (70, 35)] {
            let geometry = Geometry::new(width, height);
            for format in [PixelFormat::Rgba8, PixelFormat::Bgra8] {
                for range in [ColorRange::Limited, ColorRange::Full] {
                    for orientation in [Orientation::TopLeft, Orientation::BottomLeft] {
                        let frame = moving_rgb_frame(width as u32, height as u32, 1, format, range);
                        let picture = source_picture(&geometry, &frame, orientation).unwrap();
                        let (luma, cb, cr) = reference_conversion(&frame, orientation);
                        let crop = |plane: usize, width: usize, height: usize| {
                            let stride = picture.strides[plane];
                            (0..height)
                                .flat_map(|row| {
                                    picture.planes[plane][row * stride..][..width].to_vec()
                                })
                                .collect::<Vec<u8>>()
                        };
                        let (chroma_width, chroma_height) =
                            (geometry.chroma_width(), geometry.chroma_height());
                        let label = format!(
                            "{} {width}x{height} {format:?} {range:?} {orientation:?}",
                            isa.name()
                        );
                        assert_eq!(crop(0, width, height), luma, "{label} luma");
                        assert_eq!(crop(1, chroma_width, chroma_height), cb, "{label} Cb");
                        assert_eq!(crop(2, chroma_width, chroma_height), cr, "{label} Cr");
                    }
                }
            }
        }
    }
    zvidlib_core::simd::set_override(None);
}

/// The vector kernels are bit-exact with the scalar ones, so the encoder has
/// to write the same bytes whichever instruction set it runs on (cf. #231).
#[test]
fn every_instruction_set_encodes_byte_identical_bitstreams() {
    let _guard = zvidlib_core::simd::test_lock();
    let encode = |config: &VideoEncoderConfig, content: &dyn Fn(u32) -> VideoFrame| {
        let mut encoder = NativeVp9Encoder::new(config, &Limits::default()).unwrap();
        let mut samples = Vec::new();
        for index in 0..4 {
            let frame = content(index);
            let source = FrameSource::Cpu(CpuFrameSource {
                frame: &frame,
                orientation: Orientation::TopLeft,
            });
            samples.extend(block_on(encoder.encode(FrameIndex(u64::from(index)), source)).unwrap());
        }
        samples.extend(block_on(encoder.finish()).unwrap());
        samples
            .into_iter()
            .flat_map(|sample| sample.data)
            .collect::<Vec<u8>>()
    };
    let yuv = configuration(100, 66, PixelFormat::Yuv420p8);
    let mut bgra = configuration(67, 45, PixelFormat::Bgra8);
    bgra.color_range = ColorRange::Full;
    let streams = || {
        [
            encode(&yuv, &|index| moving_yuv_frame(100, 66, index)),
            encode(&bgra, &|index| {
                moving_rgb_frame(67, 45, index, PixelFormat::Bgra8, ColorRange::Full)
            }),
        ]
    };
    zvidlib_core::simd::set_override(Some(zvidlib_core::simd::SimdIsa::Scalar));
    let reference = streams();
    for isa in zvidlib_core::simd::available() {
        zvidlib_core::simd::set_override(Some(isa));
        assert!(
            streams() == reference,
            "{} encoded different bytes from the scalar kernels",
            isa.name()
        );
    }
    zvidlib_core::simd::set_override(None);
}

#[test]
fn loop_filter_on_sharp_content_costs_less_than_decision_noise() {
    // On content this sharp the filter has almost nothing to remove, and the
    // strict rule of `loop_filter_is_a_rate_distortion_gain` fails at some
    // quantizers by a percent or so either way (issue #590). The per-frame
    // levels it chooses do lower each frame's error; what moves the size is
    // which mode or vector later frames pick from a slightly different
    // reference, a few bytes up or down per frame. The encoder is that noisy
    // without the filter too: an unfiltered encode lands off the rate and
    // distortion curve through its neighbouring quantizers by more, 3-4% RMS
    // over q 20 to 240 and about 2% at the quantizers here. So no point may
    // lose more than that measured noise, and the points together must still
    // be a gain.
    let (mut ratios, mut deviations) = (Vec::new(), Vec::new());
    for (width, height) in [(96, 64), (192, 128)] {
        let frames: Vec<VideoFrame> = (0..12)
            .map(|index| moving_yuv_frame(width, height, index))
            .collect();
        for base_q_idx in [30_u8, 40, 130, 200, 210, 220] {
            let (deviation, ratio, _) = loop_filter_against_decision_noise(&frames, base_q_idx);
            deviations.push(deviation);
            ratios.push((width, height, base_q_idx, ratio));
        }
    }
    let noise = decision_noise(&deviations);
    assert!(
        percent(noise) < 3.0,
        "unfiltered encodes are {:.2}% off their curve",
        percent(noise)
    );
    for &(width, height, base_q_idx, ratio) in &ratios {
        assert!(
            ratio <= noise,
            "{width}x{height} q {base_q_idx}: filtered {:+.2}%, noise {:.2}%",
            percent(ratio),
            percent(noise)
        );
    }
    let mean = ratios.iter().map(|&(.., ratio)| ratio).sum::<f64>() / ratios.len() as f64;
    assert!(mean <= 0.0, "filtered {:+.2}% on average", percent(mean));
}
