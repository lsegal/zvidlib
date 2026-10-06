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

/// The visible `yuv420p` bytes of the encoder's reconstruction.
fn reconstruction(encoder: &NativeVp9Encoder) -> Vec<u8> {
    let picture = encoder.reference.as_ref().unwrap();
    let geometry = encoder.geometry;
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

fn encode_sequence(
    config: &VideoEncoderConfig,
    frames: u32,
) -> (Vec<EncodedSample>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut encoder = NativeVp9Encoder::new(config, &Limits::default()).unwrap();
    let mut samples = Vec::new();
    let mut reconstructions = Vec::new();
    let mut sources = Vec::new();
    let (width, height) = (config.coded_dimensions.width, config.coded_dimensions.height);
    for index in 0..frames {
        let frame = moving_yuv_frame(width, height, index);
        sources.push(frame.planes.iter().flat_map(|plane| plane.data.clone()).collect());
        let source = FrameSource::Cpu(CpuFrameSource {
            frame: &frame,
            orientation: Orientation::TopLeft,
        });
        let mut emitted = block_on(encoder.encode(FrameIndex(u64::from(index)), source)).unwrap();
        assert_eq!(emitted.len(), 1);
        samples.push(emitted.remove(0));
        reconstructions.push(reconstruction(&encoder));
    }
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
    invalid.hardware = HardwarePreference::Require;
    assert_eq!(
        factory.capability(&invalid),
        CodecSupport::HardwareUnavailable
    );
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
    for blob in [vec![0], vec![40, 0, 0], vec![1, 2]] {
        invalid = configuration(64, 48, PixelFormat::Yuv420p8);
        invalid.configuration = blob;
        assert!(matches!(
            factory.capability(&invalid),
            CodecSupport::InvalidConfiguration { .. }
        ));
    }
}

#[test]
fn configuration_selects_quantizer_and_keyframe_interval() {
    assert_eq!(
        parse_configuration(&[]),
        Some((DEFAULT_BASE_Q_IDX, DEFAULT_KEYFRAME_INTERVAL))
    );
    assert_eq!(
        parse_configuration(&[200]),
        Some((200, DEFAULT_KEYFRAME_INTERVAL))
    );
    assert_eq!(parse_configuration(&[30, 1, 2]), Some((30, 258)));
    assert_eq!(parse_configuration(&[0]), None);
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
    let sample = block_on(encoder.encode(FrameIndex(0), source)).unwrap().remove(0);
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
    block_on(encoder.encode(FrameIndex(0), source())).unwrap();
    assert!(block_on(encoder.finish()).unwrap().is_empty());
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

    fn assert_decoders_match_reconstruction(config: &VideoEncoderConfig, frames: u32) {
        let decoders: Vec<&str> = ["vp9", "libvpx-vp9"]
            .into_iter()
            .filter(|name| decoder_available(name))
            .collect();
        if decoders.is_empty() {
            eprintln!("skipping independent VP9 decode because ffmpeg has no VP9 decoder");
            return;
        }
        let (samples, reconstructions, _) = encode_sequence(config, frames);
        let (width, height) = (config.coded_dimensions.width, config.coded_dimensions.height);
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
        let mut config = configuration(96, 64, PixelFormat::Yuv420p8);
        config.configuration = vec![DEFAULT_BASE_Q_IDX, 0, 5];
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
}
