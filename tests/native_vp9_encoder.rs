#![cfg(not(target_arch = "wasm32"))]

//! VP9-in-MP4 output from the native VP9 encoder, checked against the
//! independent decoders in ffmpeg (its own `vp9` and, where built in, libvpx).

use std::future::Future;
use std::pin::pin;
use std::process::{Command, Stdio};
use std::task::{Context, Poll, Waker};
use zvidlib::io::{MemorySink, MemorySource};
use zvidlib::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::{
    Codec, CodecProfile, ColorRange, CpuFrameSource, FrameIndex, FrameSource, HardwarePreference,
    Limits, Mp4Demuxer, Mp4DemuxerOptions, Orientation, PixelFormat, Plane, SampleDependency,
    VideoDimensions, VideoEncoderConfig, VideoEncoderFactory, VideoFrame,
    native_vp9_video_encoder_factory,
};

const WIDTH: u32 = 160;
const HEIGHT: u32 = 90;
const FRAMES: u32 = 12;

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let mut context = Context::from_waker(Waker::noop());
    let mut future = pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
    }
}

/// An RGBA test card that pans right two pixels a frame.
fn rgba_frame(index: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let u = x + index * 2;
            let checker = if ((u / 10) + (y / 10)) % 2 == 0 { 60 } else { 0 };
            pixels.extend_from_slice(&[
                (u * 255 / (WIDTH + 2 * FRAMES)) as u8,
                (y * 255 / HEIGHT) as u8 / 2 + checker,
                (255 - u * 200 / (WIDTH + 2 * FRAMES)) as u8,
                255,
            ]);
        }
    }
    pixels
}

/// Encodes the test card into a VP9 MP4 with a key frame every five frames.
fn encode_mp4() -> (Vec<u8>, Vec<u8>) {
    let limits = Limits::default();
    let dimensions = VideoDimensions::new(WIDTH, HEIGHT, &limits).unwrap();
    let configuration = VideoEncoderConfig {
        codec: Codec::Vp9,
        profile: CodecProfile::Vp9Profile0,
        coded_dimensions: dimensions,
        input_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Prefer,
        timescale: 30,
        frame_duration: 1,
        configuration: vec![60, 0, 5],
    };
    let mut encoder = native_vp9_video_encoder_factory()
        .create(&configuration, &limits)
        .unwrap();
    let track = Mp4TrackConfig {
        encoder: encoder.config().clone(),
        format: Mp4TrackFormat::Video(dimensions),
    };
    let vpcc = encoder.config().decoder_config.clone();
    let mut muxer = block_on(Mp4Muxer::new(MemorySink::new(), vec![track], 64)).unwrap();
    for index in 0..FRAMES {
        let frame = VideoFrame::new(
            dimensions,
            PixelFormat::Rgba8,
            ColorRange::Limited,
            vec![Plane {
                data: rgba_frame(index),
                stride: (WIDTH * 4) as usize,
            }],
            &limits,
        )
        .unwrap();
        let source = FrameSource::Cpu(CpuFrameSource {
            frame: &frame,
            orientation: Orientation::TopLeft,
        });
        for sample in block_on(encoder.encode(FrameIndex(u64::from(index)), source)).unwrap() {
            block_on(muxer.write_sample(0, sample)).unwrap();
        }
    }
    assert!(block_on(encoder.finish()).unwrap().is_empty());
    let bytes = block_on(muxer.finish()).unwrap().into_inner();
    (bytes, vpcc)
}

#[test]
fn vp9_mp4_round_trips_through_the_demuxer() {
    let (bytes, vpcc) = encode_mp4();
    let demuxer = block_on(Mp4Demuxer::open(
        &MemorySource::new(bytes),
        Mp4DemuxerOptions::default(),
    ))
    .unwrap();
    assert_eq!(demuxer.tracks.len(), 1);
    let track = &demuxer.tracks[0];
    assert_eq!(track.codec, Codec::Vp9);
    assert_eq!(track.decoder_config, vpcc);
    assert_eq!(
        track.dimensions,
        Some(VideoDimensions {
            width: WIDTH,
            height: HEIGHT
        })
    );
    assert_eq!(track.samples.len(), FRAMES as usize);
    for (index, sample) in track.samples.iter().enumerate() {
        let key = index % 5 == 0;
        assert_eq!(sample.is_sync, key, "sample {index}");
        let dependency = if key {
            SampleDependency::INDEPENDENT
        } else {
            SampleDependency::DEPENDENT
        };
        assert_eq!(sample.dependency, dependency, "sample {index}");
    }
    let derived = zvidlib::derive_codec_string(Codec::Vp9, &track.decoder_config).unwrap();
    assert_eq!(derived.codec_string, "vp09.00.10.08");
}

fn ffmpeg_decoders() -> Vec<&'static str> {
    let Ok(output) = Command::new("ffmpeg")
        .args(["-hide_banner", "-decoders"])
        .output()
    else {
        return Vec::new();
    };
    let listing = String::from_utf8_lossy(&output.stdout).into_owned();
    ["vp9", "libvpx-vp9"]
        .into_iter()
        .filter(|name| listing.split_whitespace().any(|word| word == *name))
        .collect()
}

fn decode_mp4(decoder: &str, path: &std::path::Path) -> Vec<u8> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-c:v", decoder, "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        output.status.success() && output.stderr.is_empty(),
        "{decoder} could not decode the VP9 MP4: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn vp9_mp4_plays_in_independent_decoders() {
    let decoders = ffmpeg_decoders();
    if decoders.is_empty() {
        eprintln!("skipping independent VP9 decode because ffmpeg has no VP9 decoder");
        return;
    }
    let (bytes, _) = encode_mp4();
    let path = std::env::temp_dir().join(format!("zvidlib-vp9-{}.mp4", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();

    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,width,height,nb_frames",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&path)
        .output();
    if let Ok(probe) = probe {
        let report = String::from_utf8_lossy(&probe.stdout);
        assert!(report.contains("codec_name=vp9"), "{report}");
        assert!(report.contains("width=160") && report.contains("height=90"), "{report}");
        assert!(report.contains(&format!("nb_frames={FRAMES}")), "{report}");
    }

    let mut first: Option<Vec<u8>> = None;
    for decoder in decoders {
        let decoded = decode_mp4(decoder, &path);
        let frame_size = (WIDTH * HEIGHT * 3) as usize;
        assert_eq!(decoded.len(), frame_size * FRAMES as usize, "{decoder}");
        for index in 0..FRAMES {
            let rgba = rgba_frame(index);
            let source: Vec<u8> = rgba
                .chunks_exact(4)
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
                .collect();
            let frame = &decoded[index as usize * frame_size..][..frame_size];
            let error = source
                .iter()
                .zip(frame)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum::<f64>()
                / frame_size as f64;
            let psnr = 10.0 * (255.0 * 255.0 / error.max(1e-9)).log10();
            assert!(psnr > 30.0, "{decoder} frame {index}: {psnr:.1} dB");
        }
        // Every conforming decoder produces the same pictures.
        if let Some(first) = &first {
            assert!(first == &decoded, "{decoder} disagrees with the first decoder");
        } else {
            first = Some(decoded);
        }
    }
    let _ = std::fs::remove_file(&path);
}
