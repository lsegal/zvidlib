#![cfg(not(target_arch = "wasm32"))]

//! VP9-in-MP4 output from the native VP9 encoder, checked against the
//! independent decoders in ffmpeg (its own `vp9` and, where built in, libvpx).

use std::future::Future;
use std::io::Write;
use std::pin::pin;
use std::process::{Command, Stdio};
use std::task::{Context, Poll, Waker};
use zvidlib::io::{MemorySink, MemorySource};
use zvidlib::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::{
    CancellationToken, Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange,
    CpuFrameSource, DecodedVideoFrame, EncodedVideoSample, FrameIndex, FrameSource,
    HardwarePreference, Limits, Mp4Demuxer, Mp4DemuxerOptions, Orientation, PixelFormat, Plane,
    Result, SampleDependency, VideoDecoder, VideoDecoderConfig, VideoDecoderFactory,
    VideoDimensions, VideoEncoderConfig, VideoEncoderConformanceVector, VideoEncoderFactory,
    VideoFrame, native_vp9_video_encoder_factory, verify_video_encoder_conformance,
};
use zvidlib::{EncodedSample, WebmDemuxer, WebmDemuxerOptions, WebmMuxer};

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
            let checker = if ((u / 10) + (y / 10)) % 2 == 0 {
                60
            } else {
                0
            };
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

/// Encodes the test card with a key frame every five frames, returning the
/// MP4 track declaration and the samples in decode order.
fn encode_samples() -> (Mp4TrackConfig, Vec<EncodedSample>) {
    let limits = Limits::default();
    let dimensions = VideoDimensions::new(WIDTH, HEIGHT, &limits).unwrap();
    let configuration = VideoEncoderConfig {
        codec: Codec::Vp9,
        profile: CodecProfile::Vp9Profile0,
        coded_dimensions: dimensions,
        input_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Avoid,
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
    let mut samples = Vec::new();
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
        samples.extend(block_on(encoder.encode(FrameIndex(u64::from(index)), source)).unwrap());
    }
    assert!(block_on(encoder.finish()).unwrap().is_empty());
    (track, samples)
}

/// Encodes the test card into a VP9 MP4, returning it with its `vpcC`.
fn encode_mp4() -> (Vec<u8>, Vec<u8>) {
    let (track, samples) = encode_samples();
    let vpcc = track.encoder.decoder_config.clone();
    let mut muxer = block_on(Mp4Muxer::new(MemorySink::new(), vec![track], 64)).unwrap();
    for sample in samples {
        block_on(muxer.write_sample(0, sample)).unwrap();
    }
    (block_on(muxer.finish()).unwrap().into_inner(), vpcc)
}

/// Encodes the test card into a VP9 WebM.
fn encode_webm() -> Vec<u8> {
    let (track, samples) = encode_samples();
    let mut muxer = block_on(WebmMuxer::new(MemorySink::new(), vec![track], 64)).unwrap();
    for sample in samples {
        block_on(muxer.write_sample(0, sample)).unwrap();
    }
    block_on(muxer.finish()).unwrap().into_inner()
}

#[test]
fn vp9_webm_round_trips_through_the_demuxer() {
    let (_, samples) = encode_samples();
    let bytes = encode_webm();
    let demuxer = block_on(WebmDemuxer::open(
        &MemorySource::new(bytes),
        WebmDemuxerOptions::default(),
    ))
    .unwrap();
    assert_eq!(demuxer.tracks.len(), 1);
    let track = &demuxer.tracks[0];
    assert_eq!(track.codec, Codec::Vp9);
    let derived = zvidlib::derive_codec_string(Codec::Vp9, &track.decoder_config).unwrap();
    assert_eq!(derived.codec_string, "vp09.00.10.08");
    assert_eq!(derived.profile, CodecProfile::Vp9Profile0);
    assert_eq!(track.samples.len(), samples.len());
    for (index, (indexed, sample)) in track.samples.iter().zip(&samples).enumerate() {
        assert_eq!(indexed.is_sync, sample.is_sync, "sample {index}");
        assert_eq!(indexed.size as usize, sample.data.len(), "sample {index}");
    }
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

fn decode_file(decoder: &str, path: &std::path::Path) -> Vec<u8> {
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
        assert!(
            report.contains("width=160") && report.contains("height=90"),
            "{report}"
        );
        assert!(report.contains(&format!("nb_frames={FRAMES}")), "{report}");
    }

    let mut first: Option<Vec<u8>> = None;
    for decoder in decoders {
        let decoded = decode_file(decoder, &path);
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
            assert!(
                first == &decoded,
                "{decoder} disagrees with the first decoder"
            );
        } else {
            first = Some(decoded);
        }
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn vp9_webm_plays_in_independent_decoders() {
    let decoders = ffmpeg_decoders();
    if decoders.is_empty() {
        eprintln!("skipping independent VP9 decode because ffmpeg has no VP9 decoder");
        return;
    }
    let mp4 = std::env::temp_dir().join(format!("zvidlib-vp9-{}-ref.mp4", std::process::id()));
    let webm = std::env::temp_dir().join(format!("zvidlib-vp9-{}.webm", std::process::id()));
    std::fs::write(&mp4, encode_mp4().0).unwrap();
    std::fs::write(&webm, encode_webm()).unwrap();
    for decoder in decoders {
        // The same frames in either container decode to the same pictures.
        let from_webm = decode_file(decoder, &webm);
        assert_eq!(
            from_webm.len(),
            (WIDTH * HEIGHT * 3 * FRAMES) as usize,
            "{decoder}"
        );
        assert!(
            from_webm == decode_file(decoder, &mp4),
            "{decoder} decodes the WebM differently from the MP4"
        );
    }
    let _ = std::fs::remove_file(&mp4);
    let _ = std::fs::remove_file(&webm);
}

fn ivf(frames: &[&[u8]], dimensions: VideoDimensions) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend_from_slice(b"DKIF");
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(&32_u16.to_le_bytes());
    output.extend_from_slice(b"VP90");
    output.extend_from_slice(&(dimensions.width as u16).to_le_bytes());
    output.extend_from_slice(&(dimensions.height as u16).to_le_bytes());
    output.extend_from_slice(&30_u32.to_le_bytes());
    output.extend_from_slice(&1_u32.to_le_bytes());
    output.extend_from_slice(&(frames.len() as u32).to_le_bytes());
    output.extend_from_slice(&0_u32.to_le_bytes());
    for (index, frame) in frames.iter().enumerate() {
        output.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        output.extend_from_slice(&(index as u64).to_le_bytes());
        output.extend_from_slice(frame);
    }
    output
}

/// A conforming VP9 decoder for the conformance runner, backed by ffmpeg's
/// libvpx (or its own VP9 decoder where libvpx is not built in). Each
/// submission decodes again from the last random-access sample, so the
/// adapter keeps no state but that run of samples.
struct FfmpegVp9DecoderFactory {
    decoder: &'static str,
}

impl VideoDecoderFactory for FfmpegVp9DecoderFactory {
    fn capability(&self, configuration: &VideoDecoderConfig) -> CodecSupport {
        if configuration.codec == Codec::Vp9
            && configuration.profile == CodecProfile::Vp9Profile0
            && configuration.output_format == PixelFormat::Yuv420p8
        {
            CodecSupport::Supported {
                implementation: CodecImplementation::Software,
            }
        } else {
            CodecSupport::UnsupportedCodec
        }
    }

    fn create(
        &self,
        configuration: &VideoDecoderConfig,
        _limits: &Limits,
    ) -> Result<Box<dyn VideoDecoder>> {
        Ok(Box::new(FfmpegVp9Decoder {
            decoder: self.decoder,
            dimensions: configuration.coded_dimensions,
            color_range: configuration.color_range,
            run: Vec::new(),
        }))
    }
}

struct FfmpegVp9Decoder {
    decoder: &'static str,
    dimensions: VideoDimensions,
    color_range: ColorRange,
    run: Vec<Vec<u8>>,
}

impl VideoDecoder for FfmpegVp9Decoder {
    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        if sample.random_access {
            self.run.clear();
        }
        self.run.push(sample.data.clone());
        let frames: Vec<&[u8]> = self.run.iter().map(Vec::as_slice).collect();
        let mut child = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-c:v",
                self.decoder,
                "-f",
                "ivf",
                "-i",
                "pipe:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(&ivf(&frames, self.dimensions))
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success() && output.stderr.is_empty(),
            "{} failed: {}",
            self.decoder,
            String::from_utf8_lossy(&output.stderr)
        );
        let (width, height) = (
            self.dimensions.width as usize,
            self.dimensions.height as usize,
        );
        let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
        let frame_size = width * height + 2 * chroma_width * chroma_height;
        assert_eq!(output.stdout.len(), frame_size * frames.len());
        let last = &output.stdout[frame_size * (frames.len() - 1)..];
        let (luma, chroma) = last.split_at(width * height);
        let (cb, cr) = chroma.split_at(chroma_width * chroma_height);
        let frame = VideoFrame::new(
            self.dimensions,
            PixelFormat::Yuv420p8,
            self.color_range,
            vec![
                Plane {
                    data: luma.to_vec(),
                    stride: width,
                },
                Plane {
                    data: cb.to_vec(),
                    stride: chroma_width,
                },
                Plane {
                    data: cr.to_vec(),
                    stride: chroma_width,
                },
            ],
            &Limits::default(),
        )?;
        Ok(vec![DecodedVideoFrame {
            presentation_index: sample.presentation_index,
            frame,
        }])
    }

    fn drain(&mut self, _cancellation: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
        Ok(Vec::new())
    }

    fn reset(&mut self) -> Result<()> {
        self.run.clear();
        Ok(())
    }
}

/// A panning checkerboard over smooth chroma ramps.
fn yuv_frame(dimensions: VideoDimensions, index: u32) -> VideoFrame {
    let (width, height) = (dimensions.width as usize, dimensions.height as usize);
    let (chroma_width, chroma_height) = (width.div_ceil(2), height.div_ceil(2));
    let luma = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let u = x + index as usize * 2;
                (32 + u * 2 + ((u / 6 + y / 6) % 2) * 50) as u8
            })
        })
        .collect();
    let plane = |base: usize| -> Vec<u8> {
        (0..chroma_height)
            .flat_map(|y| (0..chroma_width).map(move |x| (base + x + index as usize + y) as u8))
            .collect()
    };
    VideoFrame::new(
        dimensions,
        PixelFormat::Yuv420p8,
        ColorRange::Limited,
        vec![
            Plane {
                data: luma,
                stride: width,
            },
            Plane {
                data: plane(90),
                stride: chroma_width,
            },
            Plane {
                data: plane(140),
                stride: chroma_width,
            },
        ],
        &Limits::default(),
    )
    .unwrap()
}

#[test]
fn native_vp9_encoder_passes_conformance_against_an_independent_decoder() {
    let decoders = ffmpeg_decoders();
    // Prefer libvpx, the reference decoder, when ffmpeg was built with it.
    let Some(&decoder) = decoders
        .iter()
        .find(|&&name| name == "libvpx-vp9")
        .or(decoders.first())
    else {
        eprintln!("skipping VP9 conformance because ffmpeg has no VP9 decoder");
        return;
    };
    let limits = Limits::default();
    let dimensions = VideoDimensions::new(50, 34, &limits).unwrap();
    let vector = VideoEncoderConformanceVector {
        name: "vp9-pan-key-every-4".into(),
        configuration: VideoEncoderConfig {
            codec: Codec::Vp9,
            profile: CodecProfile::Vp9Profile0,
            coded_dimensions: dimensions,
            input_format: PixelFormat::Yuv420p8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            timescale: 30,
            frame_duration: 1,
            configuration: vec![50, 0, 4],
        },
        decoder_configuration: VideoDecoderConfig {
            codec: Codec::Vp9,
            profile: CodecProfile::Vp9Profile0,
            coded_dimensions: dimensions,
            output_format: PixelFormat::Yuv420p8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: Vec::new(),
        },
        frames: (0..9).map(|index| yuv_frame(dimensions, index)).collect(),
        minimum_psnr_db: 35.0,
    };
    let report = block_on(verify_video_encoder_conformance(
        &native_vp9_video_encoder_factory(),
        &FfmpegVp9DecoderFactory { decoder },
        &vector,
        limits,
    ))
    .unwrap();
    assert_eq!(report.frames_encoded, 9);
    assert_eq!(report.packets_emitted, 9);
}

/// Issue #528's acceptance criterion: the encoder's output decodes
/// frame-accurately through zvidlib's own VP9 decoder, through the shared
/// conformance runner and under backwards and forwards seeks across key
/// frames.
#[test]
fn native_vp9_encoder_round_trips_through_the_native_decoder() {
    let limits = Limits::default();
    let dimensions = VideoDimensions::new(WIDTH, HEIGHT, &limits).unwrap();
    let frames: Vec<VideoFrame> = (0..FRAMES)
        .map(|index| {
            VideoFrame::new(
                dimensions,
                PixelFormat::Rgba8,
                ColorRange::Limited,
                vec![Plane {
                    data: rgba_frame(index),
                    stride: (WIDTH * 4) as usize,
                }],
                &limits,
            )
            .unwrap()
        })
        .collect();
    let encoder_configuration = VideoEncoderConfig {
        codec: Codec::Vp9,
        profile: CodecProfile::Vp9Profile0,
        coded_dimensions: dimensions,
        input_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Avoid,
        timescale: 30,
        frame_duration: 1,
        configuration: vec![60, 0, 5],
    };
    let decoder_configuration = VideoDecoderConfig {
        codec: Codec::Vp9,
        profile: CodecProfile::Vp9Profile0,
        coded_dimensions: dimensions,
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Avoid,
        configuration: Vec::new(),
    };
    let vector = VideoEncoderConformanceVector {
        name: "vp9-rgba-pan-key-every-5".into(),
        configuration: encoder_configuration.clone(),
        decoder_configuration: decoder_configuration.clone(),
        frames: frames.clone(),
        minimum_psnr_db: 30.0,
    };
    let report = block_on(verify_video_encoder_conformance(
        &native_vp9_video_encoder_factory(),
        &zvidlib::native_vp9_video_decoder_factory(),
        &vector,
        limits,
    ))
    .unwrap();
    assert_eq!(report.frames_encoded, u64::from(FRAMES));

    // Every frame is the same whichever order it is asked for in.
    let mut encoder = native_vp9_video_encoder_factory()
        .create(&encoder_configuration, &limits)
        .unwrap();
    let mut samples = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        let source = FrameSource::Cpu(CpuFrameSource {
            frame,
            orientation: Orientation::TopLeft,
        });
        for sample in block_on(encoder.encode(FrameIndex(index as u64), source)).unwrap() {
            samples.push(EncodedVideoSample {
                presentation_index: FrameIndex(index as u64),
                random_access: sample.is_sync,
                data: sample.data,
            });
        }
    }
    let mut configuration = decoder_configuration;
    configuration.configuration = encoder.config().decoder_config.clone();
    let decoder = zvidlib::native_vp9_video_decoder_factory();
    let cancellation = CancellationToken::new();
    let mut sequential =
        zvidlib::ExactFrameReader::new(&decoder, configuration.clone(), samples.clone(), limits)
            .unwrap();
    let in_order: Vec<VideoFrame> = (0..u64::from(FRAMES))
        .map(|index| sequential.get(FrameIndex(index), &cancellation).unwrap())
        .collect();
    let mut seeking =
        zvidlib::ExactFrameReader::new(&decoder, configuration, samples, limits).unwrap();
    for index in [11_u64, 3, 9, 0, 6, 4, 10] {
        let frame = seeking.get(FrameIndex(index), &cancellation).unwrap();
        assert!(
            frame == in_order[index as usize],
            "frame {index} differs after a seek"
        );
    }
}
