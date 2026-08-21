use std::io::Write;
use std::process::{Command, Stdio};
use std::task::{Context, Poll, Waker};
use std::{future::Future, pin::pin};
use zvidlib::{
    CancellationToken, Codec, CodecImplementation, CodecProfile, CodecSupport, ColorRange,
    CpuFrameSource, DecodedVideoFrame, EncodedVideoSample, Error, ErrorKind, FrameIndex,
    FrameSource, HardwarePreference, Limits, Orientation, PixelFormat, Plane, Result, VideoDecoder,
    VideoDecoderConfig, VideoDecoderFactory, VideoDimensions, VideoEncoderConfig,
    VideoEncoderConformanceVector, VideoEncoderFactory, VideoFrame,
    native_av1_video_encoder_factory, verify_video_encoder_conformance,
};

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let mut context = Context::from_waker(Waker::noop());
    let mut future = pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
    }
}

fn append_ivf_header(output: &mut Vec<u8>, width: u16, height: u16, frames: u32) {
    output.extend_from_slice(b"DKIF");
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(&32_u16.to_le_bytes());
    output.extend_from_slice(b"AV01");
    output.extend_from_slice(&width.to_le_bytes());
    output.extend_from_slice(&height.to_le_bytes());
    output.extend_from_slice(&30_u32.to_le_bytes());
    output.extend_from_slice(&1_u32.to_le_bytes());
    output.extend_from_slice(&frames.to_le_bytes());
    output.extend_from_slice(&0_u32.to_le_bytes());
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn decode_with_ffmpeg(data: &[u8], dimensions: VideoDimensions) -> Result<Vec<u8>> {
    let width = u16::try_from(dimensions.width)
        .map_err(|_| Error::new(ErrorKind::ResourceLimit, "test IVF width overflow"))?;
    let height = u16::try_from(dimensions.height)
        .map_err(|_| Error::new(ErrorKind::ResourceLimit, "test IVF height overflow"))?;
    let mut ivf = Vec::new();
    append_ivf_header(&mut ivf, width, height, 1);
    ivf.extend_from_slice(
        &u32::try_from(data.len())
            .map_err(|_| Error::new(ErrorKind::ResourceLimit, "test IVF frame overflow"))?
            .to_le_bytes(),
    );
    ivf.extend_from_slice(&0_u64.to_le_bytes());
    ivf.extend_from_slice(data);

    let mut child = Command::new("ffmpeg")
        .args([
            "-v", "error", "-f", "ivf", "-i", "pipe:0", "-map", "0:v:0", "-f", "rawvideo",
            "-pix_fmt", "gray", "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            Error::new(ErrorKind::Codec, format!("failed to start ffmpeg: {error}"))
        })?;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&ivf)
        .map_err(|error| Error::new(ErrorKind::Codec, format!("failed to feed ffmpeg: {error}")))?;
    let output = child.wait_with_output().map_err(|error| {
        Error::new(
            ErrorKind::Codec,
            format!("failed to wait for ffmpeg: {error}"),
        )
    })?;
    if !output.status.success() {
        return Err(Error::new(
            ErrorKind::Codec,
            format!(
                "independent ffmpeg decode failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        ));
    }
    Ok(output.stdout)
}

struct FfmpegAv1DecoderFactory;

impl VideoDecoderFactory for FfmpegAv1DecoderFactory {
    fn capability(&self, configuration: &VideoDecoderConfig) -> CodecSupport {
        if configuration.codec == Codec::Av1
            && configuration.profile == CodecProfile::Av1Main
            && configuration.output_format == PixelFormat::Gray8
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
        Ok(Box::new(FfmpegAv1Decoder {
            dimensions: configuration.coded_dimensions,
            color_range: configuration.color_range,
        }))
    }
}

struct FfmpegAv1Decoder {
    dimensions: VideoDimensions,
    color_range: ColorRange,
}

impl VideoDecoder for FfmpegAv1Decoder {
    fn submit(
        &mut self,
        sample: &EncodedVideoSample,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<DecodedVideoFrame>> {
        let pixels = decode_with_ffmpeg(&sample.data, self.dimensions)?;
        let frame = VideoFrame::new(
            self.dimensions,
            PixelFormat::Gray8,
            self.color_range,
            vec![Plane {
                data: pixels,
                stride: self.dimensions.width as usize,
            }],
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
        Ok(())
    }
}

#[test]
fn native_av1_output_decodes_with_independent_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("skipping independent AV1 decode because ffmpeg is unavailable");
        return;
    }

    let limits = Limits::default();
    let dimensions = VideoDimensions::new(17, 9, &limits).unwrap();
    let configuration = VideoEncoderConfig {
        codec: Codec::Av1,
        profile: CodecProfile::Av1Main,
        coded_dimensions: dimensions,
        input_format: PixelFormat::Gray8,
        color_range: ColorRange::Full,
        hardware: HardwarePreference::Avoid,
        timescale: 30,
        frame_duration: 1,
        configuration: Vec::new(),
    };
    let factory = native_av1_video_encoder_factory();
    let mut encoder = factory.create(&configuration, &limits).unwrap();
    let mut ivf = Vec::new();
    append_ivf_header(&mut ivf, 17, 9, 3);
    let mut expected = Vec::new();

    for index in 0..3_u64 {
        let pixels = (0..dimensions.height)
            .flat_map(|y| {
                (0..dimensions.width)
                    .map(move |x| ((x * 11 + y * 17 + index as u32 * 29) & 0xff) as u8)
            })
            .collect::<Vec<_>>();
        expected.extend_from_slice(&pixels);
        let frame = VideoFrame::new(
            dimensions,
            PixelFormat::Gray8,
            ColorRange::Full,
            vec![Plane {
                data: pixels,
                stride: dimensions.width as usize,
            }],
            &limits,
        )
        .unwrap();
        let packet = block_on(encoder.encode(
            FrameIndex(index),
            FrameSource::Cpu(CpuFrameSource {
                frame: &frame,
                orientation: Orientation::TopLeft,
            }),
        ))
        .unwrap()
        .remove(0);
        ivf.extend_from_slice(&(packet.data.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&index.to_le_bytes());
        ivf.extend_from_slice(&packet.data);
    }

    let mut child = Command::new("ffmpeg")
        .args([
            "-v", "error", "-f", "ivf", "-i", "pipe:0", "-map", "0:v:0", "-f", "rawvideo",
            "-pix_fmt", "gray", "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&ivf).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "independent ffmpeg decode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, expected);
}

#[test]
fn native_av1_passes_the_shared_encoder_conformance_runner() {
    if !ffmpeg_available() {
        eprintln!("skipping AV1 conformance because ffmpeg is unavailable");
        return;
    }

    let limits = Limits::default();
    let dimensions = VideoDimensions::new(17, 9, &limits).unwrap();
    let frames = (0..3_u32)
        .map(|frame_index| {
            let pixels = (0..dimensions.height)
                .flat_map(|y| {
                    (0..dimensions.width)
                        .map(move |x| ((x * 13 + y * 19 + frame_index * 31) & 0xff) as u8)
                })
                .collect::<Vec<_>>();
            VideoFrame::new(
                dimensions,
                PixelFormat::Gray8,
                ColorRange::Full,
                vec![Plane {
                    data: pixels,
                    stride: dimensions.width as usize,
                }],
                &limits,
            )
            .unwrap()
        })
        .collect();
    let vector = VideoEncoderConformanceVector {
        name: "native lossless monochrome AV1".into(),
        configuration: VideoEncoderConfig {
            codec: Codec::Av1,
            profile: CodecProfile::Av1Main,
            coded_dimensions: dimensions,
            input_format: PixelFormat::Gray8,
            color_range: ColorRange::Full,
            hardware: HardwarePreference::Avoid,
            timescale: 30_000,
            frame_duration: 1_001,
            configuration: Vec::new(),
        },
        decoder_configuration: VideoDecoderConfig {
            codec: Codec::Av1,
            profile: CodecProfile::Av1Main,
            coded_dimensions: dimensions,
            output_format: PixelFormat::Gray8,
            color_range: ColorRange::Full,
            hardware: HardwarePreference::Avoid,
            configuration: Vec::new(),
        },
        frames,
        minimum_psnr_db: 100.0,
    };
    let report = block_on(verify_video_encoder_conformance(
        &native_av1_video_encoder_factory(),
        &FfmpegAv1DecoderFactory,
        &vector,
        limits,
    ))
    .unwrap();
    assert_eq!(report.frames_encoded, 3);
    assert_eq!(report.packets_emitted, 3);
    assert!(report.minimum_observed_psnr_db.is_infinite());
}
