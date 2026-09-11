#![cfg(target_os = "macos")]

use std::f32::consts::PI;
use std::future::Future;
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::task::{Context, Poll, Waker};

use zvidlib::io::MemorySink;
use zvidlib::{
    AudioBuffer, AudioEncoder, AudioEncoderConfig, AudioEncoderFactory, AudioEncoderFormat, Codec,
    CodecProfile, ColorRange, CpuFrameSource, EncodedSample, EncoderConfig, EncoderFuture,
    FrameIndex, FrameRate, FrameSource, HardwarePreference, Limits, MediaOutput, Orientation,
    OutputOptions, PixelFormat, Plane, SampleRange, Timeline, VideoDimensions, VideoEncoder,
    VideoEncoderConfig, VideoEncoderFactory, VideoEncoderFormat, VideoFrame,
    native_aac_audio_encoder_factory, native_av1_video_encoder_factory,
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

fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

/// Forwards to a boxed trait object so a factory's `Box<dyn VideoEncoder>` can
/// be plugged into [`MediaOutput`], which is generic over a concrete encoder
/// type rather than a trait object.
struct BoxedVideoEncoder(Box<dyn VideoEncoder>);

impl VideoEncoder for BoxedVideoEncoder {
    fn config(&self) -> &EncoderConfig {
        self.0.config()
    }

    fn format(&self) -> VideoEncoderFormat {
        self.0.format()
    }

    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        frame: FrameSource<'a>,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        self.0.encode(index, frame)
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, Vec<EncodedSample>> {
        self.0.finish()
    }
}

/// The audio counterpart to [`BoxedVideoEncoder`].
struct BoxedAudioEncoder(Box<dyn AudioEncoder>);

impl AudioEncoder for BoxedAudioEncoder {
    fn config(&self) -> &EncoderConfig {
        self.0.config()
    }

    fn format(&self) -> AudioEncoderFormat {
        self.0.format()
    }

    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        buffer: AudioBuffer,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        self.0.encode(index, buffer)
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, zvidlib::AudioDrain> {
        self.0.finish()
    }
}

fn sine_wave(range: SampleRange, sample_rate: u32, channels: u16, frequency: f32) -> AudioBuffer {
    let frames = usize::try_from(range.len()).unwrap();
    let mut samples = Vec::with_capacity(frames * usize::from(channels));
    for frame in 0..frames {
        let t = (range.start + frame as u64) as f32 / sample_rate as f32;
        let value = (2.0 * PI * frequency * t).sin() * 0.25;
        for _ in 0..channels {
            samples.push(value);
        }
    }
    AudioBuffer::new(range, sample_rate, channels, samples, &Limits::default()).unwrap()
}

/// Mux a few frames of synthesized video and a 440 Hz sine tone through
/// [`MediaOutput`] with the native AudioToolbox AAC-LC encoder, and hand the
/// resulting MP4 to an independent ffmpeg to decode both tracks end to end -
/// the audio counterpart to `native_av1_output_decodes_with_independent_ffmpeg`.
#[test]
fn native_aac_output_decodes_with_independent_ffmpeg() {
    if !tool_available("ffmpeg") || !tool_available("ffprobe") {
        eprintln!("skipping independent AAC decode because ffmpeg/ffprobe is unavailable");
        return;
    }

    let limits = Limits::default();
    let sample_rate = 48_000_u32;
    let channels = 2_u16;

    let audio_configuration = AudioEncoderConfig {
        codec: Codec::Aac,
        profile: CodecProfile::AacLowComplexity,
        sample_rate,
        channels,
        timescale: sample_rate,
        configuration: Vec::new(),
    };
    let audio_factory = native_aac_audio_encoder_factory();
    let audio_encoder =
        BoxedAudioEncoder(audio_factory.create(&audio_configuration, &limits).unwrap());

    let dimensions = VideoDimensions::new(16, 16, &limits).unwrap();
    let video_configuration = VideoEncoderConfig {
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
    let video_factory = native_av1_video_encoder_factory();
    let video_encoder =
        BoxedVideoEncoder(video_factory.create(&video_configuration, &limits).unwrap());

    let timeline = Timeline::new(FrameRate::new(30, 1).unwrap(), sample_rate).unwrap();
    let mut output = block_on(MediaOutput::new(
        MemorySink::new(),
        video_encoder,
        audio_encoder,
        timeline,
        OutputOptions::default(),
    ))
    .unwrap();

    let samples_per_video_frame = u64::from(sample_rate / 30);
    for index in 0..6_u64 {
        let pixels = (0..dimensions.height)
            .flat_map(|y| {
                (0..dimensions.width)
                    .map(move |x| ((x * 11 + y * 17 + index as u32 * 29) & 0xff) as u8)
            })
            .collect::<Vec<_>>();
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
        block_on(output.put_video(
            FrameIndex(index),
            FrameSource::Cpu(CpuFrameSource {
                frame: &frame,
                orientation: Orientation::TopLeft,
            }),
        ))
        .unwrap();

        let start = index * samples_per_video_frame;
        let range = SampleRange::new(start, start + samples_per_video_frame).unwrap();
        let buffer = sine_wave(range, sample_rate, channels, 440.0);
        block_on(output.put_audio(FrameIndex(index), buffer)).unwrap();
    }

    let bytes = block_on(output.finish()).unwrap().into_inner();
    assert!(!bytes.is_empty(), "the muxed MP4 must not be empty");

    let path = std::env::temp_dir().join(format!(
        "zvidlib-native-aac-encoder-test-{}.mp4",
        std::process::id()
    ));
    std::fs::write(&path, &bytes).unwrap();

    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&path)
        .output();
    let decode = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-f", "null", "-"])
        .output();
    let _ = std::fs::remove_file(&path);

    let probe = probe.unwrap();
    assert!(
        probe.status.success(),
        "ffprobe could not inspect the audio track: {}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let probe_output = String::from_utf8_lossy(&probe.stdout);
    assert!(
        probe_output.contains("codec_name=aac"),
        "expected an AAC audio track, ffprobe reported: {probe_output}"
    );
    assert!(
        probe_output.contains(&format!("sample_rate={sample_rate}")),
        "expected sample rate {sample_rate}, ffprobe reported: {probe_output}"
    );
    assert!(
        probe_output.contains(&format!("channels={channels}")),
        "expected {channels} channels, ffprobe reported: {probe_output}"
    );

    let decode = decode.unwrap();
    assert!(
        decode.status.success(),
        "independent ffmpeg decode failed: {}",
        String::from_utf8_lossy(&decode.stderr)
    );
}
