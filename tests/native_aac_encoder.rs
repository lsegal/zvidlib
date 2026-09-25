#![cfg(any(target_os = "macos", windows))]

use std::f32::consts::PI;
use std::future::Future;
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::task::{Context, Poll, Waker};

use zvidlib::io::{MemorySink, MemorySource};
use zvidlib::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::{
    AacSampleReader, CancellationToken, Mp4Demuxer, Mp4DemuxerOptions, NativeAacDecoder, TrackKind,
};
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
/// [`MediaOutput`] with the native AAC-LC encoder, and hand the
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

/// Encodes `total_frames` of silence with a single full-scale-ish impulse at
/// `impulse_frame` through the native AAC-LC encoder, muxes the access units
/// and the encoder's own gapless metadata into an MP4, then demuxes and decodes
/// it back through the MP4 edit list. Returns the decoded presentation length
/// and the frame holding the loudest decoded sample.
fn encode_impulse_and_find_it(
    sample_rate: u32,
    channels: u16,
    total_frames: u64,
    impulse_frame: u64,
    configuration: Vec<u8>,
) -> (u64, u64) {
    let limits = Limits::default();
    let audio_configuration = AudioEncoderConfig {
        codec: Codec::Aac,
        profile: CodecProfile::AacLowComplexity,
        sample_rate,
        channels,
        timescale: sample_rate,
        configuration,
    };
    let mut encoder = native_aac_audio_encoder_factory()
        .create(&audio_configuration, &limits)
        .unwrap();

    // Deliberately not a multiple of the 1024-frame AAC frame, so buffers
    // straddle access units and the stream ends on a partial frame.
    let chunk = 1_000_u64;
    let mut packets = Vec::new();
    let mut start = 0;
    while start < total_frames {
        let end = (start + chunk).min(total_frames);
        let range = SampleRange::new(start, end).unwrap();
        let mut samples =
            vec![0.0_f32; usize::try_from((end - start) * u64::from(channels)).unwrap()];
        if (start..end).contains(&impulse_frame) {
            let offset = usize::try_from(impulse_frame - start).unwrap() * usize::from(channels);
            samples[offset..offset + usize::from(channels)].fill(0.9);
        }
        let buffer = AudioBuffer::new(range, sample_rate, channels, samples, &limits).unwrap();
        packets.extend(block_on(encoder.encode(FrameIndex(start / chunk), buffer)).unwrap());
        start = end;
    }
    let drain = block_on(encoder.finish()).unwrap();
    packets.extend(drain.samples);
    let encoded_frames = packets
        .iter()
        .map(|packet| u64::from(packet.duration))
        .sum::<u64>();
    assert_eq!(
        encoded_frames,
        u64::from(drain.gapless.priming) + total_frames + u64::from(drain.gapless.padding),
        "the reported priming and padding must account for exactly the emitted access units"
    );

    let mut muxer = block_on(Mp4Muxer::new(
        MemorySink::new(),
        vec![Mp4TrackConfig {
            encoder: encoder.config().clone(),
            format: Mp4TrackFormat::Audio { channels },
        }],
        packets.len(),
    ))
    .unwrap();
    for packet in packets {
        block_on(muxer.write_sample(0, packet)).unwrap();
    }
    muxer.set_audio_gapless(0, drain.gapless).unwrap();
    let bytes = block_on(muxer.finish()).unwrap().into_inner();

    let source = MemorySource::new(bytes);
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
    let track = movie
        .tracks
        .iter()
        .find(|track| track.kind == TrackKind::Audio)
        .unwrap();
    let config = track.aac_config().unwrap();
    assert_eq!(config.sample_rate, sample_rate);
    assert_eq!(config.channels, channels);
    let timing = track.audio_timing(movie.movie_timescale).unwrap();
    let encoded = block_on(track.to_encoded_audio_samples(&source, &limits)).unwrap();
    let decoder = NativeAacDecoder::new(&config, limits).unwrap();
    let mut reader =
        AacSampleReader::new(decoder, encoded, sample_rate, channels, timing, 1, limits).unwrap();
    let length = reader.presentation_length();
    let decoded = reader
        .get_range(
            SampleRange::new(0, length).unwrap(),
            &CancellationToken::new(),
        )
        .unwrap();
    let loudest = decoded
        .samples
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
        .map(|(index, _)| (index / usize::from(channels)) as u64)
        .unwrap();
    (length, loudest)
}

fn assert_impulse_round_trips(sample_rate: u32, channels: u16, configuration: Vec<u8>) {
    // Two seconds with the impulse at one: well clear of both ends, so neither
    // the encoder's priming nor its end padding can hide or clip it.
    let total_frames = u64::from(sample_rate) * 2;
    let impulse_frame = u64::from(sample_rate);
    let (length, loudest) = encode_impulse_and_find_it(
        sample_rate,
        channels,
        total_frames,
        impulse_frame,
        configuration,
    );
    assert_eq!(
        length, total_frames,
        "gapless trimming must present exactly the encoded input"
    );
    assert!(
        loudest.abs_diff(impulse_frame) <= 16,
        "an impulse at frame {impulse_frame} decoded at frame {loudest} \
         ({sample_rate} Hz, {channels} channels)"
    );
}

/// Issue #488's acceptance test: an impulse at one second decodes within 16
/// samples of frame 48,000 after a round trip through the MP4 muxer and
/// demuxer, which only holds if the encoder's reported priming is its real
/// delay and its padding matches the access units it emitted.
#[test]
fn native_aac_impulse_survives_mux_and_demux_at_48k_stereo() {
    assert_impulse_round_trips(48_000, 2, Vec::new());
}

#[test]
fn native_aac_impulse_survives_mux_and_demux_at_44_1k_mono() {
    assert_impulse_round_trips(44_100, 1, Vec::new());
}

#[test]
fn native_aac_impulse_survives_mux_and_demux_at_a_requested_bit_rate() {
    assert_impulse_round_trips(48_000, 2, 128_000_u32.to_be_bytes().to_vec());
}
