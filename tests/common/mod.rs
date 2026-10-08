//! A WebM the on-demand tests share with the bundled MP4 sample (issue #685):
//! VP9 video of a moving gradient and, beside it, Opus audio, encoded by the
//! crate's own encoders so no binary fixture is checked in.

#![allow(dead_code)]

use std::future::Future;
use std::task::{Context, Poll, Waker};

use zvidlib::io::MemorySink;
use zvidlib::mp4::{Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::transfer::{CpuFrameSource, FrameSource, Orientation};
use zvidlib::{
    AudioBuffer, AudioEncoderConfig, AudioEncoderFactory, Codec, CodecProfile, ColorRange,
    FrameIndex, HardwarePreference, Limits, PixelFormat, Plane, SampleRange, VideoDimensions,
    VideoEncoderConfig, VideoEncoderFactory, VideoFrame, WebmMuxer,
    native_opus_audio_encoder_factory, native_vp9_video_encoder_factory,
};

/// The WebM's frame rate and length: four seconds, with a key frame at frame
/// 0 and frame 60, the VP9 encoder's default interval.
pub const WEBM_RATE: u32 = 30;
pub const WEBM_FRAMES: u64 = 120;
const WIDTH: u32 = 64;
const HEIGHT: u32 = 36;

pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
    }
}

fn gradient(index: u64) -> Vec<u8> {
    (0..HEIGHT)
        .flat_map(|y| {
            (0..WIDTH).flat_map(move |x| {
                let level = ((x * 3 + y * 5 + index as u32 * 4) % 220 + 16) as u8;
                [level, level / 2 + 64, 255 - level, 255]
            })
        })
        .collect()
}

/// The WebM's bytes: VP9 video and, with `opus`, Opus audio interleaved
/// beside it in presentation order, as a WebM cluster requires.
pub fn vp9_opus_webm(opus: bool) -> Vec<u8> {
    block_on(encode(opus))
}

async fn encode(opus: bool) -> Vec<u8> {
    let limits = Limits::default();
    let dimensions = VideoDimensions::new(WIDTH, HEIGHT, &limits).unwrap();
    let mut video_encoder = native_vp9_video_encoder_factory()
        .create(
            &VideoEncoderConfig {
                codec: Codec::Vp9,
                profile: CodecProfile::Vp9Profile0,
                coded_dimensions: dimensions,
                input_format: PixelFormat::Rgba8,
                color_range: ColorRange::Limited,
                hardware: HardwarePreference::Avoid,
                timescale: WEBM_RATE,
                frame_duration: 1,
                configuration: Vec::new(),
            },
            &limits,
        )
        .unwrap();
    let mut video = Vec::new();
    for index in 0..WEBM_FRAMES {
        let frame = VideoFrame::new(
            dimensions,
            PixelFormat::Rgba8,
            ColorRange::Limited,
            vec![Plane {
                data: gradient(index),
                stride: WIDTH as usize * 4,
            }],
            &limits,
        )
        .unwrap();
        video.extend(
            video_encoder
                .encode(
                    FrameIndex(index),
                    FrameSource::Cpu(CpuFrameSource {
                        frame: &frame,
                        orientation: Orientation::TopLeft,
                    }),
                )
                .await
                .unwrap(),
        );
    }
    video.extend(video_encoder.finish().await.unwrap());
    let mut tracks = vec![Mp4TrackConfig {
        encoder: video_encoder.config().clone(),
        format: Mp4TrackFormat::Video(dimensions),
    }];
    let seconds = |pts: i64, timescale: u32| pts as f64 / f64::from(timescale);
    let mut samples: Vec<_> = video
        .into_iter()
        .map(|sample| (seconds(sample.pts, WEBM_RATE), 0, sample))
        .collect();

    let mut gapless = None;
    if opus {
        let frames = 48_000 * WEBM_FRAMES / u64::from(WEBM_RATE);
        let pcm: Vec<f32> = (0..frames)
            .flat_map(|i| {
                let t = i as f32 / 48_000.0;
                [
                    0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin(),
                    0.3 * (2.0 * std::f32::consts::PI * 660.0 * t).sin(),
                ]
            })
            .collect();
        let mut audio_encoder = native_opus_audio_encoder_factory()
            .create(
                &AudioEncoderConfig {
                    codec: Codec::Opus,
                    profile: CodecProfile::Opus,
                    sample_rate: 48_000,
                    channels: 2,
                    timescale: 48_000,
                    configuration: 128_000_u32.to_be_bytes().to_vec(),
                },
                &limits,
            )
            .unwrap();
        let buffer = AudioBuffer::new(
            SampleRange::new(0, frames).unwrap(),
            48_000,
            2,
            pcm,
            &limits,
        )
        .unwrap();
        let mut packets = audio_encoder.encode(FrameIndex(0), buffer).await.unwrap();
        let drain = audio_encoder.finish().await.unwrap();
        packets.extend(drain.samples);
        tracks.push(Mp4TrackConfig {
            encoder: audio_encoder.config().clone(),
            format: Mp4TrackFormat::Audio { channels: 2 },
        });
        samples.extend(
            packets
                .into_iter()
                .map(|packet| (seconds(packet.pts, 48_000), 1, packet)),
        );
        gapless = Some(drain.gapless);
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut muxer = WebmMuxer::new(MemorySink::new(), tracks, 100_000)
        .await
        .unwrap();
    for (_, track, sample) in samples {
        muxer.write_sample(track, sample).await.unwrap();
    }
    if let Some(gapless) = gapless {
        muxer.set_audio_gapless(1, gapless).unwrap();
    }
    muxer.finish().await.unwrap().into_inner()
}
