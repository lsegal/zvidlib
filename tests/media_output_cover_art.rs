//! The cover `MediaOutput` writes (#510): generated from a chosen frame, from
//! the last frame of a stream shorter than that, or not at all, and overridden
//! by explicit cover art. Muxing and demuxing the cover itself is
//! `zvidlib-container`'s, in `crates/zvidlib-container/tests/mp4_cover_art.rs`.

use std::future::Future;
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::task::{Context, Poll, Waker};
use zvidlib::io::{MemorySink, MemorySource};
use zvidlib::{
    AudioBuffer, AudioDrain, AudioEncoder, AudioEncoderFormat, AudioGapless, Codec, ColorRange,
    CoverArt, CoverArtFormat, CoverSource, CpuFrameSource, EncodedSample, EncoderConfig,
    EncoderFuture, ErrorKind, FrameIndex, FrameRate, FrameSource, Limits, MediaOutput, Mp4Demuxer,
    Mp4DemuxerOptions, Orientation, OutputOptions, PixelFormat, Plane, SampleDependency, Timeline,
    VideoDimensions, VideoEncoder, VideoEncoderFormat, VideoFrame,
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

fn codec_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut output = u32::try_from(payload.len() + 8)
        .unwrap()
        .to_be_bytes()
        .to_vec();
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    output
}

fn demux(bytes: Vec<u8>) -> Mp4Demuxer {
    block_on(Mp4Demuxer::open(
        &MemorySource::new(bytes),
        Mp4DemuxerOptions::default(),
    ))
    .unwrap()
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/crates/zvidlib-container/tests/fixtures/cover_art/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

fn jpeg() -> CoverArt {
    CoverArt {
        format: CoverArtFormat::Jpeg,
        data: fixture("cover.jpg"),
    }
}

fn png() -> CoverArt {
    CoverArt {
        format: CoverArtFormat::Png,
        data: fixture("cover.png"),
    }
}


/// real codec.
struct PassthroughVideoEncoder {
    config: EncoderConfig,
    format: VideoEncoderFormat,
}

impl VideoEncoder for PassthroughVideoEncoder {
    fn config(&self) -> &EncoderConfig {
        &self.config
    }

    fn format(&self) -> VideoEncoderFormat {
        self.format
    }

    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        _frame: FrameSource<'a>,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move {
            let time = i64::try_from(index.0).unwrap() * 1_000;
            Ok(vec![EncodedSample {
                data: vec![b'V', index.0 as u8],
                dts: time,
                pts: time,
                duration: 1_000,
                is_sync: true,
                dependency: SampleDependency::INDEPENDENT,
            }])
        })
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async { Ok(Vec::new()) })
    }
}

struct SilentAudioEncoder {
    config: EncoderConfig,
    format: AudioEncoderFormat,
}

impl AudioEncoder for SilentAudioEncoder {
    fn config(&self) -> &EncoderConfig {
        &self.config
    }

    fn format(&self) -> AudioEncoderFormat {
        self.format
    }

    fn encode<'a>(
        &'a mut self,
        _index: FrameIndex,
        buffer: AudioBuffer,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move {
            Ok(vec![EncodedSample {
                data: vec![b'A'],
                dts: i64::try_from(buffer.range.start).unwrap(),
                pts: i64::try_from(buffer.range.start).unwrap(),
                duration: u32::try_from(buffer.range.len()).unwrap(),
                is_sync: true,
                dependency: SampleDependency::INDEPENDENT,
            }])
        })
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, AudioDrain> {
        Box::pin(async {
            Ok(AudioDrain {
                samples: Vec::new(),
                gapless: AudioGapless::default(),
            })
        })
    }
}

const FRAME_WIDTH: u32 = 40;
const FRAME_HEIGHT: u32 = 24;

/// A frame whose every pixel encodes its index, so a cover shows which frame
/// it came from.
fn indexed_frame(index: u64) -> VideoFrame {
    let data = (0..FRAME_WIDTH * FRAME_HEIGHT)
        .flat_map(|pixel| {
            let (x, y) = (pixel % FRAME_WIDTH, pixel / FRAME_WIDTH);
            [index as u8 * 20, (x * 6) as u8, (y * 10) as u8, 255]
        })
        .collect();
    VideoFrame::new(
        VideoDimensions::new(FRAME_WIDTH, FRAME_HEIGHT, &Limits::default()).unwrap(),
        PixelFormat::Rgba8,
        ColorRange::Full,
        vec![Plane {
            data,
            stride: FRAME_WIDTH as usize * 4,
        }],
        &Limits::default(),
    )
    .unwrap()
}

/// Writes `frames` indexed frames through a `MediaOutput`, applying
/// `cover_art` after capture when given.
fn record(options: OutputOptions, frames: u64, cover_art: Option<CoverArt>) -> Vec<u8> {
    block_on(async {
        let timeline = Timeline::new(FrameRate::new(30, 1).unwrap(), 48_000).unwrap();
        let video = PassthroughVideoEncoder {
            config: EncoderConfig {
                codec: Codec::Av1,
                timescale: 30_000,
                decoder_config: codec_box(b"av1C", &[0x81, 0, 0, 0]),
            },
            format: VideoEncoderFormat {
                dimensions: VideoDimensions {
                    width: FRAME_WIDTH,
                    height: FRAME_HEIGHT,
                },
                pixel_format: PixelFormat::Rgba8,
            },
        };
        let audio = SilentAudioEncoder {
            config: EncoderConfig {
                codec: Codec::Aac,
                timescale: 48_000,
                decoder_config: codec_box(b"esds", &[0, 0, 0, 0]),
            },
            format: AudioEncoderFormat {
                sample_rate: 48_000,
                channels: 1,
            },
        };
        let mut output = MediaOutput::new(MemorySink::new(), video, audio, timeline, options)
            .await
            .unwrap();
        for index in 0..frames {
            let frame = indexed_frame(index);
            output
                .put_video(
                    FrameIndex(index),
                    FrameSource::Cpu(CpuFrameSource {
                        frame: &frame,
                        orientation: Orientation::TopLeft,
                    }),
                )
                .await
                .unwrap();
            let range = timeline
                .audio_interval_for_frame(FrameIndex(index))
                .unwrap();
            let pcm = vec![0.0; usize::try_from(range.len()).unwrap()];
            output
                .put_audio(
                    FrameIndex(index),
                    AudioBuffer::new(range, 48_000, 1, pcm, &Limits::default()).unwrap(),
                )
                .await
                .unwrap();
        }
        if cover_art.is_some() {
            output.set_cover_art(cover_art).unwrap();
        }
        output.finish().await.unwrap().into_inner()
    })
}

fn frame_cover(index: u64) -> Option<CoverArt> {
    Some(CoverArt::from_video_frame(&indexed_frame(index)).unwrap())
}

fn with_cover_source(cover_source: CoverSource) -> OutputOptions {
    OutputOptions {
        cover_source,
        ..OutputOptions::default()
    }
}

#[test]
fn media_output_covers_the_file_with_frame_four_by_default() {
    assert_eq!(OutputOptions::default().cover_source, CoverSource::Frame(4));
    let cover = demux(record(OutputOptions::default(), 10, None)).cover_art;
    assert_eq!(cover, frame_cover(4));
    assert_eq!(cover.unwrap().format, CoverArtFormat::Png);
}

#[test]
fn media_output_covers_the_file_with_a_chosen_frame() {
    let options = with_cover_source(CoverSource::Frame(7));
    assert_eq!(demux(record(options, 10, None)).cover_art, frame_cover(7));
    let options = with_cover_source(CoverSource::Frame(0));
    assert_eq!(demux(record(options, 10, None)).cover_art, frame_cover(0));
}

#[test]
fn a_stream_shorter_than_the_cover_frame_uses_its_last_frame() {
    assert_eq!(
        demux(record(OutputOptions::default(), 3, None)).cover_art,
        frame_cover(2)
    );
    assert_eq!(
        demux(record(OutputOptions::default(), 0, None)).cover_art,
        None
    );
}

#[test]
fn cover_source_none_writes_no_cover() {
    let bytes = record(with_cover_source(CoverSource::None), 10, None);
    assert!(!bytes.windows(4).any(|window| window == b"covr"));
    assert_eq!(demux(bytes).cover_art, None);
}

#[test]
fn explicit_cover_art_overrides_the_generated_cover() {
    let demuxer = demux(record(OutputOptions::default(), 10, Some(jpeg())));
    assert_eq!(demuxer.cover_art, Some(jpeg()));
    let demuxer = demux(record(
        with_cover_source(CoverSource::None),
        10,
        Some(png()),
    ));
    assert_eq!(demuxer.cover_art, Some(png()));
}

#[test]
fn ffmpeg_decodes_the_generated_cover_to_the_frame_pixels() {
    if Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_err()
    {
        eprintln!("skipping the independent decode because ffmpeg is unavailable");
        return;
    }
    let cover = demux(record(OutputOptions::default(), 10, None))
        .cover_art
        .unwrap();
    let path = std::env::temp_dir().join(format!(
        "zvidlib-generated-cover-{}.png",
        std::process::id()
    ));
    std::fs::write(&path, &cover.data).unwrap();
    let decode = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(
        decode.status.success(),
        "ffmpeg failed: {}",
        String::from_utf8_lossy(&decode.stderr)
    );
    let expected: Vec<u8> = indexed_frame(4).planes[0]
        .data
        .chunks(4)
        .flat_map(|pixel| pixel[..3].to_vec())
        .collect();
    assert_eq!(decode.stdout, expected);
}
