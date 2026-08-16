use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};
use zvidlib::io::{ByteSink, IoFuture, MemorySink, MemorySource};
use zvidlib::{
    AudioBuffer, AudioDrain, AudioEncoder, AudioEncoderFormat, AudioGapless, Codec, ColorRange,
    EncodedSample, EncoderConfig, EncoderFuture, Error, ErrorKind, FrameIndex, FrameRate, Limits,
    MediaOutput, Mp4Demuxer, Mp4DemuxerOptions, OutputOptions, PixelFormat, Plane,
    SampleDependency, SampleRange, Timeline, VideoDimensions, VideoEncoder, VideoEncoderFormat,
    VideoFrame,
};
use zvidlib::{
    ContextIdentity, CpuFrameSource, ExecutionOwner, FrameSource, GraphicsApi, GraphicsResource,
    Orientation, ResourceKind, ResourceOwnership,
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

struct DelayedVideoEncoder {
    pending: Vec<EncodedSample>,
    config: EncoderConfig,
    format: VideoEncoderFormat,
}

impl DelayedVideoEncoder {
    fn new(dimensions: VideoDimensions) -> Self {
        Self {
            pending: Vec::new(),
            config: EncoderConfig {
                codec: Codec::Av1,
                timescale: 30_000,
                decoder_config: codec_box(b"av1C", &[0x81, 0, 0, 0]),
            },
            format: VideoEncoderFormat {
                dimensions,
                pixel_format: PixelFormat::Gray8,
            },
        }
    }
}

impl VideoEncoder for DelayedVideoEncoder {
    fn config(&self) -> &EncoderConfig {
        &self.config
    }

    fn format(&self) -> VideoEncoderFormat {
        self.format
    }

    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        frame: FrameSource<'a>,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move {
            let (dimensions, pixel_format) = match frame {
                FrameSource::Cpu(source) => (source.frame.dimensions, source.frame.pixel_format),
                FrameSource::Graphics(resource) => (resource.dimensions(), resource.pixel_format()),
            };
            if dimensions != self.format.dimensions || pixel_format != self.format.pixel_format {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "video frame format changed",
                ));
            }
            let (dts, pts, is_sync, dependency) = match index.0 {
                0 => (0, 1_000, true, SampleDependency::INDEPENDENT),
                1 => (1_000, 0, false, SampleDependency::DEPENDENT),
                _ => {
                    return Err(Error::new(
                        ErrorKind::InvalidInput,
                        "fixture only accepts two frames",
                    ));
                }
            };
            self.pending.push(EncodedSample {
                data: vec![b'V', u8::try_from(index.0).unwrap()],
                dts,
                pts,
                duration: 1_000,
                is_sync,
                dependency,
            });
            if self.pending.len() > 1 {
                Ok(vec![self.pending.remove(0)])
            } else {
                Ok(Vec::new())
            }
        })
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move { Ok(std::mem::take(&mut self.pending)) })
    }
}

struct PcmFixtureEncoder {
    config: EncoderConfig,
    format: AudioEncoderFormat,
    gapless: AudioGapless,
    packets_per_encode: u32,
}

impl PcmFixtureEncoder {
    fn new() -> Self {
        Self {
            config: EncoderConfig {
                codec: Codec::Aac,
                timescale: 48_000,
                decoder_config: codec_box(b"esds", &[0, 0, 0, 0]),
            },
            format: AudioEncoderFormat {
                sample_rate: 48_000,
                channels: 2,
            },
            gapless: AudioGapless {
                priming: 1_024,
                padding: 128,
            },
            packets_per_encode: 1,
        }
    }

    fn without_gapless() -> Self {
        Self {
            gapless: AudioGapless::default(),
            ..Self::new()
        }
    }

    fn burst(packets_per_encode: u32) -> Self {
        Self {
            packets_per_encode,
            ..Self::without_gapless()
        }
    }
}

impl AudioEncoder for PcmFixtureEncoder {
    fn config(&self) -> &EncoderConfig {
        &self.config
    }

    fn format(&self) -> AudioEncoderFormat {
        self.format
    }

    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        buffer: AudioBuffer,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move {
            let duration = u32::try_from(buffer.range.len()).unwrap() / self.packets_per_encode;
            Ok((0..self.packets_per_encode)
                .map(|packet| {
                    let dts = buffer.range.start + u64::from(packet * duration);
                    EncodedSample {
                        data: vec![b'A', u8::try_from(index.0).unwrap(), packet as u8],
                        dts: i64::try_from(dts).unwrap(),
                        pts: i64::try_from(dts).unwrap(),
                        duration,
                        is_sync: true,
                        dependency: SampleDependency::INDEPENDENT,
                    }
                })
                .collect())
        })
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, AudioDrain> {
        Box::pin(async {
            Ok(AudioDrain {
                samples: Vec::new(),
                gapless: self.gapless,
            })
        })
    }
}

fn frame(dimensions: VideoDimensions, value: u8) -> VideoFrame {
    VideoFrame::new(
        dimensions,
        PixelFormat::Gray8,
        ColorRange::Full,
        vec![Plane {
            data: vec![value; 4],
            stride: 2,
        }],
        &Limits::default(),
    )
    .unwrap()
}

fn cpu_source(frame: &VideoFrame) -> FrameSource<'_> {
    FrameSource::Cpu(CpuFrameSource {
        frame,
        orientation: Orientation::TopLeft,
    })
}

fn webgl_source(dimensions: VideoDimensions) -> FrameSource<'static> {
    FrameSource::Graphics(GraphicsResource::new(
        GraphicsApi::WebGl,
        ContextIdentity(7),
        ExecutionOwner(9),
        ResourceKind::Framebuffer,
        42,
        dimensions,
        PixelFormat::Gray8,
        ColorRange::Full,
        Orientation::BottomLeft,
        ResourceOwnership::Caller,
    ))
}

fn audio(range: SampleRange) -> AudioBuffer {
    AudioBuffer::new(
        range,
        48_000,
        2,
        vec![0.0; usize::try_from(range.len() * 2).unwrap()],
        &Limits::default(),
    )
    .unwrap()
}

fn fixture() -> (
    VideoDimensions,
    Timeline,
    DelayedVideoEncoder,
    PcmFixtureEncoder,
) {
    let dimensions = VideoDimensions::new(2, 2, &Limits::default()).unwrap();
    let timeline = Timeline::new(FrameRate::new(30, 1).unwrap(), 48_000).unwrap();
    (
        dimensions,
        timeline,
        DelayedVideoEncoder::new(dimensions),
        PcmFixtureEncoder::new(),
    )
}

fn box_payload<'a>(bytes: &'a [u8], kind: &[u8; 4], occurrence: usize) -> &'a [u8] {
    let mut seen = 0;
    for index in 4..=bytes.len().saturating_sub(4) {
        if &bytes[index..index + 4] != kind {
            continue;
        }
        let size32 = u32::from_be_bytes(bytes[index - 4..index].try_into().unwrap());
        let (header, size) = if size32 == 1 && index + 12 <= bytes.len() {
            (
                16,
                usize::try_from(u64::from_be_bytes(
                    bytes[index + 4..index + 12].try_into().unwrap(),
                ))
                .unwrap(),
            )
        } else {
            (8, size32 as usize)
        };
        if size < header || index - 4 + size > bytes.len() {
            continue;
        }
        if seen == occurrence {
            return &bytes[index - 4 + header..index - 4 + size];
        }
        seen += 1;
    }
    panic!("box {:?} occurrence {occurrence} not found", kind);
}

#[test]
fn synchronized_output_round_trips_exact_mp4_indexes() {
    block_on(async {
        let (dimensions, timeline, video, audio_encoder) = fixture();
        let mut output = MediaOutput::new(
            MemorySink::new(),
            video,
            audio_encoder,
            timeline,
            OutputOptions::default(),
        )
        .await
        .unwrap();

        let skipped = output
            .put_video(FrameIndex(1), cpu_source(&frame(dimensions, 1)))
            .await;
        assert_eq!(skipped.unwrap_err().kind(), ErrorKind::InvalidInput);

        output
            .put_video(FrameIndex(0), cpu_source(&frame(dimensions, 1)))
            .await
            .unwrap();
        let repeated = output
            .put_video(FrameIndex(0), cpu_source(&frame(dimensions, 1)))
            .await;
        assert_eq!(repeated.unwrap_err().kind(), ErrorKind::InvalidInput);

        let first_audio = timeline.audio_interval_for_frame(FrameIndex(0)).unwrap();
        output
            .put_audio(FrameIndex(0), audio(first_audio))
            .await
            .unwrap();
        output
            .put_video(FrameIndex(1), webgl_source(dimensions))
            .await
            .unwrap();
        let second_audio = timeline.audio_interval_for_frame(FrameIndex(1)).unwrap();
        output
            .put_audio(FrameIndex(1), audio(second_audio))
            .await
            .unwrap();

        let bytes = output.finish().await.unwrap().into_inner();
        let demuxer = Mp4Demuxer::open(
            &MemorySource::new(bytes.clone()),
            Mp4DemuxerOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(demuxer.tracks.len(), 2);
        assert_eq!(demuxer.tracks[0].samples.len(), 2);
        assert_eq!(demuxer.tracks[0].presentation_order, vec![1, 0]);
        assert_eq!(demuxer.tracks[0].samples[0].pts, 1_000);
        assert_eq!(demuxer.tracks[0].samples[1].pts, 0);
        assert_eq!(demuxer.tracks[1].edits[0].media_time, 1_024);
        assert_eq!(&bytes[4..8], b"ftyp");
        let ftyp_size = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
        assert_eq!(&bytes[ftyp_size + 4..ftyp_size + 8], b"mdat");
        let mdat_size = usize::try_from(u64::from_be_bytes(
            bytes[ftyp_size + 8..ftyp_size + 16].try_into().unwrap(),
        ))
        .unwrap();
        assert_eq!(
            &bytes[ftyp_size + mdat_size + 4..ftyp_size + mdat_size + 8],
            b"moov"
        );
        let mdat = box_payload(&bytes, b"mdat", 0);
        assert!(mdat.windows(2).any(|value| value == b"V\0"));
        assert!(mdat.windows(2).any(|value| value == b"V\x01"));
        assert_eq!(&box_payload(&bytes, b"moov", 0)[..4], &[0, 0, 0, 108]);

        for (occurrence, expected) in [(0, [b"V\0", b"V\x01"]), (1, [b"A\0", b"A\x01"])] {
            let chunks = box_payload(&bytes, b"co64", occurrence);
            assert_eq!(u32::from_be_bytes(chunks[4..8].try_into().unwrap()), 2);
            for (sample, expected) in expected.into_iter().enumerate() {
                let start = 8 + sample * 8;
                let offset = usize::try_from(u64::from_be_bytes(
                    chunks[start..start + 8].try_into().unwrap(),
                ))
                .unwrap();
                assert_eq!(&bytes[offset..offset + 2], expected);
            }
        }

        let ctts = box_payload(&bytes, b"ctts", 0);
        assert_eq!(ctts[0], 1, "signed composition offsets use ctts v1");
        assert_eq!(u32::from_be_bytes(ctts[4..8].try_into().unwrap()), 2);
        assert_eq!(i32::from_be_bytes(ctts[12..16].try_into().unwrap()), 1_000);
        assert_eq!(i32::from_be_bytes(ctts[20..24].try_into().unwrap()), -1_000);

        let stss = box_payload(&bytes, b"stss", 0);
        assert_eq!(u32::from_be_bytes(stss[4..8].try_into().unwrap()), 1);
        assert_eq!(u32::from_be_bytes(stss[8..12].try_into().unwrap()), 1);
        assert_eq!(&box_payload(&bytes, b"sdtp", 0)[4..], &[0x20, 0x10]);

        let audio_stts = box_payload(&bytes, b"stts", 1);
        assert_eq!(u32::from_be_bytes(audio_stts[8..12].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_be_bytes(audio_stts[12..16].try_into().unwrap()),
            1_600
        );

        let edit = box_payload(&bytes, b"elst", 0);
        assert_eq!(u32::from_be_bytes(edit[4..8].try_into().unwrap()), 1);
        assert_eq!(u32::from_be_bytes(edit[8..12].try_into().unwrap()), 42);
        assert_eq!(i32::from_be_bytes(edit[12..16].try_into().unwrap()), 1_024);
    });
}

#[test]
fn incompatible_audio_and_unsynchronized_finish_are_rejected() {
    block_on(async {
        let (dimensions, timeline, video, audio_encoder) = fixture();
        let mut output = MediaOutput::new(
            MemorySink::new(),
            video,
            audio_encoder,
            timeline,
            OutputOptions::default(),
        )
        .await
        .unwrap();
        output
            .put_video(FrameIndex(0), cpu_source(&frame(dimensions, 1)))
            .await
            .unwrap();
        let range = timeline.audio_interval_for_frame(FrameIndex(0)).unwrap();
        let incompatible = AudioBuffer::new(
            range,
            48_000,
            1,
            vec![0.0; usize::try_from(range.len()).unwrap()],
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(
            output
                .put_audio(FrameIndex(0), incompatible)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            output.finish().await.unwrap_err().kind(),
            ErrorKind::InvalidState
        );
    });
}

#[derive(Debug)]
struct ObservedSink {
    inner: MemorySink,
    writes: Rc<Cell<usize>>,
    fail_write: Option<usize>,
    flushed: Rc<Cell<bool>>,
    fail_flush: bool,
}

impl ByteSink for ObservedSink {
    fn position(&self) -> u64 {
        self.inner.position()
    }

    fn write<'a>(&'a mut self, bytes: &'a [u8]) -> IoFuture<'a, ()> {
        let next = self.writes.get() + 1;
        self.writes.set(next);
        if self.fail_write == Some(next) {
            return Box::pin(async { Err(Error::new(ErrorKind::Io, "sink is full")) });
        }
        self.inner.write(bytes)
    }

    fn seek<'a>(&'a mut self, position: u64) -> IoFuture<'a, ()> {
        self.inner.seek(position)
    }

    fn flush<'a>(&'a mut self) -> IoFuture<'a, ()> {
        self.flushed.set(true);
        let fail = self.fail_flush;
        Box::pin(async move {
            if fail {
                Err(Error::new(ErrorKind::Io, "flush failed"))
            } else {
                Ok(())
            }
        })
    }
}

#[test]
fn sink_backpressure_and_flush_errors_reach_the_caller() {
    block_on(async {
        let writes = Rc::new(Cell::new(0));
        let flushed = Rc::new(Cell::new(false));
        let sink = ObservedSink {
            inner: MemorySink::new(),
            writes: writes.clone(),
            fail_write: Some(3),
            flushed,
            fail_flush: false,
        };
        let (dimensions, timeline, video, audio_encoder) = fixture();
        let mut output = MediaOutput::new(
            sink,
            video,
            audio_encoder,
            timeline,
            OutputOptions::default(),
        )
        .await
        .unwrap();
        output
            .put_video(FrameIndex(0), cpu_source(&frame(dimensions, 1)))
            .await
            .unwrap();
        let error = output
            .put_audio(
                FrameIndex(0),
                audio(timeline.audio_interval_for_frame(FrameIndex(0)).unwrap()),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Io);
        assert_eq!(
            writes.get(),
            3,
            "no packet queue hides the failed sink write"
        );

        let flush_writes = Rc::new(Cell::new(0));
        let flush_observed = Rc::new(Cell::new(false));
        let sink = ObservedSink {
            inner: MemorySink::new(),
            writes: flush_writes,
            fail_write: None,
            flushed: flush_observed.clone(),
            fail_flush: true,
        };
        let (dimensions, timeline, _, _) = fixture();
        let video = DelayedVideoEncoder::new(dimensions);
        let audio_encoder = PcmFixtureEncoder::without_gapless();
        let output = MediaOutput::new(
            sink,
            video,
            audio_encoder,
            timeline,
            OutputOptions::default(),
        )
        .await
        .unwrap();
        let error = output.finish().await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Io);
        assert!(flush_observed.get());
    });
}

#[test]
fn encoder_packet_batches_are_bounded_before_sink_writes() {
    block_on(async {
        let writes = Rc::new(Cell::new(0));
        let sink = ObservedSink {
            inner: MemorySink::new(),
            writes: writes.clone(),
            fail_write: None,
            flushed: Rc::new(Cell::new(false)),
            fail_flush: false,
        };
        let (dimensions, timeline, _, _) = fixture();
        let mut output = MediaOutput::new(
            sink,
            DelayedVideoEncoder::new(dimensions),
            PcmFixtureEncoder::burst(2),
            timeline,
            OutputOptions {
                max_samples_per_encode: 1,
                ..OutputOptions::default()
            },
        )
        .await
        .unwrap();
        let error = output
            .put_audio(
                FrameIndex(0),
                audio(timeline.audio_interval_for_frame(FrameIndex(0)).unwrap()),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);
        assert_eq!(writes.get(), 2, "only the MP4 headers reached the sink");
    });
}
