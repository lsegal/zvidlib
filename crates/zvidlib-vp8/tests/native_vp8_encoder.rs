//! Native VP8 encoding into WebM (issue #530).
//!
//! The frames of the bundled VP8 `testsrc2` WebM are decoded, re-encoded with
//! `native_vp8_video_encoder_factory`, and written with `WebmMuxer`. The
//! written file must index one cue per key frame, return every frame exactly
//! through `ExactFrameReader` whatever order it is asked for in, and look like
//! the frames it was made from. `src/vp8/tests.rs` holds the same output to
//! libvpx's decode where ffmpeg is installed.
#![cfg(not(target_arch = "wasm32"))]

use zvidlib_container::mp4::{Mp4TrackConfig, Mp4TrackFormat};
use zvidlib_container::{
    ExpectedVideoFrame, FrameDigest, VideoDecoderConformanceVector, WebmDemuxer, WebmMuxer,
    verify_video_decoder_conformance,
};
use zvidlib_core::io::{MemorySink, MemorySource};
use zvidlib_core::{
    CancellationToken, Codec, CodecProfile, ColorRange, CpuFrameSource, EncodedSample, FrameIndex,
    FrameSource, HardwarePreference, Limits, Orientation, PixelFormat, Plane, VideoDecoderConfig,
    VideoDecoderFactory, VideoDimensions, VideoEncoderConfig, VideoEncoderFactory, VideoFrame,
};
use zvidlib_vp8::{native_vp8_video_decoder_factory, native_vp8_video_encoder_factory};

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    use std::task::{Context, Poll, Waker};
    let mut context = Context::from_waker(Waker::noop());
    let mut future = Box::pin(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
    }
}

fn decoder_configuration(dimensions: VideoDimensions) -> VideoDecoderConfig {
    VideoDecoderConfig {
        codec: Codec::Vp8,
        profile: CodecProfile::Vp8,
        coded_dimensions: dimensions,
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Avoid,
        configuration: Vec::new(),
    }
}

fn encoder_configuration(
    dimensions: VideoDimensions,
    configuration: Vec<u8>,
) -> VideoEncoderConfig {
    VideoEncoderConfig {
        codec: Codec::Vp8,
        profile: CodecProfile::Vp8,
        coded_dimensions: dimensions,
        input_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Prefer,
        timescale: 25,
        frame_duration: 1,
        configuration,
    }
}

/// Every frame of the bundled 98x66 `testsrc2` clip, as RGBA.
fn source_frames() -> (VideoDimensions, Vec<VideoFrame>) {
    let source = MemorySource::new(include_bytes!("fixtures/vp8/vp8_testsrc2_98x66.webm").to_vec());
    let limits = Limits::default();
    let demuxer = block_on(WebmDemuxer::open(&source, Default::default())).unwrap();
    let track = &demuxer.tracks[0];
    let dimensions = track.dimensions.unwrap();
    let samples = block_on(track.to_encoded_video_samples(&source, &limits)).unwrap();
    let mut decoder = native_vp8_video_decoder_factory()
        .create(&decoder_configuration(dimensions), &limits)
        .unwrap();
    let cancellation = CancellationToken::new();
    let frames = samples
        .iter()
        .flat_map(|sample| decoder.submit(sample, &cancellation).unwrap())
        .map(|output| output.frame)
        .collect();
    (dimensions, frames)
}

fn encode(
    configuration: &VideoEncoderConfig,
    frames: &[VideoFrame],
    orientation: Orientation,
) -> Vec<EncodedSample> {
    let mut encoder = native_vp8_video_encoder_factory()
        .create(configuration, &Limits::default())
        .unwrap();
    let mut samples = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        samples.extend(
            block_on(encoder.encode(
                FrameIndex(index as u64),
                FrameSource::Cpu(CpuFrameSource { frame, orientation }),
            ))
            .unwrap(),
        );
    }
    samples.extend(block_on(encoder.finish()).unwrap());
    samples
}

fn mux(dimensions: VideoDimensions, timescale: u32, samples: Vec<EncodedSample>) -> Vec<u8> {
    let config = Mp4TrackConfig {
        encoder: zvidlib_core::EncoderConfig {
            codec: Codec::Vp8,
            timescale,
            decoder_config: Vec::new(),
        },
        format: Mp4TrackFormat::Video(dimensions),
    };
    block_on(async {
        let mut muxer = WebmMuxer::new(MemorySink::new(), vec![config], 1_000_000)
            .await
            .unwrap();
        for sample in samples {
            muxer.write_sample(0, sample).await.unwrap();
        }
        muxer.finish().await.unwrap().into_inner()
    })
}

fn psnr(a: &VideoFrame, b: &VideoFrame) -> f64 {
    let (a, b) = (&a.planes[0].data, &b.planes[0].data);
    let error = a
        .iter()
        .zip(b)
        .enumerate()
        .filter(|(index, _)| index % 4 != 3)
        .map(|(_, (&x, &y))| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>()
        / (a.len() / 4 * 3) as f64;
    10.0 * (255.0 * 255.0 / error.max(1e-9)).log10()
}

#[test]
fn an_encoded_webm_seeks_through_its_cues_and_decodes_exactly_in_any_order() {
    let (dimensions, frames) = source_frames();
    assert!(frames.len() >= 30);
    let limits = Limits::default();
    for (configuration, minimum_psnr) in [
        // The default quantizer.
        (Vec::new(), 28.0),
        // 200 kb/s with a key frame every ten frames.
        (
            [200_000u32.to_be_bytes(), 10u32.to_be_bytes()].concat(),
            28.0,
        ),
    ] {
        let encoder_configuration = encoder_configuration(dimensions, configuration.clone());
        let samples = encode(&encoder_configuration, &frames, Orientation::TopLeft);
        assert_eq!(samples.len(), frames.len());
        let key_frames = samples.iter().filter(|sample| sample.is_sync).count();
        assert!(key_frames >= 2, "{configuration:?}");

        let source = MemorySource::new(mux(dimensions, 25, samples));
        let demuxer = block_on(WebmDemuxer::open(&source, Default::default())).unwrap();
        assert_eq!(demuxer.doc_type, "webm");
        assert_eq!(demuxer.cues.len(), key_frames, "one cue per key frame");
        let track = &demuxer.tracks[0];
        assert_eq!(track.codec, Codec::Vp8);
        assert_eq!(track.dimensions, Some(dimensions));
        let encoded = block_on(track.to_encoded_video_samples(&source, &limits)).unwrap();

        let factory = native_vp8_video_decoder_factory();
        let configuration = decoder_configuration(dimensions);
        let mut decoder = factory.create(&configuration, &limits).unwrap();
        let cancellation = CancellationToken::new();
        let mut expected = Vec::new();
        let mut total_psnr = 0.0;
        for sample in &encoded {
            let outputs = decoder.submit(sample, &cancellation).unwrap();
            assert_eq!(outputs.len(), 1);
            let output = &outputs[0];
            total_psnr += psnr(&output.frame, &frames[expected.len()]);
            expected.push(ExpectedVideoFrame {
                presentation_index: output.presentation_index,
                digest: FrameDigest::from_frame(&output.frame).unwrap(),
            });
        }
        let average = total_psnr / frames.len() as f64;
        assert!(
            average > minimum_psnr,
            "{encoder_configuration:?}: {average:.1} dB"
        );

        let vector = VideoDecoderConformanceVector {
            name: "native VP8 encoder output".into(),
            configuration,
            expected_frames: expected,
            samples: encoded,
        };
        let report = verify_video_decoder_conformance(&factory, &vector, limits).unwrap();
        assert_eq!(report.access_patterns_verified, 3);
    }
}

#[test]
fn bgra_and_bottom_up_frames_encode_exactly_as_rgba() {
    let (dimensions, frames) = source_frames();
    let frames = &frames[..6];
    let rgba = encode(
        &encoder_configuration(dimensions, Vec::new()),
        frames,
        Orientation::TopLeft,
    );
    let height = dimensions.height as usize;
    let flipped: Vec<VideoFrame> = frames
        .iter()
        .map(|frame| {
            let stride = frame.planes[0].stride;
            let mut data = Vec::with_capacity(frame.planes[0].data.len());
            for row in (0..height).rev() {
                for pixel in frame.planes[0].data[row * stride..(row + 1) * stride].chunks(4) {
                    data.extend_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
                }
            }
            VideoFrame::new(
                dimensions,
                PixelFormat::Bgra8,
                ColorRange::Limited,
                vec![Plane { data, stride }],
                &Limits::default(),
            )
            .unwrap()
        })
        .collect();
    let configuration = VideoEncoderConfig {
        input_format: PixelFormat::Bgra8,
        ..encoder_configuration(dimensions, Vec::new())
    };
    let bgra = encode(&configuration, &flipped, Orientation::BottomLeft);
    assert_eq!(bgra, rgba);
}

#[test]
fn mp4_refuses_a_vp8_track() {
    let limits = Limits::default();
    let config = Mp4TrackConfig {
        encoder: zvidlib_core::EncoderConfig {
            codec: Codec::Vp8,
            timescale: 25,
            decoder_config: Vec::new(),
        },
        format: Mp4TrackFormat::Video(VideoDimensions::new(16, 16, &limits).unwrap()),
    };
    let error = block_on(zvidlib_container::mp4::Mp4Muxer::new(
        MemorySink::new(),
        vec![config],
        16,
    ))
    .err()
    .unwrap();
    assert_eq!(error.kind(), zvidlib_core::ErrorKind::InvalidInput);
}
