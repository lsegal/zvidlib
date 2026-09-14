//! Encode a short synthetic AV1 video into a playable MP4.
//!
//! ```console
//! cargo run --example native_encode --features native -- output.mp4
//! ```

use std::future::Future;
use std::task::{Context, Poll, Waker};

use zvidlib::io::MemorySink;
use zvidlib::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use zvidlib::{
    Codec, CodecProfile, ColorRange, CpuFrameSource, FrameIndex, FrameSource, HardwarePreference,
    Limits, Orientation, PixelFormat, Plane, VideoDimensions, VideoEncoderConfig,
    VideoEncoderFactory, VideoFrame, native_av1_video_encoder_factory,
};

fn block_on<T>(future: impl Future<Output = T>) -> T {
    let mut context = Context::from_waker(Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::yield_now();
    }
}

fn frame(dimensions: VideoDimensions, index: u64, limits: &Limits) -> VideoFrame {
    let pixels = (0..dimensions.height)
        .flat_map(|y| (0..dimensions.width).map(move |x| ((x + y + index as u32 * 5) % 256) as u8))
        .collect();
    VideoFrame::new(
        dimensions,
        PixelFormat::Gray8,
        ColorRange::Full,
        vec![Plane {
            data: pixels,
            stride: dimensions.width as usize,
        }],
        limits,
    )
    .expect("the generated frame has a valid Gray8 layout")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "zvidlib-native-encode.mp4".to_owned());
    let limits = Limits::default();
    // Keep the default quick enough to run interactively through the portable
    // software encoder; callers can adapt the frame generator for production sizes.
    let dimensions = VideoDimensions::new(96, 54, &limits)?;
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
    let mut encoder = native_av1_video_encoder_factory().create(&configuration, &limits)?;
    let mut muxer = block_on(Mp4Muxer::new(
        MemorySink::new(),
        vec![Mp4TrackConfig {
            encoder: encoder.config().clone(),
            format: Mp4TrackFormat::Video(dimensions),
        }],
        60,
    ))?;

    for index in 0..30_u64 {
        let frame = frame(dimensions, index, &limits);
        for sample in block_on(encoder.encode(
            FrameIndex(index),
            FrameSource::Cpu(CpuFrameSource {
                frame: &frame,
                orientation: Orientation::TopLeft,
            }),
        ))? {
            block_on(muxer.write_sample(0, sample))?;
        }
    }
    for sample in block_on(encoder.finish())? {
        block_on(muxer.write_sample(0, sample))?;
    }
    let bytes = block_on(muxer.finish())?.into_inner();
    std::fs::write(&output_path, bytes)?;
    println!("Wrote {output_path}. Inspect it with: ffprobe {output_path}");
    Ok(())
}
