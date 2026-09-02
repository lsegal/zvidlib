//! Temporary probe for issue #354. Not for merge.
use std::time::Instant;
use zvidlib::*;
use zvidlib::io::MemorySource;

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, Waker};
    let mut boxed = Box::pin(future);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    match boxed.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("pending"),
    }
}

fn main() -> Result<()> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/media/BigBuckBunny.mp4");
    let bytes = std::fs::read(&path).unwrap();
    let source = MemorySource::new(bytes);
    let demuxer = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))?;
    let video = demuxer.tracks.iter().find(|t| t.kind == TrackKind::Video).unwrap();
    let dimensions = video.dimensions.unwrap();
    let limits = Limits::default();
    let samples = block_on(video.to_encoded_video_samples(&source, &limits))?;
    let count = samples.len() as u64;
    let keyframes: Vec<u64> = samples.iter().filter(|s| s.random_access).map(|s| s.presentation_index.0).collect();
    println!("frames={count} keyframes={:?}", &keyframes[..keyframes.len().min(10)]);
    let factory = native_hevc_video_decoder_factory();
    let configuration = VideoDecoderConfig {
        codec: video.codec,
        profile: CodecProfile::HevcMain,
        coded_dimensions: dimensions,
        output_format: PixelFormat::Rgba8,
        color_range: ColorRange::Limited,
        hardware: HardwarePreference::Prefer,
        configuration: video.decoder_config.clone(),
    };
    println!("capability: {:?}", factory.capability(&configuration));
    let token = CancellationToken::new();
    let mut reader = ExactFrameReader::new(&factory, configuration.clone(), samples.clone(), limits)?;

    // Strided walk, as the drag does: only the published frames are converted.
    for stride in [1u64, 4, 8, 16] {
        let mut fresh = ExactFrameReader::new(&factory, configuration.clone(), samples.clone(), limits)?;
        let target = 728u64;
        let start = Instant::now();
        let mut cursor = 0u64;
        let mut worst_gap = 0f64;
        loop {
            let t = Instant::now();
            fresh.get(FrameIndex(cursor), &token)?;
            let gap = t.elapsed().as_secs_f64() * 1000.0;
            if gap > worst_gap { worst_gap = gap; }
            if cursor == target { break; }
            cursor = target.min(cursor + stride);
        }
        println!("strided walk to {target} stride {stride}: {:.0} ms, worst published-frame gap {:.1} ms, stats {:?}",
            start.elapsed().as_secs_f64()*1000.0, worst_gap, fresh.statistics());
    }

    // Sequential forward walk, reporting per-frame cost at intervals.
    let mut worst = 0f64;
    let mut bucket = Instant::now();
    for i in 0..count {
        let t = Instant::now();
        reader.get(FrameIndex(i), &token)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms > worst { worst = ms; }
        if i % 128 == 127 {
            println!("walk frames {}..={} took {:.0} ms total, worst single {:.1} ms, stats {:?}",
                i - 127, i, bucket.elapsed().as_secs_f64() * 1000.0, worst, reader.statistics());
            worst = 0.0;
            bucket = Instant::now();
        }
    }

    let report = zvidlib::hevc_hardware_readback::report();
    println!("readback: frames={} surface_copy={:?} color_convert={:?}", report.frames, report.surface_copy, report.color_convert);

    // Backwards step after arriving at the end (a drag that jitters back).
    for back in [1u64, 8, 40] {
        let target = count - 1 - back;
        let t = Instant::now();
        reader.get(FrameIndex(target), &token)?;
        println!("back {back} frames (to {target}) took {:.0} ms, stats {:?}", t.elapsed().as_secs_f64()*1000.0, reader.statistics());
    }

    // Cold jump straight to a position, from a fresh reader.
    for fraction in [0.1f64, 0.5, 0.95] {
        let mut fresh = ExactFrameReader::new(&factory, configuration.clone(), samples.clone(), limits)?;
        let target = ((count - 1) as f64 * fraction) as u64;
        let t = Instant::now();
        fresh.get(FrameIndex(target), &token)?;
        println!("cold jump to {target} took {:.0} ms", t.elapsed().as_secs_f64()*1000.0);
    }
    Ok(())
}
