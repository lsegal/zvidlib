//! The scrub preview index, on a thread.
//!
//! Issue #374 is the part of seeking that no decoder change can reach: the bundled sample codes
//! its 768 frames as a single group of pictures, so the frame four fifths of the way along the
//! bar is 613 samples of reference decoding away from the only place a decode can start. What
//! answers a drag there in under 50 ms is a picture that was decoded already.
//!
//! That index used to live here. Issue #416 moved it into the library as
//! [`zvidlib::SeekPreviews`], because it is what makes a seek meet `ARCHITECTURE.md` section
//! 3.2's requirement and every caller that scrubs a timeline needs it - the browser example as
//! much as this one. What is left here is the thread: the library's pass is driven by its caller
//! one preview at a time so the browser build can drive it from an idle callback, and a native
//! window drives it from a worker.

use std::thread::{self, JoinHandle};

use zvidlib::{
    CancellationToken, EncodedVideoSample, Error, ErrorKind, Limits, Result, SeekPreviewOptions,
    SeekPreviewPass, SeekPreviews, VideoDecoderConfig, VideoDecoderFactory, VideoFrame,
};

/// A background pass over the track that keeps a picture every stride frames.
pub struct PreviewIndex {
    previews: SeekPreviews,
    cancellation: CancellationToken,
    worker: Option<JoinHandle<()>>,
}

impl PreviewIndex {
    /// Starts the pass over `samples`, on a decoder of its own.
    ///
    /// The decoder is separate on purpose: sharing [`crate::scrub`]'s reader would make every
    /// preview the index decodes a seek that reader has to undo, and the two want opposite
    /// things from it - the index walks forwards once and never goes back, a drag jumps.
    pub fn new(
        factory: &dyn VideoDecoderFactory,
        configuration: VideoDecoderConfig,
        samples: Vec<EncodedVideoSample>,
        limits: Limits,
        frames_per_second: u64,
    ) -> Result<Self> {
        let mut pass = SeekPreviewPass::new(
            factory,
            configuration,
            samples,
            limits,
            frames_per_second,
            SeekPreviewOptions::default(),
        )?;
        let previews = pass.previews();
        let cancellation = CancellationToken::new();
        let worker_cancellation = cancellation.clone();
        let worker = thread::Builder::new()
            .name("zvidlib-preview-index".to_string())
            .spawn(move || pass.run(&worker_cancellation))
            .map_err(|error| {
                Error::new(
                    ErrorKind::InvalidInput,
                    format!("could not start the preview index thread: {error}"),
                )
            })?;
        Ok(Self {
            previews,
            cancellation,
            worker: Some(worker),
        })
    }

    /// How many of the index's positions the background pass has filled, out of how many there
    /// are, so a measurement can wait for the whole track rather than for the first picture.
    #[cfg(test)]
    fn coverage(&self) -> (usize, usize) {
        self.previews.coverage()
    }

    /// The kept picture closest to `frame`, or `None` while the pass has decoded nothing.
    ///
    /// This is a lock, a search and a clone of an already-decoded picture. Nothing here decodes,
    /// and nothing here waits on the decoder, so it is safe to call from the event loop.
    pub fn nearest(&self, frame: u64) -> Option<VideoFrame> {
        self.previews.nearest(zvidlib::FrameIndex(frame))
    }
}

impl Drop for PreviewIndex {
    fn drop(&mut self) {
        self.cancellation.cancel();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod measurement {
    use std::future::Future;
    use std::task::{Context, Poll, Waker};
    use std::time::{Duration, Instant};

    use zvidlib::io::MemorySource;
    use zvidlib::{
        Codec, CodecProfile, ColorRange, Limits, Mp4Demuxer, Mp4DemuxerOptions, PixelFormat,
        SEEK_LATENCY_BUDGET, TrackKind, VideoDecoderConfig, native_hevc_video_decoder_factory,
    };

    use super::*;

    fn block_on<F: Future>(future: F) -> F::Output {
        let mut boxed = Box::pin(future);
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        match boxed.as_mut().poll(&mut context) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("unexpected pending future"),
        }
    }

    /// Issues #374 and #416: what a scrub anywhere on the bar costs once the index is built, on
    /// the real 1080p sample through the real hardware decoder.
    ///
    /// The point of the index is that this is not a decode, so the figure it prints should be
    /// microseconds rather than the second the same position costs to decode exactly. Ignored
    /// because it is a wall-clock reading on whatever host runs it, and because building the
    /// index decodes the whole bundled track; `seek::tests` holds the same budget on every run
    /// over a synthetic single-group track.
    #[test]
    #[ignore = "host-specific preview-latency measurement"]
    fn a_preview_answers_any_position_without_decoding() {
        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/media/BigBuckBunny.mp4"
        ));
        let source = MemorySource::new(bytes.to_vec());
        let demuxer = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
        let video = demuxer
            .tracks
            .iter()
            .find(|track| track.kind == TrackKind::Video)
            .unwrap();
        let limits = Limits::default();
        let samples = block_on(video.to_encoded_video_samples(&source, &limits)).unwrap();
        let frame_count = samples.len() as u64;
        let configuration = VideoDecoderConfig {
            codec: video.codec,
            profile: CodecProfile::HevcMain,
            coded_dimensions: video.dimensions.unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: zvidlib::HardwarePreference::Prefer,
            configuration: video.decoder_config.clone(),
        };
        assert_eq!(configuration.codec, Codec::Hevc);
        let factory = native_hevc_video_decoder_factory();
        let started = Instant::now();
        let index =
            PreviewIndex::new(&factory, configuration, samples, limits, 24).expect("preview index");

        // Wait for the pass to cover the whole track, which is the state a drag finds it in a
        // second or two after the window opens.
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let (filled, total) = index.coverage();
            if filled == total || Instant::now() >= deadline {
                assert_eq!(filled, total, "the preview pass did not finish in time");
                break;
            }
            std::thread::yield_now();
        }
        let built = started.elapsed();

        // Every position on the bar, sampled a hundred ways, answered from the index.
        let mut worst = Duration::ZERO;
        for step in 0..=100_u64 {
            let frame = step * (frame_count - 1) / 100;
            let started = Instant::now();
            let preview = index.nearest(frame).expect("a preview for every position");
            worst = worst.max(started.elapsed());
            assert_eq!(preview.pixel_format, PixelFormat::Rgba8);
        }
        eprintln!(
            "preview index over {frame_count} frames built in {built:?}; worst lookup {worst:?}"
        );
        assert!(
            worst < SEEK_LATENCY_BUDGET,
            "a preview lookup took {worst:?}, over the {SEEK_LATENCY_BUDGET:?} budget"
        );
    }
}
