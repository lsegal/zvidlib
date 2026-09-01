//! Background timeline scrubbing for the native GL example.
//!
//! Scrubbing used to run [`zvidlib::PlaybackController::seek`] inline on the event-loop thread,
//! once per pointer move. A seek resets the decoder and decodes forward from the preceding
//! random-access point, so the window stopped drawing for as long as that took and the next
//! pointer move queued another one behind it.
//!
//! This module moves that work off the event loop. A [`ScrubPreviewer`] owns a second
//! [`ExactFrameReader`] on a worker thread and answers the newest requested position, dropping
//! superseded ones and cancelling a decode that a newer request has already made stale. The
//! render thread only ever polls for a completed frame.
//!
//! The other half of the cost is which frame a preview decodes. A [`KeyframeIndex`] snaps every
//! in-drag target back to its random-access point, so a preview is one intra picture instead of a
//! whole group of pictures, and the exact frame is decoded once when the pointer is released.

use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use zvidlib::{
    CancellationToken, EncodedVideoSample, Error, ErrorKind, ExactFrameReader, FrameIndex, Limits,
    Result, VideoDecoderConfig, VideoDecoderFactory, VideoFrame,
};

/// The presentation indices a decode can start from, ascending.
pub struct KeyframeIndex {
    frames: Vec<u64>,
}

impl KeyframeIndex {
    /// Collects the random-access samples' presentation indices from a track's decode-order
    /// samples.
    pub fn from_samples(samples: &[EncodedVideoSample]) -> Self {
        let mut frames: Vec<u64> = samples
            .iter()
            .filter(|sample| sample.random_access)
            .map(|sample| sample.presentation_index.0)
            .collect();
        frames.sort_unstable();
        Self { frames }
    }

    /// The newest random-access frame at or before `frame`, or `frame` itself when the track
    /// indexes none before it.
    pub fn at_or_before(&self, frame: u64) -> u64 {
        match self.frames.partition_point(|candidate| *candidate <= frame) {
            0 => frame,
            position => self.frames[position - 1],
        }
    }
}

/// The frame a timeline position `fraction` of the way along a `frame_count`-frame track selects.
pub fn target_frame(fraction: f32, frame_count: u64) -> u64 {
    let maximum = frame_count.saturating_sub(1);
    (f64::from(fraction.clamp(0.0, 1.0)) * maximum as f64).round() as u64
}

struct Request {
    frame: u64,
    cancellation: CancellationToken,
}

#[derive(Default)]
struct Queue {
    /// The newest requested frame, replacing rather than queueing behind an older one.
    pending: Option<Request>,
    /// The request the worker is decoding right now, so a newer one can cancel it.
    in_flight: Option<CancellationToken>,
    shutdown: bool,
}

/// Decodes scrub previews on a worker thread, keeping only the newest requested position.
pub struct ScrubPreviewer {
    queue: Arc<(Mutex<Queue>, Condvar)>,
    frames: Receiver<(u64, VideoFrame)>,
    /// The last frame handed to [`Self::request`], so an unchanged target costs nothing.
    requested: Option<u64>,
    worker: Option<JoinHandle<()>>,
}

impl ScrubPreviewer {
    /// Builds a previewer over its own decoder, independent of the one playback is using.
    pub fn new(
        factory: &dyn VideoDecoderFactory,
        configuration: VideoDecoderConfig,
        samples: Vec<EncodedVideoSample>,
        limits: Limits,
    ) -> Result<Self> {
        let reader = ExactFrameReader::new(factory, configuration, samples, limits)?;
        let (sender, frames) = channel();
        let queue = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        let worker_queue = Arc::clone(&queue);
        let worker = thread::Builder::new()
            .name("zvidlib-scrub-preview".to_string())
            .spawn(move || decode_previews(&worker_queue, &sender, reader))
            .map_err(|error| {
                Error::new(
                    ErrorKind::InvalidInput,
                    format!("could not start the scrub preview thread: {error}"),
                )
            })?;
        Ok(Self {
            queue,
            frames,
            requested: None,
            worker: Some(worker),
        })
    }

    /// Asks for `frame`, cancelling any older request. Returns whether this enqueued new work,
    /// and never blocks on the decode itself.
    pub fn request(&mut self, frame: u64) -> bool {
        if self.requested == Some(frame) {
            return false;
        }
        self.requested = Some(frame);
        let (lock, condvar) = &*self.queue;
        let mut state = lock.lock().expect("scrub queue poisoned");
        if let Some(superseded) = state.pending.take() {
            superseded.cancellation.cancel();
        }
        if let Some(in_flight) = state.in_flight.as_ref() {
            in_flight.cancel();
        }
        state.pending = Some(Request {
            frame,
            cancellation: CancellationToken::new(),
        });
        condvar.notify_one();
        true
    }

    /// The newest completed preview, discarding any that a later one has already superseded.
    /// Returns `None` rather than waiting when nothing has finished decoding.
    pub fn take_latest(&mut self) -> Option<VideoFrame> {
        let mut latest = None;
        loop {
            match self.frames.try_recv() {
                Ok((_, frame)) => latest = Some(frame),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return latest,
            }
        }
    }

    /// Forgets the last requested position, so the next drag re-requests it even if it lands on
    /// the same frame.
    pub fn forget_request(&mut self) {
        self.requested = None;
    }
}

impl Drop for ScrubPreviewer {
    fn drop(&mut self) {
        {
            let (lock, condvar) = &*self.queue;
            let mut state = lock.lock().expect("scrub queue poisoned");
            state.shutdown = true;
            if let Some(pending) = state.pending.take() {
                pending.cancellation.cancel();
            }
            if let Some(in_flight) = state.in_flight.as_ref() {
                in_flight.cancel();
            }
            condvar.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn decode_previews(
    queue: &Arc<(Mutex<Queue>, Condvar)>,
    sender: &Sender<(u64, VideoFrame)>,
    mut reader: ExactFrameReader,
) {
    let (lock, condvar) = &**queue;
    loop {
        let request = {
            let mut state = lock.lock().expect("scrub queue poisoned");
            loop {
                if state.shutdown {
                    return;
                }
                if let Some(request) = state.pending.take() {
                    state.in_flight = Some(request.cancellation.clone());
                    break request;
                }
                state = condvar.wait(state).expect("scrub queue poisoned");
            }
        };
        // Decoded outside the lock so a newer request can both replace this one and cancel it
        // part-way through.
        let decoded = reader.get(FrameIndex(request.frame), &request.cancellation);
        lock.lock().expect("scrub queue poisoned").in_flight = None;
        // A cancelled or failed preview is simply not drawn; the pointer has moved on, and the
        // frame the scrub finally commits to is decoded by the playback controller instead.
        if let Ok(frame) = decoded {
            if sender.send((request.frame, frame)).is_err() {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    use zvidlib::{
        CodecProfile, CodecSupport, ColorRange, DecodedVideoFrame, PixelFormat, Plane,
        VideoDecoder, VideoDimensions,
    };

    fn samples(count: u64, keyframe_every: u64) -> Vec<EncodedVideoSample> {
        (0..count)
            .map(|index| EncodedVideoSample {
                presentation_index: FrameIndex(index),
                random_access: index % keyframe_every == 0,
                data: vec![index as u8],
            })
            .collect()
    }

    #[test]
    fn a_target_snaps_back_to_the_random_access_point_that_decodes_it() {
        let index = KeyframeIndex::from_samples(&samples(12, 4));
        assert_eq!(index.at_or_before(0), 0);
        assert_eq!(index.at_or_before(3), 0);
        assert_eq!(index.at_or_before(4), 4);
        assert_eq!(index.at_or_before(7), 4);
        assert_eq!(index.at_or_before(11), 8);
        // Past the last indexed frame the caller's own target is the best answer available.
        assert_eq!(index.at_or_before(40), 8);
    }

    #[test]
    fn a_track_without_a_leading_random_access_point_returns_the_target_itself() {
        let mut without = samples(4, 1);
        for sample in &mut without {
            sample.random_access = false;
        }
        let index = KeyframeIndex::from_samples(&without);
        assert_eq!(index.at_or_before(2), 2);
    }

    #[test]
    fn a_timeline_fraction_selects_the_frame_it_points_at() {
        assert_eq!(target_frame(0.0, 101), 0);
        assert_eq!(target_frame(0.5, 101), 50);
        assert_eq!(target_frame(1.0, 101), 100);
        // Out-of-range fractions clamp to the track rather than indexing past it.
        assert_eq!(target_frame(-1.0, 101), 0);
        assert_eq!(target_frame(2.0, 101), 100);
        assert_eq!(target_frame(0.5, 0), 0);
    }

    /// Blocks inside `submit` until the test opens its gate, so a request can be observed while
    /// it is still in flight.
    struct GatedDecoder {
        gate: Arc<(Mutex<bool>, Condvar)>,
    }

    impl VideoDecoder for GatedDecoder {
        fn submit(
            &mut self,
            sample: &EncodedVideoSample,
            _: &CancellationToken,
        ) -> Result<Vec<DecodedVideoFrame>> {
            let (lock, condvar) = &*self.gate;
            let mut open = lock.lock().expect("gate poisoned");
            while !*open {
                open = condvar.wait(open).expect("gate poisoned");
            }
            drop(open);
            Ok(vec![DecodedVideoFrame {
                presentation_index: sample.presentation_index,
                frame: VideoFrame::new(
                    VideoDimensions::new(1, 1, &Limits::default())?,
                    PixelFormat::Rgba8,
                    ColorRange::Limited,
                    vec![Plane {
                        data: vec![sample.presentation_index.0 as u8, 0, 0, 255],
                        stride: 4,
                    }],
                    &Limits::default(),
                )?,
            }])
        }

        fn drain(&mut self, _: &CancellationToken) -> Result<Vec<DecodedVideoFrame>> {
            Ok(Vec::new())
        }

        fn reset(&mut self) -> Result<()> {
            Ok(())
        }
    }

    struct GatedFactory {
        gate: Arc<(Mutex<bool>, Condvar)>,
    }

    impl VideoDecoderFactory for GatedFactory {
        fn capability(&self, _: &VideoDecoderConfig) -> CodecSupport {
            CodecSupport::Supported {
                implementation: zvidlib::CodecImplementation::Software,
            }
        }

        fn create(&self, _: &VideoDecoderConfig, _: &Limits) -> Result<Box<dyn VideoDecoder>> {
            Ok(Box::new(GatedDecoder {
                gate: Arc::clone(&self.gate),
            }))
        }
    }

    fn configuration() -> VideoDecoderConfig {
        VideoDecoderConfig {
            codec: zvidlib::Codec::Hevc,
            profile: CodecProfile::HevcMain,
            coded_dimensions: VideoDimensions::new(1, 1, &Limits::default()).unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: zvidlib::HardwarePreference::Avoid,
            configuration: Vec::new(),
        }
    }

    #[test]
    fn requesting_a_preview_does_not_wait_for_it_and_the_newest_target_wins() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let factory = GatedFactory {
            gate: Arc::clone(&gate),
        };
        let mut previewer =
            ScrubPreviewer::new(&factory, configuration(), samples(12, 4), Limits::default())
                .unwrap();

        // Every request returns while the decoder is still blocked, and nothing has been drawn.
        assert!(previewer.request(0));
        assert!(previewer.take_latest().is_none());
        // An unchanged target is not re-decoded, and a changed one supersedes the first.
        assert!(!previewer.request(0));
        assert!(previewer.request(8));

        {
            let (lock, condvar) = &*gate;
            *lock.lock().unwrap() = true;
            condvar.notify_all();
        }

        let deadline = Instant::now() + Duration::from_secs(10);
        let mut latest = None;
        while Instant::now() < deadline {
            if let Some(frame) = previewer.take_latest() {
                latest = Some(frame.planes[0].data[0]);
                if latest == Some(8) {
                    break;
                }
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            latest,
            Some(8),
            "the newest requested preview is the one drawn"
        );

        // A drag that ends and starts again on the same frame asks for it afresh.
        previewer.forget_request();
        assert!(previewer.request(8));
    }
}
