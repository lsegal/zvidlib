//! Background frame decoding for the native GL example.
//!
//! Every frame this example draws is decoded on a worker thread and collected by the render
//! thread without waiting. Nothing on the event loop ever calls into the decoder.
//!
//! Scrubbing is why. #319 moved the drag's previews onto a worker but left the commit, and every
//! seek after it, decoding inline: [`zvidlib::PlaybackController::seek`] is cheap, but the
//! `current_frame` or `present` that follows it decodes forward from the preceding random-access
//! point on whichever thread calls it. The bundled sample codes its 768 frames as a single group
//! of pictures - its `stss` names exactly one sync sample - so committing a scrub four fifths of
//! the way along the bar decoded 613 frames on the event-loop thread and froze the window for
//! 8.4 seconds (issue #333).
//!
//! A [`FrameService`] owns the one [`ExactFrameReader`] the example has and answers two callers:
//!
//! * The drag asks for the frame under the pointer and draws whatever has arrived. The worker
//!   does not jump straight to it - it walks there from the target's random-access point in
//!   strides, publishing the frames it stops on, so the picture tracks the pointer from the first
//!   intra picture onwards instead of holding still for the whole decode. A newer target replaces
//!   the older one and redirects the walk; a walk that has to start over cancels the decode it is
//!   inside rather than finishing it. Only the frames the walk stops on are converted to RGBA:
//!   the reader tells the decoder that the pictures it is passing are wanted for reference only
//!   (issue #354), which is what makes the far end of the bar cost what the near end does.
//! * Playback reads through a [`FrameServiceSource`], which is the same worker behind
//!   [`zvidlib::PlaybackVideoSource`]. A frame that has not been decoded yet is reported as
//!   [`ErrorKind::WouldBlock`] and the render thread simply keeps the picture it has, so a seek
//!   never blocks the loop that draws.
//!
//! Sharing one reader between the two is also what makes committing a scrub free: the drag has
//! already walked the decoder to the frame the commit asks for, so playback's request for it is a
//! cache hit rather than a second decode through the same 613 frames.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use zvidlib::{
    CancellationToken, EncodedVideoSample, Error, ErrorKind, ExactFrameReader, FrameIndex, Limits,
    PlaybackVideoSource, Result, VideoDecoderConfig, VideoDecoderFactory, VideoFrame,
};

/// How far apart the frames a walk publishes are.
///
/// The reader decodes every sample in between either way, but only the ones a walk stops on cost
/// a colour conversion now that [`ExactFrameReader`] tells the decoder which pictures are wanted
/// (issue #354). That is what sets this: a published frame costs its own conversion plus the
/// decoding of the frames since the last one, about 6.6 ms + 2.9 ms per frame on the bundled
/// 1080p sample through VideoToolbox, so a stride of eight lands at roughly the 30 ms the issue
/// #333 budget asks for while reaching the far end of the bar in a third of the time. A stride of
/// one publishes more often and pays a conversion for every frame it passes: 7.7 s to walk to the
/// last frame of the bundled sample, against 3.0 s at eight.
const WALK_STRIDE: u64 = 8;

/// How many decoded frames the render thread can still collect.
///
/// Playback asks for one frame per redraw and a drag draws the newest, so a couple of frames of
/// slack is all a poll needs - and a 1080p RGBA frame is 8 MiB, so this is the memory the queue
/// costs.
const DELIVERY_DEPTH: usize = 3;

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

#[derive(Default)]
struct Queue {
    /// The frame the worker is walking towards, replacing rather than queueing behind an older
    /// one.
    target: Option<u64>,
    /// The frame the worker's current decode is for, so a new target can tell whether that
    /// decode is still on the way to it.
    cursor: Option<u64>,
    /// The decode the worker is inside, so a target it can no longer reach can cancel it.
    in_flight: Option<CancellationToken>,
    /// What the worker has decoded and the render thread has not collected, oldest first.
    delivered: VecDeque<(u64, VideoFrame)>,
    /// A decode failure to hand to whichever caller asks next.
    failure: Option<Error>,
    shutdown: bool,
}

impl Queue {
    /// Points the worker at `frame`, cancelling a decode that is no longer on the way there.
    ///
    /// Returns whether this changed the target: repeating one must not disturb the decode that is
    /// already serving it. Neither must moving the target further ahead - the frame being decoded
    /// still lies between the reader and the new target, so cancelling it would throw away work
    /// the new target needs and send the walk back to a random-access point. Only a target the
    /// current decode has already passed cancels it.
    fn retarget(&mut self, frame: u64) -> bool {
        if self.target == Some(frame) {
            return false;
        }
        self.target = Some(frame);
        if self.cursor.is_some_and(|cursor| cursor > frame) {
            if let Some(in_flight) = self.in_flight.as_ref() {
                in_flight.cancel();
            }
        }
        true
    }

    /// The delivered frame with exactly this index, if the render thread has not passed it yet.
    fn take_exact(&mut self, frame: u64) -> Option<VideoFrame> {
        let position = self
            .delivered
            .iter()
            .position(|(index, _)| *index == frame)?;
        // Everything older than the frame being drawn is never asked for again.
        self.delivered.drain(..position);
        self.delivered.front().map(|(_, frame)| frame.clone())
    }
}

/// Decodes on a worker thread, keeping only the newest requested position.
pub struct FrameService {
    queue: Arc<(Mutex<Queue>, Condvar)>,
    worker: Option<JoinHandle<()>>,
}

impl FrameService {
    /// Builds a service over its own decoder and the track's random-access points.
    pub fn new(
        factory: &dyn VideoDecoderFactory,
        configuration: VideoDecoderConfig,
        samples: Vec<EncodedVideoSample>,
        limits: Limits,
    ) -> Result<Self> {
        let keyframes = KeyframeIndex::from_samples(&samples);
        let reader = ExactFrameReader::new(factory, configuration, samples, limits)?;
        let queue = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        let worker_queue = Arc::clone(&queue);
        let worker = thread::Builder::new()
            .name("zvidlib-frame-service".to_string())
            .spawn(move || decode_frames(&worker_queue, reader, &keyframes))
            .map_err(|error| {
                Error::new(
                    ErrorKind::InvalidInput,
                    format!("could not start the frame decode thread: {error}"),
                )
            })?;
        Ok(Self {
            queue,
            worker: Some(worker),
        })
    }

    /// A handle the playback controller reads frames through, sharing this decoder.
    pub fn source(&self) -> FrameServiceSource {
        FrameServiceSource {
            queue: Arc::clone(&self.queue),
        }
    }

    /// Asks for `frame`, redirecting any older request. Returns whether this changed the target,
    /// and never waits for the decode itself.
    pub fn request(&mut self, frame: u64) -> bool {
        let (lock, condvar) = &*self.queue;
        let retargeted = lock.lock().expect("frame queue poisoned").retarget(frame);
        if retargeted {
            condvar.notify_one();
        }
        retargeted
    }

    /// The newest frame the worker has decoded, or `None` rather than waiting for one.
    pub fn take_latest(&mut self) -> Option<VideoFrame> {
        let (lock, _) = &*self.queue;
        let mut state = lock.lock().expect("frame queue poisoned");
        let newest = state.delivered.pop_back().map(|(_, frame)| frame);
        state.delivered.clear();
        newest
    }
}

impl Drop for FrameService {
    fn drop(&mut self) {
        {
            let (lock, condvar) = &*self.queue;
            let mut state = lock.lock().expect("frame queue poisoned");
            state.shutdown = true;
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

/// The playback controller's view of the [`FrameService`]: it asks, it does not wait.
pub struct FrameServiceSource {
    queue: Arc<(Mutex<Queue>, Condvar)>,
}

impl PlaybackVideoSource for FrameServiceSource {
    /// Returns `frame` when the worker has already decoded it, and otherwise points the worker at
    /// it and reports [`ErrorKind::WouldBlock`] instead of blocking the caller.
    fn get_exact(&mut self, frame: FrameIndex, _: &CancellationToken) -> Result<VideoFrame> {
        let (lock, condvar) = &*self.queue;
        let mut state = lock.lock().expect("frame queue poisoned");
        if let Some(decoded) = state.take_exact(frame.0) {
            return Ok(decoded);
        }
        if let Some(failure) = state.failure.take() {
            return Err(failure);
        }
        if state.retarget(frame.0) {
            condvar.notify_one();
        }
        Err(Error::new(
            ErrorKind::WouldBlock,
            "the requested frame is still decoding",
        ))
    }

    /// Keeps the decoder where it is: [`ExactFrameReader`] already resets itself when a request
    /// needs it to, and throwing its position away is what makes a seek expensive.
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Walks the reader to whatever frame the queue is pointing at, publishing what it passes.
fn decode_frames(
    queue: &Arc<(Mutex<Queue>, Condvar)>,
    mut reader: ExactFrameReader,
    keyframes: &KeyframeIndex,
) {
    let (lock, condvar) = &**queue;
    // Where the last published frame left the reader, so a forward request continues from it
    // rather than starting over at a random-access point.
    let mut position: Option<u64> = None;
    loop {
        let (mut target, mut cancellation) = {
            let mut state = lock.lock().expect("frame queue poisoned");
            loop {
                if state.shutdown {
                    return;
                }
                if let Some(target) = state.target {
                    let cancellation = CancellationToken::new();
                    state.in_flight = Some(cancellation.clone());
                    break (target, cancellation);
                }
                state = condvar.wait(state).expect("frame queue poisoned");
            }
        };
        let mut cursor = walk_start(position, target, keyframes);
        loop {
            lock.lock().expect("frame queue poisoned").cursor = Some(cursor);
            // Decoded outside the lock so a newer target can both replace this one and cancel the
            // decode part-way through.
            let decoded = reader.get(FrameIndex(cursor), &cancellation);
            let reached = decoded.is_ok();
            let mut state = lock.lock().expect("frame queue poisoned");
            if state.shutdown {
                return;
            }
            match decoded {
                Ok(frame) => {
                    position = Some(cursor);
                    while state.delivered.len() >= DELIVERY_DEPTH {
                        state.delivered.pop_front();
                    }
                    state.delivered.push_back((cursor, frame));
                }
                Err(error) if error.kind() == ErrorKind::Cancelled => {
                    // Superseded part-way through, so the reader stopped between frames and the
                    // next walk starts from a random-access point rather than from nowhere.
                    position = None;
                }
                Err(error) => {
                    // Handed to whichever caller asks next; this walk cannot make progress.
                    position = None;
                    state.failure = Some(error);
                    state.target = None;
                }
            }
            let Some(current) = state.target else {
                state.cursor = None;
                state.in_flight = None;
                break;
            };
            if current != target {
                target = current;
                cursor = walk_start(position, target, keyframes);
            } else if reached && cursor == target {
                // The walk arrived. Clearing the target rather than holding it parks the worker
                // instead of decoding the same frame again, and lets a drag that comes back to
                // this frame ask for it afresh - out of the reader's cache, for nothing.
                state.target = None;
                state.cursor = None;
                state.in_flight = None;
                break;
            } else if reached {
                cursor = target.min(cursor.saturating_add(WALK_STRIDE));
            } else {
                cursor = walk_start(position, target, keyframes);
            }
            if cancellation.is_cancelled() {
                cancellation = CancellationToken::new();
                state.in_flight = Some(cancellation.clone());
            }
        }
    }
}

/// The first frame of a walk from `position` to `target`.
///
/// A target ahead of where the reader already is continues from there, decoding only what lies
/// between. Anything else restarts at the target's random-access point, which is the frame the
/// reader would have to decode from anyway and the first one it can publish.
fn walk_start(position: Option<u64>, target: u64, keyframes: &KeyframeIndex) -> u64 {
    match position {
        Some(position) if position < target => target.min(position.saturating_add(WALK_STRIDE)),
        Some(position) if position == target => target,
        _ => keyframes.at_or_before(target),
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

    #[test]
    fn a_walk_continues_forwards_and_restarts_from_a_random_access_point_otherwise() {
        let keyframes = KeyframeIndex::from_samples(&samples(64, 16));
        // Nothing decoded yet, or a target behind the reader: start where the decode has to.
        assert_eq!(walk_start(None, 40, &keyframes), 32);
        assert_eq!(walk_start(Some(50), 40, &keyframes), 32);
        // Ahead of the reader: continue from it, one stride at a time, without overshooting.
        assert_eq!(walk_start(Some(32), 40, &keyframes), 32 + WALK_STRIDE);
        assert_eq!(walk_start(Some(39), 40, &keyframes), 40);
        assert_eq!(walk_start(Some(40), 40, &keyframes), 40);
    }

    /// How many samples a decoder may still decode before it blocks, so a test can hold a walk
    /// part-way through and see what it has published so far.
    type Gate = Arc<(Mutex<u32>, Condvar)>;

    fn gate(permits: u32) -> Gate {
        Arc::new((Mutex::new(permits), Condvar::new()))
    }

    fn grant(gate: &Gate, permits: u32) {
        *gate.0.lock().expect("gate poisoned") += permits;
        gate.1.notify_all();
    }

    /// Emits the frame it is handed, with the presentation index in the first pixel.
    struct IndexedDecoder {
        gate: Option<Gate>,
    }

    impl VideoDecoder for IndexedDecoder {
        fn submit(
            &mut self,
            sample: &EncodedVideoSample,
            _: &CancellationToken,
        ) -> Result<Vec<DecodedVideoFrame>> {
            if let Some(gate) = self.gate.as_ref() {
                let (lock, condvar) = &**gate;
                let mut permits = lock.lock().expect("gate poisoned");
                while *permits == 0 {
                    permits = condvar.wait(permits).expect("gate poisoned");
                }
                *permits -= 1;
            }
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

    struct IndexedFactory {
        gate: Option<Gate>,
    }

    impl VideoDecoderFactory for IndexedFactory {
        fn capability(&self, _: &VideoDecoderConfig) -> CodecSupport {
            CodecSupport::Supported {
                implementation: zvidlib::CodecImplementation::Software,
            }
        }

        fn create(&self, _: &VideoDecoderConfig, _: &Limits) -> Result<Box<dyn VideoDecoder>> {
            Ok(Box::new(IndexedDecoder {
                gate: self.gate.clone(),
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

    fn open(gate: Option<Gate>, samples: Vec<EncodedVideoSample>) -> FrameService {
        FrameService::new(
            &IndexedFactory { gate },
            configuration(),
            samples,
            Limits::default(),
        )
        .unwrap()
    }

    /// Polls `poll` until it answers, or gives up after ten seconds.
    fn wait_for<T>(mut poll: impl FnMut() -> Option<T>) -> Option<T> {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if let Some(value) = poll() {
                return Some(value);
            }
            thread::sleep(Duration::from_millis(2));
        }
        None
    }

    #[test]
    fn requesting_a_frame_does_not_wait_for_it_and_the_newest_target_wins() {
        let gate = gate(0);
        let mut frames = open(Some(Arc::clone(&gate)), samples(12, 4));

        // Every request returns while the decoder is still blocked, and nothing has been drawn.
        assert!(frames.request(0));
        assert!(frames.take_latest().is_none());
        // An unchanged target is not re-requested, and a changed one redirects the walk.
        assert!(!frames.request(0));
        assert!(frames.request(8));

        grant(&gate, u32::MAX / 2);

        let drawn = wait_for(|| {
            frames
                .take_latest()
                .map(|frame| frame.planes[0].data[0])
                .filter(|index| *index == 8)
        });
        assert_eq!(
            drawn,
            Some(8),
            "the newest requested frame is the one drawn"
        );
    }

    #[test]
    fn playback_is_told_a_frame_is_still_decoding_rather_than_being_made_to_wait() {
        let gate = gate(0);
        let frames = open(Some(Arc::clone(&gate)), samples(12, 4));
        let mut source = frames.source();
        let token = CancellationToken::new();

        // The very first read returns immediately, with the decoder still blocked on frame 0.
        let start = Instant::now();
        let pending = source.get_exact(FrameIndex(4), &token);
        assert!(start.elapsed() < Duration::from_secs(1), "get_exact waited");
        assert_eq!(
            pending.map(|_| ()).unwrap_err().kind(),
            ErrorKind::WouldBlock
        );

        grant(&gate, u32::MAX / 2);

        let frame = wait_for(|| source.get_exact(FrameIndex(4), &token).ok());
        assert_eq!(
            frame.map(|frame| frame.planes[0].data[0]),
            Some(4),
            "the frame is delivered once it has decoded"
        );
    }

    #[test]
    fn a_walk_through_a_single_group_of_pictures_draws_the_frames_it_passes() {
        // One random-access point, as the bundled sample has: reaching frame 9 means decoding
        // every frame from 0, and a scrub that showed nothing until it arrived is the freeze.
        let mut single_gop = samples(12, 1);
        for sample in single_gop.iter_mut().skip(1) {
            sample.random_access = false;
        }
        let gate = gate(0);
        let mut frames = open(Some(Arc::clone(&gate)), single_gop);
        frames.request(9);

        // One sample's worth of decoding is enough to publish something: the walk starts at the
        // random-access point rather than at the target, so the window has a picture long before
        // the nine frames the target needs have been decoded.
        grant(&gate, 1);
        let first = wait_for(|| frames.take_latest()).map(|frame| frame.planes[0].data[0]);
        assert_eq!(first, Some(0), "the walk draws where it starts");

        grant(&gate, u32::MAX / 2);
        let arrived = wait_for(|| {
            frames
                .take_latest()
                .map(|frame| frame.planes[0].data[0])
                .filter(|index| *index == 9)
        });
        assert_eq!(arrived, Some(9), "and then walks on to the target");
    }
}
