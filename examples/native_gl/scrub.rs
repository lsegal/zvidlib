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
//! * The drag asks for the frame under the pointer and draws it when it arrives. A newer target
//!   replaces the older one rather than queueing behind it, and one the current decode has
//!   already passed cancels that decode instead of waiting for a frame nothing will draw.
//!
//!   The worker asks for that frame and nothing else. #319 and #339 had it stop every frame on
//!   the way there and draw each one, so the picture would track the pointer - but on this
//!   sample every stop is a frame of the *start* of the movie while the pointer is at the end,
//!   and each one costs a full-resolution NV12-to-RGBA pass (6.6 ms against the 2.9 ms the
//!   decoding itself takes) for a picture that is nowhere near where the pointer is. That is
//!   what issue #354 is: the far end of the bar took 6.4 s to arrive at. `ExactFrameReader`
//!   now tells the decoder that the pictures between where it is and the frame that was asked
//!   for are wanted for reference only, so they cost their decoding and nothing else, and the
//!   frame under the pointer arrives in 1.7 s instead.
//! * Playback reads through a [`FrameServiceSource`], which is the same worker behind
//!   [`zvidlib::PlaybackVideoSource`]. A frame that has not been decoded yet is reported as
//!   [`ErrorKind::WouldBlock`] and the render thread simply keeps the picture it has, so a seek
//!   never blocks the loop that draws.
//!
//! Sharing one reader between the two is also what makes committing a scrub free: the drag has
//! already walked the decoder to the frame the commit asks for, so playback's request for it is a
//! cache hit rather than a second decode through the same 613 frames. The frames just behind it
//! are cached too - the reader keeps as many as its cache can hold rather than skipping them -
//! so the previous-frame key is a cache hit after a scrub as much as it is during playback.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use zvidlib::{
    CancellationToken, EncodedVideoSample, Error, ErrorKind, ExactFrameReader, FrameIndex, Limits,
    PlaybackVideoSource, Result, VideoDecoderConfig, VideoDecoderFactory, VideoFrame,
};

/// How many decoded frames the render thread can still collect.
///
/// Playback asks for one frame per redraw and a drag draws the newest, so a couple of frames of
/// slack is all a poll needs - and a 1080p RGBA frame is 8 MiB, so this is the memory the queue
/// costs.
const DELIVERY_DEPTH: usize = 3;

/// How often a walk towards a preview target publishes a picture.
///
/// #354 is why this is a time and not a frame count. Publishing every frame a walk passed put a
/// full-resolution NV12-to-RGBA conversion (6.6 ms against the 2.9 ms decoding costs) in front of
/// each one, and on this sample most of them are frames of the start of the movie while the
/// pointer is at the far end, so the frame that was actually asked for arrived 6.4 s late. Not
/// publishing any of them is what #355 traded for a 1.7 s arrival - and it left the picture
/// frozen for those 1.7 s, which is issue #363. A cadence buys back the motion at a bounded
/// price: about seven pictures a second however long the span is, so the same 613-frame walk
/// converts a dozen frames rather than 613 and still lands its target within a few percent of
/// the time it takes to decode nothing else at all.
const PREVIEW_INTERVAL: Duration = Duration::from_millis(150);

/// How many frames a walk decodes between published pictures.
///
/// The first step of a walk has nothing measured to go on and so publishes immediately; after
/// that the stride is whatever fits in [`PREVIEW_INTERVAL`] at the rate the walk is actually
/// decoding at, which on this sample climbs from one frame to roughly fifty as the estimate
/// settles. A decode slower than the interval still moves a frame at a time rather than stalling.
fn stride_frames(per_frame: Option<Duration>) -> u64 {
    let Some(per_frame) = per_frame else {
        return 1;
    };
    let per_frame = per_frame.as_secs_f64();
    if per_frame <= 0.0 {
        return MAXIMUM_STRIDE;
    }
    let frames = (PREVIEW_INTERVAL.as_secs_f64() / per_frame).floor();
    (frames as u64).clamp(1, MAXIMUM_STRIDE)
}

/// A ceiling on the stride, so an unmeasurably fast decoder still publishes pictures rather than
/// jumping the whole span in one silent step.
const MAXIMUM_STRIDE: u64 = 512;

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

/// The first frame of a walk from `position` towards `target`.
///
/// Decoding only runs forwards. A target ahead of where the reader already is continues from
/// there, striding towards it; anything else - a target behind it, or a reader whose position a
/// cancelled decode left unknown - restarts at the target's random-access point, which is the
/// frame the reader would have to decode from anyway and the first one it can publish.
fn walk_step(
    position: Option<u64>,
    target: u64,
    per_frame: Option<Duration>,
    keyframes: &KeyframeIndex,
) -> u64 {
    match position {
        Some(position) if position < target => {
            target.min(position.saturating_add(stride_frames(per_frame)))
        }
        Some(position) if position == target => target,
        _ => keyframes.at_or_before(target),
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
    /// Whether that target is a drag's preview, which publishes pictures on the way to it, or an
    /// exact request from playback, which wants that frame and no other.
    preview: bool,
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
    /// lies between the reader and the new target, so cancelling it would throw away reference
    /// decoding the new target needs and send the reader back to a random-access point. Only a
    /// target the current decode has already passed cancels it.
    fn retarget(&mut self, frame: u64, preview: bool) -> bool {
        if self.target == Some(frame) && self.preview == preview {
            return false;
        }
        self.target = Some(frame);
        self.preview = preview;
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
    /// Builds a service over its own decoder for the track's samples.
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

    /// Asks for `frame` as a drag's preview, redirecting any older request. Returns whether this
    /// changed the target, and never waits for the decode itself.
    ///
    /// A preview publishes the pictures the walk passes on a time cadence as well as the frame
    /// itself, so a drag keeps moving while a long span decodes (issue #363).
    pub fn request(&mut self, frame: u64) -> bool {
        let (lock, condvar) = &*self.queue;
        let retargeted = lock
            .lock()
            .expect("frame queue poisoned")
            .retarget(frame, true);
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
        if state.retarget(frame.0, false) {
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

/// Walks the reader to whatever frame the queue is pointing at, publishing pictures as it goes.
///
/// An exact request - playback's - asks for its frame and nothing else. A preview strides towards
/// its target instead, publishing a picture roughly every [`PREVIEW_INTERVAL`], so a drag over a
/// long span keeps moving rather than holding one picture until the span finishes decoding.
fn decode_frames(
    queue: &Arc<(Mutex<Queue>, Condvar)>,
    mut reader: ExactFrameReader,
    keyframes: &KeyframeIndex,
) {
    let (lock, condvar) = &**queue;
    // Where the last decoded frame left the reader, so a forward step continues from it rather
    // than starting over at a random-access point.
    let mut position: Option<u64> = None;
    // What a frame of this track is costing this walk, measured rather than assumed: it is the
    // difference between the 2.9 ms a decode takes and the 6.6 ms a published picture adds, and
    // it is what decides how many frames fit in one interval.
    let mut per_frame: Option<Duration> = None;
    loop {
        let (mut target, mut preview, mut cancellation) = {
            let mut state = lock.lock().expect("frame queue poisoned");
            loop {
                if state.shutdown {
                    return;
                }
                if let Some(target) = state.target {
                    let cancellation = CancellationToken::new();
                    state.in_flight = Some(cancellation.clone());
                    break (target, state.preview, cancellation);
                }
                state = condvar.wait(state).expect("frame queue poisoned");
            }
        };
        loop {
            let step = if preview {
                walk_step(position, target, per_frame, keyframes)
            } else {
                target
            };
            lock.lock().expect("frame queue poisoned").cursor = Some(step);
            let started = Instant::now();
            // Decoded outside the lock so a newer target can both replace this one and cancel the
            // decode part-way through. Everything between the reader's position and this frame is
            // decoded inside this one call, for reference only - which is what keeps a stride's
            // skipped frames costing their decoding and no conversion (#355).
            let decoded = reader.get(FrameIndex(step), &cancellation);
            let elapsed = started.elapsed();
            let mut state = lock.lock().expect("frame queue poisoned");
            if state.shutdown {
                return;
            }
            let mut reached = false;
            match decoded {
                Ok(frame) => {
                    // Only a step whose span is known says anything about the rate: a restart at
                    // a random-access point decoded an unknown number of frames to get there.
                    if let Some(previous) = position {
                        if let Ok(span) = u32::try_from(step.saturating_sub(previous)) {
                            if span > 0 {
                                per_frame = Some(elapsed / span);
                            }
                        }
                    }
                    position = Some(step);
                    reached = true;
                    while state.delivered.len() >= DELIVERY_DEPTH {
                        state.delivered.pop_front();
                    }
                    state.delivered.push_back((step, frame));
                }
                Err(error) if error.kind() == ErrorKind::Cancelled => {
                    // Superseded part-way through, so the reader stopped between frames and the
                    // next walk starts from a random-access point rather than from nowhere.
                    position = None;
                }
                Err(error) => {
                    // Handed to whichever caller asks next; this request cannot make progress.
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
            if current != target || state.preview != preview {
                target = current;
                preview = state.preview;
            } else if reached && step == target {
                // Arrived. Clearing the target rather than holding it parks the worker instead
                // of decoding the same frame again, and lets a drag that comes back to this
                // frame ask for it afresh - out of the reader's cache, for nothing.
                state.target = None;
                state.cursor = None;
                state.in_flight = None;
                break;
            }
            if cancellation.is_cancelled() {
                cancellation = CancellationToken::new();
                state.in_flight = Some(cancellation.clone());
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
    fn a_timeline_fraction_selects_the_frame_it_points_at() {
        assert_eq!(target_frame(0.0, 101), 0);
        assert_eq!(target_frame(0.5, 101), 50);
        assert_eq!(target_frame(1.0, 101), 100);
        // Out-of-range fractions clamp to the track rather than indexing past it.
        assert_eq!(target_frame(-1.0, 101), 0);
        assert_eq!(target_frame(2.0, 101), 100);
        assert_eq!(target_frame(0.5, 0), 0);
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

    /// Polls `poll` for a tenth of a second, for asserting that nothing arrives.
    fn wait_for_a_moment<T>(mut poll: impl FnMut() -> Option<T>) -> Option<T> {
        let deadline = Instant::now() + Duration::from_millis(100);
        while Instant::now() < deadline {
            if let Some(value) = poll() {
                return Some(value);
            }
            thread::sleep(Duration::from_millis(2));
        }
        None
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
    fn a_preview_keeps_drawing_on_the_way_to_a_far_target_without_drawing_every_frame() {
        // One random-access point, as the bundled sample has: reaching frame 9 means decoding
        // every frame from 0. Issue #354 is what those nine pictures used to cost when each was
        // drawn, and issue #363 is what the window looked like when none of them was: a drag that
        // holds one picture until the whole span finishes. A preview does neither. It publishes
        // on a cadence, so the picture moves during the walk, and the frame under the pointer is
        // still the one it ends on.
        let mut single_gop = samples(12, 1);
        for sample in single_gop.iter_mut().skip(1) {
            sample.random_access = false;
        }
        let gate = gate(0);
        let mut frames = open(Some(Arc::clone(&gate)), single_gop);
        frames.request(9);

        // Enough decoding for the frames before the target, and the picture has already moved.
        grant(&gate, 8);
        let early = wait_for(|| frames.take_latest().map(|frame| frame.planes[0].data[0]));
        assert!(
            early.is_some_and(|index| index < 9),
            "a preview draws something on the way to its target, not only the target"
        );

        grant(&gate, u32::MAX / 2);
        let arrived = wait_for(|| {
            frames
                .take_latest()
                .map(|frame| frame.planes[0].data[0])
                .filter(|index| *index == 9)
        });
        assert_eq!(
            arrived,
            Some(9),
            "the frame that was asked for is the one it ends on"
        );
    }

    #[test]
    fn an_exact_request_draws_its_own_frame_and_nothing_it_passed() {
        // Playback's path, which #354 is about: it wants the frame the clock calls for, so the
        // frames the reader passes reaching it are decoded for reference and never converted.
        let mut single_gop = samples(12, 1);
        for sample in single_gop.iter_mut().skip(1) {
            sample.random_access = false;
        }
        let gate = gate(0);
        let frames = open(Some(Arc::clone(&gate)), single_gop);
        let mut source = frames.source();
        let cancellation = CancellationToken::new();
        assert_eq!(
            source
                .get_exact(FrameIndex(9), &cancellation)
                .unwrap_err()
                .kind(),
            ErrorKind::WouldBlock
        );

        grant(&gate, 8);
        assert!(
            wait_for_a_moment(|| source.get_exact(FrameIndex(9), &cancellation).ok()).is_none(),
            "a frame an exact request passed is not drawn"
        );

        grant(&gate, u32::MAX / 2);
        let arrived = wait_for(|| source.get_exact(FrameIndex(9), &cancellation).ok());
        assert_eq!(
            arrived.map(|frame| frame.planes[0].data[0]),
            Some(9),
            "the frame that was asked for is the one drawn"
        );
    }

    #[test]
    fn a_stride_is_what_fits_in_one_publishing_interval() {
        // Nothing measured yet: publish the first picture immediately and measure from it.
        assert_eq!(stride_frames(None), 1);
        // 30 ms a frame fits five of them in the 150 ms interval.
        assert_eq!(stride_frames(Some(Duration::from_millis(30))), 5);
        // A frame slower than the whole interval still moves, one frame at a time.
        assert_eq!(stride_frames(Some(Duration::from_millis(400))), 1);
        // And an unmeasurably fast one still publishes rather than jumping a whole track blind.
        assert_eq!(stride_frames(Some(Duration::from_nanos(1))), MAXIMUM_STRIDE);
        assert_eq!(stride_frames(Some(Duration::ZERO)), MAXIMUM_STRIDE);
    }

    #[test]
    fn a_walk_continues_forwards_and_restarts_backwards() {
        let keyframes = KeyframeIndex::from_samples(&samples(12, 4));
        let rate = Some(Duration::from_millis(30));
        // Ahead of the reader: continue from where it is, one stride at a time.
        assert_eq!(walk_step(Some(2), 11, rate, &keyframes), 7);
        // Never past the target itself.
        assert_eq!(walk_step(Some(2), 5, rate, &keyframes), 5);
        // Already there.
        assert_eq!(walk_step(Some(5), 5, rate, &keyframes), 5);
        // Behind the reader, or a position a cancelled decode left unknown: restart at the
        // random-access point the target decodes from, which is the first frame it can publish.
        assert_eq!(walk_step(Some(11), 6, rate, &keyframes), 4);
        assert_eq!(walk_step(None, 6, rate, &keyframes), 4);
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
}
