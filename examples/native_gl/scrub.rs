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
    CancellationToken, DecodeStatistics, EncodedVideoSample, Error, ErrorKind, ExactFrameReader,
    FrameIndex, Limits, PlaybackVideoSource, Result, VideoDecoderConfig, VideoDecoderFactory,
    VideoFrame,
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
/// frozen for those 1.7 s, which is issue #363. A cadence buys the motion back, and this is what
/// it costs.
///
/// The value is measured rather than derived, and it has been measured twice. #379 swept it and
/// found the cadence unaffordable at any spacing a drag wants: at 150 ms the walk to frame 767
/// converted **765** of the sample's 768 pictures and the frame under the pointer arrived in
/// 7.03 s against 1.25 s with no previews, so the interval was moved out to 1.6 s, which was
/// simply where the sweep stopped paying. That knee was the reader's cache tail and not the
/// cadence: every published picture also converted the frames behind it that fit in the cache,
/// and once the stride was shorter than the tail the tails overlapped the whole span. #402 made
/// that tail the request's rather than the reader's - a walk's intermediate steps ask through
/// [`ExactFrameReader::get_step`], which converts the picture it was asked for and nothing else,
/// while the target itself is still a destination and still keeps its tail - and with it gone
/// there is no knee left to sit behind.
///
/// Re-swept by `examples/scrub_preview_profile.rs` on this project's Apple M1 through
/// VideoToolbox, three sweeps of five, five and seven runs, each cell the fastest of its runs.
/// The baseline that publishes nothing arrives in 1.30-1.32 s across the three:
///
/// | interval | arrival | published | spacing | converted |
/// | --- | --- | --- | --- | --- |
/// | 80 ms | 1.32-1.38 s | 12-16 | 82-115 ms | 24-36 |
/// | 100 ms | 1.31-1.40 s | 10-14 | 97-130 ms | 24-36 |
/// | 150 ms | 1.32-1.48 s | 12-14 | 101-122 ms | 27-51 |
/// | 200 ms | 1.33-1.48 s | 10 | 132-148 ms | 30-46 |
/// | 400 ms | 1.39-1.45 s | 7 | 199-206 ms | 38-43 |
/// | 1600 ms | 1.38-1.40 s | 5 | 277-281 ms | 38-39 |
///
/// The whole band is flat. Every interval from 80 ms to 1.6 s now lands the frame under the
/// pointer within a few percent of a walk that publishes nothing, and the run-to-run spread on
/// one interval is as wide as the spread across all of them - the two sweeps that put 150 ms
/// highest and 120 ms highest are the same host, minutes apart. What used to be a choice between
/// motion and arrival is not a trade any more, because the count the cadence controls is now the
/// count it publishes: a dozen conversions over a 767-frame walk rather than 765.
///
/// So this goes back to the 150 ms #363 chose by arithmetic, which the arithmetic was right
/// about all along and only the tail made unaffordable. It publishes about thirteen pictures on
/// the way, one every 101-122 ms, and arrives in 1.32-1.48 s. #374's background preview index
/// still draws a shrunk picture for wherever the pointer is, so this cadence owes #363 the
/// full-resolution picture catching up several times a second, which is what it now does.
///
/// [`ExactFrameReader::get_step`]: zvidlib::ExactFrameReader::get_step
const PREVIEW_INTERVAL: Duration = Duration::from_millis(150);

/// How many frames a walk decodes between published pictures at `interval`.
///
/// The first step of a walk has nothing measured to go on and so publishes immediately; after
/// that the stride is whatever fits in `interval` at the rate the walk is actually decoding at,
/// which on this sample climbs from one frame to roughly fifty as the estimate settles. A decode
/// slower than the interval still moves a frame at a time rather than stalling.
///
/// The interval is a parameter rather than [`PREVIEW_INTERVAL`] read directly so that
/// `examples/scrub_preview_profile.rs` can sweep it over the same walk this module runs; the
/// window always uses the constant.
fn stride_frames(interval: Duration, per_frame: Option<Duration>) -> u64 {
    let Some(per_frame) = per_frame else {
        return 1;
    };
    let per_frame = per_frame.as_secs_f64();
    if per_frame <= 0.0 {
        return MAXIMUM_STRIDE;
    }
    let frames = (interval.as_secs_f64() / per_frame).floor();
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
    interval: Duration,
    per_frame: Option<Duration>,
    keyframes: &KeyframeIndex,
) -> u64 {
    match position {
        Some(position) if position < target => {
            target.min(position.saturating_add(stride_frames(interval, per_frame)))
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

/// Whether a picture the worker delivered is of the frame the window is trying to show.
///
/// A walk towards a far target publishes the frames it passes, and on a track coded as one group
/// of pictures those are frames of the *start* of the movie while the pointer is at the far end.
/// Drawing them is what made a drag flash frame 0, then 15, then 30, on the way to the frame
/// under the pointer (issue #458), so the window draws a delivered picture only when this says it
/// belongs at the playhead and leaves [`zvidlib::PreviewIndex`]'s shrunk picture - which is of
/// wherever the pointer is - on screen when it does not.
///
/// `tolerance` is how stale a picture may be and still count. A drag retargets on every pointer
/// move, so the frame a decode was started for is a frame or two behind the one the pointer is on
/// by the time it lands; insisting on an exact match would keep the full-resolution picture out of
/// a slow drag entirely, and the frames within the tolerance are of the same moment of the movie.
///
/// The window's alone - `allow` rather than `expect` because the same module compiles into
/// `examples/scrub_preview_profile.rs` too, which draws nothing and so needs none of this.
#[allow(dead_code)]
pub fn frame_is_at_playhead(index: u64, playhead: u64, tolerance: u64) -> bool {
    index.abs_diff(playhead) <= tolerance
}

/// How often a drag moves the audio clock to the frame under the pointer.
///
/// It is the audio queue rather than the drag that this paces. Each
/// [`zvidlib::PlaybackController::seek`] cancels the PCM already scheduled and prerolls afresh, so
/// seeking on every pointer move of a 60 Hz drag would replace each grain about 16 ms after
/// scheduling it and the device would play a stutter of onsets rather than the sound at the
/// playhead. A grain of this length is long enough to be heard before the next seek supersedes it
/// and short enough that the sound still tracks a moving pointer.
#[allow(dead_code)]
const AUDIO_SCRUB_INTERVAL: Duration = Duration::from_millis(100);

/// Paces the seeks a drag makes to keep the audio at the playhead.
#[derive(Default)]
#[allow(dead_code)]
pub struct AudioScrubCadence {
    last: Option<Instant>,
}

#[allow(dead_code)]
impl AudioScrubCadence {
    /// Whether the audio clock should move now, recording that it did.
    ///
    /// The first call after [`Self::rearm`] is always due: a drag that presses the bar and holds
    /// the pointer still has still moved the playhead, and the sound has to go with it.
    pub fn due(&mut self, now: Instant) -> bool {
        if self
            .last
            .is_some_and(|last| now.duration_since(last) < AUDIO_SCRUB_INTERVAL)
        {
            return false;
        }
        self.last = Some(now);
        true
    }

    /// Forgets the last seek, so the next drag scrubs from its first pointer move.
    pub fn rearm(&mut self) {
        self.last = None;
    }
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
    /// How many pictures the worker has published, whether or not the render thread collected
    /// them - a picture dropped from `delivered` still cost its conversion. Counted here rather
    /// than by the caller because `examples/scrub_preview_profile.rs` charges the cadence by it
    /// and a poll can miss one.
    published: u64,
    /// The reader's counters as of the last decode, so that same profile can tell how many
    /// pictures a walk converted rather than how many it published: a published picture also
    /// converts the frames behind it that fit in the reader's cache.
    statistics: DecodeStatistics,
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
    /// Builds a service over its own decoder for the track's samples, publishing a drag's
    /// pictures at [`PREVIEW_INTERVAL`].
    pub fn new(
        factory: &dyn VideoDecoderFactory,
        configuration: VideoDecoderConfig,
        samples: Vec<EncodedVideoSample>,
        limits: Limits,
    ) -> Result<Self> {
        Self::with_preview_interval(factory, configuration, samples, limits, PREVIEW_INTERVAL)
    }

    /// The same service with the publishing cadence named, which is how
    /// `examples/scrub_preview_profile.rs` sweeps it.
    pub fn with_preview_interval(
        factory: &dyn VideoDecoderFactory,
        configuration: VideoDecoderConfig,
        samples: Vec<EncodedVideoSample>,
        limits: Limits,
        preview_interval: Duration,
    ) -> Result<Self> {
        let keyframes = KeyframeIndex::from_samples(&samples);
        let reader = ExactFrameReader::new(factory, configuration, samples, limits)?;
        let queue = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        let worker_queue = Arc::clone(&queue);
        let worker = thread::Builder::new()
            .name("zvidlib-frame-service".to_string())
            .spawn(move || decode_frames(&worker_queue, reader, &keyframes, preview_interval))
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
    ///
    /// `preview` decides what the walk towards it publishes. A preview publishes the pictures it
    /// passes on a time cadence as well as the frame itself, which is what #363 added so a drag
    /// over a long span kept moving; anything else asks for its frame and nothing else, which is
    /// what the window's drag asks for now that it draws no picture that is not at the playhead
    /// (issue #458) and is the baseline `examples/scrub_preview_profile.rs` measures the cadence
    /// against.
    pub fn request(&mut self, frame: u64, preview: bool) -> bool {
        let (lock, condvar) = &*self.queue;
        let retargeted = lock
            .lock()
            .expect("frame queue poisoned")
            .retarget(frame, preview);
        if retargeted {
            condvar.notify_one();
        }
        retargeted
    }

    /// The newest frame the worker has decoded and its presentation index, or `None` rather
    /// than waiting for one.
    ///
    /// The index is what tells a caller whether the picture it just collected is the frame under
    /// the pointer or one the walk passed on the way there. The window draws only the former -
    /// see [`frame_is_at_playhead`] - and `examples/scrub_preview_profile.rs` times the walk by
    /// it.
    pub fn take_latest(&mut self) -> Option<(u64, VideoFrame)> {
        let (lock, _) = &*self.queue;
        let mut state = lock.lock().expect("frame queue poisoned");
        let newest = state.delivered.pop_back();
        state.delivered.clear();
        newest
    }

    /// How many pictures the worker has published, and the reader's counters as of its last
    /// decode. `samples_submitted` less `samples_skipped` is what the walk converted.
    ///
    /// The window draws pictures rather than counting them, so this is
    /// `examples/scrub_preview_profile.rs`'s alone - `allow` rather than `expect` because the
    /// same module compiles into both targets and it is only dead in this one.
    #[allow(dead_code)]
    pub fn published(&self) -> (u64, DecodeStatistics) {
        let (lock, _) = &*self.queue;
        let state = lock.lock().expect("frame queue poisoned");
        (state.published, state.statistics)
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
/// `examples/scrub_preview_profile.rs` times this loop; [`PREVIEW_INTERVAL`] records what it
/// found.
fn decode_frames(
    queue: &Arc<(Mutex<Queue>, Condvar)>,
    mut reader: ExactFrameReader,
    keyframes: &KeyframeIndex,
    preview_interval: Duration,
) {
    let (lock, condvar) = &**queue;
    // Where the last decoded frame left the reader, so a forward step continues from it rather
    // than starting over at a random-access point.
    let mut position: Option<u64> = None;
    // What a frame of this track is costing this walk, measured rather than assumed: the 2.9 ms
    // a decode takes plus whatever share of a published picture's conversion the step carries,
    // and it is what decides how many frames fit in one interval. It is a feedback loop, which
    // is why the sweep in [`PREVIEW_INTERVAL`] is not monotonic: a shorter interval shortens the
    // stride, a shorter stride converts a larger fraction of what it passes, and that lengthens
    // the per-frame estimate the next stride is computed from.
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
                walk_step(position, target, preview_interval, per_frame, keyframes)
            } else {
                target
            };
            lock.lock().expect("frame queue poisoned").cursor = Some(step);
            let started = Instant::now();
            // Decoded outside the lock so a newer target can both replace this one and cancel the
            // decode part-way through. Everything between the reader's position and this frame is
            // decoded inside this one call, for reference only - which is what keeps a stride's
            // skipped frames costing their decoding and no conversion (#355).
            //
            // A step short of the target asks for its own picture and no cache tail: this walk is
            // going forwards and will never come back for the frames behind an intermediate
            // publish, and paying that tail once per published picture is what turned a 150 ms
            // cadence into a conversion of every frame in the track (#402). The target itself is
            // a destination and keeps its tail, so a committed scrub is still followed by free
            // backward steps.
            let decoded = if step == target {
                reader.get(FrameIndex(step), &cancellation)
            } else {
                reader.get_step(FrameIndex(step), &cancellation)
            };
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
                    state.published = state.published.saturating_add(1);
                    state.statistics = reader.statistics();
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
        assert!(frames.request(0, true));
        assert!(frames.take_latest().is_none());
        // An unchanged target is not re-requested, and a changed one redirects the walk.
        assert!(!frames.request(0, true));
        assert!(frames.request(8, true));

        grant(&gate, u32::MAX / 2);

        let drawn = wait_for(|| {
            frames
                .take_latest()
                .map(|(_, frame)| frame.planes[0].data[0])
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
        frames.request(9, true);

        // Enough decoding for the frames before the target, and the picture has already moved.
        grant(&gate, 8);
        let early = wait_for(|| {
            frames
                .take_latest()
                .map(|(_, frame)| frame.planes[0].data[0])
        });
        assert!(
            early.is_some_and(|index| index < 9),
            "a preview draws something on the way to its target, not only the target"
        );

        grant(&gate, u32::MAX / 2);
        let arrived = wait_for(|| {
            frames
                .take_latest()
                .map(|(_, frame)| frame.planes[0].data[0])
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
    fn a_drag_asks_for_the_frame_under_the_pointer_and_publishes_nothing_it_passed() {
        // Issue #458: the window draws no picture that is not at the playhead, so the drag asks
        // for the walk that publishes none of them. One random-access point, as the bundled
        // sample has, so everything before frame 9 is decoded to reach it.
        let mut single_gop = samples(12, 1);
        for sample in single_gop.iter_mut().skip(1) {
            sample.random_access = false;
        }
        let gate = gate(0);
        let mut frames = open(Some(Arc::clone(&gate)), single_gop);
        frames.request(9, false);

        // Enough decoding for the frames before the target, and still nothing published.
        grant(&gate, 8);
        assert!(
            wait_for_a_moment(|| frames.take_latest()).is_none(),
            "a frame the drag's walk passed is published to nobody"
        );

        grant(&gate, u32::MAX / 2);
        let arrived = wait_for(|| {
            frames
                .take_latest()
                .map(|(index, frame)| (index, frame.planes[0].data[0]))
        });
        assert_eq!(
            arrived,
            Some((9, 9)),
            "the frame under the pointer is the one that is published"
        );
    }

    #[test]
    fn only_a_picture_at_the_playhead_is_drawn() {
        // The frames a walk passes on this sample are the start of the movie while the pointer is
        // at the far end, which is what a drag used to flash (issue #458).
        assert!(!frame_is_at_playhead(0, 600, 7));
        assert!(!frame_is_at_playhead(15, 600, 7));
        assert!(!frame_is_at_playhead(30, 600, 7));
        assert!(frame_is_at_playhead(600, 600, 7));
        // A picture the pointer has moved on from since the decode started is of the same moment
        // of the movie and is still drawn, in either direction, up to the tolerance.
        assert!(frame_is_at_playhead(593, 600, 7));
        assert!(frame_is_at_playhead(607, 600, 7));
        assert!(!frame_is_at_playhead(592, 600, 7));
        assert!(!frame_is_at_playhead(608, 600, 7));
    }

    #[test]
    fn a_drag_moves_the_audio_on_a_cadence_rather_than_on_every_pointer_move() {
        let start = Instant::now();
        let mut cadence = AudioScrubCadence::default();

        // The press moves the sound to where the bar was clicked without waiting for a cadence.
        assert!(cadence.due(start));
        // The pointer moves of the next tenth of a second do not, or each seek would discard the
        // PCM the one before it queued and nothing would be heard.
        assert!(!cadence.due(start + Duration::from_millis(16)));
        assert!(!cadence.due(start + Duration::from_millis(99)));
        // Once a grain has played, the clock follows the pointer again.
        assert!(cadence.due(start + AUDIO_SCRUB_INTERVAL));
        assert!(!cadence.due(start + AUDIO_SCRUB_INTERVAL + Duration::from_millis(16)));
        // A committed scrub rearms it, so the next drag scrubs from its first pointer move
        // instead of inheriting this one's pacing.
        cadence.rearm();
        assert!(cadence.due(start + AUDIO_SCRUB_INTERVAL + Duration::from_millis(17)));
    }

    #[test]
    fn a_stride_is_what_fits_in_one_publishing_interval() {
        // Pinned rather than [`PREVIEW_INTERVAL`]: what is under test is that a stride follows
        // whatever interval it is handed, and the shipped one is a measurement that moves.
        const INTERVAL: Duration = Duration::from_millis(150);
        // Nothing measured yet: publish the first picture immediately and measure from it.
        assert_eq!(stride_frames(INTERVAL, None), 1);
        // 30 ms a frame fits five of them in a 150 ms interval.
        assert_eq!(stride_frames(INTERVAL, Some(Duration::from_millis(30))), 5);
        // The interval is the knob issue #379's profile sweeps, so the stride follows it: the
        // same decoding rate covers twice the frames in twice the time.
        assert_eq!(
            stride_frames(Duration::from_millis(300), Some(Duration::from_millis(30))),
            10
        );
        // A frame slower than the whole interval still moves, one frame at a time.
        assert_eq!(stride_frames(INTERVAL, Some(Duration::from_millis(400))), 1);
        // And an unmeasurably fast one still publishes rather than jumping a whole track blind.
        assert_eq!(
            stride_frames(INTERVAL, Some(Duration::from_nanos(1))),
            MAXIMUM_STRIDE
        );
        assert_eq!(
            stride_frames(INTERVAL, Some(Duration::ZERO)),
            MAXIMUM_STRIDE
        );
    }

    #[test]
    fn a_walk_continues_forwards_and_restarts_backwards() {
        let keyframes = KeyframeIndex::from_samples(&samples(12, 4));
        // Five frames a step, as `a_stride_is_what_fits_in_one_publishing_interval` pins.
        const INTERVAL: Duration = Duration::from_millis(150);
        let rate = Some(Duration::from_millis(30));
        // Ahead of the reader: continue from where it is, one stride at a time.
        assert_eq!(walk_step(Some(2), 11, INTERVAL, rate, &keyframes), 7);
        // Never past the target itself.
        assert_eq!(walk_step(Some(2), 5, INTERVAL, rate, &keyframes), 5);
        // Already there.
        assert_eq!(walk_step(Some(5), 5, INTERVAL, rate, &keyframes), 5);
        // Behind the reader, or a position a cancelled decode left unknown: restart at the
        // random-access point the target decodes from, which is the first frame it can publish.
        assert_eq!(walk_step(Some(11), 6, INTERVAL, rate, &keyframes), 4);
        assert_eq!(walk_step(None, 6, INTERVAL, rate, &keyframes), 4);
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
