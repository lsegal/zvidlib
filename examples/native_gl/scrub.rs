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
//!
//!   #363 put the passed pictures back on a `PREVIEW_INTERVAL` cadence so a long drag kept
//!   moving, retuned by #379 and #402, and #462 removed that cadence again. #458 is why: the
//!   window draws no picture that is not at the playhead, so the pictures a cadence published
//!   were converted and then dropped. Restoring them on a track with several random-access
//!   points, where the frames a walk passes *are* near the pointer, is not the answer either -
//!   `benches/README.md` times a cold exact seek on such a track at 18.77 ms through the
//!   hardware decoder and 31.17 ms through the software one, so the walk is over before any
//!   cadence a drag would want could publish anything. The cadence was only ever worth its cost
//!   on a single-group-of-pictures track, which is exactly the track whose passed frames are of
//!   the start of the movie while the pointer is at the end. What covers the pointer while the
//!   frame at the playhead decodes is [`zvidlib::PreviewIndex`]'s shrunk picture (#374).
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

/// The frame a timeline position `fraction` of the way along a `frame_count`-frame track selects.
pub fn target_frame(fraction: f32, frame_count: u64) -> u64 {
    let maximum = frame_count.saturating_sub(1);
    (f64::from(fraction.clamp(0.0, 1.0)) * maximum as f64).round() as u64
}

/// Whether a picture the worker delivered is of the frame the window is trying to show.
///
/// The window draws a delivered picture only when this says it belongs at the playhead, and
/// leaves [`zvidlib::PreviewIndex`]'s shrunk picture - which is of wherever the pointer is - on
/// screen when it does not. Drawing the frames a walk passed is what made a drag flash frame 0,
/// then 15, then 30, on the way to the frame under the pointer (issue #458); the worker no longer
/// publishes them at all (issue #462), and this is still what decides the pictures it does.
///
/// `tolerance` is how stale a picture may be and still count. A drag retargets on every pointer
/// move, so the frame a decode was started for is a frame or two behind the one the pointer is on
/// by the time it lands; insisting on an exact match would keep the full-resolution picture out of
/// a slow drag entirely, and the frames within the tolerance are of the same moment of the movie.
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
const AUDIO_SCRUB_INTERVAL: Duration = Duration::from_millis(100);

/// Paces the seeks a drag makes to keep the audio at the playhead.
#[derive(Default)]
pub struct AudioScrubCadence {
    last: Option<Instant>,
}

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
    /// Builds a service over its own decoder for the track's samples.
    pub fn new(
        factory: &dyn VideoDecoderFactory,
        configuration: VideoDecoderConfig,
        samples: Vec<EncodedVideoSample>,
        limits: Limits,
    ) -> Result<Self> {
        let reader = ExactFrameReader::new(factory, configuration, samples, limits)?;
        let queue = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        let worker_queue = Arc::clone(&queue);
        let worker = thread::Builder::new()
            .name("zvidlib-frame-service".to_string())
            .spawn(move || decode_frames(&worker_queue, reader))
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
    /// The walk towards it publishes that frame and nothing else. The pictures it passes are
    /// decoded for reference only, which is what keeps them costing their decoding and no
    /// conversion (#355) and what leaves the window drawing nothing that is not at the playhead
    /// (issue #458).
    pub fn request(&mut self, frame: u64) -> bool {
        let (lock, condvar) = &*self.queue;
        let retargeted = lock.lock().expect("frame queue poisoned").retarget(frame);
        if retargeted {
            condvar.notify_one();
        }
        retargeted
    }

    /// The newest frame the worker has decoded and its presentation index, or `None` rather
    /// than waiting for one.
    ///
    /// The index is what tells a caller whether the picture it just collected is still the frame
    /// under the pointer or one the pointer has since moved on from - see
    /// [`frame_is_at_playhead`].
    pub fn take_latest(&mut self) -> Option<(u64, VideoFrame)> {
        let (lock, _) = &*self.queue;
        let mut state = lock.lock().expect("frame queue poisoned");
        let newest = state.delivered.pop_back();
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

/// Walks the reader to whatever frame the queue is pointing at, publishing that frame alone.
///
/// Both callers want the same thing: playback wants the frame the clock calls for, and since
/// issue #458 the drag wants the frame under the pointer and draws nothing else. So a walk
/// converts its target and nothing it passes - the frames between the reader and the target are
/// decoded for reference only, which is what #355 bought and what #462 removed the publishing
/// cadence to leave standing.
fn decode_frames(queue: &Arc<(Mutex<Queue>, Condvar)>, mut reader: ExactFrameReader) {
    let (lock, condvar) = &**queue;
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
        loop {
            lock.lock().expect("frame queue poisoned").cursor = Some(target);
            // Decoded outside the lock so a newer target can both replace this one and cancel the
            // decode part-way through. Everything between the reader's position and this frame is
            // decoded inside this one call, for reference only - which is what keeps the frames a
            // walk passes costing their decoding and no conversion (#355). The target is a
            // destination and keeps the reader's cache tail, so a committed scrub is still
            // followed by free backward steps.
            let decoded = reader.get(FrameIndex(target), &cancellation);
            let mut state = lock.lock().expect("frame queue poisoned");
            if state.shutdown {
                return;
            }
            let mut reached = false;
            match decoded {
                Ok(frame) => {
                    reached = true;
                    while state.delivered.len() >= DELIVERY_DEPTH {
                        state.delivered.pop_front();
                    }
                    state.delivered.push_back((target, frame));
                }
                // Superseded part-way through: the reader stopped between frames and reaches the
                // next target from a random-access point of its own accord.
                Err(error) if error.kind() == ErrorKind::Cancelled => {}
                Err(error) => {
                    // Handed to whichever caller asks next; this request cannot make progress.
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
            } else if reached {
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
        frames.request(9);

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
}
