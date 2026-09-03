//! Constant-time seek previews.
//!
//! Reaching an *exact* frame in the middle of a long group of pictures costs a decode of every
//! sample it depends on, and that cost is the track's, not the decoder's: the bundled 1080p
//! sample codes its 768 frames as a single group - its `stss` names one sync sample - so frame
//! 767 is 767 reference decodes away from the only place a decode can start. Issue #395
//! measured that walk at 1.09 s inside the hardware decoder alone on an Apple Silicon host, at
//! roughly 700 1080p HEVC pictures a second. No arrangement of one decoder answers "show me
//! frame 767" inside [`SEEK_LATENCY_BUDGET`] when the frame depends on 767 others.
//!
//! What can answer inside that budget is a picture that was decoded already. A [`SeekPreviews`]
//! index keeps one shrunk picture every `stride` frames, filled in by a [`SeekPreviewPass`] over
//! the track on a decoder of its own, and [`SeekPreviews::nearest`] answers any position on the
//! timeline with a lookup and a clone. It costs no decoding and takes the same time whichever
//! end of the track is asked for, which is what makes a seek constant time rather than
//! proportional to how far it went (`ARCHITECTURE.md` section 3.2).
//!
//! The two tiers answer different questions. This one is "what is at this point of the movie",
//! which a scrub asks continuously and needs immediately; [`ExactFrameReader::get`]'s is "which
//! frame is this exactly", which a committed seek needs precisely and can wait for.
//! [`ExactFrameReader::seek`] is the entry point that reads the fast tier and never decodes.
//!
//! [`ExactFrameReader::get`]: crate::ExactFrameReader::get
//! [`ExactFrameReader::seek`]: crate::ExactFrameReader::seek

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::{
    CancellationToken, EncodedVideoSample, Error, ErrorKind, ExactFrameReader, FrameIndex, Limits,
    PixelFormat, Plane, Result, VideoDecoderConfig, VideoDecoderFactory, VideoDimensions,
    VideoFrame,
};

/// The longest a seek to any position of any track may take.
///
/// This is the requirement `ARCHITECTURE.md` section 3.2 states, kept here as a value so the
/// tests that hold the library to it and the callers that budget against it quote one number.
pub const SEEK_LATENCY_BUDGET: Duration = Duration::from_millis(50);

/// How a [`SeekPreviews`] index trades resolution, spacing, and memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SeekPreviewOptions {
    /// How far each preview is shrunk on each axis.
    ///
    /// A preview is drawn stretched back over the whole video quad, so this is a resolution the
    /// picture is recognisable at rather than one it is sharp at: a quarter on each axis is a
    /// sixteenth of the memory and reads as the right shot on a 1080p source.
    pub scale: u32,
    /// How many previews a second of playback gets, when the budget allows it.
    ///
    /// Closer together is a finer scrub but a longer pass. Half a second apart is about as
    /// coarse as a scrub can be before the picture stops answering "which shot is this".
    pub previews_per_second: u64,
    /// A ceiling on what the whole index may hold.
    ///
    /// The stride follows from this and the track length rather than being fixed, so a long
    /// track keeps fewer, further-apart previews instead of growing without bound.
    pub budget_bytes: u64,
}

impl Default for SeekPreviewOptions {
    fn default() -> Self {
        Self {
            scale: 4,
            previews_per_second: 2,
            // At a quarter scale a 1080p preview is 480x270x4 bytes, so this holds about 129.
            budget_bytes: 64 << 20,
        }
    }
}

/// The previews decoded so far, and how far apart they are.
#[derive(Debug)]
struct Store {
    stride: u64,
    /// One slot per preview position, filled in as the pass reaches it.
    slots: Vec<Option<VideoFrame>>,
}

impl Store {
    /// The kept picture closest to `frame`, preferring the one at or before it.
    ///
    /// A seek ahead of where the pass has reached gets the newest picture behind it rather than
    /// nothing, which is what makes the index useful while it is still being built.
    fn nearest(&self, frame: u64) -> Option<(FrameIndex, VideoFrame)> {
        if self.slots.is_empty() {
            return None;
        }
        let slot = ((frame / self.stride) as usize).min(self.slots.len() - 1);
        for distance in 0..self.slots.len() {
            if distance <= slot {
                if let Some(preview) = self.slots[slot - distance].as_ref() {
                    return Some((self.frame_of(slot - distance), preview.clone()));
                }
            }
            if distance > 0 {
                if let Some(preview) = self.slots.get(slot + distance).and_then(Option::as_ref) {
                    return Some((self.frame_of(slot + distance), preview.clone()));
                }
            }
        }
        None
    }

    fn frame_of(&self, slot: usize) -> FrameIndex {
        FrameIndex(slot as u64 * self.stride)
    }
}

/// A shared, bounded index of already-decoded pictures spread over a track.
///
/// Cloning shares the index rather than copying it: the pass that fills it and the reader that
/// answers seeks from it hold the same one, normally from different threads.
#[derive(Clone, Debug)]
pub struct SeekPreviews(Arc<Mutex<Store>>);

impl SeekPreviews {
    /// An empty index sized for a `frame_count`-frame track of `dimensions` at
    /// `frames_per_second`, which a [`SeekPreviewPass`] then fills.
    pub fn new(
        dimensions: VideoDimensions,
        frame_count: u64,
        frames_per_second: u64,
        options: SeekPreviewOptions,
    ) -> Result<Self> {
        if options.scale == 0 || options.previews_per_second == 0 || options.budget_bytes == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "seek previews need a nonzero scale, cadence, and memory budget",
            ));
        }
        let frame_count = frame_count.max(1);
        let stride = preview_stride(&dimensions, frame_count, frames_per_second.max(1), &options);
        let slots = usize::try_from(frame_count.div_ceil(stride)).map_err(|_| {
            Error::new(
                ErrorKind::ResourceLimit,
                "seek preview index does not fit in memory",
            )
        })?;
        Ok(Self(Arc::new(Mutex::new(Store {
            stride,
            slots: vec![None; slots],
        }))))
    }

    /// How many frames apart the previews are.
    pub fn stride(&self) -> u64 {
        self.store().stride
    }

    /// How many positions the pass has filled, out of how many there are.
    pub fn coverage(&self) -> (usize, usize) {
        let store = self.store();
        (
            store.slots.iter().filter(|slot| slot.is_some()).count(),
            store.slots.len(),
        )
    }

    /// Whether every position has a picture.
    pub fn is_complete(&self) -> bool {
        let (filled, total) = self.coverage();
        filled == total
    }

    /// The kept picture closest to `frame`, or `None` while the pass has decoded nothing.
    ///
    /// This is a lock, a bounded search, and a clone of an already-decoded picture. Nothing here
    /// decodes and nothing here waits on a decoder, so it is safe to call from an event loop and
    /// it takes the same time wherever on the track it is asked about.
    pub fn nearest(&self, frame: FrameIndex) -> Option<VideoFrame> {
        self.nearest_at(frame).map(|(_, picture)| picture)
    }

    /// [`Self::nearest`], with the frame the returned picture is actually of.
    ///
    /// A seek needs to know how far off the picture it drew is, both to label it and to decide
    /// whether the exact walk still has anywhere to go.
    pub fn nearest_at(&self, frame: FrameIndex) -> Option<(FrameIndex, VideoFrame)> {
        self.store().nearest(frame.0)
    }

    /// Records `picture` as the preview for slot `slot`, ignoring a slot past the end.
    fn fill(&self, slot: usize, picture: VideoFrame) {
        let mut store = self.store();
        if let Some(existing) = store.slots.get_mut(slot) {
            *existing = Some(picture);
        }
    }

    /// A poisoned index is a worker that panicked mid-fill, which costs the slot it was filling
    /// and nothing else: every other slot is a whole picture or `None`. Recovering keeps seeks
    /// answerable instead of turning one panicked preview into a dead fast tier.
    fn store(&self) -> MutexGuard<'_, Store> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// A forwards pass over a track that fills a [`SeekPreviews`] index.
///
/// The decoder is its own on purpose: sharing the reader that answers exact requests would make
/// every preview a seek that reader has to undo, and the two want opposite things from it - this
/// pass walks forwards once and never goes back, a scrub jumps.
///
/// The pass is driven by its caller one preview at a time rather than owning a thread, because
/// the browser build has no thread to own. A native caller runs [`Self::run`] on a worker; a
/// `wasm32` caller calls [`Self::step`] from an idle callback.
pub struct SeekPreviewPass {
    reader: ExactFrameReader,
    previews: SeekPreviews,
    scale: u32,
    limits: Limits,
    stride: u64,
    slots: usize,
    next: usize,
}

impl SeekPreviewPass {
    /// Opens a pass over `samples` and the index it fills.
    pub fn new(
        factory: &dyn VideoDecoderFactory,
        configuration: VideoDecoderConfig,
        samples: Vec<EncodedVideoSample>,
        limits: Limits,
        frames_per_second: u64,
        options: SeekPreviewOptions,
    ) -> Result<Self> {
        let dimensions = configuration.coded_dimensions;
        let frame_count = samples.len() as u64;
        let previews = SeekPreviews::new(dimensions, frame_count, frames_per_second, options)?;
        let (stride, slots) = {
            let store = previews.store();
            (store.stride, store.slots.len())
        };
        // The pass never asks for a frame twice, so it asks for every preview as a step rather
        // than a destination: the frames behind each preview would otherwise be converted at
        // full resolution for a request that never comes (issue #402).
        let reader = ExactFrameReader::new(factory, configuration, samples, limits)?;
        Ok(Self {
            reader,
            previews,
            scale: options.scale,
            limits,
            stride,
            slots,
            next: 0,
        })
    }

    /// The index this pass fills, shared with whoever answers seeks from it.
    pub fn previews(&self) -> SeekPreviews {
        self.previews.clone()
    }

    /// How many positions the pass has still to reach.
    pub fn remaining(&self) -> usize {
        self.slots.saturating_sub(self.next)
    }

    /// Decodes the next preview, returning whether there is another one after it.
    ///
    /// A preview that cannot be decoded or shrunk leaves its slot empty and the pass carries on:
    /// the index is an optimisation, and a gap in it costs a seek a fallback to the neighbour
    /// rather than an error the caller has to show. Cancellation is the one failure that stops
    /// the pass, because it means the caller has gone away.
    pub fn step(&mut self, cancellation: &CancellationToken) -> Result<bool> {
        cancellation.check()?;
        if self.next >= self.slots {
            return Ok(false);
        }
        let slot = self.next;
        self.next += 1;
        let frame = FrameIndex(slot as u64 * self.stride);
        match self.reader.get_step(frame, cancellation) {
            Ok(picture) => {
                if let Ok(preview) = shrink_frame(&picture, self.scale, &self.limits) {
                    self.previews.fill(slot, preview);
                }
            }
            Err(error) if error.kind() == ErrorKind::Cancelled => return Err(error),
            Err(_) => {}
        }
        Ok(self.next < self.slots)
    }

    /// Steps until the pass is finished or `cancellation` stops it.
    pub fn run(&mut self, cancellation: &CancellationToken) {
        while matches!(self.step(cancellation), Ok(true)) {}
    }
}

/// How many frames apart the previews are: the requested cadence, or further apart when that
/// many would not fit in the budget.
fn preview_stride(
    dimensions: &VideoDimensions,
    frame_count: u64,
    frames_per_second: u64,
    options: &SeekPreviewOptions,
) -> u64 {
    let bytes = preview_bytes(dimensions, options.scale).max(1);
    let affordable = (options.budget_bytes / bytes).max(1);
    let budgeted = frame_count.div_ceil(affordable);
    let wanted = frames_per_second.div_ceil(options.previews_per_second);
    budgeted.max(wanted).max(1)
}

/// What one preview of a `dimensions` source costs in memory at `scale`.
fn preview_bytes(dimensions: &VideoDimensions, scale: u32) -> u64 {
    let width = u64::from(dimensions.width.div_ceil(scale.max(1)).max(1));
    let height = u64::from(dimensions.height.div_ceil(scale.max(1)).max(1));
    width * height * 4
}

/// Bytes per pixel of a packed single-plane format, or `None` for one this cannot read.
const fn packed_bytes_per_pixel(format: PixelFormat) -> Option<usize> {
    match format {
        PixelFormat::Rgba8 | PixelFormat::Bgra8 => Some(4),
        PixelFormat::Rgb8 => Some(3),
        PixelFormat::Gray8 => Some(1),
        PixelFormat::Yuv420p8 => None,
    }
}

/// Averages `scale` x `scale` blocks of a packed single-plane frame into one pixel each.
///
/// Averaging rather than dropping pixels is what keeps a preview readable at a quarter scale:
/// point sampling a 1080p frame down to 480x270 aliases hard on exactly the detail - faces,
/// edges, text - that says which shot this is.
pub fn shrink_frame(frame: &VideoFrame, scale: u32, limits: &Limits) -> Result<VideoFrame> {
    let channels = match packed_bytes_per_pixel(frame.pixel_format) {
        Some(channels) if frame.planes.len() == 1 => channels,
        _ => {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "a seek preview can only be made from a packed single-plane frame",
            ));
        }
    };
    let scale = usize::try_from(scale.max(1)).unwrap_or(1);
    let source = &frame.planes[0];
    let source_width = frame.dimensions.width as usize;
    let source_height = frame.dimensions.height as usize;
    let width = source_width.div_ceil(scale).max(1);
    let height = source_height.div_ceil(scale).max(1);
    let stride = width * channels;
    let mut data = vec![0_u8; stride * height];
    for y in 0..height {
        for x in 0..width {
            let mut totals = [0_u32; 4];
            let mut count = 0_u32;
            for block_y in 0..scale {
                let source_y = y * scale + block_y;
                if source_y >= source_height {
                    break;
                }
                for block_x in 0..scale {
                    let source_x = x * scale + block_x;
                    if source_x >= source_width {
                        break;
                    }
                    let offset = source_y * source.stride + source_x * channels;
                    for (total, channel) in totals
                        .iter_mut()
                        .zip(&source.data[offset..offset + channels])
                    {
                        *total += u32::from(*channel);
                    }
                    count += 1;
                }
            }
            let offset = y * stride + x * channels;
            for (channel, total) in data[offset..offset + channels].iter_mut().zip(totals) {
                *channel = (total / count.max(1)) as u8;
            }
        }
    }
    VideoFrame::new(
        VideoDimensions::new(width as u32, height as u32, limits)?,
        frame.pixel_format,
        frame.color_range,
        vec![Plane { data, stride }],
        limits,
    )
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::{
        Codec, CodecProfile, ColorRange, HardwarePreference, uncompressed_video_decoder_factory,
    };

    fn limits() -> Limits {
        Limits::default()
    }

    fn packed(width: u32, height: u32, fill: impl Fn(u32, u32) -> [u8; 4]) -> VideoFrame {
        let stride = width as usize * 4;
        let mut data = vec![0_u8; stride * height as usize];
        for y in 0..height {
            for x in 0..width {
                let offset = y as usize * stride + x as usize * 4;
                data[offset..offset + 4].copy_from_slice(&fill(x, y));
            }
        }
        VideoFrame::new(
            VideoDimensions::new(width, height, &limits()).unwrap(),
            PixelFormat::Rgba8,
            ColorRange::Limited,
            vec![Plane { data, stride }],
            &limits(),
        )
        .unwrap()
    }

    /// A single-group-of-pictures track: one random-access point at frame zero, so every frame
    /// after it is reachable only by decoding every frame before it. This is the shape the
    /// bundled sample has and the shape the seek budget is hard for.
    fn single_group_track(frames: u64) -> (VideoDecoderConfig, Vec<EncodedVideoSample>) {
        let configuration = VideoDecoderConfig {
            codec: Codec::UncompressedVideo,
            profile: CodecProfile::UncompressedGray8,
            coded_dimensions: VideoDimensions::new(64, 64, &limits()).unwrap(),
            output_format: PixelFormat::Gray8,
            color_range: ColorRange::Full,
            hardware: HardwarePreference::Avoid,
            configuration: Vec::new(),
        };
        let samples = (0..frames)
            .map(|index| EncodedVideoSample {
                presentation_index: FrameIndex(index),
                random_access: index == 0,
                data: vec![(index % 251) as u8; 64 * 64],
            })
            .collect();
        (configuration, samples)
    }

    fn built_index(frames: u64) -> SeekPreviews {
        let (configuration, samples) = single_group_track(frames);
        let factory = uncompressed_video_decoder_factory();
        let mut pass = SeekPreviewPass::new(
            &factory,
            configuration,
            samples,
            limits(),
            24,
            SeekPreviewOptions::default(),
        )
        .expect("the pass opens over the track");
        let previews = pass.previews();
        pass.run(&CancellationToken::new());
        previews
    }

    /// A preview is the source averaged down, not point-sampled: a block that is half black and
    /// half white has to come out grey, or a scrub over fine detail flickers between the two
    /// pixels the sampling happens to land on.
    #[test]
    fn a_preview_averages_the_block_it_replaces() {
        let frame = packed(8, 4, |x, _| {
            if x % 2 == 0 {
                [0, 0, 0, 0]
            } else {
                [255, 255, 255, 255]
            }
        });
        let preview = shrink_frame(&frame, 4, &limits()).unwrap();
        assert_eq!(preview.dimensions.width, 2);
        assert_eq!(preview.dimensions.height, 1);
        assert_eq!(&preview.planes[0].data[..4], &[127, 127, 127, 127]);
        assert_eq!(&preview.planes[0].data[4..8], &[127, 127, 127, 127]);
    }

    /// A source whose size is not a multiple of the scale still produces a whole preview, with
    /// its edge pixels averaged over the part-block that exists rather than over zeroes.
    #[test]
    fn a_preview_covers_a_source_the_scale_does_not_divide() {
        let frame = packed(5, 5, |_, _| [10, 20, 30, 40]);
        let preview = shrink_frame(&frame, 4, &limits()).unwrap();
        assert_eq!(preview.dimensions.width, 2);
        assert_eq!(preview.dimensions.height, 2);
        for pixel in preview.planes[0].data.chunks_exact(4) {
            assert_eq!(pixel, [10, 20, 30, 40]);
        }
    }

    /// A frame that is not packed and single-plane is not something this can shrink, and says so
    /// rather than reading past the plane it was handed.
    #[test]
    fn a_preview_refuses_a_format_it_cannot_read() {
        let mut frame = packed(4, 4, |_, _| [0, 0, 0, 0]);
        frame.pixel_format = PixelFormat::Yuv420p8;
        assert!(shrink_frame(&frame, 4, &limits()).is_err());
    }

    /// The stride is whatever keeps the whole index inside its memory budget, so a longer track
    /// keeps previews further apart rather than more of them.
    #[test]
    fn the_stride_keeps_the_index_inside_its_budget() {
        let options = SeekPreviewOptions::default();
        let dimensions = VideoDimensions::new(1920, 1080, &limits()).unwrap();
        for frame_count in [1_u64, 768, 100_000, 10_000_000] {
            let stride = preview_stride(&dimensions, frame_count, 24, &options);
            let slots = frame_count.div_ceil(stride);
            assert!(
                slots * preview_bytes(&dimensions, options.scale) <= options.budget_bytes,
                "{frame_count} frames at stride {stride} exceeds the budget"
            );
        }
        // A short track is spaced by the frame rate rather than by the budget, and a long one the
        // other way around: the budget only takes over once half a second apart is too many.
        assert_eq!(preview_stride(&dimensions, 768, 24, &options), 12);
        assert!(preview_stride(&dimensions, 10_000_000, 24, &options) > 12);
    }

    /// A seek ahead of where the pass has reached draws the newest picture behind it, and one
    /// behind a gap draws the nearest picture either side.
    #[test]
    fn the_nearest_preview_falls_back_to_a_neighbour() {
        let picture = |value: u8| packed(4, 4, move |_, _| [value, value, value, 255]);
        let store = Store {
            stride: 10,
            slots: vec![Some(picture(1)), None, Some(picture(3)), None, None],
        };
        let value = |frame| store.nearest(frame).map(|(_, f)| f.planes[0].data[0]);
        assert_eq!(value(0), Some(1));
        assert_eq!(
            value(15),
            Some(1),
            "a gap falls back to the picture behind it"
        );
        assert_eq!(value(25), Some(3));
        assert_eq!(
            value(999),
            Some(3),
            "past the end draws the last one decoded"
        );
        assert_eq!(store.nearest(25).map(|(at, _)| at), Some(FrameIndex(20)));
        assert!(
            Store {
                stride: 10,
                slots: vec![None],
            }
            .nearest(0)
            .is_none()
        );
    }

    /// The pass covers every position of a single-group track, which is the case where nothing
    /// but an already-decoded picture can answer a seek in time.
    #[test]
    fn a_pass_covers_every_position_of_a_single_group_track() {
        let previews = built_index(768);
        assert!(previews.is_complete());
        assert_eq!(previews.coverage().1, 768_usize.div_ceil(12));
        for step in 0..=100_u64 {
            let frame = FrameIndex(step * 767 / 100);
            let (at, picture) = previews
                .nearest_at(frame)
                .expect("every position has a preview");
            assert!(at.0 <= frame.0.next_multiple_of(previews.stride()));
            assert_eq!(picture.pixel_format, PixelFormat::Gray8);
            assert_eq!(picture.dimensions.width, 16);
        }
    }

    /// The requirement issue #416 states: a seek to any position, including the far end of a
    /// track that is one group of pictures, answers inside [`SEEK_LATENCY_BUDGET`], and takes
    /// the same time whichever position it is - the far end is not slower than the near one.
    #[test]
    fn a_seek_to_any_position_answers_inside_the_latency_budget() {
        let previews = built_index(768);
        let mut worst = Duration::ZERO;
        for step in 0..=100_u64 {
            let frame = FrameIndex(step * 767 / 100);
            let started = Instant::now();
            let answer = previews.nearest(frame);
            worst = worst.max(started.elapsed());
            assert!(answer.is_some(), "frame {} has no answer", frame.0);
        }
        assert!(
            worst < SEEK_LATENCY_BUDGET,
            "the worst seek took {worst:?}, over the {SEEK_LATENCY_BUDGET:?} budget"
        );
    }

    /// A cancelled pass stops where it is rather than finishing the track, and what it filled
    /// before that still answers.
    #[test]
    fn a_cancelled_pass_stops_and_leaves_what_it_filled() {
        let (configuration, samples) = single_group_track(120);
        let factory = uncompressed_video_decoder_factory();
        let mut pass = SeekPreviewPass::new(
            &factory,
            configuration,
            samples,
            limits(),
            24,
            SeekPreviewOptions::default(),
        )
        .unwrap();
        let previews = pass.previews();
        let cancellation = CancellationToken::new();
        assert!(pass.step(&cancellation).unwrap());
        cancellation.cancel();
        assert_eq!(
            pass.step(&cancellation).unwrap_err().kind(),
            ErrorKind::Cancelled
        );
        assert_eq!(previews.coverage().0, 1);
        assert!(previews.nearest(FrameIndex(119)).is_some());
        assert!(pass.remaining() > 0);
    }

    /// An index sized for a track it has no picture of yet answers `None` rather than blocking,
    /// so a caller that seeks before the pass has produced anything falls through to its own
    /// exact request instead of stalling on the fast tier.
    #[test]
    fn an_empty_index_answers_nothing_rather_than_waiting() {
        let previews = SeekPreviews::new(
            VideoDimensions::new(64, 64, &limits()).unwrap(),
            768,
            24,
            SeekPreviewOptions::default(),
        )
        .unwrap();
        assert!(!previews.is_complete());
        assert_eq!(previews.nearest(FrameIndex(0)), None);
        assert_eq!(previews.nearest(FrameIndex(767)), None);
    }

    /// Options a track cannot be indexed under are rejected where they are given rather than
    /// dividing by zero inside the stride.
    #[test]
    fn an_index_refuses_options_it_cannot_honour() {
        let dimensions = VideoDimensions::new(64, 64, &limits()).unwrap();
        for options in [
            SeekPreviewOptions {
                scale: 0,
                ..Default::default()
            },
            SeekPreviewOptions {
                previews_per_second: 0,
                ..Default::default()
            },
            SeekPreviewOptions {
                budget_bytes: 0,
                ..Default::default()
            },
        ] {
            assert_eq!(
                SeekPreviews::new(dimensions, 768, 24, options)
                    .unwrap_err()
                    .kind(),
                ErrorKind::InvalidInput
            );
        }
    }
}
