//! A bounded preview tier over a track: one shrunk picture every *N* frames.
//!
//! [`ExactFrameReader`] answers "which frame is this" exactly, and issue #395
//! measured what that costs at an arbitrary point of a track that codes its
//! frames as one long group of pictures: the reader must decode every sample
//! from the nearest random-access point, and on a 1080p HEVC track with one such
//! point the last frame is over a second of hardware decode away. No arrangement
//! of one decoder reaches 50 ms when the frame depends on 767 others.
//!
//! What reaches 50 ms is a picture that was decoded already. A [`PreviewIndex`]
//! decodes the track once, in the background, on a decoder of its own, and keeps
//! every `stride`-th frame shrunk by [`PreviewOptions::scale`]. A drag then
//! draws the nearest kept picture the moment the pointer moves - a lookup and a
//! clone, no decoding at all - while a reader goes after the exact frame
//! underneath it and replaces the preview when it lands. The two answer
//! different questions: this one is "what is at this point of the movie", which
//! a scrub asks continuously and needs immediately, and the reader's is "which
//! frame is the pointer on", which a committed scrub needs exactly and can wait
//! for.
//!
//! It is a *tier*, not a replacement: it costs one decode pass over the track
//! and a bounded amount of memory, it is never frame-exact, and a caller that
//! only ever asks for exact frames should not build one. The pass runs forwards
//! from frame zero, which is the only order that is one pass rather than one
//! walk per preview: each request continues from where the last left the reader.
//! Until it reaches a point, a lookup there falls back to the nearest earlier
//! preview, so the picture is progressively right rather than absent.
//!
//! The pass runs on a thread of its own, so this module is native-only. The
//! browser backend has no equivalent yet; see the crate's issue tracker.

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::api::{Error, ErrorKind, Limits, Result};
use crate::codec::{
    CancellationToken, EncodedVideoSample, ExactFrameReader, VideoDecoderConfig,
    VideoDecoderFactory,
};
use crate::media::{PixelFormat, Plane, VideoDimensions, VideoFrame};
use crate::timeline::FrameIndex;

/// How the index trades memory and pass length against how fine the scrub is.
///
/// Every field has a defensible default and a reason a caller would change it,
/// which is why this is a struct rather than three arguments: a 4K source wants
/// a smaller [`scale`](Self::scale) than a 480p one, a long documentary wants
/// fewer previews per second than a short clip, and an embedded caller wants a
/// smaller [`budget_bytes`](Self::budget_bytes) than a desktop editor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreviewOptions {
    /// The source's frame rate, which is what turns
    /// [`previews_per_second`](Self::previews_per_second) into a frame stride.
    pub frames_per_second: u64,
    /// How far each preview is shrunk on each axis.
    ///
    /// A preview is drawn stretched back over the whole video quad, so this is a
    /// resolution the picture is recognisable at rather than one it is sharp at:
    /// a quarter on each axis is a sixteenth of the memory and reads as the right
    /// shot on a 1080p source. The exact frame replaces it shortly anyway.
    pub scale: u32,
    /// How many previews a second of playback gets, when memory allows it.
    ///
    /// Closer together is a finer scrub but a longer pass - every preview is a
    /// full-resolution picture the reader has to convert before it can be shrunk,
    /// and until the pass reaches the far end of the track a lookup there falls
    /// back to an earlier picture. Half a second apart is about as coarse as a
    /// scrub can be before the picture stops answering "which shot is this".
    pub previews_per_second: u64,
    /// A ceiling on what the whole index may hold.
    ///
    /// The stride follows from this and the track length rather than being fixed,
    /// so a long track keeps fewer, further-apart previews instead of growing
    /// without bound. At a quarter scale a 1080p preview is 480x270x4 bytes, so
    /// the default holds about 129 of them.
    pub budget_bytes: u64,
}

impl PreviewOptions {
    /// The defaults, for a source running at `frames_per_second`.
    pub fn for_frame_rate(frames_per_second: u64) -> Self {
        Self {
            frames_per_second,
            scale: 4,
            previews_per_second: 2,
            budget_bytes: 64 << 20,
        }
    }

    /// How many frames apart the previews are: [`previews_per_second`] of
    /// playback, or further apart when that many would not fit in the budget.
    ///
    /// [`previews_per_second`]: Self::previews_per_second
    pub fn stride(&self, dimensions: &VideoDimensions, frame_count: u64) -> u64 {
        let bytes = self.preview_bytes(dimensions).max(1);
        let affordable = (self.budget_bytes / bytes).max(1);
        let budgeted = frame_count.div_ceil(affordable);
        let wanted = self
            .frames_per_second
            .div_ceil(self.previews_per_second.max(1));
        budgeted.max(wanted).max(1)
    }

    /// What one preview of a `dimensions` source costs in memory.
    pub fn preview_bytes(&self, dimensions: &VideoDimensions) -> u64 {
        let scale = self.scale.max(1);
        let width = u64::from(dimensions.width.div_ceil(scale).max(1));
        let height = u64::from(dimensions.height.div_ceil(scale).max(1));
        width * height * 4
    }
}

/// The previews decoded so far, and how far apart they are.
struct Store {
    stride: u64,
    /// One slot per preview position, filled in as the background pass reaches it.
    slots: Vec<Option<VideoFrame>>,
}

impl Store {
    /// The kept picture closest to `frame`, preferring the one at or before it.
    ///
    /// A lookup ahead of where the pass has reached gets the newest picture
    /// behind it rather than nothing, which is what makes the index useful while
    /// it is still being built.
    fn nearest(&self, frame: u64) -> Option<VideoFrame> {
        if self.slots.is_empty() {
            return None;
        }
        let slot = ((frame / self.stride) as usize).min(self.slots.len() - 1);
        for distance in 0..self.slots.len() {
            if distance <= slot {
                if let Some(preview) = self.slots[slot - distance].as_ref() {
                    return Some(preview.clone());
                }
            }
            if distance > 0 {
                if let Some(preview) = self.slots.get(slot + distance).and_then(Option::as_ref) {
                    return Some(preview.clone());
                }
            }
        }
        None
    }
}

/// A background pass over a track that keeps a shrunk picture every `stride`
/// frames.
///
/// Dropping the index cancels the pass and joins its thread.
pub struct PreviewIndex {
    store: Arc<Mutex<Store>>,
    cancellation: CancellationToken,
    worker: Option<JoinHandle<()>>,
}

impl PreviewIndex {
    /// Starts the pass over `samples`, on a decoder of its own.
    ///
    /// The decoder is separate on purpose: sharing a caller's
    /// [`ExactFrameReader`] would make every preview the index decodes a seek
    /// that reader has to undo, and the two want opposite things from it - the
    /// index walks forwards once and never goes back, a scrub jumps.
    pub fn new(
        factory: &dyn VideoDecoderFactory,
        configuration: VideoDecoderConfig,
        samples: Vec<EncodedVideoSample>,
        limits: Limits,
        options: PreviewOptions,
    ) -> Result<Self> {
        let frame_count = samples.len().max(1) as u64;
        let stride = options.stride(&configuration.coded_dimensions, frame_count);
        let slots = frame_count.div_ceil(stride) as usize;
        let store = Arc::new(Mutex::new(Store {
            stride,
            slots: vec![None; slots],
        }));
        // The pass never asks for a frame twice, so a frame cache only costs it
        // work: the reader converts the last `max_cached_frames` pictures before
        // every target it walks to, and with previews a few frames apart that is
        // every picture in the track converted at full resolution for nobody. One
        // is the smallest the reader accepts.
        let reader = ExactFrameReader::new(
            factory,
            configuration,
            samples,
            Limits {
                max_cached_frames: 1,
                ..limits
            },
        )?;
        let cancellation = CancellationToken::new();
        let worker_store = Arc::clone(&store);
        let worker_cancellation = cancellation.clone();
        let worker = thread::Builder::new()
            .name("zvidlib-preview-index".to_string())
            .spawn(move || {
                build(
                    reader,
                    &worker_store,
                    &worker_cancellation,
                    stride,
                    slots,
                    options,
                );
            })
            .map_err(|error| {
                Error::new(
                    ErrorKind::InvalidInput,
                    format!("could not start the preview index thread: {error}"),
                )
            })?;
        Ok(Self {
            store,
            cancellation,
            worker: Some(worker),
        })
    }

    /// How many of the index's positions the background pass has filled, out of
    /// how many there are.
    ///
    /// A caller that wants the whole track covered before it shows a scrub bar
    /// polls this, or calls [`wait_for_coverage`](Self::wait_for_coverage); one
    /// that is happy to draw a progressively better picture ignores it.
    pub fn coverage(&self) -> (usize, usize) {
        match self.store.lock() {
            Ok(store) => (
                store.slots.iter().filter(|slot| slot.is_some()).count(),
                store.slots.len(),
            ),
            Err(_) => (0, 0),
        }
    }

    /// Blocks until the background pass has visited every position.
    ///
    /// The pass leaves a slot empty when its frame will not decode, so this
    /// returns once nothing more is coming rather than once every slot is full,
    /// and it returns immediately on a cancelled index.
    pub fn wait_for_coverage(&self) {
        while let Some(worker) = self.worker.as_ref() {
            if worker.is_finished() {
                return;
            }
            // The pass is decode-bound and takes seconds, so polling it is a
            // millisecond of latency on a wait measured in thousands of them,
            // against a spin that would take a core away from the decode.
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// The kept picture closest to `frame`, or `None` while the pass has decoded
    /// nothing.
    ///
    /// This is a lock, a search and a clone of an already-decoded picture.
    /// Nothing here decodes, and nothing here waits on the decoder, so it is safe
    /// to call from an event loop.
    pub fn nearest(&self, frame: u64) -> Option<VideoFrame> {
        self.store.lock().ok()?.nearest(frame)
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

/// Walks the track once, forwards, keeping a shrunk copy of every `stride`-th
/// frame.
fn build(
    mut reader: ExactFrameReader,
    store: &Arc<Mutex<Store>>,
    cancellation: &CancellationToken,
    stride: u64,
    slots: usize,
    options: PreviewOptions,
) {
    let limits = Limits::default();
    for slot in 0..slots {
        if cancellation.is_cancelled() {
            return;
        }
        let frame = slot as u64 * stride;
        // A preview that cannot be decoded leaves its slot empty and the pass
        // carries on: the index is an optimisation, and a gap in it costs a
        // fallback to the neighbour rather than an error the caller has to show.
        let Ok(picture) = reader.get(FrameIndex(frame), cancellation) else {
            continue;
        };
        let Ok(preview) = shrink(&picture, options.scale, &limits) else {
            continue;
        };
        if let Ok(mut store) = store.lock() {
            store.slots[slot] = Some(preview);
        }
    }
}

/// Averages `scale` x `scale` blocks of an RGBA frame into one pixel each.
///
/// Averaging rather than dropping pixels is what keeps a preview readable at a
/// quarter scale: point sampling a 1080p frame down to 480x270 aliases hard on
/// exactly the detail - faces, edges, text - that says which shot this is.
pub fn shrink(frame: &VideoFrame, scale: u32, limits: &Limits) -> Result<VideoFrame> {
    if frame.pixel_format != PixelFormat::Rgba8 || frame.planes.len() != 1 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "a preview can only be made from a single-plane RGBA frame",
        ));
    }
    let source = &frame.planes[0];
    let source_width = frame.dimensions.width as usize;
    let source_height = frame.dimensions.height as usize;
    let scale = scale.max(1) as usize;
    let width = source_width.div_ceil(scale).max(1);
    let height = source_height.div_ceil(scale).max(1);
    let stride = width * 4;
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
                    let offset = source_y * source.stride + source_x * 4;
                    for (total, channel) in totals.iter_mut().zip(&source.data[offset..offset + 4])
                    {
                        *total += u32::from(*channel);
                    }
                    count += 1;
                }
            }
            let offset = y * stride + x * 4;
            for (channel, total) in data[offset..offset + 4].iter_mut().zip(totals) {
                *channel = (total / count.max(1)) as u8;
            }
        }
    }
    VideoFrame::new(
        VideoDimensions::new(width as u32, height as u32, limits)?,
        PixelFormat::Rgba8,
        frame.color_range,
        vec![Plane { data, stride }],
        limits,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::ColorRange;

    const SCALE: u32 = 4;

    fn options() -> PreviewOptions {
        PreviewOptions::for_frame_rate(24)
    }

    fn rgba(width: u32, height: u32, fill: impl Fn(u32, u32) -> [u8; 4]) -> VideoFrame {
        let stride = width as usize * 4;
        let mut data = vec![0_u8; stride * height as usize];
        for y in 0..height {
            for x in 0..width {
                let offset = y as usize * stride + x as usize * 4;
                data[offset..offset + 4].copy_from_slice(&fill(x, y));
            }
        }
        VideoFrame::new(
            VideoDimensions::new(width, height, &Limits::default()).unwrap(),
            PixelFormat::Rgba8,
            ColorRange::Limited,
            vec![Plane { data, stride }],
            &Limits::default(),
        )
        .unwrap()
    }

    /// A preview is the source averaged down, not point-sampled: a block that is half black and
    /// half white has to come out grey, or a scrub over fine detail flickers between the two
    /// pixels the sampling happens to land on.
    #[test]
    fn a_preview_averages_the_block_it_replaces() {
        let frame = rgba(SCALE * 2, SCALE, |x, _| {
            if x % 2 == 0 {
                [0, 0, 0, 0]
            } else {
                [255, 255, 255, 255]
            }
        });
        let preview = shrink(&frame, SCALE, &Limits::default()).unwrap();
        assert_eq!(preview.dimensions.width, 2);
        assert_eq!(preview.dimensions.height, 1);
        assert_eq!(&preview.planes[0].data[..4], &[127, 127, 127, 127]);
        assert_eq!(&preview.planes[0].data[4..8], &[127, 127, 127, 127]);
    }

    /// A source whose size is not a multiple of the scale still produces a whole preview, with
    /// its edge pixels averaged over the part-block that exists rather than over zeroes.
    #[test]
    fn a_preview_covers_a_source_the_scale_does_not_divide() {
        let frame = rgba(SCALE + 1, SCALE + 1, |_, _| [10, 20, 30, 40]);
        let preview = shrink(&frame, SCALE, &Limits::default()).unwrap();
        assert_eq!(preview.dimensions.width, 2);
        assert_eq!(preview.dimensions.height, 2);
        for pixel in preview.planes[0].data.chunks_exact(4) {
            assert_eq!(pixel, [10, 20, 30, 40]);
        }
    }

    /// The scale is what the caller asked for, not a constant: a caller with a 4K source and a
    /// small budget picks a bigger one and gets a proportionally smaller picture.
    #[test]
    fn the_scale_is_the_callers() {
        let frame = rgba(16, 16, |_, _| [7, 7, 7, 255]);
        for scale in [1_u32, 2, 4, 8, 16] {
            let preview = shrink(&frame, scale, &Limits::default()).unwrap();
            assert_eq!(preview.dimensions.width, 16 / scale);
            assert_eq!(preview.dimensions.height, 16 / scale);
        }
        // A zero scale would divide by zero rather than mean anything, so it is read as 1.
        assert_eq!(
            shrink(&frame, 0, &Limits::default()).unwrap().dimensions,
            frame.dimensions
        );
    }

    /// The stride is whatever keeps the whole index inside its memory budget, so a longer track
    /// keeps previews further apart rather than more of them.
    #[test]
    fn the_stride_keeps_the_index_inside_its_budget() {
        let dimensions = VideoDimensions::new(1920, 1080, &Limits::default()).unwrap();
        let options = options();
        for frame_count in [1_u64, 768, 100_000, 10_000_000] {
            let stride = options.stride(&dimensions, frame_count);
            let slots = frame_count.div_ceil(stride);
            assert!(
                slots * options.preview_bytes(&dimensions) <= options.budget_bytes,
                "{frame_count} frames at stride {stride} exceeds the budget"
            );
        }
        // A short track is spaced by the frame rate rather than by the budget, and a long one the
        // other way around: the budget only takes over once half a second apart is too many.
        assert_eq!(options.stride(&dimensions, 768), 12);
        assert!(options.stride(&dimensions, 10_000_000) > 12);
    }

    /// A drag ahead of where the pass has reached draws the newest picture behind it, and one
    /// behind a gap draws the nearest picture either side.
    #[test]
    fn the_nearest_preview_falls_back_to_a_neighbour() {
        let picture = |value: u8| rgba(4, 4, move |_, _| [value, value, value, 255]);
        let store = Store {
            stride: 10,
            slots: vec![Some(picture(1)), None, Some(picture(3)), None, None],
        };
        let value = |frame| store.nearest(frame).map(|f| f.planes[0].data[0]);
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
        assert_eq!(
            Store {
                stride: 10,
                slots: vec![None]
            }
            .nearest(0),
            None
        );
    }

    /// A frame that is not single-plane RGBA is not something this can shrink, and says so
    /// rather than reading past the plane it was handed.
    #[test]
    fn a_preview_refuses_a_format_it_cannot_read() {
        let mut frame = rgba(4, 4, |_, _| [0, 0, 0, 0]);
        frame.pixel_format = PixelFormat::Yuv420p8;
        assert!(shrink(&frame, SCALE, &Limits::default()).is_err());
    }
}
