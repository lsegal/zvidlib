//! A scrub preview index for the native GL example.
//!
//! Issue #374 is the part of seeking that no decoder change can reach. The bundled sample codes
//! its 768 frames as a single group of pictures - its `stss` names exactly one sync sample - so
//! the frame four fifths of the way along the bar is 613 samples of reference decoding away from
//! the only place a decode can start. #354 stopped converting those 613 pictures to RGBA and
//! #363 kept the window moving while they decode, but the decoding itself is what is left, and
//! on this host it is 1.09 s of hardware decode for the last frame of the track. There is no
//! arrangement of one decoder that answers "show me frame 767" in under 50 ms from cold: the
//! frames it depends on have to be decoded, and a 1080p HEVC decoder that manages roughly 700
//! frames a second needs about a second to get through them.
//!
//! What can answer in under 50 ms is a picture that was decoded already. This index decodes the
//! track once, in the background, on a decoder of its own, and keeps every `stride`-th frame at a
//! quarter of its linear size. A drag then draws the nearest kept picture the moment the pointer
//! moves - a lookup and a texture upload, no decoding at all - while [`crate::scrub`]'s walk goes
//! after the exact frame underneath it and replaces the preview when it lands. The two answer
//! different questions: this one is "what is at this point of the movie", which a scrub asks
//! continuously and needs immediately, and the walk's is "which frame is the pointer on", which
//! a committed scrub needs exactly and can wait for.
//!
//! It costs one decode pass over the track and a bounded amount of memory. The pass runs forwards
//! from frame zero, which is the only order that is one pass rather than one walk per preview:
//! each request continues from where the last left the reader. Until it reaches a point, a drag
//! there falls back to the nearest earlier preview, so the picture is progressively right rather
//! than absent.

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use zvidlib::{
    CancellationToken, EncodedVideoSample, Error, ErrorKind, ExactFrameReader, FrameIndex, Limits,
    PixelFormat, Plane, Result, VideoDecoderConfig, VideoDecoderFactory, VideoDimensions,
    VideoFrame,
};

/// How far each preview is shrunk on each axis.
///
/// A preview is drawn stretched back over the whole video quad, so this is a resolution the
/// picture is recognisable at rather than one it is sharp at: a quarter on each axis is a
/// sixteenth of the memory and reads as the right shot on a 1080p source. The exact frame
/// replaces it within a second anyway.
const PREVIEW_SCALE: u32 = 4;

/// A ceiling on what the whole index may hold.
///
/// The stride follows from this and the track length rather than being fixed, so a long track
/// keeps fewer, further-apart previews instead of growing without bound. At a quarter scale a
/// 1080p preview is 480x270x4 bytes, so this holds about 129 of them.
const PREVIEW_BUDGET_BYTES: u64 = 64 << 20;

/// The previews decoded so far, and how far apart they are.
struct Store {
    stride: u64,
    /// One slot per preview position, filled in as the background pass reaches it.
    slots: Vec<Option<VideoFrame>>,
}

impl Store {
    /// The kept picture closest to `frame`, preferring the one at or before it.
    ///
    /// A drag ahead of where the pass has reached gets the newest picture behind it rather than
    /// nothing, which is what makes the index useful while it is still being built.
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

/// A background pass over the track that keeps a picture every `stride` frames.
pub struct PreviewIndex {
    store: Arc<Mutex<Store>>,
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
    ) -> Result<Self> {
        let frame_count = samples.len().max(1) as u64;
        let stride = preview_stride(&configuration.coded_dimensions, frame_count);
        let slots = frame_count.div_ceil(stride) as usize;
        let store = Arc::new(Mutex::new(Store {
            stride,
            slots: vec![None; slots],
        }));
        let reader = ExactFrameReader::new(factory, configuration, samples, limits)?;
        let cancellation = CancellationToken::new();
        let worker_store = Arc::clone(&store);
        let worker_cancellation = cancellation.clone();
        let worker = thread::Builder::new()
            .name("zvidlib-preview-index".to_string())
            .spawn(move || {
                build(reader, &worker_store, &worker_cancellation, stride, slots);
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

    /// The kept picture closest to `frame`, or `None` while the pass has decoded nothing.
    ///
    /// This is a lock, a search and a clone of an already-decoded picture. Nothing here decodes,
    /// and nothing here waits on the decoder, so it is safe to call from the event loop.
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

/// How many frames apart the previews are, from the memory they would take.
fn preview_stride(dimensions: &VideoDimensions, frame_count: u64) -> u64 {
    let bytes = preview_bytes(dimensions).max(1);
    let affordable = (PREVIEW_BUDGET_BYTES / bytes).max(1);
    frame_count.div_ceil(affordable).max(1)
}

/// What one preview of a `dimensions` source costs in memory.
fn preview_bytes(dimensions: &VideoDimensions) -> u64 {
    let width = u64::from(dimensions.width.div_ceil(PREVIEW_SCALE).max(1));
    let height = u64::from(dimensions.height.div_ceil(PREVIEW_SCALE).max(1));
    width * height * 4
}

/// Walks the track once, forwards, keeping a shrunk copy of every `stride`-th frame.
fn build(
    mut reader: ExactFrameReader,
    store: &Arc<Mutex<Store>>,
    cancellation: &CancellationToken,
    stride: u64,
    slots: usize,
) {
    let limits = Limits::default();
    for slot in 0..slots {
        if cancellation.is_cancelled() {
            return;
        }
        let frame = slot as u64 * stride;
        // A preview that cannot be decoded leaves its slot empty and the pass carries on: the
        // index is an optimisation, and a gap in it costs a fallback to the neighbour rather
        // than an error the window has to show.
        let Ok(picture) = reader.get(FrameIndex(frame), cancellation) else {
            continue;
        };
        let Ok(preview) = shrink(&picture, &limits) else {
            continue;
        };
        if let Ok(mut store) = store.lock() {
            store.slots[slot] = Some(preview);
        }
    }
}

/// Averages `PREVIEW_SCALE` x `PREVIEW_SCALE` blocks of an RGBA frame into one pixel each.
///
/// Averaging rather than dropping pixels is what keeps a preview readable at a quarter scale:
/// point sampling a 1080p frame down to 480x270 aliases hard on exactly the detail - faces,
/// edges, text - that says which shot this is.
fn shrink(frame: &VideoFrame, limits: &Limits) -> Result<VideoFrame> {
    if frame.pixel_format != PixelFormat::Rgba8 || frame.planes.len() != 1 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "a preview can only be made from a single-plane RGBA frame",
        ));
    }
    let source = &frame.planes[0];
    let source_width = frame.dimensions.width as usize;
    let source_height = frame.dimensions.height as usize;
    let scale = PREVIEW_SCALE as usize;
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
            zvidlib::ColorRange::Limited,
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
        let frame = rgba(PREVIEW_SCALE * 2, PREVIEW_SCALE, |x, _| {
            if x % 2 == 0 {
                [0, 0, 0, 0]
            } else {
                [255, 255, 255, 255]
            }
        });
        let preview = shrink(&frame, &Limits::default()).unwrap();
        assert_eq!(preview.dimensions.width, 2);
        assert_eq!(preview.dimensions.height, 1);
        assert_eq!(&preview.planes[0].data[..4], &[127, 127, 127, 127]);
        assert_eq!(&preview.planes[0].data[4..8], &[127, 127, 127, 127]);
    }

    /// A source whose size is not a multiple of the scale still produces a whole preview, with
    /// its edge pixels averaged over the part-block that exists rather than over zeroes.
    #[test]
    fn a_preview_covers_a_source_the_scale_does_not_divide() {
        let frame = rgba(PREVIEW_SCALE + 1, PREVIEW_SCALE + 1, |_, _| {
            [10, 20, 30, 40]
        });
        let preview = shrink(&frame, &Limits::default()).unwrap();
        assert_eq!(preview.dimensions.width, 2);
        assert_eq!(preview.dimensions.height, 2);
        for pixel in preview.planes[0].data.chunks_exact(4) {
            assert_eq!(pixel, [10, 20, 30, 40]);
        }
    }

    /// The stride is whatever keeps the whole index inside its memory budget, so a longer track
    /// keeps previews further apart rather than more of them.
    #[test]
    fn the_stride_keeps_the_index_inside_its_budget() {
        let dimensions = VideoDimensions::new(1920, 1080, &Limits::default()).unwrap();
        for frame_count in [1_u64, 768, 100_000, 10_000_000] {
            let stride = preview_stride(&dimensions, frame_count);
            let slots = frame_count.div_ceil(stride);
            assert!(
                slots * preview_bytes(&dimensions) <= PREVIEW_BUDGET_BYTES,
                "{frame_count} frames at stride {stride} exceeds the budget"
            );
        }
        // The bundled sample fits well inside it, so its previews stay close together.
        assert!(preview_stride(&dimensions, 768) <= 8);
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
        assert!(shrink(&frame, &Limits::default()).is_err());
    }
}
