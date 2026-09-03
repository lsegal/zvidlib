//! The browser's [`SeekPreviewSource`]: the preview tier, driven from the event
//! loop instead of from a thread.
//!
//! `ARCHITECTURE.md` section 3.2 binds a seek to
//! [`SEEK_LATENCY_BUDGET`](crate::codec::SEEK_LATENCY_BUDGET) on *any* position
//! of *any* track, and the only way to answer that on a track coded as one long
//! group of pictures is from a picture that was decoded already. On native that
//! is [`crate::previews::PreviewIndex`]. Until issue #432 the browser had no
//! equivalent, so a `wasm32` seek could only ever return `Seek::Exact` from the
//! presentation cache or `Seek::Pending`, and `examples/web_canvas` walked
//! forwards from a random-access point instead - seconds on the bundled sample,
//! whose 768 frames offer exactly one place a decode can start.
//!
//! What was native-only was never the tier, only its driver. The pass is a loop
//! and the native one runs it on a thread; a browser has no thread to give it,
//! but it does have an idle callback. So [`crate::previews`] holds the portable
//! parts - [`PreviewPass`], [`PreviewStore`], the stride derivation and the
//! shrink - and this module is the driver alone: [`WebPreviewIndex::step`]
//! decodes exactly one preview and returns to the event loop, and a caller
//! schedules the next one from `requestIdleCallback` or a
//! `requestAnimationFrame` slice. Nothing here blocks, and nothing here holds
//! the main thread for longer than one preview's decode.
//!
//! The decode session is the index's own, for the same reason the native pass
//! keeps its own decoder: the pass walks forwards once and never goes back,
//! while a scrub jumps, so sharing one session would make every preview a seek
//! the other has to undo.

use crate::api::{ErrorKind, Limits, Result};
use crate::codec::CancellationToken;
use crate::media::{ColorRange, PixelFormat, Plane, VideoDimensions, VideoFrame};
use crate::previews::{PreviewOptions, PreviewPass, PreviewStore};
use crate::timeline::FrameIndex;
use crate::web_decoder::WebVideoDecodeSession;

/// A preview pass over a browser track, advanced one preview at a time.
///
/// This owns the pass and its decode session; what answers seeks is the
/// [`store`](Self::store) handle, which is a [`SeekPreviewSource`] and can be
/// attached to an [`ExactFrameReader`] or read directly.
///
/// [`SeekPreviewSource`]: crate::codec::SeekPreviewSource
/// [`ExactFrameReader`]: crate::codec::ExactFrameReader
pub struct WebPreviewIndex {
    session: WebVideoDecodeSession,
    pass: PreviewPass,
    store: PreviewStore,
    limits: Limits,
}

impl WebPreviewIndex {
    /// Opens a decode session of its own over `track_index` of `bytes` and
    /// prepares the pass. Nothing is decoded until [`step`](Self::step) is
    /// called.
    pub async fn open(
        bytes: &[u8],
        track_index: u32,
        limits: &Limits,
        options: PreviewOptions,
    ) -> Result<Self> {
        let session = WebVideoDecodeSession::open(bytes, track_index, limits).await?;
        let pass = PreviewPass::new(options, &session.dimensions(), session.frame_count());
        let store = pass.store();
        Ok(Self {
            session,
            pass,
            store,
            limits: *limits,
        })
    }

    /// The handle a seek answers from. Cloning it shares the pictures.
    pub fn store(&self) -> PreviewStore {
        self.store.clone()
    }

    /// How many of the pass's positions are filled, out of how many there are.
    pub fn coverage(&self) -> (usize, usize) {
        self.store.coverage()
    }

    /// How many frames apart this index's previews are.
    pub fn stride(&self) -> u64 {
        self.pass.stride()
    }

    /// The frame the next preview is of, or `None` once the pass has visited
    /// every position.
    pub fn next_frame(&self) -> Option<FrameIndex> {
        self.pass.next_frame()
    }

    /// Decodes the next preview and stores it, then returns whether any position
    /// is still unvisited.
    ///
    /// One preview per call is the whole point: the caller comes back through
    /// the event loop between them, so a page stays responsive while the index
    /// fills. A frame that will not decode leaves its slot empty and the pass
    /// carries on, exactly as the native pass does - a gap costs a fallback to
    /// the neighbouring picture, not an error the caller has to show.
    ///
    /// Cancelling is the one failure that does *not* advance the pass: the
    /// position was never visited, so the next call asks for it again rather
    /// than leaving a hole a later lookup would fall through.
    pub async fn step(&mut self, cancellation: &CancellationToken) -> Result<bool> {
        let Some(frame) = self.pass.next_frame() else {
            return Ok(false);
        };
        match self.session.get(frame, cancellation).await {
            Ok((dimensions, rgba)) => match picture(dimensions, rgba, &self.limits) {
                Ok(picture) => self.pass.accept(&picture, &self.limits),
                Err(_) => self.pass.skip(),
            },
            Err(error) if error.kind() == ErrorKind::Cancelled => return Err(error),
            Err(_) => self.pass.skip(),
        }
        Ok(self.pass.next_frame().is_some())
    }
}

/// Wraps what the decode session returned as the RGBA frame the pass shrinks.
fn picture(dimensions: VideoDimensions, rgba: Vec<u8>, limits: &Limits) -> Result<VideoFrame> {
    let stride = dimensions.width as usize * 4;
    VideoFrame::new(
        dimensions,
        PixelFormat::Rgba8,
        ColorRange::Full,
        vec![Plane { data: rgba, stride }],
        limits,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{
        CodecProfile, EncodedVideoSample, ExactFrameReader, HardwarePreference,
        SEEK_LATENCY_BUDGET, Seek, VideoDecoderConfig, uncompressed_video_decoder_factory,
    };
    use crate::media::Codec;
    use std::sync::Arc;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    /// The bundled sample's shape: 768 frames of 1080p at 24 fps, which
    /// `PreviewOptions::for_frame_rate(24)` spaces twelve frames apart.
    const FRAMES: u64 = 768;

    fn dimensions(width: u32, height: u32) -> VideoDimensions {
        VideoDimensions::new(width, height, &Limits::default()).unwrap()
    }

    fn rgba(width: u32, height: u32, value: u8) -> VideoFrame {
        let stride = width as usize * 4;
        picture(
            dimensions(width, height),
            vec![value; stride * height as usize],
            &Limits::default(),
        )
        .unwrap()
    }

    /// A pass filled without a decoder, which is all a seek's contract needs:
    /// the tier's promise is that a seek reads a picture somebody already
    /// decoded, so what put it there is beside the point.
    fn filled_pass(frames: u64) -> PreviewPass {
        let mut pass = PreviewPass::new(
            PreviewOptions::for_frame_rate(24),
            &dimensions(1920, 1080),
            frames,
        );
        let mut value = 0_u8;
        while pass.next_frame().is_some() {
            pass.accept(&rgba(64, 64, value), &Limits::default());
            value = value.wrapping_add(1);
        }
        pass
    }

    /// A one-sample uncompressed reader, so the seek under test has a real
    /// `ExactFrameReader` to be answered by without a browser codec.
    fn reader() -> ExactFrameReader {
        let configuration = VideoDecoderConfig {
            codec: Codec::UncompressedVideo,
            profile: CodecProfile::UncompressedGray8,
            coded_dimensions: dimensions(1, 1),
            output_format: PixelFormat::Gray8,
            color_range: ColorRange::Full,
            hardware: HardwarePreference::Avoid,
            configuration: Vec::new(),
        };
        let samples = (0..FRAMES)
            .map(|index| EncodedVideoSample {
                presentation_index: FrameIndex(index),
                random_access: index == 0,
                data: vec![0],
            })
            .collect();
        ExactFrameReader::new(
            &uncompressed_video_decoder_factory(),
            configuration,
            samples,
            Limits::default(),
        )
        .unwrap()
    }

    /// Issue #432: the browser build has a `SeekPreviewSource` now, so a seek to
    /// a position nothing has decoded exactly is a picture rather than
    /// `Seek::Pending` - and it is still a picture without submitting a single
    /// sample, which is what makes it constant time.
    #[wasm_bindgen_test]
    fn a_browser_seek_is_answered_by_a_preview_rather_than_left_pending() {
        let mut reader = reader();
        assert_eq!(
            reader.seek(FrameIndex(FRAMES - 1)),
            Seek::Pending,
            "with no source attached there is nothing to answer from"
        );

        let pass = filled_pass(FRAMES);
        reader.set_seek_previews(Some(Arc::new(pass.store())));
        match reader.seek(FrameIndex(FRAMES - 1)) {
            Seek::Preview { frame, picture } => {
                assert!(frame.0 <= FRAMES - 1);
                assert!(FRAMES - 1 - frame.0 < pass.stride());
                assert_eq!(picture.pixel_format, PixelFormat::Rgba8);
            }
            other => panic!("a browser seek must be answered by a preview, got {other:?}"),
        }
        assert_eq!(
            reader.statistics().samples_submitted,
            0,
            "a seek answered from the tier decodes nothing"
        );
    }

    /// A lookup ahead of where the pass has reached draws the newest picture
    /// behind it rather than nothing, so a drag over the far end of the bar
    /// moves while the index is still filling.
    #[wasm_bindgen_test]
    fn a_partly_filled_pass_still_answers_the_far_end_of_the_track() {
        let mut pass = PreviewPass::new(
            PreviewOptions::for_frame_rate(24),
            &dimensions(1920, 1080),
            FRAMES,
        );
        assert_eq!(pass.store().nearest(0), None);
        pass.accept(&rgba(64, 64, 7), &Limits::default());
        let (frame, picture) = pass
            .store()
            .nearest_at(FrameIndex(FRAMES - 1))
            .expect("the picture behind the pointer");
        assert_eq!(frame, FrameIndex(0));
        assert_eq!(picture.planes[0].data[0], 7);
    }

    /// Issue #432's measurement, the browser counterpart of
    /// `tests/preview_index.rs::a_preview_answers_any_position_without_decoding`:
    /// every position of the bar, answered from the index, against the budget
    /// `ARCHITECTURE.md` section 3.2 states. The figures this prints are
    /// recorded in `benches/README.md`.
    #[wasm_bindgen_test]
    fn a_browser_preview_answers_any_position_inside_the_seek_budget() {
        let pass = filled_pass(FRAMES);
        let store = pass.store();
        let (filled, total) = store.coverage();
        assert_eq!(filled, total);

        let performance = web_sys::window()
            .and_then(|window| window.performance())
            .expect("performance.now()");
        let mut worst_ms = 0.0_f64;
        for step in 0..=100_u64 {
            let frame = step * (FRAMES - 1) / 100;
            let started = performance.now();
            let preview = store
                .nearest_at(FrameIndex(frame))
                .expect("a preview for every position");
            worst_ms = worst_ms.max(performance.now() - started);
            assert_eq!(preview.1.pixel_format, PixelFormat::Rgba8);
        }
        let budget_ms = SEEK_LATENCY_BUDGET.as_secs_f64() * 1_000.0;
        console_log!(
            "browser preview index over {FRAMES} frames, {total} positions; worst lookup {worst_ms:.3} ms"
        );
        assert!(
            worst_ms < budget_ms,
            "a browser preview lookup took {worst_ms} ms, over the {budget_ms} ms budget"
        );
    }
}
