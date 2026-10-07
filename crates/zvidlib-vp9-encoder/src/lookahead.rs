//! Coding a stream frame by frame, and choosing at the start of each group of
//! pictures whether its key frame keeps detail for the frames after it.
//!
//! A key frame at a lower lambda keeps detail every later frame predicts from,
//! which avoids an inter frame buying it back at several times the key
//! frame's size (issue #597). Whether that pays off depends on how long the
//! content persists: a pan replaces the picture within a few dozen frames,
//! and an inter frame refining a blurry key frame buys the same detail more
//! cheaply than intra coding does, so over a long pan the sharper key frame
//! leaves the group larger and blurrier (issue #618). The encoder cannot tell
//! from the key frame alone, so it codes the frames it has buffered both ways
//! and keeps whichever costs less in the frames' own rate-distortion terms,
//! the weighted key frame only by a margin.

use std::sync::{Mutex, PoisonError};

use crate::context::FrameContext;
use crate::frame::{CodingTools, FrameEncoder, Geometry, ModeInfo, Picture, lambda};

/// How much less, as a fraction of the unweighted coding's rate-distortion
/// cost, the coding with the key frame weighted must cost for the encoder to
/// keep it.
///
/// The two codings sit on different rate-distortion curves, and one lambda
/// weighs them against a single tangent, so a near tie in its terms says
/// little about which codes the group better at equal quality. With the
/// coefficient probabilities fitted to each key frame (issue #622), a 48-frame
/// pan at 96x64 and `base_q_idx` 235 kept a weighted key frame that cost 0.4%
/// less by lambda but left the stream 5% larger at equal quality (issue
/// #626). Groups whose later frames need the key frame's detail, where an
/// inter frame would otherwise buy it back at twice the key frame's size,
/// save 1.5% or more.
const WEIGHTED_KEY_MARGIN: f64 = 0.01;

/// How a stream's frames are coded.
#[derive(Clone, Copy)]
pub(super) struct StreamSettings {
    pub(super) geometry: Geometry,
    pub(super) base_q_idx: u8,
    pub(super) tools: CodingTools,
    pub(super) error_resilient: bool,
    /// The colour range key frames signal.
    pub(super) full_range: bool,
    /// Whether frames choose a loop filter level; cleared only to compare
    /// against an unfiltered encode.
    pub(super) loop_filter: bool,
}

/// One coded frame.
pub(super) struct CodedFrame {
    pub(super) data: Vec<u8>,
    pub(super) key: bool,
    /// The squared error of the visible reconstruction against the source,
    /// over every plane.
    pub(super) distortion: u64,
    #[cfg(test)]
    pub(super) reconstruction: Picture,
}

/// What the next frame codes from.
#[derive(Clone, Default)]
pub(super) struct StreamState {
    /// The previous frame's reconstruction, which the next inter frame
    /// predicts from.
    pub(super) reference: Option<Picture>,
    /// The probabilities the next frame codes with: the decoder's frame
    /// context 0, which every frame that is not error resilient adapts and
    /// saves.
    context: FrameContext,
    /// The previous frame's block modes, whose motion vectors the next inter
    /// frame takes as candidates.
    previous_mode_info: Vec<ModeInfo>,
    previous_was_key: bool,
}

impl StreamState {
    /// Codes `picture`, as a key frame when `key` is set, at a key frame
    /// lambda lowered by [`crate::frame::key_frame_weight`] when `weighted`.
    pub(super) fn code(
        &mut self,
        settings: &StreamSettings,
        picture: &Picture,
        key: bool,
        weighted: bool,
    ) -> CodedFrame {
        let reference = if key { None } else { self.reference.as_ref() };
        // Key frames and error-resilient frames reset every frame context to
        // the defaults (`setup_past_independence`).
        if key || settings.error_resilient {
            self.context = FrameContext::default();
        }
        // Each frame is shown and the same size as the one before, so a frame
        // that is not error resilient uses its motion vectors.
        let previous_mode_info = (!settings.error_resilient && !self.previous_mode_info.is_empty())
            .then_some(self.previous_mode_info.as_slice());
        let mut encoder = FrameEncoder::new(
            settings.geometry,
            picture,
            reference,
            settings.base_q_idx,
            settings.tools,
            settings.error_resilient,
            &self.context,
            previous_mode_info,
        );
        if weighted {
            encoder = encoder.weighted_key_frame();
        }
        if !settings.loop_filter {
            encoder = encoder.without_loop_filter();
        }
        let encoded = encoder.encode(settings.full_range);
        if !settings.error_resilient {
            // refresh_frame_context = 1, frame_parallel_decoding_mode = 0.
            self.context = self.context.adapted(
                &encoded.counts,
                key,
                self.previous_was_key,
                settings.tools.larger_transforms,
                encoded.allow_high_precision_mv,
            );
        }
        let distortion = visible_error(&settings.geometry, &encoded.reconstruction, picture);
        #[cfg(test)]
        let reconstruction = encoded.reconstruction.clone();
        self.reference = Some(encoded.reconstruction);
        self.previous_mode_info = encoded.mode_info;
        self.previous_was_key = key;
        CodedFrame {
            data: encoded.data,
            key,
            distortion,
            #[cfg(test)]
            reconstruction,
        }
    }

    /// Codes `pictures`, the start of a group whose first picture is its key
    /// frame, with the key frame weighted or not, whichever leaves these
    /// frames the lower total rate-distortion cost. The weighted key frame
    /// must win by [`WEIGHTED_KEY_MARGIN`].
    ///
    /// Only the key frame's lambda differs between the two; the frames after
    /// it code as they otherwise would, from whichever key frame is kept.
    ///
    /// The two codings share nothing but the state they start from, so they
    /// run side by side (issue #628): the key frames first, so that a weight
    /// that changes nothing codes the frames after it only once, then the
    /// frames after each key frame. Either coding is exactly what it would be
    /// alone.
    pub(super) fn code_group_start(
        &mut self,
        settings: &StreamSettings,
        pictures: &[Picture],
    ) -> Vec<CodedFrame> {
        // A key frame coded alone has nothing to keep its detail for.
        if pictures.len() < 2 {
            return self.code_frames(settings, pictures, true);
        }
        let mut weighted_state = self.clone();
        let (weighted_key, unweighted_key) = join(
            || weighted_state.code(settings, &pictures[0], true, true),
            || self.code(settings, &pictures[0], true, false),
        );
        let mut unweighted = vec![unweighted_key];
        // Where the weight changes nothing, the frames after the key frame
        // would code the same either way.
        if weighted_key.data == unweighted[0].data {
            unweighted.extend(self.code_frames(settings, &pictures[1..], false));
            return unweighted;
        }
        let mut weighted = vec![weighted_key];
        let (weighted_rest, unweighted_rest) = join(
            || weighted_state.code_frames(settings, &pictures[1..], false),
            || self.code_frames(settings, &pictures[1..], false),
        );
        weighted.extend(weighted_rest);
        unweighted.extend(unweighted_rest);
        let lambda = lambda(settings.base_q_idx);
        let cost = |frames: &[CodedFrame]| {
            frames
                .iter()
                .map(|frame| frame.distortion as f64 + lambda * 8.0 * frame.data.len() as f64)
                .sum::<f64>()
        };
        if cost(&weighted) < cost(&unweighted) * (1.0 - WEIGHTED_KEY_MARGIN) {
            *self = weighted_state;
            weighted
        } else {
            unweighted
        }
    }

    /// Codes `pictures` in order, unweighted, the first as a key frame when
    /// `key_first` is set and every other as an inter frame.
    fn code_frames(
        &mut self,
        settings: &StreamSettings,
        pictures: &[Picture],
        key_first: bool,
    ) -> Vec<CodedFrame> {
        pictures
            .iter()
            .enumerate()
            .map(|(index, picture)| self.code(settings, picture, key_first && index == 0, false))
            .collect()
    }
}

/// Runs `first` and `second`, `first` on a thread of its own where the host
/// has more than one core and the thread starts, and after `second`
/// otherwise.
fn join<A: Send, B>(first: impl FnOnce() -> A + Send, second: impl FnOnce() -> B) -> (A, B) {
    // Taken by the thread; left in place if it does not start.
    let first = Mutex::new(Some(first));
    let (threaded, second) = std::thread::scope(|scope| {
        let task = || {
            let first = first.lock().ok()?.take()?;
            Some(first())
        };
        let handle = std::thread::available_parallelism()
            .is_ok_and(|cores| cores.get() > 1)
            .then(|| std::thread::Builder::new().spawn_scoped(scope, task).ok())
            .flatten();
        let second = second();
        let threaded = handle.and_then(|handle| {
            handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        });
        (threaded, second)
    });
    let first = threaded.unwrap_or_else(|| {
        let first = first
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
            .expect("`first` is left in place when its thread does not run it");
        first()
    });
    (first, second)
}

/// The squared error of `reconstruction` against `source` over the visible
/// part of every plane.
fn visible_error(geometry: &Geometry, reconstruction: &Picture, source: &Picture) -> u64 {
    let mut error = 0_u64;
    for (plane, width, height) in [
        (0, geometry.width, geometry.height),
        (1, geometry.chroma_width(), geometry.chroma_height()),
        (2, geometry.chroma_width(), geometry.chroma_height()),
    ] {
        let stride = reconstruction.strides[plane];
        for row in 0..height {
            let range = row * stride..row * stride + width;
            error += reconstruction.planes[plane][range.clone()]
                .iter()
                .zip(&source.planes[plane][range])
                .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
                .sum::<u64>();
        }
    }
    error
}
