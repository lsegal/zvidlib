//! Browser `WebCodecs`-backed compressed video decoding for the `web` target.
//!
//! This bridges the browser's asynchronous `VideoDecoder`/output-callback
//! model to a simple exact-frame `get` used by [`crate::WasmVideoStream`].
//! Because decode is inherently asynchronous here, this does not implement
//! the portable, synchronous [`crate::codec::VideoDecoder`] trait; it is a
//! browser-only bridge kept out of the portable core.

use crate::codec::EncodedVideoSample;
use crate::codec_config::{box_payload, derive_codec_string};
use crate::io::MemorySource;
use crate::media::{Codec, VideoDimensions};
use crate::mp4_demux::{Mp4Demuxer, Mp4DemuxerOptions, Mp4Track};
use crate::timeline::FrameIndex;
use crate::{Error, ErrorKind, Limits, Result};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    EncodedVideoChunk, EncodedVideoChunkInit, EncodedVideoChunkType,
    VideoDecoder as JsVideoDecoder, VideoDecoderConfig as JsVideoDecoderConfig, VideoDecoderInit,
    VideoDecoderSupport, VideoFrame as JsVideoFrame, VideoFrameCopyToOptions, VideoPixelFormat,
};

/// How many samples may sit in the `WebCodecs` decoder's queue at once while
/// `get()` waits for a frame. Deep enough to keep an accelerated decoder busy
/// across an ordinary reorder window, shallow enough that the decoder cannot
/// run so far ahead that the awaited frame is evicted from the frame cache.
const MAX_IN_FLIGHT_CHUNKS: u32 = 16;

/// Resolves `resolve` on the next event-loop turn.
///
/// Reached through `globalThis` so it works in both window and worker scopes
/// without pulling in another `web-sys` feature.
fn schedule_event_loop_tick(resolve: &js_sys::Function) {
    let global = js_sys::global();
    if let Ok(set_timeout) = js_sys::Reflect::get(&global, &JsValue::from_str("setTimeout"))
        && let Ok(set_timeout) = set_timeout.dyn_into::<js_sys::Function>()
    {
        let _ = set_timeout.call2(&global, resolve, &JsValue::from_f64(0.0));
    }
}

fn codec_description(codec: Codec, decoder_config: &[u8]) -> Result<&[u8]> {
    match codec {
        Codec::Hevc => box_payload(decoder_config, b"hvcC"),
        Codec::Av1 => box_payload(decoder_config, b"av1C"),
        Codec::UncompressedVideo | Codec::Aac => Err(Error::new(
            ErrorKind::Unsupported,
            "only HEVC and AV1 have a WebCodecs decoder backend",
        )),
    }
}

fn js_to_promise(value: impl JsCast) -> js_sys::Promise {
    value.unchecked_into()
}

async fn parse_video_track(source: &MemorySource, index: u32, limits: &Limits) -> Result<Mp4Track> {
    let demuxer = Mp4Demuxer::open(
        source,
        Mp4DemuxerOptions {
            limits: *limits,
            ..Mp4DemuxerOptions::default()
        },
    )
    .await?;
    demuxer
        .tracks
        .into_iter()
        .filter(|track| track.kind == crate::codec::TrackKind::Video)
        .nth(index as usize)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "no such video track"))
}

/// Returns each video sample's presentation duration in milliseconds.
///
/// Timing is parsed independently from WebCodecs capability discovery so callers can still pace
/// fallback rendering when the browser cannot decode the track's codec.
pub async fn video_frame_durations_ms(
    bytes: &[u8],
    track_index: u32,
    limits: &Limits,
) -> Result<Vec<f64>> {
    let source = MemorySource::new(bytes.to_vec());
    let track = parse_video_track(&source, track_index, limits).await?;
    let milliseconds_per_tick = 1_000.0 / f64::from(track.timescale);
    track
        .presentation_order
        .iter()
        .map(|&decode_index| {
            track
                .samples
                .get(decode_index)
                .map(|sample| f64::from(sample.duration) * milliseconds_per_tick)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::MalformedMedia,
                        "video presentation index references a missing sample",
                    )
                })
        })
        .collect()
}

/// A lazily-configured `WebCodecs` decode session for one input video track.
pub struct WebVideoDecodeSession {
    samples: Vec<EncodedVideoSample>,
    decode_position_by_presentation: HashMap<FrameIndex, usize>,
    decoder: JsVideoDecoder,
    config: JsVideoDecoderConfig,
    pending_frames: Rc<RefCell<Vec<JsVideoFrame>>>,
    decode_error: Rc<RefCell<Option<String>>>,
    /// Bounded cache of decoded frames the decoder has already emitted,
    /// keyed by presentation index, so sequential `get()` calls within the
    /// same decode session (see `get()`) don't each trigger a fresh
    /// reset-and-redecode pass. Frames are held as live `VideoFrame`
    /// handles and converted to RGBA only when a caller actually asks for
    /// one: a real decoder runs ahead of the requested frame, and
    /// converting every frame it emits costs far more than the decode
    /// itself. Evicted frames are closed rather than dropped.
    cache: HashMap<FrameIndex, JsVideoFrame>,
    cache_order: VecDeque<FrameIndex>,
    /// Position of the next sample not yet submitted to `decoder` in the
    /// current decode session; `None` once a reset is needed. Mirrors
    /// `VideoDecoder::next_decode_position` in the portable backend
    /// (`src/codec.rs`): as long as the requested frame can be reached by
    /// continuing to submit from here, `get()` avoids resetting the
    /// decoder, which is what lets a session walk arbitrarily deep into a
    /// single GOP (see `get()` for why resets are otherwise required).
    next_decode_position: Option<usize>,
    /// Presentation frames the decoder has already emitted since the last
    /// reset. A `WebCodecs` decoder never emits the same presentation frame
    /// twice without an intervening reset, so this is the only condition
    /// that genuinely forces one; see `get()`.
    published_since_reset: HashSet<FrameIndex>,
    /// Resolver for a pending "wait for the next decoder event" `Promise`,
    /// set by `wait_for_output()` and fulfilled by the output/error
    /// closures below.
    waker: Rc<RefCell<Option<js_sys::Function>>>,
    limits: Limits,
    // Kept alive for the lifetime of `decoder`, which retains only the raw
    // `js_sys::Function` handles produced by `as_ref().unchecked_ref()`.
    _output_closure: Closure<dyn FnMut(JsVideoFrame)>,
    _error_closure: Closure<dyn FnMut(JsValue)>,
}

impl WebVideoDecodeSession {
    pub async fn open(bytes: &[u8], track_index: u32, limits: &Limits) -> Result<Self> {
        let source = MemorySource::new(bytes.to_vec());
        let track = parse_video_track(&source, track_index, limits).await?;
        let dimensions = track.dimensions.ok_or_else(|| {
            Error::new(ErrorKind::MalformedMedia, "video track has no dimensions")
        })?;
        let derived = derive_codec_string(track.codec, &track.decoder_config)?;
        let description = codec_description(track.codec, &track.decoder_config)?;

        let config = JsVideoDecoderConfig::new(&derived.codec_string);
        config.set_coded_width(dimensions.width);
        config.set_coded_height(dimensions.height);
        config.set_description_u8_array(&js_sys::Uint8Array::from(description));
        config.set_optimize_for_latency(true);

        let support: VideoDecoderSupport =
            JsFuture::from(js_to_promise(JsVideoDecoder::is_config_supported(&config)))
                .await
                .map_err(|error| normalize_js_error(error, "querying WebCodecs decoder support"))?
                .unchecked_into();
        if !support.get_supported().unwrap_or(false) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!(
                    "this browser cannot decode {} via WebCodecs",
                    derived.codec_string
                ),
            ));
        }

        let pending_frames: Rc<RefCell<Vec<JsVideoFrame>>> = Rc::new(RefCell::new(Vec::new()));
        let decode_error: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
        let waker: Rc<RefCell<Option<js_sys::Function>>> = Rc::new(RefCell::new(None));

        let output_frames = Rc::clone(&pending_frames);
        let output_waker = Rc::clone(&waker);
        let output_closure = Closure::new(move |frame: JsVideoFrame| {
            output_frames.borrow_mut().push(frame);
            if let Some(resolve) = output_waker.borrow_mut().take() {
                let _ = resolve.call0(&JsValue::NULL);
            }
        });
        let error_state = Rc::clone(&decode_error);
        let error_waker = Rc::clone(&waker);
        let error_closure = Closure::new(move |error: JsValue| {
            let message = js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                .ok()
                .and_then(|value| value.as_string())
                .unwrap_or_else(|| "WebCodecs decoder reported an error".to_owned());
            *error_state.borrow_mut() = Some(message);
            if let Some(resolve) = error_waker.borrow_mut().take() {
                let _ = resolve.call0(&JsValue::NULL);
            }
        });

        let init = VideoDecoderInit::new(
            error_closure.as_ref().unchecked_ref(),
            output_closure.as_ref().unchecked_ref(),
        );
        let decoder = JsVideoDecoder::new(&init)
            .map_err(|error| normalize_js_error(error, "constructing a WebCodecs VideoDecoder"))?;
        decoder
            .configure(&config)
            .map_err(|error| normalize_js_error(error, "configuring the WebCodecs VideoDecoder"))?;

        let mut decode_position_by_presentation = HashMap::with_capacity(track.samples.len());
        let samples = track.to_encoded_video_samples(&source, limits).await?;
        for (position, sample) in samples.iter().enumerate() {
            decode_position_by_presentation.insert(sample.presentation_index, position);
        }

        Ok(Self {
            samples,
            decode_position_by_presentation,
            decoder,
            config,
            pending_frames,
            decode_error,
            cache: HashMap::new(),
            cache_order: VecDeque::new(),
            next_decode_position: None,
            published_since_reset: HashSet::new(),
            waker,
            limits: *limits,
            _output_closure: output_closure,
            _error_closure: error_closure,
        })
    }

    fn nearest_random_access(&self, position: usize) -> usize {
        (0..=position)
            .rev()
            .find(|&candidate| self.samples[candidate].random_access)
            .unwrap_or(0)
    }

    /// Decodes and returns exactly the requested presentation frame as RGBA bytes.
    pub async fn get(
        &mut self,
        presentation_index: FrameIndex,
    ) -> Result<(VideoDimensions, Vec<u8>)> {
        if let Some(frame) = self.cache.get(&presentation_index) {
            return self.copy_frame_rgba(frame).await;
        }
        let target_position = *self
            .decode_position_by_presentation
            .get(&presentation_index)
            .ok_or_else(|| {
                Error::new(ErrorKind::InvalidInput, "presentation frame is not indexed")
            })?;
        let random_access_position = self.nearest_random_access(target_position);

        // Chrome's WebCodecs `VideoDecoder` requires the next submitted
        // chunk to be a key frame after every `flush()`, not just after
        // `configure()`. So instead of flushing after every `get()` (which
        // would force a reset-and-redecode from the nearest key frame on
        // every call), a decode session is kept open across calls and reset
        // only when it genuinely cannot produce the requested frame; see
        // `can_continue_session()`. Outputs are awaited via the output
        // callback (see `wait_for_output()`) rather than a `flush()`
        // promise, so a session can walk arbitrarily deep into a single GOP
        // without ever triggering the key-frame-after-flush requirement.
        let can_reuse = can_continue_session(
            self.next_decode_position,
            random_access_position,
            &self.published_since_reset,
            presentation_index,
        );
        if !can_reuse {
            // Only a reset re-decodes the whole span from the key frame, so
            // that is the only case the decode-work limit has to bound up
            // front; a continued session's work is bounded per call below.
            let required_span = target_position - random_access_position + 1;
            if required_span > self.limits.max_decode_samples_per_seek as usize {
                return Err(Error::new(
                    ErrorKind::ResourceLimit,
                    "exact-frame request exceeded the configured decode-work limit",
                ));
            }
            self.decoder.reset().map_err(|error| {
                normalize_js_error(error, "resetting the WebCodecs VideoDecoder")
            })?;
            self.close_pending_frames();
            *self.decode_error.borrow_mut() = None;
            self.published_since_reset.clear();
            self.decoder.configure(&self.config).map_err(|error| {
                normalize_js_error(error, "reconfiguring the WebCodecs VideoDecoder")
            })?;
            self.next_decode_position = Some(random_access_position);
        }

        let mut submitted = 0_u32;
        let mut drained_after_flush = false;
        loop {
            if let Some(message) = self.decode_error.borrow_mut().take() {
                self.next_decode_position = None;
                return Err(Error::new(ErrorKind::Codec, message));
            }
            self.drain_pending_frames();
            if self.cache.contains_key(&presentation_index) {
                let frame = &self.cache[&presentation_index];
                return self.copy_frame_rgba(frame).await;
            }
            if drained_after_flush {
                // Already flushed and drained everything the decoder had
                // to offer, and the target still isn't there: it isn't
                // coming.
                return Err(Error::new(
                    ErrorKind::Internal,
                    "decoder did not output the requested frame",
                ));
            }
            // Real decoder pipelines (particularly hardware-accelerated
            // ones) hold several samples before emitting their first output,
            // so keep the decoder fed rather than blocking on one that is
            // simply waiting for more input. Submission is flow-controlled by
            // the decoder's own queue depth: handing it the whole remaining
            // track at once would emit far more frames than
            // `Limits::max_cached_frames` can retain, evicting the very frame
            // being waited for, and would starve the event loop that delivers
            // the output callbacks in the first place.
            if let Some(position) = self.next_decode_position {
                if position < self.samples.len()
                    && self.decoder.decode_queue_size() < MAX_IN_FLIGHT_CHUNKS
                {
                    if submitted >= self.limits.max_decode_samples_per_seek {
                        return Err(Error::new(
                            ErrorKind::ResourceLimit,
                            "exact-frame request exceeded the configured decode-work limit",
                        ));
                    }
                    self.submit_at(position)?;
                    submitted += 1;
                    continue;
                }
            }
            // Nothing left to prime with. `flush()` forces the decoder to
            // drain, but flushing a large undrained backlog in one shot is
            // itself what caused the original hang this fix replaces, so
            // it's only safe to reach for once the decoder's own queue is
            // empty, i.e. it has already drained everything it can on its
            // own and is genuinely idle rather than merely busy.
            if self.decoder.decode_queue_size() == 0 {
                self.next_decode_position = None;
                JsFuture::from(js_to_promise(self.decoder.flush()))
                    .await
                    .map_err(|error| {
                        normalize_js_error(error, "flushing the WebCodecs VideoDecoder")
                    })?;
                drained_after_flush = true;
                continue;
            }
            self.wait_for_output().await;
        }
    }

    /// Moves every frame the decoder has produced so far out of
    /// `pending_frames` and into the bounded frame cache.
    fn drain_pending_frames(&mut self) {
        let frames = std::mem::take(&mut *self.pending_frames.borrow_mut());
        for frame in frames {
            let index = FrameIndex(frame.timestamp() as u64);
            self.published_since_reset.insert(index);
            self.insert_cache(index, frame);
        }
    }

    /// Closes and discards every frame the decoder emitted but that has not
    /// been drained yet, so no `VideoFrame` is ever dropped unclosed.
    fn close_pending_frames(&self) {
        for frame in std::mem::take(&mut *self.pending_frames.borrow_mut()) {
            frame.close();
        }
    }

    /// Closes every cached frame, leaving the cache empty.
    fn close_cached_frames(&mut self) {
        self.cache_order.clear();
        for (_, frame) in self.cache.drain() {
            frame.close();
        }
    }

    /// Submits the sample at `position` and advances `next_decode_position`,
    /// or invalidates the session (forcing a reset on the next `get()` call)
    /// if submission fails.
    fn submit_at(&mut self, position: usize) -> Result<()> {
        match self.submit(&self.samples[position]) {
            Ok(()) => {
                self.next_decode_position = Some(position + 1);
                Ok(())
            }
            Err(error) => {
                self.next_decode_position = None;
                Err(error)
            }
        }
    }

    /// Awaits the next decoder event (an output frame or a decode error),
    /// via a `Promise` fulfilled by whichever of the output/error closures
    /// fires next, or the next event-loop turn, whichever comes first.
    ///
    /// The event-loop tick matters: the decoder can quietly work its way
    /// through its queue without emitting anything yet, and waiting only on
    /// an output event would then park forever instead of noticing there is
    /// room to submit more.
    async fn wait_for_output(&self) {
        let waker = Rc::clone(&self.waker);
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            schedule_event_loop_tick(&resolve);
            *waker.borrow_mut() = Some(resolve);
        });
        let _ = JsFuture::from(promise).await;
    }

    fn insert_cache(&mut self, index: FrameIndex, frame: JsVideoFrame) {
        // Every `VideoFrame` owns a real platform decode buffer, so replaced
        // and evicted frames must be closed explicitly and deterministically;
        // letting one reach the garbage collector starves the decoder's buffer
        // pool and logs "A VideoFrame was garbage collected without being
        // closed".
        match self.cache.insert(index, frame) {
            Some(replaced) => replaced.close(),
            None => self.cache_order.push_back(index),
        }
        while self.cache.len() > self.limits.max_cached_frames as usize {
            let Some(oldest) = self.cache_order.pop_front() else {
                break;
            };
            if let Some(frame) = self.cache.remove(&oldest) {
                frame.close();
            }
        }
    }

    fn submit(&self, sample: &EncodedVideoSample) -> Result<()> {
        let kind = if sample.random_access {
            EncodedVideoChunkType::Key
        } else {
            EncodedVideoChunkType::Delta
        };
        let data = js_sys::Uint8Array::from(sample.data.as_slice());
        let init = EncodedVideoChunkInit::new_with_u8_array(&data, 0, kind);
        init.set_timestamp_f64(sample.presentation_index.0 as f64);
        let chunk = EncodedVideoChunk::new(&init)
            .map_err(|error| normalize_js_error(error, "constructing an EncodedVideoChunk"))?;
        self.decoder
            .decode(&chunk)
            .map_err(|error| normalize_js_error(error, "decoding a video sample"))
    }

    async fn copy_frame_rgba(&self, frame: &JsVideoFrame) -> Result<(VideoDimensions, Vec<u8>)> {
        // Use the frame's own display dimensions, not the session's track-level
        // dimensions: they can diverge (container padding/cropping), and the
        // pixel buffer below is always laid out to match the frame's actual size.
        let dimensions =
            VideoDimensions::new(frame.display_width(), frame.display_height(), &self.limits)
                .map_err(|error| Error::new(error.kind(), error.message()))?;
        let byte_length = (dimensions.width as usize)
            .checked_mul(dimensions.height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::ResourceLimit,
                    "decoded frame allocation overflow",
                )
            })?;
        let destination = js_sys::Uint8Array::new_with_length(byte_length as u32);
        let options = VideoFrameCopyToOptions::new();
        options.set_format(VideoPixelFormat::Rgba);
        JsFuture::from(js_to_promise(
            frame.copy_to_with_u8_array_and_options(&destination, &options),
        ))
        .await
        .map_err(|error| normalize_js_error(error, "copying a decoded video frame"))?;
        Ok((dimensions, destination.to_vec()))
    }
}

impl Drop for WebVideoDecodeSession {
    fn drop(&mut self) {
        self.close_pending_frames();
        self.close_cached_frames();
        let _ = self.decoder.close();
    }
}

/// Decides whether the requested presentation frame can still be reached by
/// continuing the open decode session instead of resetting the decoder.
///
/// A decoder with output reordering (HEVC hierarchical B-frames, for example)
/// must be fed samples *past* the requested frame's decode position before
/// that frame is emitted, so `next_decode_position` legitimately runs ahead of
/// the target during ordinary sequential presentation-order playback.
/// Requiring `next_decode_position <= target_position` therefore forced a
/// reset-and-redecode from the key frame on almost every call, which made
/// playback O(n^2) in decode work. The only condition that genuinely requires
/// a reset is the frame having already been emitted once since the last reset
/// (and since evicted from the cache), because a `WebCodecs` decoder will not
/// emit the same presentation frame twice without one.
fn can_continue_session(
    next_decode_position: Option<usize>,
    random_access_position: usize,
    published_since_reset: &HashSet<FrameIndex>,
    presentation_index: FrameIndex,
) -> bool {
    next_decode_position.is_some_and(|position| position >= random_access_position)
        && !published_since_reset.contains(&presentation_index)
}

fn normalize_js_error(error: JsValue, context: &str) -> Error {
    let detail = js_sys::Reflect::get(&error, &JsValue::from_str("message"))
        .ok()
        .and_then(|value| value.as_string())
        .or_else(|| error.as_string())
        .unwrap_or_else(|| "WebCodecs operation failed".to_owned());
    Error::new(ErrorKind::Codec, format!("{context}: {detail}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn reordered_output_keeps_the_session_open_past_the_target_position() {
        // Sequential presentation-order playback of hierarchical B-frames
        // leaves `next_decode_position` ahead of the next requested frame's
        // decode position. That must not force a reset, or every displayed
        // frame re-decodes the whole GOP from its key frame.
        let published = HashSet::new();
        assert!(can_continue_session(Some(9), 0, &published, FrameIndex(2)));
        assert!(can_continue_session(
            Some(120),
            0,
            &published,
            FrameIndex(3)
        ));
    }

    #[wasm_bindgen_test]
    fn a_frame_already_emitted_since_the_last_reset_requires_a_reset() {
        // A `WebCodecs` decoder will not emit the same presentation frame
        // twice, so a re-request after cache eviction genuinely needs one.
        let published = HashSet::from([FrameIndex(2)]);
        assert!(!can_continue_session(Some(9), 0, &published, FrameIndex(2)));
        assert!(can_continue_session(Some(9), 0, &published, FrameIndex(3)));
    }

    #[wasm_bindgen_test]
    fn seeking_before_the_open_session_random_access_point_requires_a_reset() {
        let published = HashSet::new();
        assert!(!can_continue_session(
            Some(4),
            30,
            &published,
            FrameIndex(31)
        ));
        assert!(can_continue_session(
            Some(30),
            30,
            &published,
            FrameIndex(31)
        ));
    }

    #[wasm_bindgen_test]
    fn an_invalidated_session_requires_a_reset() {
        let published = HashSet::new();
        assert!(!can_continue_session(None, 0, &published, FrameIndex(0)));
    }
}
