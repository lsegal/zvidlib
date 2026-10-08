//! Browser `WebCodecs`-backed compressed video decoding for the `web` target.
//!
//! This bridges the browser's asynchronous `VideoDecoder`/output-callback
//! model to a simple exact-frame `get` used by [`crate::WasmVideoStream`].
//! Because decode is inherently asynchronous here, this does not implement
//! the portable, synchronous [`crate::codec::VideoDecoder`] trait; it is a
//! browser-only bridge kept out of the portable core.
//!
//! When the browser reports a track's configuration unsupported - HEVC in
//! Chrome, Edge and WebView2 is the common case - the session falls back to
//! the crate's own portable software decoder for that codec instead, driven
//! through the same [`ExactFrameReader`] native callers use (issue #504).
//! `WebCodecs` stays preferred whenever the browser supports the config: it is
//! usually hardware and always faster, while the fallback decodes on the
//! calling thread and is meant for the handful of frames a thumbnail or a
//! preview pass needs rather than real-time playback.

use crate::av1::{Av1CodecConfigurationRecord, Av1Obu, Av1Parser};
use crate::codec::{
    CancellationToken, CodecProfile, EncodedVideoSample, ExactFrameReader, HardwarePreference,
    VideoDecoderConfig, VideoDecoderFactory,
};
use crate::codec_config::derive_codec_string;
use crate::io::MemorySource;
use crate::media::{Codec, ColorRange, PixelFormat, VideoDimensions, VideoFrame};
use crate::mp4_demux::Mp4Track;
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
use zvidlib_core::codec_config::box_payload;

/// How many samples may sit in the `WebCodecs` decoder's queue at once while
/// `get()` waits for a frame. Deep enough to keep an accelerated decoder busy
/// across an ordinary reorder window, shallow enough that the decoder cannot
/// run so far ahead that the awaited frame is evicted from the frame cache.
const MAX_IN_FLIGHT_CHUNKS: u32 = 16;

/// How many decoded `VideoFrame`s the frame cache keeps open at once, whatever
/// `Limits::max_cached_frames` allows.
///
/// Every open `VideoFrame` holds one of the decoder's output buffers, and a
/// decoder whose buffer pool is spent stops emitting frames until one is
/// closed. Chrome's stalls with as few as ten frames held (its hardware HEVC
/// decoder on Windows) or fourteen (its software AV1 decoder), so a cache of 32
/// open frames left `get()` waiting forever for a frame the decoder could not
/// produce, a dozen frames into any track. The bound is on frames at or before
/// the one asked for: frames the decoder emits after it are kept up to
/// `Limits::max_cached_frames`, which is what lets sequential reads continue the
/// decode session instead of resetting it (see `trim_cache()`).
const MAX_OPEN_FRAMES: usize = 4;

/// Resolves `resolve` on the next event-loop turn.
///
/// Reached through `globalThis` so it works in both window and worker scopes
/// without pulling in another `web-sys` feature.
pub(crate) fn schedule_event_loop_tick(resolve: &js_sys::Function) {
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
        // WebCodecs' VP8 registration takes no description.
        Codec::Vp8 => Ok(&[]),
        // Nor does VP9's: its bitstream describes itself.
        Codec::Vp9 => Ok(&[]),
        // Uncompressed video, H.264, the audio codecs, and any codec a later
        // zvidlib-core adds.
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "only HEVC, AV1, VP8 and VP9 have a WebCodecs decoder backend",
        )),
    }
}

pub(crate) fn js_to_promise(value: impl JsCast) -> js_sys::Promise {
    value.unchecked_into()
}

/// Indexes a video track of an MP4 or WebM input, whichever its signature says it is.
async fn parse_video_track(source: &MemorySource, index: u32, limits: &Limits) -> Result<Mp4Track> {
    crate::container::open_tracks(source, limits)
        .await?
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

/// Returns the presentation indices a decode can start from, ascending.
///
/// A scrub that has to move backwards cannot walk there frame by frame - the decoder only runs
/// forwards - so it restarts at the random-access point at or before its target and walks
/// forwards from there, drawing what it passes. Knowing where those points are is what lets the
/// picture follow the pointer during such a drag instead of holding still for the whole decode.
pub async fn video_random_access_points(
    bytes: &[u8],
    track_index: u32,
    limits: &Limits,
) -> Result<Vec<u64>> {
    let source = MemorySource::new(bytes.to_vec());
    let track = parse_video_track(&source, track_index, limits).await?;
    let mut points: Vec<u64> = track
        .presentation_order
        .iter()
        .enumerate()
        .filter_map(|(presentation_index, &decode_index)| {
            let sample = track.samples.get(decode_index)?;
            sample.is_sync.then_some(presentation_index as u64)
        })
        .collect();
    // Presentation order already sorts them, but a track whose samples reorder is exactly the
    // case a caller binary-searches, so the ordering is asserted rather than assumed.
    points.sort_unstable();
    Ok(points)
}

/// An exact-frame decode session for one input video track, on the browser's
/// `WebCodecs` decoder when it supports the track and on the crate's software
/// decoder otherwise.
pub struct WebVideoDecodeSession {
    /// The track's coded size, which a caller sizing a preview budget against it
    /// needs before it has decoded anything (see [`crate::web_previews`]).
    dimensions: VideoDimensions,
    frame_count: u64,
    backend: DecodeBackend,
}

enum DecodeBackend {
    WebCodecs(WebCodecsDecoder),
    Software(SoftwareDecoder),
}

/// Which decoders [`WebVideoDecodeSession::open_with`] may choose between.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BackendChoice {
    /// `WebCodecs` when the browser supports the config, software otherwise.
    Automatic,
    /// Software whatever the browser supports, so a test covers the fallback
    /// even in a browser that happens to decode the codec itself.
    #[cfg(all(test, feature = "all"))]
    SoftwareOnly,
}

impl WebVideoDecodeSession {
    pub async fn open(bytes: &[u8], track_index: u32, limits: &Limits) -> Result<Self> {
        Self::open_with(bytes, track_index, limits, BackendChoice::Automatic).await
    }

    async fn open_with(
        bytes: &[u8],
        track_index: u32,
        limits: &Limits,
        choice: BackendChoice,
    ) -> Result<Self> {
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
        if !description.is_empty() {
            config.set_description_u8_array(&js_sys::Uint8Array::from(description));
        }
        config.set_optimize_for_latency(true);

        let webcodecs_supported = match choice {
            BackendChoice::Automatic => {
                let support: VideoDecoderSupport =
                    JsFuture::from(js_to_promise(JsVideoDecoder::is_config_supported(&config)))
                        .await
                        .map_err(|error| {
                            normalize_js_error(error, "querying WebCodecs decoder support")
                        })?
                        .unchecked_into();
                support.get_supported().unwrap_or(false)
            }
            #[cfg(all(test, feature = "all"))]
            BackendChoice::SoftwareOnly => false,
        };

        let samples = track.to_encoded_video_samples(&source, limits).await?;
        // A decode-only sample, such as a hidden VP8 frame, is not a frame.
        let frame_count = track.presentation_order.len() as u64;
        let backend = if webcodecs_supported {
            DecodeBackend::WebCodecs(WebCodecsDecoder::open(config, samples, limits)?)
        } else {
            let decoder =
                SoftwareDecoder::open(&track, derived.profile, dimensions, samples, limits)
                    .map_err(|error| {
                        Error::new(
                            ErrorKind::Unsupported,
                            format!(
                                "this browser cannot decode {} via WebCodecs, and the software \
                                 decoder cannot either: {}",
                                derived.codec_string,
                                error.message()
                            ),
                        )
                    })?;
            DecodeBackend::Software(decoder)
        };
        Ok(Self {
            dimensions,
            frame_count,
            backend,
        })
    }

    /// The track's coded size.
    pub fn dimensions(&self) -> VideoDimensions {
        self.dimensions
    }

    /// How many presentation frames the track has.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Whether this session decodes in software because `WebCodecs` could not.
    #[cfg(all(test, feature = "all"))]
    fn is_software(&self) -> bool {
        matches!(self.backend, DecodeBackend::Software(_))
    }

    /// Whether this browser decodes `track_index` of `bytes` through
    /// `WebCodecs`, for the tests that exercise that session's own behavior
    /// and have nothing to check where the software fallback takes over.
    #[cfg(all(test, feature = "all"))]
    pub(crate) async fn decodes_through_webcodecs(bytes: &[u8], track_index: u32) -> bool {
        Self::open(bytes, track_index, &Limits::default())
            .await
            .is_ok_and(|session| !session.is_software())
    }

    /// Decodes and returns exactly the requested presentation frame as RGBA bytes.
    ///
    /// A single group of pictures can be hundreds of frames long, so a caller that has moved on -
    /// a timeline scrub whose pointer is already somewhere else - cancels the request rather than
    /// waiting for a frame it will not draw.
    pub async fn get(
        &mut self,
        presentation_index: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<(VideoDimensions, Vec<u8>)> {
        match &mut self.backend {
            DecodeBackend::WebCodecs(decoder) => {
                decoder.get(presentation_index, cancellation).await
            }
            DecodeBackend::Software(decoder) => decoder.get(presentation_index, cancellation).await,
        }
    }
}

/// The crate's own software decoder for `codec`.
fn software_decoder_factory(codec: Codec) -> Result<Box<dyn VideoDecoderFactory>> {
    match codec {
        #[cfg(feature = "hevc-decoder")]
        Codec::Hevc => Ok(Box::new(crate::native_hevc_video_decoder_factory())),
        #[cfg(feature = "av1-decoder")]
        Codec::Av1 => Ok(Box::new(crate::native_av1_video_decoder_factory())),
        #[cfg(feature = "vp8-decoder")]
        Codec::Vp8 => Ok(Box::new(crate::native_vp8_video_decoder_factory())),
        #[cfg(feature = "vp9-decoder")]
        Codec::Vp9 => Ok(Box::new(crate::native_vp9_video_decoder_factory())),
        // Uncompressed video, H.264, the audio codecs, any codec a later
        // zvidlib-core adds, and a decoder whose Cargo feature is off.
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "only HEVC, AV1, VP8 and VP9 have a software decoder backend, each with its \
             <codec>-decoder Cargo feature",
        )),
    }
}

/// The portable software decoder, behind the same exact-frame reader native
/// callers use.
struct SoftwareDecoder {
    reader: ExactFrameReader,
}

impl SoftwareDecoder {
    /// `profile` is the profile the track's configuration box names, so an
    /// HEVC Main 10 track is opened as Main 10 (issue #508) and a VP9 track
    /// as the VP9 profile its `vpcC` declares.
    fn open(
        track: &Mp4Track,
        profile: CodecProfile,
        dimensions: VideoDimensions,
        samples: Vec<EncodedVideoSample>,
        limits: &Limits,
    ) -> Result<Self> {
        let factory = software_decoder_factory(track.codec)?;
        // The HEVC decoder only accepts limited-range input, while the AV1
        // and VP9 decoders report whatever range the stream signals and the
        // reader holds every frame to the configured one.
        let (profile, color_range) = match track.codec {
            Codec::Av1 => (
                CodecProfile::Av1Main,
                av1_color_range(track, &samples, limits),
            ),
            Codec::Vp9 => (profile, vp9_color_range(track, &samples)),
            _ => (profile, ColorRange::Limited),
        };
        // The decoders validate the configuration record itself, so a stream
        // they cannot decode (color AV1, say) is refused here, at open,
        // rather than on its first frame.
        let configuration = VideoDecoderConfig {
            codec: track.codec,
            profile,
            coded_dimensions: dimensions,
            output_format: PixelFormat::Rgba8,
            color_range,
            hardware: HardwarePreference::Avoid,
            configuration: track.decoder_config.clone(),
        };
        let reader = ExactFrameReader::new(factory.as_ref(), configuration, samples, *limits)?;
        Ok(Self { reader })
    }

    async fn get(
        &mut self,
        presentation_index: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<(VideoDimensions, Vec<u8>)> {
        // The decode itself runs to completion on this thread, so an abort can
        // only land between requests. Yielding once first is what lets a scrub
        // that has already moved on cancel a stale request before it starts.
        yield_to_event_loop().await;
        let frame = self.reader.get(presentation_index, cancellation)?;
        Ok((frame.dimensions, packed_rgba(&frame)))
    }
}

/// The color range an AV1 track's sequence header signals: from `av1C`'s
/// `configOBUs` when it carries one, which it need not, and otherwise from the
/// first sample, a key frame that must. Limited when neither parses, which
/// leaves the reader to reject the first frame that disagrees.
fn av1_color_range(
    track: &Mp4Track,
    samples: &[EncodedVideoSample],
    limits: &Limits,
) -> ColorRange {
    let from_config = Av1CodecConfigurationRecord::parse(&track.decoder_config, limits)
        .ok()
        .and_then(|record| {
            record.config_obus.into_iter().find_map(|obu| match obu {
                Av1Obu::SequenceHeader { sequence, .. } => Some(sequence),
                _ => None,
            })
        });
    let sequence = from_config.or_else(|| {
        let mut parser = Av1Parser::new(*limits).ok()?;
        parser.parse_low_overhead(&samples.first()?.data).ok()?;
        parser.sequence
    });
    match sequence {
        Some(sequence) if sequence.color_config.color_range => ColorRange::Full,
        _ => ColorRange::Limited,
    }
}

/// The color range a VP9 track's first key frame signals, which is what
/// its pictures are decoded in. The `vpcC` box's `videoFullRangeFlag` stands
/// in when the first sample does not parse, and limited range when neither
/// says, which leaves the reader to reject the first frame that disagrees.
fn vp9_color_range(track: &Mp4Track, samples: &[EncodedVideoSample]) -> ColorRange {
    let full = samples
        .first()
        .and_then(|sample| zvidlib_vp9_syntax::chunk_full_range(&sample.data))
        .or_else(|| {
            crate::Vp9CodecConfig::parse(&track.decoder_config)
                .ok()
                .map(|config| config.video_full_range)
        });
    if full == Some(true) {
        ColorRange::Full
    } else {
        ColorRange::Limited
    }
}

/// The frame's RGBA rows without stride padding, the layout a `WebCodecs`
/// copy produces and every caller of [`WebVideoDecodeSession::get`] expects.
fn packed_rgba(frame: &VideoFrame) -> Vec<u8> {
    let plane = &frame.planes[0];
    let row = frame.dimensions.width as usize * 4;
    plane
        .data
        .chunks(plane.stride)
        .take(frame.dimensions.height as usize)
        .flat_map(|line| &line[..row])
        .copied()
        .collect()
}

async fn yield_to_event_loop() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| schedule_event_loop_tick(&resolve));
    let _ = JsFuture::from(promise).await;
}

/// A lazily-configured `WebCodecs` decode session for one input video track.
struct WebCodecsDecoder {
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
    /// (`crates/zvidlib-core/src/codec.rs`): as long as the requested frame can be reached by
    /// continuing to submit from here, `get()` avoids resetting the
    /// decoder, which is what lets a session walk arbitrarily deep into a
    /// single GOP (see `get()` for why resets are otherwise required).
    next_decode_position: Option<usize>,
    /// The random-access position the open decode session started from. Only
    /// frames at or after it have been, or can still be, submitted without a
    /// reset. Mirrors `VideoDecoder::session_start` in the portable backend.
    session_start: Option<usize>,
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

impl WebCodecsDecoder {
    fn open(
        config: JsVideoDecoderConfig,
        samples: Vec<EncodedVideoSample>,
        limits: &Limits,
    ) -> Result<Self> {
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

        let mut decode_position_by_presentation = HashMap::with_capacity(samples.len());
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
            session_start: None,
            published_since_reset: HashSet::new(),
            waker,
            limits: *limits,
            _output_closure: output_closure,
            _error_closure: error_closure,
        })
    }

    /// A random-access point presented after the target is skipped even when it comes first in
    /// decode order: the target is one of its leading pictures, which a decode starting there
    /// cannot reconstruct (issue #506).
    fn nearest_random_access(&self, position: usize) -> usize {
        let target = self.samples[position].presentation_index;
        (0..=position)
            .rev()
            .find(|&candidate| {
                let sample = &self.samples[candidate];
                sample.random_access && sample.presentation_index <= target
            })
            .unwrap_or(0)
    }

    /// Decodes and returns exactly the requested presentation frame as RGBA bytes.
    ///
    /// The token is checked on every turn of the decode loop, so canceling stops the decode
    /// part-way through instead of after it.
    async fn get(
        &mut self,
        presentation_index: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<(VideoDimensions, Vec<u8>)> {
        cancellation.check()?;
        // Frames this request has moved past are closed before anything is
        // awaited, so the decoder's output buffers they hold are free for the
        // frames it still has to emit on the way to this one.
        self.trim_cache(presentation_index);
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
            self.session_start,
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
            // Frames the old session ran ahead to can sit after the new
            // target, where `trim_cache()` keeps them, and would hold the
            // output buffers the new session needs to reach it.
            self.close_cached_frames();
            *self.decode_error.borrow_mut() = None;
            self.published_since_reset.clear();
            self.decoder.configure(&self.config).map_err(|error| {
                normalize_js_error(error, "reconfiguring the WebCodecs VideoDecoder")
            })?;
            self.next_decode_position = Some(random_access_position);
            self.session_start = Some(random_access_position);
        }

        let mut submitted = 0_u32;
        let mut drained_after_flush = false;
        loop {
            cancellation.check()?;
            if let Some(message) = self.decode_error.borrow_mut().take() {
                self.next_decode_position = None;
                return Err(Error::new(ErrorKind::Codec, message));
            }
            self.drain_pending_frames(presentation_index);
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
    /// `pending_frames` and into the bounded frame cache, never evicting
    /// `wanted`, the frame the caller is waiting for.
    fn drain_pending_frames(&mut self, wanted: FrameIndex) {
        let frames = std::mem::take(&mut *self.pending_frames.borrow_mut());
        for frame in frames {
            let index = FrameIndex(frame.timestamp() as u64);
            self.published_since_reset.insert(index);
            self.insert_cache(index, frame, wanted);
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

    fn insert_cache(&mut self, index: FrameIndex, frame: JsVideoFrame, wanted: FrameIndex) {
        // Every `VideoFrame` owns a real platform decode buffer, so replaced
        // and evicted frames must be closed explicitly and deterministically;
        // letting one reach the garbage collector starves the decoder's buffer
        // pool and logs "A VideoFrame was garbage collected without being
        // closed".
        match self.cache.insert(index, frame) {
            Some(replaced) => replaced.close(),
            None => self.cache_order.push_back(index),
        }
        self.trim_cache(wanted);
    }

    /// Closes cached frames until at most `MAX_OPEN_FRAMES` of them are
    /// presented at or before `wanted`, and the whole cache fits
    /// `Limits::max_cached_frames`. `wanted` itself always stays.
    ///
    /// Frames after `wanted` are the ones sequential playback asks for next,
    /// and the decoder never emits a frame twice without a reset, so closing
    /// one of them makes the request that reaches it re-decode from the
    /// random-access point: on a track coded as a single group of pictures,
    /// from frame 0. Evicting by age did that to every frame but the newest few
    /// of each batch the decoder ran ahead with while the caller was between
    /// requests, so a page playing the track paid for a decode from the key
    /// frame every other frame (issue #655). They cannot starve the decoder of
    /// output buffers on the way to `wanted` either: it emits in presentation
    /// order, so none of them exists until `wanted` does, and the next request
    /// past them closes them as frames behind it.
    fn trim_cache(&mut self, wanted: FrameIndex) {
        let history = (self.limits.max_cached_frames as usize).clamp(1, MAX_OPEN_FRAMES);
        while self
            .cache_order
            .iter()
            .filter(|&&cached| cached <= wanted)
            .count()
            > history
        {
            let Some(oldest) = self
                .cache_order
                .iter()
                .copied()
                .filter(|&cached| cached < wanted)
                .min()
            else {
                break;
            };
            self.evict(oldest);
        }
        let capacity = (self.limits.max_cached_frames as usize).max(1);
        while self.cache.len() > capacity {
            let Some(furthest) = self
                .cache_order
                .iter()
                .copied()
                .filter(|&cached| cached > wanted)
                .max()
            else {
                break;
            };
            self.evict(furthest);
        }
    }

    fn evict(&mut self, index: FrameIndex) {
        self.cache_order.retain(|&cached| cached != index);
        if let Some(frame) = self.cache.remove(&index) {
            frame.close();
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

impl Drop for WebCodecsDecoder {
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
///
/// A session that started at a later random-access point than the target's
/// never submitted the target at all, however far it has since walked, so it
/// must reset too: after a request for frame 40 of a 32-frame GOP opens the
/// session at 32, a request for frame 5 would otherwise walk on from 41 and
/// never reach it (issue #507).
fn can_continue_session(
    next_decode_position: Option<usize>,
    session_start: Option<usize>,
    random_access_position: usize,
    published_since_reset: &HashSet<FrameIndex>,
    presentation_index: FrameIndex,
) -> bool {
    next_decode_position.is_some_and(|position| position >= random_access_position)
        && session_start.is_some_and(|start| start <= random_access_position)
        && !published_since_reset.contains(&presentation_index)
}

pub(crate) fn normalize_js_error(error: JsValue, context: &str) -> Error {
    let detail = js_sys::Reflect::get(&error, &JsValue::from_str("message"))
        .ok()
        .and_then(|value| value.as_string())
        .or_else(|| error.as_string())
        .unwrap_or_else(|| "WebCodecs operation failed".to_owned());
    Error::new(ErrorKind::Codec, format!("{context}: {detail}"))
}

// The browser tests round-trip media through every native codec, so they
// build with the whole codec matrix.
#[cfg(all(test, feature = "all"))]
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
        assert!(can_continue_session(
            Some(9),
            Some(0),
            0,
            &published,
            FrameIndex(2)
        ));
        assert!(can_continue_session(
            Some(120),
            Some(0),
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
        assert!(!can_continue_session(
            Some(9),
            Some(0),
            0,
            &published,
            FrameIndex(2)
        ));
        assert!(can_continue_session(
            Some(9),
            Some(0),
            0,
            &published,
            FrameIndex(3)
        ));
    }

    #[wasm_bindgen_test]
    fn seeking_before_the_open_session_random_access_point_requires_a_reset() {
        let published = HashSet::new();
        assert!(!can_continue_session(
            Some(4),
            Some(0),
            30,
            &published,
            FrameIndex(31)
        ));
        assert!(can_continue_session(
            Some(30),
            Some(30),
            30,
            &published,
            FrameIndex(31)
        ));
    }

    #[wasm_bindgen_test]
    fn a_request_before_the_open_session_start_requires_a_reset() {
        // Issue #507: frame 40 of a 32-frame GOP opens the session at 32 and
        // walks it on past 40. Frame 5's random-access point is 0, before the
        // session started, so the session never submitted frame 5 however far
        // it has since walked.
        let published = HashSet::new();
        assert!(!can_continue_session(
            Some(41),
            Some(32),
            0,
            &published,
            FrameIndex(5)
        ));
        // A later request in the same GOP the session started from continues.
        assert!(can_continue_session(
            Some(41),
            Some(32),
            32,
            &published,
            FrameIndex(45)
        ));
    }

    #[wasm_bindgen_test]
    fn an_invalidated_session_requires_a_reset() {
        let published = HashSet::new();
        assert!(!can_continue_session(
            None,
            Some(0),
            0,
            &published,
            FrameIndex(0)
        ));
        assert!(!can_continue_session(
            Some(0),
            None,
            0,
            &published,
            FrameIndex(0)
        ));
    }

    const SMALL_HEVC: &[u8] =
        include_bytes!("../crates/zvidlib-hevc-decoder/tests/fixtures/bbb_hevc_512x288_gop32.mp4");

    fn digest(dimensions: VideoDimensions, rgba: Vec<u8>) -> String {
        let limits = Limits::default();
        let stride = dimensions.width as usize * 4;
        let frame = VideoFrame::new(
            dimensions,
            PixelFormat::Rgba8,
            ColorRange::Limited,
            vec![crate::media::Plane { data: rgba, stride }],
            &limits,
        )
        .unwrap();
        crate::conformance::FrameDigest::from_frame(&frame)
            .unwrap()
            .to_hex()
    }

    /// Issue #504: the software fallback decodes an HEVC Main track to the
    /// same pixels the native software decoder does. The digests are that
    /// decoder's output for this fixture on a native build, and frame 20 is
    /// reached by walking forwards from the random-access point at frame 0.
    /// Issue #506: frame 40 is decoded from the CRA at frame 32 past its
    /// leading pictures, and frame 29, one of those leading pictures, from
    /// frame 0.
    #[wasm_bindgen_test(async)]
    async fn software_fallback_decodes_hevc_like_the_native_decoder() {
        let mut session = WebVideoDecodeSession::open_with(
            SMALL_HEVC,
            0,
            &Limits::default(),
            BackendChoice::SoftwareOnly,
        )
        .await
        .unwrap();
        assert!(session.is_software());
        assert_eq!(
            (session.dimensions().width, session.dimensions().height),
            (512, 288)
        );
        assert_eq!(session.frame_count(), 768);
        let expected = [
            (
                0,
                "d405441057a0528fe3a0cda232a667177168004b7153c079080f25aafab078ce",
            ),
            (
                1,
                "2d7594d6df9dcacb1949b9f56b82e28f4c46309ae41a269619dc364d19ea7c75",
            ),
            (
                20,
                "79ac5da3a64bf292dbd490f9c92fe9321610c022dd9be117692b23ba19a281a0",
            ),
            (
                2,
                "c9213b4b6961cef722c723ed0ac1e302b99f0f72162e447191ae4c0e3b03211f",
            ),
            (
                40,
                "dedcc605951fd99f5546e98c7d5243c6e31535e37726f45b2aa7faea025d2e3c",
            ),
            (
                29,
                "50a1d7185ea2a54d245d7e3d4e8c4e182dd2e61e9bfb72b025cf688353bdc55e",
            ),
            (
                32,
                "c8aa7d647b3c001d1c0bb473ff7c9d195481a2490273c37404db5430165312f9",
            ),
        ];
        for (frame, expected) in expected {
            let (dimensions, rgba) = session
                .get(FrameIndex(frame), &CancellationToken::new())
                .await
                .unwrap();
            assert_eq!(rgba.len(), 512 * 288 * 4);
            assert_eq!(digest(dimensions, rgba), expected, "frame {frame}");
        }
    }

    /// Issue #507: a request for frame 40 opens the decode session at the
    /// random-access point at frame 32, and a following request for frame 5
    /// must restart from frame 0 rather than walk on from frame 41. Frame 5
    /// has to match what a fresh session decodes for it. Only a browser that
    /// decodes the track through `WebCodecs` runs that session.
    #[wasm_bindgen_test(async)]
    async fn a_request_before_the_open_session_start_returns_the_requested_frame() {
        let limits = Limits::default();
        let cancellation = CancellationToken::new();
        let mut fresh = WebVideoDecodeSession::open(SMALL_HEVC, 0, &limits)
            .await
            .unwrap();
        if fresh.is_software() {
            return;
        }
        let expected = fresh.get(FrameIndex(5), &cancellation).await.unwrap();

        let mut session = WebVideoDecodeSession::open(SMALL_HEVC, 0, &limits)
            .await
            .unwrap();
        session.get(FrameIndex(40), &cancellation).await.unwrap();
        let (dimensions, rgba) = session.get(FrameIndex(5), &cancellation).await.unwrap();
        assert_eq!(
            (dimensions.width, dimensions.height),
            (expected.0.width, expected.0.height)
        );
        assert!(
            rgba == expected.1,
            "frame 5 after frame 40 differs from a fresh decode"
        );
    }

    /// Issue #508: an HEVC Main 10 track is opened with the profile its `hvcC`
    /// names, so the fallback decodes it instead of refusing it. The digests
    /// are the native conformance fixture's, which were computed from an
    /// independent FFmpeg decode of the same track.
    #[wasm_bindgen_test(async)]
    async fn software_fallback_decodes_hevc_main10() {
        const MAIN10: &[u8] = include_bytes!(
            "../crates/zvidlib-hevc-decoder/tests/fixtures/bbb_hevc_main10_128x72.mp4"
        );
        let expected: Vec<&str> = include_str!(
            "../crates/zvidlib-hevc-decoder/tests/fixtures/bbb_hevc_main10_128x72_rgba.sha256"
        )
        .lines()
        .map(|line| line.split_once(' ').unwrap().1)
        .collect();
        let mut session = WebVideoDecodeSession::open_with(
            MAIN10,
            0,
            &Limits::default(),
            BackendChoice::SoftwareOnly,
        )
        .await
        .unwrap();
        assert!(session.is_software());
        assert_eq!(session.frame_count(), 12);
        for frame in [0_u64, 5, 11, 3] {
            let (dimensions, rgba) = session
                .get(FrameIndex(frame), &CancellationToken::new())
                .await
                .unwrap();
            assert_eq!((dimensions.width, dimensions.height), (128, 72));
            assert_eq!(
                digest(dimensions, rgba),
                expected[frame as usize],
                "frame {frame}"
            );
        }
    }

    /// The fallback is only for what `WebCodecs` cannot do: a browser that
    /// reports the config supported keeps decoding through `WebCodecs`.
    #[wasm_bindgen_test(async)]
    async fn webcodecs_stays_preferred_when_the_browser_supports_the_config() {
        let limits = Limits::default();
        let session = WebVideoDecodeSession::open(SMALL_HEVC, 0, &limits)
            .await
            .unwrap();
        let source = MemorySource::new(SMALL_HEVC.to_vec());
        let track = parse_video_track(&source, 0, &limits).await.unwrap();
        let derived = derive_codec_string(track.codec, &track.decoder_config).unwrap();
        let config = JsVideoDecoderConfig::new(&derived.codec_string);
        config.set_coded_width(512);
        config.set_coded_height(288);
        config.set_description_u8_array(&js_sys::Uint8Array::from(
            codec_description(track.codec, &track.decoder_config).unwrap(),
        ));
        config.set_optimize_for_latency(true);
        let support: VideoDecoderSupport =
            JsFuture::from(js_to_promise(JsVideoDecoder::is_config_supported(&config)))
                .await
                .unwrap()
                .unchecked_into();
        assert_eq!(
            session.is_software(),
            !support.get_supported().unwrap_or(false)
        );
    }

    /// A lossless monochrome AV1 track decodes through the fallback back to
    /// the exact gray levels it was encoded from.
    #[wasm_bindgen_test(async)]
    async fn software_fallback_decodes_av1_back_to_its_source() {
        use crate::codec::{VideoEncoderConfig, VideoEncoderFactory};
        use crate::io::MemorySink;
        use crate::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
        use crate::transfer::{CpuFrameSource, FrameSource, Orientation};

        let limits = Limits::default();
        let dimensions = VideoDimensions::new(32, 18, &limits).unwrap();
        let gray = |index: u64| -> Vec<u8> {
            (0..dimensions.height)
                .flat_map(|y| {
                    (0..dimensions.width)
                        .map(move |x| ((x * 7 + y * 3 + index as u32 * 5) % 256) as u8)
                })
                .collect()
        };
        let mut encoder = crate::native_av1_video_encoder_factory()
            .create(
                &VideoEncoderConfig {
                    codec: Codec::Av1,
                    profile: CodecProfile::Av1Main,
                    coded_dimensions: dimensions,
                    input_format: PixelFormat::Gray8,
                    color_range: ColorRange::Full,
                    hardware: HardwarePreference::Avoid,
                    timescale: 30,
                    frame_duration: 1,
                    configuration: Vec::new(),
                },
                &limits,
            )
            .unwrap();
        let mut muxer = Mp4Muxer::new(
            MemorySink::new(),
            vec![Mp4TrackConfig {
                encoder: encoder.config().clone(),
                format: Mp4TrackFormat::Video(dimensions),
            }],
            60,
        )
        .await
        .unwrap();
        for index in 0..3_u64 {
            let frame = VideoFrame::new(
                dimensions,
                PixelFormat::Gray8,
                ColorRange::Full,
                vec![crate::media::Plane {
                    data: gray(index),
                    stride: dimensions.width as usize,
                }],
                &limits,
            )
            .unwrap();
            let samples = encoder
                .encode(
                    FrameIndex(index),
                    FrameSource::Cpu(CpuFrameSource {
                        frame: &frame,
                        orientation: Orientation::TopLeft,
                    }),
                )
                .await
                .unwrap();
            for sample in samples {
                muxer.write_sample(0, sample).await.unwrap();
            }
        }
        for sample in encoder.finish().await.unwrap() {
            muxer.write_sample(0, sample).await.unwrap();
        }
        let bytes = muxer.finish().await.unwrap().into_inner();

        let mut session =
            WebVideoDecodeSession::open_with(&bytes, 0, &limits, BackendChoice::SoftwareOnly)
                .await
                .unwrap();
        assert!(session.is_software());
        for index in [2_u64, 0] {
            let (decoded, rgba) = session
                .get(FrameIndex(index), &CancellationToken::new())
                .await
                .unwrap();
            assert_eq!(decoded, dimensions);
            let expected: Vec<u8> = gray(index)
                .into_iter()
                .flat_map(|level| [level, level, level, 255])
                .collect();
            assert_eq!(rgba, expected, "frame {index}");
        }
    }

    /// Issue #528: the native VP9 encoder's MP4 output decodes through the
    /// browser's own WebCodecs VP9 decoder, key frames and inter frames alike,
    /// including a backwards seek across a key frame.
    #[wasm_bindgen_test(async)]
    async fn webcodecs_decodes_native_vp9_output() {
        use crate::codec::{VideoEncoderConfig, VideoEncoderFactory};
        use crate::io::MemorySink;
        use crate::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
        use crate::transfer::{CpuFrameSource, FrameSource, Orientation};

        let limits = Limits::default();
        let dimensions = VideoDimensions::new(64, 36, &limits).unwrap();
        let rgba = |index: u64| -> Vec<u8> {
            (0..dimensions.height)
                .flat_map(|y| {
                    (0..dimensions.width).flat_map(move |x| {
                        let u = x + index as u32 * 2;
                        [(u * 3) as u8, (y * 6) as u8, 200 - (u * 2) as u8, 255]
                    })
                })
                .collect()
        };
        let mut encoder = crate::native_vp9_video_encoder_factory()
            .create(
                &VideoEncoderConfig {
                    codec: Codec::Vp9,
                    profile: CodecProfile::Vp9Profile0,
                    coded_dimensions: dimensions,
                    input_format: PixelFormat::Rgba8,
                    color_range: ColorRange::Limited,
                    hardware: HardwarePreference::Avoid,
                    timescale: 30,
                    frame_duration: 1,
                    // A key frame every three frames.
                    configuration: vec![40, 0, 3],
                },
                &limits,
            )
            .unwrap();
        let mut muxer = Mp4Muxer::new(
            MemorySink::new(),
            vec![Mp4TrackConfig {
                encoder: encoder.config().clone(),
                format: Mp4TrackFormat::Video(dimensions),
            }],
            60,
        )
        .await
        .unwrap();
        for index in 0..7_u64 {
            let frame = VideoFrame::new(
                dimensions,
                PixelFormat::Rgba8,
                ColorRange::Limited,
                vec![crate::media::Plane {
                    data: rgba(index),
                    stride: dimensions.width as usize * 4,
                }],
                &limits,
            )
            .unwrap();
            let samples = encoder
                .encode(
                    FrameIndex(index),
                    FrameSource::Cpu(CpuFrameSource {
                        frame: &frame,
                        orientation: Orientation::TopLeft,
                    }),
                )
                .await
                .unwrap();
            for sample in samples {
                muxer.write_sample(0, sample).await.unwrap();
            }
        }
        // The encoder holds the start of each group back until it has seen it.
        for sample in encoder.finish().await.unwrap() {
            muxer.write_sample(0, sample).await.unwrap();
        }
        let bytes = muxer.finish().await.unwrap().into_inner();

        let Ok(mut session) = WebVideoDecodeSession::open(&bytes, 0, &limits).await else {
            // No WebCodecs VP9 decoder in this browser.
            return;
        };
        assert!(!session.is_software());
        for index in [5_u64, 1, 6] {
            let (decoded, pixels) = session
                .get(FrameIndex(index), &CancellationToken::new())
                .await
                .unwrap();
            assert_eq!(decoded, dimensions);
            let expected = rgba(index);
            let error = pixels
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum::<f64>()
                / pixels.len() as f64;
            let psnr = 10.0 * (255.0 * 255.0 / error.max(1e-9)).log10();
            assert!(psnr > 30.0, "frame {index}: {psnr:.1} dB");
        }
    }

    /// Issue #509: a color AV1 track (the bundled SVT-AV1 8-bit 4:2:0 Main
    /// sample) decodes through the fallback. The digests are FFmpeg/libdav1d's
    /// decode of the same frames converted by the crate's own BT.601 RGBA
    /// conversion, the lines of
    /// `crates/zvidlib-av1-decoder/tests/fixtures/big_buck_bunny_av1_rgba.sha256`
    /// the native conformance test checks every frame against; frame 20 is
    /// reached by walking forwards from the random-access point at frame 0.
    #[wasm_bindgen_test(async)]
    async fn software_fallback_decodes_color_av1_like_an_independent_decoder() {
        const COLOR_AV1: &[u8] = include_bytes!("../examples/media/BigBuckBunny.av1.mp4");
        let mut session = WebVideoDecodeSession::open_with(
            COLOR_AV1,
            0,
            &Limits::default(),
            BackendChoice::SoftwareOnly,
        )
        .await
        .unwrap();
        assert!(session.is_software());
        assert_eq!(
            (session.dimensions().width, session.dimensions().height),
            (960, 540)
        );
        assert_eq!(session.frame_count(), 768);
        let expected = [
            (
                0,
                "3c7fbf07cee4f4e3b72c9667061e162362019f6aea2c035fad2032180dd8742f",
            ),
            (
                1,
                "2c189d695e847d075bb765611f51287594eca1e8bbabc5a403c6d623c21dc5c7",
            ),
            (
                20,
                "387c5e72026d94a8c123c3e27977e44a84bc6ec902629e7f44b040c957979886",
            ),
        ];
        for (frame, expected) in expected {
            let (dimensions, rgba) = session
                .get(FrameIndex(frame), &CancellationToken::new())
                .await
                .unwrap();
            assert_eq!(rgba.len(), 960 * 540 * 4);
            assert_eq!(digest(dimensions, rgba), expected, "frame {frame}");
        }
    }

    /// Sequential reads through `WebCodecs` keep going past the first dozen
    /// frames. The frame cache used to hold up to `Limits::max_cached_frames`
    /// decoded `VideoFrame`s open, and Chrome's decoder stops emitting frames
    /// once that many of its output buffers are held, so `get()` waited
    /// forever for frame 14 of this track. Each read is canceled after ten
    /// seconds so a regression fails the test instead of hanging the suite.
    #[wasm_bindgen_test(async)]
    async fn sequential_webcodecs_reads_do_not_exhaust_the_decoder() {
        const COLOR_AV1: &[u8] = include_bytes!("../examples/media/BigBuckBunny.av1.mp4");
        let mut session = WebVideoDecodeSession::open(COLOR_AV1, 0, &Limits::default())
            .await
            .unwrap();
        if session.is_software() {
            return;
        }
        for frame in 0..48 {
            let cancellation = CancellationToken::new();
            let timeout = cancellation.clone();
            let cancel = Closure::once_into_js(move || timeout.cancel());
            let set_timeout: js_sys::Function =
                js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("setTimeout"))
                    .unwrap()
                    .unchecked_into();
            let timer = set_timeout
                .call2(&JsValue::NULL, &cancel, &JsValue::from_f64(10_000.0))
                .unwrap();
            let result = session.get(FrameIndex(frame), &cancellation).await;
            let clear_timeout: js_sys::Function =
                js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("clearTimeout"))
                    .unwrap()
                    .unchecked_into();
            clear_timeout.call1(&JsValue::NULL, &timer).unwrap();
            let (dimensions, rgba) =
                result.unwrap_or_else(|error| panic!("frame {frame}: {error:?}"));
            assert_eq!((dimensions.width, dimensions.height), (960, 540));
            assert_eq!(rgba.len(), 960 * 540 * 4);
        }
    }

    /// Issue #655: reads paced like playback, with time between them for the
    /// decoder to run ahead, continue one decode session instead of resetting
    /// it. Evicting by age closed the frames the decoder emitted ahead of each
    /// read, so on this single group of pictures every other read re-decoded
    /// the track from frame 0.
    #[wasm_bindgen_test(async)]
    async fn paced_sequential_reads_keep_the_frames_the_decoder_ran_ahead_to() {
        const COLOR_AV1: &[u8] = include_bytes!("../examples/media/BigBuckBunny.av1.mp4");
        let mut session = WebVideoDecodeSession::open(COLOR_AV1, 0, &Limits::default())
            .await
            .unwrap();
        if session.is_software() {
            return;
        }
        for frame in 0..24 {
            let index = FrameIndex(frame);
            if frame > 0 {
                let DecodeBackend::WebCodecs(decoder) = &session.backend else {
                    unreachable!("checked above");
                };
                let position = decoder.decode_position_by_presentation[&index];
                assert!(
                    decoder.cache.contains_key(&index)
                        || can_continue_session(
                            decoder.next_decode_position,
                            decoder.session_start,
                            decoder.nearest_random_access(position),
                            &decoder.published_since_reset,
                            index,
                        ),
                    "reading frame {frame} would reset the decoder"
                );
            }
            session
                .get(index, &CancellationToken::new())
                .await
                .unwrap_or_else(|error| panic!("frame {frame}: {error:?}"));
            let pause = js_sys::Promise::new(&mut |resolve, _reject| {
                let set_timeout: js_sys::Function =
                    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("setTimeout"))
                        .unwrap()
                        .unchecked_into();
                set_timeout
                    .call2(&JsValue::NULL, &resolve, &JsValue::from_f64(40.0))
                    .unwrap();
            });
            JsFuture::from(pause).await.unwrap();
        }
    }

    /// Issue #527: a VP9 track (libvpx's two-pass encode of the bundled
    /// sample, whose superframes carry hidden alternate reference frames)
    /// decodes through the fallback to exactly the frames libvpx decodes it
    /// to, converted by the crate's BT.601 RGBA conversion - the lines of
    /// `crates/zvidlib-vp9-decoder/tests/fixtures/vp9_bbb_256x144_rgba.sha256` the native
    /// conformance test checks. Frame 30 is reached from the key frame at 24,
    /// and frame 3 by restarting at 0.
    #[wasm_bindgen_test(async)]
    async fn software_fallback_decodes_vp9_like_libvpx() {
        const VP9: &[u8] =
            include_bytes!("../crates/zvidlib-vp9-decoder/tests/fixtures/vp9_bbb_256x144.mp4");
        let expected: Vec<&str> = include_str!(
            "../crates/zvidlib-vp9-decoder/tests/fixtures/vp9_bbb_256x144_rgba.sha256"
        )
        .lines()
        .map(|line| line.split_once(' ').unwrap().1)
        .collect();
        let mut session = WebVideoDecodeSession::open_with(
            VP9,
            0,
            &Limits::default(),
            BackendChoice::SoftwareOnly,
        )
        .await
        .unwrap();
        assert!(session.is_software());
        assert_eq!(session.frame_count(), 48);
        for frame in [0_u64, 1, 30, 3] {
            let (dimensions, rgba) = session
                .get(FrameIndex(frame), &CancellationToken::new())
                .await
                .unwrap();
            assert_eq!((dimensions.width, dimensions.height), (256, 144));
            assert_eq!(
                digest(dimensions, rgba),
                expected[frame as usize],
                "frame {frame}"
            );
        }
    }

    /// The same VP9 track opened the ordinary way, through `WebCodecs` with a
    /// `vp09.00.LL.08` codec string wherever the browser decodes VP9, and the
    /// fallback otherwise. Either way it yields whole RGBA frames of the
    /// track's size.
    #[wasm_bindgen_test(async)]
    async fn vp9_tracks_decode_through_webcodecs_or_the_fallback() {
        const VP9: &[u8] =
            include_bytes!("../crates/zvidlib-vp9-decoder/tests/fixtures/vp9_bbb_256x144.mp4");
        let source = MemorySource::new(VP9.to_vec());
        let track = parse_video_track(&source, 0, &Limits::default())
            .await
            .unwrap();
        let derived = derive_codec_string(track.codec, &track.decoder_config).unwrap();
        assert!(
            derived.codec_string.starts_with("vp09.00."),
            "{}",
            derived.codec_string
        );
        assert!(
            derived.codec_string.ends_with(".08"),
            "{}",
            derived.codec_string
        );

        let mut session = WebVideoDecodeSession::open(VP9, 0, &Limits::default())
            .await
            .unwrap();
        for frame in [0_u64, 47, 12] {
            let (dimensions, rgba) = session
                .get(FrameIndex(frame), &CancellationToken::new())
                .await
                .unwrap();
            assert_eq!((dimensions.width, dimensions.height), (256, 144));
            assert_eq!(rgba.len(), 256 * 144 * 4, "frame {frame}");
        }
    }

    /// A track neither `WebCodecs` nor the software decoder can take is still
    /// reported `Unsupported`, naming why the fallback refused it. The software
    /// AV1 decoder covers 8-bit streams, so a 10-bit Main track is refused.
    #[wasm_bindgen_test(async)]
    async fn a_track_the_software_decoder_refuses_stays_unsupported() {
        const MAIN_10_AV1: &[u8] =
            include_bytes!("../crates/zvidlib-av1-decoder/tests/fixtures/av1_main10_64x64.mp4");
        let error = match WebVideoDecodeSession::open_with(
            MAIN_10_AV1,
            0,
            &Limits::default(),
            BackendChoice::SoftwareOnly,
        )
        .await
        {
            Ok(_) => panic!("the AV1 software decoder covers 8-bit streams only"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.message().contains("software decoder cannot either"),
            "{}",
            error.message()
        );
    }

    const VP8_ALTREF: &[u8] =
        include_bytes!("../crates/zvidlib-vp8/tests/fixtures/vp8/vp8_altref_98x66.webm");

    /// Issue #537: three of this WebM track's 43 VP8 blocks are hidden
    /// alternate references. The session counts only its 40 shown frames and
    /// decodes through the hidden ones to reach any of them, in sequential,
    /// reverse and alternating order, on either backend. The software
    /// fallback matches a straight decode of every sample with the native
    /// decoder; `WebCodecs` converts to RGBA its own way, so it has to match a
    /// sequential read of a fresh `WebCodecs` session.
    #[wasm_bindgen_test(async)]
    async fn hidden_vp8_frames_are_decoded_through_but_not_counted() {
        let limits = Limits::default();
        let cancellation = CancellationToken::new();
        let source = MemorySource::new(VP8_ALTREF.to_vec());
        let track = parse_video_track(&source, 0, &limits).await.unwrap();
        let samples = track
            .to_encoded_video_samples(&source, &limits)
            .await
            .unwrap();
        assert_eq!(samples.len(), 43);
        let factory = crate::native_vp8_video_decoder_factory();
        let configuration = VideoDecoderConfig {
            codec: track.codec,
            profile: CodecProfile::Vp8,
            coded_dimensions: track.dimensions.unwrap(),
            output_format: PixelFormat::Rgba8,
            color_range: ColorRange::Limited,
            hardware: HardwarePreference::Avoid,
            configuration: Vec::new(),
        };
        let mut decoder = factory.create(&configuration, &limits).unwrap();
        let mut native = vec![String::new(); 40];
        for sample in &samples {
            for output in decoder.submit(sample, &cancellation).unwrap() {
                native[output.presentation_index.0 as usize] =
                    crate::conformance::FrameDigest::from_frame(&output.frame)
                        .unwrap()
                        .to_hex();
            }
        }
        assert!(native.iter().all(|digest| !digest.is_empty()));
        assert_eq!(
            video_frame_durations_ms(VP8_ALTREF, 0, &limits)
                .await
                .unwrap()
                .len(),
            40
        );

        let mut alternating = Vec::new();
        let (mut low, mut high) = (0, 40);
        while low < high {
            high -= 1;
            alternating.push(high);
            if low < high {
                alternating.push(low);
                low += 1;
            }
        }
        let orders = [
            (0..40).collect::<Vec<u64>>(),
            (0..40).rev().collect(),
            alternating,
        ];

        let webcodecs = WebVideoDecodeSession::open(VP8_ALTREF, 0, &limits)
            .await
            .unwrap();
        let expected = if webcodecs.is_software() {
            native.clone()
        } else {
            let mut session = webcodecs;
            let mut digests = Vec::new();
            for index in 0..40 {
                let (dimensions, rgba) =
                    session.get(FrameIndex(index), &cancellation).await.unwrap();
                digests.push(digest(dimensions, rgba));
            }
            digests
        };
        for (choice, expected) in [
            (BackendChoice::SoftwareOnly, &native),
            (BackendChoice::Automatic, &expected),
        ] {
            for order in &orders {
                let mut session = WebVideoDecodeSession::open_with(VP8_ALTREF, 0, &limits, choice)
                    .await
                    .unwrap();
                assert_eq!(session.frame_count(), 40);
                for &index in order {
                    let (dimensions, rgba) = session
                        .get(FrameIndex(index), &cancellation)
                        .await
                        .unwrap_or_else(|error| panic!("{choice:?} frame {index}: {error}"));
                    assert_eq!(
                        digest(dimensions, rgba),
                        expected[index as usize],
                        "{choice:?} frame {index}"
                    );
                }
            }
        }
    }
}
