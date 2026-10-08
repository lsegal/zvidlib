//! On-demand browser playback of an MP4 or WebM input that is never loaded
//! whole (issues #677 and #685).
//!
//! [`WasmOnDemandPlayback`] is the JavaScript face of the on-demand
//! [`PlaybackController`] of issue #672. It reads the file through a
//! [`RangeSource`]: HTTP range requests against a URL, slices of a `Blob` or
//! `File`, or an application's own `read(offset, length)`. The container is
//! detected from the input's leading bytes, and only its header and index and
//! the compressed samples playback reaches are fetched, into caches bounded by
//! the configured byte budgets.
//!
//! Every playback call is synchronous and never waits on the network. One that
//! needs a sample not loaded yet throws a `WOULD_BLOCK` error and leaves
//! playback where it was; the page awaits `prefetch()`, which loads what was
//! missing, and calls again on its next animation frame:
//!
//! ```js
//! function render() {
//!   try {
//!     const presentation = playback.present();
//!     if (presentation.picture) draw(presentation.picture);
//!   } catch (error) {
//!     if (errorCode(error) !== "WOULD_BLOCK") throw error;
//!     playback.prefetch();
//!   }
//!   requestAnimationFrame(render);
//! }
//! ```
//!
//! `WebCodecs` decodes asynchronously, so it cannot sit behind the
//! controller's synchronous reads. [`WebCodecsVideoSource`] and
//! [`BrowserAudioSource`] decode during the prefetch instead, a readahead at a
//! time, and answer the controller's reads from what they decoded. Video
//! decodes through `WebCodecs` whenever the browser supports the track, which
//! is usually in hardware (issue #680), and otherwise on the crate's software
//! decoders, which are synchronous and so sit behind the reads directly. AAC
//! audio has no software decoder in the browser build, so it always decodes
//! through `WebCodecs`. Opus audio decodes through `WebCodecs` where the
//! browser supports it, and on the crate's software decoder otherwise.
//!
//! An input with no audio track plays on a clock of its own (issue #681): the
//! `AudioContext`'s when the page gives one, with nothing scheduled on it, and
//! the page's `performance.now()` otherwise. [`SilentAudioSource`] stands in
//! for the audio the controller schedules against it.

use crate::audio::{
    AudioDecoder, AudioPacketProvider, AudioSampleReader, AudioTrackTiming, EncodedAudioSample,
};
use crate::codec::{CancellationToken, EncodedVideoSample, ExactFrameReader};
use crate::codec_config::derive_codec_string;
use crate::io::{ByteSource, CachingByteSource, IoFuture};
use crate::media::{AudioBuffer, Codec, ColorRange, PixelFormat, Plane, VideoFrame};
use crate::mp4_demux::Mp4Track;
use crate::playback::{
    AudioOutputBackend, IndexedPresentationTimeline, OnDemandVideoSource, PlaybackAudioSource,
    PlaybackController, PlaybackOptions, PlaybackVideoSource, PrefetchAudioSource,
    PrefetchVideoSource, WebAudioOutput,
};
use crate::timeline::{FrameIndex, SampleRange};
use crate::wasm_api::{
    MAX_SAFE_INTEGER, WasmVideoFrame, bigint_u64, ensure_open, js_error, owned_u8_array, parse_u64,
    property,
};
use crate::web_audio_decoder::{
    AAC_PREROLL_PACKETS, NoSoftwareDecoder, WebAudioDecoderConfig, WebCodecsAudioDecoder,
};
use crate::web_decoder::{
    WebCodecsDecoder, js_to_promise, normalize_js_error, packed_rgba, software_video_decoder,
    webcodecs_config, webcodecs_supports,
};
use crate::{
    Error, ErrorKind, Limits, Mp4SampleLoader, OPUS_PREROLL_SAMPLES, PrefetchedAudioPacketProvider,
    Result, TrackKind,
};
use js_sys::{Object, Promise, Reflect, Uint8Array};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, future_to_promise};
use web_sys::{
    AudioBufferSourceNode, AudioDecoder as JsAudioDecoder, AudioDecoderSupport,
    AudioScheduledSourceNode, BaseAudioContext,
};

#[wasm_bindgen(module = "/js/browser.js")]
extern "C" {
    #[wasm_bindgen(catch, js_name = rangeSourceSize)]
    fn range_source_size(source: &JsValue) -> std::result::Result<Promise, JsValue>;

    #[wasm_bindgen(catch, js_name = readRangeSource)]
    fn read_range_source(
        source: &JsValue,
        offset: f64,
        length: f64,
    ) -> std::result::Result<Promise, JsValue>;

    #[wasm_bindgen(js_name = clockSeconds)]
    fn clock_seconds() -> f64;
}

/// The compressed video a playback may hold when the page does not say:
/// several seconds of a 1080p stream, and room for its largest key frames.
const DEFAULT_VIDEO_BUDGET_BYTES: u64 = 16 * 1024 * 1024;

/// The compressed audio a playback may hold when the page does not say: well
/// over a minute of ordinary AAC.
const DEFAULT_AUDIO_BUDGET_BYTES: u64 = 1024 * 1024;

/// Decode-order video samples a prefetch loads past the ones the current frame
/// needs, so the frames after it play without reporting anything missing.
const VIDEO_READAHEAD_SAMPLES: usize = 24;

/// Presentation frames a `WebCodecs` prefetch decodes from the one it is
/// asked for, and so the most decoded pictures a playback holds at once: a
/// third of a second at 24 frames a second.
const DECODED_VIDEO_FRAMES: u64 = 8;

/// The page size opening an input reads its container's header and index
/// through. A WebM's index is every block's header, a few bytes each across
/// all of its clusters, so a page combines the headers of the blocks near each
/// other into one request instead of two requests a block (issue #685).
const INDEX_PAGE_BYTES: u64 = 4 * 1024;

/// The index pages opening an input holds at once. The index is read front to
/// back, so only the pages around the read in progress are worth keeping.
const INDEX_CACHE_BYTES: u64 = 64 * 1024;

/// Audio packets a prefetch loads past the ones it decodes.
const AUDIO_READAHEAD_PACKETS: usize = 16;

/// How many prefetch passes an audio decode may take to load packets the plan
/// missed before giving up on the request.
const MAX_AUDIO_LOAD_PASSES: usize = 8;

/// The ticks per second of the clock that times an input with no audio track,
/// standing in for an audio sample rate.
const VIDEO_ONLY_CLOCK_RATE: u32 = 48_000;

fn core_error(error: Error) -> JsValue {
    js_error(error.kind(), error.message())
}

/// A failed source read as an [`Error`], keeping the kind its `ZvidError`
/// code names - a URL that does not answer range requests is `UNSUPPORTED`,
/// not an I/O failure - and treating anything else as I/O.
fn source_error(error: JsValue, context: &str) -> Error {
    let detail = Reflect::get(&error, &JsValue::from_str("message"))
        .ok()
        .and_then(|value| value.as_string())
        .or_else(|| error.as_string())
        .unwrap_or_else(|| "browser read failed".to_owned());
    let code = Reflect::get(&error, &JsValue::from_str("code"))
        .ok()
        .and_then(|value| value.as_string());
    let kind = match code.as_deref() {
        Some("UNSUPPORTED") => ErrorKind::Unsupported,
        Some("INVALID_INPUT") => ErrorKind::InvalidInput,
        Some("RESOURCE_LIMIT") => ErrorKind::ResourceLimit,
        Some("CANCELED") => ErrorKind::Canceled,
        _ => ErrorKind::Io,
    };
    Error::new(kind, format!("{context}: {detail}"))
}

/// A [`ByteSource`] over a URL, `Blob` or range reader, whose reads resolve
/// only once the bytes arrive.
///
/// Clones share the source and the count of bytes read, so the video and audio
/// loaders can each own one.
#[derive(Clone)]
pub(crate) struct RangeSource {
    source: JsValue,
    len: u64,
    fetched_bytes: Rc<Cell<u64>>,
}

impl RangeSource {
    async fn open(source: JsValue) -> Result<Self> {
        let promise =
            range_source_size(&source).map_err(|error| source_error(error, "sizing the source"))?;
        let size = JsFuture::from(promise)
            .await
            .map_err(|error| source_error(error, "sizing the source"))?
            .as_f64()
            .filter(|size| size.fract() == 0.0 && *size >= 0.0 && *size <= MAX_SAFE_INTEGER as f64)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "the source's size must be a non-negative safe integer",
                )
            })?;
        Ok(Self {
            source,
            len: size as u64,
            fetched_bytes: Rc::new(Cell::new(0)),
        })
    }

    /// Every byte read from the source so far.
    pub(crate) fn fetched_bytes(&self) -> u64 {
        self.fetched_bytes.get()
    }
}

impl ByteSource for RangeSource {
    fn len(&self) -> Option<u64> {
        Some(self.len)
    }

    fn read_at<'a>(&'a self, offset: u64, destination: &'a mut [u8]) -> IoFuture<'a, usize> {
        Box::pin(async move {
            if offset >= self.len || destination.is_empty() {
                return Ok(0);
            }
            let length = (destination.len() as u64).min(self.len - offset);
            let promise = read_range_source(&self.source, offset as f64, length as f64)
                .map_err(|error| source_error(error, "reading the source"))?;
            let bytes: Uint8Array = JsFuture::from(promise)
                .await
                .map_err(|error| source_error(error, "reading the source"))?
                .dyn_into()
                .map_err(|_| Error::new(ErrorKind::Io, "a source read returned no bytes"))?;
            let count = (bytes.length() as usize).min(length as usize);
            bytes
                .subarray(0, count as u32)
                .copy_to(&mut destination[..count]);
            self.fetched_bytes
                .set(self.fetched_bytes.get() + count as u64);
            Ok(count)
        })
    }
}

/// Plays the controller's audio through a Web Audio context, scheduling each
/// buffer at the context time its media position falls on.
pub(crate) struct AudioContextBackend {
    context: BaseAudioContext,
    sample_rate: u32,
    /// The context time and media sample of the last `start`.
    anchor: (f64, u64),
    generation: u64,
    /// Scheduled nodes with their generation and the context time they end.
    nodes: Vec<(u64, f64, AudioBufferSourceNode)>,
}

impl AudioContextBackend {
    fn new(context: BaseAudioContext, sample_rate: u32) -> Self {
        Self {
            context,
            sample_rate,
            anchor: (0.0, 0),
            generation: 0,
            nodes: Vec::new(),
        }
    }

    fn stop_nodes(&mut self, keep: impl Fn(u64) -> bool) {
        self.nodes.retain(|(generation, _, node)| {
            if keep(*generation) {
                return true;
            }
            // `AudioBufferSourceNode::stop` itself is deprecated in favor of
            // the scheduled-source method it inherits.
            let scheduled: &AudioScheduledSourceNode = node;
            let _ = scheduled.stop();
            node.disconnect().ok();
            false
        });
    }
}

impl AudioOutputBackend for AudioContextBackend {
    fn clock_samples(&self) -> u64 {
        (self.context.current_time() * f64::from(self.sample_rate)).floor() as u64
    }

    fn start(&mut self, media_sample: u64) -> Result<()> {
        self.anchor = (self.context.current_time(), media_sample);
        Ok(())
    }

    fn schedule(&mut self, buffer: AudioBuffer, generation: u64) -> Result<()> {
        if generation < self.generation {
            return Ok(());
        }
        self.generation = generation;
        let now = self.context.current_time();
        self.nodes.retain(|(_, end, _)| *end > now);

        let rate = f64::from(buffer.sample_rate);
        let frames = buffer.range.len();
        let when = self.anchor.0 + (buffer.range.start as f64 - self.anchor.1 as f64) / rate;
        let end = when + frames as f64 / rate;
        if frames == 0 || end <= now {
            return Ok(());
        }
        let context = |error| normalize_js_error(error, "scheduling Web Audio playback");
        let channels = usize::from(buffer.channels);
        let audio = self
            .context
            .create_buffer(
                u32::from(buffer.channels),
                frames as u32,
                buffer.sample_rate as f32,
            )
            .map_err(context)?;
        let mut plane = vec![0.0_f32; frames as usize];
        for channel in 0..channels {
            for (sample, frame) in plane.iter_mut().zip(buffer.samples.chunks_exact(channels)) {
                *sample = frame[channel];
            }
            audio
                .copy_to_channel(&plane, channel as i32)
                .map_err(context)?;
        }
        let node = self.context.create_buffer_source().map_err(context)?;
        node.set_buffer(Some(&audio));
        node.connect_with_audio_node(&self.context.destination())
            .map_err(context)?;
        // A buffer that arrives after its start time plays from where the
        // clock is now rather than late.
        node.start_with_when_and_grain_offset(when.max(now), (now - when).max(0.0))
            .map_err(context)?;
        self.nodes.push((generation, end, node));
        Ok(())
    }

    fn cancel_queued(&mut self, generation: u64) -> Result<()> {
        self.generation = self.generation.max(generation);
        self.stop_nodes(|scheduled| scheduled >= generation);
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        self.stop_nodes(|_| false);
        Ok(())
    }
}

/// An AAC or Opus track's presentation samples, decoded from packets an
/// [`Mp4SampleLoader`] loads on demand.
///
/// `WebCodecs` decodes asynchronously, so it cannot sit behind the
/// controller's synchronous reads. [`PrefetchAudioSource::prefetch`] decodes
/// the range it is asked for and a readahead past it instead, and
/// [`PlaybackAudioSource::read`] answers from those samples, reporting
/// [`ErrorKind::WouldBlock`] for anything not decoded yet. An Opus track the
/// browser cannot decode, or decodes out of line with its packets, goes
/// through the crate's software decoder instead, as
/// [`crate::web_audio_decoder`] arranges for `MediaInput`.
pub(crate) struct BrowserAudioSource {
    reader: AudioSampleReader<Box<dyn AudioDecoder>>,
    loader: Mp4SampleLoader<RangeSource>,
    /// `None` once reads go through the reader's software decoder.
    webcodecs: Option<WebCodecsAudioDecoder>,
    /// Whether the reader's own decoder is a real software decoder rather
    /// than [`NoSoftwareDecoder`].
    has_software: bool,
    readahead_samples: u64,
    /// One contiguous run of decoded presentation samples.
    decoded: Option<AudioBuffer>,
    limits: Limits,
}

impl BrowserAudioSource {
    /// Fails with [`ErrorKind::Unsupported`] unless `audio` is an AAC track
    /// this browser decodes through `WebCodecs`, or an Opus track. `timing` is
    /// the track's timing on the decoded sample clock, as its container gives
    /// it. Reads no packet but an Opus track's last, whose first two bytes give
    /// its length.
    async fn open(
        audio: Mp4Track,
        timing: AudioTrackTiming,
        source: RangeSource,
        budget_bytes: u64,
        limits: Limits,
    ) -> Result<Self> {
        if !matches!(audio.codec, Codec::Aac | Codec::Opus) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "on-demand playback supports AAC and Opus audio tracks",
            ));
        }
        let config = WebAudioDecoderConfig::for_track(&audio)?;
        let support: AudioDecoderSupport = JsFuture::from(js_to_promise(
            JsAudioDecoder::is_config_supported(&config.to_js()),
        ))
        .await
        .map_err(|error| normalize_js_error(error, "querying WebCodecs audio decoder support"))?
        .unchecked_into();
        let webcodecs_supported = support.get_supported().unwrap_or(false);
        let software = software_audio_decoder(&audio, limits)?;
        if !webcodecs_supported && software.is_none() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!(
                    "this browser cannot decode {} via WebCodecs, and the browser build has no \
                     software decoder for it",
                    config.codec
                ),
            ));
        }
        let sample_rate = audio.audio_sample_rate()?;
        let codec = audio.codec;
        let loader = Mp4SampleLoader::new(audio, source, budget_bytes)?;
        let (packets, preroll) = if codec == Codec::Opus {
            let packets = loader.opus_packet_provider().await?;
            let preroll = opus_preroll_packets(&packets);
            (packets, preroll)
        } else {
            (loader.aac_packet_provider()?, AAC_PREROLL_PACKETS)
        };
        let has_software = software.is_some();
        let reader = AudioSampleReader::from_provider(
            software.unwrap_or_else(|| Box::new(NoSoftwareDecoder)),
            Box::new(packets),
            sample_rate,
            config.channels,
            timing,
            preroll,
            limits,
        )?;
        let webcodecs = if webcodecs_supported {
            Some(WebCodecsAudioDecoder::open(config, limits)?)
        } else {
            None
        };
        Ok(Self {
            reader,
            loader,
            webcodecs,
            has_software,
            readahead_samples: u64::from(sample_rate),
            decoded: None,
            limits,
        })
    }

    /// Whether reads go through the software decoder, because `WebCodecs`
    /// cannot decode the track or decoded it wrongly.
    #[cfg(all(test, feature = "all"))]
    fn is_software(&self) -> bool {
        self.webcodecs.is_none()
    }

    /// Drops the `WebCodecs` decoder, so reads go through software.
    #[cfg(all(test, feature = "all"))]
    fn use_software(&mut self) {
        self.webcodecs = None;
    }

    /// Decodes `range` from a fresh decoder, loading the packets it needs
    /// first, and those a decode reports missing.
    async fn decode(&mut self, range: SampleRange) -> Result<AudioBuffer> {
        let cancellation = CancellationToken::new();
        for _ in 0..MAX_AUDIO_LOAD_PASSES {
            // Every decode starts a fresh decoder, preroll included, as
            // `WebAudioDecodeSession` does.
            self.reader.reset()?;
            self.loader.load_missing().await?;
            for run in self.reader.packets_for_range(range)? {
                self.loader.load(run, AUDIO_READAHEAD_PACKETS).await?;
            }
            let result = match self.webcodecs.as_mut() {
                Some(decoder) => {
                    let result = self
                        .reader
                        .get_range_with(
                            range,
                            &cancellation,
                            async |_, packets: &[EncodedAudioSample]| decoder.decode(packets).await,
                        )
                        .await;
                    match result {
                        Err(error)
                            if self.has_software
                                && !matches!(
                                    error.kind(),
                                    ErrorKind::WouldBlock
                                        | ErrorKind::Canceled
                                        | ErrorKind::MalformedMedia
                                ) =>
                        {
                            // The browser's decode does not line up with the
                            // track. The software decoder does, so it takes
                            // over for good, starting with this range.
                            self.webcodecs = None;
                            continue;
                        }
                        result => result,
                    }
                }
                None => self.reader.get_range(range, &cancellation),
            };
            match result {
                Err(error) if error.kind() == ErrorKind::WouldBlock => continue,
                result => return result,
            }
        }
        Err(Error::new(
            ErrorKind::ResourceLimit,
            "the audio budget cannot hold the packets one prefetch decodes",
        ))
    }
}

/// The crate's software decoder for `track`, if the browser build has one:
/// Opus has one when the `opus-decoder` feature is on, and AAC never does.
fn software_audio_decoder(
    track: &Mp4Track,
    limits: Limits,
) -> Result<Option<Box<dyn AudioDecoder>>> {
    match track.codec {
        #[cfg(feature = "opus-decoder")]
        Codec::Opus => Ok(Some(Box::new(crate::NativeOpusDecoder::new(
            &track.opus_config()?,
            limits,
        )?))),
        _ => {
            let _ = limits;
            Ok(None)
        }
    }
}

/// How many packets an Opus read decodes ahead of the first one it needs:
/// enough to cover [`OPUS_PREROLL_SAMPLES`] even at the track's shortest
/// packet, as [`crate::opus_preroll_packets`] counts for packets in memory.
fn opus_preroll_packets(packets: &PrefetchedAudioPacketProvider) -> usize {
    let shortest = (0..packets.len())
        .map(|index| packets.decoded_range(index).len())
        .filter(|&length| length > 0)
        .min()
        .unwrap_or(u64::from(OPUS_PREROLL_SAMPLES));
    usize::try_from(u64::from(OPUS_PREROLL_SAMPLES).div_ceil(shortest)).unwrap_or(usize::MAX)
}

impl PlaybackAudioSource for BrowserAudioSource {
    fn sample_rate(&self) -> u32 {
        self.reader.sample_rate()
    }

    fn presentation_length(&self) -> u64 {
        self.reader.presentation_length()
    }

    fn read(&mut self, range: SampleRange, _: &CancellationToken) -> Result<AudioBuffer> {
        let Some(decoded) = self
            .decoded
            .as_ref()
            .filter(|decoded| decoded.range.start <= range.start && range.end <= decoded.range.end)
        else {
            return Err(Error::new(
                ErrorKind::WouldBlock,
                "the audio has not been decoded yet",
            ));
        };
        let channels = usize::from(decoded.channels);
        let from = (range.start - decoded.range.start) as usize * channels;
        let to = (range.end - decoded.range.start) as usize * channels;
        AudioBuffer::new(
            range,
            decoded.sample_rate,
            decoded.channels,
            decoded.samples[from..to].to_vec(),
            &self.limits,
        )
    }

    fn reset(&mut self) -> Result<()> {
        // The decoded samples are the presentation's own, so they stay valid
        // across a seek; only the loads the old position asked for are not.
        self.loader.clear_missing();
        self.reader.reset()
    }
}

impl PrefetchAudioSource for BrowserAudioSource {
    fn prefetch(&mut self, range: SampleRange) -> IoFuture<'_, ()> {
        Box::pin(async move {
            let length = self.reader.presentation_length();
            let wanted_end = range.end.saturating_add(self.readahead_samples).min(length);
            // A run that already covers the range and half the readahead is
            // enough; topping it up at every call would decode a few samples,
            // and their preroll, every frame.
            let continues = self
                .decoded
                .as_ref()
                .filter(|decoded| {
                    decoded.range.start <= range.start && range.start <= decoded.range.end
                })
                .map(|decoded| decoded.range.end);
            let start = match continues {
                Some(end)
                    if end
                        >= range
                            .end
                            .saturating_add(self.readahead_samples / 2)
                            .min(length) =>
                {
                    return Ok(());
                }
                Some(end) => end,
                None => range.start,
            };
            if start >= wanted_end {
                return Ok(());
            }
            let fresh = self.decode(SampleRange::new(start, wanted_end)?).await?;
            let merged = match self.decoded.take() {
                Some(mut decoded) if continues.is_some() => {
                    // Keep only what the controller can still ask for.
                    let channels = usize::from(decoded.channels);
                    let drop = (range.start - decoded.range.start) as usize * channels;
                    decoded.samples.drain(..drop);
                    decoded.samples.extend_from_slice(&fresh.samples);
                    AudioBuffer::new(
                        SampleRange::new(range.start, fresh.range.end)?,
                        decoded.sample_rate,
                        decoded.channels,
                        decoded.samples,
                        &self.limits,
                    )?
                }
                _ => fresh,
            };
            self.decoded = Some(merged);
            Ok(())
        })
    }
}

/// A video track's pictures, decoded through `WebCodecs` from samples an
/// [`Mp4SampleLoader`] loads on demand (issue #680).
///
/// [`PrefetchVideoSource::prefetch`] decodes the frame it is asked for and the
/// [`DECODED_VIDEO_FRAMES`] after it, loading the samples the decode reaches
/// as it goes, and [`PlaybackVideoSource::get_exact`] answers from those
/// pictures, reporting [`ErrorKind::WouldBlock`] for anything not decoded yet.
/// The pictures are RGBA copies, so holding them keeps none of the decoder's
/// output buffers; the `VideoFrame`s it emits stay within
/// [`WebCodecsDecoder`]'s own bounded cache, which closes them as it evicts
/// them.
pub(crate) struct WebCodecsVideoSource {
    decoder: WebCodecsDecoder,
    loader: Mp4SampleLoader<RangeSource>,
    frame_count: u64,
    /// Decoded pictures by presentation index, never more than
    /// [`DECODED_VIDEO_FRAMES`].
    decoded: BTreeMap<FrameIndex, VideoFrame>,
    limits: Limits,
}

impl WebCodecsVideoSource {
    /// `config` must be a configuration the browser supports for the
    /// loader's track. Reads no sample.
    fn new(
        config: web_sys::VideoDecoderConfig,
        loader: Mp4SampleLoader<RangeSource>,
        limits: Limits,
    ) -> Result<Self> {
        let decoder = WebCodecsDecoder::open(config, Box::new(loader.sample_provider()?), &limits)?;
        Ok(Self {
            decoder,
            frame_count: loader.track().presentation_order.len() as u64,
            loader,
            decoded: BTreeMap::new(),
            limits,
        })
    }

    /// Decodes `frame`, loading the samples the decode reaches that are not
    /// loaded yet, each with a readahead.
    async fn decode(&mut self, frame: FrameIndex) -> Result<VideoFrame> {
        let loader = &self.loader;
        let (dimensions, rgba) = self
            .decoder
            .get_with(frame, &CancellationToken::new(), async |position| {
                // The decode loads what it misses itself, so the provider's
                // record of it is not worth keeping.
                loader.clear_missing();
                loader
                    .load(position..position + 1, VIDEO_READAHEAD_SAMPLES)
                    .await
            })
            .await?;
        VideoFrame::new(
            dimensions,
            PixelFormat::Rgba8,
            ColorRange::Full,
            vec![Plane {
                data: rgba,
                stride: dimensions.width as usize * 4,
            }],
            &self.limits,
        )
    }
}

impl PlaybackVideoSource for WebCodecsVideoSource {
    fn get_exact(&mut self, frame: FrameIndex, _: &CancellationToken) -> Result<VideoFrame> {
        self.decoded.get(&frame).cloned().ok_or_else(|| {
            Error::new(
                ErrorKind::WouldBlock,
                "the video frame has not been decoded yet",
            )
        })
    }

    fn reset(&mut self) -> Result<()> {
        // The decoded pictures are the presentation's own, so they stay valid
        // across a seek, and the next prefetch drops those it moved away from.
        self.loader.clear_missing();
        Ok(())
    }
}

impl PrefetchVideoSource for WebCodecsVideoSource {
    fn prefetch(&mut self, frame: FrameIndex) -> IoFuture<'_, ()> {
        Box::pin(async move {
            let end = frame
                .0
                .saturating_add(DECODED_VIDEO_FRAMES)
                .min(self.frame_count);
            self.decoded
                .retain(|&index, _| frame <= index && index.0 < end);
            // Pictures that already cover the frame and half the readahead
            // are enough; topping them up at every call would wait on the
            // decoder every frame.
            let covered = (frame.0..end)
                .take_while(|&index| self.decoded.contains_key(&FrameIndex(index)))
                .count() as u64;
            if covered >= (DECODED_VIDEO_FRAMES / 2).min(end.saturating_sub(frame.0)) {
                return Ok(());
            }
            for index in frame.0 + covered..end {
                let index = FrameIndex(index);
                if !self.decoded.contains_key(&index) {
                    let picture = self.decode(index).await?;
                    self.decoded.insert(index, picture);
                }
            }
            Ok(())
        })
    }
}

/// On-demand video, decoded through `WebCodecs` when the browser supports the
/// track and on the crate's software decoder otherwise.
pub(crate) enum BrowserVideoSource {
    WebCodecs(WebCodecsVideoSource),
    Software(OnDemandVideoSource<RangeSource>),
}

impl PlaybackVideoSource for BrowserVideoSource {
    fn get_exact(
        &mut self,
        frame: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<VideoFrame> {
        match self {
            Self::WebCodecs(source) => source.get_exact(frame, cancellation),
            Self::Software(source) => source.get_exact(frame, cancellation),
        }
    }

    fn reset(&mut self) -> Result<()> {
        match self {
            Self::WebCodecs(source) => source.reset(),
            Self::Software(source) => source.reset(),
        }
    }
}

impl PrefetchVideoSource for BrowserVideoSource {
    fn prefetch(&mut self, frame: FrameIndex) -> IoFuture<'_, ()> {
        match self {
            Self::WebCodecs(source) => source.prefetch(frame),
            Self::Software(source) => source.prefetch(frame),
        }
    }
}

/// Silence for as long as the video lasts, standing in for the audio of an
/// input that has none, so the controller's clock runs over the whole video.
pub(crate) struct SilentAudioSource {
    sample_rate: u32,
    length: u64,
    limits: Limits,
}

impl PlaybackAudioSource for SilentAudioSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn presentation_length(&self) -> u64 {
        self.length
    }

    fn read(&mut self, range: SampleRange, _: &CancellationToken) -> Result<AudioBuffer> {
        AudioBuffer::new(
            range,
            self.sample_rate,
            1,
            vec![0.0; range.len() as usize],
            &self.limits,
        )
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

impl PrefetchAudioSource for SilentAudioSource {
    fn prefetch(&mut self, _: SampleRange) -> IoFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}

/// What a playback's audio comes from: the input's audio track, or silence
/// when it has none.
pub(crate) enum PlaybackAudio {
    Track(BrowserAudioSource),
    Silent(SilentAudioSource),
}

impl PlaybackAudioSource for PlaybackAudio {
    fn sample_rate(&self) -> u32 {
        match self {
            Self::Track(audio) => audio.sample_rate(),
            Self::Silent(audio) => audio.sample_rate(),
        }
    }

    fn presentation_length(&self) -> u64 {
        match self {
            Self::Track(audio) => audio.presentation_length(),
            Self::Silent(audio) => audio.presentation_length(),
        }
    }

    fn read(
        &mut self,
        range: SampleRange,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        match self {
            Self::Track(audio) => audio.read(range, cancellation),
            Self::Silent(audio) => audio.read(range, cancellation),
        }
    }

    fn reset(&mut self) -> Result<()> {
        match self {
            Self::Track(audio) => audio.reset(),
            Self::Silent(audio) => audio.reset(),
        }
    }
}

impl PrefetchAudioSource for PlaybackAudio {
    fn prefetch(&mut self, range: SampleRange) -> IoFuture<'_, ()> {
        match self {
            Self::Track(audio) => audio.prefetch(range),
            Self::Silent(audio) => audio.prefetch(range),
        }
    }
}

/// The clock that times playback of an input with no audio track.
pub(crate) enum Clock {
    /// An `AudioContext`'s `currentTime`, with nothing scheduled on it.
    AudioContext(BaseAudioContext),
    /// The page's `performance.now()`.
    Page,
}

impl Clock {
    fn seconds(&self) -> f64 {
        match self {
            Self::AudioContext(context) => context.current_time(),
            Self::Page => clock_seconds(),
        }
    }
}

/// Times playback of an input with no audio track, and plays nothing.
pub(crate) struct ClockBackend {
    clock: Clock,
    rate: u32,
}

impl AudioOutputBackend for ClockBackend {
    fn clock_samples(&self) -> u64 {
        (self.clock.seconds() * f64::from(self.rate)).floor() as u64
    }

    fn start(&mut self, _: u64) -> Result<()> {
        Ok(())
    }

    fn schedule(&mut self, _: AudioBuffer, _: u64) -> Result<()> {
        Ok(())
    }

    fn cancel_queued(&mut self, _: u64) -> Result<()> {
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Where a playback's audio goes, and the clock that times it.
pub(crate) enum BrowserOutput {
    Audio(AudioContextBackend),
    Clock(ClockBackend),
}

impl AudioOutputBackend for BrowserOutput {
    fn clock_samples(&self) -> u64 {
        match self {
            Self::Audio(output) => output.clock_samples(),
            Self::Clock(output) => output.clock_samples(),
        }
    }

    fn start(&mut self, media_sample: u64) -> Result<()> {
        match self {
            Self::Audio(output) => output.start(media_sample),
            Self::Clock(output) => output.start(media_sample),
        }
    }

    fn schedule(&mut self, buffer: AudioBuffer, generation: u64) -> Result<()> {
        match self {
            Self::Audio(output) => output.schedule(buffer, generation),
            Self::Clock(output) => output.schedule(buffer, generation),
        }
    }

    fn cancel_queued(&mut self, generation: u64) -> Result<()> {
        match self {
            Self::Audio(output) => output.cancel_queued(generation),
            Self::Clock(output) => output.cancel_queued(generation),
        }
    }

    fn stop(&mut self) -> Result<()> {
        match self {
            Self::Audio(output) => output.stop(),
            Self::Clock(output) => output.stop(),
        }
    }
}

type Controller =
    PlaybackController<BrowserVideoSource, PlaybackAudio, WebAudioOutput<BrowserOutput>>;

/// The budgets and audio context a playback is opened with.
struct OnDemandOptions {
    audio_context: Option<BaseAudioContext>,
    video_budget_bytes: u64,
    audio_budget_bytes: u64,
}

fn parse_budget(options: &JsValue, name: &str, default: u64) -> std::result::Result<u64, JsValue> {
    match property(options, name)? {
        Some(value) => parse_u64(&value, name),
        None => Ok(default),
    }
}

fn parse_on_demand_options(
    options: Option<JsValue>,
) -> std::result::Result<OnDemandOptions, JsValue> {
    let options = options.unwrap_or_else(|| Object::new().into());
    let audio_context = property(&options, "audioContext")?
        .map(|context| {
            context.dyn_into::<BaseAudioContext>().map_err(|_| {
                js_error(
                    ErrorKind::InvalidInput,
                    "audioContext must be an AudioContext or OfflineAudioContext",
                )
            })
        })
        .transpose()?;
    Ok(OnDemandOptions {
        audio_context,
        video_budget_bytes: parse_budget(&options, "videoBudgetBytes", DEFAULT_VIDEO_BUDGET_BYTES)?,
        audio_budget_bytes: parse_budget(&options, "audioBudgetBytes", DEFAULT_AUDIO_BUDGET_BYTES)?,
    })
}

fn missing_audio_context() -> JsValue {
    js_error(
        ErrorKind::InvalidInput,
        "on-demand playback of an input with audio requires an audioContext, which plays the \
         audio and whose clock times it",
    )
}

/// Which video decoders [`WasmOnDemandPlayback::open_with`] may choose between.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VideoBackend {
    /// `WebCodecs` when the browser supports the track, software otherwise.
    Automatic,
    /// Software whatever the browser supports, so a test covers it even in a
    /// browser that decodes the codec itself.
    #[cfg(all(test, feature = "all"))]
    SoftwareOnly,
}

/// Playback of an MP4 or WebM input read on demand through a URL, `Blob` or
/// range reader, timed by a Web Audio context, or by the page's clock for a
/// video with no audio track and no context.
///
/// `play`, `pause`, `seek`, `present` and `currentFrame` never wait on the
/// network. When one needs a sample that has not been loaded, it throws an
/// error whose code is `WOULD_BLOCK` and leaves playback as it was;
/// `prefetch()` loads what it needs and resolves, and the call then succeeds.
#[wasm_bindgen(js_name = OnDemandPlayback)]
pub struct WasmOnDemandPlayback {
    /// `None` while a prefetch holds the controller.
    controller: Rc<RefCell<Option<Controller>>>,
    pending_prefetch: Rc<RefCell<Option<Promise>>>,
    state: Rc<Cell<bool>>,
    source: RangeSource,
    frame_count: u64,
    width: u32,
    height: u32,
    /// The audio track's sample rate, or `None` for an input with no audio.
    sample_rate: Option<u32>,
    webcodecs_video: bool,
    video_budget_bytes: u64,
    audio_budget_bytes: u64,
}

impl WasmOnDemandPlayback {
    pub(crate) async fn open_inner(
        source: JsValue,
        options: Option<JsValue>,
    ) -> std::result::Result<Self, JsValue> {
        Self::open_with(source, options, VideoBackend::Automatic).await
    }

    async fn open_with(
        source: JsValue,
        options: Option<JsValue>,
        backend: VideoBackend,
    ) -> std::result::Result<Self, JsValue> {
        let options = parse_on_demand_options(options)?;
        let limits = Limits::default();
        let source = RangeSource::open(source).await.map_err(core_error)?;
        let index_source =
            CachingByteSource::new(source.clone(), INDEX_PAGE_BYTES, INDEX_CACHE_BYTES)
                .map_err(core_error)?;
        let media = crate::container::open_media(&index_source, &limits)
            .await
            .map_err(core_error)?;
        let video = media
            .first_track(TrackKind::Video)
            .ok_or_else(|| js_error(ErrorKind::Unsupported, "the input has no video track"))?
            .clone();
        let audio = media
            .first_track(TrackKind::Audio)
            .map(|audio| Ok((audio.clone(), media.audio_timing(audio)?)))
            .transpose()
            .map_err(core_error)?;
        // Checked before any audio is read, rather than once it has been.
        if audio.is_some() && options.audio_context.is_none() {
            return Err(missing_audio_context());
        }

        let dimensions = video
            .dimensions
            .ok_or_else(|| js_error(ErrorKind::MalformedMedia, "video track has no dimensions"))?;
        let derived =
            derive_codec_string(video.codec, &video.decoder_config).map_err(core_error)?;
        let video_loader =
            Mp4SampleLoader::new(video.clone(), source.clone(), options.video_budget_bytes)
                .map_err(core_error)?;
        // A codec `WebCodecs` has no registration for here falls to the
        // software decoder, which reports it if it cannot decode it either.
        let webcodecs = match webcodecs_config(&video, &derived, dimensions) {
            Ok(config) if backend == VideoBackend::Automatic => webcodecs_supports(&config)
                .await
                .map_err(core_error)?
                .then_some(config),
            _ => None,
        };
        let video_source = match webcodecs {
            Some(config) => BrowserVideoSource::WebCodecs(
                WebCodecsVideoSource::new(config, video_loader, limits).map_err(core_error)?,
            ),
            None => {
                // An AV1 or VP9 track's color range may be read from its
                // first sample.
                let mut leading = Vec::new();
                if matches!(video.codec, Codec::Av1 | Codec::Vp9) {
                    let first = video.samples.first().ok_or_else(|| {
                        js_error(ErrorKind::MalformedMedia, "video track contains no samples")
                    })?;
                    let mut data = vec![0_u8; first.size as usize];
                    video
                        .read_sample_into(&source, 0, &mut data)
                        .await
                        .map_err(core_error)?;
                    leading.push(EncodedVideoSample {
                        presentation_index: FrameIndex(0),
                        random_access: first.is_sync,
                        data,
                    });
                }
                let (factory, configuration) =
                    software_video_decoder(&video, derived.profile, dimensions, &leading, &limits)
                        .map_err(core_error)?;
                let video_reader = ExactFrameReader::from_provider(
                    factory.as_ref(),
                    configuration,
                    Box::new(video_loader.sample_provider().map_err(core_error)?),
                    limits,
                )
                .map_err(core_error)?;
                BrowserVideoSource::Software(OnDemandVideoSource::new(
                    video_reader,
                    video_loader,
                    VIDEO_READAHEAD_SAMPLES,
                ))
            }
        };
        let webcodecs_video = matches!(video_source, BrowserVideoSource::WebCodecs(_));
        let frame_count = video.presentation_order.len() as u64;

        let audio_source = match audio {
            Some((audio, timing)) => Some(
                BrowserAudioSource::open(
                    audio,
                    timing,
                    source.clone(),
                    options.audio_budget_bytes,
                    limits,
                )
                .await
                .map_err(core_error)?,
            ),
            None => None,
        };
        let sample_rate = audio_source.as_ref().map(BrowserAudioSource::sample_rate);
        let clock_rate = sample_rate.unwrap_or(VIDEO_ONLY_CLOCK_RATE);

        let timeline = IndexedPresentationTimeline::from_mp4_track(&video, clock_rate, &limits)
            .map_err(core_error)?;
        let (audio_source, output) = match audio_source {
            Some(audio) => (
                PlaybackAudio::Track(audio),
                BrowserOutput::Audio(AudioContextBackend::new(
                    options.audio_context.ok_or_else(missing_audio_context)?,
                    clock_rate,
                )),
            ),
            None => (
                PlaybackAudio::Silent(SilentAudioSource {
                    sample_rate: clock_rate,
                    length: timeline.end_sample(),
                    limits,
                }),
                BrowserOutput::Clock(ClockBackend {
                    clock: options
                        .audio_context
                        .map_or(Clock::Page, Clock::AudioContext),
                    rate: clock_rate,
                }),
            ),
        };
        let controller = PlaybackController::new_with_indexed_timeline(
            video_source,
            audio_source,
            WebAudioOutput(output),
            timeline,
            PlaybackOptions::for_sample_rate(clock_rate),
        )
        .map_err(core_error)?;
        Ok(Self {
            controller: Rc::new(RefCell::new(Some(controller))),
            pending_prefetch: Rc::new(RefCell::new(None)),
            state: Rc::new(Cell::new(false)),
            source,
            frame_count,
            width: dimensions.width,
            height: dimensions.height,
            sample_rate,
            webcodecs_video,
            video_budget_bytes: options.video_budget_bytes,
            audio_budget_bytes: options.audio_budget_bytes,
        })
    }

    /// Runs `step` on the controller, unless the playback is closed or a
    /// prefetch holds it, in which case the step would have had to wait.
    fn with_controller<T>(
        &self,
        step: impl FnOnce(&mut Controller) -> Result<T>,
    ) -> std::result::Result<T, JsValue> {
        ensure_open(&self.state)?;
        let mut controller = self.controller.borrow_mut();
        let controller = controller.as_mut().ok_or_else(|| {
            js_error(
                ErrorKind::WouldBlock,
                "a prefetch is still loading; try again once it resolves",
            )
        })?;
        step(controller).map_err(core_error)
    }
}

fn picture(frame: &crate::VideoFrame) -> std::result::Result<JsValue, JsValue> {
    WasmVideoFrame::rgba(
        frame.dimensions.width,
        frame.dimensions.height,
        owned_u8_array(&packed_rgba(frame)),
    )
    .map(JsValue::from)
}

fn set(target: &Object, name: &str, value: &JsValue) -> std::result::Result<(), JsValue> {
    Reflect::set(target, &JsValue::from_str(name), value).map(|_| ())
}

#[wasm_bindgen(js_class = OnDemandPlayback)]
impl WasmOnDemandPlayback {
    /// Opens `source` for on-demand playback: a URL read with HTTP range
    /// requests, a `Blob` or `File`, or an object with a numeric `size` and a
    /// `read(offset, length)` method resolving to the bytes.
    ///
    /// `options.audioContext`, an `AudioContext`, is required when the input
    /// has an audio track: playback is timed by its clock and plays through
    /// its destination. An input with no audio track is timed by the
    /// context's clock when one is given, and by `performance.now()`
    /// otherwise. `options.videoBudgetBytes` and `options.audioBudgetBytes`
    /// bound the compressed samples held at once, and default to 16 MiB and
    /// 1 MiB.
    ///
    /// Only the container's header and sample index are read here, and the
    /// first bytes of an Opus track's last packet. The video decodes through `WebCodecs` when the
    /// browser supports the track, and on the crate's software decoder
    /// otherwise; `videoDecoder` says which. AAC audio decodes through
    /// `WebCodecs`, and Opus audio through `WebCodecs` where the browser
    /// supports it and the crate's software decoder otherwise.
    pub fn open(source: JsValue, options: Option<JsValue>) -> Promise {
        future_to_promise(async move { Ok(Self::open_inner(source, options).await?.into()) })
    }

    #[wasm_bindgen(getter, js_name = frameCount)]
    pub fn frame_count(&self) -> JsValue {
        bigint_u64(self.frame_count)
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The audio track's sample rate, or `0` for an input with no audio.
    #[wasm_bindgen(getter, js_name = sampleRate)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate.unwrap_or(0)
    }

    /// Whether the input has an audio track, which plays with the video.
    #[wasm_bindgen(getter, js_name = hasAudio)]
    pub fn has_audio(&self) -> bool {
        self.sample_rate.is_some()
    }

    /// `"webcodecs"` when the video decodes through the browser's `WebCodecs`
    /// decoder, or `"software"` when it decodes on the crate's own.
    #[wasm_bindgen(getter, js_name = videoDecoder)]
    pub fn video_decoder(&self) -> String {
        if self.webcodecs_video {
            "webcodecs".to_owned()
        } else {
            "software".to_owned()
        }
    }

    #[wasm_bindgen(getter, js_name = videoBudgetBytes)]
    pub fn video_budget_bytes(&self) -> f64 {
        self.video_budget_bytes as f64
    }

    #[wasm_bindgen(getter, js_name = audioBudgetBytes)]
    pub fn audio_budget_bytes(&self) -> f64 {
        self.audio_budget_bytes as f64
    }

    /// Every byte read from the source so far, the container's header and
    /// index included.
    #[wasm_bindgen(getter, js_name = fetchedBytes)]
    pub fn fetched_bytes(&self) -> f64 {
        self.source.fetched_bytes() as f64
    }

    #[wasm_bindgen(getter, js_name = isPlaying)]
    pub fn is_playing(&self) -> bool {
        self.controller
            .borrow()
            .as_ref()
            .is_some_and(Controller::is_playing)
    }

    /// The presentation frame the playback clock is on.
    #[wasm_bindgen(js_name = currentFrameIndex)]
    pub fn current_frame_index(&self) -> std::result::Result<JsValue, JsValue> {
        self.with_controller(|controller| controller.current_frame_index())
            .map(|frame| bigint_u64(frame.0))
    }

    /// Starts playing from the current position. Throws `WOULD_BLOCK` until
    /// the audio it starts with has been loaded.
    pub fn play(&self) -> std::result::Result<(), JsValue> {
        self.with_controller(Controller::play)
    }

    pub fn pause(&self) -> std::result::Result<(), JsValue> {
        self.with_controller(Controller::pause)
    }

    /// Moves playback to `frameIndex`. A seek never needs a prefetch itself;
    /// the frame after it may.
    pub fn seek(&self, frame_index: JsValue) -> std::result::Result<(), JsValue> {
        let frame = FrameIndex(parse_u64(&frame_index, "frame index")?);
        self.with_controller(|controller| controller.seek(frame))
    }

    /// Tops the audio queue up and returns the frame the audio clock calls
    /// for, as `{ requestedFrame, frame, finished, picture }`: `frame` and
    /// `picture` are `null` when that frame was already presented, and
    /// `finished` is `true` once the clock passes the end.
    pub fn present(&self) -> std::result::Result<JsValue, JsValue> {
        let (presentation, frame) = self.with_controller(Controller::present)?;
        let result = Object::new();
        set(
            &result,
            "requestedFrame",
            &bigint_u64(presentation.requested_frame.0),
        )?;
        set(
            &result,
            "frame",
            &presentation
                .frame
                .map_or(JsValue::NULL, |frame| bigint_u64(frame.0)),
        )?;
        set(
            &result,
            "finished",
            &JsValue::from_bool(presentation.finished),
        )?;
        let picture = match frame {
            Some(frame) => picture(&frame)?,
            None => JsValue::NULL,
        };
        set(&result, "picture", &picture)?;
        Ok(result.into())
    }

    /// The frame at the current position, as a `VideoFrame`, whether or not
    /// playback is running.
    #[wasm_bindgen(js_name = currentFrame)]
    pub fn current_frame(&self) -> std::result::Result<JsValue, JsValue> {
        let frame = self.with_controller(Controller::current_frame)?;
        picture(&frame)
    }

    /// Loads what the next `present`, or while paused `currentFrame` and
    /// `play`, needs, along with a readahead, and resolves once it has.
    ///
    /// Calls made while it is loading throw `WOULD_BLOCK`. Calling it again
    /// before it resolves returns the same promise.
    pub fn prefetch(&self) -> Promise {
        if let Some(pending) = self.pending_prefetch.borrow().as_ref() {
            return pending.clone();
        }
        // The controller is taken now rather than when the promise first
        // runs, so every call from here until it settles reports the prefetch.
        let held = ensure_open(&self.state).and_then(|()| {
            self.controller
                .borrow_mut()
                .take()
                .ok_or_else(|| js_error(ErrorKind::InvalidState, "the playback is not available"))
        });
        let mut held = match held {
            Ok(held) => held,
            Err(error) => return Promise::reject(&error),
        };
        let controller = Rc::clone(&self.controller);
        let pending = Rc::clone(&self.pending_prefetch);
        let state = Rc::clone(&self.state);
        let promise = future_to_promise(async move {
            let result = held.prefetch().await;
            if state.get() {
                let _ = held.stop();
            } else {
                *controller.borrow_mut() = Some(held);
            }
            pending.borrow_mut().take();
            result.map_err(core_error)?;
            Ok(JsValue::UNDEFINED)
        });
        *self.pending_prefetch.borrow_mut() = Some(promise.clone());
        promise
    }

    #[wasm_bindgen(getter, js_name = isClosed)]
    pub fn is_closed(&self) -> bool {
        self.state.get()
    }

    /// Stops playback and releases the decoders and caches. A prefetch still
    /// loading is dropped when it settles.
    pub fn close(&mut self) {
        if !self.state.replace(true)
            && let Some(mut controller) = self.controller.borrow_mut().take()
        {
            let _ = controller.stop();
        }
    }
}

// The browser tests decode the bundled HEVC and AV1 samples, so they build
// with the whole codec matrix, as the other browser tests do.
#[cfg(all(test, feature = "all"))]
mod tests {
    use super::*;
    use crate::container::{MediaTracks, open_media};
    use crate::io::MemorySource;
    use crate::wasm_api::error_code;
    use crate::web_audio_decoder::WebAudioDecodeSession;
    use crate::web_decoder::{WebVideoDecodeSession, schedule_event_loop_tick};
    use js_sys::BigInt;
    use std::collections::HashMap;
    use wasm_bindgen_test::*;
    use web_sys::{AudioDecoderConfig as JsAudioDecoderConfig, OfflineAudioContext};

    wasm_bindgen_test_configure!(run_in_browser);

    const SAMPLE: &[u8] = include_bytes!("../examples/media/BigBuckBunny.mp4");

    /// A track `WebCodecs` decodes in the test browser, unlike the HEVC
    /// [`SAMPLE`], so the tests of the `WebCodecs` video path use it.
    const AV1_SAMPLE: &[u8] = include_bytes!("../examples/media/BigBuckBunny.av1.mp4");

    #[wasm_bindgen(module = "/js/browser.js")]
    extern "C" {
        #[wasm_bindgen(js_name = makeSuspendingReader)]
        fn make_suspending_reader(bytes: &[u8]) -> JsValue;

        #[wasm_bindgen(catch, js_name = makeObjectUrl)]
        fn make_object_url(bytes: &[u8], mime_type: &str) -> std::result::Result<String, JsValue>;

        #[wasm_bindgen(catch, js_name = makeBlob)]
        fn make_blob(bytes: &[u8], mime_type: &str) -> std::result::Result<web_sys::Blob, JsValue>;
    }

    fn assert_error_code(error: &JsValue, expected: &str) {
        assert_eq!(error_code(error).as_deref(), Some(expected));
    }

    /// An audio context whose clock stays at zero, since it never renders, so
    /// every frame a test presents is the one it seeked to.
    fn still_clock() -> BaseAudioContext {
        OfflineAudioContext::new_with_number_of_channels_and_length_and_sample_rate(
            2, 48_000, 48_000.0,
        )
        .unwrap()
        .into()
    }

    fn options(video_budget_bytes: u64) -> JsValue {
        let options = Object::new();
        set(&options, "audioContext", &still_clock()).unwrap();
        set(
            &options,
            "videoBudgetBytes",
            &JsValue::from_f64(video_budget_bytes as f64),
        )
        .unwrap();
        options.into()
    }

    async fn sample_tracks() -> (MediaTracks, MemorySource) {
        tracks_of(SAMPLE).await
    }

    async fn tracks_of(bytes: &[u8]) -> (MediaTracks, MemorySource) {
        let source = MemorySource::new(bytes.to_vec());
        let media = open_media(&source, &Limits::default()).await.unwrap();
        (media, source)
    }

    /// Opens `source` with its video on the software decoder, whatever this
    /// browser decodes through `WebCodecs`, so a test of that path compares
    /// against a software decode in every browser.
    async fn open_software(source: JsValue, options: JsValue) -> WasmOnDemandPlayback {
        let playback =
            WasmOnDemandPlayback::open_with(source, Some(options), VideoBackend::SoftwareOnly)
                .await
                .unwrap();
        assert_eq!(playback.video_decoder(), "software");
        playback
    }

    /// Whether this browser decodes the sample's AAC track through
    /// `WebCodecs`, which on-demand playback needs for its audio.
    async fn browser_decodes_sample_audio() -> bool {
        browser_decodes_audio_of(SAMPLE).await
    }

    async fn browser_decodes_audio_of(bytes: &[u8]) -> bool {
        let (media, _) = tracks_of(bytes).await;
        let audio = media.first_track(TrackKind::Audio).unwrap();
        let config = WebAudioDecoderConfig::for_track(audio).unwrap();
        let support: AudioDecoderSupport = JsFuture::from(js_to_promise(
            JsAudioDecoder::is_config_supported(&config.to_js()),
        ))
        .await
        .unwrap()
        .unchecked_into();
        let supported = support.get_supported().unwrap_or(false);
        if !supported {
            console_log!("skipped: this browser cannot decode AAC through WebCodecs");
        }
        supported
    }

    /// How many of the sample's leading frames [`eager_frames`] decodes.
    const EAGER_FRAMES: u64 = 4;

    thread_local! {
        static EAGER: RefCell<Option<Rc<Vec<Vec<u8>>>>> = const { RefCell::new(None) };
    }

    /// The sample's first [`EAGER_FRAMES`] frames, decoded in software from
    /// every sample held in memory: what on-demand playback must reproduce.
    /// Decoded once and shared, since a 1080p software decode is the slow
    /// part of these tests.
    async fn eager_frames() -> Rc<Vec<Vec<u8>>> {
        if let Some(frames) = EAGER.with_borrow(Clone::clone) {
            return frames;
        }
        let (media, source) = sample_tracks().await;
        let video = media.first_track(TrackKind::Video).unwrap();
        let limits = Limits::default();
        let samples = video
            .to_encoded_video_samples(&source, &limits)
            .await
            .unwrap();
        let derived = derive_codec_string(video.codec, &video.decoder_config).unwrap();
        let (factory, configuration) = software_video_decoder(
            video,
            derived.profile,
            video.dimensions.unwrap(),
            &samples,
            &limits,
        )
        .unwrap();
        let mut reader =
            ExactFrameReader::new(factory.as_ref(), configuration, samples, limits).unwrap();
        let cancellation = CancellationToken::new();
        let mut decoded = Vec::new();
        for frame in 0..EAGER_FRAMES {
            next_tick().await;
            decoded.push(packed_rgba(
                &reader.get(FrameIndex(frame), &cancellation).unwrap(),
            ));
        }
        let frames = Rc::new(decoded);
        EAGER.set(Some(Rc::clone(&frames)));
        frames
    }

    /// Lets the event loop run, so the test driver's polling is answered
    /// between software decodes rather than timing out behind them.
    async fn next_tick() {
        let promise = Promise::new(&mut |resolve, _| schedule_event_loop_tick(&resolve));
        JsFuture::from(promise).await.unwrap();
    }

    fn pixels(picture: &JsValue) -> Vec<u8> {
        Reflect::get(picture, &JsValue::from_str("pixels"))
            .unwrap()
            .unchecked_into::<Uint8Array>()
            .to_vec()
    }

    fn field(target: &JsValue, name: &str) -> JsValue {
        Reflect::get(target, &JsValue::from_str(name)).unwrap()
    }

    fn frame_of(presentation: &JsValue) -> u64 {
        parse_u64(&field(presentation, "frame"), "frame").unwrap()
    }

    /// Retries `step` until it stops throwing `WOULD_BLOCK`, awaiting a
    /// prefetch in between, as a page's animation-frame callback does.
    async fn until_loaded<T>(
        playback: &WasmOnDemandPlayback,
        step: impl Fn(&WasmOnDemandPlayback) -> std::result::Result<T, JsValue>,
    ) -> T {
        for _ in 0..32 {
            next_tick().await;
            match step(playback) {
                Err(error) if error_code(&error).as_deref() == Some("WOULD_BLOCK") => {
                    JsFuture::from(playback.prefetch()).await.unwrap();
                }
                result => return result.unwrap(),
            }
        }
        panic!("playback still had not loaded what it needed after 32 prefetches");
    }

    /// Plays, presents and seeks over a source whose every read suspends: no
    /// call waits on it, each reports what it is missing as `WOULD_BLOCK`, the
    /// prefetch loads it, and the frames are exactly the ones an eager decode
    /// of the whole file gives. Only a small part of the file is ever read.
    #[wasm_bindgen_test(async)]
    async fn plays_and_seeks_over_a_suspending_source_fetching_only_what_it_needs() {
        if !browser_decodes_sample_audio().await {
            return;
        }
        let reader = make_suspending_reader(SAMPLE);
        let mut playback = open_software(reader.clone(), options(1 << 20)).await;
        assert_eq!((playback.width(), playback.height()), (1920, 1080));
        assert_eq!(parse_u64(&playback.frame_count(), "frames").unwrap(), 768);
        assert!(
            playback.fetched_bytes() < 64.0 * 1024.0,
            "opening reads the movie header, not the media"
        );

        let expected = eager_frames().await;

        // Nothing is loaded yet, so starting playback says so rather than
        // waiting for its audio.
        assert_error_code(&playback.play().unwrap_err(), "WOULD_BLOCK");
        until_loaded(&playback, WasmOnDemandPlayback::play).await;
        assert!(playback.is_playing());

        let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
        assert_eq!(frame_of(&presentation), 0);
        assert_eq!(pixels(&field(&presentation, "picture")), expected[0]);
        // The clock has not moved, so there is no new frame to draw.
        let again = playback.present().unwrap();
        assert!(field(&again, "frame").is_null());
        assert!(field(&again, "picture").is_null());

        playback.seek(JsValue::from_f64(3.0)).unwrap();
        let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
        assert_eq!(frame_of(&presentation), 3);
        assert_eq!(pixels(&field(&presentation, "picture")), expected[3]);

        // While a prefetch is loading, playback calls report it instead of
        // waiting, and asking again shares the prefetch already running.
        let first = playback.prefetch();
        let second = playback.prefetch();
        assert!(Object::is(&first, &second));
        assert_error_code(&playback.present().unwrap_err(), "WOULD_BLOCK");
        JsFuture::from(first).await.unwrap();

        // Backwards, so the decode restarts from the random-access point.
        playback.seek(JsValue::from_f64(1.0)).unwrap();
        let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
        assert_eq!(frame_of(&presentation), 1);
        assert_eq!(pixels(&field(&presentation, "picture")), expected[1]);

        // Paused, the current frame comes through the same loop.
        playback.pause().unwrap();
        playback.seek(BigInt::from(2_u64).into()).unwrap();
        assert_eq!(
            parse_u64(&playback.current_frame_index().unwrap(), "frame").unwrap(),
            2
        );
        let picture = until_loaded(&playback, WasmOnDemandPlayback::current_frame).await;
        assert_eq!(pixels(&picture), expected[2]);

        let reads = field(&reader, "reads").as_f64().unwrap();
        assert!(reads > 0.0, "every byte came through the suspending reader");
        assert!(
            playback.fetched_bytes() < SAMPLE.len() as f64 / 2.0,
            "fetched {} of {} bytes",
            playback.fetched_bytes(),
            SAMPLE.len()
        );

        playback.close();
        assert!(playback.is_closed());
        assert_error_code(&playback.play().unwrap_err(), "INVALID_STATE");
    }

    /// Whether this browser decodes both of [`AV1_SAMPLE`]'s tracks through
    /// `WebCodecs`, which the tests of the `WebCodecs` video path need.
    async fn browser_decodes_av1_sample() -> bool {
        if !browser_decodes_audio_of(AV1_SAMPLE).await {
            return false;
        }
        let supported = WebVideoDecodeSession::decodes_through_webcodecs(AV1_SAMPLE, 0).await;
        if !supported {
            console_log!("skipped: this browser cannot decode the AV1 sample through WebCodecs");
        }
        supported
    }

    /// `frames` of [`AV1_SAMPLE`], decoded through `WebCodecs` from every
    /// sample held in memory: what on-demand playback must reproduce.
    async fn webcodecs_frames(frames: &[u64]) -> HashMap<u64, Vec<u8>> {
        let mut session = WebVideoDecodeSession::open(AV1_SAMPLE, 0, &Limits::default())
            .await
            .unwrap();
        let mut decoded = HashMap::new();
        for &frame in frames {
            let (_, rgba) = session
                .get(FrameIndex(frame), &CancellationToken::new())
                .await
                .unwrap();
            decoded.insert(frame, rgba);
        }
        decoded
    }

    /// Where the browser decodes the track, on-demand playback decodes its
    /// video through `WebCodecs` (issue #680): over a source whose every read
    /// suspends, no call waits on the decoder or the network, a frame not
    /// decoded yet reports `WOULD_BLOCK`, the prefetch decodes it along with
    /// the frames after it, and every frame is exactly the one `WebCodecs`
    /// decodes from the whole file in memory.
    #[wasm_bindgen_test(async)]
    async fn plays_and_seeks_through_webcodecs_over_a_suspending_source() {
        if !browser_decodes_av1_sample().await {
            return;
        }
        let reader = make_suspending_reader(AV1_SAMPLE);
        let mut playback = WasmOnDemandPlayback::open_inner(reader, Some(options(1 << 20)))
            .await
            .unwrap();
        assert_eq!(playback.video_decoder(), "webcodecs");
        let expected = webcodecs_frames(&[0, 1, 5, 40, 41]).await;

        assert_error_code(&playback.play().unwrap_err(), "WOULD_BLOCK");
        until_loaded(&playback, WasmOnDemandPlayback::play).await;
        let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
        assert_eq!(frame_of(&presentation), 0);
        assert_eq!(pixels(&field(&presentation, "picture")), expected[&0]);

        // The prefetch decoded the frames after it too, so one of those plays
        // without another.
        playback.seek(JsValue::from_f64(5.0)).unwrap();
        let presentation = playback.present().unwrap();
        assert_eq!(frame_of(&presentation), 5);
        assert_eq!(pixels(&field(&presentation, "picture")), expected[&5]);

        // Past the decoded frames, the frame is decoded on demand.
        playback.seek(JsValue::from_f64(40.0)).unwrap();
        assert_error_code(&playback.present().unwrap_err(), "WOULD_BLOCK");
        let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
        assert_eq!(frame_of(&presentation), 40);
        assert_eq!(pixels(&field(&presentation, "picture")), expected[&40]);

        // Backwards to a frame the decoder already emitted, which takes a
        // fresh decode from the random-access point.
        playback.seek(JsValue::from_f64(1.0)).unwrap();
        let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
        assert_eq!(frame_of(&presentation), 1);
        assert_eq!(pixels(&field(&presentation, "picture")), expected[&1]);

        playback.pause().unwrap();
        playback.seek(JsValue::from_f64(41.0)).unwrap();
        let picture = until_loaded(&playback, WasmOnDemandPlayback::current_frame).await;
        assert_eq!(pixels(&picture), expected[&41]);
        assert!(
            playback.fetched_bytes() < AV1_SAMPLE.len() as f64 / 2.0,
            "fetched {} of {} bytes",
            playback.fetched_bytes(),
            AV1_SAMPLE.len()
        );
        playback.close();
    }

    /// The `WebCodecs` video source holds at most [`DECODED_VIDEO_FRAMES`]
    /// decoded pictures, from the frame it was last asked to prefetch, and
    /// plays sequentially through more frames than the decoder has output
    /// buffers for, as it would stall if the frames it emits were held open.
    #[wasm_bindgen_test(async)]
    async fn webcodecs_video_holds_a_bounded_run_of_decoded_frames() {
        if !browser_decodes_av1_sample().await {
            return;
        }
        let (media, _) = tracks_of(AV1_SAMPLE).await;
        let video = media.first_track(TrackKind::Video).unwrap().clone();
        let derived = derive_codec_string(video.codec, &video.decoder_config).unwrap();
        let config = webcodecs_config(&video, &derived, video.dimensions.unwrap()).unwrap();
        let source = RangeSource::open(make_suspending_reader(AV1_SAMPLE))
            .await
            .unwrap();
        let loader = Mp4SampleLoader::new(video, source, 512 * 1024).unwrap();
        let mut video = WebCodecsVideoSource::new(config, loader, Limits::default()).unwrap();
        let expected = webcodecs_frames(&[0, 2, 7, 47]).await;
        let cancellation = CancellationToken::new();

        assert_eq!(
            video
                .get_exact(FrameIndex(0), &cancellation)
                .unwrap_err()
                .kind(),
            ErrorKind::WouldBlock
        );
        video.prefetch(FrameIndex(0)).await.unwrap();
        assert_eq!(video.decoded.len() as u64, DECODED_VIDEO_FRAMES);
        for frame in 0..DECODED_VIDEO_FRAMES {
            video.get_exact(FrameIndex(frame), &cancellation).unwrap();
        }
        assert_eq!(
            video
                .get_exact(FrameIndex(DECODED_VIDEO_FRAMES), &cancellation)
                .unwrap_err()
                .kind(),
            ErrorKind::WouldBlock
        );

        let mut prefetches = 0;
        for frame in 0..48 {
            let frame = FrameIndex(frame);
            let picture = loop {
                match video.get_exact(frame, &cancellation) {
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        prefetches += 1;
                        video.prefetch(frame).await.unwrap();
                    }
                    result => break result.unwrap(),
                }
            };
            if let Some(expected) = expected.get(&frame.0) {
                assert_eq!(&packed_rgba(&picture), expected, "frame {}", frame.0);
            }
            // A page prefetches ahead between frames, as a paced playback
            // does.
            video.prefetch(frame).await.unwrap();
            assert!(video.decoded.len() as u64 <= DECODED_VIDEO_FRAMES);
            assert!(video.decoded.keys().all(|&index| index >= frame));
            assert!(video.loader.resident_bytes() <= video.loader.budget_bytes());
        }
        assert!(
            prefetches < 48 / 4,
            "{prefetches} frames had to wait for a prefetch"
        );

        // Back to a frame the decoder already emitted and dropped.
        video.reset().unwrap();
        video.prefetch(FrameIndex(2)).await.unwrap();
        let picture = video.get_exact(FrameIndex(2), &cancellation).unwrap();
        assert_eq!(packed_rgba(&picture), expected[&2]);
        assert!(video.decoded.keys().all(|&index| index >= FrameIndex(2)));
    }

    /// A URL is read with HTTP range requests, and a `Blob` by slicing it; a
    /// `blob:` URL answers range requests as an HTTP server does.
    #[wasm_bindgen_test(async)]
    async fn opens_a_url_and_a_blob_without_reading_them_whole() {
        if !browser_decodes_sample_audio().await {
            return;
        }
        let expected = eager_frames().await;
        let url = make_object_url(SAMPLE, "video/mp4").unwrap();
        let blob = make_blob(SAMPLE, "video/mp4").unwrap();
        for source in [JsValue::from_str(&url), blob.into()] {
            let playback = open_software(source, options(1 << 20)).await;
            assert_eq!(parse_u64(&playback.frame_count(), "frames").unwrap(), 768);
            let picture = until_loaded(&playback, WasmOnDemandPlayback::current_frame).await;
            assert_eq!(pixels(&picture), expected[0]);
            assert!(
                playback.fetched_bytes() < SAMPLE.len() as f64 / 2.0,
                "fetched {} of {} bytes",
                playback.fetched_bytes(),
                SAMPLE.len()
            );
        }
    }

    /// The samples of `range` from an eager decode of the track that starts
    /// at `from`.
    async fn eager_from(
        eager: &mut WebAudioDecodeSession,
        from: u64,
        range: SampleRange,
    ) -> Vec<f32> {
        let decoded = eager
            .get_range(
                SampleRange::new(from, range.end).unwrap(),
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        let channels = usize::from(decoded.channels);
        decoded.samples[(range.start - from) as usize * channels..].to_vec()
    }

    /// The audio source decodes a readahead past each range it is asked to
    /// prefetch, continues that run rather than decoding it again, drops what
    /// playback has moved past, and starts over at a seek.
    ///
    /// Every read is exactly what the eager session decodes from the same
    /// starting point. An AAC decode that starts from its preroll differs
    /// slightly from one that ran through it, so each read is checked against
    /// an eager decode that starts where the run it comes from did.
    #[wasm_bindgen_test(async)]
    async fn audio_reads_come_from_decoded_readahead_and_match_an_eager_decode() {
        if !browser_decodes_sample_audio().await {
            return;
        }
        let (media, _) = sample_tracks().await;
        let audio = media.first_track(TrackKind::Audio).unwrap();
        let limits = Limits::default();
        let source = RangeSource::open(make_suspending_reader(SAMPLE))
            .await
            .unwrap();
        let mut audio = BrowserAudioSource::open(
            audio.clone(),
            media.audio_timing(audio).unwrap(),
            source,
            256 * 1024,
            limits,
        )
        .await
        .unwrap();
        let mut eager = WebAudioDecodeSession::open(SAMPLE, 0, &limits)
            .await
            .unwrap();
        let rate = u64::from(audio.sample_rate());
        let window = rate / 5;
        let cancellation = CancellationToken::new();
        let range = |start: u64, end: u64| SampleRange::new(start, end).unwrap();
        let read =
            |audio: &mut BrowserAudioSource, range: SampleRange| audio.read(range, &cancellation);

        let first = range(0, window);
        assert_eq!(
            read(&mut audio, first).unwrap_err().kind(),
            ErrorKind::WouldBlock
        );
        audio.prefetch(first).await.unwrap();
        assert_eq!(
            read(&mut audio, first).unwrap().samples,
            eager_from(&mut eager, 0, first).await
        );
        // A readahead of a second was decoded with it.
        let ahead = range(rate, rate + window / 2);
        assert_eq!(
            read(&mut audio, ahead).unwrap().samples,
            eager_from(&mut eager, 0, ahead).await
        );

        // Most of the readahead played: the run is continued past its end,
        // and what came before the range asked for is dropped.
        let continued_from = audio.decoded.as_ref().unwrap().range.end;
        let later = range(rate - window, rate);
        audio.prefetch(later).await.unwrap();
        let decoded = audio.decoded.as_ref().unwrap().range;
        assert_eq!(decoded.start, later.start);
        assert!(decoded.end >= later.end + rate);
        assert_eq!(
            read(&mut audio, first).unwrap_err().kind(),
            ErrorKind::WouldBlock
        );
        assert_eq!(
            read(&mut audio, later).unwrap().samples,
            eager_from(&mut eager, 0, later).await,
            "the samples decoded before the continuation are kept"
        );
        let past = range(decoded.end - window, decoded.end);
        assert_eq!(
            read(&mut audio, past).unwrap().samples,
            eager_from(&mut eager, continued_from, past).await
        );

        // A seek far ahead starts a new run there.
        audio.reset().unwrap();
        let seeked = range(10 * rate, 10 * rate + window);
        assert_eq!(
            read(&mut audio, seeked).unwrap_err().kind(),
            ErrorKind::WouldBlock
        );
        audio.prefetch(seeked).await.unwrap();
        assert_eq!(
            read(&mut audio, seeked).unwrap().samples,
            eager_from(&mut eager, seeked.start, seeked).await
        );
        assert_eq!(audio.decoded.as_ref().unwrap().range.start, seeked.start);
        assert!(audio.loader.resident_bytes() <= audio.loader.budget_bytes());
    }

    /// The Web Audio backend plays each buffer at the context time its media
    /// position falls on, relative to the last `start`, and a seek's
    /// `cancel_queued` silences what the old position scheduled.
    #[wasm_bindgen_test(async)]
    async fn audio_buffers_play_at_their_media_position_until_canceled() {
        const RATE: u32 = 48_000;
        let render = |cancel: bool| async move {
            let context =
                OfflineAudioContext::new_with_number_of_channels_and_length_and_sample_rate(
                    1,
                    RATE,
                    RATE as f32,
                )
                .unwrap();
            let mut backend = AudioContextBackend::new((*context).clone(), RATE);
            // Media sample 9_600 starts now, at context time zero, so media
            // sample 14_400 plays 4_800 samples in.
            backend.start(9_600).unwrap();
            let range = SampleRange::new(14_400, 16_800).unwrap();
            let samples = (0..range.len())
                .map(|index| index as f32 / 4_096.0)
                .collect();
            let buffer = AudioBuffer::new(range, RATE, 1, samples, &Limits::default()).unwrap();
            backend.schedule(buffer, 0).unwrap();
            if cancel {
                backend.cancel_queued(1).unwrap();
            }
            let rendered: web_sys::AudioBuffer = JsFuture::from(context.start_rendering().unwrap())
                .await
                .unwrap()
                .unchecked_into();
            rendered.get_channel_data(0).unwrap()
        };

        let played = render(false).await;
        assert!(played[..4_800].iter().all(|&sample| sample == 0.0));
        for (index, &sample) in played[4_800..7_200].iter().enumerate() {
            assert!(
                (sample - index as f32 / 4_096.0).abs() < 1e-4,
                "sample {index} of the buffer played as {sample}"
            );
        }
        assert!(played[7_200..].iter().all(|&sample| sample == 0.0));

        let canceled = render(true).await;
        assert!(canceled.iter().all(|&sample| sample == 0.0));
    }

    /// Frames per second of [`small_mp4`]'s video, and how many frames it has.
    const SMALL_RATE: u64 = 30;
    const SMALL_FRAMES: u64 = 90;

    /// The gray levels of frame `index` of [`small_mp4`], a moving gradient.
    fn small_gray(index: u64) -> Vec<u8> {
        (0..18_u32)
            .flat_map(|y| {
                (0..32_u32).map(move |x| ((x * 7 + y * 3 + index as u32 * 5) % 256) as u8)
            })
            .collect()
    }

    /// Frame `index` of [`small_mp4`] as on-demand playback presents it.
    fn small_rgba(index: u64) -> Vec<u8> {
        small_gray(index)
            .into_iter()
            .flat_map(|level| [level, level, level, 255])
            .collect()
    }

    thread_local! {
        /// [`small_mp4`]'s files, without and with Opus, once encoded.
        static SMALL: RefCell<[Option<Rc<Vec<u8>>>; 2]> = const { RefCell::new([None, None]) };
    }

    /// Three seconds of lossless monochrome 32x18 AV1 at 30 fps, which
    /// decodes quickly and exactly, and with `opus`, three seconds of Opus
    /// muxed beside it with its pre-skip and end trim. Encoded once and
    /// shared, as [`eager_frames`] is.
    async fn small_mp4(opus: bool) -> Rc<Vec<u8>> {
        if let Some(bytes) = SMALL.with_borrow(|small| small[usize::from(opus)].clone()) {
            return bytes;
        }
        let bytes = Rc::new(encode_small_mp4(opus).await);
        SMALL.with_borrow_mut(|small| small[usize::from(opus)] = Some(Rc::clone(&bytes)));
        bytes
    }

    async fn encode_small_mp4(opus: bool) -> Vec<u8> {
        use crate::codec::{VideoEncoderConfig, VideoEncoderFactory};
        use crate::io::MemorySink;
        use crate::media::{ColorRange, PixelFormat, Plane, VideoDimensions};
        use crate::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
        use crate::transfer::{CpuFrameSource, FrameSource, Orientation};
        use crate::{CodecProfile, HardwarePreference, VideoFrame};

        let limits = Limits::default();
        let dimensions = VideoDimensions::new(32, 18, &limits).unwrap();
        let mut encoder = crate::native_av1_video_encoder_factory()
            .create(
                &VideoEncoderConfig {
                    codec: Codec::Av1,
                    profile: CodecProfile::Av1Main,
                    coded_dimensions: dimensions,
                    input_format: PixelFormat::Gray8,
                    color_range: ColorRange::Full,
                    hardware: HardwarePreference::Avoid,
                    timescale: SMALL_RATE as u32,
                    frame_duration: 1,
                    configuration: Vec::new(),
                },
                &limits,
            )
            .unwrap();
        let mut tracks = vec![Mp4TrackConfig {
            encoder: encoder.config().clone(),
            format: Mp4TrackFormat::Video(dimensions),
        }];
        let audio = if opus {
            let (track, packets, gapless, _) =
                crate::web_audio_decoder::tests::opus_packets(48_000 * SMALL_FRAMES / SMALL_RATE)
                    .await;
            tracks.push(track);
            Some((packets, gapless))
        } else {
            None
        };
        let mut muxer = Mp4Muxer::new(MemorySink::new(), tracks, 100_000)
            .await
            .unwrap();
        let mut video = Vec::new();
        for index in 0..SMALL_FRAMES {
            let frame = VideoFrame::new(
                dimensions,
                PixelFormat::Gray8,
                ColorRange::Full,
                vec![Plane {
                    data: small_gray(index),
                    stride: dimensions.width as usize,
                }],
                &limits,
            )
            .unwrap();
            video.extend(
                encoder
                    .encode(
                        FrameIndex(index),
                        FrameSource::Cpu(CpuFrameSource {
                            frame: &frame,
                            orientation: Orientation::TopLeft,
                        }),
                    )
                    .await
                    .unwrap(),
            );
        }
        video.extend(encoder.finish().await.unwrap());
        for sample in video {
            muxer.write_sample(0, sample).await.unwrap();
        }
        if let Some((packets, gapless)) = audio {
            for packet in packets {
                muxer.write_sample(1, packet).await.unwrap();
            }
            muxer.set_audio_gapless(1, gapless).unwrap();
        }
        muxer.finish().await.unwrap().into_inner()
    }

    /// [`small_mp4`]'s three seconds of moving gradient as a WebM, its video
    /// encoded as `codec` (VP8 or VP9, both lossy) and with `opus`, the same
    /// Opus track muxed beside it with its `CodecDelay` and end trim.
    async fn small_webm(codec: Codec, opus: bool) -> Vec<u8> {
        use crate::codec::{VideoEncoderConfig, VideoEncoderFactory};
        use crate::io::MemorySink;
        use crate::media::{ColorRange, PixelFormat, Plane, VideoDimensions};
        use crate::mp4::{Mp4TrackConfig, Mp4TrackFormat};
        use crate::transfer::{CpuFrameSource, FrameSource, Orientation};
        use crate::{CodecProfile, HardwarePreference, VideoFrame, WebmMuxer};

        let limits = Limits::default();
        let dimensions = VideoDimensions::new(32, 18, &limits).unwrap();
        let (factory, profile): (Box<dyn VideoEncoderFactory>, _) = match codec {
            Codec::Vp8 => (
                Box::new(crate::native_vp8_video_encoder_factory()),
                CodecProfile::Vp8,
            ),
            Codec::Vp9 => (
                Box::new(crate::native_vp9_video_encoder_factory()),
                CodecProfile::Vp9Profile0,
            ),
            _ => unreachable!("small_webm encodes VP8 or VP9"),
        };
        let mut encoder = factory
            .create(
                &VideoEncoderConfig {
                    codec,
                    profile,
                    coded_dimensions: dimensions,
                    input_format: PixelFormat::Rgba8,
                    color_range: ColorRange::Limited,
                    hardware: HardwarePreference::Avoid,
                    timescale: SMALL_RATE as u32,
                    frame_duration: 1,
                    configuration: Vec::new(),
                },
                &limits,
            )
            .unwrap();
        let mut tracks = vec![Mp4TrackConfig {
            encoder: encoder.config().clone(),
            format: Mp4TrackFormat::Video(dimensions),
        }];
        let audio = if opus {
            let (track, packets, gapless, _) =
                crate::web_audio_decoder::tests::opus_packets(48_000 * SMALL_FRAMES / SMALL_RATE)
                    .await;
            tracks.push(track);
            Some((packets, gapless))
        } else {
            None
        };
        let mut muxer = WebmMuxer::new(MemorySink::new(), tracks, 100_000)
            .await
            .unwrap();
        let mut video = Vec::new();
        for index in 0..SMALL_FRAMES {
            let frame = VideoFrame::new(
                dimensions,
                PixelFormat::Rgba8,
                ColorRange::Limited,
                vec![Plane {
                    data: small_rgba(index),
                    stride: dimensions.width as usize * 4,
                }],
                &limits,
            )
            .unwrap();
            video.extend(
                encoder
                    .encode(
                        FrameIndex(index),
                        FrameSource::Cpu(CpuFrameSource {
                            frame: &frame,
                            orientation: Orientation::TopLeft,
                        }),
                    )
                    .await
                    .unwrap(),
            );
        }
        video.extend(encoder.finish().await.unwrap());
        // A WebM cluster interleaves its tracks, so the samples are written
        // in presentation-time order across both.
        let seconds = |pts: i64, timescale: u64| pts as f64 / timescale as f64;
        let mut samples: Vec<_> = video
            .into_iter()
            .map(|sample| (seconds(sample.pts, SMALL_RATE), 0, sample))
            .collect();
        let gapless = audio.map(|(packets, gapless)| {
            samples.extend(
                packets
                    .into_iter()
                    .map(|packet| (seconds(packet.pts, 48_000), 1, packet)),
            );
            gapless
        });
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, track, sample) in samples {
            muxer.write_sample(track, sample).await.unwrap();
        }
        if let Some(gapless) = gapless {
            muxer.set_audio_gapless(1, gapless).unwrap();
        }
        muxer.finish().await.unwrap().into_inner()
    }

    /// Every frame of `bytes`'s video, decoded in software from every sample
    /// held in memory: what on-demand playback of a lossy input must
    /// reproduce.
    async fn eager_frames_of(bytes: &[u8]) -> Vec<Vec<u8>> {
        let (media, source) = tracks_of(bytes).await;
        let video = media.first_track(TrackKind::Video).unwrap();
        let limits = Limits::default();
        let samples = video
            .to_encoded_video_samples(&source, &limits)
            .await
            .unwrap();
        let derived = derive_codec_string(video.codec, &video.decoder_config).unwrap();
        let (factory, configuration) = software_video_decoder(
            video,
            derived.profile,
            video.dimensions.unwrap(),
            &samples,
            &limits,
        )
        .unwrap();
        let mut reader =
            ExactFrameReader::new(factory.as_ref(), configuration, samples, limits).unwrap();
        let cancellation = CancellationToken::new();
        let mut decoded = Vec::new();
        for frame in 0..video.presentation_order.len() as u64 {
            decoded.push(packed_rgba(
                &reader.get(FrameIndex(frame), &cancellation).unwrap(),
            ));
        }
        decoded
    }

    /// Opens one of [`small_mp4`]'s files on the software video decoder,
    /// which decodes it back to exactly the gray levels it was encoded from;
    /// these tests are about its audio and its clock, not how its video
    /// decodes.
    async fn open_small(
        source: JsValue,
        options: Option<JsValue>,
    ) -> std::result::Result<WasmOnDemandPlayback, JsValue> {
        WasmOnDemandPlayback::open_with(source, options, VideoBackend::SoftwareOnly).await
    }

    /// Resolves after `milliseconds` of real time.
    async fn sleep(milliseconds: f64) {
        let promise = Promise::new(&mut |resolve, _| {
            let set_timeout: js_sys::Function =
                Reflect::get(&js_sys::global(), &JsValue::from_str("setTimeout"))
                    .unwrap()
                    .unchecked_into();
            set_timeout
                .call2(&JsValue::NULL, &resolve, &JsValue::from_f64(milliseconds))
                .unwrap();
        });
        JsFuture::from(promise).await.unwrap();
    }

    /// The largest difference between two runs of samples of the same length.
    fn largest_difference(got: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(got.len(), expected.len());
        got.iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    /// Issue #681: an Opus track opens without its packets being read, its
    /// reads come from packets loaded as playback reaches them, and they are
    /// the samples an eager decode of the whole track gives, through
    /// `WebCodecs` where the browser decodes Opus and through the software
    /// decoder either way.
    #[wasm_bindgen_test(async)]
    async fn opus_audio_reads_load_packets_on_demand_and_match_an_eager_decode() {
        // Issue #685: a WebM's Opus track, timed by its `CodecDelay` rather
        // than an edit list, loads and decodes the same way.
        let mp4 = small_mp4(true).await;
        let webm = Rc::new(small_webm(Codec::Vp9, true).await);
        for bytes in [mp4, webm] {
            opus_reads_match_an_eager_decode(&bytes).await;
        }
    }

    async fn opus_reads_match_an_eager_decode(bytes: &[u8]) {
        let source = MemorySource::new(bytes.to_vec());
        let media = open_media(&source, &Limits::default()).await.unwrap();
        let track = media.first_track(TrackKind::Audio).unwrap().clone();
        assert_eq!(track.codec, Codec::Opus);
        let packets = track.samples.len();
        let limits = Limits::default();
        let mut eager = WebAudioDecodeSession::open(bytes, 0, &limits)
            .await
            .unwrap();
        let length = eager.presentation_length();
        assert_eq!(length, 48_000 * SMALL_FRAMES / SMALL_RATE);
        let range = |start: u64, end: u64| SampleRange::new(start, end).unwrap();
        let cancellation = CancellationToken::new();

        for software in [false, true] {
            let reader = make_suspending_reader(bytes);
            let mut audio = BrowserAudioSource::open(
                track.clone(),
                media.audio_timing(&track).unwrap(),
                RangeSource::open(reader.clone()).await.unwrap(),
                64 * 1024,
                limits,
            )
            .await
            .unwrap();
            assert_eq!(
                field(&reader, "reads").as_f64().unwrap(),
                1.0,
                "opening read the last packet's header and nothing else"
            );
            if software {
                audio.use_software();
            }
            assert_eq!(audio.presentation_length(), length);
            assert_eq!(audio.sample_rate(), 48_000);

            for wanted in [
                range(0, 9_600),
                range(70_000, 80_000),
                range(20_000, 20_001),
            ] {
                audio.reset().unwrap();
                assert_eq!(
                    audio.read(wanted, &cancellation).unwrap_err().kind(),
                    ErrorKind::WouldBlock
                );
                audio.prefetch(wanted).await.unwrap();
                if wanted.start == 0 {
                    // The first second and a readahead, not the whole track.
                    assert!(!audio.loader.is_loaded(packets - 1));
                    assert!(
                        field(&reader, "reads").as_f64().unwrap() > 1.0,
                        "the prefetch read through the source"
                    );
                }
                let got = audio.read(wanted, &cancellation).unwrap().samples;
                let expected = eager
                    .get_range(wanted, &cancellation)
                    .await
                    .unwrap()
                    .samples;
                // libopus and opus-pure agree to within a few 16-bit steps,
                // and either may decode each side.
                let largest = largest_difference(&got, &expected);
                assert!(
                    largest < 1e-3,
                    "[{}, {}) differs by {largest} (software: {software})",
                    wanted.start,
                    wanted.end
                );
            }
            // A read the browser decodes wrongly would have moved it to
            // software; a correct one stays where it started.
            assert_eq!(
                audio.is_software(),
                software || !browser_decodes_opus().await
            );
            assert!(audio.loader.resident_bytes() <= audio.loader.budget_bytes());
        }
    }

    /// Whether this browser decodes stereo Opus through `WebCodecs`.
    async fn browser_decodes_opus() -> bool {
        let config = JsAudioDecoderConfig::new("opus", 2, 48_000);
        let support: AudioDecoderSupport =
            JsFuture::from(js_to_promise(JsAudioDecoder::is_config_supported(&config)))
                .await
                .unwrap()
                .unchecked_into();
        support.get_supported().unwrap_or(false)
    }

    /// Issue #681: an MP4 whose audio is Opus plays, presents and seeks over
    /// a suspending source, with its audio scheduled on the context.
    #[wasm_bindgen_test(async)]
    async fn plays_an_mp4_with_opus_audio_over_a_suspending_source() {
        let bytes = small_mp4(true).await;
        let reader = make_suspending_reader(&bytes);
        let playback = open_small(reader, Some(options(1 << 20))).await.unwrap();
        assert!(playback.has_audio());
        assert_eq!(playback.sample_rate(), 48_000);
        assert_eq!(
            parse_u64(&playback.frame_count(), "frames").unwrap(),
            SMALL_FRAMES
        );

        until_loaded(&playback, WasmOnDemandPlayback::play).await;
        let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
        assert_eq!(frame_of(&presentation), 0);
        assert_eq!(pixels(&field(&presentation, "picture")), small_rgba(0));

        playback.seek(JsValue::from_f64(45.0)).unwrap();
        let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
        assert_eq!(frame_of(&presentation), 45);
        assert_eq!(pixels(&field(&presentation, "picture")), small_rgba(45));

        playback.pause().unwrap();
        playback.seek(JsValue::from_f64(7.0)).unwrap();
        let picture = until_loaded(&playback, WasmOnDemandPlayback::current_frame).await;
        assert_eq!(pixels(&picture), small_rgba(7));
        assert!(playback.fetched_bytes() < bytes.len() as f64);
    }

    /// Issue #681: an input with no audio track plays on an audio context's
    /// clock when it is given one, with nothing scheduled on it, and plays,
    /// presents and seeks exactly as an input with audio does.
    #[wasm_bindgen_test(async)]
    async fn plays_a_video_only_mp4_on_an_audio_contexts_clock() {
        let bytes = small_mp4(false).await;
        let reader = make_suspending_reader(&bytes);
        let playback = open_small(reader.clone(), Some(options(1 << 20)))
            .await
            .unwrap();
        assert!(!playback.has_audio());
        assert_eq!(playback.sample_rate(), 0);
        assert_eq!(
            parse_u64(&playback.frame_count(), "frames").unwrap(),
            SMALL_FRAMES
        );

        assert_error_code(&playback.present().unwrap_err(), "INVALID_STATE");
        until_loaded(&playback, WasmOnDemandPlayback::play).await;
        assert!(playback.is_playing());
        let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
        assert_eq!(frame_of(&presentation), 0);
        assert_eq!(pixels(&field(&presentation, "picture")), small_rgba(0));
        // The context's clock has not moved, so there is no new frame.
        assert!(field(&playback.present().unwrap(), "frame").is_null());

        for frame in [30_u64, 12] {
            playback.seek(JsValue::from_f64(frame as f64)).unwrap();
            let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
            assert_eq!(frame_of(&presentation), frame);
            assert_eq!(pixels(&field(&presentation, "picture")), small_rgba(frame));
        }

        playback.pause().unwrap();
        playback.seek(JsValue::from_f64(89.0)).unwrap();
        let picture = until_loaded(&playback, WasmOnDemandPlayback::current_frame).await;
        assert_eq!(pixels(&picture), small_rgba(89));
        assert!(field(&reader, "reads").as_f64().unwrap() > 0.0);

        // Whichever video decoder the browser picks, the clock and seeks
        // behave the same.
        let automatic = WasmOnDemandPlayback::open_inner(
            make_suspending_reader(&bytes),
            Some(options(1 << 20)),
        )
        .await
        .unwrap();
        until_loaded(&automatic, WasmOnDemandPlayback::play).await;
        for frame in [0_u64, 30] {
            automatic.seek(JsValue::from_f64(frame as f64)).unwrap();
            let presentation = until_loaded(&automatic, WasmOnDemandPlayback::present).await;
            assert_eq!(frame_of(&presentation), frame);
            assert_eq!(pixels(&field(&presentation, "picture")).len(), 32 * 18 * 4);
        }
    }

    /// Issue #681: given no audio context, an input with no audio track plays
    /// on the page's clock, which advances in real time while playing and
    /// holds still while paused, and finishes at the end of the video.
    #[wasm_bindgen_test(async)]
    async fn plays_a_video_only_mp4_on_the_page_clock_in_real_time() {
        let bytes = small_mp4(false).await;
        let playback = open_small(make_suspending_reader(&bytes), None)
            .await
            .unwrap();
        assert!(!playback.has_audio());
        let frame_index = |playback: &WasmOnDemandPlayback| {
            parse_u64(&playback.current_frame_index().unwrap(), "frame").unwrap()
        };

        // Paused, the clock holds still.
        playback.seek(JsValue::from_f64(10.0)).unwrap();
        let picture = until_loaded(&playback, WasmOnDemandPlayback::current_frame).await;
        assert_eq!(pixels(&picture), small_rgba(10));
        sleep(150.0).await;
        assert_eq!(frame_index(&playback), 10);

        // Playing, it runs at the video's own rate: a third of a second is
        // ten frames, give or take the time the test itself takes.
        until_loaded(&playback, WasmOnDemandPlayback::play).await;
        let started = clock_seconds();
        let first = frame_index(&playback);
        sleep(334.0).await;
        let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
        let elapsed = clock_seconds() - started;
        let requested = parse_u64(&field(&presentation, "requestedFrame"), "frame").unwrap();
        let expected = first as f64 + elapsed * SMALL_RATE as f64;
        assert!(
            (requested as f64 - expected).abs() <= 2.0,
            "frame {requested} after {elapsed} s from frame {first}"
        );
        assert!(requested >= first + 9);
        if !field(&presentation, "frame").is_null() {
            assert_eq!(
                pixels(&field(&presentation, "picture")),
                small_rgba(requested)
            );
        }

        playback.pause().unwrap();
        let paused = frame_index(&playback);
        sleep(150.0).await;
        assert_eq!(frame_index(&playback), paused);

        // Played past its end, the video finishes.
        playback.seek(JsValue::from_f64(88.0)).unwrap();
        until_loaded(&playback, WasmOnDemandPlayback::play).await;
        sleep(200.0).await;
        let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
        assert_eq!(field(&presentation, "finished"), JsValue::TRUE);
    }

    /// Issue #685: a WebM plays, presents and seeks over a suspending source
    /// through the same path an MP4 does, with VP8 or VP9 video and Opus audio
    /// or none. Its frames are the ones an eager decode of the whole file
    /// gives. The file is a few kilobytes, smaller than the pages its index is
    /// read through, so what opening fetches is measured natively instead.
    #[wasm_bindgen_test(async)]
    async fn plays_webm_inputs_over_a_suspending_source() {
        for (codec, opus) in [(Codec::Vp8, true), (Codec::Vp9, true), (Codec::Vp9, false)] {
            let bytes = small_webm(codec, opus).await;
            let eager = eager_frames_of(&bytes).await;
            assert_eq!(eager.len() as u64, SMALL_FRAMES);
            let reader = make_suspending_reader(&bytes);
            let playback = open_small(reader, Some(options(1 << 20))).await.unwrap();
            assert_eq!(playback.has_audio(), opus, "{codec:?}");
            assert_eq!(playback.sample_rate(), if opus { 48_000 } else { 0 });
            assert_eq!(
                parse_u64(&playback.frame_count(), "frames").unwrap(),
                SMALL_FRAMES
            );

            until_loaded(&playback, WasmOnDemandPlayback::play).await;
            let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
            assert_eq!(frame_of(&presentation), 0);
            assert_eq!(pixels(&field(&presentation, "picture")), eager[0]);

            for frame in [45_u64, 12] {
                playback.seek(JsValue::from_f64(frame as f64)).unwrap();
                let presentation = until_loaded(&playback, WasmOnDemandPlayback::present).await;
                assert_eq!(frame_of(&presentation), frame, "{codec:?}");
                assert_eq!(
                    pixels(&field(&presentation, "picture")),
                    eager[frame as usize],
                    "{codec:?} frame {frame}"
                );
            }

            playback.pause().unwrap();
            playback.seek(JsValue::from_f64(89.0)).unwrap();
            let picture = until_loaded(&playback, WasmOnDemandPlayback::current_frame).await;
            assert_eq!(pixels(&picture), eager[89]);

            // Whichever video decoder the browser picks, the WebM opens and
            // seeks the same way.
            let automatic = WasmOnDemandPlayback::open_inner(
                make_suspending_reader(&bytes),
                Some(options(1 << 20)),
            )
            .await
            .unwrap();
            until_loaded(&automatic, WasmOnDemandPlayback::play).await;
            for frame in [0_u64, 30] {
                automatic.seek(JsValue::from_f64(frame as f64)).unwrap();
                let presentation = until_loaded(&automatic, WasmOnDemandPlayback::present).await;
                assert_eq!(frame_of(&presentation), frame);
                assert_eq!(pixels(&field(&presentation, "picture")).len(), 32 * 18 * 4);
            }
        }
    }

    /// Issue #685: a WebM opens from a URL and a `Blob` as an MP4 does.
    #[wasm_bindgen_test(async)]
    async fn opens_a_webm_from_a_url_and_a_blob() {
        let bytes = small_webm(Codec::Vp9, false).await;
        let eager = eager_frames_of(&bytes).await;
        let url = make_object_url(&bytes, "video/webm").unwrap();
        let blob = make_blob(&bytes, "video/webm").unwrap();
        for source in [JsValue::from_str(&url), blob.into()] {
            let playback = open_small(source, Some(options(1 << 20))).await.unwrap();
            playback.seek(JsValue::from_f64(20.0)).unwrap();
            let picture = until_loaded(&playback, WasmOnDemandPlayback::current_frame).await;
            assert_eq!(pixels(&picture), eager[20]);
        }
    }

    /// An input with audio needs an audio context to play it through.
    #[wasm_bindgen_test(async)]
    async fn opening_requires_an_audio_context() {
        let reader = make_suspending_reader(SAMPLE);
        let error = WasmOnDemandPlayback::open_inner(reader.clone(), None)
            .await
            .err()
            .unwrap();
        assert_error_code(&error, "INVALID_INPUT");
        let error = WasmOnDemandPlayback::open_inner(reader, Some(Object::new().into()))
            .await
            .err()
            .unwrap();
        assert_error_code(&error, "INVALID_INPUT");
    }
}
