//! On-demand browser playback of an MP4 that is never loaded whole (issue #677).
//!
//! [`WasmOnDemandPlayback`] is the JavaScript face of the on-demand
//! [`PlaybackController`] of issue #672. It reads the file through a
//! [`RangeSource`]: HTTP range requests against a URL, slices of a `Blob` or
//! `File`, or an application's own `read(offset, length)`. Only the movie
//! header and the compressed samples playback reaches are fetched, into caches
//! bounded by the configured byte budgets.
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
//! Video decodes on the crate's software decoders, which are synchronous and
//! so can sit behind the controller's synchronous reads. AAC audio has no
//! software decoder in the browser build, so [`BrowserAudioSource`] decodes it
//! through `WebCodecs` during the prefetch instead, a readahead at a time, and
//! answers the controller's reads from the decoded samples. Opus audio decodes
//! the same way, through `WebCodecs` where the browser supports it and the
//! crate's software decoder otherwise.
//!
//! An input with no audio track plays on a clock of its own (issue #681): the
//! `AudioContext`'s when the page gives one, with nothing scheduled on it, and
//! the page's `performance.now()` otherwise. [`SilentAudioSource`] stands in
//! for the audio the controller schedules against it.

use crate::audio::{AudioDecoder, AudioPacketProvider, AudioSampleReader, EncodedAudioSample};
use crate::codec::{CancellationToken, EncodedVideoSample, ExactFrameReader};
use crate::codec_config::derive_codec_string;
use crate::io::{ByteSource, IoFuture};
use crate::media::{AudioBuffer, Codec};
use crate::mp4_demux::{Mp4Demuxer, Mp4DemuxerOptions, Mp4Track};
use crate::playback::{
    AudioOutputBackend, IndexedPresentationTimeline, OnDemandVideoSource, PlaybackAudioSource,
    PlaybackController, PlaybackOptions, PrefetchAudioSource, WebAudioOutput,
};
use crate::timeline::{FrameIndex, SampleRange};
use crate::wasm_api::{
    MAX_SAFE_INTEGER, WasmVideoFrame, bigint_u64, ensure_open, js_error, owned_u8_array, parse_u64,
    property,
};
use crate::web_audio_decoder::{
    AAC_PREROLL_PACKETS, NoSoftwareDecoder, WebAudioDecoderConfig, WebCodecsAudioDecoder,
};
use crate::web_decoder::{js_to_promise, normalize_js_error, packed_rgba, software_video_decoder};
use crate::{
    Error, ErrorKind, Limits, Mp4SampleLoader, OPUS_PREROLL_SAMPLES, PrefetchedAudioPacketProvider,
    Result, TrackKind,
};
use js_sys::{Object, Promise, Reflect, Uint8Array};
use std::cell::{Cell, RefCell};
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
    /// this browser decodes through `WebCodecs`, or an Opus track. Reads no
    /// packet but an Opus track's last, whose first two bytes give its length.
    async fn open(
        audio: Mp4Track,
        movie_timescale: u32,
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
        let timing = audio.audio_timing(movie_timescale)?;
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

type Controller = PlaybackController<
    OnDemandVideoSource<RangeSource>,
    PlaybackAudio,
    WebAudioOutput<BrowserOutput>,
>;

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

fn first_track(demuxer: &Mp4Demuxer, kind: TrackKind) -> Option<&Mp4Track> {
    demuxer.tracks.iter().find(|track| track.kind == kind)
}

/// Playback of an MP4 read on demand through a URL, `Blob` or range reader,
/// timed by a Web Audio context.
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
    video_budget_bytes: u64,
    audio_budget_bytes: u64,
}

impl WasmOnDemandPlayback {
    pub(crate) async fn open_inner(
        source: JsValue,
        options: Option<JsValue>,
    ) -> std::result::Result<Self, JsValue> {
        let options = parse_on_demand_options(options)?;
        let limits = Limits::default();
        let source = RangeSource::open(source).await.map_err(core_error)?;
        let demuxer = Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())
            .await
            .map_err(core_error)?;
        let video = first_track(&demuxer, TrackKind::Video)
            .ok_or_else(|| js_error(ErrorKind::Unsupported, "the input has no video track"))?
            .clone();
        let audio = first_track(&demuxer, TrackKind::Audio).cloned();
        if audio.is_some() && options.audio_context.is_none() {
            return Err(js_error(
                ErrorKind::InvalidInput,
                "on-demand playback of an input with audio requires an audioContext, which \
                 plays the audio and whose clock times it",
            ));
        }

        let dimensions = video
            .dimensions
            .ok_or_else(|| js_error(ErrorKind::MalformedMedia, "video track has no dimensions"))?;
        let derived =
            derive_codec_string(video.codec, &video.decoder_config).map_err(core_error)?;
        // An AV1 or VP9 track's color range may be read from its first sample.
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
        let video_loader =
            Mp4SampleLoader::new(video.clone(), source.clone(), options.video_budget_bytes)
                .map_err(core_error)?;
        let video_reader = ExactFrameReader::from_provider(
            factory.as_ref(),
            configuration,
            Box::new(video_loader.sample_provider().map_err(core_error)?),
            limits,
        )
        .map_err(core_error)?;
        let frame_count = video.presentation_order.len() as u64;

        let audio_source = match audio {
            Some(audio) => Some(
                BrowserAudioSource::open(
                    audio,
                    demuxer.movie_timescale,
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
        let (audio_source, output) = match (audio_source, options.audio_context) {
            (Some(audio), Some(context)) => (
                PlaybackAudio::Track(audio),
                BrowserOutput::Audio(AudioContextBackend::new(context, clock_rate)),
            ),
            (_, context) => (
                PlaybackAudio::Silent(SilentAudioSource {
                    sample_rate: clock_rate,
                    length: timeline.end_sample(),
                    limits,
                }),
                BrowserOutput::Clock(ClockBackend {
                    clock: context.map_or(Clock::Page, Clock::AudioContext),
                    rate: clock_rate,
                }),
            ),
        };
        let controller = PlaybackController::new_with_indexed_timeline(
            OnDemandVideoSource::new(video_reader, video_loader, VIDEO_READAHEAD_SAMPLES),
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
    /// Only the movie header is read here, and an Opus track's last packet.
    /// The video plays on the crate's software decoder. AAC audio plays
    /// through `WebCodecs`, and Opus audio through `WebCodecs` where the
    /// browser supports it and the crate's software decoder otherwise.
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

    #[wasm_bindgen(getter, js_name = videoBudgetBytes)]
    pub fn video_budget_bytes(&self) -> f64 {
        self.video_budget_bytes as f64
    }

    #[wasm_bindgen(getter, js_name = audioBudgetBytes)]
    pub fn audio_budget_bytes(&self) -> f64 {
        self.audio_budget_bytes as f64
    }

    /// Every byte read from the source so far, movie header included.
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

// The browser tests decode the bundled HEVC sample, so they build with the
// whole codec matrix, as the other browser tests do.
#[cfg(all(test, feature = "all"))]
mod tests {
    use super::*;
    use crate::io::MemorySource;
    use crate::wasm_api::error_code;
    use crate::web_audio_decoder::WebAudioDecodeSession;
    use crate::web_decoder::schedule_event_loop_tick;
    use js_sys::BigInt;
    use wasm_bindgen_test::*;
    use web_sys::OfflineAudioContext;

    wasm_bindgen_test_configure!(run_in_browser);

    const SAMPLE: &[u8] = include_bytes!("../examples/media/BigBuckBunny.mp4");

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

    async fn sample_tracks() -> (Mp4Demuxer, MemorySource) {
        let source = MemorySource::new(SAMPLE.to_vec());
        let demuxer = Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())
            .await
            .unwrap();
        (demuxer, source)
    }

    /// Whether this browser decodes the sample's AAC track through
    /// `WebCodecs`, which on-demand playback needs for its audio.
    async fn browser_decodes_sample_audio() -> bool {
        let (demuxer, _) = sample_tracks().await;
        let audio = first_track(&demuxer, TrackKind::Audio).unwrap();
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
        let (demuxer, source) = sample_tracks().await;
        let video = first_track(&demuxer, TrackKind::Video).unwrap();
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
        let mut playback = WasmOnDemandPlayback::open_inner(reader.clone(), Some(options(1 << 20)))
            .await
            .unwrap();
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
            let playback = WasmOnDemandPlayback::open_inner(source, Some(options(1 << 20)))
                .await
                .unwrap();
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
        let (demuxer, _) = sample_tracks().await;
        let limits = Limits::default();
        let source = RangeSource::open(make_suspending_reader(SAMPLE))
            .await
            .unwrap();
        let mut audio = BrowserAudioSource::open(
            first_track(&demuxer, TrackKind::Audio).unwrap().clone(),
            demuxer.movie_timescale,
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
