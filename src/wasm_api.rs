//! Browser-facing wrappers for the portable core.
//!
//! Values crossing this boundary are copied into owned Rust storage. Returned
//! typed arrays are snapshots rather than views into growable WebAssembly
//! memory, and browser-owned objects are retained only as JavaScript handles.

use crate::cover::{CoverCapture, CoverSource};
use crate::io::{MemorySink, MemorySource};
use crate::mp4::{CoverArt, CoverArtFormat, Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use crate::transfer::{CpuFrameSource, FrameSource, Orientation};
use crate::web_audio_decoder::WebAudioDecodeSession;
use crate::web_decoder::{
    WebVideoDecodeSession, video_frame_durations_ms, video_random_access_points,
};
use crate::web_encoder::{
    WebAudioEncodeSession, WebVideoEncodeSession, audio_encode_capability, sample_dependency,
    video_encode_capability,
};
use crate::web_previews::WebPreviewIndex;
use crate::webm::WebmMuxer;
use crate::{
    AudioBuffer as CoreAudioBuffer, CancellationToken, Codec, CodecProfile, ColorRange, Container,
    EncodedSample, EncoderConfig, ErrorKind, FrameIndex as CoreFrameIndex, FrameRate,
    HardwarePreference, Limits, PixelFormat, Plane, PreviewOptions as CorePreviewOptions,
    PreviewStore, Rational as CoreRational, SEEK_LATENCY_BUDGET, SampleRange as CoreSampleRange,
    Timeline, VideoDimensions, VideoFrame as CoreVideoFrame,
};
use js_sys::{Array, BigInt, Float32Array, Promise, Reflect, Uint8Array};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, future_to_promise};
use web_sys::{AbortSignal, Blob};

#[cfg(test)]
use web_sys::ReadableStream;

const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

#[wasm_bindgen(module = "/js/browser.js")]
extern "C" {
    #[wasm_bindgen(catch, js_name = readBrowserSource)]
    fn read_browser_source(
        source: &JsValue,
        max_bytes: f64,
        signal: Option<&AbortSignal>,
    ) -> Result<Promise, JsValue>;

    #[wasm_bindgen(catch, js_name = makeBlob)]
    fn make_blob(bytes: &[u8], mime_type: &str) -> Result<Blob, JsValue>;

    #[wasm_bindgen(js_name = makeError)]
    fn make_error(code: &str, message: &str) -> JsValue;

    #[cfg(test)]
    #[wasm_bindgen(js_name = makeTestStream)]
    fn make_test_stream(chunks: &JsValue) -> ReadableStream;

    #[cfg(test)]
    #[wasm_bindgen(js_name = makePendingStream)]
    fn make_pending_stream() -> ReadableStream;

    #[cfg(test)]
    #[wasm_bindgen(js_name = probeTestVideo)]
    fn probe_test_video(blob: &Blob, seek_to: f64) -> Promise;
}

fn error_code_for_kind(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::InvalidInput => "INVALID_INPUT",
        ErrorKind::Unsupported => "UNSUPPORTED",
        ErrorKind::MalformedMedia => "MALFORMED_MEDIA",
        ErrorKind::ResourceLimit => "RESOURCE_LIMIT",
        ErrorKind::Io => "IO",
        ErrorKind::Codec => "CODEC",
        ErrorKind::Graphics => "GRAPHICS",
        ErrorKind::Cancelled => "CANCELLED",
        ErrorKind::InvalidState => "INVALID_STATE",
        ErrorKind::Internal => "INTERNAL",
        ErrorKind::WouldBlock => "WOULD_BLOCK",
        // A kind a later zvidlib-core adds before this table names it.
        _ => "INTERNAL",
    }
}

fn js_error(kind: ErrorKind, message: impl AsRef<str>) -> JsValue {
    make_error(error_code_for_kind(kind), message.as_ref())
}

fn reflected_string(target: &JsValue, property: &str) -> Option<String> {
    Reflect::get(target, &JsValue::from_str(property))
        .ok()
        .and_then(|value| value.as_string())
}

fn normalize_browser_error(error: JsValue, context: &str) -> JsValue {
    if reflected_string(&error, "code").is_some() {
        error
    } else {
        let detail = reflected_string(&error, "message")
            .or_else(|| error.as_string())
            .unwrap_or_else(|| "browser operation failed".to_owned());
        js_error(ErrorKind::Io, format!("{context}: {detail}"))
    }
}

/// Cancels `cancellation` when `signal` aborts, so a decode already under way stops there.
///
/// The returned closure is the live `abort` listener and has to outlive the operation it is
/// cancelling; dropping it removes the listener.
fn cancel_on_abort(signal: &AbortSignal, cancellation: &CancellationToken) -> Closure<dyn FnMut()> {
    let cancellation = cancellation.clone();
    let listener = Closure::<dyn FnMut()>::new(move || cancellation.cancel());
    signal.set_onabort(Some(listener.as_ref().unchecked_ref()));
    listener
}

fn check_signal(signal: Option<&AbortSignal>) -> Result<(), JsValue> {
    if signal.is_some_and(AbortSignal::aborted) {
        Err(js_error(
            ErrorKind::Cancelled,
            "the browser operation was cancelled",
        ))
    } else {
        Ok(())
    }
}

fn parse_u64(value: &JsValue, field: &str) -> Result<u64, JsValue> {
    if value.is_bigint() {
        return BigInt::new(value)
            .ok()
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| {
                js_error(
                    ErrorKind::InvalidInput,
                    format!("{field} must be between 0n and 18446744073709551615n"),
                )
            });
    }

    let Some(number) = value.as_f64() else {
        return Err(js_error(
            ErrorKind::InvalidInput,
            format!("{field} must be a BigInt or safe integer"),
        ));
    };
    if !number.is_finite()
        || number.fract() != 0.0
        || number < 0.0
        || number > MAX_SAFE_INTEGER as f64
    {
        return Err(js_error(
            ErrorKind::InvalidInput,
            format!("{field} must be a non-negative safe integer or BigInt"),
        ));
    }
    Ok(number as u64)
}

fn parse_i64(value: &JsValue, field: &str) -> Result<i64, JsValue> {
    if value.is_bigint() {
        return BigInt::new(value)
            .ok()
            .and_then(|value| i64::try_from(value).ok())
            .ok_or_else(|| {
                js_error(
                    ErrorKind::InvalidInput,
                    format!("{field} must fit in a signed 64-bit integer"),
                )
            });
    }

    let Some(number) = value.as_f64() else {
        return Err(js_error(
            ErrorKind::InvalidInput,
            format!("{field} must be a BigInt or safe integer"),
        ));
    };
    if !number.is_finite() || number.fract() != 0.0 || number.abs() > MAX_SAFE_INTEGER as f64 {
        return Err(js_error(
            ErrorKind::InvalidInput,
            format!("{field} must be a safe integer or BigInt"),
        ));
    }
    Ok(number as i64)
}

fn parse_allocation_limit(value: Option<&JsValue>, field: &str) -> Result<u64, JsValue> {
    let limit = match value {
        Some(value) => parse_u64(value, field)?,
        None => Limits::default().max_allocation_bytes,
    };
    if limit > MAX_SAFE_INTEGER || usize::try_from(limit).is_err() {
        return Err(js_error(
            ErrorKind::InvalidInput,
            format!("{field} exceeds this browser's addressable safe range"),
        ));
    }
    Ok(limit)
}

fn property(target: &JsValue, name: &str) -> Result<Option<JsValue>, JsValue> {
    let value = Reflect::get(target, &JsValue::from_str(name))
        .map_err(|error| normalize_browser_error(error, &format!("reading option {name}")))?;
    if value.is_null() || value.is_undefined() {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

fn parse_open_options(options: Option<JsValue>) -> Result<(u64, Option<AbortSignal>), JsValue> {
    let Some(options) = options else {
        return Ok((Limits::default().max_allocation_bytes, None));
    };
    let max_input_bytes = property(&options, "maxInputBytes")?;
    let max_input_bytes = parse_allocation_limit(max_input_bytes.as_ref(), "maxInputBytes")?;
    let signal = property(&options, "signal")?
        .map(|signal| {
            signal
                .dyn_into::<AbortSignal>()
                .map_err(|_| js_error(ErrorKind::InvalidInput, "signal must be an AbortSignal"))
        })
        .transpose()?;
    Ok((max_input_bytes, signal))
}

fn parse_playback_options(options: Option<JsValue>) -> Result<WasmPlaybackOptions, JsValue> {
    let Some(options) = options else {
        return Ok(WasmPlaybackOptions::default());
    };
    Ok(WasmPlaybackOptions {
        audio_context: property(&options, "audioContext")?,
        webgl_context: property(&options, "webglContext")?,
        canvas: property(&options, "canvas")?,
    })
}

fn bigint_u64(value: u64) -> JsValue {
    BigInt::from(value).into()
}

fn bigint_i64(value: i64) -> JsValue {
    BigInt::from(value).into()
}

fn owned_u8_array(bytes: &[u8]) -> Uint8Array {
    Uint8Array::from(bytes)
}

fn owned_f32_array(samples: &[f32]) -> Float32Array {
    Float32Array::from(samples)
}

fn ensure_open(state: &Rc<Cell<bool>>) -> Result<(), JsValue> {
    if state.get() {
        Err(js_error(
            ErrorKind::InvalidState,
            "the media session is closed",
        ))
    } else {
        Ok(())
    }
}

#[wasm_bindgen(js_name = errorCode)]
pub fn error_code(error: &JsValue) -> Option<String> {
    reflected_string(error, "code")
}

/// A zero-based presentation index represented as JavaScript `BigInt`.
#[wasm_bindgen(js_name = FrameIndex)]
pub struct WasmFrameIndex(CoreFrameIndex);

#[wasm_bindgen(js_class = FrameIndex)]
impl WasmFrameIndex {
    #[wasm_bindgen(constructor)]
    pub fn new(value: JsValue) -> Result<WasmFrameIndex, JsValue> {
        Ok(Self(CoreFrameIndex(parse_u64(&value, "frame index")?)))
    }

    #[wasm_bindgen(getter)]
    pub fn value(&self) -> JsValue {
        bigint_u64(self.0.0)
    }

    #[wasm_bindgen(js_name = toString)]
    pub fn as_string(&self) -> String {
        self.0.0.to_string()
    }
}

/// A signed timestamp scalar represented as JavaScript `BigInt`.
#[wasm_bindgen(js_name = Timestamp)]
pub struct WasmTimestamp(i64);

#[wasm_bindgen(js_class = Timestamp)]
impl WasmTimestamp {
    #[wasm_bindgen(constructor)]
    pub fn new(value: JsValue) -> Result<WasmTimestamp, JsValue> {
        Ok(Self(parse_i64(&value, "timestamp")?))
    }

    #[wasm_bindgen(getter)]
    pub fn value(&self) -> JsValue {
        bigint_i64(self.0)
    }
}

/// A normalized signed rational value.
#[wasm_bindgen(js_name = Rational)]
pub struct WasmRational(CoreRational);

#[wasm_bindgen(js_class = Rational)]
impl WasmRational {
    #[wasm_bindgen(constructor)]
    pub fn new(numerator: JsValue, denominator: JsValue) -> Result<WasmRational, JsValue> {
        let numerator = parse_i64(&numerator, "rational numerator")?;
        let denominator = parse_i64(&denominator, "rational denominator")?;
        CoreRational::new(numerator, denominator)
            .map(Self)
            .map_err(|error| js_error(error.kind(), error.message()))
    }

    #[wasm_bindgen(getter)]
    pub fn numerator(&self) -> JsValue {
        bigint_i64(self.0.numerator())
    }

    #[wasm_bindgen(getter)]
    pub fn denominator(&self) -> JsValue {
        bigint_i64(self.0.denominator())
    }
}

/// A half-open sample interval whose endpoints are JavaScript `BigInt`s.
#[wasm_bindgen(js_name = SampleRange)]
pub struct WasmSampleRange(CoreSampleRange);

#[wasm_bindgen(js_class = SampleRange)]
impl WasmSampleRange {
    #[wasm_bindgen(constructor)]
    pub fn new(start: JsValue, end: JsValue) -> Result<WasmSampleRange, JsValue> {
        let start = parse_u64(&start, "sample range start")?;
        let end = parse_u64(&end, "sample range end")?;
        CoreSampleRange::new(start, end)
            .map(Self)
            .map_err(|error| js_error(error.kind(), error.message()))
    }

    #[wasm_bindgen(getter)]
    pub fn start(&self) -> JsValue {
        bigint_u64(self.0.start)
    }

    #[wasm_bindgen(getter)]
    pub fn end(&self) -> JsValue {
        bigint_u64(self.0.end)
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> JsValue {
        bigint_u64(self.0.len())
    }
}

/// An owned CPU video frame. Pixel arrays are copied in both directions.
#[wasm_bindgen(js_name = VideoFrame)]
pub struct WasmVideoFrame(CoreVideoFrame);

#[wasm_bindgen(js_class = VideoFrame)]
impl WasmVideoFrame {
    #[wasm_bindgen(js_name = rgba)]
    pub fn rgba(width: u32, height: u32, pixels: Uint8Array) -> Result<WasmVideoFrame, JsValue> {
        let stride = usize::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(4))
            .ok_or_else(|| js_error(ErrorKind::ResourceLimit, "video stride overflow"))?;
        Self::packed(width, height, PixelFormat::Rgba8, stride, pixels)
    }

    /// A BGRA8, tightly-packed (`stride == width * 4`) frame.
    #[wasm_bindgen(js_name = bgra8)]
    pub fn bgra8(width: u32, height: u32, pixels: Uint8Array) -> Result<WasmVideoFrame, JsValue> {
        let stride = usize::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(4))
            .ok_or_else(|| js_error(ErrorKind::ResourceLimit, "video stride overflow"))?;
        Self::packed(width, height, PixelFormat::Bgra8, stride, pixels)
    }

    /// A YUV 4:2:0 planar 8-bit frame: one tightly-packed full-resolution Y
    /// plane followed by tightly-packed, half-resolution (rounded up) U and V
    /// planes, all three concatenated into `pixels`.
    #[wasm_bindgen(js_name = yuv420p8)]
    pub fn yuv420p8(
        width: u32,
        height: u32,
        pixels: Uint8Array,
    ) -> Result<WasmVideoFrame, JsValue> {
        let limits = Limits::default();
        let dimensions = VideoDimensions::new(width, height, &limits)
            .map_err(|error| js_error(error.kind(), error.message()))?;
        let w = width as usize;
        let h = height as usize;
        let chroma_w = w.div_ceil(2);
        let chroma_h = h.div_ceil(2);
        let luma_len = w
            .checked_mul(h)
            .ok_or_else(|| js_error(ErrorKind::ResourceLimit, "video plane size overflow"))?;
        let chroma_len = chroma_w
            .checked_mul(chroma_h)
            .ok_or_else(|| js_error(ErrorKind::ResourceLimit, "video plane size overflow"))?;
        let pixels = pixels.to_vec();
        let expected = luma_len
            .checked_add(chroma_len)
            .and_then(|value| value.checked_add(chroma_len))
            .ok_or_else(|| js_error(ErrorKind::ResourceLimit, "video plane size overflow"))?;
        if pixels.len() != expected {
            return Err(js_error(
                ErrorKind::InvalidInput,
                "YUV 4:2:0 frame data does not match the given width/height",
            ));
        }
        let (y, uv) = pixels.split_at(luma_len);
        let (u, v) = uv.split_at(chroma_len);
        let frame = CoreVideoFrame::new(
            dimensions,
            PixelFormat::Yuv420p8,
            ColorRange::Full,
            vec![
                Plane {
                    data: y.to_vec(),
                    stride: w,
                },
                Plane {
                    data: u.to_vec(),
                    stride: chroma_w,
                },
                Plane {
                    data: v.to_vec(),
                    stride: chroma_w,
                },
            ],
            &limits,
        )
        .map_err(|error| js_error(error.kind(), error.message()))?;
        Ok(Self(frame))
    }

    fn packed(
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        stride: usize,
        pixels: Uint8Array,
    ) -> Result<WasmVideoFrame, JsValue> {
        let limits = Limits::default();
        let dimensions = VideoDimensions::new(width, height, &limits)
            .map_err(|error| js_error(error.kind(), error.message()))?;
        let frame = CoreVideoFrame::new(
            dimensions,
            pixel_format,
            ColorRange::Full,
            vec![Plane {
                data: pixels.to_vec(),
                stride,
            }],
            &limits,
        )
        .map_err(|error| js_error(error.kind(), error.message()))?;
        Ok(Self(frame))
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.0.dimensions.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.0.dimensions.height
    }

    #[wasm_bindgen(getter)]
    pub fn pixels(&self) -> Uint8Array {
        owned_u8_array(&self.0.planes[0].data)
    }
}

/// An owned interleaved `f32` audio buffer. Sample arrays are copied.
#[wasm_bindgen(js_name = AudioBuffer)]
pub struct WasmAudioBuffer(CoreAudioBuffer);

#[wasm_bindgen(js_class = AudioBuffer)]
impl WasmAudioBuffer {
    #[wasm_bindgen(constructor)]
    pub fn new(
        range: &WasmSampleRange,
        sample_rate: u32,
        channels: u16,
        samples: Float32Array,
    ) -> Result<WasmAudioBuffer, JsValue> {
        CoreAudioBuffer::new(
            range.0,
            sample_rate,
            channels,
            samples.to_vec(),
            &Limits::default(),
        )
        .map(Self)
        .map_err(|error| js_error(error.kind(), error.message()))
    }

    #[wasm_bindgen(getter, js_name = sampleRate)]
    pub fn sample_rate(&self) -> u32 {
        self.0.sample_rate
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> u16 {
        self.0.channels
    }

    #[wasm_bindgen(getter)]
    pub fn range(&self) -> WasmSampleRange {
        WasmSampleRange(self.0.range)
    }

    #[wasm_bindgen(getter)]
    pub fn samples(&self) -> Float32Array {
        owned_f32_array(&self.0.samples)
    }
}

/// One indexed compressed AAC access unit from an input MP4 audio track.
#[wasm_bindgen(js_name = EncodedAudioSample)]
pub struct WasmEncodedAudioSample(crate::EncodedAudioSample);

#[wasm_bindgen(js_class = EncodedAudioSample)]
impl WasmEncodedAudioSample {
    #[wasm_bindgen(getter)]
    pub fn range(&self) -> WasmSampleRange {
        WasmSampleRange(self.0.decoded_range)
    }

    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Uint8Array {
        owned_u8_array(&self.0.data)
    }
}

/// AAC decoder configuration for an input MP4 audio track.
#[wasm_bindgen(js_name = AacConfig)]
pub struct WasmAacConfig(crate::AacTrackConfig);

#[wasm_bindgen(js_class = AacConfig)]
impl WasmAacConfig {
    #[wasm_bindgen(getter, js_name = audioObjectType)]
    pub fn audio_object_type(&self) -> u8 {
        self.0.audio_object_type
    }

    #[wasm_bindgen(getter, js_name = sampleRate)]
    pub fn sample_rate(&self) -> u32 {
        self.0.sample_rate
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> u16 {
        self.0.channels
    }

    #[wasm_bindgen(getter, js_name = audioSpecificConfig)]
    pub fn audio_specific_config(&self) -> Uint8Array {
        owned_u8_array(&self.0.audio_specific_config)
    }

    #[wasm_bindgen(getter)]
    pub fn codec(&self) -> String {
        format!("mp4a.40.{}", self.0.audio_object_type)
    }
}

#[wasm_bindgen(js_name = OpenOptions)]
pub struct WasmOpenOptions {
    max_input_bytes: u64,
    signal: Option<AbortSignal>,
}

#[wasm_bindgen(js_class = OpenOptions)]
impl WasmOpenOptions {
    #[wasm_bindgen(constructor)]
    pub fn new(max_input_bytes: Option<JsValue>) -> Result<WasmOpenOptions, JsValue> {
        Ok(Self {
            max_input_bytes: parse_allocation_limit(max_input_bytes.as_ref(), "maxInputBytes")?,
            signal: None,
        })
    }

    #[wasm_bindgen(getter, js_name = maxInputBytes)]
    pub fn max_input_bytes(&self) -> JsValue {
        bigint_u64(self.max_input_bytes)
    }

    #[wasm_bindgen(setter, js_name = maxInputBytes)]
    pub fn set_max_input_bytes(&mut self, value: JsValue) -> Result<(), JsValue> {
        self.max_input_bytes = parse_allocation_limit(Some(&value), "maxInputBytes")?;
        Ok(())
    }

    #[wasm_bindgen(getter)]
    pub fn signal(&self) -> Option<AbortSignal> {
        self.signal.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_signal(&mut self, signal: Option<AbortSignal>) {
        self.signal = signal;
    }
}

#[wasm_bindgen(js_name = CreateOptions)]
pub struct WasmCreateOptions {
    container: Container,
    mime_type: String,
    max_output_bytes: u64,
    frame_rate: Option<FrameRate>,
    audio_sample_rate: Option<u32>,
    video_codec: Codec,
    audio_codec: Codec,
    cover_source: CoverSource,
}

#[wasm_bindgen(js_class = CreateOptions)]
impl WasmCreateOptions {
    /// `container` is `"mp4"` (the default) or `"webm"`, whose `finish()`
    /// returns a `video/webm` Blob. WebM output carries AV1 or VP8 video and
    /// Opus or Vorbis audio.
    #[wasm_bindgen(constructor)]
    pub fn new(container: Option<String>) -> Result<WasmCreateOptions, JsValue> {
        let name = container.unwrap_or_else(|| "mp4".to_owned());
        let container = Container::from_name(&name).ok_or_else(|| {
            js_error(
                ErrorKind::Unsupported,
                format!("unsupported output container: {name}"),
            )
        })?;
        Ok(Self {
            container,
            mime_type: container.mime_type().to_owned(),
            max_output_bytes: Limits::default().max_allocation_bytes,
            frame_rate: None,
            audio_sample_rate: None,
            video_codec: Codec::Av1,
            // WebM permits no AAC, so a WebM output starts out on Opus.
            audio_codec: match container {
                Container::WebM => Codec::Opus,
                _ => Codec::Aac,
            },
            cover_source: CoverSource::default(),
        })
    }

    #[wasm_bindgen(getter)]
    pub fn container(&self) -> String {
        self.container.name().to_owned()
    }

    #[wasm_bindgen(getter, js_name = mimeType)]
    pub fn mime_type(&self) -> String {
        self.mime_type.clone()
    }

    #[wasm_bindgen(setter, js_name = mimeType)]
    pub fn set_mime_type(&mut self, value: String) -> Result<(), JsValue> {
        if value.trim().is_empty() {
            return Err(js_error(
                ErrorKind::InvalidInput,
                "mimeType cannot be empty",
            ));
        }
        self.mime_type = value;
        Ok(())
    }

    #[wasm_bindgen(getter, js_name = maxOutputBytes)]
    pub fn max_output_bytes(&self) -> JsValue {
        bigint_u64(self.max_output_bytes)
    }

    #[wasm_bindgen(setter, js_name = maxOutputBytes)]
    pub fn set_max_output_bytes(&mut self, value: JsValue) -> Result<(), JsValue> {
        self.max_output_bytes = parse_allocation_limit(Some(&value), "maxOutputBytes")?;
        Ok(())
    }

    #[wasm_bindgen(js_name = setTimeline)]
    pub fn set_timeline(
        &mut self,
        frame_rate_numerator: u32,
        frame_rate_denominator: u32,
        audio_sample_rate: u32,
    ) -> Result<(), JsValue> {
        self.frame_rate = Some(
            FrameRate::new(frame_rate_numerator, frame_rate_denominator)
                .map_err(|error| js_error(error.kind(), error.message()))?,
        );
        if audio_sample_rate == 0 {
            return Err(js_error(
                ErrorKind::InvalidInput,
                "audio sample rate must be positive",
            ));
        }
        self.audio_sample_rate = Some(audio_sample_rate);
        Ok(())
    }

    /// The browser video codec for [`WasmVideoStream::put`]: `"av1"`
    /// (default), `"hevc"`, `"vp8"` or `"vp9"`, each encoded through
    /// `WebCodecs` `VideoEncoder`. WebM permits no HEVC and MP4 has no widely
    /// supported mapping for VP8, so an `"mp4"` output takes `"av1"`, `"hevc"`
    /// or `"vp9"` and a `"webm"` output `"av1"`, `"vp8"` or `"vp9"`.
    #[wasm_bindgen(getter, js_name = videoCodec)]
    pub fn video_codec(&self) -> String {
        match self.video_codec {
            Codec::Hevc => "hevc".to_owned(),
            Codec::Vp8 => "vp8".to_owned(),
            Codec::Vp9 => "vp9".to_owned(),
            _ => "av1".to_owned(),
        }
    }

    #[wasm_bindgen(setter, js_name = videoCodec)]
    pub fn set_video_codec(&mut self, value: String) -> Result<(), JsValue> {
        self.video_codec = match value.as_str() {
            "av1" => Codec::Av1,
            "hevc" if self.container == Container::WebM => {
                return Err(js_error(
                    ErrorKind::Unsupported,
                    "WebM permits only VP8, VP9 or AV1 video; write HEVC to an mp4 output",
                ));
            }
            "hevc" => Codec::Hevc,
            "vp8" if self.container != Container::WebM => {
                return Err(js_error(
                    ErrorKind::Unsupported,
                    "MP4 has no widely supported VP8 mapping; write VP8 to a webm output",
                ));
            }
            "vp8" => Codec::Vp8,
            "vp9" => Codec::Vp9,
            other => {
                return Err(js_error(
                    ErrorKind::Unsupported,
                    format!("unsupported video codec: {other}"),
                ));
            }
        };
        Ok(())
    }

    /// The browser audio codec for [`WasmAudioStream::put`]: `"aac"` (AAC-LC),
    /// `"opus"` or `"vorbis"`. An `"mp4"` output defaults to AAC and takes AAC
    /// or Opus; MP4 has no widely supported mapping for Vorbis. A `"webm"`
    /// output defaults to Opus and takes Opus or Vorbis; WebM permits no AAC.
    /// AAC and Opus encode through `WebCodecs`, Opus through zvidlib's own
    /// encoder where the browser has no `AudioEncoder`, and Vorbis, which no
    /// browser encodes, always through zvidlib's own encoder. Opus is encoded
    /// from 48 kHz PCM, the rate it decodes at.
    #[wasm_bindgen(getter, js_name = audioCodec)]
    pub fn audio_codec(&self) -> String {
        match self.audio_codec {
            Codec::Opus => "opus".to_owned(),
            Codec::Vorbis => "vorbis".to_owned(),
            _ => "aac".to_owned(),
        }
    }

    #[wasm_bindgen(setter, js_name = audioCodec)]
    pub fn set_audio_codec(&mut self, value: String) -> Result<(), JsValue> {
        let codec = match value.as_str() {
            "aac" => Codec::Aac,
            "opus" => Codec::Opus,
            "vorbis" => Codec::Vorbis,
            other => {
                return Err(js_error(
                    ErrorKind::Unsupported,
                    format!("unsupported audio codec: {other}"),
                ));
            }
        };
        match (self.container, codec) {
            (Container::WebM, Codec::Aac) => {
                return Err(js_error(ErrorKind::Unsupported, WEBM_AUDIO_UNSUPPORTED));
            }
            (Container::Mp4, Codec::Vorbis) => {
                return Err(js_error(
                    ErrorKind::Unsupported,
                    "Vorbis has no widely supported MP4 mapping; write it to a webm output",
                ));
            }
            _ => {}
        }
        self.audio_codec = codec;
        Ok(())
    }

    /// The zero-based video frame [`WasmMediaOutput::finish`] shrinks into a
    /// PNG cover-art thumbnail when `setCoverArt()` supplied none: `4n` by
    /// default, or `null` for no generated cover. A stream with fewer frames
    /// uses its last one.
    #[wasm_bindgen(getter, js_name = coverFrame)]
    pub fn cover_frame(&self) -> JsValue {
        match self.cover_source {
            CoverSource::Frame(index) => bigint_u64(index),
            CoverSource::None => JsValue::NULL,
        }
    }

    #[wasm_bindgen(setter, js_name = coverFrame)]
    pub fn set_cover_frame(&mut self, value: JsValue) -> Result<(), JsValue> {
        self.cover_source = if value.is_null() || value.is_undefined() {
            CoverSource::None
        } else {
            CoverSource::Frame(parse_u64(&value, "coverFrame")?)
        };
        Ok(())
    }
}

#[wasm_bindgen(js_name = PlaybackOptions)]
#[derive(Clone, Default)]
pub struct WasmPlaybackOptions {
    audio_context: Option<JsValue>,
    webgl_context: Option<JsValue>,
    canvas: Option<JsValue>,
}

#[wasm_bindgen(js_class = PlaybackOptions)]
impl WasmPlaybackOptions {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmPlaybackOptions {
        Self::default()
    }

    #[wasm_bindgen(setter, js_name = audioContext)]
    pub fn set_audio_context(&mut self, value: Option<JsValue>) {
        self.audio_context = value;
    }

    #[wasm_bindgen(getter, js_name = audioContext)]
    pub fn audio_context(&self) -> Option<JsValue> {
        self.audio_context.clone()
    }

    #[wasm_bindgen(setter, js_name = webglContext)]
    pub fn set_webgl_context(&mut self, value: Option<JsValue>) {
        self.webgl_context = value;
    }

    #[wasm_bindgen(getter, js_name = webglContext)]
    pub fn webgl_context(&self) -> Option<JsValue> {
        self.webgl_context.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_canvas(&mut self, value: Option<JsValue>) {
        self.canvas = value;
    }

    #[wasm_bindgen(getter)]
    pub fn canvas(&self) -> Option<JsValue> {
        self.canvas.clone()
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum StreamDirection {
    Input,
    Output,
}

#[wasm_bindgen(js_name = VideoStream)]
pub struct WasmVideoStream {
    index: u32,
    state: Rc<Cell<bool>>,
    direction: StreamDirection,
    /// Source bytes for an input stream, shared with the owning `MediaInput`.
    bytes: Option<Rc<Vec<u8>>>,
    /// Lazily-configured `WebCodecs` decode session, built on first `get()`.
    decode_session: Rc<RefCell<Option<WebVideoDecodeSession>>>,
    /// Lazily-parsed per-presentation-frame durations in milliseconds.
    frame_durations_ms: Rc<RefCell<Option<Vec<f64>>>>,
    /// Lazily-parsed presentation indices of the track's random-access samples, ascending.
    random_access_points: Rc<RefCell<Option<Vec<u64>>>>,
    /// Shared `WebCodecs` encode state for an output video track, set for
    /// any output video track handed out by [`WasmMediaOutput::video`].
    browser_video: Option<Rc<RefCell<BrowserVideoTrack>>>,
}

#[wasm_bindgen(js_class = VideoStream)]
impl WasmVideoStream {
    #[wasm_bindgen(getter)]
    pub fn index(&self) -> u32 {
        self.index
    }

    #[wasm_bindgen(getter)]
    pub fn direction(&self) -> String {
        match self.direction {
            StreamDirection::Input => "input",
            StreamDirection::Output => "output",
        }
        .to_owned()
    }

    /// Returns the indexed frame's MP4 presentation duration in milliseconds.
    #[wasm_bindgen(js_name = frameDuration)]
    pub fn frame_duration(&self, frame_index: JsValue, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        let index = self.index;
        let bytes = self.bytes.clone();
        let durations = Rc::clone(&self.frame_durations_ms);
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            let frame_index = usize::try_from(parse_u64(&frame_index, "frame index")?)
                .map_err(|_| js_error(ErrorKind::InvalidInput, "frame index is out of range"))?;
            if direction != StreamDirection::Input {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "frameDuration is only valid on an input video stream",
                ));
            }
            let bytes = bytes.ok_or_else(|| {
                js_error(
                    ErrorKind::Unsupported,
                    "input video timing metadata is unavailable",
                )
            })?;
            if durations.borrow().is_none() {
                let parsed = video_frame_durations_ms(&bytes, index, &Limits::default())
                    .await
                    .map_err(|error| js_error(error.kind(), error.message()))?;
                *durations.borrow_mut() = Some(parsed);
            }
            check_signal(signal.as_ref())?;
            durations
                .borrow()
                .as_ref()
                .and_then(|values| values.get(frame_index))
                .copied()
                .map(JsValue::from_f64)
                .ok_or_else(|| {
                    js_error(ErrorKind::InvalidInput, "presentation frame is not indexed")
                })
        })
    }

    /// Returns the presentation indices this track's decoding can start from, ascending, as an
    /// array of `BigInt`s.
    ///
    /// A caller that has to move the picture backwards - scrubbing a timeline is the one that
    /// motivated this - cannot walk there frame by frame, because decoding only runs forwards. It
    /// restarts at the random-access point at or before its target and walks forwards from there
    /// instead, drawing what it passes, which is what keeps a backwards drag moving rather than
    /// frozen for the length of a whole group of pictures.
    #[wasm_bindgen(js_name = randomAccessPoints)]
    pub fn random_access_points(&self, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        let index = self.index;
        let bytes = self.bytes.clone();
        let points = Rc::clone(&self.random_access_points);
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            if direction != StreamDirection::Input {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "randomAccessPoints is only valid on an input video stream",
                ));
            }
            let bytes = bytes.ok_or_else(|| {
                js_error(
                    ErrorKind::Unsupported,
                    "input video timing metadata is unavailable",
                )
            })?;
            if points.borrow().is_none() {
                let parsed = video_random_access_points(&bytes, index, &Limits::default())
                    .await
                    .map_err(|error| js_error(error.kind(), error.message()))?;
                *points.borrow_mut() = Some(parsed);
            }
            check_signal(signal.as_ref())?;
            let array = Array::new();
            for point in points.borrow().as_ref().into_iter().flatten() {
                array.push(&BigInt::from(*point).into());
            }
            Ok(array.into())
        })
    }

    /// Builds this track's seek preview tier: one shrunk picture every stride
    /// frames, on a decode session of its own.
    ///
    /// `ARCHITECTURE.md` section 3.2 requires a seek to any position of any
    /// track to answer inside `seekLatencyBudgetMs()`, and on a track coded as a
    /// single group of pictures - which the bundled sample is - the only way to
    /// do that is from a picture that was decoded already. Walking there instead
    /// is seconds (issue #432).
    ///
    /// The index comes back empty. The caller advances it one preview at a time
    /// with `step()`, from `requestIdleCallback` or a `requestAnimationFrame`
    /// slice, so the page keeps its event loop while the pass fills, and
    /// `nearest()` answers from whatever it has reached so far.
    pub fn previews(&self, options: &WasmPreviewOptions, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        let owner = Rc::clone(&self.state);
        let direction = self.direction;
        let index = self.index;
        let bytes = self.bytes.clone();
        let options = options.0;
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            if direction != StreamDirection::Input {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "previews is only valid on an input video stream",
                ));
            }
            let bytes = bytes.ok_or_else(|| {
                js_error(
                    ErrorKind::Unsupported,
                    "no browser video decoder backend is registered",
                )
            })?;
            let previews = WebPreviewIndex::open(&bytes, index, &Limits::default(), options)
                .await
                .map_err(|error| js_error(error.kind(), error.message()))?;
            check_signal(signal.as_ref())?;
            Ok(WasmPreviewIndex::wrap(previews, owner).into())
        })
    }

    pub fn get(&self, frame_index: JsValue, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        let index = self.index;
        let bytes = self.bytes.clone();
        let decode_session = Rc::clone(&self.decode_session);
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            let frame_index = CoreFrameIndex(parse_u64(&frame_index, "frame index")?);
            if direction != StreamDirection::Input {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "get is only valid on an input video stream",
                ));
            }
            let bytes = bytes.ok_or_else(|| {
                js_error(
                    ErrorKind::Unsupported,
                    "no browser video decoder backend is registered",
                )
            })?;

            let mut session = decode_session.borrow_mut().take();
            if session.is_none() {
                session = Some(
                    WebVideoDecodeSession::open(&bytes, index, &Limits::default())
                        .await
                        .map_err(|error| js_error(error.kind(), error.message()))?,
                );
            }
            let mut session = session.expect("just populated above");
            check_signal(signal.as_ref())?;
            // Aborting has to reach inside the decode, not just bracket it: one group of pictures
            // is hundreds of frames on real content, and a scrub that aborts a request it has
            // already moved past must free the decoder for the newest position rather than
            // leaving it to finish the stale one first (issue #333).
            let cancellation = CancellationToken::new();
            let _abort = signal
                .as_ref()
                .map(|signal| cancel_on_abort(signal, &cancellation));
            let result = session.get(frame_index, &cancellation).await;
            *decode_session.borrow_mut() = Some(session);
            let (dimensions, rgba) =
                result.map_err(|error| js_error(error.kind(), error.message()))?;
            WasmVideoFrame::rgba(dimensions.width, dimensions.height, owned_u8_array(&rgba))
                .map(JsValue::from)
        })
    }

    pub fn put(
        &self,
        frame_index: JsValue,
        frame: &WasmVideoFrame,
        signal: Option<AbortSignal>,
    ) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        let browser_video = self.browser_video.clone();
        let core_frame = frame.0.clone();
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            let index = parse_u64(&frame_index, "frame index")?;
            if direction != StreamDirection::Output {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "put is only valid on an output video stream",
                ));
            }
            let browser_video = browser_video.ok_or_else(|| {
                js_error(
                    ErrorKind::Unsupported,
                    "no browser video encoder backend is registered for this track",
                )
            })?;
            encode_browser_video_frame(&browser_video, index, core_frame)
                .await
                .map_err(|error| js_error(error.kind(), error.message()))?;
            Ok(JsValue::UNDEFINED)
        })
    }
}

async fn parse_audio_track(
    bytes: Option<Rc<Vec<u8>>>,
    index: u32,
) -> Result<crate::Mp4Track, JsValue> {
    let bytes = bytes.ok_or_else(|| {
        js_error(
            ErrorKind::Unsupported,
            "input audio metadata is unavailable",
        )
    })?;
    let source = MemorySource::new((*bytes).clone());
    crate::container::open_tracks(&source, &Limits::default())
        .await
        .map_err(|error| js_error(error.kind(), error.message()))?
        .into_iter()
        .filter(|track| track.kind == crate::TrackKind::Audio)
        .nth(index as usize)
        .ok_or_else(|| js_error(ErrorKind::InvalidInput, "no such audio track"))
}

/// The containers `CreateOptions` accepts and `MediaInput.open` detects, by
/// the name `CreateOptions.container` takes: `["mp4", "webm"]`.
#[wasm_bindgen(js_name = supportedContainers)]
pub fn supported_containers() -> Array {
    crate::container_capabilities()
        .into_iter()
        .filter(|capability| capability.support.is_available())
        .map(|capability| JsValue::from(capability.name))
        .collect()
}

/// The longest a seek to any position of any track may take, in milliseconds.
///
/// This is the requirement `ARCHITECTURE.md` section 3.2 states, exported so a
/// browser caller budgets against the same number the library's own tests hold
/// it to rather than a copy of 50 that drifts.
#[wasm_bindgen(js_name = seekLatencyBudgetMs)]
pub fn seek_latency_budget_ms() -> f64 {
    SEEK_LATENCY_BUDGET.as_secs_f64() * 1_000.0
}

/// Reports whether this browser's `WebCodecs` bridge can encode AV1 Main,
/// HEVC Main, VP8 or VP9 profile 0 video: `codec` is `"av1"` (the default),
/// `"hevc"`, `"vp8"` or `"vp9"`.
///
/// `hardware` accepts `"require"`, `"prefer"` (the default), or `"avoid"`,
/// mirroring [`crate::HardwarePreference`]. This is a synchronous,
/// best-effort feature check: `WebCodecs`' own `isConfigSupported()` is
/// asynchronous and configuration-specific, so a caller that needs a
/// definitive answer still has to attempt a real encode via
/// [`WasmVideoStream::put`].
#[wasm_bindgen(js_name = videoEncodeSupport)]
pub fn video_encode_support(
    hardware: Option<String>,
    codec: Option<String>,
) -> Result<bool, JsValue> {
    let hardware = match hardware.as_deref() {
        None | Some("prefer") => HardwarePreference::Prefer,
        Some("require") => HardwarePreference::Require,
        Some("avoid") => HardwarePreference::Avoid,
        Some(other) => {
            return Err(js_error(
                ErrorKind::InvalidInput,
                format!("unknown hardware preference: {other}"),
            ));
        }
    };
    let (codec, profile) = match codec.as_deref() {
        None | Some("av1") => (Codec::Av1, CodecProfile::Av1Main),
        Some("hevc") => (Codec::Hevc, CodecProfile::HevcMain),
        Some("vp8") => (Codec::Vp8, CodecProfile::Vp8),
        Some("vp9") => (Codec::Vp9, CodecProfile::Vp9Profile0),
        Some(other) => {
            return Err(js_error(
                ErrorKind::Unsupported,
                format!("unsupported video codec: {other}"),
            ));
        }
    };
    let support = video_encode_capability(codec, profile, hardware);
    Ok(support.is_supported())
}

/// Reports whether this browser build can encode `codec`: `"aac"` (AAC-LC,
/// the default), which needs the browser's `WebCodecs` `AudioEncoder`, or
/// `"opus"` or `"vorbis"`, which zvidlib encodes itself where the browser
/// cannot. Which container can carry the codec is `CreateOptions.audioCodec`'s
/// concern: MP4 takes AAC or Opus, WebM Opus or Vorbis.
///
/// `hardware` accepts the same values as [`video_encode_support`], for the
/// same reason: `WebCodecs`' own `isConfigSupported()` is asynchronous and
/// configuration-specific, so a caller that needs a definitive answer still
/// has to attempt a real encode via [`WasmAudioStream::put`].
#[wasm_bindgen(js_name = audioEncodeSupport)]
pub fn audio_encode_support(
    hardware: Option<String>,
    codec: Option<String>,
) -> Result<bool, JsValue> {
    let hardware = match hardware.as_deref() {
        None | Some("prefer") => HardwarePreference::Prefer,
        Some("require") => HardwarePreference::Require,
        Some("avoid") => HardwarePreference::Avoid,
        Some(other) => {
            return Err(js_error(
                ErrorKind::InvalidInput,
                format!("unknown hardware preference: {other}"),
            ));
        }
    };
    let (codec, profile) = match codec.as_deref() {
        None | Some("aac") => (Codec::Aac, CodecProfile::AacLowComplexity),
        Some("opus") => (Codec::Opus, CodecProfile::Opus),
        Some("vorbis") => (Codec::Vorbis, CodecProfile::Vorbis),
        Some(other) => {
            return Err(js_error(
                ErrorKind::Unsupported,
                format!("unsupported audio codec: {other}"),
            ));
        }
    };
    let support = audio_encode_capability(codec, profile, hardware);
    Ok(support.is_supported())
}

/// How a preview index trades memory and pass length against how fine the scrub
/// is: the browser face of the library's `PreviewOptions`.
#[wasm_bindgen(js_name = PreviewOptions)]
#[derive(Clone, Copy)]
pub struct WasmPreviewOptions(CorePreviewOptions);

#[wasm_bindgen(js_class = PreviewOptions)]
impl WasmPreviewOptions {
    /// The defaults, for a source running at `framesPerSecond`.
    #[wasm_bindgen(constructor)]
    pub fn new(frames_per_second: f64) -> Result<WasmPreviewOptions, JsValue> {
        if !frames_per_second.is_finite() || frames_per_second < 1.0 {
            return Err(js_error(
                ErrorKind::InvalidInput,
                "a preview frame rate must be at least one frame a second",
            ));
        }
        Ok(Self(CorePreviewOptions::for_frame_rate(
            frames_per_second.round() as u64,
        )))
    }

    /// How far each preview is shrunk on each axis.
    #[wasm_bindgen(getter)]
    pub fn scale(&self) -> u32 {
        self.0.scale
    }

    #[wasm_bindgen(setter)]
    pub fn set_scale(&mut self, value: u32) {
        self.0.scale = value.max(1);
    }

    /// How many previews a second of playback gets, when memory allows it.
    #[wasm_bindgen(getter, js_name = previewsPerSecond)]
    pub fn previews_per_second(&self) -> u32 {
        self.0.previews_per_second as u32
    }

    #[wasm_bindgen(setter, js_name = previewsPerSecond)]
    pub fn set_previews_per_second(&mut self, value: u32) {
        self.0.previews_per_second = u64::from(value.max(1));
    }

    /// A ceiling on what the whole index may hold. The stride follows from this
    /// and the track length, so a long track keeps previews further apart rather
    /// than more of them.
    #[wasm_bindgen(getter, js_name = budgetBytes)]
    pub fn budget_bytes(&self) -> JsValue {
        BigInt::from(self.0.budget_bytes).into()
    }

    #[wasm_bindgen(setter, js_name = budgetBytes)]
    pub fn set_budget_bytes(&mut self, value: JsValue) -> Result<(), JsValue> {
        self.0.budget_bytes = parse_u64(&value, "preview budget")?;
        Ok(())
    }
}

/// A picture the preview tier already had, and the frame it is actually of.
///
/// The frame is part of the answer rather than a detail: a preview is
/// explicitly *not* the frame that was asked for, and a caller that draws one
/// has to be able to say so and to decide whether to go after the exact frame
/// underneath it.
#[wasm_bindgen(js_name = Preview)]
pub struct WasmPreview {
    frame: CoreFrameIndex,
    picture: CoreVideoFrame,
}

#[wasm_bindgen(js_class = Preview)]
impl WasmPreview {
    /// The frame this picture is of, as a `BigInt`.
    #[wasm_bindgen(getter)]
    pub fn frame(&self) -> JsValue {
        BigInt::from(self.frame.0).into()
    }

    /// The picture itself, shrunk by `PreviewOptions.scale`.
    #[wasm_bindgen(getter)]
    pub fn picture(&self) -> WasmVideoFrame {
        WasmVideoFrame(self.picture.clone())
    }
}

/// The browser's seek preview tier over one input video track.
///
/// `step()` is the pass and `nearest()` is the seek; they are deliberately
/// different shapes. The pass decodes and so it is a `Promise` the caller
/// schedules from an idle callback, one preview at a time. The lookup decodes
/// nothing - it reads a picture the pass already stored - so it is synchronous
/// and constant time, which is what lets a pointer move draw immediately
/// instead of waiting on a decode it cannot afford.
#[wasm_bindgen(js_name = PreviewIndex)]
pub struct WasmPreviewIndex {
    /// Taken out for the duration of a `step()` and put back after it, so the
    /// pass is never borrowed across an await point.
    index: Rc<RefCell<Option<WebPreviewIndex>>>,
    store: PreviewStore,
    stride: u64,
    /// Whether the pass has a position left to visit, as of the last `step()`.
    remaining: Rc<Cell<bool>>,
    stepping: Rc<Cell<bool>>,
    /// The owning `MediaInput`'s closed flag, so a closed input's index says so
    /// rather than decoding from bytes the caller has dropped.
    state: Rc<Cell<bool>>,
}

impl WasmPreviewIndex {
    fn wrap(previews: WebPreviewIndex, state: Rc<Cell<bool>>) -> Self {
        Self {
            store: previews.store(),
            stride: previews.stride(),
            remaining: Rc::new(Cell::new(previews.next_frame().is_some())),
            index: Rc::new(RefCell::new(Some(previews))),
            stepping: Rc::new(Cell::new(false)),
            state,
        }
    }
}

#[wasm_bindgen(js_class = PreviewIndex)]
impl WasmPreviewIndex {
    /// Decodes the next preview into the index, resolving to whether any
    /// position is still unvisited.
    ///
    /// One preview per call is the point: the caller comes back through the
    /// event loop between them, so the page stays responsive while the index
    /// fills. A frame that will not decode leaves its position empty and the
    /// pass carries on - a gap costs a fallback to the neighbouring picture,
    /// not an error. Aborting is the one failure that does not advance the
    /// pass, so the next call asks for the same position again.
    pub fn step(&self, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        let index = Rc::clone(&self.index);
        let remaining = Rc::clone(&self.remaining);
        let stepping = Rc::clone(&self.stepping);
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            if stepping.get() {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "a preview step is already under way",
                ));
            }
            stepping.set(true);
            let cancellation = CancellationToken::new();
            let _abort = signal
                .as_ref()
                .map(|signal| cancel_on_abort(signal, &cancellation));
            let mut taken = index.borrow_mut().take();
            let result = match taken.as_mut() {
                Some(previews) => previews.step(&cancellation).await,
                None => Err(crate::Error::new(
                    ErrorKind::InvalidState,
                    "the preview index is unavailable",
                )),
            };
            *index.borrow_mut() = taken;
            stepping.set(false);
            match result {
                Ok(more) => {
                    remaining.set(more);
                    Ok(JsValue::from(more))
                }
                Err(error) => Err(js_error(error.kind(), error.message())),
            }
        })
    }

    /// The kept picture closest to `frameIndex`, or `null` while the pass has
    /// decoded nothing.
    ///
    /// A lookup ahead of where the pass has reached gets the newest picture
    /// behind it rather than nothing, so a drag over a part of the bar the pass
    /// has not covered yet still moves. This is a search and a copy: it never
    /// decodes and never waits on a decoder, which is what makes it safe to
    /// call on every pointer move.
    pub fn nearest(&self, frame_index: JsValue) -> Result<Option<WasmPreview>, JsValue> {
        let frame = CoreFrameIndex(parse_u64(&frame_index, "frame index")?);
        Ok(self
            .store
            .nearest_at(frame)
            .map(|(frame, picture)| WasmPreview { frame, picture }))
    }

    /// How many frames apart this index's previews are, as a `BigInt`.
    #[wasm_bindgen(getter)]
    pub fn stride(&self) -> JsValue {
        BigInt::from(self.stride).into()
    }

    /// How many of the index's positions hold a picture.
    #[wasm_bindgen(getter)]
    pub fn filled(&self) -> u32 {
        self.store.coverage().0 as u32
    }

    /// How many positions the index has in all.
    #[wasm_bindgen(getter)]
    pub fn total(&self) -> u32 {
        self.store.coverage().1 as u32
    }

    /// Whether the pass has visited every position, so the caller can stop
    /// scheduling `step()`.
    #[wasm_bindgen(getter)]
    pub fn complete(&self) -> bool {
        !self.remaining.get()
    }
}

#[wasm_bindgen(js_name = AudioStream)]
pub struct WasmAudioStream {
    index: u32,
    state: Rc<Cell<bool>>,
    direction: StreamDirection,
    bytes: Option<Rc<Vec<u8>>>,
    timeline: Option<Timeline>,
    /// Lazily-opened decode session for an input stream, built on the first
    /// read.
    decode_session: Rc<RefCell<Option<WebAudioDecodeSession>>>,
    /// Shared `WebCodecs` encode state for an output audio track, set only
    /// for the track index this bridge currently supports encoding (track 0).
    browser_audio: Option<Rc<RefCell<BrowserAudioTrack>>>,
}

#[wasm_bindgen(js_class = AudioStream)]
impl WasmAudioStream {
    #[wasm_bindgen(getter)]
    pub fn index(&self) -> u32 {
        self.index
    }

    #[wasm_bindgen(getter)]
    pub fn direction(&self) -> String {
        match self.direction {
            StreamDirection::Input => "input",
            StreamDirection::Output => "output",
        }
        .to_owned()
    }

    #[wasm_bindgen(js_name = intervalForFrame)]
    pub fn interval_for_frame(&self, frame_index: JsValue) -> Result<WasmSampleRange, JsValue> {
        ensure_open(&self.state)?;
        let frame_index = CoreFrameIndex(parse_u64(&frame_index, "frame index")?);
        let timeline = self.timeline.ok_or_else(|| {
            js_error(
                ErrorKind::Unsupported,
                "the stream does not expose timeline metadata yet",
            )
        })?;
        timeline
            .audio_interval_for_frame(frame_index)
            .map(WasmSampleRange)
            .map_err(|error| js_error(error.kind(), error.message()))
    }

    #[wasm_bindgen(js_name = aacConfig)]
    pub fn aac_config(&self, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        let index = self.index;
        let bytes = self.bytes.clone();
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            if direction != StreamDirection::Input {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "aacConfig is only valid on an input audio stream",
                ));
            }
            let track = parse_audio_track(bytes, index).await?;
            track
                .aac_config()
                .map(WasmAacConfig)
                .map(JsValue::from)
                .map_err(|error| js_error(error.kind(), error.message()))
        })
    }

    #[wasm_bindgen(js_name = packetCount)]
    pub fn packet_count(&self, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        let index = self.index;
        let bytes = self.bytes.clone();
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            if direction != StreamDirection::Input {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "packetCount is only valid on an input audio stream",
                ));
            }
            let track = parse_audio_track(bytes, index).await?;
            Ok(bigint_u64(track.samples.len() as u64))
        })
    }

    #[wasm_bindgen(js_name = packet)]
    pub fn packet(&self, packet_index: JsValue, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        let index = self.index;
        let bytes = self.bytes.clone();
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            let packet_index = usize::try_from(parse_u64(&packet_index, "packet index")?)
                .map_err(|_| js_error(ErrorKind::InvalidInput, "packet index is out of range"))?;
            if direction != StreamDirection::Input {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "packet is only valid on an input audio stream",
                ));
            }
            let bytes = bytes.ok_or_else(|| {
                js_error(
                    ErrorKind::Unsupported,
                    "input audio packet metadata is unavailable",
                )
            })?;
            let source = MemorySource::new((*bytes).clone());
            let track = parse_audio_track(Some(bytes), index).await?;
            let packets = track
                .to_encoded_audio_samples(&source, &Limits::default())
                .await
                .map_err(|error| js_error(error.kind(), error.message()))?;
            packets
                .get(packet_index)
                .cloned()
                .map(WasmEncodedAudioSample)
                .map(JsValue::from)
                .ok_or_else(|| js_error(ErrorKind::InvalidInput, "audio packet is not indexed"))
        })
    }

    pub fn get(&self, frame_index: JsValue, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            parse_u64(&frame_index, "frame index")?;
            if direction != StreamDirection::Input {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "get is only valid on an input audio stream",
                ));
            }
            Err(js_error(
                ErrorKind::Unsupported,
                "an input audio stream has no frame timeline; read it with getRange()",
            ))
        })
    }

    /// Decodes exactly the half-open range `[start, end)` of the track's
    /// presentation samples, as `BigInt`s, resolving to an `AudioBuffer`.
    ///
    /// The range is in the track's own sample clock after its priming, end
    /// padding and edit list are applied, so sample 0 is the first sample
    /// that was encoded and `sampleCount()` is one past the last. AAC and Opus
    /// tracks decode through the browser's `WebCodecs` `AudioDecoder` where it
    /// supports them; Opus falls back to zvidlib's own decoder where it does
    /// not.
    #[wasm_bindgen(js_name = getRange)]
    pub fn get_range(&self, start: JsValue, end: JsValue, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        let index = self.index;
        let bytes = self.bytes.clone();
        let decode_session = Rc::clone(&self.decode_session);
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            let range = CoreSampleRange::new(
                parse_u64(&start, "start sample")?,
                parse_u64(&end, "end sample")?,
            )
            .map_err(|error| js_error(error.kind(), error.message()))?;
            if direction != StreamDirection::Input {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "getRange is only valid on an input audio stream",
                ));
            }
            let mut session = open_audio_session(&decode_session, bytes, index).await?;
            let cancellation = CancellationToken::new();
            let _abort = signal
                .as_ref()
                .map(|signal| cancel_on_abort(signal, &cancellation));
            let result = session.get_range(range, &cancellation).await;
            *decode_session.borrow_mut() = Some(session);
            result
                .map(WasmAudioBuffer)
                .map(JsValue::from)
                .map_err(|error| js_error(error.kind(), error.message()))
        })
    }

    /// Resolves to the number of presentation samples the track has, as a
    /// `BigInt`: the end of the range `getRange()` reads.
    #[wasm_bindgen(js_name = sampleCount)]
    pub fn sample_count(&self, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        let index = self.index;
        let bytes = self.bytes.clone();
        let decode_session = Rc::clone(&self.decode_session);
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            if direction != StreamDirection::Input {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "sampleCount is only valid on an input audio stream",
                ));
            }
            let session = open_audio_session(&decode_session, bytes, index).await?;
            let length = session.presentation_length();
            *decode_session.borrow_mut() = Some(session);
            Ok(bigint_u64(length))
        })
    }

    /// Resolves to the track's `WebCodecs` `AudioDecoderConfig`: `codec`,
    /// `sampleRate`, `numberOfChannels`, and a `description` when the codec
    /// needs one, so a caller can drive its own `AudioDecoder` with the
    /// packets `packet()` returns. Works for AAC and Opus tracks.
    #[wasm_bindgen(js_name = decoderConfig)]
    pub fn decoder_config(&self, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        let index = self.index;
        let bytes = self.bytes.clone();
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            if direction != StreamDirection::Input {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "decoderConfig is only valid on an input audio stream",
                ));
            }
            let track = parse_audio_track(bytes, index).await?;
            let config = crate::web_audio_decoder::WebAudioDecoderConfig::for_track(&track)
                .map_err(|error| js_error(error.kind(), error.message()))?;
            let object = js_sys::Object::new();
            let set =
                |key: &str, value: &JsValue| Reflect::set(&object, &JsValue::from_str(key), value);
            set("codec", &JsValue::from_str(&config.codec))?;
            set("sampleRate", &JsValue::from(config.sample_rate))?;
            set("numberOfChannels", &JsValue::from(config.channels))?;
            if let Some(description) = &config.description {
                set("description", &owned_u8_array(description))?;
            }
            Ok(object.into())
        })
    }

    pub fn put(
        &self,
        frame_index: JsValue,
        buffer: &WasmAudioBuffer,
        signal: Option<AbortSignal>,
    ) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        let browser_audio = self.browser_audio.clone();
        let sample_rate = buffer.0.sample_rate;
        let channels = buffer.0.channels;
        let range_start = buffer.0.range.start;
        let samples = buffer.0.samples.clone();
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            let index = parse_u64(&frame_index, "frame index")?;
            if direction != StreamDirection::Output {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "put is only valid on an output audio stream",
                ));
            }
            let browser_audio = browser_audio.ok_or_else(|| {
                js_error(
                    ErrorKind::Unsupported,
                    "no browser audio encoder backend is registered for this track",
                )
            })?;
            encode_browser_audio_frame(
                &browser_audio,
                index,
                sample_rate,
                channels,
                range_start,
                samples,
            )
            .await
            .map_err(|error| js_error(error.kind(), error.message()))?;
            Ok(JsValue::UNDEFINED)
        })
    }
}

/// Takes the stream's decode session out of `slot`, opening it on first use,
/// so it is never borrowed across an await point. The caller puts it back.
async fn open_audio_session(
    slot: &Rc<RefCell<Option<WebAudioDecodeSession>>>,
    bytes: Option<Rc<Vec<u8>>>,
    index: u32,
) -> Result<WebAudioDecodeSession, JsValue> {
    if let Some(session) = slot.borrow_mut().take() {
        return Ok(session);
    }
    let bytes = bytes.ok_or_else(|| {
        js_error(
            ErrorKind::Unsupported,
            "input audio metadata is unavailable",
        )
    })?;
    WebAudioDecodeSession::open(&bytes, index, &Limits::default())
        .await
        .map_err(|error| js_error(error.kind(), error.message()))
}

/// A browser input whose source bytes are owned by WebAssembly after opening.
#[wasm_bindgen(js_name = MediaInput)]
pub struct WasmMediaInput {
    bytes: Rc<Vec<u8>>,
    state: Rc<Cell<bool>>,
    container: Option<Container>,
}

/// More than the leading bytes either container's signature occupies: an MP4
/// box header, or an EBML header, which a WebM probe bounds at 4 KiB.
const SIGNATURE_PREFIX_BYTES: usize = 8 * 1024;

const WEBM_AUDIO_UNSUPPORTED: &str =
    "WebM permits only Opus or Vorbis audio, so AAC audio needs an mp4 output";

impl WasmMediaInput {
    async fn open_inner(
        source: JsValue,
        max_input_bytes: u64,
        signal: Option<AbortSignal>,
    ) -> Result<Self, JsValue> {
        check_signal(signal.as_ref())?;
        let promise = read_browser_source(&source, max_input_bytes as f64, signal.as_ref())
            .map_err(|error| normalize_browser_error(error, "opening browser media"))?;
        let value = JsFuture::from(promise)
            .await
            .map_err(|error| normalize_browser_error(error, "opening browser media"))?;
        let bytes = value.dyn_into::<Uint8Array>().map_err(|_| {
            js_error(
                ErrorKind::Internal,
                "browser source adapter returned a non-byte value",
            )
        })?;
        let bytes = bytes.to_vec();
        // Detected from the bytes themselves, never a file name or MIME type.
        let prefix = MemorySource::new(&bytes[..bytes.len().min(SIGNATURE_PREFIX_BYTES)]);
        let container = crate::probe_container(&prefix)
            .await
            .map_err(|error| js_error(error.kind(), error.message()))?;
        Ok(Self {
            bytes: Rc::new(bytes),
            state: Rc::new(Cell::new(false)),
            container,
        })
    }
}

#[wasm_bindgen(js_class = MediaInput)]
impl WasmMediaInput {
    pub fn open(source: JsValue, options: Option<JsValue>) -> Promise {
        let parsed_options = parse_open_options(options);
        future_to_promise(async move {
            let (max_input_bytes, signal) = parsed_options?;
            Self::open_inner(source, max_input_bytes, signal)
                .await
                .map(JsValue::from)
        })
    }

    #[wasm_bindgen(getter, js_name = byteLength)]
    pub fn byte_length(&self) -> Result<JsValue, JsValue> {
        ensure_open(&self.state)?;
        Ok(bigint_u64(self.bytes.len() as u64))
    }

    pub fn bytes(&self) -> Result<Uint8Array, JsValue> {
        ensure_open(&self.state)?;
        Ok(owned_u8_array(&self.bytes))
    }

    /// The container `open` detected from the source's leading bytes:
    /// `"mp4"`, `"webm"`, or `null` when neither signature matched.
    #[wasm_bindgen(getter)]
    pub fn container(&self) -> Result<Option<String>, JsValue> {
        ensure_open(&self.state)?;
        Ok(self.container.map(|container| container.name().to_owned()))
    }

    pub fn video(&self, index: u32) -> Result<WasmVideoStream, JsValue> {
        ensure_open(&self.state)?;
        Ok(WasmVideoStream {
            index,
            state: Rc::clone(&self.state),
            direction: StreamDirection::Input,
            bytes: Some(Rc::clone(&self.bytes)),
            decode_session: Rc::new(RefCell::new(None)),
            frame_durations_ms: Rc::new(RefCell::new(None)),
            random_access_points: Rc::new(RefCell::new(None)),
            browser_video: None,
        })
    }

    pub fn audio(&self, index: u32) -> Result<WasmAudioStream, JsValue> {
        ensure_open(&self.state)?;
        Ok(WasmAudioStream {
            index,
            state: Rc::clone(&self.state),
            direction: StreamDirection::Input,
            bytes: Some(Rc::clone(&self.bytes)),
            timeline: None,
            decode_session: Rc::new(RefCell::new(None)),
            browser_audio: None,
        })
    }

    #[wasm_bindgen(getter, js_name = isClosed)]
    pub fn is_closed(&self) -> bool {
        self.state.get()
    }

    pub fn close(&mut self) {
        if !self.state.replace(true) {
            self.bytes = Rc::new(Vec::new());
        }
    }
}

/// Buffered `WebCodecs` video encode state for one output track's browser
/// encoder bridge, shared between [`WasmMediaOutput`] and the
/// [`WasmVideoStream`] it hands out for that track.
///
/// Building the real MP4 sample entry requires a complete `av1C` decoder
/// configuration, which `WebCodecs` only reveals once its encoder emits its
/// first chunk. So unlike the portable [`crate::codec::VideoEncoder`] trait,
/// samples are buffered here and the [`Mp4Muxer`] is not built until
/// [`WasmMediaOutput::finish`], once every real per-sample decoder detail is
/// known.
struct BrowserVideoTrack {
    session: Option<WebVideoEncodeSession>,
    dimensions: Option<(u32, u32)>,
    decoder_config: Option<Vec<u8>>,
    samples: Vec<EncodedSample>,
    next_index: u64,
    timescale: u32,
    frame_duration: u32,
    codec: Codec,
    /// The frame this track offers as the output's generated cover art.
    cover: CoverCapture,
}

impl BrowserVideoTrack {
    fn new(timescale: u32, frame_duration: u32, codec: Codec, cover_source: CoverSource) -> Self {
        Self {
            session: None,
            dimensions: None,
            decoder_config: None,
            samples: Vec::new(),
            next_index: 0,
            timescale,
            frame_duration,
            codec,
            cover: CoverCapture::new(cover_source),
        }
    }
}

/// Encodes one presentation-order frame through `track`'s `WebCodecs` session,
/// opening the session on the first call.
async fn encode_browser_video_frame(
    track: &Rc<RefCell<BrowserVideoTrack>>,
    index: u64,
    frame: CoreVideoFrame,
) -> crate::Result<()> {
    let width = frame.dimensions.width;
    let height = frame.dimensions.height;
    let (timescale, frame_duration) = {
        let state = track.borrow();
        if state.timescale == 0 || state.frame_duration == 0 {
            return Err(crate::Error::new(
                ErrorKind::InvalidState,
                "frameRate must be set via CreateOptions to encode browser video",
            ));
        }
        if index != state.next_index {
            return Err(crate::Error::new(
                ErrorKind::InvalidInput,
                "frame index must equal the next expected output index",
            ));
        }
        if let Some(dimensions) = state.dimensions
            && dimensions != (width, height)
        {
            return Err(crate::Error::new(
                ErrorKind::InvalidInput,
                "frame dimensions changed mid-stream",
            ));
        }
        (state.timescale, state.frame_duration)
    };
    track.borrow_mut().cover.offer(
        index,
        FrameSource::Cpu(CpuFrameSource {
            frame: &frame,
            orientation: Orientation::TopLeft,
        }),
    );

    let existing_session = {
        let mut state = track.borrow_mut();
        state.session.take()
    };
    let mut session = match existing_session {
        Some(session) => session,
        None => {
            let codec = track.borrow().codec;
            let session =
                WebVideoEncodeSession::open(codec, width, height, timescale, frame_duration, None)
                    .await?;
            track.borrow_mut().dimensions = Some((width, height));
            session
        }
    };

    let key_frame = index == 0;
    let ticks = index
        .checked_mul(u64::from(frame_duration))
        .ok_or_else(|| crate::Error::new(ErrorKind::ResourceLimit, "frame timestamp overflow"))?;
    let timestamp_micros = ticks
        .checked_mul(1_000_000)
        .and_then(|value| value.checked_div(u64::from(timescale)))
        .ok_or_else(|| crate::Error::new(ErrorKind::ResourceLimit, "frame timestamp overflow"))?;
    let duration_micros = u32::try_from(
        u64::from(frame_duration)
            .checked_mul(1_000_000)
            .and_then(|value| value.checked_div(u64::from(timescale)))
            .ok_or_else(|| {
                crate::Error::new(ErrorKind::ResourceLimit, "frame duration overflow")
            })?,
    )
    .map_err(|_| crate::Error::new(ErrorKind::ResourceLimit, "frame duration does not fit"))?;

    let result = session
        .encode(&frame, timestamp_micros, duration_micros, key_frame)
        .await;

    let mut state = track.borrow_mut();
    state.session = Some(session);
    let chunk = result?;

    if state.decoder_config.is_none()
        && let Some(config) = chunk.decoder_config
    {
        state.decoder_config = Some(config);
    }
    let pts = i64::try_from(ticks).map_err(|_| {
        crate::Error::new(ErrorKind::ResourceLimit, "presentation time does not fit")
    })?;
    state.samples.push(EncodedSample {
        data: chunk.data,
        dts: pts,
        pts,
        duration: frame_duration,
        is_sync: chunk.is_sync,
        dependency: sample_dependency(chunk.is_sync),
    });
    state.next_index += 1;
    Ok(())
}

/// Buffered `WebCodecs` audio encode state for the output audio track's
/// browser encoder bridge, the audio counterpart to [`BrowserVideoTrack`].
///
/// AAC's `esds` and Opus's `dOps` decoder configurations are genuinely
/// out-of-band (unlike AV1's, which travels in the bitstream itself), but
/// `WebCodecs` still only reports them in the metadata attached to the
/// encoder's first emitted chunk rather than up front, so this defers building
/// the MP4 sample entry the same way.
struct BrowserAudioTrack {
    codec: Codec,
    session: Option<WebAudioEncodeSession>,
    /// zvidlib's own encoder, for Vorbis, which no browser encodes, and for
    /// Opus in a browser without a `WebCodecs` `AudioEncoder`.
    software: Option<Box<dyn crate::AudioEncoder>>,
    /// The trim the software encoder reported when it drained.
    software_gapless: Option<crate::AudioGapless>,
    sample_rate: Option<u32>,
    channels: Option<u16>,
    decoder_config: Option<Vec<u8>>,
    samples: Vec<EncodedSample>,
    next_put_index: u64,
    /// PCM frames put so far, from which an Opus track's end padding is
    /// measured.
    input_frames: u64,
}

impl BrowserAudioTrack {
    fn new(codec: Codec) -> Self {
        Self {
            codec,
            session: None,
            software: None,
            software_gapless: None,
            sample_rate: None,
            channels: None,
            decoder_config: None,
            samples: Vec::new(),
            next_put_index: 0,
            input_frames: 0,
        }
    }
}

/// Converts a `WebCodecs` microsecond timestamp/duration into an exact sample
/// count at `sample_rate`, which is what an MP4 audio track's timescale (set
/// equal to the sample rate) expects.
fn micros_to_samples(micros: f64, sample_rate: u32) -> crate::Result<u64> {
    let samples = (micros * f64::from(sample_rate) / 1_000_000.0).round();
    if !samples.is_finite() || samples < 0.0 || samples > u64::MAX as f64 {
        return Err(crate::Error::new(
            ErrorKind::ResourceLimit,
            "audio timestamp does not fit the track's sample clock",
        ));
    }
    Ok(samples as u64)
}

/// Encodes one buffer of interleaved `f32` PCM through `track`'s `WebCodecs`
/// session, opening the session on the first call.
async fn encode_browser_audio_frame(
    track: &Rc<RefCell<BrowserAudioTrack>>,
    index: u64,
    sample_rate: u32,
    channels: u16,
    range_start: u64,
    samples: Vec<f32>,
) -> crate::Result<()> {
    {
        let state = track.borrow();
        if state.codec == Codec::Opus && sample_rate != crate::OPUS_SAMPLE_RATE {
            return Err(crate::Error::new(
                ErrorKind::InvalidInput,
                "Opus is encoded from 48000 Hz audio, the rate it decodes at",
            ));
        }
        if index != state.next_put_index {
            return Err(crate::Error::new(
                ErrorKind::InvalidInput,
                "frame index must equal the next expected output index",
            ));
        }
        if let Some(existing) = state.sample_rate
            && existing != sample_rate
        {
            return Err(crate::Error::new(
                ErrorKind::InvalidInput,
                "audio sample rate changed mid-stream",
            ));
        }
        if let Some(existing) = state.channels
            && existing != channels
        {
            return Err(crate::Error::new(
                ErrorKind::InvalidInput,
                "audio channel count changed mid-stream",
            ));
        }
    }

    let codec = track.borrow().codec;
    if encodes_in_software(codec) {
        return encode_software_audio_frame(
            track,
            index,
            sample_rate,
            channels,
            range_start,
            samples,
        )
        .await;
    }
    let mut session = {
        let mut state = track.borrow_mut();
        match state.session.take() {
            Some(session) => session,
            None => {
                let session =
                    WebAudioEncodeSession::open(state.codec, sample_rate, channels, None)?;
                state.sample_rate = Some(sample_rate);
                state.channels = Some(channels);
                session
            }
        }
    };

    let timestamp_micros = (range_start as f64) * 1_000_000.0 / f64::from(sample_rate);
    let frames = (samples.len() / usize::from(channels.max(1))) as u64;
    let result = session.encode(&samples, timestamp_micros).await;

    let mut state = track.borrow_mut();
    state.session = Some(session);
    let chunks = result?;
    push_audio_chunks(&mut state, sample_rate, chunks)?;
    state.next_put_index += 1;
    state.input_frames += frames;
    Ok(())
}

/// Whether `codec` is encoded by zvidlib's own encoder rather than
/// `WebCodecs`: always for Vorbis, which no browser encodes, and for Opus
/// where the browser has no `AudioEncoder`.
fn encodes_in_software(codec: Codec) -> bool {
    match codec {
        Codec::Vorbis => true,
        Codec::Opus => {
            !Reflect::has(&js_sys::global(), &JsValue::from_str("AudioEncoder")).unwrap_or(false)
        }
        _ => false,
    }
}

/// Encodes one buffer through zvidlib's own Opus or Vorbis encoder, creating
/// it on the first call.
async fn encode_software_audio_frame(
    track: &Rc<RefCell<BrowserAudioTrack>>,
    index: u64,
    sample_rate: u32,
    channels: u16,
    range_start: u64,
    samples: Vec<f32>,
) -> crate::Result<()> {
    let mut encoder = {
        let mut state = track.borrow_mut();
        match state.software.take() {
            Some(encoder) => encoder,
            None => {
                let configuration = crate::AudioEncoderConfig {
                    codec: state.codec,
                    profile: if state.codec == Codec::Vorbis {
                        CodecProfile::Vorbis
                    } else {
                        CodecProfile::Opus
                    },
                    sample_rate,
                    channels,
                    timescale: sample_rate,
                    configuration: Vec::new(),
                };
                let encoder = if state.codec == Codec::Vorbis {
                    crate::AudioEncoderFactory::create(
                        &crate::native_vorbis_audio_encoder_factory(),
                        &configuration,
                        &Limits::default(),
                    )?
                } else {
                    crate::AudioEncoderFactory::create(
                        &crate::native_opus_audio_encoder_factory(),
                        &configuration,
                        &Limits::default(),
                    )?
                };
                state.sample_rate = Some(sample_rate);
                state.channels = Some(channels);
                state.decoder_config = Some(encoder.config().decoder_config.clone());
                encoder
            }
        }
    };
    let frames = (samples.len() / usize::from(channels.max(1))) as u64;
    let result = match CoreAudioBuffer::new(
        CoreSampleRange::new(range_start, range_start + frames)?,
        sample_rate,
        channels,
        samples,
        &Limits::default(),
    ) {
        Ok(buffer) => encoder.encode(CoreFrameIndex(index), buffer).await,
        Err(error) => Err(error),
    };
    let mut state = track.borrow_mut();
    state.software = Some(encoder);
    state.samples.extend(result?);
    state.next_put_index += 1;
    state.input_frames += frames;
    Ok(())
}

/// Appends every newly-produced chunk to `state.samples`, recording the first
/// decoder configuration `WebCodecs` reports.
fn push_audio_chunks(
    state: &mut BrowserAudioTrack,
    sample_rate: u32,
    chunks: Vec<crate::web_encoder::WebEncodedAudioChunk>,
) -> crate::Result<()> {
    for chunk in chunks {
        if state.decoder_config.is_none()
            && let Some(config) = chunk.decoder_config
        {
            state.decoder_config = Some(config);
        }
        // Opus packets follow one another without gaps, and the encoder's
        // delay is the track's pre-skip rather than a timestamp offset, so an
        // Opus sample starts where the one before it ended.
        let pts = match (state.codec, state.samples.last()) {
            (Codec::Opus, Some(previous)) => previous.pts + i64::from(previous.duration),
            (Codec::Opus, None) => 0,
            _ => i64::try_from(micros_to_samples(chunk.timestamp_micros, sample_rate)?).map_err(
                |_| crate::Error::new(ErrorKind::ResourceLimit, "presentation time does not fit"),
            )?,
        };
        let duration_micros = chunk.duration_micros.ok_or_else(|| {
            crate::Error::new(
                ErrorKind::Internal,
                "WebCodecs audio encoder reported a chunk with no duration",
            )
        })?;
        let duration =
            u32::try_from(micros_to_samples(duration_micros, sample_rate)?).map_err(|_| {
                crate::Error::new(ErrorKind::ResourceLimit, "frame duration does not fit")
            })?;
        state.samples.push(EncodedSample {
            data: chunk.data,
            dts: pts,
            pts,
            duration,
            is_sync: chunk.is_sync,
            dependency: sample_dependency(chunk.is_sync),
        });
    }
    Ok(())
}

/// Flushes an open browser audio encode session and, if the track produced
/// any samples, returns its complete MP4 track declaration, samples, and the
/// gapless trim its edit list needs.
async fn finalize_audio_track(
    track: &Rc<RefCell<BrowserAudioTrack>>,
) -> crate::Result<Option<(Mp4TrackConfig, Vec<EncodedSample>, crate::AudioGapless)>> {
    let pending_session = {
        let mut state = track.borrow_mut();
        state.session.take()
    };
    let pending_software = track.borrow_mut().software.take();
    if pending_session.is_none() && pending_software.is_none() && track.borrow().samples.is_empty()
    {
        return Ok(None);
    }
    if let Some(mut encoder) = pending_software {
        let drain = encoder.finish().await?;
        let mut state = track.borrow_mut();
        state.samples.extend(drain.samples);
        state.software_gapless = Some(drain.gapless);
    }
    if let Some(mut session) = pending_session {
        let remaining = session.finish().await?;
        let sample_rate = track.borrow().sample_rate.ok_or_else(|| {
            crate::Error::new(
                ErrorKind::Internal,
                "browser audio track was opened but never learned its sample rate",
            )
        })?;
        let mut state = track.borrow_mut();
        push_audio_chunks(&mut state, sample_rate, remaining)?;
    }

    let (codec, input_frames) = {
        let state = track.borrow();
        (state.codec, state.input_frames)
    };
    let (channels, decoder_config, sample_rate, samples) = {
        let mut state = track.borrow_mut();
        let channels = state.channels.ok_or_else(|| {
            crate::Error::new(
                ErrorKind::Internal,
                "browser audio track was opened but never encoded a buffer",
            )
        })?;
        let decoder_config = state.decoder_config.clone().ok_or_else(|| {
            crate::Error::new(
                ErrorKind::Internal,
                "browser audio track produced no decoder configuration",
            )
        })?;
        let sample_rate = state.sample_rate.expect("checked above via channels");
        (
            channels,
            decoder_config,
            sample_rate,
            std::mem::take(&mut state.samples),
        )
    };

    // An Opus encoder's delay is its pre-skip, and the silence it codes after
    // the input to flush that delay out is end padding, so the edit list trims
    // both and the track presents exactly what was put.
    let software_gapless = track.borrow().software_gapless;
    let gapless = if let Some(gapless) = software_gapless {
        gapless
    } else if codec == Codec::Opus {
        let priming = u32::from(crate::OpusHead::from_dops(&decoder_config)?.pre_skip);
        let encoded: u64 = samples
            .iter()
            .map(|sample| u64::from(sample.duration))
            .sum();
        let padding = encoded.saturating_sub(input_frames + u64::from(priming));
        crate::AudioGapless {
            priming,
            padding: u32::try_from(padding).map_err(|_| {
                crate::Error::new(ErrorKind::ResourceLimit, "Opus padding exceeds u32")
            })?,
        }
    } else {
        crate::AudioGapless::default()
    };
    let track_config = Mp4TrackConfig {
        encoder: EncoderConfig {
            codec,
            timescale: sample_rate,
            decoder_config,
        },
        format: Mp4TrackFormat::Audio { channels },
    };
    Ok(Some((track_config, samples, gapless)))
}

/// Flushes any open browser video encode session on every video track and any
/// open browser audio encode session and, if at least one track produced
/// samples, muxes them together into a complete MP4 or WebM. Falls back to
/// `raw_bytes` (accumulated via [`WasmMediaOutput::write_encoded_chunk`])
/// when no browser encode path was ever used, and silently omits a video
/// track that was opened via [`WasmMediaOutput::video`] but never encoded a
/// frame.
///
/// Video tracks are muxed first, in ascending output track index order,
/// renumbered to the contiguous positions the muxer expects (the original
/// indices requested through `video()` are not preserved in the output MP4);
/// the audio track, if used, is muxed last. WebM output carries no audio
/// and no cover art, and interleaves its tracks by presentation time.
/// the audio track, if used, is muxed last.
async fn finalize_browser_output(
    container: Container,
    tracks: &Rc<RefCell<BTreeMap<u32, Rc<RefCell<BrowserVideoTrack>>>>>,
    audio: &Rc<RefCell<BrowserAudioTrack>>,
    raw_bytes: Vec<u8>,
    cover_art: Option<CoverArt>,
) -> crate::Result<Vec<u8>> {
    let ordered: Vec<Rc<RefCell<BrowserVideoTrack>>> =
        tracks.borrow().values().map(Rc::clone).collect();

    let mut track_configs = Vec::new();
    let mut track_samples = Vec::new();
    let mut generated_cover = None;
    for track in &ordered {
        let pending_session = {
            let mut state = track.borrow_mut();
            state.session.take()
        };
        if let Some(mut session) = pending_session {
            let remaining = session.finish().await?;
            let mut state = track.borrow_mut();
            for chunk in remaining {
                if state.decoder_config.is_none()
                    && let Some(config) = chunk.decoder_config
                {
                    state.decoder_config = Some(config);
                }
                let ticks = state
                    .next_index
                    .checked_mul(u64::from(state.frame_duration))
                    .ok_or_else(|| {
                        crate::Error::new(ErrorKind::ResourceLimit, "presentation time overflow")
                    })?;
                let pts = i64::try_from(ticks).map_err(|_| {
                    crate::Error::new(ErrorKind::ResourceLimit, "presentation time does not fit")
                })?;
                let frame_duration = state.frame_duration;
                state.samples.push(EncodedSample {
                    data: chunk.data,
                    dts: pts,
                    pts,
                    duration: frame_duration,
                    is_sync: chunk.is_sync,
                    dependency: sample_dependency(chunk.is_sync),
                });
                state.next_index += 1;
            }
        }

        let has_activity = track.borrow().dimensions.is_some();
        if !has_activity {
            continue;
        }
        // The first muxed video track supplies the generated cover.
        if track_configs.is_empty() && container == Container::Mp4 {
            generated_cover = track.borrow().cover.cover_art();
        }

        let (coded_dimensions, decoder_config, timescale, samples, codec) = {
            let mut state = track.borrow_mut();
            let dimensions = state.dimensions.ok_or_else(|| {
                crate::Error::new(
                    ErrorKind::Internal,
                    "browser video track was opened but never encoded a frame",
                )
            })?;
            let decoder_config = state.decoder_config.clone().ok_or_else(|| {
                crate::Error::new(
                    ErrorKind::Internal,
                    "browser video track produced no decoder configuration",
                )
            })?;
            (
                dimensions,
                decoder_config,
                state.timescale,
                std::mem::take(&mut state.samples),
                state.codec,
            )
        };

        let video_dimensions =
            VideoDimensions::new(coded_dimensions.0, coded_dimensions.1, &Limits::default())?;
        track_configs.push(Mp4TrackConfig {
            encoder: EncoderConfig {
                codec,
                timescale,
                decoder_config,
            },
            format: Mp4TrackFormat::Video(video_dimensions),
        });
        track_samples.push(samples);
    }

    let mut audio_gapless = None;
    if let Some((config, samples, gapless)) = finalize_audio_track(audio).await? {
        audio_gapless = Some((track_configs.len(), gapless));
        track_configs.push(config);
        track_samples.push(samples);
    }

    if track_configs.is_empty() {
        return Ok(raw_bytes);
    }

    if container == Container::WebM {
        return mux_browser_webm(track_configs, track_samples, audio_gapless).await;
    }

    let sink = MemorySink::new();
    let mut muxer = Mp4Muxer::new(
        sink,
        track_configs,
        crate::OutputOptions::default().max_samples_per_track,
    )
    .await?;
    for (index, samples) in track_samples.into_iter().enumerate() {
        for sample in samples {
            muxer.write_sample(index, sample).await?;
        }
    }
    if let Some((index, gapless)) = audio_gapless
        && gapless != crate::AudioGapless::default()
    {
        muxer.set_audio_gapless(index, gapless)?;
    }
    muxer.set_cover_art(cover_art.or(generated_cover))?;
    let sink = muxer.finish().await?;
    Ok(sink.into_inner())
}

/// Muxes buffered browser tracks into a WebM. [`WebmMuxer`] takes
/// samples in one presentation-time order across tracks, so the per-track
/// buffers are merged by exact presentation time, each track keeping its own
/// order and earlier tracks winning ties.
async fn mux_browser_webm(
    track_configs: Vec<Mp4TrackConfig>,
    track_samples: Vec<Vec<EncodedSample>>,
    audio_gapless: Option<(usize, crate::AudioGapless)>,
) -> crate::Result<Vec<u8>> {
    let timescales: Vec<i128> = track_configs
        .iter()
        .map(|config| i128::from(config.encoder.timescale))
        .collect();
    let mut merged: Vec<(usize, EncodedSample)> = track_samples
        .into_iter()
        .enumerate()
        .flat_map(|(track, samples)| samples.into_iter().map(move |sample| (track, sample)))
        .collect();
    merged.sort_by(|(left_track, left), (right_track, right)| {
        (i128::from(left.pts) * timescales[*right_track])
            .cmp(&(i128::from(right.pts) * timescales[*left_track]))
            .then(left_track.cmp(right_track))
    });
    let mut muxer = WebmMuxer::new(
        MemorySink::new(),
        track_configs,
        crate::OutputOptions::default().max_samples_per_track,
    )
    .await?;
    // An audio track's trim is declared right after its last sample, before
    // a later video sample can write that sample's block without it.
    let mut remaining: Vec<usize> = vec![0; timescales.len()];
    for (track, _) in &merged {
        remaining[*track] += 1;
    }
    for (track, sample) in merged {
        muxer.write_sample(track, sample).await?;
        remaining[track] -= 1;
        if remaining[track] == 0
            && let Some((audio_track, gapless)) = audio_gapless
            && audio_track == track
        {
            muxer.set_audio_gapless(track, gapless)?;
        }
    }
    Ok(muxer.finish().await?.into_inner())
}

/// A browser output adapter that returns finalized bytes as an owned `Blob`.
#[wasm_bindgen(js_name = MediaOutput)]
pub struct WasmMediaOutput {
    bytes: Vec<u8>,
    mime_type: String,
    /// Which container `finish` muxes the browser-encoded tracks into.
    container: Container,
    max_output_bytes: u64,
    state: Rc<Cell<bool>>,
    timeline: Option<Timeline>,
    /// MP4 timescale/frame-duration shared by every browser-encoded video
    /// track, derived once from `CreateOptions` and applied lazily as each
    /// track index is first requested through [`WasmMediaOutput::video`].
    video_timescale: u32,
    video_frame_duration: u32,
    video_codec: Codec,
    /// One `WebCodecs` encode bridge per output video track index that has
    /// been requested through [`WasmMediaOutput::video`], keyed by that
    /// track index.
    browser_video_tracks: Rc<RefCell<BTreeMap<u32, Rc<RefCell<BrowserVideoTrack>>>>>,
    browser_audio: Rc<RefCell<BrowserAudioTrack>>,
    cover_art: Option<CoverArt>,
    cover_source: CoverSource,
}

#[wasm_bindgen(js_class = MediaOutput)]
impl WasmMediaOutput {
    pub fn create(options: &WasmCreateOptions) -> Promise {
        let mime_type = options.mime_type.clone();
        let container = options.container;
        let max_output_bytes = options.max_output_bytes;
        let video_codec = options.video_codec;
        let audio_codec = options.audio_codec;
        let cover_source = options.cover_source;
        let timeline = match (options.frame_rate, options.audio_sample_rate) {
            (Some(frame_rate), Some(sample_rate)) => Timeline::new(frame_rate, sample_rate).ok(),
            _ => None,
        };
        // Browser video encoding needs an exact MP4 timescale/frame duration
        // up front, derived the same way `Timeline` derives one: numerator as
        // timescale, denominator as the per-frame tick count. `put()` on the
        // resulting video stream reports a clear error when `frameRate` was
        // never set instead of failing here.
        let (video_timescale, video_frame_duration) = options
            .frame_rate
            .and_then(|frame_rate| {
                let rational = frame_rate.as_rational();
                let timescale = u32::try_from(rational.numerator()).ok()?;
                let frame_duration = u32::try_from(rational.denominator()).ok()?;
                Some((timescale, frame_duration))
            })
            .unwrap_or((0, 0));
        future_to_promise(async move {
            Ok(JsValue::from(Self {
                bytes: Vec::new(),
                mime_type,
                container,
                max_output_bytes,
                state: Rc::new(Cell::new(false)),
                timeline,
                video_timescale,
                video_frame_duration,
                video_codec,
                browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
                browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(audio_codec))),
                cover_art: None,
                cover_source,
            }))
        })
    }

    /// Appends already-encoded container bytes from a future muxer backend.
    #[wasm_bindgen(js_name = writeEncodedChunk)]
    pub fn write_encoded_chunk(&mut self, chunk: Uint8Array) -> Result<(), JsValue> {
        ensure_open(&self.state)?;
        let length = u64::from(chunk.length());
        let next_length = (self.bytes.len() as u64)
            .checked_add(length)
            .ok_or_else(|| js_error(ErrorKind::ResourceLimit, "output size overflow"))?;
        if next_length > self.max_output_bytes {
            return Err(js_error(
                ErrorKind::ResourceLimit,
                "browser output exceeds maxOutputBytes",
            ));
        }
        self.bytes.extend_from_slice(&chunk.to_vec());
        Ok(())
    }

    pub fn video(&self, index: u32) -> Result<WasmVideoStream, JsValue> {
        ensure_open(&self.state)?;
        let browser_video = {
            let mut tracks = self.browser_video_tracks.borrow_mut();
            Rc::clone(tracks.entry(index).or_insert_with(|| {
                Rc::new(RefCell::new(BrowserVideoTrack::new(
                    self.video_timescale,
                    self.video_frame_duration,
                    self.video_codec,
                    self.cover_source,
                )))
            }))
        };
        let browser_video = Some(browser_video);
        Ok(WasmVideoStream {
            index,
            state: Rc::clone(&self.state),
            direction: StreamDirection::Output,
            bytes: None,
            decode_session: Rc::new(RefCell::new(None)),
            frame_durations_ms: Rc::new(RefCell::new(None)),
            random_access_points: Rc::new(RefCell::new(None)),
            browser_video,
        })
    }

    /// Output audio track `index`. Track 0 encodes in the codec
    /// `CreateOptions.audioCodec` chose; other indices report `UNSUPPORTED`
    /// from `put()`.
    pub fn audio(&self, index: u32) -> Result<WasmAudioStream, JsValue> {
        ensure_open(&self.state)?;
        if self.container == Container::WebM && self.browser_audio.borrow().codec == Codec::Aac {
            return Err(js_error(ErrorKind::Unsupported, WEBM_AUDIO_UNSUPPORTED));
        }
        // The WebCodecs bridge currently supports encoding a single audio
        // track (track 0); `put()` on any other index reports Unsupported.
        let browser_audio = (index == 0).then(|| Rc::clone(&self.browser_audio));
        Ok(WasmAudioStream {
            index,
            state: Rc::clone(&self.state),
            direction: StreamDirection::Output,
            bytes: None,
            timeline: self.timeline,
            decode_session: Rc::new(RefCell::new(None)),
            browser_audio,
        })
    }

    /// Sets the cover art embedded in the finished MP4 as its file-browser
    /// thumbnail. `mimeType` is `image/jpeg` or `image/png`; `null` clears it.
    /// May be called any time before `finish()`.
    /// WebM has no cover-art element, so a `"webm"` output refuses a picture.
    #[wasm_bindgen(js_name = setCoverArt)]
    pub fn set_cover_art(
        &mut self,
        data: Option<Uint8Array>,
        mime_type: Option<String>,
    ) -> Result<(), JsValue> {
        ensure_open(&self.state)?;
        let Some(data) = data else {
            self.cover_art = None;
            return Ok(());
        };
        if self.container == Container::WebM {
            return Err(js_error(
                ErrorKind::Unsupported,
                "WebM output cannot carry cover art; setCoverArt needs an mp4 output",
            ));
        }
        let format = match mime_type.as_deref() {
            Some("image/jpeg") => CoverArtFormat::Jpeg,
            Some("image/png") => CoverArtFormat::Png,
            _ => {
                return Err(js_error(
                    ErrorKind::InvalidInput,
                    "cover art mimeType must be image/jpeg or image/png",
                ));
            }
        };
        if data.length() == 0 {
            return Err(js_error(
                ErrorKind::InvalidInput,
                "cover art data must be nonempty",
            ));
        }
        self.cover_art = Some(CoverArt {
            format,
            data: data.to_vec(),
        });
        Ok(())
    }

    pub fn finish(&mut self) -> Promise {
        let state = Rc::clone(&self.state);
        let container = self.container;
        let mime_type = self.mime_type.clone();
        let browser_video_tracks = Rc::clone(&self.browser_video_tracks);
        let browser_audio = Rc::clone(&self.browser_audio);
        let raw_bytes = std::mem::take(&mut self.bytes);
        let cover_art = self.cover_art.clone();
        future_to_promise(async move {
            ensure_open(&state)?;
            let bytes = finalize_browser_output(
                container,
                &browser_video_tracks,
                &browser_audio,
                raw_bytes,
                cover_art,
            )
            .await
            .map_err(|error| js_error(error.kind(), error.message()))?;
            let blob = make_blob(&bytes, &mime_type)
                .map_err(|error| normalize_browser_error(error, "creating output Blob"))?;
            state.set(true);
            Ok(JsValue::from(blob))
        })
    }

    #[wasm_bindgen(getter, js_name = isClosed)]
    pub fn is_closed(&self) -> bool {
        self.state.get()
    }

    pub fn close(&mut self) {
        if !self.state.replace(true) {
            self.bytes.clear();
            self.cover_art = None;
        }
    }
}

#[wasm_bindgen(js_name = Playback)]
pub struct WasmPlayback {
    state: Rc<Cell<bool>>,
    _options: WasmPlaybackOptions,
}

#[wasm_bindgen(js_class = Playback)]
impl WasmPlayback {
    pub fn create(
        video: &WasmVideoStream,
        audio: &WasmAudioStream,
        options: Option<JsValue>,
    ) -> Promise {
        let result = (|| {
            ensure_open(&video.state)?;
            ensure_open(&audio.state)?;
            if video.direction != StreamDirection::Input
                || audio.direction != StreamDirection::Input
            {
                return Err(js_error(
                    ErrorKind::InvalidInput,
                    "playback requires input video and audio streams",
                ));
            }
            if !Rc::ptr_eq(&video.state, &audio.state) {
                return Err(js_error(
                    ErrorKind::InvalidInput,
                    "playback streams must belong to the same media input",
                ));
            }
            Ok(JsValue::from(Self {
                state: Rc::new(Cell::new(false)),
                _options: parse_playback_options(options)?,
            }))
        })();
        future_to_promise(async move { result })
    }

    pub fn play(&self, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            Err(js_error(
                ErrorKind::Unsupported,
                "no browser playback backend is registered",
            ))
        })
    }

    pub fn present(&self, signal: Option<AbortSignal>) -> Promise {
        let state = Rc::clone(&self.state);
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            Err(js_error(
                ErrorKind::Unsupported,
                "no browser playback backend is registered",
            ))
        })
    }

    #[wasm_bindgen(getter, js_name = isClosed)]
    pub fn is_closed(&self) -> bool {
        self.state.get()
    }

    pub fn close(&mut self) {
        self.state.set(true);
        self._options = WasmPlaybackOptions::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use js_sys::Object;
    use wasm_bindgen_test::*;
    use web_sys::AbortController;

    wasm_bindgen_test_configure!(run_in_browser);

    fn assert_error_code(error: &JsValue, expected: &str) {
        assert_eq!(error_code(error).as_deref(), Some(expected));
        assert!(error.is_instance_of::<js_sys::Error>());
    }

    #[wasm_bindgen_test]
    fn aborting_cancels_the_decode_already_under_way() {
        // `check_signal` only brackets an operation. A scrub that replaces a request needs the
        // decode inside it to stop, which is what the bridged token gives the decode loop.
        let controller = AbortController::new().unwrap();
        let cancellation = CancellationToken::new();
        let _listener = cancel_on_abort(&controller.signal(), &cancellation);
        assert!(!cancellation.is_cancelled());
        controller.abort();
        assert!(
            cancellation.is_cancelled(),
            "aborting the signal cancels the decode it was passed to"
        );
    }

    #[wasm_bindgen_test]
    fn bigints_round_trip_and_unsafe_numbers_are_rejected() {
        let maximum = BigInt::from(u64::MAX);
        let index = WasmFrameIndex::new(maximum.into()).unwrap();
        assert_eq!(parse_u64(&index.value(), "frame").unwrap(), u64::MAX);

        let unsafe_number = JsValue::from_f64(MAX_SAFE_INTEGER as f64 + 1.0);
        let error = match WasmFrameIndex::new(unsafe_number) {
            Err(error) => error,
            Ok(_) => panic!("an unsafe integer must be rejected"),
        };
        assert_error_code(&error, "INVALID_INPUT");
    }

    #[wasm_bindgen_test(async)]
    async fn typed_array_and_blob_inputs_are_owned_copies() {
        let original = Uint8Array::from(&[1_u8, 2, 3][..]);
        let input = WasmMediaInput::open_inner(
            original.clone().into(),
            Limits::default().max_allocation_bytes,
            None,
        )
        .await
        .unwrap();
        original.set_index(0, 99);
        assert_eq!(input.bytes().unwrap().to_vec(), vec![1, 2, 3]);

        let blob = make_blob(&[4, 5, 6], "video/mp4").unwrap();
        let input =
            WasmMediaInput::open_inner(blob.into(), Limits::default().max_allocation_bytes, None)
                .await
                .unwrap();
        assert_eq!(input.bytes().unwrap().to_vec(), vec![4, 5, 6]);
    }

    /// The bundled HEVC sample decoded through `get()`.
    ///
    /// Not every headless Chrome build can decode HEVC through `WebCodecs`
    /// (it depends on platform codec licensing); where it cannot, the software
    /// fallback decodes it instead (issue #504), so either way this must be a
    /// real decoded RGBA frame. Fetches two sequential presentation frames,
    /// matching how the `web_canvas` example plays back frame by frame, and
    /// checks that each frame's reported dimensions match its own pixel buffer
    /// rather than a stale, session-wide value.
    #[wasm_bindgen_test(async)]
    async fn video_get_decodes_the_bundled_sample() {
        const SAMPLE: &[u8] = include_bytes!("../examples/media/BigBuckBunny.mp4");
        let bytes = Uint8Array::from(SAMPLE);
        let input =
            WasmMediaInput::open_inner(bytes.into(), Limits::default().max_allocation_bytes, None)
                .await
                .unwrap();
        let video = input.video(0).unwrap();
        assert_eq!(video.direction(), "input");

        let duration = JsFuture::from(video.frame_duration(BigInt::from(0_u64).into(), None))
            .await
            .unwrap()
            .as_f64()
            .unwrap();
        assert!((duration - (1_000.0 / 24.0)).abs() < 0.001);

        for frame_index in [0_u64, 1_u64] {
            let frame = JsFuture::from(video.get(BigInt::from(frame_index).into(), None))
                .await
                .expect("the software fallback decodes HEVC Main wherever WebCodecs cannot");
            // `WasmVideoFrame` doesn't implement `JsCast`, so read its
            // wasm-bindgen getters back through `Reflect` instead.
            let get_u32 = |name: &str| -> u32 {
                Reflect::get(&frame, &JsValue::from_str(name))
                    .unwrap()
                    .as_f64()
                    .unwrap() as u32
            };
            let width = get_u32("width");
            let height = get_u32("height");
            assert_eq!((width, height), (1920, 1080));
            let pixels = Reflect::get(&frame, &JsValue::from_str("pixels")).unwrap();
            let pixels: Uint8Array = pixels.unchecked_into();
            assert_eq!(pixels.length(), width * height * 4);
        }
    }

    /// Issue #527: a VP9-in-MP4 track opens and decodes through `get()`,
    /// through `WebCodecs` where the browser decodes VP9 and the software
    /// decoder otherwise, for frames on either side of its second key frame.
    #[wasm_bindgen_test(async)]
    async fn video_get_decodes_a_vp9_track() {
        const SAMPLE: &[u8] =
            include_bytes!("../crates/zvidlib-vp9-decoder/tests/fixtures/vp9_bbb_256x144.mp4");
        let bytes = Uint8Array::from(SAMPLE);
        let input =
            WasmMediaInput::open_inner(bytes.into(), Limits::default().max_allocation_bytes, None)
                .await
                .unwrap();
        let video = input.video(0).unwrap();
        for frame_index in [40_u64, 0, 23] {
            let frame = JsFuture::from(video.get(BigInt::from(frame_index).into(), None))
                .await
                .expect("VP9 decodes through WebCodecs or the software fallback");
            let get_u32 = |name: &str| -> u32 {
                Reflect::get(&frame, &JsValue::from_str(name))
                    .unwrap()
                    .as_f64()
                    .unwrap() as u32
            };
            assert_eq!((get_u32("width"), get_u32("height")), (256, 144));
            let pixels: Uint8Array = Reflect::get(&frame, &JsValue::from_str("pixels"))
                .unwrap()
                .unchecked_into();
            assert_eq!(pixels.length(), 256 * 144 * 4);
        }
    }

    /// Issue #363: scrubbing a timeline backwards has to restart its decode
    /// somewhere, and the only frames it can restart at are the track's
    /// random-access samples. The bundled sample codes its 768 frames as a
    /// single group of pictures, so frame zero is the one point it offers -
    /// which is what the `web_canvas` example walks forwards from when a drag
    /// moves back down the bar.
    #[wasm_bindgen_test(async)]
    async fn random_access_points_are_the_frames_a_decode_can_restart_at() {
        const SAMPLE: &[u8] = include_bytes!("../examples/media/BigBuckBunny.mp4");
        let bytes = Uint8Array::from(SAMPLE);
        let input =
            WasmMediaInput::open_inner(bytes.into(), Limits::default().max_allocation_bytes, None)
                .await
                .unwrap();
        let video = input.video(0).unwrap();

        let points: js_sys::Array = JsFuture::from(video.random_access_points(None))
            .await
            .unwrap()
            .unchecked_into();
        let indices: Vec<u64> = points
            .iter()
            .map(|point| parse_u64(&point, "random access point").unwrap())
            .collect();
        assert_eq!(indices, vec![0]);

        // Parsed once and cached, so a second call answers from the same list.
        let again: js_sys::Array = JsFuture::from(video.random_access_points(None))
            .await
            .unwrap()
            .unchecked_into();
        assert_eq!(again.length(), points.length());

        // An output stream indexes no samples to restart at.
        let options = WasmCreateOptions::new(None).unwrap();
        let output = WasmMediaOutput {
            bytes: Vec::new(),
            mime_type: options.mime_type,
            container: options.container,
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
            video_timescale: 0,
            video_frame_duration: 0,
            video_codec: Codec::Av1,
            browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
            browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(Codec::Aac))),
            cover_art: None,
            cover_source: CoverSource::default(),
        };
        let error = JsFuture::from(output.video(0).unwrap().random_access_points(None))
            .await
            .expect_err("an output stream has no random-access index");
        assert_error_code(&error, "INVALID_STATE");
    }

    /// Regression test for issue #38: the very last frame of a GOP has no
    /// further samples available to prime the decoder pipeline with, so
    /// `get()` must be able to fall back to draining the decoder instead of
    /// waiting forever for an output that more input would otherwise elicit.
    #[wasm_bindgen_test(async)]
    async fn video_get_decodes_the_last_frame_of_a_sparse_keyframe_gop() {
        const SAMPLE: &[u8] = include_bytes!("../examples/media/BigBuckBunny.mp4");
        const LAST_FRAME_INDEX: u64 = 767;
        // This exercises the `WebCodecs` session; the software fallback would
        // decode the same 1080p walk far too slowly for a browser test.
        if !WebVideoDecodeSession::decodes_through_webcodecs(SAMPLE, 0).await {
            return;
        }
        let bytes = Uint8Array::from(SAMPLE);
        let input =
            WasmMediaInput::open_inner(bytes.into(), Limits::default().max_allocation_bytes, None)
                .await
                .unwrap();
        let video = input.video(0).unwrap();

        match JsFuture::from(video.get(BigInt::from(LAST_FRAME_INDEX).into(), None)).await {
            Ok(frame) => {
                let get_u32 = |name: &str| -> u32 {
                    Reflect::get(&frame, &JsValue::from_str(name))
                        .unwrap()
                        .as_f64()
                        .unwrap() as u32
                };
                assert!(get_u32("width") > 0);
                assert!(get_u32("height") > 0);
            }
            Err(error) => assert_error_code(&error, "UNSUPPORTED"),
        }
    }

    /// Regression test for issue #108: sequential presentation-order playback
    /// of content with reordered decoder output used to reset the `WebCodecs`
    /// decoder and re-decode from the key frame on almost every call, which
    /// made ordinary playback quadratic in decode work and held the bundled
    /// 1080p sample under 1 fps.
    #[wasm_bindgen_test(async)]
    async fn video_get_decodes_consecutive_frames_without_restarting_the_gop() {
        const SAMPLE: &[u8] = include_bytes!("../examples/media/BigBuckBunny.mp4");
        const FRAME_COUNT: u64 = 48;
        // This exercises the `WebCodecs` session; the software fallback would
        // decode the same 1080p walk far too slowly for a browser test.
        if !WebVideoDecodeSession::decodes_through_webcodecs(SAMPLE, 0).await {
            return;
        }
        let bytes = Uint8Array::from(SAMPLE);
        let input =
            WasmMediaInput::open_inner(bytes.into(), Limits::default().max_allocation_bytes, None)
                .await
                .unwrap();
        let video = input.video(0).unwrap();

        for frame_index in 0..FRAME_COUNT {
            match JsFuture::from(video.get(BigInt::from(frame_index).into(), None)).await {
                Ok(frame) => {
                    let get_u32 = |name: &str| -> u32 {
                        Reflect::get(&frame, &JsValue::from_str(name))
                            .unwrap()
                            .as_f64()
                            .unwrap() as u32
                    };
                    let width = get_u32("width");
                    let height = get_u32("height");
                    assert!(width > 0);
                    assert!(height > 0);
                    let pixels: Uint8Array = Reflect::get(&frame, &JsValue::from_str("pixels"))
                        .unwrap()
                        .unchecked_into();
                    assert_eq!(pixels.length(), width * height * 4);
                }
                // This browser has no decoder for the track's codec at all.
                Err(error) => {
                    assert_error_code(&error, "UNSUPPORTED");
                    return;
                }
            }
        }
    }

    /// Regression test for issue #38: `BigBuckBunny.mp4` has a single key
    /// frame followed by ~768 delta frames, so decoding a frame this deep
    /// into the GOP used to exceed the old 12-frame WebCodecs batch cap and
    /// fail with `ResourceLimit` even though it was well within
    /// `Limits::max_decode_samples_per_seek`.
    #[wasm_bindgen_test(async)]
    async fn video_get_decodes_deep_into_a_sparse_keyframe_gop() {
        const SAMPLE: &[u8] = include_bytes!("../examples/media/BigBuckBunny.mp4");
        const DEEP_FRAME_INDEX: u64 = 400;
        // This exercises the `WebCodecs` session; the software fallback would
        // decode the same 1080p walk far too slowly for a browser test.
        if !WebVideoDecodeSession::decodes_through_webcodecs(SAMPLE, 0).await {
            return;
        }
        let bytes = Uint8Array::from(SAMPLE);
        let input =
            WasmMediaInput::open_inner(bytes.into(), Limits::default().max_allocation_bytes, None)
                .await
                .unwrap();
        let video = input.video(0).unwrap();

        match JsFuture::from(video.get(BigInt::from(DEEP_FRAME_INDEX).into(), None)).await {
            Ok(frame) => {
                let get_u32 = |name: &str| -> u32 {
                    Reflect::get(&frame, &JsValue::from_str(name))
                        .unwrap()
                        .as_f64()
                        .unwrap() as u32
                };
                let width = get_u32("width");
                let height = get_u32("height");
                assert!(width > 0);
                assert!(height > 0);
                let pixels = Reflect::get(&frame, &JsValue::from_str("pixels")).unwrap();
                let pixels: Uint8Array = pixels.unchecked_into();
                assert_eq!(pixels.length(), width * height * 4);
            }
            Err(error) => assert_error_code(&error, "UNSUPPORTED"),
        }
    }

    #[wasm_bindgen_test(async)]
    async fn readable_stream_input_releases_its_lock() {
        let chunks = Array::new();
        chunks.push(&Uint8Array::from(&[1_u8, 2][..]));
        chunks.push(&Uint8Array::from(&[3_u8, 4][..]));
        let stream = make_test_stream(chunks.as_ref());
        let input = WasmMediaInput::open_inner(
            stream.clone().into(),
            Limits::default().max_allocation_bytes,
            None,
        )
        .await
        .unwrap();
        assert_eq!(input.bytes().unwrap().to_vec(), vec![1, 2, 3, 4]);
        assert!(!stream.locked());
    }

    #[wasm_bindgen_test(async)]
    async fn aborting_a_pending_stream_returns_a_stable_error() {
        let controller = AbortController::new().unwrap();
        let signal = controller.signal();
        let stream = make_pending_stream();
        let pending = WasmMediaInput::open_inner(
            stream.into(),
            Limits::default().max_allocation_bytes,
            Some(signal),
        );
        controller.abort();
        let error = match pending.await {
            Err(error) => error,
            Ok(_) => panic!("an aborted stream must not open successfully"),
        };
        assert_error_code(&error, "CANCELLED");
    }

    #[wasm_bindgen_test(async)]
    async fn output_blob_owns_encoded_bytes() {
        let options = WasmCreateOptions::new(None).unwrap();
        let mut output = WasmMediaOutput {
            bytes: Vec::new(),
            mime_type: options.mime_type,
            container: options.container,
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
            video_timescale: 0,
            video_frame_duration: 0,
            video_codec: Codec::Av1,
            browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
            browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(Codec::Aac))),
            cover_art: None,
            cover_source: CoverSource::default(),
        };
        let chunk = Uint8Array::from(&[9_u8, 8, 7][..]);
        output.write_encoded_chunk(chunk.clone()).unwrap();
        chunk.set_index(0, 0);
        let blob = make_blob(&output.bytes, &output.mime_type).unwrap();
        let bytes = JsFuture::from(blob.array_buffer()).await.unwrap();
        assert_eq!(Uint8Array::new(&bytes).to_vec(), vec![9, 8, 7]);
    }

    #[wasm_bindgen_test(async)]
    async fn put_encodes_through_webcodecs_into_a_playable_mp4() {
        if !video_encode_support(None, None).unwrap() {
            // No WebCodecs AV1 encoder in this browser; nothing to verify here.
            return;
        }
        let options = WasmCreateOptions::new(None).unwrap();
        let output = WasmMediaOutput {
            bytes: Vec::new(),
            mime_type: options.mime_type,
            container: options.container,
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
            video_timescale: 30,
            video_frame_duration: 1,
            video_codec: Codec::Av1,
            browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
            browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(Codec::Aac))),
            cover_art: None,
            cover_source: CoverSource::default(),
        };
        let video = output.video(0).unwrap();
        let pixels = owned_u8_array(&[128_u8; 4 * 4 * 4]);
        let frame = WasmVideoFrame::rgba(4, 4, pixels).unwrap();
        for frame_index in 0..3_u64 {
            JsFuture::from(video.put(BigInt::from(frame_index).into(), &frame, None))
                .await
                .expect("encoding a frame through WebCodecs must succeed");
        }
        let mut output = output;
        let blob: Blob = JsFuture::from(output.finish())
            .await
            .unwrap()
            .unchecked_into();
        let array_buffer = JsFuture::from(blob.array_buffer()).await.unwrap();
        let bytes = Uint8Array::new(&array_buffer).to_vec();
        assert!(!bytes.is_empty());

        let source = MemorySource::new(bytes);
        let demuxer = crate::Mp4Demuxer::open(&source, crate::Mp4DemuxerOptions::default())
            .await
            .expect("the browser-encoded output must be a parseable MP4");
        assert_eq!(demuxer.tracks.len(), 1);
        assert_eq!(demuxer.tracks[0].kind, crate::TrackKind::Video);
        assert_eq!(demuxer.tracks[0].samples.len(), 3);
    }

    #[wasm_bindgen_test(async)]
    async fn set_cover_art_embeds_the_picture_in_the_finished_mp4() {
        if !video_encode_support(None, None).unwrap() {
            return;
        }
        let options = WasmCreateOptions::new(None).unwrap();
        let mut output = WasmMediaOutput {
            bytes: Vec::new(),
            mime_type: options.mime_type,
            container: options.container,
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
            video_timescale: 30,
            video_frame_duration: 1,
            video_codec: Codec::Av1,
            browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
            browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(Codec::Aac))),
            cover_art: None,
            cover_source: CoverSource::default(),
        };
        assert!(
            output
                .set_cover_art(
                    Some(owned_u8_array(b"GIF89a")),
                    Some("image/gif".to_owned())
                )
                .is_err()
        );
        let video = output.video(0).unwrap();
        let pixels = owned_u8_array(&[128_u8; 4 * 4 * 4]);
        let frame = WasmVideoFrame::rgba(4, 4, pixels).unwrap();
        for frame_index in 0..3_u64 {
            JsFuture::from(video.put(BigInt::from(frame_index).into(), &frame, None))
                .await
                .expect("encoding a frame through WebCodecs must succeed");
        }
        // Chosen after capture, as a recorder picking a middle frame would.
        let jpeg = [0xff_u8, 0xd8, 0xff, 0xe0, 0, 0x10, b'J', b'F', b'I', b'F'];
        output
            .set_cover_art(Some(owned_u8_array(&jpeg)), Some("image/jpeg".to_owned()))
            .unwrap();
        let blob: Blob = JsFuture::from(output.finish())
            .await
            .unwrap()
            .unchecked_into();
        let array_buffer = JsFuture::from(blob.array_buffer()).await.unwrap();
        let source = MemorySource::new(Uint8Array::new(&array_buffer).to_vec());
        let demuxer = crate::Mp4Demuxer::open(&source, crate::Mp4DemuxerOptions::default())
            .await
            .unwrap();
        assert_eq!(demuxer.tracks[0].samples.len(), 3);
        assert_eq!(
            demuxer.cover_art,
            Some(CoverArt {
                format: CoverArtFormat::Jpeg,
                data: jpeg.to_vec(),
            })
        );
    }

    #[wasm_bindgen_test(async)]
    async fn finish_generates_cover_art_from_the_cover_frame() {
        if !video_encode_support(None, None).unwrap() {
            return;
        }
        let mut options = WasmCreateOptions::new(None).unwrap();
        assert_eq!(options.cover_frame().as_f64(), None);
        assert_eq!(
            u64::try_from(BigInt::from(options.cover_frame())).unwrap(),
            4
        );
        let frames: Vec<WasmVideoFrame> = (0..3_u8)
            .map(|value| {
                WasmVideoFrame::rgba(4, 4, owned_u8_array(&[value * 60; 4 * 4 * 4])).unwrap()
            })
            .collect();
        for cover_frame in [JsValue::from(1), JsValue::NULL] {
            options.set_cover_frame(cover_frame.clone()).unwrap();
            let mut output = WasmMediaOutput {
                bytes: Vec::new(),
                mime_type: options.mime_type.clone(),
                container: options.container,
                max_output_bytes: options.max_output_bytes,
                state: Rc::new(Cell::new(false)),
                timeline: None,
                video_timescale: 30,
                video_frame_duration: 1,
                video_codec: Codec::Av1,
                browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
                browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(Codec::Aac))),
                cover_art: None,
                cover_source: options.cover_source,
            };
            let video = output.video(0).unwrap();
            for (index, frame) in frames.iter().enumerate() {
                JsFuture::from(video.put(BigInt::from(index as u64).into(), frame, None))
                    .await
                    .expect("encoding a frame through WebCodecs must succeed");
            }
            let blob: Blob = JsFuture::from(output.finish())
                .await
                .unwrap()
                .unchecked_into();
            let array_buffer = JsFuture::from(blob.array_buffer()).await.unwrap();
            let source = MemorySource::new(Uint8Array::new(&array_buffer).to_vec());
            let demuxer = crate::Mp4Demuxer::open(&source, crate::Mp4DemuxerOptions::default())
                .await
                .unwrap();
            let expected =
                (!cover_frame.is_null()).then(|| CoverArt::from_video_frame(&frames[1].0).unwrap());
            assert_eq!(demuxer.cover_art, expected);
        }
    }

    #[wasm_bindgen_test(async)]
    async fn put_encodes_hevc_through_webcodecs_into_a_playable_mp4() {
        if !video_encode_support(None, Some("hevc".to_owned())).unwrap() {
            return;
        }
        let options = WasmCreateOptions::new(None).unwrap();
        let output = WasmMediaOutput {
            bytes: Vec::new(),
            mime_type: options.mime_type,
            container: options.container,
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
            video_timescale: 30,
            video_frame_duration: 1,
            video_codec: Codec::Hevc,
            browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
            browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(Codec::Aac))),
            cover_art: None,
            cover_source: CoverSource::default(),
        };
        let video = output.video(0).unwrap();
        let frame = WasmVideoFrame::rgba(4, 4, owned_u8_array(&[128_u8; 4 * 4 * 4])).unwrap();
        for frame_index in 0..3_u64 {
            if let Err(error) =
                JsFuture::from(video.put(BigInt::from(frame_index).into(), &frame, None)).await
            {
                // The synchronous probe only establishes that this bridge
                // supports HEVC. The browser can still reject this concrete
                // configuration asynchronously when no encoder is available.
                assert_error_code(&error, "UNSUPPORTED");
                return;
            }
        }
        let mut output = output;
        let blob: Blob = JsFuture::from(output.finish())
            .await
            .unwrap()
            .unchecked_into();
        let bytes = Uint8Array::new(&JsFuture::from(blob.array_buffer()).await.unwrap()).to_vec();
        let demuxer = crate::Mp4Demuxer::open(
            &MemorySource::new(bytes),
            crate::Mp4DemuxerOptions::default(),
        )
        .await
        .expect("the browser-encoded output must be a parseable MP4");
        assert_eq!(demuxer.tracks.len(), 1);
        assert_eq!(demuxer.tracks[0].codec, Codec::Hevc);
        assert_eq!(demuxer.tracks[0].samples.len(), 3);
    }

    /// Issue #528: `videoCodec = "vp9"` encodes through WebCodecs into a
    /// demuxable `vp09` MP4 and a `V_VP9` WebM.
    #[wasm_bindgen_test(async)]
    async fn put_encodes_vp9_through_webcodecs_into_mp4_and_webm() {
        if !video_encode_support(None, Some("vp9".to_owned())).unwrap() {
            return;
        }
        for container in ["mp4", "webm"] {
            let mut options = WasmCreateOptions::new(Some(container.to_owned())).unwrap();
            options.set_video_codec("vp9".to_owned()).unwrap();
            assert_eq!(options.video_codec(), "vp9");
            let mut output = browser_output(&options);
            let video = output.video(0).unwrap();
            let frame =
                WasmVideoFrame::rgba(16, 16, owned_u8_array(&[128_u8; 16 * 16 * 4])).unwrap();
            for frame_index in 0..3_u64 {
                if let Err(error) =
                    JsFuture::from(video.put(BigInt::from(frame_index).into(), &frame, None)).await
                {
                    // As for HEVC, the browser can still refuse this concrete
                    // configuration asynchronously.
                    assert_error_code(&error, "UNSUPPORTED");
                    return;
                }
            }
            let blob: Blob = JsFuture::from(output.finish())
                .await
                .unwrap()
                .unchecked_into();
            let source = MemorySource::new(
                Uint8Array::new(&JsFuture::from(blob.array_buffer()).await.unwrap()).to_vec(),
            );
            let tracks = if container == "webm" {
                assert_eq!(blob.type_(), "video/webm");
                crate::WebmDemuxer::open(&source, crate::WebmDemuxerOptions::default())
                    .await
                    .expect("the browser-encoded output must be a parseable WebM")
                    .tracks
            } else {
                crate::Mp4Demuxer::open(&source, crate::Mp4DemuxerOptions::default())
                    .await
                    .expect("the browser-encoded output must be a parseable MP4")
                    .tracks
            };
            assert_eq!(tracks.len(), 1, "{container}");
            let track = &tracks[0];
            assert_eq!(track.codec, Codec::Vp9, "{container}");
            assert_eq!(track.samples.len(), 3, "{container}");
            assert!(track.samples[0].is_sync, "{container}");
            let derived = crate::derive_codec_string(Codec::Vp9, &track.decoder_config).unwrap();
            assert_eq!(derived.codec_string, "vp09.00.10.08", "{container}");
        }
    }

    /// Issue #474's acceptance criteria: a synchronized, playable audio+video
    /// MP4 produced entirely through `WasmMediaOutput`, mirroring
    /// `put_encodes_through_webcodecs_into_a_playable_mp4` above but with both
    /// tracks encoded.
    #[wasm_bindgen_test(async)]
    async fn put_encodes_audio_and_video_through_webcodecs_into_a_synchronized_mp4() {
        if !video_encode_support(None, None).unwrap() || !audio_encode_support(None, None).unwrap()
        {
            // No WebCodecs AV1/AAC encoder in this browser; nothing to verify here.
            return;
        }
        let options = WasmCreateOptions::new(None).unwrap();
        let output = WasmMediaOutput {
            bytes: Vec::new(),
            mime_type: options.mime_type,
            container: options.container,
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
            video_timescale: 30,
            video_frame_duration: 1,
            video_codec: Codec::Av1,
            browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
            browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(Codec::Aac))),
            cover_art: None,
            cover_source: CoverSource::default(),
        };

        let video = output.video(0).unwrap();
        let pixels = owned_u8_array(&[128_u8; 4 * 4 * 4]);
        let frame = WasmVideoFrame::rgba(4, 4, pixels).unwrap();
        for frame_index in 0..3_u64 {
            JsFuture::from(video.put(BigInt::from(frame_index).into(), &frame, None))
                .await
                .expect("encoding a video frame through WebCodecs must succeed");
        }

        let audio = output.audio(0).unwrap();
        const SAMPLE_RATE: u32 = 48_000;
        const FRAMES_PER_BUFFER: u64 = 1_024;
        let silence = vec![0.0_f32; FRAMES_PER_BUFFER as usize];
        for buffer_index in 0..4_u64 {
            let start = buffer_index * FRAMES_PER_BUFFER;
            let range =
                WasmSampleRange(CoreSampleRange::new(start, start + FRAMES_PER_BUFFER).unwrap());
            let buffer =
                WasmAudioBuffer::new(&range, SAMPLE_RATE, 1, Float32Array::from(&silence[..]))
                    .unwrap();
            JsFuture::from(audio.put(BigInt::from(buffer_index).into(), &buffer, None))
                .await
                .expect("encoding an audio buffer through WebCodecs must succeed");
        }

        // Whether this browser's `AudioEncoder` can actually produce AAC-LC
        // (rather than merely exposing the constructor `audio_encode_support`
        // checked above) is only knowable once its asynchronous error
        // callback has had a chance to fire, which `put()` never yields long
        // enough to observe (see `WebAudioEncodeSession::encode`'s doc
        // comment): a browser with no platform AAC encoder to back
        // `WebCodecs` (unlike macOS/Windows, Linux has none, mirroring the
        // native AudioToolbox/Media Foundation story) only reports that at
        // `finish()`'s `flush()`. Tolerate that specific, checked failure
        // mode; anything else is a real bug.
        let mut output = output;
        let finish_result = JsFuture::from(output.finish()).await;
        let blob: Blob = match finish_result {
            Ok(value) => value.unchecked_into(),
            Err(error) => {
                assert_error_code(&error, "CODEC");
                return;
            }
        };
        let array_buffer = JsFuture::from(blob.array_buffer()).await.unwrap();
        let bytes = Uint8Array::new(&array_buffer).to_vec();
        assert!(!bytes.is_empty());

        let source = MemorySource::new(bytes);
        let demuxer = crate::Mp4Demuxer::open(&source, crate::Mp4DemuxerOptions::default())
            .await
            .expect("the browser-encoded output must be a parseable MP4");
        assert_eq!(demuxer.tracks.len(), 2);
        assert_eq!(demuxer.tracks[0].kind, crate::TrackKind::Video);
        assert_eq!(demuxer.tracks[0].samples.len(), 3);
        assert_eq!(demuxer.tracks[1].kind, crate::TrackKind::Audio);
        assert!(
            !demuxer.tracks[1].samples.is_empty(),
            "the AAC encoder must have emitted at least one packet for 4096 input frames"
        );
    }

    /// Issue #477: `WasmMediaOutput` must encode more than one video track,
    /// not just track 0, into a single playable MP4.
    #[wasm_bindgen_test(async)]
    async fn put_encodes_multiple_video_tracks_into_a_playable_mp4() {
        if !video_encode_support(None, None).unwrap() {
            // No WebCodecs AV1 encoder in this browser; nothing to verify here.
            return;
        }
        let options = WasmCreateOptions::new(None).unwrap();
        let output = WasmMediaOutput {
            bytes: Vec::new(),
            mime_type: options.mime_type,
            container: options.container,
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
            video_timescale: 30,
            video_frame_duration: 1,
            video_codec: Codec::Av1,
            browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
            browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(Codec::Aac))),
            cover_art: None,
            cover_source: CoverSource::default(),
        };
        let track0 = output.video(0).unwrap();
        let track1 = output.video(1).unwrap();
        let pixels = owned_u8_array(&[128_u8; 4 * 4 * 4]);
        let frame = WasmVideoFrame::rgba(4, 4, pixels).unwrap();
        for frame_index in 0..3_u64 {
            JsFuture::from(track0.put(BigInt::from(frame_index).into(), &frame, None))
                .await
                .expect("encoding track 0 through WebCodecs must succeed");
        }
        for frame_index in 0..2_u64 {
            JsFuture::from(track1.put(BigInt::from(frame_index).into(), &frame, None))
                .await
                .expect("encoding track 1 through WebCodecs must succeed");
        }
        let mut output = output;
        let blob: Blob = JsFuture::from(output.finish())
            .await
            .unwrap()
            .unchecked_into();
        let array_buffer = JsFuture::from(blob.array_buffer()).await.unwrap();
        let bytes = Uint8Array::new(&array_buffer).to_vec();
        assert!(!bytes.is_empty());

        let source = MemorySource::new(bytes);
        let demuxer = crate::Mp4Demuxer::open(&source, crate::Mp4DemuxerOptions::default())
            .await
            .expect("the browser-encoded output must be a parseable MP4");
        assert_eq!(demuxer.tracks.len(), 2);
        assert_eq!(demuxer.tracks[0].kind, crate::TrackKind::Video);
        assert_eq!(demuxer.tracks[0].samples.len(), 3);
        assert_eq!(demuxer.tracks[1].kind, crate::TrackKind::Video);
        assert_eq!(demuxer.tracks[1].samples.len(), 2);
    }

    #[wasm_bindgen_test(async)]
    async fn put_encodes_yuv420p8_through_webcodecs_into_a_playable_mp4() {
        if !video_encode_support(None, None).unwrap() {
            // No WebCodecs AV1 encoder in this browser; nothing to verify here.
            return;
        }
        let options = WasmCreateOptions::new(None).unwrap();
        let output = WasmMediaOutput {
            bytes: Vec::new(),
            mime_type: options.mime_type,
            container: options.container,
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
            video_timescale: 30,
            video_frame_duration: 1,
            video_codec: Codec::Av1,
            browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
            browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(Codec::Aac))),
            cover_art: None,
            cover_source: CoverSource::default(),
        };
        let video = output.video(0).unwrap();
        // 4x4 luma plus two 2x2 chroma planes, concatenated: Y, then U, then V.
        let pixels = owned_u8_array(&[128_u8; 4 * 4 + 2 * 2 + 2 * 2]);
        let frame = WasmVideoFrame::yuv420p8(4, 4, pixels).unwrap();
        for frame_index in 0..3_u64 {
            JsFuture::from(video.put(BigInt::from(frame_index).into(), &frame, None))
                .await
                .expect("encoding a YUV 4:2:0 frame through WebCodecs must succeed");
        }
        let mut output = output;
        let blob: Blob = JsFuture::from(output.finish())
            .await
            .unwrap()
            .unchecked_into();
        let array_buffer = JsFuture::from(blob.array_buffer()).await.unwrap();
        let bytes = Uint8Array::new(&array_buffer).to_vec();
        assert!(!bytes.is_empty());

        let source = MemorySource::new(bytes);
        let demuxer = crate::Mp4Demuxer::open(&source, crate::Mp4DemuxerOptions::default())
            .await
            .expect("the browser-encoded output must be a parseable MP4");
        assert_eq!(demuxer.tracks.len(), 1);
        assert_eq!(demuxer.tracks[0].kind, crate::TrackKind::Video);
        assert_eq!(demuxer.tracks[0].samples.len(), 3);
    }

    #[wasm_bindgen_test]
    fn create_options_select_a_webm_container() {
        let options = WasmCreateOptions::new(Some("WebM".to_owned())).unwrap();
        assert_eq!(options.container(), "webm");
        assert_eq!(options.mime_type(), "video/webm");
        let mut options = options;
        options.set_video_codec("av1".to_owned()).unwrap();
        assert_error_code(
            &options.set_video_codec("hevc".to_owned()).unwrap_err(),
            "UNSUPPORTED",
        );
        options.set_video_codec("vp8".to_owned()).unwrap();
        assert_eq!(options.video_codec(), "vp8");
        let mut mp4 = WasmCreateOptions::new(None).unwrap();
        assert_error_code(
            &mp4.set_video_codec("vp8".to_owned()).unwrap_err(),
            "UNSUPPORTED",
        );
        assert_error_code(
            &WasmCreateOptions::new(Some("mkv".to_owned()))
                .err()
                .unwrap(),
            "UNSUPPORTED",
        );
        let names: Vec<_> = supported_containers()
            .iter()
            .map(|name| name.as_string().unwrap())
            .collect();
        assert_eq!(names, ["mp4", "webm"]);
    }

    fn browser_output(options: &WasmCreateOptions) -> WasmMediaOutput {
        WasmMediaOutput {
            bytes: Vec::new(),
            mime_type: options.mime_type.clone(),
            container: options.container,
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
            video_timescale: 30,
            video_frame_duration: 1,
            video_codec: options.video_codec,
            browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
            browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(options.audio_codec))),
            cover_art: None,
            cover_source: options.cover_source,
        }
    }

    #[wasm_bindgen_test]
    fn webm_output_takes_opus_or_vorbis_audio_but_refuses_aac_and_cover_art() {
        let mut options = WasmCreateOptions::new(Some("webm".to_owned())).unwrap();
        assert_eq!(options.audio_codec(), "opus");
        assert_error_code(
            &options.set_audio_codec("aac".into()).unwrap_err(),
            "UNSUPPORTED",
        );
        options.set_audio_codec("vorbis".into()).unwrap();
        let mut output = browser_output(&options);
        assert!(output.audio(0).is_ok());
        let picture = Uint8Array::from(&[0xFF_u8, 0xD8, 0xFF][..]);
        assert_error_code(
            &output
                .set_cover_art(Some(picture), Some("image/jpeg".to_owned()))
                .unwrap_err(),
            "UNSUPPORTED",
        );
        output.set_cover_art(None, None).unwrap();
    }

    /// The bundled AV1 sample's video track remuxed into WebM by
    /// [`WebmMuxer`], as a recorder writing WebM would produce it.
    async fn bundled_av1_as_webm() -> Vec<u8> {
        const SAMPLE: &[u8] = include_bytes!("../examples/media/BigBuckBunny.av1.mp4");
        let source = MemorySource::new(SAMPLE.to_vec());
        let movie = crate::Mp4Demuxer::open(&source, crate::Mp4DemuxerOptions::default())
            .await
            .unwrap();
        let track = movie
            .tracks
            .into_iter()
            .find(|track| track.kind == crate::TrackKind::Video)
            .unwrap();
        let config = Mp4TrackConfig {
            encoder: EncoderConfig {
                codec: track.codec,
                timescale: track.timescale,
                decoder_config: track.decoder_config.clone(),
            },
            format: Mp4TrackFormat::Video(track.dimensions.unwrap()),
        };
        let mut muxer = WebmMuxer::new(MemorySink::new(), vec![config], 1_000_000)
            .await
            .unwrap();
        for (index, sample) in track.samples.iter().enumerate() {
            let mut data = vec![0; sample.size as usize];
            track
                .read_sample_into(&source, index, &mut data)
                .await
                .unwrap();
            muxer
                .write_sample(
                    0,
                    EncodedSample {
                        data,
                        dts: sample.dts as i64,
                        pts: sample.pts,
                        duration: sample.duration,
                        is_sync: sample.is_sync,
                        dependency: sample.dependency,
                    },
                )
                .await
                .unwrap();
        }
        muxer.finish().await.unwrap().into_inner()
    }

    async fn frame_pixels(video: &WasmVideoStream, index: u64) -> Vec<u8> {
        let frame = JsFuture::from(video.get(BigInt::from(index).into(), None))
            .await
            .expect("an AV1 frame decodes through WebCodecs or the software fallback");
        let pixels = Reflect::get(&frame, &JsValue::from_str("pixels")).unwrap();
        pixels.unchecked_into::<Uint8Array>().to_vec()
    }

    /// Issue #526: `MediaInput.open` tells WebM from MP4 by its EBML
    /// signature, and a WebM input answers `get(n)` with exactly the picture
    /// the same AV1 track gives from MP4.
    #[wasm_bindgen_test(async)]
    async fn webm_input_is_detected_and_decodes_the_mp4_frames() {
        const SAMPLE: &[u8] = include_bytes!("../examples/media/BigBuckBunny.av1.mp4");
        let webm = bundled_av1_as_webm().await;
        // A Blob typed as MP4 is still detected by its bytes.
        let blob = make_blob(&webm, "video/mp4").unwrap();
        let limit = Limits::default().max_allocation_bytes;
        let from_webm = WasmMediaInput::open_inner(blob.into(), limit, None)
            .await
            .unwrap();
        let from_mp4 = WasmMediaInput::open_inner(Uint8Array::from(SAMPLE).into(), limit, None)
            .await
            .unwrap();
        assert_eq!(from_webm.container().unwrap().as_deref(), Some("webm"));
        assert_eq!(from_mp4.container().unwrap().as_deref(), Some("mp4"));
        let unknown =
            WasmMediaInput::open_inner(Uint8Array::from(&[1_u8, 2, 3][..]).into(), limit, None)
                .await
                .unwrap();
        assert_eq!(unknown.container().unwrap(), None);

        let webm_video = from_webm.video(0).unwrap();
        let mp4_video = from_mp4.video(0).unwrap();
        let points = JsFuture::from(webm_video.random_access_points(None))
            .await
            .unwrap();
        assert_eq!(Array::from(&points).length(), {
            let points = JsFuture::from(mp4_video.random_access_points(None))
                .await
                .unwrap();
            Array::from(&points).length()
        });
        for index in [2_u64, 0] {
            assert_eq!(
                frame_pixels(&webm_video, index).await,
                frame_pixels(&mp4_video, index).await,
                "frame {index} differs between containers"
            );
        }
    }

    /// Issue #526: the WebM the muxer writes plays in Chrome, which reads its
    /// Cues to seek.
    #[wasm_bindgen_test(async)]
    async fn written_webm_plays_and_seeks_in_the_browser() {
        let webm = bundled_av1_as_webm().await;
        let blob = make_blob(&webm, "video/webm").unwrap();
        let probe = JsFuture::from(probe_test_video(&blob, 2.0))
            .await
            .expect("the browser must load and seek the written WebM");
        let probe = Array::from(&probe);
        let value = |index: u32| probe.get(index).as_f64().unwrap();
        let (duration, seekable_end, current_time, width) =
            (value(0), value(1), value(2), value(3));
        assert!(
            duration.is_finite() && duration > 2.0,
            "duration {duration}"
        );
        assert!(
            (seekable_end - duration).abs() < 0.1,
            "seekable to {seekable_end}"
        );
        assert!((current_time - 2.0).abs() < 0.1, "seeked to {current_time}");
        assert!(width > 0.0);
    }

    #[wasm_bindgen_test(async)]
    async fn put_encodes_through_webcodecs_into_a_webm_blob() {
        if !video_encode_support(None, None).unwrap() {
            // No WebCodecs AV1 encoder in this browser; nothing to verify here.
            return;
        }
        let options = WasmCreateOptions::new(Some("webm".to_owned())).unwrap();
        let mut output = browser_output(&options);
        let video = output.video(0).unwrap();
        let pixels = owned_u8_array(&[128_u8; 4 * 4 * 4]);
        let frame = WasmVideoFrame::rgba(4, 4, pixels).unwrap();
        for frame_index in 0..3_u64 {
            JsFuture::from(video.put(BigInt::from(frame_index).into(), &frame, None))
                .await
                .expect("encoding a frame through WebCodecs must succeed");
        }
        let blob: Blob = JsFuture::from(output.finish())
            .await
            .unwrap()
            .unchecked_into();
        assert_eq!(blob.type_(), "video/webm");
        let array_buffer = JsFuture::from(blob.array_buffer()).await.unwrap();
        let source = MemorySource::new(Uint8Array::new(&array_buffer).to_vec());
        let demuxer = crate::WebmDemuxer::open(&source, crate::WebmDemuxerOptions::default())
            .await
            .expect("the browser-encoded output must be a parseable WebM");
        assert_eq!(demuxer.doc_type, "webm");
        assert_eq!(demuxer.tracks.len(), 1);
        assert_eq!(demuxer.tracks[0].codec, Codec::Av1);
        assert_eq!(demuxer.tracks[0].samples.len(), 3);
        assert!(!demuxer.cues.is_empty());
    }

    /// Issue #530: `videoCodec = "vp8"` encodes through `WebCodecs` into a
    /// WebM that zvidlib's own VP8 decoder reads and the browser plays and
    /// seeks.
    #[wasm_bindgen_test(async)]
    async fn put_encodes_vp8_through_webcodecs_into_a_seekable_webm() {
        assert!(!video_encode_support(Some("require".into()), Some("vp8".into())).unwrap());
        if !video_encode_support(None, Some("vp8".into())).unwrap() {
            return;
        }
        let mut options = WasmCreateOptions::new(Some("webm".to_owned())).unwrap();
        options.set_video_codec("vp8".to_owned()).unwrap();
        let mut output = browser_output(&options);
        let video = output.video(0).unwrap();
        // Two and a half seconds of a grey frame at 30 frames a second.
        const FRAMES: u64 = 75;
        let pixels = owned_u8_array(&[128_u8; 32 * 32 * 4]);
        let frame = WasmVideoFrame::rgba(32, 32, pixels).unwrap();
        for frame_index in 0..FRAMES {
            JsFuture::from(video.put(BigInt::from(frame_index).into(), &frame, None))
                .await
                .expect("encoding a VP8 frame through WebCodecs must succeed");
        }
        let blob: Blob = JsFuture::from(output.finish())
            .await
            .unwrap()
            .unchecked_into();
        assert_eq!(blob.type_(), "video/webm");
        let array_buffer = JsFuture::from(blob.array_buffer()).await.unwrap();
        let bytes = Uint8Array::new(&array_buffer).to_vec();
        let source = MemorySource::new(bytes.clone());
        let demuxer = crate::WebmDemuxer::open(&source, crate::WebmDemuxerOptions::default())
            .await
            .expect("the browser-encoded output must be a parseable WebM");
        let track = &demuxer.tracks[0];
        assert_eq!(track.codec, Codec::Vp8);
        assert_eq!(track.samples.len(), FRAMES as usize);
        assert!(!demuxer.cues.is_empty());

        let limits = Limits::default();
        let samples = track
            .to_encoded_video_samples(&source, &limits)
            .await
            .unwrap();
        use crate::VideoDecoderFactory;
        let factory = crate::native_vp8_video_decoder_factory();
        let mut decoder = factory
            .create(
                &crate::VideoDecoderConfig {
                    codec: Codec::Vp8,
                    profile: CodecProfile::Vp8,
                    coded_dimensions: track.dimensions.unwrap(),
                    output_format: crate::PixelFormat::Rgba8,
                    color_range: crate::ColorRange::Limited,
                    hardware: HardwarePreference::Avoid,
                    configuration: Vec::new(),
                },
                &limits,
            )
            .unwrap();
        let cancellation = crate::CancellationToken::new();
        for sample in &samples {
            let frames = decoder.submit(sample, &cancellation).unwrap();
            assert_eq!(frames.len(), 1);
            let pixels = &frames[0].frame.planes[0].data;
            assert!(pixels.iter().step_by(4).all(|&red| red.abs_diff(128) <= 6));
        }

        let blob = make_blob(&bytes, "video/webm").unwrap();
        let probe = JsFuture::from(probe_test_video(&blob, 2.0))
            .await
            .expect("the browser must load and seek the VP8 WebM");
        let probe = Array::from(&probe);
        assert!((probe.get(2).as_f64().unwrap() - 2.0).abs() < 0.1);
        assert_eq!(probe.get(3).as_f64().unwrap(), 32.0);
    }

    #[wasm_bindgen_test]
    fn media_typed_arrays_are_snapshots() {
        let source = Uint8Array::from(&[1_u8, 2, 3, 4][..]);
        let frame = WasmVideoFrame::rgba(1, 1, source.clone()).unwrap();
        source.set_index(0, 9);
        let first = frame.pixels();
        first.set_index(1, 8);
        assert_eq!(frame.pixels().to_vec(), vec![1, 2, 3, 4]);

        let range = WasmSampleRange(CoreSampleRange::new(0, 2).unwrap());
        let samples = Float32Array::from(&[0.25_f32, 0.5][..]);
        let audio = WasmAudioBuffer::new(&range, 48_000, 1, samples.clone()).unwrap();
        samples.set_index(0, 1.0);
        assert_eq!(audio.samples().to_vec(), vec![0.25, 0.5]);
    }

    #[wasm_bindgen_test]
    fn playback_options_preserve_but_do_not_mutate_browser_objects() {
        let object = Object::new();
        Reflect::set(&object, &"closed".into(), &false.into()).unwrap();
        let mut options = WasmPlaybackOptions::new();
        options.set_audio_context(Some(object.clone().into()));
        let mut playback = WasmPlayback {
            state: Rc::new(Cell::new(false)),
            _options: options,
        };
        playback.close();
        assert_eq!(
            Reflect::get(&object, &"closed".into()).unwrap().as_bool(),
            Some(false)
        );
    }

    #[wasm_bindgen_test]
    fn audio_encode_support_reports_each_codec() {
        let has_encoder =
            Reflect::has(&js_sys::global(), &JsValue::from_str("AudioEncoder")).unwrap_or(false);
        assert_eq!(audio_encode_support(None, None).unwrap(), has_encoder);
        assert_eq!(
            audio_encode_support(None, Some("aac".into())).unwrap(),
            has_encoder
        );
        // zvidlib encodes Opus and Vorbis itself where the browser cannot.
        assert!(audio_encode_support(None, Some("opus".into())).unwrap());
        assert!(audio_encode_support(None, Some("vorbis".into())).unwrap());
        assert!(!audio_encode_support(Some("require".into()), Some("opus".into())).unwrap());
        assert_error_code(
            &audio_encode_support(None, Some("flac".into())).unwrap_err(),
            "UNSUPPORTED",
        );
    }

    #[wasm_bindgen_test]
    fn create_options_select_the_audio_codec() {
        let mut options = WasmCreateOptions::new(None).unwrap();
        assert_eq!(options.audio_codec(), "aac");
        options.set_audio_codec("opus".into()).unwrap();
        assert_eq!(options.audio_codec(), "opus");
        assert_error_code(
            &options.set_audio_codec("vorbis".into()).unwrap_err(),
            "UNSUPPORTED",
        );
        assert_eq!(options.audio_codec(), "opus");
    }

    /// Issue #524: an Opus track written through the browser's `WebCodecs`
    /// encoder reads back through zvidlib with exactly the samples that were
    /// put, its pre-skip and end padding trimmed by the edit list.
    #[wasm_bindgen_test(async)]
    async fn put_encodes_opus_through_webcodecs_into_an_exact_mp4() {
        if !audio_encode_support(None, Some("opus".into())).unwrap() {
            return;
        }
        let options = WasmCreateOptions::new(None).unwrap();
        let mut output = WasmMediaOutput {
            bytes: Vec::new(),
            mime_type: options.mime_type,
            container: options.container,
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
            video_timescale: 30,
            video_frame_duration: 1,
            video_codec: Codec::Av1,
            browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
            browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(Codec::Opus))),
            cover_art: None,
            cover_source: CoverSource::default(),
        };
        const FRAMES_PER_BUFFER: u64 = 4_800;
        const BUFFERS: u64 = 5;
        let input: Vec<f32> = (0..FRAMES_PER_BUFFER * BUFFERS)
            .map(|i| 0.4 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
            .collect();
        let audio = output.audio(0).unwrap();
        for buffer_index in 0..BUFFERS {
            let start = buffer_index * FRAMES_PER_BUFFER;
            let range =
                WasmSampleRange(CoreSampleRange::new(start, start + FRAMES_PER_BUFFER).unwrap());
            let samples = &input[start as usize..(start + FRAMES_PER_BUFFER) as usize];
            let buffer =
                WasmAudioBuffer::new(&range, 48_000, 1, Float32Array::from(samples)).unwrap();
            JsFuture::from(audio.put(BigInt::from(buffer_index).into(), &buffer, None))
                .await
                .expect("encoding an audio buffer through WebCodecs must succeed");
        }
        let blob: Blob = JsFuture::from(output.finish())
            .await
            .expect("the Opus output must finish")
            .unchecked_into();
        let bytes = Uint8Array::new(&JsFuture::from(blob.array_buffer()).await.unwrap()).to_vec();

        let source = MemorySource::new(bytes);
        let movie = crate::Mp4Demuxer::open(&source, crate::Mp4DemuxerOptions::default())
            .await
            .unwrap();
        let track = &movie.tracks[0];
        assert_eq!(track.codec, Codec::Opus);
        let head = track.opus_config().unwrap();
        let packets = track
            .to_encoded_audio_samples(&source, &Limits::default())
            .await
            .unwrap();
        let mut reader = crate::AudioSampleReader::new(
            crate::NativeOpusDecoder::new(&head, Limits::default()).unwrap(),
            packets.clone(),
            48_000,
            1,
            track.audio_timing(movie.movie_timescale).unwrap(),
            crate::opus_preroll_packets(&packets),
            Limits::default(),
        )
        .unwrap();
        assert_eq!(reader.presentation_length(), FRAMES_PER_BUFFER * BUFFERS);
        let decoded = reader
            .get_range(
                CoreSampleRange::new(0, FRAMES_PER_BUFFER * BUFFERS).unwrap(),
                &CancellationToken::new(),
            )
            .unwrap()
            .samples;
        let noise: f32 = decoded
            .iter()
            .zip(&input)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let signal: f32 = input.iter().map(|v| v * v).sum();
        let snr = 10.0 * (signal / noise).log10();
        assert!(
            snr > 10.0,
            "the browser's Opus decodes {snr:.1} dB from its input: misaligned pre-skip?"
        );
    }

    /// Issue #524: Chrome's own media stack decodes an Opus MP4 zvidlib's
    /// native encoder wrote, the way a page playing the file would.
    ///
    /// Chrome trims the pre-skip twice: once through the edit list, as the
    /// Opus-in-ISOBMFF specification has players do, and again from the
    /// `dOps` box's `PreSkip`, which the specification calls informative. It
    /// does the same to the MP4s FFmpeg writes, which carry the same two
    /// fields, so the decode is accepted at either alignment.
    #[wasm_bindgen_test(async)]
    async fn the_browser_decodes_a_native_opus_mp4() {
        const PRE_SKIP: usize = 312;
        let (bytes, input) = crate::web_audio_decoder::tests::opus_mp4().await;
        let context =
            web_sys::OfflineAudioContext::new_with_number_of_channels_and_length_and_sample_rate(
                2, 48_000, 48_000.0,
            )
            .unwrap();
        let array = Uint8Array::from(bytes.as_slice());
        let decoded: web_sys::AudioBuffer = JsFuture::from(
            context
                .decode_audio_data(&array.buffer())
                .expect("decodeAudioData must accept the call"),
        )
        .await
        .expect("the browser must decode zvidlib's Opus MP4")
        .unchecked_into();
        assert_eq!(decoded.number_of_channels(), 2);
        let length = decoded.length() as usize;
        assert!(
            length == 48_000 || length == 48_000 - PRE_SKIP,
            "the browser decoded {length} frames of 48000"
        );
        let left = decoded.get_channel_data(0).unwrap();
        let reference: Vec<f32> = input.iter().step_by(2).copied().collect();
        let skipped = 48_000 - length;
        let compared = length - 2_000;
        let noise: f32 = left[..compared]
            .iter()
            .zip(&reference[skipped..skipped + compared])
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let signal: f32 = reference[skipped..skipped + compared]
            .iter()
            .map(|v| v * v)
            .sum();
        assert!(
            10.0 * (signal / noise).log10() > 10.0,
            "the browser's decode is not aligned with the input"
        );
    }

    #[wasm_bindgen_test(async)]
    async fn an_input_opus_track_reads_exact_ranges() {
        let (bytes, input) = crate::web_audio_decoder::tests::opus_mp4().await;
        let media = WasmMediaInput {
            bytes: Rc::new(bytes),
            state: Rc::new(Cell::new(false)),
            container: Some(Container::Mp4),
        };
        let audio = media.audio(0).unwrap();
        let count = JsFuture::from(audio.sample_count(None)).await.unwrap();
        assert_eq!(parse_u64(&count, "sampleCount").unwrap(), 48_000);
        let config = JsFuture::from(audio.decoder_config(None)).await.unwrap();
        assert_eq!(
            Reflect::get(&config, &"codec".into())
                .unwrap()
                .as_string()
                .as_deref(),
            Some("opus")
        );
        assert_eq!(
            Reflect::get(&config, &"numberOfChannels".into())
                .unwrap()
                .as_f64(),
            Some(2.0)
        );
        let buffer = JsFuture::from(audio.get_range(
            BigInt::from(1_000_u64).into(),
            BigInt::from(2_000_u64).into(),
            None,
        ))
        .await
        .unwrap();
        // The resolved value is the exported `AudioBuffer` class, read here
        // through its JavaScript getters as a page would.
        assert_eq!(
            Reflect::get(&buffer, &"channels".into()).unwrap().as_f64(),
            Some(2.0)
        );
        let samples: Float32Array = Reflect::get(&buffer, &"samples".into())
            .unwrap()
            .unchecked_into();
        let samples = samples.to_vec();
        assert_eq!(samples.len(), 2_000);
        let expected = &input[2_000..4_000];
        let noise: f32 = samples
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let signal: f32 = expected.iter().map(|v| v * v).sum();
        assert!(10.0 * (signal / noise).log10() > 15.0);
        assert_error_code(
            &JsFuture::from(audio.get_range(
                BigInt::from(0_u64).into(),
                BigInt::from(48_001_u64).into(),
                None,
            ))
            .await
            .unwrap_err(),
            "INVALID_INPUT",
        );
    }

    /// Encodes `seconds` of a 48 kHz stereo tone through a browser output's
    /// track 0 and returns the finished file and the PCM that was put.
    async fn browser_audio_output(options: WasmCreateOptions) -> (Vec<u8>, Vec<f32>) {
        let mut output = WasmMediaOutput {
            bytes: Vec::new(),
            mime_type: options.mime_type.clone(),
            container: options.container,
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
            video_timescale: 30,
            video_frame_duration: 1,
            video_codec: Codec::Av1,
            browser_video_tracks: Rc::new(RefCell::new(BTreeMap::new())),
            browser_audio: Rc::new(RefCell::new(BrowserAudioTrack::new(options.audio_codec))),
            cover_art: None,
            cover_source: options.cover_source,
        };
        const FRAMES_PER_BUFFER: u64 = 4_800;
        const BUFFERS: u64 = 10;
        let input: Vec<f32> = (0..FRAMES_PER_BUFFER * BUFFERS)
            .flat_map(|i| {
                let t = i as f32 / 48_000.0;
                [
                    0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin(),
                    0.3 * (2.0 * std::f32::consts::PI * 550.0 * t).sin(),
                ]
            })
            .collect();
        let audio = output.audio(0).unwrap();
        for buffer_index in 0..BUFFERS {
            let start = buffer_index * FRAMES_PER_BUFFER;
            let range =
                WasmSampleRange(CoreSampleRange::new(start, start + FRAMES_PER_BUFFER).unwrap());
            let samples = &input[2 * start as usize..2 * (start + FRAMES_PER_BUFFER) as usize];
            let buffer =
                WasmAudioBuffer::new(&range, 48_000, 2, Float32Array::from(samples)).unwrap();
            JsFuture::from(audio.put(BigInt::from(buffer_index).into(), &buffer, None))
                .await
                .expect("encoding an audio buffer must succeed");
        }
        let blob: Blob = JsFuture::from(output.finish())
            .await
            .expect("the output must finish")
            .unchecked_into();
        let bytes = Uint8Array::new(&JsFuture::from(blob.array_buffer()).await.unwrap()).to_vec();
        (bytes, input)
    }

    /// Issue #524: a WebM output carries Opus or Vorbis audio, which reads
    /// back through an input stream with exactly the samples that were put,
    /// and which Chrome's own media stack decodes.
    #[wasm_bindgen_test(async)]
    async fn a_webm_output_carries_opus_and_vorbis_audio() {
        for codec in ["opus", "vorbis"] {
            let mut options = WasmCreateOptions::new(Some("webm".to_owned())).unwrap();
            options.set_audio_codec(codec.into()).unwrap();
            let (bytes, input) = browser_audio_output(options).await;

            let media = WasmMediaInput {
                bytes: Rc::new(bytes.clone()),
                state: Rc::new(Cell::new(false)),
                container: Some(Container::WebM),
            };
            let audio = media.audio(0).unwrap();
            let config = JsFuture::from(audio.decoder_config(None)).await.unwrap();
            assert_eq!(
                Reflect::get(&config, &"codec".into())
                    .unwrap()
                    .as_string()
                    .as_deref(),
                Some(codec)
            );
            let count = JsFuture::from(audio.sample_count(None)).await.unwrap();
            assert_eq!(parse_u64(&count, "sampleCount").unwrap(), 48_000, "{codec}");
            let buffer = JsFuture::from(audio.get_range(
                BigInt::from(0_u64).into(),
                BigInt::from(48_000_u64).into(),
                None,
            ))
            .await
            .unwrap();
            let samples: Float32Array = Reflect::get(&buffer, &"samples".into())
                .unwrap()
                .unchecked_into();
            let samples = samples.to_vec();
            let noise: f32 = samples
                .iter()
                .zip(&input)
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            let signal: f32 = input.iter().map(|v| v * v).sum();
            let snr = 10.0 * (signal / noise).log10();
            assert!(snr > 15.0, "{codec} reads back {snr:.1} dB from its input");

            let context =
                web_sys::OfflineAudioContext::new_with_number_of_channels_and_length_and_sample_rate(
                    2, 48_000, 48_000.0,
                )
                .unwrap();
            let array = Uint8Array::from(bytes.as_slice());
            let decoded: web_sys::AudioBuffer = JsFuture::from(
                context
                    .decode_audio_data(&array.buffer())
                    .expect("decodeAudioData must accept the call"),
            )
            .await
            .unwrap_or_else(|_| panic!("the browser must decode zvidlib's {codec} WebM"))
            .unchecked_into();
            assert_eq!(decoded.number_of_channels(), 2);
            assert!(
                decoded.length().abs_diff(48_000) <= 960,
                "the browser decoded {} frames of {codec} where 48000 were put",
                decoded.length()
            );
        }
    }

    /// The browser's `WebCodecs` Vorbis decoder, where it has one, and
    /// zvidlib's own agree on a WebM Vorbis track; the software fallback is
    /// what a browser without one reads with.
    #[wasm_bindgen_test(async)]
    async fn software_and_webcodecs_reads_agree_on_a_vorbis_webm() {
        let mut options = WasmCreateOptions::new(Some("webm".to_owned())).unwrap();
        options.set_audio_codec("vorbis".into()).unwrap();
        let (bytes, _) = browser_audio_output(options).await;
        let cancellation = CancellationToken::new();
        let mut session = WebAudioDecodeSession::open(&bytes, 0, &Limits::default())
            .await
            .unwrap();
        let mut software = WebAudioDecodeSession::open(&bytes, 0, &Limits::default())
            .await
            .unwrap();
        for (start, end) in [(10_000, 12_000), (0, 500), (47_000, 48_000)] {
            let range = CoreSampleRange::new(start, end).unwrap();
            let got = session.get_range(range, &cancellation).await.unwrap();
            software.reset_to_software();
            let expected = software.get_range(range, &cancellation).await.unwrap();
            let largest = got
                .samples
                .iter()
                .zip(&expected.samples)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(largest < 1e-3, "[{start}, {end}) differs by {largest}");
        }
    }
}
