//! Browser-facing wrappers for the portable core.
//!
//! Values crossing this boundary are copied into owned Rust storage. Returned
//! typed arrays are snapshots rather than views into growable WebAssembly
//! memory, and browser-owned objects are retained only as JavaScript handles.

use crate::io::MemorySource;
use crate::web_decoder::{
    WebVideoDecodeSession, video_frame_durations_ms, video_random_access_points,
};
use crate::web_previews::WebPreviewIndex;
use crate::{
    AudioBuffer as CoreAudioBuffer, CancellationToken, ColorRange, ErrorKind,
    FrameIndex as CoreFrameIndex, FrameRate, Limits, PixelFormat, Plane,
    PreviewOptions as CorePreviewOptions, PreviewStore, Rational as CoreRational,
    SEEK_LATENCY_BUDGET, SampleRange as CoreSampleRange, Timeline, VideoDimensions,
    VideoFrame as CoreVideoFrame,
};
use js_sys::{Array, BigInt, Float32Array, Promise, Reflect, Uint8Array};
use std::cell::{Cell, RefCell};
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

/// An owned RGBA CPU frame. Pixel arrays are copied in both directions.
#[wasm_bindgen(js_name = VideoFrame)]
pub struct WasmVideoFrame(CoreVideoFrame);

#[wasm_bindgen(js_class = VideoFrame)]
impl WasmVideoFrame {
    #[wasm_bindgen(js_name = rgba)]
    pub fn rgba(width: u32, height: u32, pixels: Uint8Array) -> Result<WasmVideoFrame, JsValue> {
        let limits = Limits::default();
        let dimensions = VideoDimensions::new(width, height, &limits)
            .map_err(|error| js_error(error.kind(), error.message()))?;
        let stride = usize::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(4))
            .ok_or_else(|| js_error(ErrorKind::ResourceLimit, "video stride overflow"))?;
        let frame = CoreVideoFrame::new(
            dimensions,
            PixelFormat::Rgba8,
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
    container: String,
    mime_type: String,
    max_output_bytes: u64,
    frame_rate: Option<FrameRate>,
    audio_sample_rate: Option<u32>,
}

#[wasm_bindgen(js_class = CreateOptions)]
impl WasmCreateOptions {
    #[wasm_bindgen(constructor)]
    pub fn new(container: Option<String>) -> Result<WasmCreateOptions, JsValue> {
        let container = container.unwrap_or_else(|| "mp4".to_owned());
        if !container.eq_ignore_ascii_case("mp4") {
            return Err(js_error(
                ErrorKind::Unsupported,
                format!("unsupported output container: {container}"),
            ));
        }
        Ok(Self {
            container: "mp4".to_owned(),
            mime_type: "video/mp4".to_owned(),
            max_output_bytes: Limits::default().max_allocation_bytes,
            frame_rate: None,
            audio_sample_rate: None,
        })
    }

    #[wasm_bindgen(getter)]
    pub fn container(&self) -> String {
        self.container.clone()
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
        _frame: &WasmVideoFrame,
        signal: Option<AbortSignal>,
    ) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            parse_u64(&frame_index, "frame index")?;
            if direction != StreamDirection::Output {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "put is only valid on an output video stream",
                ));
            }
            Err(js_error(
                ErrorKind::Unsupported,
                "no browser video encoder backend is registered",
            ))
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
    let demuxer = crate::Mp4Demuxer::open(&source, crate::Mp4DemuxerOptions::default())
        .await
        .map_err(|error| js_error(error.kind(), error.message()))?;
    demuxer
        .tracks
        .into_iter()
        .filter(|track| track.kind == crate::TrackKind::Audio)
        .nth(index as usize)
        .ok_or_else(|| js_error(ErrorKind::InvalidInput, "no such audio track"))
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
                .ok_or_else(|| js_error(ErrorKind::InvalidInput, "AAC packet is not indexed"))
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
                "no browser audio decoder backend is registered",
            ))
        })
    }

    pub fn put(
        &self,
        frame_index: JsValue,
        _buffer: &WasmAudioBuffer,
        signal: Option<AbortSignal>,
    ) -> Promise {
        let state = Rc::clone(&self.state);
        let direction = self.direction;
        future_to_promise(async move {
            ensure_open(&state)?;
            check_signal(signal.as_ref())?;
            parse_u64(&frame_index, "frame index")?;
            if direction != StreamDirection::Output {
                return Err(js_error(
                    ErrorKind::InvalidState,
                    "put is only valid on an output audio stream",
                ));
            }
            Err(js_error(
                ErrorKind::Unsupported,
                "no browser audio encoder backend is registered",
            ))
        })
    }
}

/// A browser input whose source bytes are owned by WebAssembly after opening.
#[wasm_bindgen(js_name = MediaInput)]
pub struct WasmMediaInput {
    bytes: Rc<Vec<u8>>,
    state: Rc<Cell<bool>>,
}

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
        Ok(Self {
            bytes: Rc::new(bytes.to_vec()),
            state: Rc::new(Cell::new(false)),
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

/// A browser output adapter that returns finalized bytes as an owned `Blob`.
#[wasm_bindgen(js_name = MediaOutput)]
pub struct WasmMediaOutput {
    bytes: Vec<u8>,
    mime_type: String,
    max_output_bytes: u64,
    state: Rc<Cell<bool>>,
    timeline: Option<Timeline>,
}

#[wasm_bindgen(js_class = MediaOutput)]
impl WasmMediaOutput {
    pub fn create(options: &WasmCreateOptions) -> Promise {
        let mime_type = options.mime_type.clone();
        let max_output_bytes = options.max_output_bytes;
        let timeline = match (options.frame_rate, options.audio_sample_rate) {
            (Some(frame_rate), Some(sample_rate)) => Timeline::new(frame_rate, sample_rate).ok(),
            _ => None,
        };
        future_to_promise(async move {
            Ok(JsValue::from(Self {
                bytes: Vec::new(),
                mime_type,
                max_output_bytes,
                state: Rc::new(Cell::new(false)),
                timeline,
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
        Ok(WasmVideoStream {
            index,
            state: Rc::clone(&self.state),
            direction: StreamDirection::Output,
            bytes: None,
            decode_session: Rc::new(RefCell::new(None)),
            frame_durations_ms: Rc::new(RefCell::new(None)),
            random_access_points: Rc::new(RefCell::new(None)),
        })
    }

    pub fn audio(&self, index: u32) -> Result<WasmAudioStream, JsValue> {
        ensure_open(&self.state)?;
        Ok(WasmAudioStream {
            index,
            state: Rc::clone(&self.state),
            direction: StreamDirection::Output,
            bytes: None,
            timeline: self.timeline,
        })
    }

    pub fn finish(&mut self) -> Promise {
        let result = (|| {
            ensure_open(&self.state)?;
            let blob = make_blob(&self.bytes, &self.mime_type)
                .map_err(|error| normalize_browser_error(error, "creating output Blob"))?;
            self.state.set(true);
            self.bytes.clear();
            Ok(JsValue::from(blob))
        })();
        future_to_promise(async move { result })
    }

    #[wasm_bindgen(getter, js_name = isClosed)]
    pub fn is_closed(&self) -> bool {
        self.state.get()
    }

    pub fn close(&mut self) {
        if !self.state.replace(true) {
            self.bytes.clear();
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

    /// The bundled HEVC sample decoded through the real `WebCodecs` backend.
    ///
    /// Not every headless Chrome build can decode HEVC (it depends on
    /// platform codec licensing), so this accepts either a real decoded RGBA
    /// frame or a browser-reported `UNSUPPORTED`, and only fails on other
    /// error kinds or on structurally wrong output. Fetches two sequential
    /// presentation frames, matching how the `web_canvas` example plays back
    /// frame by frame, and checks that each frame's reported dimensions
    /// match its own pixel buffer rather than a stale, session-wide value.
    #[wasm_bindgen_test(async)]
    async fn video_get_decodes_the_bundled_sample_or_reports_unsupported() {
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
            match JsFuture::from(video.get(BigInt::from(frame_index).into(), None)).await {
                Ok(frame) => {
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
                    assert!(width > 0);
                    assert!(height > 0);
                    let pixels = Reflect::get(&frame, &JsValue::from_str("pixels")).unwrap();
                    let pixels: Uint8Array = pixels.unchecked_into();
                    assert_eq!(pixels.length(), width * height * 4);
                }
                Err(error) => assert_error_code(&error, "UNSUPPORTED"),
            }
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
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
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
            max_output_bytes: options.max_output_bytes,
            state: Rc::new(Cell::new(false)),
            timeline: None,
        };
        let chunk = Uint8Array::from(&[9_u8, 8, 7][..]);
        output.write_encoded_chunk(chunk.clone()).unwrap();
        chunk.set_index(0, 0);
        let blob = make_blob(&output.bytes, &output.mime_type).unwrap();
        let bytes = JsFuture::from(blob.array_buffer()).await.unwrap();
        assert_eq!(Uint8Array::new(&bytes).to_vec(), vec![9, 8, 7]);
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
}
