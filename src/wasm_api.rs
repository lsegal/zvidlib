//! Browser-facing wrappers for the portable core.
//!
//! Values crossing this boundary are copied into owned Rust storage. Returned
//! typed arrays are snapshots rather than views into growable WebAssembly
//! memory, and browser-owned objects are retained only as JavaScript handles.

use crate::{
    AudioBuffer as CoreAudioBuffer, ColorRange, ErrorKind, FrameIndex as CoreFrameIndex, FrameRate,
    Limits, PixelFormat, Plane, Rational as CoreRational, SampleRange as CoreSampleRange, Timeline,
    VideoDimensions, VideoFrame as CoreVideoFrame,
};
use js_sys::{BigInt, Float32Array, Promise, Reflect, Uint8Array};
use std::cell::Cell;
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
                    "get is only valid on an input video stream",
                ));
            }
            Err(js_error(
                ErrorKind::Unsupported,
                "no browser video decoder backend is registered",
            ))
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

#[wasm_bindgen(js_name = AudioStream)]
pub struct WasmAudioStream {
    index: u32,
    state: Rc<Cell<bool>>,
    direction: StreamDirection,
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
    bytes: Vec<u8>,
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
            bytes: bytes.to_vec(),
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
        })
    }

    pub fn audio(&self, index: u32) -> Result<WasmAudioStream, JsValue> {
        ensure_open(&self.state)?;
        Ok(WasmAudioStream {
            index,
            state: Rc::clone(&self.state),
            direction: StreamDirection::Input,
            timeline: None,
        })
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
        })
    }

    pub fn audio(&self, index: u32) -> Result<WasmAudioStream, JsValue> {
        ensure_open(&self.state)?;
        Ok(WasmAudioStream {
            index,
            state: Rc::clone(&self.state),
            direction: StreamDirection::Output,
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
    use js_sys::{Array, Object};
    use wasm_bindgen_test::*;
    use web_sys::AbortController;

    wasm_bindgen_test_configure!(run_in_browser);

    fn assert_error_code(error: &JsValue, expected: &str) {
        assert_eq!(error_code(error).as_deref(), Some(expected));
        assert!(error.is_instance_of::<js_sys::Error>());
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
