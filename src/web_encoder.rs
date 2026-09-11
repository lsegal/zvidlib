//! Browser `WebCodecs`-backed video encoding for the `web` target.
//!
//! This bridges the browser's asynchronous `VideoEncoder`/output-callback
//! model to a simple exact-frame `put` used by [`crate::WasmVideoStream`].
//! Because encode is inherently asynchronous here, this does not implement
//! the portable, synchronous [`crate::codec::VideoEncoder`] trait: that
//! trait's `config()` must return a complete MP4 sample-entry `decoder_config`
//! before any frame is encoded, but `WebCodecs` only reveals the real
//! `av1C`/`hvcC` description once the encoder emits its first chunk. Callers
//! ([`crate::WasmMediaOutput`]) instead buffer chunks from this session and
//! build the MP4 track configuration once that first chunk arrives.
//!
//! Scope for this initial bridge: AV1 Main profile output from RGBA8 input.
//! Broader codec and pixel-format coverage is tracked as follow-up work.

use crate::codec::{CodecImplementation, CodecSupport, SampleDependency};
use crate::web_decoder::{js_to_promise, normalize_js_error, schedule_event_loop_tick};
use crate::{Codec, CodecProfile, Error, ErrorKind, HardwarePreference, Result};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    EncodedVideoChunk, EncodedVideoChunkMetadata, VideoEncoder as JsVideoEncoder,
    VideoEncoderConfig as JsVideoEncoderConfig, VideoEncoderEncodeOptions, VideoEncoderInit,
    VideoFrame as JsVideoFrame, VideoFrameBufferInit, VideoPixelFormat,
};

/// One encoded output from a [`WebVideoEncodeSession`].
pub struct WebEncodedVideoChunk {
    pub data: Vec<u8>,
    pub is_sync: bool,
    /// Complete `av1C` configuration box (size + fourcc + payload), present
    /// only on the first chunk a session ever emits.
    pub decoder_config: Option<Vec<u8>>,
}

/// Best-effort, synchronous support check for the `WebCodecs` video encoder
/// bridge.
///
/// `WebCodecs`' own `VideoEncoder.isConfigSupported()` is asynchronous, which
/// the portable [`crate::codec::VideoEncoderFactory::capability`] contract
/// cannot express, so this checks only what a caller can know before
/// attempting to configure a real session: that the browser exposes a
/// `VideoEncoder` constructor at all, and that the requested codec/profile is
/// one this bridge implements.
pub fn video_encode_capability(
    codec: Codec,
    profile: CodecProfile,
    hardware: HardwarePreference,
) -> CodecSupport {
    if codec != Codec::Av1 || profile != CodecProfile::Av1Main {
        return CodecSupport::UnsupportedProfile;
    }
    if hardware == HardwarePreference::Require {
        // Whether the browser honors a hardware encoder is only knowable
        // asynchronously (and per-configuration), so a caller that requires
        // it cannot be promised support here.
        return CodecSupport::HardwareUnavailable;
    }
    let global = js_sys::global();
    let has_constructor =
        js_sys::Reflect::has(&global, &JsValue::from_str("VideoEncoder")).unwrap_or(false);
    if !has_constructor {
        return CodecSupport::UnsupportedCodec;
    }
    CodecSupport::Supported {
        implementation: CodecImplementation::Software,
    }
}

/// One `WebCodecs` output chunk paired with the metadata it arrived with.
type PendingChunk = (EncodedVideoChunk, Option<EncodedVideoChunkMetadata>);

/// A lazily-driven `WebCodecs` encode session producing AV1 Main chunks from
/// RGBA8 frames.
pub struct WebVideoEncodeSession {
    encoder: JsVideoEncoder,
    width: u32,
    height: u32,
    pending_chunks: Rc<RefCell<VecDeque<PendingChunk>>>,
    encode_error: Rc<RefCell<Option<String>>>,
    waker: Rc<RefCell<Option<js_sys::Function>>>,
    emitted_config: bool,
    finished: bool,
    // Kept alive for the lifetime of `encoder`.
    _output_closure: Closure<dyn FnMut(EncodedVideoChunk, JsValue)>,
    _error_closure: Closure<dyn FnMut(JsValue)>,
}

impl WebVideoEncodeSession {
    /// Opens a session targeting AV1 Main at `width`x`height`, timestamps and
    /// durations given in microseconds.
    pub fn open(width: u32, height: u32, bitrate_bits_per_second: Option<u32>) -> Result<Self> {
        let pending_chunks: Rc<RefCell<VecDeque<PendingChunk>>> =
            Rc::new(RefCell::new(VecDeque::new()));
        let encode_error: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
        let waker: Rc<RefCell<Option<js_sys::Function>>> = Rc::new(RefCell::new(None));

        let output_chunks = Rc::clone(&pending_chunks);
        let output_waker = Rc::clone(&waker);
        let output_closure = Closure::new(move |chunk: EncodedVideoChunk, metadata: JsValue| {
            let metadata: Option<EncodedVideoChunkMetadata> = metadata.dyn_into().ok();
            output_chunks.borrow_mut().push_back((chunk, metadata));
            if let Some(resolve) = output_waker.borrow_mut().take() {
                let _ = resolve.call0(&JsValue::NULL);
            }
        });
        let error_state = Rc::clone(&encode_error);
        let error_waker = Rc::clone(&waker);
        let error_closure = Closure::new(move |error: JsValue| {
            let message = js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                .ok()
                .and_then(|value| value.as_string())
                .unwrap_or_else(|| "WebCodecs encoder reported an error".to_owned());
            *error_state.borrow_mut() = Some(message);
            if let Some(resolve) = error_waker.borrow_mut().take() {
                let _ = resolve.call0(&JsValue::NULL);
            }
        });

        let init = VideoEncoderInit::new(
            error_closure.as_ref().unchecked_ref(),
            output_closure.as_ref().unchecked_ref(),
        );
        let encoder = JsVideoEncoder::new(&init)
            .map_err(|error| normalize_js_error(error, "constructing a WebCodecs VideoEncoder"))?;

        let config = JsVideoEncoderConfig::new("av01.0.00M.08", height, width);
        if let Some(bitrate) = bitrate_bits_per_second {
            config.set_bitrate(bitrate);
        }
        encoder
            .configure(&config)
            .map_err(|error| normalize_js_error(error, "configuring the WebCodecs VideoEncoder"))?;

        Ok(Self {
            encoder,
            width,
            height,
            pending_chunks,
            encode_error,
            waker,
            emitted_config: false,
            finished: false,
            _output_closure: output_closure,
            _error_closure: error_closure,
        })
    }

    /// Encodes one RGBA8, tightly-packed (`stride == width * 4`) frame and
    /// returns the chunk it produces.
    ///
    /// Blocks until exactly one output chunk is available, which keeps
    /// ordering trivial: `WebCodecs` never reorders a video encoder's output
    /// relative to submission.
    pub async fn encode(
        &mut self,
        rgba: &[u8],
        timestamp_micros: u64,
        duration_micros: u32,
        key_frame: bool,
    ) -> Result<WebEncodedVideoChunk> {
        if self.finished {
            return Err(Error::new(
                ErrorKind::InvalidState,
                "the WebCodecs video encoder session has already finished",
            ));
        }
        let required_len = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "video frame size overflow"))?;
        if rgba.len() != required_len {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "RGBA frame data does not match the configured width/height",
            ));
        }

        let data = js_sys::Uint8Array::from(rgba);
        let init = VideoFrameBufferInit::new_with_f64(
            self.height,
            self.width,
            VideoPixelFormat::Rgba,
            timestamp_micros as f64,
        );
        init.set_duration_f64(f64::from(duration_micros));
        let frame = JsVideoFrame::new_with_buffer_source_and_video_frame_buffer_init(&data, &init)
            .map_err(|error| normalize_js_error(error, "constructing a VideoFrame"))?;

        let options = VideoEncoderEncodeOptions::new();
        options.set_key_frame(key_frame);
        let result = self.encoder.encode_with_options(&frame, &options);
        frame.close();
        result.map_err(|error| normalize_js_error(error, "encoding a video frame"))?;

        self.wait_for_chunk().await
    }

    /// Flushes the encoder and returns any chunks still pending.
    pub async fn finish(&mut self) -> Result<Vec<WebEncodedVideoChunk>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        JsFuture::from(js_to_promise(self.encoder.flush()))
            .await
            .map_err(|error| normalize_js_error(error, "flushing the WebCodecs VideoEncoder"))?;
        let mut chunks = Vec::new();
        while let Some(chunk) = self.take_ready_chunk()? {
            chunks.push(chunk);
        }
        let _ = self.encoder.close();
        Ok(chunks)
    }

    async fn wait_for_chunk(&mut self) -> Result<WebEncodedVideoChunk> {
        loop {
            if let Some(message) = self.encode_error.borrow_mut().take() {
                return Err(Error::new(ErrorKind::Codec, message));
            }
            if let Some(chunk) = self.take_ready_chunk()? {
                return Ok(chunk);
            }
            self.wait_for_output().await;
        }
    }

    fn take_ready_chunk(&mut self) -> Result<Option<WebEncodedVideoChunk>> {
        let Some((chunk, metadata)) = self.pending_chunks.borrow_mut().pop_front() else {
            return Ok(None);
        };
        let destination = js_sys::Uint8Array::new_with_length(chunk.byte_length());
        chunk
            .copy_to_with_u8_array(&destination)
            .map_err(|error| normalize_js_error(error, "copying an encoded video chunk"))?;
        let data = destination.to_vec();
        let is_sync = chunk.type_() == web_sys::EncodedVideoChunkType::Key;

        let decoder_config = if !self.emitted_config {
            let description = metadata
                .and_then(|metadata| metadata.get_decoder_config())
                .and_then(|config| config.get_description());
            description
                .map(|description| {
                    let bytes = js_sys::Uint8Array::new(&description).to_vec();
                    wrap_box(b"av1C", &bytes)
                })
                .inspect(|_| {
                    self.emitted_config = true;
                })
        } else {
            None
        };

        Ok(Some(WebEncodedVideoChunk {
            data,
            is_sync,
            decoder_config,
        }))
    }

    async fn wait_for_output(&self) {
        let waker = Rc::clone(&self.waker);
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            schedule_event_loop_tick(&resolve);
            *waker.borrow_mut() = Some(resolve);
        });
        let _ = JsFuture::from(promise).await;
    }
}

impl Drop for WebVideoEncodeSession {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.encoder.close();
        }
    }
}

/// The dependency an encoded sample declares in the MP4 sample dependency
/// table, from whether `WebCodecs` marked it a key frame.
pub fn sample_dependency(is_sync: bool) -> SampleDependency {
    if is_sync {
        SampleDependency::INDEPENDENT
    } else {
        SampleDependency::DEPENDENT
    }
}

/// Wraps a raw box payload with its 32-bit size and fourcc header.
fn wrap_box(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut boxed = Vec::with_capacity(8 + payload.len());
    let size = (8 + payload.len()) as u32;
    boxed.extend_from_slice(&size.to_be_bytes());
    boxed.extend_from_slice(fourcc);
    boxed.extend_from_slice(payload);
    boxed
}
