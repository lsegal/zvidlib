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
//! Scope for this initial bridge: AV1 Main profile output. Input accepts
//! RGBA8, BGRA8, and YUV 4:2:0 planar 8-bit frames; broader pixel-format
//! coverage is tracked as follow-up work.

use crate::av1::{Av1Obu, Av1Parser};
use crate::codec::{CodecImplementation, CodecSupport, SampleDependency};
use crate::media::required_plane_layouts;
use crate::web_decoder::{js_to_promise, normalize_js_error, schedule_event_loop_tick};
use crate::{
    Codec, CodecProfile, Error, ErrorKind, HardwarePreference, Limits, PixelFormat, Result,
    VideoFrame,
};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    EncodedVideoChunk, PlaneLayout, VideoEncoder as JsVideoEncoder,
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

/// A lazily-driven `WebCodecs` encode session producing AV1 Main chunks from
/// RGBA8 frames.
pub struct WebVideoEncodeSession {
    encoder: JsVideoEncoder,
    width: u32,
    height: u32,
    pending_chunks: Rc<RefCell<VecDeque<EncodedVideoChunk>>>,
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
        let pending_chunks: Rc<RefCell<VecDeque<EncodedVideoChunk>>> =
            Rc::new(RefCell::new(VecDeque::new()));
        let encode_error: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
        let waker: Rc<RefCell<Option<js_sys::Function>>> = Rc::new(RefCell::new(None));

        let output_chunks = Rc::clone(&pending_chunks);
        let output_waker = Rc::clone(&waker);
        // The second callback argument (`EncodedVideoChunkMetadata`) is not
        // used: AV1's sequence header travels in-band in the chunk's own
        // bytes rather than in `metadata.decoderConfig.description` (see
        // `av1c_from_bitstream`), so only the chunk itself is kept.
        let output_closure = Closure::new(move |chunk: EncodedVideoChunk, _metadata: JsValue| {
            output_chunks.borrow_mut().push_back(chunk);
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

    /// Encodes one input frame and returns the chunk it produces.
    ///
    /// `frame` must match the session's configured width/height and carry a
    /// pixel format this bridge supports (RGBA8, BGRA8, or YUV 4:2:0 planar
    /// 8-bit); see [`web_pixel_format`].
    ///
    /// Blocks until exactly one output chunk is available, which keeps
    /// ordering trivial: `WebCodecs` never reorders a video encoder's output
    /// relative to submission.
    pub async fn encode(
        &mut self,
        frame: &VideoFrame,
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
        if frame.dimensions.width != self.width || frame.dimensions.height != self.height {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "video frame dimensions do not match the configured width/height",
            ));
        }
        let format = web_pixel_format(frame.pixel_format)?;
        let (bytes, layout) = pack_planes_tightly(frame)?;

        let data = js_sys::Uint8Array::from(bytes.as_slice());
        let init = VideoFrameBufferInit::new_with_f64(
            self.height,
            self.width,
            format,
            timestamp_micros as f64,
        );
        init.set_duration_f64(f64::from(duration_micros));
        if let Some(layout) = layout.as_deref() {
            init.set_layout(layout);
        }
        let js_frame =
            JsVideoFrame::new_with_buffer_source_and_video_frame_buffer_init(&data, &init)
                .map_err(|error| normalize_js_error(error, "constructing a VideoFrame"))?;

        let options = VideoEncoderEncodeOptions::new();
        options.set_key_frame(key_frame);
        let result = self.encoder.encode_with_options(&js_frame, &options);
        js_frame.close();
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
        let Some(chunk) = self.pending_chunks.borrow_mut().pop_front() else {
            return Ok(None);
        };
        let destination = js_sys::Uint8Array::new_with_length(chunk.byte_length());
        chunk
            .copy_to_with_u8_array(&destination)
            .map_err(|error| normalize_js_error(error, "copying an encoded video chunk"))?;
        let data = destination.to_vec();
        let is_sync = chunk.type_() == web_sys::EncodedVideoChunkType::Key;

        // AV1's sequence header travels in-band in the bitstream rather than
        // through `EncodedVideoChunkMetadata.decoderConfig.description`
        // (that field is for codecs like AVC/HEVC whose parameter sets are
        // genuinely out-of-band), so the real `av1C` is derived from the key
        // chunk's own bytes instead of from chunk metadata.
        let decoder_config = if !self.emitted_config && is_sync {
            av1c_from_bitstream(&data).inspect(|_| {
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

/// Maps a portable [`PixelFormat`] to the `WebCodecs` pixel format this
/// bridge feeds it as.
///
/// Only formats `WebCodecs` accepts as a `VideoFrameBufferInit.format` and
/// this bridge has been wired up for are supported; other formats are
/// rejected rather than silently reinterpreted.
fn web_pixel_format(pixel_format: PixelFormat) -> Result<VideoPixelFormat> {
    match pixel_format {
        PixelFormat::Rgba8 => Ok(VideoPixelFormat::Rgba),
        PixelFormat::Bgra8 => Ok(VideoPixelFormat::Bgra),
        PixelFormat::Yuv420p8 => Ok(VideoPixelFormat::I420),
        PixelFormat::Rgb8 | PixelFormat::Gray8 => Err(Error::new(
            ErrorKind::Unsupported,
            "this pixel format is not supported by the WebCodecs export bridge",
        )),
    }
}

/// Packs `frame`'s planes into one contiguous, tightly-packed buffer for
/// `VideoFrame`'s `BufferSource` constructor, plus an explicit `PlaneLayout`
/// describing each plane's offset and stride when there is more than one
/// plane (a single-plane buffer uses the format's own default layout).
///
/// A plane's own `stride` may be wider than its pixel data (e.g. row
/// padding); this copies out just the pixel bytes of each row so the
/// resulting buffer has no gaps `WebCodecs` would otherwise need a
/// `PlaneLayout` to skip over.
fn pack_planes_tightly(frame: &VideoFrame) -> Result<(Vec<u8>, Option<Vec<PlaneLayout>>)> {
    let layouts = required_plane_layouts(frame.dimensions, frame.pixel_format)?;
    let mut bytes = Vec::new();
    let mut plane_layout = Vec::with_capacity(layouts.len());
    for (plane, (row_bytes, rows)) in frame.planes.iter().zip(&layouts) {
        let offset = u32::try_from(bytes.len())
            .map_err(|_| Error::new(ErrorKind::ResourceLimit, "video plane offset overflow"))?;
        for row in 0..*rows {
            let start = row
                .checked_mul(plane.stride)
                .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "video plane row overflow"))?;
            let end = start
                .checked_add(*row_bytes)
                .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "video plane row overflow"))?;
            let row_data = plane.data.get(start..end).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "a video plane is shorter than its stride and height",
                )
            })?;
            bytes.extend_from_slice(row_data);
        }
        let stride = u32::try_from(*row_bytes)
            .map_err(|_| Error::new(ErrorKind::ResourceLimit, "video row size overflow"))?;
        plane_layout.push(PlaneLayout::new(offset, stride));
    }
    let layout = (layouts.len() > 1).then_some(plane_layout);
    Ok((bytes, layout))
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

/// Builds a complete `av1C` box from a key chunk's raw AV1 OBU bitstream.
///
/// `WebCodecs` does not hand back a separate decoder configuration for AV1:
/// its sequence header is one of the OBUs already in the chunk's own bytes
/// (the "low overhead bitstream format" AV1's ISO Media binding also uses for
/// `configOBUs`), so this scans for it directly rather than trusting
/// out-of-band metadata AV1 doesn't populate.
fn av1c_from_bitstream(data: &[u8]) -> Option<Vec<u8>> {
    let sequence_header_obu = extract_sequence_header_obu(data)?;
    let mut parser = Av1Parser::new(Limits::default()).ok()?;
    let parsed = parser.parse_low_overhead(sequence_header_obu).ok()?;
    let sequence = parsed.iter().find_map(|obu| match obu {
        Av1Obu::SequenceHeader { sequence, .. } => Some(sequence),
        _ => None,
    })?;
    let operating_point = sequence.operating_points.first()?;
    let color = &sequence.color_config;

    let mut payload = Vec::with_capacity(4 + sequence_header_obu.len());
    payload.push(0x81);
    payload.push((sequence.seq_profile << 5) | (operating_point.level & 0x1f));
    payload.push(
        (u8::from(operating_point.tier) << 7)
            | (u8::from(color.bit_depth > 8) << 6)
            | (u8::from(color.bit_depth == 12) << 5)
            | (u8::from(color.monochrome) << 4)
            | (u8::from(color.subsampling_x) << 3)
            | (u8::from(color.subsampling_y) << 2)
            | (color.chroma_sample_position & 3),
    );
    payload.push(0); // no initial presentation delay
    payload.extend_from_slice(sequence_header_obu);
    Some(wrap_box(b"av1C", &payload))
}

/// Scans a raw "low overhead bitstream format" OBU stream (AV1 §5.2) for its
/// first Sequence Header OBU and returns that OBU's exact bytes (header, size
/// field, and payload), which is what `av1C`'s `configOBUs` embeds verbatim.
fn extract_sequence_header_obu(bytes: &[u8]) -> Option<&[u8]> {
    const SEQUENCE_HEADER_OBU_TYPE: u8 = 1;
    let mut offset = 0;
    while offset < bytes.len() {
        let header_byte = *bytes.get(offset)?;
        let obu_type = (header_byte >> 3) & 0b1111;
        let has_extension = header_byte & 0b0000_0100 != 0;
        let has_size_field = header_byte & 0b0000_0010 != 0;
        if !has_size_field {
            // Every OBU emitted by a `WebCodecs` encoder carries a size
            // field; one that doesn't cannot be bounded without decoding it.
            return None;
        }
        let size_offset = offset + 1 + usize::from(has_extension);
        let (payload_len, leb_len) = read_leb128(bytes.get(size_offset..)?)?;
        let payload_start = size_offset + leb_len;
        let obu_len = (payload_start - offset).checked_add(usize::try_from(payload_len).ok()?)?;
        let obu = bytes.get(offset..offset.checked_add(obu_len)?)?;
        if obu_type == SEQUENCE_HEADER_OBU_TYPE {
            return Some(obu);
        }
        offset = offset.checked_add(obu_len)?;
    }
    None
}

/// Reads an AV1 `leb128()` value, returning it with the number of bytes read.
fn read_leb128(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    for (index, &byte) in bytes.iter().enumerate().take(8) {
        value |= u64::from(byte & 0x7f) << (index * 7);
        if byte & 0x80 == 0 {
            return Some((value, index + 1));
        }
    }
    None
}
