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
//! Scope for this initial bridge: AV1 Main and HEVC Main profile output from
//! RGBA8 input, and AAC-LC audio output from interleaved f32 PCM. Broader
//! codec and pixel-format coverage is tracked as follow-up work.

use crate::av1::{Av1Obu, Av1Parser};
use crate::codec::{CodecImplementation, CodecSupport, SampleDependency};
use crate::web_decoder::{js_to_promise, normalize_js_error, schedule_event_loop_tick};
use crate::{Codec, CodecProfile, Error, ErrorKind, HardwarePreference, Limits, Result};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    AudioData as JsAudioData, AudioDataInit, AudioEncoder as JsAudioEncoder,
    AudioEncoderConfig as JsAudioEncoderConfig, AudioEncoderInit, AudioSampleFormat,
    EncodedAudioChunk, EncodedAudioChunkMetadata, EncodedAudioChunkType, EncodedVideoChunk,
    EncodedVideoChunkMetadata, VideoEncoder as JsVideoEncoder,
    VideoEncoderConfig as JsVideoEncoderConfig, VideoEncoderEncodeOptions, VideoEncoderInit,
    VideoEncoderSupport, VideoFrame as JsVideoFrame, VideoFrameBufferInit, VideoPixelFormat,
};

/// One encoded output from a [`WebVideoEncodeSession`].
pub struct WebEncodedVideoChunk {
    pub data: Vec<u8>,
    pub is_sync: bool,
    /// Complete `av1C`/`hvcC` configuration box (size + fourcc + payload),
    /// present only on the first chunk a session ever emits.
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
    let supported = matches!(
        (codec, profile),
        (Codec::Av1, CodecProfile::Av1Main) | (Codec::Hevc, CodecProfile::HevcMain)
    );
    if !supported {
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

/// A lazily-driven `WebCodecs` encode session producing AV1 Main or HEVC Main
/// chunks from RGBA8 frames.
pub struct WebVideoEncodeSession {
    encoder: JsVideoEncoder,
    codec: Codec,
    width: u32,
    height: u32,
    pending_chunks: Rc<RefCell<VecDeque<(EncodedVideoChunk, JsValue)>>>,
    encode_error: Rc<RefCell<Option<String>>>,
    waker: Rc<RefCell<Option<js_sys::Function>>>,
    emitted_config: bool,
    finished: bool,
    // Kept alive for the lifetime of `encoder`.
    _output_closure: Closure<dyn FnMut(EncodedVideoChunk, JsValue)>,
    _error_closure: Closure<dyn FnMut(JsValue)>,
}

impl WebVideoEncodeSession {
    /// Opens a session targeting `codec` (AV1 Main or HEVC Main) at
    /// `width`x`height`, timestamps and durations given in microseconds.
    pub async fn open(
        codec: Codec,
        width: u32,
        height: u32,
        bitrate_bits_per_second: Option<u32>,
    ) -> Result<Self> {
        let initial_codec_string = match codec {
            Codec::Av1 => "av01.0.00M.08",
            // Main profile, tier L, level 3.1, no constraint flags. Only a
            // starting point for `configure()`: like AV1, the real profile
            // and level are read back from what the encoder actually emits.
            Codec::Hevc => "hev1.1.6.L93.B0",
            Codec::UncompressedVideo | Codec::Aac => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "the WebCodecs video encoder bridge only supports AV1 and HEVC",
                ));
            }
        };

        let config = JsVideoEncoderConfig::new(initial_codec_string, height, width);
        if let Some(bitrate) = bitrate_bits_per_second {
            config.set_bitrate(bitrate);
        }

        // `configure()` only ever synchronously validates the shape of the
        // config; a codec/profile the browser cannot actually encode is
        // reported asynchronously by closing the encoder and invoking the
        // error callback. Checking `isConfigSupported()` first means an
        // unsupported codec (e.g. no HEVC software encoder) surfaces here as
        // a normal error rather than as a "closed codec" failure on the
        // first `encode()` call.
        let support: VideoEncoderSupport =
            JsFuture::from(js_to_promise(JsVideoEncoder::is_config_supported(&config)))
                .await
                .map_err(|error| normalize_js_error(error, "querying WebCodecs encoder support"))?
                .unchecked_into();
        if !support.get_supported().unwrap_or(false) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!("this browser cannot encode {initial_codec_string} via WebCodecs"),
            ));
        }

        let pending_chunks: Rc<RefCell<VecDeque<(EncodedVideoChunk, JsValue)>>> =
            Rc::new(RefCell::new(VecDeque::new()));
        let encode_error: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
        let waker: Rc<RefCell<Option<js_sys::Function>>> = Rc::new(RefCell::new(None));

        let output_chunks = Rc::clone(&pending_chunks);
        let output_waker = Rc::clone(&waker);
        // AV1's sequence header travels in-band in the chunk's own bytes
        // (see `av1c_from_bitstream`), but HEVC's parameter sets are
        // genuinely out-of-band, so `metadata` is kept alongside the chunk
        // for `take_ready_chunk` to read `decoderConfig.description` from.
        let output_closure = Closure::new(move |chunk: EncodedVideoChunk, metadata: JsValue| {
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

        encoder
            .configure(&config)
            .map_err(|error| normalize_js_error(error, "configuring the WebCodecs VideoEncoder"))?;

        Ok(Self {
            encoder,
            codec,
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

        // AV1's sequence header travels in-band in the bitstream, so the
        // real `av1C` is derived from the key chunk's own bytes. HEVC's
        // parameter sets are genuinely out-of-band, so its `hvcC` is instead
        // read from `EncodedVideoChunkMetadata.decoderConfig.description`.
        let decoder_config = if !self.emitted_config && is_sync {
            let config = match self.codec {
                Codec::Av1 => av1c_from_bitstream(&data),
                Codec::Hevc => hvcc_from_metadata(&metadata),
                Codec::UncompressedVideo | Codec::Aac => None,
            };
            config.inspect(|_| {
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

/// One encoded output from a [`WebAudioEncodeSession`].
pub struct WebEncodedAudioChunk {
    pub data: Vec<u8>,
    pub is_sync: bool,
    pub timestamp_micros: f64,
    pub duration_micros: Option<f64>,
    /// Complete `esds` configuration box (size + fourcc + payload), present
    /// only on the first chunk a session ever emits.
    pub decoder_config: Option<Vec<u8>>,
}

/// Best-effort, synchronous support check for the `WebCodecs` audio encoder
/// bridge. See [`video_encode_capability`] for why this cannot be more than a
/// best effort: `isConfigSupported()` is asynchronous.
pub fn audio_encode_capability(
    codec: Codec,
    profile: CodecProfile,
    hardware: HardwarePreference,
) -> CodecSupport {
    if codec != Codec::Aac || profile != CodecProfile::AacLowComplexity {
        return CodecSupport::UnsupportedProfile;
    }
    if hardware == HardwarePreference::Require {
        return CodecSupport::HardwareUnavailable;
    }
    let global = js_sys::global();
    let has_constructor =
        js_sys::Reflect::has(&global, &JsValue::from_str("AudioEncoder")).unwrap_or(false);
    if !has_constructor {
        return CodecSupport::UnsupportedCodec;
    }
    CodecSupport::Supported {
        implementation: CodecImplementation::Software,
    }
}

/// A lazily-driven `WebCodecs` encode session producing AAC-LC chunks from
/// interleaved `f32` PCM.
///
/// Unlike [`WebVideoEncodeSession`], `encode()` here does not wait for an
/// output chunk before returning: an `AudioEncoder` buffers encoder
/// look-ahead/priming across several input buffers before it emits its first
/// fixed-size AAC frame (mirroring the priming the native AudioToolbox AAC-LC
/// backend has to handle explicitly), so waiting inside `encode()` for a
/// chunk that only a *later* `encode()` call would elicit would deadlock the
/// session against itself. `encode()` instead submits input and returns
/// whatever chunks already happen to be ready (often none); `finish()`
/// awaits `flush()`, which the spec guarantees does not resolve until every
/// submitted input's output has been delivered, and only then drains the
/// rest.
pub struct WebAudioEncodeSession {
    encoder: JsAudioEncoder,
    sample_rate: u32,
    channels: u16,
    pending_chunks: Rc<RefCell<VecDeque<(EncodedAudioChunk, EncodedAudioChunkMetadata)>>>,
    encode_error: Rc<RefCell<Option<String>>>,
    emitted_config: bool,
    finished: bool,
    // Kept alive for the lifetime of `encoder`.
    _output_closure: Closure<dyn FnMut(EncodedAudioChunk, EncodedAudioChunkMetadata)>,
    _error_closure: Closure<dyn FnMut(JsValue)>,
}

impl WebAudioEncodeSession {
    /// Opens a session targeting AAC-LC at `sample_rate`/`channels`,
    /// timestamps and durations given in microseconds.
    pub fn open(
        sample_rate: u32,
        channels: u16,
        bitrate_bits_per_second: Option<u32>,
    ) -> Result<Self> {
        let pending_chunks: Rc<RefCell<VecDeque<(EncodedAudioChunk, EncodedAudioChunkMetadata)>>> =
            Rc::new(RefCell::new(VecDeque::new()));
        let encode_error: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));

        let output_chunks = Rc::clone(&pending_chunks);
        let output_closure = Closure::new(
            move |chunk: EncodedAudioChunk, metadata: EncodedAudioChunkMetadata| {
                output_chunks.borrow_mut().push_back((chunk, metadata));
            },
        );
        let error_state = Rc::clone(&encode_error);
        let error_closure = Closure::new(move |error: JsValue| {
            let message = js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                .ok()
                .and_then(|value| value.as_string())
                .unwrap_or_else(|| "WebCodecs encoder reported an error".to_owned());
            *error_state.borrow_mut() = Some(message);
        });

        let init = AudioEncoderInit::new(
            error_closure.as_ref().unchecked_ref(),
            output_closure.as_ref().unchecked_ref(),
        );
        let encoder = JsAudioEncoder::new(&init)
            .map_err(|error| normalize_js_error(error, "constructing a WebCodecs AudioEncoder"))?;

        let config = JsAudioEncoderConfig::new("mp4a.40.2", u32::from(channels), sample_rate);
        if let Some(bitrate) = bitrate_bits_per_second {
            config.set_bitrate(bitrate);
        }
        encoder
            .configure(&config)
            .map_err(|error| normalize_js_error(error, "configuring the WebCodecs AudioEncoder"))?;

        Ok(Self {
            encoder,
            sample_rate,
            channels,
            pending_chunks,
            encode_error,
            emitted_config: false,
            finished: false,
            _output_closure: output_closure,
            _error_closure: error_closure,
        })
    }

    /// Encodes one buffer of interleaved `f32` PCM and returns every chunk
    /// already ready, which is often empty: an `AudioEncoder` typically
    /// buffers several buffers' worth of input before emitting its first
    /// fixed-size AAC frame.
    pub async fn encode(
        &mut self,
        samples: &[f32],
        timestamp_micros: f64,
    ) -> Result<Vec<WebEncodedAudioChunk>> {
        if self.finished {
            return Err(Error::new(
                ErrorKind::InvalidState,
                "the WebCodecs audio encoder session has already finished",
            ));
        }
        let channels = usize::from(self.channels);
        if channels == 0 || samples.len() % channels != 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "interleaved audio sample count is not a multiple of the channel count",
            ));
        }
        let number_of_frames = u32::try_from(samples.len() / channels).map_err(|_| {
            Error::new(
                ErrorKind::ResourceLimit,
                "audio buffer frame count overflow",
            )
        })?;

        let bytes: &[u8] = bytemuck_cast_f32_slice(samples);
        let data_array = js_sys::Uint8Array::from(bytes);
        let init = AudioDataInit::new_with_u8_array(
            &data_array,
            AudioSampleFormat::F32,
            u32::from(self.channels),
            number_of_frames,
            self.sample_rate as f32,
            0,
        );
        init.set_timestamp_f64(timestamp_micros);
        let audio_data = JsAudioData::new(&init)
            .map_err(|error| normalize_js_error(error, "constructing an AudioData"))?;

        let result = self.encoder.encode(&audio_data);
        audio_data.close();
        result.map_err(|error| normalize_js_error(error, "encoding an audio buffer"))?;

        if let Some(message) = self.encode_error.borrow_mut().take() {
            return Err(Error::new(ErrorKind::Codec, message));
        }
        self.take_ready_chunks()
    }

    /// Flushes the encoder - which the spec guarantees does not resolve until
    /// every chunk from every prior `encode()` call has been delivered to the
    /// output callback - and returns every chunk still pending.
    pub async fn finish(&mut self) -> Result<Vec<WebEncodedAudioChunk>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        // `flush()` itself rejects with a generic "closed codec" error when an
        // earlier `encode()` call's failure has already closed the encoder
        // (per the `WebCodecs` spec), which would otherwise mask the real
        // reason. That failure surfaces to `encode_error` no later than
        // `flush()`'s own settling, so it is checked first and preferred over
        // `flush()`'s own (less informative) rejection.
        let flush_result = JsFuture::from(js_to_promise(self.encoder.flush())).await;
        if let Some(message) = self.encode_error.borrow_mut().take() {
            let _ = self.encoder.close();
            return Err(Error::new(ErrorKind::Codec, message));
        }
        flush_result
            .map_err(|error| normalize_js_error(error, "flushing the WebCodecs AudioEncoder"))?;
        let chunks = self.take_ready_chunks()?;
        let _ = self.encoder.close();
        Ok(chunks)
    }

    fn take_ready_chunks(&mut self) -> Result<Vec<WebEncodedAudioChunk>> {
        let mut chunks = Vec::new();
        while let Some((chunk, metadata)) = self.pending_chunks.borrow_mut().pop_front() {
            let destination = js_sys::Uint8Array::new_with_length(chunk.byte_length());
            chunk
                .copy_to_with_buffer_source(&destination)
                .map_err(|error| normalize_js_error(error, "copying an encoded audio chunk"))?;
            let data = destination.to_vec();
            let is_sync = chunk.type_() == EncodedAudioChunkType::Key;

            let decoder_config = if !self.emitted_config {
                metadata
                    .get_decoder_config()
                    .and_then(|config| config.get_description())
                    .map(|description| {
                        let bytes = js_sys::Uint8Array::new(&description).to_vec();
                        esds_box(&bytes)
                    })
                    .inspect(|_| {
                        self.emitted_config = true;
                    })
            } else {
                None
            };

            chunks.push(WebEncodedAudioChunk {
                data,
                is_sync,
                timestamp_micros: chunk.timestamp(),
                duration_micros: chunk.duration(),
                decoder_config,
            });
        }
        Ok(chunks)
    }
}

impl Drop for WebAudioEncodeSession {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.encoder.close();
        }
    }
}

/// Reinterprets an `f32` PCM slice as its little-endian byte representation,
/// which is what `AudioSampleFormat::F32` expects `AudioDataInit`'s buffer to
/// contain.
fn bytemuck_cast_f32_slice(samples: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding and every bit pattern is valid, so viewing
    // it as `u8` is always sound; the returned slice borrows `samples` and
    // cannot outlive it.
    unsafe {
        std::slice::from_raw_parts(
            samples.as_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
}

/// Writes an MPEG-4 descriptor length, big-endian base-128 with the
/// continuation bit set on every byte but the last, as `esds` and its nested
/// descriptors require.
fn write_descriptor_length(out: &mut Vec<u8>, length: u32) {
    let mut chunks = [0_u8; 4];
    let mut remaining = length;
    let mut count = 0;
    loop {
        chunks[count] = (remaining & 0x7f) as u8;
        remaining >>= 7;
        count += 1;
        if remaining == 0 || count == chunks.len() {
            break;
        }
    }
    for (position, index) in (0..count).rev().enumerate() {
        let mut byte = chunks[index];
        if position != count - 1 {
            byte |= 0x80;
        }
        out.push(byte);
    }
}

fn write_descriptor(out: &mut Vec<u8>, tag: u8, content: &[u8]) {
    out.push(tag);
    write_descriptor_length(out, u32::try_from(content.len()).unwrap_or(u32::MAX));
    out.extend_from_slice(content);
}

/// Builds the complete `esds` MP4 box declaring an AAC track from the raw
/// `AudioSpecificConfig` bytes `WebCodecs` reports on its first emitted
/// chunk's `metadata.decoderConfig.description`: an `ES_Descriptor` wrapping a
/// `DecoderConfigDescriptor` (object type `0x40`, MPEG-4 audio stream type)
/// whose `DecoderSpecificInfo` is that `AudioSpecificConfig` verbatim,
/// followed by the file-format `SLConfigDescriptor`.
///
/// Unlike AV1 (see `av1c_from_bitstream`), AAC's decoder configuration is
/// genuinely out-of-band, so this trusts `WebCodecs`' own report of it rather
/// than deriving it from the bitstream.
fn esds_box(audio_specific_config: &[u8]) -> Vec<u8> {
    let mut decoder_specific_info = Vec::new();
    write_descriptor(&mut decoder_specific_info, 0x05, audio_specific_config);

    let mut decoder_config_descriptor = vec![
        0x40, // objectTypeIndication: MPEG-4 Audio
        0x15, // streamType (5, audio) << 2 | upStream (0) << 1 | reserved (1)
        0, 0, 0, // bufferSizeDB
        0, 0, 0, 0, // maxBitrate
        0, 0, 0, 0, // avgBitrate
    ];
    decoder_config_descriptor.extend_from_slice(&decoder_specific_info);

    let mut es_descriptor = vec![0, 0]; // ES_ID
    es_descriptor.push(0); // no stream dependence, no URL, no OCR stream
    write_descriptor(&mut es_descriptor, 0x04, &decoder_config_descriptor);
    write_descriptor(&mut es_descriptor, 0x06, &[0x02]); // SLConfigDescriptor, MP4 predefined

    let mut payload = vec![0, 0, 0, 0]; // FullBox version/flags
    write_descriptor(&mut payload, 0x03, &es_descriptor);

    wrap_box(b"esds", &payload)
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

/// Builds a complete `hvcC` box from a key chunk's `EncodedVideoChunkMetadata`.
///
/// Unlike AV1, `WebCodecs` gives HEVC's parameter sets to the caller
/// out-of-band: the first key chunk's metadata carries a `decoderConfig`
/// whose `description` is exactly the `HEVCDecoderConfigurationRecord`
/// payload (see `codec_config::derive_codec_string`'s `hvcC` parse), so this
/// wraps that payload with its box header rather than scanning the
/// bitstream.
fn hvcc_from_metadata(metadata: &JsValue) -> Option<Vec<u8>> {
    let metadata: &EncodedVideoChunkMetadata = metadata.dyn_ref()?;
    let decoder_config = metadata.get_decoder_config()?;
    let description = decoder_config.get_description()?;
    let bytes = js_sys::Uint8Array::new(&description).to_vec();
    Some(wrap_box(b"hvcC", &bytes))
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
