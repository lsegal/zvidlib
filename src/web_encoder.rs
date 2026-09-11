//! Browser `WebCodecs`-backed AV1 video encoding for the `web` target.
//!
//! Unlike [`crate::web_decoder`], which bridges the browser's decoder because the portable
//! [`crate::codec::VideoDecoder`] contract is fully synchronous, [`crate::codec::VideoEncoder`]
//! already returns [`crate::codec::EncoderFuture`]s. That lets this module implement the
//! portable trait directly, following the same factory (`capability()`/`create()`) plus
//! `impl VideoEncoder` shape as [`crate::av1_encoder::native_av1_video_encoder_factory`] and the
//! HEVC encoder, so [`crate::output::MediaOutput`] can drive a `WebCodecs` backend exactly as it
//! drives a native one.
//!
//! `VideoEncoder::config()` is synchronous and is read by `MediaOutput::new` before any frame is
//! encoded, so the declared [`EncoderConfig::decoder_config`] cannot wait for the browser's own
//! `EncodedVideoChunkMetadata` (which only arrives after the first encoded output). Instead this
//! backend synthesizes a minimal `av1C` box with empty `configOBUs` up front; per the AV1 Codec
//! ISO Media File Format Binding, `configOBUs` may be empty when the sequence header travels
//! in-band, which every key frame `WebCodecs` emits already does.
//!
//! Output order is assumed to match encode order (no encoder-side frame reordering), which
//! matches `latencyMode: "realtime"` below; a browser encoder that reorders AV1 output would
//! violate the MP4 muxer's contiguous-DTS requirement and surface as an error from `encode()`.

use crate::codec::{
    CodecImplementation, CodecProfile, CodecSupport, EncodedSample, EncoderConfig, EncoderFuture,
    HardwarePreference, SampleDependency, VideoEncoder, VideoEncoderConfig, VideoEncoderFactory,
    VideoEncoderFormat,
};
use crate::media::{Codec, PixelFormat};
use crate::transfer::{FrameSource, Orientation};
use crate::{Error, ErrorKind, Limits, Result};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{
    EncodedVideoChunk, EncodedVideoChunkType, LatencyMode, VideoEncoder as JsVideoEncoder,
    VideoEncoderConfig as JsVideoEncoderConfig, VideoEncoderInit, VideoFrame as JsVideoFrame,
    VideoFrameBufferInit, VideoPixelFormat,
};

/// Returns the browser `WebCodecs` AV1 video encoder factory.
pub fn web_video_encoder_factory() -> impl VideoEncoderFactory {
    WebVideoEncoderFactory
}

struct WebVideoEncoderFactory;

impl VideoEncoderFactory for WebVideoEncoderFactory {
    fn capability(&self, configuration: &VideoEncoderConfig) -> CodecSupport {
        validate_configuration(configuration)
    }

    fn create(
        &self,
        configuration: &VideoEncoderConfig,
        limits: &Limits,
    ) -> Result<Box<dyn VideoEncoder>> {
        let support = self.capability(configuration);
        if !support.is_supported() {
            return Err(capability_error(support));
        }
        WebVideoEncoder::new(configuration, limits)
            .map(|encoder| Box::new(encoder) as Box<dyn VideoEncoder>)
    }
}

/// A purely local, synchronous capability check: it validates the codec, profile, pixel format,
/// and timing this bridge can express as a `WebCodecs` configuration, but it cannot confirm the
/// current browser actually supports encoding it -- `VideoEncoder.isConfigSupported()` is
/// asynchronous and `capability()` is not. A configuration this accepts that the browser
/// declines surfaces as a [`ErrorKind::Codec`] error from the first `encode()` call instead.
fn validate_configuration(configuration: &VideoEncoderConfig) -> CodecSupport {
    if configuration.codec != Codec::Av1 {
        return CodecSupport::UnsupportedCodec;
    }
    if !matches!(
        configuration.profile,
        CodecProfile::Av1Main | CodecProfile::Av1High | CodecProfile::Av1Professional
    ) {
        return CodecSupport::UnsupportedProfile;
    }
    if configuration.hardware == HardwarePreference::Require {
        // The browser is only ever asked to *prefer* hardware acceleration; success cannot be
        // confirmed synchronously, so a caller that requires it is refused up front.
        return CodecSupport::HardwareUnavailable;
    }
    if configuration.coded_dimensions.width == 0 || configuration.coded_dimensions.height == 0 {
        return invalid_support("coded dimensions must be nonzero");
    }
    if configuration.input_format != PixelFormat::Rgba8 {
        return invalid_support(
            "the WebCodecs video encoder bridge currently requires Rgba8 input",
        );
    }
    if configuration.timescale == 0 || configuration.frame_duration == 0 {
        return invalid_support("timescale and frame duration must be nonzero");
    }
    CodecSupport::Supported {
        implementation: CodecImplementation::Software,
    }
}

fn invalid_support(reason: &str) -> CodecSupport {
    CodecSupport::InvalidConfiguration {
        reason: reason.into(),
    }
}

fn capability_error(support: CodecSupport) -> Error {
    let (kind, message) = match support {
        CodecSupport::UnsupportedCodec => (
            ErrorKind::Unsupported,
            "unsupported encoder codec".to_owned(),
        ),
        CodecSupport::UnsupportedProfile => (
            ErrorKind::Unsupported,
            "unsupported AV1 encoder profile".to_owned(),
        ),
        CodecSupport::InvalidConfiguration { reason } => {
            return Error::new(ErrorKind::InvalidInput, reason);
        }
        CodecSupport::HardwareUnavailable => (
            ErrorKind::Unsupported,
            "the WebCodecs bridge cannot guarantee hardware acceleration synchronously".to_owned(),
        ),
        CodecSupport::Supported { .. } => (
            ErrorKind::Internal,
            "encoder capability changed unexpectedly".to_owned(),
        ),
    };
    Error::new(kind, message)
}

fn codec_string(profile: CodecProfile) -> &'static str {
    match profile {
        CodecProfile::Av1High => "av01.1.08M.08",
        CodecProfile::Av1Professional => "av01.2.08M.08",
        _ => "av01.0.08M.08",
    }
}

fn seq_profile_bits(profile: CodecProfile) -> u8 {
    match profile {
        CodecProfile::Av1High => 1,
        CodecProfile::Av1Professional => 2,
        _ => 0,
    }
}

/// Synthesizes a minimal, spec-legal `av1C` box. See the module documentation for why
/// `configOBUs` is left empty rather than waiting for the browser's own encoded metadata.
fn make_av1c(profile: CodecProfile) -> Vec<u8> {
    let mut output = Vec::with_capacity(12);
    output.extend_from_slice(&12_u32.to_be_bytes());
    output.extend_from_slice(b"av1C");
    output.push(0x81);
    // seq_level_idx_0 = 8 (level 3.1): a conservative advisory value. The browser's own in-band
    // sequence header is authoritative for decode; this field only assists capability
    // negotiation by players that inspect the box without parsing the bitstream.
    output.push((seq_profile_bits(profile) << 5) | 8);
    output.push(0); // tier 0, 8-bit, not monochrome, 4:2:0 chroma (the browser's actual defaults)
    output.push(0); // no initial presentation delay
    output
}

/// One frame's presentation timing, tracked independently of the timestamp handed to the
/// browser so decode/composition timestamps stay in the caller's own tick units regardless of
/// how `WebCodecs` reports them back.
struct SubmittedFrame {
    pts: i64,
    duration: u32,
}

struct WebVideoEncoder {
    declared: EncoderConfig,
    format: VideoEncoderFormat,
    frame_duration: u32,
    timescale: u32,
    limits: Limits,
    next_index: u64,
    finished: bool,
    encoder: JsVideoEncoder,
    submitted: Rc<RefCell<VecDeque<SubmittedFrame>>>,
    pending: Rc<RefCell<VecDeque<EncodedSample>>>,
    encode_error: Rc<RefCell<Option<String>>>,
    // Kept alive for the lifetime of `encoder`, which retains only the raw `js_sys::Function`
    // handles produced by `as_ref().unchecked_ref()`.
    _output_closure: Closure<dyn FnMut(EncodedVideoChunk)>,
    _error_closure: Closure<dyn FnMut(JsValue)>,
}

impl WebVideoEncoder {
    fn new(configuration: &VideoEncoderConfig, limits: &Limits) -> Result<Self> {
        validate_limits(configuration, limits)?;

        let submitted: Rc<RefCell<VecDeque<SubmittedFrame>>> =
            Rc::new(RefCell::new(VecDeque::new()));
        let pending: Rc<RefCell<VecDeque<EncodedSample>>> = Rc::new(RefCell::new(VecDeque::new()));
        let encode_error: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));

        let output_submitted = Rc::clone(&submitted);
        let output_pending = Rc::clone(&pending);
        let output_closure = Closure::new(move |chunk: EncodedVideoChunk| {
            let Some(timing) = output_submitted.borrow_mut().pop_front() else {
                // A chunk arrived with nothing left in the submission queue: the encoder
                // reported more outputs than frames it was given, which this bridge cannot
                // place on the timeline. The next `encode()`/`finish()` call surfaces it.
                return;
            };
            let mut data = vec![0_u8; chunk.byte_length() as usize];
            if chunk.copy_to_with_u8_slice(&mut data).is_err() {
                return;
            }
            let is_sync = chunk.type_() == EncodedVideoChunkType::Key;
            output_pending.borrow_mut().push_back(EncodedSample {
                data,
                dts: timing.pts,
                pts: timing.pts,
                duration: timing.duration,
                is_sync,
                dependency: if is_sync {
                    SampleDependency::INDEPENDENT
                } else {
                    SampleDependency::DEPENDENT
                },
            });
        });

        let error_state = Rc::clone(&encode_error);
        let error_closure = Closure::new(move |error: JsValue| {
            let message = js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                .ok()
                .and_then(|value| value.as_string())
                .unwrap_or_else(|| "WebCodecs encoder reported an error".to_owned());
            *error_state.borrow_mut() = Some(message);
        });

        let init = VideoEncoderInit::new(
            error_closure.as_ref().unchecked_ref(),
            output_closure.as_ref().unchecked_ref(),
        );
        let encoder = JsVideoEncoder::new(&init)
            .map_err(|error| normalize_js_error(error, "constructing a WebCodecs VideoEncoder"))?;

        let js_config = JsVideoEncoderConfig::new(
            codec_string(configuration.profile),
            configuration.coded_dimensions.height,
            configuration.coded_dimensions.width,
        );
        js_config.set_latency_mode(LatencyMode::Realtime);
        let frame_rate =
            f64::from(configuration.timescale) / f64::from(configuration.frame_duration);
        js_config.set_framerate(frame_rate);
        encoder
            .configure(&js_config)
            .map_err(|error| normalize_js_error(error, "configuring the WebCodecs VideoEncoder"))?;

        Ok(Self {
            declared: EncoderConfig {
                codec: Codec::Av1,
                timescale: configuration.timescale,
                decoder_config: make_av1c(configuration.profile),
            },
            format: VideoEncoderFormat {
                dimensions: configuration.coded_dimensions,
                pixel_format: PixelFormat::Rgba8,
            },
            frame_duration: configuration.frame_duration,
            timescale: configuration.timescale,
            limits: *limits,
            next_index: 0,
            finished: false,
            encoder,
            submitted,
            pending,
            encode_error,
            _output_closure: output_closure,
            _error_closure: error_closure,
        })
    }

    fn take_error(&self) -> Option<Error> {
        self.encode_error
            .borrow_mut()
            .take()
            .map(|message| Error::new(ErrorKind::Codec, message))
    }

    fn drain_pending(&self) -> Vec<EncodedSample> {
        self.pending.borrow_mut().drain(..).collect()
    }

    fn encode_frame(
        &mut self,
        index: crate::FrameIndex,
        source: FrameSource<'_>,
    ) -> Result<Vec<EncodedSample>> {
        if self.finished {
            return Err(Error::new(
                ErrorKind::InvalidState,
                "cannot encode WebCodecs frames after finish",
            ));
        }
        if let Some(error) = self.take_error() {
            return Err(error);
        }
        if index.0 != self.next_index {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "WebCodecs input frame indexes must be zero-based and consecutive",
            ));
        }
        let FrameSource::Cpu(cpu) = source else {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "the WebCodecs video encoder bridge currently requires CPU frames",
            ));
        };
        let frame = cpu.frame;
        if frame.dimensions != self.format.dimensions || frame.pixel_format != PixelFormat::Rgba8 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "WebCodecs input frame does not match the configured dimensions and format",
            ));
        }
        let plane = frame.planes.first().ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                "WebCodecs Rgba8 input requires one plane",
            )
        })?;
        let width = usize::try_from(self.format.dimensions.width)
            .map_err(|_| Error::new(ErrorKind::ResourceLimit, "WebCodecs width overflow"))?;
        let height = usize::try_from(self.format.dimensions.height)
            .map_err(|_| Error::new(ErrorKind::ResourceLimit, "WebCodecs height overflow"))?;
        let row_bytes = width
            .checked_mul(4)
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "WebCodecs row size overflow"))?;
        if plane.stride < row_bytes {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "WebCodecs Rgba8 plane stride is smaller than the coded row",
            ));
        }
        let required = plane
            .stride
            .checked_mul(height)
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "WebCodecs plane size overflow"))?;
        if plane.data.len() < required {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "WebCodecs Rgba8 plane is shorter than its stride and height",
            ));
        }
        let allocation = u64::try_from(row_bytes)
            .ok()
            .and_then(|row| row.checked_mul(height as u64))
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::ResourceLimit,
                    "WebCodecs frame allocation overflow",
                )
            })?;
        if allocation > self.limits.max_allocation_bytes {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "WebCodecs frame exceeds the configured allocation limit",
            ));
        }

        let mut packed = Vec::with_capacity(row_bytes * height);
        for logical_row in 0..height {
            let stored_row = match cpu.orientation {
                Orientation::TopLeft => logical_row,
                Orientation::BottomLeft => height - logical_row - 1,
            };
            let start = stored_row * plane.stride;
            packed.extend_from_slice(&plane.data[start..start + row_bytes]);
        }

        let pts = i64::try_from(index.0)
            .ok()
            .and_then(|value| value.checked_mul(i64::from(self.frame_duration)))
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "WebCodecs timestamp overflow"))?;
        let timestamp_us = (index.0 as f64) * f64::from(self.frame_duration) * 1_000_000.0
            / f64::from(self.timescale);

        let init = VideoFrameBufferInit::new_with_f64(
            self.format.dimensions.height,
            self.format.dimensions.width,
            VideoPixelFormat::Rgba,
            timestamp_us,
        );
        let data = js_sys::Uint8Array::from(packed.as_slice());
        let video_frame = JsVideoFrame::new_with_u8_array_and_video_frame_buffer_init(&data, &init)
            .map_err(|error| normalize_js_error(error, "constructing a WebCodecs VideoFrame"))?;

        self.submitted.borrow_mut().push_back(SubmittedFrame {
            pts,
            duration: self.frame_duration,
        });
        let result = self.encoder.encode(&video_frame);
        video_frame.close();
        result.map_err(|error| {
            normalize_js_error(error, "submitting a frame to the WebCodecs VideoEncoder")
        })?;

        self.next_index = self.next_index.checked_add(1).ok_or_else(|| {
            Error::new(ErrorKind::ResourceLimit, "WebCodecs frame index overflow")
        })?;
        Ok(self.drain_pending())
    }
}

impl VideoEncoder for WebVideoEncoder {
    fn config(&self) -> &EncoderConfig {
        &self.declared
    }

    fn format(&self) -> VideoEncoderFormat {
        self.format
    }

    fn encode<'a>(
        &'a mut self,
        index: crate::FrameIndex,
        frame: FrameSource<'a>,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move { self.encode_frame(index, frame) })
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move {
            if self.finished {
                return Ok(Vec::new());
            }
            self.finished = true;
            if let Some(error) = self.take_error() {
                return Err(error);
            }
            wasm_bindgen_futures::JsFuture::from(self.encoder.flush())
                .await
                .map_err(|error| {
                    normalize_js_error(error, "flushing the WebCodecs VideoEncoder")
                })?;
            if let Some(error) = self.take_error() {
                return Err(error);
            }
            let _ = self.encoder.close();
            Ok(self.drain_pending())
        })
    }
}

fn validate_limits(configuration: &VideoEncoderConfig, limits: &Limits) -> Result<()> {
    if configuration.coded_dimensions.width > limits.max_width
        || configuration.coded_dimensions.height > limits.max_height
    {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "WebCodecs dimensions exceed the configured limits",
        ));
    }
    Ok(())
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
    use crate::{Limits, VideoDimensions};
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    fn configuration() -> VideoEncoderConfig {
        VideoEncoderConfig {
            codec: Codec::Av1,
            profile: CodecProfile::Av1Main,
            coded_dimensions: VideoDimensions::new(64, 48, &Limits::default()).unwrap(),
            input_format: PixelFormat::Rgba8,
            color_range: crate::ColorRange::Full,
            hardware: HardwarePreference::Prefer,
            timescale: 30,
            frame_duration: 1,
            configuration: Vec::new(),
        }
    }

    #[wasm_bindgen_test]
    fn capability_distinguishes_codec_profile_format_and_hardware_requirement() {
        let factory = web_video_encoder_factory();
        assert!(factory.capability(&configuration()).is_supported());

        let mut candidate = configuration();
        candidate.codec = Codec::Hevc;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::UnsupportedCodec
        );

        candidate = configuration();
        candidate.profile = CodecProfile::HevcMain;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::UnsupportedProfile
        );

        candidate = configuration();
        candidate.input_format = PixelFormat::Gray8;
        assert!(matches!(
            factory.capability(&candidate),
            CodecSupport::InvalidConfiguration { .. }
        ));

        candidate = configuration();
        candidate.hardware = HardwarePreference::Require;
        assert_eq!(
            factory.capability(&candidate),
            CodecSupport::HardwareUnavailable
        );
    }

    #[wasm_bindgen_test]
    fn av1c_declares_the_requested_seq_profile() {
        let main = make_av1c(CodecProfile::Av1Main);
        assert_eq!(&main[4..8], b"av1C");
        assert_eq!(main[9] >> 5, 0);

        let high = make_av1c(CodecProfile::Av1High);
        assert_eq!(high[9] >> 5, 1);

        let professional = make_av1c(CodecProfile::Av1Professional);
        assert_eq!(professional[9] >> 5, 2);
    }

    #[wasm_bindgen_test]
    fn codec_strings_match_the_requested_seq_profile() {
        assert_eq!(codec_string(CodecProfile::Av1Main), "av01.0.08M.08");
        assert_eq!(codec_string(CodecProfile::Av1High), "av01.1.08M.08");
        assert_eq!(codec_string(CodecProfile::Av1Professional), "av01.2.08M.08");
    }
}
