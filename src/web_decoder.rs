//! Browser `WebCodecs`-backed compressed video decoding for the `web` target.
//!
//! This bridges the browser's asynchronous `VideoDecoder`/output-callback
//! model to a simple exact-frame `get` used by [`crate::WasmVideoStream`].
//! Because decode is inherently asynchronous here, this does not implement
//! the portable, synchronous [`crate::codec::VideoDecoder`] trait; it is a
//! browser-only bridge kept out of the portable core.

use crate::codec::EncodedVideoSample;
use crate::codec_config::{box_payload, derive_codec_string};
use crate::io::MemorySource;
use crate::media::{Codec, VideoDimensions};
use crate::mp4_demux::{Mp4Demuxer, Mp4DemuxerOptions, Mp4Track};
use crate::timeline::FrameIndex;
use crate::{Error, ErrorKind, Limits, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    EncodedVideoChunk, EncodedVideoChunkInit, EncodedVideoChunkType,
    VideoDecoder as JsVideoDecoder, VideoDecoderConfig as JsVideoDecoderConfig, VideoDecoderInit,
    VideoDecoderSupport, VideoFrame as JsVideoFrame, VideoFrameCopyToOptions, VideoPixelFormat,
};

fn codec_description(codec: Codec, decoder_config: &[u8]) -> Result<&[u8]> {
    match codec {
        Codec::Hevc => box_payload(decoder_config, b"hvcC"),
        Codec::Av1 => box_payload(decoder_config, b"av1C"),
        Codec::UncompressedVideo | Codec::Aac => Err(Error::new(
            ErrorKind::Unsupported,
            "only HEVC and AV1 have a WebCodecs decoder backend",
        )),
    }
}

fn js_to_promise(value: impl JsCast) -> js_sys::Promise {
    value.unchecked_into()
}

async fn parse_video_track(source: &MemorySource, index: u32, limits: &Limits) -> Result<Mp4Track> {
    let demuxer = Mp4Demuxer::open(
        source,
        Mp4DemuxerOptions {
            limits: *limits,
            ..Mp4DemuxerOptions::default()
        },
    )
    .await?;
    demuxer
        .tracks
        .into_iter()
        .filter(|track| track.kind == crate::codec::TrackKind::Video)
        .nth(index as usize)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "no such video track"))
}

/// A lazily-configured `WebCodecs` decode session for one input video track.
pub struct WebVideoDecodeSession {
    dimensions: VideoDimensions,
    samples: Vec<EncodedVideoSample>,
    decode_position_by_presentation: HashMap<FrameIndex, usize>,
    decoder: JsVideoDecoder,
    config: JsVideoDecoderConfig,
    pending_frames: Rc<RefCell<Vec<JsVideoFrame>>>,
    decode_error: Rc<RefCell<Option<String>>>,
    next_decode_position: Option<usize>,
    limits: Limits,
    // Kept alive for the lifetime of `decoder`, which retains only the raw
    // `js_sys::Function` handles produced by `as_ref().unchecked_ref()`.
    _output_closure: Closure<dyn FnMut(JsVideoFrame)>,
    _error_closure: Closure<dyn FnMut(JsValue)>,
}

impl WebVideoDecodeSession {
    pub async fn open(bytes: &[u8], track_index: u32, limits: &Limits) -> Result<Self> {
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
        config.set_description_u8_array(&js_sys::Uint8Array::from(description));

        let support: VideoDecoderSupport =
            JsFuture::from(js_to_promise(JsVideoDecoder::is_config_supported(&config)))
                .await
                .map_err(|error| normalize_js_error(error, "querying WebCodecs decoder support"))?
                .unchecked_into();
        if !support.get_supported().unwrap_or(false) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!(
                    "this browser cannot decode {} via WebCodecs",
                    derived.codec_string
                ),
            ));
        }

        let pending_frames: Rc<RefCell<Vec<JsVideoFrame>>> = Rc::new(RefCell::new(Vec::new()));
        let decode_error: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));

        let output_frames = Rc::clone(&pending_frames);
        let output_closure = Closure::new(move |frame: JsVideoFrame| {
            output_frames.borrow_mut().push(frame);
        });
        let error_state = Rc::clone(&decode_error);
        let error_closure = Closure::new(move |error: JsValue| {
            let message = js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                .ok()
                .and_then(|value| value.as_string())
                .unwrap_or_else(|| "WebCodecs decoder reported an error".to_owned());
            *error_state.borrow_mut() = Some(message);
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

        let mut decode_position_by_presentation = HashMap::with_capacity(track.samples.len());
        let samples = track.to_encoded_video_samples(&source, limits).await?;
        for (position, sample) in samples.iter().enumerate() {
            decode_position_by_presentation.insert(sample.presentation_index, position);
        }

        Ok(Self {
            dimensions,
            samples,
            decode_position_by_presentation,
            decoder,
            config,
            pending_frames,
            decode_error,
            next_decode_position: None,
            limits: *limits,
            _output_closure: output_closure,
            _error_closure: error_closure,
        })
    }

    fn nearest_random_access(&self, position: usize) -> usize {
        (0..=position)
            .rev()
            .find(|&candidate| self.samples[candidate].random_access)
            .unwrap_or(0)
    }

    /// Decodes and returns exactly the requested presentation frame as RGBA bytes.
    pub async fn get(
        &mut self,
        presentation_index: FrameIndex,
    ) -> Result<(VideoDimensions, Vec<u8>)> {
        let target_position = *self
            .decode_position_by_presentation
            .get(&presentation_index)
            .ok_or_else(|| {
                Error::new(ErrorKind::InvalidInput, "presentation frame is not indexed")
            })?;
        let random_access_position = self.nearest_random_access(target_position);
        let can_resume = self.next_decode_position.is_some_and(|position| {
            position >= random_access_position && position <= target_position
        });
        if !can_resume {
            self.decoder.reset().map_err(|error| {
                normalize_js_error(error, "resetting the WebCodecs VideoDecoder")
            })?;
            self.pending_frames.borrow_mut().clear();
            *self.decode_error.borrow_mut() = None;
            self.decoder.configure(&self.config).map_err(|error| {
                normalize_js_error(error, "reconfiguring the WebCodecs VideoDecoder")
            })?;
            self.next_decode_position = Some(random_access_position);
        }

        let start = self.next_decode_position.expect("just set above");
        let submitted = target_position.saturating_sub(start).saturating_add(1);
        if submitted as u64 > u64::from(self.limits.max_decode_samples_per_seek) {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "exact-frame request exceeded the configured decode-work limit",
            ));
        }
        for position in start..=target_position {
            let sample = &self.samples[position];
            self.submit(sample)?;
        }
        self.next_decode_position = Some(target_position + 1);

        JsFuture::from(js_to_promise(self.decoder.flush()))
            .await
            .map_err(|error| normalize_js_error(error, "flushing the WebCodecs VideoDecoder"))?;

        if let Some(message) = self.decode_error.borrow_mut().take() {
            return Err(Error::new(ErrorKind::Codec, message));
        }

        let frames = std::mem::take(&mut *self.pending_frames.borrow_mut());
        let target_timestamp = presentation_index.0 as f64;
        let mut result = None;
        for frame in frames {
            if frame.timestamp() == target_timestamp && result.is_none() {
                result = Some(self.copy_frame_rgba(&frame).await?);
            }
            frame.close();
        }
        result.map(|rgba| (self.dimensions, rgba)).ok_or_else(|| {
            Error::new(
                ErrorKind::Internal,
                "decoder did not output the requested frame",
            )
        })
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

    async fn copy_frame_rgba(&self, frame: &JsVideoFrame) -> Result<Vec<u8>> {
        let width = frame.display_width();
        let height = frame.display_height();
        let byte_length = (width as usize)
            .checked_mul(height as usize)
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
        Ok(destination.to_vec())
    }
}

fn normalize_js_error(error: JsValue, context: &str) -> Error {
    let detail = js_sys::Reflect::get(&error, &JsValue::from_str("message"))
        .ok()
        .and_then(|value| value.as_string())
        .or_else(|| error.as_string())
        .unwrap_or_else(|| "WebCodecs operation failed".to_owned());
    Error::new(ErrorKind::Codec, format!("{context}: {detail}"))
}
