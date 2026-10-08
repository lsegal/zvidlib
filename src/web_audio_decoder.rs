//! Browser audio decoding for the `web` target: exact sample reads over an
//! input MP4's AAC or Opus track or an input WebM's Opus or Vorbis track.
//!
//! Reads go through the same [`AudioSampleReader`] native callers use, so
//! priming, padding, edits and preroll are handled identically. What decodes
//! the packets it asks for is the browser's `WebCodecs` `AudioDecoder` when it
//! supports the track, and the crate's own software decoder otherwise - the
//! same arrangement [`crate::web_decoder`] has for video. Opus and Vorbis have
//! software decoders on every target; AAC has none in the browser build, so an
//! AAC track the browser cannot decode is unsupported.
//!
//! A `WebCodecs` decoder may reset its state when it is flushed, so each read
//! decodes from a freshly configured decoder, preroll included, rather than
//! continuing the last one. Its output is checked against the packets'
//! indexed intervals, and a browser whose output does not line up - one that
//! trims samples itself, say - is not used again: the session falls back to
//! software for that read and every one after it.

use crate::audio::{AudioDecoder, AudioSampleReader, EncodedAudioSample};
use crate::codec::CancellationToken;
use crate::io::MemorySource;
use crate::media::{AudioBuffer, Codec};
use crate::mp4_demux::Mp4Track;
use crate::timeline::SampleRange;
use crate::web_decoder::{js_to_promise, normalize_js_error};
use crate::{Error, ErrorKind, Limits, Result};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    AudioData, AudioDataCopyToOptions, AudioDecoder as JsAudioDecoder,
    AudioDecoderConfig as JsAudioDecoderConfig, AudioDecoderInit, AudioDecoderSupport,
    AudioSampleFormat, EncodedAudioChunk, EncodedAudioChunkInit, EncodedAudioChunkType,
};

/// How many packets an AAC read decodes ahead of the first one it needs. An
/// AAC frame overlaps the one before it, so one is enough; the second covers
/// the decoder's own start-up.
pub(crate) const AAC_PREROLL_PACKETS: usize = 2;

/// The `WebCodecs` decoder configuration for an input audio track: its codec
/// string, sample rate and channel count, and the `description` the codec
/// needs, if any.
#[derive(Clone, Debug, PartialEq)]
pub struct WebAudioDecoderConfig {
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub description: Option<Vec<u8>>,
}

impl WebAudioDecoderConfig {
    /// The configuration for `track`, an AAC, Opus or Vorbis track.
    pub fn for_track(track: &Mp4Track) -> Result<Self> {
        match track.codec {
            Codec::Aac => {
                let config = track.aac_config()?;
                Ok(Self {
                    codec: format!("mp4a.40.{}", config.audio_object_type),
                    sample_rate: config.sample_rate,
                    channels: config.channels,
                    description: Some(config.audio_specific_config),
                })
            }
            Codec::Opus => {
                let head = track.opus_config()?;
                // A mono or stereo stream needs no description, and leaving
                // it out keeps the browser from trimming the pre-skip itself:
                // that is the reader's job, from the track's edit list. A
                // surround stream cannot be described without one.
                let description =
                    (head.channel_mapping_family != 0).then(|| head.to_identification_header());
                Ok(Self {
                    codec: "opus".to_owned(),
                    sample_rate: crate::OPUS_SAMPLE_RATE,
                    channels: u16::from(head.channels),
                    description,
                })
            }
            // WebCodecs takes Vorbis's three headers Xiph-laced, exactly as a
            // WebM track's CodecPrivate stores them.
            Codec::Vorbis => {
                let config = track.vorbis_config()?;
                Ok(Self {
                    codec: "vorbis".to_owned(),
                    sample_rate: config.sample_rate,
                    channels: u16::from(config.channels),
                    description: Some(config.to_codec_private()),
                })
            }
            _ => Err(Error::new(
                ErrorKind::Unsupported,
                "browser audio decoding supports AAC, Opus and Vorbis tracks",
            )),
        }
    }

    pub(crate) fn to_js(&self) -> JsAudioDecoderConfig {
        let config =
            JsAudioDecoderConfig::new(&self.codec, u32::from(self.channels), self.sample_rate);
        if let Some(description) = &self.description {
            config.set_description_u8_array(&js_sys::Uint8Array::from(description.as_slice()));
        }
        config
    }
}

/// Which decoders [`WebAudioDecodeSession::open_with`] may choose between.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BackendChoice {
    /// `WebCodecs` when the browser supports the config, software otherwise.
    Automatic,
    /// Software whatever the browser supports, so a test covers the fallback
    /// even in a browser that decodes the codec itself.
    #[cfg(all(test, feature = "all"))]
    SoftwareOnly,
}

/// An exact-sample read session for one input audio track.
pub struct WebAudioDecodeSession {
    reader: AudioSampleReader<Box<dyn AudioDecoder>>,
    /// Whether the reader's own decoder is a real software decoder rather
    /// than [`NoSoftwareDecoder`].
    has_software: bool,
    webcodecs: Option<WebCodecsAudioDecoder>,
}

impl WebAudioDecodeSession {
    pub async fn open(bytes: &[u8], track_index: u32, limits: &Limits) -> Result<Self> {
        Self::open_with(bytes, track_index, limits, BackendChoice::Automatic).await
    }

    async fn open_with(
        bytes: &[u8],
        track_index: u32,
        limits: &Limits,
        choice: BackendChoice,
    ) -> Result<Self> {
        let source = MemorySource::new(bytes.to_vec());
        let (track, timing) =
            crate::container::open_audio_track(&source, track_index as usize, limits).await?;
        let config = WebAudioDecoderConfig::for_track(&track)?;
        let packets = track.to_encoded_audio_samples(&source, limits).await?;
        let (software, preroll): (Option<Box<dyn AudioDecoder>>, usize) = match track.codec {
            #[cfg(feature = "opus-decoder")]
            Codec::Opus => (
                Some(Box::new(crate::NativeOpusDecoder::new(
                    &track.opus_config()?,
                    *limits,
                )?)),
                crate::opus_preroll_packets(&packets),
            ),
            // A build without the codec's feature has only WebCodecs to decode
            // it with.
            #[cfg(not(feature = "opus-decoder"))]
            Codec::Opus => (None, crate::opus_preroll_packets(&packets)),
            #[cfg(feature = "vorbis-decoder")]
            Codec::Vorbis => (
                Some(Box::new(crate::NativeVorbisDecoder::new(
                    &track.vorbis_config()?,
                    *limits,
                )?)),
                crate::VORBIS_PREROLL_PACKETS,
            ),
            #[cfg(not(feature = "vorbis-decoder"))]
            Codec::Vorbis => (None, crate::VORBIS_PREROLL_PACKETS),
            _ => (None, AAC_PREROLL_PACKETS),
        };

        let webcodecs_supported = match choice {
            BackendChoice::Automatic => {
                let support: AudioDecoderSupport = JsFuture::from(js_to_promise(
                    JsAudioDecoder::is_config_supported(&config.to_js()),
                ))
                .await
                .map_err(|error| {
                    normalize_js_error(error, "querying WebCodecs audio decoder support")
                })?
                .unchecked_into();
                support.get_supported().unwrap_or(false)
            }
            #[cfg(all(test, feature = "all"))]
            BackendChoice::SoftwareOnly => false,
        };
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
        let webcodecs = if webcodecs_supported {
            Some(WebCodecsAudioDecoder::open(config.clone(), *limits)?)
        } else {
            None
        };
        let has_software = software.is_some();
        let reader = AudioSampleReader::new(
            software.unwrap_or_else(|| Box::new(NoSoftwareDecoder)),
            packets,
            config.sample_rate,
            config.channels,
            timing,
            preroll,
            *limits,
        )?;
        Ok(Self {
            reader,
            has_software,
            webcodecs,
        })
    }

    /// The number of samples the track presents, after priming, padding and
    /// edits.
    pub fn presentation_length(&self) -> u64 {
        self.reader.presentation_length()
    }

    /// Whether reads go through the software decoder because `WebCodecs`
    /// cannot decode the track, or decoded it wrongly.
    #[cfg(all(test, feature = "all"))]
    pub(crate) fn is_software(&self) -> bool {
        self.webcodecs.is_none()
    }

    /// Drops the `WebCodecs` decoder, so reads go through software.
    #[cfg(all(test, feature = "all"))]
    pub(crate) fn reset_to_software(&mut self) {
        self.webcodecs = None;
    }

    /// Returns exactly the requested half-open range of presentation samples.
    pub async fn get_range(
        &mut self,
        range: SampleRange,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        if let Some(webcodecs) = self.webcodecs.as_mut() {
            // Every read starts a fresh decode, preroll included; see the
            // module documentation.
            self.reader.reset()?;
            let result = self
                .reader
                .get_range_with(range, cancellation, async |_, packets| {
                    webcodecs.decode(packets).await
                })
                .await;
            match result {
                Ok(buffer) => return Ok(buffer),
                Err(error) if error.kind() == ErrorKind::Canceled => return Err(error),
                Err(error) if !self.has_software => return Err(error),
                Err(_) => {
                    // The browser's decode does not line up with the track.
                    // The software decoder does, so it takes over for good.
                    self.webcodecs = None;
                }
            }
        }
        self.reader.reset()?;
        self.reader.get_range(range, cancellation)
    }
}

/// The reader's decoder when the browser build has no software decoder for a
/// track's codec, so every packet is decoded through `WebCodecs`.
pub(crate) struct NoSoftwareDecoder;

impl AudioDecoder for NoSoftwareDecoder {
    fn decode(&mut self, _: &EncodedAudioSample, _: &CancellationToken) -> Result<AudioBuffer> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "the browser build has no software decoder for this audio codec",
        ))
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A `WebCodecs` `AudioDecoder` reconfigured for every batch of packets.
pub(crate) struct WebCodecsAudioDecoder {
    config: WebAudioDecoderConfig,
    decoder: JsAudioDecoder,
    outputs: Rc<RefCell<Vec<AudioData>>>,
    error: Rc<RefCell<Option<String>>>,
    limits: Limits,
    // Kept alive for the lifetime of `decoder`, which retains only the raw
    // `js_sys::Function` handles produced by `as_ref().unchecked_ref()`.
    _output_closure: Closure<dyn FnMut(AudioData)>,
    _error_closure: Closure<dyn FnMut(JsValue)>,
}

impl WebCodecsAudioDecoder {
    pub(crate) fn open(config: WebAudioDecoderConfig, limits: Limits) -> Result<Self> {
        let outputs: Rc<RefCell<Vec<AudioData>>> = Rc::new(RefCell::new(Vec::new()));
        let error: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
        let output_sink = Rc::clone(&outputs);
        let output_closure = Closure::new(move |data: AudioData| {
            output_sink.borrow_mut().push(data);
        });
        let error_state = Rc::clone(&error);
        let error_closure = Closure::new(move |value: JsValue| {
            let message = js_sys::Reflect::get(&value, &JsValue::from_str("message"))
                .ok()
                .and_then(|value| value.as_string())
                .unwrap_or_else(|| "WebCodecs audio decoder reported an error".to_owned());
            *error_state.borrow_mut() = Some(message);
        });
        let init = AudioDecoderInit::new(
            error_closure.as_ref().unchecked_ref(),
            output_closure.as_ref().unchecked_ref(),
        );
        let decoder = JsAudioDecoder::new(&init)
            .map_err(|error| normalize_js_error(error, "constructing a WebCodecs AudioDecoder"))?;
        Ok(Self {
            config,
            decoder,
            outputs,
            error,
            limits,
            _output_closure: output_closure,
            _error_closure: error_closure,
        })
    }

    /// Decodes `packets` in order from a freshly configured decoder, returning
    /// one buffer per packet.
    pub(crate) async fn decode(&mut self, packets: &[EncodedAudioSample]) -> Result<Vec<AudioBuffer>> {
        let rate = f64::from(self.config.sample_rate);
        self.close_outputs();
        *self.error.borrow_mut() = None;
        // `reset()` returns a configured decoder to the unconfigured state;
        // on one that was never configured it is a no-op.
        let _ = self.decoder.reset();
        self.decoder
            .configure(&self.config.to_js())
            .map_err(|error| normalize_js_error(error, "configuring the WebCodecs AudioDecoder"))?;
        for packet in packets {
            let data = js_sys::Uint8Array::from(packet.data.as_slice());
            let init =
                EncodedAudioChunkInit::new_with_u8_array(&data, 0, EncodedAudioChunkType::Key);
            init.set_timestamp_f64(packet.decoded_range.start as f64 * 1_000_000.0 / rate);
            init.set_duration_f64(packet.decoded_range.len() as f64 * 1_000_000.0 / rate);
            let chunk = EncodedAudioChunk::new(&init)
                .map_err(|error| normalize_js_error(error, "constructing an audio chunk"))?;
            self.decoder
                .decode(&chunk)
                .map_err(|error| normalize_js_error(error, "decoding an audio chunk"))?;
        }
        let flushed = JsFuture::from(js_to_promise(self.decoder.flush())).await;
        let error = self.error.borrow_mut().take();
        if let Some(message) = error {
            self.close_outputs();
            return Err(Error::new(ErrorKind::Codec, message));
        }
        flushed
            .map_err(|error| normalize_js_error(error, "flushing the WebCodecs AudioDecoder"))?;
        let samples = self.take_samples()?;
        split_into_packets(
            samples,
            packets,
            self.config.sample_rate,
            self.config.channels,
            &self.limits,
        )
    }

    /// Every decoded sample so far, interleaved, in output order.
    fn take_samples(&mut self) -> Result<Vec<f32>> {
        let outputs = std::mem::take(&mut *self.outputs.borrow_mut());
        let channels = usize::from(self.config.channels);
        let mut interleaved = Vec::new();
        let mut result = Ok(());
        for data in &outputs {
            if result.is_ok() {
                result =
                    append_interleaved(data, channels, self.config.sample_rate, &mut interleaved);
            }
            data.close();
        }
        result.map(|()| interleaved)
    }

    fn close_outputs(&self) {
        for data in self.outputs.borrow_mut().drain(..) {
            data.close();
        }
    }
}

impl Drop for WebCodecsAudioDecoder {
    fn drop(&mut self) {
        self.close_outputs();
        let _ = self.decoder.close();
    }
}

/// Appends one `AudioData`'s frames to `out`, interleaved, as `f32`.
fn append_interleaved(
    data: &AudioData,
    channels: usize,
    sample_rate: u32,
    out: &mut Vec<f32>,
) -> Result<()> {
    if data.number_of_channels() as usize != channels
        || (f64::from(data.sample_rate()) - f64::from(sample_rate)).abs() > 0.5
    {
        return Err(Error::new(
            ErrorKind::Codec,
            "WebCodecs decoded audio in a different format than the track's",
        ));
    }
    let frames = data.number_of_frames() as usize;
    let start = out.len();
    out.resize(start + frames * channels, 0.0);
    let plane = js_sys::Float32Array::new_with_length(frames as u32);
    for channel in 0..channels {
        let options = AudioDataCopyToOptions::new(channel as u32);
        options.set_format(AudioSampleFormat::F32Planar);
        data.copy_to_with_buffer_source(&plane, &options)
            .map_err(|error| normalize_js_error(error, "copying decoded audio"))?;
        for (frame, value) in plane.to_vec().into_iter().enumerate() {
            out[start + frame * channels + channel] = value;
        }
    }
    Ok(())
}

/// Splits a batch's decoded samples into one buffer per packet by the
/// packets' indexed intervals, which they must fill exactly.
///
/// The one exception is the batch's first packet, which a Vorbis decoder
/// configured afresh decodes to nothing, having no block before it to overlap.
/// It is the read's preroll, which is never kept, so it is given silence.
fn split_into_packets(
    mut samples: Vec<f32>,
    packets: &[EncodedAudioSample],
    sample_rate: u32,
    channels: u16,
    limits: &Limits,
) -> Result<Vec<AudioBuffer>> {
    let expected: u64 = packets
        .iter()
        .map(|packet| packet.decoded_range.len())
        .sum();
    let channel_count = usize::from(channels);
    let first = packets
        .first()
        .map_or(0, |packet| packet.decoded_range.len());
    if packets.len() > 1
        && first > 0
        && samples.len() as u64 == (expected - first) * u64::from(channels)
    {
        let silence = first as usize * channel_count;
        samples.splice(0..0, std::iter::repeat_n(0.0, silence));
    }
    if samples.len() as u64 != expected * u64::from(channels) {
        return Err(Error::new(
            ErrorKind::Codec,
            format!(
                "WebCodecs decoded {} frames where the packets index {expected}",
                samples.len() / channel_count
            ),
        ));
    }
    let mut buffers = Vec::with_capacity(packets.len());
    let mut at = 0;
    for packet in packets {
        let count = packet.decoded_range.len() as usize * channel_count;
        buffers.push(AudioBuffer::new(
            packet.decoded_range,
            sample_rate,
            channels,
            samples[at..at + count].to_vec(),
            limits,
        )?);
        at += count;
    }
    Ok(buffers)
}

// The browser tests round-trip media through every native codec, so they
// build with the whole codec matrix.
#[cfg(all(test, feature = "all"))]
pub(crate) mod tests {
    use super::*;
    use crate::io::MemorySink;
    use crate::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
    use crate::{AudioEncoderConfig, CodecProfile, FrameIndex};
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    /// A second of 48 kHz stereo encoded by the native Opus encoder, muxed
    /// into an MP4 with its pre-skip and end trim, and the PCM it was made
    /// from.
    pub(crate) async fn opus_mp4() -> (Vec<u8>, Vec<f32>) {
        let frames = 48_000;
        let input: Vec<f32> = (0..frames)
            .flat_map(|i| {
                let t = i as f32 / 48_000.0;
                [
                    0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin(),
                    0.3 * (2.0 * std::f32::consts::PI * 660.0 * t).sin(),
                ]
            })
            .collect();
        let factory = crate::native_opus_audio_encoder_factory();
        let mut encoder = crate::AudioEncoderFactory::create(
            &factory,
            &AudioEncoderConfig {
                codec: Codec::Opus,
                profile: CodecProfile::Opus,
                sample_rate: 48_000,
                channels: 2,
                timescale: 48_000,
                configuration: 128_000_u32.to_be_bytes().to_vec(),
            },
            &Limits::default(),
        )
        .unwrap();
        let buffer = AudioBuffer::new(
            SampleRange::new(0, frames).unwrap(),
            48_000,
            2,
            input.clone(),
            &Limits::default(),
        )
        .unwrap();
        let mut samples = encoder.encode(FrameIndex(0), buffer).await.unwrap();
        let drain = encoder.finish().await.unwrap();
        samples.extend(drain.samples);
        let mut muxer = Mp4Muxer::new(
            MemorySink::new(),
            vec![Mp4TrackConfig {
                encoder: encoder.config().clone(),
                format: Mp4TrackFormat::Audio { channels: 2 },
            }],
            100_000,
        )
        .await
        .unwrap();
        for sample in samples {
            muxer.write_sample(0, sample).await.unwrap();
        }
        muxer.set_audio_gapless(0, drain.gapless).unwrap();
        (muxer.finish().await.unwrap().into_inner(), input)
    }

    async fn read(session: &mut WebAudioDecodeSession, start: u64, end: u64) -> Vec<f32> {
        session
            .get_range(
                SampleRange::new(start, end).unwrap(),
                &CancellationToken::new(),
            )
            .await
            .unwrap()
            .samples
    }

    #[wasm_bindgen_test]
    async fn software_and_webcodecs_reads_agree_on_an_opus_track() {
        let (bytes, input) = opus_mp4().await;
        let mut software = WebAudioDecodeSession::open_with(
            &bytes,
            0,
            &Limits::default(),
            BackendChoice::SoftwareOnly,
        )
        .await
        .unwrap();
        assert!(software.is_software());
        assert_eq!(software.presentation_length(), 48_000);
        let all = read(&mut software, 0, 48_000).await;
        assert_eq!(all.len(), input.len());
        let noise: f32 = all.iter().zip(&input).map(|(a, b)| (a - b).powi(2)).sum();
        let signal: f32 = input.iter().map(|v| v * v).sum();
        assert!(
            10.0 * (signal / noise).log10() > 15.0,
            "the decode is not the input"
        );

        let mut session = WebAudioDecodeSession::open(&bytes, 0, &Limits::default())
            .await
            .unwrap();
        for (start, end) in [
            (12_345, 13_000),
            (0, 960),
            (47_000, 48_000),
            (20_000, 20_001),
        ] {
            let got = read(&mut session, start, end).await;
            let expected = &all[start as usize * 2..end as usize * 2];
            assert_eq!(got.len(), expected.len());
            // libopus and opus-pure agree to within a few 16-bit steps, and
            // the session may be either.
            let largest = got
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(largest < 1e-3, "[{start}, {end}) differs by {largest}");
        }
    }

    #[wasm_bindgen_test]
    fn a_decode_that_does_not_fill_its_packets_is_refused() {
        let packets: Vec<EncodedAudioSample> = (0..2)
            .map(|index| EncodedAudioSample {
                decoded_range: SampleRange::new(index * 960, index * 960 + 960).unwrap(),
                data: vec![0],
            })
            .collect();
        let limits = Limits::default();
        assert_eq!(
            split_into_packets(vec![0.0; 1_920], &packets, 48_000, 1, &limits)
                .unwrap()
                .len(),
            2
        );
        // A browser that trimmed a pre-skip of its own decodes fewer frames
        // than the packets index.
        assert!(split_into_packets(vec![0.0; 1_608], &packets, 48_000, 1, &limits).is_err());
    }
}
