//! Opus audio: decoder configuration, native decoding and native encoding.
//!
//! Decoding and encoding run on [`opus_pure`], a pure-Rust port of libopus,
//! so both work on every target zvidlib builds for, `wasm32` included. Opus
//! always decodes at 48 kHz, which is therefore the sample clock of every Opus
//! track zvidlib reads or writes.
//!
//! The codec's own timing metadata maps onto [`crate::AudioTrackTiming`]: the
//! `OpusHead` pre-skip is the track's priming, and the end trimming an MP4
//! edit list (or a container's discard padding) records is its padding, so an
//! [`crate::AudioSampleReader`] returns exactly the samples that were encoded.

#[allow(unused_imports)]
use zvidlib_core::*;

use crate::codec::parse_audio_bit_rate;
use crate::{
    AudioBuffer, AudioDecoder, AudioDrain, AudioEncoder, AudioEncoderConfig, AudioEncoderFactory,
    AudioEncoderFormat, AudioGapless, CancellationToken, Codec, CodecImplementation, CodecProfile,
    CodecSupport, EncodedAudioSample, EncodedSample, EncoderConfig, EncoderFuture, Error,
    ErrorKind, FrameIndex, Limits, Result, SampleDependency,
};

#[doc(inline)]
pub use zvidlib_opus_syntax::{
    OPUS_PREROLL_SAMPLES, OPUS_SAMPLE_RATE, OpusHead, opus_packet_samples, opus_preroll_packets,
};

/// The longest an Opus packet can last, 120 ms at 48 kHz.
const MAX_PACKET_SAMPLES: usize = 5_760;

/// The frame the native encoder codes, 20 ms at 48 kHz.
const ENCODER_FRAME_SAMPLES: usize = 960;

/// The bit rates an Opus encoder accepts, in bits a second.
const ENCODER_BIT_RATES: std::ops::RangeInclusive<u32> = 6_000..=510_000;

enum DecoderBackend {
    /// Mapping family 0: one mono or stereo stream.
    Single(Box<opus_pure::OpusDecoder>),
    /// Mapping family 1: several streams in a Vorbis channel order.
    Multistream(opus_pure::OpusMSDecoder),
}

/// A native, pure-Rust Opus decoder producing interleaved `f32` PCM at 48 kHz.
///
/// It applies the header's output gain but not its pre-skip, which is the
/// track's priming and belongs to [`crate::AudioSampleReader`].
pub struct NativeOpusDecoder {
    backend: DecoderBackend,
    channels: u16,
    gain: f32,
    scratch: Vec<f32>,
    limits: Limits,
}

impl NativeOpusDecoder {
    pub fn new(head: &OpusHead, limits: Limits) -> Result<Self> {
        let channels = u16::from(head.channels);
        if channels > limits.max_audio_channels {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "Opus channel count is outside configured limits",
            ));
        }
        let rate = OPUS_SAMPLE_RATE as i32;
        let backend = match head.channel_mapping_family {
            0 => DecoderBackend::Single(Box::new(
                opus_pure::OpusDecoder::new(rate, usize::from(head.channels))
                    .map_err(opus_error)?,
            )),
            1 => {
                let decoder = opus_pure::OpusMSDecoder::new(rate, usize::from(head.channels), 1)
                    .map_err(|error| {
                        unsupported(format!("the Opus channel layout is unsupported: {error}"))
                    })?;
                let layout =
                    opus_pure::multistream::ChannelLayout::surround(usize::from(head.channels), 1)
                        .map_err(opus_error)?;
                if layout.nb_streams != usize::from(head.stream_count)
                    || layout.nb_coupled_streams != usize::from(head.coupled_count)
                    || layout.mapping != head.channel_mapping
                {
                    return Err(unsupported(
                        "the native Opus decoder supports the standard mapping family 1 layouts",
                    ));
                }
                DecoderBackend::Multistream(decoder)
            }
            family => {
                return Err(unsupported(format!(
                    "Opus channel mapping family {family} is unsupported"
                )));
            }
        };
        Ok(Self {
            backend,
            channels,
            gain: head.output_gain_factor(),
            scratch: vec![0.0; MAX_PACKET_SAMPLES * usize::from(channels)],
            limits,
        })
    }
}

impl AudioDecoder for NativeOpusDecoder {
    fn decode(
        &mut self,
        sample: &EncodedAudioSample,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        if cancellation.is_canceled() {
            return Err(Error::new(ErrorKind::Canceled, "Opus decode canceled"));
        }
        let decoded = match &mut self.backend {
            DecoderBackend::Single(decoder) => {
                decoder.decode(&sample.data, MAX_PACKET_SAMPLES, &mut self.scratch)
            }
            DecoderBackend::Multistream(decoder) => {
                decoder.decode(&sample.data, MAX_PACKET_SAMPLES, &mut self.scratch)
            }
        }
        .map_err(|error| codec(format!("malformed Opus packet: {error}")))?;
        if decoded as u64 != sample.decoded_range.len() {
            return Err(codec(format!(
                "Opus decoder produced {decoded} frames for an indexed {}-frame interval",
                sample.decoded_range.len()
            )));
        }
        let samples = self.scratch[..decoded * usize::from(self.channels)]
            .iter()
            .map(|value| value * self.gain)
            .collect();
        AudioBuffer::new(
            sample.decoded_range,
            OPUS_SAMPLE_RATE,
            self.channels,
            samples,
            &self.limits,
        )
    }

    fn reset(&mut self) -> Result<()> {
        match &mut self.backend {
            DecoderBackend::Single(decoder) => decoder.reset_state(),
            DecoderBackend::Multistream(decoder) => decoder.reset_state(),
        }
        .map_err(opus_error)
    }
}

/// Returns the native Opus encoder.
///
/// It is pure Rust ([`opus_pure`], a port of libopus) and so available on
/// every target, reporting [`CodecImplementation::Software`]. It encodes 48
/// kHz mono or stereo PCM, the rate Opus decodes at, so a track written with it
/// reads back on the same sample clock; resample other rates first. It codes
/// 20 ms frames in libopus's general-audio mode, and its delay becomes the
/// stream's pre-skip and the drained [`AudioGapless::priming`].
///
/// [`AudioEncoderConfig::configuration`] is either empty, leaving the bit rate
/// to the encoder (64 kb/s), or four big-endian bytes naming a target bit
/// rate from 6 to 510 kb/s, the same form the native AAC encoder takes.
pub fn native_opus_audio_encoder_factory() -> impl AudioEncoderFactory {
    OpusEncoderFactory
}

#[derive(Clone, Copy, Debug)]
struct OpusEncoderFactory;

impl AudioEncoderFactory for OpusEncoderFactory {
    fn capability(&self, configuration: &AudioEncoderConfig) -> CodecSupport {
        if configuration.codec != Codec::Opus {
            return CodecSupport::UnsupportedCodec;
        }
        if configuration.profile != CodecProfile::Opus {
            return CodecSupport::UnsupportedProfile;
        }
        let bit_rate = parse_audio_bit_rate(&configuration.configuration);
        if bit_rate.is_none()
            || bit_rate
                .flatten()
                .is_some_and(|rate| !ENCODER_BIT_RATES.contains(&rate))
        {
            return CodecSupport::InvalidConfiguration {
                reason: "native Opus encoder configuration is either empty or four big-endian \
                         bytes giving a target bit rate from 6000 to 510000 bits a second"
                    .into(),
            };
        }
        if configuration.sample_rate != OPUS_SAMPLE_RATE {
            return CodecSupport::InvalidConfiguration {
                reason: format!(
                    "the native Opus encoder takes 48000 Hz input, not {} Hz",
                    configuration.sample_rate
                ),
            };
        }
        if configuration.channels != 1 && configuration.channels != 2 {
            return CodecSupport::InvalidConfiguration {
                reason: "native Opus encoder supports mono or stereo input".into(),
            };
        }
        if configuration.timescale != OPUS_SAMPLE_RATE {
            return CodecSupport::InvalidConfiguration {
                reason: "native Opus encoder timescale must be 48000".into(),
            };
        }
        CodecSupport::Supported {
            implementation: CodecImplementation::Software,
        }
    }

    fn create(
        &self,
        configuration: &AudioEncoderConfig,
        limits: &Limits,
    ) -> Result<Box<dyn AudioEncoder>> {
        match self.capability(configuration) {
            CodecSupport::Supported { .. } => {}
            CodecSupport::InvalidConfiguration { reason } => {
                return Err(Error::new(ErrorKind::InvalidInput, reason));
            }
            _ => {
                return Err(unsupported(
                    "native Opus encoder requires the Opus codec and profile",
                ));
            }
        }
        let channels = configuration.channels;
        let mut encoder = opus_pure::OpusEncoder::new(
            OPUS_SAMPLE_RATE as i32,
            usize::from(channels),
            opus_pure::Application::Audio,
        )
        .map_err(opus_error)?;
        if let Some(bit_rate) = parse_audio_bit_rate(&configuration.configuration).flatten() {
            encoder.bitrate_bps = i32::try_from(bit_rate).expect("checked by capability");
        }
        let pre_skip = u16::try_from(encoder.lookahead())
            .map_err(|_| internal("the Opus encoder delay exceeds a pre-skip"))?;
        let head = OpusHead::new(
            u8::try_from(channels).expect("checked by capability"),
            pre_skip,
            OPUS_SAMPLE_RATE,
        )?;
        Ok(Box::new(NativeOpusEncoder {
            encoder,
            config: EncoderConfig {
                codec: Codec::Opus,
                timescale: OPUS_SAMPLE_RATE,
                decoder_config: head.to_dops(),
            },
            format: AudioEncoderFormat {
                sample_rate: OPUS_SAMPLE_RATE,
                channels,
            },
            pending: Vec::new(),
            packet: vec![0; opus_pure::MAX_PACKET_BYTES],
            pre_skip: u32::from(pre_skip),
            input_frames: 0,
            encoded_frames: 0,
            limits: *limits,
            finished: false,
        }))
    }
}

struct NativeOpusEncoder {
    encoder: opus_pure::OpusEncoder,
    config: EncoderConfig,
    format: AudioEncoderFormat,
    /// Interleaved input not yet coded because it does not fill a frame.
    pending: Vec<f32>,
    packet: Vec<u8>,
    pre_skip: u32,
    input_frames: u64,
    encoded_frames: u64,
    limits: Limits,
    finished: bool,
}

impl NativeOpusEncoder {
    fn frame_values(&self) -> usize {
        ENCODER_FRAME_SAMPLES * usize::from(self.format.channels)
    }

    /// Codes every whole frame in `pending`.
    fn encode_pending(&mut self, out: &mut Vec<EncodedSample>) -> Result<()> {
        let frame_values = self.frame_values();
        let whole = self.pending.len() / frame_values * frame_values;
        for start in (0..whole).step_by(frame_values) {
            let length = self
                .encoder
                .encode(
                    &self.pending[start..start + frame_values],
                    ENCODER_FRAME_SAMPLES,
                    &mut self.packet,
                )
                .map_err(|error| codec(format!("the Opus encoder failed: {error}")))?;
            let dts = i64::try_from(self.encoded_frames)
                .map_err(|_| Error::new(ErrorKind::ResourceLimit, "Opus timestamp overflow"))?;
            out.push(EncodedSample {
                data: self.packet[..length].to_vec(),
                dts,
                pts: dts,
                duration: ENCODER_FRAME_SAMPLES as u32,
                is_sync: true,
                dependency: SampleDependency::INDEPENDENT,
            });
            self.encoded_frames += ENCODER_FRAME_SAMPLES as u64;
        }
        self.pending.drain(..whole);
        Ok(())
    }
}

impl AudioEncoder for NativeOpusEncoder {
    fn config(&self) -> &EncoderConfig {
        &self.config
    }

    fn format(&self) -> AudioEncoderFormat {
        self.format
    }

    fn encode<'a>(
        &'a mut self,
        _index: FrameIndex,
        buffer: AudioBuffer,
    ) -> EncoderFuture<'a, Vec<EncodedSample>> {
        Box::pin(async move {
            if self.finished {
                return Err(Error::new(
                    ErrorKind::InvalidState,
                    "the Opus encoder has already finished",
                ));
            }
            if buffer.sample_rate != self.format.sample_rate
                || buffer.channels != self.format.channels
            {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "audio buffer format does not match the Opus encoder",
                ));
            }
            let pending_bytes = (self.pending.len() + buffer.samples.len()) as u64
                * std::mem::size_of::<f32>() as u64;
            if pending_bytes > self.limits.max_allocation_bytes {
                return Err(Error::new(
                    ErrorKind::ResourceLimit,
                    "pending Opus input exceeds the allocation limit",
                ));
            }
            self.input_frames += buffer.range.len();
            self.pending.extend_from_slice(&buffer.samples);
            let mut samples = Vec::new();
            self.encode_pending(&mut samples)?;
            Ok(samples)
        })
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, AudioDrain> {
        Box::pin(async move {
            if self.finished {
                return Ok(AudioDrain::default());
            }
            self.finished = true;
            // The encoder holds its delay's worth of input back, so silence
            // is coded after the input until the emitted frames cover the
            // pre-skip and every input frame.
            let needed = self.input_frames + u64::from(self.pre_skip);
            let frame = ENCODER_FRAME_SAMPLES as u64;
            let total_frames = needed.div_ceil(frame).max(1) * frame;
            let silence = (total_frames - self.encoded_frames) as usize
                * usize::from(self.format.channels)
                - self.pending.len();
            self.pending.resize(self.pending.len() + silence, 0.0);
            let mut samples = Vec::new();
            self.encode_pending(&mut samples)?;
            let padding = u32::try_from(self.encoded_frames - needed)
                .map_err(|_| internal("Opus padding exceeds u32"))?;
            Ok(AudioDrain {
                samples,
                gapless: AudioGapless {
                    priming: self.pre_skip,
                    padding,
                },
            })
        })
    }
}

fn opus_error(error: opus_pure::Error) -> Error {
    codec(format!("Opus: {error}"))
}

fn codec(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Codec, message)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

fn internal(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Internal, message)
}
