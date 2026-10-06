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

use crate::codec::parse_audio_bit_rate;
use crate::{
    AudioBuffer, AudioDecoder, AudioDrain, AudioEncoder, AudioEncoderConfig, AudioEncoderFactory,
    AudioEncoderFormat, AudioGapless, CancellationToken, Codec, CodecImplementation, CodecProfile,
    CodecSupport, EncodedAudioSample, EncodedSample, EncoderConfig, EncoderFuture, Error,
    ErrorKind, FrameIndex, Limits, Result, SampleDependency,
};

/// The sample rate Opus decodes at, and the sample clock of an Opus track.
pub const OPUS_SAMPLE_RATE: u32 = 48_000;

/// How much audio to decode ahead of a seek target before its samples are
/// trusted, 320 ms.
///
/// RFC 7845 section 4.6 asks for at least 80 ms, by which point the decoder's
/// prediction state has converged far enough to play. Its output there still
/// differs from a continuous decode's by 12 to 30 dB, though, and each further
/// 80 ms takes a good deal more off: by 320 ms a read after a seek matches a
/// read that walked there to within `f32` rounding in CELT and within a few
/// steps of 16-bit PCM in SILK, whose filters run on integers. Opus decodes so
/// much faster than real time that the extra preroll costs a few milliseconds
/// a seek.
pub const OPUS_PREROLL_SAMPLES: u32 = 15_360;

/// The longest an Opus packet can last, 120 ms at 48 kHz.
const MAX_PACKET_SAMPLES: usize = 5_760;

/// The frame the native encoder codes, 20 ms at 48 kHz.
const ENCODER_FRAME_SAMPLES: usize = 960;

/// The bit rates an Opus encoder accepts, in bits a second.
const ENCODER_BIT_RATES: std::ops::RangeInclusive<u32> = 6_000..=510_000;

/// The Opus identification header: the decoder configuration of an Opus
/// stream (RFC 7845 section 5.1).
///
/// MP4 carries it as the fields of a `dOps` box ([`Self::from_dops`],
/// [`Self::to_dops`]); Ogg, Matroska and WebM `CodecPrivate`, and a `WebCodecs`
/// `AudioDecoderConfig.description` carry the `OpusHead` packet itself
/// ([`Self::from_identification_header`],
/// [`Self::to_identification_header`]).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpusHead {
    /// The number of output channels.
    pub channels: u8,
    /// Samples at 48 kHz to discard from the start of the decoded stream: the
    /// encoder's delay.
    pub pre_skip: u16,
    /// The sample rate of the audio before it was encoded, for information
    /// only; Opus decodes at [`OPUS_SAMPLE_RATE`] regardless. Zero when
    /// unknown.
    pub input_sample_rate: u32,
    /// Gain to apply to the decoded output, in Q7.8 decibels.
    pub output_gain: i16,
    /// The channel mapping family: 0 for mono or stereo in one stream, 1 for
    /// the Vorbis channel orders of up to eight channels in several streams.
    pub channel_mapping_family: u8,
    /// The number of Opus streams in each packet.
    pub stream_count: u8,
    /// How many of those streams are coupled stereo pairs.
    pub coupled_count: u8,
    /// For each output channel, the decoded stream channel that feeds it, or
    /// 255 for silence.
    pub channel_mapping: Vec<u8>,
}

impl OpusHead {
    /// A mapping-family-0 header for a mono or stereo stream.
    pub fn new(channels: u8, pre_skip: u16, input_sample_rate: u32) -> Result<Self> {
        if channels != 1 && channels != 2 {
            return Err(invalid("an Opus mapping family 0 stream is mono or stereo"));
        }
        Ok(Self {
            channels,
            pre_skip,
            input_sample_rate,
            output_gain: 0,
            channel_mapping_family: 0,
            stream_count: 1,
            coupled_count: channels - 1,
            channel_mapping: (0..channels).collect(),
        })
    }

    /// Parses a complete `dOps` box, size and type included, as the Opus
    /// sample entry of an MP4 carries it ("Encapsulation of Opus in ISO Base
    /// Media File Format", section 4.3.2).
    pub fn from_dops(dops: &[u8]) -> Result<Self> {
        let payload = crate::codec_config::box_payload(dops, b"dOps")?;
        let declared = u32::from_be_bytes(dops[..4].try_into().expect("checked by box_payload"));
        if usize::try_from(declared).ok() != Some(dops.len()) {
            return Err(malformed("Opus dOps box size is inconsistent"));
        }
        let header = payload
            .get(..11)
            .ok_or_else(|| malformed("Opus dOps box is truncated"))?;
        if header[0] != 0 {
            return Err(unsupported(format!(
                "Opus dOps version {} is unsupported",
                header[0]
            )));
        }
        let head = Self::from_fields(
            header[1],
            u16::from_be_bytes([header[2], header[3]]),
            u32::from_be_bytes([header[4], header[5], header[6], header[7]]),
            i16::from_be_bytes([header[8], header[9]]),
            header[10],
            &payload[11..],
        )?;
        Ok(head)
    }

    /// Writes this header as a complete `dOps` box, size and type included.
    pub fn to_dops(&self) -> Vec<u8> {
        let mut payload = vec![0, self.channels];
        payload.extend_from_slice(&self.pre_skip.to_be_bytes());
        payload.extend_from_slice(&self.input_sample_rate.to_be_bytes());
        payload.extend_from_slice(&self.output_gain.to_be_bytes());
        self.push_mapping(&mut payload);
        let mut whole = u32::try_from(payload.len() + 8)
            .expect("a dOps box is at most a few hundred bytes")
            .to_be_bytes()
            .to_vec();
        whole.extend_from_slice(b"dOps");
        whole.extend_from_slice(&payload);
        whole
    }

    /// Parses an RFC 7845 `OpusHead` identification header packet, the
    /// little-endian form Ogg pages and Matroska or WebM `CodecPrivate` carry.
    pub fn from_identification_header(packet: &[u8]) -> Result<Self> {
        let header = packet
            .get(..19)
            .ok_or_else(|| malformed("Opus identification header is truncated"))?;
        if &header[..8] != b"OpusHead" {
            return Err(malformed(
                "Opus identification header does not start with OpusHead",
            ));
        }
        // The major version is the upper four bits; RFC 7845 asks decoders to
        // accept every minor version of the major version they implement.
        if header[8] >> 4 != 0 {
            return Err(unsupported(format!(
                "Opus identification header version {} is unsupported",
                header[8]
            )));
        }
        Self::from_fields(
            header[9],
            u16::from_le_bytes([header[10], header[11]]),
            u32::from_le_bytes([header[12], header[13], header[14], header[15]]),
            i16::from_le_bytes([header[16], header[17]]),
            header[18],
            &packet[19..],
        )
    }

    /// Writes this header as an RFC 7845 `OpusHead` identification header
    /// packet (version 1).
    pub fn to_identification_header(&self) -> Vec<u8> {
        let mut packet = b"OpusHead".to_vec();
        packet.push(1);
        packet.push(self.channels);
        packet.extend_from_slice(&self.pre_skip.to_le_bytes());
        packet.extend_from_slice(&self.input_sample_rate.to_le_bytes());
        packet.extend_from_slice(&self.output_gain.to_le_bytes());
        self.push_mapping(&mut packet);
        packet
    }

    /// The linear factor [`Self::output_gain`] scales decoded samples by.
    pub fn output_gain_factor(&self) -> f32 {
        10_f32.powf(f32::from(self.output_gain) / (20.0 * 256.0))
    }

    fn from_fields(
        channels: u8,
        pre_skip: u16,
        input_sample_rate: u32,
        output_gain: i16,
        channel_mapping_family: u8,
        mapping_table: &[u8],
    ) -> Result<Self> {
        if channels == 0 {
            return Err(malformed("an Opus stream has at least one channel"));
        }
        if channel_mapping_family == 0 {
            if channels > 2 {
                return Err(malformed(
                    "an Opus mapping family 0 stream is mono or stereo",
                ));
            }
            return Ok(Self {
                output_gain,
                ..Self::new(channels, pre_skip, input_sample_rate)?
            });
        }
        let table = mapping_table
            .get(..2 + usize::from(channels))
            .ok_or_else(|| malformed("Opus channel mapping table is truncated"))?;
        let (stream_count, coupled_count) = (table[0], table[1]);
        if stream_count == 0 || coupled_count > stream_count {
            return Err(malformed("Opus channel mapping has invalid stream counts"));
        }
        let decoded_channels = u16::from(stream_count) + u16::from(coupled_count);
        let channel_mapping = table[2..].to_vec();
        if channel_mapping
            .iter()
            .any(|&index| index != 255 && u16::from(index) >= decoded_channels)
        {
            return Err(malformed(
                "Opus channel mapping names a channel no stream decodes",
            ));
        }
        Ok(Self {
            channels,
            pre_skip,
            input_sample_rate,
            output_gain,
            channel_mapping_family,
            stream_count,
            coupled_count,
            channel_mapping,
        })
    }

    fn push_mapping(&self, out: &mut Vec<u8>) {
        out.push(self.channel_mapping_family);
        if self.channel_mapping_family != 0 {
            out.push(self.stream_count);
            out.push(self.coupled_count);
            out.extend_from_slice(&self.channel_mapping);
        }
    }
}

/// The number of 48 kHz samples one Opus packet decodes to, read from its
/// table-of-contents byte and frame count (RFC 6716 section 3.1).
pub fn opus_packet_samples(packet: &[u8]) -> Result<u32> {
    let samples = opus_pure::packet::samples_48k(packet).map_err(opus_error)?;
    u32::try_from(samples).map_err(|_| malformed("Opus packet duration overflows"))
}

/// The number of packets an [`crate::AudioSampleReader`] should decode ahead
/// of a seek target in an Opus track whose packets are `packets`, enough to
/// cover [`OPUS_PREROLL_SAMPLES`] even at the track's shortest packet.
pub fn opus_preroll_packets(packets: &[EncodedAudioSample]) -> usize {
    let shortest = packets
        .iter()
        .map(|packet| packet.decoded_range.len())
        .filter(|&length| length > 0)
        .min()
        .unwrap_or(u64::from(OPUS_PREROLL_SAMPLES));
    usize::try_from(u64::from(OPUS_PREROLL_SAMPLES).div_ceil(shortest)).unwrap_or(usize::MAX)
}

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
        if cancellation.is_cancelled() {
            return Err(Error::new(ErrorKind::Cancelled, "Opus decode cancelled"));
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

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

fn internal(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Internal, message)
}
