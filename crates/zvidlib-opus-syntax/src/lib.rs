//! Opus configuration records and packet timing, shared by zvidlib's
//! containers and its Opus codec.
//!
//! Opus always decodes at 48 kHz, which is therefore the sample clock of every
//! Opus track zvidlib reads or writes. [`OpusHead`] is the stream's decoder
//! configuration in both of the forms containers carry it, and
//! [`opus_packet_samples`] reads a packet's duration from its
//! table-of-contents byte, which is all a container needs to time a packet
//! without decoding it.
//!
//! This is an internal crate of [zvidlib](https://crates.io/crates/zvidlib),
//! which re-exports its public items under the paths documented there. Depend
//! on `zvidlib` rather than on this crate directly.

#[allow(unused_imports)]
use zvidlib_core::*;

use crate::{EncodedAudioSample, Error, ErrorKind, Result};

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
    let &toc = packet
        .first()
        .ok_or_else(|| invalid_packet("packet is empty"))?;
    let frames = match toc & 0x03 {
        0 => 1,
        1 | 2 => 2,
        _ => {
            let count = packet
                .get(1)
                .ok_or_else(|| invalid_packet("code 3 packet has no frame count"))?
                & 0x3F;
            if count == 0 {
                return Err(invalid_packet("code 3 packet declares zero frames"));
            }
            u32::from(count)
        }
    };
    let samples = frames * frame_samples(toc);
    // A packet cannot describe more than 120 ms, so a frame count that implies
    // more is a malformed packet rather than a very long one.
    if samples > MAX_PACKET_SAMPLES {
        return Err(invalid_packet("packet claims more than 120 ms"));
    }
    Ok(samples)
}

/// The 48 kHz samples in each frame of a packet whose table-of-contents byte
/// is `toc`: 2.5 to 20 ms for CELT, 10 or 20 ms for hybrid, and 10 to 60 ms for
/// SILK (RFC 6716 section 3.1).
fn frame_samples(toc: u8) -> u32 {
    let size = u32::from((toc >> 3) & 0x03);
    if toc & 0x80 != 0 {
        (OPUS_SAMPLE_RATE << size) / 400
    } else if toc & 0x60 == 0x60 {
        if toc & 0x08 != 0 {
            OPUS_SAMPLE_RATE / 50
        } else {
            OPUS_SAMPLE_RATE / 100
        }
    } else if size == 3 {
        OPUS_SAMPLE_RATE * 60 / 1000
    } else {
        (OPUS_SAMPLE_RATE << size) / 100
    }
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

/// The longest an Opus packet can last, 120 ms at 48 kHz.
const MAX_PACKET_SAMPLES: u32 = 5_760;

/// A packet the table of contents says is malformed, reported as the native
/// decoder reports the same packet.
fn invalid_packet(what: &str) -> Error {
    codec(format!("Opus: invalid packet: {what}"))
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
