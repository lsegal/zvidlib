//! Vorbis audio: decoder configuration, packet timing and native decoding.
//!
//! A Vorbis stream is described by three header packets - identification,
//! comment and setup - which Matroska and WebM carry Xiph-laced as
//! `CodecPrivate` and a `WebCodecs` `AudioDecoderConfig.description` takes in
//! the same form. [`VorbisConfig`] parses and writes them, and reads from the
//! setup header the one thing a container does not record exactly: how many
//! samples each audio packet decodes to.
//!
//! Decoding runs on Symphonia's pure-Rust Vorbis decoder, so it works on every
//! target zvidlib builds for, `wasm32` included.

use crate::{
    AudioBuffer, AudioDecoder, CancellationToken, EncodedAudioSample, Error, ErrorKind, Limits,
    Result, SampleRange,
};
use symphonia_core::audio::{AudioBufferRef, SampleBuffer};
use symphonia_core::codecs::{CODEC_TYPE_VORBIS, CodecParameters, Decoder, DecoderOptions};
use symphonia_core::formats::Packet;

/// How many packets an [`crate::AudioSampleReader`] must decode ahead of the
/// first one a request needs: a Vorbis packet's samples are the overlap of its
/// block with the previous packet's, so the previous packet has to be decoded
/// first.
pub const VORBIS_PREROLL_PACKETS: usize = 1;

/// The three Vorbis I header packets and what zvidlib reads from them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VorbisConfig {
    pub channels: u8,
    pub sample_rate: u32,
    /// The identification header's maximum, nominal and minimum bit rates,
    /// in bits a second; zero where the stream does not say.
    pub bitrate_maximum: i32,
    pub bitrate_nominal: i32,
    pub bitrate_minimum: i32,
    /// The short and long block sizes, in samples.
    pub block_sizes: (u16, u16),
    pub identification_header: Vec<u8>,
    pub comment_header: Vec<u8>,
    pub setup_header: Vec<u8>,
    /// Whether each mode the setup header defines codes a long block.
    mode_block_flags: Vec<bool>,
}

impl VorbisConfig {
    /// Validates and parses the three header packets.
    pub fn from_headers(
        identification_header: Vec<u8>,
        comment_header: Vec<u8>,
        setup_header: Vec<u8>,
    ) -> Result<Self> {
        let ident = &identification_header;
        if ident.len() != 30 || ident[0] != 1 || &ident[1..7] != b"vorbis" {
            return Err(malformed(
                "Vorbis identification header is not a 30-byte type 1 packet",
            ));
        }
        if u32::from_le_bytes(ident[7..11].try_into().expect("fixed")) != 0 {
            return Err(unsupported("only Vorbis I streams are supported"));
        }
        let channels = ident[11];
        let sample_rate = u32::from_le_bytes(ident[12..16].try_into().expect("fixed"));
        let bitrate = |at: usize| i32::from_le_bytes(ident[at..at + 4].try_into().expect("fixed"));
        let short_exponent = ident[28] & 0x0f;
        let long_exponent = ident[28] >> 4;
        if channels == 0 || sample_rate == 0 {
            return Err(malformed(
                "Vorbis identification header has no channels or sample rate",
            ));
        }
        if !(6..=13).contains(&short_exponent)
            || !(6..=13).contains(&long_exponent)
            || short_exponent > long_exponent
        {
            return Err(malformed(
                "Vorbis identification header has invalid block sizes",
            ));
        }
        if ident[29] & 1 == 0 {
            return Err(malformed(
                "Vorbis identification header framing bit is unset",
            ));
        }
        if comment_header.len() < 7 || comment_header[0] != 3 || &comment_header[1..7] != b"vorbis"
        {
            return Err(malformed("Vorbis comment header is not a type 3 packet"));
        }
        let mode_block_flags = parse_setup_modes(&setup_header, channels)?;
        Ok(Self {
            channels,
            sample_rate,
            bitrate_maximum: bitrate(16),
            bitrate_nominal: bitrate(20),
            bitrate_minimum: bitrate(24),
            block_sizes: (1 << short_exponent, 1 << long_exponent),
            identification_header,
            comment_header,
            setup_header,
            mode_block_flags,
        })
    }

    /// Parses the three header packets Xiph-laced, the form Matroska and WebM
    /// `CodecPrivate` use: a byte holding the packet count less one (two), the
    /// first two packets' sizes as runs of 255 ending in a smaller byte, then
    /// the packets.
    pub fn from_codec_private(codec_private: &[u8]) -> Result<Self> {
        let (&count, mut rest) = codec_private
            .split_first()
            .ok_or_else(|| malformed("Vorbis CodecPrivate is empty"))?;
        if count != 2 {
            return Err(malformed(
                "Vorbis CodecPrivate must lace three header packets",
            ));
        }
        let mut sizes = [0_usize; 2];
        for size in &mut sizes {
            loop {
                let (&byte, tail) = rest
                    .split_first()
                    .ok_or_else(|| malformed("Vorbis CodecPrivate lacing is truncated"))?;
                rest = tail;
                *size += usize::from(byte);
                if byte != 255 {
                    break;
                }
            }
        }
        if sizes[0] + sizes[1] > rest.len() {
            return Err(malformed("Vorbis CodecPrivate is shorter than its lacing"));
        }
        let (identification, rest) = rest.split_at(sizes[0]);
        let (comment, setup) = rest.split_at(sizes[1]);
        Self::from_headers(identification.to_vec(), comment.to_vec(), setup.to_vec())
    }

    /// Writes the three header packets Xiph-laced, as Matroska and WebM
    /// `CodecPrivate` and a `WebCodecs` decoder `description` take them.
    pub fn to_codec_private(&self) -> Vec<u8> {
        let mut out = vec![2];
        for size in [self.identification_header.len(), self.comment_header.len()] {
            out.extend(std::iter::repeat_n(255, size / 255));
            out.push((size % 255) as u8);
        }
        out.extend_from_slice(&self.identification_header);
        out.extend_from_slice(&self.comment_header);
        out.extend_from_slice(&self.setup_header);
        out
    }

    /// The block size an audio packet codes, read from its mode number.
    pub fn packet_block_size(&self, packet: &[u8]) -> Result<u16> {
        let mut bits = BitReader::new(packet);
        if bits.read(1)? != 0 {
            return Err(malformed("Vorbis audio packet has a header packet type"));
        }
        let mode_bits = ilog(self.mode_block_flags.len() as u32 - 1);
        let mode = usize::try_from(bits.read(mode_bits)?).expect("at most eight bits");
        let long = *self
            .mode_block_flags
            .get(mode)
            .ok_or_else(|| malformed("Vorbis audio packet names an undefined mode"))?;
        Ok(if long {
            self.block_sizes.1
        } else {
            self.block_sizes.0
        })
    }

    /// Assigns each audio packet, in stream order, the interval of decoded
    /// samples it produces: none for the first, and for every later one a
    /// quarter of its block size plus a quarter of the previous packet's.
    pub fn encoded_samples(&self, packets: Vec<Vec<u8>>) -> Result<Vec<EncodedAudioSample>> {
        let mut previous = None;
        let mut end = 0_u64;
        packets
            .into_iter()
            .map(|data| {
                let block = self.packet_block_size(&data)?;
                let length = previous.map_or(0, |previous: u16| {
                    (u64::from(previous) + u64::from(block)) / 4
                });
                previous = Some(block);
                let start = end;
                end += length;
                Ok(EncodedAudioSample {
                    decoded_range: SampleRange::new(start, end)?,
                    data,
                })
            })
            .collect()
    }
}

/// Walks a setup header (Vorbis I specification section 4.2.4) far enough to
/// validate its structure and read its modes' block flags, which come last.
fn parse_setup_modes(setup: &[u8], channels: u8) -> Result<Vec<bool>> {
    if setup.len() < 7 || setup[0] != 5 || &setup[1..7] != b"vorbis" {
        return Err(malformed("Vorbis setup header is not a type 5 packet"));
    }
    let mut bits = BitReader::new(&setup[7..]);
    let codebooks = bits.read(8)? + 1;
    for _ in 0..codebooks {
        skip_codebook(&mut bits)?;
    }
    let time_domain_transforms = bits.read(6)? + 1;
    for _ in 0..time_domain_transforms {
        if bits.read(16)? != 0 {
            return Err(malformed(
                "Vorbis setup header has a nonzero time domain type",
            ));
        }
    }
    let floors = bits.read(6)? + 1;
    for _ in 0..floors {
        skip_floor(&mut bits)?;
    }
    let residues = bits.read(6)? + 1;
    for _ in 0..residues {
        skip_residue(&mut bits)?;
    }
    let mappings = bits.read(6)? + 1;
    for _ in 0..mappings {
        skip_mapping(&mut bits, channels)?;
    }
    let modes = bits.read(6)? + 1;
    let mut block_flags = Vec::with_capacity(modes as usize);
    for _ in 0..modes {
        block_flags.push(bits.read(1)? == 1);
        if bits.read(16)? != 0 || bits.read(16)? != 0 {
            return Err(malformed(
                "Vorbis setup header mode has a nonzero window or transform type",
            ));
        }
        if bits.read(8)? >= mappings {
            return Err(malformed(
                "Vorbis setup header mode names an undefined mapping",
            ));
        }
    }
    if bits.read(1)? != 1 {
        return Err(malformed("Vorbis setup header framing bit is unset"));
    }
    Ok(block_flags)
}

fn skip_codebook(bits: &mut BitReader<'_>) -> Result<()> {
    if bits.read(24)? != 0x56_4342 {
        return Err(malformed("Vorbis codebook has an invalid sync pattern"));
    }
    let dimensions = bits.read(16)?;
    let entries = bits.read(24)?;
    if bits.read(1)? == 1 {
        // Ordered: runs of entries sharing a length.
        let mut current = 0;
        bits.read(5)?;
        while current < entries {
            current += bits.read(ilog(entries - current))?;
        }
        if current > entries {
            return Err(malformed(
                "Vorbis codebook length runs overflow its entries",
            ));
        }
    } else {
        let sparse = bits.read(1)? == 1;
        for _ in 0..entries {
            if !sparse || bits.read(1)? == 1 {
                bits.read(5)?;
            }
        }
    }
    match bits.read(4)? {
        0 => {}
        lookup_type @ (1 | 2) => {
            bits.read(32)?;
            bits.read(32)?;
            let value_bits = bits.read(4)? + 1;
            bits.read(1)?;
            let values = if lookup_type == 1 {
                lookup1_values(entries, dimensions)
            } else {
                u64::from(entries) * u64::from(dimensions)
            };
            bits.skip(values * u64::from(value_bits))?;
        }
        _ => return Err(malformed("Vorbis codebook has an invalid lookup type")),
    }
    Ok(())
}

fn skip_floor(bits: &mut BitReader<'_>) -> Result<()> {
    match bits.read(16)? {
        0 => {
            bits.read(8 + 16 + 16 + 6 + 8)?;
            let books = bits.read(4)? + 1;
            bits.skip(u64::from(books) * 8)?;
        }
        1 => {
            let partitions = bits.read(5)?;
            let mut classes = Vec::with_capacity(partitions as usize);
            for _ in 0..partitions {
                classes.push(bits.read(4)?);
            }
            let class_count = classes.iter().max().map_or(0, |&max| max + 1);
            let mut dimensions = Vec::with_capacity(class_count as usize);
            for _ in 0..class_count {
                dimensions.push(bits.read(3)? + 1);
                let subclasses = bits.read(2)?;
                if subclasses != 0 {
                    bits.read(8)?;
                }
                bits.skip((1_u64 << subclasses) * 8)?;
            }
            bits.read(2)?;
            let range_bits = bits.read(4)?;
            for class in classes {
                bits.skip(u64::from(dimensions[class as usize]) * u64::from(range_bits))?;
            }
        }
        _ => return Err(malformed("Vorbis setup header has an invalid floor type")),
    }
    Ok(())
}

fn skip_residue(bits: &mut BitReader<'_>) -> Result<()> {
    if bits.read(16)? > 2 {
        return Err(malformed("Vorbis setup header has an invalid residue type"));
    }
    bits.read(24)?;
    bits.read(24)?;
    bits.read(24)?;
    let classifications = bits.read(6)? + 1;
    bits.read(8)?;
    let mut books = 0;
    for _ in 0..classifications {
        let low = bits.read(3)?;
        let high = if bits.read(1)? == 1 { bits.read(5)? } else { 0 };
        books += (high << 3 | low).count_ones();
    }
    bits.skip(u64::from(books) * 8)
}

fn skip_mapping(bits: &mut BitReader<'_>, channels: u8) -> Result<()> {
    if bits.read(16)? != 0 {
        return Err(malformed("Vorbis setup header has an invalid mapping type"));
    }
    let submaps = if bits.read(1)? == 1 {
        bits.read(4)? + 1
    } else {
        1
    };
    let channel_bits = ilog(u32::from(channels) - 1);
    if bits.read(1)? == 1 {
        let steps = bits.read(8)? + 1;
        bits.skip(u64::from(steps) * 2 * u64::from(channel_bits))?;
    }
    if bits.read(2)? != 0 {
        return Err(malformed(
            "Vorbis setup header mapping reserved bits are set",
        ));
    }
    if submaps > 1 {
        bits.skip(u64::from(channels) * 4)?;
    }
    bits.skip(u64::from(submaps) * 24)
}

/// The number of bits needed to hold `value` (Vorbis I section 9.2.1).
fn ilog(value: u32) -> u32 {
    u32::BITS - value.leading_zeros()
}

/// The greatest integer whose `dimensions`th power is at most `entries`
/// (Vorbis I section 9.2.3).
fn lookup1_values(entries: u32, dimensions: u32) -> u64 {
    if dimensions == 0 {
        return 0;
    }
    let mut value = (f64::from(entries))
        .powf(1.0 / f64::from(dimensions))
        .floor() as u64;
    while (value + 1)
        .checked_pow(dimensions)
        .is_some_and(|power| power <= u64::from(entries))
    {
        value += 1;
    }
    while value > 0
        && value
            .checked_pow(dimensions)
            .is_none_or(|power| power > u64::from(entries))
    {
        value -= 1;
    }
    value
}

/// Reads Vorbis's least-significant-bit-first packing.
struct BitReader<'a> {
    bytes: &'a [u8],
    position: u64,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn read(&mut self, count: u32) -> Result<u32> {
        debug_assert!(count <= 32);
        let end = self.position + u64::from(count);
        if end > self.bytes.len() as u64 * 8 {
            return Err(malformed("Vorbis packet is truncated"));
        }
        let mut value = 0_u64;
        for bit in 0..u64::from(count) {
            let at = self.position + bit;
            let byte = self.bytes[(at / 8) as usize];
            value |= u64::from((byte >> (at % 8)) & 1) << bit;
        }
        self.position = end;
        Ok(value as u32)
    }

    fn skip(&mut self, count: u64) -> Result<()> {
        let end = self
            .position
            .checked_add(count)
            .filter(|&end| end <= self.bytes.len() as u64 * 8)
            .ok_or_else(|| malformed("Vorbis packet is truncated"))?;
        self.position = end;
        Ok(())
    }
}

/// A native, pure-Rust Vorbis decoder for mono and stereo streams, producing
/// interleaved `f32` PCM.
///
/// A packet decoded straight after [`AudioDecoder::reset`] has no previous
/// block to overlap and decodes to silence; [`VORBIS_PREROLL_PACKETS`] is the
/// preroll that keeps an [`crate::AudioSampleReader`] from returning it.
pub struct NativeVorbisDecoder {
    decoder: symphonia_codec_vorbis::VorbisDecoder,
    channels: u16,
    sample_rate: u32,
    /// Whether nothing has been decoded since the decoder was created or reset.
    cold: bool,
    limits: Limits,
}

impl NativeVorbisDecoder {
    pub fn new(config: &VorbisConfig, limits: Limits) -> Result<Self> {
        let channels = u16::from(config.channels);
        if channels > limits.max_audio_channels || config.sample_rate > limits.max_sample_rate {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "Vorbis channel count or sample rate is outside configured limits",
            ));
        }
        // Symphonia's Vorbis decoder, which this wraps, decodes some streams of
        // more than two channels wrongly where FFmpeg's and libvorbis's agree,
        // so those are refused rather than decoded to the wrong audio.
        if channels > 2 {
            return Err(unsupported(
                "the native Vorbis decoder supports mono and stereo streams",
            ));
        }
        let mut extra_data = config.identification_header.clone();
        extra_data.extend_from_slice(&config.setup_header);
        let mut parameters = CodecParameters::new();
        parameters
            .for_codec(CODEC_TYPE_VORBIS)
            .with_extra_data(extra_data.into_boxed_slice());
        let decoder =
            symphonia_codec_vorbis::VorbisDecoder::try_new(&parameters, &DecoderOptions::default())
                .map_err(|error| {
                    unsupported(format!("Vorbis configuration is unsupported: {error}"))
                })?;
        Ok(Self {
            decoder,
            channels,
            sample_rate: config.sample_rate,
            cold: true,
            limits,
        })
    }
}

impl AudioDecoder for NativeVorbisDecoder {
    fn decode(
        &mut self,
        sample: &EncodedAudioSample,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        if cancellation.is_cancelled() {
            return Err(Error::new(ErrorKind::Cancelled, "Vorbis decode cancelled"));
        }
        let cold = std::mem::replace(&mut self.cold, false);
        let channels = usize::from(self.channels);
        let decoded = self
            .decoder
            .decode(&Packet::new_from_slice(
                0,
                sample.decoded_range.start,
                sample.decoded_range.len(),
                &sample.data,
            ))
            .map_err(|error| codec(format!("malformed Vorbis packet: {error}")))?;
        let frames = decoded.frames();
        let expected = sample.decoded_range.len();
        let samples = if frames as u64 == expected {
            interleave(decoded)
        } else if frames == 0 && cold {
            vec![
                0.0;
                usize::try_from(expected)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(channels)
            ]
        } else {
            return Err(codec(format!(
                "Vorbis decoder produced {frames} frames for an indexed {expected}-frame interval"
            )));
        };
        AudioBuffer::new(
            sample.decoded_range,
            self.sample_rate,
            self.channels,
            samples,
            &self.limits,
        )
    }

    fn reset(&mut self) -> Result<()> {
        self.decoder.reset();
        self.cold = true;
        Ok(())
    }
}

fn interleave(decoded: AudioBufferRef<'_>) -> Vec<f32> {
    // Symphonia's interleaving copy panics on an empty buffer, which is what
    // a stream's first packet decodes to.
    if decoded.frames() == 0 {
        return Vec::new();
    }
    let mut interleaved = SampleBuffer::<f32>::new(decoded.frames() as u64, *decoded.spec());
    interleaved.copy_interleaved_ref(decoded);
    interleaved.samples().to_vec()
}

fn codec(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Codec, message)
}

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}
