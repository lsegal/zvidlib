//! The native Vorbis encoder: zvidlib's own pure-Rust port of the libvorbis
//! encoder.
//!
//! This is an internal crate of [zvidlib](https://crates.io/crates/zvidlib),
//! which re-exports [`native_vorbis_audio_encoder_factory`] from its `vorbis`
//! module. Depend on `zvidlib` rather than on this crate directly.

#[doc(hidden)]
pub mod vorbis_encoder;

use crate::codec::parse_audio_bit_rate;
use crate::vorbis_encoder::{VorbisEncoder, quality_for_nominal_bitrate};
use crate::{
    AudioBuffer, AudioDrain, AudioEncoder, AudioEncoderConfig, AudioEncoderFactory,
    AudioEncoderFormat, AudioGapless, Codec, CodecImplementation, CodecProfile, CodecSupport,
    EncodedSample, EncoderConfig, EncoderFuture, Error, ErrorKind, FrameIndex, Limits, Result,
    SampleDependency,
};

#[allow(unused_imports)]
use zvidlib_core::*;
#[allow(unused_imports)]
use zvidlib_vorbis_syntax::*;

/// The quality the native encoder uses when its configuration names no bit
/// rate: libvorbis's `-q 4`, about 128 kb/s for 44.1 kHz stereo.
const DEFAULT_QUALITY: f32 = 0.4;

/// The comment header's vendor string for streams the native encoder writes.
const VENDOR: &str = concat!(
    "zvidlib ",
    env!("CARGO_PKG_VERSION"),
    " (port of Xiph.Org libVorbis I 20200704)"
);

/// Returns the native Vorbis encoder.
///
/// It is zvidlib's own pure-Rust port of the libvorbis 1.3.7 encoder's VBR
/// mode: for the same input, quality and buffer sizes its packets are
/// bit-identical to libvorbis's own, so its output quality is the reference
/// encoder's rather than a reimplementation's. It runs on every target,
/// `wasm32` included, and reports [`CodecImplementation::Software`].
///
/// It encodes mono or stereo at every sample rate libvorbis has a setup for.
/// [`AudioEncoderConfig::configuration`] is either empty, for libvorbis's
/// quality 4 (about 128 kb/s at 44.1 kHz stereo), or four big-endian bytes
/// naming a target bit rate, which is turned into the VBR quality libvorbis
/// chooses for that nominal rate. The encoder config's `decoder_config` is the
/// stream's three header packets Xiph-laced, the `CodecPrivate` Matroska and
/// WebM carry ([`VorbisConfig::from_codec_private`]): Vorbis has no MP4 sample
/// entry, so `crate::mp4::Mp4Muxer` refuses it.
///
/// Each emitted [`EncodedSample`] spans the samples its packet decodes to - the
/// first spans none - so a stream's priming is zero, and the drained
/// [`AudioGapless::padding`] is what the last packet decodes past the input.
pub fn native_vorbis_audio_encoder_factory() -> impl AudioEncoderFactory {
    VorbisEncoderFactory
}

#[derive(Clone, Copy, Debug)]
struct VorbisEncoderFactory;

impl VorbisEncoderFactory {
    /// The libvorbis quality `configuration` asks for, or why it cannot be met.
    fn quality(configuration: &AudioEncoderConfig) -> std::result::Result<f32, String> {
        match parse_audio_bit_rate(&configuration.configuration) {
            None => Err(
                "native Vorbis encoder configuration is either empty or four big-endian \
                         bytes giving a nonzero target bit rate in bits a second"
                    .into(),
            ),
            Some(None) => Ok(DEFAULT_QUALITY),
            Some(Some(bit_rate)) => quality_for_nominal_bitrate(
                configuration.sample_rate,
                configuration.channels,
                bit_rate,
            )
            .ok_or_else(|| {
                format!(
                    "libvorbis has no quality for {bit_rate} bits a second at {} Hz with {} \
                     channel(s)",
                    configuration.sample_rate, configuration.channels
                )
            }),
        }
    }
}

impl AudioEncoderFactory for VorbisEncoderFactory {
    fn capability(&self, configuration: &AudioEncoderConfig) -> CodecSupport {
        if configuration.codec != Codec::Vorbis {
            return CodecSupport::UnsupportedCodec;
        }
        if configuration.profile != CodecProfile::Vorbis {
            return CodecSupport::UnsupportedProfile;
        }
        if configuration.timescale != configuration.sample_rate {
            return CodecSupport::InvalidConfiguration {
                reason: "native Vorbis encoder timescale must equal the sample rate".into(),
            };
        }
        let quality = match Self::quality(configuration) {
            Ok(quality) => quality,
            Err(reason) => return CodecSupport::InvalidConfiguration { reason },
        };
        // libvorbis decides which rates and channel counts it has a setup
        // for, so the encoder is asked rather than a copy of its rules.
        match VorbisEncoder::new(configuration.sample_rate, configuration.channels, quality) {
            Ok(_) => CodecSupport::Supported {
                implementation: CodecImplementation::Software,
            },
            Err(error) => CodecSupport::InvalidConfiguration {
                reason: error.to_string(),
            },
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
                    "native Vorbis encoder requires the Vorbis codec and profile",
                ));
            }
        }
        let quality = Self::quality(configuration)
            .map_err(|reason| Error::new(ErrorKind::InvalidInput, reason))?;
        let encoder =
            VorbisEncoder::new(configuration.sample_rate, configuration.channels, quality)
                .map_err(|error| Error::new(ErrorKind::InvalidInput, error.to_string()))?;
        let [identification, comment, setup] = encoder.headers(VENDOR);
        let stream = VorbisConfig::from_headers(identification, comment, setup)?;
        Ok(Box::new(NativeVorbisEncoder {
            config: EncoderConfig {
                codec: Codec::Vorbis,
                timescale: configuration.sample_rate,
                decoder_config: stream.to_codec_private(),
            },
            format: AudioEncoderFormat {
                sample_rate: configuration.sample_rate,
                channels: configuration.channels,
            },
            encoder,
            stream,
            previous_block: None,
            input_frames: 0,
            decoded_frames: 0,
            limits: *limits,
            finished: false,
        }))
    }
}

struct NativeVorbisEncoder {
    config: EncoderConfig,
    format: AudioEncoderFormat,
    encoder: VorbisEncoder,
    /// The stream's own headers, which say how long each packet decodes.
    stream: VorbisConfig,
    previous_block: Option<u16>,
    input_frames: u64,
    /// Samples the packets emitted so far decode to.
    decoded_frames: u64,
    limits: Limits,
    finished: bool,
}

impl NativeVorbisEncoder {
    /// Gives each packet the interval of decoded samples it produces.
    fn samples(&mut self, packets: Vec<(Vec<u8>, u64)>) -> Result<Vec<EncodedSample>> {
        packets
            .into_iter()
            .map(|(data, _granule)| {
                let block = self.stream.packet_block_size(&data)?;
                let length = self
                    .previous_block
                    .map_or(0, |previous| (u32::from(previous) + u32::from(block)) / 4);
                self.previous_block = Some(block);
                let dts = i64::try_from(self.decoded_frames).map_err(|_| {
                    Error::new(ErrorKind::ResourceLimit, "Vorbis timestamp overflow")
                })?;
                self.decoded_frames += u64::from(length);
                Ok(EncodedSample {
                    data,
                    dts,
                    pts: dts,
                    duration: length,
                    is_sync: true,
                    dependency: SampleDependency::INDEPENDENT,
                })
            })
            .collect()
    }
}

impl AudioEncoder for NativeVorbisEncoder {
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
                    "the Vorbis encoder has already finished",
                ));
            }
            if buffer.sample_rate != self.format.sample_rate
                || buffer.channels != self.format.channels
            {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "audio buffer format does not match the Vorbis encoder",
                ));
            }
            let bytes = buffer.samples.len() as u64 * std::mem::size_of::<f32>() as u64;
            if bytes > self.limits.max_allocation_bytes {
                return Err(Error::new(
                    ErrorKind::ResourceLimit,
                    "audio buffer exceeds the allocation limit",
                ));
            }
            self.input_frames += buffer.range.len();
            let packets = self
                .encoder
                .encode(&buffer.samples)
                .map_err(|error| codec(error.to_string()))?;
            self.samples(packets)
        })
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, AudioDrain> {
        Box::pin(async move {
            if self.finished {
                return Ok(AudioDrain::default());
            }
            self.finished = true;
            let packets = self
                .encoder
                .finish()
                .map_err(|error| codec(error.to_string()))?;
            let samples = self.samples(packets)?;
            let padding = self
                .decoded_frames
                .checked_sub(self.input_frames)
                .ok_or_else(|| {
                    codec("the Vorbis encoder decoded fewer frames than it was given")
                })?;
            Ok(AudioDrain {
                samples,
                gapless: AudioGapless {
                    priming: 0,
                    padding: u32::try_from(padding).map_err(|_| {
                        Error::new(ErrorKind::ResourceLimit, "Vorbis padding exceeds u32")
                    })?,
                },
            })
        })
    }
}

fn codec(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Codec, message)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

/// The SIMD dispatch sites in this crate, each with the instruction set it
/// resolves to right now. `zvidlib::simd::active_by_site` reports every crate's
/// sites together and documents what each one covers.
#[doc(hidden)]
#[must_use]
pub fn simd_sites() -> Vec<(&'static str, zvidlib_core::SimdIsa)> {
    vec![("vorbis_encode", vorbis_encoder::simd::active_isa())]
}
