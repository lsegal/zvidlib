//! Vorbis audio: decoder configuration, packet timing and native decoding.
//!
//! A Vorbis stream is described by three header packets - identification,
//! comment and setup - which Matroska and WebM carry Xiph-laced as
//! `CodecPrivate` and a `WebCodecs` `AudioDecoderConfig.description` takes in
//! the same form. [`VorbisConfig`] parses and writes them, and reads from the
//! setup header the one thing a container does not record exactly: how many
//! samples each audio packet decodes to.
//!
//! Decoding runs on Symphonia's pure-Rust Vorbis decoder, and encoding on
//! zvidlib's own port of the libvorbis encoder, so both work on every target
//! zvidlib builds for, `wasm32` included.

mod vorbis_decoder;
#[doc(hidden)]
pub mod vorbis_simd;

#[allow(unused_imports)]
use zvidlib_core::*;

use crate::vorbis_decoder::VorbisDecoder;
use crate::{
    AudioBuffer, AudioDecoder, CancellationToken, EncodedAudioSample, Error, ErrorKind, Limits,
    Result,
};
use symphonia_core::audio::{AudioBufferRef, SampleBuffer};
use symphonia_core::codecs::{CODEC_TYPE_VORBIS, CodecParameters, Decoder, DecoderOptions};
use symphonia_core::formats::Packet;

#[doc(inline)]
pub use zvidlib_vorbis_syntax::{VORBIS_PREROLL_PACKETS, VorbisConfig};

/// A native, pure-Rust Vorbis decoder for streams of one to eight channels,
/// producing interleaved `f32` PCM in the Vorbis channel order (Vorbis I
/// section 4.3.9).
///
/// A packet decoded straight after [`AudioDecoder::reset`] has no previous
/// block to overlap and decodes to silence; [`VORBIS_PREROLL_PACKETS`] is the
/// preroll that keeps an [`crate::AudioSampleReader`] from returning it.
pub struct NativeVorbisDecoder {
    decoder: VorbisDecoder,
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
        let mut extra_data = config.identification_header.clone();
        extra_data.extend_from_slice(&config.setup_header);
        let mut parameters = CodecParameters::new();
        parameters
            .for_codec(CODEC_TYPE_VORBIS)
            .with_extra_data(extra_data.into_boxed_slice());
        let decoder =
            VorbisDecoder::try_new(&parameters, &DecoderOptions::default()).map_err(|error| {
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
        if cancellation.is_canceled() {
            return Err(Error::new(ErrorKind::Canceled, "Vorbis decode canceled"));
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

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

/// The SIMD dispatch sites in this crate, each with the instruction set it
/// resolves to right now. `zvidlib::simd::active_by_site` reports every crate's
/// sites together and documents what each one covers.
#[doc(hidden)]
#[must_use]
pub fn simd_sites() -> Vec<(&'static str, zvidlib_core::SimdIsa)> {
    vec![("vorbis_decode", vorbis_simd::active_isa())]
}
