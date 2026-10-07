//! Symphonia AAC-LC backend for [`super::NativeAacDecoder`], on the native
//! targets whose operating system has no AAC decoder of its own: everything
//! but macOS and Windows.

use symphonia_codec_aac::AacDecoder as SymphoniaAacDecoder;
use symphonia_core::audio::{Channels, SampleBuffer};
use symphonia_core::codecs::{CODEC_TYPE_AAC, CodecParameters, Decoder as _, DecoderOptions};
use symphonia_core::formats::Packet;

use crate::{AacTrackConfig, Error, ErrorKind, Result};

pub(super) struct Decoder {
    decoder: SymphoniaAacDecoder,
    sample_rate: u32,
    channels: usize,
}

impl Decoder {
    pub(super) fn new(config: &AacTrackConfig) -> Result<Self> {
        let channels = match config.channels {
            1 => Channels::FRONT_LEFT,
            2 => Channels::FRONT_LEFT | Channels::FRONT_RIGHT,
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "the native AAC channel layout is unsupported",
                ));
            }
        };
        let mut parameters = CodecParameters::new();
        parameters
            .for_codec(CODEC_TYPE_AAC)
            .with_sample_rate(config.sample_rate)
            .with_channels(channels)
            .with_extra_data(config.audio_specific_config.clone().into_boxed_slice());
        let decoder = SymphoniaAacDecoder::try_new(&parameters, &DecoderOptions::default())
            .map_err(|error| {
                Error::new(
                    ErrorKind::Unsupported,
                    format!("native AAC configuration is unavailable: {error}"),
                )
            })?;
        Ok(Self {
            decoder,
            sample_rate: config.sample_rate,
            channels: usize::from(config.channels),
        })
    }

    /// Decodes one access unit starting at media sample `position`, appending
    /// its interleaved PCM to `out`.
    pub(super) fn decode(&mut self, data: &[u8], position: u64, out: &mut Vec<f32>) -> Result<()> {
        // Symphonia's AAC decoder reads only the payload: every access unit
        // decodes to one 1024-frame AAC-LC frame whatever duration it is given.
        let packet = Packet::new_from_slice(0, position, 1024, data);
        let decoded = self.decoder.decode(&packet).map_err(|error| {
            Error::new(
                ErrorKind::Codec,
                format!("malformed AAC access unit: {error}"),
            )
        })?;
        if decoded.spec().rate != self.sample_rate
            || decoded.spec().channels.count() != self.channels
        {
            return Err(Error::new(
                ErrorKind::Codec,
                "AAC decoder output format changed unexpectedly",
            ));
        }
        let mut interleaved = SampleBuffer::<f32>::new(decoded.frames() as u64, *decoded.spec());
        interleaved.copy_interleaved_ref(decoded);
        out.extend_from_slice(interleaved.samples());
        Ok(())
    }

    pub(super) fn reset(&mut self) -> Result<()> {
        self.decoder.reset();
        Ok(())
    }
}
