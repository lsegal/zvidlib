//! Windows Media Foundation AAC-LC backend for [`crate::AudioEncoder`].
//!
//! Encodes through Microsoft's AAC encoder MFT, a synchronous transform that
//! takes 16-bit interleaved PCM at 44.1 or 48 kHz, mono or stereo, and returns
//! one raw AAC-LC access unit (1024 PCM frames) per output sample. The
//! crate's `f32` input is converted to 16-bit on the way in.

use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::OnceLock;

use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFMediaType, IMFSample, IMFTransform, MF_E_NOTACCEPTING,
    MF_E_TRANSFORM_NEED_MORE_INPUT, MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION,
    MF_MT_AAC_PAYLOAD_TYPE, MF_MT_AUDIO_AVG_BYTES_PER_SECOND, MF_MT_AUDIO_BITS_PER_SAMPLE,
    MF_MT_AUDIO_BLOCK_ALIGNMENT, MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND,
    MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE, MF_VERSION, MFAudioFormat_AAC, MFAudioFormat_PCM,
    MFCreateMediaType, MFCreateMemoryBuffer, MFCreateSample, MFMediaType_Audio, MFStartup,
    MFT_CATEGORY_AUDIO_ENCODER, MFT_ENUM_FLAG_SORTANDFILTER, MFT_ENUM_FLAG_SYNCMFT,
    MFT_MESSAGE_COMMAND_DRAIN, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, MFT_MESSAGE_NOTIFY_END_OF_STREAM,
    MFT_MESSAGE_NOTIFY_END_STREAMING, MFT_MESSAGE_NOTIFY_START_OF_STREAM, MFT_OUTPUT_DATA_BUFFER,
    MFT_REGISTER_TYPE_INFO, MFTEnumEx,
};
use windows::Win32::System::Com::{CoIncrementMTAUsage, CoTaskMemFree};

use crate::{
    AudioBuffer, AudioDrain, AudioEncoder, AudioEncoderConfig, AudioEncoderFormat, AudioGapless,
    Codec, CodecImplementation, CodecSupport, EncodedSample, EncoderConfig, EncoderFuture, Error,
    ErrorKind, FrameIndex, Limits, Result, SampleDependency,
};

use super::{FRAME_LENGTH, esds_box, gapless_padding};

/// The average output rates, in bytes a second, the encoder MFT accepts for
/// `MF_MT_AUDIO_AVG_BYTES_PER_SECOND`: 96, 128, 160 and 192 kb/s.
const SUPPORTED_BYTES_PER_SECOND: [u32; 4] = [12_000, 16_000, 20_000, 24_000];

/// 192 kb/s, the highest rate the MFT offers, when the caller names none.
const DEFAULT_BYTES_PER_SECOND: u32 = 24_000;

/// `MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION` for AAC Profile, level 2
/// (ISO/IEC 14496-3 `audioProfileLevelIndication` `0x29`), the MFT's own
/// default; named explicitly so the output is AAC-LC however the default moves.
const AAC_PROFILE_L2: u32 = 0x29;

/// The MFT's encoder delay, in PCM frames, as a standard AAC decoder sees it.
///
/// Unlike AudioToolbox, the MFT exposes no property for this and does not
/// trim it from its output: the first access unit it emits starts at input
/// frame zero, so a decoder's own one-frame MDCT overlap is the whole delay
/// the track needs to cut. `tests/native_aac_encoder.rs` pins this with an
/// impulse through a full mux and demux.
const PRIMING_FRAMES: u32 = FRAME_LENGTH;

pub(super) fn capability(configuration: &AudioEncoderConfig, bit_rate: Option<u32>) -> CodecSupport {
    match open_transform(configuration, bit_rate) {
        Ok(_) => CodecSupport::Supported {
            implementation: CodecImplementation::Hardware,
        },
        Err(_) => CodecSupport::HardwareUnavailable,
    }
}

pub(super) fn create(
    configuration: &AudioEncoderConfig,
    bit_rate: Option<u32>,
    limits: &Limits,
) -> Result<Box<dyn AudioEncoder>> {
    let _ = limits;
    let (transform, output_size) = open_transform(configuration, bit_rate)?;
    unsafe {
        transform
            .ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)
            .map_err(|error| windows_error("the AAC encoder MFT would not start", error))?;
        transform
            .ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)
            .map_err(|error| windows_error("the AAC encoder MFT would not start", error))?;
    }
    Ok(Box::new(AacEncoder {
        transform,
        config: EncoderConfig {
            codec: Codec::Aac,
            timescale: configuration.sample_rate,
            decoder_config: esds_box(configuration.sample_rate, configuration.channels),
        },
        format: AudioEncoderFormat {
            sample_rate: configuration.sample_rate,
            channels: configuration.channels,
        },
        output_size,
        input_frames: 0,
        encoded_position: 0,
        finished: false,
    }))
}

/// Starts Media Foundation once for the process and keeps a multithreaded
/// apartment alive for it, so every thread that drives an encoder - including
/// one that never initialized COM itself - can create and call the MFT. Both
/// are deliberately never torn down: an encoder is `Send` and may be dropped
/// on any thread, which rules out pairing them with per-thread uninitialize
/// calls, and Media Foundation's startup is reference-counted process state.
fn ensure_media_foundation() -> Result<()> {
    static STARTED: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    STARTED
        .get_or_init(|| unsafe {
            CoIncrementMTAUsage()
                .map_err(|error| format!("could not initialize COM: {error}"))?;
            MFStartup(MF_VERSION, 0)
                .map_err(|error| format!("could not initialize Media Foundation: {error}"))
        })
        .clone()
        .map_err(|message| Error::new(ErrorKind::Unsupported, message))
}

/// Finds and configures the AAC encoder MFT for `configuration`, returning it
/// with the output buffer size each access unit needs.
fn open_transform(
    configuration: &AudioEncoderConfig,
    bit_rate: Option<u32>,
) -> Result<(IMFTransform, u32)> {
    if !matches!(configuration.sample_rate, 44_100 | 48_000) {
        return Err(unsupported(format!(
            "the Media Foundation AAC encoder takes 44.1 or 48 kHz input, not {} Hz",
            configuration.sample_rate
        )));
    }
    ensure_media_foundation()?;
    let transform = find_transform()?;
    let input = pcm_type(configuration)?;
    let output = aac_type(configuration, nearest_bytes_per_second(bit_rate))?;
    unsafe {
        transform
            .SetOutputType(0, &output, 0)
            .map_err(|error| windows_error("the AAC encoder MFT rejected its output type", error))?;
        transform
            .SetInputType(0, &input, 0)
            .map_err(|error| windows_error("the AAC encoder MFT rejected its input type", error))?;
        let info = transform
            .GetOutputStreamInfo(0)
            .map_err(|error| windows_error("the AAC encoder MFT has no output stream", error))?;
        Ok((transform, info.cbSize.max(8192)))
    }
}

fn find_transform() -> Result<IMFTransform> {
    let input = MFT_REGISTER_TYPE_INFO {
        guidMajorType: MFMediaType_Audio,
        guidSubtype: MFAudioFormat_PCM,
    };
    let output = MFT_REGISTER_TYPE_INFO {
        guidMajorType: MFMediaType_Audio,
        guidSubtype: MFAudioFormat_AAC,
    };
    let mut list: *mut Option<IMFActivate> = ptr::null_mut();
    let mut count = 0_u32;
    unsafe {
        MFTEnumEx(
            MFT_CATEGORY_AUDIO_ENCODER,
            MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_SORTANDFILTER,
            Some(&input),
            Some(&output),
            &mut list,
            &mut count,
        )
        .map_err(|error| windows_error("could not enumerate AAC encoder MFTs", error))?;
        // Take ownership of every activation object before freeing the array,
        // so each is released whether or not it is the one activated.
        let found = (0..count as usize)
            .filter_map(|index| (*list.add(index)).take())
            .collect::<Vec<_>>();
        CoTaskMemFree(Some(list.cast_const().cast()));
        let activate = found
            .first()
            .ok_or_else(|| unsupported("no Media Foundation AAC encoder is installed"))?;
        activate
            .ActivateObject()
            .map_err(|error| windows_error("could not activate the AAC encoder MFT", error))
    }
}

fn pcm_type(configuration: &AudioEncoderConfig) -> Result<IMFMediaType> {
    let block_alignment = u32::from(configuration.channels) * 2;
    unsafe {
        let media_type = MFCreateMediaType()
            .map_err(|error| windows_error("could not create a media type", error))?;
        media_type
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)
            .and_then(|()| media_type.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_PCM))
            .and_then(|()| media_type.SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16))
            .and_then(|()| {
                media_type.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, configuration.sample_rate)
            })
            .and_then(|()| {
                media_type.SetUINT32(
                    &MF_MT_AUDIO_NUM_CHANNELS,
                    u32::from(configuration.channels),
                )
            })
            .and_then(|()| media_type.SetUINT32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, block_alignment))
            .and_then(|()| {
                media_type.SetUINT32(
                    &MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
                    configuration.sample_rate * block_alignment,
                )
            })
            .map_err(|error| windows_error("could not describe the PCM input", error))?;
        Ok(media_type)
    }
}

fn aac_type(configuration: &AudioEncoderConfig, bytes_per_second: u32) -> Result<IMFMediaType> {
    unsafe {
        let media_type = MFCreateMediaType()
            .map_err(|error| windows_error("could not create a media type", error))?;
        media_type
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)
            .and_then(|()| media_type.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_AAC))
            .and_then(|()| media_type.SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16))
            .and_then(|()| {
                media_type.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, configuration.sample_rate)
            })
            .and_then(|()| {
                media_type.SetUINT32(
                    &MF_MT_AUDIO_NUM_CHANNELS,
                    u32::from(configuration.channels),
                )
            })
            .and_then(|()| media_type.SetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, bytes_per_second))
            // Payload type 0 is raw access units, which is what an MP4 sample
            // carries; the `AudioSpecificConfig` travels in the `esds` instead.
            .and_then(|()| media_type.SetUINT32(&MF_MT_AAC_PAYLOAD_TYPE, 0))
            .and_then(|()| {
                media_type.SetUINT32(&MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, AAC_PROFILE_L2)
            })
            .map_err(|error| windows_error("could not describe the AAC output", error))?;
        Ok(media_type)
    }
}

/// The MFT accepts only four output rates, so a requested bit rate is rounded
/// to the nearest of them; `None` takes [`DEFAULT_BYTES_PER_SECOND`].
fn nearest_bytes_per_second(bit_rate: Option<u32>) -> u32 {
    let Some(bits_per_second) = bit_rate else {
        return DEFAULT_BYTES_PER_SECOND;
    };
    let requested = bits_per_second / 8;
    SUPPORTED_BYTES_PER_SECOND
        .into_iter()
        .min_by_key(|supported| supported.abs_diff(requested))
        .unwrap_or(DEFAULT_BYTES_PER_SECOND)
}

pub(super) struct AacEncoder {
    transform: IMFTransform,
    config: EncoderConfig,
    format: AudioEncoderFormat,
    output_size: u32,
    /// Real PCM frames handed to the encoder; `finish` measures the padding
    /// against this.
    input_frames: u64,
    /// Total PCM frames covered by the access units emitted so far, in
    /// `timescale` (= sample rate) ticks; every [`EncodedSample`] is
    /// timestamped from this running clock.
    encoded_position: u64,
    finished: bool,
}

// The MFT is a free-threaded COM object living in the process's multithreaded
// apartment (see `ensure_media_foundation`), and it is only ever called from
// the thread driving this encoder's futures, one call at a time.
unsafe impl Send for AacEncoder {}

impl Drop for AacEncoder {
    fn drop(&mut self) {
        unsafe {
            let _ = self
                .transform
                .ProcessMessage(MFT_MESSAGE_NOTIFY_END_STREAMING, 0);
        }
    }
}

impl AacEncoder {
    fn submit(&mut self, samples: &[f32], out: &mut Vec<EncodedSample>) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        let channels = usize::from(self.format.channels);
        let frames = (samples.len() / channels) as u64;
        let sample = self.pcm_sample(samples, frames)?;
        self.input_frames += frames;
        match unsafe { self.transform.ProcessInput(0, &sample, 0) } {
            Ok(()) => {}
            Err(error) if error.code() == MF_E_NOTACCEPTING => {
                self.drain_output(out)?;
                unsafe { self.transform.ProcessInput(0, &sample, 0) }
                    .map_err(|error| windows_error("the AAC encoder MFT refused input", error))?;
            }
            Err(error) => {
                return Err(windows_error("the AAC encoder MFT refused input", error));
            }
        }
        self.drain_output(out)
    }

    fn pcm_sample(&self, samples: &[f32], frames: u64) -> Result<IMFSample> {
        let bytes = u32::try_from(samples.len() * 2)
            .map_err(|_| Error::new(ErrorKind::ResourceLimit, "audio buffer is too large"))?;
        let rate = i64::from(self.format.sample_rate);
        let hundred_nanoseconds = |frames: u64| i64::try_from(frames).unwrap_or(i64::MAX) * 10_000_000 / rate;
        unsafe {
            let buffer = MFCreateMemoryBuffer(bytes)
                .map_err(|error| windows_error("could not allocate a PCM buffer", error))?;
            let mut data = ptr::null_mut();
            buffer
                .Lock(&mut data, None, None)
                .map_err(|error| windows_error("could not lock a PCM buffer", error))?;
            let destination = std::slice::from_raw_parts_mut(data.cast::<i16>(), samples.len());
            for (destination, &sample) in destination.iter_mut().zip(samples) {
                *destination = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
            }
            buffer
                .Unlock()
                .map_err(|error| windows_error("could not unlock a PCM buffer", error))?;
            buffer
                .SetCurrentLength(bytes)
                .map_err(|error| windows_error("could not size a PCM buffer", error))?;
            let sample = MFCreateSample()
                .map_err(|error| windows_error("could not create a PCM sample", error))?;
            sample
                .AddBuffer(&buffer)
                .and_then(|()| sample.SetSampleTime(hundred_nanoseconds(self.input_frames)))
                .and_then(|()| sample.SetSampleDuration(hundred_nanoseconds(frames)))
                .map_err(|error| windows_error("could not describe a PCM sample", error))?;
            Ok(sample)
        }
    }

    /// Collects every access unit the MFT has ready, stopping when it asks
    /// for more input.
    fn drain_output(&mut self, out: &mut Vec<EncodedSample>) -> Result<()> {
        loop {
            let packet = unsafe {
                let buffer = MFCreateMemoryBuffer(self.output_size)
                    .map_err(|error| windows_error("could not allocate an AAC buffer", error))?;
                let sample = MFCreateSample()
                    .map_err(|error| windows_error("could not create an AAC sample", error))?;
                sample
                    .AddBuffer(&buffer)
                    .map_err(|error| windows_error("could not create an AAC sample", error))?;
                let mut buffers = [MFT_OUTPUT_DATA_BUFFER {
                    dwStreamID: 0,
                    pSample: ManuallyDrop::new(Some(sample)),
                    dwStatus: 0,
                    pEvents: ManuallyDrop::new(None),
                }];
                let mut status = 0;
                let result = self.transform.ProcessOutput(0, &mut buffers, &mut status);
                let sample = ManuallyDrop::take(&mut buffers[0].pSample);
                drop(ManuallyDrop::take(&mut buffers[0].pEvents));
                match result {
                    Ok(()) => {}
                    Err(error) if error.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => return Ok(()),
                    Err(error) => {
                        return Err(windows_error("the AAC encoder MFT failed to encode", error));
                    }
                }
                let Some(sample) = sample else { continue };
                let buffer = sample
                    .ConvertToContiguousBuffer()
                    .map_err(|error| windows_error("could not read an AAC sample", error))?;
                let mut data = ptr::null_mut();
                let mut length = 0;
                buffer
                    .Lock(&mut data, None, Some(&mut length))
                    .map_err(|error| windows_error("could not lock an AAC buffer", error))?;
                let packet = std::slice::from_raw_parts(data, length as usize).to_vec();
                buffer
                    .Unlock()
                    .map_err(|error| windows_error("could not unlock an AAC buffer", error))?;
                packet
            };
            if packet.is_empty() {
                continue;
            }
            let dts = i64::try_from(self.encoded_position)
                .map_err(|_| Error::new(ErrorKind::ResourceLimit, "AAC sample position overflowed"))?;
            out.push(EncodedSample {
                data: packet,
                dts,
                pts: dts,
                duration: FRAME_LENGTH,
                is_sync: true,
                dependency: SampleDependency::INDEPENDENT,
            });
            self.encoded_position += u64::from(FRAME_LENGTH);
        }
    }
}

impl AudioEncoder for AacEncoder {
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
            if buffer.sample_rate != self.format.sample_rate
                || buffer.channels != self.format.channels
            {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "audio buffer format does not match the AAC encoder",
                ));
            }
            let mut samples = Vec::new();
            self.submit(&buffer.samples, &mut samples)?;
            Ok(samples)
        })
    }

    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, AudioDrain> {
        Box::pin(async move {
            if self.finished {
                return Ok(AudioDrain {
                    samples: Vec::new(),
                    gapless: AudioGapless::default(),
                });
            }
            self.finished = true;
            // Draining makes the MFT encode whatever it still holds, silence
            // filling out the final partial frame, so the emitted access units
            // are the whole stream and the padding is measured from them.
            let mut samples = Vec::new();
            unsafe {
                self.transform
                    .ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0)
                    .and_then(|()| self.transform.ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0))
                    .map_err(|error| windows_error("the AAC encoder MFT would not drain", error))?;
            }
            self.drain_output(&mut samples)?;
            let padding =
                gapless_padding(self.encoded_position, PRIMING_FRAMES, self.input_frames)?;
            Ok(AudioDrain {
                samples,
                gapless: AudioGapless {
                    priming: PRIMING_FRAMES,
                    padding,
                },
            })
        })
    }
}

fn windows_error(context: &str, error: windows::core::Error) -> Error {
    Error::new(ErrorKind::Codec, format!("{context}: {error}"))
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_rates_round_to_the_nearest_rate_the_mft_offers() {
        assert_eq!(nearest_bytes_per_second(None), 24_000);
        assert_eq!(nearest_bytes_per_second(Some(96_000)), 12_000);
        assert_eq!(nearest_bytes_per_second(Some(64_000)), 12_000);
        assert_eq!(nearest_bytes_per_second(Some(130_000)), 16_000);
        assert_eq!(nearest_bytes_per_second(Some(320_000)), 24_000);
    }
}
