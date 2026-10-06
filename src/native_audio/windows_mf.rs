//! Windows Media Foundation AAC-LC backend for [`super::NativeAacDecoder`].
//!
//! Decodes through Microsoft's AAC decoder MFT, a synchronous transform that
//! takes raw AAC access units (`MF_MT_AAC_PAYLOAD_TYPE` 0) configured from the
//! track's `AudioSpecificConfig` and returns interleaved 32-bit float PCM.
//! Its output trails its input by one access unit, which
//! [`super::NativeAacDecoder`] compensates for.

use std::mem::ManuallyDrop;
use std::ptr;

use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFMediaType, IMFSample, IMFTransform, MF_E_TRANSFORM_NEED_MORE_INPUT,
    MF_E_TRANSFORM_STREAM_CHANGE, MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, MF_MT_AAC_PAYLOAD_TYPE,
    MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE,
    MF_MT_USER_DATA, MFAudioFormat_AAC, MFAudioFormat_Float, MFCreateMediaType,
    MFCreateMemoryBuffer, MFCreateSample, MFMediaType_Audio, MFT_CATEGORY_AUDIO_DECODER,
    MFT_ENUM_FLAG_SORTANDFILTER, MFT_ENUM_FLAG_SYNCMFT, MFT_MESSAGE_COMMAND_FLUSH,
    MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, MFT_MESSAGE_NOTIFY_END_STREAMING,
    MFT_MESSAGE_NOTIFY_START_OF_STREAM, MFT_OUTPUT_DATA_BUFFER, MFT_REGISTER_TYPE_INFO, MFTEnumEx,
};
use windows::Win32::System::Com::CoTaskMemFree;

use crate::aac_encoder::windows_mf::{ensure_media_foundation, windows_error};
use crate::{AacTrackConfig, Error, ErrorKind, Result};

/// `audioProfileLevelIndication` `0xFE`, "no audio profile specified": the
/// `AudioSpecificConfig` in the user data is what describes the stream.
const PROFILE_LEVEL_UNSPECIFIED: u16 = 0xfe;

/// The output buffer to offer when the MFT names no size: a 1024-frame stereo
/// access unit of `f32` is 8 KiB, and this leaves room for twice that.
const MIN_OUTPUT_BYTES: u32 = 16_384;

pub(super) struct Decoder {
    transform: IMFTransform,
    output_size: u32,
    sample_rate: u32,
}

// The MFT is a free-threaded COM object living in the process's multithreaded
// apartment (see `ensure_media_foundation`), and it is only ever called from
// the thread driving this decoder, one call at a time.
unsafe impl Send for Decoder {}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            let _ = self
                .transform
                .ProcessMessage(MFT_MESSAGE_NOTIFY_END_STREAMING, 0);
        }
    }
}

impl Decoder {
    pub(super) fn new(config: &AacTrackConfig) -> Result<Self> {
        ensure_media_foundation()?;
        let transform = find_transform()?;
        let input = aac_type(config)?;
        unsafe {
            transform.SetInputType(0, &input, 0).map_err(|error| {
                unsupported_error(
                    "the AAC decoder MFT rejected the track's configuration",
                    error,
                )
            })?;
            let output = float_output_type(&transform, config.sample_rate, config.channels)?;
            transform.SetOutputType(0, &output, 0).map_err(|error| {
                unsupported_error("the AAC decoder MFT cannot output float PCM", error)
            })?;
            let info = transform.GetOutputStreamInfo(0).map_err(|error| {
                windows_error("the AAC decoder MFT has no output stream", error)
            })?;
            transform
                .ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)
                .and_then(|()| transform.ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0))
                .map_err(|error| windows_error("the AAC decoder MFT would not start", error))?;
            Ok(Self {
                transform,
                output_size: info.cbSize.max(MIN_OUTPUT_BYTES),
                sample_rate: config.sample_rate,
            })
        }
    }

    /// Decodes one access unit starting at media sample `position`, appending
    /// whatever interleaved PCM the MFT emits for it to `out`.
    pub(super) fn decode(&mut self, data: &[u8], position: u64, out: &mut Vec<f32>) -> Result<()> {
        let sample = self.access_unit(data, position)?;
        self.process(&sample, out)
    }

    fn process(&mut self, sample: &IMFSample, out: &mut Vec<f32>) -> Result<()> {
        unsafe { self.transform.ProcessInput(0, sample, 0) }
            .map_err(|error| windows_error("the AAC decoder MFT refused an access unit", error))?;
        self.drain_output(out)
    }

    pub(super) fn reset(&mut self) -> Result<()> {
        unsafe { self.transform.ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0) }
            .map_err(|error| windows_error("the AAC decoder MFT would not flush", error))
    }

    fn access_unit(&self, data: &[u8], position: u64) -> Result<IMFSample> {
        let bytes = u32::try_from(data.len())
            .map_err(|_| Error::new(ErrorKind::ResourceLimit, "AAC access unit is too large"))?;
        let time = i64::try_from(position).unwrap_or(i64::MAX / 10_000_000) * 10_000_000
            / i64::from(self.sample_rate);
        unsafe {
            let buffer = MFCreateMemoryBuffer(bytes.max(1))
                .map_err(|error| windows_error("could not allocate an AAC buffer", error))?;
            let mut destination = ptr::null_mut();
            buffer
                .Lock(&mut destination, None, None)
                .map_err(|error| windows_error("could not lock an AAC buffer", error))?;
            ptr::copy_nonoverlapping(data.as_ptr(), destination, data.len());
            buffer
                .Unlock()
                .map_err(|error| windows_error("could not unlock an AAC buffer", error))?;
            buffer
                .SetCurrentLength(bytes)
                .map_err(|error| windows_error("could not size an AAC buffer", error))?;
            let sample = MFCreateSample()
                .map_err(|error| windows_error("could not create an AAC sample", error))?;
            sample
                .AddBuffer(&buffer)
                .and_then(|()| sample.SetSampleTime(time))
                .map_err(|error| windows_error("could not describe an AAC sample", error))?;
            Ok(sample)
        }
    }

    /// Collects every PCM buffer the MFT has ready, stopping when it asks for
    /// more input.
    fn drain_output(&mut self, out: &mut Vec<f32>) -> Result<()> {
        loop {
            unsafe {
                let buffer = MFCreateMemoryBuffer(self.output_size)
                    .map_err(|error| windows_error("could not allocate a PCM buffer", error))?;
                let sample = MFCreateSample()
                    .map_err(|error| windows_error("could not create a PCM sample", error))?;
                sample
                    .AddBuffer(&buffer)
                    .map_err(|error| windows_error("could not create a PCM sample", error))?;
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
                    // The output type is fixed to the track's rate and channel
                    // count, so a stream that wants another one is not the
                    // stream the track described.
                    Err(error) if error.code() == MF_E_TRANSFORM_STREAM_CHANGE => {
                        return Err(Error::new(
                            ErrorKind::Codec,
                            "AAC decoder output format changed unexpectedly",
                        ));
                    }
                    Err(error) => {
                        return Err(Error::new(
                            ErrorKind::Codec,
                            format!("malformed AAC access unit: {error}"),
                        ));
                    }
                }
                let Some(sample) = sample else { continue };
                let buffer = sample
                    .ConvertToContiguousBuffer()
                    .map_err(|error| windows_error("could not read a PCM sample", error))?;
                let mut data = ptr::null_mut();
                let mut length = 0;
                buffer
                    .Lock(&mut data, None, Some(&mut length))
                    .map_err(|error| windows_error("could not lock a PCM buffer", error))?;
                let pcm = std::slice::from_raw_parts(
                    data.cast::<f32>(),
                    length as usize / std::mem::size_of::<f32>(),
                );
                out.extend_from_slice(pcm);
                buffer
                    .Unlock()
                    .map_err(|error| windows_error("could not unlock a PCM buffer", error))?;
            }
        }
    }
}

fn find_transform() -> Result<IMFTransform> {
    let input = MFT_REGISTER_TYPE_INFO {
        guidMajorType: MFMediaType_Audio,
        guidSubtype: MFAudioFormat_AAC,
    };
    let output = MFT_REGISTER_TYPE_INFO {
        guidMajorType: MFMediaType_Audio,
        guidSubtype: MFAudioFormat_Float,
    };
    let mut list: *mut Option<IMFActivate> = ptr::null_mut();
    let mut count = 0_u32;
    unsafe {
        MFTEnumEx(
            MFT_CATEGORY_AUDIO_DECODER,
            MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_SORTANDFILTER,
            Some(&input),
            Some(&output),
            &mut list,
            &mut count,
        )
        .map_err(|error| windows_error("could not enumerate AAC decoder MFTs", error))?;
        // Take ownership of every activation object before freeing the array,
        // so each is released whether or not it is the one activated.
        let found = (0..count as usize)
            .filter_map(|index| (*list.add(index)).take())
            .collect::<Vec<_>>();
        CoTaskMemFree(Some(list.cast_const().cast()));
        let activate = found.first().ok_or_else(|| {
            Error::new(
                ErrorKind::Unsupported,
                "no Media Foundation AAC decoder is installed",
            )
        })?;
        activate
            .ActivateObject()
            .map_err(|error| unsupported_error("could not activate the AAC decoder MFT", error))
    }
}

fn aac_type(config: &AacTrackConfig) -> Result<IMFMediaType> {
    let user_data = heaac_user_data(&config.audio_specific_config);
    unsafe {
        let media_type = MFCreateMediaType()
            .map_err(|error| windows_error("could not create a media type", error))?;
        media_type
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)
            .and_then(|()| media_type.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_AAC))
            .and_then(|()| {
                media_type.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, config.sample_rate)
            })
            .and_then(|()| {
                media_type.SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, u32::from(config.channels))
            })
            // Payload type 0 is raw access units, which is what an MP4 sample
            // carries.
            .and_then(|()| media_type.SetUINT32(&MF_MT_AAC_PAYLOAD_TYPE, 0))
            .and_then(|()| {
                media_type.SetUINT32(
                    &MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION,
                    u32::from(PROFILE_LEVEL_UNSPECIFIED),
                )
            })
            .and_then(|()| media_type.SetBlob(&MF_MT_USER_DATA, &user_data))
            .map_err(|error| windows_error("could not describe the AAC input", error))?;
        Ok(media_type)
    }
}

/// `MF_MT_USER_DATA` for an AAC input type: the `HEAACWAVEINFO` fields that
/// follow its `WAVEFORMATEX` header, then the `AudioSpecificConfig`.
fn heaac_user_data(audio_specific_config: &[u8]) -> Vec<u8> {
    let mut data = Vec::with_capacity(12 + audio_specific_config.len());
    // wPayloadType: raw access units.
    data.extend_from_slice(&0_u16.to_le_bytes());
    data.extend_from_slice(&PROFILE_LEVEL_UNSPECIFIED.to_le_bytes());
    // wStructType 0: the `AudioSpecificConfig` follows directly.
    data.extend_from_slice(&0_u16.to_le_bytes());
    // wReserved1 and dwReserved2.
    data.extend_from_slice(&0_u16.to_le_bytes());
    data.extend_from_slice(&0_u32.to_le_bytes());
    data.extend_from_slice(audio_specific_config);
    data
}

/// Picks the float PCM type the MFT offers at the track's own rate and
/// channel count. The MFT completes its offered types with attributes of its
/// own, such as the channel mask, and rejects a hand-built one without them.
fn float_output_type(
    transform: &IMFTransform,
    sample_rate: u32,
    channels: u16,
) -> Result<IMFMediaType> {
    let mut offered = Vec::new();
    for index in 0.. {
        let Ok(media_type) = (unsafe { transform.GetOutputAvailableType(0, index) }) else {
            break;
        };
        let (subtype, rate, count) = unsafe {
            (
                media_type.GetGUID(&MF_MT_SUBTYPE).ok(),
                media_type.GetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND).ok(),
                media_type.GetUINT32(&MF_MT_AUDIO_NUM_CHANNELS).ok(),
            )
        };
        if subtype == Some(MFAudioFormat_Float)
            && rate == Some(sample_rate)
            && count == Some(u32::from(channels))
        {
            return Ok(media_type);
        }
        offered.push(format!(
            "{} Hz x {}{}",
            rate.unwrap_or(0),
            count.unwrap_or(0),
            if subtype == Some(MFAudioFormat_Float) {
                " float"
            } else {
                ""
            }
        ));
    }
    Err(Error::new(
        ErrorKind::Unsupported,
        format!(
            "the AAC decoder MFT offers no {sample_rate} Hz {channels}-channel float output              (offered: {})",
            offered.join(", ")
        ),
    ))
}

fn unsupported_error(context: &str, error: windows::core::Error) -> Error {
    Error::new(ErrorKind::Unsupported, format!("{context}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_data_is_the_heaacwaveinfo_tail_then_the_audio_specific_config() {
        assert_eq!(
            heaac_user_data(&[0x11, 0x90]),
            [0, 0, 0xfe, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x11, 0x90]
        );
    }
}
