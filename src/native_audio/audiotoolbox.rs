//! macOS AudioToolbox AAC-LC backend for [`super::NativeAacDecoder`].
//!
//! Decodes through `AudioConverter`, configured from the track's
//! `AudioSpecificConfig` as its decompression magic cookie, into interleaved
//! 32-bit float PCM. Each [`Decoder::decode`] call hands the converter exactly
//! one access unit, with its `AudioStreamPacketDescription`, through
//! [`input_proc`], and asks for the 1024 frames it decodes to.

use std::ffi::c_void;
use std::ptr;

use crate::aac_encoder::audiotoolbox::{
    AudioBufferList, AudioBufferStruct, AudioConverterDispose, AudioConverterFillComplexBuffer,
    AudioConverterNew, AudioConverterRef, AudioConverterReset, AudioConverterSetProperty,
    AudioStreamBasicDescription, AudioStreamPacketDescription, K_AUDIO_FORMAT_FLAG_IS_FLOAT,
    K_AUDIO_FORMAT_FLAG_IS_PACKED, K_AUDIO_FORMAT_LINEAR_PCM, K_AUDIO_FORMAT_MPEG4_AAC,
    NO_MORE_INPUT, OSStatus, fourcc, status_error,
};
use crate::{AacTrackConfig, Error, ErrorKind, Result};

const K_AUDIO_CONVERTER_DECOMPRESSION_MAGIC_COOKIE: u32 = fourcc(b"dmgc");

/// AAC-LC always decodes 1024 samples per access unit.
const FRAME_LENGTH: u32 = 1024;

pub(super) struct Decoder {
    converter: AudioConverterRef,
    channels: u32,
}

// The raw `AudioConverterRef` is only ever touched from the thread driving this
// decoder, synchronously and one call at a time, so there is nothing here for
// another thread to race with even though the pointer itself carries no such
// guarantee.
unsafe impl Send for Decoder {}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            AudioConverterDispose(self.converter);
        }
    }
}

impl Decoder {
    pub(super) fn new(config: &AacTrackConfig) -> Result<Self> {
        let channels = u32::from(config.channels);
        let source_format = AudioStreamBasicDescription {
            sample_rate: f64::from(config.sample_rate),
            format_id: K_AUDIO_FORMAT_MPEG4_AAC,
            format_flags: 0,
            bytes_per_packet: 0,
            frames_per_packet: FRAME_LENGTH,
            bytes_per_frame: 0,
            channels_per_frame: channels,
            bits_per_channel: 0,
            reserved: 0,
        };
        let destination_format = AudioStreamBasicDescription {
            sample_rate: f64::from(config.sample_rate),
            format_id: K_AUDIO_FORMAT_LINEAR_PCM,
            format_flags: K_AUDIO_FORMAT_FLAG_IS_FLOAT | K_AUDIO_FORMAT_FLAG_IS_PACKED,
            bytes_per_packet: 4 * channels,
            frames_per_packet: 1,
            bytes_per_frame: 4 * channels,
            channels_per_frame: channels,
            bits_per_channel: 32,
            reserved: 0,
        };
        let mut converter: AudioConverterRef = ptr::null_mut();
        let status =
            unsafe { AudioConverterNew(&source_format, &destination_format, &mut converter) };
        if status != 0 || converter.is_null() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!("AudioConverterNew could not create an AAC decoder (OSStatus {status})"),
            ));
        }
        // Owned from here, so an error below still disposes of it.
        let decoder = Self {
            converter,
            channels,
        };
        let cookie = magic_cookie(&config.audio_specific_config);
        let status = unsafe {
            AudioConverterSetProperty(
                decoder.converter,
                K_AUDIO_CONVERTER_DECOMPRESSION_MAGIC_COOKIE,
                u32::try_from(cookie.len()).unwrap_or(u32::MAX),
                cookie.as_ptr().cast(),
            )
        };
        if status != 0 {
            return Err(Error::new(
                ErrorKind::Unsupported,
                format!("AudioToolbox rejected the AAC configuration (OSStatus {status})"),
            ));
        }
        Ok(decoder)
    }

    /// Decodes one access unit, appending its interleaved PCM to `out`. The
    /// converter keeps its own stream position, so `_position` is unused.
    pub(super) fn decode(&mut self, data: &[u8], _position: u64, out: &mut Vec<f32>) -> Result<()> {
        let mut input = PendingPacket {
            data: data.as_ptr(),
            description: AudioStreamPacketDescription {
                start_offset: 0,
                variable_frames_in_packet: 0,
                data_byte_size: u32::try_from(data.len()).map_err(|_| {
                    Error::new(ErrorKind::ResourceLimit, "AAC access unit is too large")
                })?,
            },
            channels: self.channels,
            consumed: false,
        };
        let start = out.len();
        let capacity = FRAME_LENGTH as usize * self.channels as usize;
        out.resize(start + capacity, 0.0);
        let mut frames = FRAME_LENGTH;
        let mut buffer_list = AudioBufferList {
            number_buffers: 1,
            buffers: [AudioBufferStruct {
                number_channels: self.channels,
                data_byte_size: u32::try_from(capacity * std::mem::size_of::<f32>())
                    .unwrap_or(u32::MAX),
                data: out[start..].as_mut_ptr().cast(),
            }],
        };
        let status = unsafe {
            AudioConverterFillComplexBuffer(
                self.converter,
                input_proc,
                (&raw mut input).cast(),
                &mut frames,
                &mut buffer_list,
                ptr::null_mut(),
            )
        };
        out.truncate(start + frames as usize * self.channels as usize);
        if status != 0 && status != NO_MORE_INPUT {
            return Err(Error::new(
                ErrorKind::Codec,
                format!("malformed AAC access unit (OSStatus {status})"),
            ));
        }
        Ok(())
    }

    pub(super) fn reset(&mut self) -> Result<()> {
        let status: OSStatus = unsafe { AudioConverterReset(self.converter) };
        if status == 0 {
            Ok(())
        } else {
            Err(status_error(
                status,
                "AudioConverterReset could not reset the AAC decoder",
            ))
        }
    }
}

/// The single access unit offered to [`input_proc`] for one
/// `AudioConverterFillComplexBuffer` call. The description lives here so the
/// pointer handed back to the converter stays valid for the whole call.
struct PendingPacket {
    data: *const u8,
    description: AudioStreamPacketDescription,
    channels: u32,
    consumed: bool,
}

extern "C" fn input_proc(
    _converter: AudioConverterRef,
    io_number_data_packets: *mut u32,
    io_data: *mut AudioBufferList,
    out_data_packet_description: *mut *mut AudioStreamPacketDescription,
    in_user_data: *mut c_void,
) -> OSStatus {
    // Safety: `in_user_data` is the `&mut PendingPacket` `Decoder::decode`
    // passed to `AudioConverterFillComplexBuffer` for the duration of that
    // call, and this callback only runs synchronously within it.
    let state = unsafe { &mut *in_user_data.cast::<PendingPacket>() };
    unsafe {
        (*io_data).number_buffers = 1;
        (*io_data).buffers[0].number_channels = state.channels;
    }
    if state.consumed {
        // The one access unit has been handed over; asking for another is how
        // the converter learns this call has no more input.
        unsafe {
            *io_number_data_packets = 0;
            (*io_data).buffers[0].data_byte_size = 0;
            (*io_data).buffers[0].data = ptr::null_mut();
            if !out_data_packet_description.is_null() {
                *out_data_packet_description = ptr::null_mut();
            }
        }
        return NO_MORE_INPUT;
    }
    state.consumed = true;
    unsafe {
        *io_number_data_packets = 1;
        (*io_data).buffers[0].data_byte_size = state.description.data_byte_size;
        (*io_data).buffers[0].data = state.data.cast_mut().cast();
        if !out_data_packet_description.is_null() {
            *out_data_packet_description = &raw mut state.description;
        }
    }
    0
}

/// The decompression magic cookie AudioToolbox takes for MPEG-4 AAC: an
/// `ES_Descriptor` wrapping the `AudioSpecificConfig` (ISO/IEC 14496-1
/// 7.2.6.5), the same contents an MP4 `esds` box carries after its version.
fn magic_cookie(audio_specific_config: &[u8]) -> Vec<u8> {
    let specific = descriptor(0x05, audio_specific_config);
    // objectTypeIndication 0x40 (MPEG-4 audio), streamType 5 (audio) shifted
    // with the reserved bit, then a zero buffer size and bit rates.
    let mut decoder_config = vec![0x40, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    decoder_config.extend_from_slice(&specific);
    let mut es = vec![0, 0, 0];
    es.extend_from_slice(&descriptor(0x04, &decoder_config));
    descriptor(0x03, &es)
}

/// A descriptor with its size in the four-byte expandable form.
fn descriptor(tag: u8, body: &[u8]) -> Vec<u8> {
    let size = body.len() as u32;
    let mut out = vec![
        tag,
        0x80 | ((size >> 21) & 0x7f) as u8,
        0x80 | ((size >> 14) & 0x7f) as u8,
        0x80 | ((size >> 7) & 0x7f) as u8,
        (size & 0x7f) as u8,
    ];
    out.extend_from_slice(body);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_cookie_wraps_the_audio_specific_config_in_an_es_descriptor() {
        assert_eq!(
            magic_cookie(&[0x11, 0x90]),
            [
                0x03, 0x80, 0x80, 0x80, 0x1c, 0, 0, 0, //
                0x04, 0x80, 0x80, 0x80, 0x14, //
                0x40, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
                0x05, 0x80, 0x80, 0x80, 0x02, 0x11, 0x90,
            ]
        );
    }
}
