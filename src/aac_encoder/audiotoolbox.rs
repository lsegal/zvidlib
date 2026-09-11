//! macOS AudioToolbox AAC-LC backend for [`crate::AudioEncoder`].
//!
//! Encodes 32-bit float interleaved PCM to AAC-LC through `AudioConverter`'s
//! pull model: [`AudioConverterFillComplexBuffer`] calls [`input_proc`] back
//! for source frames as it needs them, one 1024-sample AAC-LC frame per call
//! here since the source and destination sample rates always match (no
//! resampling is asked of the converter).

use std::ffi::c_void;
use std::ptr;

use crate::{
    AudioBuffer, AudioDrain, AudioEncoder, AudioEncoderConfig, AudioEncoderFormat, AudioGapless,
    Codec, CodecImplementation, CodecSupport, EncodedSample, EncoderConfig, EncoderFuture, Error,
    ErrorKind, FrameIndex, Limits, Result, SampleDependency,
};

use super::esds_box;

type OSStatus = i32;
type AudioConverterRef = *mut c_void;

/// AAC-LC always encodes 1024 samples per frame.
const FRAME_LENGTH: u32 = 1024;

/// A sentinel this module's own [`input_proc`] returns from
/// `AudioConverterFillComplexBuffer`'s pull callback to say no more source
/// frames are available for the current call. Not a system `OSStatus`: the
/// converter passes whatever the callback returns straight back to its
/// caller, so this only ever needs to be distinguishable from `0` (`noErr`).
const NO_MORE_INPUT: OSStatus = 1;

#[repr(C)]
#[derive(Clone, Copy)]
struct AudioStreamBasicDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

#[repr(C)]
struct AudioBufferStruct {
    number_channels: u32,
    data_byte_size: u32,
    data: *mut c_void,
}

#[repr(C)]
struct AudioBufferList {
    number_buffers: u32,
    buffers: [AudioBufferStruct; 1],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AudioStreamPacketDescription {
    start_offset: i64,
    variable_frames_in_packet: u32,
    data_byte_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AudioConverterPrimeInfo {
    leading_frames: u32,
    trailing_frames: u32,
}

type InputProc = extern "C" fn(
    AudioConverterRef,
    *mut u32,
    *mut AudioBufferList,
    *mut *mut AudioStreamPacketDescription,
    *mut c_void,
) -> OSStatus;

#[link(name = "AudioToolbox", kind = "framework")]
unsafe extern "C" {
    fn AudioConverterNew(
        in_source_format: *const AudioStreamBasicDescription,
        in_destination_format: *const AudioStreamBasicDescription,
        out_audio_converter: *mut AudioConverterRef,
    ) -> OSStatus;

    fn AudioConverterDispose(in_audio_converter: AudioConverterRef) -> OSStatus;

    fn AudioConverterFillComplexBuffer(
        in_audio_converter: AudioConverterRef,
        in_input_data_proc: InputProc,
        in_input_data_proc_user_data: *mut c_void,
        io_output_data_packet_size: *mut u32,
        out_output_data: *mut AudioBufferList,
        out_packet_description: *mut AudioStreamPacketDescription,
    ) -> OSStatus;

    fn AudioConverterGetProperty(
        in_audio_converter: AudioConverterRef,
        in_property_id: u32,
        io_property_data_size: *mut u32,
        out_property_data: *mut c_void,
    ) -> OSStatus;
}

const fn fourcc(bytes: &[u8; 4]) -> u32 {
    ((bytes[0] as u32) << 24) | ((bytes[1] as u32) << 16) | ((bytes[2] as u32) << 8) | bytes[3] as u32
}

const K_AUDIO_FORMAT_LINEAR_PCM: u32 = fourcc(b"lpcm");
const K_AUDIO_FORMAT_MPEG4_AAC: u32 = fourcc(b"aac ");
const K_AUDIO_FORMAT_FLAG_IS_FLOAT: u32 = 1 << 0;
const K_AUDIO_FORMAT_FLAG_IS_PACKED: u32 = 1 << 3;
const K_AUDIO_CONVERTER_PROPERTY_MAXIMUM_OUTPUT_PACKET_SIZE: u32 = fourcc(b"xops");
const K_AUDIO_CONVERTER_PROPERTY_PRIME_INFO: u32 = fourcc(b"prim");

/// Generous fallback when [`K_AUDIO_CONVERTER_PROPERTY_MAXIMUM_OUTPUT_PACKET_SIZE`]
/// cannot be read: several times the largest AAC-LC frame AudioToolbox produces
/// at any bit rate this encoder would plausibly be asked for.
const FALLBACK_MAX_PACKET_SIZE: u32 = 8192;

pub(super) fn capability(configuration: &AudioEncoderConfig) -> CodecSupport {
    match new_converter(configuration) {
        Ok(converter) => {
            unsafe {
                AudioConverterDispose(converter);
            }
            CodecSupport::Supported {
                implementation: CodecImplementation::Hardware,
            }
        }
        Err(_) => CodecSupport::HardwareUnavailable,
    }
}

pub(super) fn create(
    configuration: &AudioEncoderConfig,
    limits: &Limits,
) -> Result<Box<dyn AudioEncoder>> {
    let _ = limits;
    let converter = new_converter(configuration).map_err(|status| {
        status_error(status, "AudioConverterNew could not create an AAC encoder")
    })?;
    let max_packet_size =
        query_max_output_packet_size(converter).unwrap_or(FALLBACK_MAX_PACKET_SIZE);
    Ok(Box::new(AacEncoder {
        converter,
        config: EncoderConfig {
            codec: Codec::Aac,
            timescale: configuration.sample_rate,
            decoder_config: esds_box(configuration.sample_rate, configuration.channels),
        },
        format: AudioEncoderFormat {
            sample_rate: configuration.sample_rate,
            channels: configuration.channels,
        },
        channels: u32::from(configuration.channels),
        max_packet_size,
        pending: Vec::new(),
        encoded_position: 0,
        finished: false,
    }))
}

fn new_converter(configuration: &AudioEncoderConfig) -> std::result::Result<AudioConverterRef, OSStatus> {
    let source_format = AudioStreamBasicDescription {
        sample_rate: f64::from(configuration.sample_rate),
        format_id: K_AUDIO_FORMAT_LINEAR_PCM,
        format_flags: K_AUDIO_FORMAT_FLAG_IS_FLOAT | K_AUDIO_FORMAT_FLAG_IS_PACKED,
        bytes_per_packet: 4 * u32::from(configuration.channels),
        frames_per_packet: 1,
        bytes_per_frame: 4 * u32::from(configuration.channels),
        channels_per_frame: u32::from(configuration.channels),
        bits_per_channel: 32,
        reserved: 0,
    };
    let destination_format = AudioStreamBasicDescription {
        sample_rate: f64::from(configuration.sample_rate),
        format_id: K_AUDIO_FORMAT_MPEG4_AAC,
        format_flags: 0,
        bytes_per_packet: 0,
        frames_per_packet: FRAME_LENGTH,
        bytes_per_frame: 0,
        channels_per_frame: u32::from(configuration.channels),
        bits_per_channel: 0,
        reserved: 0,
    };
    let mut converter: AudioConverterRef = ptr::null_mut();
    let status =
        unsafe { AudioConverterNew(&source_format, &destination_format, &mut converter) };
    if status != 0 || converter.is_null() {
        return Err(status);
    }
    Ok(converter)
}

fn query_max_output_packet_size(converter: AudioConverterRef) -> Option<u32> {
    let mut value: u32 = 0;
    let mut size = u32::try_from(std::mem::size_of::<u32>()).ok()?;
    let status = unsafe {
        AudioConverterGetProperty(
            converter,
            K_AUDIO_CONVERTER_PROPERTY_MAXIMUM_OUTPUT_PACKET_SIZE,
            &mut size,
            (&mut value as *mut u32).cast(),
        )
    };
    (status == 0 && value > 0).then_some(value)
}

fn query_priming_frames(converter: AudioConverterRef) -> u32 {
    let mut value = AudioConverterPrimeInfo::default();
    let mut size = match u32::try_from(std::mem::size_of::<AudioConverterPrimeInfo>()) {
        Ok(size) => size,
        Err(_) => return 0,
    };
    let status = unsafe {
        AudioConverterGetProperty(
            converter,
            K_AUDIO_CONVERTER_PROPERTY_PRIME_INFO,
            &mut size,
            (&mut value as *mut AudioConverterPrimeInfo).cast(),
        )
    };
    if status == 0 { value.leading_frames } else { 0 }
}

fn status_error(status: OSStatus, context: &str) -> Error {
    Error::new(
        ErrorKind::Codec,
        format!("{context} (OSStatus {status})"),
    )
}

/// PCM frames handed to [`input_proc`] for exactly one `AudioConverterFillComplexBuffer`
/// call. Interleaved `f32`, matching [`AudioBuffer::samples`].
struct PendingInput {
    samples: Vec<f32>,
    channels: u32,
    delivered: bool,
}

extern "C" fn input_proc(
    _converter: AudioConverterRef,
    io_number_data_packets: *mut u32,
    io_data: *mut AudioBufferList,
    out_data_packet_description: *mut *mut AudioStreamPacketDescription,
    in_user_data: *mut c_void,
) -> OSStatus {
    // Safety: `in_user_data` is the `&mut PendingInput` `fill_one_frame` passed to
    // `AudioConverterFillComplexBuffer` for the duration of that call, and this
    // callback only runs synchronously within it.
    let state = unsafe { &mut *in_user_data.cast::<PendingInput>() };
    unsafe {
        if !out_data_packet_description.is_null() {
            *out_data_packet_description = ptr::null_mut();
        }
        (*io_data).number_buffers = 1;
        (*io_data).buffers[0].number_channels = state.channels;
    }
    if state.delivered || state.samples.is_empty() {
        unsafe {
            *io_number_data_packets = 0;
            (*io_data).buffers[0].data_byte_size = 0;
            (*io_data).buffers[0].data = ptr::null_mut();
        }
        return NO_MORE_INPUT;
    }
    let frames = state.samples.len() as u32 / state.channels;
    unsafe {
        *io_number_data_packets = frames;
        (*io_data).buffers[0].data_byte_size =
            u32::try_from(state.samples.len() * std::mem::size_of::<f32>()).unwrap_or(u32::MAX);
        (*io_data).buffers[0].data = state.samples.as_mut_ptr().cast();
    }
    state.delivered = true;
    0
}

pub(super) struct AacEncoder {
    converter: AudioConverterRef,
    config: EncoderConfig,
    format: AudioEncoderFormat,
    channels: u32,
    max_packet_size: u32,
    /// Interleaved PCM buffered across `encode` calls until a full 1024-sample
    /// frame is available; `AudioBuffer`s from callers are not required to
    /// arrive in frame-sized batches.
    pending: Vec<f32>,
    /// Total PCM frames encoded so far, in `timescale` (= sample rate) ticks;
    /// every emitted [`EncodedSample`] is timestamped from this running clock
    /// rather than from the `AudioBuffer::range` that happened to trigger it,
    /// since a frame's samples can straddle two calls to `encode`.
    encoded_position: u64,
    finished: bool,
}

// The raw `AudioConverterRef` is only ever touched from the thread driving this
// encoder's `encode`/`finish` futures to completion, synchronously and one call
// at a time, so there is nothing here for another thread to race with even
// though the pointer itself carries no such guarantee.
unsafe impl Send for AacEncoder {}

impl Drop for AacEncoder {
    fn drop(&mut self) {
        unsafe {
            AudioConverterDispose(self.converter);
        }
    }
}

impl AacEncoder {
    fn frame_samples(&self) -> usize {
        (FRAME_LENGTH * self.channels) as usize
    }

    /// Encodes exactly one AAC-LC frame from `frame_pcm`, which must hold
    /// exactly [`FRAME_LENGTH`] interleaved PCM frames.
    fn encode_frame(&mut self, frame_pcm: Vec<f32>) -> Result<Vec<u8>> {
        debug_assert_eq!(frame_pcm.len(), self.frame_samples());
        let mut input = PendingInput {
            samples: frame_pcm,
            channels: self.channels,
            delivered: false,
        };
        let mut output_packets: u32 = 1;
        let mut output_buffer = vec![0_u8; self.max_packet_size as usize];
        let mut buffer_list = AudioBufferList {
            number_buffers: 1,
            buffers: [AudioBufferStruct {
                number_channels: self.channels,
                data_byte_size: self.max_packet_size,
                data: output_buffer.as_mut_ptr().cast(),
            }],
        };
        let mut packet_description = AudioStreamPacketDescription {
            start_offset: 0,
            variable_frames_in_packet: 0,
            data_byte_size: 0,
        };
        let status = unsafe {
            AudioConverterFillComplexBuffer(
                self.converter,
                input_proc,
                (&raw mut input).cast(),
                &mut output_packets,
                &mut buffer_list,
                &mut packet_description,
            )
        };
        if status != 0 {
            return Err(status_error(
                status,
                "AudioConverterFillComplexBuffer failed to encode an AAC-LC frame",
            ));
        }
        if output_packets == 0 {
            return Err(Error::new(
                ErrorKind::Codec,
                "AudioConverterFillComplexBuffer produced no AAC-LC packet for a full frame",
            ));
        }
        output_buffer.truncate(packet_description.data_byte_size as usize);
        Ok(output_buffer)
    }

    fn drain_full_frames(&mut self, running_position: &mut u64) -> Result<Vec<EncodedSample>> {
        let frame_samples = self.frame_samples();
        let mut samples = Vec::new();
        while self.pending.len() >= frame_samples {
            let frame_pcm = self.pending.drain(..frame_samples).collect::<Vec<_>>();
            let data = self.encode_frame(frame_pcm)?;
            let dts = i64::try_from(*running_position).map_err(|_| {
                Error::new(ErrorKind::ResourceLimit, "AAC sample position overflowed")
            })?;
            samples.push(EncodedSample {
                data,
                dts,
                pts: dts,
                duration: FRAME_LENGTH,
                is_sync: true,
                dependency: SampleDependency::INDEPENDENT,
            });
            *running_position += u64::from(FRAME_LENGTH);
        }
        Ok(samples)
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
            if buffer.sample_rate != self.format.sample_rate || buffer.channels != self.format.channels
            {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "audio buffer format does not match the AAC encoder",
                ));
            }
            self.pending.extend_from_slice(&buffer.samples);
            let mut position = self.encoded_position;
            let samples = self.drain_full_frames(&mut position)?;
            self.encoded_position = position;
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
            let mut position = self.encoded_position;
            let mut samples = self.drain_full_frames(&mut position)?;
            let mut padding = 0_u32;
            if !self.pending.is_empty() {
                let frame_samples = self.frame_samples();
                let present_frames = u32::try_from(self.pending.len() / usize::from(self.format.channels))
                    .unwrap_or(FRAME_LENGTH);
                padding = FRAME_LENGTH - present_frames;
                self.pending.resize(frame_samples, 0.0);
                let mut tail = self.drain_full_frames(&mut position)?;
                samples.append(&mut tail);
            }
            self.encoded_position = position;
            let priming = query_priming_frames(self.converter);
            Ok(AudioDrain {
                samples,
                gapless: AudioGapless { priming, padding },
            })
        })
    }
}
