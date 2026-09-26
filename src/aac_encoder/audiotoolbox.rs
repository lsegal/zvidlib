//! macOS AudioToolbox AAC-LC backend for [`crate::AudioEncoder`].
//!
//! Encodes 32-bit float interleaved PCM to AAC-LC through `AudioConverter`'s
//! pull model: [`AudioConverterFillComplexBuffer`] calls [`input_proc`] back
//! for source frames as it needs them, one 1024-sample AAC-LC frame per call
//! here since the source and destination sample rates always match (no
//! resampling is asked of the converter).
//!
//! `finish` tells the converter the stream has ended rather than feeding it
//! silence, so it flushes its look-ahead and final partial frame itself, and
//! the reported padding is measured from the packets it actually emitted.

use std::ffi::c_void;
use std::ptr;

use crate::{
    AudioBuffer, AudioDrain, AudioEncoder, AudioEncoderConfig, AudioEncoderFormat, AudioGapless,
    Codec, CodecImplementation, CodecSupport, EncodedSample, EncoderConfig, EncoderFuture, Error,
    ErrorKind, FrameIndex, Limits, Result, SampleDependency,
};

use super::{FRAME_LENGTH, esds_box, gapless_padding};

type OSStatus = i32;
type AudioConverterRef = *mut c_void;

/// A sentinel this module's own [`input_proc`] returns from
/// `AudioConverterFillComplexBuffer`'s pull callback to say no more source
/// frames are available for the current call. Not a system `OSStatus`: the
/// converter passes whatever the callback returns straight back to its
/// caller, so this only ever needs to be distinguishable from `0` (`noErr`),
/// which with zero frames is instead how the callback reports end of stream.
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
struct AudioValueRange {
    minimum: f64,
    maximum: f64,
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

    fn AudioConverterGetPropertyInfo(
        in_audio_converter: AudioConverterRef,
        in_property_id: u32,
        out_size: *mut u32,
        out_writable: *mut u8,
    ) -> OSStatus;

    fn AudioConverterSetProperty(
        in_audio_converter: AudioConverterRef,
        in_property_id: u32,
        in_property_data_size: u32,
        in_property_data: *const c_void,
    ) -> OSStatus;
}

const fn fourcc(bytes: &[u8; 4]) -> u32 {
    ((bytes[0] as u32) << 24)
        | ((bytes[1] as u32) << 16)
        | ((bytes[2] as u32) << 8)
        | bytes[3] as u32
}

const K_AUDIO_FORMAT_LINEAR_PCM: u32 = fourcc(b"lpcm");
const K_AUDIO_FORMAT_MPEG4_AAC: u32 = fourcc(b"aac ");
const K_AUDIO_FORMAT_FLAG_IS_FLOAT: u32 = 1 << 0;
const K_AUDIO_FORMAT_FLAG_IS_PACKED: u32 = 1 << 3;
const K_AUDIO_CONVERTER_PROPERTY_MAXIMUM_OUTPUT_PACKET_SIZE: u32 = fourcc(b"xops");
const K_AUDIO_CONVERTER_PROPERTY_PRIME_INFO: u32 = fourcc(b"prim");
const K_AUDIO_CONVERTER_ENCODE_BIT_RATE: u32 = fourcc(b"brat");
const K_AUDIO_CONVERTER_APPLICABLE_ENCODE_BIT_RATES: u32 = fourcc(b"aebr");

/// Generous fallback when [`K_AUDIO_CONVERTER_PROPERTY_MAXIMUM_OUTPUT_PACKET_SIZE`]
/// cannot be read: several times the largest AAC-LC frame AudioToolbox produces
/// at any bit rate this encoder would plausibly be asked for.
const FALLBACK_MAX_PACKET_SIZE: u32 = 8192;

pub(super) fn capability(
    configuration: &AudioEncoderConfig,
    bit_rate: Option<u32>,
) -> CodecSupport {
    match new_converter(configuration, bit_rate) {
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
    bit_rate: Option<u32>,
    limits: &Limits,
) -> Result<Box<dyn AudioEncoder>> {
    let _ = limits;
    let converter = new_converter(configuration, bit_rate).map_err(|status| {
        status_error(status, "AudioConverterNew could not create an AAC encoder")
    })?;
    let max_packet_size =
        query_max_output_packet_size(converter).unwrap_or(FALLBACK_MAX_PACKET_SIZE);
    // Read once, before any real audio flows through it: this is the encoder's
    // fixed algorithmic look-ahead for this configuration, not a measurement
    // that changes run to run, and `finish` measures the end padding past it.
    let priming_frames = query_priming_frames(converter);
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
        priming_frames,
        pending: Vec::new(),
        total_input_frames: 0,
        encoded_position: 0,
        finished: false,
    }))
}

fn new_converter(
    configuration: &AudioEncoderConfig,
    bit_rate: Option<u32>,
) -> std::result::Result<AudioConverterRef, OSStatus> {
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
    let status = unsafe { AudioConverterNew(&source_format, &destination_format, &mut converter) };
    if status != 0 || converter.is_null() {
        return Err(status);
    }
    if let Err(status) = bit_rate.map_or(Ok(()), |bits| set_bit_rate(converter, bits)) {
        unsafe {
            AudioConverterDispose(converter);
        }
        return Err(status);
    }
    Ok(converter)
}

/// Sets the encoder's target bit rate to the applicable rate nearest
/// `bits_per_second`, since the converter rejects a rate outside the ranges it
/// reports for this sample rate and channel count.
fn set_bit_rate(
    converter: AudioConverterRef,
    bits_per_second: u32,
) -> std::result::Result<(), OSStatus> {
    let target = nearest_applicable_bit_rate(converter, bits_per_second).unwrap_or(bits_per_second);
    let status = unsafe {
        AudioConverterSetProperty(
            converter,
            K_AUDIO_CONVERTER_ENCODE_BIT_RATE,
            std::mem::size_of::<u32>() as u32,
            (&raw const target).cast(),
        )
    };
    if status == 0 { Ok(()) } else { Err(status) }
}

fn nearest_applicable_bit_rate(converter: AudioConverterRef, bits_per_second: u32) -> Option<u32> {
    let mut size = 0_u32;
    let status = unsafe {
        AudioConverterGetPropertyInfo(
            converter,
            K_AUDIO_CONVERTER_APPLICABLE_ENCODE_BIT_RATES,
            &mut size,
            ptr::null_mut(),
        )
    };
    let count = size as usize / std::mem::size_of::<AudioValueRange>();
    if status != 0 || count == 0 {
        return None;
    }
    let mut ranges = vec![AudioValueRange::default(); count];
    let status = unsafe {
        AudioConverterGetProperty(
            converter,
            K_AUDIO_CONVERTER_APPLICABLE_ENCODE_BIT_RATES,
            &mut size,
            ranges.as_mut_ptr().cast(),
        )
    };
    if status != 0 {
        return None;
    }
    let requested = f64::from(bits_per_second);
    ranges
        .iter()
        .map(|range| requested.clamp(range.minimum, range.maximum.max(range.minimum)))
        .min_by(|a, b| (a - requested).abs().total_cmp(&(b - requested).abs()))
        .map(|rate| rate.round() as u32)
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
    Error::new(ErrorKind::Codec, format!("{context} (OSStatus {status})"))
}

/// The PCM backlog offered to [`input_proc`] for one `AudioConverterFillComplexBuffer`
/// call. Interleaved `f32`, matching [`AudioBuffer::samples`].
///
/// AAC-LC's encoder look-ahead means one output packet does not correspond to
/// one fixed-size chunk of input: `AudioConverterFillComplexBuffer` may invoke
/// this callback more than once per call, asking for more source frames than
/// [`FRAME_LENGTH`] before it has enough to produce a packet. So the whole
/// backlog is handed over up front, and `consumed_frames` tracks how much of
/// it the converter has actually taken; a second invocation within the same
/// call reports whatever backlog remains, which is normally none.
///
/// Once `end_of_stream` is set, an exhausted backlog is reported as the end of
/// the stream instead, which is what makes the converter flush.
struct PendingInput {
    samples: Vec<f32>,
    channels: u32,
    consumed_frames: usize,
    end_of_stream: bool,
}

extern "C" fn input_proc(
    _converter: AudioConverterRef,
    io_number_data_packets: *mut u32,
    io_data: *mut AudioBufferList,
    out_data_packet_description: *mut *mut AudioStreamPacketDescription,
    in_user_data: *mut c_void,
) -> OSStatus {
    // Safety: `in_user_data` is the `&mut PendingInput` `try_encode_one_packet`
    // passed to `AudioConverterFillComplexBuffer` for the duration of that
    // call, and this callback only runs synchronously within it.
    let state = unsafe { &mut *in_user_data.cast::<PendingInput>() };
    unsafe {
        if !out_data_packet_description.is_null() {
            *out_data_packet_description = ptr::null_mut();
        }
        (*io_data).number_buffers = 1;
        (*io_data).buffers[0].number_channels = state.channels;
    }
    let channels = state.channels as usize;
    let total_frames = state.samples.len() / channels;
    let available_frames = total_frames - state.consumed_frames;
    if available_frames == 0 {
        unsafe {
            *io_number_data_packets = 0;
            (*io_data).buffers[0].data_byte_size = 0;
            (*io_data).buffers[0].data = ptr::null_mut();
        }
        return if state.end_of_stream {
            0
        } else {
            NO_MORE_INPUT
        };
    }
    let start_sample = state.consumed_frames * channels;
    let slice = &mut state.samples[start_sample..];
    unsafe {
        *io_number_data_packets = u32::try_from(available_frames).unwrap_or(u32::MAX);
        (*io_data).buffers[0].data_byte_size =
            u32::try_from(std::mem::size_of_val(slice)).unwrap_or(u32::MAX);
        (*io_data).buffers[0].data = slice.as_mut_ptr().cast();
    }
    // The converter is handed the entire remaining backlog and is expected to
    // consume all of what it is offered before asking again, exactly as a PCM
    // source with no more to give would; there is no partial-consumption signal
    // to read back afterward.
    state.consumed_frames = total_frames;
    0
}

pub(super) struct AacEncoder {
    converter: AudioConverterRef,
    config: EncoderConfig,
    format: AudioEncoderFormat,
    channels: u32,
    max_packet_size: u32,
    /// The encoder's fixed algorithmic look-ahead, in frames, read once at
    /// creation via `kAudioConverterPropertyPrimeInfo`; reported back as
    /// [`AudioGapless::priming`].
    priming_frames: u32,
    /// PCM handed to `AudioConverter` but not yet reported as an AAC-LC
    /// packet; every `try_encode_one_packet` call offers the whole backlog and
    /// the converter buffers whatever it does not immediately need
    /// internally, so this rarely holds more than one `encode` call's worth.
    pending: Vec<f32>,
    /// Total PCM frames the caller handed to the encoder; `finish` measures
    /// the end padding as whatever the emitted packets hold past these and the
    /// priming.
    total_input_frames: u64,
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
    /// Offers the entire buffered backlog to the converter and asks for one
    /// AAC-LC packet. Returns `Ok(None)` when the backlog (however large) is
    /// not yet enough to complete one - the ordinary case mid-stream, and also
    /// how the flush in `finish`, with `end_of_stream` set, learns the
    /// converter has nothing left buffered internally.
    fn try_encode_one_packet(&mut self, end_of_stream: bool) -> Result<Option<Vec<u8>>> {
        let mut input = PendingInput {
            samples: std::mem::take(&mut self.pending),
            channels: self.channels,
            consumed_frames: 0,
            end_of_stream,
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
        // Whatever the converter did not report as consumed goes back to the
        // front of the backlog; ordinarily that is everything or nothing, since
        // `input_proc` always offers the full remainder.
        let channels = usize::try_from(self.channels).unwrap_or(1);
        let consumed_samples = (input.consumed_frames * channels).min(input.samples.len());
        self.pending = input.samples.split_off(consumed_samples);

        if output_packets == 0 {
            return if status == 0 || status == NO_MORE_INPUT {
                Ok(None)
            } else {
                Err(status_error(
                    status,
                    "AudioConverterFillComplexBuffer failed to encode an AAC-LC frame",
                ))
            };
        }
        output_buffer.truncate(packet_description.data_byte_size as usize);
        Ok(Some(output_buffer))
    }

    /// Repeatedly asks the converter for a packet until it reports it has
    /// nothing ready, which is how both an ordinary `encode` call (bounded by
    /// what backlog is actually buffered) and the end-of-stream flush in
    /// `finish`, with `end_of_stream` set, drain.
    fn drain_ready_packets(
        &mut self,
        running_position: &mut u64,
        end_of_stream: bool,
    ) -> Result<Vec<EncodedSample>> {
        let mut samples = Vec::new();
        while let Some(data) = self.try_encode_one_packet(end_of_stream)? {
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
            if buffer.sample_rate != self.format.sample_rate
                || buffer.channels != self.format.channels
            {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "audio buffer format does not match the AAC encoder",
                ));
            }
            let channels = usize::try_from(self.channels).unwrap_or(1);
            self.total_input_frames += (buffer.samples.len() / channels) as u64;
            self.pending.extend_from_slice(&buffer.samples);
            let mut position = self.encoded_position;
            let samples = self.drain_ready_packets(&mut position, false)?;
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

            // End of stream makes the converter encode everything it still
            // holds, completing the final partial frame with silence of its
            // own. Padding the input with silence here instead, as this once
            // did, never ended the stream: the converter kept its look-ahead
            // back, so the packets it emitted did not hold the padding
            // reported for them.
            let mut position = self.encoded_position;
            let samples = self.drain_ready_packets(&mut position, true)?;
            self.encoded_position = position;
            let padding = gapless_padding(
                self.encoded_position,
                self.priming_frames,
                self.total_input_frames,
            )?;
            Ok(AudioDrain {
                samples,
                gapless: AudioGapless {
                    priming: self.priming_frames,
                    padding,
                },
            })
        })
    }
}
