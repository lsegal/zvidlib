//! Native AAC-LC decoding and default-device PCM output.

use crate::{
    AacDecoder, AacTrackConfig, AudioBuffer, AudioOutputBackend, CancellationToken, Error,
    ErrorKind, Limits, Result,
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    FromSample, SampleFormat, SizedSample, Stream, StreamConfig, SupportedStreamConfigRange,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use symphonia_codec_aac::AacDecoder as SymphoniaAacDecoder;
use symphonia_core::audio::{Channels, SampleBuffer};
use symphonia_core::codecs::{CODEC_TYPE_AAC, CodecParameters, Decoder, DecoderOptions};
use symphonia_core::formats::Packet;

/// Dependency-independent AAC-LC access-unit decoder backed by Symphonia.
pub struct NativeAacDecoder {
    decoder: SymphoniaAacDecoder,
    sample_rate: u32,
    channels: u16,
    limits: Limits,
}

impl NativeAacDecoder {
    pub fn new(config: &AacTrackConfig, limits: Limits) -> Result<Self> {
        if config.audio_object_type != 2 || config.channels > 2 {
            return Err(unsupported(
                "the native AAC backend supports AAC-LC mono and stereo streams",
            ));
        }
        let channels = match config.channels {
            1 => Channels::FRONT_LEFT,
            2 => Channels::FRONT_LEFT | Channels::FRONT_RIGHT,
            _ => return Err(unsupported("the native AAC channel layout is unsupported")),
        };
        let mut parameters = CodecParameters::new();
        parameters
            .for_codec(CODEC_TYPE_AAC)
            .with_sample_rate(config.sample_rate)
            .with_channels(channels)
            .with_extra_data(config.audio_specific_config.clone().into_boxed_slice());
        let decoder = SymphoniaAacDecoder::try_new(&parameters, &DecoderOptions::default())
            .map_err(|error| {
                unsupported(format!("native AAC configuration is unavailable: {error}"))
            })?;
        Ok(Self {
            decoder,
            sample_rate: config.sample_rate,
            channels: config.channels,
            limits,
        })
    }
}

impl AacDecoder for NativeAacDecoder {
    fn decode(
        &mut self,
        sample: &crate::EncodedAudioSample,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        if cancellation.is_cancelled() {
            return Err(Error::new(ErrorKind::Cancelled, "AAC decode cancelled"));
        }
        let packet = Packet::new_from_slice(
            0,
            sample.decoded_range.start,
            sample.decoded_range.len(),
            &sample.data,
        );
        let decoded = self
            .decoder
            .decode(&packet)
            .map_err(|error| codec(format!("malformed AAC access unit: {error}")))?;
        let frames = decoded.frames() as u64;
        if frames != sample.decoded_range.len() {
            return Err(codec(format!(
                "AAC decoder produced {frames} frames for an indexed {}-frame interval",
                sample.decoded_range.len()
            )));
        }
        if decoded.spec().rate != self.sample_rate
            || decoded.spec().channels.count() != usize::from(self.channels)
        {
            return Err(codec("AAC decoder output format changed unexpectedly"));
        }
        let mut interleaved = SampleBuffer::<f32>::new(frames, *decoded.spec());
        interleaved.copy_interleaved_ref(decoded);
        AudioBuffer::new(
            sample.decoded_range,
            self.sample_rate,
            self.channels,
            interleaved.samples().to_vec(),
            &self.limits,
        )
    }

    fn reset(&mut self) -> Result<()> {
        self.decoder.reset();
        Ok(())
    }
}

struct QueuedPcm {
    samples: Vec<f32>,
    cursor: usize,
}

struct OutputState {
    queued: VecDeque<QueuedPcm>,
    generation: u64,
}

/// A raw-PCM adapter for the system default output device.
///
/// The device must natively accept the media sample rate and channel count;
/// zvidlib never silently resamples or remixes the synchronized stream.
pub struct DefaultAudioOutput {
    stream: Stream,
    state: Arc<Mutex<OutputState>>,
    clock: Arc<AtomicU64>,
    failure: Arc<Mutex<Option<String>>>,
    channels: u16,
}

impl DefaultAudioOutput {
    pub fn open(sample_rate: u32, channels: u16) -> Result<Self> {
        let device = cpal::default_host()
            .default_output_device()
            .ok_or_else(|| unsupported("no default native audio output device is available"))?;
        let sample_format = select_output_format(
            device.supported_output_configs().map_err(|error| {
                unsupported(format!("cannot query native audio output: {error}"))
            })?,
            sample_rate,
            channels,
        )
        .map_err(|offered| {
            if offered.is_empty() {
                unsupported(format!(
                    "default audio device does not support {sample_rate} Hz with {channels} channels"
                ))
            } else {
                let offered: Vec<String> =
                    offered.iter().map(SampleFormat::to_string).collect();
                unsupported(format!(
                    "default audio device supports {sample_rate} Hz with {channels} channels \
                     only in unsupported sample formats: {}",
                    offered.join(", ")
                ))
            }
        })?;
        let config = StreamConfig {
            channels,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let state = Arc::new(Mutex::new(OutputState {
            queued: VecDeque::new(),
            generation: 0,
        }));
        let clock = Arc::new(AtomicU64::new(0));
        let failure = Arc::new(Mutex::new(None));
        let stream = match sample_format {
            SampleFormat::F32 => build_stream::<f32>(&device, &config, &state, &clock, &failure)?,
            SampleFormat::I16 => build_stream::<i16>(&device, &config, &state, &clock, &failure)?,
            SampleFormat::I32 => build_stream::<i32>(&device, &config, &state, &clock, &failure)?,
            SampleFormat::F64 => build_stream::<f64>(&device, &config, &state, &clock, &failure)?,
            SampleFormat::U16 => build_stream::<u16>(&device, &config, &state, &clock, &failure)?,
            SampleFormat::I8 => build_stream::<i8>(&device, &config, &state, &clock, &failure)?,
            SampleFormat::U8 => build_stream::<u8>(&device, &config, &state, &clock, &failure)?,
            SampleFormat::I64 => build_stream::<i64>(&device, &config, &state, &clock, &failure)?,
            SampleFormat::U32 => build_stream::<u32>(&device, &config, &state, &clock, &failure)?,
            SampleFormat::U64 => build_stream::<u64>(&device, &config, &state, &clock, &failure)?,
            // `SampleFormat` is `#[non_exhaustive]`; `select_output_format` only ever
            // returns a format listed above, so this arm is for a future `cpal` variant.
            other => {
                return Err(unsupported(format!(
                    "default audio device sample format {other} is unsupported"
                )));
            }
        };
        Ok(Self {
            stream,
            state,
            clock,
            failure,
            channels,
        })
    }

    fn check_failure(&self) -> Result<()> {
        if let Some(message) = self.failure.lock().expect("audio failure lock").as_ref() {
            Err(Error::new(ErrorKind::Io, message.clone()))
        } else {
            Ok(())
        }
    }
}

impl AudioOutputBackend for DefaultAudioOutput {
    fn clock_samples(&self) -> u64 {
        self.clock.load(Ordering::Acquire)
    }

    fn start(&mut self, _media_sample: u64) -> Result<()> {
        self.check_failure()?;
        self.stream
            .play()
            .map_err(|error| io(format!("cannot start native audio output: {error}")))
    }

    fn schedule(&mut self, buffer: AudioBuffer, generation: u64) -> Result<()> {
        self.check_failure()?;
        if buffer.channels != self.channels {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "scheduled PCM channel count does not match the audio device",
            ));
        }
        let mut state = self.state.lock().expect("audio output lock");
        if generation != state.generation {
            return Err(Error::new(
                ErrorKind::Cancelled,
                "scheduled PCM belongs to a cancelled playback generation",
            ));
        }
        state.queued.push_back(QueuedPcm {
            samples: buffer.samples,
            cursor: 0,
        });
        Ok(())
    }

    fn cancel_queued(&mut self, generation: u64) -> Result<()> {
        let mut state = self.state.lock().expect("audio output lock");
        state.queued.clear();
        state.generation = generation;
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        self.stream
            .pause()
            .map_err(|error| io(format!("cannot stop native audio output: {error}")))
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    state: &Arc<Mutex<OutputState>>,
    clock: &Arc<AtomicU64>,
    failure: &Arc<Mutex<Option<String>>>,
) -> Result<Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let callback_state = Arc::clone(state);
    let callback_clock = Arc::clone(clock);
    let callback_channels = u64::from(config.channels);
    let error_state = Arc::clone(failure);
    device
        .build_output_stream(
            config,
            move |output: &mut [T], _| {
                let mut state = callback_state.lock().expect("audio output lock");
                for destination in output.iter_mut() {
                    let value = loop {
                        let Some(front) = state.queued.front_mut() else {
                            break 0.0;
                        };
                        if front.cursor < front.samples.len() {
                            let value = front.samples[front.cursor];
                            front.cursor += 1;
                            break value;
                        }
                        state.queued.pop_front();
                    };
                    *destination = T::from_sample(value);
                }
                callback_clock
                    .fetch_add(output.len() as u64 / callback_channels, Ordering::Release);
            },
            move |error| {
                *error_state.lock().expect("audio failure lock") =
                    Some(format!("native audio output failed: {error}"));
            },
            None,
        )
        .map_err(|error| unsupported(format!("cannot open native audio output: {error}")))
}

/// Output sample formats the PCM writer can convert an `f32` sample into, most
/// preferred first.
///
/// `f32` leads because that is what the decoder already holds, so the device
/// takes the samples unconverted; the integer formats follow in descending
/// precision. Every format `cpal` can name is here, so a device is only turned
/// away for its rate or channel count.
const OUTPUT_SAMPLE_FORMATS: [SampleFormat; 10] = [
    SampleFormat::F32,
    SampleFormat::I16,
    SampleFormat::I32,
    SampleFormat::F64,
    SampleFormat::U16,
    SampleFormat::I8,
    SampleFormat::U8,
    SampleFormat::I64,
    SampleFormat::U32,
    SampleFormat::U64,
];

/// Picks the sample format to open the default device in.
///
/// A device advertises one configuration per sample format it accepts, in an
/// order that is the platform's business and not a preference: Windows WASAPI
/// can lead with `u8` on a device that also offers `f32`. So the configurations
/// matching the media's rate and channel count are ranked by
/// [`OUTPUT_SAMPLE_FORMATS`] rather than taken first-come.
///
/// The error carries the formats that did match the rate and channel count but
/// that the writer cannot convert into, which is empty when nothing matched at
/// all.
fn select_output_format(
    supported: impl IntoIterator<Item = SupportedStreamConfigRange>,
    sample_rate: u32,
    channels: u16,
) -> std::result::Result<SampleFormat, Vec<SampleFormat>> {
    let offered: Vec<SampleFormat> = supported
        .into_iter()
        .filter(|config| {
            config.channels() == channels
                && config.min_sample_rate().0 <= sample_rate
                && config.max_sample_rate().0 >= sample_rate
        })
        .map(|config| config.sample_format())
        .collect();
    OUTPUT_SAMPLE_FORMATS
        .into_iter()
        .find(|format| offered.contains(format))
        .ok_or(offered)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

fn codec(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Codec, message)
}

fn io(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Io, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::MemorySource;
    use crate::{Mp4Demuxer, Mp4DemuxerOptions, TrackKind};
    use std::future::Future;
    use std::task::{Context, Poll, Waker};

    #[test]
    fn native_decoder_consumes_demuxed_aac_access_units_as_exact_f32_intervals() {
        let source =
            MemorySource::new(include_bytes!("../examples/media/BigBuckBunny.mp4").to_vec());
        let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
        let track = movie
            .tracks
            .iter()
            .find(|track| track.kind == TrackKind::Audio)
            .unwrap();
        let packets =
            block_on(track.to_encoded_audio_samples(&source, &Limits::default())).unwrap();
        let mut decoder =
            NativeAacDecoder::new(&track.aac_config().unwrap(), Limits::default()).unwrap();
        let cancellation = CancellationToken::new();
        for packet in packets.iter().take(3) {
            let decoded = decoder.decode(packet, &cancellation).unwrap();
            assert_eq!(decoded.range, packet.decoded_range);
            assert_eq!(decoded.sample_rate, 48_000);
            assert_eq!(decoded.channels, 2);
            assert_eq!(decoded.samples.len(), 2_048);
        }
        decoder.reset().unwrap();
        assert_eq!(
            decoder.decode(&packets[0], &cancellation).unwrap().range,
            packets[0].decoded_range
        );
    }

    fn config_range(
        channels: u16,
        min_rate: u32,
        max_rate: u32,
        format: SampleFormat,
    ) -> SupportedStreamConfigRange {
        SupportedStreamConfigRange::new(
            channels,
            cpal::SampleRate(min_rate),
            cpal::SampleRate(max_rate),
            cpal::SupportedBufferSize::Unknown,
            format,
        )
    }

    #[test]
    fn output_format_ranks_by_preference_rather_than_by_the_device_enumeration_order() {
        // A Windows device that lists `u8` first is still opened in `f32`;
        // taking the first matching configuration turned that device away.
        let supported = vec![
            config_range(2, 48_000, 48_000, SampleFormat::U8),
            config_range(2, 44_100, 48_000, SampleFormat::I16),
            config_range(2, 8_000, 192_000, SampleFormat::F32),
        ];
        assert_eq!(
            select_output_format(supported, 48_000, 2),
            Ok(SampleFormat::F32)
        );
    }

    #[test]
    fn output_format_accepts_a_device_that_offers_only_one_narrow_format() {
        for format in OUTPUT_SAMPLE_FORMATS {
            let supported = vec![config_range(2, 48_000, 48_000, format)];
            assert_eq!(select_output_format(supported, 48_000, 2), Ok(format));
        }
    }

    #[test]
    fn output_format_ignores_configs_with_the_wrong_rate_or_channel_count() {
        let supported = vec![
            config_range(1, 8_000, 192_000, SampleFormat::F32),
            config_range(2, 8_000, 44_100, SampleFormat::F64),
            config_range(2, 48_000, 48_000, SampleFormat::I16),
        ];
        assert_eq!(
            select_output_format(supported, 48_000, 2),
            Ok(SampleFormat::I16)
        );
    }

    #[test]
    fn output_format_reports_no_offered_format_when_the_rate_is_unavailable() {
        let supported = vec![config_range(2, 8_000, 44_100, SampleFormat::F32)];
        assert_eq!(select_output_format(supported, 48_000, 2), Err(Vec::new()));
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        let mut future = Box::pin(future);
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("memory-backed future unexpectedly suspended"),
        }
    }
}
