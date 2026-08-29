//! Native AAC-LC decoding and default-device PCM output.

use crate::{
    AacDecoder, AacTrackConfig, AudioBuffer, AudioOutputBackend, CancellationToken, Error,
    ErrorKind, Limits, Result,
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, Stream, StreamConfig};
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
        let supported = device
            .supported_output_configs()
            .map_err(|error| unsupported(format!("cannot query native audio output: {error}")))?
            .find(|config| {
                config.channels() == channels
                    && config.min_sample_rate().0 <= sample_rate
                    && config.max_sample_rate().0 >= sample_rate
            })
            .ok_or_else(|| {
                unsupported(format!(
                    "default audio device does not support {sample_rate} Hz with {channels} channels"
                ))
            })?;
        let sample_format = supported.sample_format();
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
            SampleFormat::U16 => build_stream::<u16>(&device, &config, &state, &clock, &failure)?,
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
