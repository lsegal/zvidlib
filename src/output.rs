//! Strict synchronized indexed writing built on encoder and muxer contracts.

use crate::codec::{AudioEncoder, VideoEncoder};
use crate::io::ByteSink;
use crate::mp4::{Mp4Muxer, Mp4TrackConfig, Mp4TrackFormat};
use crate::{AudioBuffer, Error, ErrorKind, FrameIndex, Result, Timeline};

/// Resource policy for a seekable output session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutputOptions {
    /// Maximum encoded samples retained in each track's finalization index.
    pub max_samples_per_track: usize,
    /// Maximum packet batch an encoder may return from one call or drain.
    pub max_samples_per_encode: usize,
}

impl Default for OutputOptions {
    fn default() -> Self {
        Self {
            max_samples_per_track: 1_000_000,
            max_samples_per_encode: 64,
        }
    }
}

/// A synchronized video/audio output session with strict `put(n)` ordering.
pub struct MediaOutput<S, V, A> {
    muxer: Mp4Muxer<S>,
    video: V,
    audio: A,
    timeline: Timeline,
    next_video: u64,
    next_audio: u64,
    max_samples_per_encode: usize,
    failure: Option<Error>,
}

impl<S, V, A> MediaOutput<S, V, A>
where
    S: ByteSink,
    V: VideoEncoder,
    A: AudioEncoder,
{
    pub async fn new(
        sink: S,
        video: V,
        audio: A,
        timeline: Timeline,
        options: OutputOptions,
    ) -> Result<Self> {
        if audio.format().sample_rate != timeline.audio_sample_rate() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "audio encoder sample rate does not match the output timeline",
            ));
        }
        if audio.config().timescale != audio.format().sample_rate {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "audio MP4 timescale must preserve the encoder sample clock",
            ));
        }
        if options.max_samples_per_encode == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "the encoder packet batch limit must be nonzero",
            ));
        }
        let tracks = vec![
            Mp4TrackConfig {
                encoder: video.config().clone(),
                format: Mp4TrackFormat::Video(video.format().dimensions),
            },
            Mp4TrackConfig {
                encoder: audio.config().clone(),
                format: Mp4TrackFormat::Audio {
                    channels: audio.format().channels,
                },
            },
        ];
        let muxer = Mp4Muxer::new(sink, tracks, options.max_samples_per_track).await?;
        Ok(Self {
            muxer,
            video,
            audio,
            timeline,
            next_video: 0,
            next_audio: 0,
            max_samples_per_encode: options.max_samples_per_encode,
            failure: None,
        })
    }

    pub fn next_video_index(&self) -> FrameIndex {
        FrameIndex(self.next_video)
    }

    pub fn next_audio_index(&self) -> FrameIndex {
        FrameIndex(self.next_audio)
    }

    /// Encodes exactly the next presentation-order video frame.
    pub async fn put_video(&mut self, index: FrameIndex, frame: V::Frame) -> Result<()> {
        self.ensure_healthy()?;
        check_index("video", self.next_video, index)?;
        let samples = match self.video.encode(index, frame).await {
            Ok(samples) => samples,
            Err(error) => return Err(self.fail(error)),
        };
        if let Err(error) = check_batch_limit(self.max_samples_per_encode, samples.len()) {
            return Err(self.fail(error));
        }
        for sample in samples {
            if let Err(error) = self.muxer.write_sample(0, sample).await {
                return Err(self.fail(error));
            }
        }
        self.next_video = advance(index)?;
        Ok(())
    }

    /// Encodes audio for exactly the next frame-aligned sample interval.
    pub async fn put_audio(&mut self, index: FrameIndex, buffer: AudioBuffer) -> Result<()> {
        self.ensure_healthy()?;
        check_index("audio", self.next_audio, index)?;
        let format = self.audio.format();
        if buffer.sample_rate != format.sample_rate || buffer.channels != format.channels {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "audio write format is incompatible with the encoder",
            ));
        }
        let expected = self.timeline.audio_interval_for_frame(index)?;
        if buffer.range != expected {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "audio write does not cover the indexed frame interval",
            ));
        }
        let samples = match self.audio.encode(index, buffer).await {
            Ok(samples) => samples,
            Err(error) => return Err(self.fail(error)),
        };
        if let Err(error) = check_batch_limit(self.max_samples_per_encode, samples.len()) {
            return Err(self.fail(error));
        }
        for sample in samples {
            if let Err(error) = self.muxer.write_sample(1, sample).await {
                return Err(self.fail(error));
            }
        }
        self.next_audio = advance(index)?;
        Ok(())
    }

    /// Drains both encoders, records gapless audio trim, finalizes indexes, and
    /// returns the flushed sink.
    pub async fn finish(mut self) -> Result<S> {
        self.ensure_healthy()?;
        if self.next_video != self.next_audio {
            return Err(Error::new(
                ErrorKind::InvalidState,
                "video and audio indexed writes must be synchronized before finish",
            ));
        }

        let delayed_video = self.video.finish().await?;
        check_batch_limit(self.max_samples_per_encode, delayed_video.len())?;
        for sample in delayed_video {
            self.muxer.write_sample(0, sample).await?;
        }
        let delayed_audio = self.audio.finish().await?;
        check_batch_limit(self.max_samples_per_encode, delayed_audio.samples.len())?;
        for sample in delayed_audio.samples {
            self.muxer.write_sample(1, sample).await?;
        }
        self.muxer.set_audio_gapless(1, delayed_audio.gapless)?;
        self.muxer.finish().await
    }

    fn ensure_healthy(&self) -> Result<()> {
        match &self.failure {
            Some(error) => Err(Error::new(
                ErrorKind::InvalidState,
                format!("output session previously failed: {error}"),
            )),
            None => Ok(()),
        }
    }

    fn fail(&mut self, error: Error) -> Error {
        self.failure = Some(error.clone());
        error
    }
}

fn check_index(stream: &str, expected: u64, actual: FrameIndex) -> Result<()> {
    if actual.0 != expected {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!(
                "{stream} put expected frame {expected}, received {}",
                actual.0
            ),
        ));
    }
    Ok(())
}

fn advance(index: FrameIndex) -> Result<u64> {
    index.0.checked_add(1).ok_or_else(|| {
        Error::new(
            ErrorKind::ResourceLimit,
            "indexed output cannot advance beyond the maximum frame index",
        )
    })
}

fn check_batch_limit(limit: usize, actual: usize) -> Result<()> {
    if actual > limit {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "encoder packet batch exceeds the configured output limit",
        ));
    }
    Ok(())
}
