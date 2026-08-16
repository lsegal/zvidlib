use crate::media::{AudioBuffer, Codec, PixelFormat, VideoDimensions};
use crate::timeline::FrameIndex;
use crate::{Result, VideoFrame};
use std::future::Future;
use std::pin::Pin;

/// A non-`Send` encoder future that works on native and single-threaded WASM.
pub type EncoderFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + 'a>>;

/// The kind of media carried by an encoded track.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrackKind {
    Video,
    Audio,
}

/// Dependency information written to the MP4 sample dependency table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampleDependency {
    pub is_leading: u8,
    pub depends_on: u8,
    pub is_depended_on: u8,
    pub has_redundancy: u8,
}

impl SampleDependency {
    pub const INDEPENDENT: Self = Self {
        is_leading: 0,
        depends_on: 2,
        is_depended_on: 0,
        has_redundancy: 0,
    };

    pub const DEPENDENT: Self = Self {
        is_leading: 0,
        depends_on: 1,
        is_depended_on: 0,
        has_redundancy: 0,
    };

    pub(crate) fn to_sdtp(self) -> u8 {
        ((self.is_leading & 3) << 6)
            | ((self.depends_on & 3) << 4)
            | ((self.is_depended_on & 3) << 2)
            | (self.has_redundancy & 3)
    }
}

/// One encoded access unit with exact decode and presentation timing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedSample {
    pub data: Vec<u8>,
    pub dts: i64,
    pub pts: i64,
    pub duration: u32,
    pub is_sync: bool,
    pub dependency: SampleDependency,
}

/// Codec and sample-entry information needed to declare an MP4 track.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncoderConfig {
    pub codec: Codec,
    pub timescale: u32,
    /// Complete codec configuration box, including its size and fourcc.
    pub decoder_config: Vec<u8>,
}

/// Video format accepted by a video encoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VideoEncoderFormat {
    pub dimensions: VideoDimensions,
    pub pixel_format: PixelFormat,
}

/// Audio format accepted by an audio encoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioEncoderFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

/// Gapless audio metadata produced when an encoder is drained.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AudioGapless {
    pub priming: u32,
    pub padding: u32,
}

/// Packets and metadata emitted while draining an audio encoder.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AudioDrain {
    pub samples: Vec<EncodedSample>,
    pub gapless: AudioGapless,
}

/// Backend-neutral video encoder contract.
///
/// The associated frame type lets CPU, GL, and WebGL transfer backends feed
/// their native frame source without exposing those handles to the muxer.
pub trait VideoEncoder {
    type Frame;

    fn config(&self) -> &EncoderConfig;
    fn format(&self) -> VideoEncoderFormat;
    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        frame: Self::Frame,
    ) -> EncoderFuture<'a, Vec<EncodedSample>>;
    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, Vec<EncodedSample>>;
}

/// Convenience contract for encoders that consume portable CPU frames.
pub trait CpuVideoEncoder: VideoEncoder<Frame = VideoFrame> {}

impl<T: VideoEncoder<Frame = VideoFrame>> CpuVideoEncoder for T {}

/// Backend-neutral audio encoder contract.
pub trait AudioEncoder {
    fn config(&self) -> &EncoderConfig;
    fn format(&self) -> AudioEncoderFormat;
    fn encode<'a>(
        &'a mut self,
        index: FrameIndex,
        buffer: AudioBuffer,
    ) -> EncoderFuture<'a, Vec<EncodedSample>>;
    fn finish<'a>(&'a mut self) -> EncoderFuture<'a, AudioDrain>;
}
