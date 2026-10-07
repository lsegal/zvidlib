//! The types every zvidlib crate shares: errors and limits, checked timeline
//! and media values, byte I/O, the codec and transfer contracts, codec
//! configuration records more than one crate reads, and the process-wide SIMD
//! override.
//!
//! This is an internal crate of [zvidlib](https://crates.io/crates/zvidlib),
//! which re-exports everything here under the paths documented there. Depend
//! on `zvidlib` rather than on this crate directly.

pub mod api;
pub mod audio;
pub mod codec;
pub mod codec_config;
pub mod io;
pub mod media;
pub mod simd;
pub mod timeline;
pub mod transfer;

pub use api::{Capability, Error, ErrorKind, Limits, Result, Support, TransferMode};
pub use audio::{
    AacDecoder, AacSampleReader, AudioDecoder, AudioEdit, AudioSampleReader, AudioTrackTiming,
    EncodedAudioSample,
};
pub use codec::{
    AudioDrain, AudioEncoder, AudioEncoderConfig, AudioEncoderFactory, AudioEncoderFormat,
    AudioGapless, CancellationToken, CodecImplementation, CodecProfile, CodecSupport,
    DecodeStatistics, DecodedVideoFrame, EncodedSample, EncodedVideoSample, EncoderConfig,
    EncoderFuture, ExactFrameReader, HardwarePreference, SEEK_LATENCY_BUDGET, SampleDependency,
    Seek, SeekPreviewSource, TrackKind, VideoDecoder, VideoDecoderConfig, VideoDecoderFactory,
    VideoEncoder, VideoEncoderConfig, VideoEncoderFactory, VideoEncoderFormat,
    uncompressed_video_decoder_factory,
};
pub use media::{
    AudioBuffer, Codec, ColorRange, Container, PixelFormat, Plane, VideoDimensions, VideoFrame,
};
pub use timeline::{FrameIndex, FrameRate, Rational, SampleRange, Timeline};
pub use transfer::{
    ColorConversion, ContextIdentity, CpuFrameDestination, CpuFrameSource, CpuPlaneDestination,
    ExecutionOwner, FrameDestination, FrameSource, GraphicsAdapter, GraphicsApi, GraphicsResource,
    Orientation, ResourceKind, ResourceOwnership, ScaleFilter, TransferCapability, TransferPolicy,
    TransferStage, execute_transfer, inspect_transfer,
};
pub use codec_config::Vp9CodecConfig;
pub use simd::SimdIsa;
