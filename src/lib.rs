//! Portable core types for frame-accurate video and synchronized audio I/O.
//!
//! The crate provides checked timeline and media values, byte I/O, codec and
//! transfer contracts, exact-frame decoding, indexed MP4 output, and a
//! browser-facing WebAssembly boundary. Production container, codec, and
//! playback backends build on these types without leaking platform-specific
//! values into the common API.

pub mod api;
pub mod codec;
pub mod io;
pub mod media;
pub mod mp4;
pub mod output;
pub mod timeline;
pub mod transfer;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod wasm_api;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub use wasm_api::*;

pub use api::{Capability, Error, ErrorKind, Limits, Result, Support, TransferMode};
pub use codec::{
    AudioDrain, AudioEncoder, AudioEncoderFormat, AudioGapless, CancellationToken,
    CodecImplementation, CodecProfile, CodecSupport, DecodeStatistics, DecodedVideoFrame,
    EncodedSample, EncodedVideoSample, EncoderConfig, EncoderFuture, ExactFrameReader,
    HardwarePreference, SampleDependency, TrackKind, VideoDecoder, VideoDecoderConfig,
    VideoDecoderFactory, VideoEncoder, VideoEncoderConfig, VideoEncoderFactory, VideoEncoderFormat,
    uncompressed_video_decoder_factory,
};
pub use media::{
    AudioBuffer, Codec, ColorRange, Container, PixelFormat, Plane, VideoDimensions, VideoFrame,
};
pub use output::{MediaOutput, OutputOptions};
pub use timeline::{FrameIndex, FrameRate, Rational, SampleRange, Timeline};
pub use transfer::{
    ColorConversion, ContextIdentity, CpuFrameDestination, CpuFrameSource, CpuPlaneDestination,
    ExecutionOwner, FrameDestination, FrameSource, GraphicsAdapter, GraphicsApi, GraphicsResource,
    Orientation, ResourceKind, ResourceOwnership, ScaleFilter, TransferCapability, TransferPolicy,
    TransferStage, execute_transfer, inspect_transfer,
};
