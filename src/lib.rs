//! Portable core types for frame-accurate video and synchronized audio I/O.
//!
//! The crate currently provides the backend-independent foundation: checked
//! timeline arithmetic, validated media values, capability descriptions, and
//! asynchronous byte I/O contracts. Container and codec backends build on
//! these types without leaking platform-specific values into the common API.

pub mod api;
pub mod codec;
pub mod io;
pub mod media;
pub mod mp4;
pub mod output;
pub mod timeline;

pub use api::{Capability, Error, ErrorKind, Limits, Result, Support, TransferMode};
pub use codec::{
    AudioDrain, AudioEncoder, AudioEncoderFormat, AudioGapless, CpuVideoEncoder, EncodedSample,
    EncoderConfig, EncoderFuture, SampleDependency, TrackKind, VideoEncoder, VideoEncoderFormat,
};
pub use media::{
    AudioBuffer, Codec, ColorRange, Container, PixelFormat, Plane, VideoDimensions, VideoFrame,
};
pub use output::{MediaOutput, OutputOptions};
pub use timeline::{FrameIndex, FrameRate, Rational, SampleRange, Timeline};
