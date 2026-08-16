//! Portable core types for frame-accurate video and synchronized audio I/O.
//!
//! The crate currently provides the backend-independent foundation: checked
//! timeline arithmetic, validated media values, capability descriptions, and
//! asynchronous byte I/O contracts. Container and codec backends build on
//! these types without leaking platform-specific values into the common API.

pub mod api;
pub mod io;
pub mod media;
pub mod timeline;
pub mod transfer;

pub use api::{Capability, Error, ErrorKind, Limits, Result, Support, TransferMode};
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
