//! Portable core types for frame-accurate video and synchronized audio I/O.
//!
//! The crate provides checked timeline and media values, byte I/O, codec and
//! transfer contracts, exact-frame decoding, indexed MP4 output, and a
//! browser-facing WebAssembly boundary. Production container, codec, and
//! playback backends build on these types without leaking platform-specific
//! values into the common API.

pub mod api;
pub mod audio;
pub mod av1;
mod av1_cdf;
mod av1_encoder;
pub mod av1_entropy;
pub mod av1_filters;
pub mod av1_inter_decoder;
pub mod av1_intra;
pub mod av1_intra_decoder;
pub mod av1_intra_pred;
pub mod av1_mc;
pub mod av1_simd;
pub mod codec;
pub mod codec_config;
pub mod conformance;
pub mod io;
pub mod media;
pub mod mp4;
pub mod mp4_demux;
pub mod output;
pub mod playback;
pub mod simd;
pub mod timeline;
pub mod transfer;

#[cfg(not(target_arch = "wasm32"))]
mod av1_decoder;

#[cfg(not(target_arch = "wasm32"))]
mod hevc;

#[cfg(not(target_arch = "wasm32"))]
mod native_audio;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod wasm_api;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod web_decoder;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub use wasm_api::*;

pub use api::{Capability, Error, ErrorKind, Limits, Result, Support, TransferMode};
pub use audio::{AacDecoder, AacSampleReader, AudioEdit, AudioTrackTiming, EncodedAudioSample};
pub use av1::{
    Av1CodecConfigurationRecord, Av1ColorConfig, Av1FrameHeader, Av1FrameType, Av1Metadata, Av1Obu,
    Av1ObuHeader, Av1ObuType, Av1OperatingPoint, Av1Parser, Av1SequenceHeader, Av1SyntaxSupport,
    Av1TileGroup,
};
pub use av1_encoder::native_av1_video_encoder_factory;
pub use av1_entropy::{AV1_CDF_MAX, Av1SymbolDecoder, validate_cdf};
pub use av1_filters::{
    CdefStrength, FilmGrainParams, FilterFrame, FilterPlane, LoopFilterParams, MatrixCoefficients,
    RestorationUnit, TxSizeGrid, apply_film_grain, apply_restoration_unit, cdef_frame,
    convert_to_rgba8, deblock_frame, super_resolution_upscale,
};
pub use av1_inter_decoder::Av1InterDecoder;
pub use av1_intra::{
    Av1IntraBlock, Av1IntraFrame, Av1IntraMode, Av1TxType, Tx1d, get_ac_quant, get_dc_quant,
    inverse_transform, inverse_wht_4x4,
};
pub use av1_intra_decoder::{decode_av1_lossless_intra, decode_av1_lossless_intra_with_tx_sizes};
pub use av1_intra_pred::{
    Av1IntraSimd, SmoothMode, add_residual_row, av1_intra_simd, directional_row, paeth_row,
    smooth_row, sum_samples,
};
pub use av1_simd::SimdIsa;
pub use codec::{
    AudioDrain, AudioEncoder, AudioEncoderFormat, AudioGapless, CancellationToken,
    CodecImplementation, CodecProfile, CodecSupport, DecodeStatistics, DecodedVideoFrame,
    EncodedSample, EncodedVideoSample, EncoderConfig, EncoderFuture, ExactFrameReader,
    HardwarePreference, SampleDependency, TrackKind, VideoDecoder, VideoDecoderConfig,
    VideoDecoderFactory, VideoEncoder, VideoEncoderConfig, VideoEncoderFactory, VideoEncoderFormat,
    uncompressed_video_decoder_factory,
};
pub use codec_config::{DerivedCodecString, derive_codec_string};
pub use conformance::{
    ExpectedVideoFrame, FrameDigest, VideoDecoderConformanceReport, VideoDecoderConformanceVector,
    VideoEncoderConformanceReport, VideoEncoderConformanceVector, verify_video_decoder_conformance,
    verify_video_encoder_conformance,
};
pub use media::{
    AudioBuffer, Codec, ColorRange, Container, PixelFormat, Plane, VideoDimensions, VideoFrame,
};
pub use mp4_demux::{
    AacTrackConfig, EditMapping, Mp4Demuxer, Mp4DemuxerOptions, Mp4Sample, Mp4Track, probe_mp4,
};
pub use output::{MediaOutput, OutputOptions};
pub use playback::{
    AudioOutputBackend, AudioOutputKind, IndexedPresentationTimeline, NativeAudioOutput,
    PlaybackAudioOutput, PlaybackAudioSource, PlaybackController, PlaybackOptions,
    PlaybackVideoSource, Presentation, WebAudioOutput,
};
pub use timeline::{FrameIndex, FrameRate, Rational, SampleRange, Timeline};
pub use transfer::{
    ColorConversion, ContextIdentity, CpuFrameDestination, CpuFrameSource, CpuPlaneDestination,
    ExecutionOwner, FrameDestination, FrameSource, GraphicsAdapter, GraphicsApi, GraphicsResource,
    Orientation, ResourceKind, ResourceOwnership, ScaleFilter, TransferCapability, TransferPolicy,
    TransferStage, execute_transfer, inspect_transfer,
};

#[cfg(not(target_arch = "wasm32"))]
pub use av1_decoder::native_av1_video_decoder_factory;
/// Prepared per-stage HEVC decoder workloads for the benchmark suite.
///
/// The HEVC engine is crate-private, so `benches/codec.rs` — an external crate —
/// cannot time its individual decode stages directly the way it times the public
/// AV1 kernels. This is the narrow surface that lets it: one prepared workload
/// per hot stage, with input construction separated from the kernel under test.
#[cfg(not(target_arch = "wasm32"))]
pub use hevc::stage_bench as hevc_stage_bench;
#[cfg(not(target_arch = "wasm32"))]
pub use hevc::native_hevc_video_decoder_factory;
#[cfg(not(target_arch = "wasm32"))]
pub use hevc::native_hevc_video_encoder_factory;
#[cfg(not(target_arch = "wasm32"))]
pub use native_audio::{DefaultAudioOutput, NativeAacDecoder};
