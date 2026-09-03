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
mod av1_cdf_tables;
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

/// The bounded preview tier over a track, for callers that need an answer at an
/// arbitrary position faster than a decode from the nearest random-access point
/// can give one. Native-only: its pass runs on a thread of its own.
#[cfg(not(target_arch = "wasm32"))]
pub mod previews;

#[cfg(not(target_arch = "wasm32"))]
mod hevc;

/// Per-stage access to the HEVC encoder for the criterion benchmark suite.
///
/// Internal and unstable: the encoder's stages are not otherwise reachable from
/// a benchmark, which is a separate crate. See `benches/hevc_encode.rs`.
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use hevc::bench as hevc_encoder_bench;

/// Per-stage access to the HEVC decoder for the criterion benchmark suite.
///
/// Internal and unstable, and the counterpart to [`hevc_encoder_bench`]: the
/// decoder's stages are not otherwise reachable from a benchmark, which is a
/// separate crate. See `benches/hevc_decode.rs`.
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use hevc::decode_bench as hevc_decoder_bench;

/// Stage attribution for a whole-frame HEVC decode.
///
/// Internal and unstable. [`hevc_decoder_bench`] measures each kernel in
/// isolation, which bounds what vectorizing a stage could buy; this reports
/// what a real decode actually spends in each stage, which is what says whether
/// that ceiling is worth reaching for. See `examples/hevc_decode_profile.rs`
/// and the breakdown recorded in `benches/README.md`.
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use hevc::decode_profile as hevc_decode_profile;

/// Decode-versus-readback attribution for the hardware HEVC backends.
///
/// Internal and unstable. A hardware backend maps its own surface and converts
/// it to RGBA inside `submit`, so a caller sees the fixed-function decode and
/// the host readback as one number; this splits them without exposing the
/// platform surface itself. See `benches/hevc_hardware.rs` and the reasoning
/// recorded in `benches/README.md`.
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub use hevc::readback as hevc_hardware_readback;

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
/// Per-stage access to the native AV1 encoder for the criterion benchmark
/// suite.
///
/// Internal and unstable, and the AV1 counterpart to [`hevc_encoder_bench`]:
/// the encoder's forward WHT, symbol coder, tile encoder and bitstream writers
/// are not otherwise reachable from a benchmark, which is a separate crate.
/// See `benches/av1_encode.rs`.
#[doc(hidden)]
pub use av1_encoder::bench as av1_encoder_bench;
pub use av1_encoder::native_av1_video_encoder_factory;
pub use av1_encoder::transform::forward_transform;
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
    HardwarePreference, SEEK_LATENCY_BUDGET, SampleDependency, Seek, SeekPreviewSource, TrackKind,
    VideoDecoder, VideoDecoderConfig, VideoDecoderFactory, VideoEncoder, VideoEncoderConfig,
    VideoEncoderFactory, VideoEncoderFormat, uncompressed_video_decoder_factory,
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
#[cfg(not(target_arch = "wasm32"))]
pub use hevc::native_hevc_video_decoder_factory;
#[cfg(not(target_arch = "wasm32"))]
pub use hevc::native_hevc_video_encoder_factory;
#[cfg(not(target_arch = "wasm32"))]
pub use native_audio::{DefaultAudioOutput, NativeAacDecoder};
#[cfg(not(target_arch = "wasm32"))]
pub use previews::{PreviewIndex, PreviewOptions};
