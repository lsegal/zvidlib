//! Portable core types for frame-accurate video and synchronized audio I/O.
//!
//! The crate provides checked timeline and media values, byte I/O, codec and
//! transfer contracts, exact-frame decoding, indexed MP4 and WebM output, and a
//! browser-facing WebAssembly boundary. Production container, codec, and
//! playback backends build on these types without leaking platform-specific
//! values into the common API.

// The library is a workspace of codec, container and shared-core crates
// (#604); these re-export their modules under the paths they had when they
// were modules of this crate.
#[cfg(any(feature = "av1-decoder", feature = "av1-encoder"))]
#[doc(inline)]
pub use zvidlib_av1::{
    av1_entropy, av1_filters, av1_inter_decoder, av1_intra, av1_intra_decoder, av1_intra_pred,
    av1_mc, av1_simd,
};
#[doc(inline)]
pub use zvidlib_av1_syntax as av1;
#[doc(inline)]
pub use zvidlib_container::{
    codec_config, conformance, container, cover, mp4, mp4_demux, prefetch, webm, webm_demux,
};
#[doc(inline)]
pub use zvidlib_core::{api, audio, codec, io, media, timeline, transfer};
/// Opus configuration records, packet timing and, with the `opus-decoder` or
/// `opus-encoder` feature, the native Opus codec.
#[cfg(any(feature = "opus-decoder", feature = "opus-encoder"))]
#[doc(inline)]
pub use zvidlib_opus as opus;
#[cfg(not(any(feature = "opus-decoder", feature = "opus-encoder")))]
#[doc(inline)]
pub use zvidlib_opus_syntax as opus;
pub mod output;
pub mod playback;
pub mod previews;
pub mod simd;
pub mod vorbis;

// The AAC encoder, and the platform bindings the macOS and Windows AAC decoders
// share with it.
#[cfg(all(
    any(
        feature = "aac-encoder",
        all(feature = "aac-decoder", any(target_os = "macos", windows))
    ),
    not(target_arch = "wasm32")
))]
use zvidlib_aac_encoder as aac_encoder;
#[cfg(feature = "av1-decoder")]
use zvidlib_av1_decoder as av1_decoder;
#[cfg(feature = "av1-encoder")]
use zvidlib_av1_encoder as av1_encoder;
#[cfg(any(feature = "vp8-decoder", feature = "vp8-encoder"))]
use zvidlib_vp8 as vp8;
#[cfg(feature = "vp9-decoder")]
use zvidlib_vp9_decoder as vp9_decoder;
#[cfg(feature = "vp9-encoder")]
use zvidlib_vp9_encoder as vp9_encoder;

#[cfg(all(feature = "hevc-encoder", not(target_arch = "wasm32")))]
#[doc(hidden)]
pub use zvidlib_hevc_encoder::bench as hevc_encoder_bench;

#[cfg(all(feature = "hevc-decoder", not(target_arch = "wasm32")))]
#[doc(hidden)]
pub use zvidlib_hevc_decoder::decode_bench as hevc_decoder_bench;

#[cfg(feature = "vp9-decoder")]
#[doc(hidden)]
pub use zvidlib_vp9_decoder::vp9_simd::bench as vp9_decoder_bench;

#[cfg(feature = "vorbis-decoder")]
#[doc(hidden)]
pub use zvidlib_vorbis_decoder::vorbis_simd::bench as vorbis_decoder_bench;

#[cfg(all(feature = "hevc-decoder", not(target_arch = "wasm32")))]
#[doc(hidden)]
pub use zvidlib_hevc_decoder::decode_profile as hevc_decode_profile;

#[cfg(all(feature = "hevc-decoder", not(target_arch = "wasm32")))]
#[doc(hidden)]
pub use zvidlib_hevc_decoder::narrow_interp as hevc_narrow_interp;

#[cfg(all(
    feature = "hevc-decoder",
    feature = "hardware",
    not(target_arch = "wasm32")
))]
#[doc(hidden)]
pub use zvidlib_hevc_decoder::readback as hevc_hardware_readback;

#[cfg(all(feature = "vp9-encoder", not(target_arch = "wasm32")))]
#[doc(hidden)]
pub use vp9_encoder::bench as vp9_encoder_bench;

#[cfg(not(target_arch = "wasm32"))]
mod native_audio;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod wasm_api;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod web_audio_decoder;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod web_decoder;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod web_encoder;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod web_playback;

/// The browser's preview driver: the same [`previews::PreviewPass`] the native
/// index runs, advanced one preview per idle callback instead of on a thread.
#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub mod web_previews;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub use wasm_api::*;

#[cfg(all(feature = "web", target_arch = "wasm32"))]
pub use web_playback::WasmOnDemandPlayback;

#[doc(no_inline)]
pub use api::{Capability, Error, ErrorKind, Limits, Result, Support, TransferMode};
#[doc(no_inline)]
pub use audio::{
    AacDecoder, AacSampleReader, AudioDecoder, AudioEdit, AudioPacketProvider, AudioSampleReader,
    AudioTrackTiming, EncodedAudioSample,
};
#[doc(no_inline)]
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
/// See `crates/zvidlib-av1-encoder/benches/av1_encode.rs`.
#[cfg(feature = "av1-encoder")]
#[doc(hidden)]
pub use av1_encoder::bench as av1_encoder_bench;
#[cfg(feature = "av1-encoder")]
#[doc(inline)]
pub use av1_encoder::native_av1_video_encoder_factory;
#[cfg(any(feature = "av1-decoder", feature = "av1-encoder"))]
#[doc(no_inline)]
pub use av1_entropy::{AV1_CDF_MAX, Av1SymbolDecoder, validate_cdf};
#[cfg(any(feature = "av1-decoder", feature = "av1-encoder"))]
#[doc(no_inline)]
pub use av1_filters::{
    CdefStrength, FilmGrainParams, FilterFrame, FilterPlane, LoopFilterParams, MatrixCoefficients,
    RestorationUnit, TxSizeGrid, apply_film_grain, apply_restoration_unit, cdef_frame,
    convert_to_rgba8, deblock_frame, super_resolution_upscale,
};
#[cfg(any(feature = "av1-decoder", feature = "av1-encoder"))]
#[doc(no_inline)]
pub use av1_inter_decoder::Av1InterDecoder;
#[cfg(any(feature = "av1-decoder", feature = "av1-encoder"))]
#[doc(no_inline)]
pub use av1_intra::{
    Av1IntraBlock, Av1IntraFrame, Av1IntraMode, Av1TxType, Tx1d, get_ac_quant, get_dc_quant,
    inverse_transform, inverse_wht_4x4,
};
#[cfg(any(feature = "av1-decoder", feature = "av1-encoder"))]
#[doc(no_inline)]
pub use av1_intra_decoder::{decode_av1_lossless_intra, decode_av1_lossless_intra_with_tx_sizes};
#[cfg(any(feature = "av1-decoder", feature = "av1-encoder"))]
#[doc(no_inline)]
pub use av1_intra_pred::{
    Av1IntraSimd, SmoothMode, add_residual_row, av1_intra_simd, directional_row, paeth_row,
    smooth_row, sum_samples,
};
#[doc(no_inline)]
pub use codec::{
    AudioDrain, AudioEncoder, AudioEncoderConfig, AudioEncoderFactory, AudioEncoderFormat,
    AudioGapless, CancellationToken, CodecImplementation, CodecProfile, CodecSupport,
    DecodeStatistics, DecodedVideoFrame, EncodedSample, EncodedVideoSample, EncoderConfig,
    EncoderFuture, ExactFrameReader, HardwarePreference, SEEK_LATENCY_BUDGET, SampleDependency,
    SampleProvider, Seek, SeekPreviewSource, TrackKind, VideoDecoder, VideoDecoderConfig,
    VideoDecoderFactory, VideoEncoder, VideoEncoderConfig, VideoEncoderFactory, VideoEncoderFormat,
    uncompressed_video_decoder_factory,
};
#[doc(no_inline)]
pub use codec_config::{DerivedCodecString, Vp9CodecConfig, derive_codec_string};
#[doc(no_inline)]
pub use conformance::{
    ExpectedVideoFrame, FrameDigest, VideoDecoderConformanceReport, VideoDecoderConformanceVector,
    VideoEncoderConformanceReport, VideoEncoderConformanceVector, verify_video_decoder_conformance,
    verify_video_encoder_conformance,
};
#[doc(no_inline)]
pub use container::{container_capabilities, probe_container};
#[doc(no_inline)]
pub use cover::{COVER_THUMBNAIL_MAX_EDGE, CoverSource, DEFAULT_COVER_FRAME};
#[doc(no_inline)]
pub use media::{
    AudioBuffer, Codec, ColorRange, Container, PixelFormat, Plane, VideoDimensions, VideoFrame,
};
#[doc(no_inline)]
pub use mp4::{CoverArt, CoverArtFormat};
#[doc(no_inline)]
pub use mp4_demux::{
    AacTrackConfig, EditMapping, Mp4AudioPacketProvider, Mp4Demuxer, Mp4DemuxerOptions, Mp4Sample,
    Mp4SampleProvider, Mp4Track, probe_mp4,
};
#[cfg(feature = "opus-decoder")]
#[doc(no_inline)]
pub use opus::NativeOpusDecoder;
#[cfg(feature = "opus-encoder")]
#[doc(no_inline)]
pub use opus::native_opus_audio_encoder_factory;
#[doc(no_inline)]
pub use opus::{
    OPUS_PREROLL_SAMPLES, OPUS_SAMPLE_RATE, OpusHead, opus_packet_samples, opus_preroll_packets,
};
#[doc(no_inline)]
pub use output::{MediaOutput, OutputOptions};
#[doc(no_inline)]
pub use playback::{
    AudioOutputBackend, AudioOutputKind, IndexedPresentationTimeline, NativeAudioOutput,
    OnDemandAudioSource, OnDemandVideoSource, PlaybackAudioOutput, PlaybackAudioSource,
    PlaybackController, PlaybackOptions, PlaybackVideoSource, PrefetchAudioSource,
    PrefetchVideoSource, Presentation, WebAudioOutput,
};
#[doc(no_inline)]
pub use prefetch::{Mp4SampleLoader, PrefetchedAudioPacketProvider, PrefetchedSampleProvider};
#[doc(no_inline)]
pub use previews::{PreviewOptions, PreviewPass, PreviewStore};
#[doc(no_inline)]
pub use timeline::{FrameIndex, FrameRate, Rational, SampleRange, Timeline};
#[doc(no_inline)]
pub use transfer::{
    ColorConversion, ContextIdentity, CpuFrameDestination, CpuFrameSource, CpuPlaneDestination,
    ExecutionOwner, FrameDestination, FrameSource, GraphicsAdapter, GraphicsApi, GraphicsResource,
    Orientation, ResourceKind, ResourceOwnership, ScaleFilter, TransferCapability, TransferPolicy,
    TransferStage, execute_transfer, inspect_transfer,
};
#[cfg(feature = "vorbis-decoder")]
#[doc(no_inline)]
pub use vorbis::NativeVorbisDecoder;
#[cfg(feature = "vorbis-encoder")]
#[doc(no_inline)]
pub use vorbis::native_vorbis_audio_encoder_factory;
#[doc(no_inline)]
pub use vorbis::{VORBIS_PREROLL_PACKETS, VorbisConfig};
/// Per-stage access to the native VP8 encoder for the criterion benchmark
/// suite.
///
/// Internal and unstable, and the VP8 counterpart to [`hevc_encoder_bench`]:
/// the encoder's distortion metrics, transforms, quantization, prediction and
/// loop filter are not otherwise reachable from a benchmark, which is a
/// separate crate. See `crates/zvidlib-vp8/benches/vp8_encode.rs`.
#[cfg(feature = "vp8-encoder")]
#[doc(hidden)]
pub use vp8::bench as vp8_encoder_bench;
#[cfg(any(feature = "av1-decoder", feature = "av1-encoder"))]
#[doc(inline)]
pub use zvidlib_av1::forward_transform;
#[doc(no_inline)]
pub use zvidlib_core::SimdIsa;

/// Per-stage access to the native VP8 decoder for the criterion benchmark
/// suite.
///
/// Internal and unstable, and the decode-side counterpart to
/// [`vp8_encoder_bench`]: the stages only the decoder takes and its whole-frame
/// decode, which a benchmark, being a separate crate, cannot otherwise reach.
/// See `crates/zvidlib-vp8/benches/vp8_decode.rs`.
#[cfg(feature = "vp8-decoder")]
#[doc(hidden)]
pub use vp8::decode_bench as vp8_decoder_bench;
#[cfg(feature = "vp9-encoder")]
#[doc(inline)]
pub use vp9_encoder::native_vp9_video_encoder_factory;
#[doc(no_inline)]
pub use webm::WebmMuxer;
#[doc(no_inline)]
pub use webm_demux::{
    WebmAudioTrim, WebmCuePoint, WebmDemuxer, WebmDemuxerOptions, WebmSeekPoint, WebmSkippedTrack,
    probe_webm,
};

#[cfg(all(feature = "aac-encoder", not(target_arch = "wasm32")))]
#[doc(inline)]
pub use aac_encoder::native_aac_audio_encoder_factory;
#[cfg(feature = "av1-decoder")]
#[doc(inline)]
pub use av1_decoder::native_av1_video_decoder_factory;
#[cfg(not(target_arch = "wasm32"))]
pub use native_audio::DefaultAudioOutput;
#[cfg(all(feature = "aac-decoder", not(target_arch = "wasm32")))]
pub use native_audio::NativeAacDecoder;
#[cfg(not(target_arch = "wasm32"))]
#[doc(no_inline)]
pub use previews::PreviewIndex;
#[cfg(feature = "vp8-decoder")]
#[doc(inline)]
pub use vp8::native_vp8_video_decoder_factory;
#[cfg(feature = "vp8-encoder")]
#[doc(inline)]
pub use vp8::native_vp8_video_encoder_factory;
#[cfg(feature = "vp9-decoder")]
#[doc(inline)]
pub use vp9_decoder::native_vp9_video_decoder_factory;
#[cfg(feature = "hevc-decoder")]
#[doc(inline)]
pub use zvidlib_hevc_decoder::native_hevc_video_decoder_factory;
#[cfg(all(feature = "hevc-encoder", not(target_arch = "wasm32")))]
#[doc(inline)]
pub use zvidlib_hevc_encoder::native_hevc_video_encoder_factory;
