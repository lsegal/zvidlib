//! zvidlib's containers: MP4 and WebM muxing and demuxing, container probing,
//! codec string derivation from codec configuration records, and the decoder
//! and encoder conformance harness built on them.
//!
//! This is an internal crate of [zvidlib](https://crates.io/crates/zvidlib),
//! which re-exports its public items under the paths documented there. Depend
//! on `zvidlib` rather than on this crate directly.

pub mod codec_config;
pub mod conformance;
pub mod container;
pub mod cover;
mod ebml;
pub mod mp4;
pub mod mp4_demux;
pub mod prefetch;
pub mod webm;
pub mod webm_demux;

#[allow(unused_imports)]
use zvidlib_av1_syntax as av1;
#[allow(unused_imports)]
use zvidlib_av1_syntax::*;
#[allow(unused_imports)]
use zvidlib_core::*;
#[allow(unused_imports)]
use zvidlib_opus_syntax as opus;
#[allow(unused_imports)]
use zvidlib_opus_syntax::*;
#[allow(unused_imports)]
use zvidlib_vorbis_syntax as vorbis;
#[allow(unused_imports)]
use zvidlib_vorbis_syntax::*;

pub use codec_config::{DerivedCodecString, derive_codec_string};
pub use conformance::{
    ExpectedVideoFrame, FrameDigest, VideoDecoderConformanceReport, VideoDecoderConformanceVector,
    VideoEncoderConformanceReport, VideoEncoderConformanceVector, verify_video_decoder_conformance,
    verify_video_encoder_conformance,
};
pub use container::{container_capabilities, probe_container};
pub use cover::{COVER_THUMBNAIL_MAX_EDGE, CoverSource, DEFAULT_COVER_FRAME};
pub use mp4::{CoverArt, CoverArtFormat};
pub use mp4_demux::{
    AacTrackConfig, EditMapping, Mp4AudioPacketProvider, Mp4Demuxer, Mp4DemuxerOptions, Mp4Sample,
    Mp4SampleProvider, Mp4Track, probe_mp4,
};
pub use prefetch::{Mp4SampleLoader, PrefetchedAudioPacketProvider, PrefetchedSampleProvider};
pub use webm::WebmMuxer;
pub use webm_demux::{
    WebmAudioTrim, WebmCuePoint, WebmDemuxer, WebmDemuxerOptions, WebmSeekPoint, WebmSkippedTrack,
    probe_webm,
};
