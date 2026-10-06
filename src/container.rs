//! Container detection and capability discovery across MP4 and WebM.

use crate::api::{Capability, Support};
use crate::io::ByteSource;
use crate::media::Container;
use crate::mp4_demux::{Mp4Demuxer, Mp4DemuxerOptions, Mp4Track, probe_mp4};
use crate::webm_demux::{WebmDemuxer, WebmDemuxerOptions, probe_webm};
use crate::{Limits, Result};

/// Identifies a source's container from its leading bytes, never from a file
/// name: an EBML header with a `webm` or `matroska` `DocType` is
/// [`Container::WebM`], and an `ftyp` or `moov` box is [`Container::Mp4`].
/// `None` when neither matches.
pub async fn probe_container<S: ByteSource + ?Sized>(source: &S) -> Result<Option<Container>> {
    if probe_webm(source).await? {
        return Ok(Some(Container::WebM));
    }
    if probe_mp4(source).await? {
        return Ok(Some(Container::Mp4));
    }
    Ok(None)
}

/// Every container zvidlib reads and writes, each named as
/// [`Container::name`] spells it. Both are portable Rust and need no platform
/// support, so each reports [`Support::Available`] on every target; whether a
/// track inside can be decoded is a separate codec capability.
pub fn container_capabilities() -> Vec<Capability> {
    Container::ALL
        .iter()
        .map(|container| Capability {
            name: container.name().to_owned(),
            support: Support::Available,
            transfer_mode: None,
        })
        .collect()
}

/// Indexes every track of an MP4 or WebM source, detected by signature. A
/// source that is neither is read as MP4, whose demuxer reports what is wrong.
#[cfg_attr(
    not(all(feature = "web", target_arch = "wasm32")),
    allow(dead_code, reason = "the browser build is the caller")
)]
pub(crate) async fn open_tracks<S: ByteSource + ?Sized>(
    source: &S,
    limits: &Limits,
) -> Result<Vec<Mp4Track>> {
    if probe_container(source).await? == Some(Container::WebM) {
        let demuxer = WebmDemuxer::open(
            source,
            WebmDemuxerOptions {
                limits: *limits,
                ..WebmDemuxerOptions::default()
            },
        )
        .await?;
        return Ok(demuxer.tracks);
    }
    let demuxer = Mp4Demuxer::open(
        source,
        Mp4DemuxerOptions {
            limits: *limits,
            ..Mp4DemuxerOptions::default()
        },
    )
    .await?;
    Ok(demuxer.tracks)
}
