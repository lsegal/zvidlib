//! Container detection and capability discovery across MP4 and WebM.

use crate::api::{Capability, Support};
use crate::audio::AudioTrackTiming;
use crate::codec::TrackKind;
use crate::io::ByteSource;
use crate::media::Container;
use crate::mp4_demux::{Mp4Demuxer, Mp4DemuxerOptions, Mp4Track, probe_mp4};
use crate::webm_demux::{WebmDemuxer, WebmDemuxerOptions, probe_webm};
use crate::{Error, ErrorKind, Limits, Result};

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
#[doc(hidden)]
pub async fn open_tracks<S: ByteSource + ?Sized>(
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

/// Indexes the `index`th audio track of an MP4 or WebM source, detected by
/// signature, together with its timing on the decoded sample clock: an MP4
/// track's edit list, or a WebM track's `CodecDelay` and `DiscardPadding`.
#[doc(hidden)]
pub async fn open_audio_track<S: ByteSource + ?Sized>(
    source: &S,
    index: usize,
    limits: &Limits,
) -> Result<(Mp4Track, AudioTrackTiming)> {
    let no_track = || Error::new(ErrorKind::InvalidInput, "no such audio track");
    if probe_container(source).await? == Some(Container::WebM) {
        let demuxer = WebmDemuxer::open(
            source,
            WebmDemuxerOptions {
                limits: *limits,
                ..WebmDemuxerOptions::default()
            },
        )
        .await?;
        let track = demuxer
            .tracks
            .iter()
            .filter(|track| track.kind == TrackKind::Audio)
            .nth(index)
            .ok_or_else(no_track)?;
        let timing = demuxer.audio_timing(track.id)?;
        return Ok((track.clone(), timing));
    }
    let demuxer = Mp4Demuxer::open(
        source,
        Mp4DemuxerOptions {
            limits: *limits,
            ..Mp4DemuxerOptions::default()
        },
    )
    .await?;
    let track = demuxer
        .tracks
        .into_iter()
        .filter(|track| track.kind == TrackKind::Audio)
        .nth(index)
        .ok_or_else(no_track)?;
    let timing = track.audio_timing(demuxer.movie_timescale)?;
    Ok((track, timing))
}
