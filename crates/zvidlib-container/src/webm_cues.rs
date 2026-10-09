//! A cued WebM opened from its header elements and `Cues` alone, its blocks
//! indexed a cue span at a time (issue #692).
//!
//! [`crate::WebmDemuxer`] indexes a WebM by reading every block header in
//! every Cluster, so opening a long file reads all of it before the first
//! frame plays. A file with `Cues` already says where its video's
//! random-access points are, and that is enough to start. [`WebmCuedIndex`]
//! reads the EBML header, `SeekHead`, `Info`, `Tracks` and `Cues` - the last
//! usually through the `SeekHead`, since muxers store `Cues` after the
//! Clusters - and the first block headers of the first Cluster, and nothing
//! else.
//!
//! The video track's cue points then divide the file into spans: each runs
//! from one cued key frame to the next, and the first from the first Cluster.
//! [`WebmCuedIndex::index_span`] reads one span's block headers, and the few
//! after it that time its last blocks, and gives every track's samples in it
//! exactly as [`crate::WebmDemuxer`] would index them. Each span starts with a
//! random-access point, so a span decodes on its own.

use std::collections::HashMap;

use crate::audio::AudioTrackTiming;
use crate::codec::TrackKind;
use crate::ebml::{self, children, read_uint};
use crate::io::ByteSource;
use crate::media::Codec;
use crate::opus::OPUS_SAMPLE_RATE;
use crate::timeline::SampleRange;
use crate::track::{Track, read_exact, scale_time};
use crate::webm_demux::{
    Clock, Header, RawCue, Scan, TrackEntry, TrackMedia, Visit, WebmAudioTrim, WebmDemuxerOptions,
    WebmSkippedTrack, audio_timing, build_track, cued_blocks, parse_cues, parse_ebml_header,
    parse_info, parse_tracks, track_id, validate_options,
};
use crate::{Error, ErrorKind, Result};

/// How many Clusters after a span's last one its scan reads at most, looking
/// for the block that follows each track's last block in the span.
const LOOKAHEAD_CLUSTERS: usize = 1;

/// How many Clusters opening reads at most looking for the first video block.
const FIRST_VIDEO_CLUSTERS: usize = 4;

/// A cued WebM opened from its header elements and `Cues` alone, whose blocks
/// are indexed a span at a time with [`Self::index_span`]; see the
/// [module documentation](self).
#[derive(Clone, Debug)]
pub struct WebmCuedIndex {
    /// `"webm"`, or `"matroska"` for a Matroska file using the same subset.
    pub doc_type: String,
    /// The segment's `Duration`, in seconds, when the file declares one.
    pub duration_seconds: Option<f64>,
    /// The indexed tracks as [`crate::WebmDemuxer::tracks`] lists them, but
    /// with no samples: [`Self::index_span`] gives them a span at a time.
    pub tracks: Vec<Track>,
    pub skipped_tracks: Vec<WebmSkippedTrack>,
    /// The [`Track::id`] of the video track whose cue points divide the file
    /// into spans: the first video track.
    pub video_track: u32,
    /// Where each span but the first starts: its cued key frame.
    boundaries: Vec<Boundary>,
    first_cluster: u64,
    segment_end: u64,
    entries: Vec<TrackEntry>,
    raw_cues: Vec<RawCue>,
    clock: Clock,
    /// Each audio track's declared `CodecDelay`.
    codec_delays: Vec<(u32, Option<u64>)>,
    options: WebmDemuxerOptions,
}

/// The cued key frame a span starts at.
#[derive(Clone, Copy, Debug)]
struct Boundary {
    cluster_offset: u64,
    /// Raw timestamp in `TimestampScale` ticks.
    time: i64,
}

/// One span of a [`WebmCuedIndex`]: every indexed track's blocks from one
/// cued video key frame to the next.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WebmCueSpan {
    /// The span's position among [`WebmCuedIndex::span_count`].
    pub index: usize,
    /// Every indexed track, in [`WebmCuedIndex::tracks`] order, with its
    /// samples in the span exactly as [`crate::WebmDemuxer`] indexes them:
    /// presentation times on the whole file's timeline, and
    /// [`Track::presentation_order`] numbering the span's own samples.
    pub tracks: Vec<Track>,
    /// Whether this is the file's last span, which holds each track's last
    /// block.
    pub is_last: bool,
    /// Each audio track's last block's `DiscardPadding` in the span, which
    /// trims the stream's end when [`Self::is_last`].
    audio_trims: Vec<WebmAudioTrim>,
}

impl WebmCueSpan {
    /// The span's samples of the track whose [`Track::id`] is `id`.
    pub fn track(&self, id: u32) -> Option<&Track> {
        self.tracks.iter().find(|track| track.id == id)
    }
}

impl WebmCuedIndex {
    /// Opens a WebM from its header elements and `Cues`, reading no Cluster
    /// but the first block headers of the first. `None` when the file is one
    /// [`crate::WebmDemuxer`] has to index by scanning: one with no `Cues`
    /// for its first video track, or whose `Info` or `Tracks` it cannot find
    /// before its Clusters or through its `SeekHead`.
    pub async fn open<S: ByteSource + ?Sized>(
        source: &S,
        options: WebmDemuxerOptions,
    ) -> Result<Option<Self>> {
        validate_options(&options)?;
        let source_len = source
            .len()
            .ok_or_else(|| unsupported("WebM indexing requires a source length"))?;
        let mut scan = Scan {
            source,
            options: &options,
            elements: 0,
        };
        let header = scan.header(0, source_len).await?;
        if header.id != ebml::EBML {
            return Err(malformed("WebM does not start with an EBML header"));
        }
        let header_end = header.known_end()?;
        let doc_type = parse_ebml_header(&scan.payload(header, header_end).await?)?;

        let mut cursor = header_end;
        let segment = loop {
            if cursor >= source_len {
                return Err(malformed("WebM does not contain a Segment"));
            }
            let header = scan.header(cursor, source_len).await?;
            match header.id {
                ebml::SEGMENT => break header,
                ebml::VOID | ebml::CRC32 => cursor = header.known_end()?,
                _ => return Err(malformed("WebM has an unexpected top-level element")),
            }
        };
        let segment_start = segment.data_start;
        let segment_end = segment.end.unwrap_or(source_len);

        let mut info = None;
        let mut entries = None;
        let mut raw_cues = None;
        let mut seeks = Vec::new();
        let mut first_cluster = None;
        let mut cursor = segment_start;
        while cursor < segment_end {
            let header = scan.header(cursor, segment_end).await?;
            if header.id == ebml::CLUSTER {
                first_cluster = Some(header.start);
                break;
            }
            if matches!(header.id, ebml::EBML | ebml::SEGMENT) {
                break;
            }
            let end = header.known_end()?;
            match header.id {
                ebml::INFO if info.is_none() => {
                    info = Some(parse_info(&scan.payload(header, end).await?)?);
                }
                ebml::TRACKS if entries.is_none() => {
                    entries = Some(parse_tracks(&scan.payload(header, end).await?, &options)?);
                }
                ebml::CUES if raw_cues.is_none() => {
                    raw_cues = Some(parse_cues(
                        &scan.payload(header, end).await?,
                        segment_start,
                    )?);
                }
                ebml::SEEK_HEAD => {
                    seeks.extend(parse_seek_head(&scan.payload(header, end).await?)?)
                }
                _ => {}
            }
            cursor = end;
        }
        let Some(first_cluster) = first_cluster else {
            return Ok(None);
        };
        // Muxers usually store `Cues`, and sometimes `Info` and `Tracks`,
        // after the Clusters, where the `SeekHead` says.
        let sought = |id| element_position(&seeks, id, segment_start);
        if info.is_none()
            && let Some(position) = sought(ebml::INFO)
            && let Some(payload) = element(&mut scan, position, ebml::INFO, segment_end).await?
        {
            info = Some(parse_info(&payload)?);
        }
        if entries.is_none()
            && let Some(position) = sought(ebml::TRACKS)
            && let Some(payload) = element(&mut scan, position, ebml::TRACKS, segment_end).await?
        {
            entries = Some(parse_tracks(&payload, &options)?);
        }
        if raw_cues.is_none()
            && let Some(position) = sought(ebml::CUES)
            && let Some(payload) = element(&mut scan, position, ebml::CUES, segment_end).await?
        {
            raw_cues = Some(parse_cues(&payload, segment_start)?);
        }
        let (Some(info), Some(entries), Some(raw_cues)) = (info, entries, raw_cues) else {
            return Ok(None);
        };
        let Some(video) = entries.iter().find(|entry| {
            matches!(
                entry.indexed.as_ref().map(|track| &track.media),
                Some(TrackMedia::Video(_))
            )
        }) else {
            return Ok(None);
        };
        let video_number = video.number;
        let mut video_cues: Vec<Boundary> = raw_cues
            .iter()
            .filter(|cue| cue.track == video_number && cue.cluster_offset >= first_cluster)
            .map(|cue| Boundary {
                cluster_offset: cue.cluster_offset,
                time: cue.time,
            })
            .collect();
        if video_cues.is_empty() {
            return Ok(None);
        }

        // The first span starts at the first Cluster, so a cue on the first
        // video frame starts no span of its own.
        let Some(first_video_time) =
            first_video_time(&mut scan, first_cluster, segment_end, video_number).await?
        else {
            return Ok(None);
        };
        video_cues.retain(|cue| cue.time > first_video_time);
        video_cues.sort_by_key(|cue| (cue.time, cue.cluster_offset));
        video_cues.dedup_by_key(|cue| cue.time);

        let clock = Clock::new(&info);
        let mut tracks = Vec::new();
        let mut skipped_tracks = Vec::new();
        let mut codec_delays = Vec::new();
        let mut templates = Vec::new();
        for entry in entries {
            let Some(track) = entry.indexed.clone() else {
                skipped_tracks.push(WebmSkippedTrack {
                    number: entry.number,
                    codec_id: entry.codec_id,
                });
                continue;
            };
            let id = track_id(entry.number)?;
            let (track, trim) = build_track(id, track, &[], None, clock, &options)?;
            if let Some(trim) = trim {
                codec_delays.push((id, trim.codec_delay_ns));
            }
            tracks.push(track);
            templates.push(entry);
        }
        Ok(Some(Self {
            doc_type,
            duration_seconds: info
                .duration
                .map(|duration| duration * info.timestamp_scale as f64 / 1e9),
            tracks,
            skipped_tracks,
            video_track: track_id(video_number)?,
            boundaries: video_cues,
            first_cluster,
            segment_end,
            entries: templates,
            raw_cues,
            clock,
            codec_delays,
            options,
        }))
    }

    /// How many spans the video track's cue points divide the file into.
    pub fn span_count(&self) -> usize {
        self.boundaries.len() + 1
    }

    /// Ticks per second of every track's [`Track::timescale`].
    pub fn timescale(&self) -> u32 {
        self.clock.timescale
    }

    /// When `span` starts, in track ticks: the presentation time of its cued
    /// key frame, or `None` for the first span, which starts with the file.
    pub fn span_start(&self, span: usize) -> Option<i64> {
        let boundary = self.boundaries.get(span.checked_sub(1)?)?;
        self.clock.to_ticks(boundary.time).ok()
    }

    /// The span whose video frames include the one presented at `time`, in
    /// track ticks: the last span starting at or before it.
    pub fn span_at(&self, time: i64) -> usize {
        self.boundaries.partition_point(|boundary| {
            self.clock
                .to_ticks(boundary.time)
                .is_ok_and(|start| start <= time)
        })
    }

    /// An audio track's timing on the decoded sample clock, as
    /// [`crate::WebmDemuxer::audio_timing`] gives it: `CodecDelay` (or an Opus
    /// track's pre-skip) is the priming, and the `DiscardPadding` of the
    /// track's last block, in `last_span` when it is the file's last span, the
    /// end padding. Without the last span the padding is zero.
    pub fn audio_timing(
        &self,
        track_id: u32,
        last_span: Option<&WebmCueSpan>,
    ) -> Result<AudioTrackTiming> {
        let track = self
            .tracks
            .iter()
            .find(|track| track.id == track_id && track.kind == TrackKind::Audio)
            .ok_or_else(|| invalid("no such WebM audio track"))?;
        let codec_delay_ns = self
            .codec_delays
            .iter()
            .find(|(id, _)| *id == track_id)
            .and_then(|&(_, delay)| delay);
        let discard_padding_ns = last_span
            .filter(|span| span.is_last)
            .and_then(|span| span.audio_trims.iter().find(|trim| trim.track == track_id))
            .map_or(0, |trim| trim.discard_padding_ns);
        audio_timing(
            track,
            &WebmAudioTrim {
                track: track_id,
                codec_delay_ns,
                discard_padding_ns,
            },
        )
    }

    /// The decoded interval of each of `span`'s packets of the Opus track
    /// `track_id`, as [`crate::TrackSampleLoader::opus_packet_provider`] gives
    /// them for [`crate::WebmDemuxer`]'s index of the whole track: counted from
    /// the track's first packet, whose presentation time is `first_pts`, by
    /// the sample table's durations, and for the track's last packet - in the
    /// last span - by its own table of contents, whose first bytes this reads
    /// from `source`.
    ///
    /// The packets before the span are counted by their presentation times
    /// rather than their durations, which agree whenever each block lasts until
    /// the next, as Opus blocks without a `BlockDuration` do.
    pub async fn opus_decoded_ranges<S: ByteSource + ?Sized>(
        &self,
        source: &S,
        span: &WebmCueSpan,
        track_id: u32,
        first_pts: i64,
    ) -> Result<Vec<SampleRange>> {
        let track = span
            .track(track_id)
            .filter(|track| track.kind == TrackKind::Audio && track.codec == Codec::Opus)
            .ok_or_else(|| invalid("no such WebM Opus track"))?;
        let Some(first) = track.samples.first() else {
            return Ok(Vec::new());
        };
        let mut track_ticks = u64::try_from(first.pts - first_pts)
            .map_err(|_| malformed("a WebM audio block precedes the track's first"))?;
        let mut ranges = Vec::with_capacity(track.samples.len());
        let mut decoded_start = scale_time(track_ticks, track.timescale, OPUS_SAMPLE_RATE)?;
        for (index, sample) in track.samples.iter().enumerate() {
            let decoded_end = if span.is_last && index + 1 == track.samples.len() {
                // A muxer shortens the last sample's duration to trim the
                // stream's end, so only the packet itself says how long it is.
                let mut head = [0_u8; 2];
                let head = &mut head[..(sample.size as usize).min(2)];
                read_exact(source, sample.offset, head).await?;
                decoded_start
                    .checked_add(u64::from(crate::opus::opus_packet_samples(head)?))
                    .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "audio timing overflow"))?
            } else {
                track_ticks = track_ticks
                    .checked_add(u64::from(sample.duration))
                    .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "audio timing overflow"))?;
                scale_time(track_ticks, track.timescale, OPUS_SAMPLE_RATE)?
            };
            if decoded_end <= decoded_start {
                return Err(malformed("audio packet has an empty decoded interval"));
            }
            ranges.push(SampleRange::new(decoded_start, decoded_end)?);
            decoded_start = decoded_end;
        }
        Ok(ranges)
    }

    /// Indexes `span`: reads the block headers from its cued key frame to the
    /// next one, and the few after that time its last blocks, and gives each
    /// track's samples in it as [`crate::WebmDemuxer`] indexes them.
    pub async fn index_span<S: ByteSource + ?Sized>(
        &self,
        source: &S,
        span: usize,
    ) -> Result<WebmCueSpan> {
        if span >= self.span_count() {
            return Err(invalid("the WebM has no such cue span"));
        }
        let mut scan = Scan {
            source,
            options: &self.options,
            elements: 0,
        };
        let start = span.checked_sub(1).map(|index| self.boundaries[index]);
        let stop = self.boundaries.get(span).copied();
        let video_number = u64::from(self.video_track);
        let codecs: HashMap<u64, Codec> = self
            .entries
            .iter()
            .filter_map(|entry| Some((entry.number, entry.indexed.as_ref()?.codec)))
            .collect();
        let mut entries = self.entries.clone();
        let mut following: HashMap<u64, i64> = HashMap::new();
        let mut state = if start.is_some() {
            SpanState::Before
        } else {
            SpanState::Within
        };
        let mut cursor = start.map_or(self.first_cluster, |start| start.cluster_offset);
        let mut lookahead_clusters = 0;
        while cursor < self.segment_end {
            let header = scan.header(cursor, self.segment_end).await?;
            match header.id {
                ebml::CLUSTER => {}
                ebml::EBML | ebml::SEGMENT => break,
                _ => {
                    cursor = header.known_end()?;
                    continue;
                }
            }
            if state == SpanState::After {
                lookahead_clusters += 1;
                if lookahead_clusters > LOOKAHEAD_CLUSTERS {
                    break;
                }
            }
            let cluster = header.start;
            let options = &self.options;
            let (end, stopped) = scan
                .cluster_blocks(header, self.segment_end, |block| {
                    let is_boundary = |boundary: Option<Boundary>| {
                        boundary.is_some_and(|boundary| {
                            cluster == boundary.cluster_offset
                                && block.track == video_number
                                && block.timestamp == boundary.time
                        })
                    };
                    if state == SpanState::Before {
                        if !is_boundary(start) {
                            return Ok(Visit::Continue);
                        }
                        state = SpanState::Within;
                    } else if state == SpanState::Within && is_boundary(stop) {
                        state = SpanState::After;
                    }
                    if state == SpanState::Within {
                        block.commit(&mut entries, cluster, options)?;
                        return Ok(Visit::Continue);
                    }
                    // Past the span: only when each track's next block comes
                    // is worth knowing, and once every track has one there is
                    // nothing more to read.
                    if let Some(&codec) = codecs.get(&block.track)
                        && block.is_shown(codec)
                    {
                        following.entry(block.track).or_insert(block.timestamp);
                    }
                    Ok(if following.len() == codecs.len() {
                        Visit::Stop
                    } else {
                        Visit::Continue
                    })
                })
                .await?;
            if state == SpanState::Before {
                return Err(malformed(
                    "a WebM cue names a block its Cluster does not hold",
                ));
            }
            if stopped {
                break;
            }
            cursor = end;
        }
        if state == SpanState::Before {
            return Err(malformed(
                "a WebM cue names a block its Cluster does not hold",
            ));
        }

        let mut tracks = Vec::with_capacity(entries.len());
        let mut audio_trims = Vec::new();
        for entry in entries {
            let Some(track) = entry.indexed else {
                continue;
            };
            let cued = cued_blocks(&track, entry.number, &self.raw_cues);
            let (track, trim) = build_track(
                track_id(entry.number)?,
                track,
                &cued,
                following.get(&entry.number).copied(),
                self.clock,
                &self.options,
            )?;
            tracks.push(track);
            audio_trims.extend(trim);
        }
        Ok(WebmCueSpan {
            index: span,
            tracks,
            is_last: stop.is_none(),
            audio_trims,
        })
    }
}

/// Where a span's scan is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpanState {
    /// In the first Cluster, before the span's cued key frame.
    Before,
    Within,
    /// Past the next span's cued key frame.
    After,
}

/// The raw timestamp of the first video block, from the first few Clusters.
async fn first_video_time<S: ByteSource + ?Sized>(
    scan: &mut Scan<'_, S>,
    first_cluster: u64,
    segment_end: u64,
    video_number: u64,
) -> Result<Option<i64>> {
    let mut cursor = first_cluster;
    let mut clusters = 0;
    while cursor < segment_end && clusters < FIRST_VIDEO_CLUSTERS {
        let header = scan.header(cursor, segment_end).await?;
        match header.id {
            ebml::CLUSTER => {}
            ebml::EBML | ebml::SEGMENT => break,
            _ => {
                cursor = header.known_end()?;
                continue;
            }
        }
        clusters += 1;
        let mut found = None;
        let (end, _) = scan
            .cluster_blocks(header, segment_end, |block| {
                if block.track != video_number {
                    return Ok(Visit::Continue);
                }
                found = Some(block.timestamp);
                Ok(Visit::Stop)
            })
            .await?;
        if found.is_some() {
            return Ok(found);
        }
        cursor = end;
    }
    Ok(None)
}

/// Each `Seek` entry of a `SeekHead`: the ID of the element it locates, and
/// that element's position from the start of the Segment's data.
fn parse_seek_head(payload: &[u8]) -> Result<Vec<(u32, u64)>> {
    let mut seeks = Vec::new();
    for child in children(payload) {
        let (id, value) = child?;
        if id != ebml::SEEK {
            continue;
        }
        let mut seek_id = None;
        let mut position = None;
        for child in children(value) {
            let (id, value) = child?;
            match id {
                // The ID as its own bytes, which is its value as an integer.
                ebml::SEEK_ID if value.len() <= 4 => {
                    seek_id = u32::try_from(read_uint(value)?).ok();
                }
                ebml::SEEK_POSITION => position = Some(read_uint(value)?),
                _ => {}
            }
        }
        if let (Some(id), Some(position)) = (seek_id, position) {
            seeks.push((id, position));
        }
    }
    Ok(seeks)
}

/// The absolute position of the element `id` the `SeekHead` locates, if it
/// locates one.
fn element_position(seeks: &[(u32, u64)], id: u32, segment_start: u64) -> Option<u64> {
    seeks
        .iter()
        .find(|(seek_id, _)| *seek_id == id)
        .and_then(|&(_, position)| segment_start.checked_add(position))
}

/// The payload of the element at `position`, if it is the `id` element a
/// `SeekHead` said it would be.
async fn element<S: ByteSource + ?Sized>(
    scan: &mut Scan<'_, S>,
    position: u64,
    id: u32,
    segment_end: u64,
) -> Result<Option<Vec<u8>>> {
    if position >= segment_end {
        return Ok(None);
    }
    let header: Header = scan.header(position, segment_end).await?;
    if header.id != id {
        return Ok(None);
    }
    let end = header.known_end()?;
    Ok(Some(scan.payload(header, end).await?))
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn unsupported(message: &str) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

fn malformed(message: &str) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{EncodedSample, EncoderConfig, SampleDependency};
    use crate::io::{IoFuture, MemorySink, MemorySource};
    use crate::media::{Codec, VideoDimensions};
    use crate::mp4::{Mp4TrackConfig, Mp4TrackFormat};
    use crate::opus::OpusHead;
    use crate::{Limits, WebmDemuxer, WebmMuxer};
    use std::cell::Cell;
    use std::future::Future;
    use std::task::{Context, Poll, Waker};

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
        }
    }

    /// Video frames a group of pictures holds in [`webm`].
    const GROUP: u64 = 10;
    /// An Opus packet's samples: 20 ms at 48 kHz.
    const PACKET: u64 = 960;

    /// A source that counts the reads it is asked for.
    struct CountingSource {
        inner: MemorySource,
        reads: Cell<usize>,
    }

    impl ByteSource for CountingSource {
        fn len(&self) -> Option<u64> {
            self.inner.len()
        }

        fn read_at<'a>(&'a self, offset: u64, destination: &'a mut [u8]) -> IoFuture<'a, usize> {
            self.reads.set(self.reads.get() + 1);
            self.inner.read_at(offset, destination)
        }
    }

    /// A WebM of `groups` groups of pictures of VP8 at 30 fps, each opening on
    /// a key frame the muxer cues and holding a hidden frame decoded only for
    /// its references, with a stereo Opus track of 20 ms packets interleaved.
    /// The frames are not real VP8; only their first bytes, which say whether
    /// a VP8 frame is shown, mean anything.
    fn webm(groups: u64) -> Vec<u8> {
        let limits = Limits::default();
        let head = OpusHead::new(2, 312, 48_000).unwrap();
        let configs = vec![
            Mp4TrackConfig {
                encoder: EncoderConfig {
                    codec: Codec::Vp8,
                    timescale: 30,
                    decoder_config: Vec::new(),
                },
                format: Mp4TrackFormat::Video(VideoDimensions::new(64, 48, &limits).unwrap()),
            },
            Mp4TrackConfig {
                encoder: EncoderConfig {
                    codec: Codec::Opus,
                    timescale: 48_000,
                    decoder_config: head.to_dops(),
                },
                format: Mp4TrackFormat::Audio { channels: 2 },
            },
        ];
        let frames = groups * GROUP;
        let mut ordered = Vec::new();
        for frame in 0..frames {
            let shown = frame % GROUP != 3;
            ordered.push((
                frame as f64 / 30.0,
                0,
                EncodedSample {
                    // A shown VP8 frame tag has bit 4 set.
                    data: vec![if shown { 0x10 } else { 0x00 }, frame as u8, 7, 7],
                    dts: frame as i64,
                    pts: frame as i64,
                    duration: 1,
                    is_sync: frame % GROUP == 0,
                    dependency: if frame % GROUP == 0 {
                        SampleDependency::INDEPENDENT
                    } else {
                        SampleDependency::DEPENDENT
                    },
                },
            ));
        }
        let packets = frames * 48_000 / 30 / PACKET;
        for packet in 0..packets {
            let start = (packet * PACKET) as i64;
            ordered.push((
                start as f64 / 48_000.0,
                1,
                EncodedSample {
                    // A 20 ms stereo CELT packet.
                    data: vec![0xFC, packet as u8, 1, 2, 3],
                    dts: start,
                    pts: start,
                    duration: PACKET as u32,
                    is_sync: true,
                    dependency: SampleDependency::INDEPENDENT,
                },
            ));
        }
        ordered.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut muxer = block_on(WebmMuxer::new(MemorySink::new(), configs, 1_000_000)).unwrap();
        for (_, track, sample) in ordered {
            block_on(muxer.write_sample(track, sample)).unwrap();
        }
        block_on(muxer.finish()).unwrap().into_inner()
    }

    fn open(bytes: &[u8]) -> (WebmDemuxer, WebmCuedIndex, MemorySource) {
        let source = MemorySource::new(bytes.to_vec());
        let whole = block_on(WebmDemuxer::open(&source, WebmDemuxerOptions::default())).unwrap();
        let cued = block_on(WebmCuedIndex::open(&source, WebmDemuxerOptions::default()))
            .unwrap()
            .expect("the muxer writes Cues");
        (whole, cued, source)
    }

    /// Every span's samples, joined, are the whole file's, sample for sample:
    /// the same bytes, presentation times, durations - a span's last frame
    /// lasting until the next span's first, its hidden frames included - and
    /// random-access points, in the same presentation order.
    #[test]
    fn spans_index_every_block_exactly_as_scanning_the_whole_file_does() {
        let (whole, cued, source) = open(&webm(6));
        assert_eq!(cued.span_count(), 6);
        assert_eq!(cued.video_track, whole.tracks[0].id);
        assert_eq!(cued.duration_seconds, whole.duration_seconds);
        assert_eq!(cued.timescale(), whole.tracks[0].timescale);
        let spans: Vec<WebmCueSpan> = (0..cued.span_count())
            .map(|span| block_on(cued.index_span(&source, span)).unwrap())
            .collect();
        assert!(spans.last().unwrap().is_last);
        assert!(spans.iter().rev().skip(1).all(|span| !span.is_last));
        for track in &whole.tracks {
            let mut samples = Vec::new();
            let mut presentation_order = Vec::new();
            for span in &spans {
                let part = span.track(track.id).unwrap();
                let base = samples.len();
                assert_eq!(
                    (part.kind, part.codec, part.timescale, &part.decoder_config),
                    (
                        track.kind,
                        track.codec,
                        track.timescale,
                        &track.decoder_config
                    )
                );
                presentation_order.extend(part.presentation_order.iter().map(|index| base + index));
                samples.extend(part.samples.iter().cloned());
            }
            assert_eq!(samples, track.samples, "track {}", track.id);
            assert_eq!(presentation_order, track.presentation_order);
        }
        // Each span after the first opens on its cued key frame, so it
        // decodes on its own.
        for span in &spans {
            let video = span.track(cued.video_track).unwrap();
            assert!(video.samples[0].is_sync);
            assert_eq!(
                cued.span_start(span.index),
                (span.index > 0).then(|| video.samples[0].pts)
            );
            assert_eq!(cued.span_at(video.samples[0].pts), span.index);
            assert_eq!(cued.span_at(video.samples.last().unwrap().pts), span.index);
        }
        let audio = whole.tracks[1].id;
        assert_eq!(
            cued.audio_timing(audio, spans.last()).unwrap(),
            whole.audio_timing(audio).unwrap()
        );
        // The Opus packets' intervals, a span at a time, are the ones the
        // whole track's index gives.
        let first_pts = spans[0].track(audio).unwrap().samples[0].pts;
        let ranges: Vec<SampleRange> = spans
            .iter()
            .flat_map(|span| {
                block_on(cued.opus_decoded_ranges(&source, span, audio, first_pts)).unwrap()
            })
            .collect();
        assert_eq!(
            ranges,
            whole.tracks[1].opus_decoded_ranges(PACKET as u32).unwrap()
        );
        assert_eq!(cued.audio_timing(audio, spans.first()).unwrap().padding, 0);
    }

    /// Issue #692: opening reads the file's header elements and `Cues`, and
    /// the first block headers, however long the file is.
    #[test]
    fn opening_reads_the_same_few_elements_however_long_the_file_is() {
        let reads = |groups| {
            let source = CountingSource {
                inner: MemorySource::new(webm(groups)),
                reads: Cell::new(0),
            };
            let cued = block_on(WebmCuedIndex::open(&source, WebmDemuxerOptions::default()))
                .unwrap()
                .unwrap();
            assert_eq!(cued.span_count() as u64, groups);
            source.reads.get()
        };
        let short = reads(2);
        let long = reads(200);
        assert_eq!(short, long);
        assert!(short < 24, "opening read {short} times");
        // Indexing the whole file reads every block header.
        let source = CountingSource {
            inner: MemorySource::new(webm(200)),
            reads: Cell::new(0),
        };
        block_on(WebmDemuxer::open(&source, WebmDemuxerOptions::default())).unwrap();
        assert!(source.reads.get() > 20 * long);
    }

    /// A WebM whose `Cues` cannot be found is one to index by scanning.
    #[test]
    fn a_webm_without_cues_is_left_to_the_whole_file_scan() {
        let mut bytes = webm(3);
        let cues = ebml::CUES.to_be_bytes();
        let at = bytes.windows(4).rposition(|window| window == cues).unwrap();
        bytes[at..at + 4].copy_from_slice(&ebml::TAGS.to_be_bytes());
        let source = MemorySource::new(bytes);
        assert!(
            block_on(WebmCuedIndex::open(&source, WebmDemuxerOptions::default()))
                .unwrap()
                .is_none()
        );
        let whole = block_on(WebmDemuxer::open(&source, WebmDemuxerOptions::default())).unwrap();
        assert!(whole.cues.is_empty());
    }

    #[test]
    fn a_span_past_the_last_is_refused() {
        let (_, cued, source) = open(&webm(2));
        assert_eq!(
            block_on(cued.index_span(&source, 2)).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
    }
}
