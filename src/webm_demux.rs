//! Bounded, read-only WebM (Matroska/EBML) probing and sample indexing.
//!
//! [`WebmDemuxer::open`] builds the same decode-order sample index
//! [`crate::Mp4Demuxer`] does, as [`Mp4Track`] values, so everything that
//! consumes an MP4 index - [`Mp4Track::to_encoded_video_samples`],
//! [`crate::ExactFrameReader`], the browser decoder - reads a WebM track the
//! same way. Only metadata elements are read whole; a block's payload is never
//! read, only its header and lacing, so indexing costs the same however large
//! the frames are.

use crate::codec::{SampleDependency, TrackKind};
use crate::ebml::{self, children, read_float, read_id, read_known_vint, read_string, read_uint};
use crate::io::ByteSource;
use crate::media::{Codec, VideoDimensions};
use crate::mp4_demux::{Mp4Sample, Mp4Track};
use crate::{Error, ErrorKind, Limits, Result};
use std::collections::BTreeMap;

/// Matroska's default `TimestampScale`: one tick is a millisecond.
const DEFAULT_TIMESTAMP_SCALE: u64 = 1_000_000;
const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;
/// The longest element header: a four-byte ID and an eight-byte size.
const MAX_HEADER: usize = 12;
/// Enough of an unlaced block for its track number, timestamp and flags, and
/// the first byte of its frame, which holds a VP8 frame's `show_frame` bit.
const BLOCK_PREFIX: usize = 12;
/// The Matroska invisible flag, in both SimpleBlock and Block flags.
const INVISIBLE: u8 = 0x08;

/// Resource limits specific to untrusted WebM structure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WebmDemuxerOptions {
    pub limits: Limits,
    /// Largest element read whole: the EBML header, `Info`, `Tracks`, `Cues`,
    /// and a laced block, whose lace sizes precede its frames.
    pub max_element_bytes: u64,
    /// Every element header visited, blocks included.
    pub max_elements: u64,
    pub max_samples_per_track: u32,
}

impl Default for WebmDemuxerOptions {
    fn default() -> Self {
        Self {
            limits: Limits::default(),
            max_element_bytes: 64 * 1024 * 1024,
            max_elements: 100_000_000,
            max_samples_per_track: 10_000_000,
        }
    }
}

/// One `CuePoint` track position: where a decode of `track` may start.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WebmCuePoint {
    /// Presentation time in the track's timescale ticks.
    pub time: i64,
    /// The Matroska track number, which is also the [`Mp4Track::id`].
    pub track: u32,
    /// Absolute byte offset of the Cluster holding the cued block.
    pub cluster_offset: u64,
}

/// Where a decode reaching a requested time can start.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WebmSeekPoint {
    /// Presentation time of the random-access point, in track ticks.
    pub time: i64,
    /// Absolute byte offset to start reading from: the cued Cluster when the
    /// file has `Cues`, the random-access block's frame otherwise.
    pub offset: u64,
    /// Whether `offset` came from the file's `Cues` rather than the scanned
    /// sample index.
    pub from_cues: bool,
}

/// A track the demuxer left out of [`WebmDemuxer::tracks`] because zvidlib
/// has no reader for it, such as an Opus or Vorbis audio track.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WebmSkippedTrack {
    pub number: u64,
    pub codec_id: String,
}

/// Parsed WebM segment metadata and indexes.
#[derive(Clone, Debug, PartialEq)]
pub struct WebmDemuxer {
    /// `"webm"`, or `"matroska"` for a Matroska file using the same subset.
    pub doc_type: String,
    /// The segment's `Duration`, in seconds, when the file declares one.
    pub duration_seconds: Option<f64>,
    /// Indexed video tracks, in `Tracks` order. Each track's
    /// [`Mp4Track::timescale`] is derived from the segment's
    /// `TimestampScale`, and a V_AV1 track's `decoder_config` is its
    /// `CodecPrivate` wrapped in an `av1C` box, as an MP4 track's is. A V_VP9
    /// track's is the `vpcC` box its `CodecPrivate` describes (see
    /// [`crate::Vp9CodecConfig`]), and a V_VP8 track has none.
    pub tracks: Vec<Mp4Track>,
    /// Cue points, in file order, for the indexed tracks. Empty when the file
    /// has no `Cues`, as a live `MediaRecorder` capture usually does.
    pub cues: Vec<WebmCuePoint>,
    pub skipped_tracks: Vec<WebmSkippedTrack>,
}

/// Bounded format detection: whether `source` starts with an EBML header
/// whose `DocType` is `webm` or `matroska`. At most the header is read.
pub async fn probe_webm<S: ByteSource + ?Sized>(source: &S) -> Result<bool> {
    let Some(source_len) = source.len() else {
        return Ok(false);
    };
    let mut prefix = [0_u8; MAX_HEADER];
    let length = read_up_to(source, 0, &mut prefix).await?;
    let prefix = &prefix[..length];
    if !prefix.starts_with(&ebml::EBML.to_be_bytes()) {
        return Ok(false);
    }
    let Ok((Some(size), size_length)) = ebml::read_vint(&prefix[4..]) else {
        return Ok(false);
    };
    let start = 4 + size_length as u64;
    if size > 4096 || start + size > source_len {
        return Ok(false);
    }
    let mut payload = vec![0_u8; size as usize];
    read_exact(source, start, &mut payload).await?;
    Ok(parse_ebml_header(&payload).is_ok())
}

impl WebmDemuxer {
    /// Scans the segment once, reading only bounded metadata elements and
    /// block headers, and builds every indexed track's sample table.
    pub async fn open<S: ByteSource + ?Sized>(
        source: &S,
        options: WebmDemuxerOptions,
    ) -> Result<Self> {
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
        let mut tracks: Option<Vec<TrackEntry>> = None;
        let mut cues = Vec::new();
        let mut cursor = segment_start;
        while cursor < segment_end {
            let header = scan.header(cursor, segment_end).await?;
            cursor = match header.id {
                ebml::CLUSTER => {
                    let tracks = tracks
                        .as_mut()
                        .ok_or_else(|| malformed("a WebM Cluster precedes the segment's Tracks"))?;
                    scan.cluster(header, segment_end, tracks).await?
                }
                ebml::INFO => {
                    let end = header.known_end()?;
                    if info.is_some() {
                        return Err(malformed("WebM has more than one Info element"));
                    }
                    info = Some(parse_info(&scan.payload(header, end).await?)?);
                    end
                }
                ebml::TRACKS => {
                    let end = header.known_end()?;
                    if tracks.is_some() {
                        return Err(malformed("WebM has more than one Tracks element"));
                    }
                    tracks = Some(parse_tracks(&scan.payload(header, end).await?, &options)?);
                    end
                }
                ebml::CUES => {
                    let end = header.known_end()?;
                    cues.extend(parse_cues(
                        &scan.payload(header, end).await?,
                        segment_start,
                    )?);
                    end
                }
                // A second Segment or EBML header chains another file onto
                // this one; only the first is indexed.
                ebml::EBML | ebml::SEGMENT => break,
                _ => header.known_end()?,
            };
        }

        let info = info.ok_or_else(|| malformed("WebM does not contain an Info element"))?;
        let entries = tracks.ok_or_else(|| malformed("WebM does not contain a Tracks element"))?;
        finish(doc_type, info, entries, cues, &options)
    }

    pub fn track(&self, id: u32) -> Option<&Mp4Track> {
        self.tracks.iter().find(|track| track.id == id)
    }

    /// The random-access point a decode reaching `time` (in track ticks)
    /// starts from: the latest cue at or before it when the file has `Cues`
    /// for the track, otherwise the latest sync sample at or before it from
    /// the scanned index. `None` when the track does not exist or nothing
    /// precedes `time`.
    pub fn seek_point(&self, track_id: u32, time: i64) -> Option<WebmSeekPoint> {
        let track = self.track(track_id)?;
        let cued = self
            .cues
            .iter()
            .filter(|cue| cue.track == track_id && cue.time <= time)
            .max_by_key(|cue| cue.time);
        if let Some(cue) = cued {
            return Some(WebmSeekPoint {
                time: cue.time,
                offset: cue.cluster_offset,
                from_cues: true,
            });
        }
        if self.cues.iter().any(|cue| cue.track == track_id) {
            return None;
        }
        track
            .samples
            .iter()
            .filter(|sample| sample.is_sync && sample.pts <= time)
            .max_by_key(|sample| sample.pts)
            .map(|sample| WebmSeekPoint {
                time: sample.pts,
                offset: sample.offset,
                from_cues: false,
            })
    }
}

#[derive(Clone, Copy, Debug)]
struct Header {
    id: u32,
    start: u64,
    data_start: u64,
    /// `None` for an unknown-size element.
    end: Option<u64>,
}

impl Header {
    fn known_end(self) -> Result<u64> {
        self.end
            .ok_or_else(|| malformed("only a Segment or Cluster may have an unknown EBML size"))
    }
}

struct Info {
    timestamp_scale: u64,
    duration: Option<f64>,
}

struct TrackEntry {
    number: u64,
    codec_id: String,
    /// `Some` for a track that is indexed.
    indexed: Option<IndexedTrack>,
}

struct IndexedTrack {
    codec: Codec,
    dimensions: VideoDimensions,
    decoder_config: Vec<u8>,
    default_duration_ns: Option<u64>,
    frames: Vec<Frame>,
    blocks: Vec<Block>,
}

#[derive(Clone, Copy)]
struct Frame {
    offset: u64,
    size: u32,
    /// `false` for a decode-only frame that is never presented, such as a
    /// hidden VP8 alternate reference stored as a block of its own.
    shown: bool,
}

struct Block {
    /// Raw timestamp in `TimestampScale` ticks.
    timestamp: i64,
    first_frame: usize,
    frames: u32,
    keyframe: bool,
    /// Whether any of the block's frames is presented.
    shown: bool,
    /// `BlockDuration`, in raw ticks.
    duration: Option<u64>,
    cluster_offset: u64,
}

struct Scan<'a, S: ?Sized> {
    source: &'a S,
    options: &'a WebmDemuxerOptions,
    elements: u64,
}

impl<S: ByteSource + ?Sized> Scan<'_, S> {
    /// Reads the element header at `start`, which must end by `parent_end`.
    async fn header(&mut self, start: u64, parent_end: u64) -> Result<Header> {
        self.elements = self
            .elements
            .checked_add(1)
            .ok_or_else(|| limit("WebM element count overflow"))?;
        if self.elements > self.options.max_elements {
            return Err(limit("WebM element count limit exceeded"));
        }
        let available = parent_end
            .checked_sub(start)
            .ok_or_else(|| malformed("EBML element starts outside its parent"))?;
        let mut bytes = [0_u8; MAX_HEADER];
        let wanted = usize::try_from(available.min(MAX_HEADER as u64)).expect("bounded");
        read_exact(self.source, start, &mut bytes[..wanted]).await?;
        let bytes = &bytes[..wanted];
        let (id, id_length) = read_id(bytes)?;
        let (size, size_length) = ebml::read_vint(&bytes[id_length..])?;
        let data_start = start + (id_length + size_length) as u64;
        let end = match size {
            Some(size) => {
                let end = data_start
                    .checked_add(size)
                    .ok_or_else(|| malformed("EBML element size overflow"))?;
                if end > parent_end {
                    return Err(malformed("EBML element exceeds its parent"));
                }
                Some(end)
            }
            None => None,
        };
        Ok(Header {
            id,
            start,
            data_start,
            end,
        })
    }

    async fn payload(&self, header: Header, end: u64) -> Result<Vec<u8>> {
        let length = end - header.data_start;
        if length > self.options.max_element_bytes {
            return Err(limit("WebM element exceeds the configured element limit"));
        }
        ensure_bytes(length, self.options, "WebM element")?;
        let mut bytes = vec![0_u8; length as usize];
        read_exact(self.source, header.data_start, &mut bytes).await?;
        Ok(bytes)
    }

    /// Indexes one Cluster's blocks and returns where the Cluster ends. An
    /// unknown-size Cluster ends at the next Segment child or EBML header.
    async fn cluster(
        &mut self,
        cluster: Header,
        segment_end: u64,
        tracks: &mut [TrackEntry],
    ) -> Result<u64> {
        let end = cluster.end.unwrap_or(segment_end);
        let mut timestamp = None;
        let mut cursor = cluster.data_start;
        while cursor < end {
            let header = self.header(cursor, end).await?;
            if cluster.end.is_none()
                && (ebml::is_segment_child(header.id)
                    || matches!(header.id, ebml::EBML | ebml::SEGMENT))
            {
                return Ok(cursor);
            }
            let element_end = header.known_end()?;
            match header.id {
                ebml::TIMESTAMP => {
                    if element_end - header.data_start > 8 {
                        return Err(malformed("WebM Cluster timestamp is too long"));
                    }
                    let value = read_uint(&self.payload(header, element_end).await?)?;
                    timestamp = Some(
                        i64::try_from(value)
                            .map_err(|_| malformed("WebM Cluster timestamp is out of range"))?,
                    );
                }
                ebml::SIMPLE_BLOCK => {
                    let timestamp = timestamp
                        .ok_or_else(|| malformed("a WebM block precedes its Cluster timestamp"))?;
                    self.block(header.data_start, element_end, timestamp, None, None)
                        .await?
                        .commit(tracks, cluster.start, self.options)?;
                }
                ebml::BLOCK_GROUP => {
                    let timestamp = timestamp
                        .ok_or_else(|| malformed("a WebM block precedes its Cluster timestamp"))?;
                    self.block_group(header, element_end, timestamp)
                        .await?
                        .commit(tracks, cluster.start, self.options)?;
                }
                _ => {}
            }
            cursor = element_end;
        }
        Ok(end)
    }

    async fn block_group(
        &mut self,
        group: Header,
        end: u64,
        timestamp: i64,
    ) -> Result<ParsedBlock> {
        let mut block = None;
        let mut duration = None;
        let mut references = 0_u32;
        let mut cursor = group.data_start;
        while cursor < end {
            let header = self.header(cursor, end).await?;
            let element_end = header.known_end()?;
            match header.id {
                ebml::BLOCK => block = Some((header.data_start, element_end)),
                ebml::BLOCK_DURATION => {
                    if element_end - header.data_start > 8 {
                        return Err(malformed("WebM BlockDuration is too long"));
                    }
                    duration = Some(read_uint(&self.payload(header, element_end).await?)?);
                }
                ebml::REFERENCE_BLOCK => references = references.saturating_add(1),
                _ => {}
            }
            cursor = element_end;
        }
        let (start, block_end) =
            block.ok_or_else(|| malformed("a WebM BlockGroup contains no Block"))?;
        // A Block's flags carry no keyframe bit; a block that references no
        // other block is the random-access point.
        self.block(start, block_end, timestamp, Some(references == 0), duration)
            .await
    }

    /// Parses a block's header and lacing. Only an unlaced block's first few
    /// bytes are read; a laced one is read whole, bounded like any element,
    /// because its lace sizes precede its frames.
    async fn block(
        &self,
        start: u64,
        end: u64,
        cluster_timestamp: i64,
        keyframe: Option<bool>,
        duration: Option<u64>,
    ) -> Result<ParsedBlock> {
        let length = end - start;
        let mut prefix = [0_u8; BLOCK_PREFIX];
        let wanted = usize::try_from(length.min(BLOCK_PREFIX as u64)).expect("bounded");
        read_exact(self.source, start, &mut prefix[..wanted]).await?;
        let prefix = &prefix[..wanted];
        let (track, track_length) = read_known_vint(prefix)?;
        let header_length = track_length + 3;
        if prefix.len() < header_length {
            return Err(malformed("WebM block header is truncated"));
        }
        let relative = i16::from_be_bytes([prefix[track_length], prefix[track_length + 1]]);
        let flags = prefix[track_length + 2];
        let lacing = (flags >> 1) & 0b11;
        let frames = if lacing == 0 {
            let size = length - header_length as u64;
            vec![(
                start + header_length as u64,
                frame_size(size)?,
                prefix.get(header_length).copied(),
            )]
        } else {
            if length > self.options.max_element_bytes {
                return Err(limit(
                    "laced WebM block exceeds the configured element limit",
                ));
            }
            ensure_bytes(length, self.options, "laced WebM block")?;
            let mut bytes = vec![0_u8; length as usize];
            read_exact(self.source, start, &mut bytes).await?;
            parse_lacing(&bytes, header_length, lacing)?
                .into_iter()
                .map(|(offset, size)| {
                    let first_byte = (size > 0).then(|| bytes[offset]);
                    Ok((start + offset as u64, frame_size(size as u64)?, first_byte))
                })
                .collect::<Result<_>>()?
        };
        Ok(ParsedBlock {
            track,
            timestamp: cluster_timestamp
                .checked_add(i64::from(relative))
                .ok_or_else(|| malformed("WebM block timestamp overflow"))?,
            keyframe: keyframe.unwrap_or(flags & 0x80 != 0),
            invisible: flags & INVISIBLE != 0,
            duration,
            frames,
        })
    }
}

struct ParsedBlock {
    track: u64,
    timestamp: i64,
    keyframe: bool,
    invisible: bool,
    duration: Option<u64>,
    /// Each frame's offset, size and first byte.
    frames: Vec<(u64, u32, Option<u8>)>,
}

impl ParsedBlock {
    fn commit(
        self,
        tracks: &mut [TrackEntry],
        cluster_offset: u64,
        options: &WebmDemuxerOptions,
    ) -> Result<()> {
        let entry = tracks
            .iter_mut()
            .find(|entry| entry.number == self.track)
            .ok_or_else(|| malformed("a WebM block references an undeclared track"))?;
        let Some(track) = entry.indexed.as_mut() else {
            return Ok(());
        };
        let total = track.frames.len() + self.frames.len();
        if total > options.max_samples_per_track as usize {
            return Err(limit("WebM track sample limit exceeded"));
        }
        ensure_allocation(
            total,
            std::mem::size_of::<Frame>() + std::mem::size_of::<Mp4Sample>(),
            options,
            "WebM sample index",
        )?;
        let first_frame = track.frames.len();
        for (offset, size, first_byte) in self.frames {
            // A VP8 frame with `show_frame` (bit 4 of its frame tag) clear is
            // decoded for its references but never presented (RFC 6386
            // section 9.1), and Matroska's invisible flag says the same of
            // the whole block.
            //
            // A VP9 block is a chunk that shows one frame, carrying hidden
            // frames ahead of it in a superframe; whether a chunk shows a
            // frame is only known from its last frame, so a block that does
            // not is identified by the invisible flag alone.
            let shown = match track.codec {
                Codec::Vp8 => !self.invisible && first_byte.is_none_or(|tag| tag & 0x10 != 0),
                Codec::Vp9 => !self.invisible,
                _ => true,
            };
            track.frames.push(Frame {
                offset,
                size,
                shown,
            });
        }
        track.blocks.push(Block {
            timestamp: self.timestamp,
            first_frame,
            frames: u32::try_from(track.frames.len() - first_frame)
                .expect("a lace holds at most 256 frames"),
            keyframe: self.keyframe,
            shown: track.frames[first_frame..].iter().any(|frame| frame.shown),
            duration: self.duration,
            cluster_offset,
        });
        Ok(())
    }
}

/// Splits a laced block into `(offset, size)` frames relative to the block.
fn parse_lacing(bytes: &[u8], header_length: usize, lacing: u8) -> Result<Vec<(usize, usize)>> {
    let count = usize::from(
        *bytes
            .get(header_length)
            .ok_or_else(|| malformed("WebM lace count is truncated"))?,
    ) + 1;
    let mut cursor = header_length + 1;
    let mut sizes = Vec::with_capacity(count);
    match lacing {
        // Xiph: each size but the last is a run of 255s plus a final byte.
        0b01 => {
            for _ in 1..count {
                let mut size = 0_usize;
                loop {
                    let byte = *bytes
                        .get(cursor)
                        .ok_or_else(|| malformed("WebM Xiph lace size is truncated"))?;
                    cursor += 1;
                    size += usize::from(byte);
                    if byte != 255 {
                        break;
                    }
                }
                sizes.push(size);
            }
        }
        // EBML: the first size, then signed differences from the previous.
        0b11 => {
            if count > 1 {
                let (first, length) = read_known_vint(&bytes[cursor..])?;
                cursor += length;
                let mut size = i64::try_from(first)
                    .map_err(|_| malformed("WebM EBML lace size is out of range"))?;
                sizes.push(size as usize);
                for _ in 2..count {
                    let (difference, length) = ebml::read_signed_vint(&bytes[cursor..])?;
                    cursor += length;
                    size = size
                        .checked_add(difference)
                        .filter(|&size| size >= 0)
                        .ok_or_else(|| malformed("WebM EBML lace size is negative"))?;
                    sizes.push(size as usize);
                }
            }
        }
        // Fixed: every frame is the same size.
        _ => {
            let data = bytes.len().saturating_sub(cursor);
            if data % count != 0 {
                return Err(malformed("WebM fixed-size lace does not divide its block"));
            }
            sizes.extend(std::iter::repeat_n(data / count, count - 1));
        }
    }
    let laced: usize = sizes.iter().sum();
    let last = bytes
        .len()
        .checked_sub(cursor)
        .and_then(|data| data.checked_sub(laced))
        .ok_or_else(|| malformed("WebM lace sizes exceed their block"))?;
    sizes.push(last);
    let mut frames = Vec::with_capacity(count);
    for size in sizes {
        if size == 0 {
            return Err(malformed("WebM lace contains an empty frame"));
        }
        frames.push((cursor, size));
        cursor += size;
    }
    Ok(frames)
}

fn parse_ebml_header(payload: &[u8]) -> Result<String> {
    let mut doc_type = None;
    for child in children(payload) {
        let (id, value) = child?;
        match id {
            ebml::EBML_READ_VERSION if read_uint(value)? > 1 => {
                return Err(unsupported(
                    "WebM requires an unsupported EBML reader version",
                ));
            }
            ebml::EBML_MAX_ID_LENGTH if read_uint(value)? > 4 => {
                return Err(unsupported(
                    "WebM element IDs longer than four bytes are unsupported",
                ));
            }
            ebml::EBML_MAX_SIZE_LENGTH if read_uint(value)? > 8 => {
                return Err(unsupported(
                    "WebM element sizes longer than eight bytes are unsupported",
                ));
            }
            ebml::DOC_TYPE_READ_VERSION if read_uint(value)? > 4 => {
                return Err(unsupported(
                    "WebM requires an unsupported Matroska reader version",
                ));
            }
            ebml::DOC_TYPE => doc_type = Some(read_string(value)?),
            _ => {}
        }
    }
    // `DocType` defaults to `matroska` when the header leaves it out.
    let doc_type = doc_type.unwrap_or_else(|| "matroska".to_owned());
    match doc_type.as_str() {
        "webm" | "matroska" => Ok(doc_type),
        _ => Err(unsupported("EBML document type is not WebM or Matroska")),
    }
}

fn parse_info(payload: &[u8]) -> Result<Info> {
    let mut info = Info {
        timestamp_scale: DEFAULT_TIMESTAMP_SCALE,
        duration: None,
    };
    for child in children(payload) {
        let (id, value) = child?;
        match id {
            ebml::TIMESTAMP_SCALE => info.timestamp_scale = read_uint(value)?,
            ebml::DURATION => info.duration = Some(read_float(value)?),
            _ => {}
        }
    }
    if info.timestamp_scale == 0 {
        return Err(malformed("WebM TimestampScale must be nonzero"));
    }
    if info
        .duration
        .is_some_and(|duration| !duration.is_finite() || duration < 0.0)
    {
        return Err(malformed("WebM Duration is not a valid time"));
    }
    Ok(info)
}

fn parse_tracks(payload: &[u8], options: &WebmDemuxerOptions) -> Result<Vec<TrackEntry>> {
    let mut entries: Vec<TrackEntry> = Vec::new();
    for child in children(payload) {
        let (id, value) = child?;
        if id != ebml::TRACK_ENTRY {
            continue;
        }
        if entries.len() >= usize::from(options.limits.max_tracks) {
            return Err(limit("WebM track count exceeds the configured limit"));
        }
        let entry = parse_track_entry(value, options)?;
        if entries.iter().any(|other| other.number == entry.number) {
            return Err(malformed("WebM declares a track number twice"));
        }
        entries.push(entry);
    }
    Ok(entries)
}

fn parse_track_entry(payload: &[u8], options: &WebmDemuxerOptions) -> Result<TrackEntry> {
    let mut number = None;
    let mut track_type = None;
    let mut codec_id = None;
    let mut codec_private = None;
    let mut default_duration = None;
    let mut encoded = false;
    let mut width = None;
    let mut height = None;
    for child in children(payload) {
        let (id, value) = child?;
        match id {
            ebml::TRACK_NUMBER => number = Some(read_uint(value)?),
            ebml::TRACK_TYPE => track_type = Some(read_uint(value)?),
            ebml::CODEC_ID => codec_id = Some(read_string(value)?),
            ebml::CODEC_PRIVATE => codec_private = Some(value),
            ebml::DEFAULT_DURATION => default_duration = Some(read_uint(value)?),
            ebml::CONTENT_ENCODINGS => encoded = true,
            ebml::VIDEO => {
                for child in children(value) {
                    let (id, value) = child?;
                    match id {
                        ebml::PIXEL_WIDTH => width = Some(read_uint(value)?),
                        ebml::PIXEL_HEIGHT => height = Some(read_uint(value)?),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    let number = number
        .filter(|&number| number != 0)
        .ok_or_else(|| malformed("WebM track has no nonzero TrackNumber"))?;
    let codec_id = codec_id.ok_or_else(|| malformed("WebM track has no CodecID"))?;
    if track_type != Some(ebml::TRACK_TYPE_VIDEO) {
        return Ok(TrackEntry {
            number,
            codec_id,
            indexed: None,
        });
    }
    let codec = video_codec(&codec_id)
        .ok_or_else(|| unsupported(format!("unsupported WebM video codec {codec_id}")))?;
    if encoded {
        return Err(unsupported(
            "compressed or encrypted WebM video tracks are unsupported",
        ));
    }
    let dimension = |value: Option<u64>, name: &str| -> Result<u32> {
        value
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| malformed(format!("WebM video track has no valid {name}")))
    };
    let dimensions = VideoDimensions::new(
        dimension(width, "PixelWidth")?,
        dimension(height, "PixelHeight")?,
        &options.limits,
    )?;
    let decoder_config = decoder_config(codec, codec_private)?;
    Ok(TrackEntry {
        number,
        codec_id,
        indexed: Some(IndexedTrack {
            codec,
            dimensions,
            decoder_config,
            default_duration_ns: default_duration.filter(|&duration| duration > 0),
            frames: Vec::new(),
            blocks: Vec::new(),
        }),
    })
}

/// The codec indexed for a Matroska video `CodecID`, or `None` when zvidlib
/// has no decoder for it.
fn video_codec(codec_id: &str) -> Option<Codec> {
    match codec_id {
        "V_AV1" => Some(Codec::Av1),
        "V_VP8" => Some(Codec::Vp8),
        "V_VP9" => Some(Codec::Vp9),
        _ => None,
    }
}

/// Wraps a track's `CodecPrivate` in the MP4 box an [`Mp4Track`] carries, so
/// decoders configure from a WebM track exactly as from an MP4 one.
fn decoder_config(codec: Codec, codec_private: Option<&[u8]>) -> Result<Vec<u8>> {
    match codec {
        Codec::Av1 => {
            let record = codec_private
                .ok_or_else(|| malformed("V_AV1 track has no CodecPrivate av1C record"))?;
            let size = u32::try_from(record.len() + 8)
                .map_err(|_| limit("V_AV1 CodecPrivate is too large"))?;
            let mut config = Vec::with_capacity(size as usize);
            config.extend_from_slice(&size.to_be_bytes());
            config.extend_from_slice(b"av1C");
            config.extend_from_slice(record);
            Ok(config)
        }
        // VP8 defines no configuration record, so any `CodecPrivate` is
        // ignored and decoders configure from each key frame's header.
        Codec::Vp8 => Ok(Vec::new()),
        // VP9's `CodecPrivate` is an optional list of features (profile,
        // level, bit depth, chroma subsampling); it becomes the `vpcC` box an
        // MP4 track would carry, with the profile 0 defaults for any it
        // leaves out.
        Codec::Vp9 => {
            let config =
                crate::Vp9CodecConfig::parse_webm_codec_private(codec_private.unwrap_or_default())?;
            Ok(config.to_vpcc())
        }
        _ => Err(unsupported(
            "codec has no WebM decoder configuration mapping",
        )),
    }
}

fn parse_cues(payload: &[u8], segment_start: u64) -> Result<Vec<RawCue>> {
    let mut cues = Vec::new();
    for child in children(payload) {
        let (id, value) = child?;
        if id != ebml::CUE_POINT {
            continue;
        }
        let mut time = None;
        let mut positions = Vec::new();
        for child in children(value) {
            let (id, value) = child?;
            match id {
                ebml::CUE_TIME => time = Some(read_uint(value)?),
                ebml::CUE_TRACK_POSITIONS => {
                    let mut track = None;
                    let mut cluster = None;
                    for child in children(value) {
                        let (id, value) = child?;
                        match id {
                            ebml::CUE_TRACK => track = Some(read_uint(value)?),
                            ebml::CUE_CLUSTER_POSITION => cluster = Some(read_uint(value)?),
                            _ => {}
                        }
                    }
                    let track = track.ok_or_else(|| malformed("WebM cue has no CueTrack"))?;
                    let cluster = cluster
                        .and_then(|cluster| segment_start.checked_add(cluster))
                        .ok_or_else(|| malformed("WebM cue has no valid CueClusterPosition"))?;
                    positions.push((track, cluster));
                }
                _ => {}
            }
        }
        let time = time.ok_or_else(|| malformed("WebM cue has no CueTime"))?;
        let time = i64::try_from(time).map_err(|_| malformed("WebM CueTime is out of range"))?;
        cues.extend(positions.into_iter().map(|(track, cluster_offset)| RawCue {
            time,
            track,
            cluster_offset,
        }));
    }
    Ok(cues)
}

struct RawCue {
    time: i64,
    track: u64,
    cluster_offset: u64,
}

/// Converts the scanned blocks into sample tables in track ticks.
fn finish(
    doc_type: String,
    info: Info,
    entries: Vec<TrackEntry>,
    raw_cues: Vec<RawCue>,
    options: &WebmDemuxerOptions,
) -> Result<WebmDemuxer> {
    // A track tick is one `TimestampScale` when that divides a second, which
    // the default of one millisecond does; otherwise a nanosecond.
    let (timescale, ticks_per_raw) = if NANOSECONDS_PER_SECOND % info.timestamp_scale == 0 {
        (NANOSECONDS_PER_SECOND / info.timestamp_scale, 1)
    } else {
        (NANOSECONDS_PER_SECOND, info.timestamp_scale)
    };
    let timescale = u32::try_from(timescale).expect("at most a billion");
    let to_ticks = |raw: i64| -> Result<i64> {
        raw.checked_mul(ticks_per_raw as i64)
            .ok_or_else(|| limit("WebM timestamp overflow"))
    };
    let segment_end = info.duration.map(|duration| {
        (duration * info.timestamp_scale as f64 / 1e9 * f64::from(timescale)) as i64
    });

    let mut tracks = Vec::new();
    let mut skipped_tracks = Vec::new();
    let mut cues = Vec::new();
    for entry in entries {
        let Some(track) = entry.indexed else {
            skipped_tracks.push(WebmSkippedTrack {
                number: entry.number,
                codec_id: entry.codec_id,
            });
            continue;
        };
        let id = u32::try_from(entry.number)
            .map_err(|_| unsupported("WebM track numbers above 2^32 are unsupported"))?;
        let mut keyframes: BTreeMap<(u64, i64), usize> = BTreeMap::new();
        for (index, block) in track.blocks.iter().enumerate() {
            keyframes.insert((block.cluster_offset, block.timestamp), index);
        }
        let mut cued = vec![false; track.blocks.len()];
        for cue in raw_cues.iter().filter(|cue| cue.track == entry.number) {
            if let Some(&index) = keyframes.get(&(cue.cluster_offset, cue.time)) {
                cued[index] = true;
            }
            cues.push(WebmCuePoint {
                time: to_ticks(cue.time)?,
                track: id,
                cluster_offset: cue.cluster_offset,
            });
        }
        let default_duration = track.default_duration_ns.map(|duration| {
            (u128::from(duration) * u128::from(timescale) / u128::from(NANOSECONDS_PER_SECOND))
                as u64
        });

        // A shown frame lasts until the next shown block, not the hidden one
        // an encoder may store a tick after it.
        let mut next_shown = vec![None; track.blocks.len()];
        let mut following = None;
        for (index, block) in track.blocks.iter().enumerate().rev() {
            next_shown[index] = following;
            if block.shown {
                following = Some(block.timestamp);
            }
        }
        let mut samples = Vec::with_capacity(track.frames.len());
        let mut shown = Vec::with_capacity(track.frames.len());
        let mut previous_frame_duration = None;
        let mut dts = 0_u64;
        for (index, block) in track.blocks.iter().enumerate() {
            let pts = to_ticks(block.timestamp)?;
            let next = next_shown[index].map(to_ticks).transpose()?;
            let frames = u64::from(block.frames);
            let duration = block
                .duration
                .map(|duration| duration.saturating_mul(ticks_per_raw))
                .or_else(|| {
                    next.and_then(|next| u64::try_from(next.checked_sub(pts)?).ok())
                        .filter(|&delta| delta > 0)
                })
                .or_else(|| default_duration.map(|duration| duration.saturating_mul(frames)))
                .or_else(|| {
                    segment_end
                        .and_then(|end| u64::try_from(end.checked_sub(pts)?).ok())
                        .filter(|&delta| delta > 0)
                })
                .or_else(|| previous_frame_duration.map(|duration: u64| duration * frames))
                .unwrap_or(frames)
                .max(frames);
            let frame_duration = duration / frames;
            previous_frame_duration = Some(frame_duration);
            let keyframe = block.keyframe || cued[index];
            let mut frame_pts = pts;
            for frame in 0..block.frames {
                let Frame {
                    offset,
                    size,
                    shown: frame_shown,
                } = track.frames[block.first_frame + frame as usize];
                shown.push(frame_shown);
                let this_duration = if frame + 1 == block.frames {
                    duration - frame_duration * (frames - 1)
                } else {
                    frame_duration
                };
                dts = dts.max(u64::try_from(frame_pts.max(0)).expect("nonnegative"));
                samples.push(Mp4Sample {
                    offset,
                    size,
                    dts,
                    pts: frame_pts,
                    duration: u32::try_from(this_duration)
                        .map_err(|_| limit("WebM frame duration is out of range"))?,
                    dependency: if keyframe {
                        SampleDependency::INDEPENDENT
                    } else {
                        SampleDependency::DEPENDENT
                    },
                    is_sync: keyframe,
                });
                frame_pts = frame_pts
                    .checked_add(this_duration as i64)
                    .ok_or_else(|| limit("WebM timestamp overflow"))?;
            }
        }
        let mut duration = 0_u64;
        for sample in &samples {
            duration = duration.max(
                sample
                    .dts
                    .checked_add(u64::from(sample.duration))
                    .ok_or_else(|| limit("track duration overflow"))?,
            );
        }
        ensure_allocation(
            samples.len(),
            std::mem::size_of::<usize>(),
            options,
            "presentation index",
        )?;
        // A decode-only frame is decoded whenever a seek passes through it,
        // but it is not a presentation frame.
        let mut presentation_order: Vec<usize> =
            (0..samples.len()).filter(|&index| shown[index]).collect();
        presentation_order.sort_by_key(|&index| {
            let sample = &samples[index];
            (sample.pts, sample.dts, index)
        });
        tracks.push(Mp4Track {
            id,
            kind: TrackKind::Video,
            codec: track.codec,
            timescale,
            duration,
            dimensions: Some(track.dimensions),
            channels: None,
            sample_rate: None,
            decoder_config: track.decoder_config,
            edits: Vec::new(),
            samples,
            presentation_order,
        });
    }
    Ok(WebmDemuxer {
        doc_type,
        duration_seconds: info
            .duration
            .map(|duration| duration * info.timestamp_scale as f64 / 1e9),
        tracks,
        cues,
        skipped_tracks,
    })
}

fn frame_size(size: u64) -> Result<u32> {
    if size == 0 {
        return Err(malformed("WebM block contains an empty frame"));
    }
    u32::try_from(size).map_err(|_| limit("WebM frame is too large"))
}

fn validate_options(options: &WebmDemuxerOptions) -> Result<()> {
    if options.max_element_bytes == 0 || options.max_elements == 0 {
        return Err(invalid("WebM structure limits must be nonzero"));
    }
    if options.max_samples_per_track == 0 {
        return Err(invalid("the WebM sample limit must be nonzero"));
    }
    Ok(())
}

fn ensure_bytes(length: u64, options: &WebmDemuxerOptions, what: &str) -> Result<()> {
    if length > options.limits.max_allocation_bytes {
        return Err(limit(format!(
            "{what} exceeds the configured allocation limit"
        )));
    }
    Ok(())
}

fn ensure_allocation(
    count: usize,
    element_size: usize,
    options: &WebmDemuxerOptions,
    what: &str,
) -> Result<()> {
    let bytes = count
        .checked_mul(element_size)
        .ok_or_else(|| limit(format!("{what} allocation overflow")))?;
    ensure_bytes(bytes as u64, options, what)
}

async fn read_up_to<S: ByteSource + ?Sized>(
    source: &S,
    offset: u64,
    destination: &mut [u8],
) -> Result<usize> {
    let mut read = 0;
    while read < destination.len() {
        let count = source
            .read_at(offset + read as u64, &mut destination[read..])
            .await?;
        if count == 0 {
            break;
        }
        read += count;
    }
    Ok(read)
}

async fn read_exact<S: ByteSource + ?Sized>(
    source: &S,
    offset: u64,
    destination: &mut [u8],
) -> Result<()> {
    if read_up_to(source, offset, destination).await? != destination.len() {
        return Err(malformed("WebM element is truncated"));
    }
    Ok(())
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}

fn limit(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ebml::{write_element, write_id, write_string, write_uint, write_vint};
    use crate::io::MemorySource;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let mut context = Context::from_waker(Waker::noop());
        let mut future = Box::pin(future);
        loop {
            if let Poll::Ready(value) = Pin::new(&mut future).poll(&mut context) {
                return value;
            }
        }
    }

    const AV1C: [u8; 4] = [0x81, 0, 0, 0];

    fn ebml_header(doc_type: &str) -> Vec<u8> {
        let mut payload = Vec::new();
        write_uint(&mut payload, ebml::EBML_VERSION, 1);
        write_string(&mut payload, ebml::DOC_TYPE, doc_type);
        write_uint(&mut payload, ebml::DOC_TYPE_READ_VERSION, 2);
        let mut output = Vec::new();
        write_element(&mut output, ebml::EBML, &payload);
        output
    }

    fn info(scale: Option<u64>, duration: Option<f64>) -> Vec<u8> {
        let mut payload = Vec::new();
        if let Some(scale) = scale {
            write_uint(&mut payload, ebml::TIMESTAMP_SCALE, scale);
        }
        if let Some(duration) = duration {
            ebml::write_float(&mut payload, ebml::DURATION, duration);
        }
        let mut output = Vec::new();
        write_element(&mut output, ebml::INFO, &payload);
        output
    }

    fn video_entry(number: u64, codec_id: &str, default_duration: Option<u64>) -> Vec<u8> {
        let mut entry = Vec::new();
        write_uint(&mut entry, ebml::TRACK_NUMBER, number);
        write_uint(&mut entry, ebml::TRACK_TYPE, 1);
        write_string(&mut entry, ebml::CODEC_ID, codec_id);
        write_element(&mut entry, ebml::CODEC_PRIVATE, &AV1C);
        if let Some(duration) = default_duration {
            write_uint(&mut entry, ebml::DEFAULT_DURATION, duration);
        }
        let mut video = Vec::new();
        write_uint(&mut video, ebml::PIXEL_WIDTH, 64);
        write_uint(&mut video, ebml::PIXEL_HEIGHT, 48);
        write_element(&mut entry, ebml::VIDEO, &video);
        let mut output = Vec::new();
        write_element(&mut output, ebml::TRACK_ENTRY, &entry);
        output
    }

    fn opus_entry(number: u64) -> Vec<u8> {
        let mut entry = Vec::new();
        write_uint(&mut entry, ebml::TRACK_NUMBER, number);
        write_uint(&mut entry, ebml::TRACK_TYPE, 2);
        write_string(&mut entry, ebml::CODEC_ID, "A_OPUS");
        let mut output = Vec::new();
        write_element(&mut output, ebml::TRACK_ENTRY, &entry);
        output
    }

    fn tracks(entries: &[Vec<u8>]) -> Vec<u8> {
        let mut output = Vec::new();
        write_element(&mut output, ebml::TRACKS, &entries.concat());
        output
    }

    /// A block body: track number, relative timestamp, flags, then `body`.
    fn block_body(track: u64, relative: i16, flags: u8, body: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        write_vint(&mut output, track);
        output.extend_from_slice(&relative.to_be_bytes());
        output.push(flags);
        output.extend_from_slice(body);
        output
    }

    fn simple_block(track: u64, relative: i16, keyframe: bool, data: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let flags = if keyframe { 0x80 } else { 0 };
        write_element(
            &mut output,
            ebml::SIMPLE_BLOCK,
            &block_body(track, relative, flags, data),
        );
        output
    }

    /// A Cluster of known size, or of unknown size as a live recorder writes.
    fn cluster(timestamp: u64, blocks: &[Vec<u8>], known_size: bool) -> Vec<u8> {
        let mut payload = Vec::new();
        write_uint(&mut payload, ebml::TIMESTAMP, timestamp);
        payload.extend_from_slice(&blocks.concat());
        let mut output = Vec::new();
        if known_size {
            write_element(&mut output, ebml::CLUSTER, &payload);
        } else {
            write_id(&mut output, ebml::CLUSTER);
            output.push(0xFF);
            output.extend_from_slice(&payload);
        }
        output
    }

    fn file(doc_type: &str, children: &[Vec<u8>], known_size: bool) -> Vec<u8> {
        let mut output = ebml_header(doc_type);
        let payload = children.concat();
        if known_size {
            write_element(&mut output, ebml::SEGMENT, &payload);
        } else {
            write_id(&mut output, ebml::SEGMENT);
            output.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
            output.extend_from_slice(&payload);
        }
        output
    }

    fn open(bytes: Vec<u8>) -> Result<WebmDemuxer> {
        let source = MemorySource::new(bytes);
        block_on(WebmDemuxer::open(&source, WebmDemuxerOptions::default()))
    }

    fn summary(track: &Mp4Track) -> Vec<(i64, u32, bool)> {
        track
            .samples
            .iter()
            .map(|sample| (sample.pts, sample.duration, sample.is_sync))
            .collect()
    }

    #[test]
    fn unknown_size_segment_and_clusters_without_cues_index_by_scan() {
        // A MediaRecorder-style capture: no Cues, no Duration, every size unknown.
        let bytes = file(
            "webm",
            &[
                info(None, None),
                tracks(&[video_entry(1, "V_AV1", Some(33_000_000)), opus_entry(2)]),
                cluster(
                    0,
                    &[
                        simple_block(1, 0, true, &[1, 1]),
                        simple_block(2, 0, true, &[9]),
                        simple_block(1, 33, false, &[2, 2, 2]),
                    ],
                    false,
                ),
                cluster(
                    66,
                    &[
                        simple_block(1, 0, true, &[3]),
                        simple_block(1, 33, false, &[4]),
                    ],
                    false,
                ),
            ],
            false,
        );
        let demuxer = open(bytes.clone()).unwrap();
        assert_eq!(demuxer.doc_type, "webm");
        assert!(demuxer.cues.is_empty());
        assert_eq!(demuxer.duration_seconds, None);
        assert_eq!(
            demuxer.skipped_tracks,
            vec![WebmSkippedTrack {
                number: 2,
                codec_id: "A_OPUS".to_owned(),
            }]
        );
        assert_eq!(demuxer.tracks.len(), 1);
        let track = &demuxer.tracks[0];
        assert_eq!(track.timescale, 1_000);
        assert_eq!(
            track.dimensions,
            Some(VideoDimensions {
                width: 64,
                height: 48
            })
        );
        assert_eq!(
            track.decoder_config,
            [0, 0, 0, 12, b'a', b'v', b'1', b'C', 0x81, 0, 0, 0]
        );
        // The last frame takes its length from DefaultDuration.
        assert_eq!(
            summary(track),
            [
                (0, 33, true),
                (33, 33, false),
                (66, 33, true),
                (99, 33, false)
            ]
        );
        let payloads: Vec<_> = track
            .samples
            .iter()
            .map(|sample| {
                let offset = sample.offset as usize;
                bytes[offset..offset + sample.size as usize].to_vec()
            })
            .collect();
        assert_eq!(payloads, [vec![1, 1], vec![2, 2, 2], vec![3], vec![4]]);
        // Without Cues, seeking falls back to the scanned sync samples.
        let seek = demuxer.seek_point(1, 98).unwrap();
        assert!(!seek.from_cues);
        assert_eq!(seek.time, 66);
        assert_eq!(seek.offset, track.samples[2].offset);
        assert!(demuxer.seek_point(3, 0).is_none());
    }

    #[test]
    fn lacing_splits_blocks_into_evenly_timed_frames() {
        // Xiph sizes 300 (255 + 45) and 2, then the 4-byte remainder.
        let mut xiph = vec![2, 255, 45, 2];
        xiph.extend(std::iter::repeat_n(7, 306));
        // EBML sizes 5, then 5 + (-2) = 3, then the 1-byte remainder.
        let mut ebml_laced = vec![2, 0x85, 0xBF - 2];
        ebml_laced.extend(std::iter::repeat_n(8, 9));
        // Fixed: three frames of two bytes.
        let mut fixed = vec![2];
        fixed.extend(std::iter::repeat_n(6, 6));
        let laced = |flags: u8, body: &[u8], relative: i16| {
            let mut output = Vec::new();
            write_element(
                &mut output,
                ebml::SIMPLE_BLOCK,
                &block_body(1, relative, 0x80 | flags, body),
            );
            output
        };
        let bytes = file(
            "matroska",
            &[
                info(None, Some(90.0)),
                tracks(&[video_entry(1, "V_AV1", None)]),
                cluster(
                    0,
                    &[
                        laced(0b010, &xiph, 0),
                        laced(0b110, &ebml_laced, 30),
                        laced(0b100, &fixed, 60),
                    ],
                    true,
                ),
            ],
            true,
        );
        let demuxer = open(bytes).unwrap();
        assert_eq!(demuxer.doc_type, "matroska");
        assert_eq!(demuxer.duration_seconds, Some(0.09));
        let track = &demuxer.tracks[0];
        let sizes: Vec<_> = track.samples.iter().map(|sample| sample.size).collect();
        assert_eq!(sizes, [300, 2, 4, 5, 3, 1, 2, 2, 2]);
        // Each block's 30 ms is shared by its three frames; the last block's
        // length comes from the segment Duration.
        let times: Vec<_> = track.samples.iter().map(|sample| sample.pts).collect();
        assert_eq!(times, [0, 10, 20, 30, 40, 50, 60, 70, 80]);
        assert!(track.samples.iter().all(|sample| sample.duration == 10));
        let mut offset = track.samples[0].offset;
        for sample in &track.samples[..3] {
            assert_eq!(sample.offset, offset);
            offset += u64::from(sample.size);
        }
    }

    #[test]
    fn malformed_lacing_is_rejected() {
        for (flags, body) in [
            // Xiph sizes larger than the block.
            (0b010_u8, vec![1, 200, 1]),
            // Fixed lacing that does not divide the data.
            (0b100, vec![1, 1, 2, 3]),
            // EBML lacing whose difference makes a size negative.
            (0b110, vec![2, 0x81, 0x80, 1, 1, 1]),
        ] {
            let mut block = Vec::new();
            write_element(
                &mut block,
                ebml::SIMPLE_BLOCK,
                &block_body(1, 0, 0x80 | flags, &body),
            );
            let bytes = file(
                "webm",
                &[
                    info(None, None),
                    tracks(&[video_entry(1, "V_AV1", None)]),
                    cluster(0, &[block], true),
                ],
                true,
            );
            assert_eq!(open(bytes).unwrap_err().kind(), ErrorKind::MalformedMedia);
        }
    }

    #[test]
    fn hidden_vp8_frames_are_decoded_but_not_presented() {
        // Issue #537. A VP8 frame tag's bit 4 is `show_frame`, and bit 0 is
        // clear on a key frame. The hidden frame one tick after frame 40 is
        // how vpxenc stores an alternate reference, and the block at 120 is
        // hidden by its Matroska invisible flag instead. The AV1 track shows
        // the same first bytes are only read as a VP8 frame tag.
        let mut invisible = Vec::new();
        write_element(
            &mut invisible,
            ebml::SIMPLE_BLOCK,
            &block_body(1, 120, INVISIBLE, &[0x11, 0]),
        );
        let bytes = file(
            "webm",
            &[
                info(None, None),
                tracks(&[video_entry(1, "V_VP8", None), video_entry(2, "V_AV1", None)]),
                cluster(
                    0,
                    &[
                        simple_block(1, 0, true, &[0x10, 0]),
                        simple_block(2, 0, true, &[0x00, 0]),
                        simple_block(1, 40, false, &[0x11, 0]),
                        simple_block(1, 41, false, &[0x01, 0]),
                        simple_block(2, 40, false, &[0x01, 0]),
                        simple_block(1, 80, false, &[0x11, 0]),
                        invisible,
                        simple_block(1, 160, false, &[0x11, 0]),
                    ],
                    true,
                ),
            ],
            true,
        );
        let source = MemorySource::new(bytes.clone());
        let demuxer = open(bytes).unwrap();
        let vp8 = &demuxer.tracks[0];
        assert_eq!(vp8.samples.len(), 6);
        assert_eq!(vp8.presentation_order, vec![0, 1, 3, 5]);
        // Frame 40 lasts until the next shown frame, not the hidden one.
        assert_eq!(vp8.presentation_sample(1).unwrap().duration, 40);
        let samples = block_on(vp8.to_encoded_video_samples(&source, &Limits::default())).unwrap();
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.presentation_index.0)
                .collect::<Vec<_>>(),
            vec![0, 1, 4, 2, 5, 3]
        );
        assert_eq!(demuxer.tracks[1].presentation_order, vec![0, 1]);
    }

    #[test]
    fn block_groups_are_keyframes_unless_they_reference_a_block() {
        let group = |relative: i16, reference: bool, duration: Option<u64>| {
            let mut payload = Vec::new();
            write_element(
                &mut payload,
                ebml::BLOCK,
                &block_body(1, relative, 0, &[5, 5]),
            );
            if let Some(duration) = duration {
                write_uint(&mut payload, ebml::BLOCK_DURATION, duration);
            }
            if reference {
                write_element(&mut payload, ebml::REFERENCE_BLOCK, &[0xE0]);
            }
            let mut output = Vec::new();
            write_element(&mut output, ebml::BLOCK_GROUP, &payload);
            output
        };
        // A TimestampScale that does not divide a second counts in nanoseconds.
        let bytes = file(
            "webm",
            &[
                info(Some(3_000_000), None),
                tracks(&[video_entry(1, "V_AV1", None)]),
                cluster(10, &[group(0, false, None), group(2, true, Some(4))], true),
            ],
            true,
        );
        let demuxer = open(bytes).unwrap();
        let track = &demuxer.tracks[0];
        assert_eq!(track.timescale, 1_000_000_000);
        assert_eq!(
            summary(track),
            [
                (30_000_000, 6_000_000, true),
                (36_000_000, 12_000_000, false)
            ]
        );
        assert_eq!(track.samples[1].dependency, SampleDependency::DEPENDENT);
    }

    #[test]
    fn cues_mark_random_access_and_answer_seeks() {
        let first = cluster(
            0,
            &[
                simple_block(1, 0, true, &[1]),
                simple_block(1, 40, false, &[2]),
            ],
            true,
        );
        let second_start = info(None, Some(120.0)).len()
            + tracks(&[video_entry(1, "V_AV1", None)]).len()
            + first.len();
        // The second Cluster's keyframe flag is missing; its cue restores it.
        let second = cluster(80, &[simple_block(1, 0, false, &[3])], true);
        let mut cue_positions = Vec::new();
        write_uint(&mut cue_positions, ebml::CUE_TRACK, 1);
        write_uint(
            &mut cue_positions,
            ebml::CUE_CLUSTER_POSITION,
            second_start as u64,
        );
        let mut point = Vec::new();
        write_uint(&mut point, ebml::CUE_TIME, 80);
        write_element(&mut point, ebml::CUE_TRACK_POSITIONS, &cue_positions);
        let mut points = Vec::new();
        write_element(&mut points, ebml::CUE_POINT, &point);
        let mut cues = Vec::new();
        write_element(&mut cues, ebml::CUES, &points);
        let bytes = file(
            "webm",
            &[
                info(None, Some(120.0)),
                tracks(&[video_entry(1, "V_AV1", None)]),
                first,
                second,
                cues,
            ],
            true,
        );
        let demuxer = open(bytes.clone()).unwrap();
        let track = &demuxer.tracks[0];
        assert_eq!(
            summary(track),
            [(0, 40, true), (40, 40, false), (80, 40, true)]
        );
        assert_eq!(demuxer.cues.len(), 1);
        let cue = demuxer.cues[0];
        assert_eq!(
            &bytes[cue.cluster_offset as usize..][..4],
            &[0x1F, 0x43, 0xB6, 0x75]
        );
        let seek = demuxer.seek_point(1, 100).unwrap();
        assert!(seek.from_cues);
        assert_eq!((seek.time, seek.offset), (80, cue.cluster_offset));
        // Before the only cue there is nowhere cued to start.
        assert!(demuxer.seek_point(1, 50).is_none());
    }

    #[test]
    fn structure_errors_are_reported_with_stable_kinds() {
        let blocks = || cluster(0, &[simple_block(1, 0, true, &[1])], true);
        let av1 = || tracks(&[video_entry(1, "V_AV1", None)]);
        let cases = [
            // No Tracks before the Cluster.
            (
                file("webm", &[info(None, None), blocks()], true),
                ErrorKind::MalformedMedia,
            ),
            // No Info at all.
            (
                file("webm", &[av1(), blocks()], true),
                ErrorKind::MalformedMedia,
            ),
            (
                file(
                    "webm",
                    &[
                        info(None, None),
                        tracks(&[video_entry(1, "V_THEORA", None)]),
                        blocks(),
                    ],
                    true,
                ),
                ErrorKind::Unsupported,
            ),
            // A block for a track the file never declared.
            (
                file(
                    "webm",
                    &[
                        info(None, None),
                        av1(),
                        cluster(0, &[simple_block(3, 0, true, &[1])], true),
                    ],
                    true,
                ),
                ErrorKind::MalformedMedia,
            ),
            (file("mkv3d", &[], true), ErrorKind::Unsupported),
        ];
        for (bytes, kind) in cases {
            assert_eq!(open(bytes).unwrap_err().kind(), kind);
        }
        // A Segment that claims more bytes than the file holds.
        let mut truncated = file("webm", &[info(None, None), av1(), blocks()], true);
        truncated.truncate(truncated.len() - 1);
        assert_eq!(
            open(truncated).unwrap_err().kind(),
            ErrorKind::MalformedMedia
        );
    }

    #[test]
    fn element_and_sample_limits_apply() {
        let bytes = file(
            "webm",
            &[
                info(None, None),
                tracks(&[video_entry(1, "V_AV1", None)]),
                cluster(
                    0,
                    &[
                        simple_block(1, 0, true, &[1]),
                        simple_block(1, 1, false, &[2]),
                    ],
                    true,
                ),
            ],
            true,
        );
        let source = MemorySource::new(bytes);
        let limited = |options: WebmDemuxerOptions| {
            block_on(WebmDemuxer::open(&source, options))
                .unwrap_err()
                .kind()
        };
        for options in [
            WebmDemuxerOptions {
                max_samples_per_track: 1,
                ..WebmDemuxerOptions::default()
            },
            WebmDemuxerOptions {
                max_elements: 4,
                ..WebmDemuxerOptions::default()
            },
            WebmDemuxerOptions {
                max_element_bytes: 8,
                ..WebmDemuxerOptions::default()
            },
        ] {
            assert_eq!(limited(options), ErrorKind::ResourceLimit);
        }
    }

    #[test]
    fn probing_reads_the_doc_type_and_nothing_else() {
        let webm = MemorySource::new(file("webm", &[], true));
        let matroska = MemorySource::new(file("matroska", &[], true));
        let other = MemorySource::new(file("other", &[], true));
        let mp4 = MemorySource::new(b"\0\0\0\x18ftypisom".to_vec());
        assert!(block_on(probe_webm(&webm)).unwrap());
        assert!(block_on(probe_webm(&matroska)).unwrap());
        assert!(!block_on(probe_webm(&other)).unwrap());
        assert!(!block_on(probe_webm(&mp4)).unwrap());
        assert!(!block_on(probe_webm(&MemorySource::new(vec![0x1A]))).unwrap());
    }
}
