//! Bounded, read-only ISO BMFF/MP4 probing and sample indexing.

use crate::audio::{AudioEdit, AudioTrackTiming, EncodedAudioSample};
use crate::codec::{EncodedVideoSample, SampleDependency, TrackKind};
use crate::io::ByteSource;
use crate::media::{Codec, VideoDimensions};
use crate::timeline::FrameIndex;
use crate::{Error, ErrorKind, Limits, Result};
use std::collections::{BTreeMap, BTreeSet};

const HEADER_SIZE: usize = 8;

/// Resource limits specific to untrusted MP4 structure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mp4DemuxerOptions {
    pub limits: Limits,
    pub max_box_bytes: u64,
    pub max_boxes: u32,
    pub max_box_depth: u8,
    pub max_samples_per_track: u32,
}

impl Default for Mp4DemuxerOptions {
    fn default() -> Self {
        Self {
            limits: Limits::default(),
            max_box_bytes: 64 * 1024 * 1024,
            max_boxes: 100_000,
            max_box_depth: 16,
            max_samples_per_track: 10_000_000,
        }
    }
}

/// A movie-to-media timeline edit. A media time of `-1` is an empty edit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EditMapping {
    pub segment_duration: u64,
    pub media_time: i64,
    pub media_rate_integer: i16,
    pub media_rate_fraction: i16,
}

/// One encoded sample in decode order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mp4Sample {
    pub offset: u64,
    pub size: u32,
    pub dts: u64,
    pub pts: i64,
    pub duration: u32,
    pub dependency: SampleDependency,
    pub is_sync: bool,
}

/// Read-only metadata and indexes for one MP4 track.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mp4Track {
    pub id: u32,
    pub kind: TrackKind,
    pub codec: Codec,
    pub timescale: u32,
    pub duration: u64,
    pub dimensions: Option<VideoDimensions>,
    pub channels: Option<u16>,
    pub sample_rate: Option<u32>,
    /// Complete codec configuration box, including its header.
    pub decoder_config: Vec<u8>,
    pub edits: Vec<EditMapping>,
    /// Samples in decode order.
    pub samples: Vec<Mp4Sample>,
    /// Decode-order indexes sorted by PTS, then DTS, then decode index.
    pub presentation_order: Vec<usize>,
}

impl Mp4Track {
    /// Returns a sample by its zero-based presentation index.
    pub fn presentation_sample(&self, index: usize) -> Option<&Mp4Sample> {
        self.presentation_order
            .get(index)
            .and_then(|&decode_index| self.samples.get(decode_index))
    }

    /// Reads one decode-order sample without changing caller-visible source state.
    pub async fn read_sample_into<S: ByteSource + ?Sized>(
        &self,
        source: &S,
        decode_index: usize,
        destination: &mut [u8],
    ) -> Result<()> {
        let sample = self
            .samples
            .get(decode_index)
            .ok_or_else(|| invalid("MP4 sample index is out of range"))?;
        if destination.len() != sample.size as usize {
            return Err(invalid("sample destination size does not match the index"));
        }
        read_exact(source, sample.offset, destination).await
    }

    /// Reads every decode-order sample and returns owned
    /// [`EncodedVideoSample`] values keyed by presentation order, ready for
    /// a [`crate::codec::VideoDecoderFactory`] backend.
    ///
    /// The zero-based position of each sample within
    /// [`Self::presentation_order`] becomes its `presentation_index`, and
    /// `is_sync` becomes `random_access`. Total decoded sample bytes are
    /// bounded by `limits.max_allocation_bytes`.
    pub async fn to_encoded_video_samples<S: ByteSource + ?Sized>(
        &self,
        source: &S,
        limits: &Limits,
    ) -> Result<Vec<EncodedVideoSample>> {
        if self.kind != TrackKind::Video {
            return Err(unsupported("encoded video samples require a video track"));
        }
        let mut presentation_index_by_decode = vec![0_u64; self.samples.len()];
        for (presentation_index, &decode_index) in self.presentation_order.iter().enumerate() {
            let presentation_index = u64::try_from(presentation_index)
                .map_err(|_| limit("presentation index overflow"))?;
            presentation_index_by_decode[decode_index] = presentation_index;
        }

        let mut total_bytes = 0_u64;
        let mut samples = Vec::with_capacity(self.samples.len());
        for (decode_index, sample) in self.samples.iter().enumerate() {
            total_bytes = total_bytes
                .checked_add(u64::from(sample.size))
                .ok_or_else(|| limit("encoded sample allocation overflow"))?;
            if total_bytes > limits.max_allocation_bytes {
                return Err(limit(
                    "encoded video samples exceed the configured allocation limit",
                ));
            }
            let mut data = vec![0_u8; sample.size as usize];
            self.read_sample_into(source, decode_index, &mut data)
                .await?;
            samples.push(EncodedVideoSample {
                presentation_index: FrameIndex(presentation_index_by_decode[decode_index]),
                random_access: sample.is_sync,
                data,
            });
        }
        Ok(samples)
    }

    /// Parses the AAC AudioSpecificConfig carried by this track's `esds` box.
    pub fn aac_config(&self) -> Result<AacTrackConfig> {
        if self.kind != TrackKind::Audio || self.codec != Codec::Aac {
            return Err(unsupported("AAC configuration requires an AAC audio track"));
        }
        let sample_rate = self
            .sample_rate
            .ok_or_else(|| malformed("AAC sample rate is missing"))?;
        let channels = self
            .channels
            .ok_or_else(|| malformed("AAC channel count is missing"))?;
        parse_aac_config(&self.decoder_config, sample_rate, channels)
    }

    /// Reads every indexed AAC access unit from its validated byte range.
    ///
    /// Packet intervals use the decoded PCM sample clock and remain contiguous
    /// even when the MP4 track timescale differs from the AAC sample rate.
    pub async fn to_encoded_audio_samples<S: ByteSource + ?Sized>(
        &self,
        source: &S,
        limits: &Limits,
    ) -> Result<Vec<EncodedAudioSample>> {
        let config = self.aac_config()?;
        let mut total_bytes = 0_u64;
        let mut decoded_start = 0_u64;
        let mut track_ticks = 0_u64;
        let mut packets = Vec::with_capacity(self.samples.len());
        for (decode_index, sample) in self.samples.iter().enumerate() {
            total_bytes = total_bytes
                .checked_add(u64::from(sample.size))
                .ok_or_else(|| limit("encoded AAC allocation overflow"))?;
            if total_bytes > limits.max_allocation_bytes {
                return Err(limit(
                    "encoded AAC samples exceed the configured allocation limit",
                ));
            }
            track_ticks = track_ticks
                .checked_add(u64::from(sample.duration))
                .ok_or_else(|| limit("AAC track timing overflow"))?;
            let decoded_end = scale_time(track_ticks, self.timescale, config.sample_rate)?;
            if decoded_end <= decoded_start {
                return Err(malformed("AAC packet has an empty decoded interval"));
            }
            let mut data = vec![0_u8; sample.size as usize];
            self.read_sample_into(source, decode_index, &mut data)
                .await?;
            packets.push(EncodedAudioSample {
                decoded_range: crate::SampleRange::new(decoded_start, decoded_end)?,
                data,
            });
            decoded_start = decoded_end;
        }
        if packets.is_empty() {
            return Err(malformed("AAC track contains no samples"));
        }
        Ok(packets)
    }

    /// Converts MP4 edit-list timing to the presentation sample clock used by
    /// [`crate::AacSampleReader`], including decoder priming and end padding.
    pub fn audio_timing(&self, movie_timescale: u32) -> Result<AudioTrackTiming> {
        let config = self.aac_config()?;
        let decoded_length = scale_time(self.duration, self.timescale, config.sample_rate)?;
        if self.edits.is_empty() {
            return Ok(AudioTrackTiming::default());
        }
        let mut presentation_start = 0_u64;
        let mut edits = Vec::with_capacity(self.edits.len());
        let mut first_media = None;
        let mut last_media_end = 0_u64;
        for edit in &self.edits {
            if edit.media_rate_integer != 1 || edit.media_rate_fraction != 0 {
                return Err(unsupported("AAC edit rates other than 1.0 are unsupported"));
            }
            let length = scale_time(edit.segment_duration, movie_timescale, config.sample_rate)?;
            if length == 0 {
                return Err(malformed("AAC edit has an empty presentation interval"));
            }
            let presentation_end = presentation_start
                .checked_add(length)
                .ok_or_else(|| limit("AAC edit presentation overflow"))?;
            let media_start = if edit.media_time == -1 {
                None
            } else {
                if edit.media_time < 0 {
                    return Err(malformed("AAC edit has an invalid negative media time"));
                }
                let start = scale_time(edit.media_time as u64, self.timescale, config.sample_rate)?;
                let end = start
                    .checked_add(length)
                    .ok_or_else(|| limit("AAC edit media range overflow"))?;
                first_media = Some(first_media.map_or(start, |value: u64| value.min(start)));
                last_media_end = last_media_end.max(end);
                Some(start)
            };
            edits.push(AudioEdit {
                presentation: crate::SampleRange::new(presentation_start, presentation_end)?,
                media_start,
            });
            presentation_start = presentation_end;
        }
        let priming = u32::try_from(first_media.unwrap_or(0))
            .map_err(|_| limit("AAC priming exceeds the supported range"))?;
        let padding = u32::try_from(decoded_length.saturating_sub(last_media_end))
            .map_err(|_| limit("AAC padding exceeds the supported range"))?;
        Ok(AudioTrackTiming {
            priming,
            padding,
            track_offset: 0,
            edits,
        })
    }
}

/// Decoder configuration extracted from an AAC `mp4a` / `esds` description.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AacTrackConfig {
    pub audio_object_type: u8,
    pub sample_rate: u32,
    pub channels: u16,
    pub audio_specific_config: Vec<u8>,
}

/// Parsed MP4 movie metadata and indexes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mp4Demuxer {
    pub movie_timescale: u32,
    pub tracks: Vec<Mp4Track>,
}

/// Bounded format detection. At most 32 bytes are requested.
pub async fn probe_mp4<S: ByteSource + ?Sized>(source: &S) -> Result<bool> {
    if source
        .len()
        .is_some_and(|length| length < HEADER_SIZE as u64)
    {
        return Ok(false);
    }
    let mut bytes = [0_u8; HEADER_SIZE];
    let mut read = 0;
    while read < bytes.len() {
        let count = source.read_at(read as u64, &mut bytes[read..]).await?;
        if count == 0 {
            return Ok(false);
        }
        read += count;
    }
    let size = u32::from_be_bytes(bytes[..4].try_into().expect("fixed slice"));
    if size != 0 && size != 1 && size < 8 {
        return Ok(false);
    }
    Ok(&bytes[4..8] == b"ftyp" || &bytes[4..8] == b"moov")
}

impl Mp4Demuxer {
    /// Incrementally scans top-level boxes and reads only bounded metadata boxes.
    pub async fn open<S: ByteSource + ?Sized>(
        source: &S,
        options: Mp4DemuxerOptions,
    ) -> Result<Self> {
        validate_options(&options)?;
        let source_len = source
            .len()
            .ok_or_else(|| unsupported("MP4 indexing requires a source length"))?;
        let mut cursor = 0_u64;
        let mut budget = Budget::new(&options);
        let mut movie = None;
        let mut fragments = Vec::new();
        let mut media_ranges = Vec::new();
        while cursor < source_len {
            let header = read_header(source, cursor, source_len, &mut budget).await?;
            match &header.kind {
                b"moov" => {
                    if movie.is_some() {
                        return Err(malformed("multiple movie boxes are not supported"));
                    }
                    let bytes = read_payload(source, header, &options).await?;
                    movie = Some(parse_moov(&bytes, &mut budget, &options)?);
                }
                b"moof" => {
                    let bytes = read_payload(source, header, &options).await?;
                    ensure_allocation(
                        fragments.len() + 1,
                        std::mem::size_of::<Fragment>(),
                        &options,
                        "movie fragments",
                    )?;
                    fragments.push(parse_moof(&bytes, header.start, &mut budget, &options)?);
                }
                b"mdat" => {
                    ensure_allocation(
                        media_ranges.len() + 1,
                        std::mem::size_of::<(u64, u64)>(),
                        &options,
                        "media data ranges",
                    )?;
                    media_ranges.push((header.payload_start, header.end));
                }
                _ => {}
            }
            cursor = header.end;
        }
        let ParsedMovie {
            mut demuxer,
            fragment_defaults,
        } = movie.ok_or_else(|| malformed("MP4 does not contain a movie box"))?;
        append_fragments(&mut demuxer, fragments, &fragment_defaults, &options)?;
        for track in &mut demuxer.tracks {
            track
                .samples
                .sort_by_key(|sample| (sample.dts, sample.offset));
            validate_samples(track, source_len, &media_ranges)?;
            for sample in &track.samples {
                track.duration = track.duration.max(
                    sample
                        .dts
                        .checked_add(u64::from(sample.duration))
                        .ok_or_else(|| limit("track duration overflow"))?,
                );
            }
            ensure_allocation(
                track.samples.len(),
                std::mem::size_of::<usize>(),
                &options,
                "presentation index",
            )?;
            track.presentation_order = (0..track.samples.len()).collect();
            track.presentation_order.sort_by_key(|&index| {
                let sample = &track.samples[index];
                (sample.pts, sample.dts, index)
            });
        }
        Ok(demuxer)
    }

    pub fn track(&self, id: u32) -> Option<&Mp4Track> {
        self.tracks.iter().find(|track| track.id == id)
    }
}

#[derive(Clone, Copy)]
struct Header {
    start: u64,
    payload_start: u64,
    end: u64,
    kind: [u8; 4],
}

struct Budget {
    boxes: u32,
    max_boxes: u32,
}

impl Budget {
    fn new(options: &Mp4DemuxerOptions) -> Self {
        Self {
            boxes: 0,
            max_boxes: options.max_boxes,
        }
    }

    fn box_seen(&mut self) -> Result<()> {
        self.boxes = self
            .boxes
            .checked_add(1)
            .ok_or_else(|| limit("MP4 box count overflow"))?;
        if self.boxes > self.max_boxes {
            return Err(limit("MP4 box count limit exceeded"));
        }
        Ok(())
    }
}

async fn read_header<S: ByteSource + ?Sized>(
    source: &S,
    start: u64,
    parent_end: u64,
    budget: &mut Budget,
) -> Result<Header> {
    budget.box_seen()?;
    let remaining = parent_end
        .checked_sub(start)
        .ok_or_else(|| malformed("box offset exceeds parent"))?;
    if remaining < 8 {
        return Err(malformed("truncated MP4 box header"));
    }
    let mut basic = [0_u8; 16];
    read_exact(source, start, &mut basic[..8]).await?;
    let size32 = u32::from_be_bytes(basic[..4].try_into().expect("fixed slice"));
    let kind = basic[4..8].try_into().expect("fixed slice");
    let (size, header_size) = match size32 {
        0 => (remaining, 8_u64),
        1 => {
            if remaining < 16 {
                return Err(malformed("truncated extended MP4 box header"));
            }
            read_exact(source, start + 8, &mut basic[8..16]).await?;
            (
                u64::from_be_bytes(basic[8..16].try_into().expect("fixed slice")),
                16,
            )
        }
        value => (u64::from(value), 8),
    };
    if size < header_size {
        return Err(malformed("MP4 box is smaller than its header"));
    }
    let end = start
        .checked_add(size)
        .ok_or_else(|| malformed("MP4 box end overflow"))?;
    if end > parent_end {
        return Err(malformed("MP4 box exceeds its parent or source"));
    }
    Ok(Header {
        start,
        payload_start: start + header_size,
        end,
        kind,
    })
}

async fn read_exact<S: ByteSource + ?Sized>(
    source: &S,
    mut offset: u64,
    mut output: &mut [u8],
) -> Result<()> {
    while !output.is_empty() {
        let read = source.read_at(offset, output).await?;
        if read == 0 {
            return Err(malformed("unexpected end of MP4 source"));
        }
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| malformed("read offset overflow"))?;
        output = &mut output[read..];
    }
    Ok(())
}

async fn read_payload<S: ByteSource + ?Sized>(
    source: &S,
    header: Header,
    options: &Mp4DemuxerOptions,
) -> Result<Vec<u8>> {
    let length = header.end - header.payload_start;
    if length > options.max_box_bytes || length > options.limits.max_allocation_bytes {
        return Err(limit("MP4 metadata box allocation limit exceeded"));
    }
    let mut bytes = vec![
        0;
        usize::try_from(length)
            .map_err(|_| limit("MP4 box is too large for this platform"))?
    ];
    read_exact(source, header.payload_start, &mut bytes).await?;
    Ok(bytes)
}

#[derive(Default)]
struct TrackBuilder {
    id: Option<u32>,
    kind: Option<TrackKind>,
    codec: Option<Codec>,
    timescale: Option<u32>,
    duration: u64,
    dimensions: Option<VideoDimensions>,
    channels: Option<u16>,
    sample_rate: Option<u32>,
    decoder_config: Vec<u8>,
    edits: Vec<EditMapping>,
    stts: Vec<(u32, u32)>,
    ctts: Vec<(u32, i64)>,
    stsc: Vec<(u32, u32, u32)>,
    sizes: Vec<u32>,
    chunks: Vec<u64>,
    sync: Option<BTreeSet<u32>>,
    dependencies: Vec<SampleDependency>,
    samples: Vec<Mp4Sample>,
    ignored: bool,
}

#[derive(Default)]
struct MovieBuilder {
    timescale: Option<u32>,
    tracks: Vec<TrackBuilder>,
    trex: BTreeMap<u32, Trex>,
}

struct ParsedMovie {
    demuxer: Mp4Demuxer,
    fragment_defaults: BTreeMap<u32, Trex>,
}

fn parse_moov(
    bytes: &[u8],
    budget: &mut Budget,
    options: &Mp4DemuxerOptions,
) -> Result<ParsedMovie> {
    let mut builder = MovieBuilder::default();
    for child in children(bytes, 1, budget, options)? {
        match &child.kind {
            b"mvhd" => builder.timescale = Some(parse_mvhd(child.payload)?),
            b"trak" => builder
                .tracks
                .push(parse_trak(child.payload, budget, options)?),
            b"mvex" => parse_mvex(child.payload, &mut builder.trex, budget, options)?,
            _ => {}
        }
    }
    if builder.tracks.len() > usize::from(options.limits.max_tracks) {
        return Err(limit("MP4 track count limit exceeded"));
    }
    let movie_timescale = builder
        .timescale
        .ok_or_else(|| malformed("movie header is missing"))?;
    if movie_timescale == 0 {
        return Err(malformed("movie timescale is zero"));
    }
    let mut ids = BTreeSet::new();
    let mut tracks = Vec::with_capacity(builder.tracks.len());
    for track in builder.tracks {
        let id = track
            .id
            .ok_or_else(|| malformed("track header is missing"))?;
        if id == 0 || !ids.insert(id) {
            return Err(malformed("MP4 track IDs must be unique and nonzero"));
        }
        if track.ignored {
            continue;
        }
        tracks.push(finalize_track(track, options)?);
    }
    Ok(ParsedMovie {
        demuxer: Mp4Demuxer {
            movie_timescale,
            tracks,
        },
        fragment_defaults: builder.trex,
    })
}

fn parse_mvhd(payload: &[u8]) -> Result<u32> {
    let (version, body) = full(payload)?;
    let offset = match version {
        0 => 8,
        1 => 16,
        _ => return Err(malformed("unsupported mvhd version")),
    };
    be_u32(body, offset)
}

fn parse_trak(
    bytes: &[u8],
    budget: &mut Budget,
    options: &Mp4DemuxerOptions,
) -> Result<TrackBuilder> {
    let mut track = TrackBuilder::default();
    for child in children(bytes, 2, budget, options)? {
        match &child.kind {
            b"tkhd" => parse_tkhd(child.payload, &mut track, options)?,
            b"edts" => parse_edts(child.payload, &mut track, budget, options)?,
            b"mdia" => parse_mdia(child.payload, &mut track, budget, options)?,
            _ => {}
        }
    }
    Ok(track)
}

fn parse_tkhd(payload: &[u8], track: &mut TrackBuilder, options: &Mp4DemuxerOptions) -> Result<()> {
    let (version, body) = full(payload)?;
    let (id_offset, width_offset) = match version {
        0 => (8, 72),
        1 => (16, 84),
        _ => return Err(malformed("unsupported tkhd version")),
    };
    track.id = Some(be_u32(body, id_offset)?);
    let width = be_u32(body, width_offset)? >> 16;
    let height = be_u32(body, width_offset + 4)? >> 16;
    if width != 0 || height != 0 {
        track.dimensions = Some(VideoDimensions::new(width, height, &options.limits)?);
    }
    Ok(())
}

fn parse_edts(
    bytes: &[u8],
    track: &mut TrackBuilder,
    budget: &mut Budget,
    options: &Mp4DemuxerOptions,
) -> Result<()> {
    for child in children(bytes, 3, budget, options)? {
        if &child.kind != b"elst" {
            continue;
        }
        let (version, body) = full(child.payload)?;
        let count = count(body, 0, options.max_samples_per_track, "edit")?;
        ensure_allocation(
            count,
            std::mem::size_of::<EditMapping>(),
            options,
            "edit list",
        )?;
        let entry_size = if version == 1 {
            20
        } else if version == 0 {
            12
        } else {
            return Err(malformed("unsupported elst version"));
        };
        require(
            body,
            4,
            count
                .checked_mul(entry_size)
                .ok_or_else(|| malformed("edit table overflow"))?,
        )?;
        for index in 0..count {
            let at = 4 + index * entry_size;
            let (segment_duration, media_time, rate_at) = if version == 1 {
                (be_u64(body, at)?, be_i64(body, at + 8)?, at + 16)
            } else {
                (
                    u64::from(be_u32(body, at)?),
                    i64::from(be_i32(body, at + 4)?),
                    at + 8,
                )
            };
            track.edits.push(EditMapping {
                segment_duration,
                media_time,
                media_rate_integer: be_i16(body, rate_at)?,
                media_rate_fraction: be_i16(body, rate_at + 2)?,
            });
        }
    }
    Ok(())
}

fn parse_mdia(
    bytes: &[u8],
    track: &mut TrackBuilder,
    budget: &mut Budget,
    options: &Mp4DemuxerOptions,
) -> Result<()> {
    for child in children(bytes, 3, budget, options)? {
        match &child.kind {
            b"mdhd" => parse_mdhd(child.payload, track)?,
            b"hdlr" => parse_hdlr(child.payload, track)?,
            b"minf" => parse_minf(child.payload, track, budget, options)?,
            _ => {}
        }
    }
    Ok(())
}

fn parse_mdhd(payload: &[u8], track: &mut TrackBuilder) -> Result<()> {
    let (version, body) = full(payload)?;
    let (timescale_at, duration_at) = match version {
        0 => (8, 12),
        1 => (16, 20),
        _ => return Err(malformed("unsupported mdhd version")),
    };
    let timescale = be_u32(body, timescale_at)?;
    if timescale == 0 {
        return Err(malformed("track timescale is zero"));
    }
    track.timescale = Some(timescale);
    track.duration = if version == 0 {
        u64::from(be_u32(body, duration_at)?)
    } else {
        be_u64(body, duration_at)?
    };
    Ok(())
}

fn parse_hdlr(payload: &[u8], track: &mut TrackBuilder) -> Result<()> {
    let (_, body) = full(payload)?;
    track.kind = match slice(body, 4, 4)? {
        b"vide" => Some(TrackKind::Video),
        b"soun" => Some(TrackKind::Audio),
        _ => {
            track.ignored = true;
            None
        }
    };
    Ok(())
}

fn parse_minf(
    bytes: &[u8],
    track: &mut TrackBuilder,
    budget: &mut Budget,
    options: &Mp4DemuxerOptions,
) -> Result<()> {
    if track.ignored {
        return Ok(());
    }
    for child in children(bytes, 4, budget, options)? {
        if &child.kind == b"stbl" {
            parse_stbl(child.payload, track, budget, options)?;
        }
    }
    Ok(())
}

fn parse_stbl(
    bytes: &[u8],
    track: &mut TrackBuilder,
    budget: &mut Budget,
    options: &Mp4DemuxerOptions,
) -> Result<()> {
    for child in children(bytes, 5, budget, options)? {
        match &child.kind {
            b"stsd" => parse_stsd(child.payload, track, budget, options)?,
            b"stts" => track.stts = parse_u32_pairs(child.payload, options, "stts")?,
            b"ctts" => track.ctts = parse_ctts(child.payload, options)?,
            b"stsc" => track.stsc = parse_u32_triples(child.payload, options)?,
            b"stsz" => track.sizes = parse_stsz(child.payload, options)?,
            b"stz2" => track.sizes = parse_stz2(child.payload, options)?,
            b"stco" => track.chunks = parse_offsets(child.payload, options, false)?,
            b"co64" => track.chunks = parse_offsets(child.payload, options, true)?,
            b"stss" => track.sync = Some(parse_sync(child.payload, options)?),
            b"sdtp" => track.dependencies = parse_sdtp(child.payload, options)?,
            _ => {}
        }
    }
    Ok(())
}

fn parse_stsd(
    payload: &[u8],
    track: &mut TrackBuilder,
    budget: &mut Budget,
    options: &Mp4DemuxerOptions,
) -> Result<()> {
    if track.ignored {
        return Ok(());
    }
    let (_, body) = full(payload)?;
    let entries = count(body, 0, 16, "sample description")?;
    if entries != 1 {
        return Err(unsupported(
            "exactly one MP4 sample description is supported",
        ));
    }
    let entry = one_child(&body[4..], 6, budget, options)?;
    let (codec, prefix, kind) = match &entry.kind {
        b"hvc1" | b"hev1" => (Codec::Hevc, 78, TrackKind::Video),
        b"av01" => (Codec::Av1, 78, TrackKind::Video),
        b"mp4a" => (Codec::Aac, 28, TrackKind::Audio),
        _ => return Err(unsupported("unsupported MP4 sample entry")),
    };
    if let Some(handler) = track.kind
        && handler != kind
    {
        return Err(malformed("sample entry and track handler disagree"));
    }
    if kind == TrackKind::Video {
        let width = u32::from(be_u16(entry.payload, 24)?);
        let height = u32::from(be_u16(entry.payload, 26)?);
        track.dimensions = Some(VideoDimensions::new(width, height, &options.limits)?);
    } else {
        track.channels = Some(be_u16(entry.payload, 16)?);
        let fixed_rate = be_u32(entry.payload, 24)?;
        if fixed_rate & 0xffff != 0 {
            return Err(unsupported("fractional AAC sample rates are unsupported"));
        }
        track.sample_rate = Some(fixed_rate >> 16);
    }
    let config_kind = match codec {
        Codec::Hevc => b"hvcC",
        Codec::Av1 => b"av1C",
        Codec::Aac => b"esds",
        Codec::UncompressedVideo => {
            return Err(unsupported(
                "uncompressed MP4 sample entries are not supported",
            ));
        }
    };
    let nested = children(&entry.payload[prefix..], 7, budget, options)?;
    let config = nested
        .into_iter()
        .find(|child| &child.kind == config_kind)
        .ok_or_else(|| malformed("codec configuration box is missing"))?;
    track.decoder_config = config.whole.to_vec();
    track.codec = Some(codec);
    Ok(())
}

fn parse_u32_pairs(
    payload: &[u8],
    options: &Mp4DemuxerOptions,
    label: &str,
) -> Result<Vec<(u32, u32)>> {
    let (_, body) = full(payload)?;
    let count = count(body, 0, options.max_samples_per_track, label)?;
    ensure_allocation(count, std::mem::size_of::<(u32, u32)>(), options, label)?;
    require(body, 4, checked_product(count, 8, label)?)?;
    (0..count)
        .map(|i| Ok((be_u32(body, 4 + i * 8)?, be_u32(body, 8 + i * 8)?)))
        .collect()
}

fn parse_ctts(payload: &[u8], options: &Mp4DemuxerOptions) -> Result<Vec<(u32, i64)>> {
    let (version, body) = full(payload)?;
    let count = count(body, 0, options.max_samples_per_track, "ctts")?;
    ensure_allocation(count, std::mem::size_of::<(u32, i64)>(), options, "ctts")?;
    require(body, 4, checked_product(count, 8, "ctts")?)?;
    (0..count)
        .map(|i| {
            let raw = be_u32(body, 8 + i * 8)?;
            Ok((
                be_u32(body, 4 + i * 8)?,
                if version == 0 {
                    i64::from(raw)
                } else if version == 1 {
                    i64::from(raw as i32)
                } else {
                    return Err(malformed("unsupported ctts version"));
                },
            ))
        })
        .collect()
}

fn parse_u32_triples(payload: &[u8], options: &Mp4DemuxerOptions) -> Result<Vec<(u32, u32, u32)>> {
    let (_, body) = full(payload)?;
    let count = count(body, 0, options.max_samples_per_track, "stsc")?;
    ensure_allocation(
        count,
        std::mem::size_of::<(u32, u32, u32)>(),
        options,
        "stsc",
    )?;
    require(body, 4, checked_product(count, 12, "stsc")?)?;
    (0..count)
        .map(|i| {
            Ok((
                be_u32(body, 4 + i * 12)?,
                be_u32(body, 8 + i * 12)?,
                be_u32(body, 12 + i * 12)?,
            ))
        })
        .collect()
}

fn parse_stsz(payload: &[u8], options: &Mp4DemuxerOptions) -> Result<Vec<u32>> {
    let (_, body) = full(payload)?;
    let fixed = be_u32(body, 0)?;
    let count = count(body, 4, options.max_samples_per_track, "sample")?;
    ensure_allocation(count, std::mem::size_of::<u32>(), options, "sample sizes")?;
    if fixed != 0 {
        return Ok(vec![fixed; count]);
    }
    require(body, 8, checked_product(count, 4, "sample sizes")?)?;
    (0..count).map(|i| be_u32(body, 8 + i * 4)).collect()
}

fn parse_stz2(payload: &[u8], options: &Mp4DemuxerOptions) -> Result<Vec<u32>> {
    let (_, body) = full(payload)?;
    let field_size = slice(body, 3, 1)?[0];
    let count = count(body, 4, options.max_samples_per_track, "sample")?;
    ensure_allocation(count, std::mem::size_of::<u32>(), options, "sample sizes")?;
    let encoded_bytes = match field_size {
        4 => count.div_ceil(2),
        8 => count,
        16 => checked_product(count, 2, "compact sample sizes")?,
        _ => return Err(malformed("unsupported compact sample size width")),
    };
    let encoded = slice(body, 8, encoded_bytes)?;
    let mut sizes = Vec::with_capacity(count);
    for index in 0..count {
        sizes.push(match field_size {
            4 => {
                let byte = encoded[index / 2];
                u32::from(if index % 2 == 0 {
                    byte >> 4
                } else {
                    byte & 0x0f
                })
            }
            8 => u32::from(encoded[index]),
            16 => u32::from(u16::from_be_bytes(
                encoded[index * 2..index * 2 + 2]
                    .try_into()
                    .expect("validated compact sample size"),
            )),
            _ => unreachable!("validated compact sample size width"),
        });
    }
    Ok(sizes)
}

fn parse_offsets(payload: &[u8], options: &Mp4DemuxerOptions, wide: bool) -> Result<Vec<u64>> {
    let (_, body) = full(payload)?;
    let count = count(body, 0, options.max_samples_per_track, "chunk")?;
    ensure_allocation(count, std::mem::size_of::<u64>(), options, "chunk offsets")?;
    let width = if wide { 8 } else { 4 };
    require(body, 4, checked_product(count, width, "chunk offsets")?)?;
    (0..count)
        .map(|i| {
            if wide {
                be_u64(body, 4 + i * 8)
            } else {
                be_u32(body, 4 + i * 4).map(u64::from)
            }
        })
        .collect()
}

fn parse_sync(payload: &[u8], options: &Mp4DemuxerOptions) -> Result<BTreeSet<u32>> {
    let (_, body) = full(payload)?;
    let count = count(body, 0, options.max_samples_per_track, "sync sample")?;
    ensure_allocation(count, std::mem::size_of::<u32>(), options, "sync samples")?;
    require(body, 4, checked_product(count, 4, "sync samples")?)?;
    (0..count).map(|i| be_u32(body, 4 + i * 4)).collect()
}

fn parse_sdtp(payload: &[u8], options: &Mp4DemuxerOptions) -> Result<Vec<SampleDependency>> {
    let (_, body) = full(payload)?;
    if body.len() > options.max_samples_per_track as usize {
        return Err(limit("sample dependency count limit exceeded"));
    }
    ensure_allocation(
        body.len(),
        std::mem::size_of::<SampleDependency>(),
        options,
        "sample dependencies",
    )?;
    Ok(body
        .iter()
        .map(|&v| SampleDependency {
            is_leading: v >> 6,
            depends_on: (v >> 4) & 3,
            is_depended_on: (v >> 2) & 3,
            has_redundancy: v & 3,
        })
        .collect())
}

fn finalize_track(mut b: TrackBuilder, options: &Mp4DemuxerOptions) -> Result<Mp4Track> {
    let id = b.id.ok_or_else(|| malformed("track ID is missing"))?;
    let kind = b
        .kind
        .ok_or_else(|| malformed("track handler is missing"))?;
    let codec = b
        .codec
        .ok_or_else(|| malformed("sample description is missing"))?;
    let timescale = b
        .timescale
        .ok_or_else(|| malformed("media header is missing"))?;
    if b.sizes.len() > options.max_samples_per_track as usize {
        return Err(limit("sample count limit exceeded"));
    }
    ensure_allocation(
        b.sizes.len(),
        std::mem::size_of::<Mp4Sample>(),
        options,
        "sample index",
    )?;
    ensure_allocation(
        b.sizes.len(),
        std::mem::size_of::<i64>(),
        options,
        "expanded timing table",
    )?;
    if b.sync.as_ref().is_some_and(|sync| {
        sync.iter()
            .any(|&index| index == 0 || index as usize > b.sizes.len())
    }) {
        return Err(malformed("sync sample index is out of range"));
    }
    if !b.dependencies.is_empty() && b.dependencies.len() != b.sizes.len() {
        return Err(malformed("sample dependency count does not match samples"));
    }
    if !b.sizes.is_empty() {
        let durations = expand_runs(&b.stts, b.sizes.len(), "decode timing")?;
        let offsets = if b.ctts.is_empty() {
            vec![0; b.sizes.len()]
        } else {
            expand_signed_runs(&b.ctts, b.sizes.len(), "composition timing")?
        };
        let sample_offsets = expand_chunks(&b.stsc, &b.chunks, &b.sizes)?;
        let mut dts = 0_u64;
        for i in 0..b.sizes.len() {
            let duration = durations[i];
            if duration == 0 {
                return Err(malformed("sample duration is zero"));
            }
            let pts = i64::try_from(dts)
                .map_err(|_| limit("decode timestamp exceeds i64"))?
                .checked_add(offsets[i])
                .ok_or_else(|| malformed("presentation timestamp overflow"))?;
            let number = u32::try_from(i + 1).map_err(|_| limit("sample index overflow"))?;
            let is_sync = b.sync.as_ref().is_none_or(|s| s.contains(&number));
            let dependency = b.dependencies.get(i).copied().unwrap_or(if is_sync {
                SampleDependency::INDEPENDENT
            } else {
                SampleDependency::DEPENDENT
            });
            b.samples.push(Mp4Sample {
                offset: sample_offsets[i],
                size: b.sizes[i],
                dts,
                pts,
                duration,
                dependency,
                is_sync,
            });
            dts = dts
                .checked_add(u64::from(duration))
                .ok_or_else(|| limit("decode timestamp overflow"))?;
        }
    }
    Ok(Mp4Track {
        id,
        kind,
        codec,
        timescale,
        duration: b.duration,
        dimensions: b.dimensions,
        channels: b.channels,
        sample_rate: b.sample_rate,
        decoder_config: b.decoder_config,
        edits: b.edits,
        samples: b.samples,
        presentation_order: Vec::new(),
    })
}

fn expand_runs(runs: &[(u32, u32)], expected: usize, label: &str) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(expected);
    for &(count, value) in runs {
        let count = usize::try_from(count).map_err(|_| limit(format!("{label} count overflow")))?;
        if out.len().checked_add(count).is_none_or(|n| n > expected) {
            return Err(malformed(format!("{label} count exceeds samples")));
        }
        out.extend(std::iter::repeat_n(value, count));
    }
    if out.len() != expected {
        return Err(malformed(format!("{label} count does not match samples")));
    }
    Ok(out)
}
fn expand_signed_runs(runs: &[(u32, i64)], expected: usize, label: &str) -> Result<Vec<i64>> {
    let mut out = Vec::with_capacity(expected);
    for &(count, value) in runs {
        let count = count as usize;
        if out.len().checked_add(count).is_none_or(|n| n > expected) {
            return Err(malformed(format!("{label} count exceeds samples")));
        }
        out.extend(std::iter::repeat_n(value, count));
    }
    if out.len() != expected {
        return Err(malformed(format!("{label} count does not match samples")));
    }
    Ok(out)
}

fn expand_chunks(stsc: &[(u32, u32, u32)], chunks: &[u64], sizes: &[u32]) -> Result<Vec<u64>> {
    if sizes.is_empty() {
        return Ok(Vec::new());
    }
    if stsc.is_empty() || chunks.is_empty() {
        return Err(malformed("sample-to-chunk or chunk offsets are missing"));
    }
    if stsc[0].0 != 1 {
        return Err(malformed("first stsc entry must start at chunk one"));
    }
    if stsc.windows(2).any(|entries| entries[0].0 >= entries[1].0) {
        return Err(malformed("stsc first-chunk values are not increasing"));
    }
    if stsc.iter().any(|entry| entry.1 == 0 || entry.2 != 1) {
        return Err(malformed("invalid or unsupported stsc entry"));
    }
    let mut out = Vec::with_capacity(sizes.len());
    let mut sample = 0;
    for (ci, &chunk_offset) in chunks.iter().enumerate() {
        let chunk_number = u32::try_from(ci + 1).map_err(|_| limit("chunk index overflow"))?;
        let entry = stsc
            .iter()
            .rev()
            .find(|e| e.0 <= chunk_number)
            .ok_or_else(|| malformed("chunk has no stsc mapping"))?;
        let mut offset = chunk_offset;
        for _ in 0..entry.1 {
            if sample >= sizes.len() {
                return Err(malformed("chunks contain more samples than stsz"));
            }
            out.push(offset);
            offset = offset
                .checked_add(u64::from(sizes[sample]))
                .ok_or_else(|| malformed("sample offset overflow"))?;
            sample += 1;
        }
    }
    if sample != sizes.len() {
        return Err(malformed("chunks contain fewer samples than stsz"));
    }
    Ok(out)
}

#[derive(Clone, Copy, Default)]
struct Trex {
    duration: u32,
    size: u32,
    flags: u32,
}
fn parse_mvex(
    bytes: &[u8],
    trex: &mut BTreeMap<u32, Trex>,
    budget: &mut Budget,
    options: &Mp4DemuxerOptions,
) -> Result<()> {
    for child in children(bytes, 2, budget, options)? {
        if &child.kind == b"trex" {
            let (_, body) = full(child.payload)?;
            let id = be_u32(body, 0)?;
            trex.insert(
                id,
                Trex {
                    duration: be_u32(body, 8)?,
                    size: be_u32(body, 12)?,
                    flags: be_u32(body, 16)?,
                },
            );
        }
    }
    Ok(())
}

struct Fragment {
    moof_start: u64,
    trafs: Vec<Traf>,
}
#[derive(Default)]
struct Traf {
    track_id: u32,
    base: u64,
    decode_time: u64,
    default_duration: u32,
    default_size: u32,
    default_flags: u32,
    runs: Vec<Run>,
}
struct Run {
    data_offset: Option<i32>,
    first_flags: Option<u32>,
    samples: Vec<FragSample>,
}
#[derive(Default)]
struct FragSample {
    duration: Option<u32>,
    size: Option<u32>,
    flags: Option<u32>,
    cto: i64,
}

fn parse_moof(
    bytes: &[u8],
    start: u64,
    budget: &mut Budget,
    options: &Mp4DemuxerOptions,
) -> Result<Fragment> {
    let mut trafs = Vec::new();
    for child in children(bytes, 1, budget, options)? {
        if &child.kind == b"traf" {
            trafs.push(parse_traf(child.payload, start, budget, options)?);
        }
    }
    Ok(Fragment {
        moof_start: start,
        trafs,
    })
}
fn parse_traf(
    bytes: &[u8],
    moof_start: u64,
    budget: &mut Budget,
    options: &Mp4DemuxerOptions,
) -> Result<Traf> {
    let mut traf = Traf {
        base: moof_start,
        ..Traf::default()
    };
    for child in children(bytes, 2, budget, options)? {
        match &child.kind {
            b"tfhd" => parse_tfhd(child.payload, &mut traf, moof_start)?,
            b"tfdt" => parse_tfdt(child.payload, &mut traf)?,
            b"trun" => traf.runs.push(parse_trun(child.payload, options)?),
            _ => {}
        }
    }
    if traf.track_id == 0 {
        return Err(malformed("fragment track ID is missing"));
    }
    Ok(traf)
}
fn parse_tfhd(payload: &[u8], traf: &mut Traf, moof_start: u64) -> Result<()> {
    let (_, flags, body) = full_flags(payload)?;
    let mut at = 0;
    traf.track_id = be_u32(body, at)?;
    at += 4;
    if flags & 1 != 0 {
        traf.base = be_u64(body, at)?;
        at += 8;
    } else if flags & 0x020000 != 0 {
        traf.base = moof_start;
    }
    if flags & 2 != 0 {
        at += 4;
    }
    if flags & 8 != 0 {
        traf.default_duration = be_u32(body, at)?;
        at += 4;
    }
    if flags & 0x10 != 0 {
        traf.default_size = be_u32(body, at)?;
        at += 4;
    }
    if flags & 0x20 != 0 {
        traf.default_flags = be_u32(body, at)?;
    }
    Ok(())
}
fn parse_tfdt(payload: &[u8], traf: &mut Traf) -> Result<()> {
    let (version, body) = full(payload)?;
    traf.decode_time = if version == 1 {
        be_u64(body, 0)?
    } else if version == 0 {
        u64::from(be_u32(body, 0)?)
    } else {
        return Err(malformed("unsupported tfdt version"));
    };
    Ok(())
}
fn parse_trun(payload: &[u8], options: &Mp4DemuxerOptions) -> Result<Run> {
    let (version, flags, body) = full_flags(payload)?;
    let count = count(body, 0, options.max_samples_per_track, "fragment sample")?;
    ensure_allocation(
        count,
        std::mem::size_of::<FragSample>(),
        options,
        "fragment samples",
    )?;
    let mut at = 4;
    let data_offset = if flags & 1 != 0 {
        let v = be_i32(body, at)?;
        at += 4;
        Some(v)
    } else {
        None
    };
    let first_flags = if flags & 4 != 0 {
        let v = be_u32(body, at)?;
        at += 4;
        Some(v)
    } else {
        None
    };
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        let mut s = FragSample::default();
        if flags & 0x100 != 0 {
            s.duration = Some(be_u32(body, at)?);
            at += 4;
        }
        if flags & 0x200 != 0 {
            s.size = Some(be_u32(body, at)?);
            at += 4;
        }
        if flags & 0x400 != 0 {
            s.flags = Some(be_u32(body, at)?);
            at += 4;
        }
        if flags & 0x800 != 0 {
            let raw = be_u32(body, at)?;
            at += 4;
            s.cto = if version == 0 {
                i64::from(raw)
            } else if version == 1 {
                i64::from(raw as i32)
            } else {
                return Err(malformed("unsupported trun version"));
            };
        }
        samples.push(s);
    }
    Ok(Run {
        data_offset,
        first_flags,
        samples,
    })
}

fn append_fragments(
    movie: &mut Mp4Demuxer,
    fragments: Vec<Fragment>,
    trex: &BTreeMap<u32, Trex>,
    options: &Mp4DemuxerOptions,
) -> Result<()> {
    for fragment in fragments {
        for traf in fragment.trafs {
            let track = movie
                .tracks
                .iter_mut()
                .find(|t| t.id == traf.track_id)
                .ok_or_else(|| malformed("fragment references an unknown track"))?;
            let defaults = trex.get(&traf.track_id).copied().unwrap_or_default();
            let mut dts = traf.decode_time;
            let mut next_data = None;
            for run in traf.runs {
                let mut offset = if let Some(relative) = run.data_offset {
                    add_signed(traf.base, relative)?
                } else {
                    next_data.unwrap_or(fragment.moof_start)
                };
                for (i, s) in run.samples.into_iter().enumerate() {
                    if track.samples.len() >= options.max_samples_per_track as usize {
                        return Err(limit("fragment sample count limit exceeded"));
                    }
                    ensure_allocation(
                        track.samples.len() + 1,
                        std::mem::size_of::<Mp4Sample>(),
                        options,
                        "sample index",
                    )?;
                    let duration = s
                        .duration
                        .or(nonzero(traf.default_duration))
                        .or(nonzero(defaults.duration))
                        .ok_or_else(|| malformed("fragment sample duration is missing"))?;
                    let size = s
                        .size
                        .or(nonzero(traf.default_size))
                        .or(nonzero(defaults.size))
                        .ok_or_else(|| malformed("fragment sample size is missing"))?;
                    let flags = s
                        .flags
                        .or(if i == 0 { run.first_flags } else { None })
                        .unwrap_or(if traf.default_flags != 0 {
                            traf.default_flags
                        } else {
                            defaults.flags
                        });
                    let is_sync = flags & 0x0001_0000 == 0;
                    let dependency = dependency_from_flags(flags, is_sync);
                    let pts = i64::try_from(dts)
                        .map_err(|_| limit("fragment DTS exceeds i64"))?
                        .checked_add(s.cto)
                        .ok_or_else(|| malformed("fragment PTS overflow"))?;
                    track.samples.push(Mp4Sample {
                        offset,
                        size,
                        dts,
                        pts,
                        duration,
                        dependency,
                        is_sync,
                    });
                    offset = offset
                        .checked_add(u64::from(size))
                        .ok_or_else(|| malformed("fragment data offset overflow"))?;
                    dts = dts
                        .checked_add(u64::from(duration))
                        .ok_or_else(|| malformed("fragment DTS overflow"))?;
                }
                next_data = Some(offset);
            }
        }
    }
    Ok(())
}
fn dependency_from_flags(flags: u32, is_sync: bool) -> SampleDependency {
    let value = ((flags >> 20) & 0xff) as u8;
    let parsed = SampleDependency {
        is_leading: value >> 6,
        depends_on: (value >> 4) & 3,
        is_depended_on: (value >> 2) & 3,
        has_redundancy: value & 3,
    };
    if value == 0 {
        if is_sync {
            SampleDependency::INDEPENDENT
        } else {
            SampleDependency::DEPENDENT
        }
    } else {
        parsed
    }
}
fn nonzero(v: u32) -> Option<u32> {
    (v != 0).then_some(v)
}
fn add_signed(base: u64, delta: i32) -> Result<u64> {
    if delta >= 0 {
        base.checked_add(delta as u64)
    } else {
        base.checked_sub(u64::from(delta.unsigned_abs()))
    }
    .ok_or_else(|| malformed("fragment data offset overflow"))
}

fn validate_samples(track: &Mp4Track, source_len: u64, media_ranges: &[(u64, u64)]) -> Result<()> {
    for sample in &track.samples {
        if sample.size == 0 {
            return Err(malformed("sample size is zero"));
        }
        let end = sample
            .offset
            .checked_add(u64::from(sample.size))
            .ok_or_else(|| malformed("sample byte range overflow"))?;
        if end > source_len {
            return Err(malformed("sample byte range exceeds source"));
        }
        if !media_ranges
            .iter()
            .any(|&(start, range_end)| sample.offset >= start && end <= range_end)
        {
            return Err(malformed("sample byte range is outside media data"));
        }
    }
    Ok(())
}

struct Child<'a> {
    kind: [u8; 4],
    payload: &'a [u8],
    whole: &'a [u8],
}
fn children<'a>(
    bytes: &'a [u8],
    depth: u8,
    budget: &mut Budget,
    options: &Mp4DemuxerOptions,
) -> Result<Vec<Child<'a>>> {
    if depth > options.max_box_depth {
        return Err(limit("MP4 box nesting limit exceeded"));
    }
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        budget.box_seen()?;
        if bytes.len() - at < 8 {
            return Err(malformed("truncated child box header"));
        }
        let size32 = be_u32(bytes, at)?;
        let (kind, header, size) = if size32 == 1 {
            (
                slice(bytes, at + 4, 4)?.try_into().expect("fixed"),
                16,
                usize::try_from(be_u64(bytes, at + 8)?)
                    .map_err(|_| limit("extended child box too large"))?,
            )
        } else {
            (
                slice(bytes, at + 4, 4)?.try_into().expect("fixed"),
                8,
                if size32 == 0 {
                    bytes.len() - at
                } else {
                    size32 as usize
                },
            )
        };
        if size < header {
            return Err(malformed("child box is smaller than its header"));
        }
        let end = at
            .checked_add(size)
            .ok_or_else(|| malformed("child box size overflow"))?;
        if end > bytes.len() {
            return Err(malformed("child box exceeds its parent"));
        }
        ensure_allocation(
            out.len() + 1,
            std::mem::size_of::<Child<'_>>(),
            options,
            "child boxes",
        )?;
        out.push(Child {
            kind,
            payload: &bytes[at + header..end],
            whole: &bytes[at..end],
        });
        at = end;
    }
    Ok(out)
}
fn one_child<'a>(
    bytes: &'a [u8],
    depth: u8,
    budget: &mut Budget,
    options: &Mp4DemuxerOptions,
) -> Result<Child<'a>> {
    let mut values = children(bytes, depth, budget, options)?;
    if values.len() != 1 {
        return Err(malformed("sample description entry count is inconsistent"));
    }
    Ok(values.remove(0))
}
fn full(payload: &[u8]) -> Result<(u8, &[u8])> {
    let (version, _, body) = full_flags(payload)?;
    Ok((version, body))
}
fn full_flags(payload: &[u8]) -> Result<(u8, u32, &[u8])> {
    require(payload, 0, 4)?;
    Ok((
        payload[0],
        u32::from_be_bytes([0, payload[1], payload[2], payload[3]]),
        &payload[4..],
    ))
}
fn count(bytes: &[u8], at: usize, max: u32, label: &str) -> Result<usize> {
    let value = be_u32(bytes, at)?;
    if value > max {
        return Err(limit(format!("{label} count limit exceeded")));
    }
    Ok(value as usize)
}
fn checked_product(count: usize, width: usize, label: &str) -> Result<usize> {
    count
        .checked_mul(width)
        .ok_or_else(|| malformed(format!("{label} byte count overflow")))
}
fn ensure_allocation(
    count: usize,
    element_size: usize,
    options: &Mp4DemuxerOptions,
    label: &str,
) -> Result<()> {
    let bytes = checked_product(count, element_size, label)?;
    if bytes as u128 > u128::from(options.limits.max_allocation_bytes) {
        return Err(limit(format!("{label} allocation limit exceeded")));
    }
    Ok(())
}
fn require(bytes: &[u8], at: usize, len: usize) -> Result<()> {
    let end = at
        .checked_add(len)
        .ok_or_else(|| malformed("box field offset overflow"))?;
    if end > bytes.len() {
        return Err(malformed("truncated MP4 box fields"));
    }
    Ok(())
}
fn slice(bytes: &[u8], at: usize, len: usize) -> Result<&[u8]> {
    require(bytes, at, len)?;
    Ok(&bytes[at..at + len])
}

fn scale_time(value: u64, source_timescale: u32, destination_rate: u32) -> Result<u64> {
    if source_timescale == 0 || destination_rate == 0 {
        return Err(malformed("media timescale and sample rate must be nonzero"));
    }
    let numerator = u128::from(value)
        .checked_mul(u128::from(destination_rate))
        .ok_or_else(|| limit("media time scaling overflow"))?;
    let rounded = numerator
        .checked_add(u128::from(source_timescale) / 2)
        .ok_or_else(|| limit("media time rounding overflow"))?
        / u128::from(source_timescale);
    u64::try_from(rounded).map_err(|_| limit("scaled media time cannot be represented"))
}

fn parse_aac_config(esds: &[u8], entry_rate: u32, entry_channels: u16) -> Result<AacTrackConfig> {
    if esds.len() < 13 || &esds[4..8] != b"esds" {
        return Err(malformed("AAC decoder configuration is not an esds box"));
    }
    let payload = &esds[12..];
    let mut unsupported_type = None;
    for at in 0..payload.len() {
        if payload[at] != 0x05 {
            continue;
        }
        let Ok((length, header)) = descriptor_length(&payload[at + 1..]) else {
            continue;
        };
        let start = at + 1 + header;
        let Some(end) = start.checked_add(length) else {
            continue;
        };
        if end > payload.len() || length < 2 {
            continue;
        }
        let asc = &payload[start..end];
        let Ok((object_type, sample_rate, channel_config)) = parse_audio_specific_config(asc)
        else {
            continue;
        };
        if object_type != 2 {
            unsupported_type = Some(object_type);
            continue;
        }
        if sample_rate != entry_rate {
            return Err(malformed(
                "AAC AudioSpecificConfig sample rate disagrees with mp4a",
            ));
        }
        if u16::from(channel_config) != entry_channels {
            return Err(malformed(
                "AAC AudioSpecificConfig channel count disagrees with mp4a",
            ));
        }
        return Ok(AacTrackConfig {
            audio_object_type: object_type,
            sample_rate,
            channels: entry_channels,
            audio_specific_config: asc.to_vec(),
        });
    }
    if let Some(object_type) = unsupported_type {
        Err(unsupported(format!(
            "AAC audio object type {object_type} is unsupported; AAC-LC is required"
        )))
    } else {
        Err(malformed(
            "AAC esds does not contain a valid DecoderSpecificInfo descriptor",
        ))
    }
}

fn descriptor_length(bytes: &[u8]) -> Result<(usize, usize)> {
    let mut length = 0_usize;
    for (index, &byte) in bytes.iter().take(4).enumerate() {
        length = length
            .checked_shl(7)
            .and_then(|value| value.checked_add(usize::from(byte & 0x7f)))
            .ok_or_else(|| malformed("AAC descriptor length overflow"))?;
        if byte & 0x80 == 0 {
            return Ok((length, index + 1));
        }
    }
    Err(malformed("AAC descriptor has an invalid length"))
}

fn parse_audio_specific_config(bytes: &[u8]) -> Result<(u8, u32, u8)> {
    let mut bits = AscBits::new(bytes);
    let mut object_type = bits.read(5)? as u8;
    if object_type == 31 {
        object_type = 32_u8
            .checked_add(bits.read(6)? as u8)
            .ok_or_else(|| malformed("AAC audio object type overflow"))?;
    }
    let frequency_index = bits.read(4)? as usize;
    const FREQUENCIES: [u32; 13] = [
        96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025,
        8_000, 7_350,
    ];
    let sample_rate = if frequency_index == 15 {
        bits.read(24)?
    } else {
        *FREQUENCIES
            .get(frequency_index)
            .ok_or_else(|| malformed("AAC AudioSpecificConfig has an invalid sample rate"))?
    };
    let channel_config = bits.read(4)? as u8;
    if channel_config == 0 || channel_config > 7 {
        return Err(unsupported(
            "AAC program-config-element channel layouts are unsupported",
        ));
    }
    Ok((object_type, sample_rate, channel_config))
}

struct AscBits<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> AscBits<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn read(&mut self, count: usize) -> Result<u32> {
        if count > 32
            || self
                .position
                .checked_add(count)
                .is_none_or(|end| end > self.bytes.len() * 8)
        {
            return Err(malformed("truncated AAC AudioSpecificConfig"));
        }
        let mut value = 0_u32;
        for _ in 0..count {
            value = (value << 1)
                | u32::from((self.bytes[self.position / 8] >> (7 - self.position % 8)) & 1);
            self.position += 1;
        }
        Ok(value)
    }
}
fn be_u16(b: &[u8], a: usize) -> Result<u16> {
    Ok(u16::from_be_bytes(
        slice(b, a, 2)?.try_into().expect("fixed"),
    ))
}
fn be_i16(b: &[u8], a: usize) -> Result<i16> {
    Ok(i16::from_be_bytes(
        slice(b, a, 2)?.try_into().expect("fixed"),
    ))
}
fn be_u32(b: &[u8], a: usize) -> Result<u32> {
    Ok(u32::from_be_bytes(
        slice(b, a, 4)?.try_into().expect("fixed"),
    ))
}
fn be_i32(b: &[u8], a: usize) -> Result<i32> {
    Ok(i32::from_be_bytes(
        slice(b, a, 4)?.try_into().expect("fixed"),
    ))
}
fn be_u64(b: &[u8], a: usize) -> Result<u64> {
    Ok(u64::from_be_bytes(
        slice(b, a, 8)?.try_into().expect("fixed"),
    ))
}
fn be_i64(b: &[u8], a: usize) -> Result<i64> {
    Ok(i64::from_be_bytes(
        slice(b, a, 8)?.try_into().expect("fixed"),
    ))
}
fn validate_options(o: &Mp4DemuxerOptions) -> Result<()> {
    if o.max_box_bytes < 8
        || o.max_boxes == 0
        || o.max_box_depth == 0
        || o.max_samples_per_track == 0
    {
        return Err(invalid("MP4 parser limits must be nonzero"));
    }
    Ok(())
}
fn invalid(m: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, m)
}
fn malformed(m: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedMedia, m)
}
fn limit(m: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, m)
}
fn unsupported(m: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{IoFuture, MemorySource};
    use std::cell::Cell;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let mut future = Box::pin(future);
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        loop {
            match Pin::new(&mut future).poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    fn boxed(kind: &[u8; 4], payload: impl AsRef<[u8]>) -> Vec<u8> {
        let payload = payload.as_ref();
        let mut bytes = u32::try_from(payload.len() + 8)
            .unwrap()
            .to_be_bytes()
            .to_vec();
        bytes.extend_from_slice(kind);
        bytes.extend_from_slice(payload);
        bytes
    }

    fn full_box(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
        let mut payload = vec![version];
        payload.extend_from_slice(&flags.to_be_bytes()[1..]);
        payload.extend_from_slice(body);
        boxed(kind, payload)
    }

    fn container(kind: &[u8; 4], children: &[Vec<u8>]) -> Vec<u8> {
        boxed(kind, children.concat())
    }

    fn fragmented_fixture() -> Vec<u8> {
        let mut mvhd = vec![0; 16];
        mvhd[8..12].copy_from_slice(&1_000_u32.to_be_bytes());

        let mut tkhd = vec![0; 80];
        tkhd[8..12].copy_from_slice(&1_u32.to_be_bytes());
        tkhd[72..76].copy_from_slice(&(2_u32 << 16).to_be_bytes());
        tkhd[76..80].copy_from_slice(&(2_u32 << 16).to_be_bytes());

        let mut mdhd = vec![0; 20];
        mdhd[8..12].copy_from_slice(&1_000_u32.to_be_bytes());
        let mut hdlr = vec![0; 8];
        hdlr[4..8].copy_from_slice(b"vide");

        let mut sample_entry = vec![0; 78];
        sample_entry[6..8].copy_from_slice(&1_u16.to_be_bytes());
        sample_entry[24..26].copy_from_slice(&2_u16.to_be_bytes());
        sample_entry[26..28].copy_from_slice(&2_u16.to_be_bytes());
        sample_entry.extend_from_slice(&boxed(b"av1C", [0x81, 0, 0, 0]));
        let sample_entry = boxed(b"av01", sample_entry);
        let mut stsd_body = 1_u32.to_be_bytes().to_vec();
        stsd_body.extend_from_slice(&sample_entry);
        let stbl = container(
            b"stbl",
            &[
                full_box(b"stsd", 0, 0, &stsd_body),
                full_box(b"stts", 0, 0, &0_u32.to_be_bytes()),
                full_box(b"stsc", 0, 0, &0_u32.to_be_bytes()),
                full_box(b"stsz", 0, 0, &[0; 8]),
                full_box(b"stco", 0, 0, &0_u32.to_be_bytes()),
            ],
        );
        let minf = container(b"minf", &[stbl]);
        let mdia = container(
            b"mdia",
            &[
                full_box(b"mdhd", 0, 0, &mdhd),
                full_box(b"hdlr", 0, 0, &hdlr),
                minf,
            ],
        );
        let trak = container(b"trak", &[full_box(b"tkhd", 0, 7, &tkhd), mdia]);

        let mut trex = Vec::new();
        trex.extend_from_slice(&1_u32.to_be_bytes());
        trex.extend_from_slice(&1_u32.to_be_bytes());
        trex.extend_from_slice(&1_000_u32.to_be_bytes());
        trex.extend_from_slice(&4_u32.to_be_bytes());
        trex.extend_from_slice(&0_u32.to_be_bytes());
        let mvex = container(b"mvex", &[full_box(b"trex", 0, 0, &trex)]);
        let moov = container(b"moov", &[full_box(b"mvhd", 0, 0, &mvhd), trak, mvex]);

        let tfhd = full_box(b"tfhd", 0, 0x020000, &1_u32.to_be_bytes());
        let tfdt = full_box(b"tfdt", 1, 0, &0_u64.to_be_bytes());
        let make_moof = |data_offset: i32| {
            let mut trun = 2_u32.to_be_bytes().to_vec();
            trun.extend_from_slice(&data_offset.to_be_bytes());
            trun.extend_from_slice(&1_000_i32.to_be_bytes());
            trun.extend_from_slice(&(-1_000_i32).to_be_bytes());
            let traf = container(
                b"traf",
                &[
                    tfhd.clone(),
                    tfdt.clone(),
                    full_box(b"trun", 1, 0x801, &trun),
                ],
            );
            container(b"moof", &[traf])
        };
        let provisional = make_moof(0);
        let moof = make_moof(i32::try_from(provisional.len() + 8).unwrap());
        let mdat = boxed(b"mdat", b"AAAABBBB");
        [boxed(b"ftyp", b"isom\0\0\0\0isom"), moov, moof, mdat].concat()
    }

    struct PartialSource {
        inner: MemorySource,
        largest_request: Cell<usize>,
    }

    impl ByteSource for PartialSource {
        fn len(&self) -> Option<u64> {
            self.inner.len()
        }

        fn read_at<'a>(&'a self, offset: u64, destination: &'a mut [u8]) -> IoFuture<'a, usize> {
            self.largest_request
                .set(self.largest_request.get().max(destination.len()));
            let limited = destination.len().min(3);
            self.inner.read_at(offset, &mut destination[..limited])
        }
    }

    #[test]
    fn fragmented_mp4_expands_decode_and_presentation_indexes_incrementally() {
        block_on(async {
            let source = PartialSource {
                inner: MemorySource::new(fragmented_fixture()),
                largest_request: Cell::new(0),
            };
            assert!(probe_mp4(&source).await.unwrap());
            let movie = Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())
                .await
                .unwrap();
            let track = movie.track(1).unwrap();
            assert_eq!(track.codec, Codec::Av1);
            assert_eq!(track.samples.len(), 2);
            assert_eq!(track.samples[0].dts, 0);
            assert_eq!(track.samples[0].pts, 1_000);
            assert_eq!(track.samples[1].dts, 1_000);
            assert_eq!(track.samples[1].pts, 0);
            assert_eq!(track.presentation_order, vec![1, 0]);
            assert_eq!(track.samples[1].offset, track.samples[0].offset + 4);
            assert_eq!(track.presentation_sample(0), Some(&track.samples[1]));
            let mut encoded = [0; 4];
            track
                .read_sample_into(&source, 1, &mut encoded)
                .await
                .unwrap();
            assert_eq!(&encoded, b"BBBB");
            assert!(source.largest_request.get() < source.inner.as_slice().len());
        });
    }

    #[test]
    fn malformed_sizes_and_limits_fail_without_panicking() {
        block_on(async {
            let valid = fragmented_fixture();
            for length in 0..valid.len() {
                let _result = Mp4Demuxer::open(
                    &MemorySource::new(valid[..length].to_vec()),
                    Mp4DemuxerOptions::default(),
                )
                .await;
            }
            for declared in 0_u32..8 {
                let mut bytes = valid.clone();
                bytes[..4].copy_from_slice(&declared.to_be_bytes());
                let result =
                    Mp4Demuxer::open(&MemorySource::new(bytes), Mp4DemuxerOptions::default()).await;
                assert!(result.is_err(), "top-level size {declared} was accepted");
            }

            let options = Mp4DemuxerOptions {
                max_box_bytes: 32,
                ..Mp4DemuxerOptions::default()
            };
            let error = Mp4Demuxer::open(&MemorySource::new(valid), options)
                .await
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::ResourceLimit);
        });
    }

    #[test]
    fn compact_sample_sizes_expand_at_every_supported_width() {
        let options = Mp4DemuxerOptions::default();
        for (width, count, encoded, expected) in [
            (4_u8, 3_u32, vec![0x12, 0x30], vec![1, 2, 3]),
            (8, 3, vec![1, 2, 255], vec![1, 2, 255]),
            (16, 2, vec![0, 1, 1, 0], vec![1, 256]),
        ] {
            let mut body = vec![0; 4];
            body[3] = width;
            body.extend_from_slice(&count.to_be_bytes());
            body.extend_from_slice(&encoded);
            assert_eq!(
                parse_stz2(&full_box(b"stz2", 0, 0, &body)[8..], &options).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn to_encoded_video_samples_reads_the_bundled_hevc_sample_in_presentation_order() {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/media/BigBuckBunny.mp4"
        ))
        .expect("bundled sample video is checked into the repository");
        let source = MemorySource::new(bytes);
        let demuxer = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())).unwrap();
        let track = demuxer
            .tracks
            .iter()
            .find(|track| track.kind == TrackKind::Video)
            .expect("sample video has a video track");
        assert_eq!(track.codec, Codec::Hevc);
        assert!(!track.decoder_config.is_empty());

        let samples =
            block_on(track.to_encoded_video_samples(&source, &Limits::default())).unwrap();
        assert_eq!(samples.len(), track.samples.len());
        assert!(
            samples[0].random_access,
            "decode must start at a sync sample"
        );
        let mut seen: Vec<u64> = samples.iter().map(|s| s.presentation_index.0).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            samples.len(),
            "presentation indexes must be unique"
        );
        assert!(samples.iter().all(|sample| !sample.data.is_empty()));

        let derived =
            crate::codec_config::derive_codec_string(track.codec, &track.decoder_config).unwrap();
        assert!(derived.codec_string.starts_with("hev1."));
    }

    #[test]
    fn bundled_aac_track_exposes_indexed_packets_configuration_and_gapless_timing() {
        block_on(async {
            let source =
                MemorySource::new(include_bytes!("../examples/media/BigBuckBunny.mp4").to_vec());
            let movie = Mp4Demuxer::open(&source, Mp4DemuxerOptions::default())
                .await
                .unwrap();
            let track = movie
                .tracks
                .iter()
                .find(|track| track.kind == TrackKind::Audio)
                .unwrap();
            let config = track.aac_config().unwrap();
            assert_eq!(config.audio_object_type, 2);
            assert_eq!(config.sample_rate, 48_000);
            assert_eq!(config.channels, 2);
            assert_eq!(config.audio_specific_config, [0x11, 0x90, 0x56, 0xe5, 0x00]);

            let packets = track
                .to_encoded_audio_samples(&source, &Limits::default())
                .await
                .unwrap();
            assert_eq!(packets.len(), 1_501);
            assert_eq!(
                packets[0].decoded_range,
                crate::SampleRange::new(0, 1_024).unwrap()
            );
            assert_eq!(
                packets[1].decoded_range,
                crate::SampleRange::new(1_024, 2_048).unwrap()
            );
            assert_eq!(packets.last().unwrap().decoded_range.end, 1_537_024);
            assert_eq!(packets[0].data.len(), track.samples[0].size as usize);

            let timing = track.audio_timing(movie.movie_timescale).unwrap();
            assert_eq!(timing.priming, 1_024);
            assert_eq!(timing.padding, 0);
            assert_eq!(timing.edits.len(), 1);
            assert_eq!(
                timing.edits[0].presentation,
                crate::SampleRange::new(0, 1_536_000).unwrap()
            );
            assert_eq!(timing.edits[0].media_start, Some(1_024));
        });
    }
}
