//! Deterministic seekable WebM muxing.
//!
//! [`WebmMuxer`] writes the same guarantees [`crate::mp4::Mp4Muxer`] does:
//! payload bytes go to the sink as each sample arrives, and `finish` seeks
//! back to fill in what only the end of the stream knows - the Segment and
//! Cluster sizes, the `Duration`, and a `SeekHead` naming the `Cues` written
//! after the last Cluster - so the file is seekable in a browser and in
//! [`crate::WebmDemuxer`] without a remux.
//!
//! WebM permits only VP8, VP9 and AV1 video and Vorbis or Opus audio. zvidlib
//! encodes AV1, so output is AV1 video only: an AAC track is refused rather
//! than written into a file no WebM player would accept.

use crate::codec::{EncodedSample, TrackKind};
use crate::ebml::{self, write_element, write_float, write_id, write_string, write_uint};
use crate::io::ByteSink;
use crate::media::Codec;
use crate::mp4::{Mp4TrackConfig, Mp4TrackFormat};
use crate::{Error, ErrorKind, Result};

/// One tick of every block timestamp written: a millisecond, Matroska's
/// default and the scale every WebM player expects.
const TIMESTAMP_SCALE_NS: u64 = 1_000_000;
const TICKS_PER_SECOND: u64 = 1_000;
/// Track numbers are written as one-byte block headers.
const MAX_TRACKS: usize = 126;
/// The Seek entries `finish` fills in, in `SeekHead` order.
const SEEK_TARGETS: [u32; 3] = [ebml::INFO, ebml::TRACKS, ebml::CUES];
const MUXING_APP: &str = concat!("zvidlib ", env!("CARGO_PKG_VERSION"));

struct TrackState {
    number: u64,
    timescale: u32,
    is_video: bool,
    samples: usize,
}

struct OpenCluster {
    /// Absolute offset of the Cluster element.
    start: u64,
    /// Absolute offset of the Cluster's payload, after its fixed-size size.
    data_start: u64,
    timestamp: u64,
    blocks: usize,
}

struct CueRecord {
    time: u64,
    track: u64,
    /// Relative to the Segment payload, as `CueClusterPosition` is.
    cluster_position: u64,
    /// Relative to the Cluster payload, as `CueRelativePosition` is.
    relative_position: u64,
}

/// A seekable WebM muxer that writes payload bytes immediately.
///
/// Track declarations are the same [`Mp4TrackConfig`] values
/// [`crate::mp4::Mp4Muxer`] takes, so one encoder configuration feeds either
/// container. Samples from every track are written in one nondecreasing
/// presentation-time order, because a WebM Cluster interleaves its tracks
/// rather than storing each in a run of its own.
pub struct WebmMuxer<S> {
    sink: S,
    tracks: Vec<TrackState>,
    segment_size_position: u64,
    segment_data_start: u64,
    header_position: u64,
    header_length: usize,
    info_position: u64,
    tracks_position: u64,
    cluster: Option<OpenCluster>,
    cues: Vec<CueRecord>,
    last_timestamp: u64,
    end_ticks: f64,
    max_samples_per_track: usize,
    finished: bool,
}

impl<S: ByteSink> WebmMuxer<S> {
    pub async fn new(
        mut sink: S,
        configs: Vec<Mp4TrackConfig>,
        max_samples_per_track: usize,
    ) -> Result<Self> {
        if !sink.is_seekable() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "seekable WebM output requires a seekable sink",
            ));
        }
        if configs.is_empty() {
            return Err(invalid("a WebM must declare at least one track"));
        }
        if configs.len() > MAX_TRACKS {
            return Err(Error::new(ErrorKind::ResourceLimit, "too many WebM tracks"));
        }
        if max_samples_per_track == 0 {
            return Err(invalid("the WebM sample limit must be nonzero"));
        }
        for config in &configs {
            validate_track_config(config)?;
        }

        sink.write(&ebml_header()).await?;
        let mut segment = Vec::new();
        write_id(&mut segment, ebml::SEGMENT);
        let segment_size_position = sink.position() + segment.len() as u64;
        ebml::write_vint_fixed(&mut segment, 0, 8);
        sink.write(&segment).await?;
        let segment_data_start = sink.position();

        let header_position = sink.position();
        let placeholder = header_region([0; 3], 0.0, true);
        let info_offset = placeholder.info_offset;
        sink.write(&placeholder.bytes).await?;
        let info_position = header_position + info_offset as u64;
        let tracks_position = sink.position();
        sink.write(&tracks_element(&configs)?).await?;

        let tracks = configs
            .iter()
            .enumerate()
            .map(|(index, config)| TrackState {
                number: index as u64 + 1,
                timescale: config.encoder.timescale,
                is_video: config.kind() == TrackKind::Video,
                samples: 0,
            })
            .collect();
        Ok(Self {
            sink,
            tracks,
            segment_size_position,
            segment_data_start,
            header_position,
            header_length: placeholder.bytes.len(),
            info_position,
            tracks_position,
            cluster: None,
            cues: Vec::new(),
            last_timestamp: 0,
            end_ticks: 0.0,
            max_samples_per_track,
            finished: false,
        })
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Writes one sample as a SimpleBlock. A video random-access sample starts
    /// a new Cluster and gets a cue point, so every Cluster opens on one.
    pub async fn write_sample(&mut self, track: usize, sample: EncodedSample) -> Result<()> {
        if self.finished {
            return Err(invalid_state("cannot write a WebM after finish"));
        }
        let state = self
            .tracks
            .get(track)
            .ok_or_else(|| invalid("WebM track index is out of range"))?;
        if state.samples >= self.max_samples_per_track {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "WebM track sample limit exceeded",
            ));
        }
        if sample.duration == 0 {
            return Err(invalid("encoded sample duration must be nonzero"));
        }
        if sample.data.is_empty() {
            return Err(invalid("encoded sample data must be nonempty"));
        }
        let pts = u64::try_from(sample.pts)
            .map_err(|_| invalid("negative presentation timestamps are not supported"))?;
        let timestamp = to_ticks(pts, state.timescale)?;
        if timestamp < self.last_timestamp {
            return Err(invalid(
                "WebM samples must be written in nondecreasing presentation-time order across tracks",
            ));
        }
        let number = state.number;
        let starts_cluster = sample.is_sync && state.is_video;
        let end = (pts as f64 + f64::from(sample.duration)) * TICKS_PER_SECOND as f64
            / f64::from(state.timescale);

        let needs_cluster = match &self.cluster {
            None => true,
            Some(cluster) => {
                (starts_cluster && cluster.blocks > 0)
                    || timestamp - cluster.timestamp > i16::MAX as u64
            }
        };
        if needs_cluster {
            self.close_cluster().await?;
            self.open_cluster(timestamp).await?;
        }
        let cluster = self.cluster.as_mut().expect("opened above");
        let relative = i16::try_from(timestamp - cluster.timestamp).expect("checked above");
        if sample.is_sync {
            self.cues.push(CueRecord {
                time: timestamp,
                track: number,
                cluster_position: cluster.start - self.segment_data_start,
                relative_position: self.sink.position() - cluster.data_start,
            });
        }
        let mut block = Vec::with_capacity(16);
        write_id(&mut block, ebml::SIMPLE_BLOCK);
        ebml::write_vint(&mut block, 4 + sample.data.len() as u64);
        ebml::write_vint(&mut block, number);
        block.extend_from_slice(&relative.to_be_bytes());
        block.push(if sample.is_sync { 0x80 } else { 0x00 });
        self.sink.write(&block).await?;
        self.sink.write(&sample.data).await?;
        cluster.blocks += 1;
        self.last_timestamp = timestamp;
        self.end_ticks = self.end_ticks.max(end);
        self.tracks[track].samples += 1;
        Ok(())
    }

    pub async fn finish(mut self) -> Result<S> {
        if self.finished {
            return Err(invalid_state("WebM muxer was already finished"));
        }
        self.close_cluster().await?;
        let cues_position = (!self.cues.is_empty()).then(|| self.sink.position());
        if !self.cues.is_empty() {
            let cues = cues_element(&self.cues);
            self.sink.write(&cues).await?;
        }
        let end = self.sink.position();

        let relative = |position: u64| position - self.segment_data_start;
        let header = header_region(
            [
                relative(self.info_position),
                relative(self.tracks_position),
                cues_position.map_or(0, relative),
            ],
            self.end_ticks,
            cues_position.is_some(),
        );
        if header.bytes.len() != self.header_length {
            return Err(internal("WebM header region changed size"));
        }
        self.sink.seek(self.header_position).await?;
        self.sink.write(&header.bytes).await?;
        self.patch_size(self.segment_size_position, end - self.segment_data_start)
            .await?;
        self.sink.seek(end).await?;
        self.sink.flush().await?;
        self.finished = true;
        Ok(self.sink)
    }

    async fn open_cluster(&mut self, timestamp: u64) -> Result<()> {
        let start = self.sink.position();
        let mut header = Vec::with_capacity(20);
        write_id(&mut header, ebml::CLUSTER);
        ebml::write_vint_fixed(&mut header, 0, 8);
        let data_start = start + header.len() as u64;
        write_uint(&mut header, ebml::TIMESTAMP, timestamp);
        self.sink.write(&header).await?;
        self.cluster = Some(OpenCluster {
            start,
            data_start,
            timestamp,
            blocks: 0,
        });
        Ok(())
    }

    async fn close_cluster(&mut self) -> Result<()> {
        let Some(cluster) = self.cluster.take() else {
            return Ok(());
        };
        let end = self.sink.position();
        self.patch_size(cluster.data_start - 8, end - cluster.data_start)
            .await?;
        self.sink.seek(end).await
    }

    /// Rewrites the eight-byte size at `position`.
    async fn patch_size(&mut self, position: u64, size: u64) -> Result<()> {
        let mut bytes = Vec::with_capacity(8);
        ebml::write_vint_fixed(&mut bytes, size, 8);
        self.sink.seek(position).await?;
        self.sink.write(&bytes).await
    }
}

fn validate_track_config(config: &Mp4TrackConfig) -> Result<()> {
    if config.encoder.timescale == 0 {
        return Err(invalid("a WebM track timescale must be nonzero"));
    }
    match (config.encoder.codec, config.format) {
        (Codec::Av1, Mp4TrackFormat::Video(_)) => {}
        (Codec::Aac, _) | (_, Mp4TrackFormat::Audio { .. }) => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "WebM output is video-only: WebM permits only Vorbis or Opus audio, so an AAC track cannot be written to it",
            ));
        }
        (Codec::Hevc | Codec::H264, _) => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "WebM permits only VP8, VP9 or AV1 video; write HEVC or H.264 to MP4",
            ));
        }
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "codec is not a WebM output codec",
            ));
        }
    }
    codec_private(config).map(|_| ())
}

/// The track's `CodecID` and `CodecPrivate`. A V_VP8 or V_VP9 arm joins V_AV1
/// here once zvidlib encodes them (the sibling sub-issues of #523).
fn codec_private(config: &Mp4TrackConfig) -> Result<(&'static str, &[u8])> {
    match config.encoder.codec {
        Codec::Av1 => {
            let whole = &config.encoder.decoder_config;
            let declared = whole
                .get(..4)
                .map(|size| u32::from_be_bytes(size.try_into().expect("four bytes")));
            if whole.len() < 12
                || declared.and_then(|size| usize::try_from(size).ok()) != Some(whole.len())
                || &whole[4..8] != b"av1C"
            {
                return Err(invalid(
                    "AV1 codec configuration must be a complete av1C box",
                ));
            }
            // WebM's CodecPrivate is the AV1CodecConfigurationRecord itself.
            Ok(("V_AV1", &whole[8..]))
        }
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "codec is not a WebM output codec",
        )),
    }
}

fn to_ticks(pts: u64, timescale: u32) -> Result<u64> {
    let ticks = (u128::from(pts) * u128::from(TICKS_PER_SECOND) + u128::from(timescale) / 2)
        / u128::from(timescale);
    u64::try_from(ticks)
        .map_err(|_| Error::new(ErrorKind::ResourceLimit, "WebM timestamp overflow"))
}

fn ebml_header() -> Vec<u8> {
    let mut payload = Vec::new();
    write_uint(&mut payload, ebml::EBML_VERSION, 1);
    write_uint(&mut payload, ebml::EBML_READ_VERSION, 1);
    write_uint(&mut payload, ebml::EBML_MAX_ID_LENGTH, 4);
    write_uint(&mut payload, ebml::EBML_MAX_SIZE_LENGTH, 8);
    write_string(&mut payload, ebml::DOC_TYPE, "webm");
    write_uint(&mut payload, ebml::DOC_TYPE_VERSION, 4);
    // SimpleBlock needs a version 2 reader.
    write_uint(&mut payload, ebml::DOC_TYPE_READ_VERSION, 2);
    let mut output = Vec::new();
    write_element(&mut output, ebml::EBML, &payload);
    output
}

struct HeaderRegion {
    bytes: Vec<u8>,
    info_offset: usize,
}

/// The `SeekHead` and `Info` at the start of the Segment. Every value `finish`
/// learns late is written at a fixed width, so the region is the same length
/// as the placeholder written first and is rewritten in place. Without cues,
/// the Cues entry becomes a Void of the same length.
fn header_region(positions: [u64; 3], duration: f64, with_cues: bool) -> HeaderRegion {
    let mut seek_head = Vec::new();
    for (target, position) in SEEK_TARGETS.into_iter().zip(positions) {
        let mut seek = Vec::new();
        let mut id = Vec::new();
        write_id(&mut id, target);
        write_element(&mut seek, ebml::SEEK_ID, &id);
        ebml::write_uint_fixed(&mut seek, ebml::SEEK_POSITION, position);
        let mut entry = Vec::new();
        write_element(&mut entry, ebml::SEEK, &seek);
        if target == ebml::CUES && !with_cues {
            let filler = entry.len() - 2;
            entry.clear();
            write_element(&mut entry, ebml::VOID, &vec![0; filler]);
        }
        seek_head.extend_from_slice(&entry);
    }
    let mut info = Vec::new();
    write_uint(&mut info, ebml::TIMESTAMP_SCALE, TIMESTAMP_SCALE_NS);
    write_float(&mut info, ebml::DURATION, duration);
    write_string(&mut info, ebml::MUXING_APP, MUXING_APP);
    write_string(&mut info, ebml::WRITING_APP, MUXING_APP);

    let mut bytes = Vec::new();
    write_element(&mut bytes, ebml::SEEK_HEAD, &seek_head);
    let info_offset = bytes.len();
    write_element(&mut bytes, ebml::INFO, &info);
    HeaderRegion { bytes, info_offset }
}

fn tracks_element(configs: &[Mp4TrackConfig]) -> Result<Vec<u8>> {
    let mut tracks = Vec::new();
    for (index, config) in configs.iter().enumerate() {
        let number = index as u64 + 1;
        let (codec_id, private) = codec_private(config)?;
        let Mp4TrackFormat::Video(dimensions) = config.format else {
            return Err(internal("WebM track configuration was not validated"));
        };
        let mut entry = Vec::new();
        write_uint(&mut entry, ebml::TRACK_NUMBER, number);
        write_uint(&mut entry, ebml::TRACK_UID, number);
        write_uint(&mut entry, ebml::TRACK_TYPE, ebml::TRACK_TYPE_VIDEO);
        write_uint(&mut entry, ebml::FLAG_LACING, 0);
        write_string(&mut entry, ebml::CODEC_ID, codec_id);
        write_element(&mut entry, ebml::CODEC_PRIVATE, private);
        let mut video = Vec::new();
        write_uint(&mut video, ebml::PIXEL_WIDTH, u64::from(dimensions.width));
        write_uint(&mut video, ebml::PIXEL_HEIGHT, u64::from(dimensions.height));
        write_element(&mut entry, ebml::VIDEO, &video);
        write_element(&mut tracks, ebml::TRACK_ENTRY, &entry);
    }
    let mut output = Vec::new();
    write_element(&mut output, ebml::TRACKS, &tracks);
    Ok(output)
}

fn cues_element(cues: &[CueRecord]) -> Vec<u8> {
    let mut points = Vec::new();
    for cue in cues {
        let mut positions = Vec::new();
        write_uint(&mut positions, ebml::CUE_TRACK, cue.track);
        write_uint(
            &mut positions,
            ebml::CUE_CLUSTER_POSITION,
            cue.cluster_position,
        );
        write_uint(
            &mut positions,
            ebml::CUE_RELATIVE_POSITION,
            cue.relative_position,
        );
        let mut point = Vec::new();
        write_uint(&mut point, ebml::CUE_TIME, cue.time);
        write_element(&mut point, ebml::CUE_TRACK_POSITIONS, &positions);
        write_element(&mut points, ebml::CUE_POINT, &point);
    }
    let mut output = Vec::new();
    write_element(&mut output, ebml::CUES, &points);
    output
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn invalid_state(message: &str) -> Error {
    Error::new(ErrorKind::InvalidState, message)
}

fn internal(message: &str) -> Error {
    Error::new(ErrorKind::Internal, message)
}
