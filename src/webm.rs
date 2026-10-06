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
//! encodes VP8, VP9 and AV1 video and Opus and Vorbis audio, so those are what
//! it writes: an AAC track is refused rather than written into a file no WebM
//! player would accept.
//!
//! An audio track's encoder delay is written as its `CodecDelay` and its end
//! padding as its last block's `DiscardPadding`, so a reader trims the decoded
//! stream to exactly the samples that were encoded. The padding is only known
//! once the encoder drains, after its last packets are handed over, so each
//! audio track's newest block is held back until a later block or `finish`
//! writes it.

use crate::codec::{AudioGapless, EncodedSample, TrackKind};
use crate::ebml::{
    self, write_element, write_float, write_id, write_int, write_string, write_uint,
};
use crate::io::ByteSink;
use crate::media::Codec;
use crate::mp4::{Mp4TrackConfig, Mp4TrackFormat};
use crate::opus::OpusHead;
use crate::vorbis::VorbisConfig;
use crate::{Error, ErrorKind, Result, Vp9CodecConfig};

/// One tick of every block timestamp written: a millisecond, Matroska's
/// default and the scale every WebM player expects.
const TIMESTAMP_SCALE_NS: u64 = 1_000_000;
const TICKS_PER_SECOND: u64 = 1_000;
/// Track numbers are written as one-byte block headers.
const MAX_TRACKS: usize = 126;
/// The Seek entries `finish` fills in, in `SeekHead` order.
const SEEK_TARGETS: [u32; 3] = [ebml::INFO, ebml::TRACKS, ebml::CUES];
const MUXING_APP: &str = concat!("zvidlib ", env!("CARGO_PKG_VERSION"));
const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;
/// How much Opus to decode ahead of a seek target, the `SeekPreRoll` the
/// Matroska Opus mapping asks for.
const OPUS_SEEK_PRE_ROLL_NS: u64 = 80_000_000;
/// How long a Cluster of a file with no video runs before the next one opens.
/// A video file opens one at each random-access sample instead.
const AUDIO_CLUSTER_TICKS: u64 = 5_000;

struct TrackState {
    number: u64,
    timescale: u32,
    is_video: bool,
    samples: usize,
    audio: Option<AudioTrackState>,
}

struct AudioTrackState {
    /// The decoded samples the track's `CodecDelay` declares.
    codec_delay: u32,
    padding: u32,
    /// The track's newest block, not yet written.
    held: Option<HeldBlock>,
    /// Whether a block of the track has been written.
    wrote_block: bool,
    /// Whether [`WebmMuxer::set_audio_gapless`] has ended the track, so the
    /// held block is its last.
    complete: bool,
}

struct HeldBlock {
    timestamp: u64,
    data: Vec<u8>,
    /// End of the block's samples, in the track's timescale.
    end: u64,
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
    has_video: bool,
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

        let tracks: Vec<TrackState> = configs
            .iter()
            .enumerate()
            .map(|(index, config)| {
                Ok(TrackState {
                    number: index as u64 + 1,
                    timescale: config.encoder.timescale,
                    is_video: config.kind() == TrackKind::Video,
                    samples: 0,
                    audio: match config.format {
                        Mp4TrackFormat::Video(_) => None,
                        Mp4TrackFormat::Audio { .. } => Some(AudioTrackState {
                            codec_delay: codec_delay(config)?,
                            padding: 0,
                            held: None,
                            wrote_block: false,
                            complete: false,
                        }),
                    },
                })
            })
            .collect::<Result<_>>()?;
        let has_video = tracks.iter().any(|track| track.is_video);
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
            has_video,
            finished: false,
        })
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Writes one sample. A video random-access sample starts a new Cluster
    /// and gets a cue point, so every Cluster opens on one; in a file without
    /// video, a Cluster opens every few seconds and its first audio block is
    /// cued instead. An audio sample is held back until a later sample or
    /// [`Self::finish`] writes it, so its track's last block can carry the
    /// end padding [`Self::set_audio_gapless`] declares.
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
        if state.audio.as_ref().is_some_and(|audio| audio.complete) {
            return Err(invalid_state(
                "a WebM audio track's gapless trim was set, which ends the track",
            ));
        }
        // A Vorbis stream's first packet decodes no samples, so an audio
        // sample may last no time at all.
        if sample.duration == 0 && state.audio.is_none() {
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
        self.flush_held(Some(timestamp)).await?;
        let state = &mut self.tracks[track];
        state.samples += 1;
        let number = state.number;
        if let Some(audio) = state.audio.as_mut() {
            audio.held = Some(HeldBlock {
                timestamp,
                data: sample.data,
                end: pts + u64::from(sample.duration),
            });
            self.last_timestamp = timestamp;
            return Ok(());
        }
        let end = (pts as f64 + f64::from(sample.duration)) * TICKS_PER_SECOND as f64
            / f64::from(state.timescale);
        self.write_block(number, timestamp, &sample.data, sample.is_sync, true, None)
            .await?;
        self.end_ticks = self.end_ticks.max(end);
        Ok(())
    }

    /// Declares an audio track's gapless trim and ends the track. The
    /// priming must be the delay its codec configuration already declares - an
    /// Opus stream's pre-skip, zero for Vorbis - and the padding is written as
    /// the track's last block's `DiscardPadding`, so it has to be set after the
    /// track's last sample and before a sample of another track is written past
    /// that sample's time.
    pub fn set_audio_gapless(&mut self, track: usize, gapless: AudioGapless) -> Result<()> {
        let state = self
            .tracks
            .get_mut(track)
            .ok_or_else(|| invalid("WebM track index is out of range"))?;
        let audio = state
            .audio
            .as_mut()
            .ok_or_else(|| invalid("gapless metadata belongs to an audio track"))?;
        if gapless.priming != audio.codec_delay {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "WebM audio priming must equal the codec's own delay, which its CodecDelay declares",
            ));
        }
        if gapless.padding != 0 && audio.held.is_none() && audio.wrote_block {
            return Err(invalid_state(
                "a WebM audio track's end padding must be set before later samples write its last block",
            ));
        }
        audio.padding = gapless.padding;
        audio.complete = true;
        Ok(())
    }

    pub async fn finish(mut self) -> Result<S> {
        if self.finished {
            return Err(invalid_state("WebM muxer was already finished"));
        }
        self.flush_held(None).await?;
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

    /// Writes every held audio block timestamped at or before `before`, or
    /// every one when it is `None` - then each is its track's last, and
    /// carries the track's end padding.
    async fn flush_held(&mut self, before: Option<u64>) -> Result<()> {
        loop {
            let next = self
                .tracks
                .iter()
                .enumerate()
                .filter_map(|(index, track)| {
                    let held = track.audio.as_ref()?.held.as_ref()?;
                    before
                        .is_none_or(|before| held.timestamp <= before)
                        .then_some((held.timestamp, index))
                })
                .min();
            let Some((_, index)) = next else {
                return Ok(());
            };
            let state = &mut self.tracks[index];
            let number = state.number;
            let timescale = state.timescale;
            let audio = state.audio.as_mut().expect("held blocks are audio");
            let held = audio.held.take().expect("found above");
            audio.wrote_block = true;
            let last = before.is_none() || audio.complete;
            let padding = if last { audio.padding } else { 0 };
            let discard_padding = (padding > 0)
                .then(|| {
                    let nanoseconds = u128::from(padding) * u128::from(NANOSECONDS_PER_SECOND)
                        / u128::from(timescale);
                    i64::try_from(nanoseconds)
                        .map_err(|_| Error::new(ErrorKind::ResourceLimit, "WebM padding overflow"))
                })
                .transpose()?;
            let presented_end = held
                .end
                .saturating_sub(u64::from(audio.codec_delay) + u64::from(padding));
            let end = presented_end as f64 * TICKS_PER_SECOND as f64 / f64::from(timescale);
            self.write_block(
                number,
                held.timestamp,
                &held.data,
                true,
                false,
                discard_padding,
            )
            .await?;
            self.end_ticks = self.end_ticks.max(end);
        }
    }

    /// Writes one block, opening a Cluster first when it needs one.
    async fn write_block(
        &mut self,
        number: u64,
        timestamp: u64,
        data: &[u8],
        is_sync: bool,
        is_video: bool,
        discard_padding_ns: Option<i64>,
    ) -> Result<()> {
        let starts_cluster = is_sync && is_video;
        let needs_cluster = match &self.cluster {
            None => true,
            Some(cluster) => {
                (starts_cluster && cluster.blocks > 0)
                    || timestamp - cluster.timestamp > i16::MAX as u64
                    || (!self.has_video && timestamp - cluster.timestamp >= AUDIO_CLUSTER_TICKS)
            }
        };
        if needs_cluster {
            self.close_cluster().await?;
            self.open_cluster(timestamp).await?;
        }
        let cluster = self.cluster.as_mut().expect("opened above");
        let relative = i16::try_from(timestamp - cluster.timestamp).expect("checked above");
        // Every video random-access sample is cued. Audio is cued once a
        // Cluster, at its first block, in a file with no video to cue.
        let cued = if is_video {
            is_sync
        } else {
            !self.has_video && cluster.blocks == 0
        };
        if cued {
            self.cues.push(CueRecord {
                time: timestamp,
                track: number,
                cluster_position: cluster.start - self.segment_data_start,
                relative_position: self.sink.position() - cluster.data_start,
            });
        }
        let mut block = Vec::with_capacity(32);
        match discard_padding_ns {
            None => {
                write_id(&mut block, ebml::SIMPLE_BLOCK);
                ebml::write_vint(&mut block, 4 + data.len() as u64);
                ebml::write_vint(&mut block, number);
                block.extend_from_slice(&relative.to_be_bytes());
                block.push(if is_sync { 0x80 } else { 0x00 });
                self.sink.write(&block).await?;
                self.sink.write(data).await?;
            }
            // `DiscardPadding` belongs to a BlockGroup, whose Block carries
            // no keyframe flag.
            Some(padding) => {
                let mut padding_element = Vec::new();
                write_int(&mut padding_element, ebml::DISCARD_PADDING, padding);
                let mut block_header = Vec::new();
                write_id(&mut block_header, ebml::BLOCK);
                ebml::write_vint(&mut block_header, 4 + data.len() as u64);
                ebml::write_vint(&mut block_header, number);
                block_header.extend_from_slice(&relative.to_be_bytes());
                block_header.push(0x00);
                let group_length = (block_header.len() + data.len() + padding_element.len()) as u64;
                write_id(&mut block, ebml::BLOCK_GROUP);
                ebml::write_vint(&mut block, group_length);
                block.extend_from_slice(&block_header);
                self.sink.write(&block).await?;
                self.sink.write(data).await?;
                self.sink.write(&padding_element).await?;
            }
        }
        cluster.blocks += 1;
        self.last_timestamp = self.last_timestamp.max(timestamp);
        Ok(())
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
        (Codec::Av1 | Codec::Vp8 | Codec::Vp9, Mp4TrackFormat::Video(_)) => {}
        (Codec::Opus | Codec::Vorbis, Mp4TrackFormat::Audio { channels }) => {
            let declared = match config.encoder.codec {
                Codec::Opus => {
                    u16::from(OpusHead::from_dops(&config.encoder.decoder_config)?.channels)
                }
                _ => u16::from(
                    VorbisConfig::from_codec_private(&config.encoder.decoder_config)?.channels,
                ),
            };
            if declared != channels {
                return Err(invalid(
                    "the audio codec configuration's channel count disagrees with the track format",
                ));
            }
        }
        (Codec::Aac, _) | (_, Mp4TrackFormat::Audio { .. }) => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "WebM permits only Vorbis or Opus audio; write AAC to MP4",
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

/// The decoded samples a track's `CodecDelay` declares: an Opus stream's
/// pre-skip, and none for any other codec.
fn codec_delay(config: &Mp4TrackConfig) -> Result<u32> {
    match config.encoder.codec {
        Codec::Opus => Ok(u32::from(
            OpusHead::from_dops(&config.encoder.decoder_config)?.pre_skip,
        )),
        _ => Ok(0),
    }
}

/// The track's `CodecID` and `CodecPrivate`.
fn codec_private(config: &Mp4TrackConfig) -> Result<(&'static str, Vec<u8>)> {
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
            Ok(("V_AV1", whole[8..].to_vec()))
        }
        // VP8 carries everything a decoder needs in its key frames, so the
        // Matroska VP8 mapping has no CodecPrivate.
        Codec::Vp8 => {
            if !config.encoder.decoder_config.is_empty() {
                return Err(invalid("VP8 has no codec configuration record"));
            }
            Ok(("V_VP8", Vec::new()))
        }
        Codec::Vp9 => {
            // The WebM VP9 codec mapping's CodecPrivate is a list of
            // (ID, length, value) features: profile (1), level (2), bit
            // depth (3) and chroma subsampling (4), all from the `vpcC`.
            let vpcc = Vp9CodecConfig::parse_vpcc(&config.encoder.decoder_config)
                .map_err(|_| invalid("VP9 codec configuration must be a complete vpcC box"))?;
            let mut features = Vec::with_capacity(12);
            for (id, value) in [
                (1, vpcc.profile),
                (2, vpcc.level),
                (3, vpcc.bit_depth),
                (4, vpcc.chroma_subsampling),
            ] {
                features.extend_from_slice(&[id, 1, value]);
            }
            Ok(("V_VP9", features))
        }
        // The Matroska Opus mapping's CodecPrivate is the RFC 7845 OpusHead.
        Codec::Opus => Ok((
            "A_OPUS",
            OpusHead::from_dops(&config.encoder.decoder_config)?.to_identification_header(),
        )),
        // A Vorbis encoder's configuration is already the Xiph-laced
        // headers Matroska carries.
        Codec::Vorbis => Ok(("A_VORBIS", config.encoder.decoder_config.clone())),
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
        let mut entry = Vec::new();
        write_uint(&mut entry, ebml::TRACK_NUMBER, number);
        write_uint(&mut entry, ebml::TRACK_UID, number);
        match config.format {
            Mp4TrackFormat::Video(dimensions) => {
                write_uint(&mut entry, ebml::TRACK_TYPE, ebml::TRACK_TYPE_VIDEO);
                write_uint(&mut entry, ebml::FLAG_LACING, 0);
                write_string(&mut entry, ebml::CODEC_ID, codec_id);
                if !private.is_empty() {
                    write_element(&mut entry, ebml::CODEC_PRIVATE, &private);
                }
                let mut video = Vec::new();
                write_uint(&mut video, ebml::PIXEL_WIDTH, u64::from(dimensions.width));
                write_uint(&mut video, ebml::PIXEL_HEIGHT, u64::from(dimensions.height));
                write_element(&mut entry, ebml::VIDEO, &video);
            }
            Mp4TrackFormat::Audio { channels } => {
                write_uint(&mut entry, ebml::TRACK_TYPE, ebml::TRACK_TYPE_AUDIO);
                write_uint(&mut entry, ebml::FLAG_LACING, 0);
                write_string(&mut entry, ebml::CODEC_ID, codec_id);
                write_element(&mut entry, ebml::CODEC_PRIVATE, &private);
                let delay = codec_delay(config)?;
                if config.encoder.codec == Codec::Opus {
                    // Opus's delay and preroll are on its 48 kHz clock.
                    let delay_ns = u64::from(delay) * NANOSECONDS_PER_SECOND
                        / u64::from(crate::OPUS_SAMPLE_RATE);
                    write_uint(&mut entry, ebml::CODEC_DELAY, delay_ns);
                    write_uint(&mut entry, ebml::SEEK_PRE_ROLL, OPUS_SEEK_PRE_ROLL_NS);
                }
                let mut audio = Vec::new();
                write_float(
                    &mut audio,
                    ebml::SAMPLING_FREQUENCY,
                    f64::from(config.encoder.timescale),
                );
                write_uint(&mut audio, ebml::CHANNELS, u64::from(channels));
                write_element(&mut entry, ebml::AUDIO, &audio);
            }
        }
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
