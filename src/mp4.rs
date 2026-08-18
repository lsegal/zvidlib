//! Deterministic seekable ISO BMFF/MP4 muxing.

use crate::codec::{AudioGapless, EncodedSample, EncoderConfig, SampleDependency, TrackKind};
use crate::io::ByteSink;
use crate::media::{Codec, VideoDimensions};
use crate::{Error, ErrorKind, Result};

const MOVIE_TIMESCALE: u32 = 1_000;

/// Media-specific fields used to create an MP4 sample entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mp4TrackFormat {
    Video(VideoDimensions),
    Audio { channels: u16 },
}

/// A track declaration passed to [`Mp4Muxer`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mp4TrackConfig {
    pub encoder: EncoderConfig,
    pub format: Mp4TrackFormat,
}

impl Mp4TrackConfig {
    pub fn kind(&self) -> TrackKind {
        match self.format {
            Mp4TrackFormat::Video(_) => TrackKind::Video,
            Mp4TrackFormat::Audio { .. } => TrackKind::Audio,
        }
    }
}

#[derive(Clone, Debug)]
struct SampleRecord {
    offset: u64,
    size: u32,
    dts: i64,
    pts: i64,
    duration: u32,
    is_sync: bool,
    dependency: SampleDependency,
}

#[derive(Clone, Debug)]
struct TrackState {
    id: u32,
    config: Mp4TrackConfig,
    samples: Vec<SampleRecord>,
    gapless: AudioGapless,
}

/// A seekable MP4 muxer that writes payload bytes immediately.
///
/// Each `write_sample` awaits the underlying sink before returning, so a slow
/// sink naturally applies backpressure without an unbounded packet queue.
pub struct Mp4Muxer<S> {
    sink: S,
    tracks: Vec<TrackState>,
    mdat_start: u64,
    max_samples_per_track: usize,
    finished: bool,
}

impl<S: ByteSink> Mp4Muxer<S> {
    pub async fn new(
        mut sink: S,
        configs: Vec<Mp4TrackConfig>,
        max_samples_per_track: usize,
    ) -> Result<Self> {
        if !sink.is_seekable() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "ordinary MP4 output requires a seekable sink",
            ));
        }
        if configs.is_empty() {
            return Err(invalid("an MP4 must declare at least one track"));
        }
        if max_samples_per_track == 0 {
            return Err(invalid("the MP4 sample limit must be nonzero"));
        }
        for config in &configs {
            validate_track_config(config)?;
        }

        sink.write(&ftyp_box()?).await?;
        let mdat_start = sink.position();
        let mut header = Vec::with_capacity(16);
        header.extend_from_slice(&1_u32.to_be_bytes());
        header.extend_from_slice(b"mdat");
        header.extend_from_slice(&0_u64.to_be_bytes());
        sink.write(&header).await?;

        let tracks = configs
            .into_iter()
            .enumerate()
            .map(|(index, config)| {
                Ok(TrackState {
                    id: u32::try_from(index + 1).map_err(|_| limit("too many MP4 tracks"))?,
                    config,
                    samples: Vec::new(),
                    gapless: AudioGapless::default(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            sink,
            tracks,
            mdat_start,
            max_samples_per_track,
            finished: false,
        })
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    pub async fn write_sample(&mut self, track: usize, sample: EncodedSample) -> Result<()> {
        if self.finished {
            return Err(invalid_state("cannot write an MP4 after finish"));
        }
        let state = self
            .tracks
            .get_mut(track)
            .ok_or_else(|| invalid("MP4 track index is out of range"))?;
        if state.samples.len() >= self.max_samples_per_track {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "MP4 track sample limit exceeded",
            ));
        }
        validate_sample(state, &sample)?;
        let size = u32::try_from(sample.data.len()).map_err(|_| {
            Error::new(
                ErrorKind::ResourceLimit,
                "encoded sample is too large for MP4",
            )
        })?;
        let record = SampleRecord {
            offset: self.sink.position(),
            size,
            dts: sample.dts,
            pts: sample.pts,
            duration: sample.duration,
            is_sync: sample.is_sync,
            dependency: sample.dependency,
        };
        self.sink.write(&sample.data).await?;
        state.samples.push(record);
        Ok(())
    }

    pub fn set_audio_gapless(&mut self, track: usize, gapless: AudioGapless) -> Result<()> {
        let state = self
            .tracks
            .get_mut(track)
            .ok_or_else(|| invalid("MP4 track index is out of range"))?;
        if state.config.kind() != TrackKind::Audio {
            return Err(invalid("gapless metadata belongs to an audio track"));
        }
        state.gapless = gapless;
        Ok(())
    }

    pub async fn finish(mut self) -> Result<S> {
        if self.finished {
            return Err(invalid_state("MP4 muxer was already finished"));
        }
        validate_gapless(&self.tracks)?;
        let payload_end = self.sink.position();
        let mdat_size = payload_end
            .checked_sub(self.mdat_start)
            .ok_or_else(|| internal("sink position moved before the MP4 media-data box"))?;
        self.sink.seek(self.mdat_start + 8).await?;
        self.sink.write(&mdat_size.to_be_bytes()).await?;
        self.sink.seek(payload_end).await?;
        let moov = moov_box(&self.tracks)?;
        self.sink.write(&moov).await?;
        self.sink.flush().await?;
        self.finished = true;
        Ok(self.sink)
    }
}

fn validate_track_config(config: &Mp4TrackConfig) -> Result<()> {
    if config.encoder.timescale == 0 {
        return Err(invalid("an MP4 track timescale must be nonzero"));
    }
    if config.encoder.decoder_config.len() < 8 {
        return Err(invalid(
            "codec configuration must contain a complete MP4 box",
        ));
    }
    let declared = u32::from_be_bytes(
        config.encoder.decoder_config[..4]
            .try_into()
            .map_err(|_| invalid("invalid codec configuration box"))?,
    );
    if usize::try_from(declared).ok() != Some(config.encoder.decoder_config.len()) {
        return Err(invalid("codec configuration box size is inconsistent"));
    }
    let expected_config = match config.encoder.codec {
        Codec::UncompressedVideo => {
            return Err(invalid(
                "the uncompressed conformance codec is not an MP4 output codec",
            ));
        }
        Codec::Hevc => b"hvcC",
        Codec::Av1 => b"av1C",
        Codec::Aac => b"esds",
    };
    if &config.encoder.decoder_config[4..8] != expected_config {
        return Err(invalid("codec configuration box type is incompatible"));
    }
    match (config.encoder.codec, config.format) {
        (Codec::Hevc | Codec::Av1, Mp4TrackFormat::Video(_))
        | (Codec::Aac, Mp4TrackFormat::Audio { .. }) => Ok(()),
        _ => Err(invalid("codec and MP4 track kind are incompatible")),
    }
}

fn validate_sample(track: &TrackState, sample: &EncodedSample) -> Result<()> {
    if sample.duration == 0 {
        return Err(invalid("encoded sample duration must be nonzero"));
    }
    if sample.data.is_empty() {
        return Err(invalid("encoded sample data must be nonempty"));
    }
    if sample.dependency.is_leading > 3
        || sample.dependency.depends_on > 3
        || sample.dependency.is_depended_on > 3
        || sample.dependency.has_redundancy > 3
    {
        return Err(invalid("sample dependency fields must fit in two bits"));
    }
    if sample.dts < 0 {
        return Err(invalid("negative decode timestamps are not supported"));
    }
    let offset = sample
        .pts
        .checked_sub(sample.dts)
        .ok_or_else(|| invalid("composition offset overflow"))?;
    i32::try_from(offset).map_err(|_| invalid("composition offset exceeds MP4 version 1 range"))?;
    if let Some(previous) = track.samples.last() {
        let expected = previous
            .dts
            .checked_add(i64::from(previous.duration))
            .ok_or_else(|| invalid("decode timestamp overflow"))?;
        if sample.dts != expected {
            return Err(invalid(
                "encoded samples must have contiguous decode timestamps",
            ));
        }
    } else if sample.dts != 0 {
        return Err(invalid("the first encoded sample must start at DTS zero"));
    }
    Ok(())
}

fn validate_gapless(tracks: &[TrackState]) -> Result<()> {
    for track in tracks {
        if track.config.kind() != TrackKind::Audio {
            continue;
        }
        let duration = track_duration(track)?;
        let trim = u64::from(track.gapless.priming) + u64::from(track.gapless.padding);
        if trim > duration {
            return Err(invalid("audio priming and padding exceed encoded duration"));
        }
    }
    Ok(())
}

fn ftyp_box() -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    payload.extend_from_slice(b"isom");
    payload.extend_from_slice(&0x200_u32.to_be_bytes());
    payload.extend_from_slice(b"isomiso6mp41");
    make_box(*b"ftyp", payload)
}

fn moov_box(tracks: &[TrackState]) -> Result<Vec<u8>> {
    let mut payload = mvhd_box(tracks)?;
    for track in tracks {
        payload.extend_from_slice(&trak_box(track)?);
    }
    make_box(*b"moov", payload)
}

fn mvhd_box(tracks: &[TrackState]) -> Result<Vec<u8>> {
    let duration = tracks.iter().try_fold(0_u64, |maximum, track| {
        Ok::<_, Error>(maximum.max(presentation_duration(track, MOVIE_TIMESCALE)?))
    })?;
    let mut body = Vec::new();
    body.extend_from_slice(&0_u32.to_be_bytes());
    body.extend_from_slice(&0_u32.to_be_bytes());
    body.extend_from_slice(&MOVIE_TIMESCALE.to_be_bytes());
    body.extend_from_slice(&u32_checked(duration, "movie duration")?.to_be_bytes());
    body.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
    body.extend_from_slice(&0x0100_u16.to_be_bytes());
    body.extend_from_slice(&[0; 10]);
    append_identity_matrix(&mut body);
    body.extend_from_slice(&[0; 24]);
    body.extend_from_slice(
        &u32::try_from(tracks.len() + 1)
            .map_err(|_| limit("too many MP4 tracks"))?
            .to_be_bytes(),
    );
    full_box(*b"mvhd", 0, 0, body)
}

fn trak_box(track: &TrackState) -> Result<Vec<u8>> {
    let mut payload = tkhd_box(track)?;
    if track.config.kind() == TrackKind::Audio
        && (track.gapless.priming != 0 || track.gapless.padding != 0)
    {
        payload.extend_from_slice(&edts_box(track)?);
    }
    payload.extend_from_slice(&mdia_box(track)?);
    make_box(*b"trak", payload)
}

fn tkhd_box(track: &TrackState) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    body.extend_from_slice(&0_u32.to_be_bytes());
    body.extend_from_slice(&0_u32.to_be_bytes());
    body.extend_from_slice(&track.id.to_be_bytes());
    body.extend_from_slice(&0_u32.to_be_bytes());
    let duration = presentation_duration(track, MOVIE_TIMESCALE)?;
    body.extend_from_slice(&u32_checked(duration, "track movie duration")?.to_be_bytes());
    body.extend_from_slice(&[0; 8]);
    body.extend_from_slice(&0_u16.to_be_bytes());
    body.extend_from_slice(&0_u16.to_be_bytes());
    let volume = if track.config.kind() == TrackKind::Audio {
        0x0100_u16
    } else {
        0
    };
    body.extend_from_slice(&volume.to_be_bytes());
    body.extend_from_slice(&0_u16.to_be_bytes());
    append_identity_matrix(&mut body);
    let (width, height) = match track.config.format {
        Mp4TrackFormat::Video(dimensions) => (dimensions.width, dimensions.height),
        Mp4TrackFormat::Audio { .. } => (0, 0),
    };
    body.extend_from_slice(
        &width
            .checked_shl(16)
            .ok_or_else(|| limit("MP4 width overflow"))?
            .to_be_bytes(),
    );
    body.extend_from_slice(
        &height
            .checked_shl(16)
            .ok_or_else(|| limit("MP4 height overflow"))?
            .to_be_bytes(),
    );
    full_box(*b"tkhd", 0, 7, body)
}

fn edts_box(track: &TrackState) -> Result<Vec<u8>> {
    let presentation = presentation_duration(track, MOVIE_TIMESCALE)?;
    let media_time = i32::try_from(track.gapless.priming)
        .map_err(|_| limit("audio priming exceeds MP4 edit range"))?;
    let mut body = Vec::new();
    body.extend_from_slice(&1_u32.to_be_bytes());
    body.extend_from_slice(&u32_checked(presentation, "audio edit duration")?.to_be_bytes());
    body.extend_from_slice(&media_time.to_be_bytes());
    body.extend_from_slice(&1_i16.to_be_bytes());
    body.extend_from_slice(&0_i16.to_be_bytes());
    make_box(*b"edts", full_box(*b"elst", 0, 0, body)?)
}

fn mdia_box(track: &TrackState) -> Result<Vec<u8>> {
    let mut payload = mdhd_box(track)?;
    payload.extend_from_slice(&hdlr_box(track.config.kind())?);
    payload.extend_from_slice(&minf_box(track)?);
    make_box(*b"mdia", payload)
}

fn mdhd_box(track: &TrackState) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    body.extend_from_slice(&0_u32.to_be_bytes());
    body.extend_from_slice(&0_u32.to_be_bytes());
    body.extend_from_slice(&track.config.encoder.timescale.to_be_bytes());
    body.extend_from_slice(&u32_checked(track_duration(track)?, "media duration")?.to_be_bytes());
    body.extend_from_slice(&0x55c4_u16.to_be_bytes());
    body.extend_from_slice(&0_u16.to_be_bytes());
    full_box(*b"mdhd", 0, 0, body)
}

fn hdlr_box(kind: TrackKind) -> Result<Vec<u8>> {
    let mut body = vec![0; 4];
    body.extend_from_slice(match kind {
        TrackKind::Video => b"vide",
        TrackKind::Audio => b"soun",
    });
    body.extend_from_slice(&[0; 12]);
    body.extend_from_slice(match kind {
        TrackKind::Video => b"zvidlib video\0",
        TrackKind::Audio => b"zvidlib audio\0",
    });
    full_box(*b"hdlr", 0, 0, body)
}

fn minf_box(track: &TrackState) -> Result<Vec<u8>> {
    let header = match track.config.kind() {
        TrackKind::Video => {
            let mut body = Vec::new();
            body.extend_from_slice(&0_u16.to_be_bytes());
            body.extend_from_slice(&[0; 6]);
            full_box(*b"vmhd", 0, 1, body)?
        }
        TrackKind::Audio => {
            let mut body = Vec::new();
            body.extend_from_slice(&0_i16.to_be_bytes());
            body.extend_from_slice(&0_u16.to_be_bytes());
            full_box(*b"smhd", 0, 0, body)?
        }
    };
    let mut payload = header;
    payload.extend_from_slice(&dinf_box()?);
    payload.extend_from_slice(&stbl_box(track)?);
    make_box(*b"minf", payload)
}

fn dinf_box() -> Result<Vec<u8>> {
    let url = full_box(*b"url ", 0, 1, Vec::new())?;
    let mut dref_body = 1_u32.to_be_bytes().to_vec();
    dref_body.extend_from_slice(&url);
    make_box(*b"dinf", full_box(*b"dref", 0, 0, dref_body)?)
}

fn stbl_box(track: &TrackState) -> Result<Vec<u8>> {
    let mut payload = stsd_box(track)?;
    payload.extend_from_slice(&stts_box(track)?);
    if track.samples.iter().any(|sample| sample.pts != sample.dts) {
        payload.extend_from_slice(&ctts_box(track)?);
    }
    payload.extend_from_slice(&stsc_box()?);
    payload.extend_from_slice(&stsz_box(track)?);
    payload.extend_from_slice(&co64_box(track)?);
    if track.config.kind() == TrackKind::Video {
        payload.extend_from_slice(&stss_box(track)?);
    }
    payload.extend_from_slice(&sdtp_box(track)?);
    make_box(*b"stbl", payload)
}

fn stsd_box(track: &TrackState) -> Result<Vec<u8>> {
    let entry = match track.config.format {
        Mp4TrackFormat::Video(dimensions) => video_sample_entry(track, dimensions)?,
        Mp4TrackFormat::Audio { channels } => audio_sample_entry(track, channels)?,
    };
    let mut body = 1_u32.to_be_bytes().to_vec();
    body.extend_from_slice(&entry);
    full_box(*b"stsd", 0, 0, body)
}

fn video_sample_entry(track: &TrackState, dimensions: VideoDimensions) -> Result<Vec<u8>> {
    let mut body = vec![0; 6];
    body.extend_from_slice(&1_u16.to_be_bytes());
    body.extend_from_slice(&[0; 16]);
    body.extend_from_slice(
        &u16::try_from(dimensions.width)
            .map_err(|_| limit("MP4 video width exceeds u16"))?
            .to_be_bytes(),
    );
    body.extend_from_slice(
        &u16::try_from(dimensions.height)
            .map_err(|_| limit("MP4 video height exceeds u16"))?
            .to_be_bytes(),
    );
    body.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    body.extend_from_slice(&0x0048_0000_u32.to_be_bytes());
    body.extend_from_slice(&0_u32.to_be_bytes());
    body.extend_from_slice(&1_u16.to_be_bytes());
    body.extend_from_slice(&[0; 32]);
    body.extend_from_slice(&0x0018_u16.to_be_bytes());
    body.extend_from_slice(&u16::MAX.to_be_bytes());
    body.extend_from_slice(&track.config.encoder.decoder_config);
    make_box(
        match track.config.encoder.codec {
            Codec::UncompressedVideo => {
                return Err(internal(
                    "uncompressed conformance codec used for an MP4 video sample entry",
                ));
            }
            Codec::Hevc => *b"hvc1",
            Codec::Av1 => *b"av01",
            Codec::Aac => return Err(internal("AAC used for a video sample entry")),
        },
        body,
    )
}

fn audio_sample_entry(track: &TrackState, channels: u16) -> Result<Vec<u8>> {
    let mut body = vec![0; 6];
    body.extend_from_slice(&1_u16.to_be_bytes());
    body.extend_from_slice(&[0; 8]);
    body.extend_from_slice(&channels.to_be_bytes());
    body.extend_from_slice(&16_u16.to_be_bytes());
    body.extend_from_slice(&0_u16.to_be_bytes());
    body.extend_from_slice(&0_u16.to_be_bytes());
    body.extend_from_slice(
        &track
            .config
            .encoder
            .timescale
            .checked_shl(16)
            .ok_or_else(|| limit("MP4 audio sample rate overflow"))?
            .to_be_bytes(),
    );
    body.extend_from_slice(&track.config.encoder.decoder_config);
    make_box(*b"mp4a", body)
}

fn stts_box(track: &TrackState) -> Result<Vec<u8>> {
    let groups = run_lengths(track.samples.iter().map(|sample| sample.duration));
    let mut body = u32_checked(groups.len() as u64, "stts entry count")?
        .to_be_bytes()
        .to_vec();
    for (count, duration) in groups {
        body.extend_from_slice(&count.to_be_bytes());
        body.extend_from_slice(&duration.to_be_bytes());
    }
    full_box(*b"stts", 0, 0, body)
}

fn ctts_box(track: &TrackState) -> Result<Vec<u8>> {
    let mut offsets = Vec::with_capacity(track.samples.len());
    for sample in &track.samples {
        offsets.push(
            i32::try_from(sample.pts - sample.dts)
                .map_err(|_| invalid("composition offset exceeds MP4 range"))?,
        );
    }
    let groups = run_lengths(offsets.into_iter());
    let mut body = u32_checked(groups.len() as u64, "ctts entry count")?
        .to_be_bytes()
        .to_vec();
    for (count, offset) in groups {
        body.extend_from_slice(&count.to_be_bytes());
        body.extend_from_slice(&offset.to_be_bytes());
    }
    full_box(*b"ctts", 1, 0, body)
}

fn stsc_box() -> Result<Vec<u8>> {
    let mut body = 1_u32.to_be_bytes().to_vec();
    body.extend_from_slice(&1_u32.to_be_bytes());
    body.extend_from_slice(&1_u32.to_be_bytes());
    body.extend_from_slice(&1_u32.to_be_bytes());
    full_box(*b"stsc", 0, 0, body)
}

fn stsz_box(track: &TrackState) -> Result<Vec<u8>> {
    let mut body = 0_u32.to_be_bytes().to_vec();
    body.extend_from_slice(&u32_checked(track.samples.len() as u64, "sample count")?.to_be_bytes());
    for sample in &track.samples {
        body.extend_from_slice(&sample.size.to_be_bytes());
    }
    full_box(*b"stsz", 0, 0, body)
}

fn co64_box(track: &TrackState) -> Result<Vec<u8>> {
    let mut body = u32_checked(track.samples.len() as u64, "chunk count")?
        .to_be_bytes()
        .to_vec();
    for sample in &track.samples {
        body.extend_from_slice(&sample.offset.to_be_bytes());
    }
    full_box(*b"co64", 0, 0, body)
}

fn stss_box(track: &TrackState) -> Result<Vec<u8>> {
    let sync: Vec<_> = track
        .samples
        .iter()
        .enumerate()
        .filter(|(_, sample)| sample.is_sync)
        .map(|(index, _)| u32::try_from(index + 1).map_err(|_| limit("sync sample index overflow")))
        .collect::<Result<_>>()?;
    let mut body = u32_checked(sync.len() as u64, "sync sample count")?
        .to_be_bytes()
        .to_vec();
    for index in sync {
        body.extend_from_slice(&index.to_be_bytes());
    }
    full_box(*b"stss", 0, 0, body)
}

fn sdtp_box(track: &TrackState) -> Result<Vec<u8>> {
    full_box(
        *b"sdtp",
        0,
        0,
        track
            .samples
            .iter()
            .map(|sample| sample.dependency.to_sdtp())
            .collect(),
    )
}

fn track_duration(track: &TrackState) -> Result<u64> {
    let Some(last) = track.samples.last() else {
        return Ok(0);
    };
    let end = last
        .dts
        .checked_add(i64::from(last.duration))
        .ok_or_else(|| limit("track duration overflow"))?;
    u64::try_from(end).map_err(|_| invalid("track duration is negative"))
}

fn presentation_duration(track: &TrackState, timescale: u32) -> Result<u64> {
    let mut duration = presentation_end(track)?;
    if track.config.kind() == TrackKind::Audio {
        duration = duration
            .checked_sub(u64::from(track.gapless.priming) + u64::from(track.gapless.padding))
            .ok_or_else(|| invalid("audio gapless trim exceeds its duration"))?;
    }
    scale_duration(duration, track.config.encoder.timescale, timescale)
}

fn presentation_end(track: &TrackState) -> Result<u64> {
    track.samples.iter().try_fold(0_u64, |maximum, sample| {
        let end = sample
            .pts
            .checked_add(i64::from(sample.duration))
            .ok_or_else(|| limit("presentation duration overflow"))?;
        let end = u64::try_from(end)
            .map_err(|_| invalid("encoded presentation ends before time zero"))?;
        Ok(maximum.max(end))
    })
}

fn scale_duration(value: u64, source: u32, target: u32) -> Result<u64> {
    let scaled = u128::from(value)
        .checked_mul(u128::from(target))
        .ok_or_else(|| limit("MP4 duration scaling overflow"))?
        / u128::from(source);
    u64::try_from(scaled).map_err(|_| limit("MP4 duration cannot be represented"))
}

fn run_lengths<T: Copy + Eq>(values: impl Iterator<Item = T>) -> Vec<(u32, T)> {
    let mut groups: Vec<(u32, T)> = Vec::new();
    for value in values {
        if let Some((count, previous)) = groups.last_mut()
            && *previous == value
            && *count < u32::MAX
        {
            *count += 1;
        } else {
            groups.push((1, value));
        }
    }
    groups
}

fn append_identity_matrix(output: &mut Vec<u8>) {
    output.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
    output.extend_from_slice(&0_u32.to_be_bytes());
    output.extend_from_slice(&0_u32.to_be_bytes());
    output.extend_from_slice(&0_u32.to_be_bytes());
    output.extend_from_slice(&0x0001_0000_u32.to_be_bytes());
    output.extend_from_slice(&0_u32.to_be_bytes());
    output.extend_from_slice(&0_u32.to_be_bytes());
    output.extend_from_slice(&0_u32.to_be_bytes());
    output.extend_from_slice(&0x4000_0000_u32.to_be_bytes());
}

fn full_box(kind: [u8; 4], version: u8, flags: u32, mut body: Vec<u8>) -> Result<Vec<u8>> {
    if flags > 0x00ff_ffff {
        return Err(internal("full-box flags exceed 24 bits"));
    }
    let mut payload = Vec::with_capacity(body.len() + 4);
    payload.push(version);
    payload.extend_from_slice(&flags.to_be_bytes()[1..]);
    payload.append(&mut body);
    make_box(kind, payload)
}

fn make_box(kind: [u8; 4], payload: Vec<u8>) -> Result<Vec<u8>> {
    let size = payload
        .len()
        .checked_add(8)
        .ok_or_else(|| limit("MP4 box size overflow"))?;
    let mut output = Vec::with_capacity(size);
    output.extend_from_slice(
        &u32::try_from(size)
            .map_err(|_| limit("MP4 box exceeds 32-bit size"))?
            .to_be_bytes(),
    );
    output.extend_from_slice(&kind);
    output.extend_from_slice(&payload);
    Ok(output)
}

fn u32_checked(value: u64, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| limit(format!("{label} exceeds MP4 version 0 range")))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn invalid_state(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidState, message)
}

fn limit(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

fn internal(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Internal, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::EncoderConfig;
    use crate::io::{IoFuture, NonSeekableSink};
    use crate::media::{Codec, VideoDimensions};
    use std::cell::Cell;
    use std::future::Future;
    use std::rc::Rc;
    use std::task::{Context, Poll, Waker};

    /// Wraps [`NonSeekableSink`] to record whether `write()` was ever called,
    /// so the test can prove rejection happens before any bytes are written.
    struct TrackingSink {
        inner: NonSeekableSink,
        wrote: Rc<Cell<bool>>,
    }

    impl ByteSink for TrackingSink {
        fn position(&self) -> u64 {
            self.inner.position()
        }

        fn is_seekable(&self) -> bool {
            self.inner.is_seekable()
        }

        fn write<'a>(&'a mut self, bytes: &'a [u8]) -> IoFuture<'a, ()> {
            self.wrote.set(true);
            self.inner.write(bytes)
        }

        fn seek<'a>(&'a mut self, position: u64) -> IoFuture<'a, ()> {
            self.inner.seek(position)
        }
    }

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        match Box::pin(future).as_mut().poll(&mut context) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("MP4 muxer test unexpectedly returned a pending future"),
        }
    }

    fn codec_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut output = u32::try_from(payload.len() + 8)
            .unwrap()
            .to_be_bytes()
            .to_vec();
        output.extend_from_slice(kind);
        output.extend_from_slice(payload);
        output
    }

    fn video_track_config() -> Mp4TrackConfig {
        Mp4TrackConfig {
            encoder: EncoderConfig {
                codec: Codec::Av1,
                timescale: 1_000,
                decoder_config: codec_box(b"av1C", &[0x81, 0, 0, 0]),
            },
            format: Mp4TrackFormat::Video(VideoDimensions {
                width: 16,
                height: 16,
            }),
        }
    }

    #[test]
    fn ordinary_mp4_creation_rejects_non_seekable_sink_before_writing() {
        let wrote = Rc::new(Cell::new(false));
        let sink = TrackingSink {
            inner: NonSeekableSink::default(),
            wrote: wrote.clone(),
        };
        let error = match block_on(Mp4Muxer::new(sink, vec![video_track_config()], 16)) {
            Ok(_) => panic!("expected a non-seekable sink to be rejected"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            !wrote.get(),
            "sink must not be written to before the seekability check"
        );
    }
}
