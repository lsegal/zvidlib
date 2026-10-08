//! The container-neutral track index every demuxer produces.
//!
//! [`crate::Mp4Demuxer`] and [`crate::WebmDemuxer`] both index a track as a
//! [`Track`]: its codec, timing and decode-order [`TrackSample`] table, with
//! its codec configuration in the box form an MP4 sample entry carries. So
//! everything built on an index - [`Track::to_encoded_video_samples`],
//! [`Track::to_encoded_audio_samples`], the on-demand [`TrackSampleProvider`]
//! and [`TrackAudioPacketProvider`], and [`crate::TrackSampleLoader`] - works
//! the same over either container.

use crate::audio::{AudioEdit, AudioPacketProvider, AudioTrackTiming, EncodedAudioSample};
use crate::codec::{EncodedVideoSample, SampleDependency, SampleProvider, TrackKind};
use crate::io::ByteSource;
use crate::media::{Codec, VideoDimensions};
use crate::mp4_demux::{AacTrackConfig, parse_aac_config};
use crate::opus::{OPUS_SAMPLE_RATE, OpusHead};
use crate::timeline::FrameIndex;
use crate::vorbis::VorbisConfig;
use crate::{Error, ErrorKind, Limits, Result};
use std::borrow::Cow;
use std::future::Future;

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
pub struct TrackSample {
    pub offset: u64,
    pub size: u32,
    pub dts: u64,
    pub pts: i64,
    pub duration: u32,
    pub dependency: SampleDependency,
    pub is_sync: bool,
}

/// Read-only metadata and indexes for one demuxed track, from any container.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Track {
    pub id: u32,
    pub kind: TrackKind,
    pub codec: Codec,
    pub timescale: u32,
    pub duration: u64,
    pub dimensions: Option<VideoDimensions>,
    pub channels: Option<u16>,
    pub sample_rate: Option<u32>,
    /// The track's ISO 639-2/T language code, such as `"eng"` or `"fra"`,
    /// from an MP4 track's `mdhd` box: `"und"` when the file marks the
    /// language undetermined, and `None` when the box holds no valid code. A
    /// WebM track's is always `None`.
    pub language: Option<String>,
    /// Complete codec configuration box, including its header, in the form an
    /// MP4 sample entry carries it whichever container the track came from.
    pub decoder_config: Vec<u8>,
    pub edits: Vec<EditMapping>,
    /// Samples in decode order.
    pub samples: Vec<TrackSample>,
    /// Decode-order indexes of the presentation frames, sorted by PTS, then
    /// DTS, then decode index. A decode-only sample, such as a hidden VP8
    /// frame stored as a WebM block of its own, is decoded on the way to the
    /// frames after it but is never presented, so it is not listed.
    pub presentation_order: Vec<usize>,
    /// The first byte of every sample, in decode order, of a WebM Vorbis
    /// track, which [`crate::WebmDemuxer`] reads anyway while scanning block
    /// headers. A Vorbis packet's mode number is in it, so every packet's
    /// decoded interval is known without reading the packets again; empty for
    /// every other track.
    pub vorbis_packet_heads: Vec<u8>,
}

impl Track {
    /// Returns a sample by its zero-based presentation index.
    pub fn presentation_sample(&self, index: usize) -> Option<&TrackSample> {
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
            .ok_or_else(|| invalid("sample index is out of range"))?;
        if destination.len() != sample.size as usize {
            return Err(invalid("sample destination size does not match the index"));
        }
        read_exact(source, sample.offset, destination).await
    }

    /// The presentation identity, in [`FrameIndex`] terms, of every
    /// decode-order sample: [`Self::presentation_order`]'s inverse, with a
    /// decode-only sample (one `presentation_order` does not list) given an
    /// identity past the last presentation frame so it never collides with
    /// a frame a caller asks for. Reads no sample data.
    pub(crate) fn presentation_index_by_decode(&self) -> Result<Vec<u64>> {
        let mut presentation_index_by_decode = vec![None; self.samples.len()];
        for (presentation_index, &decode_index) in self.presentation_order.iter().enumerate() {
            let presentation_index = u64::try_from(presentation_index)
                .map_err(|_| limit("presentation index overflow"))?;
            *presentation_index_by_decode
                .get_mut(decode_index)
                .ok_or_else(|| malformed("presentation order references a missing sample"))? =
                Some(presentation_index);
        }
        let mut next_decode_only = self.presentation_order.len() as u64;
        Ok(presentation_index_by_decode
            .into_iter()
            .map(|index| {
                index.unwrap_or_else(|| {
                    next_decode_only += 1;
                    next_decode_only - 1
                })
            })
            .collect())
    }

    /// Reads every decode-order sample and returns owned
    /// [`EncodedVideoSample`] values keyed by presentation order, ready for
    /// a [`crate::codec::VideoDecoderFactory`] backend.
    ///
    /// The zero-based position of each sample within
    /// [`Self::presentation_order`] becomes its `presentation_index`, and
    /// `is_sync` becomes `random_access`. A decode-only sample, which
    /// `presentation_order` does not list, takes an identity past the last
    /// presentation frame, in decode order, so it never collides with a frame
    /// a caller asks for. Total decoded sample bytes are bounded by
    /// `limits.max_allocation_bytes`.
    pub async fn to_encoded_video_samples<S: ByteSource + ?Sized>(
        &self,
        source: &S,
        limits: &Limits,
    ) -> Result<Vec<EncodedVideoSample>> {
        if self.kind != TrackKind::Video {
            return Err(unsupported("encoded video samples require a video track"));
        }
        let presentation_index_by_decode = self.presentation_index_by_decode()?;

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

    /// Parses the Opus identification header carried by this track's `dOps`
    /// box.
    pub fn opus_config(&self) -> Result<OpusHead> {
        if self.kind != TrackKind::Audio || self.codec != Codec::Opus {
            return Err(unsupported(
                "Opus configuration requires an Opus audio track",
            ));
        }
        let head = OpusHead::from_dops(&self.decoder_config)?;
        if self.channels != Some(u16::from(head.channels)) {
            return Err(malformed(
                "Opus dOps channel count disagrees with the sample entry",
            ));
        }
        Ok(head)
    }

    /// Parses the three Vorbis header packets a WebM Vorbis track carries as
    /// its `decoder_config` (see [`crate::WebmDemuxer`]); Vorbis has no MP4
    /// sample entry.
    pub fn vorbis_config(&self) -> Result<VorbisConfig> {
        if self.kind != TrackKind::Audio || self.codec != Codec::Vorbis {
            return Err(unsupported(
                "Vorbis configuration requires a Vorbis audio track",
            ));
        }
        VorbisConfig::from_codec_private(&self.decoder_config)
    }

    /// The sample rate this audio track decodes at: the AAC
    /// `AudioSpecificConfig`'s or the Vorbis identification header's, or 48
    /// kHz for Opus, which always decodes at that rate whatever its input was.
    pub fn audio_sample_rate(&self) -> Result<u32> {
        match self.codec {
            Codec::Aac => Ok(self.aac_config()?.sample_rate),
            Codec::Opus => {
                self.opus_config()?;
                Ok(OPUS_SAMPLE_RATE)
            }
            Codec::Vorbis => Ok(self.vorbis_config()?.sample_rate),
            _ => Err(unsupported(
                "audio packets require an AAC, Opus or Vorbis track",
            )),
        }
    }

    /// Reads every indexed audio packet - AAC access units, Opus or Vorbis
    /// packets - from its validated byte range.
    ///
    /// Packet intervals use the decoded PCM sample clock and remain contiguous
    /// even when the track timescale differs from the decoded sample rate.
    pub async fn to_encoded_audio_samples<S: ByteSource + ?Sized>(
        &self,
        source: &S,
        limits: &Limits,
    ) -> Result<Vec<EncodedAudioSample>> {
        let sample_rate = self.audio_sample_rate()?;
        // A Vorbis packet's length depends on the one before it, so the
        // intervals are assigned once every packet has been read.
        let vorbis = if self.codec == Codec::Vorbis {
            Some(self.vorbis_config()?)
        } else {
            None
        };
        // An AAC packet's interval comes from the sample table alone.
        let aac_ranges = if self.codec == Codec::Aac {
            Some(self.aac_decoded_ranges(sample_rate)?)
        } else {
            None
        };
        let mut total_bytes = 0_u64;
        let mut decoded_start = 0_u64;
        let mut packets = Vec::with_capacity(self.samples.len());
        for (decode_index, sample) in self.samples.iter().enumerate() {
            total_bytes = total_bytes
                .checked_add(u64::from(sample.size))
                .ok_or_else(|| limit("encoded audio allocation overflow"))?;
            if total_bytes > limits.max_allocation_bytes {
                return Err(limit(
                    "encoded audio samples exceed the configured allocation limit",
                ));
            }
            let mut data = vec![0_u8; sample.size as usize];
            self.read_sample_into(source, decode_index, &mut data)
                .await?;
            if vorbis.is_some() {
                packets.push(EncodedAudioSample {
                    decoded_range: crate::SampleRange::new(0, 0)?,
                    data,
                });
                continue;
            }
            if let Some(ranges) = &aac_ranges {
                packets.push(EncodedAudioSample {
                    decoded_range: ranges[decode_index],
                    data,
                });
                continue;
            }
            // An Opus packet's own table of contents says how long it
            // decodes. Muxers shorten the last sample's duration to trim
            // the stream's end - FFmpeg does - so the sample table cannot.
            let decoded_end = decoded_start
                .checked_add(u64::from(crate::opus::opus_packet_samples(&data)?))
                .ok_or_else(|| limit("audio track timing overflow"))?;
            if decoded_end <= decoded_start {
                return Err(malformed("audio packet has an empty decoded interval"));
            }
            packets.push(EncodedAudioSample {
                decoded_range: crate::SampleRange::new(decoded_start, decoded_end)?,
                data,
            });
            decoded_start = decoded_end;
        }
        if packets.is_empty() {
            return Err(malformed("audio track contains no samples"));
        }
        if let Some(vorbis) = vorbis {
            return vorbis.encoded_samples(packets.into_iter().map(|packet| packet.data).collect());
        }
        Ok(packets)
    }

    /// The decoded PCM interval of every AAC packet, from the sample table's
    /// durations alone. Reads no sample data.
    pub(crate) fn aac_decoded_ranges(&self, sample_rate: u32) -> Result<Vec<crate::SampleRange>> {
        self.table_decoded_ranges(&self.samples, sample_rate)
    }

    /// The decoded PCM interval of every Opus packet: the sample table's
    /// durations for all but the last, and `last_packet_samples`, read from
    /// the last packet's own table of contents, for that one. A muxer shortens
    /// the last sample's duration to trim the stream's end, as
    /// [`Self::to_encoded_audio_samples`] explains, so the table cannot give
    /// it. Reads no sample data.
    pub(crate) fn opus_decoded_ranges(
        &self,
        last_packet_samples: u32,
    ) -> Result<Vec<crate::SampleRange>> {
        let (_, leading) = self
            .samples
            .split_last()
            .ok_or_else(|| malformed("audio track contains no samples"))?;
        let mut ranges = self.table_decoded_ranges(leading, OPUS_SAMPLE_RATE)?;
        let start = ranges.last().map_or(0, |range| range.end);
        let end = start
            .checked_add(u64::from(last_packet_samples))
            .ok_or_else(|| limit("audio track timing overflow"))?;
        if end <= start {
            return Err(malformed("audio packet has an empty decoded interval"));
        }
        ranges.push(crate::SampleRange::new(start, end)?);
        Ok(ranges)
    }

    /// The decoded PCM interval of every Vorbis packet, the ones
    /// [`Self::to_encoded_audio_samples`] gives, from the block size each
    /// packet's first byte names. Reads no sample data: the first bytes are
    /// [`Self::vorbis_packet_heads`], which [`crate::WebmDemuxer`] records.
    /// Fails with [`ErrorKind::Unsupported`] for a track that does not record
    /// them.
    pub(crate) fn vorbis_decoded_ranges(&self) -> Result<Vec<crate::SampleRange>> {
        let config = self.vorbis_config()?;
        if self.samples.is_empty() {
            return Err(malformed("audio track contains no samples"));
        }
        if self.vorbis_packet_heads.len() != self.samples.len() {
            return Err(unsupported(
                "a Vorbis track's packet intervals need the first byte of every packet, which \
                 the WebM demuxer records",
            ));
        }
        let block_sizes = self
            .vorbis_packet_heads
            .iter()
            .map(|&head| config.packet_block_size(&[head]))
            .collect::<Result<Vec<_>>>()?;
        VorbisConfig::decoded_ranges(block_sizes)
    }

    /// The decoded PCM interval of each of `samples`, the track's leading
    /// samples, from their sample-table durations.
    fn table_decoded_ranges(
        &self,
        samples: &[TrackSample],
        sample_rate: u32,
    ) -> Result<Vec<crate::SampleRange>> {
        let mut decoded_start = 0_u64;
        let mut track_ticks = 0_u64;
        let mut ranges = Vec::with_capacity(samples.len());
        for sample in samples {
            track_ticks = track_ticks
                .checked_add(u64::from(sample.duration))
                .ok_or_else(|| limit("audio track timing overflow"))?;
            let decoded_end = scale_time(track_ticks, self.timescale, sample_rate)?;
            if decoded_end <= decoded_start {
                return Err(malformed("audio packet has an empty decoded interval"));
            }
            ranges.push(crate::SampleRange::new(decoded_start, decoded_end)?);
            decoded_start = decoded_end;
        }
        Ok(ranges)
    }

    /// Converts the track's MP4 edit-list timing to the presentation sample
    /// clock used by [`crate::AudioSampleReader`], including decoder priming
    /// and end padding.
    ///
    /// An Opus track without an edit list still has its `dOps` pre-skip
    /// trimmed as priming, and ends where its sample table does: the packets
    /// themselves can decode past that end, which is how a muxer that writes
    /// no edit list trims the stream.
    pub fn audio_timing(&self, movie_timescale: u32) -> Result<AudioTrackTiming> {
        let sample_rate = self.audio_sample_rate()?;
        let decoded_length = scale_time(self.duration, self.timescale, sample_rate)?;
        if self.edits.is_empty() {
            if self.codec != Codec::Opus {
                return Ok(AudioTrackTiming::default());
            }
            let priming = self.opus_config()?.pre_skip;
            let length = decoded_length
                .checked_sub(u64::from(priming))
                .filter(|&length| length > 0)
                .ok_or_else(|| malformed("Opus pre-skip covers the whole track"))?;
            return Ok(AudioTrackTiming {
                priming: u32::from(priming),
                padding: 0,
                track_offset: 0,
                edits: vec![AudioEdit {
                    presentation: crate::SampleRange::new(0, length)?,
                    media_start: Some(u64::from(priming)),
                }],
            });
        }
        let mut presentation_start = 0_u64;
        let mut edits = Vec::with_capacity(self.edits.len());
        let mut first_media = None;
        let mut last_media_end = 0_u64;
        for edit in &self.edits {
            if edit.media_rate_integer != 1 || edit.media_rate_fraction != 0 {
                return Err(unsupported(
                    "audio edit rates other than 1.0 are unsupported",
                ));
            }
            let length = scale_time(edit.segment_duration, movie_timescale, sample_rate)?;
            if length == 0 {
                return Err(malformed("audio edit has an empty presentation interval"));
            }
            let presentation_end = presentation_start
                .checked_add(length)
                .ok_or_else(|| limit("audio edit presentation overflow"))?;
            let media_start = if edit.media_time == -1 {
                None
            } else {
                if edit.media_time < 0 {
                    return Err(malformed("audio edit has an invalid negative media time"));
                }
                let start = scale_time(edit.media_time as u64, self.timescale, sample_rate)?;
                let end = start
                    .checked_add(length)
                    .ok_or_else(|| limit("audio edit media range overflow"))?;
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
            .map_err(|_| limit("audio priming exceeds the supported range"))?;
        let padding = u32::try_from(decoded_length.saturating_sub(last_media_end))
            .map_err(|_| limit("audio padding exceeds the supported range"))?;
        Ok(AudioTrackTiming {
            priming,
            padding,
            track_offset: 0,
            edits,
        })
    }
}

/// A [`SampleProvider`] that reads a video track's compressed bytes from a
/// [`ByteSource`] on demand, holding only `track`'s index (and whatever `S`
/// itself caches) rather than every sample's bytes.
///
/// Built from a container-neutral [`Track`] index, so it works over MP4 and
/// WebM alike.
///
/// `S::read_at`'s future must resolve the first time it is polled: this
/// provider is used from [`ExactFrameReader::get`][crate::codec::ExactFrameReader::get],
/// which is synchronous, so [`Self::read`] polls the read once and reports
/// [`ErrorKind::Unsupported`] rather than blocking if the source does not
/// finish immediately. [`MemorySource`][crate::io::MemorySource] and a
/// [`CachingByteSource`][crate::io::CachingByteSource] wrapping one both
/// satisfy this; a source that genuinely waits on I/O (a network fetch, for
/// example) needs its own asynchronous reader rather than this provider.
///
/// `S` must be `Send` so the provider - and a reader built over it - can be
/// moved to a decode thread, matching [`SampleProvider`]'s own requirement.
pub struct TrackSampleProvider<S> {
    track: Track,
    source: S,
    presentation_index_by_decode: Vec<u64>,
}

impl<S: ByteSource + Send> TrackSampleProvider<S> {
    /// Fails if `track` is not a video track; reads no sample data.
    pub fn new(track: Track, source: S) -> Result<Self> {
        if track.kind != TrackKind::Video {
            return Err(unsupported("a sample provider requires a video track"));
        }
        let presentation_index_by_decode = track.presentation_index_by_decode()?;
        Ok(Self {
            track,
            source,
            presentation_index_by_decode,
        })
    }

    pub fn track(&self) -> &Track {
        &self.track
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn into_source(self) -> S {
        self.source
    }
}

impl<S: ByteSource + Send> SampleProvider for TrackSampleProvider<S> {
    fn len(&self) -> usize {
        self.track.samples.len()
    }

    fn is_random_access(&self, decode_index: usize) -> bool {
        self.track.samples[decode_index].is_sync
    }

    fn presentation_index(&self, decode_index: usize) -> FrameIndex {
        FrameIndex(self.presentation_index_by_decode[decode_index])
    }

    fn read(&self, decode_index: usize) -> Result<Cow<'_, [u8]>> {
        let size = self
            .track
            .samples
            .get(decode_index)
            .ok_or_else(|| invalid("sample index is out of range"))?
            .size as usize;
        let mut data = vec![0_u8; size];
        poll_once(
            self.track
                .read_sample_into(&self.source, decode_index, &mut data),
        )
        .ok_or_else(|| {
            unsupported(
                "TrackSampleProvider requires a byte source whose reads complete \
                     synchronously; it cannot drive one that suspends",
            )
        })??;
        Ok(Cow::Owned(data))
    }
}

/// An [`AudioPacketProvider`] that reads an AAC or Vorbis track's compressed
/// packets from a [`ByteSource`] on demand, holding only `track`'s index (and
/// whatever `S` itself caches) rather than every packet's bytes.
///
/// The audio counterpart of [`TrackSampleProvider`], with the same requirement
/// that `S::read_at`'s future resolve the first time it is polled: a source
/// that suspends makes [`Self::read`] report [`ErrorKind::Unsupported`]
/// rather than block.
///
/// [`Self::new`] builds the whole index without reading any packet. An AAC
/// packet's decoded interval comes from the track's sample durations, and a
/// Vorbis packet's from the block sizes of it and the packet before it, which
/// [`Track::vorbis_packet_heads`] names. An Opus packet's interval is in its
/// own table-of-contents byte, which only the packet has, so Opus tracks use
/// [`crate::TrackSampleLoader::opus_packet_provider`] or
/// [`Track::to_encoded_audio_samples`]; [`Self::new`] rejects them.
pub struct TrackAudioPacketProvider<S> {
    track: Track,
    source: S,
    decoded_ranges: Vec<crate::SampleRange>,
}

impl<S: ByteSource + Send> TrackAudioPacketProvider<S> {
    /// Fails if `track` is not an AAC or Vorbis audio track; reads no packet
    /// data.
    pub fn new(track: Track, source: S) -> Result<Self> {
        if track.kind != TrackKind::Audio || !matches!(track.codec, Codec::Aac | Codec::Vorbis) {
            return Err(unsupported(
                "an on-demand audio packet provider requires an AAC or Vorbis audio track",
            ));
        }
        if track.samples.is_empty() {
            return Err(malformed("audio track contains no samples"));
        }
        let decoded_ranges = if track.codec == Codec::Vorbis {
            track.vorbis_decoded_ranges()?
        } else {
            track.aac_decoded_ranges(track.audio_sample_rate()?)?
        };
        Ok(Self {
            track,
            source,
            decoded_ranges,
        })
    }

    pub fn track(&self) -> &Track {
        &self.track
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn into_source(self) -> S {
        self.source
    }
}

impl<S: ByteSource + Send> AudioPacketProvider for TrackAudioPacketProvider<S> {
    fn len(&self) -> usize {
        self.decoded_ranges.len()
    }

    fn decoded_range(&self, index: usize) -> crate::SampleRange {
        self.decoded_ranges[index]
    }

    fn read(&self, index: usize) -> Result<Cow<'_, [u8]>> {
        let size = self
            .track
            .samples
            .get(index)
            .ok_or_else(|| invalid("sample index is out of range"))?
            .size as usize;
        let mut data = vec![0_u8; size];
        poll_once(self.track.read_sample_into(&self.source, index, &mut data)).ok_or_else(
            || {
                unsupported(
                    "TrackAudioPacketProvider requires a byte source whose reads complete \
                     synchronously; it cannot drive one that suspends",
                )
            },
        )??;
        Ok(Cow::Owned(data))
    }
}

/// Polls a future once and returns its value if it was ready, without
/// blocking or spinning. Every `ByteSource` this crate ships resolves on its
/// first poll; this is how a synchronous caller, such as
/// [`TrackSampleProvider::read`], drives one without an async runtime.
fn poll_once<T>(future: impl Future<Output = T>) -> Option<T> {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(value) => Some(value),
        std::task::Poll::Pending => None,
    }
}

pub(crate) async fn read_exact<S: ByteSource + ?Sized>(
    source: &S,
    mut offset: u64,
    mut output: &mut [u8],
) -> Result<()> {
    while !output.is_empty() {
        let read = source.read_at(offset, output).await?;
        if read == 0 {
            return Err(malformed("unexpected end of media source"));
        }
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| malformed("read offset overflow"))?;
        output = &mut output[read..];
    }
    Ok(())
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
