//! Exact audio sample reads with gapless and edit-list timeline mapping.
//!
//! [`AudioSampleReader`] is codec-neutral: it maps presentation sample ranges
//! through priming, padding and edits onto a track's packets and asks an
//! [`AudioDecoder`] for exactly the packets it needs. AAC, Opus and Vorbis
//! tracks all read through it.

use crate::{
    AudioBuffer, CancellationToken, Error, ErrorKind, FrameIndex, Limits, Result, SampleRange,
    Timeline,
};
use std::borrow::Cow;
use std::collections::BTreeMap;

/// One compressed audio packet - an AAC access unit, an Opus packet or a
/// Vorbis packet - and the decoded interval it covers.
///
/// The interval may be empty: a Vorbis stream's first audio packet only primes
/// the decoder's overlap and decodes no samples.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedAudioSample {
    pub decoded_range: SampleRange,
    pub data: Vec<u8>,
}

/// A stateful audio decoder. Output must use the packet's decoded sample
/// clock: the buffer for a packet covers exactly its `decoded_range`, which is
/// empty for a packet that decodes nothing.
pub trait AudioDecoder {
    fn decode(
        &mut self,
        sample: &EncodedAudioSample,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer>;
    fn reset(&mut self) -> Result<()>;
}

/// The name [`AudioDecoder`] had while AAC was the only audio codec.
pub use self::AudioDecoder as AacDecoder;

/// Supplies [`AudioSampleReader`] with compressed audio packets by index, so
/// a reader does not have to own every packet of a track up front.
///
/// An owned `Vec<EncodedAudioSample>` implements this directly and is what
/// [`AudioSampleReader::new`] wraps, so that constructor and its callers are
/// unchanged. [`AudioSampleReader::from_provider`] accepts any other
/// implementation, such as one that reads through a container track's index
/// and a [`crate::io::ByteSource`] on demand.
///
/// `len` and `decoded_range` answer from the index alone and must not read
/// packet data; only [`Self::read`] may do that, and only for the one packet
/// it is asked for. A packet's decoded interval must be knowable without
/// reading any packet's bytes: an AAC packet's comes from the track's sample
/// durations, which this holds for. An Opus packet's own table of contents
/// gives its interval, and a Vorbis packet's depends on the block size of the
/// packet before it, so neither decides its boundary without the data a
/// provider is meant not to hold. `zvidlib-container`'s
/// `Mp4AudioPacketProvider` is the on-demand provider for AAC tracks;
/// `Mp4Track::to_encoded_audio_samples` remains how Opus and Vorbis tracks
/// are read.
///
/// Implementations used from a decode thread must be `Send`.
pub trait AudioPacketProvider: Send {
    /// The number of packets.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn decoded_range(&self, index: usize) -> SampleRange;
    /// Returns the compressed bytes of one packet.
    fn read(&self, index: usize) -> Result<Cow<'_, [u8]>>;
}

impl AudioPacketProvider for Vec<EncodedAudioSample> {
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn decoded_range(&self, index: usize) -> SampleRange {
        self[index].decoded_range
    }

    fn read(&self, index: usize) -> Result<Cow<'_, [u8]>> {
        Ok(Cow::Borrowed(&self[index].data))
    }
}

impl<D: AudioDecoder + ?Sized> AudioDecoder for Box<D> {
    fn decode(
        &mut self,
        sample: &EncodedAudioSample,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        (**self).decode(sample, cancellation)
    }

    fn reset(&mut self) -> Result<()> {
        (**self).reset()
    }
}

/// One MP4 edit in the presentation sample clock.
///
/// `media_start == None` is an empty edit and produces silence. A media edit's
/// source interval has the same length as its presentation interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioEdit {
    pub presentation: SampleRange,
    pub media_start: Option<u64>,
}

/// Preserved timing metadata needed to expose gapless presentation samples.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AudioTrackTiming {
    pub priming: u32,
    pub padding: u32,
    /// Positive values delay the track; negative values trim its media start.
    pub track_offset: i64,
    pub edits: Vec<AudioEdit>,
}

/// Exact presentation-range access over a track's sequential audio packets.
pub struct AudioSampleReader<D> {
    decoder: D,
    packets: Box<dyn AudioPacketProvider>,
    decoded: BTreeMap<u64, AudioBuffer>,
    /// Inclusive packet-index bounds of the contiguous run held in `decoded`.
    ///
    /// The run is exactly what the decoder has produced since its last reset,
    /// minus any packets evicted from its front, so `resident.1 + 1` is the
    /// packet the decoder is positioned to decode next without a reset.
    resident: Option<(usize, usize)>,
    resident_bytes: u64,
    timing: AudioTrackTiming,
    sample_rate: u32,
    channels: u16,
    decoded_length: u64,
    presentation_length: u64,
    preroll_packets: usize,
    limits: Limits,
}

/// The name [`AudioSampleReader`] had while AAC was the only audio codec.
pub type AacSampleReader<D> = AudioSampleReader<D>;

impl<D: AudioDecoder> AudioSampleReader<D> {
    /// `preroll_packets` is how many packets before the first one a request
    /// needs are decoded first after a reset, so the decoder's state has
    /// converged by the time it reaches the samples that are returned.
    pub fn new(
        decoder: D,
        packets: Vec<EncodedAudioSample>,
        sample_rate: u32,
        channels: u16,
        timing: AudioTrackTiming,
        preroll_packets: usize,
        limits: Limits,
    ) -> Result<Self> {
        Self::from_provider(
            decoder,
            Box::new(packets),
            sample_rate,
            channels,
            timing,
            preroll_packets,
            limits,
        )
    }

    /// Builds a reader over any [`AudioPacketProvider`], such as one that
    /// reads compressed bytes from a container track and a
    /// [`crate::io::ByteSource`] on demand instead of owning every packet.
    ///
    /// Behaves exactly as [`Self::new`], which is this constructor with an
    /// owned `Vec<EncodedAudioSample>` as the provider.
    pub fn from_provider(
        decoder: D,
        packets: Box<dyn AudioPacketProvider>,
        sample_rate: u32,
        channels: u16,
        timing: AudioTrackTiming,
        preroll_packets: usize,
        limits: Limits,
    ) -> Result<Self> {
        if packets.is_empty() {
            return Err(invalid("an audio reader requires at least one packet"));
        }
        if sample_rate == 0 || sample_rate > limits.max_sample_rate {
            return Err(limit("audio sample rate is outside configured limits"));
        }
        if channels == 0 || channels > limits.max_audio_channels {
            return Err(limit("audio channel count is outside configured limits"));
        }
        let mut expected = 0;
        for index in 0..packets.len() {
            let range = packets.decoded_range(index);
            if range.start != expected {
                return Err(invalid("audio packet sample intervals must be contiguous"));
            }
            expected = range.end;
        }
        if expected == 0 {
            return Err(invalid("audio packets decode no samples"));
        }
        let trimmed = expected
            .checked_sub(u64::from(timing.priming) + u64::from(timing.padding))
            .ok_or_else(|| invalid("audio priming and padding exceed decoded duration"))?;
        validate_edits(&timing, expected)?;
        let presentation_length = if timing.edits.is_empty() {
            if timing.track_offset >= 0 {
                trimmed.checked_add(timing.track_offset.unsigned_abs())
            } else {
                trimmed.checked_sub(timing.track_offset.unsigned_abs())
            }
            .ok_or_else(|| invalid("track offset exceeds gapless audio duration"))?
        } else {
            timing.edits.iter().try_fold(0, |length, edit| {
                Ok::<_, Error>(
                    length.max(
                        shifted_edit(*edit, timing.track_offset)?
                            .map_or(0, |shifted| shifted.presentation.end),
                    ),
                )
            })?
        };
        Ok(Self {
            decoder,
            packets,
            decoded: BTreeMap::new(),
            resident: None,
            resident_bytes: 0,
            timing,
            sample_rate,
            channels,
            decoded_length: expected,
            presentation_length,
            preroll_packets,
            limits,
        })
    }

    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub const fn channels(&self) -> u16 {
        self.channels
    }
    pub const fn presentation_length(&self) -> u64 {
        self.presentation_length
    }
    pub fn timing(&self) -> &AudioTrackTiming {
        &self.timing
    }

    /// Returns exactly the requested contiguous half-open presentation range.
    pub fn get_range(
        &mut self,
        range: SampleRange,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        let (mut output, mappings) = self.start_request(range, cancellation)?;
        for mapping in mappings {
            let Some(media) = mapping.media else { continue };
            if let Some(plan) = self.plan_decode(media)? {
                if plan.reset {
                    self.decoder.reset()?;
                }
                for index in plan.from..=plan.last {
                    if cancellation.is_canceled() {
                        return Err(canceled());
                    }
                    let packet = self.packet_at(index)?;
                    let buffer = self.decoder.decode(&packet, cancellation)?;
                    self.accept_decoded(&plan, index, buffer)?;
                }
                self.evict_behind(plan.first);
            }
            self.copy_media(media, mapping.output_offset, &mut output)?;
        }
        AudioBuffer::new(range, self.sample_rate, self.channels, output, &self.limits)
    }

    /// [`Self::get_range`] with the packets it needs decoded by `decode`
    /// rather than the reader's own decoder, for a decoder that can only be
    /// driven asynchronously.
    ///
    /// `decode` is handed whether it must reset its state first, and the
    /// packets to decode in order; it returns one buffer per packet, each
    /// covering exactly its packet's interval. The reader's own decoder is not
    /// used.
    #[doc(hidden)]
    pub async fn get_range_with<F>(
        &mut self,
        range: SampleRange,
        cancellation: &CancellationToken,
        mut decode: F,
    ) -> Result<AudioBuffer>
    where
        F: AsyncFnMut(bool, &[EncodedAudioSample]) -> Result<Vec<AudioBuffer>>,
    {
        let (mut output, mappings) = self.start_request(range, cancellation)?;
        for mapping in mappings {
            let Some(media) = mapping.media else { continue };
            if let Some(plan) = self.plan_decode(media)? {
                let window = (plan.from..=plan.last)
                    .map(|index| self.packet_at(index))
                    .collect::<Result<Vec<_>>>()?;
                let buffers = decode(plan.reset, &window).await?;
                if cancellation.is_canceled() {
                    return Err(canceled());
                }
                if buffers.len() != plan.last - plan.from + 1 {
                    return Err(Error::new(
                        ErrorKind::Codec,
                        "audio decoder returned the wrong number of packets",
                    ));
                }
                for (index, buffer) in (plan.from..=plan.last).zip(buffers) {
                    self.accept_decoded(&plan, index, buffer)?;
                }
                self.evict_behind(plan.first);
            }
            self.copy_media(media, mapping.output_offset, &mut output)?;
        }
        AudioBuffer::new(range, self.sample_rate, self.channels, output, &self.limits)
    }

    /// Validates a request and allocates its silent output.
    fn start_request(
        &self,
        range: SampleRange,
        cancellation: &CancellationToken,
    ) -> Result<(Vec<f32>, Vec<Mapping>)> {
        if range.end > self.presentation_length {
            return Err(invalid("audio request exceeds the presentation duration"));
        }
        if cancellation.is_canceled() {
            return Err(canceled());
        }
        let sample_count = range
            .len()
            .checked_mul(u64::from(self.channels))
            .ok_or_else(|| limit("audio request size overflow"))?;
        let bytes = sample_count
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| limit("audio request allocation overflow"))?;
        if bytes > self.limits.max_allocation_bytes {
            return Err(limit("audio request exceeds the allocation limit"));
        }
        let output = vec![
            0.0;
            usize::try_from(sample_count)
                .map_err(|_| limit("audio request cannot be represented"))?
        ];
        Ok((output, self.mappings(range)?))
    }

    pub fn get(
        &mut self,
        timeline: Timeline,
        frame: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        if timeline.audio_sample_rate() != self.sample_rate {
            return Err(invalid("timeline and audio sample rates do not match"));
        }
        self.get_range(timeline.audio_interval_for_frame(frame)?, cancellation)
    }

    /// The packets whose compressed bytes [`Self::get_range`] of `range` reads from the provider,
    /// as runs of packet indexes, so a caller can load them before asking.
    ///
    /// Each run is what one of the request's media ranges decodes from the reader's current
    /// state: nothing when its packets are already decoded, the packets after the decoded run
    /// when the request continues it, and the preroll packets onwards when it resets. A request
    /// whose provider reports [`ErrorKind::WouldBlock`] for a packet keeps what it decoded before
    /// it, so asking again once that packet is loaded carries on from there.
    pub fn packets_for_range(&self, range: SampleRange) -> Result<Vec<std::ops::Range<usize>>> {
        if range.end > self.presentation_length {
            return Err(invalid("audio request exceeds the presentation duration"));
        }
        let mut runs = Vec::new();
        for mapping in self.mappings(range)? {
            let Some(media) = mapping.media else { continue };
            if let Planned::Decode(plan) = self.plan(media)? {
                runs.push(plan.from..plan.last + 1);
            }
        }
        Ok(runs)
    }

    /// Cancels are caller-owned; reset drops queued decode state before a seek.
    pub fn reset(&mut self) -> Result<()> {
        self.decoder.reset()?;
        self.discard_resident();
        Ok(())
    }

    /// Builds the owned, transient [`EncodedAudioSample`] the decoder is
    /// given for one packet, reading its bytes from the provider.
    fn packet_at(&self, index: usize) -> Result<EncodedAudioSample> {
        Ok(EncodedAudioSample {
            decoded_range: self.packets.decoded_range(index),
            data: self.packets.read(index)?.into_owned(),
        })
    }

    fn discard_resident(&mut self) {
        self.decoded.clear();
        self.resident = None;
        self.resident_bytes = 0;
    }

    /// Packets retained behind the request are dropped once the resident run
    /// exceeds either bound `Limits` already defines for a decode: at most
    /// `max_decode_samples_per_seek` packets, holding at most
    /// `max_allocation_bytes` of decoded samples. Only packets before
    /// `keep_from` are eligible, so a request never evicts what it just
    /// decoded for itself.
    fn evict_behind(&mut self, keep_from: usize) {
        let Some((mut start, end)) = self.resident else {
            return;
        };
        let max_packets = self.limits.max_decode_samples_per_seek as usize;
        while start < keep_from
            && (end - start + 1 > max_packets
                || self.resident_bytes > self.limits.max_allocation_bytes)
        {
            // An empty packet's buffer was never cached, and its start is the
            // key of the packet after it.
            let range = self.packets.decoded_range(start);
            if !range.is_empty()
                && let Some(buffer) = self.decoded.remove(&range.start)
            {
                self.resident_bytes = self.resident_bytes.saturating_sub(buffer_bytes(&buffer));
            }
            start += 1;
        }
        self.resident = Some((start, end));
    }

    fn mappings(&self, request: SampleRange) -> Result<Vec<Mapping>> {
        let mut mappings = Vec::new();
        if self.timing.edits.is_empty() {
            let presentation_start = self.timing.track_offset.max(0) as u64;
            let media_start = u64::from(self.timing.priming)
                .checked_add(self.timing.track_offset.min(0).unsigned_abs())
                .ok_or_else(|| limit("audio media offset overflow"))?;
            let playable_end = self.decoded_length - u64::from(self.timing.padding);
            let length = playable_end.saturating_sub(media_start);
            let presentation_end = presentation_start
                .checked_add(length)
                .ok_or_else(|| limit("audio presentation range overflow"))?;
            push_intersection(
                &mut mappings,
                request,
                SampleRange::new(presentation_start, presentation_end)?,
                Some(media_start),
            )?;
        } else {
            for edit in &self.timing.edits {
                let Some(edit) = shifted_edit(*edit, self.timing.track_offset)? else {
                    continue;
                };
                push_intersection(&mut mappings, request, edit.presentation, edit.media_start)?;
            }
        }
        Ok(mappings)
    }

    /// Decides what `range` of the media needs decoded, or `None` when every
    /// packet it covers is already resident.
    ///
    /// A request whose window sits inside the resident run needs nothing. A
    /// request that starts inside it, or immediately after it, and reaches
    /// past its end is the forward-sequential playback case: the decoder is
    /// already positioned on the next packet, so the run is extended in place
    /// rather than resetting and re-decoding the preroll. Anything else -
    /// cold, backwards, or separated by a gap - takes the reset path, which is
    /// what a real seek needs, and discards the resident run.
    fn plan_decode(&mut self, range: SampleRange) -> Result<Option<DecodePlan>> {
        match self.plan(range)? {
            Planned::Resident { first } => {
                self.evict_behind(first);
                Ok(None)
            }
            Planned::Decode(plan) => {
                if plan.reset {
                    self.discard_resident();
                }
                Ok(Some(plan))
            }
        }
    }

    /// [`Self::plan_decode`]'s decision, without acting on it.
    fn plan(&self, range: SampleRange) -> Result<Planned> {
        let first = (0..self.packets.len())
            .find(|&index| self.packets.decoded_range(index).end > range.start)
            .ok_or_else(|| invalid("audio edit maps beyond decoded samples"))?;
        let last = (0..self.packets.len())
            .rfind(|&index| self.packets.decoded_range(index).start < range.end)
            .ok_or_else(|| invalid("audio edit maps before decoded samples"))?;
        let seek_start = first.saturating_sub(self.preroll_packets);
        if last - seek_start + 1 > self.limits.max_decode_samples_per_seek as usize {
            return Err(limit(
                "audio request exceeded the configured decode-work limit",
            ));
        }
        match self.resident {
            Some((resident_start, resident_end))
                if first >= resident_start && first <= resident_end + 1 =>
            {
                if last <= resident_end {
                    return Ok(Planned::Resident { first });
                }
                Ok(Planned::Decode(DecodePlan {
                    reset: false,
                    from: resident_end + 1,
                    last,
                    first,
                    preroll_end: resident_end + 1,
                }))
            }
            _ => Ok(Planned::Decode(DecodePlan {
                reset: true,
                from: seek_start,
                last,
                first,
                preroll_end: first,
            })),
        }
    }

    /// Takes in the buffer `plan` decoded for packet `index`.
    fn accept_decoded(
        &mut self,
        plan: &DecodePlan,
        index: usize,
        buffer: AudioBuffer,
    ) -> Result<()> {
        let decoded_range = self.packets.decoded_range(index);
        if buffer.range != decoded_range
            || buffer.sample_rate != self.sample_rate
            || buffer.channels != self.channels
        {
            return Err(Error::new(
                ErrorKind::Codec,
                "audio decoder output does not match its packet interval or format",
            ));
        }
        // Preroll only brings a reset decoder's state up to date. What it
        // decodes to is not the stream's audio - a Vorbis packet with no block
        // before it to overlap decodes to silence - so it is never kept for a
        // later request to read.
        if index < plan.preroll_end {
            return Ok(());
        }
        if !buffer.range.is_empty() {
            self.resident_bytes = self.resident_bytes.saturating_add(buffer_bytes(&buffer));
            self.decoded.insert(buffer.range.start, buffer);
        }
        self.resident = Some((self.resident.map_or(index, |(start, _)| start), index));
        Ok(())
    }

    fn copy_media(&self, range: SampleRange, output_offset: u64, output: &mut [f32]) -> Result<()> {
        let channels = usize::from(self.channels);
        // Walk back from the last packet that can overlap. The resident run
        // now spans more than one request, so visiting every buffer would make
        // each read cost the size of the cache instead of the size of the read.
        for buffer in self
            .decoded
            .range(..range.end)
            .rev()
            .map(|(_, buffer)| buffer)
        {
            if buffer.range.end <= range.start {
                break;
            }
            let start = range.start.max(buffer.range.start);
            let end = range.end.min(buffer.range.end);
            if start >= end {
                continue;
            }
            let input_start =
                usize::try_from((start - buffer.range.start) * u64::from(self.channels))
                    .map_err(|_| limit("audio input offset overflow"))?;
            let output_start =
                usize::try_from((output_offset + start - range.start) * u64::from(self.channels))
                    .map_err(|_| limit("audio output offset overflow"))?;
            let count = usize::try_from(end - start)
                .map_err(|_| limit("audio copy size overflow"))?
                .checked_mul(channels)
                .ok_or_else(|| limit("audio copy size overflow"))?;
            output[output_start..output_start + count]
                .copy_from_slice(&buffer.samples[input_start..input_start + count]);
        }
        Ok(())
    }
}

struct Mapping {
    media: Option<SampleRange>,
    output_offset: u64,
}

/// What one media range of a request needs.
enum Planned {
    /// Every packet it covers is resident; `first` is the first of them.
    Resident {
        first: usize,
    },
    Decode(DecodePlan),
}

/// The packets one media range of a request needs decoded.
struct DecodePlan {
    /// Whether the decoder has to be reset before `from`.
    reset: bool,
    from: usize,
    last: usize,
    /// The first packet holding samples the request reads.
    first: usize,
    /// Packets before this one are preroll, decoded but not kept.
    preroll_end: usize,
}

fn push_intersection(
    out: &mut Vec<Mapping>,
    request: SampleRange,
    presentation: SampleRange,
    media_start: Option<u64>,
) -> Result<()> {
    let start = request.start.max(presentation.start);
    let end = request.end.min(presentation.end);
    if start >= end {
        return Ok(());
    }
    let media = media_start
        .map(|base| {
            let source_start = base
                .checked_add(start - presentation.start)
                .ok_or_else(|| limit("audio edit mapping overflow"))?;
            let source_end = base
                .checked_add(end - presentation.start)
                .ok_or_else(|| limit("audio edit mapping overflow"))?;
            SampleRange::new(source_start, source_end)
        })
        .transpose()?;
    out.push(Mapping {
        media,
        output_offset: start - request.start,
    });
    Ok(())
}

fn validate_edits(timing: &AudioTrackTiming, decoded_length: u64) -> Result<()> {
    let mut end = 0;
    for edit in &timing.edits {
        if edit.presentation.start < end || edit.presentation.is_empty() {
            return Err(invalid(
                "audio edits must be nonempty and ordered without overlap",
            ));
        }
        if let Some(media_start) = edit.media_start {
            let media_end = media_start
                .checked_add(edit.presentation.len())
                .ok_or_else(|| limit("audio edit range overflow"))?;
            if media_start < u64::from(timing.priming)
                || media_end > decoded_length - u64::from(timing.padding)
            {
                return Err(invalid(
                    "audio edit includes priming, padding, or samples outside the track",
                ));
            }
        }
        end = edit.presentation.end;
    }
    Ok(())
}

fn shifted_edit(edit: AudioEdit, offset: i64) -> Result<Option<AudioEdit>> {
    let shifted_start = i128::from(edit.presentation.start) + i128::from(offset);
    let shifted_end = i128::from(edit.presentation.end) + i128::from(offset);
    if shifted_end <= 0 {
        return Ok(None);
    }
    let clipped = shifted_start.min(0).unsigned_abs();
    let start = u64::try_from(shifted_start.max(0))
        .map_err(|_| limit("shifted audio edit start cannot be represented"))?;
    let end = u64::try_from(shifted_end)
        .map_err(|_| limit("shifted audio edit end cannot be represented"))?;
    let media_start = edit
        .media_start
        .map(|value| {
            value
                .checked_add(
                    u64::try_from(clipped)
                        .map_err(|_| limit("shifted audio edit clip cannot be represented"))?,
                )
                .ok_or_else(|| limit("shifted audio edit media offset overflow"))
        })
        .transpose()?;
    Ok(Some(AudioEdit {
        presentation: SampleRange::new(start, end)?,
        media_start,
    }))
}

fn buffer_bytes(buffer: &AudioBuffer) -> u64 {
    buffer.samples.len() as u64 * std::mem::size_of::<f32>() as u64
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}
fn limit(message: &str) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}
fn canceled() -> Error {
    Error::new(ErrorKind::Canceled, "audio decode canceled")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct DecoderCounts {
        resets: std::rc::Rc<std::cell::Cell<usize>>,
        decodes: std::rc::Rc<std::cell::Cell<usize>>,
    }

    #[derive(Default)]
    struct FixtureDecoder {
        counts: DecoderCounts,
    }

    impl AudioDecoder for FixtureDecoder {
        fn decode(
            &mut self,
            sample: &EncodedAudioSample,
            cancellation: &CancellationToken,
        ) -> Result<AudioBuffer> {
            if cancellation.is_canceled() {
                return Err(canceled());
            }
            self.counts.decodes.set(self.counts.decodes.get() + 1);
            let values = (sample.decoded_range.start..sample.decoded_range.end)
                .map(|value| value as f32)
                .collect();
            AudioBuffer::new(sample.decoded_range, 48_000, 1, values, &Limits::default())
        }

        fn reset(&mut self) -> Result<()> {
            self.counts.resets.set(self.counts.resets.get() + 1);
            Ok(())
        }
    }

    fn packets_of(count: u64) -> Vec<EncodedAudioSample> {
        (0..count)
            .map(|index| EncodedAudioSample {
                decoded_range: SampleRange::new(index * 4, index * 4 + 4).unwrap(),
                data: vec![index as u8],
            })
            .collect()
    }

    /// A counted reader over `count` four-sample packets and no gapless trims.
    fn counted_reader(
        count: u64,
        preroll: usize,
        limits: Limits,
    ) -> (AudioSampleReader<FixtureDecoder>, DecoderCounts) {
        let decoder = FixtureDecoder::default();
        let counts = decoder.counts.clone();
        let reader = AudioSampleReader::new(
            decoder,
            packets_of(count),
            48_000,
            1,
            AudioTrackTiming::default(),
            preroll,
            limits,
        )
        .unwrap();
        (reader, counts)
    }

    /// Packets that answer only once the test has marked them loaded, and report
    /// [`ErrorKind::WouldBlock`] until then, the way a provider over a network source does.
    struct LoadOnDemand {
        packets: Vec<EncodedAudioSample>,
        loaded: std::sync::Arc<std::sync::Mutex<std::collections::BTreeSet<usize>>>,
    }

    impl AudioPacketProvider for LoadOnDemand {
        fn len(&self) -> usize {
            self.packets.len()
        }

        fn decoded_range(&self, index: usize) -> SampleRange {
            self.packets[index].decoded_range
        }

        fn read(&self, index: usize) -> Result<Cow<'_, [u8]>> {
            if self.loaded.lock().unwrap().contains(&index) {
                Ok(Cow::Borrowed(&self.packets[index].data))
            } else {
                Err(Error::new(ErrorKind::WouldBlock, "not loaded yet"))
            }
        }
    }

    /// Issue #672: a request whose provider has not loaded a packet yet reports `WouldBlock`,
    /// keeps what it decoded before it, and carries on from that packet once it is loaded.
    #[test]
    fn a_request_waiting_on_an_unloaded_packet_resumes_where_it_stopped() {
        let loaded = std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeSet::new()));
        let decoder = FixtureDecoder::default();
        let counts = decoder.counts.clone();
        let mut reader = AudioSampleReader::from_provider(
            decoder,
            Box::new(LoadOnDemand {
                packets: packets_of(10),
                loaded: loaded.clone(),
            }),
            48_000,
            1,
            AudioTrackTiming::default(),
            1,
            Limits::default(),
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        let range = SampleRange::new(9, 20).unwrap();

        // Packets 2..=4 hold samples 8..20, and one packet of preroll comes before them.
        assert_eq!(reader.packets_for_range(range).unwrap(), vec![1..5]);
        let error = reader.get_range(range, &cancellation).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::WouldBlock);

        loaded.lock().unwrap().extend([1, 2, 3]);
        let error = reader.get_range(range, &cancellation).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::WouldBlock);
        assert_eq!(counts.decodes.get(), 3);
        // Packets 2 and 3 were kept, so only packet 4 is still needed.
        assert_eq!(reader.packets_for_range(range).unwrap(), vec![4..5]);

        loaded.lock().unwrap().insert(4);
        let buffer = reader.get_range(range, &cancellation).unwrap();
        let expected: Vec<f32> = (9..20).map(|value| value as f32).collect();
        assert_eq!(buffer.samples, expected);
        assert_eq!(counts.decodes.get(), 4);

        // Already decoded: nothing to load. Continuing: just the next packet.
        assert!(reader.packets_for_range(range).unwrap().is_empty());
        assert_eq!(
            reader
                .packets_for_range(SampleRange::new(16, 24).unwrap())
                .unwrap(),
            vec![5..6]
        );
        assert!(
            reader
                .packets_for_range(SampleRange::new(0, 41).unwrap())
                .is_err()
        );
    }

    fn packets() -> Vec<EncodedAudioSample> {
        (0..3)
            .map(|index| EncodedAudioSample {
                decoded_range: SampleRange::new(index * 4, index * 4 + 4).unwrap(),
                data: vec![index as u8],
            })
            .collect()
    }

    #[test]
    fn gapless_offset_reads_return_exact_contiguous_ranges() {
        let timing = AudioTrackTiming {
            priming: 2,
            padding: 2,
            track_offset: 1,
            edits: Vec::new(),
        };
        let mut reader = AudioSampleReader::new(
            FixtureDecoder::default(),
            packets(),
            48_000,
            1,
            timing.clone(),
            1,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(reader.timing(), &timing);
        assert_eq!(reader.presentation_length(), 9);
        let buffer = reader
            .get_range(SampleRange::new(0, 9).unwrap(), &CancellationToken::new())
            .unwrap();
        assert_eq!(buffer.range, SampleRange { start: 0, end: 9 });
        assert_eq!(
            buffer.samples,
            vec![0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn edit_lists_preserve_empty_edits_and_media_mapping() {
        let timing = AudioTrackTiming {
            priming: 1,
            padding: 1,
            track_offset: 1,
            edits: vec![
                AudioEdit {
                    presentation: SampleRange::new(0, 2).unwrap(),
                    media_start: None,
                },
                AudioEdit {
                    presentation: SampleRange::new(2, 5).unwrap(),
                    media_start: Some(4),
                },
            ],
        };
        let mut reader = AudioSampleReader::new(
            FixtureDecoder::default(),
            packets(),
            48_000,
            1,
            timing,
            0,
            Limits::default(),
        )
        .unwrap();
        let buffer = reader
            .get_range(SampleRange::new(0, 6).unwrap(), &CancellationToken::new())
            .unwrap();
        assert_eq!(buffer.samples, vec![0.0, 0.0, 0.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn cancellation_stops_decode_before_queued_work_runs() {
        let mut reader = AudioSampleReader::new(
            FixtureDecoder::default(),
            packets(),
            48_000,
            1,
            AudioTrackTiming::default(),
            0,
            Limits::default(),
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error = reader
            .get_range(SampleRange::new(0, 1).unwrap(), &cancellation)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Canceled);
    }

    #[test]
    fn forward_sequential_reads_decode_each_packet_once() {
        let (mut reader, counts) = counted_reader(8, 2, Limits::default());
        let cancellation = CancellationToken::new();
        // Start past the preroll depth so the first read pays for it, then
        // walk forward one packet at a time.
        for step in 3..8 {
            let range = SampleRange::new(step * 4, step * 4 + 4).unwrap();
            let buffer = reader.get_range(range, &cancellation).unwrap();
            assert_eq!(
                buffer.samples,
                (step * 4..step * 4 + 4)
                    .map(|v| v as f32)
                    .collect::<Vec<_>>()
            );
        }
        // Packets 1 through 7: the two preroll units ahead of packet 3 plus
        // each requested unit, decoded once each and never re-decoded.
        assert_eq!(counts.decodes.get(), 7);
        assert_eq!(counts.resets.get(), 1);
    }

    #[test]
    fn backward_and_distant_requests_still_reset_with_preroll() {
        let (mut reader, counts) = counted_reader(8, 1, Limits::default());
        let cancellation = CancellationToken::new();
        reader
            .get_range(SampleRange::new(12, 16).unwrap(), &cancellation)
            .unwrap();
        assert_eq!(counts.resets.get(), 1);
        // Backwards of the resident run.
        reader
            .get_range(SampleRange::new(0, 4).unwrap(), &cancellation)
            .unwrap();
        assert_eq!(counts.resets.get(), 2);
        // Forward but separated by a gap from the resident run.
        let buffer = reader
            .get_range(SampleRange::new(28, 32).unwrap(), &cancellation)
            .unwrap();
        assert_eq!(counts.resets.get(), 3);
        assert_eq!(buffer.samples, vec![28.0, 29.0, 30.0, 31.0]);
    }

    #[test]
    fn resident_packets_stay_within_the_configured_bound() {
        let limits = Limits {
            max_decode_samples_per_seek: 3,
            ..Limits::default()
        };
        let (mut reader, _) = counted_reader(16, 0, limits);
        let cancellation = CancellationToken::new();
        for step in 0..16 {
            reader
                .get_range(
                    SampleRange::new(step * 4, step * 4 + 4).unwrap(),
                    &cancellation,
                )
                .unwrap();
            assert!(reader.decoded.len() <= 3, "resident run grew unbounded");
        }
        // The allocation bound evicts too, independently of the packet count.
        let limits = Limits {
            // One four-sample packet is 16 bytes, so the request itself fits
            // but a second resident packet does not.
            max_allocation_bytes: 16,
            ..Limits::default()
        };
        let (mut reader, _) = counted_reader(16, 0, limits);
        for step in 0..16 {
            reader
                .get_range(
                    SampleRange::new(step * 4, step * 4 + 4).unwrap(),
                    &cancellation,
                )
                .unwrap();
            assert!(reader.decoded.len() <= 2, "allocation bound did not evict");
        }
    }

    /// A Vorbis stream's first packet decodes nothing: an empty interval
    /// sharing its start with the packet after it.
    fn packets_after_an_empty_one(count: u64) -> Vec<EncodedAudioSample> {
        std::iter::once(EncodedAudioSample {
            decoded_range: SampleRange::new(0, 0).unwrap(),
            data: vec![0xff],
        })
        .chain(packets_of(count))
        .collect()
    }

    #[test]
    fn a_packet_that_decodes_nothing_is_decoded_but_never_cached() {
        let decoder = FixtureDecoder::default();
        let counts = decoder.counts.clone();
        let mut reader = AudioSampleReader::new(
            decoder,
            packets_after_an_empty_one(4),
            48_000,
            1,
            AudioTrackTiming::default(),
            1,
            Limits::default(),
        )
        .unwrap();
        assert_eq!(reader.presentation_length(), 16);
        let cancellation = CancellationToken::new();
        let buffer = reader
            .get_range(SampleRange::new(0, 6).unwrap(), &cancellation)
            .unwrap();
        assert_eq!(buffer.samples, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        // The empty packet was the preroll ahead of the first packet with
        // samples, and it holds no entry of its own.
        assert_eq!(counts.decodes.get(), 3);
        assert_eq!(reader.decoded.len(), 2);
    }

    /// Decodes the first packet after a reset wrong, as a decoder with no
    /// earlier state to continue from does.
    #[derive(Default)]
    struct ColdStartDecoder {
        cold: bool,
    }

    impl AudioDecoder for ColdStartDecoder {
        fn decode(
            &mut self,
            sample: &EncodedAudioSample,
            _: &CancellationToken,
        ) -> Result<AudioBuffer> {
            let cold = std::mem::take(&mut self.cold);
            let values = (sample.decoded_range.start..sample.decoded_range.end)
                .map(|value| if cold { -1.0 } else { value as f32 })
                .collect();
            AudioBuffer::new(sample.decoded_range, 48_000, 1, values, &Limits::default())
        }

        fn reset(&mut self) -> Result<()> {
            self.cold = true;
            Ok(())
        }
    }

    #[test]
    fn preroll_output_is_never_served_to_a_later_request() {
        let mut reader = AudioSampleReader::new(
            ColdStartDecoder::default(),
            packets_of(4),
            48_000,
            1,
            AudioTrackTiming::default(),
            1,
            Limits::default(),
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        let later = reader
            .get_range(SampleRange::new(8, 12).unwrap(), &cancellation)
            .unwrap();
        assert_eq!(later.samples, vec![8.0, 9.0, 10.0, 11.0]);
        // Packet 1 was this decode's preroll. Reading it now must decode it
        // again behind its own preroll rather than return the cold output.
        let earlier = reader
            .get_range(SampleRange::new(4, 8).unwrap(), &cancellation)
            .unwrap();
        assert_eq!(earlier.samples, vec![4.0, 5.0, 6.0, 7.0]);
    }

    fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => value,
            std::task::Poll::Pending => panic!("the decode closure never suspends"),
        }
    }

    #[test]
    fn an_asynchronous_decoder_is_driven_as_the_readers_own_would_be() {
        let (mut reader, _) = counted_reader(8, 1, Limits::default());
        let mut fixture = FixtureDecoder::default();
        let mut calls = Vec::new();
        let cancellation = CancellationToken::new();
        let mut read = |reader: &mut AudioSampleReader<FixtureDecoder>, start: u64, end: u64| {
            block_on(reader.get_range_with(
                SampleRange::new(start, end).unwrap(),
                &cancellation,
                async |reset, packets: &[EncodedAudioSample]| {
                    calls.push((reset, packets.len()));
                    packets
                        .iter()
                        .map(|packet| fixture.decode(packet, &CancellationToken::new()))
                        .collect()
                },
            ))
            .unwrap()
            .samples
        };
        // Cold: reset, one preroll packet and the requested one.
        assert_eq!(read(&mut reader, 12, 16), vec![12.0, 13.0, 14.0, 15.0]);
        // Sequential: continues without a reset or preroll.
        assert_eq!(read(&mut reader, 16, 20), vec![16.0, 17.0, 18.0, 19.0]);
        // Already resident: nothing to decode.
        assert_eq!(
            read(&mut reader, 13, 18),
            vec![13.0, 14.0, 15.0, 16.0, 17.0]
        );
        // Backwards: a seek.
        assert_eq!(read(&mut reader, 0, 2), vec![0.0, 1.0]);
        assert_eq!(calls, vec![(true, 2), (false, 1), (true, 1)]);
    }

    #[test]
    fn packets_that_decode_nothing_at_all_are_rejected() {
        let error = AudioSampleReader::new(
            FixtureDecoder::default(),
            vec![EncodedAudioSample {
                decoded_range: SampleRange::new(0, 0).unwrap(),
                data: vec![0],
            }],
            48_000,
            1,
            AudioTrackTiming::default(),
            0,
            Limits::default(),
        )
        .err()
        .unwrap();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}
