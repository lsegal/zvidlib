//! Exact AAC sample reads with gapless and edit-list timeline mapping.

use crate::{
    AudioBuffer, CancellationToken, Error, ErrorKind, FrameIndex, Limits, Result, SampleRange,
    Timeline,
};
use std::collections::BTreeMap;

/// One compressed AAC access unit and the decoded interval it covers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedAudioSample {
    pub decoded_range: SampleRange,
    pub data: Vec<u8>,
}

/// A stateful AAC decoder. Output must use the packet's decoded sample clock.
pub trait AacDecoder {
    fn decode(
        &mut self,
        sample: &EncodedAudioSample,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer>;
    fn reset(&mut self) -> Result<()>;
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

/// Exact presentation-range access over sequential AAC packets.
pub struct AacSampleReader<D> {
    decoder: D,
    packets: Vec<EncodedAudioSample>,
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

impl<D: AacDecoder> AacSampleReader<D> {
    pub fn new(
        decoder: D,
        packets: Vec<EncodedAudioSample>,
        sample_rate: u32,
        channels: u16,
        timing: AudioTrackTiming,
        preroll_packets: usize,
        limits: Limits,
    ) -> Result<Self> {
        if packets.is_empty() {
            return Err(invalid("an AAC reader requires at least one packet"));
        }
        if sample_rate == 0 || sample_rate > limits.max_sample_rate {
            return Err(limit("AAC sample rate is outside configured limits"));
        }
        if channels == 0 || channels > limits.max_audio_channels {
            return Err(limit("AAC channel count is outside configured limits"));
        }
        let mut expected = 0;
        for packet in &packets {
            if packet.decoded_range.start != expected || packet.decoded_range.is_empty() {
                return Err(invalid(
                    "AAC packet sample intervals must be nonempty and contiguous",
                ));
            }
            expected = packet.decoded_range.end;
        }
        let trimmed = expected
            .checked_sub(u64::from(timing.priming) + u64::from(timing.padding))
            .ok_or_else(|| invalid("AAC priming and padding exceed decoded duration"))?;
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
        if range.end > self.presentation_length {
            return Err(invalid("audio request exceeds the presentation duration"));
        }
        if cancellation.is_cancelled() {
            return Err(cancelled());
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
        let mut output = vec![
            0.0;
            usize::try_from(sample_count)
                .map_err(|_| limit("audio request cannot be represented"))?
        ];
        for mapping in self.mappings(range)? {
            let Some(media) = mapping.media else { continue };
            self.ensure_decoded(media, cancellation)?;
            self.copy_media(media, mapping.output_offset, &mut output)?;
        }
        AudioBuffer::new(range, self.sample_rate, self.channels, output, &self.limits)
    }

    pub fn get(
        &mut self,
        timeline: Timeline,
        frame: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        if timeline.audio_sample_rate() != self.sample_rate {
            return Err(invalid("timeline and AAC sample rates do not match"));
        }
        self.get_range(timeline.audio_interval_for_frame(frame)?, cancellation)
    }

    /// Cancels are caller-owned; reset drops queued decode state before a seek.
    pub fn reset(&mut self) -> Result<()> {
        self.decoder.reset()?;
        self.discard_resident();
        Ok(())
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
            if let Some(buffer) = self
                .decoded
                .remove(&self.packets[start].decoded_range.start)
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

    fn ensure_decoded(
        &mut self,
        range: SampleRange,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        let first = self
            .packets
            .iter()
            .position(|packet| packet.decoded_range.end > range.start)
            .ok_or_else(|| invalid("audio edit maps beyond decoded AAC samples"))?;
        let last = self
            .packets
            .iter()
            .rposition(|packet| packet.decoded_range.start < range.end)
            .ok_or_else(|| invalid("audio edit maps before decoded AAC samples"))?;
        let seek_start = first.saturating_sub(self.preroll_packets);
        if last - seek_start + 1 > self.limits.max_decode_samples_per_seek as usize {
            return Err(limit(
                "AAC request exceeded the configured decode-work limit",
            ));
        }
        // A request whose window sits inside the resident run needs nothing. A
        // request that starts inside it, or immediately after it, and reaches
        // past its end is the forward-sequential playback case: the decoder is
        // already positioned on the next packet, so extend the run in place
        // rather than resetting and re-decoding the preroll. Anything else -
        // cold, backwards, or separated by a gap - takes the reset path, which
        // is what a real seek needs.
        let decode_from = match self.resident {
            Some((resident_start, resident_end))
                if first >= resident_start && first <= resident_end + 1 =>
            {
                if last <= resident_end {
                    self.evict_behind(first);
                    return Ok(());
                }
                resident_end + 1
            }
            _ => {
                self.decoder.reset()?;
                self.discard_resident();
                seek_start
            }
        };
        for index in decode_from..=last {
            if cancellation.is_cancelled() {
                return Err(cancelled());
            }
            let packet = &self.packets[index];
            let buffer = self.decoder.decode(packet, cancellation)?;
            if buffer.range != packet.decoded_range
                || buffer.sample_rate != self.sample_rate
                || buffer.channels != self.channels
            {
                return Err(Error::new(
                    ErrorKind::Codec,
                    "AAC decoder output does not match its packet interval or format",
                ));
            }
            self.resident_bytes = self.resident_bytes.saturating_add(buffer_bytes(&buffer));
            self.decoded.insert(buffer.range.start, buffer);
            self.resident = Some((self.resident.map_or(index, |(start, _)| start), index));
        }
        self.evict_behind(first);
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

impl<D: AacDecoder> crate::PlaybackAudioSource for AacSampleReader<D> {
    fn sample_rate(&self) -> u32 {
        self.sample_rate()
    }

    fn presentation_length(&self) -> u64 {
        self.presentation_length()
    }

    fn read(
        &mut self,
        range: SampleRange,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        self.get_range(range, cancellation)
    }

    fn reset(&mut self) -> Result<()> {
        self.reset()
    }
}

struct Mapping {
    media: Option<SampleRange>,
    output_offset: u64,
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
fn cancelled() -> Error {
    Error::new(ErrorKind::Cancelled, "AAC decode cancelled")
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

    impl AacDecoder for FixtureDecoder {
        fn decode(
            &mut self,
            sample: &EncodedAudioSample,
            cancellation: &CancellationToken,
        ) -> Result<AudioBuffer> {
            if cancellation.is_cancelled() {
                return Err(cancelled());
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
    ) -> (AacSampleReader<FixtureDecoder>, DecoderCounts) {
        let decoder = FixtureDecoder::default();
        let counts = decoder.counts.clone();
        let reader = AacSampleReader::new(
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
        let mut reader = AacSampleReader::new(
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
        let mut reader = AacSampleReader::new(
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
        let mut reader = AacSampleReader::new(
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
        assert_eq!(error.kind(), ErrorKind::Cancelled);
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
}
