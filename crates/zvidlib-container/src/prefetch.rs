//! Non-blocking access to a track's compressed samples over a [`ByteSource`]
//! whose reads genuinely suspend, such as a browser `fetch` or HTTP range
//! request (issue #672).
//!
//! [`TrackSampleProvider`] reads through its source from inside
//! [`ExactFrameReader::get`], which is synchronous, so it can only drive a
//! source that resolves on the first poll. This module splits the read in two
//! instead. A [`TrackSampleLoader`] owns the track's index and the source, and
//! loads samples asynchronously into a byte-budgeted cache. The providers it
//! hands out answer a reader's synchronous reads from that cache alone and
//! report [`ErrorKind::WouldBlock`] for a sample that is not loaded yet rather
//! than blocking. A provider remembers every sample it was asked for and did
//! not have, so the caller awaits [`TrackSampleLoader::load_missing`] and asks
//! again:
//!
//! ```ignore
//! let frame = loop {
//!     match reader.get(frame_index, &cancellation) {
//!         Err(error) if error.kind() == ErrorKind::WouldBlock => {
//!             loader.load_missing().await?;
//!         }
//!         result => break result?,
//!     }
//! };
//! ```
//!
//! Both readers leave their state resumable when a provider read fails, so the
//! retry carries on from the sample that was missing rather than decoding the
//! run again from its random-access point. Loading the run a request needs
//! before making it turns those per-sample round trips into one batch:
//! [`ExactFrameReader::decode_positions_for`] and
//! [`AudioSampleReader::packets_for_range`] name it, and
//! [`TrackSampleLoader::load`] loads it together with a playback readahead.
//!
//! [`TrackSampleProvider`]: crate::track::TrackSampleProvider
//! [`ExactFrameReader::get`]: crate::codec::ExactFrameReader::get
//! [`ExactFrameReader::decode_positions_for`]: crate::codec::ExactFrameReader::decode_positions_for
//! [`AudioSampleReader::packets_for_range`]: crate::audio::AudioSampleReader::packets_for_range

use crate::audio::AudioPacketProvider;
use crate::codec::{SampleProvider, TrackKind};
use crate::io::ByteSource;
use crate::media::Codec;
use crate::timeline::{FrameIndex, SampleRange};
use crate::track::{Track, read_exact};
use crate::{Error, ErrorKind, Result};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard};

/// The most bytes one coalesced read of adjacent samples asks the source for.
///
/// Adjacent samples are fetched together so a run costs one range request
/// rather than one per sample, but a single unbounded request would delay the
/// first sample of the run behind the whole of it.
const MAX_COALESCED_READ_BYTES: u64 = 4 * 1024 * 1024;

/// The most bytes between two samples that a coalesced read reads through and
/// discards rather than splitting into two requests.
///
/// A WebM stores each sample in a block of its own, so a few header bytes
/// separate every sample from the next, and the blocks of other tracks lie
/// between a track's samples wherever the file interleaves them. Reading
/// through a gap this small costs less than another round trip to the source
/// (issue #685).
const MAX_COALESCED_GAP_BYTES: u64 = 16 * 1024;

/// Loads a track's compressed samples from a [`ByteSource`] asynchronously,
/// ahead of the synchronous readers that decode them.
///
/// The loader holds the track's index, the source and a cache of loaded
/// samples bounded by `budget_bytes`, evicting the least recently used
/// samples once it is full. [`Self::sample_provider`] and
/// [`Self::audio_packet_provider`] build providers that share the cache, for
/// [`ExactFrameReader::from_provider`] and
/// [`AudioSampleReader::from_provider`]. Those providers never touch the
/// source themselves, so `S` need not be `Send` and its reads may suspend for
/// as long as they like: a browser `fetch` is the case this exists for.
///
/// [`ExactFrameReader::from_provider`]: crate::codec::ExactFrameReader::from_provider
/// [`AudioSampleReader::from_provider`]: crate::audio::AudioSampleReader::from_provider
pub struct TrackSampleLoader<S> {
    track: Track,
    source: S,
    cache: SharedCache,
}

impl<S: ByteSource> TrackSampleLoader<S> {
    /// Fails if `budget_bytes` cannot hold the track's largest sample, which
    /// could then never be loaded. Reads no sample data.
    pub fn new(track: Track, source: S, budget_bytes: u64) -> Result<Self> {
        let largest = track
            .samples
            .iter()
            .map(|sample| u64::from(sample.size))
            .max()
            .ok_or_else(|| invalid("an on-demand sample loader requires a nonempty track"))?;
        if budget_bytes < largest {
            return Err(limit(
                "the sample cache budget cannot hold the track's largest sample",
            ));
        }
        Ok(Self {
            track,
            source,
            cache: SharedCache::new(budget_bytes),
        })
    }

    pub fn track(&self) -> &Track {
        &self.track
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    /// A [`SampleProvider`] for a video track, answering from this loader's
    /// cache. Fails if the track is not a video track.
    pub fn sample_provider(&self) -> Result<PrefetchedSampleProvider> {
        if self.track.kind != TrackKind::Video {
            return Err(unsupported("a sample provider requires a video track"));
        }
        Ok(PrefetchedSampleProvider {
            presentation_index_by_decode: self.track.presentation_index_by_decode()?,
            random_access: self
                .track
                .samples
                .iter()
                .map(|sample| sample.is_sync)
                .collect(),
            cache: self.cache.clone(),
        })
    }

    /// An [`AudioPacketProvider`] for an audio track, answering from this
    /// loader's cache, with `decoded_ranges` giving each packet's decoded
    /// interval in decode order.
    ///
    /// The intervals are the caller's because whether they can be known
    /// without reading every packet depends on the codec, as
    /// [`AudioPacketProvider`] explains. An Opus or Vorbis packet's depends
    /// on packet bytes; an AAC track should use [`Self::aac_packet_provider`],
    /// which derives them from the track's index, and an Opus track
    /// [`Self::opus_packet_provider`]. Fails if the track is not
    /// an audio track or `decoded_ranges` does not have one interval per
    /// sample.
    pub fn audio_packet_provider(
        &self,
        decoded_ranges: Vec<SampleRange>,
    ) -> Result<PrefetchedAudioPacketProvider> {
        if self.track.kind != TrackKind::Audio {
            return Err(unsupported(
                "an audio packet provider requires an audio track",
            ));
        }
        if decoded_ranges.len() != self.track.samples.len() {
            return Err(invalid(
                "an audio packet provider needs one decoded interval per sample",
            ));
        }
        Ok(PrefetchedAudioPacketProvider {
            decoded_ranges,
            check_opus_durations: false,
            cache: self.cache.clone(),
        })
    }

    /// An [`AudioPacketProvider`] for an Opus track, answering from this
    /// loader's cache, without reading every packet to learn its interval.
    ///
    /// Each packet's decoded interval is its sample-table duration, except
    /// the last packet's, which comes from its own table of contents: a muxer
    /// trims the stream's end by shortening that duration. Only that packet's
    /// first two bytes are read here. The table is then checked a packet at a
    /// time, as playback reaches each one: a packet whose table of contents
    /// disagrees with its duration is read as [`ErrorKind::MalformedMedia`],
    /// so a track whose intervals would differ from the ones
    /// [`Track::to_encoded_audio_samples`] gives is refused rather than
    /// played out of time. Fails if the track is not an Opus audio track.
    pub async fn opus_packet_provider(&self) -> Result<PrefetchedAudioPacketProvider> {
        if self.track.kind != TrackKind::Audio || self.track.codec != Codec::Opus {
            return Err(unsupported(
                "an Opus audio packet provider requires an Opus audio track",
            ));
        }
        // `new` refuses an empty track.
        let last = &self.track.samples[self.track.samples.len() - 1];
        let mut head = [0_u8; 2];
        let head = &mut head[..(last.size as usize).min(2)];
        read_exact(&self.source, last.offset, head).await?;
        let decoded_ranges = self
            .track
            .opus_decoded_ranges(crate::opus::opus_packet_samples(head)?)?;
        let mut provider = self.audio_packet_provider(decoded_ranges)?;
        provider.check_opus_durations = true;
        Ok(provider)
    }

    /// An [`AudioPacketProvider`] for an AAC track, answering from this
    /// loader's cache, with each packet's decoded interval taken from the
    /// track's sample durations.
    ///
    /// The intervals are the ones [`TrackAudioPacketProvider`] and
    /// [`Track::to_encoded_audio_samples`] give the same track, and
    /// building them reads no packet data. Fails if the track is not an AAC
    /// audio track: Opus tracks use [`Self::opus_packet_provider`], and Vorbis
    /// tracks [`Self::audio_packet_provider`] with intervals the caller
    /// supplies.
    ///
    /// [`TrackAudioPacketProvider`]: crate::track::TrackAudioPacketProvider
    pub fn aac_packet_provider(&self) -> Result<PrefetchedAudioPacketProvider> {
        if self.track.kind != TrackKind::Audio || self.track.codec != Codec::Aac {
            return Err(unsupported(
                "an AAC audio packet provider requires an AAC audio track",
            ));
        }
        let decoded_ranges = self
            .track
            .aac_decoded_ranges(self.track.audio_sample_rate()?)?;
        self.audio_packet_provider(decoded_ranges)
    }

    /// Loads every sample in `required` that is not already loaded, then up
    /// to `readahead` samples after it, as far as the budget allows.
    ///
    /// `required` is clamped to the track. Adjacent samples stored together,
    /// or separated only by a few kilobytes such as a WebM's block headers,
    /// are read together. The samples loaded here are the most recently used,
    /// so making room for them evicts other samples first.
    ///
    /// A `required` run larger than the whole budget - the walk to a frame
    /// deep inside a long group of pictures - is loaded from its start for as
    /// far as it fits, with no readahead. The reader decodes that much, reports
    /// [`ErrorKind::WouldBlock`] for the next sample, and resumes from there,
    /// so the run streams through the budget over several loads rather than
    /// having to fit in it at once.
    pub async fn load(&self, required: Range<usize>, readahead: usize) -> Result<()> {
        let len = self.track.samples.len();
        let required_end = required.end.min(len);
        let start = required.start.min(required_end);
        let mut batch_bytes = 0_u64;
        let mut end = start;
        while end < len && (end < required_end || end - required_end < readahead) {
            let size = u64::from(self.track.samples[end].size);
            if batch_bytes + size > self.cache.budget_bytes {
                break;
            }
            batch_bytes += size;
            end += 1;
        }
        self.fetch(start..end).await
    }

    /// Loads every sample a provider was asked for and did not have since the
    /// last call, returning how many that was.
    ///
    /// This is what a caller awaits after a reader reports
    /// [`ErrorKind::WouldBlock`], before asking the reader again.
    pub async fn load_missing(&self) -> Result<usize> {
        let missing = std::mem::take(&mut self.cache.lock().missing);
        let count = missing.len();
        let mut indexes = missing.into_iter().peekable();
        while let Some(start) = indexes.next() {
            let mut end = start + 1;
            while indexes.next_if_eq(&end).is_some() {
                end += 1;
            }
            self.load(start..end, 0).await?;
        }
        Ok(count)
    }

    /// Forgets the samples providers have reported missing, as a seek does:
    /// what the old position needed is no longer worth loading.
    pub fn clear_missing(&self) {
        self.cache.lock().missing.clear();
    }

    pub fn is_loaded(&self, decode_index: usize) -> bool {
        self.cache.lock().samples.contains_key(&decode_index)
    }

    /// The compressed bytes currently held, which never exceeds
    /// [`Self::budget_bytes`].
    pub fn resident_bytes(&self) -> u64 {
        self.cache.lock().resident_bytes
    }

    pub fn budget_bytes(&self) -> u64 {
        self.cache.budget_bytes
    }

    /// Loads the samples of `range` that are not resident and marks the rest
    /// as just used, in order, so the whole range ends up the most recent.
    ///
    /// The cache is never locked across a read, so a provider can still
    /// answer a reader while a load is waiting on the source.
    async fn fetch(&self, range: Range<usize>) -> Result<()> {
        let samples = &self.track.samples;
        let mut index = range.start;
        while index < range.end {
            if self.cache.lock().touch(index) {
                index += 1;
                continue;
            }
            let offset = samples[index].offset;
            let mut run_end = index + 1;
            // The bytes from the run's first sample to the end of its last.
            let mut run_bytes = u64::from(samples[index].size);
            while run_end < range.end && !self.is_loaded(run_end) {
                let next = &samples[run_end];
                let Some(gap) = next.offset.checked_sub(offset + run_bytes) else {
                    break;
                };
                let next_bytes = run_bytes + gap + u64::from(next.size);
                if gap > MAX_COALESCED_GAP_BYTES || next_bytes > MAX_COALESCED_READ_BYTES {
                    break;
                }
                run_bytes = next_bytes;
                run_end += 1;
            }
            let mut bytes = vec![
                0_u8;
                usize::try_from(run_bytes).map_err(|_| limit(
                    "coalesced sample read cannot be represented"
                ))?
            ];
            read_exact(&self.source, offset, &mut bytes).await?;
            let mut cache = self.cache.lock();
            for (decode_index, sample) in samples.iter().enumerate().take(run_end).skip(index) {
                let start = (sample.offset - offset) as usize;
                cache.insert(
                    decode_index,
                    Arc::from(&bytes[start..start + sample.size as usize]),
                );
            }
            index = run_end;
        }
        Ok(())
    }
}

/// A [`SampleProvider`] that answers from a [`TrackSampleLoader`]'s cache and
/// reports [`ErrorKind::WouldBlock`] for a sample that is not loaded yet.
pub struct PrefetchedSampleProvider {
    presentation_index_by_decode: Vec<u64>,
    random_access: Vec<bool>,
    cache: SharedCache,
}

impl SampleProvider for PrefetchedSampleProvider {
    fn len(&self) -> usize {
        self.random_access.len()
    }

    fn is_random_access(&self, decode_index: usize) -> bool {
        self.random_access[decode_index]
    }

    fn presentation_index(&self, decode_index: usize) -> FrameIndex {
        FrameIndex(self.presentation_index_by_decode[decode_index])
    }

    fn read(&self, decode_index: usize) -> Result<Cow<'_, [u8]>> {
        if decode_index >= self.len() {
            return Err(invalid("sample index is out of range"));
        }
        self.cache.read(decode_index)
    }
}

/// An [`AudioPacketProvider`] that answers from a [`TrackSampleLoader`]'s
/// cache and reports [`ErrorKind::WouldBlock`] for a packet that is not
/// loaded yet.
pub struct PrefetchedAudioPacketProvider {
    decoded_ranges: Vec<SampleRange>,
    /// Whether each packet is an Opus packet whose interval came from the
    /// sample table, so its own table of contents must agree with it.
    check_opus_durations: bool,
    cache: SharedCache,
}

impl AudioPacketProvider for PrefetchedAudioPacketProvider {
    fn len(&self) -> usize {
        self.decoded_ranges.len()
    }

    fn decoded_range(&self, index: usize) -> SampleRange {
        self.decoded_ranges[index]
    }

    fn read(&self, index: usize) -> Result<Cow<'_, [u8]>> {
        if index >= self.len() {
            return Err(invalid("sample index is out of range"));
        }
        let packet = self.cache.read(index)?;
        if self.check_opus_durations
            && u64::from(crate::opus::opus_packet_samples(&packet)?)
                != self.decoded_ranges[index].len()
        {
            return Err(Error::new(
                ErrorKind::MalformedMedia,
                "an Opus packet's duration disagrees with the track's sample table",
            ));
        }
        Ok(packet)
    }
}

/// The cache a [`TrackSampleLoader`] fills and its providers read, shared
/// between them.
#[derive(Clone)]
struct SharedCache {
    state: Arc<Mutex<CacheState>>,
    budget_bytes: u64,
}

struct CacheState {
    /// Loaded samples by decode index, with the use they were last touched at.
    samples: HashMap<usize, (Arc<[u8]>, u64)>,
    /// The same samples by the use they were last touched at, oldest first.
    by_use: BTreeMap<u64, usize>,
    next_use: u64,
    resident_bytes: u64,
    budget_bytes: u64,
    /// Samples a provider was asked for and did not have.
    missing: BTreeSet<usize>,
}

impl SharedCache {
    fn new(budget_bytes: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(CacheState {
                samples: HashMap::new(),
                by_use: BTreeMap::new(),
                next_use: 0,
                resident_bytes: 0,
                budget_bytes,
                missing: BTreeSet::new(),
            })),
            budget_bytes,
        }
    }

    fn lock(&self) -> MutexGuard<'_, CacheState> {
        // Nothing holding the lock can leave the cache inconsistent partway,
        // so a panic elsewhere while it was held does not invalidate it.
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn read(&self, index: usize) -> Result<Cow<'static, [u8]>> {
        let mut state = self.lock();
        if state.touch(index) {
            return Ok(Cow::Owned(state.samples[&index].0.to_vec()));
        }
        state.missing.insert(index);
        Err(Error::new(
            ErrorKind::WouldBlock,
            "the compressed sample is not loaded yet",
        ))
    }
}

impl CacheState {
    /// Marks a resident sample as the most recently used; `false` if it is
    /// not resident.
    fn touch(&mut self, index: usize) -> bool {
        let next_use = self.next_use;
        let Some((_, last_use)) = self.samples.get_mut(&index) else {
            return false;
        };
        self.by_use.remove(last_use);
        *last_use = next_use;
        self.by_use.insert(next_use, index);
        self.next_use += 1;
        true
    }

    fn insert(&mut self, index: usize, bytes: Arc<[u8]>) {
        if self.touch(index) {
            // A concurrent load got here first.
            return;
        }
        self.resident_bytes += bytes.len() as u64;
        self.samples.insert(index, (bytes, self.next_use));
        self.by_use.insert(self.next_use, index);
        self.next_use += 1;
        self.missing.remove(&index);
        while self.resident_bytes > self.budget_bytes {
            let Some((_, oldest)) = self.by_use.pop_first() else {
                break;
            };
            if let Some((evicted, _)) = self.samples.remove(&oldest) {
                self.resident_bytes -= evicted.len() as u64;
            }
        }
    }
}

fn invalid(message: &str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn limit(message: &str) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

fn unsupported(message: &str) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::SampleDependency;
    use crate::io::{IoFuture, MemorySource};
    use crate::track::TrackSample;
    use std::cell::Cell;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    /// A source whose every read suspends once before completing, the way a
    /// network fetch does, and counts the reads it was asked for.
    struct SuspendingSource {
        inner: MemorySource,
        reads: Cell<usize>,
    }

    impl ByteSource for SuspendingSource {
        fn len(&self) -> Option<u64> {
            self.inner.len()
        }

        fn read_at<'a>(&'a self, offset: u64, destination: &'a mut [u8]) -> IoFuture<'a, usize> {
            self.reads.set(self.reads.get() + 1);
            Box::pin(async move {
                YieldOnce(false).await;
                self.inner.read_at(offset, destination).await
            })
        }
    }

    struct YieldOnce(bool);

    impl Future for YieldOnce {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
            if self.0 {
                Poll::Ready(())
            } else {
                self.0 = true;
                context.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        let mut future = std::pin::pin!(future);
        let mut context = Context::from_waker(Waker::noop());
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
        }
    }

    /// A video track of `sizes.len()` samples stored back to back from
    /// offset 0, every fourth a random-access point, each filled with its own
    /// decode index.
    fn track_and_source(sizes: &[u32]) -> (Track, SuspendingSource) {
        let gaps = vec![0; sizes.len()];
        gapped_track_and_source(sizes, &gaps)
    }

    /// [`track_and_source`] with `gaps[index]` bytes of something else, such
    /// as a WebM block header, stored before each sample.
    fn gapped_track_and_source(sizes: &[u32], gaps: &[usize]) -> (Track, SuspendingSource) {
        let mut bytes = Vec::new();
        let mut samples = Vec::new();
        for (index, &size) in sizes.iter().enumerate() {
            bytes.extend(std::iter::repeat_n(0xff, gaps[index]));
            samples.push(TrackSample {
                offset: bytes.len() as u64,
                size,
                dts: index as u64,
                pts: index as i64,
                duration: 1,
                dependency: SampleDependency::INDEPENDENT,
                is_sync: index % 4 == 0,
            });
            bytes.extend(std::iter::repeat_n(index as u8, size as usize));
        }
        let track = Track {
            id: 1,
            kind: TrackKind::Video,
            codec: Codec::Hevc,
            timescale: 30,
            duration: sizes.len() as u64,
            dimensions: None,
            channels: None,
            sample_rate: None,
            decoder_config: Vec::new(),
            edits: Vec::new(),
            presentation_order: (0..sizes.len()).collect(),
            samples,
        };
        let source = SuspendingSource {
            inner: MemorySource::new(bytes),
            reads: Cell::new(0),
        };
        (track, source)
    }

    #[test]
    fn a_provider_reports_an_unloaded_sample_instead_of_blocking() {
        let (track, source) = track_and_source(&[10; 8]);
        let loader = TrackSampleLoader::new(track, source, 1_000).unwrap();
        let provider = loader.sample_provider().unwrap();

        let error = provider.read(3).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::WouldBlock);
        assert_eq!(
            loader.source().reads.get(),
            0,
            "the provider touched the source"
        );

        assert_eq!(block_on(loader.load_missing()).unwrap(), 1);
        assert_eq!(provider.read(3).unwrap().as_ref(), &[3; 10]);
        // Nothing is missing any more, so nothing more is loaded.
        assert_eq!(block_on(loader.load_missing()).unwrap(), 0);
    }

    #[test]
    fn loading_a_run_coalesces_contiguous_samples_and_adds_readahead() {
        let (track, source) = track_and_source(&[10; 16]);
        let loader = TrackSampleLoader::new(track, source, 1_000).unwrap();
        let provider = loader.sample_provider().unwrap();

        block_on(loader.load(4..7, 3)).unwrap();
        assert_eq!(loader.source().reads.get(), 1, "the run was not coalesced");
        for index in 4..10 {
            assert_eq!(provider.read(index).unwrap().as_ref(), &[index as u8; 10]);
        }
        assert!(!loader.is_loaded(3));
        assert!(!loader.is_loaded(10));

        // Loading what is already resident reads nothing.
        block_on(loader.load(4..10, 0)).unwrap();
        assert_eq!(loader.source().reads.get(), 1);
    }

    /// Issue #685: samples a few bytes apart, as a WebM's blocks are, are
    /// still read together, while a gap too large to be worth reading through
    /// splits the run.
    #[test]
    fn loading_a_run_reads_through_small_gaps_between_samples() {
        let mut gaps = vec![12; 8];
        gaps[6] = MAX_COALESCED_GAP_BYTES as usize + 1;
        let (track, source) = gapped_track_and_source(&[10; 8], &gaps);
        let loader = TrackSampleLoader::new(track, source, 1_000).unwrap();
        let provider = loader.sample_provider().unwrap();

        block_on(loader.load(0..6, 0)).unwrap();
        assert_eq!(loader.source().reads.get(), 1, "the run was not coalesced");
        block_on(loader.load(0..8, 0)).unwrap();
        assert_eq!(
            loader.source().reads.get(),
            2,
            "the large gap was read through"
        );
        for index in 0..8 {
            assert_eq!(provider.read(index).unwrap().as_ref(), &[index as u8; 10]);
        }
    }

    #[test]
    fn the_cache_stays_within_its_budget_and_evicts_the_least_recently_used() {
        let (track, source) = track_and_source(&[10; 16]);
        let loader = TrackSampleLoader::new(track, source, 40).unwrap();

        block_on(loader.load(0..4, 0)).unwrap();
        assert_eq!(loader.resident_bytes(), 40);
        block_on(loader.load(8..10, 0)).unwrap();
        assert_eq!(loader.resident_bytes(), 40);
        assert!(!loader.is_loaded(0) && !loader.is_loaded(1));
        assert!(loader.is_loaded(2) && loader.is_loaded(3));
        assert!(loader.is_loaded(8) && loader.is_loaded(9));

        // Readahead stops where it would no longer fit beside the request.
        block_on(loader.load(12..14, 10)).unwrap();
        assert!((12..16).all(|index| loader.is_loaded(index)));
        assert_eq!(loader.resident_bytes(), 40);

        // A run larger than the budget loads from its start as far as it fits.
        block_on(loader.load(0..6, 10)).unwrap();
        assert!((0..4).all(|index| loader.is_loaded(index)));
        assert!(!loader.is_loaded(4) && !loader.is_loaded(5));
        assert_eq!(loader.resident_bytes(), 40);
    }

    #[test]
    fn a_budget_smaller_than_the_largest_sample_is_refused() {
        let (track, source) = track_and_source(&[10, 50, 10]);
        let error = TrackSampleLoader::new(track, source, 49)
            .err()
            .expect("the budget cannot hold sample 1");
        assert_eq!(error.kind(), ErrorKind::ResourceLimit);
    }

    #[test]
    fn clearing_missing_samples_forgets_what_an_old_position_needed() {
        let (track, source) = track_and_source(&[10; 8]);
        let loader = TrackSampleLoader::new(track, source, 1_000).unwrap();
        let provider = loader.sample_provider().unwrap();
        provider.read(1).unwrap_err();
        loader.clear_missing();
        assert_eq!(block_on(loader.load_missing()).unwrap(), 0);
        assert_eq!(loader.source().reads.get(), 0);
    }

    #[test]
    fn an_audio_provider_answers_from_the_same_cache() {
        let (mut track, source) = track_and_source(&[4; 4]);
        track.kind = TrackKind::Audio;
        let loader = TrackSampleLoader::new(track, source, 1_000).unwrap();
        assert_eq!(
            loader.sample_provider().err().unwrap().kind(),
            ErrorKind::Unsupported
        );
        let ranges: Vec<_> = (0..4)
            .map(|index| SampleRange::new(index * 1024, (index + 1) * 1024).unwrap())
            .collect();
        assert_eq!(
            loader
                .audio_packet_provider(ranges[..3].to_vec())
                .err()
                .unwrap()
                .kind(),
            ErrorKind::InvalidInput
        );
        let provider = loader.audio_packet_provider(ranges.clone()).unwrap();
        assert_eq!(provider.decoded_range(2), ranges[2]);
        assert_eq!(provider.read(2).unwrap_err().kind(), ErrorKind::WouldBlock);
        block_on(loader.load_missing()).unwrap();
        assert_eq!(provider.read(2).unwrap().as_ref(), &[2; 4]);
    }
}
