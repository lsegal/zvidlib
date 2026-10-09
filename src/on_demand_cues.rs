//! Playing a cued WebM whose blocks are indexed a cue span at a time, as
//! playback reaches them (issue #692), for the native
//! [`crate::OnDemandPlayer`] and the browser's `OnDemandPlayback`.
//!
//! [`WebmCuedIndex`] opens a WebM from its header elements and `Cues`, and
//! indexes its blocks a span at a time: from one cued video key frame to the
//! next. How many frames come before a span is unknown until every span before
//! it is indexed, so a frame here is not numbered: its [`FrameIndex`] is its
//! presentation time in track ticks, and playback is addressed by time.
//!
//! [`CuedTimeline`] maps the playback clock to those frames a span at a time.
//! [`CuedVideoSource`] decodes each span on a source of its own - a span opens
//! on a key frame, so it decodes alone, to the pictures the whole file's index
//! gives - and loads every span's samples into one byte budget.
//! [`AudioWindow`] keeps one contiguous run of spans' audio packets that grows
//! as playback goes on, so the audio decodes straight across span boundaries
//! as it does from the whole file's index, and starts a new run where a seek
//! lands.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::audio::{AudioDecoder, AudioSampleReader, AudioTrackTiming};
use crate::codec::CancellationToken;
use crate::io::{ByteSource, IoFuture};
use crate::media::{AudioBuffer, Codec};
use crate::on_demand::VIDEO_READAHEAD_SAMPLES;
use crate::playback::{
    LazyPresentationTimeline, PlaybackAudioSource, PlaybackVideoSource, PrefetchAudioSource,
    PrefetchVideoSource, scale_to_audio_samples,
};
use crate::timeline::{FrameIndex, SampleRange};
use crate::track::Track;
use crate::{
    Error, ErrorKind, Limits, OPUS_PREROLL_SAMPLES, PrefetchedAudioPacketProvider, Result,
    SampleCache, TrackKind, TrackSampleLoader, VideoFrame, WebmCueSpan, WebmCuedIndex,
    WebmDemuxerOptions, probe_webm,
};

/// A cued WebM's index and the spans of it indexed so far, shared by the
/// timeline and the sources that play it, any of which indexes a span when
/// playback first reaches it.
pub(crate) struct CuedSpans<I> {
    index: Rc<WebmCuedIndex>,
    /// What spans are indexed through: normally a page cache over the input,
    /// so the headers of neighboring blocks come in one read.
    source: Rc<I>,
    spans: Rc<RefCell<BTreeMap<usize, Rc<WebmCueSpan>>>>,
}

impl<I> Clone for CuedSpans<I> {
    fn clone(&self) -> Self {
        Self {
            index: Rc::clone(&self.index),
            source: Rc::clone(&self.source),
            spans: Rc::clone(&self.spans),
        }
    }
}

/// What [`CuedSpans::open`] found an input to be.
pub(crate) enum Opened<I> {
    /// A cued WebM, played as its spans are indexed.
    Cued(CuedSpans<I>),
    /// Anything else, which the whole file's index plays: the source back,
    /// with whatever it cached of the input's header.
    Whole(I),
}

impl<I: ByteSource> CuedSpans<I> {
    /// Opens `source` as a cued WebM and indexes its first span. The whole
    /// file's index plays anything else: an MP4, a WebM without `Cues`, one
    /// with a Vorbis audio track - a Vorbis packet's position depends on the
    /// length of every packet before it - or one with an audio track that has
    /// no packet in the first span, which its packets' positions are counted
    /// from.
    pub(crate) async fn open(source: I, limits: &Limits) -> Result<Opened<I>> {
        if !probe_webm(&source).await? {
            return Ok(Opened::Whole(source));
        }
        let options = WebmDemuxerOptions {
            limits: *limits,
            ..WebmDemuxerOptions::default()
        };
        let Some(index) = WebmCuedIndex::open(&source, options).await? else {
            return Ok(Opened::Whole(source));
        };
        if index
            .tracks
            .iter()
            .any(|track| track.kind == TrackKind::Audio && track.codec != Codec::Opus)
        {
            return Ok(Opened::Whole(source));
        }
        let first = index.index_span(&source, 0).await?;
        if first
            .tracks
            .iter()
            .any(|track| track.kind == TrackKind::Audio && track.samples.is_empty())
        {
            return Ok(Opened::Whole(source));
        }
        Ok(Opened::Cued(Self {
            index: Rc::new(index),
            source: Rc::new(source),
            spans: Rc::new(RefCell::new(BTreeMap::from([(0, Rc::new(first))]))),
        }))
    }

    pub(crate) fn index(&self) -> &WebmCuedIndex {
        &self.index
    }

    /// `span`, if it has been indexed.
    pub(crate) fn indexed(&self, span: usize) -> Option<Rc<WebmCueSpan>> {
        self.spans.borrow().get(&span).cloned()
    }

    /// `span`, indexing it first if it has not been.
    pub(crate) async fn span(&self, span: usize) -> Result<Rc<WebmCueSpan>> {
        if let Some(indexed) = self.indexed(span) {
            return Ok(indexed);
        }
        let indexed = Rc::new(self.index.index_span(&*self.source, span).await?);
        self.spans.borrow_mut().insert(span, Rc::clone(&indexed));
        Ok(indexed)
    }

    /// The file's last span, once it has been indexed.
    pub(crate) fn last(&self) -> Option<Rc<WebmCueSpan>> {
        self.indexed(self.index.span_count() - 1)
    }

    /// The span holding `frame`, whose identity is its presentation time.
    fn span_of(&self, frame: FrameIndex) -> usize {
        self.index
            .span_at(i64::try_from(frame.0).unwrap_or(i64::MAX))
    }

    /// The span's video track.
    fn video<'a>(&self, span: &'a WebmCueSpan) -> Result<&'a Track> {
        span.track(self.index.video_track)
            .ok_or_else(|| Error::new(ErrorKind::Internal, "a cue span lost its video track"))
    }

    /// How long the presentation is estimated to run, in samples at
    /// `sample_rate`, before its last span is indexed: the segment's
    /// `Duration`, but never less than a second past where the last span
    /// starts.
    fn estimated_length(&self, sample_rate: u32) -> u64 {
        let declared = self.index.duration_seconds.map_or(0, |seconds| {
            (seconds * f64::from(sample_rate)).round() as u64
        });
        let last_start = self
            .index
            .span_start(self.index.span_count() - 1)
            .and_then(|ticks| u64::try_from(ticks).ok())
            .and_then(|ticks| {
                scale_to_audio_samples(ticks, self.index.timescale(), sample_rate).ok()
            })
            .unwrap_or(0);
        declared.max(last_start.saturating_add(u64::from(sample_rate)))
    }
}

/// A presented frame's identity: its presentation time in track ticks.
fn frame_identity(pts: i64) -> Result<FrameIndex> {
    u64::try_from(pts).map(FrameIndex).map_err(|_| {
        Error::new(
            ErrorKind::MalformedMedia,
            "negative video PTS is unsupported for indexed playback",
        )
    })
}

/// Each presented frame of a span's video track, in presentation order: its
/// identity, which is its presentation time, and the samples of a
/// `sample_rate` clock it is presented for, as
/// [`crate::IndexedPresentationTimeline::from_track`] times it.
fn frame_intervals(
    video: &Track,
    sample_rate: u32,
) -> Result<impl Iterator<Item = Result<(FrameIndex, SampleRange)>> + '_> {
    Ok(video.presentation_order.iter().map(move |&index| {
        let sample = &video.samples[index];
        let pts = frame_identity(sample.pts)?.0;
        let end = pts.checked_add(u64::from(sample.duration)).ok_or_else(|| {
            Error::new(
                ErrorKind::ResourceLimit,
                "video presentation timing overflow",
            )
        })?;
        Ok((
            FrameIndex(pts),
            SampleRange::new(
                scale_to_audio_samples(pts, video.timescale, sample_rate)?,
                scale_to_audio_samples(end, video.timescale, sample_rate)?,
            )?,
        ))
    }))
}

fn not_indexed() -> Error {
    Error::new(
        ErrorKind::WouldBlock,
        "this part of the WebM has not been indexed yet",
    )
}

/// A cued WebM's presentation timeline, indexed a span at a time; see the
/// [module documentation](self).
pub(crate) struct CuedTimeline<I> {
    spans: CuedSpans<I>,
    sample_rate: u32,
    /// Where each span after the first starts, in samples of the clock.
    starts: Vec<u64>,
}

impl<I: ByteSource> CuedTimeline<I> {
    pub(crate) fn new(spans: CuedSpans<I>, sample_rate: u32, limits: &Limits) -> Result<Self> {
        if sample_rate == 0 || sample_rate > limits.max_sample_rate {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "playback audio sample rate is outside configured limits",
            ));
        }
        let index = spans.index();
        let starts = (1..index.span_count())
            .map(|span| {
                let ticks = index
                    .span_start(span)
                    .and_then(|ticks| u64::try_from(ticks).ok())
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::MalformedMedia,
                            "negative video PTS is unsupported for indexed playback",
                        )
                    })?;
                scale_to_audio_samples(ticks, index.timescale(), sample_rate)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            spans,
            sample_rate,
            starts,
        })
    }

    fn span_for_sample(&self, sample: u64) -> usize {
        self.starts.partition_point(|&start| start <= sample)
    }
}

impl<I: ByteSource> LazyPresentationTimeline for CuedTimeline<I> {
    fn frame_for_audio_sample(&self, sample: u64) -> Result<FrameIndex> {
        let span = self.span_for_sample(sample);
        let indexed = self.spans.indexed(span).ok_or_else(not_indexed)?;
        let mut last = None;
        for interval in frame_intervals(self.spans.video(&indexed)?, self.sample_rate)? {
            let (frame, range) = interval?;
            if range.end > sample {
                return Ok(frame);
            }
            last = Some(frame);
        }
        // Between the span's last frame and the next span: the next span's
        // first frame, its cued key frame, is the first to end after it.
        if let Some(next) = self.spans.index().span_start(span + 1) {
            return Ok(FrameIndex(u64::try_from(next).unwrap_or(0)));
        }
        last.ok_or_else(|| Error::new(ErrorKind::MalformedMedia, "a WebM cue span has no frame"))
    }

    fn audio_interval_for_frame(&self, frame: FrameIndex) -> Result<SampleRange> {
        let indexed = self
            .spans
            .indexed(self.spans.span_of(frame))
            .ok_or_else(not_indexed)?;
        for interval in frame_intervals(self.spans.video(&indexed)?, self.sample_rate)? {
            let (identity, range) = interval?;
            if identity == frame {
                return Ok(range);
            }
        }
        Err(Error::new(
            ErrorKind::InvalidInput,
            "presentation frame is not indexed",
        ))
    }

    fn prepare(&mut self, sample: u64) -> IoFuture<'_, ()> {
        let span = self.span_for_sample(sample);
        Box::pin(async move { self.spans.span(span).await.map(drop) })
    }
}

/// Builds the source one span's video decodes on from a loader of the span's
/// samples: a loader into the playback's shared cache, keying its samples from
/// the base given.
pub(crate) type SpanVideoFactory<V> = Box<dyn Fn(Track, &SampleCache, usize) -> Result<V>>;

/// A cued WebM's video, decoded a span at a time; see the
/// [module documentation](self).
pub(crate) struct CuedVideoSource<I, V> {
    spans: CuedSpans<I>,
    cache: SampleCache,
    make: SpanVideoFactory<V>,
    /// The span playback is in.
    current: Option<SpanVideo<V>>,
    /// The span after it, loaded ahead as playback nears its end.
    next: Option<SpanVideo<V>>,
    /// The cache key the next span's first sample takes.
    next_base: usize,
}

struct SpanVideo<V> {
    span: usize,
    /// Each presented frame's identity, by its position in the span's
    /// presentation order.
    frames: Vec<FrameIndex>,
    source: V,
}

impl<V> SpanVideo<V> {
    /// The span source's own presentation index for `frame`.
    fn local(&self, frame: FrameIndex) -> Result<FrameIndex> {
        self.frames
            .binary_search(&frame)
            .map(|position| FrameIndex(position as u64))
            .map_err(|_| Error::new(ErrorKind::InvalidInput, "presentation frame is not indexed"))
    }
}

impl<I: ByteSource, V: PrefetchVideoSource> CuedVideoSource<I, V> {
    /// Video decoded on the sources `make` builds, holding at most
    /// `budget_bytes` of compressed samples across every span. Reads nothing.
    pub(crate) fn new(spans: CuedSpans<I>, budget_bytes: u64, make: SpanVideoFactory<V>) -> Self {
        Self {
            spans,
            cache: SampleCache::new(budget_bytes),
            make,
            current: None,
            next: None,
            next_base: 0,
        }
    }

    /// The compressed video bytes loaded now, across every span.
    #[cfg(all(any(unix, windows), not(target_arch = "wasm32")))]
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.cache.resident_bytes()
    }

    fn build(&mut self, span: &WebmCueSpan) -> Result<SpanVideo<V>> {
        let track = self.spans.video(span)?.clone();
        let frames = track
            .presentation_order
            .iter()
            .map(|&index| frame_identity(track.samples[index].pts))
            .collect::<Result<_>>()?;
        let base = self.next_base;
        self.next_base = base.saturating_add(track.samples.len());
        Ok(SpanVideo {
            span: span.index,
            frames,
            source: (self.make)(track, &self.cache, base)?,
        })
    }

    /// The source of `span`, if it is the current one or the next, which then
    /// becomes the current one.
    fn holding(&mut self, span: usize) -> Option<&mut SpanVideo<V>> {
        if self.current.as_ref().is_none_or(|video| video.span != span)
            && self.next.as_ref().is_some_and(|video| video.span == span)
        {
            self.current = self.next.take();
        }
        self.current.as_mut().filter(|video| video.span == span)
    }
}

impl<I: ByteSource, V: PrefetchVideoSource> PlaybackVideoSource for CuedVideoSource<I, V> {
    fn get_exact(
        &mut self,
        frame: FrameIndex,
        cancellation: &CancellationToken,
    ) -> Result<VideoFrame> {
        let span = self.spans.span_of(frame);
        let video = self.holding(span).ok_or_else(not_indexed)?;
        let local = video.local(frame)?;
        video.source.get_exact(local, cancellation)
    }

    fn reset(&mut self) -> Result<()> {
        for video in self.current.iter_mut().chain(self.next.iter_mut()) {
            video.source.reset()?;
        }
        Ok(())
    }
}

impl<I: ByteSource, V: PrefetchVideoSource> PrefetchVideoSource for CuedVideoSource<I, V> {
    fn prefetch(&mut self, frame: FrameIndex) -> IoFuture<'_, ()> {
        Box::pin(async move {
            let span = self.spans.span_of(frame);
            if self.holding(span).is_none() {
                let indexed = self.spans.span(span).await?;
                self.current = Some(self.build(&indexed)?);
            }
            let current = self.current.as_mut().expect("the span was just built");
            let local = current.local(frame)?;
            current.source.prefetch(local).await?;
            // Near the span's end, the next span's first frames are loaded
            // too, so playback crosses into it without waiting.
            let remaining = current.frames.len().saturating_sub(local.0 as usize);
            if remaining > VIDEO_READAHEAD_SAMPLES || span + 1 >= self.spans.index().span_count() {
                return Ok(());
            }
            if self.next.as_ref().is_none_or(|next| next.span != span + 1) {
                let indexed = self.spans.span(span + 1).await?;
                self.next = Some(self.build(&indexed)?);
            }
            let next = self.next.as_mut().expect("the next span was just built");
            next.source.prefetch(FrameIndex(0)).await
        })
    }
}

/// Silence for as long as a cued WebM's video lasts, standing in for the
/// audio of one with no audio track, as [`crate::on_demand::SilentAudioSource`]
/// does for an input indexed whole. How long the video lasts is only known
/// once its last span is indexed, and estimated until then.
pub(crate) struct CuedSilence<I> {
    pub(crate) spans: CuedSpans<I>,
    pub(crate) sample_rate: u32,
    pub(crate) limits: Limits,
}

impl<I: ByteSource> PlaybackAudioSource for CuedSilence<I> {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn presentation_length(&self) -> u64 {
        let exact = self.spans.last().and_then(|last| {
            let video = self.spans.video(&last).ok()?;
            let mut end = None;
            for interval in frame_intervals(video, self.sample_rate).ok()? {
                end = Some(interval.ok()?.1.end);
            }
            end
        });
        exact.unwrap_or_else(|| self.spans.estimated_length(self.sample_rate))
    }

    fn read(&mut self, range: SampleRange, _: &CancellationToken) -> Result<AudioBuffer> {
        AudioBuffer::new(
            range,
            self.sample_rate,
            1,
            vec![0.0; range.len() as usize],
            &self.limits,
        )
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

impl<I: ByteSource> PrefetchAudioSource for CuedSilence<I> {
    fn prefetch(&mut self, _: SampleRange) -> IoFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}

/// The packets of a cued WebM's Opus track an [`AudioSampleReader`] reads:
/// one contiguous run of spans' packets, which grows as playback goes on and
/// starts again where a seek lands; see the [module documentation](self).
pub(crate) struct AudioWindow<I, S> {
    spans: CuedSpans<I>,
    /// The track's metadata, without samples.
    track: Track,
    source: S,
    budget_bytes: u64,
    priming: u32,
    /// The presentation time of the track's first packet, which its packets'
    /// positions are counted from.
    first_pts: i64,
    /// How many packets a read decodes ahead of the first it needs.
    preroll: usize,
    first_span: usize,
    last_span: usize,
    /// The window's packets' decoded intervals, in decode order.
    ranges: Vec<SampleRange>,
    loader: TrackSampleLoader<S>,
    /// Whether the reader has the window's latest packets.
    current: bool,
    /// The presentation's length, once the window has reached the last span.
    length: Option<u64>,
}

impl<I: ByteSource, S: ByteSource + Clone> AudioWindow<I, S> {
    /// A window over the first span's packets of the audio track `track`,
    /// one of [`WebmCuedIndex::tracks`], loading at most `budget_bytes` of
    /// them at once from `source`. Returns the window and the packets and
    /// timing to build its reader with. Reads no packet but the track's last,
    /// when the file has one span, whose first bytes give its length.
    pub(crate) async fn open(
        spans: CuedSpans<I>,
        track: &Track,
        source: S,
        budget_bytes: u64,
    ) -> Result<(Self, PrefetchedAudioPacketProvider, AudioTrackTiming)> {
        let first = spans.span(0).await?;
        let first_pts = first
            .track(track.id)
            .and_then(|part| part.samples.first())
            .map(|sample| sample.pts)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unsupported,
                    "a WebM audio track with no packet in its first cue span cannot be played \
                     while its index is read",
                )
            })?;
        let priming = spans.index().audio_timing(track.id, None)?.priming;
        let ranges = spans
            .index()
            .opus_decoded_ranges(&*spans.source, &first, track.id, first_pts)
            .await?;
        // As many packets as cover the Opus preroll at the first span's
        // shortest, which in an ordinary stream is every packet's length.
        let shortest = ranges
            .iter()
            .map(|range| range.len())
            .filter(|&length| length > 0)
            .min()
            .unwrap_or(u64::from(OPUS_PREROLL_SAMPLES));
        let preroll = usize::try_from(u64::from(OPUS_PREROLL_SAMPLES).div_ceil(shortest))
            .unwrap_or(usize::MAX);
        let part = first.track(track.id).expect("checked above").clone();
        let loader = TrackSampleLoader::new(part, source.clone(), budget_bytes)?;
        let mut window = Self {
            spans,
            track: track.clone(),
            source,
            budget_bytes,
            priming,
            first_pts,
            preroll,
            first_span: 0,
            last_span: 0,
            ranges,
            loader,
            current: true,
            length: None,
        };
        let (packets, timing) = window.packets()?;
        Ok((window, packets, timing))
    }

    pub(crate) fn loader(&self) -> &TrackSampleLoader<S> {
        &self.loader
    }

    pub(crate) fn preroll(&self) -> usize {
        self.preroll
    }

    /// The presentation's length: exact once the window has reached the last
    /// span, and estimated until then.
    pub(crate) fn presentation_length(&self) -> u64 {
        self.length
            .unwrap_or_else(|| self.spans.estimated_length(crate::opus::OPUS_SAMPLE_RATE))
    }

    fn window_end(&self) -> u64 {
        self.ranges.last().map_or(0, |range| range.end)
    }

    /// How many of the window's packets end by `media`.
    fn packets_before(&self, media: u64) -> usize {
        self.ranges.partition_point(|range| range.end <= media)
    }

    /// Whether a read of `media` can go on in the window as it is: it starts
    /// in it, with the packets a read that resets the decoder decodes ahead of
    /// its first, unless the window starts with the track.
    fn holds_start(&self, media: u64) -> bool {
        self.ranges
            .first()
            .is_some_and(|first| first.start <= media && media <= self.window_end())
            && (self.first_span == 0 || self.packets_before(media) >= self.preroll)
    }

    /// Whether `reader` reads `range` of the presentation from the window as
    /// it is, without [`Self::cover`] moving or growing it.
    #[cfg(all(any(unix, windows), not(target_arch = "wasm32")))]
    pub(crate) fn covers<D: AudioDecoder>(
        &self,
        range: SampleRange,
        reader: &AudioSampleReader<D>,
    ) -> bool {
        self.current
            && self.holds_start(range.start + u64::from(self.priming))
            && range.end <= reader.presentation_length()
    }

    /// Moves or grows the window so `reader` reads `range` of the
    /// presentation from it, indexing the spans that takes, and hands the
    /// reader the window's packets if they changed.
    pub(crate) async fn cover<D: AudioDecoder>(
        &mut self,
        range: SampleRange,
        reader: &mut AudioSampleReader<D>,
    ) -> Result<()> {
        let media_start = range.start + u64::from(self.priming);
        let media_end = range.end + u64::from(self.priming);
        if !self.holds_start(media_start) {
            self.restart(media_start).await?;
        }
        let count = self.spans.index().span_count();
        while self.window_end() < media_end && self.last_span + 1 < count {
            self.extend().await?;
        }
        if !self.current {
            let (packets, timing) = self.packets()?;
            reader.replace_packets(Box::new(packets), timing)?;
            self.current = true;
        }
        Ok(())
    }

    /// The window's packets for its reader and the timing they play with,
    /// refusing a budget that cannot hold one of them with its preroll, as
    /// [`crate::on_demand::audio_packets`] refuses one for a whole track.
    fn packets(&mut self) -> Result<(PrefetchedAudioPacketProvider, AudioTrackTiming)> {
        let packets = self
            .loader
            .opus_packet_provider_with_ranges(self.ranges.clone())?;
        self.loader.check_audio_budget(&packets, self.preroll)?;
        let last = (self.last_span + 1 == self.spans.index().span_count())
            .then(|| self.spans.last())
            .flatten();
        let timing = self
            .spans
            .index()
            .audio_timing(self.track.id, last.as_deref())?;
        if last.is_some() {
            self.length = Some(
                self.window_end()
                    .saturating_sub(u64::from(timing.priming) + u64::from(timing.padding)),
            );
        }
        Ok((packets, timing))
    }

    /// `span`'s samples of the track and their decoded intervals.
    async fn span_packets(&self, span: usize) -> Result<(Track, Vec<SampleRange>)> {
        let indexed = self.spans.span(span).await?;
        let part = indexed
            .track(self.track.id)
            .cloned()
            .ok_or_else(|| Error::new(ErrorKind::Internal, "a cue span lost an audio track"))?;
        let ranges = self
            .spans
            .index()
            .opus_decoded_ranges(&*self.spans.source, &indexed, self.track.id, self.first_pts)
            .await?;
        Ok((part, ranges))
    }

    /// Adds the span after the window's last.
    async fn extend(&mut self) -> Result<()> {
        let (part, ranges) = self.span_packets(self.last_span + 1).await?;
        self.loader.append(&part.samples)?;
        self.ranges.extend(ranges);
        self.last_span += 1;
        self.current = false;
        Ok(())
    }

    /// Starts a window over the spans holding `media` and the packets a read
    /// from it decodes ahead of it, with a loader of its own.
    async fn restart(&mut self, media: u64) -> Result<()> {
        let index = self.spans.index();
        let ticks = self.first_pts.saturating_add(
            i64::try_from(
                media * u64::from(index.timescale()) / u64::from(crate::opus::OPUS_SAMPLE_RATE),
            )
            .unwrap_or(i64::MAX),
        );
        let count = index.span_count();
        let mut first = index.span_at(ticks);
        let (samples, ranges, last) = loop {
            let (part, ranges) = self.span_packets(first).await?;
            let mut samples = part.samples;
            let mut ranges = ranges;
            let mut last = first;
            // A packet is in the span its block is stored in, which may be a
            // later span than its time suggests.
            while ranges.last().is_none_or(|range| range.end <= media) && last + 1 < count {
                last += 1;
                let (part, more) = self.span_packets(last).await?;
                samples.extend(part.samples);
                ranges.extend(more);
            }
            let before = ranges.partition_point(|range| range.end <= media);
            let starts_before = ranges.first().is_some_and(|range| range.start <= media);
            // Past the track's last packet, the window holds the packets
            // before it, from which its length is known.
            let past_the_end = last + 1 == count && !ranges.is_empty() && before == ranges.len();
            if first == 0 || past_the_end || (starts_before && before >= self.preroll) {
                break (samples, ranges, last);
            }
            first -= 1;
        };
        self.loader = TrackSampleLoader::new(
            Track {
                presentation_order: (0..samples.len()).collect(),
                samples,
                ..self.track.clone()
            },
            self.source.clone(),
            self.budget_bytes,
        )?;
        self.first_span = first;
        self.last_span = last;
        self.ranges = ranges;
        self.current = false;
        Ok(())
    }
}

/// The packets an audio track's reader reads: the whole track's, indexed when
/// the input opened, or a cued WebM's window of them.
pub(crate) enum AudioPackets<I, S> {
    Whole(TrackSampleLoader<S>),
    Cued(Box<AudioWindow<I, S>>),
}

impl<I: ByteSource, S: ByteSource + Clone> AudioPackets<I, S> {
    pub(crate) fn loader(&self) -> &TrackSampleLoader<S> {
        match self {
            Self::Whole(loader) => loader,
            Self::Cued(window) => window.loader(),
        }
    }

    pub(crate) fn presentation_length<D: AudioDecoder>(
        &self,
        reader: &AudioSampleReader<D>,
    ) -> u64 {
        match self {
            Self::Whole(_) => reader.presentation_length(),
            Self::Cued(window) => window.presentation_length(),
        }
    }

    /// Whether `reader` reads `range` from what it has now.
    #[cfg(all(any(unix, windows), not(target_arch = "wasm32")))]
    pub(crate) fn covers<D: AudioDecoder>(
        &self,
        range: SampleRange,
        reader: &AudioSampleReader<D>,
    ) -> bool {
        match self {
            Self::Whole(_) => true,
            Self::Cued(window) => window.covers(range, reader),
        }
    }

    /// Makes `reader` read `range`, and as far again as `readahead` past it,
    /// from what it has, indexing whatever that takes.
    pub(crate) async fn cover<D: AudioDecoder>(
        &mut self,
        range: SampleRange,
        readahead: u64,
        reader: &mut AudioSampleReader<D>,
    ) -> Result<()> {
        let Self::Cued(window) = self else {
            return Ok(());
        };
        let end = range
            .end
            .saturating_add(readahead)
            .min(window.presentation_length())
            .max(range.end);
        window
            .cover(SampleRange::new(range.start, end)?, reader)
            .await
    }
}

/// An audio track's presentation samples, decoded from packets loaded on
/// demand, from the whole track's index or a cued WebM's window of it: the
/// [`crate::OnDemandAudioSource`] of [`crate::OnDemandPlayer`].
#[cfg(all(any(unix, windows), not(target_arch = "wasm32")))]
pub(crate) struct PacketAudioSource<D, I, S> {
    reader: AudioSampleReader<D>,
    packets: AudioPackets<I, S>,
    readahead_packets: usize,
}

#[cfg(all(any(unix, windows), not(target_arch = "wasm32")))]
impl<D: AudioDecoder, I: ByteSource, S: ByteSource + Clone> PacketAudioSource<D, I, S> {
    pub(crate) fn new(
        reader: AudioSampleReader<D>,
        packets: AudioPackets<I, S>,
        readahead_packets: usize,
    ) -> Self {
        Self {
            reader,
            packets,
            readahead_packets,
        }
    }

    pub(crate) fn loader(&self) -> &TrackSampleLoader<S> {
        self.packets.loader()
    }
}

#[cfg(all(any(unix, windows), not(target_arch = "wasm32")))]
impl<D: AudioDecoder, I: ByteSource, S: ByteSource + Clone> PlaybackAudioSource
    for PacketAudioSource<D, I, S>
{
    fn sample_rate(&self) -> u32 {
        self.reader.sample_rate()
    }

    fn presentation_length(&self) -> u64 {
        self.packets.presentation_length(&self.reader)
    }

    fn read(
        &mut self,
        range: SampleRange,
        cancellation: &CancellationToken,
    ) -> Result<AudioBuffer> {
        if !self.packets.covers(range, &self.reader) {
            return Err(not_indexed());
        }
        self.reader.get_range(range, cancellation)
    }

    fn reset(&mut self) -> Result<()> {
        self.packets.loader().clear_missing();
        self.reader.reset()
    }
}

#[cfg(all(any(unix, windows), not(target_arch = "wasm32")))]
impl<D: AudioDecoder, I: ByteSource, S: ByteSource + Clone> PrefetchAudioSource
    for PacketAudioSource<D, I, S>
{
    fn prefetch(&mut self, range: SampleRange) -> IoFuture<'_, ()> {
        Box::pin(async move {
            let readahead = u64::from(self.reader.sample_rate());
            self.packets
                .cover(range, readahead, &mut self.reader)
                .await?;
            let loader = self.packets.loader();
            loader.load_missing().await?;
            // Covering the range may have found the presentation shorter than
            // it was estimated to be when the range was asked for.
            let end = range
                .end
                .min(self.packets.presentation_length(&self.reader));
            if range.start >= end {
                return Ok(());
            }
            for run in self
                .reader
                .packets_for_range(SampleRange::new(range.start, end)?)?
            {
                loader.load(run, self.readahead_packets).await?;
            }
            Ok(())
        })
    }
}
