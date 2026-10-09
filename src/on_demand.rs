//! Opening an MP4 or WebM for on-demand playback, shared by the native
//! [`crate::OnDemandPlayer`] and the browser's `OnDemandPlayback` (issues #689
//! and #685).
//!
//! Both open the input through `crate::container::open_media`, whichever
//! container it is, and read only its header and index up front and then the
//! compressed samples playback reaches, through a [`TrackSampleLoader`] per
//! track bounded by a byte budget. What they share is how a track becomes a source the
//! [`crate::PlaybackController`] plays: which of the crate's decoders a video
//! track opens on and with what configuration, how an audio track's packets
//! are indexed and how many of them a read decodes ahead, and the silence that
//! stands in for an input with no audio track.

use crate::audio::AudioPacketProvider;
use crate::av1::{Av1CodecConfigurationRecord, Av1Obu, Av1Parser};
use crate::codec::{
    CancellationToken, CodecProfile, EncodedVideoSample, ExactFrameReader, HardwarePreference,
    VideoDecoderConfig, VideoDecoderFactory,
};
use crate::codec_config::derive_codec_string;
use crate::io::{ByteSource, IoFuture};
use crate::media::{AudioBuffer, Codec, ColorRange, PixelFormat, VideoDimensions};
use crate::playback::{OnDemandVideoSource, PlaybackAudioSource, PrefetchAudioSource};
use crate::timeline::{FrameIndex, SampleRange};
use crate::track::Track;
use crate::{
    Error, ErrorKind, Limits, OPUS_PREROLL_SAMPLES, PrefetchedAudioPacketProvider, Result,
    TrackSampleLoader, VORBIS_PREROLL_PACKETS,
};

/// The compressed video a playback may hold when the caller does not say:
/// several seconds of a 1080p stream, and room for its largest key frames.
pub(crate) const DEFAULT_VIDEO_BUDGET_BYTES: u64 = 16 * 1024 * 1024;

/// The compressed audio a playback may hold when the caller does not say: well
/// over a minute of ordinary AAC.
pub(crate) const DEFAULT_AUDIO_BUDGET_BYTES: u64 = 1024 * 1024;

/// Decode-order video samples a prefetch loads past the ones the current frame
/// needs, so the frames after it play without reporting anything missing.
pub(crate) const VIDEO_READAHEAD_SAMPLES: usize = 24;

/// Audio packets a prefetch loads past the ones it decodes.
pub(crate) const AUDIO_READAHEAD_PACKETS: usize = 16;

/// The ticks per second of the clock that times an input with no audio track,
/// standing in for an audio sample rate.
pub(crate) const VIDEO_ONLY_CLOCK_RATE: u32 = 48_000;

/// How many AAC access units an audio read decodes ahead of the first it
/// needs. Each AAC frame overlaps the one before it, so one is enough; the
/// second covers the decoder's own start-up.
pub(crate) const AAC_PREROLL_PACKETS: usize = 2;

/// `video` decoded on the crate's own decoder from samples a fresh
/// [`TrackSampleLoader`] of `budget_bytes` loads on demand from `source`.
///
/// Reads only an AV1 or VP9 track's first sample, whose color range the
/// decoder is configured with when the track's configuration record does not
/// say.
pub(crate) async fn crate_video_source<S: ByteSource>(
    video: &Track,
    source: S,
    budget_bytes: u64,
    hardware: HardwarePreference,
    limits: Limits,
) -> Result<OnDemandVideoSource<S>> {
    // Before any read, so a budget that cannot hold the largest sample is
    // refused without one.
    let loader = TrackSampleLoader::new(video.clone(), source, budget_bytes)?;
    let decoding = VideoDecoding::open(video, loader.source(), hardware, limits).await?;
    decoding.source(loader)
}

/// The crate's decoder for a video track and the configuration it decodes the
/// track with, chosen once when the input opens, from which a reader is built
/// over any loader of the track's samples: the whole track's, or one cue
/// span's of a lazily indexed WebM (issue #692).
pub(crate) struct VideoDecoding {
    factory: Box<dyn VideoDecoderFactory>,
    configuration: VideoDecoderConfig,
    limits: Limits,
}

impl VideoDecoding {
    /// The decoder for `video`, whose first sample - read from `source`, for
    /// an AV1 or VP9 track only - is the stream's first.
    pub(crate) async fn open<S: ByteSource>(
        video: &Track,
        source: &S,
        hardware: HardwarePreference,
        limits: Limits,
    ) -> Result<Self> {
        let dimensions = video.dimensions.ok_or_else(|| {
            Error::new(ErrorKind::MalformedMedia, "video track has no dimensions")
        })?;
        let derived = derive_codec_string(video.codec, &video.decoder_config)?;
        let mut leading = Vec::new();
        if matches!(video.codec, Codec::Av1 | Codec::Vp9) {
            let first = video.samples.first().ok_or_else(|| {
                Error::new(ErrorKind::MalformedMedia, "video track contains no samples")
            })?;
            let mut data = vec![0_u8; first.size as usize];
            video.read_sample_into(source, 0, &mut data).await?;
            leading.push(EncodedVideoSample {
                presentation_index: FrameIndex(0),
                random_access: first.is_sync,
                data,
            });
        }
        let (factory, configuration) = crate_video_decoder(
            video,
            derived.profile,
            dimensions,
            &leading,
            hardware,
            &limits,
        )?;
        Ok(Self {
            factory,
            configuration,
            limits,
        })
    }

    /// The track `loader` loads, decoded on demand.
    pub(crate) fn source<S: ByteSource>(
        &self,
        loader: TrackSampleLoader<S>,
    ) -> Result<OnDemandVideoSource<S>> {
        let reader = ExactFrameReader::from_provider(
            self.factory.as_ref(),
            self.configuration.clone(),
            Box::new(loader.sample_provider()?),
            self.limits,
        )?;
        Ok(OnDemandVideoSource::new(
            reader,
            loader,
            VIDEO_READAHEAD_SAMPLES,
        ))
    }
}

/// The packets of the loader's AAC, Opus or Vorbis track, indexed without
/// reading them, and how many of them a read decodes ahead of the first it
/// needs. Reads no packet but an Opus track's last, whose first bytes give its
/// length; a Vorbis track's intervals come from the first bytes the WebM
/// demuxer recorded. Fails with [`ErrorKind::ResourceLimit`] if the loader's
/// budget cannot hold a packet together with its preroll packets.
pub(crate) async fn audio_packets<S: ByteSource>(
    loader: &TrackSampleLoader<S>,
) -> Result<(PrefetchedAudioPacketProvider, usize)> {
    let (packets, preroll) = match loader.track().codec {
        Codec::Opus => {
            let packets = loader.opus_packet_provider().await?;
            let preroll = opus_preroll_packets(&packets);
            (packets, preroll)
        }
        Codec::Aac => (loader.aac_packet_provider()?, AAC_PREROLL_PACKETS),
        Codec::Vorbis => (loader.vorbis_packet_provider()?, VORBIS_PREROLL_PACKETS),
        _ => {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "on-demand playback supports AAC, Opus and Vorbis audio tracks",
            ));
        }
    };
    // A budget that cannot hold a packet with its preroll would load the same
    // prefix of a read's packets on every prefetch, forever (issue #694).
    loader.check_audio_budget(&packets, preroll)?;
    Ok((packets, preroll))
}

/// How many packets an Opus read decodes ahead of the first one it needs:
/// enough to cover [`OPUS_PREROLL_SAMPLES`] even at the track's shortest
/// packet, as [`crate::opus_preroll_packets`] counts for packets in memory.
fn opus_preroll_packets(packets: &PrefetchedAudioPacketProvider) -> usize {
    let shortest = (0..packets.len())
        .map(|index| packets.decoded_range(index).len())
        .filter(|&length| length > 0)
        .min()
        .unwrap_or(u64::from(OPUS_PREROLL_SAMPLES));
    usize::try_from(u64::from(OPUS_PREROLL_SAMPLES).div_ceil(shortest)).unwrap_or(usize::MAX)
}

/// Silence for as long as the video lasts, standing in for the audio of an
/// input that has none, so the controller's clock runs over the whole video.
pub(crate) struct SilentAudioSource {
    pub(crate) sample_rate: u32,
    pub(crate) length: u64,
    pub(crate) limits: Limits,
}

impl PlaybackAudioSource for SilentAudioSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn presentation_length(&self) -> u64 {
        self.length
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

impl PrefetchAudioSource for SilentAudioSource {
    fn prefetch(&mut self, _: SampleRange) -> IoFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}

/// The crate's own decoder for `codec`.
fn decoder_factory(codec: Codec) -> Result<Box<dyn VideoDecoderFactory>> {
    match codec {
        #[cfg(feature = "hevc-decoder")]
        Codec::Hevc => Ok(Box::new(crate::native_hevc_video_decoder_factory())),
        #[cfg(feature = "av1-decoder")]
        Codec::Av1 => Ok(Box::new(crate::native_av1_video_decoder_factory())),
        #[cfg(feature = "vp8-decoder")]
        Codec::Vp8 => Ok(Box::new(crate::native_vp8_video_decoder_factory())),
        #[cfg(feature = "vp9-decoder")]
        Codec::Vp9 => Ok(Box::new(crate::native_vp9_video_decoder_factory())),
        // Uncompressed video, H.264, the audio codecs, any codec a later
        // zvidlib-core adds, and a decoder whose Cargo feature is off.
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "only HEVC, AV1, VP8 and VP9 have a decoder in this crate, each with its \
             <codec>-decoder Cargo feature",
        )),
    }
}

/// The crate's own decoder for `track` and the configuration it decodes
/// the track with.
///
/// `profile` is the profile the track's configuration box names, so an HEVC
/// Main 10 track is opened as Main 10 (issue #508) and a VP9 track as the VP9
/// profile its `vpcC` declares. `samples` are the track's leading samples in
/// decode order: an AV1 or VP9 track's color range is read from the first one
/// when its configuration record does not say, so a caller that loads samples
/// on demand need only pass that one. `hardware` is the preference the
/// decoder is opened with: the browser avoids a platform backend, which it
/// has none of, while native playback prefers one where the codec has it.
pub(crate) fn crate_video_decoder(
    track: &Track,
    profile: CodecProfile,
    dimensions: VideoDimensions,
    samples: &[EncodedVideoSample],
    hardware: HardwarePreference,
    limits: &Limits,
) -> Result<(Box<dyn VideoDecoderFactory>, VideoDecoderConfig)> {
    let factory = decoder_factory(track.codec)?;
    // The HEVC decoder only accepts limited-range input, while the AV1
    // and VP9 decoders report whatever range the stream signals and the
    // reader holds every frame to the configured one.
    let (profile, color_range) = match track.codec {
        Codec::Av1 => (
            CodecProfile::Av1Main,
            av1_color_range(track, samples, limits),
        ),
        Codec::Vp9 => (profile, vp9_color_range(track, samples)),
        _ => (profile, ColorRange::Limited),
    };
    // The decoders validate the configuration record itself, so a stream
    // they cannot decode (color AV1, say) is refused here, at open,
    // rather than on its first frame.
    let configuration = VideoDecoderConfig {
        codec: track.codec,
        profile,
        coded_dimensions: dimensions,
        output_format: PixelFormat::Rgba8,
        color_range,
        hardware,
        configuration: track.decoder_config.clone(),
    };
    Ok((factory, configuration))
}

/// The color range an AV1 track's sequence header signals: from `av1C`'s
/// `configOBUs` when it carries one, which it need not, and otherwise from the
/// first sample, a key frame that must. Limited when neither parses, which
/// leaves the reader to reject the first frame that disagrees.
fn av1_color_range(track: &Track, samples: &[EncodedVideoSample], limits: &Limits) -> ColorRange {
    let from_config = Av1CodecConfigurationRecord::parse(&track.decoder_config, limits)
        .ok()
        .and_then(|record| {
            record.config_obus.into_iter().find_map(|obu| match obu {
                Av1Obu::SequenceHeader { sequence, .. } => Some(sequence),
                _ => None,
            })
        });
    let sequence = from_config.or_else(|| {
        let mut parser = Av1Parser::new(*limits).ok()?;
        parser.parse_low_overhead(&samples.first()?.data).ok()?;
        parser.sequence
    });
    match sequence {
        Some(sequence) if sequence.color_config.color_range => ColorRange::Full,
        _ => ColorRange::Limited,
    }
}

/// The color range a VP9 track's first key frame signals, which is what
/// its pictures are decoded in. The `vpcC` box's `videoFullRangeFlag` stands
/// in when the first sample does not parse, and limited range when neither
/// says, which leaves the reader to reject the first frame that disagrees.
fn vp9_color_range(track: &Track, samples: &[EncodedVideoSample]) -> ColorRange {
    let full = samples
        .first()
        .and_then(|sample| zvidlib_vp9_syntax::chunk_full_range(&sample.data))
        .or_else(|| {
            crate::Vp9CodecConfig::parse(&track.decoder_config)
                .ok()
                .map(|config| config.video_full_range)
        });
    if full == Some(true) {
        ColorRange::Full
    } else {
        ColorRange::Limited
    }
}
