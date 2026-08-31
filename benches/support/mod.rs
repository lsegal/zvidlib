//! Shared fixture loading, synthetic inputs, and throughput reporting for the
//! criterion benchmark suite.
//!
//! Everything expensive here is decoded or demuxed exactly once per process and
//! cached in a `OnceLock`, so a benchmark's per-iteration cost is codec work and
//! nothing else. Fixtures are the ones already checked into the repository: the
//! bundled `examples/media/BigBuckBunny.mp4` HEVC Main sample and the AV1
//! elementary streams under `tests/fixtures/codec/`.
//!
//! [`isa`] adds the scalar-vs-SIMD axis on top: it runs a workload once per
//! instruction set `zvidlib::simd::available()` reports, asserts every arm is
//! bit-exact with scalar before timing it, and names the arms `<codec>/<isa>`.

// Each bench target (`benches/codec.rs`, `benches/av1_decode.rs`,
// `benches/audio_decode.rs`, `benches/audio_mux.rs`, `benches/hevc_encode.rs`)
// is its own crate root and compiles this whole module, but uses only the
// fixtures its own measurements need. Unused-here is not dead:
// `cargo clippy --all-targets` would otherwise fail one target for helpers
// another target depends on.
#![allow(dead_code)]

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;
use std::task::{Context, Poll, Waker};

pub mod isa;

use criterion::Throughput;
use zvidlib::io::MemorySource;
use zvidlib::{
    AacTrackConfig, AudioTrackTiming, Codec, CodecProfile, ColorRange, EncodedAudioSample,
    EncodedVideoSample, FilterPlane, Limits, Mp4Demuxer, Mp4DemuxerOptions, Mp4Track, PixelFormat,
    Plane, TrackKind, TxSizeGrid, VideoDecoderConfig, VideoDimensions, VideoFrame,
    decode_av1_lossless_intra,
};

/// Whether the crate was built with the additive `simd` cargo feature.
///
/// The crate's vector kernels are selected by runtime CPU feature detection, so
/// this feature gates no code today. It exists so a benchmark run records which
/// arm it measured; the per-codec tickets that add compile-time SIMD paths hang
/// them off the same switch.
pub fn simd_enabled() -> bool {
    cfg!(feature = "simd")
}

/// The `simd=on` / `simd=off` suffix every criterion group name carries.
pub fn simd_tag() -> &'static str {
    if simd_enabled() {
        "simd=on"
    } else {
        "simd=off"
    }
}

/// Builds a criterion group name of the form `hevc_decode/simd=off`.
pub fn group_name(stage: &str) -> String {
    format!("{stage}/{}", simd_tag())
}

/// The amount of pixel work one benchmark iteration performs.
///
/// Wall time alone is not comparable across fixtures of different resolutions,
/// so benchmarks report `Throughput::Elements(frames)` (criterion prints
/// `elem/s`, i.e. frames per second) together with the megapixels each frame
/// carries, which converts that rate to megapixels per second.
#[derive(Clone, Copy, Debug)]
pub struct FrameWork {
    pub frames: u64,
    pub width: u64,
    pub height: u64,
}

impl FrameWork {
    pub fn new(frames: u64, width: u64, height: u64) -> Self {
        Self {
            frames,
            width,
            height,
        }
    }

    /// Frames per iteration, which criterion turns into a frames/sec rate.
    pub fn elements(&self) -> Throughput {
        Throughput::Elements(self.frames)
    }

    /// Megapixels touched per iteration.
    pub fn megapixels(&self) -> f64 {
        (self.frames * self.width * self.height) as f64 / 1e6
    }

    /// Megapixels per second for a measured per-iteration duration.
    pub fn megapixels_per_second(&self, per_iteration: std::time::Duration) -> f64 {
        let seconds = per_iteration.as_secs_f64();
        if seconds <= 0.0 {
            return 0.0;
        }
        self.megapixels() / seconds
    }
}

/// Registers a benchmark's throughput and prints the megapixel scale criterion's
/// own `elem/s` line does not carry.
///
/// `frames/s * megapixels-per-frame = megapixels/s`, so printing the factor once
/// per benchmark makes every reported rate convertible without re-deriving each
/// fixture's resolution.
pub fn report_throughput<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    id: &str,
    work: FrameWork,
) {
    group.throughput(work.elements());
    println!(
        "# {id}: {} frame(s)/iter at {}x{} = {:.4} Mpx/iter (frames/s x {:.4} = Mpx/s)",
        work.frames,
        work.width,
        work.height,
        work.megapixels(),
        work.megapixels() / work.frames.max(1) as f64,
    );
}

/// Minimal executor for the crate's `async` I/O entry points.
pub fn block_on<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = Box::pin(future);
    loop {
        match Pin::new(&mut future).poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn from_hex(hex: &str) -> Vec<u8> {
    let hex = hex.trim();
    assert_eq!(hex.len() & 1, 0, "hex fixture must have an even length");
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16)
                .expect("hex fixture must be valid hex")
        })
        .collect()
}

/// The standardized AV1 Main lossless monochrome intra vector (17x9).
pub fn av1_lossless_intra_stream() -> &'static [u8] {
    static STREAM: OnceLock<Vec<u8>> = OnceLock::new();
    STREAM.get_or_init(|| {
        from_hex(include_str!(
            "../../tests/fixtures/codec/av1_lossless_17x9.hex"
        ))
    })
}

/// The standardized AV1 inter + `show_existing_frame` vector (16x16).
pub fn av1_inter_stream() -> &'static [u8] {
    static STREAM: OnceLock<Vec<u8>> = OnceLock::new();
    STREAM.get_or_init(|| {
        from_hex(include_str!(
            "../../tests/fixtures/codec/av1_inter_show_existing_16x16.hex"
        ))
    })
}

/// Byte ranges of the temporal units in [`av1_inter_stream`], split at each
/// temporal-delimiter OBU (`obu_type == 2`).
pub fn av1_inter_temporal_units() -> &'static [std::ops::Range<usize>] {
    static UNITS: OnceLock<Vec<std::ops::Range<usize>>> = OnceLock::new();
    UNITS.get_or_init(|| {
        let stream = av1_inter_stream();
        let mut starts = Vec::new();
        let mut cursor = 0usize;
        while cursor < stream.len() {
            let start = cursor;
            let header = stream[cursor];
            cursor += 1;
            let obu_type = (header >> 3) & 0x0f;
            assert_ne!(header & 0x02, 0, "fixture OBU must carry a size field");
            let mut payload_len = 0usize;
            let mut shift = 0usize;
            loop {
                let byte = stream[cursor];
                cursor += 1;
                payload_len |= usize::from(byte & 0x7f) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            cursor += payload_len;
            assert!(cursor <= stream.len(), "fixture OBU length is in bounds");
            if obu_type == 2 {
                starts.push(start);
            }
        }
        starts
            .iter()
            .enumerate()
            .map(|(index, &start)| {
                let end = starts.get(index + 1).copied().unwrap_or(stream.len());
                start..end
            })
            .collect()
    })
}

/// The AV1 intra vector decoded once, for benchmarks that consume decoded planes
/// rather than measuring the decode itself.
pub fn av1_lossless_intra_frame() -> &'static VideoFrame {
    static FRAME: OnceLock<VideoFrame> = OnceLock::new();
    FRAME.get_or_init(|| {
        decode_av1_lossless_intra(av1_lossless_intra_stream(), &Limits::default())
            .expect("the checked-in AV1 intra vector decodes")
    })
}

/// The bundled 1920x1080 HEVC Main sample, demuxed once into decoder-ready
/// samples and a matching decoder configuration.
pub struct BundledHevcSample {
    pub configuration: VideoDecoderConfig,
    pub samples: Vec<EncodedVideoSample>,
    pub width: u64,
    pub height: u64,
}

/// Demuxes `examples/media/BigBuckBunny.mp4` once per process.
///
/// This is the only fixture large enough to matter: it is ~3 MB of bitstream and
/// 768 presentation frames, so it belongs to the long-running benchmark group
/// rather than the default fast path.
pub fn bundled_hevc_sample() -> &'static BundledHevcSample {
    static SAMPLE: OnceLock<BundledHevcSample> = OnceLock::new();
    SAMPLE.get_or_init(|| {
        let limits = Limits::default();
        let source =
            MemorySource::new(include_bytes!("../../examples/media/BigBuckBunny.mp4").to_vec());
        let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))
            .expect("the bundled sample is a readable MP4");
        let track = movie.track(1).expect("the bundled sample has track 1");
        let dimensions = track
            .dimensions
            .expect("the bundled sample's video track is dimensioned");
        let samples = block_on(track.to_encoded_video_samples(&source, &limits))
            .expect("the bundled sample's video samples are readable");
        BundledHevcSample {
            configuration: VideoDecoderConfig {
                codec: Codec::Hevc,
                profile: CodecProfile::HevcMain,
                coded_dimensions: dimensions,
                output_format: PixelFormat::Rgba8,
                color_range: ColorRange::Limited,
                // Benchmarks measure the crate's own decode work, not whichever
                // fixed-function block the host happens to ship.
                hardware: zvidlib::HardwarePreference::Avoid,
                configuration: track.decoder_config.clone(),
            },
            samples,
            width: u64::from(dimensions.width),
            height: u64::from(dimensions.height),
        }
    })
}

/// Builds a deterministic synthetic YUV420 frame sequence for encoder inputs.
///
/// Encoder benchmarks need frames without paying for a decode first, and the
/// content has a moving gradient plus low-amplitude noise so neither prediction
/// nor entropy coding degenerates into an unrepresentative best case.
pub fn synthetic_yuv420_sequence(width: u32, height: u32, frames: usize) -> Vec<VideoFrame> {
    let limits = Limits::default();
    let dimensions =
        VideoDimensions::new(width, height, &limits).expect("synthetic dimensions are valid");
    let (luma_w, luma_h) = (width as usize, height as usize);
    let (chroma_w, chroma_h) = (luma_w.div_ceil(2), luma_h.div_ceil(2));
    (0..frames)
        .map(|frame| {
            let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ frame as u64;
            let mut next_noise = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 58) as i32
            };
            let shift = (frame * 3) as i32;
            let luma = (0..luma_h)
                .flat_map(|y| (0..luma_w).map(move |x| (x, y)))
                .map(|(x, y)| {
                    let gradient = (x as i32 + y as i32 + shift) / 2;
                    ((gradient + next_noise()) & 0xff) as u8
                })
                .collect::<Vec<_>>();
            let chroma = |offset: i32| {
                (0..chroma_h)
                    .flat_map(|y| (0..chroma_w).map(move |x| (x, y)))
                    .map(|(x, y)| (128 + ((x as i32 - y as i32 + shift + offset) % 24) - 12) as u8)
                    .collect::<Vec<_>>()
            };
            VideoFrame::new(
                dimensions,
                PixelFormat::Yuv420p8,
                ColorRange::Limited,
                vec![
                    Plane {
                        data: luma,
                        stride: luma_w,
                    },
                    Plane {
                        data: chroma(0),
                        stride: chroma_w,
                    },
                    Plane {
                        data: chroma(7),
                        stride: chroma_w,
                    },
                ],
                &limits,
            )
            .expect("synthetic YUV420 frames are valid")
        })
        .collect()
}

// ---------------------------------------------------------------------------
// AV1 software decoder fixtures.
//
// The AV1 groups need two kinds of input the fixtures above do not provide: a
// whole AV1 stream at a resolution where per-frame overhead is negligible, and
// the structured synthetic planes the kernel-level measurements run on. The
// checked-in AV1 vectors are 17x9 and 16x16 conformance streams — correct, but
// far too small to time a decoder with — so the stream below is produced by the
// crate's own AV1 encoder once per process and the planes are generated.
// ---------------------------------------------------------------------------

/// Luma width of [`synthetic_av1_stream`].
pub const AV1_STREAM_WIDTH: u32 = 320;
/// Luma height of [`synthetic_av1_stream`].
pub const AV1_STREAM_HEIGHT: u32 = 180;
/// Frames in [`synthetic_av1_stream`].
///
/// Enough that per-call decoder setup is a small share of one iteration, while
/// keeping a criterion sample well under a second on the software decoder.
pub const AV1_STREAM_FRAMES: usize = 8;

/// An AV1 elementary stream and the decoder configuration that decodes it.
pub struct SyntheticAv1Stream {
    pub configuration: VideoDecoderConfig,
    pub samples: Vec<EncodedVideoSample>,
    pub width: u64,
    pub height: u64,
}

/// Encodes a deterministic monochrome sequence with the crate's native AV1
/// encoder, once per process.
///
/// The native AV1 decoder implements the bounded Main-profile 8-bit lossless
/// monochrome subset its module documents, so a "representative stream" for it
/// is one this crate's own encoder produces. Encoding is hoisted here — a
/// `OnceLock` outside every timed loop — so a decode benchmark measures decode
/// and nothing else.
pub fn synthetic_av1_stream() -> &'static SyntheticAv1Stream {
    use zvidlib::{
        CpuFrameSource, FrameIndex, FrameSource, HardwarePreference, Orientation, VideoDimensions,
        VideoEncoderConfig, VideoEncoderFactory, native_av1_video_encoder_factory,
    };

    static STREAM: OnceLock<SyntheticAv1Stream> = OnceLock::new();
    STREAM.get_or_init(|| {
        let limits = Limits::default();
        let dimensions = VideoDimensions::new(AV1_STREAM_WIDTH, AV1_STREAM_HEIGHT, &limits)
            .expect("the synthetic AV1 dimensions are valid");
        let mut encoder = native_av1_video_encoder_factory()
            .create(
                &VideoEncoderConfig {
                    codec: Codec::Av1,
                    profile: CodecProfile::Av1Main,
                    coded_dimensions: dimensions,
                    input_format: PixelFormat::Gray8,
                    color_range: ColorRange::Full,
                    hardware: HardwarePreference::Avoid,
                    timescale: 30,
                    frame_duration: 1,
                    configuration: Vec::new(),
                },
                &limits,
            )
            .expect("the native AV1 encoder is constructible");

        let mut packets = Vec::new();
        for (index, luma) in
            av1_gray8_planes(AV1_STREAM_WIDTH, AV1_STREAM_HEIGHT, AV1_STREAM_FRAMES)
                .into_iter()
                .enumerate()
        {
            let frame = VideoFrame::new(
                dimensions,
                PixelFormat::Gray8,
                ColorRange::Full,
                vec![Plane {
                    data: luma,
                    stride: AV1_STREAM_WIDTH as usize,
                }],
                &limits,
            )
            .expect("synthetic monochrome frames are valid");
            packets.extend(
                block_on(encoder.encode(
                    FrameIndex(index as u64),
                    FrameSource::Cpu(CpuFrameSource {
                        frame: &frame,
                        orientation: Orientation::TopLeft,
                    }),
                ))
                .expect("the synthetic frame encodes"),
            );
        }
        packets.extend(block_on(encoder.finish()).expect("the encoder finishes"));
        assert_eq!(
            packets.len(),
            AV1_STREAM_FRAMES,
            "the encoder emits one packet per submitted frame"
        );

        SyntheticAv1Stream {
            configuration: VideoDecoderConfig {
                codec: Codec::Av1,
                profile: CodecProfile::Av1Main,
                coded_dimensions: dimensions,
                output_format: PixelFormat::Rgba8,
                color_range: ColorRange::Full,
                hardware: HardwarePreference::Avoid,
                configuration: encoder.config().decoder_config.clone(),
            },
            samples: packets
                .into_iter()
                .enumerate()
                .map(|(index, packet)| EncodedVideoSample {
                    presentation_index: FrameIndex(index as u64),
                    random_access: packet.is_sync,
                    data: packet.data,
                })
                .collect(),
            width: u64::from(AV1_STREAM_WIDTH),
            height: u64::from(AV1_STREAM_HEIGHT),
        }
    })
}

/// Deterministic 8-bit monochrome planes for the AV1 encoder.
///
/// Borrows [`synthetic_yuv420_sequence`]'s luma so the encoded stream carries
/// the same moving gradient plus low-amplitude noise every other synthetic
/// fixture does, rather than content that degenerates into a best case.
fn av1_gray8_planes(width: u32, height: u32, frames: usize) -> Vec<Vec<u8>> {
    synthetic_yuv420_sequence(width, height, frames)
        .into_iter()
        .map(|frame| frame.planes.into_iter().next().expect("luma plane").data)
        .collect()
}

/// Deterministic frame content with enough local structure that the in-loop
/// filters' data-dependent branches are actually taken.
///
/// Ported from the ad-hoc `tests/av1_simd_bench.rs` this suite replaces; the
/// generators there were the useful part of that file.
pub fn av1_structured_plane(width: usize, height: usize) -> FilterPlane {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut data = Vec::with_capacity(width * height);
    for index in 0..width * height {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = (state >> 56) as i32 & 0x1f;
        let gradient = ((index % width) / 8 + (index / width) / 8) as i32;
        data.push(((gradient + noise) & 0xff) as u8);
    }
    FilterPlane::from_samples(width, height, data, &Limits::default())
        .expect("the structured plane fits the default limits")
}

/// Near-flat block content.
///
/// The wide (8-tap and 14-tap) deblocking filters are gated on a flatness
/// check, so they only do work on content like this — which is exactly why
/// this generator, not [`av1_structured_plane`], is the input to the wide-filter
/// measurement.
pub fn av1_flat_blocks_plane(width: usize, height: usize) -> FilterPlane {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut data = Vec::with_capacity(width * height);
    for index in 0..width * height {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let (x, y) = (index % width, index / width);
        let block = ((x / 32 + y / 32) % 5) as i32;
        data.push((100 + block * 6 + ((state >> 60) as i32 & 1)) as u8);
    }
    FilterPlane::from_samples(width, height, data, &Limits::default())
        .expect("the flat-block plane fits the default limits")
}

/// A frame-wide grid of 32x32 transform blocks, which is what makes every luma
/// edge select the 14-tap deblocking filter (AV1 spec §7.14.5).
pub fn av1_wide_tx_grid(width: usize, height: usize) -> TxSizeGrid {
    let mut grid = TxSizeGrid::new(width, height);
    for y in (0..height).step_by(32) {
        for x in (0..width).step_by(32) {
            grid.set_block(x, y, 32, 32);
        }
    }
    grid
}

/// The amount of audio work one benchmark iteration performs.
///
/// Audio has no pixels, so [`FrameWork`]'s megapixel scale says nothing about
/// it. The comparable pair for a sample-clock workload is the one the AAC decode
/// groups report: `Throughput::Elements(samples)`, which criterion prints as a
/// per-channel samples/sec rate, and the x-realtime factor that rate divides
/// into — a track covering 32 s of audio muxed in 8 ms is 4000x realtime.
/// Both sides of the write path and both sides of the read path report on this
/// same scale, so mux, demux, and decode numbers are directly comparable.
#[derive(Clone, Copy, Debug)]
pub struct AudioWork {
    /// Samples per channel, i.e. the length of the covered sample interval.
    pub samples: u64,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioWork {
    pub fn new(samples: u64, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    /// Samples per iteration, which criterion turns into a samples/sec rate.
    pub fn elements(&self) -> Throughput {
        Throughput::Elements(self.samples)
    }

    /// Seconds of audio one iteration covers.
    pub fn seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples as f64 / f64::from(self.sample_rate)
    }

    /// How many times faster than realtime a measured iteration ran.
    ///
    /// This is the playback-relevant figure: anything at or below 1.0 cannot
    /// sustain real-time output, and the margin above it is the headroom the
    /// path under test leaves for everything else on the thread.
    pub fn realtime_factor(&self, per_iteration: std::time::Duration) -> f64 {
        let elapsed = per_iteration.as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }
        self.seconds() / elapsed
    }

    /// Samples per second for a measured per-iteration duration.
    pub fn samples_per_second(&self, per_iteration: std::time::Duration) -> f64 {
        let elapsed = per_iteration.as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }
        self.samples as f64 / elapsed
    }
}

/// Registers an audio benchmark's throughput and prints the realtime scale
/// criterion's own `elem/s` line does not carry.
///
/// `samples/s / sample_rate = x-realtime`, so printing the sample rate and the
/// covered duration once per benchmark makes every reported rate convertible
/// without re-deriving the fixture's timing.
pub fn report_audio_throughput<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    id: &str,
    work: AudioWork,
) {
    group.throughput(work.elements());
    println!(
        "# {id}: {} sample(s)/iter at {} Hz x{}ch = {:.4} s of audio/iter (samples/s / {} = x-realtime)",
        work.samples,
        work.sample_rate,
        work.channels,
        work.seconds(),
        work.sample_rate,
    );
}

/// The bundled sample's AAC track, demuxed once per process.
///
/// `examples/media/BigBuckBunny.mp4` is the only real audio fixture checked into
/// the repository: 1,501 AAC-LC access units at 48 kHz stereo covering 1,536,000
/// decoded samples (32 s), with a one-edit edit list that produces 1,024 samples
/// of decoder priming. That makes it the read-side counterpart to the synthetic
/// write-path fixtures — real packet sizes, a real sample table, and real
/// gapless timing rather than a uniform synthetic track.
pub struct BundledAacTrack {
    pub source: MemorySource,
    pub movie: Mp4Demuxer,
    pub track_index: usize,
    /// The parsed `esds` configuration, which is what a decoder is built from.
    pub config: AacTrackConfig,
    pub sample_rate: u32,
    pub channels: u16,
    /// Decoded samples per channel the whole track covers.
    pub decoded_samples: u64,
    pub packets: Vec<EncodedAudioSample>,
    pub timing: AudioTrackTiming,
}

impl BundledAacTrack {
    pub fn track(&self) -> &Mp4Track {
        &self.movie.tracks[self.track_index]
    }

    /// The work the whole track represents, for throughput reporting.
    pub fn work(&self) -> AudioWork {
        AudioWork::new(self.decoded_samples, self.sample_rate, self.channels)
    }

    /// The first `count` access units, or all of them when the track is shorter.
    pub fn prefix(&self, count: usize) -> &[EncodedAudioSample] {
        &self.packets[..count.min(self.packets.len())]
    }

    /// Decoded samples covered by [`BundledAacTrack::prefix`].
    pub fn prefix_samples(&self, count: usize) -> u64 {
        self.prefix(count)
            .last()
            .map_or(0, |packet| packet.decoded_range.end)
    }
}

/// The bundled sample's bytes, held once so `Mp4Demuxer::open` can be timed
/// without re-reading the file.
pub fn bundled_mp4_bytes() -> &'static [u8] {
    include_bytes!("../../examples/media/BigBuckBunny.mp4")
}

/// Demuxes the single AAC track out of an in-memory MP4.
fn demux_aac_track(bytes: Vec<u8>, label: &str) -> BundledAacTrack {
    let limits = Limits::default();
    let source = MemorySource::new(bytes);
    let movie = block_on(Mp4Demuxer::open(&source, Mp4DemuxerOptions::default()))
        .unwrap_or_else(|error| panic!("{label} is a readable MP4: {error}"));
    let track_index = movie
        .tracks
        .iter()
        .position(|track| track.kind == TrackKind::Audio && track.codec == Codec::Aac)
        .unwrap_or_else(|| panic!("{label} has an AAC audio track"));
    let track = &movie.tracks[track_index];
    let config = track
        .aac_config()
        .unwrap_or_else(|error| panic!("{label} carries an AAC AudioSpecificConfig: {error}"));
    let packets = block_on(track.to_encoded_audio_samples(&source, &limits))
        .unwrap_or_else(|error| panic!("{label}'s AAC access units are readable: {error}"));
    let timing = track
        .audio_timing(movie.movie_timescale)
        .unwrap_or_else(|error| panic!("{label}'s edit list maps to the sample clock: {error}"));
    let decoded_samples = packets
        .last()
        .unwrap_or_else(|| panic!("{label}'s AAC track is not empty"))
        .decoded_range
        .end;
    BundledAacTrack {
        sample_rate: config.sample_rate,
        channels: config.channels,
        config,
        decoded_samples,
        packets,
        timing,
        track_index,
        movie,
        source,
    }
}

/// Demuxes the bundled sample's AAC track once per process.
pub fn bundled_aac_track() -> &'static BundledAacTrack {
    static TRACK: OnceLock<BundledAacTrack> = OnceLock::new();
    TRACK.get_or_init(|| {
        demux_aac_track(
            bundled_mp4_bytes().to_vec(),
            "the bundled BigBuckBunny sample",
        )
    })
}

/// A mono 48 kHz AAC-LC fixture with a real priming edit list.
///
/// The bundled sample is stereo, but `NativeAacDecoder` accepts AAC-LC mono as
/// well and rejects everything beyond stereo, so mono is the other half of the
/// backend's entire supported input space.
pub fn aac_mono_track() -> &'static BundledAacTrack {
    static TRACK: OnceLock<BundledAacTrack> = OnceLock::new();
    TRACK.get_or_init(|| {
        demux_aac_track(
            include_bytes!("../../tests/fixtures/codec/aac_lc_mono_48k.m4a").to_vec(),
            "the mono AAC-LC fixture",
        )
    })
}

/// Builds a deterministic synthetic RGBA8 frame sequence for encoder inputs.
///
/// [`synthetic_yuv420_sequence`] is the right input for the encoder's *later*
/// stages, which consume YUV420 planes. The public HEVC encoder's own input
/// format is RGBA8, so a whole-frame encode benchmark needs the same content in
/// that format: the same moving gradient plus low-amplitude noise, so neither
/// prediction nor entropy coding degenerates into an unrepresentative best case,
/// and still no decode cost folded into the measurement.
pub fn synthetic_rgba8_sequence(width: u32, height: u32, frames: usize) -> Vec<VideoFrame> {
    let limits = Limits::default();
    let dimensions =
        VideoDimensions::new(width, height, &limits).expect("synthetic dimensions are valid");
    let (w, h) = (width as usize, height as usize);
    (0..frames)
        .map(|frame| {
            let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ frame as u64;
            let mut next_noise = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 58) as i32
            };
            let shift = (frame * 3) as i32;
            let mut data = Vec::with_capacity(w * h * 4);
            for y in 0..h {
                for x in 0..w {
                    let gradient = (x as i32 + y as i32 + shift) / 2;
                    let noise = next_noise();
                    data.push(((gradient + noise) & 0xff) as u8);
                    data.push(((gradient / 2 + (x as i32 - y as i32 + shift)) & 0xff) as u8);
                    data.push(((gradient / 3 + noise * 2) & 0xff) as u8);
                    data.push(0xff);
                }
            }
            VideoFrame::new(
                dimensions,
                PixelFormat::Rgba8,
                ColorRange::Limited,
                vec![Plane {
                    data,
                    stride: w * 4,
                }],
                &limits,
            )
            .expect("synthetic RGBA8 frames are valid")
        })
        .collect()
}
