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

// Every bench target in `benches/` compiles this module, and no single target
// uses all of it — `codec.rs` never touches the AV1 stream fixtures, and
// `av1_decode.rs` never touches the HEVC sample. Unused-item warnings here
// would therefore be one per target per fixture and would say nothing true.
#![allow(dead_code)]

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;
use std::task::{Context, Poll, Waker};

pub mod isa;

use criterion::Throughput;
use zvidlib::io::MemorySource;
use zvidlib::{
    Codec, CodecProfile, ColorRange, EncodedVideoSample, FilterPlane, Limits, Mp4Demuxer,
    Mp4DemuxerOptions, PixelFormat, Plane, TxSizeGrid, VideoDecoderConfig, VideoDimensions,
    VideoFrame, decode_av1_lossless_intra,
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
fn block_on<T>(future: impl Future<Output = T>) -> T {
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
