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

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;
use std::task::{Context, Poll, Waker};

pub mod isa;

use criterion::Throughput;
use zvidlib::io::MemorySource;
use zvidlib::{
    Codec, CodecProfile, ColorRange, EncodedVideoSample, Limits, Mp4Demuxer, Mp4DemuxerOptions,
    PixelFormat, Plane, VideoDecoderConfig, VideoDimensions, VideoFrame, decode_av1_lossless_intra,
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
