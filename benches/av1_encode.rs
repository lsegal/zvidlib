//! Criterion benchmarks for the native AV1 encoder (`zvidlib::av1_encoder`).
//!
//! Two axes at once. The first is *where the encoder's time goes*: a whole-frame
//! group that drives the public [`zvidlib::native_av1_video_encoder_factory`]
//! end to end, plus one group per pipeline stage — the forward WHT, symbol
//! coding, tile encoding, and bitstream/header writing — so the whole-frame
//! number can be attributed rather than merely observed.
//!
//! The second is the instruction set. Every group here runs once per entry in
//! `zvidlib::simd::available()` through [`support::isa::bench_across_isas`], the
//! same runner the decode-side groups in `benches/codec.rs` use. Only the
//! forward WHT currently dispatches to a vector kernel
//! (`zvidlib::av1_simd::fwht4x4`); the entropy and bitstream stages are scalar
//! and are expected to read identically across arms. Those arms are reported
//! rather than omitted: a stage whose scalar and vector arms match is exactly
//! the evidence that it is a candidate for vectorization work, and dropping it
//! would hide that.
//!
//! See `benches/README.md` for how to run and filter the suite.

// `benches/support/` is shared with the `codec` bench target and neither target
// uses all of it; the fixtures this one does not touch are live in that one.
#[allow(dead_code)]
mod support;

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::av1_encoder::headers::{
    self, Av1StillConfig, assemble_temporal_unit, frame_header_payload, sequence_header_payload,
};
use zvidlib::av1_encoder::leb128::{leb128_len, write_leb128};
use zvidlib::av1_encoder::symbol::SymbolEncoder;
use zvidlib::av1_encoder::tile::FrameEncoder;
use zvidlib::av1_encoder::wht::fwht4x4;
use zvidlib::av1_encoder::{bitwriter::BitWriter, cdf};
use zvidlib::{
    Codec, CodecProfile, ColorRange, CpuFrameSource, FrameIndex, FrameSource, HardwarePreference,
    Limits, Orientation, PixelFormat, Plane, VideoDimensions, VideoEncoderConfig,
    VideoEncoderFactory, VideoFrame, native_av1_video_encoder_factory,
};

use support::FrameWork;
use support::isa::{IsaWorkload, bench_across_isas};

/// The two whole-frame resolutions, as `(width, height, frames per iteration)`.
///
/// The encoder codes every 4x4 block of every frame through the symbol coder, so
/// 1080p costs roughly nine times what 360p does; one frame per iteration keeps
/// the large arm's criterion sample within the same window as the small one's.
const RESOLUTIONS: [(u32, u32, usize); 2] = [(640, 360, 4), (1920, 1080, 1)];

/// Resolution of the plane the per-stage groups run over.
///
/// The stage groups measure cost per unit of pixel work, not the resolution
/// scaling the whole-frame groups already cover, so one size is enough — and the
/// smaller one keeps a stage arm comfortably under criterion's window on all
/// four ISAs.
const STAGE_WIDTH: usize = 640;
const STAGE_HEIGHT: usize = 360;

/// Deterministic 8-bit monochrome planes, borrowed from the shared synthetic
/// YUV420 sequence's luma.
///
/// The native AV1 encoder is lossless all-intra `Gray8` (see
/// `zvidlib::av1_encoder`), so it takes the luma plane directly. Building the
/// input from the same generator the rest of the suite uses keeps the content —
/// a moving gradient plus low-amplitude noise, so neither DC prediction nor the
/// entropy coder degenerates — identical across benchmarks.
fn gray8_planes(width: u32, height: u32, frames: usize) -> Vec<Vec<u8>> {
    support::synthetic_yuv420_sequence(width, height, frames)
        .into_iter()
        .map(|frame| {
            frame
                .planes
                .into_iter()
                .next()
                .expect("YUV420 has luma")
                .data
        })
        .collect()
}

/// The plane the per-stage groups read, generated once per process.
fn stage_plane() -> &'static [u8] {
    static PLANE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    PLANE.get_or_init(|| {
        gray8_planes(STAGE_WIDTH as u32, STAGE_HEIGHT as u32, 1)
            .into_iter()
            .next()
            .expect("one frame was requested")
    })
}

/// Wraps a monochrome plane as a `Gray8` [`VideoFrame`] the encoder accepts.
fn gray8_frame(plane: Vec<u8>, dimensions: VideoDimensions) -> VideoFrame {
    let stride = dimensions.width as usize;
    VideoFrame::new(
        dimensions,
        PixelFormat::Gray8,
        ColorRange::Full,
        vec![Plane {
            data: plane,
            stride,
        }],
        &Limits::default(),
    )
    .expect("a Gray8 plane matching its dimensions is a valid frame")
}

/// Whole-frame encode through the public encoder factory, at both resolutions.
///
/// Timed at the same level a caller sees: `VideoEncoderFactory::create` once per
/// iteration, then one `encode` per frame. Returning the concatenated bitstream
/// gives the per-ISA bit-exactness guard the encoder's actual output to compare,
/// which for an entropy coder is the strictest possible check — a single
/// diverged coefficient changes every byte after it.
fn av1_encode_whole_frame(criterion: &mut Criterion) {
    for (width, height, frames) in RESOLUTIONS {
        let limits = Limits::default();
        let dimensions =
            VideoDimensions::new(width, height, &limits).expect("bench dimensions are valid");
        let inputs: Vec<VideoFrame> = gray8_planes(width, height, frames)
            .into_iter()
            .map(|plane| gray8_frame(plane, dimensions))
            .collect();
        let configuration = VideoEncoderConfig {
            codec: Codec::Av1,
            profile: CodecProfile::Av1Main,
            coded_dimensions: dimensions,
            input_format: PixelFormat::Gray8,
            color_range: ColorRange::Full,
            // Measure the crate's own encoder, not whichever fixed-function
            // block the host happens to ship.
            hardware: HardwarePreference::Avoid,
            timescale: 30,
            frame_duration: 1,
            configuration: Vec::new(),
        };
        let label = format!("av1_encode_{width}x{height}");
        let workload = IsaWorkload::new(
            &label,
            FrameWork::new(frames as u64, u64::from(width), u64::from(height)),
        );
        bench_across_isas(criterion, &workload, || {
            let mut encoder = native_av1_video_encoder_factory()
                .create(&configuration, &limits)
                .expect("the native AV1 encoder accepts a Gray8 configuration");
            let mut bitstream = Vec::new();
            for (index, frame) in inputs.iter().enumerate() {
                let source = FrameSource::Cpu(CpuFrameSource {
                    frame,
                    orientation: Orientation::TopLeft,
                });
                let samples = support::block_on(encoder.encode(FrameIndex(index as u64), source))
                    .expect("the synthetic frame encodes");
                for sample in samples {
                    bitstream.extend_from_slice(&sample.data);
                }
            }
            bitstream
        });
    }
}

/// Forward 4x4 Walsh-Hadamard transform (`av1_encoder::wht`).
///
/// The only encoder stage with a vector kernel today: `fwht4x4` dispatches
/// through `zvidlib::av1_simd`, so this is the group where the ISA axis is
/// expected to move. One iteration transforms every 4x4 block of the stage
/// plane against a DC prediction, which is the residual the lossless all-intra
/// encoder actually feeds it.
fn av1_encode_wht(criterion: &mut Criterion) {
    let plane = stage_plane();
    let workload = IsaWorkload::new(
        "av1_encode_wht",
        FrameWork::new(1, STAGE_WIDTH as u64, STAGE_HEIGHT as u64),
    );
    bench_across_isas(criterion, &workload, || {
        let mut accumulator = 0i64;
        let mut residual = [0i32; 16];
        for block_y in (0..STAGE_HEIGHT).step_by(4) {
            for block_x in (0..STAGE_WIDTH).step_by(4) {
                // DC prediction from the block's left column, as `tile.rs` does
                // at the frame's left edge; the exact predictor does not matter
                // here, only that the residual is data-dependent.
                let dc = i32::from(plane[block_y * STAGE_WIDTH + block_x]);
                for row in 0..4 {
                    let base = (block_y + row) * STAGE_WIDTH + block_x;
                    for column in 0..4 {
                        residual[row * 4 + column] = i32::from(plane[base + column]) - dc;
                    }
                }
                for coefficient in fwht4x4(black_box(&residual)) {
                    accumulator = accumulator
                        .wrapping_mul(31)
                        .wrapping_add(i64::from(coefficient));
                }
            }
        }
        accumulator.to_le_bytes().to_vec()
    });
}

/// Symbol coding and CDF lookup (`av1_encoder::symbol` and `av1_encoder::cdf`).
///
/// The arithmetic coder is the encoder's hot inner loop, so it is measured
/// directly rather than only through the tile stage. The symbol stream is
/// derived from the plane so the coder sees a realistic mix of probable and
/// improbable symbols instead of one steady-state probability, and the returned
/// bytes are the coder's real output.
///
/// This stage is scalar and its arms are expected to read alike. That is the
/// measurement, not a gap in it: it says the entropy coder, not the transform,
/// is what a vectorization effort would have to attack.
fn av1_encode_symbol(criterion: &mut Criterion) {
    const SYMBOLS: usize = 1 << 16;
    let plane = stage_plane();
    let workload = IsaWorkload::new(
        "av1_encode_symbol",
        FrameWork::new(1, STAGE_WIDTH as u64, STAGE_HEIGHT as u64),
    );
    bench_across_isas(criterion, &workload, || {
        let mut encoder = SymbolEncoder::new();
        for index in 0..SYMBOLS {
            let sample = usize::from(plane[index % plane.len()]);
            let context = sample & 3;
            encoder.encode_symbol(sample & 3, &cdf::PARTITION_W8[context]);
            encoder.encode_symbol(sample % 10, &cdf::PARTITION_W16[context]);
            encoder.encode_literal((sample as u32) & 0x3f, 6);
        }
        encoder.finish()
    });
}

/// Whole-tile encoding (`av1_encoder::tile`): partition iteration, DC intra
/// prediction, the forward WHT, and coefficient coding with full context
/// derivation.
///
/// This is the stage the whole-frame group spends nearly all of its time in, and
/// the one that composes the WHT and symbol stages above, so its arms are what
/// the two of them have to add up to.
fn av1_encode_tile(criterion: &mut Criterion) {
    let plane = stage_plane();
    let workload = IsaWorkload::new(
        "av1_encode_tile",
        FrameWork::new(1, STAGE_WIDTH as u64, STAGE_HEIGHT as u64),
    );
    bench_across_isas(criterion, &workload, || {
        FrameEncoder::new(plane, STAGE_WIDTH, STAGE_HEIGHT).encode()
    });
}

/// Bitstream writing and headers (`av1_encoder::bitwriter`, `headers`, and
/// `leb128`).
///
/// Per frame this is a few hundred bytes against a tile payload measured in tens
/// of kilobytes, so one iteration writes the headers [`HEADER_REPEATS`] times to
/// lift the stage above criterion's timer resolution. Read it as a per-frame
/// fixed cost divided by that count, not as a share of the whole-frame number.
fn av1_encode_bitstream(criterion: &mut Criterion) {
    /// Header assemblies per iteration; see [`av1_encode_bitstream`].
    const HEADER_REPEATS: usize = 512;

    let stream = Av1StillConfig {
        seq_profile: 0,
        seq_level_idx_0: headers::pick_level(1920, 1080, 30, 1).expect("1080p is within level 6.0"),
        seq_tier_0: 0,
        high_bitdepth: false,
        twelve_bit: false,
        monochrome: true,
        chroma_subsampling_x: 1,
        chroma_subsampling_y: 1,
        chroma_sample_position: 0,
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
        full_range: true,
    };
    let workload = IsaWorkload::new(
        "av1_encode_bitstream",
        FrameWork::new(1, STAGE_WIDTH as u64, STAGE_HEIGHT as u64),
    );
    bench_across_isas(criterion, &workload, || {
        let mut unit = Vec::new();
        for repeat in 0..HEADER_REPEATS {
            let order_hint = (repeat as u32) & 7;
            let sequence = sequence_header_payload(&stream, 1920, 1080);
            let frame = frame_header_payload(1920, 1080, 480, 270, order_hint);

            // The raw writers underneath the headers, exercised at the widths
            // the frame header uses them at.
            let mut writer = BitWriter::new();
            for bit in 0..256u32 {
                writer.put_bit((bit & 1) as u8);
                writer.put_bits(bit, 9);
            }
            writer.byte_align();
            let mut leb = Vec::new();
            let mut leb_bytes = 0usize;
            for value in 0..64u64 {
                let size = value * 12_345;
                leb_bytes += leb128_len(size);
                write_leb128(&mut leb, size);
            }
            assert_eq!(leb_bytes, leb.len(), "leb128_len agrees with write_leb128");

            unit = assemble_temporal_unit(&sequence, &frame);
            unit.extend_from_slice(writer.as_bytes());
            unit.extend_from_slice(&leb);
        }
        unit
    });
}

criterion_group!(
    benches,
    av1_encode_whole_frame,
    av1_encode_wht,
    av1_encode_symbol,
    av1_encode_tile,
    av1_encode_bitstream
);
criterion_main!(benches);
