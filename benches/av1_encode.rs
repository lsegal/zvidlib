//! Criterion benchmarks for zvidlib's native AV1 encoder.
//!
//! Two axes, both required to read an encoder number correctly:
//!
//! * **Whole-frame versus per-stage.** The whole-frame groups encode a
//!   synthetic monochrome sequence through the public
//!   [`zvidlib::native_av1_video_encoder_factory`] at two resolutions and
//!   report frames/sec and megapixels/sec. The per-stage groups time the
//!   pipeline's individual stages through [`zvidlib::av1_encoder_bench`], so
//!   the tile encoder's cost — which dominates — is not mistaken for
//!   bitstream-writing cost. The kernel groups below them measure the forward
//!   transforms on their own.
//! * **Instruction set.** Every group runs once per entry in
//!   `zvidlib::simd::available()` through `support::isa::bench_across_isas`,
//!   which pins the crate-wide override, asserts it reached every dispatch
//!   family, and checks each arm is bit-exact with scalar before timing it.
//!
//! Inputs are the synthetic sequences from `benches/support`, never a decoded
//! file: decoding first would fold decoder cost into the encoder numbers.
//!
//! The AV1 encoder's forward transforms are the counterpart to the inverse
//! transforms `benches/av1_decode.rs` measures: the same block sizes, the same
//! `Av1TxType` families, and the same `av1_simd` dispatch site, run in the
//! encoding direction. They are measured here rather than beside the inverse
//! sweep because they are encoder work, and a decoder target that reports
//! encoder numbers is a target whose scope cannot be read off its name.
//!
//! # Groups
//!
//! | Group | Stage |
//! | --- | --- |
//! | `av1_encode_{640x360,1920x1080}` | whole-frame encode through the public factory |
//! | `av1_encode_*_wht` | the forward 4x4 WHT, `src/av1_encoder/wht.rs` |
//! | `av1_encode_*_symbol` | symbol coding over the static CDF tables, `src/av1_encoder/symbol.rs` and `cdf.rs` |
//! | `av1_encode_*_tile` | tile encoding, `src/av1_encoder/tile.rs` |
//! | `av1_encode_*_bitstream` | headers, bit writing and LEB128 framing, `src/av1_encoder/{bitwriter,headers,leb128}.rs` |
//! | `av1_forward_dct_{4x4,8x8,16x16,32x32}` | forward DCT, `src/av1_encoder/transform.rs` through `zvidlib::forward_transform` |
//! | `av1_forward_adst_8x8`, `av1_forward_flipadst_16x16` | the forward ADST family, including a flipped type |
//!
//! The kernel groups' block counts and coefficient generator are the ones they
//! were introduced with (issue #140, in `tests/av1_simd_bench.rs`, and then in
//! `benches/av1_decode.rs`), so their numbers stay directly comparable with the
//! inverse-transform groups and with everything reported for them before the
//! move.
//!
//! ## The SIMD axis, and where it is expected to read flat
//!
//! The forward transforms and the forward 4x4 WHT are this encoder's only
//! vectorized kernels; they dispatch through `zvidlib::av1_simd`. Symbol
//! coding, CDF handling, bitstream writing and OBU framing are scalar and are
//! expected to stay that way, so those arms are expected to read identically
//! under every instruction set. That is the measured result the issue asks
//! for, not a broken benchmark: it says where the encoder's time actually goes
//! and what is worth vectorizing next. It is also why every group asserts
//! through `simd::active_by_site()` that the override landed rather than
//! inferring it from the clock.
//!
//! ## Stage coverage
//!
//! [`report_stage_coverage`] prints the stage list on every run, so a group
//! missing from the output reads as a broken run rather than as a stage that
//! costs nothing.
//!
//! See `benches/README.md` for how to run and filter the suite.

mod support;

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use zvidlib::av1_encoder_bench as encoder_bench;
use zvidlib::{
    Av1TxType, Codec, CodecProfile, ColorRange, CpuFrameSource, FrameIndex, FrameSource,
    HardwarePreference, Limits, Orientation, PixelFormat, VideoDimensions, VideoEncoderConfig,
    VideoEncoderFactory, VideoFrame, forward_transform, native_av1_video_encoder_factory,
};

use support::FrameWork;
use support::isa::{IsaWorkload, bench_across_isas, checksum};

/// The two resolutions the whole-frame and per-stage groups run, the ones the
/// tracking issue names. Both are encodable as-is: the AV1 encoder pads to the
/// 4x4 transform grid itself, so neither needs the divisibility adjustment
/// `benches/hevc_encode.rs` documents.
const SMALL: (u32, u32) = (640, 360);

/// The 1080p resolution, the second of the two.
const LARGE: (u32, u32) = (1920, 1080);

/// Frames per whole-frame iteration. Small enough that one criterion sample of
/// the 1080p group stays well inside the measurement window.
const WHOLE_FRAME_FRAMES: usize = 2;

/// CDF-coded symbols the symbol-coder group encodes per 4x4 block.
///
/// The tile encoder codes a partition, a skip flag and a mode symbol per block
/// before its coefficients, so three per block keeps this group the same scale
/// of work as the tile group it is factored out of — and, because it is
/// per-block, keeps the two resolutions measuring proportionally different
/// amounts of work rather than the same fixed count twice.
const SYMBOLS_PER_BLOCK: usize = 3;

/// The `base_q_idx` the per-stage groups encode at.
///
/// Zero is the lossless WHT profile, which is what the public encoder emits by
/// default and therefore the path the whole-frame groups exercise.
const BENCH_QINDEX: u8 = 0;

/// Prints the stages this target measures.
///
/// A benchmark suite reports what it measured; a stage that silently stopped
/// being measured has to read as a broken run, not as a stage that costs
/// nothing.
fn report_stage_coverage(_: &mut Criterion) {
    println!(
        "# av1_encode: every stage of the native AV1 encoder is benchmarked below: the forward\n\
         # 4x4 WHT, symbol (range) coding over the static CDF tables, whole-tile encoding\n\
         # (superblock iteration, DC_PRED, coefficient coding), sequence/frame header writing\n\
         # with OBU LEB128 framing, and whole-frame encode through the public factory at\n\
         # 640x360 and 1920x1080, alongside the forward DCT/ADST kernel sweep.\n\
         # av1_encode: the forward transforms and the forward WHT are the encoder's only SIMD\n\
         # dispatch families, so the symbol, bitstream and (dominantly scalar) tile arms are\n\
         # expected to read flat across instruction sets. That is the measured result, not a\n\
         # broken bench: it is what says entropy coding is the next vectorization target."
    );
}

/// The encoder configuration the whole-frame groups measure.
fn encoder_config(width: u32, height: u32) -> VideoEncoderConfig {
    VideoEncoderConfig {
        codec: Codec::Av1,
        profile: CodecProfile::Av1Main,
        coded_dimensions: VideoDimensions::new(width, height, &Limits::default())
            .expect("benchmark dimensions are valid"),
        input_format: PixelFormat::Gray8,
        color_range: ColorRange::Limited,
        // Measure the crate's own encode work, not whichever fixed-function
        // block the host happens to ship.
        hardware: HardwarePreference::Avoid,
        timescale: 90_000,
        frame_duration: 3_000,
        // Empty selects the lossless WHT profile.
        configuration: Vec::new(),
    }
}

/// Encodes a fixed-length synthetic sequence through the public encoder, once
/// per instruction set.
fn whole_frame(criterion: &mut Criterion, (width, height): (u32, u32), group: &str) {
    let frames = support::synthetic_gray8_sequence(width, height, WHOLE_FRAME_FRAMES);
    let configuration = encoder_config(width, height);
    let workload = IsaWorkload::new(
        group,
        FrameWork::new(
            WHOLE_FRAME_FRAMES as u64,
            u64::from(width),
            u64::from(height),
        ),
    );
    bench_across_isas(criterion, &workload, || {
        let mut encoder = native_av1_video_encoder_factory()
            .create(&configuration, &Limits::default())
            .expect("the native AV1 encoder is constructible");
        let mut bitstream = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            let source = FrameSource::Cpu(CpuFrameSource {
                frame,
                orientation: Orientation::TopLeft,
            });
            for sample in support::block_on(encoder.encode(FrameIndex(index as u64), source))
                .expect("the synthetic sequence encodes")
            {
                bitstream.extend_from_slice(&sample.data);
            }
        }
        for sample in support::block_on(encoder.finish()).expect("the encoder flushes") {
            bitstream.extend_from_slice(&sample.data);
        }
        bitstream
    });
}

/// One tightly packed 8-bit luma plane of the synthetic sequence, the input
/// every per-stage group runs over.
fn luma_plane(width: u32, height: u32) -> Vec<u8> {
    let frame: VideoFrame = support::synthetic_gray8_sequence(width, height, 1)
        .into_iter()
        .next()
        .expect("the synthetic sequence has one frame");
    let plane = frame.planes.first().expect("Gray8 has one plane");
    let width = width as usize;
    (0..height as usize)
        .flat_map(|row| plane.data[row * plane.stride..row * plane.stride + width].to_vec())
        .collect()
}

/// The forward 4x4 WHT over a whole plane: the lossless path's transform, and
/// the one per-stage group with a vector path.
fn wht(criterion: &mut Criterion, (width, height): (u32, u32), group_prefix: &str) {
    let plane = luma_plane(width, height);
    let name = format!("{group_prefix}_wht");
    let workload = IsaWorkload::new(
        &name,
        FrameWork::new(1, u64::from(width), u64::from(height)),
    );
    bench_across_isas(criterion, &workload, || {
        encoder_bench::fwht4x4_plane(&plane, width as usize, height as usize)
    });
}

/// Symbol coding and CDF handling, factored out of the tile encoder so the
/// entropy coder's cost is visible on its own.
fn symbol(criterion: &mut Criterion, (width, height): (u32, u32), group_prefix: &str) {
    let symbols = (width as usize / 4) * (height as usize / 4) * SYMBOLS_PER_BLOCK;
    let name = format!("{group_prefix}_symbol");
    let workload = IsaWorkload::new(
        &name,
        FrameWork::new(1, u64::from(width), u64::from(height)),
    );
    bench_across_isas(criterion, &workload, || {
        encoder_bench::symbol_encode(symbols)
    });
}

/// Whole-tile encoding: superblock and partition iteration, DC intra
/// prediction, the forward transform, and coefficient coding with full context
/// derivation. This is where a whole-frame encode spends nearly all its time.
fn tile(criterion: &mut Criterion, (width, height): (u32, u32), group_prefix: &str) {
    let plane = luma_plane(width, height);
    let name = format!("{group_prefix}_tile");
    let workload = IsaWorkload::new(
        &name,
        FrameWork::new(1, u64::from(width), u64::from(height)),
    );
    bench_across_isas(criterion, &workload, || {
        encoder_bench::tile_encode(&plane, width as usize, height as usize, BENCH_QINDEX)
    });
}

/// Header writing and OBU framing: the bit writer, the sequence and frame
/// header syntax, and the LEB128 size fields.
///
/// The tile payload is encoded once, outside the timed closure, so this group
/// measures only the bitstream stage wrapped around it.
fn bitstream(criterion: &mut Criterion, (width, height): (u32, u32), group_prefix: &str) {
    let plane = luma_plane(width, height);
    let tile_data =
        encoder_bench::tile_encode(&plane, width as usize, height as usize, BENCH_QINDEX);
    let name = format!("{group_prefix}_bitstream");
    let mut workload = IsaWorkload::new(
        &name,
        FrameWork::new(1, u64::from(width), u64::from(height)),
    );
    // Header writing is microseconds of work next to a tile encode, so the
    // default frame-scale windows would spend seconds resolving noise.
    workload.measurement_time = Duration::from_secs(2);
    workload.warm_up_time = Duration::from_millis(300);
    workload.sample_size = 100;
    bench_across_isas(criterion, &workload, || {
        encoder_bench::write_temporal_unit(width, height, 0, BENCH_QINDEX, &tile_data)
    });
}

fn av1_encode_whole_frame(criterion: &mut Criterion) {
    whole_frame(criterion, SMALL, "av1_encode_640x360");
    whole_frame(criterion, LARGE, "av1_encode_1920x1080");
}

fn av1_encode_stages(criterion: &mut Criterion) {
    for (size, prefix) in [
        (SMALL, "av1_encode_640x360"),
        (LARGE, "av1_encode_1920x1080"),
    ] {
        wht(criterion, size, prefix);
        symbol(criterion, size, prefix);
        tile(criterion, size, prefix);
        bitstream(criterion, size, prefix);
    }
}

/// Luma dimensions the kernel-level groups run over, matching
/// `benches/av1_decode.rs`. One 1080p plane is large enough that per-call
/// dispatch overhead is negligible next to the vectorized inner loops, and it
/// is the size these measurements have always used, so their numbers stay
/// comparable across the move.
const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;

/// Criterion windows for the kernel groups.
///
/// Each group is measured once per available instruction set, so the default
/// five-second window would stretch a plain `cargo bench --bench av1_encode`
/// out for no extra resolution. Two seconds over a 1080p plane is still
/// hundreds of iterations of work per sample.
fn kernel_workload<'a>(codec: &'a str, work: FrameWork) -> IsaWorkload<'a> {
    IsaWorkload {
        measurement_time: Duration::from_secs(2),
        warm_up_time: Duration::from_millis(300),
        ..IsaWorkload::new(codec, work)
    }
}

// ---------------------------------------------------------------------------
// Forward transforms (src/av1_encoder/transform.rs, src/av1_simd/transforms.rs)
// ---------------------------------------------------------------------------

/// Every forward transform size and family the vector kernels cover, applied
/// over a whole frame's worth of blocks.
fn av1_forward_transforms(criterion: &mut Criterion) {
    for (name, size, tx_type) in [
        ("av1_forward_dct_4x4", 4usize, Av1TxType::DctDct),
        ("av1_forward_dct_8x8", 8, Av1TxType::DctDct),
        ("av1_forward_dct_16x16", 16, Av1TxType::DctDct),
        ("av1_forward_dct_32x32", 32, Av1TxType::DctDct),
        ("av1_forward_adst_8x8", 8, Av1TxType::AdstAdst),
        ("av1_forward_flipadst_16x16", 16, Av1TxType::FlipadstAdst),
    ] {
        let residual: Vec<i32> = (0..size * size)
            .map(|index| (index as i32 * 53) % 511 - 255)
            .collect();
        let blocks = (WIDTH / size) * (HEIGHT / size);
        let covered_width = (WIDTH / size * size) as u64;
        let covered_height = (HEIGHT / size * size) as u64;
        let work = FrameWork::new(1, covered_width, covered_height);
        let workload = kernel_workload(name, work);
        bench_across_isas(criterion, &workload, || {
            let mut digest = 0u64;
            for _ in 0..blocks {
                let coefficients = forward_transform(&residual, size, tx_type);
                digest ^= checksum(&coefficients[0].to_le_bytes());
            }
            digest.to_le_bytes().to_vec()
        });
    }
}

criterion_group!(
    benches,
    report_stage_coverage,
    av1_encode_whole_frame,
    av1_encode_stages,
    av1_forward_transforms
);
criterion_main!(benches);
