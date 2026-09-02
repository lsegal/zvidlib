//! Benchmark-only access to the AV1 encoder's individual pipeline stages.
//!
//! `crate::av1_encoder` is a private module and criterion benchmarks are a
//! separate crate, so the per-stage groups in `benches/av1_encode.rs` cannot
//! reach the forward WHT, the symbol coder, the tile encoder, or the bitstream
//! writers through the public API. The public
//! [`crate::native_av1_video_encoder_factory`] runs all of them at once, which
//! is exactly what a per-stage breakdown must avoid.
//!
//! This module is that access and nothing more: thin wrappers that own their
//! inputs, return plain bytes, and add no logic the benchmark could
//! accidentally measure instead of the encoder. It is `#[doc(hidden)]` and
//! explicitly not part of the stable API, matching the
//! [`crate::hevc_encoder_bench`] convention.
//!
//! Each wrapper returns the bytes that identify its result, because
//! `benches/support/isa.rs` compares those bytes across instruction sets before
//! timing anything: a stage whose return value did not depend on the kernels
//! under test would silently disarm that guard.

use super::headers::{
    self, Av1StillConfig, ORDER_HINT_BITS, assemble_temporal_unit, frame_header_payload,
    sequence_header_payload,
};
use super::symbol::SymbolEncoder;
use super::tile::{CoeffScratch, FrameEncoder};
use super::wht::{fwht4x4, fwht4x4_scalar, iwht4x4};
use super::{cdf, stream_configuration};
use crate::ColorRange;

/// The predictor the WHT group subtracts, the mid-point of the 8-bit range.
///
/// The tile encoder's own `DC_PRED` predictor depends on neighbouring
/// reconstructions, which is tile work rather than transform work; a fixed
/// predictor keeps this group measuring only the butterfly.
const BENCH_PREDICTOR: i32 = 128;

/// Runs the forward 4x4 WHT over every block of one 8-bit luma plane.
///
/// This is the encoder's only vectorized kernel on the lossless path — it
/// dispatches through `crate::av1_simd::fwht4x4` — so it is the one per-stage
/// group whose arms are expected to move with the instruction set.
///
/// Returns an order-sensitive fold of every coefficient rather than the
/// coefficients themselves: a 1080p plane is 130k blocks, and allocating two
/// megabytes per iteration would measure the allocator as much as the
/// transform. The fold still depends on every output value, so the
/// bit-exactness guard keeps its teeth.
#[must_use]
pub fn fwht4x4_plane(plane: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert!(
        plane.len() >= width * height,
        "plane is shorter than its dimensions"
    );
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    let mut residual = [0i32; 16];
    for block_y in (0..height & !3).step_by(4) {
        for block_x in (0..width & !3).step_by(4) {
            for row in 0..4 {
                let start = (block_y + row) * width + block_x;
                for column in 0..4 {
                    residual[row * 4 + column] = i32::from(plane[start + column]) - BENCH_PREDICTOR;
                }
            }
            for coefficient in fwht4x4(&residual) {
                digest ^= coefficient as u32 as u64;
                digest = digest.wrapping_mul(0x1000_0000_01b3);
            }
        }
    }
    digest.to_le_bytes().to_vec()
}

/// The lossless coefficients of one 8-bit luma plane, block by block, as the
/// flat input [`iwht4x4_plane`] runs over.
///
/// Built outside the timed loop so the inverse group measures the inverse WHT
/// and not the forward one, and built through `wht::fwht4x4_scalar` rather than
/// the dispatching `wht::fwht4x4` so the input is the same bytes on every host
/// and under every instruction-set override — a setup step that varied with
/// the arm would make the inverse group's bit-exactness guard compare
/// different inputs rather than different kernels.
#[must_use]
pub fn wht4x4_coefficients(plane: &[u8], width: usize, height: usize) -> Vec<i32> {
    assert!(
        plane.len() >= width * height,
        "plane is shorter than its dimensions"
    );
    let mut coefficients = Vec::new();
    let mut residual = [0i32; 16];
    for block_y in (0..height & !3).step_by(4) {
        for block_x in (0..width & !3).step_by(4) {
            for row in 0..4 {
                let start = (block_y + row) * width + block_x;
                for column in 0..4 {
                    residual[row * 4 + column] = i32::from(plane[start + column]) - BENCH_PREDICTOR;
                }
            }
            coefficients.extend_from_slice(&fwht4x4_scalar(&residual));
        }
    }
    coefficients
}

/// Runs the lossless 4x4 inverse WHT over every block of
/// [`wht4x4_coefficients`]'s output.
///
/// The counterpart to [`fwht4x4_plane`] and the other half of the same
/// dispatch family: `crate::av1_simd::iwht4x4`. It has its own group because a
/// dispatch decision taken on the forward direction's measurement is a
/// decision taken on a different kernel — the forward transform runs three
/// `transpose4`s where the inverse runs two, so the shuffle pressure that
/// settled the forward arm is not the inverse arm's shuffle pressure.
///
/// Returns an order-sensitive fold of every reconstructed residual rather than
/// the residuals themselves, for the same reason [`fwht4x4_plane`] does.
#[must_use]
pub fn iwht4x4_plane(coefficients: &[i32]) -> Vec<u8> {
    assert!(
        coefficients.len() % 16 == 0,
        "coefficients are whole 4x4 blocks"
    );
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    let mut block = [0i32; 16];
    for quantized in coefficients.chunks_exact(16) {
        block.copy_from_slice(quantized);
        for residual in iwht4x4(&block) {
            digest ^= residual as u32 as u64;
            digest = digest.wrapping_mul(0x1000_0000_01b3);
        }
    }
    digest.to_le_bytes().to_vec()
}

/// Drives the symbol (range) coder over a deterministic symbol stream taken
/// from the encoder's own CDF tables.
///
/// `symbols` is the number of CDF-coded symbols to encode; each is followed by
/// two equiprobable literal bits, the mix the tile encoder actually produces.
/// The tables cycled through are the partition, skip and intra-mode CDFs the
/// tile encoder codes with, so the alphabet sizes match real work.
///
/// The M0 frame sets `disable_cdf_update = 1`, so the tables are static and no
/// adaptation cost is included; that absence is the measurement, and it is why
/// the issue's "symbol coding and CDF adaptation" stage reads as pure coding
/// cost here.
#[must_use]
pub fn symbol_encode(symbols: usize) -> Vec<u8> {
    let mut encoder = SymbolEncoder::new();
    for index in 0..symbols {
        match index % 4 {
            0 => encoder.encode_symbol(index % 4, &cdf::PARTITION_W8[index % 4]),
            1 => encoder.encode_symbol(index % 2, &cdf::SKIP[index % 3]),
            2 => encoder.encode_symbol(index % 13, &cdf::INTRA_FRAME_Y_MODE_DC_DC),
            _ => encoder.encode_symbol(index % 10, &cdf::PARTITION_W32[index % 4]),
        }
        encoder.encode_literal((index % 4) as u32, 2);
    }
    encoder.finish()
}

/// Encodes one whole tile — superblock and partition iteration, `DC_PRED`
/// intra prediction, the forward transform, and coefficient coding with full
/// context derivation — and returns the symbol-coded tile bytes.
///
/// `qindex` selects the profile: `0` is the lossless WHT path, nonzero is the
/// quantized path through `super::transform::forward_transform`.
#[must_use]
pub fn tile_encode(plane: &[u8], width: usize, height: usize, qindex: u8) -> Vec<u8> {
    FrameEncoder::new(plane, width, height, qindex).encode()
}

/// Derives the §8.3.2 `coeff_base` and `coeff_br` contexts for every 4x4
/// transform block of one 8-bit luma plane.
///
/// This is the pass `tile.rs` runs over a whole block before its serial symbol
/// loop, and the only part of coefficient coding that is data-parallel — the
/// range coder around it is serial by construction. It is benchmarked on its
/// own because `av1_encode_stage_tile` cannot show it: the same profile that
/// motivated this kernel puts 64% of a lossless tile encode inside
/// `SymbolEncoder::encode_symbol`, which dilutes any context-derivation
/// speedup to within run-to-run noise. This group is where the
/// `av1_coeff_ctx` dispatch family's scalar-versus-vector delta is actually
/// visible.
///
/// The coefficients fed to it are the plane's own lossless WHT output, so the
/// magnitude distribution — and therefore the neighbour sums the contexts are
/// made of — is the one the encoder really derives contexts over, not a
/// synthetic one.
///
/// Returns an order-sensitive fold of every derived context rather than the
/// contexts themselves, for the same reason [`fwht4x4_plane`] does: the fold
/// depends on every value, so `benches/support/isa.rs`'s bit-exactness guard
/// keeps its teeth without allocating a plane's worth of output per iteration.
#[must_use]
pub fn coeff_context_plane(plane: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert!(
        plane.len() >= width * height,
        "plane is shorter than its dimensions"
    );
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    let mut residual = [0i32; 16];
    let mut scratch = CoeffScratch::default();
    for block_y in (0..height & !3).step_by(4) {
        for block_x in (0..width & !3).step_by(4) {
            for row in 0..4 {
                let start = (block_y + row) * width + block_x;
                for column in 0..4 {
                    residual[row * 4 + column] = i32::from(plane[start + column]) - BENCH_PREDICTOR;
                }
            }
            scratch.derive(&fwht4x4(&residual), 4);
            for &context in scratch.base.iter().chain(scratch.br.iter()) {
                digest ^= context as u32 as u64;
                digest = digest.wrapping_mul(0x1000_0000_01b3);
            }
        }
    }
    digest.to_le_bytes().to_vec()
}

/// Writes the sequence header, the frame header, and the OBU framing that
/// wraps them and `tile_data` into one temporal unit.
///
/// This is the bitstream stage in full: `bitwriter.rs` writes both headers bit
/// by bit, `headers.rs` lays out their syntax, and `leb128.rs` encodes each
/// OBU's size field. It is scalar and expected to stay that way, so its arms
/// are expected to read identically across instruction sets.
#[must_use]
pub fn write_temporal_unit(
    width: u32,
    height: u32,
    order_hint: u32,
    base_q_idx: u8,
    tile_data: &[u8],
) -> Vec<u8> {
    let stream = bench_stream_configuration(width, height);
    let sequence = sequence_header_payload(&stream, width, height);
    let mi_cols = 2 * ((width as usize + 7) >> 3);
    let mi_rows = 2 * ((height as usize + 7) >> 3);
    let mut frame = frame_header_payload(
        width,
        height,
        mi_cols as u32,
        mi_rows as u32,
        order_hint % (1 << ORDER_HINT_BITS),
        base_q_idx,
    );
    frame.extend_from_slice(tile_data);
    assemble_temporal_unit(&sequence, &frame)
}

/// The sequence configuration the bitstream group writes, chosen by the same
/// [`headers::pick_level`] the public encoder uses so the header syntax
/// measured here is the header syntax it emits.
fn bench_stream_configuration(width: u32, height: u32) -> Av1StillConfig {
    let level = headers::pick_level(width, height, 90_000, 3_000)
        .expect("benchmark dimensions are within level 6.0");
    stream_configuration(ColorRange::Limited, level)
}
