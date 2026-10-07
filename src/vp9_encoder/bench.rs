//! Benchmark-only access to the native VP9 encoder's pixel kernels.
//!
//! `crate::vp9_encoder` is a private module and criterion benchmarks are a
//! separate crate, so the per-stage groups in `benches/vp9_encode.rs` cannot
//! reach the forward transforms, the quantizer, the distortion metrics, the
//! predictors or the input conversion through the public API, which runs them
//! all at once inside the rate-distortion search.
//!
//! These are thin wrappers over whole planes: each walks a plane block by
//! block, calls the encoder's own entry point for the stage (the same function
//! the search calls, dispatching through the `vp9_encode` SIMD site), and
//! returns the bytes that identify its result, because `benches/support/isa.rs`
//! compares those bytes across instruction sets before timing anything. They
//! are `#[doc(hidden)]` and not part of the stable API.

use super::dsp::{
    IntraMode, ReferencePlane, TransformScratch, TxType, forward_transform, predict_inter,
    predict_intra,
};
use super::frame::Geometry;
use super::simd::{self, Quantizer};
use super::tables::{AC_QLOOKUP, DC_QLOOKUP};
use crate::{Orientation, VideoFrame};

/// The quantizer index the stage groups quantize at: the encoder's default.
const BENCH_Q_IDX: usize = super::DEFAULT_BASE_Q_IDX as usize;

/// The top-left corners of the whole `size` x `size` blocks of a plane.
fn blocks(width: usize, height: usize, size: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..height / size).flat_map(move |by| (0..width / size).map(move |bx| (bx * size, by * size)))
}

/// Takes the residual of every whole `4 << tx_size` block of `current`
/// against `reference`, forward-transforms it and quantizes it, as
/// `code_residual` does; returns the levels.
///
/// Blocks below 32x32 alternate through the four transform types, so the DCT
/// and ADST paths are both timed.
pub fn forward_transform_quantize(
    current: &[u8],
    reference: &[u8],
    width: usize,
    height: usize,
    tx_size: usize,
) -> Vec<u8> {
    const TYPES: [TxType; 4] = [
        TxType::DctDct,
        TxType::AdstDct,
        TxType::DctAdst,
        TxType::AdstAdst,
    ];
    let n = 4 << tx_size;
    let quantizer = Quantizer {
        dc_q: DC_QLOOKUP[BENCH_Q_IDX],
        ac_q: AC_QLOOKUP[BENCH_Q_IDX],
        half_step: tx_size == 3,
        rounding: 0.25,
    };
    let mut residual = vec![0; n * n];
    let mut coefficients = vec![0.0; n * n];
    let mut levels = vec![0; n * n];
    let mut scratch = TransformScratch::new();
    let mut dequantized = vec![0; n * n];
    let mut output = Vec::new();
    for (index, (x, y)) in blocks(width, height, n).enumerate() {
        let start = y * width + x;
        simd::residual(
            &current[start..],
            width,
            &reference[start..],
            width,
            n,
            &mut residual,
        );
        let tx_type = if tx_size == 3 {
            TxType::DctDct
        } else {
            TYPES[index % 4]
        };
        forward_transform(&residual, tx_size, tx_type, &mut coefficients, &mut scratch);
        simd::quantize(&coefficients, &quantizer, &mut levels, &mut dequantized);
        output.extend(levels.iter().map(|&level| level as u8));
    }
    output
}

/// A whole-sample motion search for every whole `size` x `size` block of
/// `current`: the SAD of each vector in a 9x9 grid of four-sample steps that
/// stays inside `reference`, with the encoder's early termination against the
/// best so far. Returns each block's best SAD.
pub fn motion_search_sad(
    current: &[u8],
    reference: &[u8],
    width: usize,
    height: usize,
    size: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    for (x, y) in blocks(width, height, size) {
        let source = &current[y * width + x..];
        let mut best = u32::MAX;
        for dy in -4_isize..=4 {
            for dx in -4_isize..=4 {
                let (rx, ry) = (x as isize + dx * 4, y as isize + dy * 4);
                if rx < 0 || ry < 0 || rx as usize + size > width || ry as usize + size > height {
                    continue;
                }
                let start = ry as usize * width + rx as usize;
                best = best.min(simd::sad(
                    source,
                    width,
                    &reference[start..],
                    width,
                    size,
                    best,
                ));
            }
        }
        output.extend_from_slice(&best.to_le_bytes());
    }
    output
}

/// The squared error between two planes, per 8x8 block and over the whole
/// plane, as the mode decisions and the loop filter level search take it.
pub fn block_and_plane_sse(a: &[u8], b: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut output: Vec<u8> = blocks(width, height, 8)
        .flat_map(|(x, y)| {
            let start = y * width + x;
            simd::sse(&a[start..], width, &b[start..], width, 8, 8).to_le_bytes()
        })
        .collect();
    output.extend_from_slice(&simd::sse(a, width, b, width, width, height).to_le_bytes());
    output
}

/// The 8-tap motion-compensated prediction of every whole `size` x `size`
/// block from `reference`, each with a different sub-sample vector, half of
/// them reaching past the plane's edges.
pub fn inter_prediction(reference: &[u8], width: usize, height: usize, size: usize) -> Vec<u8> {
    let plane = ReferencePlane {
        pixels: reference,
        stride: width,
        width,
        height,
    };
    let mut prediction = vec![0; size * size];
    let mut output = Vec::new();
    for (index, (x, y)) in blocks(width, height, size).enumerate() {
        let index = index as i32;
        let (mv_row, mv_col) = if index % 2 == 0 {
            (index % 16 - 8 + 16 * 2, 7 * index % 16 - 16 * 3)
        } else {
            (3 * index % 16, 5 * index % 16)
        };
        predict_inter(&plane, x, y, size, mv_row, mv_col, &mut prediction);
        output.extend_from_slice(&prediction);
    }
    output
}

/// TM intra prediction of every whole `size` x `size` block of `plane` that
/// has neighbours above and to the left, predicting from the plane itself.
pub fn tm_prediction(plane: &[u8], width: usize, height: usize, size: usize) -> Vec<u8> {
    let mut plane = plane.to_vec();
    for (x, y) in blocks(width, height, size).filter(|&(x, y)| x > 0 && y > 0) {
        predict_intra(&mut plane, width, x, y, size, IntraMode::Tm, true, true);
    }
    plane
}

/// Converts an RGBA8 or BGRA8 frame to the encoder's 8-aligned 4:2:0 source
/// picture, returning its three planes.
pub fn rgba_to_yuv420(frame: &VideoFrame) -> Vec<u8> {
    let geometry = Geometry::new(
        frame.dimensions.width as usize,
        frame.dimensions.height as usize,
    );
    let picture = super::source_picture(&geometry, frame, Orientation::TopLeft)
        .expect("a valid RGBA8 or BGRA8 frame converts");
    picture.planes.concat()
}
