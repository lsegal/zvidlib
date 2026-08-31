//! Encoder-side forward transform (§8.6.4) and quantization (§8.6.3).
//!
//! The decoder half of these two stages already exists: §8.6.3 scaling is
//! [`crate::hevc::engine::transform::scale_coefficients`] and §8.6.4 is
//! [`crate::hevc::engine::transform::inverse_transform`], both driven by
//! [`crate::hevc::engine::transform::residual_block`]. This module is their
//! encode-side inverse, and it is deliberately the *only* new numerical stage
//! in the encoder: the reconstruction path never uses these kernels to
//! reproduce a decoder's output. It quantizes, and then reconstructs by
//! calling the decoder's own `residual_block` on the quantized levels, so the
//! encoder's reference picture is by construction the picture a conforming
//! decoder derives from the levels that were actually written.
//!
//! That split matters for what has to be exact. [`forward_transform`] and
//! [`quantize`] only decide *which* levels get coded — a different rounding
//! rule costs quality, never conformance — while [`reconstruct_residual`]
//! runs the spec's own reconstruction and therefore has to be, and is,
//! literally the decoder's code.
//!
//! ## Forward transform
//!
//! §8.6.4 synthesizes `r = Mᵀ · d · M` from the equation-8-316..8-321 basis
//! `M`, one 1-D pass over the columns and one over the rows. The analysis
//! that inverts it is `d = M · r · Mᵀ`, which is the same basis read in the
//! transposed direction — [`forward_dct_1d`] and [`forward_dst4_1d`] — applied
//! to the rows first and then the columns. The two intermediate right shifts
//! keep the coefficients inside the §7.4.5 `[ CoeffMin, CoeffMax ]` dynamic
//! range the inverse assumes at its input.
//!
//! ## Quantization
//!
//! §8.6.3 scales a level by `levelScale[ qP % 6 ] << ( qP / 6 )` and rounds
//! down by `bdShift`. [`quantize`] is the reciprocal: a multiply by the
//! matching `quantScale[ qP % 6 ]`, a `qBits` right shift derived from the
//! same `qP / 6` and transform dynamic range, and a deadzone rounding offset
//! that is wider for intra blocks than for inter ones — the standard choice,
//! since an intra block's reconstruction error feeds every block that predicts
//! from it.

use crate::hevc::engine::transform::{
    BlockParams, Component, PredMode, TransformError, coeff_range, forward_dct_1d,
    forward_dst4_1d, residual_block,
};

/// `MAX_TR_DYNAMIC_RANGE` — the coefficient dynamic range the §8.6.3 /
/// §8.6.4 shifts are dimensioned around (§7.4.5 equations 7-27..7-30 with
/// `extended_precision_processing_flag == 0`).
const TRANSFORM_DYNAMIC_RANGE: i32 = 15;

/// The reciprocal of [`crate::hevc::engine::transform::LEVEL_SCALE`]:
/// `quantScale[ k ] ≈ ( 1 << 14 ) * 2^( 6 − k ) / levelScale[ k ]`, so that
/// quantizing at `qP` and scaling back at `qP` is the identity up to the
/// rounding both directions apply.
const QUANT_SCALE: [i64; 6] = [26214, 23302, 20560, 18396, 16384, 14564];

/// The `qBits` fixed-point position the quantizer's multiply is taken at.
const QUANT_SHIFT: i32 = 14;

/// The deadzone rounding offset numerators, over a 512 denominator: intra
/// blocks round at 1/3, inter blocks at 1/6. An intra block is a prediction
/// source for everything coded after it, so it is worth spending bits to keep
/// its reconstruction closer.
const INTRA_ROUNDING: i64 = 171;
/// The inter-block deadzone numerator (see [`INTRA_ROUNDING`]).
const INTER_ROUNDING: i64 = 85;

/// `log2( n_tbs )` for the four legal transform-block sides.
fn log2_tbs(n_tbs: usize) -> Option<u32> {
    match n_tbs {
        4 => Some(2),
        8 => Some(3),
        16 => Some(4),
        32 => Some(5),
        _ => None,
    }
}

/// `true` when §8.6.4 codes this block with the equation-8-316 alternate
/// (DST-VII) transform: 4x4 luma in an intra coding unit.
fn uses_dst(n_tbs: usize, pred_mode: PredMode, component: Component) -> bool {
    n_tbs == 4 && matches!(pred_mode, PredMode::Intra) && !component.is_chroma()
}

/// One forward 1-D pass over `input`, selecting the §8.6.4.2 `trType`.
fn forward_1d(input: &[i64], n_tbs: usize, dst: bool) -> Vec<i64> {
    if dst {
        forward_dst4_1d(input)
    } else {
        forward_dct_1d(input, n_tbs)
    }
}

/// §8.6.4 in the analysis direction — turn an `(nTbS)x(nTbS)` residual block
/// into transform coefficients.
///
/// `residual` is row-major by `y` (`residual[ y * nTbS + x ]`), matching the
/// layout [`residual_block`] returns and [`quantize`] consumes. `pred_mode`
/// and `component` select the same `trType` the inverse will, so a block coded
/// with the 4x4 alternate transform is analysed with it too.
///
/// # Panics
/// Panics if `n_tbs` is not 4 / 8 / 16 / 32 or `residual` is not
/// `n_tbs * n_tbs` long — both are encoder-internal invariants.
#[must_use]
pub(crate) fn forward_transform(
    residual: &[i32],
    n_tbs: usize,
    pred_mode: PredMode,
    component: Component,
    bit_depth: u8,
) -> Vec<i32> {
    let log2 = log2_tbs(n_tbs).expect("forward_transform called with an illegal nTbS");
    assert_eq!(residual.len(), n_tbs * n_tbs, "residual block size mismatch");
    let dst = uses_dst(n_tbs, pred_mode, component);
    let (coeff_min, coeff_max) = coeff_range(bit_depth, false);
    let (lo, hi) = (i64::from(coeff_min), i64::from(coeff_max));

    // The two shifts that mirror the inverse's §8.6.2 equation-8-299 `bdShift`
    // and the §8.6.4 intermediate clip, so both passes land back inside the
    // coefficient dynamic range.
    let shift1 = log2 as i32 + i32::from(bit_depth) - 9;
    let shift2 = log2 as i32 + 6;

    // Pass 1 — every row through the analysis basis (`t = r · Mᵀ`).
    let mut tmp = vec![0i64; n_tbs * n_tbs];
    let mut row = vec![0i64; n_tbs];
    for y in 0..n_tbs {
        for (x, cell) in row.iter_mut().enumerate() {
            *cell = i64::from(residual[y * n_tbs + x]);
        }
        for (u, v) in forward_1d(&row, n_tbs, dst).into_iter().enumerate() {
            tmp[y * n_tbs + u] = shift_round(v, shift1).clamp(lo, hi);
        }
    }

    // Pass 2 — every column through the same basis (`d = M · t`).
    let mut out = vec![0i32; n_tbs * n_tbs];
    let mut column = vec![0i64; n_tbs];
    for x in 0..n_tbs {
        for (y, cell) in column.iter_mut().enumerate() {
            *cell = tmp[y * n_tbs + x];
        }
        for (u, v) in forward_1d(&column, n_tbs, dst).into_iter().enumerate() {
            out[u * n_tbs + x] = shift_round(v, shift2).clamp(lo, hi) as i32;
        }
    }
    out
}

/// `( value + ( 1 << ( shift − 1 ) ) ) >> shift`, the offset-round both
/// forward passes share (`shift` is always positive for the dimensioned
/// 8..=16-bit depths).
fn shift_round(value: i64, shift: i32) -> i64 {
    debug_assert!(shift > 0);
    (value + (1 << (shift - 1))) >> shift
}

/// §8.6.3 in the quantizing direction — turn transform coefficients into the
/// `TransCoeffLevel` array §7.3.8.11 codes.
///
/// `q_p` is the block's §8.6.1-derived `qP`, the same value the decoder's
/// scaling process will be given. `intra` selects the wider deadzone rounding
/// offset. The result is row-major by `y` and clipped to the §7.4.5
/// `[ CoeffMin, CoeffMax ]` conformance range.
///
/// # Panics
/// Panics if `n_tbs` is not 4 / 8 / 16 / 32 or `coefficients` is not
/// `n_tbs * n_tbs` long.
#[must_use]
pub(crate) fn quantize(
    coefficients: &[i32],
    n_tbs: usize,
    q_p: u32,
    bit_depth: u8,
    intra: bool,
) -> Vec<i32> {
    let log2 = log2_tbs(n_tbs).expect("quantize called with an illegal nTbS");
    assert_eq!(
        coefficients.len(),
        n_tbs * n_tbs,
        "coefficient block size mismatch"
    );
    // The §8.6.3 `bdShift` read backwards: what the scaling process will shift
    // right by is what the quantizer has to shift left into.
    let transform_shift = TRANSFORM_DYNAMIC_RANGE - i32::from(bit_depth) - log2 as i32;
    let q_bits = QUANT_SHIFT + (q_p / 6) as i32 + transform_shift;
    debug_assert!(q_bits > 0);
    let scale = QUANT_SCALE[(q_p % 6) as usize];
    let offset = ((1i64 << (q_bits - 1)) * if intra { INTRA_ROUNDING } else { INTER_ROUNDING }) / 512;
    let (coeff_min, coeff_max) = coeff_range(bit_depth, false);

    coefficients
        .iter()
        .map(|&c| {
            let level = ((i64::from(c).unsigned_abs() as i64) * scale + offset) >> q_bits;
            let level = level.min(i64::from(coeff_max)) as i32;
            if c < 0 { -level } else { level }.clamp(coeff_min, coeff_max)
        })
        .collect()
}

/// The decoder's own §8.6.2 reconstruction of one transform block, run on the
/// encoder side over the levels that were quantized for it.
///
/// This is what keeps the encoder's reference picture and a decoder's in step:
/// it is [`residual_block`], not an encoder approximation of it, so whatever
/// the quantizer rounded away is reflected identically on both sides.
///
/// # Errors
/// [`TransformError`] as for [`residual_block`] — an illegal block size,
/// length mismatch, or bit depth, none of which the encoder produces.
pub(crate) fn reconstruct_residual(
    levels: &[i32],
    n_tbs: usize,
    q_p: u32,
    pred_mode: PredMode,
    component: Component,
    bit_depth: u8,
) -> Result<Vec<i32>, TransformError> {
    residual_block(
        levels,
        None,
        BlockParams {
            n_tbs,
            q_p,
            component,
            pred_mode,
            bit_depth,
            extended_precision: false,
            transquant_bypass: false,
            transform_skip: false,
            transform_skip_rotation_enabled: false,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic residual block with both signs and a DC offset.
    fn residual(n: usize, seed: i32) -> Vec<i32> {
        (0..n * n)
            .map(|i| {
                let (x, y) = ((i % n) as i32, (i / n) as i32);
                (x * 3 - y * 2 + seed) % 61 - 30
            })
            .collect()
    }

    fn sse(a: &[i32], b: &[i32]) -> i64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = i64::from(x) - i64::from(y);
                d * d
            })
            .sum()
    }

    #[test]
    fn a_flat_residual_transforms_to_a_dc_only_block() {
        // The row-0 basis is constant, so a constant residual has no AC
        // content — the single sharpest check that the analysis basis is the
        // transpose of the synthesis one and not its rotation.
        for n in [4usize, 8, 16, 32] {
            let flat = vec![20i32; n * n];
            let coeffs = forward_transform(&flat, n, PredMode::Inter, Component::Luma, 8);
            assert_ne!(coeffs[0], 0, "{n}x{n}: the DC coefficient vanished");
            assert!(
                coeffs[1..].iter().all(|&c| c == 0),
                "{n}x{n}: a flat residual produced AC coefficients"
            );
        }
    }

    #[test]
    fn quantizing_at_qp_zero_and_reconstructing_is_near_lossless() {
        // qP 0 is the finest step the scaling process offers; the round trip
        // there is the transform's own rounding error and nothing else.
        for n in [4usize, 8, 16] {
            let r = residual(n, 7);
            let coeffs = forward_transform(&r, n, PredMode::Inter, Component::Luma, 8);
            let levels = quantize(&coeffs, n, 0, 8, false);
            let back =
                reconstruct_residual(&levels, n, 0, PredMode::Inter, Component::Luma, 8).unwrap();
            let per_sample = sse(&r, &back) / (n * n) as i64;
            assert!(
                per_sample <= 4,
                "{n}x{n}: qP 0 round trip lost {per_sample} per sample"
            );
        }
    }

    #[test]
    fn a_coarser_qp_costs_more_distortion_and_fewer_levels() {
        // The property the whole stage exists for: raising qP must actually
        // throw information away, or the encoder is still lossless.
        let n = 16;
        let r = residual(n, 3);
        let coeffs = forward_transform(&r, n, PredMode::Intra, Component::Luma, 8);
        let mut previous_nonzero = usize::MAX;
        let mut previous_error = -1i64;
        for qp in [10u32, 26, 40] {
            let levels = quantize(&coeffs, n, qp, 8, true);
            let nonzero = levels.iter().filter(|&&l| l != 0).count();
            let back =
                reconstruct_residual(&levels, n, qp, PredMode::Intra, Component::Luma, 8).unwrap();
            let error = sse(&r, &back);
            assert!(
                nonzero <= previous_nonzero,
                "qP {qp} coded more levels than the finer step"
            );
            assert!(error > previous_error, "qP {qp} did not cost more error");
            previous_nonzero = nonzero;
            previous_error = error;
        }
        assert!(
            previous_error > 0,
            "a lossy quantizer reproduced the residual exactly"
        );
    }

    #[test]
    fn the_intra_deadzone_keeps_more_levels_than_the_inter_one() {
        let n = 8;
        let r = residual(n, 11);
        let coeffs = forward_transform(&r, n, PredMode::Intra, Component::Luma, 8);
        let intra = quantize(&coeffs, n, 34, 8, true);
        let inter = quantize(&coeffs, n, 34, 8, false);
        let magnitude = |levels: &[i32]| levels.iter().map(|l| i64::from(l.abs())).sum::<i64>();
        assert!(
            magnitude(&intra) >= magnitude(&inter),
            "the intra rounding offset must not be the narrower one"
        );
    }

    #[test]
    fn the_four_by_four_intra_luma_block_uses_the_alternate_transform() {
        // §8.6.4 takes the DST only for 4x4 intra luma, so the same residual
        // analysed as chroma or as inter must produce different coefficients.
        let r = residual(4, 5);
        let dst = forward_transform(&r, 4, PredMode::Intra, Component::Luma, 8);
        let dct_inter = forward_transform(&r, 4, PredMode::Inter, Component::Luma, 8);
        let dct_chroma = forward_transform(&r, 4, PredMode::Intra, Component::Cb, 8);
        assert_ne!(dst, dct_inter, "4x4 intra luma did not take the DST path");
        assert_eq!(
            dct_inter, dct_chroma,
            "only 4x4 intra luma may take the DST path"
        );
    }

    #[test]
    fn a_zero_residual_quantizes_to_no_levels_at_all() {
        let n = 16;
        let coeffs = forward_transform(&vec![0i32; n * n], n, PredMode::Intra, Component::Luma, 8);
        assert!(coeffs.iter().all(|&c| c == 0));
        assert!(quantize(&coeffs, n, 26, 8, true).iter().all(|&l| l == 0));
    }
}
