//! Encoder-side forward transform and quantization — the inverse of the
//! decoder's §8.6.3 scaling and §8.6.4 transformation processes.
//!
//! The decoder turns `TransCoeffLevel` into a residual array `r`; this
//! module turns a residual array into `TransCoeffLevel`, using the same
//! integer matrices and the same quantization step ladder so a
//! forward-then-inverse pass reproduces its input up to the quantization
//! error the chosen `qP` implies.
//!
//! Two stages, mirroring the two the decoder runs in the opposite order:
//!
//! * [`forward_transform`] — the separable analysis transform. It is the
//!   transpose of the §8.6.4.2 synthesis: where the decoder computes
//!   `r[ i ] = Σ_j transMatrix[ j ][ i ] * d[ j ]`, the encoder computes
//!   `c[ u ] = Σ_x transMatrix[ u ][ x ] * r[ x ]`. Both share the
//!   `trType` selection (the 4x4 DST-VII of equation 8-316 for
//!   `MODE_INTRA` 4x4 luma, the equation 8-318 DCT-II otherwise) and both
//!   run through the vectorized butterfly in
//!   [`crate::hevc::engine::transform_simd`] — the forward pass simply
//!   feeds it the transposed basis, so the two directions share one
//!   kernel and one bit-exactness guarantee.
//! * [`quantize`] — the reciprocal of the §8.6.3 scaling loop. The
//!   decoder multiplies by `levelScale[ qP % 6 ]` and shifts left by
//!   `qP / 6`; the encoder multiplies by [`QUANT_SCALE`]`[ qP % 6 ]` and
//!   shifts right by `qbits`, with the same `ScalingFactor` matrix
//!   applied as a per-position reciprocal. It dispatches through
//!   [`super::quant_simd`], its own [`crate::simd`] site.
//!
//! [`transform_and_quantize`] runs both for one transform block, and
//! [`luma_qp`] / [`chroma_qp`] derive the §8.6.1 `qP` the two stages
//! need from a slice `QpY`.
//!
//! ## Normalization
//!
//! The forward transform's two passes shift by `log2( nTbS ) + bitDepth
//! − 9` and `log2( nTbS ) + 6`, which together with the decoder's `7`
//! and equation 8-299's `bdShift` make the round trip unit-gain: the
//! §8.6.4.2 matrices satisfy `Tᵀ T = nTbS * 2^12 * I`, so the two
//! directions contribute `2 * log2( nTbS ) + 24` bits of gain and the
//! four shifts remove exactly that many. Intermediates are clipped to
//! the 16-bit dynamic range the decoder's `coeffMin` / `coeffMax` also
//! use, so no stage can hand the next one a value it could not have
//! parsed from a bitstream.
//!
//! ## Scope
//!
//! This is the transform-domain numerics only. Choosing which blocks to
//! code, deriving `cu_qp_delta`, rate-distortion-optimized quantization,
//! and writing the resulting levels as `residual_coding()` syntax are
//! the surrounding encoder's work; this module stops at the
//! `(nTbS)x(nTbS)` level array.

use super::quant_simd::{self, QuantParams};
use crate::hevc::engine::scaling_list::ScalingFactorMatrix;
use crate::hevc::engine::transform::{
    Component, DCT32, DST4, PredMode, TransformError, coeff_range, log2_tbs,
};
use crate::hevc::engine::transform_simd::{self, Backend};

/// The reciprocal of [`crate::hevc::engine::transform::LEVEL_SCALE`],
/// indexed by `qP % 6`.
///
/// Each entry is `round( 2^20 / levelScale[ k ] )`, so
/// `QUANT_SCALE[ k ] * LEVEL_SCALE[ k ] ≈ 2^20` and the encoder's right
/// shift cancels the decoder's left one to within half a step. These are
/// the same six constants the reference encoder uses.
pub const QUANT_SCALE: [i32; 6] = [26214, 23302, 20560, 18396, 16384, 14564];

/// Rounding offset numerator for intra blocks, as `171 / 512` of
/// `1 << qbits` — the deadzone the reference encoder applies where intra
/// residual is expensive to get wrong.
const INTRA_ROUND: i64 = 171;

/// Rounding offset numerator for inter blocks, `85 / 512` — half the
/// intra offset, biasing inter residual towards zero.
const INTER_ROUND: i64 = 85;

/// The number of fractional bits [`INTRA_ROUND`] / [`INTER_ROUND`] are
/// expressed in.
const ROUND_SHIFT: u32 = 9;

/// Builds the transposed §8.6.4.2 DCT-II basis for one `log2( nTbS )`.
///
/// The decoder reads `transMatrix[ i ][ j * stride ]` as
/// `DCT32[ j * stride ][ i ]`; the forward direction needs the transpose
/// of that, `basis[ x * nTbS + u ] = DCT32[ u * stride ][ x ]`, so the
/// same butterfly kernel computes `c[ u ] = Σ_x transMatrix[ u ][ x ] *
/// r[ x ]` without a second implementation. Only the leading
/// `nTbS * nTbS` entries are used; the table is sized for the largest
/// transform so all four sizes share one array type.
const fn forward_dct_basis(log2: usize) -> [i32; 1024] {
    let n = 1usize << log2;
    let stride = 1usize << (5 - log2);
    let mut basis = [0i32; 1024];
    let mut x = 0;
    while x < n {
        let mut u = 0;
        while u < n {
            basis[x * n + u] = DCT32[u * stride][x];
            u += 1;
        }
        x += 1;
    }
    basis
}

/// The four transposed DCT-II bases, indexed by `log2( nTbS ) − 2`.
static FORWARD_DCT: [[i32; 1024]; 4] = [
    forward_dct_basis(2),
    forward_dct_basis(3),
    forward_dct_basis(4),
    forward_dct_basis(5),
];

/// The transposed 4x4 DST-VII basis, `basis[ x * 4 + u ] = DST4[ u ][ x ]`.
const fn forward_dst_basis() -> [i32; 16] {
    let mut basis = [0i32; 16];
    let mut x = 0;
    while x < 4 {
        let mut u = 0;
        while u < 4 {
            basis[x * 4 + u] = DST4[u][x];
            u += 1;
        }
        x += 1;
    }
    basis
}

/// The transposed 4x4 DST-VII basis.
static FORWARD_DST: [i32; 16] = forward_dst_basis();

/// The §8.6.4 `trType` selection, stated from the encoder's side: the
/// 4x4 alternate (DST-VII) transform is used only for `MODE_INTRA` 4x4
/// luma blocks.
#[must_use]
pub fn use_dst(n_tbs: usize, pred_mode: PredMode, component: Component) -> bool {
    n_tbs == 4 && matches!(pred_mode, PredMode::Intra) && !component.is_chroma()
}

/// The dynamic range the forward transform's intermediate pass is
/// clipped to, matching the fixed `[ −32768, 32767 ]` coefficient range
/// of §7.4.5 without `extended_precision_processing_flag`.
const INTERMEDIATE_MIN: i32 = -32768;
/// Upper bound of [`INTERMEDIATE_MIN`]'s range.
const INTERMEDIATE_MAX: i32 = 32767;

/// Forward (analysis) transform for one residual block.
///
/// `residual` is the `(nTbS)x(nTbS)` residual array, row-major by `y`,
/// in the same layout [`crate::hevc::engine::transform::residual_block`]
/// returns. The result is the transform coefficient array `c`, also
/// row-major, clipped into the `[ coeffMin, coeffMax ]` range the
/// decoder's §8.6.3 accepts.
///
/// # Errors
/// [`TransformError::InvalidBlockSize`] for a non-4/8/16/32 `n_tbs`,
/// [`TransformError::LengthMismatch`] if `residual` is not
/// `n_tbs * n_tbs` long, [`TransformError::InvalidBitDepth`] for a
/// `bit_depth` outside 8..=16.
pub fn forward_transform(
    residual: &[i32],
    n_tbs: usize,
    tr_type: bool,
    bit_depth: u8,
    extended_precision: bool,
) -> Result<Vec<i32>, TransformError> {
    forward_transform_with_backend(
        quant_simd::detected(),
        residual,
        n_tbs,
        tr_type,
        bit_depth,
        extended_precision,
    )
}

/// [`forward_transform`] with an explicitly chosen SIMD backend.
///
/// Production callers want [`forward_transform`], which picks the best
/// backend the host supports. This entry point exists so the
/// bit-exactness tests and the encoder benchmark can drive every backend
/// the machine can run, including [`Backend::Scalar`].
///
/// # Errors
/// As for [`forward_transform`].
pub fn forward_transform_with_backend(
    backend: Backend,
    residual: &[i32],
    n_tbs: usize,
    tr_type: bool,
    bit_depth: u8,
    extended_precision: bool,
) -> Result<Vec<i32>, TransformError> {
    let log2 = log2_tbs(n_tbs).ok_or(TransformError::InvalidBlockSize(n_tbs))?;
    if !(8..=16).contains(&bit_depth) {
        return Err(TransformError::InvalidBitDepth(bit_depth));
    }
    let count = n_tbs * n_tbs;
    if residual.len() != count {
        return Err(TransformError::LengthMismatch {
            expected: count,
            got: residual.len(),
        });
    }
    let basis: &[i32] = if tr_type {
        debug_assert_eq!(n_tbs, 4, "the alternate transform is 4x4 only");
        &FORWARD_DST
    } else {
        &FORWARD_DCT[log2 as usize - 2][..count]
    };

    // The two passes together undo `2 * log2( nTbS ) + 24` bits of
    // matrix gain minus what the inverse direction removes; see the
    // module note on normalization. `shift1` is at least 1 for every
    // dimensioned (bitDepth >= 8, log2 >= 2) input, so the rounding
    // offsets below are well-formed.
    let shift1 = log2 + u32::from(bit_depth) - 9;
    let shift2 = log2 + 6;
    let round1 = 1i32 << (shift1 - 1);
    let round2 = 1i32 << (shift2 - 1);
    let (coeff_min, coeff_max) = coeff_range(bit_depth, extended_precision);

    let mut column = vec![0i32; n_tbs];
    let mut pass = vec![0i32; n_tbs];
    // Pass 1 transforms the columns, writing its output transposed so
    // pass 2 reads contiguous rows.
    let mut intermediate = vec![0i32; count];
    for x in 0..n_tbs {
        for (y, c) in column.iter_mut().enumerate() {
            *c = residual[y * n_tbs + x];
        }
        transform_simd::transform_1d(backend, &column, &mut pass, basis, n_tbs, 1);
        for (u, &v) in pass.iter().enumerate() {
            intermediate[u * n_tbs + x] =
                ((v + round1) >> shift1).clamp(INTERMEDIATE_MIN, INTERMEDIATE_MAX);
        }
    }

    // Pass 2 transforms the rows of the transposed intermediate, i.e.
    // the columns of the original, completing `T r Tᵀ`.
    let mut coefficients = vec![0i32; count];
    for u in 0..n_tbs {
        let row = &intermediate[u * n_tbs..][..n_tbs];
        transform_simd::transform_1d(backend, row, &mut pass, basis, n_tbs, 1);
        for (v, &value) in pass.iter().enumerate() {
            let rounded = (i64::from(value) + i64::from(round2)) >> shift2;
            coefficients[u * n_tbs + v] =
                rounded.clamp(i64::from(coeff_min), i64::from(coeff_max)) as i32;
        }
    }
    Ok(coefficients)
}

/// The `qbits` right shift that inverts §8.6.3's `<< ( qP / 6 )` and
/// `>> bdShift` for one block.
///
/// `bdShift` is `bitDepth + log2( nTbS ) + 10 − log2TransformRange`, so
/// `qbits = 24 + qP / 6 − bdShift` makes the encoder's `>> qbits` and
/// the decoder's scaling compose to the identity: `QUANT_SCALE * m *
/// LEVEL_SCALE` contributes `2^20 * 16 = 2^24` of gain.
#[must_use]
fn qbits(q_p: u32, bit_depth: u8, log2: u32, extended_precision: bool) -> i32 {
    let log2_transform_range = if extended_precision {
        core::cmp::max(15, i32::from(bit_depth) + 6)
    } else {
        15
    };
    let bd_shift = i32::from(bit_depth) + log2 as i32 + 10 - log2_transform_range;
    24 + (q_p / 6) as i32 - bd_shift
}

/// Quantization — the reciprocal of the §8.6.3 scaling process.
///
/// Inputs:
/// * `coefficients` — the forward-transform output, row-major by `y`.
/// * `n_tbs` — the block side `nTbS` (4 / 8 / 16 / 32).
/// * `q_p` — the §8.6.1-derived quantization parameter, as
///   [`crate::hevc::engine::transform::scale_coefficients`] takes it.
/// * `bit_depth` / `extended_precision` — fix `bdShift` and the
///   `coeffMin` / `coeffMax` clip, exactly as they do on the decode side.
/// * `intra` — selects the rounding offset ([`INTRA_ROUND`] vs
///   [`INTER_ROUND`]).
/// * `scaling` — the per-position `ScalingFactor` matrix `m[ x ][ y ]`
///   when `scaling_list_enabled_flag == 1`, else `None` for the flat 16.
///   It is inverted into a per-position quantization factor here, so the
///   caller passes the same matrix the decoder will dequantize with.
///
/// Output: the `TransCoeffLevel` array, row-major by `y`.
///
/// # Errors
/// [`TransformError`] as for [`forward_transform`], plus
/// [`TransformError::LengthMismatch`] when `scaling` does not match the
/// block size.
pub fn quantize(
    coefficients: &[i32],
    n_tbs: usize,
    q_p: u32,
    bit_depth: u8,
    extended_precision: bool,
    intra: bool,
    scaling: Option<&ScalingFactorMatrix>,
) -> Result<Vec<i32>, TransformError> {
    quantize_with_backend(
        quant_simd::detected(),
        coefficients,
        n_tbs,
        q_p,
        bit_depth,
        extended_precision,
        intra,
        scaling,
    )
}

/// [`quantize`] with an explicitly chosen SIMD backend, for the
/// bit-exactness tests and the benchmark.
///
/// # Errors
/// As for [`quantize`].
#[allow(clippy::too_many_arguments)]
pub fn quantize_with_backend(
    backend: Backend,
    coefficients: &[i32],
    n_tbs: usize,
    q_p: u32,
    bit_depth: u8,
    extended_precision: bool,
    intra: bool,
    scaling: Option<&ScalingFactorMatrix>,
) -> Result<Vec<i32>, TransformError> {
    let log2 = log2_tbs(n_tbs).ok_or(TransformError::InvalidBlockSize(n_tbs))?;
    if !(8..=16).contains(&bit_depth) {
        return Err(TransformError::InvalidBitDepth(bit_depth));
    }
    let count = n_tbs * n_tbs;
    if coefficients.len() != count {
        return Err(TransformError::LengthMismatch {
            expected: count,
            got: coefficients.len(),
        });
    }
    if let Some(m) = scaling {
        let m_count = m.dim as usize * m.dim as usize;
        if m_count != count {
            return Err(TransformError::LengthMismatch {
                expected: count,
                got: m_count,
            });
        }
    }

    let qbits = qbits(q_p, bit_depth, log2, extended_precision);
    let quant_scale = QUANT_SCALE[(q_p % 6) as usize];
    let (coeff_min, coeff_max) = coeff_range(bit_depth, extended_precision);
    // `171 << ( qbits − 9 )` written so a `qbits` below 9 — reachable
    // only at the extreme end of the extended-precision range — stays
    // exact instead of shifting by a negative amount.
    let round_add = if qbits >= 0 {
        ((if intra { INTRA_ROUND } else { INTER_ROUND }) << qbits) >> ROUND_SHIFT
    } else {
        0
    };

    // §8.6.3 applies `m[ x ][ y ] * levelScale`; the encoder applies the
    // reciprocal `( quantScale << 4 ) / m[ x ][ y ]`, the flat-16 case
    // collapsing to `quantScale` itself. The division is integer, which
    // is what makes the pair invert to within one quantization step
    // rather than exactly.
    let factors = scaling.map(|m| {
        m.coef
            .iter()
            .map(|&value| (quant_scale << 4) / i32::from(value.max(1)))
            .collect::<Vec<i32>>()
    });

    let mut levels = vec![0i32; count];
    quant_simd::quant_block(
        backend,
        &mut levels,
        coefficients,
        factors.as_deref(),
        QuantParams {
            quant_scale,
            qbits: qbits.max(0) as u32,
            round_add,
            coeff_min,
            coeff_max,
        },
    );
    Ok(levels)
}

/// Inputs the forward transform and quantization share for one
/// transform block, gathered so [`transform_and_quantize`] has a stable
/// signature as encoder-side RDPCM and cross-component prediction are
/// added.
#[derive(Debug, Clone, Copy)]
pub struct ForwardBlockParams {
    /// `nTbS` — the transform-block side (4 / 8 / 16 / 32).
    pub n_tbs: usize,
    /// `qP` — the §8.6.1-derived quantization parameter for this block.
    pub q_p: u32,
    /// The colour component `cIdx`.
    pub component: Component,
    /// `CuPredMode[ xTbY ][ yTbY ]`.
    pub pred_mode: PredMode,
    /// `BitDepthY` for luma, `BitDepthC` for chroma.
    pub bit_depth: u8,
    /// `extended_precision_processing_flag`.
    pub extended_precision: bool,
}

/// Runs the forward transform and then quantization for one block — the
/// encoder-side counterpart of
/// [`crate::hevc::engine::transform::residual_block`].
///
/// # Errors
/// [`TransformError`] as for [`forward_transform`] / [`quantize`].
pub fn transform_and_quantize(
    residual: &[i32],
    scaling: Option<&ScalingFactorMatrix>,
    params: ForwardBlockParams,
) -> Result<Vec<i32>, TransformError> {
    transform_and_quantize_with_backend(quant_simd::detected(), residual, scaling, params)
}

/// [`transform_and_quantize`] with an explicitly chosen SIMD backend.
///
/// # Errors
/// As for [`transform_and_quantize`].
pub fn transform_and_quantize_with_backend(
    backend: Backend,
    residual: &[i32],
    scaling: Option<&ScalingFactorMatrix>,
    params: ForwardBlockParams,
) -> Result<Vec<i32>, TransformError> {
    let tr_type = use_dst(params.n_tbs, params.pred_mode, params.component);
    let coefficients = forward_transform_with_backend(
        backend,
        residual,
        params.n_tbs,
        tr_type,
        params.bit_depth,
        params.extended_precision,
    )?;
    quantize_with_backend(
        backend,
        &coefficients,
        params.n_tbs,
        params.q_p,
        params.bit_depth,
        params.extended_precision,
        matches!(params.pred_mode, PredMode::Intra),
        scaling,
    )
}

/// `QpBdOffsetY = 6 * bit_depth_luma_minus8` (§7.4.3.2.1, eq. 7-4).
#[inline]
fn qp_bd_offset(bit_depth: u8) -> i32 {
    6 * (i32::from(bit_depth) - 8)
}

/// §8.6.1 eq. 8-284 — `Qp′Y = QpY + QpBdOffsetY`, the `qP` a luma
/// transform block quantizes with.
#[must_use]
pub fn luma_qp(qp_y: i32, bit_depth: u8) -> u32 {
    (qp_y + qp_bd_offset(bit_depth)).max(0) as u32
}

/// Table 8-10 — `ChromaArrayType == 1` chroma-QP mapping `QpC = f(qPi)`.
#[inline]
fn qpc_420(qpi: i32) -> i32 {
    match qpi {
        x if x < 30 => x,
        30 => 29,
        31 => 30,
        32 => 31,
        33 => 32,
        34 | 35 => 33,
        36 | 37 => 34,
        38 | 39 => 35,
        40 | 41 => 36,
        42 | 43 => 37,
        x => x - 6,
    }
}

/// §8.6.1 — derive `Qp′Cb` / `Qp′Cr`, the `qP` a chroma transform block
/// quantizes with.
///
/// `qPiCx = Clip3( −QpBdOffsetC, 57, QpY + cQpOffset )` with `cQpOffset`
/// the summed PPS, slice, and CU chroma offsets; for
/// `ChromaArrayType == 1` `qPCx = qPC_table( qPiCx )` (Table 8-10), for
/// the other chroma types `qPCx = Min( qPiCx, 51 )`; then
/// `Qp′Cx = qPCx + QpBdOffsetC` (eq. 8-260). This is the same mapping
/// [`crate::hevc::engine::recon`] applies on the decode side, so an
/// encoder and decoder given the same offsets agree on `qP`.
#[must_use]
pub fn chroma_qp(qp_y: i32, chroma_qp_offset: i32, bit_depth_c: u8, chroma_array_type: u8) -> u32 {
    let qp_bd_c = qp_bd_offset(bit_depth_c);
    let qpi = (qp_y + chroma_qp_offset).clamp(-qp_bd_c, 57);
    let qpc = if chroma_array_type == 1 {
        qpc_420(qpi)
    } else {
        qpi.min(51)
    };
    (qpc + qp_bd_c).max(0) as u32
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::engine::transform::{
        LEVEL_SCALE, inverse_transform, scale_coefficients,
    };

    /// Deterministic pseudo-random residual generator. A fixed LCG keeps
    /// the round-trip tests reproducible across hosts, which matters
    /// because they assert numeric bounds rather than exact values.
    fn residual(n_tbs: usize, bit_depth: u8, seed: u64) -> Vec<i32> {
        let span = (1i32 << bit_depth) - 1;
        let mut state = seed;
        (0..n_tbs * n_tbs)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as i32 % (2 * span + 1)) - span
            })
            .collect()
    }

    /// §8.6.2 eq. 8-299 `bdShift` for the non-transform-skip path, the
    /// last shift of the inverse direction. Applying it here lets the
    /// round-trip test compare against the original residual without
    /// going through the whole `residual_block` orchestration.
    fn apply_bd_shift(r: &[i32], bit_depth: u8) -> Vec<i32> {
        let bd_shift = 20 - i32::from(bit_depth);
        let round = 1i64 << (bd_shift - 1);
        r.iter()
            .map(|&v| ((i64::from(v) + round) >> bd_shift) as i32)
            .collect()
    }

    /// The forward transform followed by the decoder's inverse must
    /// reproduce the residual: the four normalization shifts cancel the
    /// matrices' gain exactly, so only rounding error remains.
    #[test]
    fn forward_then_inverse_reproduces_the_residual() {
        for (index, n_tbs) in [4usize, 8, 16, 32].into_iter().enumerate() {
            let source = residual(n_tbs, 8, 0x5eed + index as u64);
            let coefficients = forward_transform(&source, n_tbs, false, 8, false)
                .expect("the block size and bit depth are valid");
            let inverse = inverse_transform(
                &coefficients,
                n_tbs,
                PredMode::Inter,
                Component::Luma,
                8,
                false,
            )
            .expect("the coefficients invert");
            let round_trip = apply_bd_shift(&inverse, 8);
            let worst = source
                .iter()
                .zip(round_trip.iter())
                .map(|(&a, &b)| (a - b).abs())
                .max()
                .expect("blocks are non-empty");
            assert!(
                worst <= 3,
                "{n_tbs}x{n_tbs} round trip drifted by {worst}, expected rounding error only"
            );
        }
    }

    /// The same round trip through the 4x4 DST-VII path, which uses a
    /// different matrix and so a different rounding profile.
    #[test]
    fn forward_then_inverse_reproduces_the_residual_for_the_dst() {
        let source = residual(4, 8, 0xd57);
        let coefficients =
            forward_transform(&source, 4, true, 8, false).expect("4x4 is a valid DST block");
        let inverse = inverse_transform(&coefficients, 4, PredMode::Intra, Component::Luma, 8, false)
            .expect("the coefficients invert");
        let round_trip = apply_bd_shift(&inverse, 8);
        let worst = source
            .iter()
            .zip(round_trip.iter())
            .map(|(&a, &b)| (a - b).abs())
            .max()
            .expect("the block is non-empty");
        assert!(worst <= 3, "the DST round trip drifted by {worst}");
    }

    /// Quantization must invert §8.6.3 scaling: dequantizing a quantized
    /// coefficient has to land within one quantization step of the
    /// original, at every `qP` residue and every block size.
    #[test]
    fn quantize_inverts_the_scaling_process() {
        for n_tbs in [4usize, 8, 16, 32] {
            let log2 = log2_tbs(n_tbs).expect("the size is legal");
            let source = residual(n_tbs, 8, 0xc0ffee + n_tbs as u64);
            let coefficients = forward_transform(&source, n_tbs, false, 8, false)
                .expect("the residual transforms");
            for q_p in [0u32, 7, 13, 26, 34, 45, 51] {
                let levels = quantize(&coefficients, n_tbs, q_p, 8, false, false, None)
                    .expect("the coefficients quantize");
                let scaled = scale_coefficients(&levels, n_tbs, q_p, 8, false, None)
                    .expect("the levels dequantize");
                // One step is what a level of 1 dequantizes to: the
                // flat-16 scaling factor times levelScale, shifted by
                // the same bdShift §8.6.3 applies.
                let bd_shift = 8 + log2 + 10 - 15;
                let step = ((16i64 * i64::from(LEVEL_SCALE[(q_p % 6) as usize]))
                    << (q_p / 6))
                    >> bd_shift;
                for (&want, &got) in coefficients.iter().zip(scaled.iter()) {
                    let error = (i64::from(want) - i64::from(got)).abs();
                    assert!(
                        error <= step,
                        "{n_tbs}x{n_tbs} qP {q_p}: {want} dequantized to {got}, \
                         off by {error} with a step of {step}"
                    );
                }
            }
        }
    }

    /// The intra rounding offset is the larger one, so intra
    /// quantization never rounds a coefficient further towards zero than
    /// inter quantization does.
    #[test]
    fn the_intra_rounding_offset_is_never_smaller_than_the_inter_one() {
        let source = residual(8, 8, 0x1_7a);
        let coefficients =
            forward_transform(&source, 8, false, 8, false).expect("the residual transforms");
        let intra = quantize(&coefficients, 8, 30, 8, false, true, None).expect("intra quantizes");
        let inter = quantize(&coefficients, 8, 30, 8, false, false, None).expect("inter quantizes");
        for (&i, &p) in intra.iter().zip(inter.iter()) {
            assert!(
                i.abs() >= p.abs(),
                "intra level {i} is closer to zero than inter level {p}"
            );
        }
        assert_ne!(intra, inter, "the two rounding offsets must differ somewhere");
    }

    /// A scaling list that halves every position must roughly double the
    /// levels, because the encoder inverts the same matrix the decoder
    /// multiplies by.
    #[test]
    fn the_scaling_list_is_inverted_not_reapplied() {
        let source = residual(4, 8, 0x5ca1e);
        let coefficients =
            forward_transform(&source, 4, false, 8, false).expect("the residual transforms");
        let matrix = ScalingFactorMatrix {
            dim: 4,
            coef: vec![8u16; 16],
        };
        let flat = quantize(&coefficients, 4, 26, 8, false, false, None).expect("flat quantizes");
        let listed = quantize(&coefficients, 4, 26, 8, false, false, Some(&matrix))
            .expect("the listed block quantizes");
        for (&f, &l) in flat.iter().zip(listed.iter()) {
            assert!(
                l.abs() >= f.abs(),
                "halving the scaling factor should not shrink level {l} below {f}"
            );
        }
        // And the pair still inverts: dequantizing with the same matrix
        // lands back near the original coefficients.
        let scaled = scale_coefficients(&listed, 4, 26, 8, false, Some(&matrix))
            .expect("the levels dequantize");
        let bd_shift = 8 + 2 + 10 - 15;
        let step = ((8i64 * i64::from(LEVEL_SCALE[26 % 6])) << (26 / 6)) >> bd_shift;
        for (&want, &got) in coefficients.iter().zip(scaled.iter()) {
            let error = (i64::from(want) - i64::from(got)).abs();
            assert!(error <= step, "{want} dequantized to {got}, off by {error}");
        }
    }

    /// Every vector backend the host can run must be bit-identical to the
    /// scalar reference, for both stages and every block size.
    #[test]
    fn every_backend_is_bit_exact_with_scalar() {
        let matrix = ScalingFactorMatrix {
            dim: 8,
            coef: (0..64).map(|i| 16 + (i as u16 % 9)).collect(),
        };
        for n_tbs in [4usize, 8, 16, 32] {
            for tr_type in [false, true] {
                if tr_type && n_tbs != 4 {
                    continue;
                }
                let source = residual(n_tbs, 8, 0xbee5 + n_tbs as u64 + u64::from(tr_type));
                let reference =
                    forward_transform_with_backend(Backend::Scalar, &source, n_tbs, tr_type, 8, false)
                        .expect("the scalar reference transforms");
                for backend in quant_simd::supported_backends() {
                    let actual =
                        forward_transform_with_backend(backend, &source, n_tbs, tr_type, 8, false)
                            .expect("every backend transforms");
                    assert_eq!(
                        actual, reference,
                        "{backend:?} forward transform diverged at {n_tbs}x{n_tbs}"
                    );
                }
                for q_p in [0u32, 11, 26, 51] {
                    let scaling = (n_tbs == 8).then_some(&matrix);
                    let want = quantize_with_backend(
                        Backend::Scalar,
                        &reference,
                        n_tbs,
                        q_p,
                        8,
                        false,
                        tr_type,
                        scaling,
                    )
                    .expect("the scalar reference quantizes");
                    for backend in quant_simd::supported_backends() {
                        let got = quantize_with_backend(
                            backend, &reference, n_tbs, q_p, 8, false, tr_type, scaling,
                        )
                        .expect("every backend quantizes");
                        assert_eq!(
                            got, want,
                            "{backend:?} quantization diverged at {n_tbs}x{n_tbs} qP {q_p}"
                        );
                    }
                }
            }
        }
    }

    /// §8.6.4 `trType`: the alternate transform is 4x4 intra luma only.
    #[test]
    fn the_alternate_transform_is_intra_4x4_luma_only() {
        assert!(use_dst(4, PredMode::Intra, Component::Luma));
        assert!(!use_dst(4, PredMode::Inter, Component::Luma));
        assert!(!use_dst(4, PredMode::Intra, Component::Cb));
        assert!(!use_dst(4, PredMode::Intra, Component::Cr));
        assert!(!use_dst(8, PredMode::Intra, Component::Luma));
    }

    /// A flat residual is pure DC, so only coefficient 0 may be non-zero
    /// — the property that makes the transposed basis demonstrably the
    /// analysis direction rather than a second copy of the synthesis one.
    #[test]
    fn a_flat_residual_produces_only_a_dc_coefficient() {
        for n_tbs in [4usize, 8, 16, 32] {
            let source = vec![100i32; n_tbs * n_tbs];
            let coefficients =
                forward_transform(&source, n_tbs, false, 8, false).expect("the block transforms");
            assert!(coefficients[0] > 0, "{n_tbs}x{n_tbs} lost its DC term");
            for (index, &value) in coefficients.iter().enumerate().skip(1) {
                assert_eq!(value, 0, "{n_tbs}x{n_tbs} coefficient {index} is not zero");
            }
        }
    }

    /// The forward transform rejects the same malformed inputs the
    /// decoder's §8.6.3 / §8.6.4 entry points do.
    #[test]
    fn malformed_blocks_are_rejected() {
        assert_eq!(
            forward_transform(&[0; 36], 6, false, 8, false),
            Err(TransformError::InvalidBlockSize(6))
        );
        assert_eq!(
            forward_transform(&[0; 15], 4, false, 8, false),
            Err(TransformError::LengthMismatch {
                expected: 16,
                got: 15
            })
        );
        assert_eq!(
            forward_transform(&[0; 16], 4, false, 7, false),
            Err(TransformError::InvalidBitDepth(7))
        );
        assert_eq!(
            quantize(&[0; 16], 4, 26, 8, false, false, None).map(|l| l.len()),
            Ok(16)
        );
        let mismatched = ScalingFactorMatrix {
            dim: 8,
            coef: vec![16; 64],
        };
        assert_eq!(
            quantize(&[0; 16], 4, 26, 8, false, false, Some(&mismatched)),
            Err(TransformError::LengthMismatch {
                expected: 16,
                got: 64
            })
        );
    }

    /// §8.6.1: `Qp′Y` adds `QpBdOffsetY`, and the chroma mapping follows
    /// Table 8-10 for 4:2:0 while 4:4:4 only clamps.
    #[test]
    fn qp_derivation_follows_section_8_6_1() {
        assert_eq!(luma_qp(26, 8), 26);
        assert_eq!(luma_qp(26, 10), 38);
        // Table 8-10 is identity below 30 and flattens through 30..43.
        assert_eq!(chroma_qp(29, 0, 8, 1), 29);
        assert_eq!(chroma_qp(37, 0, 8, 1), 34);
        assert_eq!(chroma_qp(44, 0, 8, 1), 38);
        // ChromaArrayType 3 takes Min( qPi, 51 ) instead.
        assert_eq!(chroma_qp(37, 0, 8, 3), 37);
        assert_eq!(chroma_qp(55, 0, 8, 3), 51);
        // The PPS/slice offset moves qPi before the mapping.
        assert_eq!(chroma_qp(37, -5, 8, 3), 32);
    }
}
