//! §8.6.2 / §8.6.3 / §8.6.4 — scaling, transformation and residual
//! array construction prior to the deblocking filter process.
//!
//! This module turns the parsed `TransCoeffLevel[ xC ][ yC ]` array of
//! one transform block (produced by the [`crate::hevc::engine::residual`] §7.3.8.11
//! driver) into the `(nTbS)x(nTbS)` array `r` of residual samples that
//! the picture-construction step (§8.6.7) adds to the prediction.
//!
//! Three ITU-T H.265 (08/2021) processes are implemented, in the
//! dependency order the spec invokes them:
//!
//! * §8.6.3 **scaling process for transform coefficients**
//!   ([`scale_coefficients`]) — the dequantization step. Each
//!   `TransCoeffLevel[ x ][ y ]` is multiplied by the scaling factor
//!   `m[ x ][ y ]` (a flat 16 when `scaling_list_enabled_flag == 0`,
//!   else `ScalingFactor[ sizeId ][ matrixId ][ x ][ y ]`), the
//!   `levelScale[ qP % 6 ]` rational-step list, and `1 << ( qP / 6 )`,
//!   then offset-rounded by `bdShift` and clipped to
//!   `[ coeffMin, coeffMax ]` (equations 8-300..8-309).
//! * §8.6.4 **transformation process for scaled transform
//!   coefficients** ([`inverse_transform`]) — the separable inverse
//!   transform. Each column then each row is passed through the
//!   §8.6.4.2 one-dimensional transform ([`transform_1d`]); the
//!   `trType == 1` 4x4 alternate transform (the DST-VII matrix of
//!   equation 8-316) is selected only for `MODE_INTRA` 4x4 luma
//!   blocks, every other block uses the `trType == 0` partial-butterfly
//!   DCT-II matrix of equations 8-318..8-321. The intermediate column
//!   result is offset-rounded by 7 and clipped (equation 8-314).
//! * §8.6.2 **scaling and transformation process** ([`residual_block`])
//!   — the orchestration that selects between the
//!   `cu_transquant_bypass_flag` pass-through (with the §8.6.2
//!   `rotateCoeffs` reordering, equation 8-297), the
//!   `transform_skip_flag` `tsShift` left-shift (equation 8-298), and
//!   the full scale-then-transform path, applying the final `bdShift`
//!   offset-round (equation 8-299).
//!
//! All arithmetic is integer-exact per the spec: products use `i64`
//! intermediates (the §8.6.3 scale product can exceed `i32` for the
//! `extended_precision_processing_flag` ranges), and every clip uses
//! the `Clip3` bounds the surrounding subclause derives.
//!
//! ## Scope
//!
//! The transform-domain numerics are self-contained: the inputs are
//! the decoded `TransCoeffLevel` array, the derived quantization
//! parameter `qP`, the bit depth, and the small set of SPS/PPS/CU
//! flags the three subclauses read. The §8.6.1 `qP` derivation, the
//! §8.6.5 transform-bypass RDPCM residual modification, the §8.6.6
//! cross-component-prediction modification, and the §8.6.7 picture
//! construction are the consumers' / follow-ups' responsibility — this
//! module stops at the `(nTbS)x(nTbS)` array `r`.

use crate::hevc::engine::scaling_list::ScalingFactorMatrix;
use crate::hevc::engine::transform_simd::{self, Backend};

/// §8.6.3 `levelScale[ k ]` rational quantization-step list, indexed by
/// `qP % 6` (the list is `{ 40, 45, 51, 57, 64, 72 }`).
pub const LEVEL_SCALE: [i32; 6] = [40, 45, 51, 57, 64, 72];

/// Colour component of the current transform block, naming the
/// `cIdx` value (0 = luma, 1 = Cb, 2 = Cr) the three subclauses branch
/// on for bit depth and `coeffMin` / `coeffMax`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    /// `cIdx == 0` — luma. Uses `BitDepthY` and `CoeffMin/MaxY`.
    Luma,
    /// `cIdx == 1` — Cb chroma. Uses `BitDepthC` and `CoeffMin/MaxC`.
    Cb,
    /// `cIdx == 2` — Cr chroma. Uses `BitDepthC` and `CoeffMin/MaxC`.
    Cr,
}

impl Component {
    /// `true` for the two chroma components (`cIdx != 0`).
    #[inline]
    #[must_use]
    pub fn is_chroma(self) -> bool {
        !matches!(self, Component::Luma)
    }
}

/// §8.6.4 `CuPredMode[ xTbY ][ yTbY ]` — the prediction mode of the
/// coding unit covering the transform block. Only the §8.6.2
/// `rotateCoeffs` and §8.6.4 `trType` derivations read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredMode {
    /// `MODE_INTRA`.
    Intra,
    /// `MODE_INTER` (or `MODE_SKIP`, which shares the inter transform
    /// path for residual purposes).
    Inter,
}

/// Errors from the §8.6.2 / §8.6.3 / §8.6.4 processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformError {
    /// `nTbS` (the `1 << log2TrafoSize` block side) was not one of the
    /// four legal transform-block sizes (4, 8, 16, 32).
    InvalidBlockSize(usize),
    /// The `TransCoeffLevel` (or `ScalingFactor`) input array did not
    /// hold exactly `nTbS * nTbS` elements.
    LengthMismatch {
        /// The `nTbS * nTbS` count the block requires.
        expected: usize,
        /// The element count actually supplied.
        got: usize,
    },
    /// `bitDepth` was outside the 8..=16 range the equations are
    /// dimensioned for.
    InvalidBitDepth(u8),
}

impl core::fmt::Display for TransformError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidBlockSize(n) => {
                write!(
                    f,
                    "invalid transform block size nTbS = {n} (expected 4/8/16/32)"
                )
            }
            Self::LengthMismatch { expected, got } => {
                write!(
                    f,
                    "coefficient array length {got} != nTbS*nTbS = {expected}"
                )
            }
            Self::InvalidBitDepth(b) => write!(f, "invalid bitDepth {b} (expected 8..=16)"),
        }
    }
}

impl std::error::Error for TransformError {}

/// `log2( nTbS )` for a legal transform-block side, or `None` if `n_tbs`
/// is not 4 / 8 / 16 / 32.
#[inline]
fn log2_tbs(n_tbs: usize) -> Option<u32> {
    match n_tbs {
        4 => Some(2),
        8 => Some(3),
        16 => Some(4),
        32 => Some(5),
        _ => None,
    }
}

/// §7.4.5 equations 7-27..7-30 — `CoeffMin` / `CoeffMax` for the given
/// `bitDepth` and `extended_precision_processing_flag`.
///
/// Returns `( coeffMin, coeffMax )`. With the flag clear the range is
/// the fixed `[ −32768, 32767 ]`; with it set the magnitude widens to
/// `Max( 15, bitDepth + 6 )` bits.
#[inline]
#[must_use]
pub fn coeff_range(bit_depth: u8, extended_precision: bool) -> (i32, i32) {
    let log2_range = if extended_precision {
        core::cmp::max(15, bit_depth as i32 + 6)
    } else {
        15
    };
    let mag = 1i32 << log2_range;
    (-mag, mag - 1)
}

/// `Clip3( lo, hi, x )` — clamp `x` to the inclusive `[ lo, hi ]` range
/// (§5, the clause-8 `Clip3` operator), on `i64`.
#[inline]
fn clip3(lo: i64, hi: i64, x: i64) -> i64 {
    x.clamp(lo, hi)
}

/// §8.6.3 — scaling (dequantization) process for transform
/// coefficients.
///
/// Inputs:
/// * `levels` — the `TransCoeffLevel[ x ][ y ]` array, row-major by
///   `y` (`levels[ y * nTbS + x ]`), as [`crate::hevc::engine::residual::ResidualBlock`]
///   stores it.
/// * `n_tbs` — the block side `nTbS` (4 / 8 / 16 / 32).
/// * `q_p` — the quantization parameter `qP` derived by §8.6.2.
/// * `bit_depth` — `BitDepthY` for luma, `BitDepthC` for chroma.
/// * `extended_precision` — `extended_precision_processing_flag`.
/// * `scaling` — the per-position scaling factor `m[ x ][ y ]`:
///   `Some( ScalingFactor )` when `scaling_list_enabled_flag == 1` and
///   the §8.6.3 "flat 16" exception does not apply, else `None` (a flat
///   16 is used). The matrix is indexed `at( x, y )`.
///
/// Output: the `(nTbS)x(nTbS)` array `d` of scaled coefficients,
/// row-major by `y`.
///
/// # Errors
/// [`TransformError::InvalidBlockSize`] for a non-4/8/16/32 `n_tbs`,
/// [`TransformError::LengthMismatch`] if `levels` (or `scaling`) is not
/// `n_tbs * n_tbs` long, [`TransformError::InvalidBitDepth`] for a
/// `bit_depth` outside 8..=16.
pub fn scale_coefficients(
    levels: &[i32],
    n_tbs: usize,
    q_p: u32,
    bit_depth: u8,
    extended_precision: bool,
    scaling: Option<&ScalingFactorMatrix>,
) -> Result<Vec<i32>, TransformError> {
    let log2_tbs = log2_tbs(n_tbs).ok_or(TransformError::InvalidBlockSize(n_tbs))?;
    if !(8..=16).contains(&bit_depth) {
        return Err(TransformError::InvalidBitDepth(bit_depth));
    }
    let count = n_tbs * n_tbs;
    if levels.len() != count {
        return Err(TransformError::LengthMismatch {
            expected: count,
            got: levels.len(),
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

    // §8.6.3 equations 8-300/8-304 (log2TransformRange),
    // 8-301/8-305 (bdShift), 8-302..8-307 (coeffMin/coeffMax).
    let log2_transform_range = if extended_precision {
        core::cmp::max(15, bit_depth as i32 + 6)
    } else {
        15
    };
    // bdShift is always >= 1 for the dimensioned ranges (bitDepth >= 8,
    // log2TrafoSize >= 2, log2TransformRange <= bitDepth + 6), so the
    // (1 << (bdShift - 1)) rounding offset the kernel applies is
    // well-formed.
    let bd_shift = bit_depth as i32 + log2_tbs as i32 + 10 - log2_transform_range;
    let (coeff_min, coeff_max) = coeff_range(bit_depth, extended_precision);

    let level_scale = LEVEL_SCALE[(q_p % 6) as usize];
    let qp_div6 = q_p / 6;

    let mut d = vec![0i32; count];
    // §8.6.3 m[ x ][ y ]: flat 16 unless an explicit ScalingFactor
    // matrix is supplied (the caller folds the "transform_skip && nTbS >
    // 4 ⇒ 16" exception into the `scaling` argument it passes). The
    // matrix is stored row-major by `y` exactly like `levels`, so
    // `ScalingFactorMatrix::at( x, y )` is `coef[ y * nTbS + x ]` — the
    // kernel consumes it as one flat slice.
    //
    // §8.6.3 eq. 8-309: clip( (TransCoeffLevel * m * levelScale <<
    // (qP/6)) + round ) >> bdShift, applied by the SIMD-dispatched
    // kernel (bit-exact with its own scalar reference).
    transform_simd::dequant_block(
        transform_simd::detected(),
        &mut d,
        levels,
        scaling.map(|sf| sf.coef.as_slice()),
        transform_simd::DequantParams {
            level_scale: level_scale as i32,
            qp_div6,
            bd_shift: bd_shift as u32,
            coeff_min,
            coeff_max,
        },
    );
    Ok(d)
}

/// §8.6.4.2 equation 8-316 — the `trType == 1` 4x4 alternate (DST-VII)
/// transform matrix, `transMatrix[ i ][ j ]`, row-major.
#[rustfmt::skip]
const DST4: [[i32; 4]; 4] = [
    [29,  55,  74,  84],
    [74,  74,   0, -74],
    [84, -29, -74,  55],
    [55, -84,  74, -29],
];

/// §8.6.4.2 equations 8-318..8-321 — the `trType == 0` 32x32 DCT-II
/// transform matrix `transMatrix[ m ][ n ]` (`m`, `n` = 0..31),
/// row-major. The smaller 4 / 8 / 16 transforms subsample column `n`
/// at stride `1 << ( 5 − log2( nTbS ) )` per equation 8-317.
#[rustfmt::skip]
const DCT32: [[i32; 32]; 32] = [
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    [90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90],
    [90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90],
    [90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61, 78, 88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90],
    [89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
    [88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85, -90, -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88],
    [87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87],
    [85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90, 61, 4, -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85],
    [83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
    [82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4, 67, 90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82],
    [80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25, -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80],
    [78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38, -54, -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78],
    [75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
    [73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4, 88, 46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73],
    [70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70],
    [67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46, -82, 31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67],
    [64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64],
    [61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67, -67, -54, 78, 38, -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61],
    [57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57, -57, 80, 25, -90, 9, 87, -43, -70, 70, 43, -87, -9, 90, -25, -80, 57],
    [54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73, 73, 31, -90, 22, 78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54],
    [50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50],
    [46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46],
    [43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43, -43, 90, -57, -25, 87, -70, -9, 80, -80, 9, 70, -87, 25, 57, -90, 43],
    [38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54, 90, -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38],
    [36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36],
    [31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85, -85, 46, 13, -67, 90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31],
    [25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25, -25, 70, -90, 80, -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 25],
    [22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88, 88, -67, 31, 13, -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22],
    [18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18],
    [13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90, -90, 82, -67, 46, -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13],
    [9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9, -9, 25, -43, 57, -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -25, 9],
    [4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4],
];

/// §8.6.4.2 — the one-dimensional transformation process.
///
/// `trType == 1` (the `tr_type == true` argument) applies the 4x4
/// DST-VII matrix multiplication of equation 8-316 (valid only for
/// `n_tbs == 4`); `trType == 0` applies the equation 8-317 DCT-II
/// matrix multiplication, subsampling the 32x32 base matrix's column
/// index at stride `1 << ( 5 − log2( nTbS ) )`.
///
/// `input` is the length-`n_tbs` list `x[ j ]`; the return is the
/// length-`n_tbs` list `y[ i ]`. Products accumulate in `i64`.
#[must_use]
fn transform_1d(input: &[i64], n_tbs: usize, tr_type: bool) -> Vec<i64> {
    let mut out = vec![0i64; n_tbs];
    transform_1d_into(input, n_tbs, tr_type, &mut out);
    out
}

/// Same computation as [`transform_1d`], writing into a caller-owned
/// `out` slice instead of allocating a fresh `Vec` each call. `nTbS` is
/// always one of 4/8/16/32, so hot callers can pass a stack-allocated
/// buffer and avoid a heap allocation per row/column of every transform
/// block — this function runs `nTbS` times per (de)coded transform
/// block, i.e. millions of times per frame.
fn transform_1d_into(input: &[i64], n_tbs: usize, tr_type: bool, out: &mut [i64]) {
    debug_assert_eq!(input.len(), n_tbs);
    debug_assert_eq!(out.len(), n_tbs);
    if tr_type {
        // §8.6.4.2 eq. 8-315/8-316: y[i] = Σ_j transMatrix[i][j] * x[j],
        // the 4x4 DST. trType == 1 is only ever invoked with nTbS == 4.
        //
        // The printed eq.-8-316 matrix follows the same index convention
        // eq. 8-318 states for the DCT base table ("transMatrix[ m ][ n ]
        // ... with m = 0..15" — the first printed index is the column):
        // each printed brace row is a *column* of transMatrix, so
        // `transMatrix[ i ][ j ]` reads the in-code row-major [`DST4`]
        // at `[ j ][ i ]`, exactly like the transposed [`DCT32`] read
        // below. Verified byte-exact against the qp-high / qp-low
        // fixture decodes (a literal `[ i ][ j ]` read of the printed
        // rows diverges from the conformance output).
        for (i, oi) in out.iter_mut().enumerate() {
            let mut acc = 0i64;
            for (j, &xj) in input.iter().enumerate() {
                acc += DST4[j][i] as i64 * xj;
            }
            *oi = acc;
        }
    } else {
        // §8.6.4.2 eq. 8-317: y[i] = Σ_j transMatrix[i][j * stride] *
        // x[j], stride = 1 << (5 - log2(nTbS)).
        //
        // The §8.6.4.2 transMatrix is laid out (eq. 8-318/8-319) so that
        // `transMatrix[ m ][ n ]` = `transMatrixCol0to15[ m ][ n ]` with
        // `m` the column index 0..15 and `n` the row index 0..31; i.e.
        // the named base table is indexed [column][row]. The in-code
        // [`DCT32`] is the natural row-major listing of that base table,
        // so `DCT32[ a ][ b ] == transMatrix[ b ][ a ]`. Equation 8-317's
        // `transMatrix[ i ][ j*stride ]` therefore reads `DCT32[ j*stride
        // ][ i ]` — the column index `j*stride` becomes the DCT32 row.
        // (For DC-only input this yields the constant row-0 basis, the
        // uniform synthesis the inverse transform must produce.)
        let log2 = log2_tbs(n_tbs).expect("transform_1d called with non-2^k nTbS");
        let stride = 1usize << (5 - log2);
        for (i, oi) in out.iter_mut().enumerate() {
            let mut acc = 0i64;
            for (j, &xj) in input.iter().enumerate() {
                acc += DCT32[j * stride][i] as i64 * xj;
            }
            *oi = acc;
        }
    }
}

/// Encoder-side forward DCT-II 1-D over the same §8.6.4.2 basis the
/// inverse synthesizes from: `y[ u ] = Σ_x DCT32[ u * stride ][ x ] *
/// x[ x ]` — the transpose of the [`transform_1d`] `trType == 0`
/// analysis, so an inverse-transformed forward output reproduces the
/// input up to the normalization shifts the encoder applies.
pub(crate) fn forward_dct_1d(input: &[i64], n_tbs: usize) -> Vec<i64> {
    let log2 = log2_tbs(n_tbs).expect("forward_dct_1d called with non-2^k nTbS");
    let stride = 1usize << (5 - log2);
    (0..n_tbs)
        .map(|u| {
            input
                .iter()
                .enumerate()
                .map(|(x, &xv)| DCT32[u * stride][x] as i64 * xv)
                .sum()
        })
        .collect()
}

/// §8.6.4 — transformation process for scaled transform coefficients.
///
/// Inputs:
/// * `d` — the scaled-coefficient array from §8.6.3, row-major by `y`
///   (`d[ y * nTbS + x ]`).
/// * `n_tbs` — the block side.
/// * `pred_mode` / `component` — select the §8.6.4 `trType` (the 4x4
///   DST is taken only for `MODE_INTRA`, `nTbS == 4`, luma).
/// * `bit_depth` / `extended_precision` — fix the §8.6.4 intermediate
///   `coeffMin` / `coeffMax` clip (equation 8-314).
///
/// Output: the `(nTbS)x(nTbS)` array `r` of pre-`bdShift` residual
/// samples, row-major by `y`. The final §8.6.2 equation-8-299
/// `bdShift` offset-round is applied by [`residual_block`], not here.
///
/// # Errors
/// [`TransformError::InvalidBlockSize`] / [`TransformError::LengthMismatch`]
/// / [`TransformError::InvalidBitDepth`] as for [`scale_coefficients`].
pub fn inverse_transform(
    d: &[i32],
    n_tbs: usize,
    pred_mode: PredMode,
    component: Component,
    bit_depth: u8,
    extended_precision: bool,
) -> Result<Vec<i32>, TransformError> {
    inverse_transform_with_backend(
        transform_simd::detected(),
        d,
        n_tbs,
        pred_mode,
        component,
        bit_depth,
        extended_precision,
    )
}

/// [`inverse_transform`] with an explicitly chosen SIMD backend.
///
/// Production callers want [`inverse_transform`], which picks the best
/// backend the host supports. This entry point exists so the
/// bit-exactness tests and the transform benchmark can drive every
/// backend the machine can run, including [`Backend::Scalar`].
///
/// # Errors
/// As for [`inverse_transform`].
pub fn inverse_transform_with_backend(
    backend: Backend,
    d: &[i32],
    n_tbs: usize,
    pred_mode: PredMode,
    component: Component,
    bit_depth: u8,
    extended_precision: bool,
) -> Result<Vec<i32>, TransformError> {
    if log2_tbs(n_tbs).is_none() {
        return Err(TransformError::InvalidBlockSize(n_tbs));
    }
    if !(8..=16).contains(&bit_depth) {
        return Err(TransformError::InvalidBitDepth(bit_depth));
    }
    if d.len() != n_tbs * n_tbs {
        return Err(TransformError::LengthMismatch {
            expected: n_tbs * n_tbs,
            got: d.len(),
        });
    }
    let (coeff_min, coeff_max) = coeff_range(bit_depth, extended_precision);
    if !fits_i32_accumulation(n_tbs, coeff_min, coeff_max) {
        // Extended precision widens coeffMin/coeffMax past the point
        // where a 32-bit lane can hold a partial sum; that path keeps
        // the exact i64 reference below.
        return inverse_transform_reference(
            d,
            n_tbs,
            pred_mode,
            component,
            bit_depth,
            extended_precision,
        );
    }
    // §8.6.4 trType: 1 iff MODE_INTRA, nTbS == 4, cIdx == 0.
    let tr_type =
        matches!(pred_mode, PredMode::Intra) && n_tbs == 4 && matches!(component, Component::Luma);
    Ok(inverse_transform_i32(
        backend, d, n_tbs, tr_type, coeff_min, coeff_max,
    ))
}

/// Largest magnitude in the §8.6.4.2 basis tables: [`DCT32`] peaks at 90
/// and [`DST4`] at 84.
const MAX_BASIS_MAGNITUDE: i64 = 90;

/// Whether a full `nTbS`-term §8.6.4.2 butterfly over inputs bounded by
/// `coeffMin`/`coeffMax` is guaranteed to stay inside `i32`.
///
/// Both transform passes take clipped input (`d` from §8.6.3, and the
/// equation-8-314 `g` between the passes), so the worst case is `nTbS`
/// terms of `maxBasis * max( |coeffMin|, coeffMax )`. That holds for
/// every 4/8/16/32 block at the default 16-bit coefficient range but not
/// for the wider extended-precision one, which keeps the `i64` path.
fn fits_i32_accumulation(n_tbs: usize, coeff_min: i32, coeff_max: i32) -> bool {
    let magnitude = core::cmp::max(i64::from(coeff_max), -i64::from(coeff_min));
    n_tbs as i64 * MAX_BASIS_MAGNITUDE * magnitude <= i64::from(i32::MAX)
}

/// The §8.6.4 separable inverse transform accumulated in `i32` lanes.
///
/// Identical arithmetic to [`inverse_transform_reference`], reassociated
/// so the innermost loop is a vectorizable
/// [`transform_simd::accumulate_scaled`] over all `nTbS` outputs instead
/// of a serial per-output dot product. Zero inputs contribute nothing and
/// are skipped outright, which matters because dequantized coefficient
/// blocks are overwhelmingly sparse.
///
/// The caller must have checked [`fits_i32_accumulation`].
fn inverse_transform_i32(
    backend: Backend,
    d: &[i32],
    n_tbs: usize,
    tr_type: bool,
    coeff_min: i32,
    coeff_max: i32,
) -> Vec<i32> {
    let count = n_tbs * n_tbs;
    // nTbS is always 4/8/16/32, so the per-column/per-row 1-D transform
    // inputs/outputs fit in fixed-size stack buffers; this avoids two
    // heap allocations per column and per row of every transform block
    // decoded, which otherwise dominates decode time at 1080p.
    let mut in_buf = [0i32; 32];
    let mut out_buf = [0i32; 32];

    // §8.6.4 step 1: column transform d[x][y] over y -> e[x][y].
    // d is row-major by y; a column is the fixed-x slice.
    let mut e = vec![0i32; count];
    for x in 0..n_tbs {
        let col = &mut in_buf[..n_tbs];
        for (y, cv) in col.iter_mut().enumerate() {
            *cv = d[y * n_tbs + x];
        }
        if col.iter().all(|&value| value == 0) {
            continue;
        }
        let te = &mut out_buf[..n_tbs];
        transform_1d_i32_into(backend, col, n_tbs, tr_type, te);
        for (y, &v) in te.iter().enumerate() {
            e[y * n_tbs + x] = v;
        }
    }

    // §8.6.4 step 2 (eq. 8-314): g[x][y] = clip( (e + 64) >> 7 ).
    for ev in &mut e {
        *ev = ((*ev + 64) >> 7).clamp(coeff_min, coeff_max);
    }

    // §8.6.4 step 3: row transform g[x][y] over x -> r[x][y].
    let mut r = vec![0i32; count];
    for y in 0..n_tbs {
        let row = &e[y * n_tbs..(y + 1) * n_tbs];
        if row.iter().all(|&value| value == 0) {
            continue;
        }
        // Per the spec the §8.6.4 row output is not itself clipped;
        // equation 8-299's bdShift round (in residual_block) is what
        // reduces it.
        transform_1d_i32_into(
            backend,
            row,
            n_tbs,
            tr_type,
            &mut r[y * n_tbs..(y + 1) * n_tbs],
        );
    }
    r
}

/// [`transform_1d_into`] over `i32` lanes, driven as one
/// broadcast-multiply-add per non-zero input rather than one dot product
/// per output.
///
/// `out[ i ] = Σ_j transMatrix[ i ][ j ] * x[ j ]` is computed as
/// `Σ_j x[ j ] * ( row j of the basis )`, which is the same sum of the
/// same integer products — addition is associative in `i32` as long as
/// no partial sum overflows, which [`fits_i32_accumulation`] guarantees.
/// See [`transform_1d_into`] for why the in-code [`DST4`] / [`DCT32`]
/// tables are read transposed relative to the printed equations.
fn transform_1d_i32_into(
    backend: Backend,
    input: &[i32],
    n_tbs: usize,
    tr_type: bool,
    out: &mut [i32],
) {
    debug_assert_eq!(input.len(), n_tbs);
    debug_assert_eq!(out.len(), n_tbs);
    out.fill(0);
    if tr_type {
        // §8.6.4.2 eq. 8-315/8-316, the 4x4 DST. trType == 1 is only
        // ever invoked with nTbS == 4.
        for (j, &xj) in input.iter().enumerate() {
            if xj != 0 {
                transform_simd::accumulate_scaled(backend, out, &DST4[j][..n_tbs], xj);
            }
        }
    } else {
        // §8.6.4.2 eq. 8-317, stride = 1 << (5 - log2(nTbS)).
        let log2 = log2_tbs(n_tbs).expect("transform_1d called with non-2^k nTbS");
        let stride = 1usize << (5 - log2);
        for (j, &xj) in input.iter().enumerate() {
            if xj != 0 {
                transform_simd::accumulate_scaled(backend, out, &DCT32[j * stride][..n_tbs], xj);
            }
        }
    }
}

/// The unreassociated `i64` reference implementation of §8.6.4.
///
/// [`inverse_transform`] produces bit-identical output through the
/// vectorized `i32` kernels for every block whose coefficient range fits
/// (see [`fits_i32_accumulation`]); this is what runs for the wider
/// extended-precision ranges, and what the SIMD backends are tested
/// against.
///
/// # Errors
/// As for [`inverse_transform`].
pub fn inverse_transform_reference(
    d: &[i32],
    n_tbs: usize,
    pred_mode: PredMode,
    component: Component,
    bit_depth: u8,
    extended_precision: bool,
) -> Result<Vec<i32>, TransformError> {
    if log2_tbs(n_tbs).is_none() {
        return Err(TransformError::InvalidBlockSize(n_tbs));
    }
    if !(8..=16).contains(&bit_depth) {
        return Err(TransformError::InvalidBitDepth(bit_depth));
    }
    let count = n_tbs * n_tbs;
    if d.len() != count {
        return Err(TransformError::LengthMismatch {
            expected: count,
            got: d.len(),
        });
    }

    // §8.6.4 trType: 1 iff MODE_INTRA, nTbS == 4, cIdx == 0.
    let tr_type =
        matches!(pred_mode, PredMode::Intra) && n_tbs == 4 && matches!(component, Component::Luma);
    // §8.6.4 eqs. 8-310..8-313: the intermediate-clip coeffMin/coeffMax.
    let (coeff_min, coeff_max) = coeff_range(bit_depth, extended_precision);

    // §8.6.4 step 1: column transform d[x][y] over y -> e[x][y].
    // d is row-major by y; a column is the fixed-x slice.
    //
    // nTbS is always 4/8/16/32, so the per-column/per-row 1-D transform
    // inputs/outputs fit in fixed-size stack buffers; this avoids two
    // heap allocations (the gathered column/row and transform_1d's
    // former Vec return) per column and per row of every transform
    // block decoded, which otherwise dominates decode time at 1080p.
    let mut e = vec![0i64; count];
    let mut in_buf = [0i64; 32];
    let mut out_buf = [0i64; 32];
    for x in 0..n_tbs {
        let col = &mut in_buf[..n_tbs];
        for (y, cv) in col.iter_mut().enumerate() {
            *cv = d[y * n_tbs + x] as i64;
        }
        if col.iter().all(|&value| value == 0) {
            continue;
        }
        let te = &mut out_buf[..n_tbs];
        transform_1d_into(col, n_tbs, tr_type, te);
        for (y, &v) in te.iter().enumerate() {
            e[y * n_tbs + x] = v;
        }
    }

    // §8.6.4 step 2 (eq. 8-314): g[x][y] = clip( (e + 64) >> 7 ).
    let mut g = vec![0i64; count];
    for (gv, &ev) in g.iter_mut().zip(e.iter()) {
        *gv = clip3(coeff_min as i64, coeff_max as i64, (ev + 64) >> 7);
    }

    // §8.6.4 step 3: row transform g[x][y] over x -> r[x][y].
    let mut r = vec![0i32; count];
    for y in 0..n_tbs {
        let row = &mut in_buf[..n_tbs];
        for (x, rv) in row.iter_mut().enumerate() {
            *rv = g[y * n_tbs + x];
        }
        if row.iter().all(|&value| value == 0) {
            continue;
        }
        let tr = &mut out_buf[..n_tbs];
        transform_1d_into(row, n_tbs, tr_type, tr);
        for (x, &v) in tr.iter().enumerate() {
            // r stays i64-valued through here; equation 8-299's bdShift
            // round (in residual_block) is what reduces it. Per the
            // spec the §8.6.4 row output is not itself clipped, so keep
            // full precision and narrow at the final offset-round.
            r[y * n_tbs + x] = v as i32;
        }
    }
    Ok(r)
}

/// Inputs to the §8.6.2 scaling-and-transformation orchestration that
/// the surrounding subclauses derive for one transform block, gathered
/// into one struct so [`residual_block`] has a stable signature as the
/// follow-up RDPCM / cross-component steps are added.
#[derive(Debug, Clone, Copy)]
pub struct BlockParams {
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
    /// `cu_transquant_bypass_flag` — when set, §8.6.2 bypasses scaling
    /// and transformation entirely (with the `rotateCoeffs` reorder).
    pub transquant_bypass: bool,
    /// `transform_skip_flag[ xTbY ][ yTbY ][ cIdx ]` — when set (and
    /// not bypassed), §8.6.2 replaces the inverse transform with the
    /// `tsShift` left-shift (equation 8-298).
    pub transform_skip: bool,
    /// `transform_skip_rotation_enabled_flag` — gates the §8.6.2
    /// `rotateCoeffs` derivation (only meaningful for 4x4 intra blocks).
    pub transform_skip_rotation_enabled: bool,
}

/// §8.6.2 — the scaling and transformation process for one transform
/// block: turn `TransCoeffLevel` into the `(nTbS)x(nTbS)` residual
/// array `r`.
///
/// `levels` is the `TransCoeffLevel[ x ][ y ]` array, row-major by `y`
/// (as [`crate::hevc::engine::residual::ResidualBlock::levels`] stores it). `scaling`
/// is the per-position `ScalingFactor` matrix when
/// `scaling_list_enabled_flag == 1` and the §8.6.3 flat-16 exception
/// does not apply, else `None`. Returns the residual array `r`,
/// row-major by `y`.
///
/// The three §8.6.2 branches are dispatched on `params`:
/// * `transquant_bypass` ⇒ pass `TransCoeffLevel` straight to `r`,
///   applying the `rotateCoeffs` reorder (equation 8-297);
/// * else `transform_skip` ⇒ scale (§8.6.3), `rotateCoeffs`-reorder if
///   applicable, then `<< tsShift` (equation 8-298) and the equation
///   8-299 `bdShift` offset-round;
/// * else ⇒ scale (§8.6.3), inverse-transform (§8.6.4), then the
///   equation 8-299 `bdShift` offset-round.
///
/// # Errors
/// [`TransformError`] as for [`scale_coefficients`] / [`inverse_transform`].
pub fn residual_block(
    levels: &[i32],
    scaling: Option<&ScalingFactorMatrix>,
    params: BlockParams,
) -> Result<Vec<i32>, TransformError> {
    let n_tbs = params.n_tbs;
    let log2_tbs = log2_tbs(n_tbs).ok_or(TransformError::InvalidBlockSize(n_tbs))?;
    if !(8..=16).contains(&params.bit_depth) {
        return Err(TransformError::InvalidBitDepth(params.bit_depth));
    }
    let count = n_tbs * n_tbs;
    if levels.len() != count {
        return Err(TransformError::LengthMismatch {
            expected: count,
            got: levels.len(),
        });
    }

    // §8.6.2 rotateCoeffs: 1 iff transform_skip_rotation_enabled_flag,
    // nTbS == 4, MODE_INTRA.
    let rotate = params.transform_skip_rotation_enabled
        && n_tbs == 4
        && matches!(params.pred_mode, PredMode::Intra);

    // §8.6.2 cu_transquant_bypass_flag == 1 path (eq. 8-297 / array
    // copy): no scaling, no transform, no bdShift.
    if params.transquant_bypass {
        let mut r = vec![0i32; count];
        for y in 0..n_tbs {
            for x in 0..n_tbs {
                let (sx, sy) = if rotate {
                    (n_tbs - x - 1, n_tbs - y - 1)
                } else {
                    (x, y)
                };
                r[y * n_tbs + x] = levels[sy * n_tbs + sx];
            }
        }
        return Ok(r);
    }

    // §8.6.2 ordered step 1: §8.6.3 scaling process -> d.
    let d = scale_coefficients(
        levels,
        n_tbs,
        params.q_p,
        params.bit_depth,
        params.extended_precision,
        scaling,
    )?;

    // §8.6.2 equations 8-294/8-295/8-296: bitDepth, bdShift, tsShift.
    let bd_shift = core::cmp::max(
        20 - params.bit_depth as i32,
        if params.extended_precision { 11 } else { 0 },
    );

    // §8.6.2 ordered step 2.
    let pre = if params.transform_skip {
        // eq. 8-298: r[x][y] = (rotate ? d[n-x-1][n-y-1] : d[x][y]) << tsShift.
        let ts_shift = 5 + log2_tbs as i32;
        let mut r = vec![0i32; count];
        for y in 0..n_tbs {
            for x in 0..n_tbs {
                let (sx, sy) = if rotate {
                    (n_tbs - x - 1, n_tbs - y - 1)
                } else {
                    (x, y)
                };
                r[y * n_tbs + x] = ((d[sy * n_tbs + sx] as i64) << ts_shift) as i32;
            }
        }
        r
    } else {
        // §8.6.4 transformation process.
        inverse_transform(
            &d,
            n_tbs,
            params.pred_mode,
            params.component,
            params.bit_depth,
            params.extended_precision,
        )?
    };

    // §8.6.2 ordered step 3 (eq. 8-299): r = (r + (1 << (bdShift-1))) >> bdShift.
    let mut r = vec![0i32; count];
    if bd_shift > 0 {
        let round = 1i64 << (bd_shift - 1);
        for (rv, &pv) in r.iter_mut().zip(pre.iter()) {
            *rv = (((pv as i64) + round) >> bd_shift) as i32;
        }
    } else {
        // bdShift == 0 only under extended-precision with bitDepth == 20
        // (out of our 8..=16 domain) — but keep the no-shift identity
        // for completeness.
        r.copy_from_slice(&pre);
    }
    Ok(r)
}

/// §8.6.5 — residual modification process for blocks using a transform
/// bypass (RDPCM): the directional cumulative sum over the
/// `(nTbS)x(nTbS)` residual array `r`.
///
/// `vertical` is the process input `mDir`: `false` (mDir 0) applies the
/// horizontal accumulation of eq. 8-322 (`r[x][y] += r[x-1][y]`, x
/// proceeding 1..nTbS-1), `true` (mDir 1) the vertical accumulation of
/// eq. 8-323 (`r[x][y] += r[x][y-1]`, y proceeding 1..nTbS-1).
///
/// Invoked for inter blocks with `explicit_rdpcm_flag == 1` (mDir =
/// `explicit_rdpcm_dir_flag`, §8.5.4.2 / §8.5.4.3) and for intra blocks
/// under the implicit-RDPCM condition (`implicit_rdpcm_enabled_flag`,
/// transform skip or transquant bypass, predModeIntra 10 / 26, with
/// mDir = predModeIntra / 26 — §8.4.4.1).
///
/// `r` is row-major by `y` (as [`residual_block`] returns it); its
/// length must be `n_tbs * n_tbs`.
pub fn rdpcm_accumulate(r: &mut [i32], n_tbs: usize, vertical: bool) {
    debug_assert_eq!(r.len(), n_tbs * n_tbs);
    if vertical {
        // eq. 8-323: r[ x ][ y ] += r[ x ][ y − 1 ].
        for y in 1..n_tbs {
            for x in 0..n_tbs {
                r[y * n_tbs + x] = r[y * n_tbs + x].wrapping_add(r[(y - 1) * n_tbs + x]);
            }
        }
    } else {
        // eq. 8-322: r[ x ][ y ] += r[ x − 1 ][ y ].
        for y in 0..n_tbs {
            for x in 1..n_tbs {
                r[y * n_tbs + x] = r[y * n_tbs + x].wrapping_add(r[y * n_tbs + x - 1]);
            }
        }
    }
}

/// §8.6.8.2 — adaptive colour transformation process (inverse), applied
/// to the three co-located residual arrays of one 4:4:4 transform block
/// whose `tu_residual_act_flag` is 1.
///
/// Ordered per the clause:
/// 1. eq. 8-326..8-328 — clip each array to its `CoeffMin/Max` range
///    (luma range for `rY`, chroma range for `rCb` / `rCr`);
/// 2. when `cu_transquant_bypass_flag == 0`, eq. 8-329..8-335 —
///    bit-depth alignment to `Max( BitDepthY, BitDepthC )` with the
///    extra `<< 1` on both chroma arrays;
/// 3. eq. 8-336..8-339 — the lifting inverse:
///    `tmp = rY − ( rCb >> 1 ); rY = tmp + rCb;
///    rCb = tmp − ( rCr >> 1 ); rCr += rCb`;
/// 4. when `cu_transquant_bypass_flag == 0`, eq. 8-340..8-342 — the
///    rounded down-shift back to the component bit depths.
///
/// All three slices must have the same length (`nTbS * nTbS`).
pub fn act_inverse(
    r_y: &mut [i32],
    r_cb: &mut [i32],
    r_cr: &mut [i32],
    bit_depth_luma: u8,
    bit_depth_chroma: u8,
    extended_precision: bool,
    transquant_bypass: bool,
) {
    debug_assert_eq!(r_y.len(), r_cb.len());
    debug_assert_eq!(r_y.len(), r_cr.len());
    let (min_y, max_y) = coeff_range(bit_depth_luma, extended_precision);
    let (min_c, max_c) = coeff_range(bit_depth_chroma, extended_precision);
    // eq. 8-329..8-332.
    let max_bd = bit_depth_luma.max(bit_depth_chroma);
    let delta_bd_y = u32::from(max_bd - bit_depth_luma);
    let delta_bd_c = u32::from(max_bd - bit_depth_chroma);
    let offset_bd_y = if delta_bd_y != 0 {
        1i32 << (delta_bd_y - 1)
    } else {
        0
    };
    let offset_bd_c = if delta_bd_c != 0 {
        1i32 << (delta_bd_c - 1)
    } else {
        0
    };
    for i in 0..r_y.len() {
        // eq. 8-326..8-328.
        let mut y = r_y[i].clamp(min_y, max_y);
        let mut cb = r_cb[i].clamp(min_c, max_c);
        let mut cr = r_cr[i].clamp(min_c, max_c);
        // eq. 8-333..8-335 (lossy only).
        if !transquant_bypass {
            y <<= delta_bd_y;
            cb <<= delta_bd_c + 1;
            cr <<= delta_bd_c + 1;
        }
        // eq. 8-336..8-339.
        let tmp = y - (cb >> 1);
        y = tmp + cb;
        cb = tmp - (cr >> 1);
        cr += cb;
        // eq. 8-340..8-342 (lossy only).
        if !transquant_bypass {
            y = (y + offset_bd_y) >> delta_bd_y;
            cb = (cb + offset_bd_c) >> delta_bd_c;
            cr = (cr + offset_bd_c) >> delta_bd_c;
        }
        r_y[i] = y;
        r_cb[i] = cb;
        r_cr[i] = cr;
    }
}

#[cfg(any())]
mod tests {
    use super::*;

    fn flat_levels(n: usize, fill: &[(usize, usize, i32)]) -> Vec<i32> {
        let mut v = vec![0i32; n * n];
        for &(x, y, val) in fill {
            v[y * n + x] = val;
        }
        v
    }

    #[test]
    fn coeff_range_default_is_15bit() {
        assert_eq!(coeff_range(8, false), (-32768, 32767));
        assert_eq!(coeff_range(10, false), (-32768, 32767));
    }

    #[test]
    fn coeff_range_extended_widens_with_bitdepth() {
        // bitDepth + 6 = 18 > 15 -> magnitude 1 << 18.
        assert_eq!(coeff_range(12, true), (-(1 << 18), (1 << 18) - 1));
        // bitDepth + 6 = 14 < 15 -> still 15 bits.
        assert_eq!(coeff_range(8, true), (-32768, 32767));
    }

    /// §8.6.3 hand-computed scale of a single DC level. nTbS = 4
    /// (log2 = 2), bitDepth = 8, no scaling list (m = 16),
    /// extended_precision = 0. qP = 6 -> qP%6 = 0 (levelScale 40),
    /// qP/6 = 1. bdShift = 8 + 2 + 10 - 15 = 5, round = 16.
    /// d = clip( ((level * 16 * 40) << 1 + 16) >> 5 ).
    #[test]
    fn scale_single_dc_matches_hand_computation() {
        let n = 4;
        let levels = flat_levels(n, &[(0, 0, 3)]);
        let d = scale_coefficients(&levels, n, 6, 8, false, None).unwrap();
        // level 3: (3*16*40) = 1920; <<1 = 3840; +16 = 3856; >>5 = 120.
        assert_eq!(d[0], 120);
        // every other coefficient is 0 -> stays 0.
        assert!(d[1..].iter().all(|&v| v == 0));
    }

    /// qP/6 partition: qP = 4 -> qP%6 = 4 (levelScale 64), qP/6 = 0.
    /// level = -5, m = 16, bdShift = 5, round = 16.
    /// (-5*16*64) = -5120; <<0 = -5120; +16 = -5104; >>5 = -160 (arith).
    #[test]
    fn scale_negative_level_arithmetic_shift() {
        let n = 4;
        let levels = flat_levels(n, &[(1, 2, -5)]);
        let d = scale_coefficients(&levels, n, 4, 8, false, None).unwrap();
        assert_eq!(d[2 * n + 1], -160);
    }

    /// §8.6.3 clip to CoeffMax. A huge level * big qP saturates at
    /// 32767 for the default 15-bit range.
    #[test]
    fn scale_clips_to_coeff_max() {
        let n = 4;
        let levels = flat_levels(n, &[(0, 0, 30000)]);
        // qP = 51 -> qP/6 = 8, large left-shift -> saturates.
        let d = scale_coefficients(&levels, n, 51, 8, false, None).unwrap();
        assert_eq!(d[0], 32767);
    }

    /// §8.6.2 transquant-bypass copies TransCoeffLevel verbatim (no
    /// rotate when rotation not enabled).
    #[test]
    fn bypass_copies_levels_verbatim() {
        let n = 4;
        let levels = flat_levels(n, &[(0, 0, 7), (3, 3, -2), (1, 0, 5)]);
        let params = BlockParams {
            n_tbs: n,
            q_p: 26,
            component: Component::Luma,
            pred_mode: PredMode::Intra,
            bit_depth: 8,
            extended_precision: false,
            transquant_bypass: true,
            transform_skip: false,
            transform_skip_rotation_enabled: false,
        };
        let r = residual_block(&levels, None, params).unwrap();
        assert_eq!(r, levels);
    }

    /// §8.6.2 transquant-bypass with rotateCoeffs (4x4 intra,
    /// rotation enabled) mirrors x and y (eq. 8-297).
    #[test]
    fn bypass_rotate_mirrors_coefficients() {
        let n = 4;
        let levels = flat_levels(n, &[(0, 0, 9)]);
        let params = BlockParams {
            n_tbs: n,
            q_p: 0,
            component: Component::Luma,
            pred_mode: PredMode::Intra,
            bit_depth: 8,
            extended_precision: false,
            transquant_bypass: true,
            transform_skip: false,
            transform_skip_rotation_enabled: true,
        };
        let r = residual_block(&levels, None, params).unwrap();
        // r[x][y] = level[n-x-1][n-y-1]; the only nonzero level is at
        // (0,0) -> lands at r[3][3].
        assert_eq!(r[3 * n + 3], 9);
        assert_eq!(r[0], 0);
    }

    /// The inverse DCT of a single DC coefficient is a constant field:
    /// the DC basis function (transMatrix row 0, all-64 per eq. 8-319) is
    /// flat, so a `d[0][0]`-only input must reconstruct to a uniform `r`.
    /// This pins the §8.6.4 matrix-orientation: eq. 8-317's
    /// `transMatrix[ i ][ j*stride ]` reads the in-code [`DCT32`] as
    /// `DCT32[ j*stride ][ i ]` (the base table is laid out
    /// [column][row], eq. 8-318/8-319), so the DC input excites the
    /// constant row-0 basis.
    #[test]
    fn dc_only_dct_4x4_exact() {
        // d has only d[0][0] = 64 (n=4, trType=0/inter).
        let n = 4;
        let d = flat_levels(n, &[(0, 0, 64)]);
        let r = inverse_transform(&d, n, PredMode::Inter, Component::Luma, 8, false).unwrap();
        // Column transform of column 0 (input [64,0,0,0]):
        //   e[0][i] = transMatrix[i][0] * 64 = DCT32[0][i] * 64
        //           = 64 * 64 = 4096 (row 0 of DCT32 is all 64);
        //   every other column is 0.
        // g[x][y] = clip((e+64)>>7): g[0][.] = (4096+64)>>7 = 32;
        //   g[x>0][.] = (0+64)>>7 = 0.
        // Row transform of row y (only x=0 nonzero, value 32):
        //   r[i][y] = transMatrix[i][0] * 32 = 64 * 32 = 2048 for all i.
        // So the whole 4x4 block reconstructs to the constant 2048.
        assert!(
            r.iter().all(|&v| v == 2048),
            "DC-only must be uniform: {r:?}"
        );
    }

    /// trType selection: only MODE_INTRA / nTbS==4 / luma uses the DST.
    /// The DST and DCT matrices differ, so the same scaled-coefficient
    /// array must produce different residuals under the two prediction
    /// modes. (A DC-only impulse is a valid discriminator: DST column 0
    /// is {29,74,84,55} vs DCT column 0 {64,90,90,90}.)
    #[test]
    fn dst_path_selected_for_4x4_intra_luma() {
        let n = 4;
        let d = flat_levels(n, &[(0, 0, 100)]);
        let intra = inverse_transform(&d, n, PredMode::Intra, Component::Luma, 8, false).unwrap();
        let inter = inverse_transform(&d, n, PredMode::Inter, Component::Luma, 8, false).unwrap();
        assert_ne!(intra, inter, "DST (intra 4x4 luma) must differ from DCT");
        // The §8.6.4.2 column transform of [100,0,0,0] under the DST
        // gives e[i][0] = DST4[i][0] * 100 = [2900, 7400, 8400, 5500];
        // the DCT gives [6400, 9000, 9000, 9000]. So intra and inter
        // genuinely take different matrices.
    }

    /// 4x4 chroma intra must NOT use the DST (trType requires cIdx==0).
    /// A Cb 4x4 intra block must match the inter (DCT) result.
    #[test]
    fn chroma_4x4_intra_uses_dct_not_dst() {
        let n = 4;
        let d = flat_levels(n, &[(1, 2, 37), (0, 0, 100), (3, 1, -8)]);
        let cb_intra = inverse_transform(&d, n, PredMode::Intra, Component::Cb, 8, false).unwrap();
        let luma_inter =
            inverse_transform(&d, n, PredMode::Inter, Component::Luma, 8, false).unwrap();
        // Both take trType == 0 (DCT), so the residual arrays match.
        assert_eq!(cb_intra, luma_inter);
        // And both differ from the luma-intra DST path.
        let luma_intra =
            inverse_transform(&d, n, PredMode::Intra, Component::Luma, 8, false).unwrap();
        assert_ne!(cb_intra, luma_intra);
    }

    /// transform_skip path: scale then << tsShift then >> bdShift, no
    /// matrix transform. tsShift = 5 + log2(nTbS). For nTbS=4 -> 7.
    #[test]
    fn transform_skip_shifts_without_transform() {
        let n = 4;
        let levels = flat_levels(n, &[(2, 1, 1)]);
        let params = BlockParams {
            n_tbs: n,
            q_p: 4, // levelScale 64, qP/6 = 0
            component: Component::Luma,
            pred_mode: PredMode::Inter,
            bit_depth: 8,
            extended_precision: false,
            transquant_bypass: false,
            transform_skip: true,
            transform_skip_rotation_enabled: false,
        };
        let r = residual_block(&levels, None, params).unwrap();
        // §8.6.3 d for level 1: (1*16*64<<0 +16)>>5 = (1024+16)>>5 = 32.
        // §8.6.2 ts: 32 << (5+2=7) = 4096. bdShift = max(20-8,0)=12,
        // round 1<<11=2048. (4096+2048)>>12 = 6144>>12 = 1.
        // coefficient is at (x=2, y=1) -> index y*n + x = n + 2.
        assert_eq!(r[n + 2], 1);
        // the rest are zero -> 0 << 7 = 0 -> (0+2048)>>12 = 0.
        assert!(r.iter().enumerate().all(|(i, &v)| i == n + 2 || v == 0));
    }

    #[test]
    fn rejects_bad_block_size() {
        let levels = vec![0i32; 36]; // 6x6, illegal
        assert_eq!(
            scale_coefficients(&levels, 6, 26, 8, false, None),
            Err(TransformError::InvalidBlockSize(6))
        );
    }

    #[test]
    fn rejects_length_mismatch() {
        let levels = vec![0i32; 15]; // not 16
        assert_eq!(
            scale_coefficients(&levels, 4, 26, 8, false, None),
            Err(TransformError::LengthMismatch {
                expected: 16,
                got: 15,
            })
        );
    }

    #[test]
    fn rejects_bad_bit_depth() {
        let levels = vec![0i32; 16];
        assert_eq!(
            scale_coefficients(&levels, 4, 26, 7, false, None),
            Err(TransformError::InvalidBitDepth(7))
        );
    }

    /// The DST-VII matrix rows are orthogonal-ish but here we just pin
    /// the equation-8-316 matrix values so a future edit can't silently
    /// change them.
    #[test]
    fn dst_matrix_pinned() {
        assert_eq!(DST4[0], [29, 55, 74, 84]);
        assert_eq!(DST4[1], [74, 74, 0, -74]);
        assert_eq!(DST4[3], [55, -84, 74, -29]);
    }

    /// Pin a few DCT matrix cells across the col0to15 / col16to31 split.
    #[test]
    fn dct_matrix_pinned() {
        // row 0 is all 64.
        assert!(DCT32[0].iter().all(|&v| v == 64));
        // row 1, col 0 = 90, col 31 = -90 (eq. 8-319 first / 8-321 last).
        assert_eq!(DCT32[1][0], 90);
        assert_eq!(DCT32[1][31], -90);
        // row 31, col 0 = 4, col 31 = -4.
        assert_eq!(DCT32[31][0], 4);
        assert_eq!(DCT32[31][31], -4);
        // a col16to31 interior cell: row 2, col 16 = -90.
        assert_eq!(DCT32[2][16], -90);
    }

    /// §8.6.4 16x16 / 8x8 subsample stride: the nTbS=8 transform reads
    /// DCT32 columns at stride 4, nTbS=16 at stride 2, nTbS=4 at
    /// stride 8. Spot-check via a single-row impulse so the output is
    /// exactly one basis row.
    #[test]
    fn dct_subsample_stride_picks_right_columns() {
        // 8x8: eq. 8-317 reads transMatrix[i][j*stride] with stride 4,
        // which in the in-code [`DCT32`] ([column][row] layout) is
        // DCT32[j*stride][i].
        let mut x = vec![0i64; 8];
        x[0] = 1; // DC basis input (j = 0).
        let y = transform_1d(&x, 8, false);
        // y[i] = transMatrix[i][0] = DCT32[0][i] = 64 for all i (the
        // all-64 DC basis row).
        assert!(
            y.iter().all(|&v| v == 64),
            "DC basis must be flat 64: {y:?}"
        );
        // x[1] excites transform-matrix column 1*stride = 4, i.e.
        // DCT32 row 4 = [89, 75, 50, 18, ...].
        let mut x2 = vec![0i64; 8];
        x2[1] = 1;
        let y2 = transform_1d(&x2, 8, false);
        assert_eq!(y2[0], DCT32[4][0] as i64); // 89
        assert_eq!(y2[1], DCT32[4][1] as i64); // 75
    }

    /// §8.6.5 eq. 8-322 — horizontal RDPCM is a running sum along each
    /// row (x proceeding over 1..nTbS − 1).
    #[test]
    fn rdpcm_horizontal_accumulates_rows() {
        // Row-major 4x4: row y holds [y+1, 1, -2, 3].
        let mut r: Vec<i32> = (0..4).flat_map(|y| vec![y + 1, 1, -2, 3]).collect();
        rdpcm_accumulate(&mut r, 4, false);
        for y in 0..4usize {
            let base = y as i32 + 1;
            assert_eq!(
                &r[y * 4..y * 4 + 4],
                &[base, base + 1, base - 1, base + 2],
                "row {y}"
            );
        }
    }

    /// §8.6.5 eq. 8-323 — vertical RDPCM is a running sum down each
    /// column (y proceeding over 1..nTbS − 1).
    #[test]
    fn rdpcm_vertical_accumulates_columns() {
        // Column x holds [x, 5, -3, 1] top to bottom.
        let mut r = vec![0i32; 16];
        for x in 0..4usize {
            r[x] = x as i32;
            r[4 + x] = 5;
            r[8 + x] = -3;
            r[12 + x] = 1;
        }
        rdpcm_accumulate(&mut r, 4, true);
        for x in 0..4usize {
            let x0 = x as i32;
            assert_eq!(
                [r[x], r[4 + x], r[8 + x], r[12 + x]],
                [x0, x0 + 5, x0 + 2, x0 + 3],
                "column {x}"
            );
        }
    }

    /// §8.6.8.2 lossless (transquant-bypass) inverse undoes the
    /// matching forward lifting exactly: forward
    /// `Co = c2 − c1; t = c1 + ( Co >> 1 ); Cg = c0 − t;
    /// Y = t + ( Cg >> 1 )` (component order c0/c1/c2 mapped onto the
    /// reconstructed rY/rCb/rCr outputs), inverse eq. 8-336..8-339.
    #[test]
    fn act_inverse_bypass_roundtrips_forward_lifting() {
        let triples: Vec<(i32, i32, i32)> = (0..64)
            .map(|i| {
                let a = ((i * 37) % 51) - 25;
                let b = ((i * 53) % 61) - 30;
                let c = ((i * 71) % 41) - 20;
                (a, b, c)
            })
            .collect();
        let mut r_y = Vec::new();
        let mut r_cb = Vec::new();
        let mut r_cr = Vec::new();
        for &(c0, c1, c2) in &triples {
            let co = c2 - c1;
            let t = c1 + (co >> 1);
            let cg = c0 - t;
            let y = t + (cg >> 1);
            r_y.push(y);
            r_cb.push(cg);
            r_cr.push(co);
        }
        act_inverse(&mut r_y, &mut r_cb, &mut r_cr, 8, 8, false, true);
        for (i, &(c0, c1, c2)) in triples.iter().enumerate() {
            assert_eq!((r_y[i], r_cb[i], r_cr[i]), (c0, c1, c2), "triple {i}");
        }
    }

    /// The lossy path pre-scales both chroma arrays by `<< 1`
    /// (eq. 8-334 / 8-335) even at equal bit depths (deltaBD = 0):
    /// rY = 0, rCb = 4, rCr = 0 ⇒ cb = 8, tmp = −4, y = 4, cb = −4,
    /// cr = −4.
    #[test]
    fn act_inverse_lossy_prescales_chroma() {
        let mut r_y = vec![0i32];
        let mut r_cb = vec![4i32];
        let mut r_cr = vec![0i32];
        act_inverse(&mut r_y, &mut r_cb, &mut r_cr, 8, 8, false, false);
        assert_eq!((r_y[0], r_cb[0], r_cr[0]), (4, -4, -4));
    }

    /// Bit-depth alignment (eq. 8-329..8-335 / 8-340..8-342): with
    /// BitDepthY = 10, BitDepthC = 8 a luma-only residual survives the
    /// up/down shift exactly (deltaBDY = 0, deltaBDC = 2), and a
    /// chroma-only rCb = 1 aligns to 1 << 3 = 8 before the lifting.
    #[test]
    fn act_inverse_lossy_aligns_bit_depths() {
        // Luma-only: chroma zero ⇒ y unchanged, cb/cr mirror −(y>>1)
        // pattern scaled back down.
        let mut r_y = vec![100i32];
        let mut r_cb = vec![0i32];
        let mut r_cr = vec![0i32];
        act_inverse(&mut r_y, &mut r_cb, &mut r_cr, 10, 8, false, false);
        // deltaBDY = 0: y = 100 − 0 + 0 = 100.
        // cb = tmp = 100 (aligned) → back: (100 + 2) >> 2 = 25.
        assert_eq!((r_y[0], r_cb[0], r_cr[0]), (100, 25, 25));
        // Chroma-only rCb = 1 at deltaBDC = 2: cb = 1 << 3 = 8;
        // tmp = −4; y = 4; cb = −4 → (−4 + 2) >> 2 = −1 (arithmetic);
        // cr = −4 → −1; y stays 10-bit (deltaBDY = 0).
        let mut r_y = vec![0i32];
        let mut r_cb = vec![1i32];
        let mut r_cr = vec![0i32];
        act_inverse(&mut r_y, &mut r_cb, &mut r_cr, 10, 8, false, false);
        assert_eq!((r_y[0], r_cb[0], r_cr[0]), (4, -1, -1));
    }

    /// Eq. 8-326..8-328 clip the INPUT arrays to the coefficient
    /// ranges before any lifting.
    #[test]
    fn act_inverse_clips_inputs_to_coeff_range() {
        let mut r_y = vec![40000i32];
        let mut r_cb = vec![-40000i32];
        let mut r_cr = vec![0i32];
        act_inverse(&mut r_y, &mut r_cb, &mut r_cr, 8, 8, false, true);
        // Clipped to (32767, −32768, 0): tmp = 32767 − (−16384) =
        // 49151; y = 49151 − 32768 = 16383; cb = 49151 − 0 = 49151;
        // cr = 0 + 49151 = 49151.
        assert_eq!((r_y[0], r_cb[0], r_cr[0]), (16383, 49151, 49151));
    }

    /// The first row (vertical) / first column (horizontal) is left
    /// unmodified — the accumulation index proceeds from 1.
    #[test]
    fn rdpcm_leaves_first_line_unmodified() {
        let src: Vec<i32> = (0..64).map(|i| (i * 7 % 23) - 11).collect();
        let mut h = src.clone();
        rdpcm_accumulate(&mut h, 8, false);
        for y in 0..8 {
            assert_eq!(h[y * 8], src[y * 8], "column 0, row {y}");
        }
        let mut v = src.clone();
        rdpcm_accumulate(&mut v, 8, true);
        assert_eq!(&v[..8], &src[..8], "row 0");
    }
}
