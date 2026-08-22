//! §7.3.4 `scaling_list_data()` parse + §7.4.5 `ScalingList` derivation.
//!
//! H.265 carries an optional set of quantization scaling lists in the
//! SPS (`sps_scaling_list_data_present_flag == 1`) and PPS
//! (`pps_scaling_list_data_present_flag == 1`). The
//! `scaling_list_data()` syntax structure (ITU-T Rec. H.265 §7.3.4)
//! signals, for each of the 24 (sizeId, matrixId) slots, either:
//!
//! * a **prediction** from an earlier list of the same `sizeId`
//!   (`scaling_list_pred_mode_flag == 0`), where
//!   `scaling_list_pred_matrix_id_delta` selects the reference list
//!   (delta 0 means "use the default list", per §7.4.5); or
//! * an **explicit** delta-coded list
//!   (`scaling_list_pred_mode_flag == 1`), built from a running
//!   `nextCoef` accumulator modulo 256, optionally preceded by a DC
//!   coefficient (`scaling_list_dc_coef_minus8`) for sizeId > 1.
//!
//! This module parses that structure and derives the flat
//! `ScalingList[sizeId][matrixId][i]` coefficient arrays plus the
//! per-slot DC coefficients, applying the §7.4.5 default-list and
//! prediction-inference rules (equations 7-42 and 7-43).
//!
//! [`ScalingListData::scaling_factors`] then expands the flat lists
//! into the two-dimensional `ScalingFactor[sizeId][matrixId][x][y]`
//! quantization matrices (§7.4.5 equations 7-44..7-51), placing each
//! flat coefficient at the `(x, y)` cell given by the §6.5.3 up-right
//! diagonal scan order ([`crate::hevc::engine::scan`]) and applying the
//! `2x`/`4x` block-upsampling for the 16x16 / 32x32 sizes plus the
//! DC-coefficient `[0][0]` override.

use crate::hevc::engine::bitreader::{BitReader, BitReaderError};
use crate::hevc::engine::scan::up_right_diagonal;

/// Number of `sizeId` values (4x4, 8x8, 16x16, 32x32) — §7.4.5
/// Table 7-3.
pub const NUM_SIZE_IDS: usize = 4;

/// Number of `matrixId` values per `sizeId` — §7.4.5 Table 7-4
/// (intra Y/Cb/Cr + inter Y/Cb/Cr).
pub const NUM_MATRIX_IDS: usize = 6;

/// Maximum number of coefficients in a flat scaling list. `sizeId` 0
/// carries 16; `sizeId` 1..3 carry 64
/// (`Min(64, 1 << (4 + (sizeId << 1)))`).
pub const MAX_COEF_NUM: usize = 64;

/// Number of explicit coefficients for a given `sizeId`:
/// `Min(64, 1 << (4 + (sizeId << 1)))` (§7.3.4) — 16 for `sizeId` 0,
/// 64 otherwise.
fn coef_num(size_id: usize) -> usize {
    (1usize << (4 + (size_id << 1))).min(MAX_COEF_NUM)
}

/// Errors that can arise while parsing `scaling_list_data()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingListError {
    /// The RBSP ran out of bits before the structure was fully parsed.
    Truncated,
    /// `scaling_list_pred_matrix_id_delta` exceeded the §7.4.5 bound
    /// (`matrixId` for `sizeId <= 2`, `matrixId / 3` for `sizeId == 3`),
    /// which would index a non-existent or future reference list.
    PredMatrixIdDeltaOutOfRange {
        /// The `sizeId` of the offending slot.
        size_id: u8,
        /// The `matrixId` of the offending slot.
        matrix_id: u8,
        /// The (illegal) `scaling_list_pred_matrix_id_delta` value.
        got: u32,
    },
    /// `scaling_list_dc_coef_minus8` was outside the §7.4.5 range of
    /// −7 to 247, inclusive.
    DcCoefOutOfRange {
        /// The `sizeId` of the offending slot (always > 1).
        size_id: u8,
        /// The `matrixId` of the offending slot.
        matrix_id: u8,
        /// The (illegal) `scaling_list_dc_coef_minus8` value.
        got: i32,
    },
    /// `scaling_list_delta_coef` was outside the §7.4.5 range of −128
    /// to 127, inclusive.
    DeltaCoefOutOfRange {
        /// The `sizeId` of the offending slot.
        size_id: u8,
        /// The `matrixId` of the offending slot.
        matrix_id: u8,
        /// The coefficient index `i`.
        index: u8,
        /// The (illegal) `scaling_list_delta_coef` value.
        got: i32,
    },
    /// A derived coefficient `ScalingList[sizeId][matrixId][i]` was not
    /// greater than 0, violating the §7.4.5 conformance requirement.
    NonPositiveCoef {
        /// The `sizeId` of the offending slot.
        size_id: u8,
        /// The `matrixId` of the offending slot.
        matrix_id: u8,
        /// The coefficient index `i`.
        index: u8,
    },
    /// An unexpected bitstream-level error surfaced from the reader.
    Bitstream(BitReaderError),
}

impl core::fmt::Display for ScalingListError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => f.write_str("scaling_list_data() truncated"),
            Self::PredMatrixIdDeltaOutOfRange {
                size_id,
                matrix_id,
                got,
            } => write!(
                f,
                "scaling_list_pred_matrix_id_delta[{size_id}][{matrix_id}] out of range: {got}"
            ),
            Self::DcCoefOutOfRange {
                size_id,
                matrix_id,
                got,
            } => write!(
                f,
                "scaling_list_dc_coef_minus8[{}][{matrix_id}] out of range: {got}",
                size_id - 2
            ),
            Self::DeltaCoefOutOfRange {
                size_id,
                matrix_id,
                index,
                got,
            } => write!(
                f,
                "scaling_list_delta_coef[{size_id}][{matrix_id}][{index}] out of range: {got}"
            ),
            Self::NonPositiveCoef {
                size_id,
                matrix_id,
                index,
            } => write!(
                f,
                "ScalingList[{size_id}][{matrix_id}][{index}] is not greater than 0"
            ),
            Self::Bitstream(e) => write!(f, "bitstream error during scaling_list_data(): {e}"),
        }
    }
}

impl std::error::Error for ScalingListError {}

impl From<BitReaderError> for ScalingListError {
    fn from(e: BitReaderError) -> Self {
        match e {
            BitReaderError::EndOfBuffer => Self::Truncated,
            other => Self::Bitstream(other),
        }
    }
}

/// Default 4x4 list — §7.4.5 Table 7-5: every coefficient is 16 for
/// all six `matrixId`s.
const DEFAULT_4X4: [u16; 16] = [16; 16];

/// Default 8x8 intra list (`matrixId` 0..2) — §7.4.5 Table 7-6, rows
/// for `ScalingList[1..3][0..2]`. Also serves as the default for the
/// 16x16 and 32x32 intra lists (sizeId 1, 2, 3).
const DEFAULT_8X8_INTRA: [u16; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, // i = 0..15
    17, 18, 18, 17, 18, 21, 19, 20, 21, 20, 19, 21, 24, 22, 22, 24, // i = 16..31
    24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31, // i = 32..47
    29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115, // i = 48..63
];

/// Default 8x8 inter list (`matrixId` 3..5) — §7.4.5 Table 7-6, rows
/// for `ScalingList[1..3][3..5]`. Also serves as the default for the
/// 16x16 and 32x32 inter lists.
const DEFAULT_8X8_INTER: [u16; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, // i = 0..15
    18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24, // i = 16..31
    24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28, // i = 32..47
    28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91, // i = 48..63
];

/// The §7.4.5 default `ScalingList[sizeId][matrixId][i]` values, for
/// `i = 0..coefNum(sizeId) - 1`. `sizeId == 0` is the flat 4x4 list;
/// every other `sizeId` reuses the 8x8 intra/inter defaults
/// (`matrixId < 3` → intra, `matrixId >= 3` → inter).
pub fn default_scaling_list(size_id: usize, matrix_id: usize) -> &'static [u16] {
    if size_id == 0 {
        &DEFAULT_4X4
    } else if matrix_id < 3 {
        &DEFAULT_8X8_INTRA
    } else {
        &DEFAULT_8X8_INTER
    }
}

/// One parsed-and-derived scaling list: the flat coefficient array
/// `ScalingList[sizeId][matrixId][i]` plus its DC coefficient (the
/// §7.4.5 `scaling_list_dc_coef_minus8 + 8`, only meaningful for
/// `sizeId > 1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingListMatrix {
    /// The flat coefficients, length `coefNum(sizeId)` (16 for
    /// `sizeId` 0, else 64).
    pub coef: Vec<u16>,
    /// `scaling_list_dc_coef_minus8 + 8` — the DC coefficient for
    /// `sizeId > 1`, supplying `ScalingFactor[sizeId][matrixId][0][0]`.
    /// Inferred to 8 (and unused) for `sizeId <= 1`.
    pub dc_coef: u16,
}

/// The full parsed-and-derived `scaling_list_data()` structure: all 24
/// `(sizeId, matrixId)` slots. Slots not signalled for `sizeId == 3`
/// (matrixId 1, 2, 4, 5 are skipped by the `matrixId += 3` step)
/// retain their default value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingListData {
    /// `[sizeId][matrixId]` — the derived per-slot scaling lists.
    pub lists: [[ScalingListMatrix; NUM_MATRIX_IDS]; NUM_SIZE_IDS],
}

impl ScalingListData {
    /// The all-default `scaling_list_data()`: every slot carries its
    /// §7.4.5 Table 7-5 / Table 7-6 default list with a DC scaling
    /// factor of 16 (`scaling_list_dc_coef_minus8` inferred to 8).
    ///
    /// This is the active scaling-list data when
    /// `scaling_list_enabled_flag == 1` but neither the SPS nor the PPS
    /// carries an explicit `scaling_list_data()` body.
    #[must_use]
    pub fn all_default() -> Self {
        let lists: [[ScalingListMatrix; NUM_MATRIX_IDS]; NUM_SIZE_IDS] =
            core::array::from_fn(|size_id| {
                core::array::from_fn(|matrix_id| ScalingListMatrix {
                    coef: default_scaling_list(size_id, matrix_id).to_vec(),
                    // §7.4.5: the inferred scaling_list_dc_coef_minus8
                    // is 8 ⇒ ScalingFactor[sizeId][matrixId][0][0] = 16
                    // (matching the default lists' first coefficient).
                    dc_coef: 16,
                })
            });
        Self { lists }
    }

    /// Parse `scaling_list_data()` (§7.3.4) and derive the per-slot
    /// `ScalingList` coefficient arrays (§7.4.5), applying the
    /// default-list and prediction-inference rules.
    pub fn parse(br: &mut BitReader<'_>) -> Result<Self, ScalingListError> {
        // Seed every slot with its default; signalled slots overwrite.
        // sizeId == 3 only visits matrixId 0 and 3 (the `+= 3` step),
        // so the other four slots stay at their (unused) default.
        let mut lists = Self::all_default().lists;

        for (size_id, size_lists) in lists.iter_mut().enumerate() {
            // matrixId += (sizeId == 3) ? 3 : 1  (§7.3.4)
            let step = if size_id == 3 { 3 } else { 1 };
            let mut matrix_id = 0usize;
            while matrix_id < NUM_MATRIX_IDS {
                let pred_mode_flag = br.u1()? != 0;
                if !pred_mode_flag {
                    // Prediction from a reference list of the same sizeId.
                    let delta = br.ue()?;
                    let max_delta = if size_id <= 2 {
                        matrix_id as u32
                    } else {
                        matrix_id as u32 / 3
                    };
                    if delta > max_delta {
                        return Err(ScalingListError::PredMatrixIdDeltaOutOfRange {
                            size_id: size_id as u8,
                            matrix_id: matrix_id as u8,
                            got: delta,
                        });
                    }
                    if delta == 0 {
                        // §7.4.5: inferred from the default scaling list.
                        let def = default_scaling_list(size_id, matrix_id);
                        size_lists[matrix_id] = ScalingListMatrix {
                            coef: def.to_vec(),
                            // §7.4.5: scaling_list_dc_coef_minus8 is
                            // inferred to be 8 for the default list,
                            // i.e. a DC scaling factor of 16.
                            dc_coef: 16,
                        };
                    } else {
                        // refMatrixId = matrixId − delta * step  (7-42),
                        // and copy the reference list (7-43) including
                        // its DC coefficient (§7.4.5 last paragraph).
                        let ref_matrix_id = matrix_id - (delta as usize) * step;
                        size_lists[matrix_id] = size_lists[ref_matrix_id].clone();
                    }
                } else {
                    // Explicitly signalled list.
                    let n = coef_num(size_id);
                    let mut next_coef: i32 = 8;
                    let mut dc_coef: u16 = 8;
                    if size_id > 1 {
                        let dc = br.se()?;
                        if !(-7..=247).contains(&dc) {
                            return Err(ScalingListError::DcCoefOutOfRange {
                                size_id: size_id as u8,
                                matrix_id: matrix_id as u8,
                                got: dc,
                            });
                        }
                        next_coef = dc + 8;
                        dc_coef = (dc + 8) as u16;
                    }
                    let mut coef = Vec::with_capacity(n);
                    for i in 0..n {
                        let delta_coef = br.se()?;
                        // §7.4.5: "The value of scaling_list_delta_coef
                        // shall be in the range of −128 to 127,
                        // inclusive." (Also keeps the eq. nextCoef sum
                        // below inside i32.)
                        if !(-128..=127).contains(&delta_coef) {
                            return Err(ScalingListError::DeltaCoefOutOfRange {
                                size_id: size_id as u8,
                                matrix_id: matrix_id as u8,
                                index: i as u8,
                                got: delta_coef,
                            });
                        }
                        // nextCoef = (nextCoef + delta_coef + 256) % 256
                        next_coef = (next_coef + delta_coef + 256).rem_euclid(256);
                        if next_coef <= 0 {
                            return Err(ScalingListError::NonPositiveCoef {
                                size_id: size_id as u8,
                                matrix_id: matrix_id as u8,
                                index: i as u8,
                            });
                        }
                        coef.push(next_coef as u16);
                    }
                    size_lists[matrix_id] = ScalingListMatrix { coef, dc_coef };
                }
                matrix_id += step;
            }
        }

        Ok(Self { lists })
    }

    /// Derive the §7.4.5 `ScalingFactor[sizeId][matrixId][x][y]`
    /// quantization matrices (equations 7-44..7-51) from the parsed
    /// flat lists.
    ///
    /// `chroma_array_type` is the §7.4.3.2.1 `ChromaArrayType`
    /// (`separate_colour_plane_flag ? 0 : chroma_format_idc`). It only
    /// affects the 32x32 chroma matrices: equations 7-50 / 7-51 derive
    /// `ScalingFactor[3][matrixId]` for `matrixId ∈ {1, 2, 4, 5}` from
    /// the corresponding 16x16 lists (`ScalingList[2][matrixId]`) **only
    /// when `ChromaArrayType == 3`** (4:4:4). For other chroma formats
    /// those 32x32 chroma matrices are not used, and they are left as
    /// all-zero so a consumer that reads one signals its own bug.
    pub fn scaling_factors(&self, chroma_array_type: u8) -> ScalingFactors {
        // ScanOrder[2][0] — up-right diagonal scan of a 4x4 block
        // (16 positions); ScanOrder[3][0] — of an 8x8 block (64).
        let scan_4 = up_right_diagonal(4);
        let scan_8 = up_right_diagonal(8);

        let mut out = ScalingFactors {
            factors: core::array::from_fn(|size_id| {
                let dim = 1usize << (2 + size_id); // 4, 8, 16, 32
                core::array::from_fn(|_matrix_id| ScalingFactorMatrix {
                    dim: dim as u8,
                    coef: vec![0u16; dim * dim],
                })
            }),
        };

        for matrix_id in 0..NUM_MATRIX_IDS {
            // 4x4 (sizeId 0) — equation 7-44: place ScalingList[0][m][i]
            // at (ScanOrder[2][0][i][0], ScanOrder[2][0][i][1]).
            place(
                &mut out.factors[0][matrix_id],
                &self.lists[0][matrix_id].coef,
                &scan_4,
                1,
            );

            // 8x8 (sizeId 1) — equation 7-45: 8x8 scan, no upsampling.
            place(
                &mut out.factors[1][matrix_id],
                &self.lists[1][matrix_id].coef,
                &scan_8,
                1,
            );

            // 16x16 (sizeId 2) — equation 7-46: 8x8 scan, each entry
            // replicated into a 2x2 block. Then 7-47 overrides [0][0]
            // with the DC coefficient.
            place(
                &mut out.factors[2][matrix_id],
                &self.lists[2][matrix_id].coef,
                &scan_8,
                2,
            );
            set_dc(
                &mut out.factors[2][matrix_id],
                self.lists[2][matrix_id].dc_coef,
            );
        }

        // 32x32 (sizeId 3) — only matrixId 0 (intra Y) and 3 (inter Y)
        // are signalled (the `matrixId += 3` step). Equation 7-48 uses
        // the 8x8 scan with each entry replicated into a 4x4 block,
        // 7-49 overrides [0][0] with the DC coefficient.
        for &matrix_id in &[0usize, 3usize] {
            place(
                &mut out.factors[3][matrix_id],
                &self.lists[3][matrix_id].coef,
                &scan_8,
                4,
            );
            set_dc(
                &mut out.factors[3][matrix_id],
                self.lists[3][matrix_id].dc_coef,
            );
        }

        // 32x32 chroma (matrixId 1, 2, 4, 5) — equations 7-50 / 7-51,
        // applicable only for ChromaArrayType == 3 (4:4:4). The flat
        // list comes from the 16x16 (sizeId 2) list of the same
        // matrixId, and the DC coefficient also from sizeId 2.
        if chroma_array_type == 3 {
            for &matrix_id in &[1usize, 2usize, 4usize, 5usize] {
                place(
                    &mut out.factors[3][matrix_id],
                    &self.lists[2][matrix_id].coef,
                    &scan_8,
                    4,
                );
                set_dc(
                    &mut out.factors[3][matrix_id],
                    self.lists[2][matrix_id].dc_coef,
                );
            }
        }

        out
    }
}

/// Scatter a flat scaling list into a `ScalingFactor` matrix along the
/// up-right diagonal `scan`, replicating each coefficient into a
/// `rep`x`rep` block (`rep == 1` for 4x4 / 8x8, 2 for 16x16, 4 for
/// 32x32 — the `x * rep + k` / `y * rep + j` indexing of equations
/// 7-46 / 7-48 / 7-50).
fn place(
    matrix: &mut ScalingFactorMatrix,
    coef: &[u16],
    scan: &[crate::hevc::engine::scan::ScanPos],
    rep: usize,
) {
    let dim = matrix.dim as usize;
    for (i, value) in coef.iter().copied().enumerate() {
        let pos = scan[i];
        let bx = pos.x as usize * rep;
        let by = pos.y as usize * rep;
        for j in 0..rep {
            for k in 0..rep {
                let x = bx + k;
                let y = by + j;
                matrix.coef[y * dim + x] = value;
            }
        }
    }
}

/// Override `ScalingFactor[sizeId][matrixId][0][0]` with the DC
/// coefficient (equations 7-47 / 7-49 / 7-51).
fn set_dc(matrix: &mut ScalingFactorMatrix, dc_coef: u16) {
    matrix.coef[0] = dc_coef;
}

/// One derived §7.4.5 `ScalingFactor[sizeId][matrixId][x][y]`
/// quantization matrix, stored row-major (`coef[y * dim + x]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingFactorMatrix {
    /// Side length of the (square) matrix: 4, 8, 16, or 32 — `1 << (2 +
    /// sizeId)`.
    pub dim: u8,
    /// The `dim * dim` scaling factors, row-major: the value at column
    /// `x`, row `y` is `coef[y * dim + x]`.
    pub coef: Vec<u16>,
}

impl ScalingFactorMatrix {
    /// `ScalingFactor[..][..][x][y]` — the scaling factor at column `x`
    /// (horizontal), row `y` (vertical). Panics if `x` or `y` is `>=
    /// dim`.
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u16 {
        let dim = self.dim as usize;
        assert!(x < dim && y < dim, "ScalingFactor index out of range");
        self.coef[y * dim + x]
    }
}

/// The full set of §7.4.5 `ScalingFactor[sizeId][matrixId][x][y]`
/// quantization matrices, derived from a [`ScalingListData`] via
/// [`ScalingListData::scaling_factors`].
///
/// The 32x32 chroma matrices (`factors[3][1]`, `[3][2]`, `[3][4]`,
/// `[3][5]`) are all-zero unless `ChromaArrayType == 3`; see
/// [`ScalingListData::scaling_factors`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingFactors {
    /// `[sizeId][matrixId]` — the derived quantization matrices.
    pub factors: [[ScalingFactorMatrix; NUM_MATRIX_IDS]; NUM_SIZE_IDS],
}

#[cfg(any())]
mod tests {
    use super::*;

    /// Pack a sequence of `(value, bit_width)` MSB-first into bytes.
    fn pack(fields: &[(u64, u32)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut cur = 0u8;
        let mut nbits = 0u32;
        for &(val, w) in fields {
            for i in (0..w).rev() {
                let bit = ((val >> i) & 1) as u8;
                cur = (cur << 1) | bit;
                nbits += 1;
                if nbits == 8 {
                    out.push(cur);
                    cur = 0;
                    nbits = 0;
                }
            }
        }
        if nbits > 0 {
            cur <<= 8 - nbits;
            out.push(cur);
        }
        out
    }

    /// ue(v) codeword for a small codeNum, as `(value, width)` fields.
    /// codeNum 0 → '1'; codeNum N → prefix of `m` zeros, a 1, then
    /// `m`-bit suffix where `m = floor(log2(N+1))`.
    fn ue(code_num: u32) -> Vec<(u64, u32)> {
        let x = code_num + 1;
        let m = 31 - x.leading_zeros(); // floor(log2(x))
        // m leading zeros + the implicit leading 1 of `x` (so `x` in
        // (m+1) bits) gives the standard Exp-Golomb codeword.
        vec![(0, m), (x as u64, m + 1)]
    }

    /// se(v) codeword: map value → codeNum per Table 9-3, then ue().
    fn se(value: i32) -> Vec<(u64, u32)> {
        let code_num = if value > 0 {
            (2 * value - 1) as u32
        } else {
            (-2 * value) as u32
        };
        ue(code_num)
    }

    /// A `scaling_list_data()` where every one of the 24 slots is
    /// pred_mode_flag=0, delta=0 (i.e. "use the default list").
    /// 24 slots visited: sizeId 0/1/2 each 6, sizeId 3 only 2.
    fn all_default_bits() -> Vec<(u64, u32)> {
        let mut f = Vec::new();
        for size_id in 0..4 {
            let step = if size_id == 3 { 3 } else { 1 };
            let mut m = 0;
            while m < 6 {
                f.push((0, 1)); // scaling_list_pred_mode_flag = 0
                f.extend(ue(0)); // scaling_list_pred_matrix_id_delta = 0
                m += step;
            }
        }
        f
    }

    #[test]
    fn all_default_lists_match_tables() {
        let bits = all_default_bits();
        let buf = pack(&bits);
        let mut br = BitReader::new(&buf);
        let data = ScalingListData::parse(&mut br).unwrap();

        // 4x4 — Table 7-5: all 16.
        for m in 0..6 {
            assert_eq!(data.lists[0][m].coef, vec![16u16; 16]);
        }
        // 8x8 intra (matrixId 0..2) — Table 7-6.
        for m in 0..3 {
            assert_eq!(data.lists[1][m].coef, DEFAULT_8X8_INTRA.to_vec());
        }
        // 8x8 inter (matrixId 3..5).
        for m in 3..6 {
            assert_eq!(data.lists[1][m].coef, DEFAULT_8X8_INTER.to_vec());
        }
        // 16x16 reuses the 8x8 defaults; the default DC is 16 (§7.4.5:
        // scaling_list_dc_coef_minus8 inferred to 8 ⇒ 8 + 8).
        assert_eq!(data.lists[2][0].coef, DEFAULT_8X8_INTRA.to_vec());
        assert_eq!(data.lists[2][0].dc_coef, 16);
        assert_eq!(data.lists[2][3].coef, DEFAULT_8X8_INTER.to_vec());
        // 32x32 visits only matrixId 0 (intra) and 3 (inter).
        assert_eq!(data.lists[3][0].coef, DEFAULT_8X8_INTRA.to_vec());
        assert_eq!(data.lists[3][3].coef, DEFAULT_8X8_INTER.to_vec());
        assert_eq!(data.lists[3][0].dc_coef, 16);
    }

    /// Coefficient-numbering: sizeId 0 carries 16, sizeId 1..3 carry 64.
    #[test]
    fn coef_num_per_size_id() {
        assert_eq!(coef_num(0), 16);
        assert_eq!(coef_num(1), 64);
        assert_eq!(coef_num(2), 64);
        assert_eq!(coef_num(3), 64);
    }

    /// An explicitly-signalled 4x4 list (sizeId 0, matrixId 0): all 16
    /// slots carry delta 0, so every coefficient equals the
    /// `nextCoef` seed (8). Verifies the running modulo accumulator and
    /// that no DC coefficient is read for sizeId <= 1.
    #[test]
    fn explicit_flat_4x4_list() {
        let mut f = Vec::new();
        // sizeId 0, matrixId 0: pred_mode = 1, 16 delta_coef = 0.
        f.push((1, 1));
        for _ in 0..16 {
            f.extend(se(0));
        }
        // Remaining 23 slots: default (pred_mode = 0, delta = 0).
        for size_id in 0..4 {
            let step = if size_id == 3 { 3 } else { 1 };
            let mut m = 0;
            while m < 6 {
                if !(size_id == 0 && m == 0) {
                    f.push((0, 1));
                    f.extend(ue(0));
                }
                m += step;
            }
        }
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let data = ScalingListData::parse(&mut br).unwrap();
        // nextCoef seeds at 8; every delta is 0 ⇒ all coefs = 8.
        assert_eq!(data.lists[0][0].coef, vec![8u16; 16]);
    }

    /// Explicit 16x16 list (sizeId 2) reads a DC coefficient first.
    /// dc = +4 ⇒ scaling_list_dc_coef_minus8 = 4 ⇒ dc_coef = 12,
    /// nextCoef seeds at 12; deltas all 0 ⇒ every coef = 12.
    #[test]
    fn explicit_16x16_reads_dc_coef() {
        let mut f = Vec::new();
        // Default sizeId 0 and 1 (12 slots).
        for _ in 0..2 {
            for _ in 0..6 {
                f.push((0, 1));
                f.extend(ue(0));
            }
        }
        // sizeId 2, matrixId 0: explicit with DC = +4.
        f.push((1, 1));
        f.extend(se(4)); // scaling_list_dc_coef_minus8 = 4
        for _ in 0..64 {
            f.extend(se(0));
        }
        // sizeId 2, matrixId 1..5: default.
        for _ in 1..6 {
            f.push((0, 1));
            f.extend(ue(0));
        }
        // sizeId 3, matrixId 0 and 3: default.
        for _ in 0..2 {
            f.push((0, 1));
            f.extend(ue(0));
        }
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let data = ScalingListData::parse(&mut br).unwrap();
        assert_eq!(data.lists[2][0].dc_coef, 12);
        assert_eq!(data.lists[2][0].coef, vec![12u16; 64]);
    }

    /// Prediction with delta != 0 copies the referenced list (eq 7-43),
    /// including its DC coefficient. Set sizeId 2, matrixId 0 explicit
    /// with DC=+4, then matrixId 1 predicts from it (delta = 1 ⇒
    /// refMatrixId = 0).
    #[test]
    fn pred_matrix_id_delta_copies_reference() {
        let mut f = Vec::new();
        for _ in 0..2 {
            for _ in 0..6 {
                f.push((0, 1));
                f.extend(ue(0));
            }
        }
        // sizeId 2, matrixId 0: explicit, DC = +4, all deltas 0.
        f.push((1, 1));
        f.extend(se(4));
        for _ in 0..64 {
            f.extend(se(0));
        }
        // sizeId 2, matrixId 1: predicts from matrixId 0 (delta = 1).
        f.push((0, 1));
        f.extend(ue(1));
        // sizeId 2, matrixId 2..5: default.
        for _ in 2..6 {
            f.push((0, 1));
            f.extend(ue(0));
        }
        // sizeId 3, matrixId 0 and 3: default.
        for _ in 0..2 {
            f.push((0, 1));
            f.extend(ue(0));
        }
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let data = ScalingListData::parse(&mut br).unwrap();
        // matrixId 1 == copy of matrixId 0 (coefs and DC).
        assert_eq!(data.lists[2][1].coef, data.lists[2][0].coef);
        assert_eq!(data.lists[2][1].dc_coef, 12);
    }

    /// `scaling_list_pred_matrix_id_delta` out of range is rejected:
    /// for sizeId 0, matrixId 0 the max allowed delta is 0, so delta=1
    /// must error.
    #[test]
    fn rejects_pred_matrix_id_delta_out_of_range() {
        let mut f = Vec::new();
        f.push((0, 1)); // pred_mode = 0
        f.extend(ue(1)); // delta = 1 > matrixId(0)
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let err = ScalingListData::parse(&mut br).unwrap_err();
        assert_eq!(
            err,
            ScalingListError::PredMatrixIdDeltaOutOfRange {
                size_id: 0,
                matrix_id: 0,
                got: 1,
            }
        );
    }

    /// A delta-coef sequence that drives `nextCoef` to 0 (modulo 256)
    /// must be rejected (§7.4.5: ScalingList coefficient > 0).
    #[test]
    fn rejects_non_positive_coef() {
        let mut f = Vec::new();
        // sizeId 0, matrixId 0: explicit. nextCoef seeds at 8; a
        // delta_coef of −8 wraps it to 0.
        f.push((1, 1));
        f.extend(se(-8));
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let err = ScalingListData::parse(&mut br).unwrap_err();
        assert_eq!(
            err,
            ScalingListError::NonPositiveCoef {
                size_id: 0,
                matrix_id: 0,
                index: 0,
            }
        );
    }

    /// Fuzz regression (r282 `parse_annexb`): §7.4.5 bounds
    /// `scaling_list_delta_coef` to −128..=127, inclusive. An
    /// unbounded `se(v)` value previously overflowed the i32
    /// `nextCoef + scaling_list_delta_coef + 256` sum.
    #[test]
    fn rejects_delta_coef_out_of_range() {
        let mut f = Vec::new();
        // sizeId 0, matrixId 0: explicit list; first delta_coef
        // = −129 (codeNum 258), one past the §7.4.5 lower bound.
        f.push((1, 1));
        f.extend(se(-129));
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let err = ScalingListData::parse(&mut br).unwrap_err();
        assert_eq!(
            err,
            ScalingListError::DeltaCoefOutOfRange {
                size_id: 0,
                matrix_id: 0,
                index: 0,
                got: -129,
            }
        );
    }

    /// `scaling_list_dc_coef_minus8` outside −7..=247 is rejected.
    /// se(v) for −8 (codeNum 16) gives dc = −8 < −7.
    #[test]
    fn rejects_dc_coef_out_of_range() {
        let mut f = Vec::new();
        for _ in 0..2 {
            for _ in 0..6 {
                f.push((0, 1));
                f.extend(ue(0));
            }
        }
        // sizeId 2, matrixId 0: explicit with DC = −8 (out of range).
        f.push((1, 1));
        f.extend(se(-8));
        let buf = pack(&f);
        let mut br = BitReader::new(&buf);
        let err = ScalingListData::parse(&mut br).unwrap_err();
        assert_eq!(
            err,
            ScalingListError::DcCoefOutOfRange {
                size_id: 2,
                matrix_id: 0,
                got: -8,
            }
        );
    }

    /// Truncation midway through the structure surfaces `Truncated`.
    #[test]
    fn truncation_is_reported() {
        // Just one bit — not enough for the first slot.
        let buf = [0u8; 0];
        let mut br = BitReader::new(&buf);
        assert_eq!(
            ScalingListData::parse(&mut br).unwrap_err(),
            ScalingListError::Truncated
        );
    }

    /// Parse the all-default `scaling_list_data()` and derive its
    /// `ScalingFactor` matrices.
    fn all_default_factors(chroma_array_type: u8) -> ScalingFactors {
        let buf = pack(&all_default_bits());
        let mut br = BitReader::new(&buf);
        ScalingListData::parse(&mut br)
            .unwrap()
            .scaling_factors(chroma_array_type)
    }

    /// 4x4 (sizeId 0) `ScalingFactor` matrices are 4x4, and — since the
    /// default 4x4 list is flat 16 — every cell is 16 (equation 7-44).
    /// Also exercises the [`ScalingFactorMatrix::at`] accessor.
    #[test]
    fn factor_4x4_default_all_16() {
        let f = all_default_factors(1);
        for m in 0..NUM_MATRIX_IDS {
            let mat = &f.factors[0][m];
            assert_eq!(mat.dim, 4);
            assert_eq!(mat.coef.len(), 16);
            assert!(mat.coef.iter().all(|&c| c == 16));
            assert_eq!(mat.at(2, 3), 16);
        }
    }

    /// 8x8 (sizeId 1) default intra `ScalingFactor` is the 8x8 intra
    /// table scattered along the up-right diagonal scan (equation 7-45).
    /// Spot-check the diagonal placement: `ScalingList[1][0][i]` lands
    /// at `(ScanOrder[3][0][i][0], ScanOrder[3][0][i][1])`.
    #[test]
    fn factor_8x8_intra_follows_diagonal_scan() {
        let f = all_default_factors(1);
        let mat = &f.factors[1][0];
        assert_eq!(mat.dim, 8);
        let scan = crate::hevc::engine::scan::up_right_diagonal(8);
        for (i, &expected) in DEFAULT_8X8_INTRA.iter().enumerate() {
            let p = scan[i];
            assert_eq!(
                mat.at(p.x as usize, p.y as usize),
                expected,
                "8x8 intra coef i={i} misplaced"
            );
        }
        // i = 0 maps to (0, 0): the first table entry is 16.
        assert_eq!(mat.at(0, 0), DEFAULT_8X8_INTRA[0]);
    }

    /// 16x16 (sizeId 2) uses the 8x8 scan with each entry replicated
    /// into a 2x2 block (equation 7-46), then the DC coefficient
    /// overrides `[0][0]` (equation 7-47). Build an explicit list with
    /// DC = +4 ⇒ `dc_coef = 12`, which also seeds `next_coef`; with all
    /// deltas 0 every flat list coef is 12, so the whole matrix is 12 —
    /// confirming both the 2x2 replication blankets the grid and the DC
    /// override lands on `[0][0]`. A second slot (DC = 0, delta seeds an
    /// i=0 coef of 8) separates the DC override from the list body.
    #[test]
    fn factor_16x16_replicates_2x2_and_dc_override() {
        // Default sizeId 0 and 1.
        let mut bits = Vec::new();
        for _ in 0..2 {
            for _ in 0..6 {
                bits.push((0, 1));
                bits.extend(ue(0));
            }
        }
        // sizeId 2, matrixId 0: explicit, DC = +4 (dc_coef 12), deltas 0
        // ⇒ every flat coef = 12 (next_coef seeds from the DC value).
        bits.push((1, 1));
        bits.extend(se(4));
        for _ in 0..64 {
            bits.extend(se(0));
        }
        // sizeId 2, matrixId 1: explicit, DC = -8+? -> use dc=0 so
        // dc_coef = 8, then a first delta of 0 ⇒ list coef 8 throughout;
        // the DC override is then a no-op (8 == 8) — but matrixId 2 sets
        // DC -1 to separate the override from the list body.
        bits.push((1, 1));
        bits.extend(se(0)); // dc_coef 8
        for _ in 0..64 {
            bits.extend(se(0));
        }
        // sizeId 2, matrixId 2: explicit, DC = -1 (dc_coef 7) but list
        // body driven to 8 (next_coef seeds at 7, first delta +1 ⇒ 8).
        bits.push((1, 1));
        bits.extend(se(-1)); // dc_coef 7
        bits.extend(se(1)); // i=0: next_coef 7+1 = 8
        for _ in 1..64 {
            bits.extend(se(0)); // remaining deltas 0 ⇒ all 8
        }
        // sizeId 2, matrixId 3..5 default; sizeId 3 matrixId 0/3 default.
        for _ in 3..6 {
            bits.push((0, 1));
            bits.extend(ue(0));
        }
        for _ in 0..2 {
            bits.push((0, 1));
            bits.extend(ue(0));
        }
        let buf = pack(&bits);
        let mut br = BitReader::new(&buf);
        let data = ScalingListData::parse(&mut br).unwrap();
        let f = data.scaling_factors(1);

        // matrixId 0: DC seeds list, all 12.
        let mat0 = &f.factors[2][0];
        assert_eq!(mat0.dim, 16);
        assert_eq!(mat0.coef.len(), 256);
        assert!(mat0.coef.iter().all(|&c| c == 12));

        // matrixId 2: list body 8, DC override 7 at [0][0] only — proves
        // the override is isolated to the single corner cell while the
        // 2x2 replication blankets the rest with the list value.
        let mat2 = &f.factors[2][2];
        assert_eq!(mat2.at(0, 0), 7);
        assert_eq!(mat2.at(1, 0), 8);
        assert_eq!(mat2.at(0, 1), 8);
        assert_eq!(mat2.at(1, 1), 8);
        for y in 0..16 {
            for x in 0..16 {
                if x == 0 && y == 0 {
                    continue;
                }
                assert_eq!(mat2.at(x, y), 8, "cell ({x},{y})");
            }
        }
    }

    /// 32x32 (sizeId 3) only derives matrixId 0 (intra Y) and 3
    /// (inter Y) for non-4:4:4 chroma; each 8x8-scan entry replicates
    /// into a 4x4 block (equation 7-48) and the DC overrides [0][0]
    /// (7-49). The chroma matrices (1,2,4,5) stay all-zero.
    #[test]
    fn factor_32x32_replicates_4x4_chroma_zero_when_not_444() {
        let f = all_default_factors(1); // ChromaArrayType 1 (4:2:0)
        let intra = &f.factors[3][0];
        assert_eq!(intra.dim, 32);
        assert_eq!(intra.coef.len(), 1024);
        // The default DC is 16 (scaling_list_dc_coef_minus8 inferred
        // to 8), matching the i=0 list coefficient.
        assert_eq!(intra.at(0, 0), 16);
        assert_eq!(intra.at(1, 0), 16);
        assert_eq!(intra.at(3, 3), 16);
        // Inter Y (matrixId 3) is also derived.
        assert_eq!(f.factors[3][3].at(1, 0), 16);
        // Chroma matrices are untouched (all-zero) for ChromaArrayType
        // != 3.
        for &m in &[1usize, 2, 4, 5] {
            assert!(f.factors[3][m].coef.iter().all(|&c| c == 0));
        }
    }

    /// With `ChromaArrayType == 3` (4:4:4) the 32x32 chroma matrices
    /// (matrixId 1, 2, 4, 5) are derived from the 16x16 (sizeId 2)
    /// lists of the same matrixId (equations 7-50 / 7-51), so they are
    /// no longer all-zero.
    #[test]
    fn factor_32x32_chroma_derived_when_444() {
        let f = all_default_factors(3);
        for &m in &[1usize, 2, 4, 5] {
            let mat = &f.factors[3][m];
            assert_eq!(mat.dim, 32);
            // Derived from the default 16x16 list (intra for m<3, inter
            // otherwise) ⇒ not all-zero; the default DC 16 at [0][0].
            assert!(mat.coef.iter().any(|&c| c != 0));
            assert_eq!(mat.at(0, 0), 16);
        }
        // The chroma 32x32 matrix matches the 16x16 luma-scan placement
        // of the same matrixId's 8x8 default table, replicated 4x4.
        let mat = &f.factors[3][1];
        let scan = crate::hevc::engine::scan::up_right_diagonal(8);
        // i=1 (intra default table entry) lands at scan[1] scaled by 4.
        let p = scan[1];
        assert_eq!(
            mat.at(p.x as usize * 4, p.y as usize * 4),
            DEFAULT_8X8_INTRA[1]
        );
    }
}
