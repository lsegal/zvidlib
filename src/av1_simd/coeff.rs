//! Vectorized `coeff_base` / `coeff_br` context derivation (§8.3.2) for the
//! native AV1 encoder's coefficient coding loop.
//!
//! # What is data-parallel here and what is not
//!
//! The symbol coder itself is serial by construction: every symbol updates the
//! CDF and the range coder state the next symbol is written against. The
//! *context derivation* around it is not. `getCoeffBaseCtx` and `getCoeffBrCtx`
//! read only the neighbours *down and to the right* of a coefficient
//! (`Sig_Ref_Diff_Offset` and `Mag_Ref_Offset` for `TX_CLASS_2D` are all
//! non-negative), and the encoder walks the up-right diagonal scan **backwards**
//! from the end-of-block. Every neighbour a position consults therefore sits on
//! a strictly later anti-diagonal, so it has already been coded and its level is
//! final — and positions at or past the end-of-block are zero by the definition
//! of `eob`. The whole level plane is consequently known before the first symbol
//! is written, and every position's context can be derived in one pass, ahead of
//! the serial loop, independently of every other position.
//!
//! That pass is what this module vectorizes: the clamped neighbour magnitude
//! sums, the `(mag + 1) >> 1` saturation, and the position-dependent context
//! offsets, computed `LANES` coefficients at a time along each row of the
//! transform block.
//!
//! # The padded level plane
//!
//! A neighbour at `(row + dr, col + dc)` is defined as zero once it leaves the
//! block, and both offset tables reach at most two rows down and two columns
//! right. Rather than branch per lane, the caller keeps the levels in a plane
//! padded by [`MAX_ROW_OFFSET`] zero rows and [`MAX_COL_OFFSET`] zero columns
//! (plus a vector's worth of slack so a full-width load at the last column
//! stays in bounds), which turns each neighbour into an unaligned load. The
//! padding is written once when the plane is sized and never again: a block only
//! ever overwrites the `size x size` interior, so the per-block cost is the
//! clamped copy of the levels themselves and nothing more.
//!
//! # Bit-exactness
//!
//! The kernel is a lane-by-lane transliteration of `tile.rs`'s scalar
//! `coeff_base_ctx` and `coeff_br_ctx`, including the DC position's special
//! cases, and levels are clamped to 15 on the way into the plane so a lane can
//! never overflow: 5 neighbours x 15 fits in a handful of bits. The encoded
//! bitstream is identical under every instruction set, which `tests/av1_simd.rs`
//! and the encoder's own round-trip tests assert.

use super::SimdIsa;
use super::vector::{I32x, MAX_LANES};
use crate::av1_encoder::cdf::{MAG_REF_OFFSET_2D, SIG_REF_DIFF_OFFSET_2D};
use std::sync::OnceLock;

/// Rows below a coefficient that `Sig_Ref_Diff_Offset[TX_CLASS_2D]` reaches.
pub(crate) const MAX_ROW_OFFSET: usize = 2;
/// Columns right of a coefficient that `Sig_Ref_Diff_Offset[TX_CLASS_2D]` reaches.
pub(crate) const MAX_COL_OFFSET: usize = 2;

/// The `coeff_br` context offset for a position outside the top-left 2x2 corner.
const BR_FAR_OFFSET: i32 = 14;
/// The `coeff_br` context offset for a non-DC position inside that corner.
const BR_NEAR_OFFSET: i32 = 7;

/// Row stride of the padded level plane for a `size x size` transform block.
///
/// The `MAX_LANES` slack lets the widest vector load start at the last column
/// plus [`MAX_COL_OFFSET`] and still read whole lanes from inside the buffer;
/// the lanes past the block's width land on padding zeros and are dropped by the
/// masked store.
#[must_use]
pub(crate) const fn padded_stride(size: usize) -> usize {
    size + MAX_COL_OFFSET + MAX_LANES
}

/// Number of rows in the padded level plane for a `size x size` block.
#[must_use]
pub(crate) const fn padded_rows(size: usize) -> usize {
    size + MAX_ROW_OFFSET
}

/// Total length of the padded level plane for a `size x size` block.
#[must_use]
pub(crate) const fn padded_len(size: usize) -> usize {
    padded_rows(size) * padded_stride(size)
}

/// The instruction set this dispatch family will actually run.
///
/// Registered as the `av1_coeff_ctx` site of [`crate::simd::active_by_site`].
/// Like every other site it consults [`crate::simd::set_override`] first and
/// falls back to its own cached CPU probe, so pinning an instruction set reaches
/// the coefficient contexts and a benchmark can prove that it did.
#[must_use]
pub fn active_isa() -> SimdIsa {
    static DETECTED: OnceLock<SimdIsa> = OnceLock::new();
    crate::simd::override_isa().unwrap_or_else(|| *DETECTED.get_or_init(super::detected_isa))
}

/// Whether `isa` has a vector kernel in this build; `false` means the caller
/// must derive the contexts with its scalar reference.
#[must_use]
pub(crate) fn has_vector_kernel(isa: SimdIsa) -> bool {
    match isa {
        SimdIsa::Scalar => false,
        #[cfg(target_arch = "x86_64")]
        SimdIsa::Sse41 | SimdIsa::Avx2 => true,
        #[cfg(target_arch = "aarch64")]
        SimdIsa::Neon => true,
        _ => false,
    }
}

/// Writes the zero padding of a `size x size` padded level plane, sizing
/// `plane` to [`padded_len`] first. The interior is left for
/// [`fill_padded_levels`]; the padding written here is never touched again.
pub(crate) fn reset_padded_plane(plane: &mut Vec<i32>, size: usize) {
    plane.clear();
    plane.resize(padded_len(size), 0);
}

/// Copies the magnitudes of `coefficients` into the interior of a plane already
/// padded by [`reset_padded_plane`], clamping to 15 — the largest magnitude
/// either context sum distinguishes, and the clamp that keeps a 32-bit lane far
/// from overflow.
pub(crate) fn fill_padded_levels(plane: &mut [i32], coefficients: &[i32], size: usize) {
    let stride = padded_stride(size);
    for row in 0..size {
        let source = &coefficients[row * size..row * size + size];
        let target = &mut plane[row * stride..row * stride + size];
        for (slot, &value) in target.iter_mut().zip(source) {
            *slot = value.abs().min(15);
        }
    }
}

/// Derives every position's `coeff_base` and `coeff_br` context from a padded
/// level plane, `V::LANES` coefficients at a time.
///
/// `base_out` and `br_out` are `size * size` long and indexed by raster position,
/// matching the scalar reference they are checked against.
///
/// # Safety
///
/// Only callable from a wrapper that has verified `V`'s CPU feature.
pub(crate) unsafe fn block_contexts<V: I32x>(
    plane: &[i32],
    size: usize,
    base_out: &mut [i32],
    br_out: &mut [i32],
) {
    unsafe {
        let stride = padded_stride(size);
        let one = V::splat(1);
        let three = V::splat(3);
        let fifteen = V::splat(15);
        let mut lanes = [0i32; MAX_LANES];
        for row in 0..size {
            let mut column = 0usize;
            while column < size {
                for (lane, slot) in lanes.iter_mut().enumerate().take(V::LANES) {
                    *slot = (column + lane) as i32;
                }
                let columns = V::load(&lanes);

                let mut base_mag = V::zero();
                for &(dr, dc) in &SIG_REF_DIFF_OFFSET_2D {
                    let neighbour = V::load(&plane[(row + dr) * stride + column + dc..]);
                    base_mag = base_mag.add(neighbour.min(three));
                }
                let mut br_mag = V::zero();
                for &(dr, dc) in &MAG_REF_OFFSET_2D {
                    let neighbour = V::load(&plane[(row + dr) * stride + column + dc..]);
                    br_mag = br_mag.add(neighbour.min(fifteen));
                }

                // `coeff_base_ctx_offset` is a function of the anti-diagonal
                // alone: 1 on the first two, 6 on the next two, 21 beyond.
                let diagonal = columns.add(V::splat(row as i32));
                let base_offset = V::select(
                    diagonal.le(one),
                    V::splat(1),
                    V::select(diagonal.le(V::splat(3)), V::splat(6), V::splat(21)),
                );
                let base = base_mag
                    .add(one)
                    .sra::<1>()
                    .min(V::splat(4))
                    .add(base_offset);

                // The `coeff_br` offset splits on the top-left 2x2 corner.
                let near = if row < 2 { columns.le(one) } else { V::zero() };
                let br_offset = V::select(near, V::splat(BR_NEAR_OFFSET), V::splat(BR_FAR_OFFSET));
                let br = br_mag.add(one).sra::<1>().min(V::splat(6)).add(br_offset);

                let count = (size - column).min(V::LANES);
                base.store_masked(&mut base_out[row * size + column..], count);
                br.store_masked(&mut br_out[row * size + column..], count);
                column += V::LANES;
            }
        }
        // The DC position is the one lane the uniform formulas cannot express:
        // its `coeff_base` context is 0 outright, and its `coeff_br` context is
        // the magnitude alone with no positional offset. Lane 0 of row 0 was
        // computed with the near-corner offset, so removing that recovers it.
        base_out[0] = 0;
        br_out[0] -= BR_NEAR_OFFSET;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1_cdf::coeff_base_ctx_offset;

    /// The vector kernel folds `coeff_base_ctx_offset` into an anti-diagonal
    /// comparison and the `coeff_br` offset into a 2x2 corner test. Both are
    /// only legal as long as the tables they stand in for say so.
    #[test]
    fn the_folded_context_offsets_match_the_specification_tables() {
        for row in 0..64usize {
            for column in 0..64usize {
                let expected = match row + column {
                    0 | 1 => 1,
                    2 | 3 => 6,
                    _ => 21,
                };
                assert_eq!(
                    coeff_base_ctx_offset(row, column),
                    expected,
                    "row {row}, column {column}"
                );
            }
        }
    }

    /// The padded plane exists so a neighbour load never leaves the buffer.
    #[test]
    fn the_padding_covers_every_neighbour_offset() {
        for &(dr, dc) in SIG_REF_DIFF_OFFSET_2D.iter().chain(&MAG_REF_OFFSET_2D) {
            assert!(dr <= MAX_ROW_OFFSET, "row offset {dr}");
            assert!(dc <= MAX_COL_OFFSET, "column offset {dc}");
        }
        for size in [4usize, 8, 16, 32, 64] {
            let stride = padded_stride(size);
            let last = (size - 1 + MAX_ROW_OFFSET) * stride + (size - 1 + MAX_COL_OFFSET);
            assert!(
                last + MAX_LANES <= padded_len(size),
                "a widest-vector load at the last position leaves the plane at size {size}"
            );
        }
    }
}
