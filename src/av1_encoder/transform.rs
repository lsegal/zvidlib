//! AV1 encoder-side forward DCT and ADST kernels (issue #140).
//!
//! [`crate::av1_intra::inverse_transform`] implements the non-lossless
//! *inverse* transform set: the 4-, 8-, 16-, and 32-point inverse DCT and the
//! 4-, 8-, and 16-point inverse ADST. Encoding needs the other direction, and
//! until now the only forward transform in the crate was the lossless 4x4
//! Walsh-Hadamard transform in [`super::wht`].
//!
//! # Lineage
//!
//! These kernels are the transposes of the inverse ones, not an independently
//! chosen transform. Writing the inverse kernel of size `N` as
//!
//! ```text
//! x[n] = sum_k X[k] * b(k, n)
//! ```
//!
//! the forward kernel here is `X[k] = sum_n x[n] * b(k, n)` over the *same*
//! `b`, so a forward/inverse round trip is well defined by construction. The
//! basis is held at the inverse kernels' own 2^14 fixed-point scale and is
//! generated from the same `cos(j*pi/64)` and `sin(j*pi/9)` constants those
//! kernels multiply by ([`COSPI_64`], [`SINPI_9`]):
//!
//! ```text
//! DCT,  N points:  b(k, n) = c(k) * cos(pi * (2n + 1) * k / (2N)),  c(0) = 1/sqrt(2), else 1
//! ADST, N = 8, 16: b(k, n) = sin(pi * (2k + 1) * (2n + 1) / (4N))
//! ADST, N = 4:     b(k, n) = (2 * sqrt(2) / 3) * sin(pi * (2k + 1) * (n + 1) / 9)
//! ```
//!
//! `tests::basis_matches_inverse_kernels` asserts entry-for-entry that the
//! generated tables are the impulse responses of the inverse kernels, and
//! `tests::basis_matches_double_precision` asserts they are the rounded
//! `f64` evaluation of the formulas above. Both hold for all 1696 entries.
//!
//! # Rounding schedule and scale
//!
//! Each 1-D pass is one `dct_round_shift` per output, `(v + 2^13) >> 14`,
//! exactly as in the inverse kernels. A 1-D forward followed by a 1-D inverse
//! has gain `N / 2`, so a 2-D round trip has gain `(N / 2)^2` while the
//! inverse applies its own `>> transform_shift(N)`. The forward pass therefore
//! carries the difference, `2^transform_shift(N) / (N / 2)^2`, as a left shift
//! of the input ([`pre_shift`]) and a rounding right shift between the two
//! passes ([`mid_shift`]):
//!
//! | `N` | `pre_shift` | `mid_shift` | net scale |
//! |-----|-------------|-------------|-----------|
//! | 4   | 2           | 0           | `2^2`     |
//! | 8   | 1           | 0           | `2^1`     |
//! | 16  | 0           | 0           | `1`       |
//! | 32  | 0           | 2           | `2^-2`    |
//!
//! so `inverse_transform(forward_transform(r), .., 1, 1) == r` up to the
//! kernels' own rounding error, which the round-trip tests bound.

use crate::av1_intra::{Av1TxType, Tx1d};

/// `round(2^14 * cos(j * pi / 64))` for `j` in `0..=32`. Entries 1 through 31
/// are the `COSPI_j_64` constants the inverse kernels use, unchanged.
pub(crate) const COSPI_64: [i32; 33] = [
    16384, 16364, 16305, 16207, 16069, 15893, 15679, 15426, 15137, 14811, 14449, 14053, 13623,
    13160, 12665, 12140, 11585, 11003, 10394, 9760, 9102, 8423, 7723, 7005, 6270, 5520, 4756, 3981,
    3196, 2404, 1606, 804, 0,
];

/// `round(2^14 * (2 * sqrt(2) / 3) * sin(j * pi / 9))` for `j` in `0..=9`.
/// Entries 1 through 4 are the `SINPI_j_9` constants of the inverse 4-point
/// ADST, unchanged; the rest follow from `sin(pi - t) == sin(t)`.
pub(crate) const SINPI_9: [i32; 10] = [0, 5283, 9929, 13377, 15212, 15212, 13377, 9929, 5283, 0];

/// `round(2^14 * cos(index * pi / 64))` for any integer `index`, using the
/// period-128 symmetry of the cosine to fold onto [`COSPI_64`].
const fn cospi_64(index: i64) -> i32 {
    let mut index = index % 128;
    if index < 0 {
        index += 128;
    }
    if index > 64 {
        index = 128 - index;
    }
    if index > 32 {
        -COSPI_64[(64 - index) as usize]
    } else {
        COSPI_64[index as usize]
    }
}

/// `round(2^14 * sin(index * pi / 64))`, as `cos((32 - index) * pi / 64)`.
const fn sinpi_64(index: i64) -> i32 {
    cospi_64(32 - index)
}

/// `round(2^14 * (2 * sqrt(2) / 3) * sin(index * pi / 9))`, using the
/// period-18 symmetry of the sine to fold onto [`SINPI_9`].
const fn sinpi_9(index: i64) -> i32 {
    let mut index = index % 18;
    if index < 0 {
        index += 18;
    }
    if index > 9 {
        -SINPI_9[(index - 9) as usize]
    } else {
        SINPI_9[index as usize]
    }
}

/// Forward DCT basis entry `b(k, n)` for an `n`-point transform.
const fn dct_entry(points: usize, k: i64, n: i64) -> i32 {
    if k == 0 {
        // c(0) = 1/sqrt(2), which at this scale is exactly COSPI_16_64.
        COSPI_64[16]
    } else {
        cospi_64((2 * n + 1) * k * (32 / points as i64))
    }
}

/// Forward ADST basis entry `b(k, n)` for an `n`-point transform.
const fn adst_entry(points: usize, k: i64, n: i64) -> i32 {
    if points == 4 {
        sinpi_9((2 * k + 1) * (n + 1))
    } else {
        sinpi_64((2 * k + 1) * (2 * n + 1) * (16 / points as i64))
    }
}

/// Defines one row-major `points x points` basis table, `table[k * n + i]`.
macro_rules! basis_table {
    ($name:ident, $points:literal, $entry:ident) => {
        pub(crate) const $name: [i32; $points * $points] = {
            let mut table = [0i32; $points * $points];
            let mut k = 0;
            while k < $points {
                let mut i = 0;
                while i < $points {
                    table[k * $points + i] = $entry($points, k as i64, i as i64);
                    i += 1;
                }
                k += 1;
            }
            table
        };
    };
}

basis_table!(FDCT4, 4, dct_entry);
basis_table!(FDCT8, 8, dct_entry);
basis_table!(FDCT16, 16, dct_entry);
basis_table!(FDCT32, 32, dct_entry);
basis_table!(FADST4, 4, adst_entry);
basis_table!(FADST8, 8, adst_entry);
basis_table!(FADST16, 16, adst_entry);

/// The basis table for one 1-D forward kernel, or `None` at a size that
/// kernel is not defined for (ADST above 16 points).
pub(crate) fn basis(kind: Tx1d, points: usize) -> Option<&'static [i32]> {
    Some(match (kind, points) {
        (Tx1d::Dct, 4) => &FDCT4,
        (Tx1d::Dct, 8) => &FDCT8,
        (Tx1d::Dct, 16) => &FDCT16,
        (Tx1d::Dct, 32) => &FDCT32,
        (Tx1d::Adst, 4) => &FADST4,
        (Tx1d::Adst, 8) => &FADST8,
        (Tx1d::Adst, 16) => &FADST16,
        _ => return None,
    })
}

/// Left shift applied to the residual before the row pass. See the scale
/// table in the module docs.
pub(crate) const fn pre_shift(size: usize) -> u32 {
    match size {
        4 => 2,
        8 => 1,
        _ => 0,
    }
}

/// Rounding right shift applied between the row and column passes. See the
/// scale table in the module docs.
pub(crate) const fn mid_shift(size: usize) -> u32 {
    if size == 32 { 2 } else { 0 }
}

/// Upper bound on how much one forward 1-D pass can grow a value:
/// `max_k sum_n |b(k, n)|`, rounded up over both the DCT and the ADST of that
/// size. The DCT's `k == 0` row is the widest at every size (`N / sqrt(2)`).
const fn pass_gain(size: usize) -> i32 {
    match size {
        4 => 3,
        8 => 6,
        16 => 12,
        _ => 23,
    }
}

/// Largest residual magnitude for which every `i32` intermediate of the
/// vectorized forward transform stays in range, in the style of
/// `av1_simd::transforms::input_limit`.
///
/// The row pass multiplies the pre-shifted input by at most `pass_gain`, the
/// mid shift divides it back down, and the column pass multiplies by
/// `pass_gain` again; the spare factor of four covers the rounding slack the
/// split-accumulator dot product adds to its high half. That leaves roughly
/// 2^23 at 4 points through 2^21 at 32 points - four orders of magnitude
/// above the +/-255 an 8-bit residual can reach - so a real encode never
/// falls back, and the guard only exists so the two paths cannot disagree.
pub(crate) fn input_limit(size: usize) -> i32 {
    let gain = pass_gain(size);
    ((i32::MAX / 4) >> pre_shift(size)) / ((gain * gain) >> mid_shift(size))
}

/// Largest magnitude the row pass may leave behind for the column pass to
/// stay in range. Checked at run time so the bound above is verified rather
/// than merely argued.
pub(crate) fn staged_limit(size: usize) -> i32 {
    (i32::MAX / 4) / pass_gain(size)
}

/// The inverse kernels' `dct_round_shift`.
fn round_shift14(value: i64) -> i64 {
    (value + (1 << 13)) >> 14
}

/// One forward 1-D pass over `input`, writing `size` coefficients.
fn forward_1d(kind: Tx1d, input: &[i64], output: &mut [i64]) {
    let points = input.len();
    let Some(table) = basis(kind, points) else {
        output.copy_from_slice(input);
        return;
    };
    for (k, coefficient) in output.iter_mut().enumerate() {
        let row = &table[k * points..(k + 1) * points];
        let sum: i64 = row
            .iter()
            .zip(input.iter())
            .map(|(&b, &x)| i64::from(b) * x)
            .sum();
        *coefficient = round_shift14(sum);
    }
}

/// Applies the non-lossless forward transform for one `size x size` block,
/// returning row-major coefficients ready for quantization.
///
/// `size` must be 4, 8, 16, or 32, and `residual` is that many samples in
/// row-major order. The ADST kernels are only defined for 4, 8, and 16
/// points, so a 32-point block runs the DCT along both axes regardless of
/// `tx_type`, matching [`crate::av1_intra::inverse_transform`].
///
/// This is the exact counterpart of that inverse: with unit quantizers,
/// `inverse_transform(&forward_transform(r, n, t), n, t, 1, 1)` reproduces
/// `r` up to the kernels' rounding error.
#[must_use]
pub fn forward_transform(residual: &[i32], size: usize, tx_type: Av1TxType) -> Vec<i32> {
    debug_assert_eq!(residual.len(), size * size);
    debug_assert!(matches!(size, 4 | 8 | 16 | 32));
    if tx_type == Av1TxType::Idtx {
        // The 2-D identity has no butterfly pass and no scaling, mirroring
        // the inverse identity's short path.
        return residual.to_vec();
    }

    let (mut column, mut row, lr_flip, ud_flip) = tx_type.kernels();
    if size == 32 {
        column = Tx1d::Dct;
        row = Tx1d::Dct;
    }

    // The vectorized kernels work in 32-bit lanes and decline residuals whose
    // magnitudes could overflow one, so an input the guard rejects simply
    // falls through to the scalar passes below.
    let mut output = vec![0i32; size * size];
    if crate::av1_simd::forward_transform(
        crate::av1_simd::active_isa(),
        residual,
        size,
        column,
        row,
        lr_flip,
        ud_flip,
        &mut output,
    ) {
        return output;
    }

    // `FLIPADST` reverses the inverse transform's *output* along an axis, so
    // the forward transform reverses its input along the same axis.
    let pre = pre_shift(size);
    let mut source = vec![0i64; size * size];
    for target_row in 0..size {
        let source_row = if ud_flip {
            size - 1 - target_row
        } else {
            target_row
        };
        for target_column in 0..size {
            let source_column = if lr_flip {
                size - 1 - target_column
            } else {
                target_column
            };
            source[target_row * size + target_column] =
                i64::from(residual[source_row * size + source_column]) << pre;
        }
    }

    let mut staged = vec![0i64; size * size];
    let mut scratch = vec![0i64; size];
    for index in 0..size {
        let start = index * size;
        forward_1d(row, &source[start..start + size], &mut scratch);
        staged[start..start + size].copy_from_slice(&scratch);
    }
    let mid = mid_shift(size);
    for value in &mut staged {
        *value = (*value + ((1 << mid) >> 1)) >> mid;
    }
    for column_index in 0..size {
        let values: Vec<i64> = (0..size).map(|r| staged[r * size + column_index]).collect();
        forward_1d(column, &values, &mut scratch);
        for (row_index, &value) in scratch.iter().enumerate() {
            output[row_index * size + column_index] =
                value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        }
    }
    output
}
