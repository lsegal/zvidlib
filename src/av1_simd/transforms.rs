//! Vectorized AV1 transform butterfly passes.
//!
//! Covers the lossless 4x4 Walsh-Hadamard transform in both directions
//! ([`crate::av1_encoder`]) and the 4- and 8-point inverse DCT used by the
//! non-lossless reconstruct ([`crate::av1_intra::inverse_transform`]).
//!
//! Both scalar references compute in `i64`. These kernels compute in `i32`
//! lanes, which is only bit-exact while no intermediate overflows, so each
//! dispatcher first checks the pass input against a conservative magnitude
//! bound (see [`WHT_INPUT_LIMIT`] and [`DCT_INPUT_LIMIT`]) and falls back to
//! the scalar path otherwise. Real bitstream coefficients sit orders of
//! magnitude below those bounds; the guard exists so a hostile or malformed
//! stream cannot make the two paths disagree.
//!
//! Data flow is the standard separable shape: load rows, transpose, run the
//! first pass with one lane per row, transpose, run the second pass, store
//! rows. An 8x8 transpose is assembled from four 4x4 transposes of its
//! quadrants.

use super::vector::{I32x, Transpose4};

/// Largest input magnitude for which the `i32` Walsh-Hadamard butterflies
/// cannot overflow. The transform grows a value by at most a factor of 4 per
/// pass over two passes plus the `* 4` dequant, so `2^18` leaves 2^26 headroom.
pub(crate) const WHT_INPUT_LIMIT: i32 = 1 << 18;

/// Largest input magnitude for which the `i32` inverse-DCT butterflies cannot
/// overflow. The widest intermediate is `(step2[6] - step2[5]) * COSPI_16_64`
/// in the 8-point transform, whose operands reach about `2.8x` the pass input;
/// `30000 * 2.8 * 11585` stays comfortably inside `i32`.
pub(crate) const DCT_INPUT_LIMIT: i32 = 30_000;

const COSPI_4_64: i32 = 16069;
const COSPI_8_64: i32 = 15137;
const COSPI_12_64: i32 = 13623;
const COSPI_16_64: i32 = 11585;
const COSPI_20_64: i32 = 9102;
const COSPI_24_64: i32 = 6270;
const COSPI_28_64: i32 = 3196;

/// True when every value is small enough for the `i32` kernels to stay exact.
pub(crate) fn within_limit(values: &[i32], limit: i32) -> bool {
    let limit = limit as u32;
    values.iter().all(|&v| v.unsigned_abs() <= limit)
}

// ---------------------------------------------------------------------
// Butterflies (one lane per independent row/column)
// ---------------------------------------------------------------------

#[inline(always)]
unsafe fn round_shift14<V: I32x>(value: V) -> V {
    unsafe { value.add(V::splat(1 << 13)).sra::<14>() }
}

/// Vector form of `av1_intra::inverse_dct4_1d`.
#[inline(always)]
unsafe fn dct4<V: I32x>(input: [V; 4]) -> [V; 4] {
    unsafe {
        let c16 = V::splat(COSPI_16_64);
        let step0 = round_shift14(input[0].add(input[2]).mul(c16));
        let step1 = round_shift14(input[0].sub(input[2]).mul(c16));
        let step2 = round_shift14(
            input[1]
                .mul(V::splat(COSPI_24_64))
                .sub(input[3].mul(V::splat(COSPI_8_64))),
        );
        let step3 = round_shift14(
            input[1]
                .mul(V::splat(COSPI_8_64))
                .add(input[3].mul(V::splat(COSPI_24_64))),
        );
        [
            step0.add(step3),
            step1.add(step2),
            step1.sub(step2),
            step0.sub(step3),
        ]
    }
}

/// Vector form of `av1_intra::inverse_dct8_1d`.
#[inline(always)]
unsafe fn dct8<V: I32x>(input: [V; 8]) -> [V; 8] {
    unsafe {
        let c16 = V::splat(COSPI_16_64);
        let s1_0 = input[0];
        let s1_1 = input[2];
        let s1_2 = input[4];
        let s1_3 = input[6];
        let s1_4 = round_shift14(
            input[1]
                .mul(V::splat(COSPI_28_64))
                .sub(input[7].mul(V::splat(COSPI_4_64))),
        );
        let s1_7 = round_shift14(
            input[1]
                .mul(V::splat(COSPI_4_64))
                .add(input[7].mul(V::splat(COSPI_28_64))),
        );
        let s1_5 = round_shift14(
            input[5]
                .mul(V::splat(COSPI_12_64))
                .sub(input[3].mul(V::splat(COSPI_20_64))),
        );
        let s1_6 = round_shift14(
            input[5]
                .mul(V::splat(COSPI_20_64))
                .add(input[3].mul(V::splat(COSPI_12_64))),
        );

        let s2_0 = round_shift14(s1_0.add(s1_2).mul(c16));
        let s2_1 = round_shift14(s1_0.sub(s1_2).mul(c16));
        let s2_2 = round_shift14(
            s1_1.mul(V::splat(COSPI_24_64))
                .sub(s1_3.mul(V::splat(COSPI_8_64))),
        );
        let s2_3 = round_shift14(
            s1_1.mul(V::splat(COSPI_8_64))
                .add(s1_3.mul(V::splat(COSPI_24_64))),
        );
        let s2_4 = s1_4.add(s1_5);
        let s2_5 = s1_4.sub(s1_5);
        let s2_6 = s1_7.sub(s1_6);
        let s2_7 = s1_6.add(s1_7);

        let s3_0 = s2_0.add(s2_3);
        let s3_1 = s2_1.add(s2_2);
        let s3_2 = s2_1.sub(s2_2);
        let s3_3 = s2_0.sub(s2_3);
        let s3_4 = s2_4;
        let s3_5 = round_shift14(s2_6.sub(s2_5).mul(c16));
        let s3_6 = round_shift14(s2_5.add(s2_6).mul(c16));
        let s3_7 = s2_7;

        [
            s3_0.add(s3_7),
            s3_1.add(s3_6),
            s3_2.add(s3_5),
            s3_3.add(s3_4),
            s3_3.sub(s3_4),
            s3_2.sub(s3_5),
            s3_1.sub(s3_6),
            s3_0.sub(s3_7),
        ]
    }
}

/// Vector form of `av1_encoder::wht::iwht_1d`.
#[inline(always)]
unsafe fn iwht<V: I32x, const SHIFT: i32>(t: [V; 4]) -> [V; 4] {
    unsafe {
        let mut a = t[0].sra::<SHIFT>();
        let mut c = t[1].sra::<SHIFT>();
        let mut d = t[2].sra::<SHIFT>();
        let mut b = t[3].sra::<SHIFT>();
        a = a.add(c);
        d = d.sub(b);
        let e = a.sub(d).sra::<1>();
        b = e.sub(b);
        c = e.sub(c);
        a = a.sub(b);
        d = d.add(c);
        [a, b, c, d]
    }
}

/// Vector form of `av1_encoder::wht::iwht_1d_inverse`.
#[inline(always)]
unsafe fn iwht_inverse<V: I32x>(o: [V; 4]) -> [V; 4] {
    unsafe {
        let a1 = o[0].add(o[1]);
        let d1 = o[3].sub(o[2]);
        let e = a1.sub(d1).sra::<1>();
        let in3 = e.sub(o[1]);
        let in1 = e.sub(o[2]);
        let in0 = a1.sub(in1);
        let in2 = d1.add(in3);
        [in0, in1, in2, in3]
    }
}

// ---------------------------------------------------------------------
// Block drivers
// ---------------------------------------------------------------------

#[inline(always)]
unsafe fn load_rows4<V: I32x + Transpose4>(block: &[i32]) -> [V; 4] {
    unsafe {
        [
            V::load(&block[0..]),
            V::load(&block[4..]),
            V::load(&block[8..]),
            V::load(&block[12..]),
        ]
    }
}

#[inline(always)]
unsafe fn store_rows4<V: I32x>(rows: [V; 4], block: &mut [i32]) {
    unsafe {
        for (index, row) in rows.into_iter().enumerate() {
            row.store(&mut block[index * 4..]);
        }
    }
}

/// Bit-exact vector form of `av1_encoder::wht::iwht4x4`.
///
/// # Safety
/// The caller must have verified `V`'s instruction set is available, and every
/// entry of `quant` must satisfy `|q| <= WHT_INPUT_LIMIT`.
pub(crate) unsafe fn iwht4x4<V: I32x + Transpose4>(quant: &[i32; 16]) -> [i32; 16] {
    unsafe {
        let rows = load_rows4::<V>(quant);
        // Lane j of `columns[k]` is element k of row j: one lane per row, so the
        // row pass below transforms all four rows at once.
        let columns = V::transpose4(rows);
        let four = V::splat(4);
        let scaled = [
            columns[0].mul(four),
            columns[1].mul(four),
            columns[2].mul(four),
            columns[3].mul(four),
        ];
        let row_pass = iwht::<V, 2>(scaled);
        // Transposing back leaves one lane per column, which is what the column
        // pass needs.
        let staged = V::transpose4(row_pass);
        let column_pass = iwht::<V, 0>(staged);
        let mut out = [0i32; 16];
        store_rows4(column_pass, &mut out);
        out
    }
}

/// Bit-exact vector form of `av1_encoder::wht::fwht4x4`.
///
/// # Safety
/// The caller must have verified `V`'s instruction set is available, and every
/// entry of `residual` must satisfy `|r| <= WHT_INPUT_LIMIT`.
pub(crate) unsafe fn fwht4x4<V: I32x + Transpose4>(residual: &[i32; 16]) -> [i32; 16] {
    unsafe {
        // The forward transform undoes the decoder's column pass first, and the
        // loaded rows already carry one lane per column.
        let rows = load_rows4::<V>(residual);
        let column_pass = iwht_inverse(rows);
        let staged = V::transpose4(column_pass);
        let row_pass = iwht_inverse(staged);
        let out_rows = V::transpose4(row_pass);
        let mut out = [0i32; 16];
        store_rows4(out_rows, &mut out);
        out
    }
}

#[inline(always)]
unsafe fn store_i16_clamped<V: I32x>(row: V, out: &mut [i16]) {
    unsafe {
        let mut scratch = [0i32; super::vector::MAX_LANES];
        row.clamp(V::splat(i16::MIN as i32), V::splat(i16::MAX as i32))
            .store(&mut scratch);
        for (slot, &value) in out.iter_mut().zip(scratch.iter()) {
            *slot = value as i16;
        }
    }
}

/// Bit-exact vector form of the `DCT_DCT` branch of
/// `av1_intra::inverse_transform` for a 4x4 block.
///
/// # Safety
/// The caller must have verified `V`'s instruction set is available and that
/// `dequantized` is within [`DCT_INPUT_LIMIT`].
pub(crate) unsafe fn inverse_dct4x4<V: I32x + Transpose4>(
    dequantized: &[i32],
    out: &mut [i16],
) -> bool {
    unsafe {
        let rows = load_rows4::<V>(dequantized);
        let row_pass = dct4(V::transpose4(rows));
        let mut staged = [0i32; 16];
        store_rows4(V::transpose4(row_pass), &mut staged);
        if !within_limit(&staged, DCT_INPUT_LIMIT) {
            return false;
        }
        let column_pass = dct4(load_rows4::<V>(&staged));
        let bias = V::splat(1 << 3);
        for (index, row) in column_pass.into_iter().enumerate() {
            store_i16_clamped(row.add(bias).sra::<4>(), &mut out[index * 4..index * 4 + 4]);
        }
        true
    }
}

/// Transposes an 8x8 block held as `rows[j][h]`, the `h`-th four columns of row
/// `j`, into `t[c][g]`, the `c`-th column over rows `4g..4g+4`.
#[inline(always)]
unsafe fn transpose8<V: I32x + Transpose4>(rows: [[V; 2]; 8]) -> [[V; 2]; 8] {
    unsafe {
        let mut out = [[V::zero(); 2]; 8];
        for g in 0..2 {
            for h in 0..2 {
                let quadrant = V::transpose4([
                    rows[4 * g][h],
                    rows[4 * g + 1][h],
                    rows[4 * g + 2][h],
                    rows[4 * g + 3][h],
                ]);
                for (k, value) in quadrant.into_iter().enumerate() {
                    out[4 * h + k][g] = value;
                }
            }
        }
        out
    }
}

#[inline(always)]
unsafe fn dct8_pass<V: I32x>(block: [[V; 2]; 8]) -> [[V; 2]; 8] {
    unsafe {
        let mut out = [[V::zero(); 2]; 8];
        for g in 0..2 {
            let lanes = dct8([
                block[0][g],
                block[1][g],
                block[2][g],
                block[3][g],
                block[4][g],
                block[5][g],
                block[6][g],
                block[7][g],
            ]);
            for (index, value) in lanes.into_iter().enumerate() {
                out[index][g] = value;
            }
        }
        out
    }
}

/// Bit-exact vector form of the `DCT_DCT` branch of
/// `av1_intra::inverse_transform` for an 8x8 block.
///
/// # Safety
/// The caller must have verified `V`'s instruction set is available and that
/// `dequantized` is within [`DCT_INPUT_LIMIT`].
pub(crate) unsafe fn inverse_dct8x8<V: I32x + Transpose4>(
    dequantized: &[i32],
    out: &mut [i16],
) -> bool {
    unsafe {
        let mut rows = [[V::zero(); 2]; 8];
        for (j, row) in rows.iter_mut().enumerate() {
            row[0] = V::load(&dequantized[j * 8..]);
            row[1] = V::load(&dequantized[j * 8 + 4..]);
        }
        let row_pass = dct8_pass(transpose8(rows));
        let staged_vectors = transpose8(row_pass);
        let mut staged = [0i32; 64];
        for (j, row) in staged_vectors.iter().enumerate() {
            row[0].store(&mut staged[j * 8..]);
            row[1].store(&mut staged[j * 8 + 4..]);
        }
        if !within_limit(&staged, DCT_INPUT_LIMIT) {
            return false;
        }
        let column_pass = dct8_pass(staged_vectors);
        let bias = V::splat(1 << 4);
        for (j, row) in column_pass.into_iter().enumerate() {
            store_i16_clamped(row[0].add(bias).sra::<5>(), &mut out[j * 8..j * 8 + 4]);
            store_i16_clamped(row[1].add(bias).sra::<5>(), &mut out[j * 8 + 4..j * 8 + 8]);
        }
        true
    }
}
