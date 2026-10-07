//! The two-dimensional inverse transforms and add-to-prediction of
//! `vp9_dec::recon::inverse_transform_add`, with one row or column per lane.
//!
//! The data flow is the AV1 transforms' (`av1_simd::transforms`): each pass
//! gathers four rows into one lane apiece with 4x4 register transposes, runs
//! the 1-D transform, and writes its output transposed, so the row pass
//! leaves the intermediate's columns in rows for the column pass, which adds
//! its output straight to the prediction. The transposes are four lanes wide,
//! so AVX2 hosts run these through the SSE4.1 instantiation, as they do the
//! AV1 transforms.

// Loops over vectors index rather than iterate: an iterator adapter or an
// `array::from_fn` closure over vector values is a separate function the
// inliner can leave outside the `#[target_feature]` wrapper, compiled at the
// baseline instruction set (#341).
#![allow(clippy::needless_range_loop)]

use super::idct1d::{iadst4, iadst8, iadst16, idct4, idct8, idct16, idct32};
use super::wide::W;
use crate::av1_simd::vector::{I32x, Transpose4};
use crate::vp9_dec::recon::{ADST_DCT, DCT_ADST, DCT_DCT};

/// The largest input magnitude for which no intermediate value of
/// `iadst4` leaves `i32`, so 32-bit lanes reproduce the scalar `i64`
/// arithmetic exactly. Derived by bounding every intermediate by the sum of
/// the absolute values of its terms, stage by stage; the DCTs need no limit
/// because they truncate every stage to 16 bits.
pub(crate) const ADST4_INPUT_LIMIT: u32 = 28_932;
/// [`ADST4_INPUT_LIMIT`] for `iadst8`.
pub(crate) const ADST8_INPUT_LIMIT: u32 = 13_905;
/// [`ADST4_INPUT_LIMIT`] for `iadst16`.
pub(crate) const ADST16_INPUT_LIMIT: u32 = 5_429;

/// Whether the vector kernel can take an `n`x`n` block: the coefficients
/// are all there and every destination row it loads and stores is in
/// bounds.
pub(super) fn covers(coefficients: &[i32], dest: &[u8], stride: usize, n: usize) -> bool {
    coefficients.len() >= n * n && stride >= n && dest.len() >= (n - 1) * stride + n
}

fn adst_limit(n: usize) -> u32 {
    match n {
        4 => ADST4_INPUT_LIMIT,
        8 => ADST8_INPUT_LIMIT,
        _ => ADST16_INPUT_LIMIT,
    }
}

fn within(values: &[i32], limit: u32) -> bool {
    values.iter().all(|value| value.unsigned_abs() <= limit)
}

/// The vector form of `vp9_dec::recon::inverse_transform_add`. Returns
/// `false`, with `dest` untouched, when an ADST input exceeds its limit.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn inverse_transform_add<V: I32x + Transpose4>(
    coefficients: &[i32],
    dest: &mut [u8],
    stride: usize,
    tx_size: u8,
    tx_type: u8,
    eob: usize,
    lossless: bool,
) -> bool {
    unsafe {
        if lossless {
            iwht4x4_add::<V>(coefficients, dest, stride, eob);
            return true;
        }
        let (n, shift): (usize, i32) = match tx_size {
            0 => (4, 4),
            1 => (8, 5),
            2 => (16, 6),
            _ => (32, 6),
        };
        let (rows, row_adst, column_adst) = if tx_type == DCT_DCT || tx_size == 3 {
            if eob == 1 {
                dc_only::<V>(coefficients[0], dest, stride, n, shift);
                return true;
            }
            let rows = match tx_size {
                0 => 4,
                1 if eob <= 12 => 4,
                1 => 8,
                2 if eob <= 10 => 4,
                2 if eob <= 38 => 8,
                2 => 16,
                _ if eob <= 34 => 8,
                _ if eob <= 135 => 16,
                _ => 32,
            };
            (rows, false, false)
        } else {
            // `tx_type` names the vertical (column) transform first.
            match tx_type {
                ADST_DCT => (n, false, true),
                DCT_ADST => (n, true, false),
                _ => (n, true, true),
            }
        };
        let block = Block {
            input: coefficients,
            dest,
            stride,
            rows,
            row_adst,
            column_adst,
            shift,
        };
        match n {
            4 => inverse_2d_4::<V>(block),
            8 => inverse_2d_8::<V>(block),
            16 => inverse_2d_16::<V>(block),
            _ => inverse_2d_32::<V>(block),
        }
    }
}

/// One block for an `inverse_2d_*` driver.
struct Block<'a> {
    input: &'a [i32],
    dest: &'a mut [u8],
    stride: usize,
    /// How many leading rows of `input` the end-of-block position leaves
    /// nonzero; the rest are treated as zero, as the scalar code does.
    rows: usize,
    row_adst: bool,
    column_adst: bool,
    /// The final `ROUND_POWER_OF_TWO` shift.
    shift: i32,
}

/// Gathers rows `4 * group..4 * group + 4` of the `n`-wide row-major `src`
/// into `lanes`, one row per lane: lane `j` of `lanes[k]` is element `k` of
/// row `4 * group + j`.
#[inline(always)]
unsafe fn gather_rows<V: I32x + Transpose4>(
    src: &[i32],
    n: usize,
    group: usize,
    lanes: &mut [W<V>],
) {
    unsafe {
        let base = 4 * group * n;
        for quad in 0..n / 4 {
            let at = base + 4 * quad;
            let tile = V::transpose4([
                V::load(&src[at..]),
                V::load(&src[at + n..]),
                V::load(&src[at + 2 * n..]),
                V::load(&src[at + 3 * n..]),
            ]);
            for j in 0..4 {
                lanes[4 * quad + j] = W(tile[j]);
            }
        }
    }
}

/// Defines the separable `N`x`N` inverse transform driver for one size:
/// the first `rows` rows of the input through the row transform, then every
/// column through the column transform, and `ROUND_POWER_OF_TWO(result,
/// shift)` added to `dest`, as the scalar `inverse_2d` does.
macro_rules! inverse_2d {
    ($name:ident, $n:literal, $dct:ident, $adst:ident) => {
        #[inline(always)]
        unsafe fn $name<V: I32x + Transpose4>(block: Block<'_>) -> bool {
            unsafe {
                let Block {
                    input,
                    dest,
                    stride,
                    rows,
                    row_adst,
                    column_adst,
                    shift,
                } = block;
                if row_adst && !within(&input[..rows * $n], adst_limit($n)) {
                    return false;
                }
                let mut lanes = [W::<V>::zero(); $n];
                let mut out = [W::<V>::zero(); $n];

                // Row pass, four rows per iteration. `middle` holds the
                // output transposed: its row `k` is column `k` of the
                // intermediate block, whose rows past `rows` stay zero.
                let mut middle = [0i32; $n * $n];
                for group in 0..rows / 4 {
                    gather_rows(input, $n, group, &mut lanes);
                    if row_adst {
                        $adst(&lanes, &mut out);
                    } else {
                        $dct(&lanes, &mut out);
                    }
                    for k in 0..$n {
                        out[k].0.store(&mut middle[k * $n + 4 * group..]);
                    }
                }
                if column_adst && !within(&middle, adst_limit($n)) {
                    return false;
                }

                // Column pass, four columns per iteration, each output row
                // added straight to the prediction.
                let round = V::splat(1 << (shift - 1));
                for group in 0..$n / 4 {
                    gather_rows(&middle, $n, group, &mut lanes);
                    if column_adst {
                        $adst(&lanes, &mut out);
                    } else {
                        $dct(&lanes, &mut out);
                    }
                    for y in 0..$n {
                        let at = y * stride + 4 * group;
                        let residual = out[y].0.add(round).sra_var(shift);
                        V::load_u8(&dest[at..])
                            .add(residual)
                            .store_u8_clamped(&mut dest[at..]);
                    }
                }
                true
            }
        }
    };
}

inverse_2d!(inverse_2d_4, 4, idct4, iadst4);
inverse_2d!(inverse_2d_8, 8, idct8, iadst8);
inverse_2d!(inverse_2d_16, 16, idct16, iadst16);
// VP9 has no 32-point ADST, and the entry point never asks for one.
inverse_2d!(inverse_2d_32, 32, idct32, idct32);

/// The DC-only shortcut every DCT size takes when `eob == 1`.
#[inline(always)]
unsafe fn dc_only<V: I32x>(dc: i32, dest: &mut [u8], stride: usize, n: usize, shift: i32) {
    unsafe {
        let round = |x: i64| (x + (1 << 13)) >> 14;
        let out = round(i64::from(dc as i16) * 11585) as i32;
        let out = round(i64::from(out) * 11585) as i32;
        let a1 = V::splat((out + (1 << (shift - 1))) >> shift);
        for y in 0..n {
            for x in (0..n).step_by(V::LANES) {
                let at = y * stride + x;
                V::load_u8(&dest[at..])
                    .add(a1)
                    .store_u8_clamped(&mut dest[at..]);
            }
        }
    }
}

/// One `vpx_iwht4x4_16_add` butterfly on four vectors.
#[inline(always)]
unsafe fn wht_butterfly<V: I32x>(input: [V; 4]) -> [V; 4] {
    unsafe {
        let [mut a1, mut c1, mut d1, mut b1] = input;
        a1 = a1.add(c1);
        d1 = d1.sub(b1);
        let e1 = a1.sub(d1).sra::<1>();
        b1 = e1.sub(b1);
        c1 = e1.sub(c1);
        a1 = a1.sub(b1);
        d1 = d1.add(c1);
        [a1, b1, c1, d1]
    }
}

/// The 4x4 inverse Walsh-Hadamard transform of lossless frames, and its
/// DC-only form.
#[inline(always)]
unsafe fn iwht4x4_add<V: I32x + Transpose4>(
    input: &[i32],
    dest: &mut [u8],
    stride: usize,
    eob: usize,
) {
    unsafe {
        let rows = if eob <= 1 {
            let mut a1 = input[0] >> 2;
            let e1 = a1 >> 1;
            a1 -= e1;
            let tmp = [a1, e1, e1, e1];
            let mut first = [0i32; 4];
            let mut rest = [0i32; 4];
            for x in 0..4 {
                rest[x] = tmp[x] >> 1;
                first[x] = tmp[x] - rest[x];
            }
            let rest = V::load(&rest);
            [V::load(&first), rest, rest, rest]
        } else {
            // Row pass, one row per lane: tap `k` is column `k` of every row.
            let mut taps = V::transpose4([
                V::load(&input[0..]),
                V::load(&input[4..]),
                V::load(&input[8..]),
                V::load(&input[12..]),
            ]);
            for k in 0..4 {
                taps[k] = taps[k].sra::<2>();
            }
            // Column pass, one column per lane: tap `y` is row `y`.
            wht_butterfly(V::transpose4(wht_butterfly(taps)))
        };
        for y in 0..4 {
            let at = y * stride;
            V::load_u8(&dest[at..])
                .add(rows[y])
                .store_u8_clamped(&mut dest[at..]);
        }
    }
}
