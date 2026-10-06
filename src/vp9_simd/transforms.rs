//! The two-dimensional inverse transforms and add-to-prediction of
//! `vp9_dec::recon::inverse_transform_add`, with one row or column per lane.
//!
//! The row pass runs a group of rows at once, one per lane, so it reads its
//! input down the columns; the column pass runs a group of columns at once
//! and reads the row pass's output along the rows. Both passes call one
//! [`transform`] site, so each 1-D transform is inlined once per instruction
//! set rather than once per pass.

use super::idct1d::{iadst4, iadst8, iadst16, idct4, idct8, idct16, idct32};
use super::wide::W;
use crate::av1_simd::vector::{I32x, MAX_LANES};
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
pub(super) unsafe fn inverse_transform_add<V: I32x>(
    coefficients: &[i32],
    dest: &mut [u8],
    stride: usize,
    tx_size: u8,
    tx_type: u8,
    eob: usize,
    lossless: bool,
) -> bool {
    unsafe {
        let (n, shift): (usize, i32) = match tx_size {
            _ if lossless => (4, 0),
            0 => (4, 4),
            1 => (8, 5),
            2 => (16, 6),
            _ => (32, 6),
        };
        // Every row and column group is a whole vector; the dispatcher
        // narrows AVX2 to SSE4.1 for the 4x4 sizes so this never fails there.
        if n < V::LANES {
            return false;
        }
        if lossless {
            iwht4x4_add::<V>(coefficients, dest, stride, eob);
            return true;
        }
        if tx_type == DCT_DCT || tx_size == 3 {
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
            return inverse_2d::<V>(coefficients, dest, stride, n, rows, false, false, shift);
        }
        // `tx_type` names the vertical (column) transform first.
        let (column_adst, row_adst) = match tx_type {
            ADST_DCT => (true, false),
            DCT_ADST => (false, true),
            _ => (true, true),
        };
        inverse_2d::<V>(
            coefficients,
            dest,
            stride,
            n,
            n,
            row_adst,
            column_adst,
            shift,
        )
    }
}

/// The 1-D transform of size `n`, on `n` vectors of independent lanes.
#[inline(always)]
unsafe fn transform<V: I32x>(n: usize, adst: bool, input: &[W<V>], output: &mut [W<V>]) {
    unsafe {
        match (n, adst) {
            (4, false) => idct4(input, output),
            (4, true) => iadst4(input, output),
            (8, false) => idct8(input, output),
            (8, true) => iadst8(input, output),
            (16, false) => idct16(input, output),
            (16, true) => iadst16(input, output),
            _ => idct32(input, output),
        }
    }
}

/// `inverse_2d` of the scalar reconstruction: transforms the first `rows`
/// rows of the `n`x`n` `input` (the rest are zero), then every column, and
/// adds `ROUND_POWER_OF_TWO(result, shift)` to `dest`.
#[inline(always)]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
unsafe fn inverse_2d<V: I32x>(
    input: &[i32],
    dest: &mut [u8],
    stride: usize,
    n: usize,
    rows: usize,
    row_adst: bool,
    column_adst: bool,
    shift: i32,
) -> bool {
    unsafe {
        if row_adst && !within(&input[..rows * n], adst_limit(n)) {
            return false;
        }
        let lanes = V::LANES;
        let mut middle = [0i32; 32 * 32];
        let mut vectors_in = [W::<V>::zero(); 32];
        let mut vectors_out = [W::<V>::zero(); 32];
        let mut scratch = [0i32; MAX_LANES];

        // Two passes through one `transform` call: the rows of `input` into
        // `middle`, then the columns of `middle` into `dest`.
        for pass in 0..2 {
            let (adst, groups) = if pass == 0 {
                (row_adst, rows.div_ceil(lanes))
            } else {
                if column_adst && !within(&middle[..n * n], adst_limit(n)) {
                    return false;
                }
                (column_adst, n / lanes)
            };
            for group in 0..groups {
                let first = group * lanes;
                for (k, vector) in vectors_in.iter_mut().enumerate().take(n) {
                    *vector = if pass == 0 {
                        // Lane `j` is row `first + j`, read down column `k`.
                        for (lane, slot) in scratch.iter_mut().enumerate().take(lanes) {
                            let row = first + lane;
                            *slot = if row < rows { input[row * n + k] } else { 0 };
                        }
                        W(V::load(&scratch))
                    } else {
                        // Lane `j` is column `first + j`, read along row `k`.
                        W(V::load(&middle[k * n + first..]))
                    };
                }
                transform(n, adst, &vectors_in[..n], &mut vectors_out[..n]);
                if pass == 0 {
                    for (k, vector) in vectors_out.iter().enumerate().take(n) {
                        vector.0.store(&mut scratch);
                        for (lane, &value) in scratch.iter().enumerate().take(lanes) {
                            let row = first + lane;
                            if row < n {
                                middle[row * n + k] = value;
                            }
                        }
                    }
                } else {
                    let round = V::splat(1 << (shift - 1));
                    for (y, vector) in vectors_out.iter().enumerate().take(n) {
                        let at = y * stride + first;
                        let residual = vector.0.add(round).sra_var(shift);
                        V::load_u8(&dest[at..])
                            .add(residual)
                            .store_u8_clamped(&mut dest[at..]);
                    }
                }
            }
        }
        true
    }
}

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

/// One `vpx_iwht4x4_16_add` butterfly on four vectors, after the given
/// input shift.
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
/// DC-only form. `V` has four lanes: callers narrow AVX2 to SSE4.1.
#[inline(always)]
unsafe fn iwht4x4_add<V: I32x>(input: &[i32], dest: &mut [u8], stride: usize, eob: usize) {
    unsafe {
        let mut rows = [0i32; 16];
        if eob <= 1 {
            let mut a1 = input[0] >> 2;
            let e1 = a1 >> 1;
            a1 -= e1;
            let tmp = [a1, e1, e1, e1];
            let mut first = [0i32; MAX_LANES];
            let mut rest = [0i32; MAX_LANES];
            for x in 0..4 {
                rest[x] = tmp[x] >> 1;
                first[x] = tmp[x] - rest[x];
            }
            rows[..4].copy_from_slice(&first[..4]);
            for y in 1..4 {
                rows[y * 4..y * 4 + 4].copy_from_slice(&rest[..4]);
            }
        } else {
            // Row pass, one row per lane: tap `k` is column `k` of every row.
            let mut scratch = [0i32; MAX_LANES];
            let taps: [V; 4] = core::array::from_fn(|k| {
                for (row, slot) in scratch.iter_mut().enumerate().take(4) {
                    *slot = input[row * 4 + k];
                }
                V::load(&scratch).sra::<2>()
            });
            let columns = wht_butterfly(taps);
            for (k, column) in columns.into_iter().enumerate() {
                column.store(&mut scratch);
                for row in 0..4 {
                    rows[row * 4 + k] = scratch[row];
                }
            }
            // Column pass, one column per lane: tap `y` is row `y`.
            let taps: [V; 4] = core::array::from_fn(|y| V::load(&rows[y * 4..]));
            for (y, row) in wht_butterfly(taps).into_iter().enumerate() {
                row.store(&mut scratch);
                rows[y * 4..y * 4 + 4].copy_from_slice(&scratch[..4]);
            }
        }
        for y in 0..4 {
            let at = y * stride;
            V::load_u8(&dest[at..])
                .add(V::load(&rows[y * 4..]))
                .store_u8_clamped(&mut dest[at..]);
        }
    }
}
