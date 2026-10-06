//! The intra predictors of `vp9_dec::recon::predict_intra` for 4x4 to
//! 32x32 blocks.
//!
//! Every directional predictor in VP9 is built from a handful of one-
//! dimensional runs of two- and three-tap averages of the edge pixels; each
//! row of the block is then one of those runs, shifted. So the averages are
//! computed with one pixel per lane, and the rows are written as copies of
//! the runs or of earlier rows, following the same recurrences the scalar
//! code (and libvpx's `d*_predictor`) uses. DC sums its edge with vectors and
//! TM computes `left + above - above_left` a vector at a time.

use crate::av1_simd::vector::I32x;
use crate::vp9_dec::recon::{
    D45_PRED, D63_PRED, D117_PRED, D135_PRED, D207_PRED, DC_PRED, H_PRED, TM_PRED, V_PRED,
};

/// Large enough for a 64-entry run, rounded up to whole vectors, plus the
/// two taps an average reads past its last position.
const RUN: usize = 80;

/// `out[i] = avg2(src[i], src[i + 1])` for `i` in `0..count`, rounded up to
/// whole vectors.
#[inline(always)]
unsafe fn avg2_run<V: I32x>(src: &[u8; RUN], count: usize) -> [u8; RUN] {
    unsafe {
        let mut out = [0u8; RUN];
        for i in (0..count).step_by(V::LANES) {
            V::load_u8(&src[i..])
                .add(V::load_u8(&src[i + 1..]))
                .add(V::splat(1))
                .sra::<1>()
                .store_u8_clamped(&mut out[i..]);
        }
        out
    }
}

/// `out[i] = avg3(src[i], src[i + 1], src[i + 2])` for `i` in `0..count`,
/// rounded up to whole vectors.
#[inline(always)]
unsafe fn avg3_run<V: I32x>(src: &[u8; RUN], count: usize) -> [u8; RUN] {
    unsafe {
        let mut out = [0u8; RUN];
        for i in (0..count).step_by(V::LANES) {
            let middle = V::load_u8(&src[i + 1..]);
            V::load_u8(&src[i..])
                .add(middle)
                .add(middle)
                .add(V::load_u8(&src[i + 2..]))
                .add(V::splat(2))
                .sra::<2>()
                .store_u8_clamped(&mut out[i..]);
        }
        out
    }
}

/// The sum of `values[..count]`, `count` a whole number of vectors.
#[inline(always)]
unsafe fn sum<V: I32x>(values: &[u8], count: usize) -> u32 {
    unsafe {
        let mut total = V::zero();
        for i in (0..count).step_by(V::LANES) {
            total = total.add(V::load_u8(&values[i..]));
        }
        total.hsum() as u32
    }
}

fn avg2(a: u8, b: u8) -> u8 {
    ((u32::from(a) + u32::from(b) + 1) >> 1) as u8
}

fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((u32::from(a) + 2 * u32::from(b) + u32::from(c) + 2) >> 2) as u8
}

/// Writes row `y` of the block from `row[..bs]`.
#[inline(always)]
fn put(dest: &mut [u8], stride: usize, bs: usize, y: usize, row: &[u8]) {
    dest[y * stride..y * stride + bs].copy_from_slice(&row[..bs]);
}

/// The vector form of `vp9_dec::recon::predict_intra`: `above[0]` is the
/// above-left pixel and `above[1..=2 * bs]` the row above, `left[..bs]` the
/// column to the left. Returns `false` for a block narrower than a vector.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    clippy::too_many_lines
)]
pub(super) unsafe fn predict_intra<V: I32x>(
    dest: &mut [u8],
    stride: usize,
    bs: usize,
    mode: u8,
    above: &[u8; 65],
    left: &[u8; 32],
    have_left: bool,
    have_above: bool,
) -> bool {
    unsafe {
        if bs < V::LANES {
            return false;
        }
        // Copies padded so that a vector load at any run position stays in
        // bounds. `a[i + 1]` is the scalar code's `above(i)`.
        let mut a = [0u8; RUN];
        a[..65].copy_from_slice(above);
        let mut l = [0u8; RUN];
        l[..32].copy_from_slice(left);
        let top_left = a[0];

        match mode {
            DC_PRED => {
                let value = match (have_left, have_above) {
                    (false, false) => 128,
                    (true, false) => (sum::<V>(&l, bs) + (bs as u32 >> 1)) / bs as u32,
                    (false, true) => (sum::<V>(&a[1..], bs) + (bs as u32 >> 1)) / bs as u32,
                    (true, true) => {
                        let count = 2 * bs as u32;
                        (sum::<V>(&l, bs) + sum::<V>(&a[1..], bs) + (count >> 1)) / count
                    }
                } as u8;
                for y in 0..bs {
                    dest[y * stride..y * stride + bs].fill(value);
                }
            }
            V_PRED => {
                for y in 0..bs {
                    put(dest, stride, bs, y, &a[1..]);
                }
            }
            H_PRED => {
                for y in 0..bs {
                    dest[y * stride..y * stride + bs].fill(l[y]);
                }
            }
            TM_PRED => {
                let top_left = V::splat(i32::from(top_left));
                for y in 0..bs {
                    let side = V::splat(i32::from(l[y])).sub(top_left);
                    for x in (0..bs).step_by(V::LANES) {
                        V::load_u8(&a[1 + x..])
                            .add(side)
                            .store_u8_clamped(&mut dest[y * stride + x..]);
                    }
                }
            }
            D45_PRED => {
                // Position `i` of the run is `avg3(above(i), above(i + 1),
                // above(i + 2))`, and the replicated last above pixel from
                // where that would reach past it.
                let mut run = avg3_run::<V>(&a, 2 * bs);
                run.copy_within(1..2 * bs, 0);
                run[2 * bs - 2..2 * bs].fill(a[2 * bs]);
                for y in 0..bs {
                    put(dest, stride, bs, y, &run[y..]);
                }
            }
            D63_PRED => {
                // `avg2_run(a)[i + 1]` and `avg3_run(a)[i + 1]` are the
                // two- and three-tap averages starting at `above(i)`.
                let two = avg2_run::<V>(&a, 2 * bs);
                let three = avg3_run::<V>(&a, 2 * bs);
                for y in 0..bs {
                    let run = if y & 1 == 1 { &three } else { &two };
                    put(dest, stride, bs, y, &run[y / 2 + 1..]);
                }
            }
            D207_PRED => {
                // libvpx's `d207_predictor`: column 0 is the two-tap and
                // column 1 the three-tap averages down the left edge, and
                // each row repeats the one below it shifted two columns
                // left, so row `r` is the interleaved columns from `2 * r`.
                let mut columns = [avg2_run::<V>(&l, bs), avg3_run::<V>(&l, bs)];
                columns[0][bs - 1] = l[bs - 1];
                columns[1][bs - 2] = avg3(l[bs - 2], l[bs - 1], l[bs - 1]);
                columns[1][bs - 1] = l[bs - 1];
                let mut interleaved = [l[bs - 1]; 3 * 32];
                for r in 0..bs {
                    interleaved[2 * r] = columns[0][r];
                    interleaved[2 * r + 1] = columns[1][r];
                }
                for y in 0..bs {
                    put(dest, stride, bs, y, &interleaved[2 * y..]);
                }
            }
            D117_PRED => {
                // libvpx's `d117_predictor`: rows 0 and 1 from the above
                // edge, column 0 from the left, and every later row the one
                // two above it shifted a column right.
                let above_three = avg3_run::<V>(&a, 2 * bs);
                let left_three = avg3_run::<V>(&l, bs);
                let row0 = avg2_run::<V>(&a, 2 * bs);
                put(dest, stride, bs, 0, &row0);
                let mut row1 = [0u8; 32];
                row1[0] = avg3(l[0], top_left, a[1]);
                row1[1..bs].copy_from_slice(&above_three[..bs - 1]);
                put(dest, stride, bs, 1, &row1);
                for y in 2..bs {
                    let first = if y == 2 {
                        avg3(top_left, l[0], l[1])
                    } else {
                        left_three[y - 3]
                    };
                    dest[y * stride] = first;
                    let from = (y - 2) * stride;
                    dest.copy_within(from..from + bs - 1, y * stride + 1);
                }
            }
            D135_PRED => {
                // libvpx's `d135_predictor`: one border run, down the left
                // edge (reversed), round the corner and along the top; row
                // `y` starts `y` positions further towards the left edge.
                let above_three = avg3_run::<V>(&a, 2 * bs);
                let left_three = avg3_run::<V>(&l, bs);
                let mut border = [0u8; 64];
                for i in 0..bs - 2 {
                    border[i] = left_three[bs - 3 - i];
                }
                border[bs - 2] = avg3(top_left, l[0], l[1]);
                border[bs - 1] = avg3(l[0], top_left, a[1]);
                border[bs] = avg3(top_left, a[1], a[2]);
                border[bs + 1..2 * bs - 1].copy_from_slice(&above_three[1..bs - 1]);
                for y in 0..bs {
                    put(dest, stride, bs, y, &border[bs - 1 - y..]);
                }
            }
            _ => {
                // D153: libvpx's `d153_predictor`. Columns 0 and 1 from the
                // left edge, row 0 from the above one, and every later row
                // the one above it shifted two columns right.
                let left_two = avg2_run::<V>(&l, bs);
                let left_three = avg3_run::<V>(&l, bs);
                let above_three = avg3_run::<V>(&a, 2 * bs);
                let mut row0 = [0u8; 32];
                row0[0] = avg2(top_left, l[0]);
                row0[1] = avg3(l[0], top_left, a[1]);
                row0[2..bs].copy_from_slice(&above_three[..bs - 2]);
                put(dest, stride, bs, 0, &row0);
                for y in 1..bs {
                    dest[y * stride] = left_two[y - 1];
                    dest[y * stride + 1] = if y == 1 {
                        avg3(top_left, l[0], l[1])
                    } else {
                        left_three[y - 2]
                    };
                    let from = (y - 1) * stride;
                    dest.copy_within(from..from + bs - 2, y * stride + 2);
                }
            }
        }
        true
    }
}
