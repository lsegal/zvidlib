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

// Loops over vectors index rather than iterate: an iterator adapter or an
// `array::from_fn` closure over vector values is a separate function the
// inliner can leave outside the `#[target_feature]` wrapper, compiled at the
// baseline instruction set (#341).
#![allow(clippy::needless_range_loop)]

use crate::vp9_dec::recon::{
    D45_PRED, D63_PRED, D117_PRED, D135_PRED, D207_PRED, DC_PRED, H_PRED, TM_PRED, V_PRED,
};
use zvidlib_core::simd::vector::I32x;

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
///
/// Each block size is its own constant-length copy, which compiles to a few
/// vector moves rather than a call into `memcpy` per row.
#[inline(always)]
fn put(dest: &mut [u8], stride: usize, bs: usize, y: usize, row: &[u8]) {
    let at = y * stride;
    match bs {
        4 => dest[at..at + 4].copy_from_slice(&row[..4]),
        8 => dest[at..at + 8].copy_from_slice(&row[..8]),
        16 => dest[at..at + 16].copy_from_slice(&row[..16]),
        _ => dest[at..at + 32].copy_from_slice(&row[..32]),
    }
}

/// Fills row `y` of the block with `value`, with [`put`]'s constant-length
/// stores.
#[inline(always)]
fn fill(dest: &mut [u8], stride: usize, bs: usize, y: usize, value: u8) {
    put(dest, stride, bs, y, &[value; 32]);
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
        // DC, V, H and TM read no further than the edges themselves.
        let top_left = above[0];
        match mode {
            DC_PRED => {
                let value = match (have_left, have_above) {
                    (false, false) => 128,
                    (true, false) => (sum::<V>(left, bs) + (bs as u32 >> 1)) / bs as u32,
                    (false, true) => (sum::<V>(&above[1..], bs) + (bs as u32 >> 1)) / bs as u32,
                    (true, true) => {
                        let count = 2 * bs as u32;
                        (sum::<V>(left, bs) + sum::<V>(&above[1..], bs) + (count >> 1)) / count
                    }
                } as u8;
                for y in 0..bs {
                    fill(dest, stride, bs, y, value);
                }
                return true;
            }
            V_PRED => {
                for y in 0..bs {
                    put(dest, stride, bs, y, &above[1..]);
                }
                return true;
            }
            H_PRED => {
                for y in 0..bs {
                    fill(dest, stride, bs, y, left[y]);
                }
                return true;
            }
            TM_PRED => {
                let top_left = V::splat(i32::from(top_left));
                for y in 0..bs {
                    let side = V::splat(i32::from(left[y])).sub(top_left);
                    for x in (0..bs).step_by(V::LANES) {
                        V::load_u8(&above[1 + x..])
                            .add(side)
                            .store_u8_clamped(&mut dest[y * stride + x..]);
                    }
                }
                return true;
            }
            _ => {}
        }

        // The directional predictors read runs of averages, so they work on
        // copies padded far enough that a vector load at any run position
        // stays in bounds. `a[i + 1]` is the scalar code's `above(i)`.
        let mut a = [0u8; RUN];
        a[..65].copy_from_slice(above);
        let mut l = [0u8; RUN];
        l[..32].copy_from_slice(left);

        match mode {
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
                // two above it shifted a column right. So row `y` is a
                // window into the row of its parity, extended to the left
                // by the column-0 values that shift into it: `runs[p][bs +
                // i]` is row `p` and `runs[p][bs - m]` is column 0 of row
                // `p + 2 * m`, and row `y` starts at `bs - y / 2`.
                let above_three = avg3_run::<V>(&a, 2 * bs);
                let left_three = avg3_run::<V>(&l, bs);
                let row0 = avg2_run::<V>(&a, 2 * bs);
                let column = |row: usize| {
                    if row == 2 {
                        avg3(top_left, l[0], l[1])
                    } else {
                        left_three[row - 3]
                    }
                };
                let mut runs = [[0u8; 64]; 2];
                runs[0][bs..2 * bs].copy_from_slice(&row0[..bs]);
                runs[1][bs] = avg3(l[0], top_left, a[1]);
                runs[1][bs + 1..2 * bs].copy_from_slice(&above_three[..bs - 1]);
                for parity in 0..2 {
                    for m in 1..=(bs - 1 - parity) / 2 {
                        runs[parity][bs - m] = column(parity + 2 * m);
                    }
                }
                for y in 0..bs {
                    put(dest, stride, bs, y, &runs[y & 1][bs - y / 2..]);
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
                // the one above it shifted two columns right. So row `y` is
                // the window from `2 * (bs - y)` of one run: row 0 from
                // `2 * bs`, and before it the column 0 and column 1 values
                // of each row in turn, interleaved.
                let left_two = avg2_run::<V>(&l, bs);
                let left_three = avg3_run::<V>(&l, bs);
                let above_three = avg3_run::<V>(&a, 2 * bs);
                let mut run = [0u8; 96];
                run[2 * bs] = avg2(top_left, l[0]);
                run[2 * bs + 1] = avg3(l[0], top_left, a[1]);
                run[2 * bs + 2..3 * bs].copy_from_slice(&above_three[..bs - 2]);
                for row in 1..bs {
                    run[2 * (bs - row)] = left_two[row - 1];
                    run[2 * (bs - row) + 1] = if row == 1 {
                        avg3(top_left, l[0], l[1])
                    } else {
                        left_three[row - 2]
                    };
                }
                for y in 0..bs {
                    put(dest, stride, bs, y, &run[2 * (bs - y)..]);
                }
            }
        }
        true
    }
}
