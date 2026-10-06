//! The VP8 kernels, written once over [`I32x`] and instantiated per
//! instruction set by the `#[target_feature]` wrappers in [`super`].
//!
//! Each kernel is a lane-by-lane transliteration of the scalar routine it
//! replaces, in the same `i32` arithmetic and with the same truncations to
//! 16 bits, so the two agree bit for bit rather than approximately. The 4x4
//! kernels hold one block row per 4-lane vector and turn rows into columns
//! with [`Transpose4`] between their two passes; the pixel kernels (six-tap
//! prediction, `TM_PRED`, the loop filter and quantization) run one sample per
//! lane at whatever width the vector has.
//!
//! Every kernel is `#[inline(always)]`. As in `av1_simd`, that is what lets
//! its wrapper's `#[target_feature]` reach the intrinsics inside it; a kernel
//! the inliner left standing would be compiled at the baseline instruction set
//! and `.github/scripts/check_simd_target_features.py` fails on it.

use crate::av1_simd::vector::{I32x, Transpose4};

const COS_PI8_SQRT2_MINUS1: i32 = 20091;
const SIN_PI8_SQRT2: i32 = 35468;

/// `x as i16` widened back to `i32`, for the places libvpx stores a value in
/// `short`.
#[inline(always)]
unsafe fn truncate16<V: I32x>(x: V) -> V {
    unsafe { x.sll::<16>().sra::<16>() }
}

/// All-ones where a lane is nonzero.
#[inline(always)]
unsafe fn nonzero<V: I32x>(x: V) -> V {
    unsafe { x.abs().gt(V::zero()) }
}

// --- Distortion --------------------------------------------------------------

/// The halved sum of absolute 4x4 Hadamard-transformed values of the block
/// whose rows are `rows`, as `satd4` in `frame_encoder.rs` computes it.
///
/// The scalar code transforms rows and then columns; this transforms columns
/// (across the row vectors) and then rows (after a transpose). Both compute
/// the same exact integer `H * X * H^T`, only read out transposed, and the sum
/// of absolute values does not depend on the order it is read in.
#[inline(always)]
unsafe fn hadamard_sum<V: Transpose4>([r0, r1, r2, r3]: [V; 4]) -> u32 {
    unsafe {
        let (s0, s1, d0, d1) = (r0.add(r1), r2.add(r3), r0.sub(r1), r2.sub(r3));
        let [t0, t1, t2, t3] = V::transpose4([s0.add(s1), s0.sub(s1), d0.add(d1), d0.sub(d1)]);
        let (s0, s1, d0, d1) = (t0.add(t1), t2.add(t3), t0.sub(t1), t2.sub(t3));
        let sum = s0
            .add(s1)
            .abs()
            .add(s0.sub(s1).abs())
            .add(d0.add(d1).abs())
            .add(d0.sub(d1).abs());
        sum.hsum() as u32 / 2
    }
}

/// The rows of `source - prediction` for the 4x4 block at `offset`.
#[inline(always)]
unsafe fn residual_rows<V: I32x>(
    source: &[u8],
    prediction: &[u8],
    offset: usize,
    stride: usize,
) -> [V; 4] {
    unsafe {
        let row = |r: usize| {
            let at = offset + r * stride;
            (&source[at..at + 4], &prediction[at..at + 4])
        };
        let ((s0, p0), (s1, p1), (s2, p2), (s3, p3)) = (row(0), row(1), row(2), row(3));
        [
            V::load_u8(s0).sub(V::load_u8(p0)),
            V::load_u8(s1).sub(V::load_u8(p1)),
            V::load_u8(s2).sub(V::load_u8(p2)),
            V::load_u8(s3).sub(V::load_u8(p3)),
        ]
    }
}

/// `satd4` of a residual block in raster order.
#[inline(always)]
pub(super) unsafe fn satd4<V: Transpose4>(block: &[i16; 16]) -> u32 {
    unsafe {
        hadamard_sum::<V>([
            V::load_i16(&block[0..]),
            V::load_i16(&block[4..]),
            V::load_i16(&block[8..]),
            V::load_i16(&block[12..]),
        ])
    }
}

/// The SATD of the `size`x`size` block at `origin` of two planes of one
/// stride: the sum of every 4x4 block's [`satd4`].
#[inline(always)]
pub(super) unsafe fn satd<V: Transpose4>(
    source: &[u8],
    prediction: &[u8],
    origin: usize,
    stride: usize,
    size: usize,
) -> u32 {
    unsafe {
        let mut sum = 0;
        for block_y in 0..size / 4 {
            for block_x in 0..size / 4 {
                let offset = origin + block_y * 4 * stride + block_x * 4;
                sum += hadamard_sum::<V>(residual_rows(source, prediction, offset, stride));
            }
        }
        sum
    }
}

// --- Forward transforms and quantization ------------------------------------

/// libvpx's forward 4x4 DCT (`vp8_short_fdct4x4_c`) of the residual of the
/// 4x4 block at `offset`, raster order.
#[inline(always)]
pub(super) unsafe fn residual_dct<V: Transpose4>(
    source: &[u8],
    prediction: &[u8],
    offset: usize,
    stride: usize,
) -> [i16; 16] {
    unsafe {
        // Lane `r` of column `j` is input row `r`, so the row pass below
        // computes every row at once.
        let [c0, c1, c2, c3] = V::transpose4(residual_rows(source, prediction, offset, stride));
        let a1 = c0.add(c3).sll::<3>();
        let b1 = c1.add(c2).sll::<3>();
        let c1 = c1.sub(c2).sll::<3>();
        let d1 = c0.sub(c3).sll::<3>();
        let (k2217, k5352) = (V::splat(2217), V::splat(5352));
        let t0 = a1.add(b1);
        let t2 = a1.sub(b1);
        let t1 = c1
            .mul(k2217)
            .add(d1.mul(k5352))
            .add(V::splat(14500))
            .sra::<12>();
        let t3 = d1
            .mul(k2217)
            .sub(c1.mul(k5352))
            .add(V::splat(7500))
            .sra::<12>();

        // Back to rows, so the column pass computes every column at once.
        let [r0, r1, r2, r3] = V::transpose4([t0, t1, t2, t3]);
        let a1 = r0.add(r3);
        let b1 = r1.add(r2);
        let c1 = r1.sub(r2);
        let d1 = r0.sub(r3);
        let seven = V::splat(7);
        let out0 = a1.add(b1).add(seven).sra::<4>();
        let out2 = a1.sub(b1).add(seven).sra::<4>();
        // Adding one where `d1 != 0` is subtracting its all-ones mask.
        let out1 = c1
            .mul(k2217)
            .add(d1.mul(k5352))
            .add(V::splat(12000))
            .sra::<16>()
            .sub(nonzero(d1));
        let out3 = d1
            .mul(k2217)
            .sub(c1.mul(k5352))
            .add(V::splat(51000))
            .sra::<16>();
        let mut output = [0i16; 16];
        out0.store_i16(&mut output[0..]);
        out1.store_i16(&mut output[4..]);
        out2.store_i16(&mut output[8..]);
        out3.store_i16(&mut output[12..]);
        output
    }
}

/// libvpx's forward Walsh-Hadamard transform (`vp8_short_walsh4x4_c`) of the
/// 16 luma DC coefficients in raster order.
#[inline(always)]
pub(super) unsafe fn forward_walsh<V: Transpose4>(input: &[i16; 16]) -> [i16; 16] {
    unsafe {
        let [c0, c1, c2, c3] = V::transpose4([
            V::load_i16(&input[0..]),
            V::load_i16(&input[4..]),
            V::load_i16(&input[8..]),
            V::load_i16(&input[12..]),
        ]);
        let a1 = c0.add(c2).sll::<2>();
        let d1 = c1.add(c3).sll::<2>();
        let c1 = c1.sub(c3).sll::<2>();
        let b1 = c0.sub(c2).sll::<2>();
        let t0 = a1.add(d1).sub(nonzero(a1));
        let t1 = b1.add(c1);
        let t2 = b1.sub(c1);
        let t3 = a1.sub(d1);

        let [r0, r1, r2, r3] = V::transpose4([t0, t1, t2, t3]);
        let a1 = r0.add(r2);
        let d1 = r1.add(r3);
        let c1 = r1.sub(r3);
        let b1 = r0.sub(r2);
        // `(v + (v < 0) + 3) >> 3`; the comparison mask is -1 where it holds.
        let round = |v: V| v.sub(V::zero().gt(v)).add(V::splat(3)).sra::<3>();
        let mut output = [0i16; 16];
        round(a1.add(d1)).store_i16(&mut output[0..]);
        round(b1.add(c1)).store_i16(&mut output[4..]);
        round(b1.sub(c1)).store_i16(&mut output[8..]);
        round(a1.sub(d1)).store_i16(&mut output[12..]);
        output
    }
}

/// Quantizes a block's coefficients, returning the levels and their
/// dequantized values, as `quantize` in `frame_encoder.rs` does.
///
/// The scalar code walks the block in zigzag order, but the only thing it
/// takes from that order is whether a coefficient is the first one, and the
/// first zigzag position is raster position 0. So this works in raster order,
/// with the DC step and bias in lane 0, and clears position 0 afterwards when
/// `first` skips it.
#[inline(always)]
pub(super) unsafe fn quantize<V: I32x>(
    coefficients: &[i16; 16],
    [dc, ac]: [i32; 2],
    first: usize,
) -> ([i16; 16], [i16; 16]) {
    unsafe {
        let mut steps = [ac; 16];
        steps[0] = dc;
        let mut biases = [ac / 4; 16];
        biases[0] = dc / 2;
        let mut levels = [0i16; 16];
        let mut dequantized = [0i16; 16];
        let zero = V::zero();
        let mut at = 0;
        while at < 16 {
            let value = V::load_i16(&coefficients[at..]);
            let step = V::load(&steps[at..]);
            let level = value
                .abs()
                .add(V::load(&biases[at..]))
                .div_nonneg(step)
                .min(V::splat(2048));
            let level = V::select(zero.gt(value), zero.sub(level), level);
            level.store_i16(&mut levels[at..]);
            level.mul(step).store_i16(&mut dequantized[at..]);
            at += V::LANES;
        }
        if first > 0 {
            levels[0] = 0;
            dequantized[0] = 0;
        }
        (levels, dequantized)
    }
}

// --- Inverse transforms --------------------------------------------------------

/// One pass of the inverse DCT over four vectors.
#[inline(always)]
unsafe fn idct_pass<V: I32x>([i0, i1, i2, i3]: [V; 4]) -> (V, V, V, V) {
    unsafe {
        let (sin, cos) = (V::splat(SIN_PI8_SQRT2), V::splat(COS_PI8_SQRT2_MINUS1));
        let a1 = i0.add(i2);
        let b1 = i0.sub(i2);
        let c1 = i1.mul(sin).sra::<16>().sub(i3.add(i3.mul(cos).sra::<16>()));
        let d1 = i1.add(i1.mul(cos).sra::<16>()).add(i3.mul(sin).sra::<16>());
        (a1, b1, c1, d1)
    }
}

/// Inverse 4x4 DCT of dequantized `coefficients` (raster order), added to the
/// 4x4 block at `offset` of `plane`, as `idct_add` in `predict.rs` does.
#[inline(always)]
pub(super) unsafe fn idct_add<V: Transpose4>(
    coefficients: &[i16; 16],
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) {
    unsafe {
        // The vertical pass runs over the rows as they are, one column a lane.
        let (a1, b1, c1, d1) = idct_pass([
            V::load_i16(&coefficients[0..]),
            V::load_i16(&coefficients[4..]),
            V::load_i16(&coefficients[8..]),
            V::load_i16(&coefficients[12..]),
        ]);
        let temp = [
            truncate16(a1.add(d1)),
            truncate16(b1.add(c1)),
            truncate16(b1.sub(c1)),
            truncate16(a1.sub(d1)),
        ];

        // The horizontal pass, one row a lane.
        let (a1, b1, c1, d1) = idct_pass(V::transpose4(temp));
        let four = V::splat(4);
        let rows = V::transpose4([
            truncate16(a1.add(d1).add(four).sra::<3>()),
            truncate16(b1.add(c1).add(four).sra::<3>()),
            truncate16(b1.sub(c1).add(four).sra::<3>()),
            truncate16(a1.sub(d1).add(four).sra::<3>()),
        ]);
        for (row, residual) in rows.into_iter().enumerate() {
            let line = &mut plane[offset + row * stride..offset + row * stride + 4];
            V::load_u8(line).add(residual).store_u8_clamped(line);
        }
    }
}

/// Inverse Walsh-Hadamard transform of the dequantized Y2 block, as
/// `inverse_walsh` in `predict.rs` computes it.
#[inline(always)]
pub(super) unsafe fn inverse_walsh<V: Transpose4>(input: &[i16; 16]) -> [i16; 16] {
    unsafe {
        let r0 = V::load_i16(&input[0..]);
        let r1 = V::load_i16(&input[4..]);
        let r2 = V::load_i16(&input[8..]);
        let r3 = V::load_i16(&input[12..]);
        let a1 = r0.add(r3);
        let b1 = r1.add(r2);
        let c1 = r1.sub(r2);
        let d1 = r0.sub(r3);
        let [i0, i1, i2, i3] = V::transpose4([
            truncate16(a1.add(b1)),
            truncate16(c1.add(d1)),
            truncate16(a1.sub(b1)),
            truncate16(d1.sub(c1)),
        ]);
        let a1 = i0.add(i3);
        let b1 = i1.add(i2);
        let c1 = i1.sub(i2);
        let d1 = i0.sub(i3);
        let three = V::splat(3);
        let rows = V::transpose4([
            a1.add(b1).add(three).sra::<3>(),
            c1.add(d1).add(three).sra::<3>(),
            a1.sub(b1).add(three).sra::<3>(),
            d1.sub(c1).add(three).sra::<3>(),
        ]);
        let mut output = [0i16; 16];
        for (row, values) in rows.into_iter().enumerate() {
            values.store_i16(&mut output[row * 4..]);
        }
        output
    }
}

// --- Prediction --------------------------------------------------------------

/// The two filter passes of `predict_inter`: `taps` horizontally over the
/// `(width + 5)`-wide, `(height + 5)`-tall `window`, then vertically into the
/// `width`x`height` block at `destination` of `output`. `width` is a multiple
/// of `V::LANES`.
///
/// A tap of zero contributes nothing to its sum, so it is skipped rather than
/// multiplied; the bilinear filters and the six-tap filters at even fractions
/// have two to four of them.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) unsafe fn sixtap<V: I32x>(
    window: &[u8],
    width: usize,
    height: usize,
    horizontal: &[i32; 6],
    vertical: &[i32; 6],
    output: &mut [u8],
    destination: usize,
    stride: usize,
) {
    unsafe {
        let window_width = width + 5;
        let round = V::splat(64);
        let mut first = [0u8; 16 * 21];
        for row in 0..height + 5 {
            let line = &window[row * window_width..(row + 1) * window_width];
            let mut column = 0;
            while column < width {
                let mut sum = round;
                for (tap_index, &tap) in horizontal.iter().enumerate() {
                    if tap != 0 {
                        let samples = V::load_u8(&line[column + tap_index..]);
                        sum = sum.add(samples.mul(V::splat(tap)));
                    }
                }
                sum.sra::<7>()
                    .store_u8_clamped(&mut first[row * width + column..]);
                column += V::LANES;
            }
        }
        for row in 0..height {
            let mut column = 0;
            while column < width {
                let mut sum = round;
                for (tap_index, &tap) in vertical.iter().enumerate() {
                    if tap != 0 {
                        let samples = V::load_u8(&first[(row + tap_index) * width + column..]);
                        sum = sum.add(samples.mul(V::splat(tap)));
                    }
                }
                sum.sra::<7>()
                    .store_u8_clamped(&mut output[destination + row * stride + column..]);
                column += V::LANES;
            }
        }
    }
}

/// `TM_PRED` of an `size`x`size` block: each sample is its left neighbour
/// plus the one above it minus the corner, clamped. `above` holds the `size`
/// samples above the block, without the corner.
#[inline(always)]
pub(super) unsafe fn tm_predict<V: I32x>(
    above: &[u8],
    left: &[u8],
    corner: i32,
    size: usize,
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) {
    unsafe {
        for (row, &left) in left.iter().enumerate().take(size) {
            let left = V::splat(i32::from(left) - corner);
            let mut column = 0;
            while column < size {
                V::load_u8(&above[column..])
                    .add(left)
                    .store_u8_clamped(&mut plane[offset + row * stride + column..]);
                column += V::LANES;
            }
        }
    }
}

// --- Loop filter ---------------------------------------------------------------

/// The thresholds `filter_edge` in `loop_filter.rs` takes, already resolved
/// for the edge being filtered.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeLimits {
    /// The macroblock- or subblock-edge limit, whichever this edge uses.
    pub edge: i32,
    pub interior: i32,
    pub hev_threshold: i32,
    pub macroblock_edge: bool,
    pub simple: bool,
}

/// Filters one vector of segments across an edge in place: `p[i]` and `q[i]`
/// hold the `i`th sample on either side of it, one segment a lane.
///
/// The scalar `Segment` code branches per segment between not filtering, the
/// common adjustment with or without the outer taps, and the macroblock
/// filter. This computes each outcome a lane could take and selects per lane,
/// which gives every lane exactly the values its branch would have.
#[inline(always)]
unsafe fn filter_segments<V: I32x>(p: &mut [V; 4], q: &mut [V; 4], limits: EdgeLimits) {
    unsafe {
        let difference = |a: V, b: V| a.sub(b).abs();
        let (low, high) = (V::splat(-128), V::splat(127));
        let clamp = |x: V| x.clamp(low, high);
        let bias = V::splat(128);
        let to_unsigned = |x: V| clamp(x).add(bias);

        let mut mask = difference(p[0], q[0])
            .sll::<1>()
            .add(difference(p[1], q[1]).sra::<1>())
            .le(V::splat(limits.edge));
        if !limits.simple {
            let interior = V::splat(limits.interior);
            for (a, b) in [
                (p[3], p[2]),
                (p[2], p[1]),
                (p[1], p[0]),
                (q[3], q[2]),
                (q[2], q[1]),
                (q[1], q[0]),
            ] {
                mask = mask.and(difference(a, b).le(interior));
            }
        }
        if !mask.any() {
            return;
        }

        let p1 = p[1].sub(bias);
        let p0 = p[0].sub(bias);
        let q0 = q[0].sub(bias);
        let q1 = q[1].sub(bias);
        let base = q0.sub(p0).mul(V::splat(3));
        let outer = clamp(base.add(clamp(p1.sub(q1))));
        // The common adjustment of `p0`/`q0` for the filter value `a`.
        let common = |a: V| {
            let f1 = clamp(a.add(V::splat(4))).sra::<3>();
            let f2 = clamp(a.add(V::splat(3))).sra::<3>();
            (to_unsigned(p0.add(f2)), to_unsigned(q0.sub(f1)), f1)
        };

        if limits.simple {
            let (new_p0, new_q0, _) = common(outer);
            p[0] = V::select(mask, new_p0, p[0]);
            q[0] = V::select(mask, new_q0, q[0]);
            return;
        }

        let threshold = V::splat(limits.hev_threshold);
        let hev = difference(p[1], p[0])
            .gt(threshold)
            .or(difference(q[1], q[0]).gt(threshold));
        // Lanes that change more than `p0`/`q0`.
        let wide = hev.andnot(mask);
        if limits.macroblock_edge {
            // High edge variance takes the common adjustment with the outer
            // taps; anything else the macroblock filter, whose `w` is that
            // same clamped value.
            let (hev_p0, hev_q0, _) = common(outer);
            let tap = |weight: i32| clamp(outer.mul(V::splat(weight)).add(V::splat(63)).sra::<7>());
            let (a0, a1, a2) = (tap(27), tap(18), tap(9));
            let p2 = p[2].sub(bias);
            let q2 = q[2].sub(bias);
            p[0] = V::select(mask, V::select(hev, hev_p0, to_unsigned(p0.add(a0))), p[0]);
            q[0] = V::select(mask, V::select(hev, hev_q0, to_unsigned(q0.sub(a0))), q[0]);
            p[1] = V::select(wide, to_unsigned(p1.add(a1)), p[1]);
            q[1] = V::select(wide, to_unsigned(q1.sub(a1)), q[1]);
            p[2] = V::select(wide, to_unsigned(p2.add(a2)), p[2]);
            q[2] = V::select(wide, to_unsigned(q2.sub(a2)), q[2]);
        } else {
            // The common adjustment, with the outer taps where the edge
            // variance is high and adjusting `p1`/`q1` where it is not.
            let a = V::select(hev, outer, clamp(base));
            let (new_p0, new_q0, f1) = common(a);
            let a = f1.add(V::splat(1)).sra::<1>();
            p[0] = V::select(mask, new_p0, p[0]);
            q[0] = V::select(mask, new_q0, q[0]);
            p[1] = V::select(wide, to_unsigned(p1.add(a)), p[1]);
            q[1] = V::select(wide, to_unsigned(q1.sub(a)), q[1]);
        }
    }
}

/// Filters `count` segments across a horizontal edge whose first `q0` is
/// `data[at]`: the segments run along a row, and each crosses the edge
/// vertically. `count` is a multiple of `V::LANES`.
#[inline(always)]
pub(super) unsafe fn filter_horizontal_edge<V: I32x>(
    data: &mut [u8],
    at: usize,
    stride: usize,
    count: usize,
    limits: EdgeLimits,
) {
    unsafe {
        let mut segment = 0;
        while segment < count {
            let at = at + segment;
            let row = |i: usize| at + i * stride;
            let above = |i: usize| at - (i + 1) * stride;
            let mut p = [
                V::load_u8(&data[above(0)..]),
                V::load_u8(&data[above(1)..]),
                V::load_u8(&data[above(2)..]),
                V::load_u8(&data[above(3)..]),
            ];
            let mut q = [
                V::load_u8(&data[row(0)..]),
                V::load_u8(&data[row(1)..]),
                V::load_u8(&data[row(2)..]),
                V::load_u8(&data[row(3)..]),
            ];
            filter_segments(&mut p, &mut q, limits);
            for i in 0..3 {
                p[i].store_u8_clamped(&mut data[above(i)..]);
                q[i].store_u8_clamped(&mut data[row(i)..]);
            }
            segment += V::LANES;
        }
    }
}

/// Filters `count` segments across a vertical edge whose first `q0` is
/// `data[at]`: one segment a row, each crossing the edge horizontally.
/// `count` is a multiple of `V::LANES`.
///
/// A segment's eight samples are two little-endian words of its row, `p3` to
/// `p0` and `q0` to `q3`, so each side is read and written as one word a lane.
#[inline(always)]
pub(super) unsafe fn filter_vertical_edge<V: I32x>(
    data: &mut [u8],
    at: usize,
    stride: usize,
    count: usize,
    limits: EdgeLimits,
) {
    unsafe {
        let byte = V::splat(0xff);
        let mut segment = 0;
        while segment < count {
            let at = at + segment * stride;
            let before = V::load_u32_rows(data, at - 4, stride);
            let after = V::load_u32_rows(data, at, stride);
            let mut p = [
                before.srl::<24>(),
                before.srl::<16>().and(byte),
                before.srl::<8>().and(byte),
                before.and(byte),
            ];
            let mut q = [
                after.and(byte),
                after.srl::<8>().and(byte),
                after.srl::<16>().and(byte),
                after.srl::<24>(),
            ];
            filter_segments(&mut p, &mut q, limits);
            p[3].or(p[2].sll::<8>())
                .or(p[1].sll::<16>())
                .or(p[0].sll::<24>())
                .store_u32_rows(data, at - 4, stride);
            q[0].or(q[1].sll::<8>())
                .or(q[2].sll::<16>())
                .or(q[3].sll::<24>())
                .store_u32_rows(data, at, stride);
            segment += V::LANES;
        }
    }
}
