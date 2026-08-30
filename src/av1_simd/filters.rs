//! Vectorized AV1 in-loop filter pixel kernels.
//!
//! Each kernel is the elementwise transliteration of the corresponding scalar
//! routine in [`crate::av1_filters`], evaluated for `V::LANES` output positions
//! at a time. All arithmetic that the scalar code performs in `i32` is
//! performed in `i32` lanes here, and the two `i64` accumulators the scalar code
//! uses (CDEF's direction search and the self-guided box statistics) are proven
//! to fit in `i32` for 8-bit input, so the results are bit-exact rather than
//! approximately equal.
//!
//! Every kernel here assumes its caller has already established that all sample
//! reads land inside the plane; the dispatchers in [`super`] keep the
//! edge-clamped positions on the scalar path.

use super::vector::{I32x, MAX_LANES};

// ---------------------------------------------------------------------
// Deblocking: narrow (4-tap) filter, spec 7.14.6.2
// ---------------------------------------------------------------------

#[inline(always)]
unsafe fn clamp_i8<V: I32x>(value: V) -> V {
    unsafe { value.clamp(V::splat(-128), V::splat(127)) }
}

#[inline(always)]
unsafe fn clamp_pixel<V: I32x>(value: V) -> V {
    unsafe { value.clamp(V::zero(), V::splat(255)) }
}

/// Vector form of `filter_mask` + `hev_mask` + `narrow_filter`: returns the new
/// `(p1, p0, q0, q1)` with lanes whose boundary mask fails left untouched.
#[inline(always)]
unsafe fn narrow_filter_lanes<V: I32x>(
    p1: V,
    p0: V,
    q0: V,
    q1: V,
    limit: i32,
    blimit: i32,
    thresh: i32,
) -> [V; 4] {
    unsafe {
        let dp = p1.sub(p0).abs();
        let dq = q1.sub(q0).abs();
        let boundary = p0
            .sub(q0)
            .abs()
            .mul(V::splat(2))
            .add(p1.sub(q1).abs().sra::<1>());
        let mask = dp
            .le(V::splat(limit))
            .and(dq.le(V::splat(limit)))
            .and(boundary.le(V::splat(blimit)));

        let thresh = V::splat(thresh);
        let hev = dp.gt(thresh).or(dq.gt(thresh));

        let bias = V::splat(128);
        let ps1 = p1.sub(bias);
        let ps0 = p0.sub(bias);
        let qs0 = q0.sub(bias);
        let qs1 = q1.sub(bias);

        let filter = V::select(hev, clamp_i8(ps1.sub(qs1)), V::zero());
        let filter = clamp_i8(filter.add(qs0.sub(ps0).mul(V::splat(3))));
        let filter1 = clamp_i8(filter.add(V::splat(4))).sra::<3>();
        let filter2 = clamp_i8(filter.add(V::splat(3))).sra::<3>();

        let new_q0 = clamp_pixel(qs0.sub(filter1).add(bias));
        let new_p0 = clamp_pixel(ps0.add(filter2).add(bias));
        let outer = filter1.add(V::splat(1)).sra::<1>();
        let new_p1 = V::select(hev, p1, clamp_pixel(ps1.add(outer).add(bias)));
        let new_q1 = V::select(hev, q1, clamp_pixel(qs1.sub(outer).add(bias)));

        [
            V::select(mask, new_p1, p1),
            V::select(mask, new_p0, p0),
            V::select(mask, new_q0, q0),
            V::select(mask, new_q1, q1),
        ]
    }
}

/// Filters `V::LANES` positions of the horizontal edge above row `y`, starting
/// at column `x0`. The four taps live in rows `y - 2 ..= y + 1`, so each one is
/// a contiguous byte run and loads directly.
///
/// # Safety
/// `V`'s instruction set must be available, `y >= 2`, `y + 1 < height`, and
/// `x0 + V::LANES <= width`.
pub(crate) unsafe fn deblock_narrow_horizontal<V: I32x>(
    data: &mut [u8],
    stride: usize,
    x0: usize,
    y: usize,
    limit: i32,
    blimit: i32,
    thresh: i32,
) {
    unsafe {
        let rows = [
            (y - 2) * stride + x0,
            (y - 1) * stride + x0,
            y * stride + x0,
            (y + 1) * stride + x0,
        ];
        let taps = [
            V::load_u8(&data[rows[0]..]),
            V::load_u8(&data[rows[1]..]),
            V::load_u8(&data[rows[2]..]),
            V::load_u8(&data[rows[3]..]),
        ];
        let out = narrow_filter_lanes(taps[0], taps[1], taps[2], taps[3], limit, blimit, thresh);
        for (offset, value) in rows.into_iter().zip(out) {
            value.store_u8_clamped(&mut data[offset..]);
        }
    }
}

/// Filters `V::LANES` positions of the vertical edge left of column `x`,
/// starting at row `y0`. The four taps are contiguous *within* a row but the
/// filtered positions run down the column, so the taps are staged through a
/// small scratch transpose; the per-position arithmetic still runs vectorized.
///
/// # Safety
/// `V`'s instruction set must be available, `x >= 2`, `x + 1 < stride`, and
/// `y0 + V::LANES <= height`.
pub(crate) unsafe fn deblock_narrow_vertical<V: I32x>(
    data: &mut [u8],
    stride: usize,
    x: usize,
    y0: usize,
    limit: i32,
    blimit: i32,
    thresh: i32,
) {
    unsafe {
        let mut staged = [[0u8; MAX_LANES]; 4];
        for lane in 0..V::LANES {
            let base = (y0 + lane) * stride + x - 2;
            for (tap, column) in staged.iter_mut().enumerate() {
                column[lane] = data[base + tap];
            }
        }
        let taps = [
            V::load_u8(&staged[0]),
            V::load_u8(&staged[1]),
            V::load_u8(&staged[2]),
            V::load_u8(&staged[3]),
        ];
        let out = narrow_filter_lanes(taps[0], taps[1], taps[2], taps[3], limit, blimit, thresh);
        for (tap, value) in out.into_iter().enumerate() {
            value.store_u8_clamped(&mut staged[tap]);
        }
        for lane in 0..V::LANES {
            let base = (y0 + lane) * stride + x - 2;
            for (tap, column) in staged.iter().enumerate() {
                data[base + tap] = column[lane];
            }
        }
    }
}

// ---------------------------------------------------------------------
// CDEF, spec 7.15
// ---------------------------------------------------------------------

/// Accumulates the direction-search `(sum, sum_sq)` of sample differences over
/// one 8x8 block along the `(dr, dc)` offset.
///
/// Both accumulators are computed in `i32`: differences are bounded by 255, so
/// the block totals stay under `64 * 255^2`, well inside `i32`, and match the
/// scalar `i64` accumulation exactly.
///
/// # Safety
/// `V`'s instruction set must be available and every sampled position, offset
/// included, must be inside the plane.
pub(crate) unsafe fn cdef_direction_stats<V: I32x>(
    data: &[u8],
    stride: usize,
    x0: usize,
    y0: usize,
    dr: i32,
    dc: i32,
) -> (i32, i32) {
    unsafe {
        let mut sum = V::zero();
        let mut sum_sq = V::zero();
        for row in 0..8usize {
            let mut column = 0;
            while column < 8 {
                let a_index = (y0 + row) * stride + x0 + column;
                let b_index = (((y0 + row) as isize + dr as isize) * stride as isize
                    + (x0 + column) as isize
                    + dc as isize) as usize;
                let diff = V::load_u8(&data[a_index..]).sub(V::load_u8(&data[b_index..]));
                sum = sum.add(diff);
                sum_sq = sum_sq.add(diff.mul(diff));
                column += V::LANES;
            }
        }
        (sum.hsum(), sum_sq.hsum())
    }
}

/// One CDEF tap: weight and `(row, column)` offset from the filtered sample.
pub(crate) type CdefTap = (i32, i32, i32);

/// Vector form of the spec's `constrain`, with the damping adjustment already
/// resolved by the caller (it depends only on the strength and damping).
#[inline(always)]
unsafe fn constrain_lanes<V: I32x>(diff: V, threshold: i32, damping_adj: i32) -> V {
    unsafe {
        let magnitude = diff.abs();
        let clipped = V::splat(threshold)
            .sub(magnitude.sra_var(damping_adj))
            .clamp(V::zero(), magnitude);
        clipped.mul(diff.signum())
    }
}

/// Precomputes `constrain`'s damping shift for one strength.
pub(crate) fn constrain_damping_adjustment(threshold: i32, damping: i32) -> i32 {
    (damping - (31 - threshold.max(1).leading_zeros() as i32)).max(0)
}

/// Filters `V::LANES` samples of row `y` starting at column `x0`, writing them
/// to `dst` (which is the destination row's slice starting at `x0`).
///
/// # Safety
/// `V`'s instruction set must be available, every tap offset applied to every
/// filtered sample must be inside the plane, and `dst` must hold at least
/// `V::LANES` bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cdef_filter_row<V: I32x>(
    data: &[u8],
    stride: usize,
    x0: usize,
    y: usize,
    primary: &[CdefTap],
    primary_strength: i32,
    primary_damping_adj: i32,
    secondary: &[CdefTap],
    secondary_strength: i32,
    secondary_damping_adj: i32,
    total_weight: i32,
    dst: &mut [u8],
) {
    unsafe {
        let center = V::load_u8(&data[y * stride + x0..]);
        let mut sum = V::zero();
        for &(weight, dr, dc) in primary {
            let index = (((y as isize + dr as isize) * stride as isize)
                + (x0 as isize + dc as isize)) as usize;
            let diff = V::load_u8(&data[index..]).sub(center);
            sum = sum.add(
                constrain_lanes(diff, primary_strength, primary_damping_adj).mul(V::splat(weight)),
            );
        }
        for &(weight, dr, dc) in secondary {
            let index = (((y as isize + dr as isize) * stride as isize)
                + (x0 as isize + dc as isize)) as usize;
            let diff = V::load_u8(&data[index..]).sub(center);
            sum = sum.add(
                constrain_lanes(diff, secondary_strength, secondary_damping_adj)
                    .mul(V::splat(weight)),
            );
        }
        if total_weight == 0 {
            center.store_u8_clamped(dst);
            return;
        }
        let rounded = sum.add(V::splat(1 << 3)).sra::<4>();
        center.add(rounded).store_u8_clamped(dst);
    }
}

// ---------------------------------------------------------------------
// Loop restoration: Wiener, spec 7.17.2
// ---------------------------------------------------------------------

/// Runs the Wiener horizontal pass for `V::LANES` samples of row `y` starting
/// at column `x0`, writing the rounded intermediate values to `out`.
///
/// # Safety
/// `V`'s instruction set must be available, `x0 >= 3`, `x0 + V::LANES + 3 <=
/// width`, and `out` must hold at least `V::LANES` values.
pub(crate) unsafe fn wiener_horizontal_row<V: I32x>(
    data: &[u8],
    stride: usize,
    x0: usize,
    y: usize,
    taps: [i32; 3],
    center_tap: i32,
    out: &mut [i32],
) {
    unsafe {
        let row = y * stride;
        let mut sum = V::load_u8(&data[row + x0..]).mul(V::splat(center_tap));
        for (k, &tap) in taps.iter().enumerate() {
            let offset = 3 - k;
            let minus = V::load_u8(&data[row + x0 - offset..]);
            let plus = V::load_u8(&data[row + x0 + offset..]);
            sum = sum.add(minus.add(plus).mul(V::splat(tap)));
        }
        sum.add(V::splat(1 << 2)).sra::<3>().store(out);
    }
}

/// Runs the Wiener vertical pass for `V::LANES` columns of `row` over the
/// horizontal pass's `intermediate` buffer, writing clipped samples to `dst`.
///
/// # Safety
/// `V`'s instruction set must be available, `row >= 3`, `row + 3 < height`,
/// `column + V::LANES <= width`, and `dst` must hold at least `V::LANES` bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn wiener_vertical_row<V: I32x>(
    intermediate: &[i32],
    width: usize,
    row: usize,
    column: usize,
    taps: [i32; 3],
    center_tap: i32,
    dst: &mut [u8],
) {
    unsafe {
        let base = row * width + column;
        let mut sum = V::load(&intermediate[base..]).mul(V::splat(center_tap));
        for (k, &tap) in taps.iter().enumerate() {
            let offset = (3 - k) * width;
            let minus = V::load(&intermediate[base - offset..]);
            let plus = V::load(&intermediate[base + offset..]);
            sum = sum.add(minus.add(plus).mul(V::splat(tap)));
        }
        sum.add(V::splat(1 << 10)).sra::<11>().store_u8_clamped(dst);
    }
}

// ---------------------------------------------------------------------
// Loop restoration: self-guided box statistics, spec 7.17.3
// ---------------------------------------------------------------------

/// Accumulates `(sum, sum_sq)` over the `(2r+1)x(2r+1)` window centered on each
/// of `V::LANES` consecutive samples of row `y` starting at column `x0`.
///
/// 8-bit samples bound the window sum by `49 * 255` and the sum of squares by
/// `49 * 255^2`, so the scalar reference's `i64` accumulators and these `i32`
/// lanes agree exactly.
///
/// # Safety
/// `V`'s instruction set must be available and the whole window of every
/// filtered sample must be inside the plane.
pub(crate) unsafe fn box_stats_row<V: I32x>(
    data: &[u8],
    stride: usize,
    x0: usize,
    y: usize,
    radius: usize,
    sums: &mut [i32],
    sums_sq: &mut [i32],
) {
    unsafe {
        let mut sum = V::zero();
        let mut sum_sq = V::zero();
        for dy in 0..=(2 * radius) {
            let row = (y + dy - radius) * stride;
            for dx in 0..=(2 * radius) {
                let value = V::load_u8(&data[row + x0 + dx - radius..]);
                sum = sum.add(value);
                sum_sq = sum_sq.add(value.mul(value));
            }
        }
        sum.store(sums);
        sum_sq.store(sums_sq);
    }
}
