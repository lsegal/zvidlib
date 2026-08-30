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
//! Unlike the first version of this module, no filter position stays on the
//! scalar path because of where it sits in the plane. Sample reads go through
//! [`load_row`] / [`gather_column`], which reproduce
//! [`crate::av1_filters::FilterPlane::get_clamped`]'s edge replication, and
//! writes go through the `*_masked` stores so a partial row or column at the
//! right or bottom edge of a plane touches exactly the positions the caller
//! asked for. Frame borders, tile/superblock boundaries and partial
//! rows/columns therefore all run on the vector path; only a target without any
//! supported instruction set falls back to [`crate::av1_filters`]'s scalar code.

use super::vector::{I32x, MAX_LANES};

/// The plane geometry a kernel needs to reproduce the scalar path's edge
/// clamping: the row stride plus the logical sample bounds.
#[derive(Clone, Copy)]
pub(crate) struct Geometry {
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

impl Geometry {
    #[inline(always)]
    fn clamp_x(&self, x: isize) -> usize {
        x.clamp(0, self.width as isize - 1) as usize
    }

    #[inline(always)]
    fn clamp_y(&self, y: isize) -> usize {
        y.clamp(0, self.height as isize - 1) as usize
    }

    /// Byte offset of the start of row `y`, clamped to the plane.
    #[inline(always)]
    fn row(&self, y: isize) -> usize {
        self.clamp_y(y) * self.stride
    }
}

/// Loads `V::LANES` samples of the row that starts at byte offset `row_base`,
/// beginning at column `x`, replicating the nearest in-plane sample for columns
/// outside the plane.
///
/// The row offset is a parameter rather than a row index so callers that read
/// several vectors from one row (the Wiener horizontal pass, the box
/// statistics) pay for the clamp and the multiply once instead of once per
/// load.
///
/// # Safety
/// `V`'s instruction set must be available and `row_base` must be the start of
/// a row of `data`.
#[inline(always)]
unsafe fn load_lanes<V: I32x>(data: &[u8], geom: Geometry, row_base: usize, x: isize) -> V {
    unsafe { load_lanes_inside(data, geom, row_base, x, columns_inside::<V>(geom, x)) }
}

/// True when a whole `V::LANES` window starting at column `x` is inside the
/// plane. A kernel whose taps share one window (the Wiener horizontal pass, the
/// box statistics) evaluates this once and hands it to every load.
#[inline(always)]
fn columns_inside<V: I32x>(geom: Geometry, x: isize) -> bool {
    x >= 0 && (x as usize) + V::LANES <= geom.width
}

/// [`load_lanes`] with the in-plane test already made by the caller.
///
/// # Safety
/// As [`load_lanes`], and `inside` must equal [`columns_inside`] for `x`.
#[inline(always)]
unsafe fn load_lanes_inside<V: I32x>(
    data: &[u8],
    geom: Geometry,
    row_base: usize,
    x: isize,
    inside: bool,
) -> V {
    unsafe {
        if inside {
            V::load_u8(&data[row_base + x as usize..])
        } else {
            let mut staged = [0u8; MAX_LANES];
            for (lane, slot) in staged.iter_mut().enumerate().take(V::LANES) {
                *slot = data[row_base + geom.clamp_x(x + lane as isize)];
            }
            V::load_u8(&staged)
        }
    }
}

/// Loads `V::LANES` samples of row `y` starting at column `x`, replicating the
/// nearest in-plane sample for any position outside it, exactly as the scalar
/// `get_clamped` does.
///
/// A row is contiguous, so a clamped *row* index costs nothing: only a column
/// range that leaves the plane needs the staging buffer, which is at most the
/// first and last vector of each row.
///
/// # Safety
/// `V`'s instruction set must be available.
#[inline(always)]
unsafe fn load_row<V: I32x>(data: &[u8], geom: Geometry, x: isize, y: isize) -> V {
    unsafe { load_lanes(data, geom, geom.row(y), x) }
}

/// Gathers column `x` of the `V::LANES` rows starting at `y0` into one vector,
/// with the same edge replication as [`load_row`]. Used by the vertical-edge
/// deblocking kernel, whose filtered positions run down a column.
///
/// # Safety
/// `V`'s instruction set must be available.
#[inline(always)]
unsafe fn gather_column<V: I32x>(data: &[u8], geom: Geometry, x: isize, y0: usize) -> V {
    unsafe {
        let column = geom.clamp_x(x);
        let mut staged = [0u8; MAX_LANES];
        for (lane, slot) in staged.iter_mut().enumerate().take(V::LANES) {
            *slot = data[geom.row((y0 + lane) as isize) + column];
        }
        V::load_u8(&staged)
    }
}

/// Writes the first `count` lanes down column `x`, starting at row `y0`.
///
/// # Safety
/// `V`'s instruction set must be available, `x < geom.width`, and
/// `y0 + count <= geom.height`.
#[inline(always)]
unsafe fn scatter_column<V: I32x>(
    value: V,
    data: &mut [u8],
    geom: Geometry,
    x: usize,
    y0: usize,
    count: usize,
) {
    unsafe {
        let mut staged = [0u8; MAX_LANES];
        value.store_u8_clamped_masked(&mut staged, count);
        for (lane, &byte) in staged.iter().enumerate().take(count) {
            data[(y0 + lane) * geom.stride + x] = byte;
        }
    }
}

/// Loads `V::LANES` `i32` values starting at `base`, staging the load when the
/// buffer's tail is shorter than a whole vector. Only the first `count` lanes
/// are meaningful; the rest are padding the caller never stores.
///
/// # Safety
/// `V`'s instruction set must be available and `base + count <= src.len()`.
#[inline(always)]
unsafe fn load_i32_padded<V: I32x>(src: &[i32], base: usize, count: usize) -> V {
    unsafe {
        if base + V::LANES <= src.len() {
            V::load(&src[base..])
        } else {
            let mut staged = [0i32; MAX_LANES];
            staged[..count].copy_from_slice(&src[base..base + count]);
            V::load(&staged)
        }
    }
}

// ---------------------------------------------------------------------
// Deblocking, spec 7.14.6
// ---------------------------------------------------------------------

/// Samples the widest filter reads: offsets `-7..=6` around the edge
/// (`p6..q6`).
const WIDE_TAPS: usize = 14;
/// Samples the widest filter writes: offsets `-6..=5`, which is a superset of
/// what the 8-tap (`-3..=2`) and narrow (`-2..=1`) filters write.
const WIDE_OUTPUTS: usize = 12;
/// Tap index of the sample at offset `0` (`q0`).
const EDGE: usize = 7;
/// Threshold (8-bit sample domain) used by the flatness checks that gate the
/// wide filters; mirrors `av1_filters::FLAT_THRESH`.
const FLAT_THRESH: i32 = 1;

#[inline(always)]
unsafe fn clamp_i8<V: I32x>(value: V) -> V {
    unsafe { value.clamp(V::splat(-128), V::splat(127)) }
}

#[inline(always)]
unsafe fn clamp_pixel<V: I32x>(value: V) -> V {
    unsafe { value.clamp(V::zero(), V::splat(255)) }
}

/// The two inner-tap gradients `|p1 - p0|` and `|q1 - q0|`, which both
/// `filter_mask` and `hev_mask` are expressed in.
#[inline(always)]
unsafe fn edge_deltas<V: I32x>(p1: V, p0: V, q0: V, q1: V) -> (V, V) {
    unsafe { (p1.sub(p0).abs(), q1.sub(q0).abs()) }
}

/// Vector form of `av1_filters::filter_mask`, given [`edge_deltas`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn filter_mask_lanes<V: I32x>(
    dp: V,
    dq: V,
    p1: V,
    p0: V,
    q0: V,
    q1: V,
    limit: i32,
    blimit: i32,
) -> V {
    unsafe {
        let boundary = p0
            .sub(q0)
            .abs()
            .mul(V::splat(2))
            .add(p1.sub(q1).abs().sra::<1>());
        dp.le(V::splat(limit))
            .and(dq.le(V::splat(limit)))
            .and(boundary.le(V::splat(blimit)))
    }
}

/// Vector form of `av1_filters::filter_mask_wide`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn filter_mask_wide_lanes<V: I32x>(
    p3: V,
    p2: V,
    p1: V,
    q1: V,
    q2: V,
    q3: V,
    limit: i32,
) -> V {
    unsafe {
        let limit = V::splat(limit);
        p3.sub(p2)
            .abs()
            .le(limit)
            .and(p2.sub(p1).abs().le(limit))
            .and(q2.sub(q1).abs().le(limit))
            .and(q3.sub(q2).abs().le(limit))
    }
}

/// Vector form of `av1_filters::flat_mask`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn flat_mask_lanes<V: I32x>(
    p0: V,
    p_a: V,
    p_b: V,
    p_c: V,
    q0: V,
    q_a: V,
    q_b: V,
    q_c: V,
) -> V {
    unsafe {
        let thresh = V::splat(FLAT_THRESH);
        p_a.sub(p0)
            .abs()
            .le(thresh)
            .and(p_b.sub(p0).abs().le(thresh))
            .and(p_c.sub(p0).abs().le(thresh))
            .and(q_a.sub(q0).abs().le(thresh))
            .and(q_b.sub(q0).abs().le(thresh))
            .and(q_c.sub(q0).abs().le(thresh))
    }
}

/// Vector form of `hev_mask` + `narrow_filter`: returns the new
/// `(p1, p0, q0, q1)` with lanes whose boundary `mask` fails left untouched.
/// `dp` / `dq` are [`edge_deltas`], which the caller already needed for `mask`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn narrow_filter_lanes<V: I32x>(
    mask: V,
    dp: V,
    dq: V,
    p1: V,
    p0: V,
    q0: V,
    q1: V,
    thresh: i32,
) -> [V; 4] {
    unsafe {
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

/// Symmetric triangular taper weights for an `N`-tap wide filter: entry
/// `(weights[i][j], denominators[i])` reproduces the `j`-th term and the
/// divisor of `av1_filters::wide_taper_filter`'s output `i`. Only the inner
/// outputs `1..N - 1` are used; the outermost taps only widen the window.
const fn taper_table<const N: usize>() -> ([[i32; N]; N], [i32; N]) {
    let mut weights = [[0i32; N]; N];
    let mut denominators = [0i32; N];
    let n = N as i32;
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < N {
            let distance = i as i32 - j as i32;
            let weight = n - if distance < 0 { -distance } else { distance };
            weights[i][j] = weight;
            denominators[i] += weight;
            j += 1;
        }
        i += 1;
    }
    (weights, denominators)
}

const TAPER8: ([[i32; 8]; 8], [i32; 8]) = taper_table::<8>();
const TAPER14: ([[i32; 14]; 14], [i32; 14]) = taper_table::<14>();

/// One wide-filter output: `(sum(weight * tap) + denominator / 2) /
/// denominator`, the exact integer result the scalar reference computes in
/// `i64`.
///
/// Every tap is an 8-bit sample and every weight is positive, so the numerator
/// is non-negative and at most `255 * denominator` (at most `49_980` for the
/// widest window), which keeps it inside an `i32` lane and inside
/// [`I32x::div_small_nonneg`]'s exact range.
#[inline(always)]
unsafe fn taper_output<V: I32x>(taps: &[V], weights: &[i32], denominator: i32) -> V {
    unsafe {
        let mut sum = V::zero();
        for (&tap, &weight) in taps.iter().zip(weights.iter()) {
            sum = sum.add(tap.mul(V::splat(weight)));
        }
        sum.add(V::splat(denominator / 2))
            .div_small_nonneg(denominator)
    }
}

/// Applies spec §7.14.6's filter cascade to `V::LANES` edge positions at once.
///
/// `taps[k]` holds the samples at offset `k - 7` from the edge (so `taps[6]`
/// and `taps[7]` are `p0`/`q0`), and `sizes` holds each position's filter
/// length from §7.14.5 (4, 8, or 14) — per lane, because a run of positions can
/// straddle transform blocks of different sizes. The result holds the new
/// samples for offsets `-6..=5`; lanes and offsets that the selected filter
/// does not write come back as the original sample, so the caller can store the
/// whole window unconditionally.
#[inline(always)]
unsafe fn deblock_filter_lanes<V: I32x>(
    taps: &[V; WIDE_TAPS],
    sizes: V,
    limit: i32,
    blimit: i32,
    thresh: i32,
) -> [V; WIDE_OUTPUTS] {
    unsafe {
        let (p3, p2, p1, p0) = (
            taps[EDGE - 4],
            taps[EDGE - 3],
            taps[EDGE - 2],
            taps[EDGE - 1],
        );
        let (q0, q1, q2, q3) = (taps[EDGE], taps[EDGE + 1], taps[EDGE + 2], taps[EDGE + 3]);

        let mut out = [p0; WIDE_OUTPUTS];
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = taps[index + 1];
        }

        let (dp, dq) = edge_deltas(p1, p0, q0, q1);
        let mask = filter_mask_lanes(dp, dq, p1, p0, q0, q1, limit, blimit);
        let narrow = narrow_filter_lanes(mask, dp, dq, p1, p0, q0, q1, thresh);
        out[4..8].copy_from_slice(&narrow);

        // Both wide filters need the boundary mask, a filter length that
        // reaches that far, and their flatness gate; the 14-tap filter is a
        // strict refinement of the 8-tap one, matching the scalar cascade's
        // 14 -> 8 -> narrow fallback.
        let wide8 = mask
            .and(sizes.gt(V::splat(7)))
            .and(filter_mask_wide_lanes(p3, p2, p1, q1, q2, q3, limit))
            .and(flat_mask_lanes(p0, p1, p2, p3, q0, q1, q2, q3));
        if !wide8.any() {
            return out;
        }

        let (weights, denominators) = &TAPER8;
        let window = &taps[EDGE - 4..EDGE + 4];
        for (index, slot) in out[3..9].iter_mut().enumerate() {
            let output = index + 1;
            let filtered = taper_output(window, &weights[output], denominators[output]);
            *slot = V::select(wide8, filtered, *slot);
        }

        let wide14 = wide8.and(sizes.gt(V::splat(13))).and(flat_mask_lanes(
            p0,
            taps[EDGE - 5],
            taps[EDGE - 6],
            taps[EDGE - 7],
            q0,
            taps[EDGE + 4],
            taps[EDGE + 5],
            taps[EDGE + 6],
        ));
        if !wide14.any() {
            return out;
        }

        let (weights, denominators) = &TAPER14;
        for (index, slot) in out.iter_mut().enumerate() {
            let output = index + 1;
            let filtered = taper_output(&taps[..], &weights[output], denominators[output]);
            *slot = V::select(wide14, filtered, *slot);
        }
        out
    }
}

/// Stores `count` lanes into row `row` at column `x0`, skipping rows outside
/// the plane (the scalar path's `put` does the same).
///
/// # Safety
/// `V`'s instruction set must be available and `x0 + count <= geom.width`.
#[inline(always)]
unsafe fn store_edge_row<V: I32x>(
    value: V,
    data: &mut [u8],
    geom: Geometry,
    row: isize,
    x0: usize,
    count: usize,
) {
    unsafe {
        if row < 0 || row >= geom.height as isize {
            return;
        }
        let base = row as usize * geom.stride + x0;
        value.store_u8_clamped_masked(&mut data[base..], count);
    }
}

/// The per-lane filter lengths of the `count` positions at `offset`, and
/// whether any of them exceeds the narrow filter.
///
/// `sizes` is either empty (no transform-size metadata, so every edge is
/// narrow) or padded to a whole number of vectors so a chunk always loads.
///
/// # Safety
/// `V`'s instruction set must be available.
#[inline(always)]
unsafe fn chunk_sizes<V: I32x>(sizes: &[i32], offset: usize, count: usize) -> (V, bool) {
    unsafe {
        if sizes.is_empty() {
            return (V::splat(4), false);
        }
        let wide = sizes[offset..offset + count].iter().any(|&size| size > 4);
        (V::load(&sizes[offset..]), wide)
    }
}

/// Filters one chunk of a horizontal edge with the narrow (4-tap) filter.
/// `rows` holds the byte offsets of rows `y - 2 ..= y + 1`, which the caller's
/// edge grid keeps inside the plane.
///
/// # Safety
/// `V`'s instruction set must be available, `lanes <= V::LANES`,
/// `x + lanes <= geom.width`, and `inside` must equal [`columns_inside`] for
/// `x` (whole chunks always satisfy it, so the caller knows it without
/// testing).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn narrow_chunk_horizontal<V: I32x>(
    data: &mut [u8],
    geom: Geometry,
    rows: [usize; 4],
    x: usize,
    lanes: usize,
    inside: bool,
    limit: i32,
    blimit: i32,
    thresh: i32,
) {
    unsafe {
        let taps = rows.map(|row| load_lanes_inside::<V>(data, geom, row, x as isize, inside));
        let (dp, dq) = edge_deltas(taps[0], taps[1], taps[2], taps[3]);
        let mask = filter_mask_lanes(dp, dq, taps[0], taps[1], taps[2], taps[3], limit, blimit);
        let out = narrow_filter_lanes(mask, dp, dq, taps[0], taps[1], taps[2], taps[3], thresh);
        for (row, value) in rows.into_iter().zip(out) {
            value.store_u8_clamped_masked(&mut data[row + x..], lanes);
        }
    }
}

/// Filters one chunk of a vertical edge with the narrow (4-tap) filter. The
/// window is columns `x - 2 ..= x + 1` of one row per lane, which the caller's
/// edge grid keeps inside the plane, so one row offset per lane serves all four
/// taps.
///
/// `full` says the chunk ends before the plane's last row, so it needs no row
/// clamping at all; every chunk but the last of a column satisfies it.
///
/// # Safety
/// `V`'s instruction set must be available, `lanes <= V::LANES`,
/// `y + lanes <= geom.height`, `full == (y + V::LANES <= geom.height)`,
/// `2 <= x` and `x + 1 < geom.width`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn narrow_chunk_vertical<V: I32x>(
    data: &mut [u8],
    geom: Geometry,
    x: usize,
    y: usize,
    lanes: usize,
    full: bool,
    limit: i32,
    blimit: i32,
    thresh: i32,
) {
    unsafe {
        let mut staged = [[0u8; MAX_LANES]; 4];
        for lane in 0..V::LANES {
            let row = if full {
                (y + lane) * geom.stride
            } else {
                geom.row((y + lane) as isize)
            };
            let base = row + x - 2;
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
        let (dp, dq) = edge_deltas(taps[0], taps[1], taps[2], taps[3]);
        let mask = filter_mask_lanes(dp, dq, taps[0], taps[1], taps[2], taps[3], limit, blimit);
        let out = narrow_filter_lanes(mask, dp, dq, taps[0], taps[1], taps[2], taps[3], thresh);
        for (tap, value) in out.into_iter().enumerate() {
            value.store_u8_clamped_masked(&mut staged[tap], V::LANES);
        }
        for lane in 0..lanes {
            let base = (y + lane) * geom.stride + x - 2;
            for (tap, column) in staged.iter().enumerate() {
                data[base + tap] = column[lane];
            }
        }
    }
}

/// Filters `count` consecutive positions of the horizontal edge above row `y`,
/// starting at column `x0`. Each tap lives in its own row, so every load is a
/// contiguous byte run; rows outside the plane replicate the nearest edge row
/// and are never written back.
///
/// Positions whose §7.14.5 filter length (from `sizes`, per position) is the
/// narrow one only touch the four narrow taps, which is why the common case —
/// every chroma edge, and every luma edge of a frame decoded without
/// transform-size metadata — does not pay for the wide filters' fourteen loads.
///
/// # Safety
/// `V`'s instruction set must be available, `x0 + count <= geom.width`,
/// `2 <= y` and `y + 1 < geom.height`, and `sizes` must be empty or at least
/// `count` rounded up to a whole number of vectors long.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn deblock_edge_horizontal<V: I32x>(
    data: &mut [u8],
    geom: Geometry,
    x0: usize,
    y: usize,
    count: usize,
    limit: i32,
    blimit: i32,
    thresh: i32,
    sizes: &[i32],
) {
    unsafe {
        // The narrow window is rows `y - 2 ..= y + 1`, which the caller's edge
        // grid keeps inside the plane, so the narrow path skips the row
        // clamping and the per-row bounds checks the wide filters need.
        let narrow_rows = [
            (y - 2) * geom.stride,
            (y - 1) * geom.stride,
            y * geom.stride,
            (y + 1) * geom.stride,
        ];
        let mut done = 0;
        while done < count {
            let x = x0 + done;
            let lanes = V::LANES.min(count - done);
            let (sizes_lanes, wide) = chunk_sizes::<V>(sizes, done, lanes);
            if !wide {
                narrow_chunk_horizontal::<V>(
                    data,
                    geom,
                    narrow_rows,
                    x,
                    lanes,
                    lanes == V::LANES,
                    limit,
                    blimit,
                    thresh,
                );
                done += lanes;
                continue;
            }
            let mut taps = [V::zero(); WIDE_TAPS];
            for (index, tap) in taps.iter_mut().enumerate() {
                let row = y as isize + index as isize - EDGE as isize;
                *tap = load_row::<V>(data, geom, x as isize, row);
            }
            let out = deblock_filter_lanes(&taps, sizes_lanes, limit, blimit, thresh);
            for (index, value) in out.into_iter().enumerate() {
                let row = y as isize + index as isize - (EDGE as isize - 1);
                store_edge_row(value, data, geom, row, x, lanes);
            }
            done += lanes;
        }
    }
}

/// Filters `count` consecutive positions of the vertical edge left of column
/// `x`, starting at row `y0`, with the same narrow/wide split as
/// [`deblock_edge_horizontal`]. The taps are contiguous *within* a row but the
/// filtered positions run down the column, so each tap is gathered into a
/// vector and scattered back; the per-position arithmetic still runs
/// vectorized.
///
/// # Safety
/// `V`'s instruction set must be available, `y0 + count <= geom.height`,
/// `2 <= x` and `x + 1 < geom.width`, and `sizes` must be empty or at least
/// `count` rounded up to a whole number of vectors long.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn deblock_edge_vertical<V: I32x>(
    data: &mut [u8],
    geom: Geometry,
    x: usize,
    y0: usize,
    count: usize,
    limit: i32,
    blimit: i32,
    thresh: i32,
    sizes: &[i32],
) {
    unsafe {
        let mut done = 0;
        while done < count {
            let y = y0 + done;
            let lanes = V::LANES.min(count - done);
            let (sizes_lanes, wide) = chunk_sizes::<V>(sizes, done, lanes);
            if !wide {
                narrow_chunk_vertical::<V>(
                    data,
                    geom,
                    x,
                    y,
                    lanes,
                    lanes == V::LANES,
                    limit,
                    blimit,
                    thresh,
                );
                done += lanes;
                continue;
            }
            let mut taps = [V::zero(); WIDE_TAPS];
            for (index, tap) in taps.iter_mut().enumerate() {
                let column = x as isize + index as isize - EDGE as isize;
                *tap = gather_column::<V>(data, geom, column, y);
            }
            let out = deblock_filter_lanes(&taps, sizes_lanes, limit, blimit, thresh);
            for (index, value) in out.into_iter().enumerate() {
                let column = x as isize + index as isize - (EDGE as isize - 1);
                if column >= 0 && column < geom.width as isize {
                    scatter_column(value, data, geom, column as usize, y, lanes);
                }
            }
            done += lanes;
        }
    }
}

// ---------------------------------------------------------------------
// CDEF, spec 7.15
// ---------------------------------------------------------------------

/// Accumulates the direction-search `(sum, sum_sq)` of sample differences over
/// one 8x8 block along the `(dr, dc)` offset. Samples outside the plane — the
/// offset itself, and the partial blocks at the right and bottom edges of a
/// frame — replicate the nearest edge sample, exactly as the scalar path's
/// `get_clamped` does, and every block still contributes 64 samples.
///
/// Both accumulators are computed in `i32`: differences are bounded by 255, so
/// the block totals stay under `64 * 255^2`, well inside `i32`, and match the
/// scalar `i64` accumulation exactly.
///
/// # Safety
/// `V`'s instruction set must be available.
pub(crate) unsafe fn cdef_direction_stats<V: I32x>(
    data: &[u8],
    geom: Geometry,
    x0: usize,
    y0: usize,
    dr: i32,
    dc: i32,
) -> (i32, i32) {
    unsafe {
        let mut sum = V::zero();
        let mut sum_sq = V::zero();
        for row in 0..8isize {
            let y = y0 as isize + row;
            let (a_row, b_row) = (geom.row(y), geom.row(y + dr as isize));
            let mut column = 0isize;
            while column < 8 {
                let x = x0 as isize + column;
                let a: V = load_lanes(data, geom, a_row, x);
                let b: V = load_lanes(data, geom, b_row, x + dc as isize);
                let diff = a.sub(b);
                sum = sum.add(diff);
                sum_sq = sum_sq.add(diff.mul(diff));
                column += V::LANES as isize;
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

/// Filters `count` consecutive samples of row `y` starting at column `x0`,
/// writing them to `dst` (the destination row's slice starting at `x0`). Taps
/// that leave the plane replicate the nearest edge sample.
///
/// # Safety
/// `V`'s instruction set must be available and `dst` must hold at least `count`
/// bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cdef_filter_row<V: I32x>(
    data: &[u8],
    geom: Geometry,
    x0: usize,
    y: usize,
    count: usize,
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
        let center_row = geom.row(y as isize);
        let mut done = 0;
        while done < count {
            let x = (x0 + done) as isize;
            let lanes = V::LANES.min(count - done);
            let center: V = load_lanes(data, geom, center_row, x);
            let mut sum = V::zero();
            for &(weight, dr, dc) in primary {
                let row = geom.row(y as isize + dr as isize);
                let tap: V = load_lanes(data, geom, row, x + dc as isize);
                sum = sum.add(
                    constrain_lanes(tap.sub(center), primary_strength, primary_damping_adj)
                        .mul(V::splat(weight)),
                );
            }
            for &(weight, dr, dc) in secondary {
                let row = geom.row(y as isize + dr as isize);
                let tap: V = load_lanes(data, geom, row, x + dc as isize);
                sum = sum.add(
                    constrain_lanes(tap.sub(center), secondary_strength, secondary_damping_adj)
                        .mul(V::splat(weight)),
                );
            }
            let out = if total_weight == 0 {
                center
            } else {
                center.add(sum.add(V::splat(1 << 3)).sra::<4>())
            };
            out.store_u8_clamped_masked(&mut dst[done..], lanes);
            done += lanes;
        }
    }
}

// ---------------------------------------------------------------------
// Loop restoration: Wiener, spec 7.17.2
// ---------------------------------------------------------------------

/// Runs the Wiener horizontal pass for `count` consecutive samples of row `y`
/// starting at column `x0`, writing the rounded intermediate values to `out`.
/// The 7-tap window's reach outside the plane replicates the nearest edge
/// sample.
///
/// # Safety
/// `V`'s instruction set must be available and `out` must hold at least `count`
/// values.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn wiener_horizontal_row<V: I32x>(
    data: &[u8],
    geom: Geometry,
    x0: usize,
    y: usize,
    count: usize,
    taps: [i32; 3],
    center_tap: i32,
    out: &mut [i32],
) {
    unsafe {
        let row = geom.row(y as isize);
        let mut done = 0;
        while done < count {
            let x = (x0 + done) as isize;
            let lanes = V::LANES.min(count - done);
            // The whole 7-tap window shares one in-plane test: away from the
            // left and right borders every tap loads directly.
            let inside = columns_inside::<V>(geom, x - 3) && columns_inside::<V>(geom, x + 3);
            let center: V = load_lanes_inside(data, geom, row, x, inside);
            let mut sum = center.mul(V::splat(center_tap));
            for (k, &tap) in taps.iter().enumerate() {
                let offset = 3 - k as isize;
                let minus: V = load_lanes_inside(data, geom, row, x - offset, inside);
                let plus: V = load_lanes_inside(data, geom, row, x + offset, inside);
                sum = sum.add(minus.add(plus).mul(V::splat(tap)));
            }
            sum.add(V::splat(1 << 2))
                .sra::<3>()
                .store_masked(&mut out[done..], lanes);
            done += lanes;
        }
    }
}

/// Runs the Wiener vertical pass for `count` consecutive columns of `row` over
/// the horizontal pass's `intermediate` buffer (a `width` by `height`
/// restoration region), writing clipped samples to `dst`. Rows outside the
/// region clamp to its first or last row, matching the scalar path.
///
/// # Safety
/// `V`'s instruction set must be available, `column + count <= width`, and
/// `dst` must hold at least `count` bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn wiener_vertical_row<V: I32x>(
    intermediate: &[i32],
    width: usize,
    height: usize,
    row: usize,
    column: usize,
    count: usize,
    taps: [i32; 3],
    center_tap: i32,
    dst: &mut [u8],
) {
    unsafe {
        let interior_rows = row >= 3 && row + 3 < height;
        let mut done = 0;
        while done < count {
            let lanes = V::LANES.min(count - done);
            let base = row * width + column + done;
            let mut sum;
            if interior_rows && column + done + V::LANES <= width {
                // Interior rows reach their taps by adding whole row strides,
                // which is the shape the vast majority of a unit takes.
                sum = V::load(&intermediate[base..]).mul(V::splat(center_tap));
                for (k, &tap) in taps.iter().enumerate() {
                    let offset = (3 - k) * width;
                    let minus = V::load(&intermediate[base - offset..]);
                    let plus = V::load(&intermediate[base + offset..]);
                    sum = sum.add(minus.add(plus).mul(V::splat(tap)));
                }
            } else {
                let clamp_row = |offset: isize| -> usize {
                    (row as isize + offset).clamp(0, height as isize - 1) as usize * width
                        + column
                        + done
                };
                sum = load_i32_padded::<V>(intermediate, base, lanes).mul(V::splat(center_tap));
                for (k, &tap) in taps.iter().enumerate() {
                    let offset = 3 - k as isize;
                    let minus: V = load_i32_padded(intermediate, clamp_row(-offset), lanes);
                    let plus: V = load_i32_padded(intermediate, clamp_row(offset), lanes);
                    sum = sum.add(minus.add(plus).mul(V::splat(tap)));
                }
            }
            sum.add(V::splat(1 << 10))
                .sra::<11>()
                .store_u8_clamped_masked(&mut dst[done..], lanes);
            done += lanes;
        }
    }
}

// ---------------------------------------------------------------------
// Loop restoration: self-guided box statistics, spec 7.17.3
// ---------------------------------------------------------------------

/// Accumulates `(sum, sum_sq)` over the `(2r+1)x(2r+1)` window centered on each
/// of `count` consecutive samples of row `y` starting at column `x0`. Window
/// positions outside the plane replicate the nearest edge sample and still
/// count toward the window, exactly as the scalar `box_stats` does.
///
/// 8-bit samples bound the window sum by `49 * 255` and the sum of squares by
/// `49 * 255^2`, so the scalar reference's `i64` accumulators and these `i32`
/// lanes agree exactly.
///
/// # Safety
/// `V`'s instruction set must be available and `sums` / `sums_sq` must hold at
/// least `count` values.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn box_stats_row<V: I32x>(
    data: &[u8],
    geom: Geometry,
    x0: usize,
    y: usize,
    count: usize,
    radius: usize,
    sums: &mut [i32],
    sums_sq: &mut [i32],
) {
    unsafe {
        let radius = radius as isize;
        let mut done = 0;
        while done < count {
            let x = (x0 + done) as isize;
            let lanes = V::LANES.min(count - done);
            let inside =
                columns_inside::<V>(geom, x - radius) && columns_inside::<V>(geom, x + radius);
            let mut sum = V::zero();
            let mut sum_sq = V::zero();
            for dy in -radius..=radius {
                let row = geom.row(y as isize + dy);
                for dx in -radius..=radius {
                    let value: V = load_lanes_inside(data, geom, row, x + dx, inside);
                    sum = sum.add(value);
                    sum_sq = sum_sq.add(value.mul(value));
                }
            }
            sum.store_masked(&mut sums[done..], lanes);
            sum_sq.store_masked(&mut sums_sq[done..], lanes);
            done += lanes;
        }
    }
}
