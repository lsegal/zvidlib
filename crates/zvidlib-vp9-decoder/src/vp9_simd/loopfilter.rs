//! The loop filters of `vp9_dec::loopfilter` (`filter4`, `filter8` and the
//! 16-wide filter, with their `filter_mask`, `hev_mask`, `flat_mask4` and
//! `flat_mask5` decisions), one edge position per lane.
//!
//! Positions along an edge never read each other's pixels, so a whole run
//! of an edge is filtered at once. Each lane's branch between the filters is
//! a select between all of their results, which is exactly the one result
//! its branch would have computed. A horizontal edge's taps are rows of
//! consecutive pixels and load directly; a vertical edge's taps sit side by
//! side within each row, so they are read as one 32-bit word per row and
//! split into bytes.

// Loops over vectors index rather than iterate: an iterator adapter or an
// `array::from_fn` closure over vector values is a separate function the
// inliner can leave outside the `#[target_feature]` wrapper, compiled at the
// baseline instruction set (#341).
#![allow(clippy::needless_range_loop)]

pub(crate) use crate::vp9_dec::loopfilter::Thresholds;
use zvidlib_core::simd::vector::I32x;

/// Which filter an edge takes: `lpf4`, `lpf8` or `lpf16`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Taps {
    Four,
    Eight,
    Sixteen,
}

impl Taps {
    /// How many pixels the filter reads on each side of the edge.
    fn reach(self) -> usize {
        match self {
            Taps::Four | Taps::Eight => 4,
            Taps::Sixteen => 8,
        }
    }
}

/// Whether every pixel the vector kernel reads lies inside `0..len`. The
/// 16-wide scalar filter only reads its outer taps when the inner ones are
/// flat, so near a plane's edge the two can differ in what they touch; the
/// scalar path keeps those positions.
pub(super) fn covers(
    len: usize,
    start: isize,
    step: isize,
    along: isize,
    count: usize,
    taps: Taps,
) -> bool {
    let reach = taps.reach() as isize;
    if step <= 0 || along <= 0 || count == 0 {
        return false;
    }
    let first = start - reach * step;
    let last = start + (count as isize - 1) * along + (reach - 1) * step;
    first >= 0 && (last as usize) < len
}

/// `|a - b|`.
#[inline(always)]
unsafe fn d<V: I32x>(a: V, b: V) -> V {
    unsafe { a.sub(b).abs() }
}

/// `signed_char_clamp`.
#[inline(always)]
unsafe fn clamp8<V: I32x>(value: V) -> V {
    unsafe { value.clamp(V::splat(-128), V::splat(127)) }
}

/// Reads taps `range` (a whole number of 4-tap words) of the positions from
/// `base` into `px`, where tap `i` is `i - reach` steps across the edge.
#[inline(always)]
unsafe fn load_taps<V: I32x>(
    data: &[u8],
    base: isize,
    step: isize,
    along: isize,
    reach: usize,
    range: core::ops::Range<usize>,
    px: &mut [V; 16],
) {
    unsafe {
        if along == 1 {
            for i in range {
                let at = base + (i as isize - reach as isize) * step;
                px[i] = V::load_u8(&data[at as usize..]);
            }
        } else {
            let mask = V::splat(0xff);
            for word in range.start / 4..range.end / 4 {
                let at = (base - reach as isize) as usize + 4 * word;
                let bytes = V::load_u32_rows(data, at, along as usize);
                px[4 * word] = bytes.and(mask);
                px[4 * word + 1] = bytes.srl::<8>().and(mask);
                px[4 * word + 2] = bytes.srl::<16>().and(mask);
                px[4 * word + 3] = bytes.srl::<24>();
            }
        }
    }
}

/// `ROUND_POWER_OF_TWO(sum, 3)` of the `filter8` taps.
#[inline(always)]
unsafe fn round3<V: I32x>(sum: V) -> V {
    unsafe { sum.add(V::splat(4)).sra::<3>() }
}

/// The vector form of `lpf4`, `lpf8` and `lpf16`: `count` edge positions
/// from `start`, `step` crossing the edge and `along` moving along it.
/// Returns `false`, having changed nothing, when `count` is not a whole
/// number of vectors or the edge is neither horizontal nor vertical.
#[inline(always)]
#[allow(clippy::too_many_lines)]
pub(super) unsafe fn filter_edge<V: I32x>(
    data: &mut [u8],
    start: isize,
    step: isize,
    along: isize,
    count: usize,
    t: Thresholds,
    taps: Taps,
) -> bool {
    unsafe {
        let lanes = V::LANES;
        let horizontal = along == 1;
        if count % lanes != 0 || !(horizontal || step == 1) {
            return false;
        }
        let reach = taps.reach();
        let ones = V::splat(-1);
        let one = V::splat(1);
        let limit = V::splat(i32::from(t.lim));
        let blimit = V::splat(i32::from(t.mblim));
        let thresh = V::splat(i32::from(t.hev_thr));
        let bias = V::splat(128);

        for chunk in (0..count).step_by(lanes) {
            let base = start + chunk as isize * along;
            // `px[i]` is the pixel `i - reach` steps across the edge, so
            // `px[reach - 1]` is p0 and `px[reach]` is q0. The eight taps
            // nearest the edge are read now; the 16-wide filter's outer
            // eight only once some position turns out to need them.
            let o = reach - 4;
            let mut px = [V::zero(); 16];
            load_taps(data, base, step, along, reach, o..o + 8, &mut px);
            let [p3, p2, p1, p0, q0, q1, q2, q3] = [
                px[o],
                px[o + 1],
                px[o + 2],
                px[o + 3],
                px[o + 4],
                px[o + 5],
                px[o + 6],
                px[o + 7],
            ];

            // filter_mask
            let exceed = d(p3, p2)
                .gt(limit)
                .or(d(p2, p1).gt(limit))
                .or(d(p1, p0).gt(limit))
                .or(d(q1, q0).gt(limit))
                .or(d(q2, q1).gt(limit))
                .or(d(q3, q2).gt(limit))
                .or(d(p0, q0)
                    .add(d(p0, q0))
                    .add(d(p1, q1).sra::<1>())
                    .gt(blimit));
            let mask = exceed.andnot(ones);

            // filter4
            let ps1 = p1.sub(bias);
            let ps0 = p0.sub(bias);
            let qs0 = q0.sub(bias);
            let qs1 = q1.sub(bias);
            let hev = d(p1, p0).gt(thresh).or(d(q1, q0).gt(thresh));
            let mut filter = hev.and(clamp8(ps1.sub(qs1)));
            let step3 = qs0.sub(ps0);
            filter = mask.and(clamp8(filter.add(step3).add(step3).add(step3)));
            let filter1 = clamp8(filter.add(V::splat(4))).sra::<3>();
            let filter2 = clamp8(filter.add(V::splat(3))).sra::<3>();
            let outer = hev.andnot(filter1.add(one).sra::<1>());
            let mut out = px;
            out[o + 4] = clamp8(qs0.sub(filter1)).add(bias);
            out[o + 3] = clamp8(ps0.add(filter2)).add(bias);
            out[o + 5] = clamp8(qs1.sub(outer)).add(bias);
            out[o + 2] = clamp8(ps1.add(outer)).add(bias);
            // The taps the filters applied so far changed.
            let mut changed = o + 2..o + 6;

            // flat_mask4, then filter8 where both it and the filter mask
            // hold. A position that is not flat keeps its filter4 result,
            // so with none flat there is nothing more to compute.
            let flat = d(p1, p0)
                .gt(one)
                .or(d(q1, q0).gt(one))
                .or(d(p2, p0).gt(one))
                .or(d(q2, q0).gt(one))
                .or(d(p3, p0).gt(one))
                .or(d(q3, q0).gt(one))
                .andnot(mask);
            if taps != Taps::Four && flat.any() {
                let seven = [
                    p3.add(p3).add(p3).add(p2).add(p2).add(p1).add(p0).add(q0),
                    p3.add(p3).add(p2).add(p1).add(p1).add(p0).add(q0).add(q1),
                    p3.add(p2).add(p1).add(p0).add(p0).add(q0).add(q1).add(q2),
                    p2.add(p1).add(p0).add(q0).add(q0).add(q1).add(q2).add(q3),
                    p1.add(p0).add(q0).add(q1).add(q1).add(q2).add(q3).add(q3),
                    p0.add(q0).add(q1).add(q2).add(q2).add(q3).add(q3).add(q3),
                ];
                for k in 0..6 {
                    out[o + 1 + k] = V::select(flat, round3(seven[k]), out[o + 1 + k]);
                }
                changed = o + 1..o + 7;

                if taps == Taps::Sixteen {
                    // flat_mask5 on the outer taps, then the 15-tap filter
                    // where it, flat_mask4 and the filter mask all hold.
                    load_taps(data, base, step, along, reach, 0..4, &mut px);
                    load_taps(data, base, step, along, reach, 12..16, &mut px);
                    let mut not_flat2 = V::zero();
                    for k in 0..4 {
                        not_flat2 = not_flat2
                            .or(d(px[3 - k], p0).gt(one))
                            .or(d(px[12 + k], q0).gt(one));
                    }
                    let flat2 = not_flat2.andnot(flat);
                    if flat2.any() {
                        for k in (0..4).chain(12..16) {
                            out[k] = px[k];
                        }
                        // The taps past either end repeat p7 or q7, so the
                        // window sum slides by one tap in and one out.
                        let mut window = V::zero();
                        for k in -7..=7isize {
                            window = window.add(px[(1 + k).clamp(0, 15) as usize]);
                        }
                        for position in 1..15usize {
                            if position > 1 {
                                window = window
                                    .add(px[(position + 7).min(15)])
                                    .sub(px[position.saturating_sub(8)]);
                            }
                            let value = window.add(px[position]).add(V::splat(8)).sra::<4>();
                            out[position] = V::select(flat2, value, out[position]);
                        }
                        changed = 1..15;
                    }
                }
            }

            if horizontal {
                for i in changed {
                    let at = base + (i as isize - reach as isize) * step;
                    out[i].store_u8_clamped(&mut data[at as usize..]);
                }
            } else {
                // Whole words, so the unchanged taps in them are written
                // back as they were read.
                for word in changed.start / 4..changed.end.div_ceil(4) {
                    let at = (base - reach as isize) as usize + 4 * word;
                    out[4 * word]
                        .or(out[4 * word + 1].sll::<8>())
                        .or(out[4 * word + 2].sll::<16>())
                        .or(out[4 * word + 3].sll::<24>())
                        .store_u32_rows(data, at, along as usize);
                }
            }
        }
        true
    }
}
