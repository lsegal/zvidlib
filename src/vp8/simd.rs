//! SSE4.1, AVX2 and NEON kernels for the VP8 decoder's hot loops (issue #568):
//! the inverse DCT and Walsh-Hadamard transforms, the six-tap and bilinear
//! sub-pixel filters, intra prediction, and the loop filter.
//!
//! Every kernel is written once, generic over [`I32x`], and instantiated per
//! instruction set behind a `#[target_feature]` entry point, the way the AV1
//! kernels in [`crate::av1_simd`] are. The entry points are `#[inline(never)]`
//! so each one stays a distinct symbol that the generic kernel is inlined
//! *into*; `.github/scripts/check_simd_target_features.py` then fails the
//! build if a kernel was left standing on its own, compiled at the baseline
//! instruction set with every intrinsic an out-of-line call (issue #341).
//!
//! The scalar functions in [`super::predict`] and [`super::loop_filter`] stay
//! the reference: every kernel here is bit-exact with them, which the tests at
//! the bottom of this file check on randomized and edge-case inputs, so the
//! instruction set only ever changes speed. A dispatcher returns `false` when
//! the active instruction set has no kernel - scalar, and every target other
//! than x86_64 and aarch64, `wasm32` included - and the caller runs the scalar
//! reference.
//!
//! The dispatch site is reported as `vp8_decode` by
//! [`crate::simd::active_by_site`] and follows [`crate::simd::set_override`].
//! The VP8 encoder reconstructs its reference frames through the same
//! functions, so it runs these kernels too.

use std::sync::OnceLock;

use crate::simd::SimdIsa;

/// The instruction set the VP8 kernels run on: the crate-wide override when
/// one is set, otherwise this host's widest, probed once.
#[inline]
#[must_use]
pub(crate) fn active_isa() -> SimdIsa {
    if let Some(isa) = crate::simd::override_isa() {
        return isa;
    }
    static DETECTED: OnceLock<SimdIsa> = OnceLock::new();
    *DETECTED.get_or_init(crate::simd::detected)
}

/// The thresholds of one loop-filter edge, already chosen for macroblock or
/// subblock edges.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeThresholds {
    pub edge_limit: i32,
    pub interior: i32,
    pub hev_threshold: i32,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod kernels {
    //! The generic kernels. Each is `#[inline(always)]` so that it is
    //! compiled with the target feature of the entry point that inlines it.

    use super::EdgeThresholds;
    use crate::av1_simd::vector::{I32x, Transpose4};
    use crate::vp8::tables::{
        B_DC_PRED, B_HD_PRED, B_HE_PRED, B_HU_PRED, B_LD_PRED, B_RD_PRED, B_TM_PRED, B_VE_PRED,
        B_VL_PRED, B_VR_PRED,
    };

    const COS_PI8_SQRT2_MINUS1: i32 = 20091;
    const SIN_PI8_SQRT2: i32 = 35468;

    /// Wraps every lane to 16 bits, as libvpx's `short` intermediates do.
    #[inline(always)]
    unsafe fn wrap16<V: I32x>(value: V) -> V {
        unsafe { value.sll::<16>().sra::<16>() }
    }

    /// `(value * constant) >> 16`. Every product fits in 32 bits because the
    /// inputs are 16-bit.
    #[inline(always)]
    unsafe fn mul_shift16<V: I32x>(value: V, constant: i32) -> V {
        unsafe { value.mul(V::splat(constant)).sra::<16>() }
    }

    /// One 4-point inverse DCT pass over four lanes at once.
    #[inline(always)]
    unsafe fn idct4<V: I32x>(input: [V; 4]) -> [V; 4] {
        unsafe {
            let [i0, i1, i2, i3] = input;
            let a1 = i0.add(i2);
            let b1 = i0.sub(i2);
            let c1 =
                mul_shift16(i1, SIN_PI8_SQRT2).sub(i3.add(mul_shift16(i3, COS_PI8_SQRT2_MINUS1)));
            let d1 = i1
                .add(mul_shift16(i1, COS_PI8_SQRT2_MINUS1))
                .add(mul_shift16(i3, SIN_PI8_SQRT2));
            [a1.add(d1), b1.add(c1), b1.sub(c1), a1.sub(d1)]
        }
    }

    #[inline(always)]
    unsafe fn load_block<V: I32x>(block: &[i16; 16]) -> [V; 4] {
        let mut wide = [0i32; 16];
        for (wide, &value) in wide.iter_mut().zip(block) {
            *wide = i32::from(value);
        }
        unsafe {
            [
                V::load(&wide[0..4]),
                V::load(&wide[4..8]),
                V::load(&wide[8..12]),
                V::load(&wide[12..16]),
            ]
        }
    }

    /// Adds one residual row per vector to the 4x4 block at `offset`.
    #[inline(always)]
    unsafe fn add_rows<V: Transpose4>(
        rows: [V; 4],
        plane: &mut [u8],
        offset: usize,
        stride: usize,
    ) {
        for (row, residual) in rows.into_iter().enumerate() {
            let line = &mut plane[offset + row * stride..offset + row * stride + 4];
            unsafe { V::load_u8(line).add(residual).store_u8_clamped(line) };
        }
    }

    /// [`crate::vp8::predict::idct_add_scalar`]. The first pass runs down the
    /// four columns at once, the second across the four rows after a
    /// transpose.
    #[inline(always)]
    pub(super) unsafe fn idct_add<V: Transpose4>(
        coefficients: &[i16; 16],
        plane: &mut [u8],
        offset: usize,
        stride: usize,
    ) {
        unsafe {
            let [c0, c1, c2, c3] = idct4(load_block::<V>(coefficients));
            let columns = [wrap16(c0), wrap16(c1), wrap16(c2), wrap16(c3)];
            let [r0, r1, r2, r3] = idct4(V::transpose4(columns));
            let round = V::splat(4);
            let residual = V::transpose4([
                wrap16(r0.add(round).sra::<3>()),
                wrap16(r1.add(round).sra::<3>()),
                wrap16(r2.add(round).sra::<3>()),
                wrap16(r3.add(round).sra::<3>()),
            ]);
            add_rows(residual, plane, offset, stride);
        }
    }

    /// [`crate::vp8::predict::idct_dc_add_scalar`].
    #[inline(always)]
    pub(super) unsafe fn idct_dc_add<V: Transpose4>(
        dc: i16,
        plane: &mut [u8],
        offset: usize,
        stride: usize,
    ) {
        unsafe {
            let residual = V::splat((i32::from(dc) + 4) >> 3);
            add_rows([residual; 4], plane, offset, stride);
        }
    }

    /// [`crate::vp8::predict::inverse_walsh_scalar`].
    #[inline(always)]
    pub(super) unsafe fn inverse_walsh<V: Transpose4>(input: &[i16; 16]) -> [i16; 16] {
        unsafe {
            let [i0, i4, i8, i12] = load_block::<V>(input);
            let a1 = i0.add(i12);
            let b1 = i4.add(i8);
            let c1 = i4.sub(i8);
            let d1 = i0.sub(i12);
            let columns = [
                wrap16(a1.add(b1)),
                wrap16(c1.add(d1)),
                wrap16(a1.sub(b1)),
                wrap16(d1.sub(c1)),
            ];
            let [i0, i1, i2, i3] = V::transpose4(columns);
            let a1 = i0.add(i3);
            let b1 = i1.add(i2);
            let c1 = i1.sub(i2);
            let d1 = i0.sub(i3);
            let round = V::splat(3);
            let rows = V::transpose4([
                a1.add(b1).add(round).sra::<3>(),
                c1.add(d1).add(round).sra::<3>(),
                a1.sub(b1).add(round).sra::<3>(),
                d1.sub(c1).add(round).sra::<3>(),
            ]);
            let mut wide = [0i32; 16];
            for (row, lane) in rows.into_iter().enumerate() {
                lane.store(&mut wide[row * 4..row * 4 + 4]);
            }
            wide.map(|value| value as i16)
        }
    }

    /// The two passes of [`crate::vp8::predict::predict_inter`] over the
    /// gathered `window`: the horizontal filter over every window row, then
    /// the vertical filter into `output`. `width` must be a multiple of
    /// `V::LANES`.
    ///
    /// Zero taps are skipped, which is exact, and is most of the bilinear
    /// filter. A whole-sample vertical displacement makes the vertical pass
    /// the identity, so the horizontal pass then writes `output` directly.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(super) unsafe fn filter_block<V: I32x>(
        window: &[u8],
        window_width: usize,
        width: usize,
        height: usize,
        horizontal: &[i32; 6],
        vertical: &[i32; 6],
        output: &mut [u8],
        destination: usize,
        stride: usize,
    ) {
        const IDENTITY: [i32; 6] = [0, 0, 128, 0, 0, 0];
        unsafe {
            if *vertical == IDENTITY {
                for row in 0..height {
                    filter_row::<V>(
                        &window[(row + 2) * window_width..],
                        1,
                        width,
                        horizontal,
                        &mut output[destination + row * stride..],
                    );
                }
                return;
            }

            let mut first = [0u8; 16 * 21];
            for row in 0..height + 5 {
                filter_row::<V>(
                    &window[row * window_width..],
                    1,
                    width,
                    horizontal,
                    &mut first[row * width..(row + 1) * width],
                );
            }
            for row in 0..height {
                filter_row::<V>(
                    &first[row * width..],
                    width,
                    width,
                    vertical,
                    &mut output[destination + row * stride..],
                );
            }
        }
    }

    /// One row of `width` outputs of a six-tap filter whose taps are `step`
    /// apart in `source`: `clamp255((64 + sum(source * tap)) >> 7)`.
    #[inline(always)]
    unsafe fn filter_row<V: I32x>(
        source: &[u8],
        step: usize,
        width: usize,
        taps: &[i32; 6],
        out: &mut [u8],
    ) {
        unsafe {
            let mut column = 0;
            while column < width {
                let mut sum = V::splat(64);
                for (index, &tap) in taps.iter().enumerate() {
                    if tap != 0 {
                        let samples = V::load_u8(&source[index * step + column..]);
                        sum = sum.add(samples.mul(V::splat(tap)));
                    }
                }
                sum.sra::<7>().store_u8_clamped(&mut out[column..]);
                column += V::LANES;
            }
        }
    }

    /// TM prediction of an `n`x`n` block, `n` a multiple of `V::LANES`.
    /// `above` holds the pixel above-left at index 0.
    #[inline(always)]
    pub(super) unsafe fn tm_block<V: I32x>(
        above: &[u8],
        left: &[u8],
        n: usize,
        plane: &mut [u8],
        offset: usize,
        stride: usize,
    ) {
        unsafe {
            let corner = V::splat(i32::from(above[0]));
            let mut top = [V::zero(); 16];
            for (chunk, lane) in top.iter_mut().enumerate().take(n / V::LANES) {
                *lane = V::load_u8(&above[1 + chunk * V::LANES..]).sub(corner);
            }
            for (row, &left) in left.iter().enumerate().take(n) {
                let left = V::splat(i32::from(left));
                let line = &mut plane[offset + row * stride..offset + row * stride + n];
                for (chunk, &top) in top.iter().enumerate().take(n / V::LANES) {
                    top.add(left)
                        .store_u8_clamped(&mut line[chunk * V::LANES..]);
                }
            }
        }
    }

    /// Where a subblock mode takes each of its 16 samples from (raster
    /// order). `A3 + j` is `avg3` centred on edge sample `j + 1`, `A2 + j` is
    /// `avg2` of edge samples `j` and `j + 1`, and `E + j` is edge sample `j`
    /// itself, the edge running `l3, l3, l2, l1, l0, p, a0, .., a7, a7, ..`.
    const A3: u8 = 0;
    const A2: u8 = 16;
    const E: u8 = 32;
    #[rustfmt::skip]
    const SUBBLOCK_SOURCES: [[u8; 16]; 8] = [
        // B_VE_PRED
        [A3 + 5, A3 + 6, A3 + 7, A3 + 8, A3 + 5, A3 + 6, A3 + 7, A3 + 8,
         A3 + 5, A3 + 6, A3 + 7, A3 + 8, A3 + 5, A3 + 6, A3 + 7, A3 + 8],
        // B_HE_PRED
        [A3 + 3, A3 + 3, A3 + 3, A3 + 3, A3 + 2, A3 + 2, A3 + 2, A3 + 2,
         A3 + 1, A3 + 1, A3 + 1, A3 + 1, A3, A3, A3, A3],
        // B_LD_PRED
        [A3 + 6, A3 + 7, A3 + 8, A3 + 9, A3 + 7, A3 + 8, A3 + 9, A3 + 10,
         A3 + 8, A3 + 9, A3 + 10, A3 + 11, A3 + 9, A3 + 10, A3 + 11, A3 + 12],
        // B_RD_PRED
        [A3 + 4, A3 + 5, A3 + 6, A3 + 7, A3 + 3, A3 + 4, A3 + 5, A3 + 6,
         A3 + 2, A3 + 3, A3 + 4, A3 + 5, A3 + 1, A3 + 2, A3 + 3, A3 + 4],
        // B_VR_PRED
        [A2 + 5, A2 + 6, A2 + 7, A2 + 8, A3 + 4, A3 + 5, A3 + 6, A3 + 7,
         A3 + 3, A2 + 5, A2 + 6, A2 + 7, A3 + 2, A3 + 4, A3 + 5, A3 + 6],
        // B_VL_PRED
        [A2 + 6, A2 + 7, A2 + 8, A2 + 9, A3 + 6, A3 + 7, A3 + 8, A3 + 9,
         A2 + 7, A2 + 8, A2 + 9, A3 + 10, A3 + 7, A3 + 8, A3 + 9, A3 + 11],
        // B_HD_PRED
        [A2 + 4, A3 + 4, A3 + 5, A3 + 6, A2 + 3, A3 + 3, A2 + 4, A3 + 4,
         A2 + 2, A3 + 2, A2 + 3, A3 + 3, A2 + 1, A3 + 1, A2 + 2, A3 + 2],
        // B_HU_PRED
        [A2 + 3, A3 + 2, A2 + 2, A3 + 1, A2 + 2, A3 + 1, A2 + 1, A3,
         A2 + 1, A3, E, E, E, E, E, E],
    ];

    /// [`crate::vp8::predict::predict_subblock_scalar`]. TM is computed for
    /// all 16 samples at once; every directional mode reads its samples out
    /// of the edge's `avg2` and `avg3` rows, which are computed a vector at a
    /// time.
    #[inline(always)]
    pub(super) unsafe fn subblock<V: I32x>(
        mode: u8,
        above: &[u8; 9],
        left: &[u8; 4],
        plane: &mut [u8],
        offset: usize,
        stride: usize,
    ) {
        let mut block = [0u8; 16];
        unsafe {
            match mode {
                B_DC_PRED => {
                    let sum: u32 = above[1..5]
                        .iter()
                        .chain(left)
                        .map(|&value| u32::from(value))
                        .sum();
                    block = [((sum + 4) >> 3) as u8; 16];
                }
                B_TM_PRED => {
                    let mut lefts = [0i32; 16];
                    let mut tops = [0i32; 16];
                    for index in 0..16 {
                        lefts[index] = i32::from(left[index >> 2]);
                        tops[index] = i32::from(above[1 + (index & 3)]);
                    }
                    let corner = V::splat(i32::from(above[0]));
                    let mut index = 0;
                    while index < 16 {
                        V::load(&lefts[index..])
                            .add(V::load(&tops[index..]))
                            .sub(corner)
                            .store_u8_clamped(&mut block[index..]);
                        index += V::LANES;
                    }
                }
                B_VE_PRED | B_HE_PRED | B_LD_PRED | B_RD_PRED | B_VR_PRED | B_VL_PRED
                | B_HD_PRED | B_HU_PRED => {
                    let mut edge = [i32::from(above[8]); 24];
                    edge[0] = i32::from(left[3]);
                    for (index, &value) in left.iter().rev().enumerate() {
                        edge[1 + index] = i32::from(value);
                    }
                    for (index, &value) in above.iter().enumerate() {
                        edge[5 + index] = i32::from(value);
                    }
                    // `A3`, `A2` and then the edge itself, as `SUBBLOCK_SOURCES`
                    // indexes them.
                    let mut table = [0i32; 48];
                    let mut index = 0;
                    while index < 16 {
                        let x = V::load(&edge[index..]);
                        let y = V::load(&edge[index + 1..]);
                        let z = V::load(&edge[index + 2..]);
                        let avg3 = x.add(y).add(y).add(z).add(V::splat(2)).sra::<2>();
                        let avg2 = x.add(y).add(V::splat(1)).sra::<1>();
                        avg3.store(&mut table[index..]);
                        avg2.store(&mut table[16 + index..]);
                        index += V::LANES;
                    }
                    table[32..48].copy_from_slice(&edge[..16]);
                    let sources = &SUBBLOCK_SOURCES[usize::from(mode - B_VE_PRED)];
                    for (sample, &source) in block.iter_mut().zip(sources) {
                        *sample = table[usize::from(source)] as u8;
                    }
                }
                _ => unreachable!("subblock intra modes are 0..=9"),
            }
        }
        for (row, line) in block.chunks_exact(4).enumerate() {
            plane[offset + row * stride..offset + row * stride + 4].copy_from_slice(line);
        }
    }

    /// The loop filter of [`crate::vp8::loop_filter`] across one edge for
    /// `V::LANES` segments at once. `p[i]` and `q[i]` are the samples `i + 1`
    /// before and `i` after the edge, one segment per lane.
    #[inline(always)]
    unsafe fn filter_lanes<V: I32x>(
        p: &mut [V; 4],
        q: &mut [V; 4],
        thresholds: EdgeThresholds,
        macroblock_edge: bool,
        simple: bool,
    ) {
        unsafe {
            let bias = V::splat(128);
            let edge_difference = abs_diff(p[0], q[0])
                .add(abs_diff(p[0], q[0]))
                .add(abs_diff(p[1], q[1]).sra::<1>());
            let mut mask = edge_difference.le(V::splat(thresholds.edge_limit));
            if !simple {
                let interior = V::splat(thresholds.interior);
                for side in [&*p, &*q] {
                    for i in 0..3 {
                        mask = mask.and(abs_diff(side[i + 1], side[i]).le(interior));
                    }
                }
            }
            if !mask.any() {
                return;
            }

            let p1 = p[1].sub(bias);
            let p0 = p[0].sub(bias);
            let q0 = q[0].sub(bias);
            let q1 = q[1].sub(bias);
            let outer = clamp_i8(p1.sub(q1));
            let step = q0.sub(p0);
            let step3 = step.add(step).add(step);

            if simple {
                let a = clamp_i8(step3.add(outer));
                let f1 = clamp_i8(a.add(V::splat(4))).sra::<3>();
                let f2 = clamp_i8(a.add(V::splat(3))).sra::<3>();
                q[0] = V::select(mask, s2u(q0.sub(f1)), q[0]);
                p[0] = V::select(mask, s2u(p0.add(f2)), p[0]);
                return;
            }

            let threshold = V::splat(thresholds.hev_threshold);
            let hev = abs_diff(p[1], p[0])
                .gt(threshold)
                .or(abs_diff(q[1], q[0]).gt(threshold));

            if macroblock_edge {
                // High edge variance takes the common adjustment with the
                // outer taps, whose `a` is exactly the macroblock filter's
                // `w`; everywhere else the macroblock filter.
                let w = clamp_i8(step3.add(outer));
                let f1 = clamp_i8(w.add(V::splat(4))).sra::<3>();
                let f2 = clamp_i8(w.add(V::splat(3))).sra::<3>();
                let common_q0 = s2u(q0.sub(f1));
                let common_p0 = s2u(p0.add(f2));
                let a0 = macroblock_tap(w, 27);
                let a1 = macroblock_tap(w, 18);
                let a2 = macroblock_tap(w, 9);
                // Lanes the macroblock filter leaves alone: unfiltered, or
                // high edge variance.
                let unchanged = mask.andnot(V::splat(-1)).or(hev);
                let q2 = q[2].sub(bias);
                let p2 = p[2].sub(bias);
                q[0] = V::select(mask, V::select(hev, common_q0, s2u(q0.sub(a0))), q[0]);
                p[0] = V::select(mask, V::select(hev, common_p0, s2u(p0.add(a0))), p[0]);
                q[1] = V::select(unchanged, q[1], s2u(q1.sub(a1)));
                p[1] = V::select(unchanged, p[1], s2u(p1.add(a1)));
                q[2] = V::select(unchanged, q[2], s2u(q2.sub(a2)));
                p[2] = V::select(unchanged, p[2], s2u(p2.add(a2)));
            } else {
                let a = clamp_i8(step3.add(hev.and(outer)));
                let f1 = clamp_i8(a.add(V::splat(4))).sra::<3>();
                let f2 = clamp_i8(a.add(V::splat(3))).sra::<3>();
                q[0] = V::select(mask, s2u(q0.sub(f1)), q[0]);
                p[0] = V::select(mask, s2u(p0.add(f2)), p[0]);
                // `p1` and `q1` move only where the outer taps were not used.
                let a = f1.add(V::splat(1)).sra::<1>();
                let unchanged = mask.andnot(V::splat(-1)).or(hev);
                q[1] = V::select(unchanged, q[1], s2u(q1.sub(a)));
                p[1] = V::select(unchanged, p[1], s2u(p1.add(a)));
            }
        }
    }

    #[inline(always)]
    unsafe fn abs_diff<V: I32x>(a: V, b: V) -> V {
        unsafe { a.sub(b).abs() }
    }

    #[inline(always)]
    unsafe fn clamp_i8<V: I32x>(value: V) -> V {
        unsafe { value.clamp(V::splat(-128), V::splat(127)) }
    }

    /// A signed sample back to unsigned, saturating.
    #[inline(always)]
    unsafe fn s2u<V: I32x>(value: V) -> V {
        unsafe { clamp_i8(value).add(V::splat(128)) }
    }

    /// `clamp_i8((weight * w + 63) >> 7)`, one of the macroblock filter's
    /// three adjustments.
    #[inline(always)]
    unsafe fn macroblock_tap<V: I32x>(w: V, weight: i32) -> V {
        unsafe { clamp_i8(w.mul(V::splat(weight)).add(V::splat(63)).sra::<7>()) }
    }

    /// Filters `count` segments of a horizontal edge (the samples across it
    /// are a row apart, the segments consecutive). `count` must be a multiple
    /// of `V::LANES`.
    #[inline(always)]
    pub(super) unsafe fn filter_horizontal_edge<V: I32x>(
        data: &mut [u8],
        at: usize,
        stride: usize,
        count: usize,
        macroblock_edge: bool,
        thresholds: EdgeThresholds,
        simple: bool,
    ) {
        unsafe {
            let mut segment = 0;
            while segment < count {
                let base = at + segment;
                let mut p = [V::zero(); 4];
                let mut q = [V::zero(); 4];
                for i in 0..4 {
                    p[i] = V::load_u8(&data[base - (i + 1) * stride..]);
                    q[i] = V::load_u8(&data[base + i * stride..]);
                }
                filter_lanes(&mut p, &mut q, thresholds, macroblock_edge, simple);
                for i in 0..3 {
                    p[i].store_u8_clamped(&mut data[base - (i + 1) * stride..]);
                    q[i].store_u8_clamped(&mut data[base + i * stride..]);
                }
                segment += V::LANES;
            }
        }
    }

    /// Filters `count` segments of a vertical edge (the samples across it
    /// are consecutive, the segments a row apart), reading and writing each
    /// side of a segment as one 32-bit word. `count` must be a multiple of
    /// `V::LANES`.
    #[inline(always)]
    pub(super) unsafe fn filter_vertical_edge<V: I32x>(
        data: &mut [u8],
        at: usize,
        stride: usize,
        count: usize,
        macroblock_edge: bool,
        thresholds: EdgeThresholds,
        simple: bool,
    ) {
        unsafe {
            let byte = V::splat(0xff);
            let mut segment = 0;
            while segment < count {
                let base = at + segment * stride;
                // `before` holds p3, p2, p1, p0 from its low byte up, `after`
                // q0, q1, q2, q3.
                let before = V::load_u32_rows(data, base - 4, stride);
                let after = V::load_u32_rows(data, base, stride);
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
                filter_lanes(&mut p, &mut q, thresholds, macroblock_edge, simple);
                p[3].or(p[2].sll::<8>())
                    .or(p[1].sll::<16>())
                    .or(p[0].sll::<24>())
                    .store_u32_rows(data, base - 4, stride);
                q[0].or(q[1].sll::<8>())
                    .or(q[2].sll::<16>())
                    .or(q[3].sll::<24>())
                    .store_u32_rows(data, base, stride);
                segment += V::LANES;
            }
        }
    }
}

/// Generates the three per-instruction-set entry points of one kernel.
/// `sse` and `avx2` name the vector type each x86_64 entry point instantiates
/// the kernel with: the 4x4 kernels have no 8-lane shape, so their AVX2 entry
/// point runs the 4-lane kernel compiled with AVX2 enabled.
macro_rules! entry_points {
    (
        $(#[$meta:meta])*
        fn [$sse_name:ident, $avx_name:ident, $neon_name:ident](
            $($arg:ident : $ty:ty),* $(,)?
        ) $(-> $ret:ty)? = $kernel:ident, avx2 = $avx_vector:ident;
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse4.1")]
        #[inline(never)]
        $(#[$meta])*
        unsafe fn $sse_name($($arg: $ty),*) $(-> $ret)? {
            unsafe { kernels::$kernel::<crate::av1_simd::vector::Sse4>($($arg),*) }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[inline(never)]
        $(#[$meta])*
        unsafe fn $avx_name($($arg: $ty),*) $(-> $ret)? {
            unsafe { kernels::$kernel::<crate::av1_simd::vector::$avx_vector>($($arg),*) }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline(never)]
        $(#[$meta])*
        unsafe fn $neon_name($($arg: $ty),*) $(-> $ret)? {
            unsafe { kernels::$kernel::<crate::av1_simd::vector::Neon>($($arg),*) }
        }
    };
}

entry_points! {
    fn [idct_add_sse41, idct_add_avx2, idct_add_neon](
        coefficients: &[i16; 16], plane: &mut [u8], offset: usize, stride: usize,
    ) = idct_add, avx2 = Sse4;
}
entry_points! {
    fn [idct_dc_add_sse41, idct_dc_add_avx2, idct_dc_add_neon](
        dc: i16, plane: &mut [u8], offset: usize, stride: usize,
    ) = idct_dc_add, avx2 = Sse4;
}
entry_points! {
    fn [inverse_walsh_sse41, inverse_walsh_avx2, inverse_walsh_neon](
        input: &[i16; 16],
    ) -> [i16; 16] = inverse_walsh, avx2 = Sse4;
}
entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [filter_block_sse41, filter_block_avx2, filter_block_neon](
        window: &[u8], window_width: usize, width: usize, height: usize,
        horizontal: &[i32; 6], vertical: &[i32; 6],
        output: &mut [u8], destination: usize, stride: usize,
    ) = filter_block, avx2 = Avx2;
}
entry_points! {
    #[allow(clippy::too_many_arguments)]
    fn [filter_block4_sse41, filter_block4_avx2, filter_block4_neon](
        window: &[u8], window_width: usize, width: usize, height: usize,
        horizontal: &[i32; 6], vertical: &[i32; 6],
        output: &mut [u8], destination: usize, stride: usize,
    ) = filter_block, avx2 = Sse4;
}
entry_points! {
    fn [tm_block_sse41, tm_block_avx2, tm_block_neon](
        above: &[u8], left: &[u8], n: usize, plane: &mut [u8], offset: usize, stride: usize,
    ) = tm_block, avx2 = Avx2;
}
entry_points! {
    fn [subblock_sse41, subblock_avx2, subblock_neon](
        mode: u8, above: &[u8; 9], left: &[u8; 4], plane: &mut [u8], offset: usize, stride: usize,
    ) = subblock, avx2 = Avx2;
}
entry_points! {
    fn [horizontal_edge_sse41, horizontal_edge_avx2, horizontal_edge_neon](
        data: &mut [u8], at: usize, stride: usize, count: usize,
        macroblock_edge: bool, thresholds: EdgeThresholds, simple: bool,
    ) = filter_horizontal_edge, avx2 = Avx2;
}
entry_points! {
    fn [vertical_edge_sse41, vertical_edge_avx2, vertical_edge_neon](
        data: &mut [u8], at: usize, stride: usize, count: usize,
        macroblock_edge: bool, thresholds: EdgeThresholds, simple: bool,
    ) = filter_vertical_edge, avx2 = Avx2;
}

/// Runs the entry point for `isa`, evaluating to `true`, or to `false` when
/// `isa` has no kernel here.
macro_rules! dispatch {
    ($isa:expr, [$sse:ident, $avx:ident, $neon:ident]($($arg:expr),* $(,)?)) => {
        match $isa {
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Sse41 => {
                unsafe { $sse($($arg),*) };
                true
            }
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx2 => {
                unsafe { $avx($($arg),*) };
                true
            }
            #[cfg(target_arch = "aarch64")]
            SimdIsa::Neon => {
                unsafe { $neon($($arg),*) };
                true
            }
            _ => {
                // Only the arms above use the arguments, and a target with
                // none of them compiles none of the arms.
                let _ = ($(&$arg,)*);
                false
            }
        }
    };
}

// Every `isa` below has passed through `active_isa`, whose value is either
// this host's detected instruction set or an override `simd::set_override`
// has already clamped to what the host can execute. That is the safety
// argument for every `unsafe` call the dispatchers make.

/// The vector [`crate::vp8::predict::idct_add_scalar`].
pub(crate) fn idct_add(
    isa: SimdIsa,
    coefficients: &[i16; 16],
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) -> bool {
    dispatch!(
        isa,
        [idct_add_sse41, idct_add_avx2, idct_add_neon](coefficients, plane, offset, stride)
    )
}

/// The vector [`crate::vp8::predict::idct_dc_add_scalar`].
pub(crate) fn idct_dc_add(
    isa: SimdIsa,
    dc: i16,
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) -> bool {
    dispatch!(
        isa,
        [idct_dc_add_sse41, idct_dc_add_avx2, idct_dc_add_neon](dc, plane, offset, stride)
    )
}

/// The vector [`crate::vp8::predict::inverse_walsh_scalar`].
pub(crate) fn inverse_walsh(isa: SimdIsa, input: &[i16; 16]) -> Option<[i16; 16]> {
    match isa {
        #[cfg(target_arch = "x86_64")]
        SimdIsa::Sse41 => Some(unsafe { inverse_walsh_sse41(input) }),
        #[cfg(target_arch = "x86_64")]
        SimdIsa::Avx2 => Some(unsafe { inverse_walsh_avx2(input) }),
        #[cfg(target_arch = "aarch64")]
        SimdIsa::Neon => Some(unsafe { inverse_walsh_neon(input) }),
        _ => {
            let _ = input;
            None
        }
    }
}

/// The sub-pixel filter passes of [`crate::vp8::predict::predict_inter`]
/// over a gathered `window` of `width + 5` by `height + 5` samples. `width`
/// is 4, 8 or 16.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_block(
    isa: SimdIsa,
    window: &[u8],
    window_width: usize,
    width: usize,
    height: usize,
    horizontal: &[i32; 6],
    vertical: &[i32; 6],
    output: &mut [u8],
    destination: usize,
    stride: usize,
) -> bool {
    debug_assert!(matches!(width, 4 | 8 | 16) && height <= 16);
    if width < 8 {
        dispatch!(
            isa,
            [filter_block4_sse41, filter_block4_avx2, filter_block4_neon](
                window,
                window_width,
                width,
                height,
                horizontal,
                vertical,
                output,
                destination,
                stride,
            )
        )
    } else {
        dispatch!(
            isa,
            [filter_block_sse41, filter_block_avx2, filter_block_neon](
                window,
                window_width,
                width,
                height,
                horizontal,
                vertical,
                output,
                destination,
                stride,
            )
        )
    }
}

/// TM prediction of an 8x8 or 16x16 block; `above` holds the pixel
/// above-left at index 0.
pub(crate) fn tm_block(
    isa: SimdIsa,
    above: &[u8],
    left: &[u8],
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) -> bool {
    let n = left.len();
    debug_assert!(matches!(n, 8 | 16) && above.len() > n);
    dispatch!(
        isa,
        [tm_block_sse41, tm_block_avx2, tm_block_neon](above, left, n, plane, offset, stride)
    )
}

/// The vector [`crate::vp8::predict::predict_subblock_scalar`].
pub(crate) fn subblock(
    isa: SimdIsa,
    mode: u8,
    above: &[u8; 9],
    left: &[u8; 4],
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) -> bool {
    dispatch!(
        isa,
        [subblock_sse41, subblock_avx2, subblock_neon](mode, above, left, plane, offset, stride)
    )
}

/// The loop filter across `count` (8 or 16) segments of one edge. `step` is
/// the distance between samples across the edge: `1` for a vertical edge,
/// `stride` for a horizontal one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_edge(
    isa: SimdIsa,
    data: &mut [u8],
    at: usize,
    step: usize,
    stride: usize,
    count: usize,
    macroblock_edge: bool,
    thresholds: EdgeThresholds,
    simple: bool,
) -> bool {
    debug_assert!(matches!(count, 8 | 16));
    if step == 1 {
        dispatch!(
            isa,
            [vertical_edge_sse41, vertical_edge_avx2, vertical_edge_neon](
                data,
                at,
                stride,
                count,
                macroblock_edge,
                thresholds,
                simple,
            )
        )
    } else {
        dispatch!(
            isa,
            [
                horizontal_edge_sse41,
                horizontal_edge_avx2,
                horizontal_edge_neon
            ](data, at, stride, count, macroblock_edge, thresholds, simple,)
        )
    }
}

#[cfg(test)]
mod tests;
