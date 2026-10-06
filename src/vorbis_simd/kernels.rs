//! The Vorbis synthesis kernels: a scalar reference for each, and a vector
//! kernel generic over [`F32x`] that is a lane-by-lane transliteration of it.
//!
//! Every vector kernel computes each output element with the same IEEE 754
//! operations, on the same operands, in the same order as its scalar
//! reference, so the two agree bit for bit rather than to a tolerance. Lengths
//! that are not a whole number of vectors finish on the scalar reference.
//!
//! Every generic kernel is `#[inline(always)]`. That is a codegen requirement,
//! not a speed hint: see the note on the entry points in [`super`].

// Symphonia
// Copyright (c) 2019-2022 The Project Symphonia Developers.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The scalar references are Symphonia's (`symphonia-codec-vorbis` 0.5.5 and
// `symphonia-core` 0.5.5's IMDCT), restructured for this crate's vector kernels.

use super::imdct::Plan;
use super::vector::F32x;

// ---------------------------------------------------------------------------
// Overlap-add and output clamping (Vorbis I section 4.3.8)
// ---------------------------------------------------------------------------

/// `out[i] = left[i] * win_rev[i] + right[i] * win[i]`, where `win_rev` is
/// `win` reversed: the falling slope of the previous block's window times its
/// saved right half, plus the rising slope times this block's left half.
pub(crate) fn overlap_add_scalar(
    out: &mut [f32],
    left: &[f32],
    right: &[f32],
    win: &[f32],
    win_rev: &[f32],
) {
    for ((((out, &s0), &s1), &w0), &w1) in out.iter_mut().zip(left).zip(right).zip(win_rev).zip(win)
    {
        *out = s0 * w0 + s1 * w1;
    }
}

#[inline(always)]
pub(crate) unsafe fn overlap_add<V: F32x>(
    out: &mut [f32],
    left: &[f32],
    right: &[f32],
    win: &[f32],
    win_rev: &[f32],
) {
    let whole = out.len() - out.len() % V::LANES;
    let lanes = V::LANES;
    for ((((out, s0), s1), w0), w1) in out[..whole]
        .chunks_exact_mut(lanes)
        .zip(left.chunks_exact(lanes))
        .zip(right.chunks_exact(lanes))
        .zip(win_rev.chunks_exact(lanes))
        .zip(win.chunks_exact(lanes))
    {
        unsafe {
            let (s0, s1, w0, w1) = (V::load(s0), V::load(s1), V::load(w0), V::load(w1));
            s0.mul(w0).add(s1.mul(w1)).store(out);
        }
    }
    overlap_add_scalar(
        &mut out[whole..],
        &left[whole..],
        &right[whole..],
        &win[whole..],
        &win_rev[whole..],
    );
}

/// Clamps every sample to `-1.0..=1.0` exactly as [`f32::clamp`] does,
/// leaving NaN as it is.
pub(crate) fn clamp_unit_scalar(buf: &mut [f32]) {
    for s in buf {
        *s = s.clamp(-1.0, 1.0);
    }
}

#[inline(always)]
pub(crate) unsafe fn clamp_unit<V: F32x>(buf: &mut [f32]) {
    let whole = buf.len() - buf.len() % V::LANES;
    unsafe {
        let low = V::splat(-1.0);
        let high = V::splat(1.0);
        for chunk in buf[..whole].chunks_exact_mut(V::LANES) {
            V::load(chunk).raise_to(low).lower_to(high).store(chunk);
        }
    }
    clamp_unit_scalar(&mut buf[whole..]);
}

// ---------------------------------------------------------------------------
// Inverse channel coupling (Vorbis I section 4.3.5)
// ---------------------------------------------------------------------------

/// Undoes one square-polar coupling step in place.
pub(crate) fn inverse_coupling_scalar(magnitude: &mut [f32], angle: &mut [f32]) {
    for (m, a) in magnitude.iter_mut().zip(angle.iter_mut()) {
        let (new_m, new_a) = if *m > 0.0 {
            if *a > 0.0 {
                (*m, *m - *a)
            } else {
                (*m + *a, *m)
            }
        } else if *a > 0.0 {
            (*m, *m + *a)
        } else {
            (*m - *a, *m)
        };
        *m = new_m;
        *a = new_a;
    }
}

#[inline(always)]
pub(crate) unsafe fn inverse_coupling<V: F32x>(magnitude: &mut [f32], angle: &mut [f32]) {
    let whole = magnitude.len() - magnitude.len() % V::LANES;
    unsafe {
        let zero = V::splat(0.0);
        for (m_out, a_out) in magnitude[..whole]
            .chunks_exact_mut(V::LANES)
            .zip(angle.chunks_exact_mut(V::LANES))
        {
            let m = V::load(m_out);
            let a = V::load(a_out);
            let m_pos = m.gt(zero);
            let a_pos = a.gt(zero);
            let sum = m.add(a);
            let diff = m.sub(a);
            // With `a > 0` the magnitude is kept and the angle becomes
            // `m - a` or `m + a`; otherwise the angle takes the magnitude and
            // the magnitude becomes `m + a` or `m - a`. `m > 0` picks which.
            let when_a_pos = V::select(m_pos, diff, sum);
            let when_a_not_pos = V::select(m_pos, sum, diff);
            V::select(a_pos, m, when_a_not_pos).store(m_out);
            V::select(a_pos, when_a_pos, m).store(a_out);
        }
    }
    inverse_coupling_scalar(&mut magnitude[whole..], &mut angle[whole..]);
}

// ---------------------------------------------------------------------------
// Floor curve times residue (Vorbis I section 4.3.6)
// ---------------------------------------------------------------------------

/// `floor[i] *= residue[i]`.
pub(crate) fn apply_floor_scalar(floor: &mut [f32], residue: &[f32]) {
    for (f, &r) in floor.iter_mut().zip(residue) {
        *f *= r;
    }
}

#[inline(always)]
pub(crate) unsafe fn apply_floor<V: F32x>(floor: &mut [f32], residue: &[f32]) {
    let whole = floor.len() - floor.len() % V::LANES;
    for (f, r) in floor[..whole]
        .chunks_exact_mut(V::LANES)
        .zip(residue.chunks_exact(V::LANES))
    {
        unsafe { V::load(f).mul(V::load(r)).store(f) };
    }
    apply_floor_scalar(&mut floor[whole..], &residue[whole..]);
}

// ---------------------------------------------------------------------------
// Inverse MDCT (Vorbis I section 4.3.7)
//
// An `n`-point spectrum becomes `2n` samples through an `n/2`-point complex
// FFT: a pre-twiddle folds the spectrum into complex input, a radix-2
// decimation-in-time FFT transforms it, and a post-twiddle rotates the result
// back and unfolds it into the four quarters of the output. The arithmetic is
// Symphonia's `Imdct` (the one this replaced), with its recursive FFT laid
// out iteratively over split real and imaginary arrays so a stage is plain
// elementwise work.
// ---------------------------------------------------------------------------

/// Pre-twiddle: for `i` in `0..n/2`, with `e = spec[2i]` and
/// `s = spec[n - 1 - 2i]`, writes `(s * -w.im - e * w.re, e * w.im - s * w.re)`.
///
/// That is Symphonia's `(odd * w.im - even * w.re, odd * w.re + even * w.im)`
/// with `odd = -s` folded into a negated twiddle table, which rounds the same
/// and needs no sign flip per element.
pub(crate) fn pre_twiddle_scalar(plan: &Plan, spec: &[f32], re: &mut [f32], im: &mut [f32]) {
    pre_twiddle_range(plan, spec, re, im, 0);
}

fn pre_twiddle_range(plan: &Plan, spec: &[f32], re: &mut [f32], im: &mut [f32], from: usize) {
    let n = spec.len();
    for i in from..n / 2 {
        let e = spec[2 * i];
        let s = spec[n - 1 - 2 * i];
        re[i] = s * plan.tw_neg_im[i] - e * plan.tw_re[i];
        im[i] = e * plan.tw_im[i] - s * plan.tw_re[i];
    }
}

#[inline(always)]
unsafe fn pre_twiddle<V: F32x>(plan: &Plan, spec: &[f32], re: &mut [f32], im: &mut [f32]) {
    let n = spec.len();
    let n2 = n / 2;
    let lanes = V::LANES;
    let whole = n2 - n2 % lanes;
    let mut i = 0;
    while i < whole {
        unsafe {
            // `spec[2i .. 2i + 2L]` holds the even samples of this run, and
            // `spec[n - 2i - 2L .. n - 2i]` the odd ones in descending order.
            let (e, _) = V::load(&spec[2 * i..]).deinterleave(V::load(&spec[2 * i + lanes..]));
            let tail = n - 2 * i - 2 * lanes;
            let (_, s) = V::load(&spec[tail..]).deinterleave(V::load(&spec[tail + lanes..]));
            let s = s.reverse();
            let c = V::load(&plan.tw_re[i..]);
            let d = V::load(&plan.tw_im[i..]);
            let nd = V::load(&plan.tw_neg_im[i..]);
            s.mul(nd).sub(e.mul(c)).store(&mut re[i..]);
            e.mul(d).sub(s.mul(c)).store(&mut im[i..]);
        }
        i += lanes;
    }
    pre_twiddle_range(plan, spec, re, im, whole);
}

/// Bit-reverses the FFT input into `re`/`im` and runs the first two radix-2
/// stages, whose twiddles are `1` and `-i`, as one radix-4 pass. Pure
/// gathering and trivial butterflies, shared unchanged by every arm.
pub(crate) fn bit_reverse_radix4(
    plan: &Plan,
    src_re: &[f32],
    src_im: &[f32],
    re: &mut [f32],
    im: &mut [f32],
) {
    for (k, quad) in plan.perm.chunks_exact(4).enumerate() {
        let at = |j: usize| {
            let p = quad[j] as usize;
            (src_re[p], src_im[p])
        };
        let (x0r, x0i) = at(0);
        let (x1r, x1i) = at(1);
        let (x2r, x2i) = at(2);
        let (x3r, x3i) = at(3);
        // Stage 1: pairs.
        let (a0r, a0i) = (x0r + x1r, x0i + x1i);
        let (a1r, a1i) = (x0r - x1r, x0i - x1i);
        let (a2r, a2i) = (x2r + x3r, x2i + x3i);
        let (a3r, a3i) = (x2r - x3r, x2i - x3i);
        // Stage 2: `a2` with twiddle 1, `a3` with twiddle -i, which turns
        // `(r, i)` into `(i, -r)`.
        let b = 4 * k;
        re[b] = a0r + a2r;
        im[b] = a0i + a2i;
        re[b + 2] = a0r - a2r;
        im[b + 2] = a0i - a2i;
        re[b + 1] = a1r + a3i;
        im[b + 1] = a1i - a3r;
        re[b + 3] = a1r - a3i;
        im[b + 3] = a1i + a3r;
    }
}

/// One radix-2 stage of half-length `half`: each block of `2 * half` has its
/// odd half rotated by `w_j = exp(-2 pi i j / (2 half))` and butterflied into
/// its even half.
pub(crate) fn fft_stage_scalar(plan: &Plan, re: &mut [f32], im: &mut [f32], half: usize) {
    let (wr, wi) = plan.stage_twiddles(half);
    for base in (0..re.len()).step_by(2 * half) {
        let (e_re, o_re) = re[base..base + 2 * half].split_at_mut(half);
        let (e_im, o_im) = im[base..base + 2 * half].split_at_mut(half);
        for j in 0..half {
            let tr = wr[j] * o_re[j] - wi[j] * o_im[j];
            let ti = wr[j] * o_im[j] + wi[j] * o_re[j];
            let (er, ei) = (e_re[j], e_im[j]);
            e_re[j] = er + tr;
            e_im[j] = ei + ti;
            o_re[j] = er - tr;
            o_im[j] = ei - ti;
        }
    }
}

/// [`fft_stage_scalar`], `V::LANES` butterflies at a time. `half` must be a
/// multiple of `V::LANES`.
#[inline(always)]
unsafe fn fft_stage<V: F32x>(plan: &Plan, re: &mut [f32], im: &mut [f32], half: usize) {
    debug_assert_eq!(half % V::LANES, 0);
    let (wr, wi) = plan.stage_twiddles(half);
    for base in (0..re.len()).step_by(2 * half) {
        let (e_re, o_re) = re[base..base + 2 * half].split_at_mut(half);
        let (e_im, o_im) = im[base..base + 2 * half].split_at_mut(half);
        let lanes = V::LANES;
        for (((((e_re, e_im), o_re), o_im), wr), wi) in e_re
            .chunks_exact_mut(lanes)
            .zip(e_im.chunks_exact_mut(lanes))
            .zip(o_re.chunks_exact_mut(lanes))
            .zip(o_im.chunks_exact_mut(lanes))
            .zip(wr.chunks_exact(lanes))
            .zip(wi.chunks_exact(lanes))
        {
            unsafe {
                let w_re = V::load(wr);
                let w_im = V::load(wi);
                let or = V::load(o_re);
                let oi = V::load(o_im);
                let tr = w_re.mul(or).sub(w_im.mul(oi));
                let ti = w_re.mul(oi).add(w_im.mul(or));
                let er = V::load(e_re);
                let ei = V::load(e_im);
                er.add(tr).store(e_re);
                ei.add(ti).store(e_im);
                er.sub(tr).store(o_re);
                ei.sub(ti).store(o_im);
            }
        }
    }
}

/// Post-twiddle, in place: `x = w * conj(x)`, which is
/// `(w.re * x.re + w.im * x.im, w.im * x.re - w.re * x.im)`.
pub(crate) fn post_twiddle_scalar(plan: &Plan, re: &mut [f32], im: &mut [f32]) {
    post_twiddle_range(plan, re, im, 0);
}

fn post_twiddle_range(plan: &Plan, re: &mut [f32], im: &mut [f32], from: usize) {
    for i in from..re.len() {
        let (c, d) = (plan.tw_re[i], plan.tw_im[i]);
        let (xr, xi) = (re[i], im[i]);
        re[i] = c * xr + d * xi;
        im[i] = d * xr - c * xi;
    }
}

#[inline(always)]
unsafe fn post_twiddle<V: F32x>(plan: &Plan, re: &mut [f32], im: &mut [f32]) {
    let whole = re.len() - re.len() % V::LANES;
    let lanes = V::LANES;
    for (((xr_out, xi_out), c), d) in re[..whole]
        .chunks_exact_mut(lanes)
        .zip(im.chunks_exact_mut(lanes))
        .zip(plan.tw_re.chunks_exact(lanes))
        .zip(plan.tw_im.chunks_exact(lanes))
    {
        unsafe {
            let (c, d) = (V::load(c), V::load(d));
            let (xr, xi) = (V::load(xr_out), V::load(xi_out));
            c.mul(xr).add(d.mul(xi)).store(xr_out);
            d.mul(xr).sub(c.mul(xi)).store(xi_out);
        }
    }
    post_twiddle_range(plan, re, im, whole);
}

/// Unfolds the post-twiddled values into the four `n/2` quarters of the
/// output. With `A` the first `n/4` values and `B` the rest, and `r(X)[j]`
/// meaning `X[n/4 - 1 - j]`, quarter by quarter the even and odd output
/// samples are:
///
/// | quarter | `[2j]` | `[2j + 1]` |
/// | --- | --- | --- |
/// | 0 | `-B.re[j]` | `-r(A.im)[j]` |
/// | 1 | `A.im[j]` | `r(B.re)[j]` |
/// | 2 | `B.im[j]` | `r(A.re)[j]` |
/// | 3 | `A.re[j]` | `r(B.im)[j]` |
///
/// Only moves and sign flips, so it is exact in any arm.
pub(crate) fn unfold_scalar(re: &[f32], im: &[f32], out: &mut [f32]) {
    unfold_range(re, im, out, 0);
}

fn unfold_range(re: &[f32], im: &[f32], out: &mut [f32], from: usize) {
    let n2 = re.len();
    let n4 = n2 / 2;
    let (a_re, b_re) = re.split_at(n4);
    let (a_im, b_im) = im.split_at(n4);
    let (q0, rest) = out.split_at_mut(n2);
    let (q1, rest) = rest.split_at_mut(n2);
    let (q2, q3) = rest.split_at_mut(n2);
    for j in from..n4 {
        let r = n4 - 1 - j;
        q0[2 * j] = -b_re[j];
        q0[2 * j + 1] = -a_im[r];
        q1[2 * j] = a_im[j];
        q1[2 * j + 1] = b_re[r];
        q2[2 * j] = b_im[j];
        q2[2 * j + 1] = a_re[r];
        q3[2 * j] = a_re[j];
        q3[2 * j + 1] = b_im[r];
    }
}

#[inline(always)]
unsafe fn unfold<V: F32x>(re: &[f32], im: &[f32], out: &mut [f32]) {
    let n2 = re.len();
    let n4 = n2 / 2;
    let lanes = V::LANES;
    let whole = n4 - n4 % lanes;
    let (a_re, b_re) = re.split_at(n4);
    let (a_im, b_im) = im.split_at(n4);
    let (q0, rest) = out.split_at_mut(n2);
    let (q1, rest) = rest.split_at_mut(n2);
    let (q2, q3) = rest.split_at_mut(n2);
    let mut j = 0;
    while j < whole {
        unsafe {
            // `r(X)[j .. j + L]` is `X[n/4 - j - L .. n/4 - j]` reversed.
            let r = n4 - j - lanes;
            let q = 2 * j;
            store_interleaved(
                V::load(&b_re[j..]).neg(),
                V::load(&a_im[r..]).reverse().neg(),
                &mut q0[q..],
            );
            store_interleaved(
                V::load(&a_im[j..]),
                V::load(&b_re[r..]).reverse(),
                &mut q1[q..],
            );
            store_interleaved(
                V::load(&b_im[j..]),
                V::load(&a_re[r..]).reverse(),
                &mut q2[q..],
            );
            store_interleaved(
                V::load(&a_re[j..]),
                V::load(&b_im[r..]).reverse(),
                &mut q3[q..],
            );
        }
        j += lanes;
    }
    unfold_range(re, im, out, whole);
}

/// Stores `even[0], odd[0], even[1], odd[1], ...` to the front of `dst`.
#[inline(always)]
unsafe fn store_interleaved<V: F32x>(even: V, odd: V, dst: &mut [f32]) {
    unsafe {
        let (low, high) = even.interleave(odd);
        low.store(dst);
        high.store(&mut dst[V::LANES..]);
    }
}

/// Working buffers for one inverse MDCT, each `n/2` long.
pub(crate) struct Scratch<'a> {
    pub(crate) folded_re: &'a mut [f32],
    pub(crate) folded_im: &'a mut [f32],
    pub(crate) re: &'a mut [f32],
    pub(crate) im: &'a mut [f32],
}

/// The scalar reference inverse MDCT.
pub(crate) fn imdct_scalar(plan: &Plan, spec: &[f32], out: &mut [f32], scratch: Scratch<'_>) {
    let Scratch {
        folded_re,
        folded_im,
        re,
        im,
    } = scratch;
    pre_twiddle_scalar(plan, spec, folded_re, folded_im);
    bit_reverse_radix4(plan, folded_re, folded_im, re, im);
    let mut half = 4;
    while half < re.len() {
        fft_stage_scalar(plan, re, im, half);
        half *= 2;
    }
    post_twiddle_scalar(plan, re, im);
    unfold_scalar(re, im, out);
}

/// The vector inverse MDCT. `V` is the widest vector; `N` is a four-lane
/// vector of the same instruction set, which takes the `half == 4` FFT stage
/// when `V` is wider than that (AVX2), so no stage is left to scalar code.
#[inline(always)]
pub(crate) unsafe fn imdct<V: F32x, N: F32x>(
    plan: &Plan,
    spec: &[f32],
    out: &mut [f32],
    scratch: Scratch<'_>,
) {
    let Scratch {
        folded_re,
        folded_im,
        re,
        im,
    } = scratch;
    unsafe {
        pre_twiddle::<V>(plan, spec, folded_re, folded_im);
        bit_reverse_radix4(plan, folded_re, folded_im, re, im);
        let mut half = 4;
        while half < re.len() {
            if half % V::LANES == 0 {
                fft_stage::<V>(plan, re, im, half);
            } else if half % N::LANES == 0 {
                fft_stage::<N>(plan, re, im, half);
            } else {
                fft_stage_scalar(plan, re, im, half);
            }
            half *= 2;
        }
        post_twiddle::<V>(plan, re, im);
        unfold::<V>(re, im, out);
    }
}
