//! The vector kernels, written once over [`F32x4`] and instantiated per
//! instruction set by the entry points in [`super`].
//!
//! Each kernel mirrors one scalar loop of the encoder and is held to it bit
//! for bit: it issues the same IEEE-754 operations on the same operands in the
//! same order, and only changes *which* iterations run side by side. A loop
//! is only vectorized here when its iterations are independent of each other;
//! anything that carries a value from one iteration to the next stays scalar.
//! The comment on each kernel names the scalar function it must agree with,
//! and `super::tests` compares the two on random and edge-case input.
//!
//! The kernels index through raw pointers wherever the bound follows from
//! the shape of the loop, after checking it up front. Indices read out of
//! data (the bit-reversal table, the bark windows, the companding index) are
//! bounds-checked one by one, so a bad table can only panic.

use super::vector::{F32x4, I32x4};
use crate::vorbis_encoder::floor1::{self, FitSums};
use crate::vorbis_encoder::mdct::{self, MdctLookup};
use crate::vorbis_encoder::psy::NoiseAcc;
use crate::vorbis_encoder::smallft;

// ---------------------------------------------------------------------
// MDCT (mdct.rs `MdctLookup::forward_scalar`)
// ---------------------------------------------------------------------

/// The even-indexed floats of `p[0..16)`, split by residue mod 4:
/// `((p0, p4, p8, p12), (p2, p6, p10, p14))`. Reads `p[0..15)` only, so a
/// stride-4 walk can run right up to the end of its buffer.
#[inline(always)]
unsafe fn stride4_pairs<V: F32x4>(p: *const f32) -> (V, V) {
    unsafe {
        let (e01, _) = V::load(p).uzp(V::load(p.add(4)));
        // `(p11, p12, p13, p14)` with its pairs swapped puts p12 and p14 in
        // the even lanes without reading p15.
        let (e23, _) = V::load(p.add(8)).uzp(V::load(p.add(11)).swap_pairs());
        e01.uzp(e23)
    }
}

/// `(r1 * s + r0 * c, r1 * c - r0 * s)` per pair, for `r = (r0, r1)` and
/// `tc = (c, s)` — the rotation every MDCT butterfly applies.
#[inline(always)]
unsafe fn rotate_butterfly<V: F32x4>(r: V, tc: V) -> V {
    unsafe {
        let q1 = r.dup_odd().mul(tc.swap_pairs());
        let q2 = r.dup_even().mul(tc);
        // `x + (-y)` is IEEE-754's definition of `x - y`, so the odd lanes
        // round exactly as the scalar subtraction does.
        q1.add(q2.neg_odd())
    }
}

/// `(r1 * c + r0 * s, r1 * s - r0 * c)` per pair, for `r = (r1, r0)` and
/// `tc = (c, s)` — the rotation of `mdct_bitreverse`.
#[inline(always)]
unsafe fn rotate_bitreverse<V: F32x4>(r: V, tc: V) -> V {
    unsafe {
        let q1 = r.dup_even().mul(tc);
        let q2 = r.dup_odd().mul(tc.swap_pairs());
        q1.add(q2.neg_odd())
    }
}

/// Which of `mdct_forward`'s three pre-rotation loops is running.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Presum {
    /// `r0 = x[x0+2] + x[x1]`, `r1 = x[x0] + x[x1+2]`
    Add,
    /// `r0 = x[x0+2] - x[x1]`, `r1 = x[x0] - x[x1+2]`
    Sub,
    /// `r0 = -x[x0+2] - x[x1]`, `r1 = -x[x0] - x[x1+2]`
    NegSub,
}

/// Four iterations of one pre-rotation loop: reads `input` stepping `x0` down
/// and `x1` up by 16 and `trig` down from `t`, writes `w2[i..i+8)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn presum_group<V: F32x4>(
    kind: Presum,
    input: *const f32,
    trig: *const f32,
    w2: *mut f32,
    x0: usize,
    x1: usize,
    t: usize,
    i: usize,
) {
    unsafe {
        // Iteration k reads x0-4(k+1) and x0-4(k+1)+2, which are the
        // reversed lanes of a stride-4 walk starting at x0-16.
        let (lo, lo2) = stride4_pairs::<V>(input.add(x0 - 16));
        let (a_x0, a_x0p2) = (lo.reverse(), lo2.reverse());
        let (a_x1, a_x1p2) = stride4_pairs::<V>(input.add(x1));
        let (c, s) = V::load(trig.add(t - 8)).uzp(V::load(trig.add(t - 4)));
        let (c, s) = (c.reverse(), s.reverse());
        let (r0, r1) = match kind {
            Presum::Add => (a_x0p2.add(a_x1), a_x0.add(a_x1p2)),
            Presum::Sub => (a_x0p2.sub(a_x1), a_x0.sub(a_x1p2)),
            Presum::NegSub => (a_x0p2.neg().sub(a_x1), a_x0.neg().sub(a_x1p2)),
        };
        let even = r1.mul(s).add(r0.mul(c));
        let odd = r1.mul(c).sub(r0.mul(s));
        let (lo, hi) = even.zip(odd);
        lo.store(w2.add(i));
        hi.store(w2.add(i + 4));
    }
}

/// `mdct_butterfly_first` (`trigint == 4`) and `mdct_butterfly_generic`.
#[inline(always)]
unsafe fn butterfly_generic<V: F32x4>(
    trig: *const f32,
    x: *mut f32,
    points: usize,
    trigint: usize,
) {
    unsafe {
        let mut x1 = points - 8;
        let mut x2 = points / 2 - 8;
        let mut ti = 0;
        loop {
            let a = x.add(x1);
            let b = x.add(x2);
            // Lanes 0-1 are the pair at +4 (twiddle ti+trigint), lanes 2-3
            // the pair at +6 (twiddle ti).
            let (va, vb) = (V::load(a.add(4)), V::load(b.add(4)));
            va.add(vb).store(a.add(4));
            let tc = V::load_pairs(trig.add(ti + trigint), trig.add(ti));
            rotate_butterfly(va.sub(vb), tc).store(b.add(4));
            // The pairs at +0 and +2 take the third and second twiddles.
            let (va, vb) = (V::load(a), V::load(b));
            va.add(vb).store(a);
            let tc = V::load_pairs(trig.add(ti + 3 * trigint), trig.add(ti + 2 * trigint));
            rotate_butterfly(va.sub(vb), tc).store(b);

            ti += 4 * trigint;
            if x2 < 8 {
                break;
            }
            x1 -= 8;
            x2 -= 8;
        }
    }
}

/// `mdct_bitreverse` over `x[..n]`: one scalar iteration (two index pairs)
/// per vector. The bit-reversal indices are table data, so every pair they
/// name is bounds-checked before it is loaded.
#[inline(always)]
unsafe fn bitreverse<V: F32x4>(m: &MdctLookup, x: &mut [f32]) {
    let n = m.n;
    let (wlo, xs) = x[..n].split_at_mut(n >> 1);
    let xs = &*xs;
    let bit = &m.bitrev[..n >> 2];
    let pair = |b: i32| xs[b as usize..b as usize + 2].as_ptr();
    unsafe {
        let wlo = wlo.as_mut_ptr();
        let trig = m.trig.as_ptr().add(n);
        let half = V::splat(0.5);
        let mut w0 = 0usize;
        let mut w1 = n >> 1;
        let mut t = 0usize;
        loop {
            let v0 = V::load_pairs(pair(bit[t]), pair(bit[t + 2]));
            let v1 = V::load_pairs(pair(bit[t + 1]), pair(bit[t + 3]));
            let sum = v0.add(v1);
            let diff = v0.sub(v1);
            // (r1, r0) = (x0 + x1, x0' - x1') and its rotation (r2, r3)
            let r23 = rotate_bitreverse(sum.blend_odd(diff), V::load(trig.add(t)));
            // (r0, r1) of the second half = ((x0' + x1') * .5, (x0 - x1) * .5)
            let h = diff.blend_odd(sum).swap_pairs().mul(half);
            h.add(r23).store(wlo.add(w0));
            w1 -= 4;
            // (r0 - r2, r3 - r1), the second index pair landing lower
            h.blend_odd(r23)
                .sub(r23.blend_odd(h))
                .swap_halves()
                .store(wlo.add(w1));
            t += 4;
            w0 += 4;
            if w0 >= w1 {
                break;
            }
        }
    }
}

/// [`MdctLookup::forward_scalar`], for `n >= 64` (every Vorbis block size and
/// the envelope's 128-point transform).
#[inline(always)]
pub(super) unsafe fn mdct_forward<V: F32x4>(
    m: &MdctLookup,
    input: &[f32],
    out: &mut [f32],
    w: &mut [f32],
) {
    let n = m.n;
    assert!(n >= 64 && n.is_power_of_two());
    assert!(input.len() >= n && out.len() >= n / 2 && w.len() >= n);
    assert!(m.trig.len() >= n + n / 4 && m.bitrev.len() >= n / 4);
    let n2 = n >> 1;
    let n4 = n >> 2;
    let n8 = n >> 3;
    unsafe {
        let ip = input.as_ptr();
        let tp = m.trig.as_ptr();
        let wp = w.as_mut_ptr();
        let w2 = wp.add(n2);

        let mut x0 = n2 + n4;
        let mut x1 = x0 + 1;
        let mut t = n2;
        let mut i = 0;
        while i < n8 {
            presum_group::<V>(Presum::Add, ip, tp, w2, x0, x1, t, i);
            x0 -= 16;
            x1 += 16;
            t -= 8;
            i += 8;
        }
        x1 = 1;
        while i < n2 - n8 {
            presum_group::<V>(Presum::Sub, ip, tp, w2, x0, x1, t, i);
            x0 -= 16;
            x1 += 16;
            t -= 8;
            i += 8;
        }
        x0 = n;
        while i < n2 {
            presum_group::<V>(Presum::NegSub, ip, tp, w2, x0, x1, t, i);
            x0 -= 16;
            x1 += 16;
            t -= 8;
            i += 8;
        }

        // mdct_butterflies, with the 32-point tail left scalar
        let mut stages = m.log2n - 6;
        if stages > 0 {
            butterfly_generic::<V>(tp, w2, n2, 4);
        }
        let mut stage = 1;
        loop {
            stages -= 1;
            if stages <= 0 {
                break;
            }
            let p = n2 >> stage;
            for j in 0..(1usize << stage) {
                butterfly_generic::<V>(tp, w2.add(p * j), p, 4 << stage);
            }
            stage += 1;
        }
        let tail = core::slice::from_raw_parts_mut(w2, n2);
        for chunk in tail.chunks_exact_mut(32) {
            mdct::butterfly_32(chunk);
        }

        bitreverse::<V>(m, core::slice::from_raw_parts_mut(wp, n));

        // rotate + window
        let scale = V::splat(m.scale);
        let op = out.as_mut_ptr();
        let rot = tp.add(n2);
        let mut i = 0;
        while i < n4 {
            let (we, wo) = V::load(wp.add(2 * i)).uzp(V::load(wp.add(2 * i + 4)));
            let (c, s) = V::load(rot.add(2 * i)).uzp(V::load(rot.add(2 * i + 4)));
            we.mul(c).add(wo.mul(s)).mul(scale).store(op.add(i));
            we.mul(s)
                .sub(wo.mul(c))
                .mul(scale)
                .reverse()
                .store(op.add(n2 - 4 - i));
            i += 4;
        }
    }
}

// ---------------------------------------------------------------------
// Real FFT (smallft.rs)
// ---------------------------------------------------------------------

/// `(wr * re + wi * im, wr * im - wi * re)` per pair, for `w = (wr, wi)` and
/// `x = (re, im)` — the twiddle product of `dradf2`/`dradf4`.
#[inline(always)]
unsafe fn twiddle<V: F32x4>(w: V, x: V) -> V {
    unsafe {
        let q1 = w.dup_even().mul(x);
        let q2 = w.dup_odd().mul(x.swap_pairs());
        q1.add(q2.neg_odd())
    }
}

/// The first loop of [`smallft::dradf4_first`] for `ido == 1`, four rows at a
/// time: each row's four outputs are one column of a 4x4 transpose.
#[inline(always)]
pub(super) unsafe fn dradf4_ido1<V: F32x4>(l1: usize, cc: &[f32], ch: &mut [f32]) {
    assert!(cc.len() >= 4 * l1 && ch.len() >= 4 * l1);
    let t0 = l1;
    let mut k = 0;
    unsafe {
        let cp = cc.as_ptr();
        let hp = ch.as_mut_ptr();
        while k + 4 <= l1 {
            let c0 = V::load(cp.add(k));
            let c1 = V::load(cp.add(t0 + k));
            let c2 = V::load(cp.add(2 * t0 + k));
            let c3 = V::load(cp.add(3 * t0 + k));
            let tr1 = c1.add(c3);
            let tr2 = c0.add(c2);
            let o0 = tr1.add(tr2);
            let o3 = tr2.sub(tr1);
            let o1 = c0.sub(c2);
            let o2 = c3.sub(c1);
            let (a, b) = o0.zip(o1);
            let (c, d) = o2.zip(o3);
            a.low_halves(c).store(hp.add(4 * k));
            a.high_halves(c).store(hp.add(4 * k + 4));
            b.low_halves(d).store(hp.add(4 * k + 8));
            b.high_halves(d).store(hp.add(4 * k + 12));
            k += 4;
        }
    }
    smallft::dradf4_first_rows(1, l1, cc, ch, k);
}

/// The twiddle loop of `dradf4` (`ido > 2`): two butterflies (`i`, `i + 2`)
/// per vector, the odd one out per row through the scalar step.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn dradf4_twiddle<V: F32x4>(
    ido: usize,
    l1: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
    wa2: &[f32],
    wa3: &[f32],
) {
    let t0 = l1 * ido;
    assert!(cc.len() >= 4 * t0 && ch.len() >= 4 * t0);
    assert!(wa1.len() >= ido && wa2.len() >= ido && wa3.len() >= ido);
    let t6 = ido << 1;
    for row in 0..l1 {
        let t1 = row * ido;
        let mut i = 2;
        unsafe {
            let cp = cc.as_ptr();
            let hp = ch.as_mut_ptr();
            while i + 2 < ido {
                let x0 = V::load(cp.add(t1 + i - 1));
                let c2 = twiddle(
                    V::load(wa1.as_ptr().add(i - 2)),
                    V::load(cp.add(t1 + t0 + i - 1)),
                );
                let c3 = twiddle(
                    V::load(wa2.as_ptr().add(i - 2)),
                    V::load(cp.add(t1 + 2 * t0 + i - 1)),
                );
                let c4 = twiddle(
                    V::load(wa3.as_ptr().add(i - 2)),
                    V::load(cp.add(t1 + 3 * t0 + i - 1)),
                );
                // (tr1, ti1) and (tr4, ti4) = (cr4 - cr2, ci2 - ci4)
                let s24 = c2.add(c4);
                let m = c4.blend_odd(c2).sub(c2.blend_odd(c4));
                // (tr2, ti2) and (tr3, ti3)
                let p = x0.add(c3);
                let d = x0.sub(c3);

                let t4 = (t1 << 2) + i;
                let t5 = (t1 << 2) + t6 - i;
                s24.add(p).store(hp.add(t4 - 1));
                // (tr3 - ti4, tr4 - ti3), the second butterfly landing lower
                d.trn1(m).sub(m.trn2(d)).swap_halves().store(hp.add(t5 - 3));
                // (ti4 + tr3, tr4 + ti3)
                m.swap_pairs().add(d).store(hp.add(t4 + t6 - 1));
                // (tr2 - tr1, ti1 - ti2)
                p.blend_odd(s24)
                    .sub(s24.blend_odd(p))
                    .swap_halves()
                    .store(hp.add(t5 + t6 - 3));
                i += 4;
            }
        }
        while i < ido {
            smallft::dradf4_twiddle_step(ido, t0, t1, i, cc, ch, wa1, wa2, wa3);
            i += 2;
        }
    }
}

/// The twiddle loop of `dradf2` (`ido > 2`), two butterflies per vector.
#[inline(always)]
pub(super) unsafe fn dradf2_twiddle<V: F32x4>(
    ido: usize,
    l1: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
) {
    let t0 = l1 * ido;
    assert!(cc.len() >= 2 * t0 && ch.len() >= 2 * t0 && wa1.len() >= ido);
    for row in 0..l1 {
        let t1 = row * ido;
        let t2 = t0 + t1;
        let mut i = 2;
        unsafe {
            let cp = cc.as_ptr();
            let hp = ch.as_mut_ptr();
            while i + 2 < ido {
                let c = twiddle(
                    V::load(wa1.as_ptr().add(i - 2)),
                    V::load(cp.add(t2 + i - 1)),
                );
                let y = V::load(cp.add(t1 + i - 1));
                y.add(c).store(hp.add(2 * t1 + i - 1));
                // (cc[t5-1] - tr2, ti2 - cc[t5]), the second one landing lower
                y.blend_odd(c)
                    .sub(c.blend_odd(y))
                    .swap_halves()
                    .store(hp.add(2 * t1 + 2 * ido - i - 3));
                i += 4;
            }
        }
        while i < ido {
            smallft::dradf2_twiddle_step(ido, t0, t1, i, cc, ch, wa1);
            i += 2;
        }
    }
}

// ---------------------------------------------------------------------
// Noise masking (psy.rs `bark_noise_hybridmp`, `PsyLook::noisemask`)
// ---------------------------------------------------------------------

/// Lanes of `noise_fit`: the least-squares (A, B, D) for four windows.
#[inline(always)]
unsafe fn noise_fit<V: F32x4>(hi: [V; 5], lo: [V; 5], reflect: bool) -> (V, V, V) {
    unsafe {
        let [hn, hx, hxx, hy, hxy] = hi;
        let [ln, lx, lxx, ly, lxy] = lo;
        let (t_n, t_xx, t_y) = if reflect {
            (hn.add(ln), hxx.add(lxx), hy.add(ly))
        } else {
            (hn.sub(ln), hxx.sub(lxx), hy.sub(ly))
        };
        let t_x = hx.sub(lx);
        let t_xy = hxy.sub(lxy);
        let a = t_y.mul(t_xx).sub(t_x.mul(t_xy));
        let b = t_n.mul(t_xy).sub(t_x.mul(t_y));
        let d = t_n.mul(t_xx).sub(t_x.mul(t_x));
        (a, b, d)
    }
}

/// `x` for four consecutive bins: `i as f32` is exact, as the scalar loops'
/// `x += 1.` counter is, for every block size Vorbis has.
#[inline(always)]
unsafe fn bin_x<V: F32x4>(i: usize) -> V {
    unsafe { V::from_array([i as f32, (i + 1) as f32, (i + 2) as f32, (i + 3) as f32]) }
}

/// The (N, X, XX, Y, XY) sums of four bins, one bin per lane. The first four
/// sums of a `repr(C)` entry are one 16-byte load, so four loads and a 4x4
/// transpose replace sixteen scalar gathers; XY is gathered.
///
/// The indices come from the bark tables and the window arithmetic, so each
/// is bounds-checked here rather than trusted.
#[inline(always)]
unsafe fn gather_acc<V: F32x4>(acc: &[NoiseAcc], idx: [usize; 4]) -> [V; 5] {
    let e = idx.map(|i| &acc[i]);
    unsafe {
        let [v0, v1, v2, v3] = e.map(|e| V::load((e as *const NoiseAcc).cast::<f32>()));
        let (a, b) = v0.zip(v1);
        let (c, d) = v2.zip(v3);
        [
            a.low_halves(c),
            a.high_halves(c),
            b.low_halves(d),
            b.high_halves(d),
            V::from_array(e.map(|e| e.xy)),
        ]
    }
}

/// [`psy::noise_bark_scalar`](crate::vorbis_encoder::psy): the bark-window
/// fits of `bark_noise_hybridmp`'s first pass over `start..end`.
#[inline(always)]
pub(super) unsafe fn noise_bark<V: F32x4>(
    acc: &[NoiseAcc],
    b: &[i32],
    start: usize,
    end: usize,
    reflect: bool,
    offset: f32,
    noise: &mut [f32],
) {
    assert!(end <= b.len() && end <= noise.len());
    let mut i = start;
    unsafe {
        let zero = V::splat(0.);
        let off = V::splat(offset);
        while i + 4 <= end {
            let w = [b[i], b[i + 1], b[i + 2], b[i + 3]];
            let hi = w.map(|w| (w & 0xffff) as usize);
            let lo = w.map(|w| (if reflect { -(w >> 16) } else { w >> 16 }) as usize);
            let (a, bb, d) = noise_fit(gather_acc::<V>(acc, hi), gather_acc(acc, lo), reflect);
            let r = a.add(bin_x::<V>(i).mul(bb)).div(d);
            let r = r.select_lt(zero, zero, r);
            r.sub(off).store(noise.as_mut_ptr().add(i));
            i += 4;
        }
    }
    crate::vorbis_encoder::psy::noise_bark_scalar(acc, b, i, end, reflect, offset, noise);
}

/// [`psy::noise_fixed_scalar`](crate::vorbis_encoder::psy): the fixed-width
/// fits of `bark_noise_hybridmp`'s second pass over `start..end`, which lower
/// `noise` wherever they come out below it.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn noise_fixed<V: F32x4>(
    acc: &[NoiseAcc],
    fixed: i32,
    start: usize,
    end: usize,
    reflect: bool,
    offset: f32,
    noise: &mut [f32],
) {
    assert!(end <= noise.len());
    let half = (fixed / 2) as isize;
    // A negative edge wraps to a huge index, which `gather_acc` rejects.
    let edges = |i: usize| {
        let hi = i as isize + half;
        let lo = hi - fixed as isize;
        (hi as usize, (if reflect { -lo } else { lo }) as usize)
    };
    let mut i = start;
    unsafe {
        let off = V::splat(offset);
        while i + 4 <= end {
            let [e0, e1, e2, e3] = [i, i + 1, i + 2, i + 3].map(edges);
            let his = gather_acc::<V>(acc, [e0.0, e1.0, e2.0, e3.0]);
            let los = gather_acc::<V>(acc, [e0.1, e1.1, e2.1, e3.1]);
            let (a, bb, d) = noise_fit(his, los, reflect);
            let r = a.add(bin_x::<V>(i).mul(bb)).div(d).sub(off);
            let np = noise.as_mut_ptr().add(i);
            let cur = V::load(np);
            r.select_lt(cur, r, cur).store(np);
            i += 4;
        }
    }
    crate::vorbis_encoder::psy::noise_fixed_scalar(acc, fixed, i, end, reflect, offset, noise);
}

/// [`psy::noise_extrapolate_scalar`](crate::vorbis_encoder::psy): the tail of
/// either pass, where the last fit (`a`, `b`, `d`) is extrapolated.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn noise_extrapolate<V: F32x4>(
    fit: (f32, f32, f32),
    start: usize,
    end: usize,
    offset: f32,
    lower_only: bool,
    noise: &mut [f32],
) {
    assert!(end <= noise.len());
    let mut i = start;
    unsafe {
        let (a, bb, d) = (V::splat(fit.0), V::splat(fit.1), V::splat(fit.2));
        let zero = V::splat(0.);
        let off = V::splat(offset);
        while i + 4 <= end {
            let r = a.add(bin_x::<V>(i).mul(bb)).div(d);
            let np = noise.as_mut_ptr().add(i);
            if lower_only {
                let r = r.sub(off);
                let cur = V::load(np);
                r.select_lt(cur, r, cur).store(np);
            } else {
                r.select_lt(zero, zero, r).sub(off).store(np);
            }
            i += 4;
        }
    }
    crate::vorbis_encoder::psy::noise_extrapolate_scalar(fit, i, end, offset, lower_only, noise);
}

/// [`psy::noise_compand_scalar`](crate::vorbis_encoder::psy): the residual
/// against the noise fit plus the companding curve indexed by the fit.
#[inline(always)]
pub(super) unsafe fn noise_compand<V: F32x4>(
    logmdct: &[f32],
    work: &[f32],
    logmask: &mut [f32],
    compand: &[f32],
) {
    let n = logmask.len();
    assert!(logmdct.len() >= n && work.len() >= n);
    let levels = compand.len();
    assert!(levels > 0);
    let top = (levels - 1) as f64;
    let mut i = 0;
    unsafe {
        while i + 4 <= n {
            let mp = logmask.as_mut_ptr().add(i);
            let db = V::load(mp).add_f64_trunc_clamp(0.5, top).to_array();
            // Indexed with bounds checks: the clamp above is the only thing
            // keeping these in range, and a 40-entry table makes the checks
            // cheap.
            let curve = V::from_array(db.map(|d| compand[d as usize]));
            V::load(logmdct.as_ptr().add(i))
                .sub(V::load(work.as_ptr().add(i)))
                .add(curve)
                .store(mp);
            i += 4;
        }
    }
    crate::vorbis_encoder::psy::noise_compand_scalar(logmdct, work, logmask, compand, i);
}

// ---------------------------------------------------------------------
// Floor fitting (floor1.rs `accumulate_fit`)
// ---------------------------------------------------------------------

/// [`floor1::accumulate_fit_scalar`]'s sums over bins `x0..=x1`.
///
/// Integer addition is associative, so per-lane partial sums added together
/// at the end give exactly the scalar total.
#[inline(always)]
pub(super) unsafe fn accumulate_fit<V: F32x4>(
    flr: &[f32],
    mdct: &[f32],
    x0: i32,
    x1: i32,
    twofitatten: f32,
) -> FitSums {
    if x1 < x0 {
        return FitSums::default();
    }
    assert!(x0 >= 0);
    let end = (x1 + 1) as usize;
    assert!(flr.len() >= end && mdct.len() >= end);
    let mut i = x0 as usize;
    let mut tail = FitSums::default();
    unsafe {
        let zero = V::I::splat(0);
        let (mut xa, mut ya, mut x2a, mut y2a, mut xya, mut na) =
            (zero, zero, zero, zero, zero, zero);
        let (mut xb, mut yb, mut x2b, mut y2b, mut xyb, mut nb) =
            (zero, zero, zero, zero, zero, zero);
        let scale = V::splat(7.314_285_8);
        let bias = V::splat(1023.5);
        let att = V::splat(twofitatten);
        let four = V::I::splat(4);
        let ones = V::I::splat(-1);
        let mut idx = V::I::from_array([i as i32, i as i32 + 1, i as i32 + 2, i as i32 + 3]);
        while i + 4 <= end {
            let f = V::load(flr.as_ptr().add(i));
            let m = V::load(mdct.as_ptr().add(i));
            let q = f.mul(scale).add(bias).trunc_clamp(1023.);
            let zero_q = q.eq_zero();
            let above = m.add(att).ge_mask(f);
            let in_a = above.and_not(zero_q);
            let in_b = ones.and_not(above).and_not(zero_q);
            // A lane outside a set contributes 0 to each sum and -0 to the
            // count (a mask lane is -1).
            let (ia, qa) = (idx.and(in_a), q.and(in_a));
            xa = xa.add(ia);
            ya = ya.add(qa);
            x2a = x2a.add(ia.mul(ia));
            y2a = y2a.add(qa.mul(qa));
            xya = xya.add(ia.mul(qa));
            na = na.sub(in_a);
            let (ib, qb) = (idx.and(in_b), q.and(in_b));
            xb = xb.add(ib);
            yb = yb.add(qb);
            x2b = x2b.add(ib.mul(ib));
            y2b = y2b.add(qb.mul(qb));
            xyb = xyb.add(ib.mul(qb));
            nb = nb.sub(in_b);
            idx = idx.add(four);
            i += 4;
        }
        if i < end {
            tail = floor1::accumulate_fit_scalar(flr, mdct, i as i32, x1, twofitatten);
        }
        FitSums {
            xa: xa.sum().wrapping_add(tail.xa),
            ya: ya.sum().wrapping_add(tail.ya),
            x2a: x2a.sum().wrapping_add(tail.x2a),
            y2a: y2a.sum().wrapping_add(tail.y2a),
            xya: xya.sum().wrapping_add(tail.xya),
            na: na.sum().wrapping_add(tail.na),
            xb: xb.sum().wrapping_add(tail.xb),
            yb: yb.sum().wrapping_add(tail.yb),
            x2b: x2b.sum().wrapping_add(tail.x2b),
            y2b: y2b.sum().wrapping_add(tail.y2b),
            xyb: xyb.sum().wrapping_add(tail.xyb),
            nb: nb.sum().wrapping_add(tail.nb),
        }
    }
}

// ---------------------------------------------------------------------
// Log spectra (mapping0.rs)
// ---------------------------------------------------------------------

/// [`mapping0::log_mdct_scalar`](crate::vorbis_encoder::mapping0).
#[inline(always)]
pub(super) unsafe fn log_mdct<V: F32x4>(mdct: &[f32], logmdct: &mut [f32]) {
    let n = logmdct.len();
    assert!(mdct.len() >= n);
    let mut j = 0;
    unsafe {
        while j + 4 <= n {
            V::load(mdct.as_ptr().add(j))
                .todb()
                .add_f64(0.345)
                .store(logmdct.as_mut_ptr().add(j));
            j += 4;
        }
    }
    crate::vorbis_encoder::mapping0::log_mdct_scalar(mdct, logmdct, j);
}

/// [`mapping0::log_fft_scalar`](crate::vorbis_encoder::mapping0): bins
/// `1..n/2` of the power spectrum, written in place over the FFT output.
///
/// Bin `k` reads floats `2k-1` and `2k` and writes float `k`, so writing four
/// bins after reading them never overwrites anything a later bin reads.
#[inline(always)]
pub(super) unsafe fn log_fft<V: F32x4>(pcm: &mut [f32], n: usize, scale_db: f32) {
    assert!(pcm.len() >= n);
    let half = n / 2;
    let mut k = 1;
    unsafe {
        let p = pcm.as_mut_ptr();
        let sdb = V::splat(scale_db);
        let h = V::splat(0.5);
        while k + 4 <= half {
            let (re, im) = V::load(p.add(2 * k - 1)).uzp(V::load(p.add(2 * k + 3)));
            let temp = re.mul(re).add(im.mul(im));
            sdb.add(h.mul(temp.todb())).add_f64(0.345).store(p.add(k));
            k += 4;
        }
    }
    crate::vorbis_encoder::mapping0::log_fft_scalar(pcm, n, scale_db, k);
}
