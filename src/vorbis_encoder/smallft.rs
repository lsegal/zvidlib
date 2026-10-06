//! Port of the forward half of smallft.c (`drft_init`, `drft_forward` and the
//! radix-2/4 passes). The encoder only ever transforms power-of-two block
//! sizes, which factor into 4s and 2s, so the generic odd-radix pass
//! (`dradfg`) is never reached and is not ported.

use crate::simd::SimdIsa;

/// Port of `drft_lookup` (smallft.h).
#[derive(Debug, Clone)]
pub(crate) struct DrftLookup {
    n: usize,
    /// `trigcache + n` in C (the first n floats of the C trigcache are the
    /// scratch buffer, which this port passes separately)
    wa: Vec<f32>,
    ifac: [i32; 32],
}

impl DrftLookup {
    /// Port of smallft.c `drft_init` / `fdrffti` / `drfti1`.
    pub(crate) fn new(n: usize) -> Self {
        let mut l = DrftLookup {
            n,
            wa: vec![0f32; 2 * n],
            ifac: [0; 32],
        };
        if n > 1 {
            drfti1(n as i32, &mut l.wa, &mut l.ifac);
        }
        l
    }

    /// Port of smallft.c `drft_forward`. `ch` is scratch of at least n floats.
    /// Runs the `isa` kernels of [`super::simd`], which agree with the scalar
    /// passes bit for bit.
    pub(crate) fn forward(&self, isa: SimdIsa, data: &mut [f32], ch: &mut [f32]) {
        if self.n == 1 {
            return;
        }
        drftf1(
            isa,
            self.n as i32,
            data,
            &mut ch[..self.n],
            &self.wa,
            &self.ifac,
        );
    }
}

/// Port of smallft.c `drfti1`.
fn drfti1(n: i32, wa: &mut [f32], ifac: &mut [i32; 32]) {
    const NTRYH: [i32; 4] = [4, 2, 3, 5];
    let tpi: f32 = 6.283_185_5;
    let mut ntry = 0;
    let mut j: i32 = -1;
    let mut nl = n;
    let mut nf = 0i32;

    'l101: loop {
        j += 1;
        if j < 4 {
            ntry = NTRYH[j as usize];
        } else {
            ntry += 2;
        }
        loop {
            let nq = nl / ntry;
            let nr = nl - ntry * nq;
            if nr != 0 {
                continue 'l101;
            }
            nf += 1;
            ifac[(nf + 1) as usize] = ntry;
            nl = nq;
            if ntry == 2 && nf != 1 {
                for i in 1..nf {
                    let ib = nf - i + 1;
                    ifac[(ib + 1) as usize] = ifac[ib as usize];
                }
                ifac[2] = 2;
            }
            if nl == 1 {
                break 'l101;
            }
        }
    }
    ifac[0] = n;
    ifac[1] = nf;
    let argh = tpi / n as f32;
    let mut is = 0usize;
    let nfm1 = nf - 1;
    let mut l1 = 1;

    if nfm1 == 0 {
        return;
    }
    for k1 in 0..nfm1 {
        let ip = ifac[(k1 + 2) as usize];
        let mut ld = 0;
        let l2 = l1 * ip;
        let ido = n / l2;
        let ipm = ip - 1;
        for _ in 0..ipm {
            ld += l1;
            let mut i = is;
            let argld = ld as f32 * argh;
            let mut fi = 0f32;
            let mut ii = 2;
            while ii < ido {
                fi += 1.;
                let arg = fi * argld;
                wa[i] = f64::from(arg).cos() as f32;
                wa[i + 1] = f64::from(arg).sin() as f32;
                i += 2;
                ii += 2;
            }
            is += ido as usize;
        }
        l1 = l2;
    }
}

/// Port of smallft.c `dradf2`. The twiddle loop runs the `isa` kernels of
/// [`super::simd`], which agree with [`dradf2_twiddle_scalar`] bit for bit.
fn dradf2(isa: SimdIsa, ido: usize, l1: usize, cc: &[f32], ch: &mut [f32], wa1: &[f32]) {
    let mut t1 = 0;
    let t0 = l1 * ido;
    let mut t2 = t0;
    let t3 = ido << 1;
    for _ in 0..l1 {
        ch[t1 << 1] = cc[t1] + cc[t2];
        ch[(t1 << 1) + t3 - 1] = cc[t1] - cc[t2];
        t1 += ido;
        t2 += ido;
    }

    if ido < 2 {
        return;
    }
    if ido != 2 {
        super::simd::dradf2_twiddle(isa, ido, l1, cc, ch, wa1);
        if ido % 2 == 1 {
            return;
        }
    }

    // L105
    t1 = ido;
    let mut t3 = ido - 1;
    let mut t2 = t3 + t0;
    for _ in 0..l1 {
        ch[t1] = -cc[t2];
        ch[t1 - 1] = cc[t3];
        t1 += ido << 1;
        t2 += ido;
        t3 += ido;
    }
}

/// The twiddle loop of `dradf2` (`ido > 2`), one butterfly per `i`.
pub(super) fn dradf2_twiddle_scalar(
    ido: usize,
    l1: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
) {
    let t0 = l1 * ido;
    for row in 0..l1 {
        let mut i = 2;
        while i < ido {
            dradf2_twiddle_step(ido, t0, row * ido, i, cc, ch, wa1);
            i += 2;
        }
    }
}

/// Butterfly `i` of row `t1 / ido` of `dradf2`'s twiddle loop.
#[inline(always)]
pub(super) fn dradf2_twiddle_step(
    ido: usize,
    t0: usize,
    t1: usize,
    i: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
) {
    let t3 = t0 + t1 + i;
    let t4 = (t1 << 1) + (ido << 1) - i;
    let t5 = t1 + i;
    let t6 = (t1 << 1) + i;
    let tr2 = wa1[i - 2] * cc[t3 - 1] + wa1[i - 1] * cc[t3];
    let ti2 = wa1[i - 2] * cc[t3] - wa1[i - 1] * cc[t3 - 1];
    ch[t6] = cc[t5] + ti2;
    ch[t4] = ti2 - cc[t5];
    ch[t6 - 1] = cc[t5 - 1] + tr2;
    ch[t4 - 1] = cc[t5 - 1] - tr2;
}

/// Port of smallft.c `dradf4`. The `ido == 1` first loop and the twiddle
/// loop run the `isa` kernels of [`super::simd`], which agree with
/// [`dradf4_first_rows`] and [`dradf4_twiddle_scalar`] bit for bit.
#[allow(clippy::too_many_arguments)]
fn dradf4(
    isa: SimdIsa,
    ido: usize,
    l1: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
    wa2: &[f32],
    wa3: &[f32],
) {
    let hsqt2: f32 = 0.707_106_77;
    let t0 = l1 * ido;

    if ido == 1 {
        super::simd::dradf4_ido1(isa, l1, cc, ch);
    } else {
        dradf4_first_rows(ido, l1, cc, ch, 0);
    }

    if ido < 2 {
        return;
    }
    if ido != 2 {
        super::simd::dradf4_twiddle(isa, ido, l1, cc, ch, wa1, wa2, wa3);
        if ido & 1 != 0 {
            return;
        }
    }

    // L105
    let mut t1 = t0 + ido - 1;
    let mut t2 = t1 + (t0 << 1);
    let t3 = ido << 2;
    let mut t4 = ido;
    let t5 = ido << 1;
    let mut t6 = ido;

    for _ in 0..l1 {
        let ti1 = -hsqt2 * (cc[t1] + cc[t2]);
        let tr1 = hsqt2 * (cc[t1] - cc[t2]);

        ch[t4 - 1] = tr1 + cc[t6 - 1];
        ch[t4 + t5 - 1] = cc[t6 - 1] - tr1;

        ch[t4] = ti1 - cc[t1 + t0];
        ch[t4 + t5] = ti1 + cc[t1 + t0];

        t1 += ido;
        t2 += ido;
        t4 += t3;
        t6 += ido;
    }
}

/// The first loop of `dradf4`, from row `first` on.
pub(super) fn dradf4_first_rows(ido: usize, l1: usize, cc: &[f32], ch: &mut [f32], first: usize) {
    let t0 = l1 * ido;

    let mut t3 = first * ido;
    let mut t1 = t0 + t3;
    let mut t4 = (t0 << 1) + t3;
    let mut t2 = t0 + (t0 << 1) + t3;

    for _ in first..l1 {
        let tr1 = cc[t1] + cc[t2];
        let tr2 = cc[t3] + cc[t4];

        let mut t5 = t3 << 2;
        ch[t5] = tr1 + tr2;
        ch[(ido << 2) + t5 - 1] = tr2 - tr1;
        t5 += ido << 1;
        ch[t5 - 1] = cc[t3] - cc[t4];
        ch[t5] = cc[t2] - cc[t1];

        t1 += ido;
        t2 += ido;
        t3 += ido;
        t4 += ido;
    }
}

/// The twiddle loop of `dradf4` (`ido > 2`), one butterfly per `i`.
#[allow(clippy::too_many_arguments)]
pub(super) fn dradf4_twiddle_scalar(
    ido: usize,
    l1: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
    wa2: &[f32],
    wa3: &[f32],
) {
    let t0 = l1 * ido;
    for row in 0..l1 {
        let mut i = 2;
        while i < ido {
            dradf4_twiddle_step(ido, t0, row * ido, i, cc, ch, wa1, wa2, wa3);
            i += 2;
        }
    }
}

/// Butterfly `i` of row `t1 / ido` of `dradf4`'s twiddle loop.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn dradf4_twiddle_step(
    ido: usize,
    t0: usize,
    t1: usize,
    i: usize,
    cc: &[f32],
    ch: &mut [f32],
    wa1: &[f32],
    wa2: &[f32],
    wa3: &[f32],
) {
    let t6 = ido << 1;
    let t2 = t1 + i;
    let t4 = (t1 << 2) + i;
    let t5 = t6 + (t1 << 2) - i;

    let mut t3 = t2 + t0;
    let cr2 = wa1[i - 2] * cc[t3 - 1] + wa1[i - 1] * cc[t3];
    let ci2 = wa1[i - 2] * cc[t3] - wa1[i - 1] * cc[t3 - 1];
    t3 += t0;
    let cr3 = wa2[i - 2] * cc[t3 - 1] + wa2[i - 1] * cc[t3];
    let ci3 = wa2[i - 2] * cc[t3] - wa2[i - 1] * cc[t3 - 1];
    t3 += t0;
    let cr4 = wa3[i - 2] * cc[t3 - 1] + wa3[i - 1] * cc[t3];
    let ci4 = wa3[i - 2] * cc[t3] - wa3[i - 1] * cc[t3 - 1];

    let tr1 = cr2 + cr4;
    let tr4 = cr4 - cr2;
    let ti1 = ci2 + ci4;
    let ti4 = ci2 - ci4;

    let ti2 = cc[t2] + ci3;
    let ti3 = cc[t2] - ci3;
    let tr2 = cc[t2 - 1] + cr3;
    let tr3 = cc[t2 - 1] - cr3;

    ch[t4 - 1] = tr1 + tr2;
    ch[t4] = ti1 + ti2;

    ch[t5 - 1] = tr3 - ti4;
    ch[t5] = tr4 - ti3;

    ch[t4 + t6 - 1] = ti4 + tr3;
    ch[t4 + t6] = tr4 + ti3;

    ch[t5 + t6 - 1] = tr2 - tr1;
    ch[t5 + t6] = ti1 - ti2;
}

/// Port of smallft.c `drftf1` (radix 2 and 4 only).
fn drftf1(isa: SimdIsa, n: i32, c: &mut [f32], ch: &mut [f32], wa: &[f32], ifac: &[i32; 32]) {
    let n = n as usize;
    let nf = ifac[1] as usize;
    let mut na = 1;
    let mut l2 = n;
    let mut iw = n;

    for k1 in 0..nf {
        let kh = nf - k1;
        let ip = ifac[kh + 1] as usize;
        let l1 = l2 / ip;
        let ido = n / l2;
        iw -= (ip - 1) * ido;
        na = 1 - na;

        match ip {
            4 => {
                let ix2 = iw + ido;
                let ix3 = ix2 + ido;
                let (w1, w2, w3) = (&wa[iw - 1..], &wa[ix2 - 1..], &wa[ix3 - 1..]);
                if na != 0 {
                    dradf4(isa, ido, l1, ch, c, w1, w2, w3);
                } else {
                    dradf4(isa, ido, l1, c, ch, w1, w2, w3);
                }
            }
            2 => {
                if na != 0 {
                    dradf2(isa, ido, l1, ch, c, &wa[iw - 1..]);
                } else {
                    dradf2(isa, ido, l1, c, ch, &wa[iw - 1..]);
                }
            }
            _ => unreachable!("smallft: only radix 2/4 factors occur for power-of-two sizes"),
        }
        l2 = l1;
    }

    if na == 1 {
        return;
    }
    c[..n].copy_from_slice(&ch[..n]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_c_literals() {
        assert_eq!(
            6.283_185_5_f32,
            "6.28318530717958648".parse::<f32>().unwrap()
        );
        assert_eq!(
            0.707_106_77_f32,
            ".70710678118654752".parse::<f32>().unwrap()
        );
    }

    /// Output packing: r0, r1, i1, r2, i2, ..., r(n/2) (FORTRAN order).
    #[test]
    fn forward_matches_dft() {
        for &n in &[64usize, 256, 2048] {
            let l = DrftLookup::new(n);
            let x: Vec<f32> = (0..n).map(|i| ((i * 31) % 17) as f32 - 8.0).collect();
            let mut d = x.clone();
            let mut ch = vec![0f32; n];
            l.forward(SimdIsa::Scalar, &mut d, &mut ch);
            for k in 0..=n / 2 {
                let (mut re, mut im) = (0f64, 0f64);
                for (i, &xi) in x.iter().enumerate() {
                    let a = -2. * std::f64::consts::PI * (i * k) as f64 / n as f64;
                    re += f64::from(xi) * a.cos();
                    im += f64::from(xi) * a.sin();
                }
                let got_re = if k == 0 { d[0] } else { d[2 * k - 1] };
                assert!((f64::from(got_re) - re).abs() < 1e-2, "n={n} k={k}");
                if k != 0 && k != n / 2 {
                    assert!((f64::from(d[2 * k]) - im).abs() < 1e-2, "n={n} k={k}");
                }
            }
        }
    }
}
