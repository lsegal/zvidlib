//! Port of mdct.c (float build): `mdct_init` and `mdct_forward` with its
//! butterflies and bit-reversal. Operation order is kept identical to the C
//! code so results are bit-exact.

use super::os::rint;
use std::f64::consts::PI;

const C_PI3_8: f32 = 0.382_683_43;
const C_PI2_8: f32 = 0.707_106_77;
const C_PI1_8: f32 = 0.923_879_5;

/// Port of `mdct_lookup` (mdct.h).
#[derive(Debug, Clone)]
pub(crate) struct MdctLookup {
    n: usize,
    log2n: i32,
    trig: Vec<f32>,
    bitrev: Vec<i32>,
    scale: f32,
}

impl MdctLookup {
    /// Port of mdct.c `mdct_init`.
    pub(crate) fn new(n: usize) -> Self {
        let mut bitrev = vec![0i32; n / 4];
        let mut t = vec![0f32; n + n / 4];
        let n2 = n >> 1;
        let log2n = rint((n as f32 as f64).ln() / (2.0f32 as f64).ln()) as i32;
        let nf = n as f64;
        for i in 0..n / 4 {
            let fi = i as f64;
            t[i * 2] = ((PI / nf) * (4. * fi)).cos() as f32;
            t[i * 2 + 1] = (-((PI / nf) * (4. * fi)).sin()) as f32;
            t[n2 + i * 2] = ((PI / (2. * nf)) * (2. * fi + 1.)).cos() as f32;
            t[n2 + i * 2 + 1] = ((PI / (2. * nf)) * (2. * fi + 1.)).sin() as f32;
        }
        for i in 0..n / 8 {
            let fi = i as f64;
            t[n + i * 2] = (((PI / nf) * (4. * fi + 2.)).cos() * 0.5) as f32;
            t[n + i * 2 + 1] = (-((PI / nf) * (4. * fi + 2.)).sin() * 0.5) as f32;
        }
        // bitreverse lookup
        let mask = (1i32 << (log2n - 1)) - 1;
        let msb = 1i32 << (log2n - 2);
        for i in 0..(n / 8) as i32 {
            let mut acc = 0i32;
            let mut j = 0;
            while (msb >> j) != 0 {
                if (msb >> j) & i != 0 {
                    acc |= 1 << j;
                }
                j += 1;
            }
            bitrev[(i * 2) as usize] = ((!acc) & mask) - 1;
            bitrev[(i * 2 + 1) as usize] = acc;
        }
        MdctLookup {
            n,
            log2n,
            trig: t,
            bitrev,
            scale: 4.0f32 / n as f32,
        }
    }

    /// Port of mdct.c `mdct_forward`: `out` receives n/2 coefficients.
    /// `w` is scratch space of at least n floats.
    pub(crate) fn forward(&self, input: &[f32], out: &mut [f32], w: &mut [f32]) {
        let n = self.n;
        let n2 = n >> 1;
        let n4 = n >> 2;
        let n8 = n >> 3;
        let trig = &self.trig;
        let w = &mut w[..n];
        {
            let w2 = &mut w[n2..];
            let mut x0 = n2 + n4;
            let mut x1 = x0 + 1;
            let mut t = n2;
            let mut i = 0;
            while i < n8 {
                x0 -= 4;
                t -= 2;
                let r0 = input[x0 + 2] + input[x1];
                let r1 = input[x0] + input[x1 + 2];
                w2[i] = r1 * trig[t + 1] + r0 * trig[t];
                w2[i + 1] = r1 * trig[t] - r0 * trig[t + 1];
                x1 += 4;
                i += 2;
            }
            x1 = 1;
            while i < n2 - n8 {
                t -= 2;
                x0 -= 4;
                let r0 = input[x0 + 2] - input[x1];
                let r1 = input[x0] - input[x1 + 2];
                w2[i] = r1 * trig[t + 1] + r0 * trig[t];
                w2[i + 1] = r1 * trig[t] - r0 * trig[t + 1];
                x1 += 4;
                i += 2;
            }
            x0 = n;
            while i < n2 {
                t -= 2;
                x0 -= 4;
                let r0 = -input[x0 + 2] - input[x1];
                let r1 = -input[x0] - input[x1 + 2];
                w2[i] = r1 * trig[t + 1] + r0 * trig[t];
                w2[i + 1] = r1 * trig[t] - r0 * trig[t + 1];
                x1 += 4;
                i += 2;
            }
        }
        self.butterflies(&mut w[n2..], n2);
        self.bitreverse(w);

        // rotate + window
        let mut t = n2;
        let mut wi = 0;
        for i in 0..n4 {
            let x0 = n2 - 1 - i;
            out[i] = (w[wi] * trig[t] + w[wi + 1] * trig[t + 1]) * self.scale;
            out[x0] = (w[wi] * trig[t + 1] - w[wi + 1] * trig[t]) * self.scale;
            wi += 2;
            t += 2;
        }
    }

    /// Port of mdct.c `mdct_butterflies`.
    fn butterflies(&self, x: &mut [f32], points: usize) {
        let trig = &self.trig[..];
        let mut stages = self.log2n - 5;
        stages -= 1;
        if stages > 0 {
            butterfly_first(trig, x, points);
        }
        let mut i = 1;
        loop {
            stages -= 1;
            if stages <= 0 {
                break;
            }
            for j in 0..(1usize << i) {
                let p = points >> i;
                butterfly_generic(trig, &mut x[p * j..p * (j + 1)], p, 4 << i);
            }
            i += 1;
        }
        let mut j = 0;
        while j < points {
            butterfly_32(&mut x[j..j + 32]);
            j += 32;
        }
    }

    /// Port of mdct.c `mdct_bitreverse`.
    fn bitreverse(&self, x: &mut [f32]) {
        let n = self.n;
        let (wlo, xs) = x.split_at_mut(n >> 1);
        let bit = &self.bitrev;
        let trig = &self.trig[n..];
        let mut w0 = 0usize;
        let mut w1 = n >> 1;
        let mut t = 0usize;
        let mut b = 0usize;
        loop {
            let x0 = bit[b] as usize;
            let x1 = bit[b + 1] as usize;
            let r0 = xs[x0 + 1] - xs[x1 + 1];
            let r1 = xs[x0] + xs[x1];
            let r2 = r1 * trig[t] + r0 * trig[t + 1];
            let r3 = r1 * trig[t + 1] - r0 * trig[t];
            w1 -= 4;
            let r0 = (xs[x0 + 1] + xs[x1 + 1]) * 0.5;
            let r1 = (xs[x0] - xs[x1]) * 0.5;
            wlo[w0] = r0 + r2;
            wlo[w1 + 2] = r0 - r2;
            wlo[w0 + 1] = r1 + r3;
            wlo[w1 + 3] = r3 - r1;

            let x0 = bit[b + 2] as usize;
            let x1 = bit[b + 3] as usize;
            let r0 = xs[x0 + 1] - xs[x1 + 1];
            let r1 = xs[x0] + xs[x1];
            let r2 = r1 * trig[t + 2] + r0 * trig[t + 3];
            let r3 = r1 * trig[t + 3] - r0 * trig[t + 2];
            let r0 = (xs[x0 + 1] + xs[x1 + 1]) * 0.5;
            let r1 = (xs[x0] - xs[x1]) * 0.5;
            wlo[w0 + 2] = r0 + r2;
            wlo[w1] = r0 - r2;
            wlo[w0 + 3] = r1 + r3;
            wlo[w1 + 1] = r3 - r1;

            t += 4;
            b += 4;
            w0 += 4;
            if w0 >= w1 {
                break;
            }
        }
    }
}

/// Port of mdct.c `mdct_butterfly_first`.
fn butterfly_first(t: &[f32], x: &mut [f32], points: usize) {
    let mut x1 = points as isize - 8;
    let mut x2 = (points >> 1) as isize - 8;
    let mut ti = 0usize;
    while x2 >= 0 {
        let a = x1 as usize;
        let b = x2 as usize;
        let r0 = x[a + 6] - x[b + 6];
        let r1 = x[a + 7] - x[b + 7];
        x[a + 6] += x[b + 6];
        x[a + 7] += x[b + 7];
        x[b + 6] = r1 * t[ti + 1] + r0 * t[ti];
        x[b + 7] = r1 * t[ti] - r0 * t[ti + 1];

        let r0 = x[a + 4] - x[b + 4];
        let r1 = x[a + 5] - x[b + 5];
        x[a + 4] += x[b + 4];
        x[a + 5] += x[b + 5];
        x[b + 4] = r1 * t[ti + 5] + r0 * t[ti + 4];
        x[b + 5] = r1 * t[ti + 4] - r0 * t[ti + 5];

        let r0 = x[a + 2] - x[b + 2];
        let r1 = x[a + 3] - x[b + 3];
        x[a + 2] += x[b + 2];
        x[a + 3] += x[b + 3];
        x[b + 2] = r1 * t[ti + 9] + r0 * t[ti + 8];
        x[b + 3] = r1 * t[ti + 8] - r0 * t[ti + 9];

        let r0 = x[a] - x[b];
        let r1 = x[a + 1] - x[b + 1];
        x[a] += x[b];
        x[a + 1] += x[b + 1];
        x[b] = r1 * t[ti + 13] + r0 * t[ti + 12];
        x[b + 1] = r1 * t[ti + 12] - r0 * t[ti + 13];

        x1 -= 8;
        x2 -= 8;
        ti += 16;
    }
}

/// Port of mdct.c `mdct_butterfly_generic`.
fn butterfly_generic(t: &[f32], x: &mut [f32], points: usize, trigint: usize) {
    let mut x1 = points as isize - 8;
    let mut x2 = (points >> 1) as isize - 8;
    let mut ti = 0usize;
    while x2 >= 0 {
        let a = x1 as usize;
        let b = x2 as usize;
        let r0 = x[a + 6] - x[b + 6];
        let r1 = x[a + 7] - x[b + 7];
        x[a + 6] += x[b + 6];
        x[a + 7] += x[b + 7];
        x[b + 6] = r1 * t[ti + 1] + r0 * t[ti];
        x[b + 7] = r1 * t[ti] - r0 * t[ti + 1];
        ti += trigint;

        let r0 = x[a + 4] - x[b + 4];
        let r1 = x[a + 5] - x[b + 5];
        x[a + 4] += x[b + 4];
        x[a + 5] += x[b + 5];
        x[b + 4] = r1 * t[ti + 1] + r0 * t[ti];
        x[b + 5] = r1 * t[ti] - r0 * t[ti + 1];
        ti += trigint;

        let r0 = x[a + 2] - x[b + 2];
        let r1 = x[a + 3] - x[b + 3];
        x[a + 2] += x[b + 2];
        x[a + 3] += x[b + 3];
        x[b + 2] = r1 * t[ti + 1] + r0 * t[ti];
        x[b + 3] = r1 * t[ti] - r0 * t[ti + 1];
        ti += trigint;

        let r0 = x[a] - x[b];
        let r1 = x[a + 1] - x[b + 1];
        x[a] += x[b];
        x[a + 1] += x[b + 1];
        x[b] = r1 * t[ti + 1] + r0 * t[ti];
        x[b + 1] = r1 * t[ti] - r0 * t[ti + 1];
        ti += trigint;

        x1 -= 8;
        x2 -= 8;
    }
}

/// Port of mdct.c `mdct_butterfly_8`.
#[inline]
fn butterfly_8(x: &mut [f32]) {
    let r0 = x[6] + x[2];
    let r1 = x[6] - x[2];
    let r2 = x[4] + x[0];
    let r3 = x[4] - x[0];

    x[6] = r0 + r2;
    x[4] = r0 - r2;

    let r0 = x[5] - x[1];
    let r2 = x[7] - x[3];
    x[0] = r1 + r0;
    x[2] = r1 - r0;

    let r0 = x[5] + x[1];
    let r1 = x[7] + x[3];
    x[3] = r2 + r3;
    x[1] = r2 - r3;
    x[7] = r1 + r0;
    x[5] = r1 - r0;
}

/// Port of mdct.c `mdct_butterfly_16`.
#[inline]
fn butterfly_16(x: &mut [f32]) {
    let r0 = x[1] - x[9];
    let r1 = x[0] - x[8];

    x[8] += x[0];
    x[9] += x[1];
    x[0] = (r0 + r1) * C_PI2_8;
    x[1] = (r0 - r1) * C_PI2_8;

    let r0 = x[3] - x[11];
    let r1 = x[10] - x[2];
    x[10] += x[2];
    x[11] += x[3];
    x[2] = r0;
    x[3] = r1;

    let r0 = x[12] - x[4];
    let r1 = x[13] - x[5];
    x[12] += x[4];
    x[13] += x[5];
    x[4] = (r0 - r1) * C_PI2_8;
    x[5] = (r0 + r1) * C_PI2_8;

    let r0 = x[14] - x[6];
    let r1 = x[15] - x[7];
    x[14] += x[6];
    x[15] += x[7];
    x[6] = r0;
    x[7] = r1;

    butterfly_8(&mut x[..8]);
    butterfly_8(&mut x[8..16]);
}

/// Port of mdct.c `mdct_butterfly_32`.
#[inline]
fn butterfly_32(x: &mut [f32]) {
    let r0 = x[30] - x[14];
    let r1 = x[31] - x[15];

    x[30] += x[14];
    x[31] += x[15];
    x[14] = r0;
    x[15] = r1;

    let r0 = x[28] - x[12];
    let r1 = x[29] - x[13];
    x[28] += x[12];
    x[29] += x[13];
    x[12] = r0 * C_PI1_8 - r1 * C_PI3_8;
    x[13] = r0 * C_PI3_8 + r1 * C_PI1_8;

    let r0 = x[26] - x[10];
    let r1 = x[27] - x[11];
    x[26] += x[10];
    x[27] += x[11];
    x[10] = (r0 - r1) * C_PI2_8;
    x[11] = (r0 + r1) * C_PI2_8;

    let r0 = x[24] - x[8];
    let r1 = x[25] - x[9];
    x[24] += x[8];
    x[25] += x[9];
    x[8] = r0 * C_PI3_8 - r1 * C_PI1_8;
    x[9] = r1 * C_PI3_8 + r0 * C_PI1_8;

    let r0 = x[22] - x[6];
    let r1 = x[7] - x[23];
    x[22] += x[6];
    x[23] += x[7];
    x[6] = r1;
    x[7] = r0;

    let r0 = x[4] - x[20];
    let r1 = x[5] - x[21];
    x[20] += x[4];
    x[21] += x[5];
    x[4] = r1 * C_PI1_8 + r0 * C_PI3_8;
    x[5] = r1 * C_PI3_8 - r0 * C_PI1_8;

    let r0 = x[2] - x[18];
    let r1 = x[3] - x[19];
    x[18] += x[2];
    x[19] += x[3];
    x[2] = (r1 + r0) * C_PI2_8;
    x[3] = (r1 - r0) * C_PI2_8;

    let r0 = x[0] - x[16];
    let r1 = x[1] - x[17];
    x[16] += x[0];
    x[17] += x[1];
    x[0] = r1 * C_PI3_8 + r0 * C_PI1_8;
    x[1] = r1 * C_PI1_8 - r0 * C_PI3_8;

    butterfly_16(&mut x[..16]);
    butterfly_16(&mut x[16..32]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_c_literals() {
        assert_eq!(C_PI3_8, ".38268343236508977175".parse::<f32>().unwrap());
        assert_eq!(C_PI2_8, ".70710678118654752441".parse::<f32>().unwrap());
        assert_eq!(C_PI1_8, ".92387953251128675613".parse::<f32>().unwrap());
    }

    /// The fast MDCT must agree with the O(n^2) definition
    /// X[k] = 4/n * sum x[i] cos(2pi/n (i + 1/2 + n/4)(k + 1/2)).
    #[test]
    fn forward_matches_definition() {
        for &n in &[64usize, 256, 2048] {
            let m = MdctLookup::new(n);
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 7919) % 97) as f32 / 97.0 - 0.5)
                .collect();
            let mut out = vec![0f32; n / 2];
            let mut w = vec![0f32; n];
            m.forward(&x, &mut out, &mut w);
            for (k, &got) in out.iter().enumerate() {
                let mut acc = 0f64;
                for (i, &xi) in x.iter().enumerate() {
                    let a =
                        2. * PI / n as f64 * (i as f64 + 0.5 + n as f64 / 4.) * (k as f64 + 0.5);
                    acc += f64::from(xi) * a.cos();
                }
                // libvorbis' forward MDCT is scaled by 4/n
                let want = acc * 4. / n as f64;
                assert!(
                    (f64::from(got) - want).abs() < 1e-4,
                    "n={n} k={k} got={got} want={want}"
                );
            }
        }
    }
}
