//! Port of lpc.c (`vorbis_lpc_from_data`, `vorbis_lpc_predict`), used by the
//! analysis side to extrapolate the stream edges (block.c).

/// Port of lpc.c `vorbis_lpc_from_data`: Levinson-Durbin LPC of order `m`
/// over `data` (n = data.len()). Writes `m` coefficients to `lpci` and
/// returns the excitation energy.
pub(crate) fn lpc_from_data(data: &[f32], lpci: &mut [f32], m: usize) -> f32 {
    let n = data.len();
    let mut aut = vec![0f64; m + 1];
    let mut lpc = vec![0f64; m];

    // autocorrelation, p+1 lag coefficients
    for j in (0..=m).rev() {
        let mut d = 0f64; // double needed for accumulator depth
        for i in j..n {
            d += f64::from(data[i]) * f64::from(data[i - j]);
        }
        aut[j] = d;
    }

    // Generate lpc coefficients from autocorr values
    let mut error = aut[0] * (1. + 1e-10);
    let epsilon = 1e-9 * aut[0] + 1e-10;

    'done: {
        for i in 0..m {
            let mut r = -aut[i + 1];
            if error < epsilon {
                for v in &mut lpc[i..] {
                    *v = 0.;
                }
                break 'done;
            }
            for j in 0..i {
                r -= lpc[j] * aut[i - j];
            }
            r /= error;

            lpc[i] = r;
            let mut j = 0;
            while j < i / 2 {
                let tmp = lpc[j];
                lpc[j] += r * lpc[i - 1 - j];
                lpc[i - 1 - j] += r * tmp;
                j += 1;
            }
            if i & 1 != 0 {
                lpc[j] += lpc[j] * r;
            }
            error *= 1. - r * r;
        }
    }

    // slightly damp the filter
    let g = 0.99;
    let mut damp = g;
    for v in lpc.iter_mut() {
        *v *= damp;
        damp *= g;
    }
    for (o, &v) in lpci.iter_mut().zip(&lpc) {
        *o = v as f32;
    }
    error as f32
}

/// Port of lpc.c `vorbis_lpc_predict`: run the predictor `coeff` (order m)
/// primed with `prime[..m]` and write `data.len()` predicted samples.
pub(crate) fn lpc_predict(coeff: &[f32], prime: &[f32], m: usize, data: &mut [f32]) {
    let mut work = vec![0f32; m + data.len()];
    work[..m].copy_from_slice(&prime[..m]);
    for (i, d) in data.iter_mut().enumerate() {
        let mut y = 0f32;
        let mut o = i;
        let mut p = m;
        for _ in 0..m {
            p -= 1;
            y -= work[o] * coeff[p];
            o += 1;
        }
        work[o] = y;
        *d = y;
    }
}
