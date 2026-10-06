//! Port of psy.c: the psychoacoustic model (ATH, tone masking curves, noise
//! masking, offset/mix, the AoTuV M1 tweak) and the coupled quantization /
//! noise normalization (`_vp_couple_quantize_normalize`).
//!
//! Float/double usage mirrors the C code expression by expression; comments
//! give the C source where the promotions are not obvious.

use super::os::{rint, unitnorm};
use super::tables::gen_misc::{ATH, FLOOR1_FROMDB_LOOKUP, TONEMASKS};
use super::tables::types::{InfoMapping0, InfoPsy, InfoPsyGlobal};

const NEGINF: f32 = -9999.;
const STEREO_THRESHHOLDS: [f64; 9] = [0.0, 0.5, 1.0, 1.5, 2.5, 4.5, 8.5, 16.5, 9e10];
const STEREO_THRESHHOLDS_LIMITED: [f64; 9] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 4.5, 8.5, 9e10];

pub(crate) const P_BANDS: usize = 17;
const P_LEVELS: usize = 8;
const P_LEVEL_0: f64 = 30.;
const P_NOISECURVES: usize = 3;
const NOISE_COMPAND_LEVELS: i32 = 40;
const EHMER_OFFSET: i32 = 16;
const EHMER_MAX: usize = 56;
const MAX_ATH: usize = 88;

/// scales.h `toOC` with a double argument: `log(n)*1.442695f-5.965784f`
// libvorbis' truncated constants must stay exactly as in C
#[allow(clippy::approx_constant)]
#[inline]
fn to_oc(n: f64) -> f64 {
    n.ln() * f64::from(1.442_695_f32) - f64::from(5.965_784_f32)
}

/// scales.h `fromOC` with a double argument: `exp((o+5.965784f)*.693147f)`
// libvorbis' truncated constants must stay exactly as in C
#[allow(clippy::approx_constant)]
#[inline]
fn from_oc(o: f64) -> f64 {
    ((o + f64::from(5.965_784_f32)) * f64::from(0.693_147_f32)).exp()
}

/// scales.h `toBARK` for an integer (C `long`) argument `n`:
/// `13.1f*atan(.00074f*(n))+2.24f*atan((n)*(n)*1.85e-8f)+1e-4f*(n)`.
/// `(n)*(n)` is evaluated in 64 bits as with LP64 `long`; Windows builds of
/// libvorbis (32-bit `long`) overflow here for sample rates above ~92.7 kHz.
#[inline]
fn to_bark_long(n: i64) -> f64 {
    let a = f64::from(13.1_f32) * f64::from(0.00074_f32 * n as f32).atan();
    // (n)*(n) is a long product, converted to float for the multiply
    let b = f64::from(2.24_f32) * f64::from((n * n) as f32 * 1.85e-8_f32).atan();
    let c = 1e-4_f32 * n as f32;
    a + b + f64::from(c)
}

/// Port of `vorbis_look_psy` (psy.h).
#[derive(Debug, Clone)]
pub(crate) struct PsyLook {
    n: usize,
    pub(crate) vi: InfoPsy,
    /// [P_BANDS][P_LEVELS][EHMER_MAX+2]; elements 0/1 are the fenceposts
    tonecurves: Vec<[[f32; EHMER_MAX + 2]; P_LEVELS]>,
    noiseoffset: [Vec<f32>; P_NOISECURVES],
    ath: Vec<f32>,
    octave: Vec<i32>,
    bark: Vec<i32>,
    firstoc: i32,
    shiftoc: i32,
    eighth_octave_lines: i32,
    total_octave_lines: i32,
    m_val: f32,
}

/// Scratch buffers for the per-block psy functions (the C code uses alloca).
#[derive(Debug, Default)]
pub(crate) struct PsyScratch {
    acc: Vec<NoiseAcc>,
    work: Vec<f32>,
    seed: Vec<f32>,
    posstack: Vec<i32>,
    ampstack: Vec<f32>,
    // couple/quantize
    raw: Vec<f32>,
    quant: Vec<f32>,
    floor: Vec<f32>,
    flag: Vec<i32>,
    sort: Vec<usize>,
}

impl PsyLook {
    /// Port of psy.c `_vp_psy_init`.
    pub(crate) fn new(vi: &InfoPsy, gi: &InfoPsyGlobal, n: usize, rate: i32) -> Self {
        let rate_l = i64::from(rate);
        let n_l = n as i64;
        let eighth_octave_lines = gi.eighth_octave_lines;
        // C: rint(log(gi->eighth_octave_lines*8.f)/log(2.f))-1
        let shiftoc = rint(f64::from(eighth_octave_lines as f32 * 8.).ln() / f64::from(2.0f32).ln())
            as i32
            - 1;
        let oc_scale = f64::from(1i32 << (shiftoc + 1));
        // C: toOC(.25f*rate*.5/n)*(1<<(p->shiftoc+1))-gi->eighth_octave_lines
        let firstoc = (to_oc(f64::from(0.25f32 * rate as f32) * 0.5 / n as f64) * oc_scale
            - f64::from(eighth_octave_lines)) as i32;
        // C: toOC((n+.25f)*rate*.5/n)*(1<<(p->shiftoc+1))+.5f
        let maxoc = (to_oc(f64::from((n as f32 + 0.25) * rate as f32) * 0.5 / n as f64) * oc_scale
            + 0.5) as i32;
        let total_octave_lines = maxoc - firstoc + 1;

        // AoTuV HF weighting
        let m_val: f32 = if rate < 26000 {
            0.
        } else if rate < 38000 {
            0.94 // 32kHz
        } else if rate > 46000 {
            1.275 // 48kHz
        } else {
            1.
        };

        // set up the lookups for a given blocksize and sample rate
        let mut ath = vec![0f32; n];
        let mut j = 0usize;
        for i in 0..MAX_ATH - 1 {
            // C: int endpos=rint(fromOC((i+1)*.125-2.)*2*n/rate);
            let endpos =
                rint(from_oc((i + 1) as f64 * 0.125 - 2.) * 2. * n as f64 / f64::from(rate)) as i32;
            let mut base = ATH[i];
            if (j as i32) < endpos {
                let delta = (ATH[i + 1] - base) / (endpos - j as i32) as f32;
                while (j as i32) < endpos && j < n {
                    ath[j] = (f64::from(base) + 100.) as f32;
                    base += delta;
                    j += 1;
                }
            }
        }
        while j < n {
            ath[j] = ath[j - 1];
            j += 1;
        }

        let mut bark = vec![0i32; n];
        let mut lo: i64 = -99;
        let mut hi: i64 = 1;
        let step = rate_l / (2 * n_l);
        for i in 0..n_l {
            let bark_i = to_bark_long(step * i) as f32;
            while lo + i64::from(vi.noisewindowlomin) < i
                && to_bark_long(step * lo) < f64::from(bark_i - vi.noisewindowlo)
            {
                lo += 1;
            }
            while hi <= n_l
                && (hi < i + i64::from(vi.noisewindowhimin)
                    || to_bark_long(step * hi) < f64::from(bark_i + vi.noisewindowhi))
            {
                hi += 1;
            }
            bark[i as usize] = (((lo - 1) as i32) << 16).wrapping_add((hi - 1) as i32);
        }

        let mut octave = vec![0i32; n];
        for (i, o) in octave.iter_mut().enumerate() {
            // C: toOC((i+.25f)*.5*rate/n)*(1<<(p->shiftoc+1))+.5f
            *o = (to_oc(f64::from(i as f32 + 0.25) * 0.5 * f64::from(rate) / n as f64) * oc_scale
                + 0.5) as i32;
        }

        let bin_hz = (f64::from(rate) * 0.5 / n as f64) as f32;
        let tonecurves =
            setup_tone_curves(&vi.toneatt, bin_hz, n, vi.tone_centerboost, vi.tone_decay);

        // set up rolling noise median
        let mut noiseoffset: [Vec<f32>; P_NOISECURVES] =
            [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
        for i in 0..n {
            // C: float halfoc=toOC((i+.5)*rate/(2.*n))*2.;
            let mut halfoc =
                (to_oc((i as f64 + 0.5) * f64::from(rate) / (2. * n as f64)) * 2.) as f32;
            if halfoc < 0. {
                halfoc = 0.;
            }
            if halfoc >= (P_BANDS - 1) as f32 {
                halfoc = (P_BANDS - 1) as f32;
            }
            let inthalfoc = halfoc as i32 as usize;
            let del = halfoc - inthalfoc as f32;
            // C reads noiseoff[j][17] (out of bounds) when inthalfoc==16, but
            // multiplies it by del==0; clamp the index instead.
            let hi_idx = (inthalfoc + 1).min(P_BANDS - 1);
            for (j, no) in noiseoffset.iter_mut().enumerate() {
                // C: noiseoff[j][inthalfoc]*(1.-del) + noiseoff[j][inthalfoc+1]*del
                // (first product in double, second one in float)
                no[i] = (f64::from(vi.noiseoff[j][inthalfoc]) * (1. - f64::from(del))
                    + f64::from(vi.noiseoff[j][hi_idx] * del)) as f32;
            }
        }

        PsyLook {
            n,
            vi: *vi,
            tonecurves,
            noiseoffset,
            ath,
            octave,
            bark,
            firstoc,
            shiftoc,
            eighth_octave_lines,
            total_octave_lines,
            m_val,
        }
    }

    /// Port of psy.c `_vp_noisemask`.
    pub(crate) fn noisemask(&self, s: &mut PsyScratch, logmdct: &[f32], logmask: &mut [f32]) {
        let n = self.n;
        bark_noise_hybridmp(s, n, &self.bark, logmdct, logmask, 140., -1);

        let mut work = std::mem::take(&mut s.work);
        work.clear();
        work.extend((0..n).map(|i| logmdct[i] - logmask[i]));

        bark_noise_hybridmp(
            s,
            n,
            &self.bark,
            &work,
            logmask,
            0.,
            self.vi.noisewindowfixed,
        );

        for i in 0..n {
            work[i] = logmdct[i] - work[i];
        }

        for i in 0..n {
            // C: int dB=logmask[i]+.5;
            let db = (f64::from(logmask[i]) + 0.5) as i32;
            let db = db.clamp(0, NOISE_COMPAND_LEVELS - 1);
            logmask[i] = work[i] + self.vi.noisecompand[db as usize];
        }
        s.work = work;
    }

    /// Port of psy.c `_vp_tonemask`.
    pub(crate) fn tonemask(
        &self,
        s: &mut PsyScratch,
        logfft: &[f32],
        logmask: &mut [f32],
        global_specmax: f32,
        local_specmax: f32,
    ) {
        let n = self.n;
        let total = self.total_octave_lines as usize;
        let mut seed = std::mem::take(&mut s.seed);
        seed.clear();
        seed.resize(total, NEGINF);
        let mut att = local_specmax + self.vi.ath_adjatt;

        // set the ATH (floating below localmax, not global max by a
        // specified att)
        if att < self.vi.ath_maxatt {
            att = self.vi.ath_maxatt;
        }
        for (lm, &a) in logmask[..n].iter_mut().zip(&self.ath) {
            *lm = a + att;
        }

        // tone masking
        self.seed_loop(logfft, logmask, &mut seed, global_specmax);
        self.max_seeds(s, &mut seed, logmask);
        s.seed = seed;
    }

    /// Port of psy.c `seed_loop`.
    fn seed_loop(&self, f: &[f32], flr: &[f32], seed: &mut [f32], specmax: f32) {
        let n = self.n;
        let dboffset = self.vi.max_curve_db - specmax;

        // prime the working vector with peak values
        let mut i = 0;
        while i < n {
            let mut max = f[i];
            let mut oc = self.octave[i];
            while i + 1 < n && self.octave[i + 1] == oc {
                i += 1;
                if f[i] > max {
                    max = f[i];
                }
            }

            if max + 6. > flr[i] {
                oc >>= self.shiftoc;
                oc = oc.clamp(0, P_BANDS as i32 - 1);
                seed_curve(
                    seed,
                    &self.tonecurves[oc as usize],
                    max,
                    self.octave[i] - self.firstoc,
                    self.total_octave_lines,
                    self.eighth_octave_lines,
                    dboffset,
                );
            }
            i += 1;
        }
    }

    /// Port of psy.c `max_seeds`.
    fn max_seeds(&self, s: &mut PsyScratch, seed: &mut [f32], flr: &mut [f32]) {
        let n = self.total_octave_lines as i64;
        let linesper = self.eighth_octave_lines;
        let mut linpos = 0usize;

        seed_chase(s, seed, linesper, n); // for masking

        let mut pos = i64::from(self.octave[0] - self.firstoc - (linesper >> 1));

        while linpos + 1 < self.n {
            let mut min_v = seed[pos as usize];
            let mut end =
                i64::from(((self.octave[linpos] + self.octave[linpos + 1]) >> 1) - self.firstoc);
            if min_v > self.vi.tone_abs_limit {
                min_v = self.vi.tone_abs_limit;
            }
            while pos < end {
                pos += 1;
                let sp = seed[pos as usize];
                if (sp > NEGINF && sp < min_v) || min_v == NEGINF {
                    min_v = sp;
                }
            }

            end = pos + i64::from(self.firstoc);
            while linpos < self.n && i64::from(self.octave[linpos]) <= end {
                if flr[linpos] < min_v {
                    flr[linpos] = min_v;
                }
                linpos += 1;
            }
        }

        let min_v = seed[self.total_octave_lines as usize - 1];
        while linpos < self.n {
            if flr[linpos] < min_v {
                flr[linpos] = min_v;
            }
            linpos += 1;
        }
    }

    /// Port of psy.c `_vp_offset_and_mix`.
    pub(crate) fn offset_and_mix(
        &self,
        noise: &[f32],
        tone: &[f32],
        offset_select: usize,
        logmask: &mut [f32],
        mdct: &mut [f32],
        logmdct: &[f32],
    ) {
        let n = self.n;
        let toneatt = self.vi.tone_masteratt[offset_select];
        let cx = self.m_val;

        for i in 0..n {
            let mut val = noise[i] + self.noiseoffset[offset_select][i];
            if val > self.vi.noisemaxsupp {
                val = self.vi.noisemaxsupp;
            }
            let t = tone[i] + toneatt;
            logmask[i] = if val < t { t } else { val };

            // AoTuV M1: relative compensation of the MDCT based on the mask
            if offset_select == 1 {
                let coeffi: f32 = -17.2; // coeffi is a -17.2dB threshold
                val -= logmdct[i]; // val == mdct line value relative to floor in dB
                let de: f32 = if val > coeffi {
                    // C: de = 1.0-((val-coeffi)*0.005*cx);
                    let mut de = (1.0 - (f64::from(val - coeffi) * 0.005 * f64::from(cx))) as f32;
                    if de < 0. {
                        de = 0.0001;
                    }
                    de
                } else {
                    (1.0 - (f64::from(val - coeffi) * 0.0003 * f64::from(cx))) as f32
                };
                mdct[i] *= de;
            }
        }
    }
}

/// psy.c `min_curve`
fn min_curve(c: &mut [f32; EHMER_MAX], c2: &[f32; EHMER_MAX]) {
    for i in 0..EHMER_MAX {
        if c2[i] < c[i] {
            c[i] = c2[i];
        }
    }
}

/// psy.c `max_curve`
fn max_curve(c: &mut [f32; EHMER_MAX], c2: &[f32; EHMER_MAX]) {
    for i in 0..EHMER_MAX {
        if c2[i] > c[i] {
            c[i] = c2[i];
        }
    }
}

/// psy.c `attenuate_curve`
fn attenuate_curve(c: &mut [f32; EHMER_MAX], att: f32) {
    for v in c.iter_mut() {
        *v += att;
    }
}

/// Port of psy.c `setup_tone_curves`.
fn setup_tone_curves(
    curveatt_db: &[f32; P_BANDS],
    bin_hz: f32,
    n: usize,
    center_boost: f32,
    center_decay_rate: f32,
) -> Vec<[[f32; EHMER_MAX + 2]; P_LEVELS]> {
    let mut ath = [0f32; EHMER_MAX];
    let mut workc = vec![[[0f32; EHMER_MAX]; P_LEVELS]; P_BANDS];
    let mut athc = [[0f32; EHMER_MAX]; P_LEVELS];
    let mut brute_buffer = vec![0f32; n];
    let mut ret = vec![[[0f32; EHMER_MAX + 2]; P_LEVELS]; P_BANDS];
    let bin_hz_d = f64::from(bin_hz);

    for i in 0..P_BANDS {
        // we add back in the ATH to avoid low level curves falling off to
        // -infinity and unnecessarily cutting off high level curves in the
        // curve limiting (last step).

        // A half-band's settings must be valid over the whole band, and
        // it's better to mask too little than too much
        let ath_offset = i * 4;
        for (j, a) in ath.iter_mut().enumerate() {
            let mut min = 999f32;
            for k in 0..4 {
                let v = if j + k + ath_offset < MAX_ATH {
                    ATH[j + k + ath_offset]
                } else {
                    ATH[MAX_ATH - 1]
                };
                if min > v {
                    min = v;
                }
            }
            *a = min;
        }

        // copy curves into working space, replicate the 50dB curve to 30
        // and 40, replicate the 100dB curve to 110
        for j in 0..6 {
            workc[i][j + 2] = TONEMASKS[i][j];
        }
        workc[i][0] = TONEMASKS[i][0];
        workc[i][1] = TONEMASKS[i][0];

        // apply centered curve boost/decay
        for curve in workc[i].iter_mut() {
            for (k, w) in curve.iter_mut().enumerate() {
                // C: float adj=center_boost+abs(EHMER_OFFSET-k)*center_decay_rate;
                let mut adj =
                    center_boost + (EHMER_OFFSET - k as i32).abs() as f32 * center_decay_rate;
                if adj < 0. && center_boost > 0. {
                    adj = 0.;
                }
                if adj > 0. && center_boost < 0. {
                    adj = 0.;
                }
                *w += adj;
            }
        }

        // normalize curves so the driving amplitude is 0dB
        // make temp curves with the ATH overlayed
        for j in 0..P_LEVELS {
            // C: curveatt_dB[i]+100.-(j<2?2:j)*10.-P_LEVEL_0
            let jj = if j < 2 { 2 } else { j };
            let att = (f64::from(curveatt_db[i]) + 100. - jj as f64 * 10. - P_LEVEL_0) as f32;
            attenuate_curve(&mut workc[i][j], att);
            athc[j] = ath;
            // C: +100.-j*10.f-P_LEVEL_0
            let att2 = (100. - f64::from(j as f32 * 10.) - P_LEVEL_0) as f32;
            attenuate_curve(&mut athc[j], att2);
            max_curve(&mut athc[j], &workc[i][j]);
        }

        // Now limit the louder curves.
        for j in 1..P_LEVELS {
            let prev = athc[j - 1];
            min_curve(&mut athc[j], &prev);
            min_curve(&mut workc[i][j], &athc[j]);
        }
    }

    let ni = n as i32;
    for i in 0..P_BANDS {
        // which octave curves will we be compositing?
        let bin = (from_oc(i as f64 * 0.5) / bin_hz_d).floor() as i32;
        // C: ceil(toOC(bin*binHz+1)*2)  (bin*binHz+1 is a float expression)
        let mut lo_curve = (to_oc(f64::from(bin as f32 * bin_hz + 1.)) * 2.).ceil() as i32;
        let mut hi_curve = (to_oc(f64::from((bin + 1) as f32 * bin_hz)) * 2.).floor() as i32;
        if lo_curve > i as i32 {
            lo_curve = i as i32;
        }
        if lo_curve < 0 {
            lo_curve = 0;
        }
        if hi_curve >= P_BANDS as i32 {
            hi_curve = P_BANDS as i32 - 1;
        }

        for m in 0..P_LEVELS {
            for b in brute_buffer.iter_mut() {
                *b = 999.;
            }

            // render the curve into bins, then pull values back into curve.
            // The point is that any inherent subsampling aliasing results in
            // a safe minimum
            let mut k = lo_curve;
            while k <= hi_curve {
                let curve = &workc[k as usize][m];
                render_curve(&mut brute_buffer, curve, k, bin_hz_d, ni);
                k += 1;
            }

            // be equally paranoid about being valid up to next half ocatve
            if i + 1 < P_BANDS {
                let curve = &workc[i + 1][m];
                render_curve(&mut brute_buffer, curve, i as i32, bin_hz_d, ni);
            }

            for j in 0..EHMER_MAX {
                let bin = (from_oc(j as f64 * 0.125 + i as f64 * 0.5 - 2.) / bin_hz_d) as i32;
                ret[i][m][j + 2] = if bin < 0 || bin >= ni {
                    -999.
                } else {
                    brute_buffer[bin as usize]
                };
            }

            // add fenceposts
            let mut j = 0;
            while j < EHMER_OFFSET as usize {
                if ret[i][m][j + 2] > -200. {
                    break;
                }
                j += 1;
            }
            ret[i][m][0] = j as f32;

            let mut j = EHMER_MAX - 1;
            while j > EHMER_OFFSET as usize + 1 {
                if ret[i][m][j + 2] > -200. {
                    break;
                }
                j -= 1;
            }
            ret[i][m][1] = j as f32;
        }
    }
    ret
}

/// The bin-rendering inner loop of `setup_tone_curves`, shared by its two
/// copies (octave `oc` selects the frequency placement, `curve` the data).
fn render_curve(brute_buffer: &mut [f32], curve: &[f32; EHMER_MAX], oc: i32, bin_hz: f64, n: i32) {
    let mut l: i32 = 0;
    for (j, &cv) in curve.iter().enumerate() {
        let mut lo_bin = (from_oc(j as f64 * 0.125 + f64::from(oc) * 0.5 - 2.0625) / bin_hz) as i32;
        let mut hi_bin =
            (from_oc(j as f64 * 0.125 + f64::from(oc) * 0.5 - 1.9375) / bin_hz + 1.) as i32;
        if lo_bin < 0 {
            lo_bin = 0;
        }
        if lo_bin > n {
            lo_bin = n;
        }
        if lo_bin < l {
            l = lo_bin;
        }
        if hi_bin < 0 {
            hi_bin = 0;
        }
        if hi_bin > n {
            hi_bin = n;
        }
        while l < hi_bin && l < n {
            if brute_buffer[l as usize] > cv {
                brute_buffer[l as usize] = cv;
            }
            l += 1;
        }
    }
    let last = curve[EHMER_MAX - 1];
    while l < n {
        if brute_buffer[l as usize] > last {
            brute_buffer[l as usize] = last;
        }
        l += 1;
    }
}

/// Port of psy.c `seed_curve`.
fn seed_curve(
    seed: &mut [f32],
    curves: &[[f32; EHMER_MAX + 2]; P_LEVELS],
    amp: f32,
    oc: i32,
    n: i32,
    linesper: i32,
    dboffset: f32,
) {
    // C: int choice=(int)((amp+dBoffset-P_LEVEL_0)*.1f);
    let choice = ((f64::from(amp + dboffset) - P_LEVEL_0) * f64::from(0.1f32)) as i32;
    let choice = choice.clamp(0, P_LEVELS as i32 - 1);
    let posts = &curves[choice as usize];
    let curve = &posts[2..];
    let post1 = posts[1] as i32;
    // C: seedptr=oc+(posts[0]-EHMER_OFFSET)*linesper-(linesper>>1);  (float)
    let mut seedptr = (oc as f32 + (posts[0] - EHMER_OFFSET as f32) * linesper as f32
        - (linesper >> 1) as f32) as i32;

    let mut i = posts[0] as i32;
    while i < post1 {
        if seedptr > 0 {
            let lin = amp + curve[i as usize];
            if seed[seedptr as usize] < lin {
                seed[seedptr as usize] = lin;
            }
        }
        seedptr += linesper;
        if seedptr >= n {
            break;
        }
        i += 1;
    }
}

/// Port of psy.c `seed_chase`.
fn seed_chase(s: &mut PsyScratch, seeds: &mut [f32], linesper: i32, n: i64) {
    let nu = n as usize;
    s.posstack.clear();
    s.posstack.resize(nu, 0);
    s.ampstack.clear();
    s.ampstack.resize(nu, 0.);
    let posstack = &mut s.posstack;
    let ampstack = &mut s.ampstack;
    let mut stack = 0usize;
    let mut pos = 0usize;

    for (i, &sv) in seeds.iter().enumerate().take(nu) {
        if stack < 2 {
            posstack[stack] = i as i32;
            ampstack[stack] = sv;
            stack += 1;
        } else {
            loop {
                if sv < ampstack[stack - 1] {
                    posstack[stack] = i as i32;
                    ampstack[stack] = sv;
                    stack += 1;
                    break;
                } else {
                    if (i as i32) < posstack[stack - 1] + linesper
                        && stack > 1
                        && ampstack[stack - 1] <= ampstack[stack - 2]
                        && (i as i32) < posstack[stack - 2] + linesper
                    {
                        // we completely overlap, making stack-1 irrelevant. pop it
                        stack -= 1;
                        continue;
                    }
                    posstack[stack] = i as i32;
                    ampstack[stack] = sv;
                    stack += 1;
                    break;
                }
            }
        }
    }

    // the stack now contains only the positions that are relevant. Scan
    // 'em straight through
    for i in 0..stack {
        let mut endpos = if i < stack - 1 && ampstack[i + 1] > ampstack[i] {
            posstack[i + 1] as usize
        } else {
            // +1 is important, else bin 0 is discarded in short frames
            (posstack[i] + linesper + 1) as usize
        };
        if endpos > nu {
            endpos = nu;
        }
        while pos < endpos {
            seeds[pos] = ampstack[i];
            pos += 1;
        }
    }
}

/// Running sums of `bark_noise_hybridmp`: N, X, XX, Y, XY at one index (the
/// C code keeps five parallel arrays).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NoiseAcc {
    n: f32,
    x: f32,
    xx: f32,
    y: f32,
    xy: f32,
}

/// One least-squares evaluation of `bark_noise_hybridmp`. `reflect` selects
/// the window-straddles-zero form (sums of the mirrored lower part).
/// Returns (A, B, D).
#[inline(always)]
fn noise_fit(hi: &NoiseAcc, lo: &NoiseAcc, reflect: bool) -> (f32, f32, f32) {
    let (t_n, t_xx, t_y) = if reflect {
        (hi.n + lo.n, hi.xx + lo.xx, hi.y + lo.y)
    } else {
        (hi.n - lo.n, hi.xx - lo.xx, hi.y - lo.y)
    };
    let t_x = hi.x - lo.x;
    let t_xy = hi.xy - lo.xy;

    let a = t_y * t_xx - t_x * t_xy;
    let b = t_n * t_xy - t_x * t_y;
    let d = t_n * t_xx - t_x * t_x;
    (a, b, d)
}

/// Port of psy.c `bark_noise_hybridmp`.
fn bark_noise_hybridmp(
    s: &mut PsyScratch,
    n: usize,
    b: &[i32],
    f: &[f32],
    noise: &mut [f32],
    offset: f32,
    fixed: i32,
) {
    // every element is written by the accumulation loop below
    s.acc.resize(n, NoiseAcc::default());
    let acc = &mut s.acc[..n];
    let noise = &mut noise[..n];
    let b = &b[..n];

    let mut t = NoiseAcc::default();

    let mut a = 0f32;
    let mut bb = 0f32;
    let mut d = 1f32;

    let mut y = f[0] + offset;
    if y < 1. {
        y = 1.;
    }
    // C: w = y * y * .5;
    let w = (f64::from(y * y) * 0.5) as f32;

    t.n += w;
    t.x += w;
    t.y += w * y;
    acc[0] = t;

    let mut x = 1f32;
    for (ai, &fi) in acc.iter_mut().zip(f).skip(1) {
        let mut y = fi + offset;
        if y < 1. {
            y = 1.;
        }
        let w = y * y;

        t.n += w;
        t.x += w * x;
        t.xx += w * x * x;
        t.y += w * y;
        t.xy += w * x * y;

        *ai = t;
        x += 1.;
    }

    let ni = n as i32;
    let mut i = 0usize;
    let mut x = 0f32;
    while i < n {
        let lo = b[i] >> 16;
        let hi = b[i] & 0xffff;
        if lo >= 0 || -lo >= ni {
            break;
        }
        if hi >= ni {
            break;
        }
        (a, bb, d) = noise_fit(&acc[hi as usize], &acc[(-lo) as usize], true);
        let mut r = (a + x * bb) / d;
        if r < 0. {
            r = 0.;
        }
        noise[i] = r - offset;
        i += 1;
        x += 1.;
    }

    while i < n {
        let lo = b[i] >> 16;
        let hi = b[i] & 0xffff;
        if lo < 0 || lo >= ni {
            break;
        }
        if hi >= ni {
            break;
        }
        (a, bb, d) = noise_fit(&acc[hi as usize], &acc[lo as usize], false);
        let mut r = (a + x * bb) / d;
        if r < 0. {
            r = 0.;
        }
        noise[i] = r - offset;
        i += 1;
        x += 1.;
    }

    while i < n {
        let mut r = (a + x * bb) / d;
        if r < 0. {
            r = 0.;
        }
        noise[i] = r - offset;
        i += 1;
        x += 1.;
    }

    if fixed <= 0 {
        return;
    }

    let mut i = 0usize;
    let mut x = 0f32;
    while i < n {
        let hi = i as i32 + fixed / 2;
        let lo = hi - fixed;
        if hi >= ni {
            break;
        }
        if lo >= 0 {
            break;
        }
        (a, bb, d) = noise_fit(&acc[hi as usize], &acc[(-lo) as usize], true);
        let r = (a + x * bb) / d;
        if r - offset < noise[i] {
            noise[i] = r - offset;
        }
        i += 1;
        x += 1.;
    }
    while i < n {
        let hi = i as i32 + fixed / 2;
        let lo = hi - fixed;
        if hi >= ni {
            break;
        }
        if lo < 0 {
            break;
        }
        (a, bb, d) = noise_fit(&acc[hi as usize], &acc[lo as usize], false);
        let r = (a + x * bb) / d;
        if r - offset < noise[i] {
            noise[i] = r - offset;
        }
        i += 1;
        x += 1.;
    }
    while i < n {
        let r = (a + x * bb) / d;
        if r - offset < noise[i] {
            noise[i] = r - offset;
        }
        i += 1;
        x += 1.;
    }
}

/// psy.c `_vp_ampmax_decay`
pub(crate) fn ampmax_decay(amp: f32, gi: &InfoPsyGlobal, blocksize_w: i32, rate: i32) -> f32 {
    let n = blocksize_w / 2;
    let secs = n as f32 / rate as f32;
    let mut amp = amp + secs * gi.ampmax_att_per_sec;
    if amp < -9999. {
        amp = -9999.;
    }
    amp
}

/// Port of psy.c `flag_lossless`.
#[allow(clippy::too_many_arguments)]
fn flag_lossless(
    limit: i32,
    prepoint: f32,
    postpoint: f32,
    mdct: &[f32],
    floor: &[f32],
    flag: &mut [i32],
    i: i32,
    jn: usize,
) {
    for j in 0..jn {
        let point = if j as i32 >= limit - i {
            postpoint
        } else {
            prepoint
        };
        // C: float r = fabs(mdct[j])/floor[j];  (computed in double)
        let r = (f64::from(mdct[j].abs()) / f64::from(floor[j])) as f32;
        flag[j] = if r < point { 0 } else { 1 };
    }
}

/// Port of psy.c `noise_normalize`. `q` holds quantized energy (flagged) or
/// |raw| (unflagged) on input and the quantized energy on output.
#[allow(clippy::too_many_arguments)]
fn noise_normalize(
    vi: &InfoPsy,
    sort: &mut Vec<usize>,
    limit: i32,
    r: &[f32],
    q: &mut [f32],
    f: &[f32],
    flags: Option<&[i32]>,
    i: i32,
    n: usize,
    out: &mut [i32],
) -> f32 {
    let mut start = if vi.normal_p != 0 {
        vi.normal_start - i
    } else {
        n as i32
    };
    if start > n as i32 {
        start = n as i32;
    }
    let start = start.max(0) as usize;

    // force classic behavior where only energy in the current band is considered
    let mut acc = 0f32;

    let unflagged = |j: usize| flags.is_none_or(|fl| fl[j] == 0);

    // still responsible for populating *out where noise norm not in
    // effect.  There's no need to [re]populate *q in these areas
    for j in 0..start {
        if unflagged(j) {
            let ve = q[j] / f[j];
            let v = rint(f64::from(ve).sqrt()) as i32;
            out[j] = if r[j] < 0. { -v } else { v };
        }
    }

    // sort magnitudes for noise norm portion of partition
    sort.clear();
    for j in start..n {
        if unflagged(j) {
            let ve = q[j] / f[j];
            // Despite all the new, more capable coupling code, for now we
            // implement noise norm as it has been up to this point. Only
            // consider promotions to unit magnitude from 0.  In addition
            // the only energy error counted is quantizations to zero.
            // also-- the original point code only applied noise norm at > pointlimit
            if ve < 0.25 && (flags.is_none() || j as i32 >= limit - i) {
                acc += ve;
                sort.push(j); // q is fabs(r) for unflagged element
            } else {
                // For now: no acc adjustment for nonzero quantization.
                // populate *out and q as this value is final.
                let v = rint(f64::from(ve).sqrt()) as i32;
                out[j] = if r[j] < 0. { -v } else { v };
                q[j] = out[j].wrapping_mul(out[j]) as f32 * f[j];
            }
        }
    }

    if !sort.is_empty() {
        // noise norm to do: qsort with apsort (descending by q). The sort is
        // stable here; C's qsort order for equal values is unspecified.
        sort.sort_by(|&a, &b| q[b].partial_cmp(&q[a]).unwrap_or(std::cmp::Ordering::Equal));
        for &k in sort.iter() {
            if f64::from(acc) >= vi.normal_thresh {
                out[k] = unitnorm(r[k]) as i32;
                acc -= 1.;
                q[k] = f[k];
            } else {
                out[k] = 0;
                q[k] = 0.;
            }
        }
    }
    acc
}

/// Port of psy.c `_vp_couple_quantize_normalize`. `iwork` holds the integer
/// floor (masking curve) on input and the quantized, coupled residue on output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn couple_quantize_normalize(
    s: &mut PsyScratch,
    blobno: usize,
    g: &InfoPsyGlobal,
    p: &PsyLook,
    vi: &InfoMapping0,
    mdct: &[Vec<f32>],
    iwork: &mut [Vec<i32>],
    nonzero: &mut [bool],
    sliding_lowpass: i32,
    ch: usize,
) {
    let n = p.n;
    let partition = if p.vi.normal_p != 0 {
        p.vi.normal_partition as usize
    } else {
        16
    };
    let limit = g.coupling_pointlimit[p.vi.blockflag as usize][blobno];
    let prepoint = STEREO_THRESHHOLDS[g.coupling_prepointamp[blobno] as usize] as f32;
    let mut postpoint = STEREO_THRESHHOLDS[g.coupling_postpointamp[blobno] as usize] as f32;

    // The threshold of a stereo is changed with the size of n
    if n > 1000 {
        postpoint = STEREO_THRESHHOLDS_LIMITED[g.coupling_postpointamp[blobno] as usize] as f32;
    }

    let steps = vi.coupling_steps as usize;
    let mut raw = std::mem::take(&mut s.raw);
    let mut quant = std::mem::take(&mut s.quant);
    let mut floor = std::mem::take(&mut s.floor);
    let mut flag = std::mem::take(&mut s.flag);
    let mut sort = std::mem::take(&mut s.sort);
    raw.clear();
    raw.resize(ch * partition, 0.);
    quant.clear();
    quant.resize(ch * partition, 0.);
    floor.clear();
    floor.resize(ch * partition, 0.);
    flag.clear();
    flag.resize(ch * partition, 0);
    let mut acc = vec![0f32; ch + steps];
    let mut nz = vec![false; ch];

    let mut i = 0usize;
    while i < n {
        let jn = if partition > n - i { n - i } else { partition };
        let mut track = 0usize;
        let ii = i as i32;

        nz.copy_from_slice(nonzero);

        // prefill
        for v in flag.iter_mut() {
            *v = 0;
        }
        for k in 0..ch {
            let base = k * partition;
            let iout = &mut iwork[k][i..i + jn];
            if nz[k] {
                for j in 0..jn {
                    floor[base + j] = FLOOR1_FROMDB_LOOKUP[iout[j] as usize];
                }

                flag_lossless(
                    limit,
                    prepoint,
                    postpoint,
                    &mdct[k][i..i + jn],
                    &floor[base..base + jn],
                    &mut flag[base..base + jn],
                    ii,
                    jn,
                );

                for j in 0..jn {
                    let m = mdct[k][i + j];
                    let e = m * m;
                    quant[base + j] = e;
                    raw[base + j] = e;
                    if m < 0. {
                        raw[base + j] *= -1.;
                    }
                    floor[base + j] *= floor[base + j];
                }

                acc[track] = noise_normalize(
                    &p.vi,
                    &mut sort,
                    limit,
                    &raw[base..base + jn],
                    &mut quant[base..base + jn],
                    &floor[base..base + jn],
                    None,
                    ii,
                    jn,
                    iout,
                );
            } else {
                for j in 0..jn {
                    floor[base + j] = 1e-10;
                    raw[base + j] = 0.;
                    quant[base + j] = 0.;
                    flag[base + j] = 0;
                    iout[j] = 0;
                }
                acc[track] = 0.;
            }
            track += 1;
        }

        // coupling
        for step in 0..steps {
            let mi = vi.coupling_mag[step] as usize;
            let ai = vi.coupling_ang[step] as usize;
            let (mb, ab) = (mi * partition, ai * partition);

            if nz[mi] || nz[ai] {
                nz[mi] = true;
                nz[ai] = true;

                for j in 0..jn {
                    if (j as i32) < sliding_lowpass - ii {
                        if flag[mb + j] != 0 || flag[ab + j] != 0 {
                            // lossless coupling
                            raw[mb + j] = raw[mb + j].abs() + raw[ab + j].abs();
                            quant[mb + j] += quant[ab + j];
                            flag[mb + j] = 1;
                            flag[ab + j] = 1;

                            // couple iM/iA
                            let am = iwork[mi][i + j];
                            let bm = iwork[ai][i + j];
                            // (wrapping integer ops: C's int arithmetic, which
                            // only overflows for absurd input amplitudes)
                            if am.wrapping_abs() > bm.wrapping_abs() {
                                iwork[ai][i + j] = if am > 0 {
                                    am.wrapping_sub(bm)
                                } else {
                                    bm.wrapping_sub(am)
                                };
                            } else {
                                iwork[ai][i + j] = if bm > 0 {
                                    am.wrapping_sub(bm)
                                } else {
                                    bm.wrapping_sub(am)
                                };
                                iwork[mi][i + j] = bm;
                            }

                            // collapse two equivalent tuples to one
                            if iwork[ai][i + j] >= iwork[mi][i + j].wrapping_abs().wrapping_mul(2) {
                                iwork[ai][i + j] = iwork[ai][i + j].wrapping_neg();
                                iwork[mi][i + j] = iwork[mi][i + j].wrapping_neg();
                            }
                        } else {
                            // lossy (point) coupling
                            if (j as i32) < limit - ii {
                                // dipole
                                raw[mb + j] += raw[ab + j];
                                quant[mb + j] = raw[mb + j].abs();
                            } else {
                                // elliptical
                                let e = raw[mb + j].abs() + raw[ab + j].abs();
                                quant[mb + j] = e;
                                raw[mb + j] = if raw[mb + j] + raw[ab + j] < 0. {
                                    -e
                                } else {
                                    e
                                };
                            }
                            raw[ab + j] = 0.;
                            quant[ab + j] = 0.;
                            flag[ab + j] = 1;
                            iwork[ai][i + j] = 0;
                        }
                    }
                    let fsum = floor[mb + j] + floor[ab + j];
                    floor[mb + j] = fsum;
                    floor[ab + j] = fsum;
                }
                // normalize the resulting mag vector
                acc[track] = noise_normalize(
                    &p.vi,
                    &mut sort,
                    limit,
                    &raw[mb..mb + jn],
                    &mut quant[mb..mb + jn],
                    &floor[mb..mb + jn],
                    Some(&flag[mb..mb + jn]),
                    ii,
                    jn,
                    &mut iwork[mi][i..i + jn],
                );
                track += 1;
            }
        }
        i += partition;
    }

    for i in 0..steps {
        // make sure coupling a zero and a nonzero channel results in two
        // nonzero channels.
        let (m, a) = (vi.coupling_mag[i] as usize, vi.coupling_ang[i] as usize);
        if nonzero[m] || nonzero[a] {
            nonzero[m] = true;
            nonzero[a] = true;
        }
    }

    s.raw = raw;
    s.quant = quant;
    s.floor = floor;
    s.flag = flag;
    s.sort = sort;
}
