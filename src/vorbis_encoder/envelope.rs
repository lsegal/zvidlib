//! Port of envelope.c: the pre-echo/transient detector that drives the
//! long/short block decision (`_ve_envelope_init`, `_ve_amp`,
//! `_ve_envelope_search`, `_ve_envelope_mark`, `_ve_envelope_shift`).

use super::mdct::MdctLookup;
use super::os::todb;
use super::tables::types::InfoPsyGlobal;
use std::f64::consts::PI;

const VE_PRE: usize = 16;
const VE_WIN: i64 = 4;
const VE_POST: i64 = 2;
const VE_AMP: usize = VE_PRE + VE_POST as usize - 1;
pub(crate) const VE_BANDS: usize = 7;
const VE_NEARDC: usize = 15;
const VE_MINSTRETCH: i32 = 2;
const VE_MAXSTRETCH: i32 = 12;

/// Port of `envelope_filter_state` (envelope.h).
#[derive(Debug, Clone, Default)]
struct FilterState {
    ampbuf: [f32; VE_AMP],
    ampptr: usize,
    near_dc: [f32; VE_NEARDC],
    near_dc_acc: f32,
    near_dc_partialacc: f32,
    nearptr: usize,
}

/// Port of `envelope_band` (envelope.h).
#[derive(Debug, Clone)]
struct Band {
    begin: usize,
    end: usize,
    window: Vec<f32>,
    total: f32,
}

/// Port of `envelope_lookup` (envelope.h).
#[derive(Debug, Clone)]
pub(crate) struct EnvelopeLookup {
    ch: usize,
    winlength: usize,
    searchstep: i64,
    minenergy: f32,
    mdct: MdctLookup,
    mdct_win: Vec<f32>,
    band: Vec<Band>,
    filter: Vec<FilterState>,
    stretch: i32,
    mark: Vec<i32>,
    current: i64,
    curmark: i64,
    cursor: i64,
    // scratch
    vec: Vec<f32>,
    vec_out: Vec<f32>,
    work: Vec<f32>,
}

/// The part of `vorbis_dsp_state` the envelope search reads.
pub(crate) struct DspView<'a> {
    pub(crate) pcm: &'a [Vec<f32>],
    pub(crate) pcm_current: i64,
    pub(crate) center_w: i64,
    pub(crate) w: usize,
    pub(crate) lw: usize,
    pub(crate) nw: usize,
    pub(crate) blocksizes: [i32; 2],
}

impl EnvelopeLookup {
    /// Port of envelope.c `_ve_envelope_init`.
    pub(crate) fn new(gi: &InfoPsyGlobal, channels: usize, blocksizes: [i32; 2]) -> Self {
        let n = 128usize;
        let mut mdct_win = vec![0f32; n];
        for (i, w) in mdct_win.iter_mut().enumerate() {
            *w = (i as f64 / (n as f64 - 1.) * PI).sin() as f32;
            *w *= *w;
        }
        // magic follows
        let begins = [2, 4, 6, 9, 13, 17, 22];
        let ends = [4, 5, 6, 8, 8, 8, 8];
        let mut band = Vec::with_capacity(VE_BANDS);
        for j in 0..VE_BANDS {
            let n = ends[j];
            let mut window = vec![0f32; n];
            let mut total = 0f32;
            for (i, w) in window.iter_mut().enumerate() {
                *w = ((i as f64 + 0.5) / n as f64 * PI).sin() as f32;
                total += *w;
            }
            band.push(Band {
                begin: begins[j],
                end: n,
                window,
                total: (1. / f64::from(total)) as f32,
            });
        }
        EnvelopeLookup {
            ch: channels,
            winlength: n,
            searchstep: 64,
            minenergy: gi.preecho_minenergy,
            mdct: MdctLookup::new(n),
            mdct_win,
            band,
            filter: vec![FilterState::default(); VE_BANDS * channels],
            stretch: 0,
            mark: vec![0; 128],
            current: 0,
            curmark: 0,
            cursor: i64::from(blocksizes[1] / 2),
            vec: vec![0f32; n],
            vec_out: vec![0f32; n / 2],
            work: vec![0f32; n],
        }
    }

    /// Port of envelope.c `_ve_amp` for one channel's filter bank.
    fn amp(&mut self, gi: &InfoPsyGlobal, data: &[f32], fbase: usize) -> i32 {
        let n = self.winlength;
        let mut ret = 0;
        let min_v = self.minenergy;

        // stretch is used to gradually lengthen the number of windows
        // considered previous-to-potential-trigger
        let stretch = VE_MINSTRETCH.max(self.stretch / 2);
        let mut penalty = gi.stretch_penalty - (self.stretch / 2 - VE_MINSTRETCH) as f32;
        if penalty < 0. {
            penalty = 0.;
        }
        if penalty > gi.stretch_penalty {
            penalty = gi.stretch_penalty;
        }

        // window and transform
        for ((v, &d), &w) in self.vec.iter_mut().zip(&data[..n]).zip(&self.mdct_win) {
            *v = d * w;
        }
        self.mdct
            .forward(&self.vec, &mut self.vec_out, &mut self.work);
        let vec = &mut self.vec_out;

        // near-DC spreading function
        let mut decay;
        {
            let f = &mut self.filter[fbase];
            let temp = (f64::from(vec[0] * vec[0])
                + 0.7 * f64::from(vec[1]) * f64::from(vec[1])
                + 0.2 * f64::from(vec[2]) * f64::from(vec[2])) as f32;
            let ptr = f.nearptr;

            // the accumulation is regularly refreshed from scratch to avoid
            // floating point creep
            if ptr == 0 {
                f.near_dc_acc = f.near_dc_partialacc + temp;
                decay = f.near_dc_acc;
                f.near_dc_partialacc = temp;
            } else {
                f.near_dc_acc += temp;
                decay = f.near_dc_acc;
                f.near_dc_partialacc += temp;
            }
            f.near_dc_acc -= f.near_dc[ptr];
            f.near_dc[ptr] = temp;

            decay = (f64::from(decay) * (1. / (VE_NEARDC as f64 + 1.))) as f32;
            f.nearptr += 1;
            if f.nearptr >= VE_NEARDC {
                f.nearptr = 0;
            }
            decay = (f64::from(todb(decay)) * 0.5 - 15.) as f32;
        }

        // perform spreading and limiting, also smooth the spectrum
        let mut i = 0;
        while i < n / 2 {
            let mut val = vec[i] * vec[i] + vec[i + 1] * vec[i + 1];
            val = todb(val) * 0.5;
            if val < decay {
                val = decay;
            }
            if val < min_v {
                val = min_v;
            }
            vec[i >> 1] = val;
            decay = (f64::from(decay) - 8.) as f32;
            i += 2;
        }

        // perform preecho/postecho triggering by band
        for j in 0..VE_BANDS {
            let band = &self.band[j];
            let mut acc = 0f32;
            for i in 0..band.end {
                acc += vec[i + band.begin] * band.window[i];
            }
            acc *= band.total;

            // convert amplitude to delta
            let f = &mut self.filter[fbase + j];
            let this = f.ampptr;
            let mut premax = -99999f32;
            let mut premin = 99999f32;
            let mut p = this as isize - 1;
            if p < 0 {
                p += VE_AMP as isize;
            }
            let postmax = fmax(acc, f.ampbuf[p as usize]);
            let postmin = fmin(acc, f.ampbuf[p as usize]);
            for _ in 0..stretch {
                p -= 1;
                if p < 0 {
                    p += VE_AMP as isize;
                }
                premax = fmax(premax, f.ampbuf[p as usize]);
                premin = fmin(premin, f.ampbuf[p as usize]);
            }
            let valmin = postmin - premin;
            let valmax = postmax - premax;

            f.ampbuf[this] = acc;
            f.ampptr += 1;
            if f.ampptr >= VE_AMP {
                f.ampptr = 0;
            }

            // look at min/max, decide trigger
            if valmax > gi.preecho_thresh[j] + penalty {
                ret |= 1;
                ret |= 4;
            }
            if valmin < gi.postecho_thresh[j] - penalty {
                ret |= 2;
            }
        }
        ret
    }

    /// Port of envelope.c `_ve_envelope_search`. Returns 1 (next block is
    /// long), 0 (short) or -1 (need more data).
    pub(crate) fn search(&mut self, v: &DspView<'_>, gi: &InfoPsyGlobal) -> i32 {
        let step = self.searchstep;
        let first = (self.current / step).max(0);
        let last = v.pcm_current / step - VE_WIN;

        // make sure we have enough storage to match the PCM
        let need = (last + VE_WIN + VE_POST).max(0) as usize;
        if need > self.mark.len() {
            self.mark.resize(need, 0);
        }

        let mut j = first;
        while j < last {
            let mut ret = 0;
            self.stretch += 1;
            if self.stretch > VE_MAXSTRETCH * 2 {
                self.stretch = VE_MAXSTRETCH * 2;
            }
            for i in 0..self.ch {
                let off = (step * j) as usize;
                let pcm = &v.pcm[i][off..off + self.winlength];
                ret |= self.amp(gi, pcm, i * VE_BANDS);
            }
            let ju = j as usize;
            self.mark[ju + VE_POST as usize] = 0;
            if ret & 1 != 0 {
                self.mark[ju] = 1;
                self.mark[ju + 1] = 1;
            }
            if ret & 2 != 0 {
                self.mark[ju] = 1;
                if j > 0 {
                    self.mark[ju - 1] = 1;
                }
            }
            if ret & 4 != 0 {
                self.stretch = -1;
            }
            j += 1;
        }

        self.current = last * step;

        let center_w = v.center_w;
        let bs = v.blocksizes;
        let test_w =
            center_w + i64::from(bs[v.w] / 4) + i64::from(bs[1] / 2) + i64::from(bs[0] / 4);

        let mut j = self.cursor;
        while j < self.current - step {
            // account for postecho working back one window
            if j >= test_w {
                return 1;
            }
            self.cursor = j;
            if self.mark[(j / step) as usize] != 0 && j > center_w {
                self.curmark = j;
                if j >= test_w {
                    return 1;
                }
                return 0;
            }
            j += step;
        }
        -1
    }

    /// Port of envelope.c `_ve_envelope_mark`.
    pub(crate) fn mark(&self, v: &DspView<'_>) -> bool {
        let bs = v.blocksizes;
        let center_w = v.center_w;
        let mut begin_w = center_w - i64::from(bs[v.w] / 4);
        let mut end_w = center_w + i64::from(bs[v.w] / 4);
        if v.w != 0 {
            begin_w -= i64::from(bs[v.lw] / 4);
            end_w += i64::from(bs[v.nw] / 4);
        } else {
            begin_w -= i64::from(bs[0] / 4);
            end_w += i64::from(bs[0] / 4);
        }

        if self.curmark >= begin_w && self.curmark < end_w {
            return true;
        }
        let first = begin_w / self.searchstep;
        let last = end_w / self.searchstep;
        (first..last).any(|i| self.mark[i as usize] != 0)
    }

    /// Port of envelope.c `_ve_envelope_shift`.
    pub(crate) fn shift(&mut self, shift: i64) {
        // adjust for placing marks ahead of ve->current
        let smallsize = (self.current / self.searchstep + VE_POST) as usize;
        let smallshift = (shift / self.searchstep) as usize;
        self.mark.copy_within(smallshift..smallsize, 0);
        self.current -= shift;
        if self.curmark >= 0 {
            self.curmark -= shift;
        }
        self.cursor -= shift;
    }
}

/// C `max(x,y)` macro from os.h: `((x)<(y)?(y):(x))`
#[inline]
fn fmax(x: f32, y: f32) -> f32 {
    if x < y { y } else { x }
}

/// C `min(x,y)` macro from os.h: `((x)>(y)?(y):(x))`
#[inline]
fn fmin(x: f32, y: f32) -> f32 {
    if x > y { y } else { x }
}
