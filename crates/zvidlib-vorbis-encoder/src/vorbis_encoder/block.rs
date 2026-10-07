//! Port of the analysis side of block.c (`vorbis_analysis_init`,
//! `vorbis_analysis_buffer`, `vorbis_analysis_wrote` including the LPC
//! extrapolation of both stream edges, `vorbis_analysis_blockout`) and of the
//! packet hand-off of analysis.c / bitrate.c in VBR mode.

use super::bitpack::OggPackBuffer;
use super::envelope::{DspView, EnvelopeLookup};
use super::lpc::{lpc_from_data, lpc_predict};
use super::mapping0::{Backend, Block};
use super::psy::ampmax_decay;
use super::setup::CodecSetup;

/// `vorbis_dsp_state` + `private_state` + the (reused) `vorbis_block`, i.e.
/// everything the analysis loop of libvorbis keeps between calls.
#[derive(Debug)]
pub(crate) struct Analysis {
    pub(crate) backend: Backend,
    ve: EnvelopeLookup,
    pcm: Vec<Vec<f32>>,
    pcm_current: i64,
    center_w: i64,
    lw: usize,
    w: usize,
    nw: usize,
    /// 0: not at EOF yet; >0: index of the last real sample; -1: done
    eofflag: i64,
    granulepos: i64,
    preextrapolate: bool,
    /// `vorbis_look_psy_global.ampmax`
    g_ampmax: f32,
    /// `vorbis_block_internal.ampmax` of the reused vorbis_block
    vbi_ampmax: f32,
    block: Block,
    opb: OggPackBuffer,
}

impl Analysis {
    /// Port of block.c `vorbis_analysis_init` (+ `_vds_shared_init`).
    pub(crate) fn new(ci: CodecSetup) -> Option<Self> {
        let channels = ci.channels as usize;
        let bs = ci.blocksizes;
        let ve = EnvelopeLookup::new(&ci.psy_g_param, channels, bs);
        let backend = Backend::new(ci)?;
        let storage = bs[1] as usize;
        Some(Analysis {
            backend,
            ve,
            pcm: vec![vec![0f32; storage]; channels],
            pcm_current: i64::from(bs[1] / 2),
            center_w: i64::from(bs[1] / 2),
            lw: 0,
            w: 0,
            nw: 0,
            eofflag: 0,
            granulepos: 0,
            preextrapolate: false,
            g_ampmax: -9999.,
            vbi_ampmax: -9999.,
            block: Block::default(),
            opb: OggPackBuffer::new(),
        })
    }

    /// Whether the start of the stream has been extrapolated yet (after which
    /// write chunking no longer affects the output).
    pub(crate) fn preextrapolated(&self) -> bool {
        self.preextrapolate
    }

    fn ci(&self) -> &CodecSetup {
        &self.backend.ci
    }

    /// Port of block.c `vorbis_analysis_buffer`: make room for `vals` more
    /// samples per channel.
    fn buffer(&mut self, vals: usize) {
        let need = self.pcm_current as usize + vals;
        if need >= self.pcm[0].len() {
            let storage = self.pcm_current as usize + vals * 2;
            for p in &mut self.pcm {
                p.resize(storage, 0.);
            }
        }
    }

    /// `vorbis_analysis_buffer` + copy + `vorbis_analysis_wrote(frames)` for
    /// interleaved input (`interleaved.len() == frames * channels`).
    pub(crate) fn write_interleaved(&mut self, interleaved: &[f32]) {
        let ch = self.pcm.len();
        let frames = interleaved.len() / ch;
        if frames == 0 {
            return;
        }
        self.buffer(frames);
        let start = self.pcm_current as usize;
        for (c, p) in self.pcm.iter_mut().enumerate() {
            for (dst, src) in p[start..start + frames]
                .iter_mut()
                .zip(interleaved[c..].iter().step_by(ch))
            {
                *dst = *src;
            }
        }
        self.wrote(frames);
    }

    /// Port of block.c `_preextrapolate_helper`.
    fn preextrapolate_helper(&mut self) {
        let order = 16;
        let mut lpc = [0f32; 16];
        let cur = self.pcm_current as usize;
        let center = self.center_w as usize;
        self.preextrapolate = true;

        if self.pcm_current - self.center_w > (order * 2) as i64 {
            // safety
            let mut work = vec![0f32; cur];
            for p in &mut self.pcm {
                // need to run the extrapolation in reverse!
                for j in 0..cur {
                    work[j] = p[cur - j - 1];
                }
                // prime as above
                lpc_from_data(&work[..cur - center], &mut lpc, order);
                // run the predictor filter
                let (prime, data) = work.split_at_mut(cur - center);
                lpc_predict(
                    &lpc,
                    &prime[cur - center - order..],
                    order,
                    &mut data[..center],
                );
                for j in 0..cur {
                    p[cur - j - 1] = work[j];
                }
            }
        }
    }

    /// Port of block.c `vorbis_analysis_wrote` with `vals > 0`.
    fn wrote(&mut self, vals: usize) {
        self.pcm_current += vals as i64;
        // we may want to reverse extrapolate the beginning of a stream too...
        // in case we're beginning on a cliff!
        if !self.preextrapolate
            && self.pcm_current - self.center_w > i64::from(self.ci().blocksizes[1])
        {
            self.preextrapolate_helper();
        }
    }

    /// Port of block.c `vorbis_analysis_wrote(v, 0)`: mark end of stream and
    /// extrapolate a few blocks past it.
    pub(crate) fn wrote_eof(&mut self) {
        let order = 32;
        let mut lpc = [0f32; 32];
        let bs1 = self.ci().blocksizes[1] as usize;

        // if it wasn't done earlier (very short sample)
        if !self.preextrapolate {
            self.preextrapolate_helper();
        }

        // We're encoding the end of the stream. Just make sure we have
        // [at least] a few full blocks of zeroes at the end; actually we
        // extrapolate (LPC) rather than drop a large amplitude off a cliff.
        self.buffer(bs1 * 3);
        self.eofflag = self.pcm_current;
        self.pcm_current += (bs1 * 3) as i64;
        let eof = self.eofflag as usize;
        let cur = self.pcm_current as usize;

        for p in &mut self.pcm {
            if self.eofflag > (order * 2) as i64 {
                // extrapolate with LPC to fill in; make a predictor filter
                let n = eof.min(bs1);
                lpc_from_data(&p[eof - n..eof], &mut lpc, order);
                // run the predictor filter
                let prime: Vec<f32> = p[eof - order..eof].to_vec();
                lpc_predict(&lpc, &prime, order, &mut p[eof..cur]);
            } else {
                // not enough data to extrapolate; zeroes will do.
                for v in &mut p[eof..cur] {
                    *v = 0.;
                }
            }
        }
    }

    fn view(&self) -> DspView<'_> {
        DspView {
            pcm: &self.pcm,
            pcm_current: self.pcm_current,
            center_w: self.center_w,
            w: self.w,
            lw: self.lw,
            nw: self.nw,
            blocksizes: self.ci().blocksizes,
        }
    }

    /// Port of block.c `vorbis_analysis_blockout`: returns `true` when a block
    /// was produced into `self.block`.
    fn blockout(&mut self) -> bool {
        let bs = self.ci().blocksizes;
        let begin_w = self.center_w - i64::from(bs[self.w] / 2);

        // check to see if we're started...
        if !self.preextrapolate {
            return false;
        }
        // check to see if we're done...
        if self.eofflag == -1 {
            return false;
        }

        // By our invariant, we have lW, W and centerW set. Search for the
        // next boundary so we can determine nW (the next window size) which
        // lets us compute the shape of the current block's window. We do an
        // envelope search even on a single blocksize; we may still be
        // throwing more bits at impulses, and envelope search handles marking
        // impulses too.
        {
            let gi = self.backend.ci.psy_g_param;
            let view = DspView {
                pcm: &self.pcm,
                pcm_current: self.pcm_current,
                center_w: self.center_w,
                w: self.w,
                lw: self.lw,
                nw: self.nw,
                blocksizes: bs,
            };
            let bp = self.ve.search(&view, &gi);
            if bp == -1 {
                if self.eofflag == 0 {
                    return false; // not enough data currently to search for a full long block
                }
                self.nw = 0;
            } else if bs[0] == bs[1] {
                self.nw = 0;
            } else {
                self.nw = bp as usize;
            }
        }

        let center_next = self.center_w + i64::from(bs[self.w] / 4) + i64::from(bs[self.nw] / 4);
        {
            // center of next block + next block maximum right side.
            let blockbound = center_next + i64::from(bs[self.nw] / 2);
            if self.pcm_current < blockbound {
                return false; // not enough data yet
            }
        }

        // fill in the block. Note that for a short window, lW and nW are
        // *short* regardless of actual settings in the stream
        let blocktype = if self.w != 0 {
            if self.lw == 0 || self.nw == 0 {
                0 // BLOCKTYPE_TRANSITION
            } else {
                1 // BLOCKTYPE_LONG
            }
        } else if self.ve.mark(&self.view()) {
            0 // BLOCKTYPE_IMPULSE
        } else {
            1 // BLOCKTYPE_PADDING
        };

        let pcmend = bs[self.w] as usize;
        self.block.lw = self.lw;
        self.block.w = self.w;
        self.block.nw = self.nw;
        self.block.blocktype = blocktype;
        self.block.granulepos = self.granulepos;
        self.block.pcmend = pcmend;
        self.block.eofflag = false;

        // this tracks 'strongest peak' for later psychoacoustics
        if self.vbi_ampmax > self.g_ampmax {
            self.g_ampmax = self.vbi_ampmax;
        }
        self.g_ampmax = ampmax_decay(
            self.g_ampmax,
            &self.backend.ci.psy_g_param,
            bs[self.w],
            self.backend.ci.rate,
        );
        self.vbi_ampmax = self.g_ampmax;

        let channels = self.pcm.len();
        self.block.pcm.resize_with(channels, Vec::new);
        let bw = begin_w as usize;
        for (dst, src) in self.block.pcm.iter_mut().zip(&self.pcm) {
            dst.clear();
            dst.extend_from_slice(&src[bw..bw + pcmend]);
        }

        // handle eof detection: eof==0 means that we've not yet received EOF
        // eof>0 marks the last 'real' sample in pcm[]; eof<0 'no more to do'
        if self.eofflag != 0 && self.center_w >= self.eofflag {
            self.eofflag = -1;
            self.block.eofflag = true;
            return true;
        }

        // advance storage vectors and clean up
        let new_center_next = i64::from(bs[1] / 2);
        let movement_w = center_next - new_center_next;

        if movement_w > 0 {
            self.ve.shift(movement_w);
            self.pcm_current -= movement_w;
            let mw = movement_w as usize;
            let cur = self.pcm_current as usize;
            for p in &mut self.pcm {
                p.copy_within(mw..mw + cur, 0);
            }

            self.lw = self.w;
            self.w = self.nw;
            self.center_w = new_center_next;

            if self.eofflag != 0 {
                self.eofflag -= movement_w;
                if self.eofflag <= 0 {
                    self.eofflag = -1;
                }
                // do not add padding to end of stream!
                if self.center_w >= self.eofflag {
                    self.granulepos += movement_w - (self.center_w - self.eofflag);
                } else {
                    self.granulepos += movement_w;
                }
            } else {
                self.granulepos += movement_w;
            }
        }
        true
    }

    /// Drain every complete block: `vorbis_analysis_blockout` +
    /// `vorbis_analysis` + `vorbis_bitrate_addblock`/`flushpacket` (VBR).
    pub(crate) fn drain(&mut self, out: &mut Vec<(Vec<u8>, u64)>) -> bool {
        let mut eos = false;
        while self.blockout() {
            self.opb.reset();
            let mut ampmax = self.vbi_ampmax;
            self.backend
                .mapping0_forward(&mut self.block, &mut ampmax, &mut self.opb);
            self.vbi_ampmax = ampmax;
            out.push((self.opb.to_vec(), self.block.granulepos as u64));
            if self.block.eofflag {
                eos = true;
                break;
            }
        }
        eos
    }
}
