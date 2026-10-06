//! Port of mapping0.c `mapping0_forward` (VBR path: a single packet blob,
//! `PACKETBLOBS/2`) together with the encoder backend state it uses
//! (`private_state` lookups set up by block.c `_vds_shared_init`).

use super::bitpack::OggPackBuffer;
use super::codebook::Codebook;
use super::floor1::LookFloor1;
use super::mdct::MdctLookup;
use super::os::{ilog, todb};
use super::psy::{PsyLook, PsyScratch, couple_quantize_normalize};
use super::res0::LookResidue0;
use super::setup::{CodecSetup, PACKETBLOBS};
use super::smallft::DrftLookup;
use super::window::apply_window;

/// The encoder-side lookups of `private_state` (codec_internal.h).
#[derive(Debug)]
pub(crate) struct Backend {
    pub(crate) ci: CodecSetup,
    modebits: u32,
    window: [usize; 2],
    transform: [MdctLookup; 2],
    fft_look: [DrftLookup; 2],
    books: Vec<Codebook>,
    psy: Vec<PsyLook>,
    flr: Vec<LookFloor1>,
    residue: Vec<LookResidue0>,
    scratch: Scratch,
}

/// Per-block working storage (the C code uses `_vorbis_block_alloc`/alloca).
#[derive(Debug, Default)]
struct Scratch {
    gmdct: Vec<Vec<f32>>,
    iwork: Vec<Vec<i32>>,
    noise: Vec<f32>,
    tone: Vec<f32>,
    mdct_w: Vec<f32>,
    fft_ch: Vec<f32>,
    res_work: Vec<i32>,
    psy: PsyScratch,
}

/// One analysis block (`vorbis_block`): `pcmend` samples per channel (raw
/// PCM; windowed in place by `mapping0_forward`) and the block-shape flags.
#[derive(Debug, Default)]
pub(crate) struct Block {
    pub(crate) pcm: Vec<Vec<f32>>,
    pub(crate) lw: usize,
    pub(crate) w: usize,
    pub(crate) nw: usize,
    pub(crate) pcmend: usize,
    pub(crate) blocktype: usize,
    pub(crate) eofflag: bool,
    pub(crate) granulepos: i64,
}

impl Backend {
    /// The encode half of block.c `_vds_shared_init` (+ the lookups that
    /// `vorbis_analysis_init` sets up for the backends).
    pub(crate) fn new(ci: CodecSetup) -> Option<Self> {
        let bs = ci.blocksizes;
        let books = ci
            .books
            .iter()
            .map(|b| Codebook::init_encode(b))
            .collect::<Option<Vec<_>>>()?;
        let psy = ci
            .psy_param
            .iter()
            .map(|p| {
                PsyLook::new(
                    p,
                    &ci.psy_g_param,
                    (bs[p.blockflag as usize] / 2) as usize,
                    ci.rate,
                )
            })
            .collect();
        let flr = ci.floors.iter().map(LookFloor1::new).collect();
        let residue = ci.residues.iter().map(LookResidue0::new).collect();
        Some(Backend {
            modebits: ilog((ci.modes.len() - 1) as u32) as u32,
            window: [
                (ilog(bs[0] as u32) - 7) as usize,
                (ilog(bs[1] as u32) - 7) as usize,
            ],
            transform: [
                MdctLookup::new(bs[0] as usize),
                MdctLookup::new(bs[1] as usize),
            ],
            fft_look: [
                DrftLookup::new(bs[0] as usize),
                DrftLookup::new(bs[1] as usize),
            ],
            books,
            psy,
            flr,
            residue,
            scratch: Scratch::default(),
            ci,
        })
    }

    /// Port of mapping0.c `mapping0_forward` followed by the VBR packet
    /// hand-off of analysis.c `vorbis_analysis`. `ampmax` is the block's
    /// `vorbis_block_internal.ampmax` (in: global max, out: updated).
    pub(crate) fn mapping0_forward(
        &mut self,
        vb: &mut Block,
        ampmax: &mut f32,
        opb: &mut OggPackBuffer,
    ) {
        let channels = self.ci.channels as usize;
        let n = vb.pcmend;
        let half = n / 2;
        let modenumber = vb.w;
        let info = &self.ci.maps[modenumber];
        let psy_index = vb.blocktype + if vb.w != 0 { 2 } else { 0 };
        let mut global_ampmax = *ampmax;
        let mut local_ampmax = vec![0f32; channels];
        let mut nonzero = vec![false; channels];

        let sc = &mut self.scratch;
        sc.gmdct.resize_with(channels, Vec::new);
        sc.iwork.resize_with(channels, Vec::new);
        for i in 0..channels {
            sc.gmdct[i].clear();
            sc.gmdct[i].resize(half, 0.);
            sc.iwork[i].clear();
            sc.iwork[i].resize(half, 0);
        }
        sc.mdct_w.resize(n, 0.);
        sc.fft_ch.resize(n, 0.);

        for (i, lamp) in local_ampmax.iter_mut().enumerate() {
            let scale = 4.0f32 / n as f32;
            // + .345 is a hack; the original todB estimation used on IEEE 754
            // compliant machines had a bug that returned dB values about a
            // third of a decibel too high. (see mapping0.c)
            let scale_db = (f64::from(todb(scale)) + 0.345) as f32;
            let pcm = &mut vb.pcm[i];

            // window the PCM data
            apply_window(pcm, &self.window, &self.ci.blocksizes, vb.lw, vb.w, vb.nw);

            // transform the PCM data; only MDCT right now....
            self.transform[vb.w].forward(pcm, &mut sc.gmdct[i], &mut sc.mdct_w);

            // FFT yields more accurate tonal estimation (not phase sensitive)
            self.fft_look[vb.w].forward(pcm, &mut sc.fft_ch);
            // logfft aliases pcm (written in place, as in C)
            pcm[0] = (f64::from(scale_db + todb(pcm[0])) + 0.345) as f32;
            *lamp = pcm[0];
            let mut j = 1;
            while j < n - 1 {
                let temp = pcm[j] * pcm[j] + pcm[j + 1] * pcm[j + 1];
                // C: temp=logfft[(j+1)>>1]=scale_dB+.5f*todB(&temp) + .345;
                let temp = (f64::from(scale_db + 0.5 * todb(temp)) + 0.345) as f32;
                pcm[(j + 1) >> 1] = temp;
                if temp > *lamp {
                    *lamp = temp;
                }
                j += 2;
            }

            if *lamp > 0. {
                *lamp = 0.;
            }
            if *lamp > global_ampmax {
                global_ampmax = *lamp;
            }
        }

        sc.noise.resize(half, 0.);
        sc.tone.resize(half, 0.);
        let psy_look = &self.psy[psy_index];
        let mut floor_posts: Vec<Option<Vec<i32>>> = Vec::with_capacity(channels);
        for (i, &lamp) in local_ampmax.iter().enumerate() {
            // the encoder setup assumes that all the modes used by any
            // specific bitrate tweaking use the same floor
            let submap = info.chmuxlist[i] as usize;
            let mdct = &mut sc.gmdct[i];
            let (logfft, logmdct) = vb.pcm[i].split_at_mut(half);

            for j in 0..half {
                logmdct[j] = (f64::from(todb(mdct[j])) + 0.345) as f32;
            }

            // first step; noise masking
            psy_look.noisemask(&mut sc.psy, logmdct, &mut sc.noise);

            // second step: 'all the other crap'; tone masking, peak limiting and ATH
            psy_look.tonemask(&mut sc.psy, logfft, &mut sc.tone, global_ampmax, lamp);

            // third step; we offset the noise vectors, overlay tone masking.
            // logmask aliases logfft.
            psy_look.offset_and_mix(&sc.noise, &sc.tone, 1, logfft, mdct, logmdct);

            // this algorithm is hardwired to floor 1 for now
            let floor = &self.flr[info.floorsubmap[submap] as usize];
            floor_posts.push(floor.fit(logmdct, logfft));
        }

        *ampmax = global_ampmax;

        // the next phases are performed once for vbr-only:
        // 1) encode actual mode being used
        // 2) encode the floor for each channel, compute coded mask curve/res
        // 3) normalize and couple.
        // 4) encode residue
        let k = PACKETBLOBS / 2;

        // Encode the packet type
        opb.write(0, 1);
        // Encode the modenumber; frame mode, pre,post windowsize, then dispatch
        opb.write(modenumber as u32, self.modebits);
        if vb.w != 0 {
            opb.write(vb.lw as u32, 1);
            opb.write(vb.nw as u32, 1);
        }

        // encode floor, compute masking curve, sep out residue
        for i in 0..channels {
            let submap = info.chmuxlist[i] as usize;
            let floor = &self.flr[info.floorsubmap[submap] as usize];
            nonzero[i] = floor.encode(
                opb,
                &self.books,
                floor_posts[i].as_deref_mut(),
                &mut sc.iwork[i],
                half,
            );
        }

        // quantize/couple
        couple_quantize_normalize(
            &mut sc.psy,
            k,
            &self.ci.psy_g_param,
            psy_look,
            info,
            &sc.gmdct,
            &mut sc.iwork,
            &mut nonzero,
            self.ci.psy_g_param.sliding_lowpass[vb.w][k],
            channels,
        );

        // classify and encode by submap
        for i in 0..info.submaps as usize {
            let resnum = info.residuesubmap[i] as usize;
            let rtype = self.ci.residue_types[resnum];
            let look = &self.residue[resnum];
            let mut zerobundle = Vec::with_capacity(channels);
            let mut bundle: Vec<&mut [i32]> = Vec::with_capacity(channels);
            for (j, iw) in sc.iwork.iter_mut().enumerate() {
                if info.chmuxlist[j] as usize == i {
                    zerobundle.push(nonzero[j]);
                    bundle.push(&mut iw[..]);
                }
            }
            let classifications = {
                let ro: Vec<&[i32]> = bundle.iter().map(|v| &**v).collect();
                look.class(rtype, &ro, &zerobundle)
            };
            look.forward(
                rtype,
                opb,
                &self.books,
                &mut bundle,
                &zerobundle,
                classifications.as_deref(),
                half,
                &mut sc.res_work,
            );
        }
    }
}
