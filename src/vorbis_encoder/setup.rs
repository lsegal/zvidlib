//! Port of the VBR part of vorbisenc.c: template selection and the
//! "highlevel" setup that turns a quality value into a complete
//! `codec_setup_info` (books, floors, residues, mappings, modes, psy params).

use super::tables::gen_modes::{MODE_TEMPLATE, PSY_INFO_TEMPLATE, SETUP_LIST};
use super::tables::types::{
    AdjStereo, Att3, CompandBlock, InfoFloor1, InfoMapping0, InfoMode, InfoPsy, InfoPsyGlobal,
    InfoResidue0, MappingTemplate, Noise3, NoiseGuard, ResidueTemplate, SetupDataTemplate,
    StaticCodebook, VpAdjBlock,
};

/// `PACKETBLOBS` (codec_internal.h)
pub(crate) const PACKETBLOBS: usize = 15;

/// Port of `highlevel_byblocktype` (highlevel.h).
#[derive(Debug, Clone, Copy, Default)]
struct HighLevelByBlockType {
    tone_mask_setting: f64,
    tone_peaklimit_setting: f64,
    noise_bias_setting: f64,
    noise_compand_setting: f64,
}

/// Port of `highlevel_encode_setup` (highlevel.h), VBR members only.
#[derive(Debug, Clone)]
struct HighLevel {
    setup: &'static SetupDataTemplate,
    base_setting: f64,
    impulse_noisetune: f64,
    impulse_block_p: bool,
    noise_normalize_p: bool,
    stereo_point_setting: f64,
    lowpass_khz: f64,
    ath_floating_db: f64,
    ath_absolute_db: f64,
    amplitude_track_db_per_sec: f64,
    trigger_setting: f64,
    block: [HighLevelByBlockType; 4],
}

/// The parts of `vorbis_info` + `codec_setup_info` the encoder needs.
#[derive(Debug, Clone)]
pub(crate) struct CodecSetup {
    pub(crate) channels: i32,
    pub(crate) rate: i32,
    pub(crate) bitrate_upper: i32,
    pub(crate) bitrate_nominal: i32,
    pub(crate) bitrate_lower: i32,
    pub(crate) blocksizes: [i32; 2],
    pub(crate) modes: Vec<InfoMode>,
    /// mapping type is always 0
    pub(crate) maps: Vec<InfoMapping0>,
    /// floor type is always 1
    pub(crate) floors: Vec<InfoFloor1>,
    pub(crate) residue_types: Vec<i32>,
    pub(crate) residues: Vec<InfoResidue0>,
    pub(crate) books: Vec<&'static StaticCodebook>,
    pub(crate) psy_param: Vec<InfoPsy>,
    pub(crate) psy_g_param: InfoPsyGlobal,
}

/// Port of vorbisenc.c `get_setup_template`. `req` is a quality (when
/// `q_or_bitrate` is false) or a bitrate in bits/s for all channels.
/// Returns the template and the `base_setting`.
pub(crate) fn get_setup_template(
    ch: i64,
    srate: i64,
    mut req: f64,
    q_or_bitrate: bool,
) -> Option<(&'static SetupDataTemplate, f64)> {
    if q_or_bitrate {
        req /= ch as f64;
    }
    for t in SETUP_LIST {
        if t.coupling_restriction != -1 && i64::from(t.coupling_restriction) != ch {
            continue;
        }
        if srate < i64::from(t.samplerate_min_restriction)
            || srate > i64::from(t.samplerate_max_restriction)
        {
            continue;
        }
        let mappings = t.mappings as usize;
        let map = if q_or_bitrate {
            t.rate_mapping
        } else {
            t.quality_mapping
        };
        // the template matches. Does the requested quality mode fall within
        // this template's modes?
        if req < map[0] || req > map[mappings] {
            continue;
        }
        let mut j = 0;
        while j < mappings {
            if req >= map[j] && req < map[j + 1] {
                break;
            }
            j += 1;
        }
        let base_setting = if j == mappings {
            j as f64 - 0.001
        } else {
            let low = map[j] as f32;
            let high = map[j + 1] as f32;
            let del = ((req - f64::from(low)) / f64::from(high - low)) as f32;
            // C: *base_setting = j + del;  (int + float -> float)
            f64::from(j as f32 + del)
        };
        return Some((t, base_setting));
    }
    None
}

/// `is=s; ds=s-is;` interpolation helper used throughout vorbisenc.c.
#[inline]
fn split(s: f64) -> (usize, f64) {
    let is = s as i32;
    (is as usize, s - f64::from(is))
}

/// Port of vorbisenc.c `vorbis_encode_setup_setting`.
fn setup_setting(setup: &'static SetupDataTemplate, base_setting: f64) -> HighLevel {
    let (is, ds) = split(base_setting);
    let lowpass_khz = setup.psy_lowpass[is] * (1. - ds) + setup.psy_lowpass[is + 1] * ds;
    let ath_floating_db = f64::from(setup.psy_ath_float[is]) * (1. - ds)
        + f64::from(setup.psy_ath_float[is + 1]) * ds;
    let ath_absolute_db =
        f64::from(setup.psy_ath_abs[is]) * (1. - ds) + f64::from(setup.psy_ath_abs[is + 1]) * ds;
    let b = HighLevelByBlockType {
        tone_mask_setting: base_setting,
        tone_peaklimit_setting: base_setting,
        noise_bias_setting: base_setting,
        noise_compand_setting: base_setting,
    };
    HighLevel {
        setup,
        base_setting,
        impulse_noisetune: 0.,
        impulse_block_p: true,
        noise_normalize_p: true,
        stereo_point_setting: base_setting,
        lowpass_khz,
        ath_floating_db,
        ath_absolute_db,
        amplitude_track_db_per_sec: -6.,
        trigger_setting: base_setting,
        block: [b; 4],
    }
}

/// Port of vorbisenc.c `book_dup_or_new` (pointer identity, as in C).
fn book_dup_or_new(books: &mut Vec<&'static StaticCodebook>, book: &'static StaticCodebook) -> i32 {
    if let Some(i) = books.iter().position(|b| std::ptr::eq(*b, book)) {
        return i as i32;
    }
    books.push(book);
    (books.len() - 1) as i32
}

struct Builder {
    ci: CodecSetup,
    hi: HighLevel,
    psy_set: [bool; 4],
}

impl Builder {
    /// Port of vorbisenc.c `vorbis_encode_floor_setup`.
    fn floor_setup(
        &mut self,
        s: f64,
        books: &'static [&'static [Option<&'static StaticCodebook>]],
        params: &'static [InfoFloor1],
        x: &[i32],
    ) {
        let is = s as i32 as usize;
        let idx = x[is] as usize;
        let mut f = params[idx];
        let nbooks = self.ci.books.len() as i32;
        let mut maxclass = -1;
        let mut maxbook = -1;
        for i in 0..f.partitions as usize {
            maxclass = maxclass.max(f.partitionclass[i]);
        }
        for i in 0..(maxclass + 1) as usize {
            maxbook = maxbook.max(f.class_book[i]);
            f.class_book[i] += nbooks;
            for k in 0..(1usize << f.class_subs[i]) {
                maxbook = maxbook.max(f.class_subbook[i][k]);
                if f.class_subbook[i][k] >= 0 {
                    f.class_subbook[i][k] += nbooks;
                }
            }
        }
        // the floor book lists of the supported templates never hold NULL
        // where a floor actually references a book
        for b in books[idx].iter().take((maxbook + 1) as usize).flatten() {
            self.ci.books.push(b);
        }
        self.ci.floors.push(f);
    }

    /// Port of vorbisenc.c `vorbis_encode_global_psych_setup`.
    fn global_psych_setup(&mut self, s: f64, input: &'static [InfoPsyGlobal], x: &[f64]) {
        let (is, ds) = split(s);
        let mut g = input[x[is] as i32 as usize];
        let ds = x[is] * (1. - ds) + x[is + 1] * ds;
        let mut is = ds as i32;
        let mut ds = ds - f64::from(is);
        if ds == 0. && is > 0 {
            is -= 1;
            ds = 1.;
        }
        let is = is as usize;
        // interpolate the trigger threshholds (only the first 4 bands, as in C)
        for i in 0..4 {
            g.preecho_thresh[i] = (f64::from(input[is].preecho_thresh[i]) * (1. - ds)
                + f64::from(input[is + 1].preecho_thresh[i]) * ds)
                as f32;
            g.postecho_thresh[i] = (f64::from(input[is].postecho_thresh[i]) * (1. - ds)
                + f64::from(input[is + 1].postecho_thresh[i]) * ds)
                as f32;
        }
        g.ampmax_att_per_sec = self.hi.amplitude_track_db_per_sec as f32;
        self.ci.psy_g_param = g;
    }

    /// Port of vorbisenc.c `vorbis_encode_global_stereo` (non-managed).
    fn global_stereo(&mut self, p: Option<&'static [AdjStereo]>) {
        // C: float s=hi->stereo_point_setting; int is=s; double ds=s-is;
        let s = self.hi.stereo_point_setting as f32;
        let is = s as i32;
        let ds = f64::from(s) - f64::from(is);
        let is = is as usize;
        let rate = f64::from(self.ci.rate);
        let bs = self.ci.blocksizes;
        let g = &mut self.ci.psy_g_param;
        if let Some(p) = p {
            g.coupling_prepointamp = p[is].pre;
            g.coupling_postpointamp = p[is].post;
            let half = PACKETBLOBS / 2;
            let khz = (f64::from(p[is].khz[half]) * (1. - ds) + f64::from(p[is + 1].khz[half]) * ds)
                as f32;
            for i in 0..PACKETBLOBS {
                g.coupling_pointlimit[0][i] =
                    (f64::from(khz) * 1000. / rate * f64::from(bs[0])) as i32;
                g.coupling_pointlimit[1][i] =
                    (f64::from(khz) * 1000. / rate * f64::from(bs[1])) as i32;
                g.coupling_pkhz[i] = khz as i32;
            }
            let khz = (f64::from(p[is].lowpass_khz[half]) * (1. - ds)
                + f64::from(p[is + 1].lowpass_khz[half]) * ds) as f32;
            for i in 0..PACKETBLOBS {
                g.sliding_lowpass[0][i] = (f64::from(khz) * 1000. / rate * f64::from(bs[0])) as i32;
                g.sliding_lowpass[1][i] = (f64::from(khz) * 1000. / rate * f64::from(bs[1])) as i32;
            }
        } else {
            for i in 0..PACKETBLOBS {
                g.sliding_lowpass[0][i] = bs[0];
                g.sliding_lowpass[1][i] = bs[1];
            }
        }
    }

    /// Port of vorbisenc.c `vorbis_encode_psyset_setup`.
    fn psyset_setup(
        &mut self,
        s: f64,
        nn_start: &[i32],
        nn_partition: &[i32],
        nn_thresh: &[f64],
        block: usize,
    ) {
        let is = s as i32 as usize;
        let mut p = PSY_INFO_TEMPLATE;
        p.blockflag = (block >> 1) as i32;
        if self.hi.noise_normalize_p {
            p.normal_p = 1;
            p.normal_start = nn_start[is];
            p.normal_partition = nn_partition[is];
            p.normal_thresh = nn_thresh[is];
        }
        if self.ci.psy_param.len() <= block {
            self.ci.psy_param.resize(block + 1, PSY_INFO_TEMPLATE);
        }
        self.ci.psy_param[block] = p;
        self.psy_set[block] = true;
    }

    /// Port of vorbisenc.c `vorbis_encode_tonemask_setup`.
    fn tonemask_setup(
        &mut self,
        s: f64,
        block: usize,
        att: &[Att3],
        max: &[i32],
        input: &[VpAdjBlock],
    ) {
        let (is, ds) = split(s);
        let p = &mut self.ci.psy_param[block];
        for k in 0..3 {
            p.tone_masteratt[k] =
                (f64::from(att[is].att[k]) * (1. - ds) + f64::from(att[is + 1].att[k]) * ds) as f32;
        }
        p.tone_centerboost =
            (f64::from(att[is].boost) * (1. - ds) + f64::from(att[is + 1].boost) * ds) as f32;
        p.tone_decay =
            (f64::from(att[is].decay) * (1. - ds) + f64::from(att[is + 1].decay) * ds) as f32;
        p.max_curve_db = (f64::from(max[is]) * (1. - ds) + f64::from(max[is + 1]) * ds) as f32;
        for i in 0..17 {
            p.toneatt[i] = (f64::from(input[is].block[i]) * (1. - ds)
                + f64::from(input[is + 1].block[i]) * ds) as f32;
        }
    }

    /// Port of vorbisenc.c `vorbis_encode_compand_setup`.
    fn compand_setup(&mut self, s: f64, block: usize, input: &[CompandBlock], x: &[f64]) {
        let (is, ds) = split(s);
        let ds = x[is] * (1. - ds) + x[is + 1] * ds;
        let mut is = ds as i32;
        let mut ds = ds - f64::from(is);
        if ds == 0. && is > 0 {
            is -= 1;
            ds = 1.;
        }
        let is = is as usize;
        let p = &mut self.ci.psy_param[block];
        for i in 0..40 {
            p.noisecompand[i] = (f64::from(input[is].data[i]) * (1. - ds)
                + f64::from(input[is + 1].data[i]) * ds) as f32;
        }
    }

    /// Port of vorbisenc.c `vorbis_encode_peak_setup`.
    fn peak_setup(&mut self, s: f64, block: usize, suppress: &[i32]) {
        let (is, ds) = split(s);
        self.ci.psy_param[block].tone_abs_limit =
            (f64::from(suppress[is]) * (1. - ds) + f64::from(suppress[is + 1]) * ds) as f32;
    }

    /// Port of vorbisenc.c `vorbis_encode_noisebias_setup`.
    fn noisebias_setup(
        &mut self,
        s: f64,
        block: usize,
        suppress: &[i32],
        input: &[Noise3],
        guard: &[NoiseGuard],
        userbias: f64,
    ) {
        let (is, ds) = split(s);
        let p = &mut self.ci.psy_param[block];
        p.noisemaxsupp =
            (f64::from(suppress[is]) * (1. - ds) + f64::from(suppress[is + 1]) * ds) as f32;
        p.noisewindowlomin = guard[block].lo;
        p.noisewindowhimin = guard[block].hi;
        p.noisewindowfixed = guard[block].fixed;
        for j in 0..3 {
            for i in 0..17 {
                p.noiseoff[j][i] = (f64::from(input[is].data[j][i]) * (1. - ds)
                    + f64::from(input[is + 1].data[j][i]) * ds)
                    as f32;
            }
        }
        // impulse blocks may take a user specified bias to boost the
        // nominal/high noise encoding depth
        for j in 0..3 {
            let min = p.noiseoff[j][0] + 6.; // the lowest it can go
            for i in 0..17 {
                p.noiseoff[j][i] = (f64::from(p.noiseoff[j][i]) + userbias) as f32;
                if p.noiseoff[j][i] < min {
                    p.noiseoff[j][i] = min;
                }
            }
        }
    }

    /// Port of vorbisenc.c `vorbis_encode_ath_setup`.
    fn ath_setup(&mut self, block: usize) {
        let p = &mut self.ci.psy_param[block];
        p.ath_adjatt = self.hi.ath_floating_db as f32;
        p.ath_maxatt = self.hi.ath_absolute_db as f32;
    }

    /// Port of vorbisenc.c `vorbis_encode_blocksize_setup`.
    fn blocksize_setup(&mut self, s: f64, shortb: &[i32], longb: &[i32]) {
        let is = s as i32 as usize;
        self.ci.blocksizes = [shortb[is], longb[is]];
    }

    /// Port of vorbisenc.c `vorbis_encode_residue_setup` (non-managed).
    fn residue_setup(&mut self, number: usize, block: usize, res: &'static ResidueTemplate) {
        let mut r = *res.res;
        if self.ci.residues.len() <= number {
            self.ci.residues.resize(number + 1, r);
            self.ci.residue_types.resize(number + 1, 0);
        }
        r.grouping = res.grouping;
        self.ci.residue_types[number] = res.res_type;

        // fill in all the books
        let partitions = r.partitions as usize;
        for i in 0..partitions {
            for k in 0..4 {
                if res.books_base.books[i][k].is_some() {
                    r.secondstages[i] |= 1 << k;
                }
            }
        }
        r.groupbook = book_dup_or_new(&mut self.ci.books, res.book_aux);
        let mut booklist = 0;
        for i in 0..partitions {
            for k in 0..4 {
                if let Some(b) = res.books_base.books[i][k] {
                    let bookid = book_dup_or_new(&mut self.ci.books, b);
                    r.booklist[booklist] = bookid;
                    booklist += 1;
                }
            }
        }

        // lowpass setup/pointlimit
        let mut freq = self.hi.lowpass_khz * 1000.;
        let nyq = f64::from(self.ci.rate) / 2.;
        let blocksize = f64::from(self.ci.blocksizes[block] >> 1);
        if freq > nyq {
            freq = nyq;
        }
        // in the floor, the granularity can be very fine
        self.ci.floors[block].n = (freq / nyq * blocksize) as i32;

        match res.limit_type {
            1 => {
                freq = f64::from(self.ci.psy_g_param.coupling_pkhz[PACKETBLOBS / 2]) * 1000.;
                if freq > nyq {
                    freq = nyq;
                }
            }
            2 => freq = 250.,
            _ => {}
        }

        let grouping = r.grouping;
        let blocksize_i = self.ci.blocksizes[block] >> 1;
        if self.ci.residue_types[number] == 2 {
            // residue 2 bundles together multiple channels; count the channels
            // of the first submap that references this residue
            let mut ch = 0;
            for mi in &self.ci.maps {
                if ch != 0 {
                    break;
                }
                for j in 0..mi.submaps as usize {
                    if ch != 0 {
                        break;
                    }
                    if mi.residuesubmap[j] as usize == number {
                        for k in 0..self.ci.channels as usize {
                            if mi.chmuxlist[k] as usize == j {
                                ch += 1;
                            }
                        }
                    }
                }
            }
            r.end = ((freq / nyq * blocksize * f64::from(ch)) / f64::from(grouping) + 0.9) as i32
                * grouping;
            if r.end > blocksize_i * ch {
                r.end = blocksize_i * ch / grouping * grouping;
            }
        } else {
            r.end = ((freq / nyq * blocksize) / f64::from(grouping) + 0.9) as i32 * grouping;
            if r.end > blocksize_i {
                r.end = blocksize_i / grouping * grouping;
            }
        }
        if r.end == 0 {
            r.end = grouping;
        }
        self.ci.residues[number] = r;
    }

    /// Port of vorbisenc.c `vorbis_encode_map_n_res_setup`.
    fn map_n_res_setup(&mut self, s: f64, maps: &'static [MappingTemplate]) {
        let is = s as i32 as usize;
        let map = maps[is].map;
        let res = maps[is].res;
        let modes = if self.ci.blocksizes[0] == self.ci.blocksizes[1] {
            1
        } else {
            2
        };
        for i in 0..modes {
            self.ci.modes.push(MODE_TEMPLATE[i]);
            self.ci.maps.push(map[i]);
            for j in 0..map[i].submaps as usize {
                let number = map[i].residuesubmap[j] as usize;
                self.residue_setup(number, i, &res[number]);
            }
        }
    }
}

/// Port of vorbisenc.c `setting_to_approx_bitrate`.
fn setting_to_approx_bitrate(hi: &HighLevel, ch: i32) -> f64 {
    let (is, ds) = split(hi.base_setting);
    let r = hi.setup.rate_mapping;
    (r[is] * (1. - ds) + r[is + 1] * ds) * f64::from(ch)
}

/// Port of `vorbis_encode_setup_vbr` followed by `vorbis_encode_setup_init`,
/// i.e. `vorbis_encode_init_vbr`. Returns `None` where libvorbis returns an
/// error (no template for the rate/channels/quality).
pub(crate) fn encode_init_vbr(channels: i32, rate: i32, quality: f32) -> Option<CodecSetup> {
    if rate <= 0 || !(1..=255).contains(&channels) {
        return None;
    }
    // C: quality+=.0000001; (float += double)
    let mut quality = (f64::from(quality) + 0.0000001) as f32;
    if quality >= 1. {
        quality = 0.9999;
    }
    let (setup, base_setting) = get_setup_template(
        i64::from(channels),
        i64::from(rate),
        f64::from(quality),
        false,
    )?;
    let hi = setup_setting(setup, base_setting);
    Some(setup_init(hi, channels, rate))
}

/// Port of vorbisenc.c `vorbis_encode_setup_init` (non-managed paths).
fn setup_init(mut hi: HighLevel, channels: i32, rate: i32) -> CodecSetup {
    let i0 = if hi.impulse_block_p { 0 } else { 1 };
    hi.ath_floating_db = hi.ath_floating_db.clamp(-200., -80.);
    hi.amplitude_track_db_per_sec = hi.amplitude_track_db_per_sec.clamp(-99999., 0.);
    let setup = hi.setup;

    let ci = CodecSetup {
        channels,
        rate,
        bitrate_upper: 0,
        bitrate_nominal: 0,
        bitrate_lower: 0,
        blocksizes: [0, 0],
        modes: Vec::new(),
        maps: Vec::new(),
        floors: Vec::new(),
        residue_types: Vec::new(),
        residues: Vec::new(),
        books: Vec::new(),
        psy_param: Vec::new(),
        psy_g_param: setup.global_params[0],
    };
    let mut b = Builder {
        ci,
        hi,
        psy_set: [false; 4],
    };
    let bs = b.hi.base_setting;

    b.blocksize_setup(bs, setup.blocksize_short, setup.blocksize_long);
    let singleblock = b.ci.blocksizes[0] == b.ci.blocksizes[1];

    // floor setup
    for i in 0..setup.floor_mappings as usize {
        b.floor_setup(
            bs,
            setup.floor_books,
            setup.floor_params,
            setup.floor_mapping_list[i],
        );
    }

    // setup of [mostly] short block detection and stereo
    b.global_psych_setup(
        b.hi.trigger_setting,
        setup.global_params,
        setup.global_mapping,
    );
    b.global_stereo(setup.stereo_modes);

    // basic psych setup and noise normalization
    let nt = setup.psy_noise_normal_thresh;
    b.psyset_setup(
        bs,
        setup.psy_noise_normal_start[0],
        setup.psy_noise_normal_partition[0],
        nt,
        0,
    );
    b.psyset_setup(
        bs,
        setup.psy_noise_normal_start[0],
        setup.psy_noise_normal_partition[0],
        nt,
        1,
    );
    if !singleblock {
        b.psyset_setup(
            bs,
            setup.psy_noise_normal_start[1],
            setup.psy_noise_normal_partition[1],
            nt,
            2,
        );
        b.psyset_setup(
            bs,
            setup.psy_noise_normal_start[1],
            setup.psy_noise_normal_partition[1],
            nt,
            3,
        );
    }

    // tone masking setup
    let blk = b.hi.block;
    let att = setup.psy_tone_masteratt;
    let db0 = setup.psy_tone_0db;
    b.tonemask_setup(
        blk[i0].tone_mask_setting,
        0,
        att,
        db0,
        setup.psy_tone_adj_impulse,
    );
    b.tonemask_setup(
        blk[1].tone_mask_setting,
        1,
        att,
        db0,
        setup.psy_tone_adj_other,
    );
    if !singleblock {
        b.tonemask_setup(
            blk[2].tone_mask_setting,
            2,
            att,
            db0,
            setup.psy_tone_adj_other,
        );
        let long = setup.psy_tone_adj_long.unwrap_or(setup.psy_tone_adj_other);
        b.tonemask_setup(blk[3].tone_mask_setting, 3, att, db0, long);
    }

    // noise companding setup
    let comp = setup.psy_noise_compand;
    let short_map = setup.psy_noise_compand_short_mapping;
    b.compand_setup(blk[i0].noise_compand_setting, 0, comp, short_map);
    b.compand_setup(blk[1].noise_compand_setting, 1, comp, short_map);
    if !singleblock {
        let long_map = setup.psy_noise_compand_long_mapping.unwrap_or(short_map);
        b.compand_setup(blk[2].noise_compand_setting, 2, comp, long_map);
        b.compand_setup(blk[3].noise_compand_setting, 3, comp, long_map);
    }

    // peak guarding setup
    let sup = setup.psy_tone_dbsuppress;
    b.peak_setup(blk[i0].tone_peaklimit_setting, 0, sup);
    b.peak_setup(blk[1].tone_peaklimit_setting, 1, sup);
    if !singleblock {
        b.peak_setup(blk[2].tone_peaklimit_setting, 2, sup);
        b.peak_setup(blk[3].tone_peaklimit_setting, 3, sup);
    }

    // noise bias setup
    let nsup = setup.psy_noise_dbsuppress;
    let guards = setup.psy_noiseguards;
    let userbias = if i0 == 0 { b.hi.impulse_noisetune } else { 0. };
    b.noisebias_setup(
        blk[i0].noise_bias_setting,
        0,
        nsup,
        setup.psy_noise_bias_impulse,
        guards,
        userbias,
    );
    b.noisebias_setup(
        blk[1].noise_bias_setting,
        1,
        nsup,
        setup.psy_noise_bias_padding,
        guards,
        0.,
    );
    if !singleblock {
        let trans = setup
            .psy_noise_bias_trans
            .unwrap_or(setup.psy_noise_bias_padding);
        let long = setup
            .psy_noise_bias_long
            .unwrap_or(setup.psy_noise_bias_padding);
        b.noisebias_setup(blk[2].noise_bias_setting, 2, nsup, trans, guards, 0.);
        b.noisebias_setup(blk[3].noise_bias_setting, 3, nsup, long, guards, 0.);
    }

    b.ath_setup(0);
    b.ath_setup(1);
    if !singleblock {
        b.ath_setup(2);
        b.ath_setup(3);
    }

    b.map_n_res_setup(bs, setup.maps);

    // set bitrate readonlies (non-managed: no bitrate_av/min/max)
    b.ci.bitrate_nominal = setting_to_approx_bitrate(&b.hi, channels) as i32;
    b.ci.bitrate_lower = 0;
    b.ci.bitrate_upper = 0;
    debug_assert!(b.psy_set.iter().take(b.ci.psy_param.len()).all(|&x| x));
    b.ci
}

/// Port of the bitrate -> setting mapping that `vorbis_encode_setup_managed`
/// performs through `get_setup_template(..., q_or_bitrate=1, ...)`, converted
/// to the equivalent VBR quality by interpolating the template's
/// `quality_mapping` at the resulting `base_setting`.
pub(crate) fn quality_for_bitrate(rate: i64, channels: i64, nominal: f64) -> Option<f32> {
    let (t, base_setting) = get_setup_template(channels, rate, nominal, true)?;
    let (is, ds) = split(base_setting);
    let q = &t.quality_mapping;
    let quality = if is + 1 < q.len() {
        q[is] * (1. - ds) + q[is + 1] * ds
    } else {
        q[is]
    };
    Some(quality as f32)
}
