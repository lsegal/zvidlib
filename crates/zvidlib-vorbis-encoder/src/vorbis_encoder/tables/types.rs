//! Rust mirrors of the libvorbis structs that hold the encoder's static
//! configuration data. Field names follow the C members (lower-cased where the
//! C name has capitals, e.g. `max_curve_dB` -> `max_curve_db`).
//!
//! C `long` members are `i32` here: every value stored in them fits, and the
//! Windows LLP64 build of libvorbis uses 32-bit `long` as well.

/// Port of `static_codebook` (codebook.h).
#[derive(Debug)]
pub struct StaticCodebook {
    pub dim: i32,
    pub entries: i32,
    /// Codeword lengths in bits; 0 marks an unused entry.
    pub lengthlist: &'static [u8],
    pub maptype: i32,
    /// Packed 32-bit VQ float (see `_float32_unpack`).
    pub q_min: i32,
    pub q_delta: i32,
    pub q_quant: i32,
    pub q_sequencep: i32,
    pub quantlist: Option<&'static [i32]>,
}

/// Port of `vorbis_info_floor1` (backends.h).
#[derive(Debug, Clone, Copy)]
pub struct InfoFloor1 {
    pub partitions: i32,
    pub partitionclass: [i32; 31],
    pub class_dim: [i32; 16],
    pub class_subs: [i32; 16],
    pub class_book: [i32; 16],
    pub class_subbook: [[i32; 8]; 16],
    pub mult: i32,
    pub postlist: [i32; 65],
    pub maxover: f32,
    pub maxunder: f32,
    pub maxerr: f32,
    pub twofitweight: f32,
    pub twofitatten: f32,
    pub n: i32,
}

/// Port of `vorbis_info_residue0` (backends.h).
#[derive(Debug, Clone, Copy)]
pub struct InfoResidue0 {
    pub begin: i32,
    pub end: i32,
    pub grouping: i32,
    pub partitions: i32,
    /// decode-side only (set by `res0_unpack`)
    #[allow(dead_code)]
    pub partvals: i32,
    pub groupbook: i32,
    pub secondstages: [i32; 64],
    pub booklist: [i32; 512],
    pub classmetric1: [i32; 64],
    pub classmetric2: [i32; 64],
}

/// Port of `vorbis_info_mapping0` (backends.h).
#[derive(Debug, Clone, Copy)]
pub struct InfoMapping0 {
    pub submaps: i32,
    pub chmuxlist: [i32; 256],
    pub floorsubmap: [i32; 16],
    pub residuesubmap: [i32; 16],
    pub coupling_steps: i32,
    pub coupling_mag: [i32; 256],
    pub coupling_ang: [i32; 256],
}

/// Port of `vorbis_info_mode` (codec_internal.h).
#[derive(Debug, Clone, Copy)]
pub struct InfoMode {
    pub blockflag: i32,
    pub windowtype: i32,
    pub transformtype: i32,
    pub mapping: i32,
}

/// Port of `vorbis_info_psy_global` (psy.h).
#[derive(Debug, Clone, Copy)]
pub struct InfoPsyGlobal {
    pub eighth_octave_lines: i32,
    pub preecho_thresh: [f32; 7],
    pub postecho_thresh: [f32; 7],
    pub stretch_penalty: f32,
    pub preecho_minenergy: f32,
    pub ampmax_att_per_sec: f32,
    pub coupling_pkhz: [i32; 15],
    pub coupling_pointlimit: [[i32; 15]; 2],
    pub coupling_prepointamp: [i32; 15],
    pub coupling_postpointamp: [i32; 15],
    pub sliding_lowpass: [[i32; 15]; 2],
}

/// Port of `vorbis_info_psy` (psy.h).
#[derive(Debug, Clone, Copy)]
pub struct InfoPsy {
    pub blockflag: i32,
    pub ath_adjatt: f32,
    pub ath_maxatt: f32,
    pub tone_masteratt: [f32; 3],
    pub tone_centerboost: f32,
    pub tone_decay: f32,
    pub tone_abs_limit: f32,
    pub toneatt: [f32; 17],
    /// unused by libvorbis 1.3.7
    #[allow(dead_code)]
    pub noisemaskp: i32,
    pub noisemaxsupp: f32,
    pub noisewindowlo: f32,
    pub noisewindowhi: f32,
    pub noisewindowlomin: i32,
    pub noisewindowhimin: i32,
    pub noisewindowfixed: i32,
    pub noiseoff: [[f32; 17]; 3],
    pub noisecompand: [f32; 40],
    pub max_curve_db: f32,
    pub normal_p: i32,
    pub normal_start: i32,
    pub normal_partition: i32,
    pub normal_thresh: f64,
}

/// Port of `att3` (vorbisenc.c).
#[derive(Debug)]
pub struct Att3 {
    pub att: [i32; 3],
    pub boost: f32,
    pub decay: f32,
}

/// Port of `adj_stereo` (vorbisenc.c).
#[derive(Debug)]
pub struct AdjStereo {
    pub pre: [i32; 15],
    pub post: [i32; 15],
    pub khz: [f32; 15],
    pub lowpass_khz: [f32; 15],
}

/// Port of `noiseguard` (vorbisenc.c).
#[derive(Debug)]
pub struct NoiseGuard {
    pub lo: i32,
    pub hi: i32,
    pub fixed: i32,
}

/// Port of `noise3` (vorbisenc.c).
#[derive(Debug)]
pub struct Noise3 {
    pub data: [[i32; 17]; 3],
}

/// Port of `vp_adjblock` (vorbisenc.c).
#[derive(Debug)]
pub struct VpAdjBlock {
    pub block: [i32; 17],
}

/// Port of `compandblock` (vorbisenc.c).
#[derive(Debug)]
pub struct CompandBlock {
    pub data: [i32; 40],
}

/// Port of `static_bookblock` (vorbisenc.c).
#[derive(Debug)]
pub struct StaticBookBlock {
    pub books: [[Option<&'static StaticCodebook>; 4]; 12],
}

/// Port of `vorbis_residue_template` (vorbisenc.c), without the
/// `book_aux_managed`/`books_base_managed` members that only bitrate-managed
/// mode uses.
#[derive(Debug)]
pub struct ResidueTemplate {
    pub res_type: i32,
    pub limit_type: i32,
    pub grouping: i32,
    pub res: &'static InfoResidue0,
    pub book_aux: &'static StaticCodebook,
    pub books_base: &'static StaticBookBlock,
}

/// Port of `vorbis_mapping_template` (vorbisenc.c).
#[derive(Debug)]
pub struct MappingTemplate {
    pub map: &'static [InfoMapping0],
    pub res: &'static [ResidueTemplate],
}

/// Port of `ve_setup_data_template` (vorbisenc.c). `Option` marks the
/// pointers some templates leave `NULL`.
#[derive(Debug)]
pub struct SetupDataTemplate {
    pub mappings: i32,
    pub rate_mapping: &'static [f64],
    pub quality_mapping: &'static [f64],
    pub coupling_restriction: i32,
    pub samplerate_min_restriction: i32,
    pub samplerate_max_restriction: i32,
    pub blocksize_short: &'static [i32],
    pub blocksize_long: &'static [i32],
    pub psy_tone_masteratt: &'static [Att3],
    pub psy_tone_0db: &'static [i32],
    pub psy_tone_dbsuppress: &'static [i32],
    pub psy_tone_adj_impulse: &'static [VpAdjBlock],
    pub psy_tone_adj_long: Option<&'static [VpAdjBlock]>,
    pub psy_tone_adj_other: &'static [VpAdjBlock],
    pub psy_noiseguards: &'static [NoiseGuard],
    pub psy_noise_bias_impulse: &'static [Noise3],
    pub psy_noise_bias_padding: &'static [Noise3],
    pub psy_noise_bias_trans: Option<&'static [Noise3]>,
    pub psy_noise_bias_long: Option<&'static [Noise3]>,
    pub psy_noise_dbsuppress: &'static [i32],
    pub psy_noise_compand: &'static [CompandBlock],
    pub psy_noise_compand_short_mapping: &'static [f64],
    pub psy_noise_compand_long_mapping: Option<&'static [f64]>,
    pub psy_noise_normal_start: [&'static [i32]; 2],
    pub psy_noise_normal_partition: [&'static [i32]; 2],
    pub psy_noise_normal_thresh: &'static [f64],
    pub psy_ath_float: &'static [i32],
    pub psy_ath_abs: &'static [i32],
    pub psy_lowpass: &'static [f64],
    pub global_params: &'static [InfoPsyGlobal],
    pub global_mapping: &'static [f64],
    pub stereo_modes: Option<&'static [AdjStereo]>,
    pub floor_books: &'static [&'static [Option<&'static StaticCodebook>]],
    pub floor_params: &'static [InfoFloor1],
    pub floor_mappings: i32,
    pub floor_mapping_list: &'static [&'static [i32]],
    pub maps: &'static [MappingTemplate],
}
