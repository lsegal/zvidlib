//! Symbolic constants from the AV1 specification (section 3 and the
//! semantics tables of section 6), named as the specification names them.

#![allow(dead_code)]

pub(crate) const REFS_PER_FRAME: usize = 7;
pub(crate) const TOTAL_REFS_PER_FRAME: usize = 8;
pub(crate) const NUM_REF_FRAMES: usize = 8;
pub(crate) const MAX_SEGMENTS: usize = 8;
pub(crate) const SEG_LVL_ALT_Q: usize = 0;
pub(crate) const SEG_LVL_ALT_LF_Y_V: usize = 1;
pub(crate) const SEG_LVL_REF_FRAME: usize = 5;
pub(crate) const SEG_LVL_SKIP: usize = 6;
pub(crate) const SEG_LVL_GLOBALMV: usize = 7;
pub(crate) const SEG_LVL_MAX: usize = 8;
pub(crate) const MAX_LOOP_FILTER: i32 = 63;
pub(crate) const MI_SIZE: usize = 4;
pub(crate) const MI_SIZE_LOG2: usize = 2;
pub(crate) const MAX_TILE_WIDTH: usize = 4096;
pub(crate) const MAX_TILE_AREA: usize = 4096 * 2304;
pub(crate) const MAX_TILE_ROWS: usize = 64;
pub(crate) const MAX_TILE_COLS: usize = 64;
pub(crate) const PRIMARY_REF_NONE: usize = 7;
pub(crate) const SUPERRES_NUM: usize = 8;
pub(crate) const SUPERRES_DENOM_MIN: usize = 9;
pub(crate) const SUPERRES_DENOM_BITS: usize = 3;
pub(crate) const SUPERRES_FILTER_BITS: i32 = 6;
pub(crate) const SUPERRES_FILTER_TAPS: i32 = 8;
pub(crate) const SUPERRES_FILTER_OFFSET: i32 = 3;
pub(crate) const SUPERRES_SCALE_BITS: i32 = 14;
pub(crate) const SUPERRES_SCALE_MASK: i32 = (1 << 14) - 1;
pub(crate) const SUPERRES_EXTRA_BITS: i32 = 8;
pub(crate) const RESTORATION_TILESIZE_MAX: usize = 256;
pub(crate) const WARPEDMODEL_PREC_BITS: i32 = 16;
pub(crate) const GM_ABS_TRANS_BITS: i32 = 12;
pub(crate) const GM_ABS_TRANS_ONLY_BITS: i32 = 9;
pub(crate) const GM_ABS_ALPHA_BITS: i32 = 12;
pub(crate) const GM_ALPHA_PREC_BITS: i32 = 15;
pub(crate) const GM_TRANS_PREC_BITS: i32 = 6;
pub(crate) const GM_TRANS_ONLY_PREC_BITS: i32 = 3;
pub(crate) const DIV_LUT_PREC_BITS: i32 = 14;
pub(crate) const DIV_LUT_BITS: i32 = 8;
pub(crate) const LEAST_SQUARES_SAMPLES_MAX: usize = 8;
pub(crate) const LS_MV_MAX: i32 = 256;
pub(crate) const WARPEDMODEL_TRANS_CLAMP: i32 = 1 << 23;
pub(crate) const WARPEDMODEL_NONDIAGAFFINE_CLAMP: i32 = 1 << 13;
pub(crate) const WARPEDPIXEL_PREC_SHIFTS: i32 = 1 << 6;
pub(crate) const WARPEDDIFF_PREC_BITS: i32 = 10;
pub(crate) const WARP_PARAM_REDUCE_BITS: i32 = 6;
pub(crate) const MAX_SB_SIZE: usize = 128;
pub(crate) const MASK_MASTER_SIZE: usize = 64;
pub(crate) const MAX_FRAME_DISTANCE: i32 = 31;
pub(crate) const MAX_OFFSET_WIDTH: i32 = 8;
pub(crate) const MAX_OFFSET_HEIGHT: i32 = 0;
pub(crate) const REF_CAT_LEVEL: u32 = 640;
pub(crate) const MAX_REF_MV_STACK_SIZE: usize = 8;
pub(crate) const MFMV_STACK_SIZE: i32 = 3;
pub(crate) const MV_BORDER: i32 = 128;
pub(crate) const REFMVS_LIMIT: i32 = (1 << 12) - 1;
pub(crate) const REF_SCALE_SHIFT: i32 = 14;
pub(crate) const SUBPEL_BITS: i32 = 4;
pub(crate) const SUBPEL_MASK: i32 = 15;
pub(crate) const SCALE_SUBPEL_BITS: i32 = 10;
pub(crate) const FILTER_BITS: i32 = 7;
pub(crate) const NUM_BASE_LEVELS: i32 = 2;
pub(crate) const COEFF_BASE_RANGE: i32 = 12;
pub(crate) const BR_CDF_SIZE: i32 = 4;
pub(crate) const SIG_COEF_CONTEXTS: usize = 42;
pub(crate) const SIG_COEF_CONTEXTS_EOB: usize = 4;
pub(crate) const SIG_COEF_CONTEXTS_2D: usize = 26;
pub(crate) const DELTA_Q_SMALL: u32 = 3;
pub(crate) const DELTA_LF_SMALL: u32 = 3;
pub(crate) const FRAME_LF_COUNT: usize = 4;
pub(crate) const MAX_VARTX_DEPTH: usize = 2;
pub(crate) const MAX_ANGLE_DELTA: i32 = 3;
pub(crate) const ANGLE_STEP: i32 = 3;
pub(crate) const PALETTE_COLORS: usize = 8;
pub(crate) const PALETTE_NUM_NEIGHBORS: usize = 3;
pub(crate) const INTRA_FILTER_SCALE_BITS: i32 = 4;
pub(crate) const SGRPROJ_PARAMS_BITS: u32 = 4;
pub(crate) const SGRPROJ_PRJ_SUBEXP_K: u32 = 4;
pub(crate) const SGRPROJ_PRJ_BITS: i32 = 7;
pub(crate) const SGRPROJ_RST_BITS: i32 = 4;
pub(crate) const SGRPROJ_MTABLE_BITS: i32 = 20;
pub(crate) const SGRPROJ_RECIP_BITS: i32 = 12;
pub(crate) const SGRPROJ_SGR_BITS: i32 = 8;
pub(crate) const INTRABC_DELAY_PIXELS: i32 = 256;
pub(crate) const INTRABC_DELAY_SB64: i32 = 4;
pub(crate) const SELECT_SCREEN_CONTENT_TOOLS: u32 = 2;
pub(crate) const SELECT_INTEGER_MV: u32 = 2;

// Frame types.
pub(crate) const KEY_FRAME: u8 = 0;
pub(crate) const INTER_FRAME: u8 = 1;
pub(crate) const INTRA_ONLY_FRAME: u8 = 2;
pub(crate) const SWITCH_FRAME: u8 = 3;

// Reference frames.
pub(crate) const NONE: i8 = -1;
pub(crate) const INTRA_FRAME: i8 = 0;
pub(crate) const LAST_FRAME: i8 = 1;
pub(crate) const LAST2_FRAME: i8 = 2;
pub(crate) const LAST3_FRAME: i8 = 3;
pub(crate) const GOLDEN_FRAME: i8 = 4;
pub(crate) const BWDREF_FRAME: i8 = 5;
pub(crate) const ALTREF2_FRAME: i8 = 6;
pub(crate) const ALTREF_FRAME: i8 = 7;

// Block sizes.
pub(crate) const BLOCK_4X4: usize = 0;
pub(crate) const BLOCK_4X8: usize = 1;
pub(crate) const BLOCK_8X4: usize = 2;
pub(crate) const BLOCK_8X8: usize = 3;
pub(crate) const BLOCK_8X16: usize = 4;
pub(crate) const BLOCK_16X8: usize = 5;
pub(crate) const BLOCK_16X16: usize = 6;
pub(crate) const BLOCK_32X32: usize = 9;
pub(crate) const BLOCK_64X64: usize = 12;
pub(crate) const BLOCK_128X128: usize = 15;
pub(crate) const BLOCK_SIZES: usize = 22;
pub(crate) const BLOCK_INVALID: usize = 22;

// Partition types.
pub(crate) const PARTITION_NONE: usize = 0;
pub(crate) const PARTITION_HORZ: usize = 1;
pub(crate) const PARTITION_VERT: usize = 2;
pub(crate) const PARTITION_SPLIT: usize = 3;
pub(crate) const PARTITION_HORZ_A: usize = 4;
pub(crate) const PARTITION_HORZ_B: usize = 5;
pub(crate) const PARTITION_VERT_A: usize = 6;
pub(crate) const PARTITION_VERT_B: usize = 7;
pub(crate) const PARTITION_HORZ_4: usize = 8;
pub(crate) const PARTITION_VERT_4: usize = 9;

// Transform sizes.
pub(crate) const TX_4X4: usize = 0;
pub(crate) const TX_8X8: usize = 1;
pub(crate) const TX_16X16: usize = 2;
pub(crate) const TX_32X32: usize = 3;
pub(crate) const TX_64X64: usize = 4;
pub(crate) const TX_16X32: usize = 9;
pub(crate) const TX_32X16: usize = 10;
pub(crate) const TX_16X64: usize = 17;
pub(crate) const TX_64X16: usize = 18;
pub(crate) const TX_SIZES: usize = 5;
pub(crate) const TX_SIZES_ALL: usize = 19;

// Transform modes.
pub(crate) const ONLY_4X4: u8 = 0;
pub(crate) const TX_MODE_LARGEST: u8 = 1;
pub(crate) const TX_MODE_SELECT: u8 = 2;

// Transform types.
pub(crate) const DCT_DCT: u8 = 0;
pub(crate) const ADST_DCT: u8 = 1;
pub(crate) const DCT_ADST: u8 = 2;
pub(crate) const ADST_ADST: u8 = 3;
pub(crate) const FLIPADST_DCT: u8 = 4;
pub(crate) const DCT_FLIPADST: u8 = 5;
pub(crate) const FLIPADST_FLIPADST: u8 = 6;
pub(crate) const ADST_FLIPADST: u8 = 7;
pub(crate) const FLIPADST_ADST: u8 = 8;
pub(crate) const IDTX: u8 = 9;
pub(crate) const V_DCT: u8 = 10;
pub(crate) const H_DCT: u8 = 11;
pub(crate) const V_ADST: u8 = 12;
pub(crate) const H_ADST: u8 = 13;
pub(crate) const V_FLIPADST: u8 = 14;
pub(crate) const H_FLIPADST: u8 = 15;

// Transform sets.
pub(crate) const TX_SET_DCTONLY: usize = 0;
pub(crate) const TX_SET_INTRA_1: usize = 1;
pub(crate) const TX_SET_INTRA_2: usize = 2;
pub(crate) const TX_SET_INTER_1: usize = 1;
pub(crate) const TX_SET_INTER_2: usize = 2;
pub(crate) const TX_SET_INTER_3: usize = 3;

// Transform classes.
pub(crate) const TX_CLASS_2D: usize = 0;
pub(crate) const TX_CLASS_HORIZ: usize = 1;
pub(crate) const TX_CLASS_VERT: usize = 2;

// Prediction modes (YMode values).
pub(crate) const DC_PRED: u8 = 0;
pub(crate) const V_PRED: u8 = 1;
pub(crate) const H_PRED: u8 = 2;
pub(crate) const D45_PRED: u8 = 3;
pub(crate) const D135_PRED: u8 = 4;
pub(crate) const D113_PRED: u8 = 5;
pub(crate) const D157_PRED: u8 = 6;
pub(crate) const D203_PRED: u8 = 7;
pub(crate) const D67_PRED: u8 = 8;
pub(crate) const SMOOTH_PRED: u8 = 9;
pub(crate) const SMOOTH_V_PRED: u8 = 10;
pub(crate) const SMOOTH_H_PRED: u8 = 11;
pub(crate) const PAETH_PRED: u8 = 12;
pub(crate) const UV_CFL_PRED: u8 = 13;
pub(crate) const NEARESTMV: u8 = 14;
pub(crate) const NEARMV: u8 = 15;
pub(crate) const GLOBALMV: u8 = 16;
pub(crate) const NEWMV: u8 = 17;
pub(crate) const NEAREST_NEARESTMV: u8 = 18;
pub(crate) const NEAR_NEARMV: u8 = 19;
pub(crate) const NEAREST_NEWMV: u8 = 20;
pub(crate) const NEW_NEARESTMV: u8 = 21;
pub(crate) const NEAR_NEWMV: u8 = 22;
pub(crate) const NEW_NEARMV: u8 = 23;
pub(crate) const GLOBAL_GLOBALMV: u8 = 24;
pub(crate) const NEW_NEWMV: u8 = 25;

// Interintra modes.
pub(crate) const II_DC_PRED: u8 = 0;
pub(crate) const II_V_PRED: u8 = 1;
pub(crate) const II_H_PRED: u8 = 2;
pub(crate) const II_SMOOTH_PRED: u8 = 3;

// Compound types.
pub(crate) const COMPOUND_WEDGE: u8 = 0;
pub(crate) const COMPOUND_DIFFWTD: u8 = 1;
pub(crate) const COMPOUND_AVERAGE: u8 = 2;
pub(crate) const COMPOUND_INTRA: u8 = 3;
pub(crate) const COMPOUND_DISTANCE: u8 = 4;

// Motion modes.
pub(crate) const SIMPLE: u8 = 0;
pub(crate) const OBMC: u8 = 1;
pub(crate) const LOCALWARP: u8 = 2;

// Global motion types.
pub(crate) const IDENTITY: u8 = 0;
pub(crate) const TRANSLATION: u8 = 1;
pub(crate) const ROTZOOM: u8 = 2;
pub(crate) const AFFINE: u8 = 3;

// Interpolation filters.
pub(crate) const EIGHTTAP: u8 = 0;
pub(crate) const EIGHTTAP_SMOOTH: u8 = 1;
pub(crate) const EIGHTTAP_SHARP: u8 = 2;
pub(crate) const BILINEAR: u8 = 3;
pub(crate) const SWITCHABLE: u8 = 4;

// MV joints.
pub(crate) const MV_JOINT_ZERO: usize = 0;
pub(crate) const MV_JOINT_HNZVZ: usize = 1;
pub(crate) const MV_JOINT_HZVNZ: usize = 2;
pub(crate) const MV_JOINT_HNZVNZ: usize = 3;
pub(crate) const MV_INTRABC_CONTEXT: usize = 1;
pub(crate) const CLASS0_SIZE: i32 = 2;

// Loop restoration types.
pub(crate) const RESTORE_NONE: u8 = 0;
pub(crate) const RESTORE_WIENER: u8 = 1;
pub(crate) const RESTORE_SGRPROJ: u8 = 2;
pub(crate) const RESTORE_SWITCHABLE: u8 = 3;

// Compound reference types.
pub(crate) const SINGLE_REFERENCE: usize = 0;
pub(crate) const COMPOUND_REFERENCE: usize = 1;
pub(crate) const UNIDIR_COMP_REFERENCE: usize = 0;
pub(crate) const BIDIR_COMP_REFERENCE: usize = 1;

// Chroma-from-luma sign values.
pub(crate) const CFL_SIGN_ZERO: usize = 0;
pub(crate) const CFL_SIGN_NEG: usize = 1;
pub(crate) const CFL_SIGN_POS: usize = 2;

/// `Split_Tx_Size` (spec section 10); the specification's own listing of
/// this table is missing a separator, so it is carried by hand.
pub(crate) static SPLIT_TX_SIZE: [u8; TX_SIZES_ALL] =
    [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10];

pub(crate) const fn block_width(size: usize) -> usize {
    4 * super::tables::NUM_4X4_BLOCKS_WIDE[size] as usize
}

pub(crate) const fn block_height(size: usize) -> usize {
    4 * super::tables::NUM_4X4_BLOCKS_HIGH[size] as usize
}
