//! Default CDF tables (AV1 §9.4), the 4×4 scan (§9.2), and the constant context-offset tables
//! (§8.3.2) needed by the M0 lossless encoder.
//!
//! Only the tables with no decoder-side counterpart are defined here. Every coefficient CDF —
//! `txb_skip`, `eob_pt`, `eob_extra`, `coeff_base`, `coeff_base_eob`, `coeff_br` and `dc_sign` —
//! along with `tx_depth`, the scan orders, the context offsets and `ext_tx`, is reached through
//! the accessors re-exported from [`crate::av1_cdf`] at the bottom of this module, so the
//! encoder and [`crate::av1_intra_decoder`] necessarily agree on every table they share and on
//! the quantizer and transform-size context each symbol is selected by. CDFs are stored in the spec's cumulative form (rising to 32768)
//! with the trailing adaptation-count element dropped, so each row is ready to pass straight to
//! [`gamut_bitstream::SymbolEncoder::encode_symbol`] (the M0 frame sets `disable_cdf_update = 1`,
//! so the tables are never adapted). These values are extracted verbatim from the specification.

// --- Partition CDFs (Default_Partition_W*_Cdf), indexed [ctx]. ---
/// `Default_Partition_W8_Cdf` (4 partitions: NONE/HORZ/VERT/SPLIT).
// Adapted and modified from gamut, Copyright (c) 2026 Justin Chung, MIT licensed.
pub static PARTITION_W8: [[u16; 4]; 4] = [
    [19132, 25510, 30392, 32768],
    [13928, 19855, 28540, 32768],
    [12522, 23679, 28629, 32768],
    [9896, 18783, 25853, 32768],
];
/// `Default_Partition_W16_Cdf` (10 partition types).
pub static PARTITION_W16: [[u16; 10]; 4] = [
    [
        15597, 20929, 24571, 26706, 27664, 28821, 29601, 30571, 31902, 32768,
    ],
    [
        7925, 11043, 16785, 22470, 23971, 25043, 26651, 28701, 29834, 32768,
    ],
    [
        5414, 13269, 15111, 20488, 22360, 24500, 25537, 26336, 32117, 32768,
    ],
    [
        2662, 6362, 8614, 20860, 23053, 24778, 26436, 27829, 31171, 32768,
    ],
];
/// `Default_Partition_W32_Cdf`.
pub static PARTITION_W32: [[u16; 10]; 4] = [
    [
        18462, 20920, 23124, 27647, 28227, 29049, 29519, 30178, 31544, 32768,
    ],
    [
        7689, 9060, 12056, 24992, 25660, 26182, 26951, 28041, 29052, 32768,
    ],
    [
        6015, 9009, 10062, 24544, 25409, 26545, 27071, 27526, 32047, 32768,
    ],
    [
        1394, 2208, 2796, 28614, 29061, 29466, 29840, 30185, 31899, 32768,
    ],
];
/// `Default_Partition_W64_Cdf`.
pub static PARTITION_W64: [[u16; 10]; 4] = [
    [
        20137, 21547, 23078, 29566, 29837, 30261, 30524, 30892, 31724, 32768,
    ],
    [
        6732, 7490, 9497, 27944, 28250, 28515, 28969, 29630, 30104, 32768,
    ],
    [
        5945, 7663, 8348, 28683, 29117, 29749, 30064, 30298, 32238, 32768,
    ],
    [
        870, 1212, 1487, 31198, 31394, 31574, 31743, 31881, 32332, 32768,
    ],
];

/// `Default_Skip_Cdf`, indexed [ctx].
pub static SKIP: [[u16; 2]; 3] = [[31671, 32768], [16515, 32768], [4576, 32768]];

/// `Default_Intra_Frame_Y_Mode_Cdf[0][0]` (above/left both `DC_PRED`).
pub static INTRA_FRAME_Y_MODE_DC_DC: [u16; 13] = [
    15588, 17027, 19338, 20218, 20682, 21110, 21825, 23244, 24189, 28165, 29093, 30466, 32768,
];

/// `Default_Uv_Mode_Cfl_Not_Allowed_Cdf[DC_PRED]` (blocks larger than 4×4 in 4:4:4 lossless).
pub static UV_MODE_CFL_NOT_ALLOWED_DC: [u16; 13] = [
    22631, 24152, 25378, 25661, 25986, 26520, 27055, 27923, 28244, 30059, 30941, 31961, 32768,
];
/// `Default_Uv_Mode_Cfl_Allowed_Cdf[DC_PRED]` (4×4 blocks in lossless).
pub static UV_MODE_CFL_ALLOWED_DC: [u16; 14] = [
    10407, 11208, 12900, 13181, 13823, 14175, 14899, 15656, 15986, 20086, 20995, 22455, 24212,
    32768,
];

// --- Coefficient CDFs (quant context 0, TX_4X4). ---

/// `Default_Scan_4x4` (§9.2): up-right diagonal scan order over the 16 positions.
pub static DEFAULT_SCAN_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// `Coeff_Base_Ctx_Offset[TX_4X4]` (§8.3.2), indexed `[min(row,4)][min(col,4)]`.
pub static COEFF_BASE_CTX_OFFSET_4X4: [[u8; 5]; 5] = [
    [0, 1, 6, 6, 0],
    [1, 6, 6, 21, 0],
    [6, 6, 21, 21, 0],
    [6, 21, 21, 21, 0],
    [0, 0, 0, 0, 0],
];

/// `Sig_Ref_Diff_Offset[TX_CLASS_2D]` (§8.3.2): `(row, col)` neighbour offsets for `coeff_base`.
pub static SIG_REF_DIFF_OFFSET_2D: [(usize, usize); 5] = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)];

/// `Mag_Ref_Offset_With_Tx_Class[TX_CLASS_2D]` (§8.3.2): neighbour offsets for `coeff_br`.
pub static MAG_REF_OFFSET_2D: [(usize, usize); 3] = [(0, 1), (1, 0), (1, 1)];

// --- Symbols shared verbatim with the decoder. ---
//
// The coefficient CDFs are reached through these accessors rather than through the
// `qctx = 0`, `TX_4X4` copies above, so the encoder selects the specification's real quantizer
// and transform-size contexts exactly as the decoders do.
pub use crate::av1_cdf::{
    coeff_base_cdf, coeff_base_ctx_offset, coeff_base_eob_cdf, coeff_br_cdf, coeff_qctx,
    coeff_tx_size_ctx, dc_sign_cdf, eob_extra_cdf, eob_pt_cdf, get_tx_set, tx_depth_cdf,
    tx_type_cdf, tx_type_inverse_set, txb_skip_cdf, up_right_diagonal_scan,
};
