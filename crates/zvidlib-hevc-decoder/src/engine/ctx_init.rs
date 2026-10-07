//! §9.3.2.2 context-variable initialization — the `initValue` tables
//! (Tables 9-5 .. 9-42) and the per-slice context-array setup.
//!
//! This module is a clean-room transcription from the ITU-T Rec. H.265
//! | ISO/IEC 23008-2 specification text (`docs/video/h265/`):
//!
//! * The 8-bit `initValue` tables for every context-coded syntax
//!   element of clauses 7.3.8.1 through 7.3.8.12 (except
//!   `end_of_slice_segment_flag`, `end_of_subset_one_bit` and
//!   `pcm_flag`, which use the §9.3.2.2 NOTE 2 non-adapting
//!   `ctxTable == 0` state — [`ContextModel::terminate_state`]).
//!   Each constant is laid out flat over the table's printed ctxIdx
//!   axis, exactly as in the spec.
//! * The Table 9-4 association of ctxIdx ranges to the three
//!   initialization types: most tables devote a contiguous span of
//!   `N` ctxIdx slots to each `initType` column, but several are
//!   irregular (`part_mode`, `cbf_cb`/`cbf_cr`, the
//!   `abs_mvd_greater0_flag`/`abs_mvd_greater1_flag` interleave,
//!   `transform_skip_flag`, the `explicit_rdpcm_*` pairs and the
//!   `sig_coeff_flag` transform-skip tail) — see the per-bank
//!   helpers.
//! * [`SliceContexts`] — the whole per-slice context array, one bank
//!   per syntax element (or per Table 9-4 shared-variable group),
//!   initialised for a given `initType` and `SliceQpY` through the
//!   §9.3.2.2 process ([`ContextModel::init`], equations 9-4..9-6).
//!   [`SliceContexts::for_slice`] additionally applies the
//!   equation 9-7 `initType` derivation from `slice_type` and
//!   `cabac_init_flag` ([`crate::hevc::engine::cabac::init_type`]).
//!
//! # Shared context variables
//!
//! Table 9-4 assigns the *same* `(ctxTable, ctxIdx)` slots to a few
//! element groups, which therefore share one adapting context
//! variable per `initType`:
//!
//! * `sao_merge_left_flag` / `sao_merge_up_flag` (Table 9-5),
//! * `sao_type_idx_luma` / `sao_type_idx_chroma` (Table 9-6),
//! * `ref_idx_l0` / `ref_idx_l1` (Table 9-18),
//! * `mvp_l0_flag` / `mvp_l1_flag` (Table 9-19),
//! * `cbf_cb` / `cbf_cr` (Table 9-22),
//! * `copy_above_palette_indices_flag` /
//!   `copy_above_indices_for_final_run_flag` (Table 9-41),
//! * the chroma planes of `transform_skip_flag[ ][ ][ 1 ]` /
//!   `[ 2 ]` (Table 9-25), `explicit_rdpcm_flag[ ][ ][ 1 ]` / `[ 2 ]`
//!   (Table 9-32) and `explicit_rdpcm_dir_flag[ ][ ][ 1 ]` / `[ 2 ]`
//!   (Table 9-33).
//!
//! # Inter-only elements in I slices
//!
//! Table 9-4 leaves the `initType == 0` column empty for the elements
//! that can only occur in P / B slices (`cu_skip_flag`,
//! `pred_mode_flag`, `rqt_root_cbf`, `merge_flag`, `merge_idx`,
//! `inter_pred_idc`, `ref_idx_l0`/`l1`, `mvp_l0_flag`/`l1`,
//! `abs_mvd_greater0_flag`/`greater1_flag`, `explicit_rdpcm_flag` and
//! `explicit_rdpcm_dir_flag`) — no initialization is specified
//! because a conforming I slice never decodes them. [`SliceContexts`]
//! still materialises those banks so the array shape is
//! `initType`-independent; at `initType == 0` they are populated with
//! the §9.3.2.2 NOTE 2 non-adapting state
//! ([`ContextModel::terminate_state`], `pStateIdx = 63`,
//! `valMps = 0`), which no table-derived context can take
//! (equations 9-4..9-6 bound `pStateIdx` to 0..=62), making an
//! accidental read recognisable. The same placeholder fills the
//! `part_mode` slots 1..=3 at `initType == 0` (Table 9-4 gives
//! `part_mode` only ctxIdx 0 there: the I-slice §9.3.4.2 binarization
//! uses a single context-coded bin).

use crate::hevc::engine::cabac::{ContextModel, init_type};
use crate::hevc::engine::residual::ResidualContexts;
use crate::hevc::engine::slice::SliceType;

// ---------------------------------------------------------------------
// initValue tables (Tables 9-5 .. 9-42)
// ---------------------------------------------------------------------

/// Table 9-5 — `initValue` for ctxIdx of `sao_merge_left_flag` and
/// `sao_merge_up_flag` (one shared context per `initType`; ctxIdx
/// 0 / 1 / 2 for `initType` 0 / 1 / 2).
pub const TABLE_9_5_SAO_MERGE_FLAG: [u8; 3] = [153, 153, 153];

/// Table 9-6 — `initValue` for ctxIdx of `sao_type_idx_luma` and
/// `sao_type_idx_chroma` (one shared context per `initType`).
pub const TABLE_9_6_SAO_TYPE_IDX: [u8; 3] = [200, 185, 160];

/// Table 9-7 — `initValue` for ctxIdx of `split_cu_flag` (ctxIdx
/// 0..2 / 3..5 / 6..8 for `initType` 0 / 1 / 2 per Table 9-4).
pub const TABLE_9_7_SPLIT_CU_FLAG: [u8; 9] = [139, 141, 157, 107, 139, 126, 107, 139, 126];

/// Table 9-8 — `initValue` for ctxIdx of `cu_transquant_bypass_flag`.
pub const TABLE_9_8_CU_TRANSQUANT_BYPASS_FLAG: [u8; 3] = [154, 154, 154];

/// Table 9-9 — `initValue` for ctxIdx of `cu_skip_flag` (ctxIdx 0..2 /
/// 3..5 for `initType` 1 / 2; the element does not occur in I slices).
pub const TABLE_9_9_CU_SKIP_FLAG: [u8; 6] = [197, 185, 201, 197, 185, 201];

/// Table 9-10 — `initValue` for ctxIdx of `pred_mode_flag` (ctxIdx
/// 0 / 1 for `initType` 1 / 2; inter-only).
pub const TABLE_9_10_PRED_MODE_FLAG: [u8; 2] = [149, 134];

/// Table 9-11 — `initValue` for ctxIdx of `part_mode`. Irregular
/// Table 9-4 mapping: ctxIdx 0 for `initType` 0 (the I-slice
/// binarization has one context-coded bin), ctxIdx 1..4 / 5..8 for
/// `initType` 1 / 2.
pub const TABLE_9_11_PART_MODE: [u8; 9] = [184, 154, 139, 154, 154, 154, 139, 154, 154];

/// Table 9-12 — `initValue` for ctxIdx of `prev_intra_luma_pred_flag`.
pub const TABLE_9_12_PREV_INTRA_LUMA_PRED_FLAG: [u8; 3] = [184, 154, 183];

/// Table 9-13 — `initValue` for ctxIdx of `intra_chroma_pred_mode`.
pub const TABLE_9_13_INTRA_CHROMA_PRED_MODE: [u8; 3] = [63, 152, 152];

/// Table 9-14 — `initValue` for ctxIdx of `rqt_root_cbf` (ctxIdx 0 / 1
/// for `initType` 1 / 2; inter-only).
pub const TABLE_9_14_RQT_ROOT_CBF: [u8; 2] = [79, 79];

/// Table 9-15 — `initValue` for ctxIdx of `merge_flag` (`initType`
/// 1 / 2; inter-only).
pub const TABLE_9_15_MERGE_FLAG: [u8; 2] = [110, 154];

/// Table 9-16 — `initValue` for ctxIdx of `merge_idx` (`initType`
/// 1 / 2; inter-only).
pub const TABLE_9_16_MERGE_IDX: [u8; 2] = [122, 137];

/// Table 9-17 — `initValue` for ctxIdx of `inter_pred_idc` (ctxIdx
/// 0..4 / 5..9 for `initType` 1 / 2; inter-only).
pub const TABLE_9_17_INTER_PRED_IDC: [u8; 10] = [95, 79, 63, 31, 31, 95, 79, 63, 31, 31];

/// Table 9-18 — `initValue` for ctxIdx of `ref_idx_l0` and
/// `ref_idx_l1` (shared bank; ctxIdx 0..1 / 2..3 for `initType` 1 / 2;
/// inter-only).
pub const TABLE_9_18_REF_IDX: [u8; 4] = [153, 153, 153, 153];

/// Table 9-19 — `initValue` for ctxIdx of `mvp_l0_flag` and
/// `mvp_l1_flag` (shared bank; ctxIdx 0 / 1 for `initType` 1 / 2;
/// inter-only).
pub const TABLE_9_19_MVP_FLAG: [u8; 2] = [168, 168];

/// Table 9-20 — `initValue` for ctxIdx of `split_transform_flag`
/// (ctxIdx 0..2 / 3..5 / 6..8 for `initType` 0 / 1 / 2).
pub const TABLE_9_20_SPLIT_TRANSFORM_FLAG: [u8; 9] = [153, 138, 138, 124, 138, 94, 224, 167, 122];

/// Table 9-21 — `initValue` for ctxIdx of `cbf_luma` (ctxIdx 0..1 /
/// 2..3 / 4..5 for `initType` 0 / 1 / 2).
pub const TABLE_9_21_CBF_LUMA: [u8; 6] = [111, 141, 153, 111, 153, 111];

/// Table 9-22 — `initValue` for ctxIdx of `cbf_cb` and `cbf_cr`
/// (shared bank). Irregular Table 9-4 mapping: ctxIdx 0..3 / 4..7 /
/// 8..11 carry the four trafo-depth contexts for `initType` 0 / 1 / 2
/// and ctxIdx 12 / 13 / 14 carry the fifth (`ctxInc == 4`) context
/// for `initType` 0 / 1 / 2.
#[rustfmt::skip]
pub const TABLE_9_22_CBF_CHROMA: [u8; 15] = [
    94, 138, 182, 154,
    149, 107, 167, 154,
    149, 92, 167, 154,
    154, 154, 154,
];

/// Table 9-23 — `initValue` for ctxIdx of `abs_mvd_greater0_flag` and
/// `abs_mvd_greater1_flag`. Irregular Table 9-4 mapping (inter-only):
/// `abs_mvd_greater0_flag` takes ctxIdx 0 / 2 for `initType` 1 / 2 and
/// `abs_mvd_greater1_flag` takes ctxIdx 1 / 3.
pub const TABLE_9_23_ABS_MVD_GREATER_FLAG: [u8; 4] = [140, 198, 169, 198];

/// Table 9-24 — `initValue` for ctxIdx of `cu_qp_delta_abs` (ctxIdx
/// 0..1 / 2..3 / 4..5 for `initType` 0 / 1 / 2).
pub const TABLE_9_24_CU_QP_DELTA_ABS: [u8; 6] = [154, 154, 154, 154, 154, 154];

/// Table 9-25 — `initValue` for ctxIdx of `transform_skip_flag`.
/// Irregular Table 9-4 mapping: ctxIdx 0 / 1 / 2 carry the luma
/// (`cIdx == 0`) context for `initType` 0 / 1 / 2 and ctxIdx 3 / 4 / 5
/// the context shared by both chroma planes (`cIdx == 1` / `2`).
pub const TABLE_9_25_TRANSFORM_SKIP_FLAG: [u8; 6] = [139, 139, 139, 139, 139, 139];

/// Table 9-26 — `initValue` for ctxIdx of `last_sig_coeff_x_prefix`
/// (ctxIdx 0..17 / 18..35 / 36..53 for `initType` 0 / 1 / 2).
#[rustfmt::skip]
pub const TABLE_9_26_LAST_SIG_COEFF_X_PREFIX: [u8; 54] = [
    110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111,  79, 108, 123,  63,
    125, 110,  94, 110,  95,  79, 125, 111, 110,  78, 110, 111, 111,  95,  94, 108, 123, 108,
    125, 110, 124, 110,  95,  94, 125, 111, 111,  79, 125, 126, 111, 111,  79, 108, 123,  93,
];

/// Table 9-27 — `initValue` for ctxIdx of `last_sig_coeff_y_prefix`
/// (ctxIdx 0..17 / 18..35 / 36..53 for `initType` 0 / 1 / 2). The
/// printed values coincide with Table 9-26 entry-for-entry, but the
/// two elements adapt separate context variables.
#[rustfmt::skip]
pub const TABLE_9_27_LAST_SIG_COEFF_Y_PREFIX: [u8; 54] = [
    110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111,  79, 108, 123,  63,
    125, 110,  94, 110,  95,  79, 125, 111, 110,  78, 110, 111, 111,  95,  94, 108, 123, 108,
    125, 110, 124, 110,  95,  94, 125, 111, 111,  79, 125, 126, 111, 111,  79, 108, 123,  93,
];

/// Table 9-28 — `initValue` for ctxIdx of `coded_sub_block_flag`
/// (ctxIdx 0..3 / 4..7 / 8..11 for `initType` 0 / 1 / 2).
#[rustfmt::skip]
pub const TABLE_9_28_CODED_SUB_BLOCK_FLAG: [u8; 12] = [
    91, 171, 134, 141,
    121, 140, 61, 154,
    121, 140, 61, 154,
];

/// Table 9-29 — `initValue` for ctxIdx of `sig_coeff_flag`. Irregular
/// Table 9-4 mapping: ctxIdx 0..41 / 42..83 / 84..125 carry the 42
/// §9.3.4.2.5 contexts for `initType` 0 / 1 / 2 and ctxIdx 126..127 /
/// 128..129 / 130..131 carry the two equation 9-40 transform-skip
/// contexts (luma, chroma) for `initType` 0 / 1 / 2.
#[rustfmt::skip]
pub const TABLE_9_29_SIG_COEFF_FLAG: [u8; 132] = [
    111, 111, 125, 110, 110,  94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107,
    125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152,
    136, 152, 136, 153, 136, 139, 111, 136, 139, 111, 155, 154, 139, 153, 139, 123,
    123,  63, 153, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 166,
    183, 140, 136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121, 167, 151, 183,
    140, 151, 183, 140, 170, 154, 139, 153, 139, 123, 123,  63, 124, 166, 183, 140,
    136, 153, 154, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 170,
    153, 138, 138, 122, 121, 122, 121, 167, 151, 183, 140, 151, 183, 140, 141, 111,
    140, 140, 140, 140,
];

/// Table 9-30 — `initValue` for ctxIdx of
/// `coeff_abs_level_greater1_flag` (ctxIdx 0..23 / 24..47 / 48..71 for
/// `initType` 0 / 1 / 2).
#[rustfmt::skip]
pub const TABLE_9_30_COEFF_ABS_LEVEL_GREATER1_FLAG: [u8; 72] = [
    140,  92, 137, 138, 140, 152, 138, 139, 153,  74, 149,  92, 139, 107, 122, 152,
    140, 179, 166, 182, 140, 227, 122, 197, 154, 196, 196, 167, 154, 152, 167, 182,
    182, 134, 149, 136, 153, 121, 136, 137, 169, 194, 166, 167, 154, 167, 137, 182,
    154, 196, 167, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 122,
    169, 208, 166, 167, 154, 152, 167, 182,
];

/// Table 9-31 — `initValue` for ctxIdx of
/// `coeff_abs_level_greater2_flag` (ctxIdx 0..5 / 6..11 / 12..17 for
/// `initType` 0 / 1 / 2).
#[rustfmt::skip]
pub const TABLE_9_31_COEFF_ABS_LEVEL_GREATER2_FLAG: [u8; 18] = [
    138, 153, 136, 167, 152, 152,
    107, 167,  91, 122, 107, 167,
    107, 167,  91, 107, 107, 167,
];

/// Table 9-32 — `initValue` for ctxIdx of `explicit_rdpcm_flag`.
/// Irregular Table 9-4 mapping (inter-only): ctxIdx 0 / 1 carry the
/// luma (`cIdx == 0`) context for `initType` 1 / 2 and ctxIdx 2 / 3
/// the context shared by both chroma planes.
pub const TABLE_9_32_EXPLICIT_RDPCM_FLAG: [u8; 4] = [139, 139, 139, 139];

/// Table 9-33 — `initValue` for ctxIdx of `explicit_rdpcm_dir_flag`
/// (same irregular mapping as Table 9-32; inter-only).
pub const TABLE_9_33_EXPLICIT_RDPCM_DIR_FLAG: [u8; 4] = [139, 139, 139, 139];

/// Table 9-34 — `initValue` for ctxIdx of `cu_chroma_qp_offset_flag`.
pub const TABLE_9_34_CU_CHROMA_QP_OFFSET_FLAG: [u8; 3] = [154, 154, 154];

/// Table 9-35 — `initValue` for ctxIdx of `cu_chroma_qp_offset_idx`.
pub const TABLE_9_35_CU_CHROMA_QP_OFFSET_IDX: [u8; 3] = [154, 154, 154];

/// Table 9-36 — `initValue` for ctxIdx of `log2_res_scale_abs_plus1`
/// (ctxIdx 0..7 / 8..15 / 16..23 for `initType` 0 / 1 / 2).
pub const TABLE_9_36_LOG2_RES_SCALE_ABS_PLUS1: [u8; 24] = [154; 24];

/// Table 9-37 — `initValue` for ctxIdx of `res_scale_sign_flag`
/// (ctxIdx 0..1 / 2..3 / 4..5 for `initType` 0 / 1 / 2).
pub const TABLE_9_37_RES_SCALE_SIGN_FLAG: [u8; 6] = [154; 6];

/// Table 9-38 — `initValue` for ctxIdx of `palette_mode_flag`.
pub const TABLE_9_38_PALETTE_MODE_FLAG: [u8; 3] = [154, 154, 154];

/// Table 9-39 — `initValue` for ctxIdx of `tu_residual_act_flag`.
pub const TABLE_9_39_TU_RESIDUAL_ACT_FLAG: [u8; 3] = [154, 154, 154];

/// Table 9-40 — `initValue` for ctxIdx of `palette_run_prefix` (ctxIdx
/// 0..7 / 8..15 / 16..23 for `initType` 0 / 1 / 2).
pub const TABLE_9_40_PALETTE_RUN_PREFIX: [u8; 24] = [154; 24];

/// Table 9-41 — `initValue` for ctxIdx of
/// `copy_above_palette_indices_flag` and
/// `copy_above_indices_for_final_run_flag` (shared bank: Table 9-4
/// assigns both elements ctxIdx 0 / 1 / 2 of this table).
pub const TABLE_9_41_PALETTE_COPY_ABOVE_FLAG: [u8; 3] = [154, 154, 154];

/// Table 9-42 — `initValue` for ctxIdx of `palette_transpose_flag`.
pub const TABLE_9_42_PALETTE_TRANSPOSE_FLAG: [u8; 3] = [154, 154, 154];

// ---------------------------------------------------------------------
// Table 9-4 ctxIdx-span selection helpers
// ---------------------------------------------------------------------

/// Select the `N` per-`initType` `initValue` entries from a table
/// whose Table 9-4 mapping is the regular contiguous layout (ctxIdx
/// `0..N` / `N..2N` / `2N..3N` for `initType` 0 / 1 / 2).
///
/// # Panics
///
/// Panics when `init_type > 2` or `table.len() != 3 * N`.
#[must_use]
pub fn uniform_init_values<const N: usize>(table: &[u8], init_type: u8) -> [u8; N] {
    assert!(init_type <= 2, "initType must be 0, 1 or 2 (equation 9-7)");
    assert_eq!(table.len(), 3 * N, "table is not a 3-initType layout");
    let base = N * init_type as usize;
    core::array::from_fn(|i| table[base + i])
}

/// Select the `N` per-`initType` `initValue` entries from a table for
/// an inter-only element (Table 9-4 leaves `initType == 0` empty;
/// ctxIdx `0..N` / `N..2N` map to `initType` 1 / 2). Returns `None`
/// for `init_type == 0` — the element cannot occur in an I slice.
///
/// # Panics
///
/// Panics when `init_type > 2` or `table.len() != 2 * N`.
#[must_use]
pub fn inter_init_values<const N: usize>(table: &[u8], init_type: u8) -> Option<[u8; N]> {
    assert!(init_type <= 2, "initType must be 0, 1 or 2 (equation 9-7)");
    assert_eq!(table.len(), 2 * N, "table is not a 2-initType layout");
    if init_type == 0 {
        return None;
    }
    let base = N * (init_type as usize - 1);
    Some(core::array::from_fn(|i| table[base + i]))
}

/// The per-`initType` `sig_coeff_flag` `initValue` bank (44 entries):
/// the 42 §9.3.4.2.5 contexts from the Table 9-29 ctxIdx span
/// `42 * initType ..`, followed by the two equation 9-40
/// transform-skip contexts (luma then chroma) from ctxIdx
/// `126 + 2 * initType ..`. Matches the
/// [`crate::hevc::engine::residual::ResidualContexts`] bank layout (slots 42 / 43
/// are the transform-skip contexts).
///
/// # Panics
///
/// Panics when `init_type > 2`.
#[must_use]
pub fn sig_coeff_flag_init_values(init_type: u8) -> [u8; 44] {
    assert!(init_type <= 2, "initType must be 0, 1 or 2 (equation 9-7)");
    let t = init_type as usize;
    core::array::from_fn(|i| {
        if i < 42 {
            TABLE_9_29_SIG_COEFF_FLAG[42 * t + i]
        } else {
            TABLE_9_29_SIG_COEFF_FLAG[126 + 2 * t + (i - 42)]
        }
    })
}

/// Initialize an inter-only bank: table entries through
/// [`ContextModel::init`] for `initType` 1 / 2, the
/// [`ContextModel::terminate_state`] placeholder for `initType` 0
/// (see the module docs).
fn inter_bank<const N: usize>(table: &[u8], init_type: u8, slice_qp_y: i32) -> [ContextModel; N] {
    match inter_init_values::<N>(table, init_type) {
        Some(values) => values.map(|v| ContextModel::init(v, slice_qp_y)),
        None => [ContextModel::terminate_state(); N],
    }
}

/// Initialize a regular three-`initType` bank.
fn uniform_bank<const N: usize>(table: &[u8], init_type: u8, slice_qp_y: i32) -> [ContextModel; N] {
    uniform_init_values::<N>(table, init_type).map(|v| ContextModel::init(v, slice_qp_y))
}

// ---------------------------------------------------------------------
// SliceContexts
// ---------------------------------------------------------------------

/// The complete per-slice CABAC context array of §9.3.2.2 / Table 9-4:
/// one bank per syntax element (or per Table 9-4 shared-variable
/// group; see the module docs), each slot a `(pStateIdx, valMps)`
/// [`ContextModel`]. Bank-internal indices are the *bank-relative*
/// ctxIdx (the §9.3.4.2 `ctxInc` value), not the absolute Table 9-4
/// ctxIdx.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceContexts {
    /// `sao_merge_left_flag` / `sao_merge_up_flag` (Table 9-5, shared).
    pub sao_merge_flag: [ContextModel; 1],
    /// `sao_type_idx_luma` / `sao_type_idx_chroma` (Table 9-6, shared).
    pub sao_type_idx: [ContextModel; 1],
    /// `split_cu_flag` (Table 9-7).
    pub split_cu_flag: [ContextModel; 3],
    /// `cu_transquant_bypass_flag` (Table 9-8).
    pub cu_transquant_bypass_flag: [ContextModel; 1],
    /// `cu_skip_flag` (Table 9-9; inter-only).
    pub cu_skip_flag: [ContextModel; 3],
    /// `palette_mode_flag` (Table 9-38).
    pub palette_mode_flag: [ContextModel; 1],
    /// `pred_mode_flag` (Table 9-10; inter-only).
    pub pred_mode_flag: [ContextModel; 1],
    /// `part_mode` (Table 9-11; slot 0 only at `initType == 0`).
    pub part_mode: [ContextModel; 4],
    /// `prev_intra_luma_pred_flag` (Table 9-12).
    pub prev_intra_luma_pred_flag: [ContextModel; 1],
    /// `intra_chroma_pred_mode` (Table 9-13).
    pub intra_chroma_pred_mode: [ContextModel; 1],
    /// `rqt_root_cbf` (Table 9-14; inter-only).
    pub rqt_root_cbf: [ContextModel; 1],
    /// `merge_flag` (Table 9-15; inter-only).
    pub merge_flag: [ContextModel; 1],
    /// `merge_idx` (Table 9-16; inter-only).
    pub merge_idx: [ContextModel; 1],
    /// `inter_pred_idc` (Table 9-17; inter-only).
    pub inter_pred_idc: [ContextModel; 5],
    /// `ref_idx_l0` / `ref_idx_l1` (Table 9-18, shared; inter-only).
    pub ref_idx: [ContextModel; 2],
    /// `mvp_l0_flag` / `mvp_l1_flag` (Table 9-19, shared; inter-only).
    pub mvp_flag: [ContextModel; 1],
    /// `split_transform_flag` (Table 9-20).
    pub split_transform_flag: [ContextModel; 3],
    /// `cbf_luma` (Table 9-21).
    pub cbf_luma: [ContextModel; 2],
    /// `cbf_cb` / `cbf_cr` (Table 9-22, shared; slot 4 is the
    /// Table 9-4 ctxIdx 12 / 13 / 14 fifth context).
    pub cbf_chroma: [ContextModel; 5],
    /// `abs_mvd_greater0_flag` (Table 9-23 ctxIdx 0 / 2; inter-only).
    pub abs_mvd_greater0_flag: [ContextModel; 1],
    /// `abs_mvd_greater1_flag` (Table 9-23 ctxIdx 1 / 3; inter-only).
    pub abs_mvd_greater1_flag: [ContextModel; 1],
    /// `tu_residual_act_flag` (Table 9-39).
    pub tu_residual_act_flag: [ContextModel; 1],
    /// `log2_res_scale_abs_plus1` (Table 9-36).
    pub log2_res_scale_abs_plus1: [ContextModel; 8],
    /// `res_scale_sign_flag` (Table 9-37).
    pub res_scale_sign_flag: [ContextModel; 2],
    /// `transform_skip_flag` (Table 9-25; slot 0 luma, slot 1 the
    /// shared chroma context).
    pub transform_skip_flag: [ContextModel; 2],
    /// `explicit_rdpcm_flag` (Table 9-32; slot 0 luma, slot 1 the
    /// shared chroma context; inter-only).
    pub explicit_rdpcm_flag: [ContextModel; 2],
    /// `explicit_rdpcm_dir_flag` (Table 9-33; same layout; inter-only).
    pub explicit_rdpcm_dir_flag: [ContextModel; 2],
    /// The §7.3.8.11 `residual_coding( )` banks (Tables 9-26..9-31).
    pub residual: ResidualContexts,
    /// `palette_run_prefix` (Table 9-40).
    pub palette_run_prefix: [ContextModel; 8],
    /// `copy_above_palette_indices_flag` /
    /// `copy_above_indices_for_final_run_flag` (Table 9-41, shared).
    pub palette_copy_above_flag: [ContextModel; 1],
    /// `palette_transpose_flag` (Table 9-42).
    pub palette_transpose_flag: [ContextModel; 1],
    /// `cu_qp_delta_abs` (Table 9-24).
    pub cu_qp_delta_abs: [ContextModel; 2],
    /// `cu_chroma_qp_offset_flag` (Table 9-34).
    pub cu_chroma_qp_offset_flag: [ContextModel; 1],
    /// `cu_chroma_qp_offset_idx` (Table 9-35).
    pub cu_chroma_qp_offset_idx: [ContextModel; 1],
    /// §9.3.2.1 palette predictor variables (`PredictorPaletteSize` /
    /// `PredictorPaletteEntries`) — parse state that initializes,
    /// stores and synchronizes ALONGSIDE the context variables
    /// (§9.3.2.3 / §9.3.2.4 / §9.3.2.5), so it travels inside this
    /// struct. [`Self::init`] leaves it empty; the slice driver
    /// assigns the PPS / SPS initializer-derived predictor at every
    /// §9.3.2.2 re-initialization point.
    pub palette_predictor: crate::hevc::engine::palette::PalettePredictor,
}

impl SliceContexts {
    /// The number of distinct adapting context variables per
    /// `initType` (shared Table 9-4 groups counted once).
    pub const CONTEXT_COUNT: usize = 185;

    /// §9.3.2.2 initialization of the whole context array for one
    /// `initType` (0, 1 or 2) and `SliceQpY` (equation 7-54). Every
    /// table entry goes through [`ContextModel::init`]
    /// (equations 9-4..9-6); inter-only banks at `initType == 0` take
    /// the non-adapting placeholder described in the module docs.
    ///
    /// # Panics
    ///
    /// Panics when `init_type > 2`.
    #[must_use]
    pub fn init(init_type: u8, slice_qp_y: i32) -> Self {
        assert!(init_type <= 2, "initType must be 0, 1 or 2 (equation 9-7)");
        let t = init_type as usize;
        let init = |v: u8| ContextModel::init(v, slice_qp_y);
        // Table 9-11 `part_mode`: ctxIdx 0 at initType 0, 1..4 / 5..8
        // at initType 1 / 2.
        let part_mode = if init_type == 0 {
            let mut bank = [ContextModel::terminate_state(); 4];
            bank[0] = init(TABLE_9_11_PART_MODE[0]);
            bank
        } else {
            core::array::from_fn(|i| init(TABLE_9_11_PART_MODE[1 + 4 * (t - 1) + i]))
        };
        // Table 9-22 `cbf_cb` / `cbf_cr`: ctxIdx 4t..4t+3 plus 12 + t.
        let cbf_chroma = core::array::from_fn(|i| {
            init(if i < 4 {
                TABLE_9_22_CBF_CHROMA[4 * t + i]
            } else {
                TABLE_9_22_CBF_CHROMA[12 + t]
            })
        });
        // Table 9-23 interleave: greater0 at ctxIdx 0 / 2, greater1 at
        // ctxIdx 1 / 3 for initType 1 / 2.
        let (abs_mvd_greater0_flag, abs_mvd_greater1_flag) = if init_type == 0 {
            (
                [ContextModel::terminate_state(); 1],
                [ContextModel::terminate_state(); 1],
            )
        } else {
            (
                [init(TABLE_9_23_ABS_MVD_GREATER_FLAG[2 * (t - 1)])],
                [init(TABLE_9_23_ABS_MVD_GREATER_FLAG[2 * (t - 1) + 1])],
            )
        };
        // Table 9-25 `transform_skip_flag`: luma at ctxIdx t, the
        // shared chroma context at ctxIdx 3 + t.
        let transform_skip_flag = [
            init(TABLE_9_25_TRANSFORM_SKIP_FLAG[t]),
            init(TABLE_9_25_TRANSFORM_SKIP_FLAG[3 + t]),
        ];
        // Tables 9-32 / 9-33 (inter-only): luma at ctxIdx t − 1, the
        // shared chroma context at ctxIdx 2 + (t − 1).
        let rdpcm_bank = |table: &[u8; 4]| {
            if init_type == 0 {
                [ContextModel::terminate_state(); 2]
            } else {
                [init(table[t - 1]), init(table[2 + (t - 1)])]
            }
        };
        Self {
            sao_merge_flag: uniform_bank(&TABLE_9_5_SAO_MERGE_FLAG, init_type, slice_qp_y),
            sao_type_idx: uniform_bank(&TABLE_9_6_SAO_TYPE_IDX, init_type, slice_qp_y),
            split_cu_flag: uniform_bank(&TABLE_9_7_SPLIT_CU_FLAG, init_type, slice_qp_y),
            cu_transquant_bypass_flag: uniform_bank(
                &TABLE_9_8_CU_TRANSQUANT_BYPASS_FLAG,
                init_type,
                slice_qp_y,
            ),
            cu_skip_flag: inter_bank(&TABLE_9_9_CU_SKIP_FLAG, init_type, slice_qp_y),
            palette_mode_flag: uniform_bank(&TABLE_9_38_PALETTE_MODE_FLAG, init_type, slice_qp_y),
            pred_mode_flag: inter_bank(&TABLE_9_10_PRED_MODE_FLAG, init_type, slice_qp_y),
            part_mode,
            prev_intra_luma_pred_flag: uniform_bank(
                &TABLE_9_12_PREV_INTRA_LUMA_PRED_FLAG,
                init_type,
                slice_qp_y,
            ),
            intra_chroma_pred_mode: uniform_bank(
                &TABLE_9_13_INTRA_CHROMA_PRED_MODE,
                init_type,
                slice_qp_y,
            ),
            rqt_root_cbf: inter_bank(&TABLE_9_14_RQT_ROOT_CBF, init_type, slice_qp_y),
            merge_flag: inter_bank(&TABLE_9_15_MERGE_FLAG, init_type, slice_qp_y),
            merge_idx: inter_bank(&TABLE_9_16_MERGE_IDX, init_type, slice_qp_y),
            inter_pred_idc: inter_bank(&TABLE_9_17_INTER_PRED_IDC, init_type, slice_qp_y),
            ref_idx: inter_bank(&TABLE_9_18_REF_IDX, init_type, slice_qp_y),
            mvp_flag: inter_bank(&TABLE_9_19_MVP_FLAG, init_type, slice_qp_y),
            split_transform_flag: uniform_bank(
                &TABLE_9_20_SPLIT_TRANSFORM_FLAG,
                init_type,
                slice_qp_y,
            ),
            cbf_luma: uniform_bank(&TABLE_9_21_CBF_LUMA, init_type, slice_qp_y),
            cbf_chroma,
            abs_mvd_greater0_flag,
            abs_mvd_greater1_flag,
            tu_residual_act_flag: uniform_bank(
                &TABLE_9_39_TU_RESIDUAL_ACT_FLAG,
                init_type,
                slice_qp_y,
            ),
            log2_res_scale_abs_plus1: uniform_bank(
                &TABLE_9_36_LOG2_RES_SCALE_ABS_PLUS1,
                init_type,
                slice_qp_y,
            ),
            res_scale_sign_flag: uniform_bank(
                &TABLE_9_37_RES_SCALE_SIGN_FLAG,
                init_type,
                slice_qp_y,
            ),
            transform_skip_flag,
            explicit_rdpcm_flag: rdpcm_bank(&TABLE_9_32_EXPLICIT_RDPCM_FLAG),
            explicit_rdpcm_dir_flag: rdpcm_bank(&TABLE_9_33_EXPLICIT_RDPCM_DIR_FLAG),
            residual: ResidualContexts::init(init_type, slice_qp_y),
            palette_run_prefix: uniform_bank(&TABLE_9_40_PALETTE_RUN_PREFIX, init_type, slice_qp_y),
            palette_copy_above_flag: uniform_bank(
                &TABLE_9_41_PALETTE_COPY_ABOVE_FLAG,
                init_type,
                slice_qp_y,
            ),
            palette_transpose_flag: uniform_bank(
                &TABLE_9_42_PALETTE_TRANSPOSE_FLAG,
                init_type,
                slice_qp_y,
            ),
            cu_qp_delta_abs: uniform_bank(&TABLE_9_24_CU_QP_DELTA_ABS, init_type, slice_qp_y),
            cu_chroma_qp_offset_flag: uniform_bank(
                &TABLE_9_34_CU_CHROMA_QP_OFFSET_FLAG,
                init_type,
                slice_qp_y,
            ),
            cu_chroma_qp_offset_idx: uniform_bank(
                &TABLE_9_35_CU_CHROMA_QP_OFFSET_IDX,
                init_type,
                slice_qp_y,
            ),
            palette_predictor: crate::hevc::engine::palette::PalettePredictor::default(),
        }
    }

    /// Initialize the context array for a slice: derives `initType`
    /// from `slice_type` and `cabac_init_flag` per equation 9-7
    /// ([`init_type`]), then runs [`SliceContexts::init`].
    /// `cabac_init_flag` is the §7.3.6.1 header flag (inferred 0 when
    /// absent — pass `false` for I slices or when
    /// `cabac_init_present_flag == 0`); `slice_qp_y` is `SliceQpY`
    /// per equation 7-54 (`26 + init_qp_minus26 + slice_qp_delta`).
    #[must_use]
    pub fn for_slice(slice_type: SliceType, cabac_init_flag: bool, slice_qp_y: i32) -> Self {
        let raw = match slice_type {
            SliceType::B => 0,
            SliceType::P => 1,
            SliceType::I => 2,
        };
        Self::init(init_type(raw, cabac_init_flag), slice_qp_y)
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(any())]
mod tests {
    use super::*;

    /// Every table's printed ctxIdx span (Tables 9-5..9-42).
    #[test]
    fn table_shapes() {
        assert_eq!(TABLE_9_5_SAO_MERGE_FLAG.len(), 3);
        assert_eq!(TABLE_9_6_SAO_TYPE_IDX.len(), 3);
        assert_eq!(TABLE_9_7_SPLIT_CU_FLAG.len(), 9);
        assert_eq!(TABLE_9_8_CU_TRANSQUANT_BYPASS_FLAG.len(), 3);
        assert_eq!(TABLE_9_9_CU_SKIP_FLAG.len(), 6);
        assert_eq!(TABLE_9_10_PRED_MODE_FLAG.len(), 2);
        assert_eq!(TABLE_9_11_PART_MODE.len(), 9);
        assert_eq!(TABLE_9_12_PREV_INTRA_LUMA_PRED_FLAG.len(), 3);
        assert_eq!(TABLE_9_13_INTRA_CHROMA_PRED_MODE.len(), 3);
        assert_eq!(TABLE_9_14_RQT_ROOT_CBF.len(), 2);
        assert_eq!(TABLE_9_15_MERGE_FLAG.len(), 2);
        assert_eq!(TABLE_9_16_MERGE_IDX.len(), 2);
        assert_eq!(TABLE_9_17_INTER_PRED_IDC.len(), 10);
        assert_eq!(TABLE_9_18_REF_IDX.len(), 4);
        assert_eq!(TABLE_9_19_MVP_FLAG.len(), 2);
        assert_eq!(TABLE_9_20_SPLIT_TRANSFORM_FLAG.len(), 9);
        assert_eq!(TABLE_9_21_CBF_LUMA.len(), 6);
        assert_eq!(TABLE_9_22_CBF_CHROMA.len(), 15);
        assert_eq!(TABLE_9_23_ABS_MVD_GREATER_FLAG.len(), 4);
        assert_eq!(TABLE_9_24_CU_QP_DELTA_ABS.len(), 6);
        assert_eq!(TABLE_9_25_TRANSFORM_SKIP_FLAG.len(), 6);
        assert_eq!(TABLE_9_26_LAST_SIG_COEFF_X_PREFIX.len(), 54);
        assert_eq!(TABLE_9_27_LAST_SIG_COEFF_Y_PREFIX.len(), 54);
        assert_eq!(TABLE_9_28_CODED_SUB_BLOCK_FLAG.len(), 12);
        assert_eq!(TABLE_9_29_SIG_COEFF_FLAG.len(), 132);
        assert_eq!(TABLE_9_30_COEFF_ABS_LEVEL_GREATER1_FLAG.len(), 72);
        assert_eq!(TABLE_9_31_COEFF_ABS_LEVEL_GREATER2_FLAG.len(), 18);
        assert_eq!(TABLE_9_32_EXPLICIT_RDPCM_FLAG.len(), 4);
        assert_eq!(TABLE_9_33_EXPLICIT_RDPCM_DIR_FLAG.len(), 4);
        assert_eq!(TABLE_9_34_CU_CHROMA_QP_OFFSET_FLAG.len(), 3);
        assert_eq!(TABLE_9_35_CU_CHROMA_QP_OFFSET_IDX.len(), 3);
        assert_eq!(TABLE_9_36_LOG2_RES_SCALE_ABS_PLUS1.len(), 24);
        assert_eq!(TABLE_9_37_RES_SCALE_SIGN_FLAG.len(), 6);
        assert_eq!(TABLE_9_38_PALETTE_MODE_FLAG.len(), 3);
        assert_eq!(TABLE_9_39_TU_RESIDUAL_ACT_FLAG.len(), 3);
        assert_eq!(TABLE_9_40_PALETTE_RUN_PREFIX.len(), 24);
        assert_eq!(TABLE_9_41_PALETTE_COPY_ABOVE_FLAG.len(), 3);
        assert_eq!(TABLE_9_42_PALETTE_TRANSPOSE_FLAG.len(), 3);
        // The all-154 tables really are uniform.
        assert!(TABLE_9_24_CU_QP_DELTA_ABS.iter().all(|&v| v == 154));
        assert!(
            TABLE_9_36_LOG2_RES_SCALE_ABS_PLUS1
                .iter()
                .all(|&v| v == 154)
        );
        assert!(TABLE_9_37_RES_SCALE_SIGN_FLAG.iter().all(|&v| v == 154));
        assert!(TABLE_9_40_PALETTE_RUN_PREFIX.iter().all(|&v| v == 154));
        // The all-139 tables (transform-skip / RDPCM group).
        assert!(TABLE_9_25_TRANSFORM_SKIP_FLAG.iter().all(|&v| v == 139));
        assert!(TABLE_9_32_EXPLICIT_RDPCM_FLAG.iter().all(|&v| v == 139));
        assert!(TABLE_9_33_EXPLICIT_RDPCM_DIR_FLAG.iter().all(|&v| v == 139));
    }

    /// Tables 9-26 and 9-27 print identical values (the elements still
    /// adapt independent variables).
    #[test]
    fn last_sig_coeff_tables_print_identically() {
        assert_eq!(
            TABLE_9_26_LAST_SIG_COEFF_X_PREFIX,
            TABLE_9_27_LAST_SIG_COEFF_Y_PREFIX
        );
    }

    fn cm(p_state_idx: u8, val_mps: u8) -> ContextModel {
        ContextModel {
            p_state_idx,
            val_mps,
        }
    }

    /// §9.3.2.2 derived `(pStateIdx, valMps)` pins across QPs for
    /// elements drawn from regular three-`initType` tables. Expected
    /// values hand-evaluated from equations 9-4..9-6 on the cited
    /// table entries.
    #[test]
    fn derived_state_pins_uniform_tables() {
        // Table 9-7 `split_cu_flag`, initType 0 slot 0 (initValue 139).
        assert_eq!(SliceContexts::init(0, 0).split_cu_flag[0], cm(8, 1));
        assert_eq!(SliceContexts::init(0, 26).split_cu_flag[0], cm(0, 0));
        assert_eq!(SliceContexts::init(0, 51).split_cu_flag[0], cm(7, 0));
        // Table 9-7 `split_cu_flag`, initType 2 slot 0 (initValue 107).
        assert_eq!(SliceContexts::init(2, 17).split_cu_flag[0], cm(7, 0));
        assert_eq!(SliceContexts::init(2, 26).split_cu_flag[0], cm(16, 0));
        assert_eq!(SliceContexts::init(2, 51).split_cu_flag[0], cm(39, 0));
        // Table 9-6 `sao_type_idx`, initType 0 (initValue 200): the
        // valMps flip across the QP range.
        assert_eq!(SliceContexts::init(0, 0).sao_type_idx[0], cm(15, 0));
        assert_eq!(SliceContexts::init(0, 26).sao_type_idx[0], cm(8, 1));
        assert_eq!(SliceContexts::init(0, 51).sao_type_idx[0], cm(31, 1));
        // Table 9-6 `sao_type_idx`, initType 2 (initValue 160):
        // slopeIdx 10 ⇒ m == 5; the state barely moves with QP.
        assert_eq!(SliceContexts::init(2, 0).sao_type_idx[0], cm(62, 0));
        assert_eq!(SliceContexts::init(2, 51).sao_type_idx[0], cm(62, 0));
        // Table 9-20 `split_transform_flag`, initType 2 slot 0
        // (initValue 224).
        assert_eq!(SliceContexts::init(2, 0).split_transform_flag[0], cm(62, 0));
        assert_eq!(
            SliceContexts::init(2, 37).split_transform_flag[0],
            cm(22, 0)
        );
        assert_eq!(SliceContexts::init(2, 51).split_transform_flag[0], cm(0, 0));
        // Table 9-21 `cbf_luma`, initType 0 slot 0 (initValue 111).
        assert_eq!(SliceContexts::init(0, 17).cbf_luma[0], cm(24, 1));
        assert_eq!(SliceContexts::init(0, 51).cbf_luma[0], cm(7, 0));
        // QP clamp (equation 9-6 Clip3( 0, 51, SliceQpY )): out-of-range
        // QPs pin to the endpoints.
        assert_eq!(
            SliceContexts::init(0, -7).cbf_luma[0],
            SliceContexts::init(0, 0).cbf_luma[0]
        );
        assert_eq!(
            SliceContexts::init(0, 60).cbf_luma[0],
            SliceContexts::init(0, 51).cbf_luma[0]
        );
    }

    /// Pins for the inter-only banks (Table 9-4 `initType` 1 / 2
    /// columns).
    #[test]
    fn derived_state_pins_inter_tables() {
        // Table 9-9 `cu_skip_flag`, initType 1 slot 0 (initValue 197).
        assert_eq!(SliceContexts::init(1, 17).cu_skip_flag[0], cm(24, 0));
        assert_eq!(SliceContexts::init(1, 26).cu_skip_flag[0], cm(15, 0));
        assert_eq!(SliceContexts::init(1, 51).cu_skip_flag[0], cm(7, 1));
        // Table 9-16 `merge_idx`, initType 1 (initValue 122).
        assert_eq!(SliceContexts::init(1, 0).merge_idx[0], cm(0, 1));
        assert_eq!(SliceContexts::init(1, 26).merge_idx[0], cm(16, 0));
        assert_eq!(SliceContexts::init(1, 51).merge_idx[0], cm(31, 0));
        // Table 9-17 `inter_pred_idc`, initType 1 slot 0 (initValue 95).
        assert_eq!(SliceContexts::init(1, 26).inter_pred_idc[0], cm(7, 1));
        assert_eq!(SliceContexts::init(1, 51).inter_pred_idc[0], cm(23, 0));
        // The initType 2 column starts at the table's second span:
        // Table 9-17 ctxIdx 5 also holds 95.
        assert_eq!(
            SliceContexts::init(2, 26).inter_pred_idc[0],
            SliceContexts::init(1, 26).inter_pred_idc[0]
        );
        // Table 9-23 interleave: `abs_mvd_greater0_flag` initType 2
        // reads ctxIdx 2 (initValue 169), `abs_mvd_greater1_flag`
        // initType 1 reads ctxIdx 1 (initValue 198).
        assert_eq!(
            SliceContexts::init(2, 26).abs_mvd_greater0_flag[0],
            cm(0, 1)
        );
        assert_eq!(
            SliceContexts::init(2, 51).abs_mvd_greater0_flag[0],
            cm(7, 1)
        );
        assert_eq!(
            SliceContexts::init(1, 26).abs_mvd_greater1_flag[0],
            cm(7, 0)
        );
        assert_eq!(
            SliceContexts::init(1, 51).abs_mvd_greater1_flag[0],
            cm(15, 1)
        );
    }

    /// Pins for the irregular Table 9-4 layouts (`part_mode`,
    /// `cbf_cb`/`cbf_cr`).
    #[test]
    fn derived_state_pins_irregular_tables() {
        // Table 9-11 `part_mode` initType 0: only ctxIdx 0
        // (initValue 184) is initialized; slots 1..3 take the
        // non-adapting placeholder.
        let i_slice = SliceContexts::init(0, 26);
        assert_eq!(i_slice.part_mode[0], cm(0, 1));
        for slot in 1..4 {
            assert_eq!(i_slice.part_mode[slot], ContextModel::terminate_state());
        }
        assert_eq!(SliceContexts::init(0, 0).part_mode[0], cm(15, 0));
        assert_eq!(SliceContexts::init(0, 51).part_mode[0], cm(15, 1));
        // `part_mode` initType 1 slot 1 reads Table 9-11 ctxIdx 2
        // (initValue 139); initType 2 slot 1 reads ctxIdx 6 (also 139).
        assert_eq!(SliceContexts::init(1, 26).part_mode[1], cm(0, 0));
        assert_eq!(
            SliceContexts::init(2, 26).part_mode[1],
            SliceContexts::init(1, 26).part_mode[1]
        );
        // Table 9-22 `cbf_cb`/`cbf_cr` initType 0: slot 0 reads ctxIdx 0
        // (initValue 94), slot 4 reads ctxIdx 12 (initValue 154).
        assert_eq!(SliceContexts::init(0, 0).cbf_chroma[0], cm(32, 1));
        assert_eq!(SliceContexts::init(0, 26).cbf_chroma[0], cm(0, 0));
        assert_eq!(SliceContexts::init(0, 51).cbf_chroma[0], cm(31, 0));
        for qp in [0, 26, 51] {
            // initValue 154 ⇒ m == 0, n == 64 ⇒ preCtxState == 64 at
            // every QP: pStateIdx 0, valMps 1.
            assert_eq!(SliceContexts::init(0, qp).cbf_chroma[4], cm(0, 1));
        }
        // Table 9-25 `transform_skip_flag` (initValue 139 everywhere):
        // luma and chroma slots agree at each initType.
        let b = SliceContexts::init(2, 26);
        assert_eq!(b.transform_skip_flag[0], b.transform_skip_flag[1]);
        assert_eq!(b.transform_skip_flag[0], cm(0, 0));
    }

    /// Whole-array smoke test per `initType`: every table-derived
    /// context lands in the equations-9-4..9-6 range
    /// (`pStateIdx <= 62`), and the `initType == 0` placeholder
    /// (`pStateIdx == 63`) appears exactly on the inter-only banks.
    #[test]
    fn whole_array_smoke_per_init_type() {
        for init_type in 0u8..=2 {
            for qp in [0, 26, 51] {
                let ctx = SliceContexts::init(init_type, qp);
                let inter_only: Vec<&[ContextModel]> = vec![
                    &ctx.cu_skip_flag,
                    &ctx.pred_mode_flag,
                    &ctx.rqt_root_cbf,
                    &ctx.merge_flag,
                    &ctx.merge_idx,
                    &ctx.inter_pred_idc,
                    &ctx.ref_idx,
                    &ctx.mvp_flag,
                    &ctx.abs_mvd_greater0_flag,
                    &ctx.abs_mvd_greater1_flag,
                    &ctx.explicit_rdpcm_flag,
                    &ctx.explicit_rdpcm_dir_flag,
                ];
                let every_type: Vec<&[ContextModel]> = vec![
                    &ctx.sao_merge_flag,
                    &ctx.sao_type_idx,
                    &ctx.split_cu_flag,
                    &ctx.cu_transquant_bypass_flag,
                    &ctx.palette_mode_flag,
                    &ctx.prev_intra_luma_pred_flag,
                    &ctx.intra_chroma_pred_mode,
                    &ctx.split_transform_flag,
                    &ctx.cbf_luma,
                    &ctx.cbf_chroma,
                    &ctx.tu_residual_act_flag,
                    &ctx.log2_res_scale_abs_plus1,
                    &ctx.res_scale_sign_flag,
                    &ctx.transform_skip_flag,
                    &ctx.residual.last_sig_coeff_x_prefix,
                    &ctx.residual.last_sig_coeff_y_prefix,
                    &ctx.residual.coded_sub_block_flag,
                    &ctx.residual.sig_coeff_flag,
                    &ctx.residual.coeff_abs_level_greater1_flag,
                    &ctx.residual.coeff_abs_level_greater2_flag,
                    &ctx.palette_run_prefix,
                    &ctx.palette_copy_above_flag,
                    &ctx.palette_transpose_flag,
                    &ctx.cu_qp_delta_abs,
                    &ctx.cu_chroma_qp_offset_flag,
                    &ctx.cu_chroma_qp_offset_idx,
                ];
                let mut total = 0usize;
                for bank in &every_type {
                    for c in bank.iter() {
                        assert!(c.p_state_idx <= 62, "table-derived state out of range");
                        assert!(c.val_mps <= 1);
                        total += 1;
                    }
                }
                // `part_mode` slot 0 is table-derived at every
                // initType; slots 1..3 only for initType 1 / 2.
                assert!(ctx.part_mode[0].p_state_idx <= 62);
                total += 4;
                for bank in &inter_only {
                    for c in bank.iter() {
                        if init_type == 0 {
                            assert_eq!(*c, ContextModel::terminate_state());
                        } else {
                            assert!(c.p_state_idx <= 62);
                        }
                        total += 1;
                    }
                }
                if init_type == 0 {
                    for slot in 1..4 {
                        assert_eq!(ctx.part_mode[slot], ContextModel::terminate_state());
                    }
                } else {
                    for slot in 1..4 {
                        assert!(ctx.part_mode[slot].p_state_idx <= 62);
                    }
                }
                assert_eq!(total, SliceContexts::CONTEXT_COUNT);
            }
        }
    }

    /// Equation 9-7 routing through [`SliceContexts::for_slice`]: the
    /// `cabac_init_flag` swap between P and B, and its irrelevance for
    /// I slices.
    #[test]
    fn for_slice_eq_9_7_routing() {
        let qp = 30;
        assert_eq!(
            SliceContexts::for_slice(SliceType::I, false, qp),
            SliceContexts::init(0, qp)
        );
        assert_eq!(
            SliceContexts::for_slice(SliceType::I, true, qp),
            SliceContexts::init(0, qp)
        );
        assert_eq!(
            SliceContexts::for_slice(SliceType::P, false, qp),
            SliceContexts::init(1, qp)
        );
        assert_eq!(
            SliceContexts::for_slice(SliceType::P, true, qp),
            SliceContexts::init(2, qp)
        );
        assert_eq!(
            SliceContexts::for_slice(SliceType::B, false, qp),
            SliceContexts::init(2, qp)
        );
        assert_eq!(
            SliceContexts::for_slice(SliceType::B, true, qp),
            SliceContexts::init(1, qp)
        );
        // P with the flag and B without select the same array; P
        // without and B with likewise.
        assert_eq!(
            SliceContexts::for_slice(SliceType::P, true, qp),
            SliceContexts::for_slice(SliceType::B, false, qp)
        );
        assert_eq!(
            SliceContexts::for_slice(SliceType::P, false, qp),
            SliceContexts::for_slice(SliceType::B, true, qp)
        );
        // The three initTypes produce three distinct arrays.
        assert_ne!(SliceContexts::init(0, qp), SliceContexts::init(1, qp));
        assert_ne!(SliceContexts::init(1, qp), SliceContexts::init(2, qp));
        assert_ne!(SliceContexts::init(0, qp), SliceContexts::init(2, qp));
    }

    /// The span-selection helpers reject malformed inputs and slice
    /// the right ctxIdx ranges.
    #[test]
    fn span_selection_helpers() {
        assert_eq!(
            uniform_init_values::<3>(&TABLE_9_7_SPLIT_CU_FLAG, 1),
            [107, 139, 126]
        );
        assert_eq!(
            inter_init_values::<3>(&TABLE_9_9_CU_SKIP_FLAG, 2),
            Some([197, 185, 201])
        );
        assert_eq!(inter_init_values::<3>(&TABLE_9_9_CU_SKIP_FLAG, 0), None);
        let sig0 = sig_coeff_flag_init_values(0);
        assert_eq!(sig0[0], 111);
        assert_eq!(sig0[41], 111);
        // The transform-skip tail: Table 9-29 ctxIdx 126 / 127.
        assert_eq!(sig0[42], 141);
        assert_eq!(sig0[43], 111);
        let sig1 = sig_coeff_flag_init_values(1);
        assert_eq!(sig1[0], 155);
        assert_eq!(sig1[41], 140);
        let sig2 = sig_coeff_flag_init_values(2);
        assert_eq!(sig2[0], 170);
        assert_eq!(sig2[41], 140);
    }

    #[test]
    #[should_panic(expected = "initType must be 0, 1 or 2")]
    fn init_rejects_init_type_3() {
        let _ = SliceContexts::init(3, 26);
    }
}
