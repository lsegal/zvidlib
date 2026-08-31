//! §7.3.8.1 .. §7.3.8.6 slice-data CABAC syntax-element walk.
//!
//! This module is the upper rung of the §7.3.8 slice-data parse loop:
//! it drives the CABAC engine through the per-CTU syntax structures —
//! §7.3.8.3 `sao( )`, §7.3.8.4 `coding_tree_unit( )` /
//! `coding_quadtree( )`, §7.3.8.5 `coding_unit( )`, and §7.3.8.6
//! `prediction_unit( )` — composing the §7.3.8.3 / §7.3.8.5 / §7.3.8.6
//! leaf decode primitives ([`crate::hevc::engine::binarization`]) with the already
//! implemented §7.3.8.8 `transform_tree( )` recursion
//! ([`crate::hevc::engine::transform_tree`]) and its §7.3.8.10 `transform_unit( )`
//! leaf.
//!
//! The driver produces a structured parse tree (`CodingTreeUnit` →
//! `CodingQuadtree` → `CodingUnit`) rather than reconstructed samples:
//! it decodes the complete CABAC syntax-element stream of a CTU, which
//! is the prerequisite the §8.4 / §8.5 picture-reconstruction passes
//! consume. Picture-level neighbour availability (§6.4.1) and the
//! `CtDepth` / `cu_skip_flag` neighbour grids feeding the §9.3.4.2.2
//! `split_cu_flag` / `cu_skip_flag` ctxInc derivations are carried by
//! the picture-level [`PictureParseState`] (per-4×4-cell grids gated on
//! the §6.4.1 slice / tile availability), so a coding block's left /
//! above neighbour reads work across CTU boundaries.
//!
//! The §6.5.1 quantization-group reset (`IsCuQpDeltaCoded`,
//! `CuQpDeltaVal`, `IsCuChromaQpOffsetCoded`) is performed by the
//! §7.3.8.4 `coding_quadtree( )` walk at each node whose `log2CbSize`
//! meets the `Log2MinCuQpDeltaSize` / `Log2MinCuChromaQpOffsetSize`
//! threshold, mirroring the syntax table; the resulting
//! [`crate::hevc::engine::transform_unit::QuantGroupState`] is threaded into the
//! transform tree.

use crate::hevc::engine::binarization::{
    CuPredMode, InterPredIdc, LumaIntraModeSource, MvdComponent, PartMode, PartModeResult,
    cu_pred_mode_from_flag, cu_pred_mode_from_skip, cu_skip_flag_ctx_inc, decode_cu_skip_flag,
    decode_cu_transquant_bypass_flag, decode_end_of_slice_segment_flag, decode_inter_pred_idc,
    decode_intra_chroma_pred_mode, decode_merge_flag, decode_merge_idx, decode_mpm_idx,
    decode_mvd_pair, decode_mvp_flag, decode_part_mode, decode_pcm_flag, decode_pred_mode_flag,
    decode_prev_intra_luma_pred_flag, decode_ref_idx, decode_rem_intra_luma_pred_mode,
    decode_rqt_root_cbf, decode_sao_band_position, decode_sao_eo_class, decode_sao_merge_flag,
    decode_sao_offset_abs, decode_sao_offset_sign, decode_sao_type_idx, decode_split_cu_flag,
    derive_intra_pred_mode_c, derive_intra_pred_mode_y, intra_luma_cand_mode_list,
    luma_intra_mode_source_from_flag, split_cu_flag_ctx_inc,
};
use crate::hevc::engine::cabac::CabacEngine;
use crate::hevc::engine::ctx_init::SliceContexts;
use crate::hevc::engine::intra_mode_field::{IntraModeField, Neighbour};
use crate::hevc::engine::residual::ResidualCodingError;
use crate::hevc::engine::transform_tree::{
    TransformTree, TransformTreeParams, decode_transform_tree,
};
use crate::hevc::engine::transform_unit::{
    CuPredMode as TuCuPredMode, QuantGroupState, TransformUnitParams,
};

/// Map the §7.3.8.5 [`binarization::CuPredMode`](CuPredMode) (which
/// carries the `MODE_SKIP` not-present variant) to the two-state
/// [`crate::hevc::engine::transform_unit::CuPredMode`] the transform tree / unit
/// consume. A skip CU never enters the transform tree (it has no
/// residual), so `Skip` collapses to `Inter` defensively.
fn to_tu_pred_mode(m: CuPredMode) -> TuCuPredMode {
    match m {
        CuPredMode::Intra => TuCuPredMode::Intra,
        CuPredMode::Inter | CuPredMode::Skip => TuCuPredMode::Inter,
    }
}

/// Per-CTU sequence / picture / slice constants the §7.3.8 walk reads.
/// These derive from the active SPS / PPS / slice header (§7.4.3) and
/// are constant for one slice segment's worth of CTUs.
#[derive(Debug, Clone, Copy)]
pub struct SliceDataParams {
    /// `CtbLog2SizeY` (§7.4.3.2) — the luma coding-tree-block log2 size.
    pub ctb_log2_size_y: u32,
    /// `MinCbLog2SizeY` (§7.4.3.2) — the minimum luma coding-block log2
    /// size; bounds the §7.3.8.4 `split_cu_flag` presence gate.
    pub min_cb_log2_size_y: u32,
    /// `MaxTbLog2SizeY` (§7.4.3.2).
    pub max_tb_log2_size_y: u32,
    /// `MinTbLog2SizeY` (§7.4.3.2).
    pub min_tb_log2_size_y: u32,
    /// `pic_width_in_luma_samples` (§7.4.3.2.1).
    pub pic_width_in_luma_samples: u32,
    /// `pic_height_in_luma_samples` (§7.4.3.2.1).
    pub pic_height_in_luma_samples: u32,
    /// `ChromaArrayType` (0 = monochrome, 1 = 4:2:0, 2 = 4:2:2,
    /// 3 = 4:4:4).
    pub chroma_array_type: u8,
    /// `BitDepthY` (§7.4.3.2.1) — used by §7.3.8.3 `sao_offset_abs`.
    pub bit_depth_luma: u32,
    /// `BitDepthC` (§7.4.3.2.1) — used by §7.3.8.3 `sao_offset_abs`.
    pub bit_depth_chroma: u32,
    /// `slice_type == I` (§7.4.7.1).
    pub slice_type_is_i: bool,
    /// `slice_type == B` (§7.4.7.1).
    pub slice_type_is_b: bool,
    /// `slice_sao_luma_flag` (§7.4.7.1).
    pub slice_sao_luma_flag: bool,
    /// `slice_sao_chroma_flag` (§7.4.7.1).
    pub slice_sao_chroma_flag: bool,
    /// `transquant_bypass_enabled_flag` (§7.4.3.3.1).
    pub transquant_bypass_enabled_flag: bool,
    /// `cu_qp_delta_enabled_flag` (§7.4.3.3.1).
    pub cu_qp_delta_enabled_flag: bool,
    /// `Log2MinCuQpDeltaSize` (§7.4.3.3.1) = `CtbLog2SizeY −
    /// diff_cu_qp_delta_depth`.
    pub log2_min_cu_qp_delta_size: u32,
    /// per-slice `cu_chroma_qp_offset_enabled_flag` (§7.4.9.10).
    pub cu_chroma_qp_offset_enabled_flag: bool,
    /// `Log2MinCuChromaQpOffsetSize` (§7.4.3.3.1).
    pub log2_min_cu_chroma_qp_offset_size: u32,
    /// `chroma_qp_offset_list_len_minus1` (§7.4.3.3.1).
    pub chroma_qp_offset_list_len_minus1: u32,
    /// `amp_enabled_flag` (§7.4.3.2.1).
    pub amp_enabled_flag: bool,
    /// PCM block: `pcm_enabled_flag` (§7.4.3.2.1).
    pub pcm_enabled_flag: bool,
    /// `Log2MinIpcmCbSizeY` (§7.4.3.2.1).
    pub log2_min_ipcm_cb_size_y: u32,
    /// `Log2MaxIpcmCbSizeY` (§7.4.3.2.1).
    pub log2_max_ipcm_cb_size_y: u32,
    /// `PcmBitDepthY` (§7.4.3.2.1 equation 7-25; meaningful only when
    /// [`Self::pcm_enabled_flag`]).
    pub pcm_bit_depth_luma: u32,
    /// `PcmBitDepthC` (§7.4.3.2.1 equation 7-26).
    pub pcm_bit_depth_chroma: u32,
    /// `max_transform_hierarchy_depth_intra` (§7.4.3.2.1).
    pub max_transform_hierarchy_depth_intra: u32,
    /// `max_transform_hierarchy_depth_inter` (§7.4.3.2.1).
    pub max_transform_hierarchy_depth_inter: u32,
    /// `MaxNumMergeCand` (§7.4.7.1) — `5 −
    /// five_minus_max_num_merge_cand`.
    pub max_num_merge_cand: u32,
    /// `num_ref_idx_l0_active_minus1` (§7.4.7.1).
    pub num_ref_idx_l0_active_minus1: u32,
    /// `num_ref_idx_l1_active_minus1` (§7.4.7.1).
    pub num_ref_idx_l1_active_minus1: u32,
    /// `mvd_l1_zero_flag` (§7.4.7.1).
    pub mvd_l1_zero_flag: bool,
    /// PPS `sign_data_hiding_enabled_flag` (§7.4.3.3.1).
    pub sign_data_hiding_enabled_flag: bool,
    /// PPS `cross_component_prediction_enabled_flag` (§7.4.3.3.1).
    pub cross_component_prediction_enabled_flag: bool,
    /// SCC `residual_adaptive_colour_transform_enabled_flag`
    /// (§7.4.3.3.1).
    pub residual_adaptive_colour_transform_enabled_flag: bool,
    /// PPS `transform_skip_enabled_flag` (§7.4.3.3.1) — the §7.3.8.11
    /// `transform_skip_flag` presence gate.
    pub transform_skip_enabled_flag: bool,
    /// `Log2MaxTransformSkipSize` (§7.4.3.3.2) —
    /// `log2_max_transform_skip_block_size_minus2 + 2` (2 when the PPS
    /// range extension is absent).
    pub log2_max_transform_skip_size: u32,
    /// SPS range extension `implicit_rdpcm_enabled_flag` (§7.4.3.2.2)
    /// — part of the §7.3.8.11 `signHidden` condition.
    pub implicit_rdpcm_enabled_flag: bool,
    /// SPS range extension `explicit_rdpcm_enabled_flag` (§7.4.3.2.2)
    /// — the §7.3.8.11 `explicit_rdpcm_flag` presence gate.
    pub explicit_rdpcm_enabled_flag: bool,
    /// SPS range extension `transform_skip_context_enabled_flag`
    /// (§7.4.3.2.2) — the §9.3.4.2.5 transform-skip sig-ctx gate.
    pub transform_skip_context_enabled_flag: bool,
    /// SPS range extension `persistent_rice_adaptation_enabled_flag`
    /// (§7.4.3.2.2) — the §9.3.3.11 StatCoeff Rice path.
    pub persistent_rice_adaptation_enabled_flag: bool,
    /// SPS range extension `cabac_bypass_alignment_enabled_flag`
    /// (§7.4.3.2.2) — the §9.3.4.3.6 aligned bypass decoding gate.
    pub cabac_bypass_alignment_enabled_flag: bool,
    /// SPS range extension `extended_precision_processing_flag`
    /// (§7.4.3.2.2) — the §9.3.3.4 limited-EGk escape suffix gate.
    pub extended_precision_processing_flag: bool,
    /// SCC `palette_mode_enabled_flag` (§7.4.3.2.3) — the §7.3.8.5
    /// `palette_mode_flag` presence gate.
    pub palette_mode_enabled_flag: bool,
    /// SCC `palette_max_size` (§7.4.3.2.3).
    pub palette_max_size: u32,
    /// `PaletteMaxPredictorSize` (eq. 7-35).
    pub palette_max_predictor_size: u32,
}

/// §7.4.9.3 decoded SAO parameters for one colour component of one CTB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SaoComponent {
    /// `SaoTypeIdx[cIdx][rx][ry]` — 0 (not applied), 1 (band offset),
    /// 2 (edge offset).
    pub sao_type_idx: u8,
    /// `sao_offset_abs[cIdx][rx][ry][0..4]` magnitudes.
    pub offset_abs: [u32; 4],
    /// `sao_offset_sign[cIdx][rx][ry][0..4]` (band offset only).
    pub offset_sign: [u8; 4],
    /// `sao_band_position[cIdx][rx][ry]` (band offset only).
    pub band_position: u8,
    /// `SaoEoClass[cIdx][rx][ry]` (edge offset only).
    pub eo_class: u8,
}

/// §7.3.8.3 decoded SAO parameters for one CTB (all three components).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SaoCtbParams {
    /// `sao_merge_left_flag` — this CTB copies the left CTB's params.
    pub merge_left: bool,
    /// `sao_merge_up_flag` — this CTB copies the above CTB's params.
    pub merge_up: bool,
    /// Per-component parameters: `[Y, Cb, Cr]`. Only populated when
    /// neither merge flag is set (otherwise the caller resolves the
    /// merged source).
    pub components: [SaoComponent; 3],
}

/// One decoded §7.3.8.5 `coding_unit( )`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodingUnit {
    /// Luma top-left position `(x0, y0)`.
    pub x0: u32,
    /// Luma top-left position `(x0, y0)`.
    pub y0: u32,
    /// `log2CbSize`.
    pub log2_cb_size: u32,
    /// `CuPredMode[x0][y0]`.
    pub cu_pred_mode: CuPredMode,
    /// `cu_transquant_bypass_flag`.
    pub cu_transquant_bypass_flag: bool,
    /// `PartMode` (§7.4.9.5).
    pub part_mode: PartMode,
    /// `pcm_flag[x0][y0]`.
    pub pcm_flag: bool,
    /// §7.3.8.7 PCM sample payload (`Some` iff [`Self::pcm_flag`]),
    /// already scaled to the picture bit depth per §8.4.1
    /// equation 8-12 (`pcm_sample << (BitDepth − PcmBitDepth)`).
    pub pcm: Option<PcmSamples>,
    /// §7.3.8.13 palette coding unit payload (`Some` iff
    /// `palette_mode_flag == 1`; such a CU has no prediction units and
    /// no transform tree).
    pub palette: Option<Box<crate::hevc::engine::palette::PaletteCu>>,
    /// Decoded prediction units (intra: empty — the luma/chroma intra
    /// modes carry the prediction; inter: 1..=4 entries).
    pub prediction_units: Vec<PredictionUnit>,
    /// `prev_intra_luma_pred_flag` / `mpm_idx` /
    /// `rem_intra_luma_pred_mode` per luma prediction block (intra
    /// only), in §7.3.8.5 PB-loop order.
    pub intra_luma: Vec<IntraLumaMode>,
    /// `intra_chroma_pred_mode` values (intra only).
    pub intra_chroma_pred_mode: Vec<u8>,
    /// `rqt_root_cbf` (inter only; intra CUs always enter the tree).
    pub rqt_root_cbf: bool,
    /// The decoded §7.3.8.8 transform tree, present when the CU codes
    /// residual (`!pcm_flag && (intra || rqt_root_cbf)`).
    pub transform_tree: Option<TransformTree>,
}

/// §7.3.8.5 per-luma-prediction-block intra-mode signalling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntraLumaMode {
    /// `prev_intra_luma_pred_flag[xPb][yPb]`.
    pub prev_intra_luma_pred_flag: bool,
    /// `mpm_idx[xPb][yPb]` (present when `prev_intra_luma_pred_flag`).
    pub mpm_idx: Option<u8>,
    /// `rem_intra_luma_pred_mode[xPb][yPb]` (present otherwise).
    pub rem_intra_luma_pred_mode: Option<u8>,
}

/// §7.3.8.6 one decoded `prediction_unit( )`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictionUnit {
    /// `merge_flag[x0][y0]` (`false` for a skip CU, which carries only
    /// `merge_idx`).
    pub merge_flag: bool,
    /// `merge_idx[x0][y0]` (present for skip or merge).
    pub merge_idx: Option<u8>,
    /// `inter_pred_idc[x0][y0]` (present for non-merge B PUs; a P-slice
    /// non-merge PU is `PRED_L0`).
    pub inter_pred_idc: Option<InterPredIdc>,
    /// `ref_idx_l0[x0][y0]` (present for L0/BI non-merge PUs).
    pub ref_idx_l0: Option<u8>,
    /// `mvd_coding(…, 0)` (present for L0/BI non-merge PUs).
    pub mvd_l0: Option<[MvdComponent; 2]>,
    /// `mvp_l0_flag[x0][y0]`.
    pub mvp_l0_flag: Option<u8>,
    /// `ref_idx_l1[x0][y0]` (present for L1/BI non-merge PUs).
    pub ref_idx_l1: Option<u8>,
    /// `mvd_coding(…, 1)` (present for L1/BI non-merge PUs unless the
    /// `mvd_l1_zero_flag && PRED_BI` zero-inference path applies).
    pub mvd_l1: Option<[MvdComponent; 2]>,
    /// `mvp_l1_flag[x0][y0]`.
    pub mvp_l1_flag: Option<u8>,
}

/// One node of a decoded §7.3.8.4 `coding_quadtree( )`: a split node
/// with up to four in-picture children, or a leaf coding unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodingQuadtree {
    /// `split_cu_flag == 1`: the present (in-picture) children, raster
    /// order; off-picture children are absent (the §7.3.8.4 boundary
    /// `if( x1 < … )` guards).
    Split(Vec<CodingQuadtree>),
    /// `split_cu_flag == 0`: a leaf coding unit.
    Leaf(Box<CodingUnit>),
}

/// One decoded §7.3.8.2 `coding_tree_unit( )`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodingTreeUnit {
    /// The §7.3.8.3 SAO parameters, when `slice_sao_luma_flag ||
    /// slice_sao_chroma_flag`.
    pub sao: Option<SaoCtbParams>,
    /// The §7.3.8.4 coding-quadtree root.
    pub quadtree: CodingQuadtree,
}

/// Picture-level parse state threaded across the CTUs of a picture.
///
/// The §7.3.8 walk needs the **derived** `IntraPredModeY` /
/// `IntraPredModeC` of every intra prediction block *during* parsing:
/// the §7.4.9.11 `residual_coding( )` scan order of a 4×4 / 8×8 intra
/// transform block depends on the actual prediction mode, and that mode
/// comes from the §8.4.2 most-probable-mode derivation over the left /
/// above neighbour blocks — potentially in a different CTU. This state
/// carries the per-4×4 [`IntraModeField`] plus the per-CTB
/// `SliceAddrRs` / `TileId` grids that gate the §6.4.1 neighbour
/// availability.
#[derive(Debug)]
pub struct PictureParseState {
    field: IntraModeField,
    ctb_log2: u32,
    pic_width: u32,
    pic_height: u32,
    pic_w_ctbs: u32,
    /// Per-CTB `(SliceAddrRs, TileId)`, `None` until the CTU is begun.
    ctb_info: Vec<Option<(u32, u32)>>,
    /// The current CTU's `(SliceAddrRs, TileId)`.
    cur: (u32, u32),
    /// Per-4×4-cell `CtDepth` (−1 = not yet decoded) — the §9.3.4.2.2
    /// `split_cu_flag` ctxInc neighbour reads.
    ct_depth: Vec<i8>,
    /// Per-4×4-cell `cu_skip_flag` — the §9.3.4.2.2 `cu_skip_flag`
    /// ctxInc neighbour reads.
    cu_skip: Vec<u8>,
    w_cells: usize,
    h_cells: usize,
}

impl PictureParseState {
    /// A fresh per-picture state sized from the slice-data params.
    #[must_use]
    pub fn new(params: &SliceDataParams) -> Self {
        let ctb = 1u32 << params.ctb_log2_size_y;
        let w_ctbs = params.pic_width_in_luma_samples.div_ceil(ctb);
        let h_ctbs = params.pic_height_in_luma_samples.div_ceil(ctb);
        let w_cells = (params.pic_width_in_luma_samples as usize).div_ceil(4);
        let h_cells = (params.pic_height_in_luma_samples as usize).div_ceil(4);
        Self {
            field: IntraModeField::new(
                params.pic_width_in_luma_samples as usize,
                params.pic_height_in_luma_samples as usize,
                params.ctb_log2_size_y,
            ),
            ctb_log2: params.ctb_log2_size_y,
            pic_width: params.pic_width_in_luma_samples,
            pic_height: params.pic_height_in_luma_samples,
            pic_w_ctbs: w_ctbs,
            ctb_info: vec![None; (w_ctbs * h_ctbs) as usize],
            cur: (0, 0),
            ct_depth: vec![-1; w_cells * h_cells],
            cu_skip: vec![0; w_cells * h_cells],
            w_cells,
            h_cells,
        }
    }

    fn cell(&self, x: u32, y: u32) -> usize {
        ((y as usize) >> 2).min(self.h_cells - 1) * self.w_cells
            + ((x as usize) >> 2).min(self.w_cells - 1)
    }

    /// Record a coding block's `CtDepth` + `cu_skip_flag` over its area
    /// (the §9.3.4.2.2 neighbour state for later blocks).
    fn record_cu_depth(&mut self, x0: u32, y0: u32, log2_cb_size: u32, depth: u8, skip: u8) {
        let n = 1u32 << log2_cb_size;
        let x1 = (x0 + n).min(self.pic_width);
        let y1 = (y0 + n).min(self.pic_height);
        for y in (y0..y1).step_by(4) {
            for x in (x0..x1).step_by(4) {
                let c = self.cell(x, y);
                self.ct_depth[c] = depth as i8;
                self.cu_skip[c] = skip;
            }
        }
    }

    /// `(CtDepth, available)` of the neighbour of `(x0, y0)` — the
    /// §9.3.4.2.2 `split_cu_flag` ctxInc read, gated on the §6.4.1
    /// availability (in-picture, decoded, same slice, same tile).
    fn neighbour_ct_depth(&self, x0: u32, y0: u32, neighbour: Neighbour) -> (u32, bool) {
        let (x_nb, y_nb) = match neighbour {
            Neighbour::Left => (x0.wrapping_sub(1), y0),
            Neighbour::Above => (x0, y0.wrapping_sub(1)),
        };
        if !self.neighbour_available(x0, y0, neighbour) {
            return (0, false);
        }
        let d = self.ct_depth[self.cell(x_nb, y_nb)];
        if d < 0 {
            return (0, false);
        }
        (d as u32, true)
    }

    /// `(cu_skip_flag, available)` of the neighbour of `(x0, y0)` — the
    /// §9.3.4.2.2 `cu_skip_flag` ctxInc read.
    fn neighbour_cu_skip(&self, x0: u32, y0: u32, neighbour: Neighbour) -> (u8, bool) {
        let (x_nb, y_nb) = match neighbour {
            Neighbour::Left => (x0.wrapping_sub(1), y0),
            Neighbour::Above => (x0, y0.wrapping_sub(1)),
        };
        if !self.neighbour_available(x0, y0, neighbour) {
            return (0, false);
        }
        let c = self.cell(x_nb, y_nb);
        if self.ct_depth[c] < 0 {
            return (0, false);
        }
        (self.cu_skip[c], true)
    }

    /// Mark the CTU at `(x_ctb, y_ctb)` as belonging to slice segment
    /// sequence `slice_addr_rs` and tile `tile_id` — called before its
    /// syntax decode so same-CTU neighbour queries resolve.
    pub fn begin_ctu(&mut self, x_ctb: u32, y_ctb: u32, slice_addr_rs: u32, tile_id: u32) {
        let rs = (y_ctb >> self.ctb_log2) * self.pic_w_ctbs + (x_ctb >> self.ctb_log2);
        if let Some(slot) = self.ctb_info.get_mut(rs as usize) {
            *slot = Some((slice_addr_rs, tile_id));
        }
        self.cur = (slice_addr_rs, tile_id);
    }

    /// §6.4.1 availability of the left / above neighbour of the block at
    /// `(x_pb, y_pb)`: in-picture, already decoded (the mode field's
    /// `written` test), same slice and same tile.
    fn neighbour_available(&self, x_pb: u32, y_pb: u32, neighbour: Neighbour) -> bool {
        let (x_nb, y_nb) = match neighbour {
            Neighbour::Left => (x_pb as i64 - 1, y_pb as i64),
            Neighbour::Above => (x_pb as i64, y_pb as i64 - 1),
        };
        if x_nb < 0
            || y_nb < 0
            || x_nb >= i64::from(self.pic_width)
            || y_nb >= i64::from(self.pic_height)
        {
            return false;
        }
        let rs =
            ((y_nb as u32) >> self.ctb_log2) * self.pic_w_ctbs + ((x_nb as u32) >> self.ctb_log2);
        match self.ctb_info.get(rs as usize).copied().flatten() {
            // §6.4.1: a neighbour in a different slice segment sequence
            // or a different tile is unavailable.
            Some(info) => info == self.cur,
            // The neighbour's CTU has not been decoded yet.
            None => false,
        }
    }

    /// §8.4.2 — derive `IntraPredModeY` for one luma prediction block
    /// from the decoded signalling + the neighbour mode field, then
    /// record it for later neighbours.
    fn derive_and_record_luma_mode(
        &mut self,
        x_pb: u32,
        y_pb: u32,
        n_pb: u32,
        luma: &IntraLumaMode,
    ) -> u8 {
        let avail_a = self.neighbour_available(x_pb, y_pb, Neighbour::Left);
        let avail_b = self.neighbour_available(x_pb, y_pb, Neighbour::Above);
        let cand_a =
            self.field
                .cand_intra_pred_mode(x_pb as usize, y_pb as usize, Neighbour::Left, avail_a);
        let cand_b = self.field.cand_intra_pred_mode(
            x_pb as usize,
            y_pb as usize,
            Neighbour::Above,
            avail_b,
        );
        let cand_list = intra_luma_cand_mode_list(cand_a, cand_b);
        let source = luma_intra_mode_source_from_flag(u8::from(luma.prev_intra_luma_pred_flag));
        let field_val = match source {
            LumaIntraModeSource::Mpm => luma.mpm_idx.unwrap_or(0),
            LumaIntraModeSource::Remaining => luma.rem_intra_luma_pred_mode.unwrap_or(0),
        };
        let mode = derive_intra_pred_mode_y(cand_list, source, field_val);
        self.field
            .record_intra_pb(x_pb as usize, y_pb as usize, n_pb as usize, mode, false);
        mode
    }

    /// Record a PCM coding unit (its neighbours see `INTRA_DC`).
    fn record_pcm_cu(&mut self, x0: u32, y0: u32, n_cb: u32) {
        self.field
            .record_intra_pb(x0 as usize, y0 as usize, n_cb as usize, 1, true);
    }

    /// Record an inter / skip coding unit (its neighbours see
    /// `INTRA_DC`).
    fn record_non_intra_cu(&mut self, x0: u32, y0: u32, n_cb: u32, mode: CuPredMode) {
        self.field
            .record_non_intra_cu(x0 as usize, y0 as usize, n_cb as usize, mode);
    }

    /// Debug: the parse-time §8.4.2 derived mode recorded at a luma
    /// location.
    #[doc(hidden)]
    #[must_use]
    pub fn debug_mode_at(&self, x: u32, y: u32) -> Option<u8> {
        self.field.recorded_mode(x as usize, y as usize)
    }
}

/// Decode one §7.3.8.3 `sao( rx, ry )` syntax structure.
///
/// `merge_left_allowed` / `merge_up_allowed` are the §7.3.8.3 presence
/// conditions for the two merge flags (`rx > 0 && leftCtbInSliceSeg &&
/// leftCtbInTile` and the symmetric up condition), already evaluated by
/// the caller against the §6.5 tile / slice geometry. When the merge
/// path is taken the per-component fields are left at their defaults.
pub fn decode_sao(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    merge_left_allowed: bool,
    merge_up_allowed: bool,
) -> Result<SaoCtbParams, ResidualCodingError> {
    let mut out = SaoCtbParams::default();

    if merge_left_allowed {
        out.merge_left = decode_sao_merge_flag(engine, &mut ctx.sao_merge_flag[0])? != 0;
    }
    if merge_up_allowed && !out.merge_left {
        out.merge_up = decode_sao_merge_flag(engine, &mut ctx.sao_merge_flag[0])? != 0;
    }
    if out.merge_left || out.merge_up {
        return Ok(out);
    }

    let num_comp = if params.chroma_array_type != 0 { 3 } else { 1 };
    for c_idx in 0..num_comp {
        let read = (params.slice_sao_luma_flag && c_idx == 0)
            || (params.slice_sao_chroma_flag && c_idx > 0);
        if !read {
            continue;
        }
        // sao_type_idx_luma (cIdx 0) / sao_type_idx_chroma (cIdx 1) are
        // both read from the wire (sharing the Table 9-5 bank). For
        // cIdx == 2 the §7.3.8.3 syntax has no read — SaoTypeIdx[2] is
        // inferred equal to SaoTypeIdx[1] (§7.4.9.3).
        let type_idx = if c_idx < 2 {
            decode_sao_type_idx(engine, &mut ctx.sao_type_idx[0])?
        } else {
            out.components[1].sao_type_idx
        };
        out.components[c_idx as usize].sao_type_idx = type_idx;

        if type_idx != 0 {
            let bit_depth = if c_idx == 0 {
                params.bit_depth_luma
            } else {
                params.bit_depth_chroma
            };
            for i in 0..4 {
                out.components[c_idx as usize].offset_abs[i] =
                    decode_sao_offset_abs(engine, bit_depth)?;
            }
            if type_idx == 1 {
                // Band offset: per-i sign + band position.
                for i in 0..4 {
                    if out.components[c_idx as usize].offset_abs[i] != 0 {
                        out.components[c_idx as usize].offset_sign[i] =
                            decode_sao_offset_sign(engine)?;
                    }
                }
                out.components[c_idx as usize].band_position = decode_sao_band_position(engine)?;
            } else {
                // Edge offset: eo_class for cIdx 0 and 1 (cIdx 2 shares
                // cIdx 1's eo_class per §7.4.9.3).
                if c_idx == 0 || c_idx == 1 {
                    out.components[c_idx as usize].eo_class = decode_sao_eo_class(engine)?;
                } else {
                    out.components[2].eo_class = out.components[1].eo_class;
                }
            }
        }
    }
    // §7.4.9.3: SaoTypeIdx[2] / eo_class[2] inherit cIdx 1.
    if num_comp == 3 {
        out.components[2].sao_type_idx = out.components[1].sao_type_idx;
        if out.components[2].sao_type_idx == 2 {
            out.components[2].eo_class = out.components[1].eo_class;
        }
    }
    Ok(out)
}

/// Build the constant (non-geometry) part of a §7.3.8.10
/// `TransformUnitParams` template from the slice-data params + CU
/// context. The §7.3.8.8 transform-tree walk overwrites the per-node
/// geometry / cbf fields before each leaf.
fn tu_template(
    params: &SliceDataParams,
    cu_pred_mode: CuPredMode,
    cu_transquant_bypass_flag: bool,
    part_mode_2nx2n: bool,
) -> TransformUnitParams {
    TransformUnitParams {
        log2_trafo_size: 0,
        trafo_depth: 0,
        blk_idx: 0,
        cu_pred_mode: to_tu_pred_mode(cu_pred_mode),
        chroma_array_type: params.chroma_array_type,
        cbf_luma: false,
        cbf_cb: false,
        cbf_cb_lower: false,
        cbf_cr: false,
        cbf_cr_lower: false,
        intra_pred_mode_y: 0,
        intra_pred_mode_c: 0,
        intra_chroma_pred_mode: 0,
        cu_qp_delta_enabled_flag: params.cu_qp_delta_enabled_flag,
        cu_chroma_qp_offset_enabled_flag: params.cu_chroma_qp_offset_enabled_flag,
        chroma_qp_offset_list_len_minus1: params.chroma_qp_offset_list_len_minus1,
        cu_transquant_bypass_flag,
        sign_data_hiding_enabled_flag: params.sign_data_hiding_enabled_flag,
        cross_component_prediction_enabled_flag: params.cross_component_prediction_enabled_flag,
        residual_adaptive_colour_transform_enabled_flag: params
            .residual_adaptive_colour_transform_enabled_flag,
        transform_skip_enabled_flag: params.transform_skip_enabled_flag,
        log2_max_transform_skip_size: params.log2_max_transform_skip_size,
        implicit_rdpcm_enabled_flag: params.implicit_rdpcm_enabled_flag,
        explicit_rdpcm_enabled_flag: params.explicit_rdpcm_enabled_flag,
        transform_skip_context_enabled_flag: params.transform_skip_context_enabled_flag,
        persistent_rice_adaptation_enabled_flag: params.persistent_rice_adaptation_enabled_flag,
        cabac_bypass_alignment_enabled_flag: params.cabac_bypass_alignment_enabled_flag,
        extended_precision_processing_flag: params.extended_precision_processing_flag,
        bit_depth_luma: params.bit_depth_luma as u8,
        bit_depth_chroma: params.bit_depth_chroma as u8,
        part_mode_2nx2n,
        intra_chroma_pred_mode_corners: [0; 4],
    }
}

/// Decode one §7.3.8.6 `prediction_unit( x0, y0, nPbW, nPbH )`.
fn decode_prediction_unit(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    cu_skip_flag: bool,
    ct_depth: u32,
    n_pb_w: u32,
    n_pb_h: u32,
) -> Result<PredictionUnit, ResidualCodingError> {
    let mut pu = PredictionUnit {
        merge_flag: false,
        merge_idx: None,
        inter_pred_idc: None,
        ref_idx_l0: None,
        mvd_l0: None,
        mvp_l0_flag: None,
        ref_idx_l1: None,
        mvd_l1: None,
        mvp_l1_flag: None,
    };

    if cu_skip_flag {
        if params.max_num_merge_cand > 1 {
            pu.merge_idx = Some(decode_merge_idx(
                engine,
                &mut ctx.merge_idx[0],
                params.max_num_merge_cand,
            )?);
        } else {
            pu.merge_idx = Some(0);
        }
        return Ok(pu);
    }

    // MODE_INTER.
    pu.merge_flag = decode_merge_flag(engine, &mut ctx.merge_flag[0])? != 0;
    if pu.merge_flag {
        if params.max_num_merge_cand > 1 {
            pu.merge_idx = Some(decode_merge_idx(
                engine,
                &mut ctx.merge_idx[0],
                params.max_num_merge_cand,
            )?);
        } else {
            pu.merge_idx = Some(0);
        }
        return Ok(pu);
    }

    // Non-merge inter.
    let pred_idc = if params.slice_type_is_b {
        // inter_pred_idc bin 0 ctxInc = (nPbW+nPbH != 12) ? CtDepth : 4;
        // bin 1 ctxInc = 4. Bank slot layout: indices 0..=3 are the
        // CtDepth-keyed slots, index 4 is the shared bin-1 slot.
        let b0_slot = if n_pb_w + n_pb_h != 12 {
            ct_depth.min(3) as usize
        } else {
            4
        };
        // Borrow two distinct slots from the inter_pred_idc bank.
        let idc = decode_inter_pred_idc_banked(engine, ctx, b0_slot, n_pb_w, n_pb_h)?;
        pu.inter_pred_idc = Some(idc);
        idc
    } else {
        // P slice: a non-merge inter PB is PRED_L0 (§7.4.9.6) — no
        // inter_pred_idc on the wire.
        InterPredIdc::PredL0
    };

    if pred_idc != InterPredIdc::PredL1 {
        // L0 path.
        if params.num_ref_idx_l0_active_minus1 > 0 {
            pu.ref_idx_l0 = Some(decode_ref_idx_l0(engine, ctx, params)?);
        } else {
            pu.ref_idx_l0 = Some(0);
        }
        pu.mvd_l0 = Some(decode_mvd_pair_banked(engine, ctx)?);
        pu.mvp_l0_flag = Some(decode_mvp_flag(engine, &mut ctx.mvp_flag[0])?);
    }
    if pred_idc != InterPredIdc::PredL0 {
        // L1 path.
        if params.num_ref_idx_l1_active_minus1 > 0 {
            pu.ref_idx_l1 = Some(decode_ref_idx_l1(engine, ctx, params)?);
        } else {
            pu.ref_idx_l1 = Some(0);
        }
        if params.mvd_l1_zero_flag && pred_idc == InterPredIdc::PredBi {
            // MvdL1 inferred zero; mvd_coding not read.
            pu.mvd_l1 = None;
        } else {
            pu.mvd_l1 = Some(decode_mvd_pair_banked(engine, ctx)?);
        }
        pu.mvp_l1_flag = Some(decode_mvp_flag(engine, &mut ctx.mvp_flag[0])?);
    }

    Ok(pu)
}

/// Helper: decode `inter_pred_idc` borrowing two distinct slots from
/// the `inter_pred_idc[5]` bank (bin-0 slot `b0_slot`, bin-1 slot 4).
fn decode_inter_pred_idc_banked(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    b0_slot: usize,
    n_pb_w: u32,
    n_pb_h: u32,
) -> Result<InterPredIdc, ResidualCodingError> {
    // The bank is `[ContextModel; 5]`; bin-0 slot is in 0..=4 and bin-1
    // slot is fixed at 4. When they collide (b0_slot == 4, the
    // nPbW+nPbH==12 single-bin case) only bin 0 is read, so bin 1's
    // context is never dereferenced — a throwaway copy is harmless.
    if b0_slot == 4 {
        let mut dummy = ctx.inter_pred_idc[4];
        let r = decode_inter_pred_idc(
            engine,
            &mut ctx.inter_pred_idc[4],
            &mut dummy,
            n_pb_w,
            n_pb_h,
        )?;
        return Ok(r);
    }
    let (head, tail) = ctx.inter_pred_idc.split_at_mut(4);
    let b0 = &mut head[b0_slot];
    let b1 = &mut tail[0];
    Ok(decode_inter_pred_idc(engine, b0, b1, n_pb_w, n_pb_h)?)
}

/// Decode one `mvd_coding( )` invocation (both components in the
/// §7.3.8.9 interleaved bin order), borrowing the two
/// `abs_mvd_greater0_flag` / `abs_mvd_greater1_flag` contexts.
fn decode_mvd_pair_banked(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
) -> Result<[MvdComponent; 2], ResidualCodingError> {
    Ok(decode_mvd_pair(
        engine,
        &mut ctx.abs_mvd_greater0_flag[0],
        &mut ctx.abs_mvd_greater1_flag[0],
    )?)
}

fn decode_ref_idx_l0(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
) -> Result<u8, ResidualCodingError> {
    let (c0, c1) = ctx.ref_idx.split_at_mut(1);
    Ok(decode_ref_idx(
        engine,
        &mut c0[0],
        &mut c1[0],
        params.num_ref_idx_l0_active_minus1,
    )?)
}

fn decode_ref_idx_l1(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
) -> Result<u8, ResidualCodingError> {
    let (c0, c1) = ctx.ref_idx.split_at_mut(1);
    Ok(decode_ref_idx(
        engine,
        &mut c0[0],
        &mut c1[0],
        params.num_ref_idx_l1_active_minus1,
    )?)
}

/// Decode one §7.3.8.5 `coding_unit( x0, y0, log2CbSize )`.
#[allow(clippy::too_many_arguments)]
fn decode_coding_unit(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    state: &mut PictureParseState,
    qg: &mut QuantGroupState,
    x0: u32,
    y0: u32,
    log2_cb_size: u32,
    ct_depth: u32,
) -> Result<CodingUnit, ResidualCodingError> {
    let n_cb_s = 1u32 << log2_cb_size;

    let cu_transquant_bypass_flag = if params.transquant_bypass_enabled_flag {
        decode_cu_transquant_bypass_flag(engine, &mut ctx.cu_transquant_bypass_flag[0])? != 0
    } else {
        false
    };

    // cu_skip_flag (P/B only).
    let cu_skip_flag = if !params.slice_type_is_i {
        let (l_skip, l_avail) = state.neighbour_cu_skip(x0, y0, Neighbour::Left);
        let (a_skip, a_avail) = state.neighbour_cu_skip(x0, y0, Neighbour::Above);
        let inc = cu_skip_flag_ctx_inc(l_skip, l_avail, a_skip, a_avail) as usize;
        decode_cu_skip_flag(engine, &mut ctx.cu_skip_flag[inc])? != 0
    } else {
        false
    };
    state.record_cu_depth(x0, y0, log2_cb_size, ct_depth as u8, cu_skip_flag as u8);

    let mut cu = CodingUnit {
        x0,
        y0,
        log2_cb_size,
        cu_pred_mode: CuPredMode::Intra,
        cu_transquant_bypass_flag,
        part_mode: PartMode::Part2Nx2N,
        pcm_flag: false,
        pcm: None,
        palette: None,
        prediction_units: Vec::new(),
        intra_luma: Vec::new(),
        intra_chroma_pred_mode: Vec::new(),
        rqt_root_cbf: false,
        transform_tree: None,
    };

    if cu_skip_flag {
        // §7.4.9.5: MODE_SKIP. One prediction_unit covering the CU.
        cu.cu_pred_mode = cu_pred_mode_from_skip(params.slice_type_is_i, 1).unwrap();
        state.record_non_intra_cu(x0, y0, n_cb_s, cu.cu_pred_mode);
        let pu = decode_prediction_unit(engine, ctx, params, true, ct_depth, n_cb_s, n_cb_s)?;
        cu.prediction_units.push(pu);
        return Ok(cu);
    }

    // pred_mode_flag (P/B), else inferred MODE_INTRA on I slices.
    let cu_pred_mode = if params.slice_type_is_i {
        CuPredMode::Intra
    } else {
        let flag = decode_pred_mode_flag(engine, &mut ctx.pred_mode_flag[0])?;
        cu_pred_mode_from_flag(flag)
    };
    cu.cu_pred_mode = cu_pred_mode;

    // §7.3.8.5: palette_mode_flag, gated on the SCC enable, an intra
    // CU and log2CbSize <= MaxTbLog2SizeY; a palette CU replaces the
    // whole prediction + transform-coding tail of coding_unit( ).
    if params.palette_mode_enabled_flag
        && cu_pred_mode == CuPredMode::Intra
        && log2_cb_size <= params.max_tb_log2_size_y
        // palette_mode_flag: one Table 9-38 context-coded bin.
        && engine.decode_decision(&mut ctx.palette_mode_flag[0])? != 0
    {
        let pp = crate::hevc::engine::palette::PaletteParams {
            palette_max_size: params.palette_max_size,
            palette_max_predictor_size: params.palette_max_predictor_size,
            chroma_array_type: params.chroma_array_type,
            bit_depth_luma: params.bit_depth_luma,
            bit_depth_chroma: params.bit_depth_chroma,
            cu_transquant_bypass_flag: cu.cu_transquant_bypass_flag,
            cu_qp_delta_enabled_flag: params.cu_qp_delta_enabled_flag,
            cu_chroma_qp_offset_enabled_flag: params.cu_chroma_qp_offset_enabled_flag,
            chroma_qp_offset_list_len_minus1: params.chroma_qp_offset_list_len_minus1,
        };
        let pal = crate::hevc::engine::palette::decode_palette_coding(
            engine,
            ctx,
            &pp,
            qg,
            1usize << log2_cb_size,
        )?;
        // A palette CU carries no IntraPredModeY; record INTRA_DC for
        // the §8.4.2 neighbour derivation (the PCM convention — the
        // spec never derives a mode for palette blocks).
        state.record_pcm_cu(x0, y0, n_cb_s);
        cu.palette = Some(Box::new(pal));
        return Ok(cu);
    }

    // part_mode: present when MODE_INTER or log2CbSize == MinCbLog2SizeY.
    let part_present =
        cu_pred_mode != CuPredMode::Intra || log2_cb_size == params.min_cb_log2_size_y;
    let part_result: PartModeResult = if part_present {
        decode_part_mode_banked(engine, ctx, cu_pred_mode, log2_cb_size, params)?
    } else {
        crate::hevc::engine::binarization::part_mode_inferred()
    };
    cu.part_mode = part_result.part_mode;

    if cu_pred_mode == CuPredMode::Intra {
        decode_intra_cu(
            engine,
            ctx,
            params,
            state,
            &mut cu,
            x0,
            y0,
            log2_cb_size,
            part_result,
            qg,
        )?;
    } else {
        state.record_non_intra_cu(x0, y0, n_cb_s, cu_pred_mode);
        decode_inter_cu(
            engine,
            ctx,
            params,
            &mut cu,
            x0,
            y0,
            log2_cb_size,
            ct_depth,
            part_result,
            qg,
        )?;
    }

    Ok(cu)
}

/// Decode `part_mode` borrowing three distinct slots from the
/// `part_mode[4]` bank. Bin 0 → slot 0, bin 1 → slot 1, bin 2 → slot 2
/// (`log2CbSize == MinCbLog2SizeY`) or slot 3 (`>`).
fn decode_part_mode_banked(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    cu_pred_mode: CuPredMode,
    log2_cb_size: u32,
    params: &SliceDataParams,
) -> Result<PartModeResult, ResidualCodingError> {
    let bin2_slot = if log2_cb_size == params.min_cb_log2_size_y {
        2usize
    } else {
        3usize
    };
    // Borrow slot 0, 1 and bin2_slot (2 or 3) — all distinct.
    let (lo, hi) = ctx.part_mode.split_at_mut(2);
    let (c0, c1) = lo.split_at_mut(1);
    let bin2 = &mut hi[bin2_slot - 2];
    Ok(decode_part_mode(
        engine,
        &mut c0[0],
        &mut c1[0],
        bin2,
        cu_pred_mode,
        log2_cb_size,
        params.min_cb_log2_size_y,
        params.amp_enabled_flag,
    )?)
}

/// §7.3.8.7 PCM sample payload of one coding unit. Values are stored
/// already scaled to the picture bit depth (§8.4.1 equation 8-12 for
/// luma and its chroma analogue): `pcm_sample << (BitDepth −
/// PcmBitDepth)` — the reconstruction writes them into the picture
/// verbatim.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PcmSamples {
    /// `pcm_sample_luma[i]` in raster order, `nCbS * nCbS` values.
    pub luma: Vec<u16>,
    /// Cb samples in raster order, `(nCbS / SubWidthC) * (nCbS /
    /// SubHeightC)` values (empty when `ChromaArrayType == 0`).
    pub cb: Vec<u16>,
    /// Cr samples in raster order (same count as `cb`).
    pub cr: Vec<u16>,
}

/// §7.3.8.7 `pcm_sample( x0, y0, log2CbSize )` — the byte-aligned raw
/// sample payload of a `pcm_flag == 1` coding unit, plus the §9.3.1
/// engine re-initialization that follows it.
fn decode_pcm_sample(
    engine: &mut CabacEngine<'_>,
    params: &SliceDataParams,
    log2_cb_size: u32,
) -> Result<PcmSamples, ResidualCodingError> {
    // §9.3.4.3.5 leaves the raw bit position immediately after the
    // pcm_flag terminate bin; the pcm_alignment_zero_bit run pads it
    // to the byte boundary.
    engine.pcm_align()?;

    let n = 1usize << log2_cb_size;
    let shift_y = params
        .bit_depth_luma
        .saturating_sub(params.pcm_bit_depth_luma) as u16;
    let mut luma = Vec::with_capacity(n * n);
    for _ in 0..n * n {
        let v = engine.read_raw_bits(params.pcm_bit_depth_luma as u8)? as u16;
        luma.push(v << shift_y);
    }

    let (mut cb, mut cr) = (Vec::new(), Vec::new());
    if params.chroma_array_type != 0 {
        let (sub_w, sub_h) = crate::hevc::engine::picture::sub_wh_c(params.chroma_array_type);
        let count = (n / sub_w) * (n / sub_h);
        let shift_c = params
            .bit_depth_chroma
            .saturating_sub(params.pcm_bit_depth_chroma) as u16;
        // §7.4.9.7: the first half of pcm_sample_chroma is Cb, the
        // second half Cr, each in raster order.
        for plane in [&mut cb, &mut cr] {
            plane.reserve(count);
            for _ in 0..count {
                let v = engine.read_raw_bits(params.pcm_bit_depth_chroma as u8)? as u16;
                plane.push(v << shift_c);
            }
        }
    }

    // §9.3.1: the decoding engine is re-initialized (§9.3.2.6) after
    // the PCM data; context variables persist.
    engine.init_engine()?;
    Ok(PcmSamples { luma, cb, cr })
}

#[allow(clippy::too_many_arguments)]
fn decode_intra_cu(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    state: &mut PictureParseState,
    cu: &mut CodingUnit,
    x0: u32,
    y0: u32,
    log2_cb_size: u32,
    part_result: PartModeResult,
    qg: &mut QuantGroupState,
) -> Result<(), ResidualCodingError> {
    let n_cb_s = 1u32 << log2_cb_size;

    // PCM gate.
    let pcm_present = part_result.part_mode == PartMode::Part2Nx2N
        && params.pcm_enabled_flag
        && log2_cb_size >= params.log2_min_ipcm_cb_size_y
        && log2_cb_size <= params.log2_max_ipcm_cb_size_y;
    if pcm_present {
        cu.pcm_flag = decode_pcm_flag(engine)? != 0;
    }
    if cu.pcm_flag {
        // §7.3.8.7: pcm_alignment_zero_bit run, the raw u(v) sample
        // payload, then the §9.3.2.6 engine re-initialization (§9.3.1).
        state.record_pcm_cu(x0, y0, n_cb_s);
        cu.pcm = Some(decode_pcm_sample(engine, params, log2_cb_size)?);
        return Ok(());
    }

    // Luma intra mode signalling group.
    let pb_offset = if part_result.part_mode == PartMode::PartNxN {
        n_cb_s / 2
    } else {
        n_cb_s
    };
    let n_pb = (n_cb_s / pb_offset) as usize; // 1 or 2 per axis
    let count = n_pb * n_pb;

    let mut prev_flags = Vec::with_capacity(count);
    for _ in 0..count {
        let f =
            decode_prev_intra_luma_pred_flag(engine, &mut ctx.prev_intra_luma_pred_flag[0])? != 0;
        prev_flags.push(f);
    }
    for &f in &prev_flags {
        let mut entry = IntraLumaMode {
            prev_intra_luma_pred_flag: f,
            mpm_idx: None,
            rem_intra_luma_pred_mode: None,
        };
        if f {
            entry.mpm_idx = Some(decode_mpm_idx(engine)?);
        } else {
            entry.rem_intra_luma_pred_mode = Some(decode_rem_intra_luma_pred_mode(engine)?);
        }
        cu.intra_luma.push(entry);
    }

    // Chroma intra mode.
    if params.chroma_array_type == 3 {
        for _ in 0..count {
            cu.intra_chroma_pred_mode
                .push(decode_intra_chroma_pred_mode(
                    engine,
                    &mut ctx.intra_chroma_pred_mode[0],
                )?);
        }
    } else if params.chroma_array_type != 0 {
        cu.intra_chroma_pred_mode
            .push(decode_intra_chroma_pred_mode(
                engine,
                &mut ctx.intra_chroma_pred_mode[0],
            )?);
    }

    // §8.4.2 / §8.4.3 — derive the actual `IntraPredModeY` per luma
    // prediction block (recording each into the picture-level mode
    // field so later blocks' MPM derivation sees it) and the derived
    // `IntraPredModeC`: the §7.4.9.11 mode-dependent scan of the
    // `residual_coding( )` invocations below reads them.
    let pb_offsets: [(u32, u32); 4] = [
        (0, 0),
        (pb_offset, 0),
        (0, pb_offset),
        (pb_offset, pb_offset),
    ];
    let mut modes_y = [0u32; 4];
    for (i, luma) in cu.intra_luma.iter().enumerate() {
        let (dx, dy) = pb_offsets[i];
        modes_y[i] =
            u32::from(state.derive_and_record_luma_mode(x0 + dx, y0 + dy, pb_offset, luma));
    }
    for i in cu.intra_luma.len()..4 {
        modes_y[i] = modes_y[0];
    }
    let mut raw_chroma = [0u8; 4];
    let mut modes_c = [0u32; 4];
    for i in 0..4 {
        let raw = if cu.intra_chroma_pred_mode.len() == 4 {
            cu.intra_chroma_pred_mode[i]
        } else {
            cu.intra_chroma_pred_mode.first().copied().unwrap_or(0)
        };
        raw_chroma[i] = raw;
        // §8.4.3: IntraPredModeC derives from the co-located luma mode —
        // per PB for the ChromaArrayType == 3 PART_NxN case, else from
        // the CU's first PB.
        let luma_for_c = if cu.intra_chroma_pred_mode.len() == 4 {
            modes_y[i]
        } else {
            modes_y[0]
        };
        modes_c[i] = u32::from(derive_intra_pred_mode_c(
            raw,
            luma_for_c as u8,
            params.chroma_array_type == 2,
        ));
    }

    // Intra CUs always enter the transform tree (rqt_root_cbf is not
    // coded; cbf_luma presence at the root is unconditional).
    let max_trafo_depth =
        params.max_transform_hierarchy_depth_intra + part_result.intra_split_flag as u32;
    let mut template = tu_template(
        params,
        CuPredMode::Intra,
        cu.cu_transquant_bypass_flag,
        part_result.part_mode == PartMode::Part2Nx2N,
    );
    template.intra_pred_mode_y = modes_y[0];
    template.intra_pred_mode_c = modes_c[0];
    template.intra_chroma_pred_mode = raw_chroma[0];
    template.intra_chroma_pred_mode_corners = raw_chroma;
    let tt_params = TransformTreeParams {
        max_tb_log2_size_y: params.max_tb_log2_size_y,
        min_tb_log2_size_y: params.min_tb_log2_size_y,
        max_trafo_depth,
        intra_split_flag: part_result.intra_split_flag,
        inter_split_flag: false,
        cu_pred_mode: TuCuPredMode::Intra,
        chroma_array_type: params.chroma_array_type,
        tu_template: template,
        cu_x0: x0,
        cu_y0: y0,
        log2_cb_size,
        intra_pred_mode_y_corners: modes_y,
        intra_pred_mode_c_corners: modes_c,
    };
    let tree = decode_transform_tree(
        engine,
        ctx,
        &tt_params,
        qg,
        x0,
        y0,
        x0,
        y0,
        log2_cb_size,
        0,
        0,
        false,
        false,
        false,
        false,
    )?;
    cu.transform_tree = Some(tree);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_inter_cu(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    cu: &mut CodingUnit,
    x0: u32,
    y0: u32,
    log2_cb_size: u32,
    ct_depth: u32,
    part_result: PartModeResult,
    qg: &mut QuantGroupState,
) -> Result<(), ResidualCodingError> {
    let n = 1u32 << log2_cb_size;
    let pm = part_result.part_mode;

    // §7.3.8.5: emit the prediction_unit calls per PartMode.
    let pu_rects: Vec<(u32, u32, u32, u32)> = match pm {
        PartMode::Part2Nx2N => vec![(x0, y0, n, n)],
        PartMode::Part2NxN => vec![(x0, y0, n, n / 2), (x0, y0 + n / 2, n, n / 2)],
        PartMode::PartNx2N => vec![(x0, y0, n / 2, n), (x0 + n / 2, y0, n / 2, n)],
        PartMode::Part2NxnU => vec![(x0, y0, n, n / 4), (x0, y0 + n / 4, n, n * 3 / 4)],
        PartMode::Part2NxnD => vec![(x0, y0, n, n * 3 / 4), (x0, y0 + n * 3 / 4, n, n / 4)],
        PartMode::PartNLx2N => vec![(x0, y0, n / 4, n), (x0 + n / 4, y0, n * 3 / 4, n)],
        PartMode::PartNRx2N => vec![(x0, y0, n * 3 / 4, n), (x0 + n * 3 / 4, y0, n / 4, n)],
        PartMode::PartNxN => vec![
            (x0, y0, n / 2, n / 2),
            (x0 + n / 2, y0, n / 2, n / 2),
            (x0, y0 + n / 2, n / 2, n / 2),
            (x0 + n / 2, y0 + n / 2, n / 2, n / 2),
        ],
    };
    for (px, py, pw, ph) in pu_rects {
        let _ = (px, py);
        let pu = decode_prediction_unit(engine, ctx, params, false, ct_depth, pw, ph)?;
        cu.prediction_units.push(pu);
    }

    // rqt_root_cbf: present unless PART_2Nx2N + merge.
    let single_merge = pm == PartMode::Part2Nx2N
        && cu
            .prediction_units
            .first()
            .map(|p| p.merge_flag)
            .unwrap_or(false);
    let rqt_root_cbf = if !single_merge {
        decode_rqt_root_cbf(engine, &mut ctx.rqt_root_cbf[0])? != 0
    } else {
        // §7.4.9.5: not present ⇒ inferred 1.
        true
    };
    cu.rqt_root_cbf = rqt_root_cbf;

    if rqt_root_cbf {
        let inter_split =
            params.max_transform_hierarchy_depth_inter == 0 && pm != PartMode::Part2Nx2N;
        let max_trafo_depth = params.max_transform_hierarchy_depth_inter;
        let tt_params = TransformTreeParams {
            max_tb_log2_size_y: params.max_tb_log2_size_y,
            min_tb_log2_size_y: params.min_tb_log2_size_y,
            max_trafo_depth,
            intra_split_flag: false,
            inter_split_flag: inter_split,
            cu_pred_mode: TuCuPredMode::Inter,
            chroma_array_type: params.chroma_array_type,
            tu_template: tu_template(
                params,
                CuPredMode::Inter,
                cu.cu_transquant_bypass_flag,
                pm == PartMode::Part2Nx2N,
            ),
            cu_x0: x0,
            cu_y0: y0,
            log2_cb_size,
            // Inter CUs never take the §7.4.9.11 intra-scan branch.
            intra_pred_mode_y_corners: [0; 4],
            intra_pred_mode_c_corners: [0; 4],
        };
        let tree = decode_transform_tree(
            engine,
            ctx,
            &tt_params,
            qg,
            x0,
            y0,
            x0,
            y0,
            log2_cb_size,
            0,
            0,
            false,
            false,
            false,
            false,
        )?;
        cu.transform_tree = Some(tree);
    }
    Ok(())
}

/// Decode one §7.3.8.4 `coding_quadtree( x0, y0, log2CbSize, cqtDepth )`.
#[allow(clippy::too_many_arguments)]
pub fn decode_coding_quadtree(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    state: &mut PictureParseState,
    qg: &mut QuantGroupState,
    x0: u32,
    y0: u32,
    log2_cb_size: u32,
    cqt_depth: u32,
) -> Result<CodingQuadtree, ResidualCodingError> {
    let size = 1u32 << log2_cb_size;
    let fits_w = x0 + size <= params.pic_width_in_luma_samples;
    let fits_h = y0 + size <= params.pic_height_in_luma_samples;

    // split_cu_flag presence gate (§7.3.8.4).
    let split_present = fits_w && fits_h && log2_cb_size > params.min_cb_log2_size_y;
    let split = if split_present {
        let (l_depth, l_avail) = state.neighbour_ct_depth(x0, y0, Neighbour::Left);
        let (a_depth, a_avail) = state.neighbour_ct_depth(x0, y0, Neighbour::Above);
        let inc = split_cu_flag_ctx_inc(l_depth, l_avail, a_depth, a_avail, cqt_depth) as usize;
        decode_split_cu_flag(engine, &mut ctx.split_cu_flag[inc])? != 0
    } else {
        // §7.4.9.4 inference: 1 when the block extends past the picture
        // boundary OR log2CbSize > MinCbLog2SizeY, else 0.
        (!fits_w || !fits_h) || log2_cb_size > params.min_cb_log2_size_y
    };

    // §6.5.1 quantization-group resets at the QG threshold.
    if params.cu_qp_delta_enabled_flag && log2_cb_size >= params.log2_min_cu_qp_delta_size {
        qg.is_cu_qp_delta_coded = false;
        qg.cu_qp_delta_val = 0;
    }
    if params.cu_chroma_qp_offset_enabled_flag
        && log2_cb_size >= params.log2_min_cu_chroma_qp_offset_size
    {
        qg.is_cu_chroma_qp_offset_coded = false;
    }

    if split {
        let half = 1u32 << (log2_cb_size - 1);
        let x1 = x0 + half;
        let y1 = y0 + half;
        let child_log2 = log2_cb_size - 1;
        let child_depth = cqt_depth + 1;
        let mut children = Vec::with_capacity(4);
        // First child always present.
        children.push(decode_coding_quadtree(
            engine,
            ctx,
            params,
            state,
            qg,
            x0,
            y0,
            child_log2,
            child_depth,
        )?);
        if x1 < params.pic_width_in_luma_samples {
            children.push(decode_coding_quadtree(
                engine,
                ctx,
                params,
                state,
                qg,
                x1,
                y0,
                child_log2,
                child_depth,
            )?);
        }
        if y1 < params.pic_height_in_luma_samples {
            children.push(decode_coding_quadtree(
                engine,
                ctx,
                params,
                state,
                qg,
                x0,
                y1,
                child_log2,
                child_depth,
            )?);
        }
        if x1 < params.pic_width_in_luma_samples && y1 < params.pic_height_in_luma_samples {
            children.push(decode_coding_quadtree(
                engine,
                ctx,
                params,
                state,
                qg,
                x1,
                y1,
                child_log2,
                child_depth,
            )?);
        }
        Ok(CodingQuadtree::Split(children))
    } else {
        let cu = decode_coding_unit(
            engine,
            ctx,
            params,
            state,
            qg,
            x0,
            y0,
            log2_cb_size,
            cqt_depth,
        )?;
        Ok(CodingQuadtree::Leaf(Box::new(cu)))
    }
}

/// Decode one §7.3.8.2 `coding_tree_unit( )` rooted at CTB top-left
/// `(x_ctb, y_ctb)`.
///
/// `sao_merge_left_allowed` / `sao_merge_up_allowed` are the §7.3.8.3
/// merge-flag presence conditions (the slice-segment / tile boundary
/// tests), evaluated by the caller.
#[allow(clippy::too_many_arguments)]
pub fn decode_coding_tree_unit(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    x_ctb: u32,
    y_ctb: u32,
    sao_merge_left_allowed: bool,
    sao_merge_up_allowed: bool,
) -> Result<CodingTreeUnit, ResidualCodingError> {
    // Standalone entry point: a fresh per-picture parse state (no
    // cross-CTU intra-mode memory). Multi-CTU pictures must use
    // [`decode_coding_tree_unit_in_picture`] with a shared
    // [`PictureParseState`] so the §8.4.2 MPM derivation sees the true
    // cross-CTU neighbour modes.
    let mut state = PictureParseState::new(params);
    decode_coding_tree_unit_in_picture(
        engine,
        ctx,
        params,
        &mut state,
        x_ctb,
        y_ctb,
        0,
        0,
        sao_merge_left_allowed,
        sao_merge_up_allowed,
    )
}

/// Decode one §7.3.8.2 `coding_tree_unit( )` with the shared per-picture
/// parse state (the §8.4.2 intra-mode neighbour field + the per-CTB
/// slice / tile availability grids). `slice_addr_rs` is the CTB's
/// `SliceAddrRs`; `tile_id` its §6.5.1 `TileId`.
#[allow(clippy::too_many_arguments)]
pub fn decode_coding_tree_unit_in_picture(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &SliceDataParams,
    state: &mut PictureParseState,
    x_ctb: u32,
    y_ctb: u32,
    slice_addr_rs: u32,
    tile_id: u32,
    sao_merge_left_allowed: bool,
    sao_merge_up_allowed: bool,
) -> Result<CodingTreeUnit, ResidualCodingError> {
    // Issue #189 stage attribution: one scope per CTU covers the whole
    // §7.3.8 syntax walk. The §7.3.8.11 residual scope nests inside it, so
    // what is left here is the non-coefficient CABAC decode.
    let _profile =
        crate::hevc::engine::profile::scope(crate::hevc::engine::profile::Stage::SliceData);
    let mut qg = QuantGroupState::default();
    state.begin_ctu(x_ctb, y_ctb, slice_addr_rs, tile_id);

    let sao = if params.slice_sao_luma_flag || params.slice_sao_chroma_flag {
        Some(decode_sao(
            engine,
            ctx,
            params,
            sao_merge_left_allowed,
            sao_merge_up_allowed,
        )?)
    } else {
        None
    };

    let quadtree = decode_coding_quadtree(
        engine,
        ctx,
        params,
        state,
        &mut qg,
        x_ctb,
        y_ctb,
        params.ctb_log2_size_y,
        0,
    )?;

    Ok(CodingTreeUnit { sao, quadtree })
}

/// Decode the §7.3.8.1 `end_of_slice_segment_flag` that follows each
/// CTU. Re-exported here for the slice-data loop convenience; it is the
/// §9.3.4.3.5 terminate path.
pub fn end_of_slice_segment_flag(
    engine: &mut CabacEngine<'_>,
) -> Result<bool, ResidualCodingError> {
    Ok(decode_end_of_slice_segment_flag(engine)? != 0)
}
