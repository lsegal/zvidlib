//! §7.3.8.8 `transform_tree( x0, y0, xBase, yBase, log2TrafoSize,
//! trafoDepth, blkIdx )` recursion.
//!
//! This module drives the residual-quadtree split that the §7.3.8.5
//! `coding_unit( )` body enters (under `rqt_root_cbf`) and that bottoms
//! out in the §7.3.8.10 `transform_unit( )` leaf already implemented in
//! [`crate::hevc::engine::transform_unit`]. It is the missing rung between the
//! coding-unit walk and the transform-unit leaf: it reads
//! `split_transform_flag`, the per-node `cbf_cb` / `cbf_cr` chroma
//! coded-block flags, recurses into four quarter-size nodes when split,
//! and at each leaf reads `cbf_luma` then invokes
//! [`crate::hevc::engine::transform_unit::decode_transform_unit`].
//!
//! The driver mirrors the §7.3.8.8 syntax table exactly, including the
//! inheritance gate (`trafoDepth == 0 || cbf_cb[xBase][yBase][trafoDepth
//! − 1]`) under which a node only re-reads a chroma cbf when its parent
//! had a non-zero one, the §7.4.9.8 inference rules for the flags that
//! are not present on the wire, the `ChromaArrayType == 2` lower-half
//! companion cbf reads, and the §7.3.8.8 `cbf_luma` presence condition
//! (only read at a leaf when the CU is intra, the depth is non-zero, or
//! a chroma cbf is set).
//!
//! The §6.5.1 quantization-group reset (`IsCuQpDeltaCoded`,
//! `CuQpDeltaVal`, `IsCuChromaQpOffsetCoded`) at the start of each
//! quantization group is the §7.3.8.4 `coding_quadtree( )` caller's
//! responsibility; this module threads one [`QuantGroupState`] through
//! the whole tree (each `transform_unit( )` leaf reads / updates it in
//! place). The §8.6 dequantization / inverse transform that turns the
//! decoded coefficient blocks into sample residuals likewise remains the
//! caller's responsibility.

use crate::hevc::engine::binarization::{
    cbf_cb_ctx_inc, cbf_chroma_inferred, cbf_cr_ctx_inc, cbf_luma_ctx_inc, cbf_luma_inferred,
    decode_cbf_cb, decode_cbf_cr, decode_cbf_luma, decode_split_transform_flag,
    split_transform_flag_ctx_inc, split_transform_flag_inferred,
};
use crate::hevc::engine::cabac::CabacEngine;
use crate::hevc::engine::ctx_init::SliceContexts;
use crate::hevc::engine::residual::ResidualCodingError;
use crate::hevc::engine::transform_unit::{
    CuPredMode, QuantGroupState, TransformUnit, TransformUnitParams, decode_transform_unit,
};

/// Geometry + coding-unit context shared by every node of one
/// §7.3.8.8 `transform_tree( )` recursion. Every field is constant for
/// the duration of the tree (it derives from the §7.3.8.5
/// `coding_unit( )` the tree roots in) — only the per-node
/// `(x0, y0, log2TrafoSize, trafoDepth, blkIdx)` and the inherited
/// chroma-cbf state vary as the recursion descends.
#[derive(Debug, Clone, Copy)]
pub struct TransformTreeParams {
    /// `MaxTbLog2SizeY` (§7.4.3.2) — the maximum luma transform-block
    /// size; bounds the §7.3.8.8 split presence gate and the §7.4.9.8
    /// "log2TrafoSize > MaxTbLog2SizeY ⇒ forced split" inference.
    pub max_tb_log2_size_y: u32,
    /// `MinTbLog2SizeY` (§7.4.3.2) — the minimum luma transform-block
    /// size; bounds the split presence gate.
    pub min_tb_log2_size_y: u32,
    /// `MaxTrafoDepth` (§7.3.8.5) — the maximum residual-quadtree
    /// depth for this coding unit (`max_transform_hierarchy_depth_intra
    /// + IntraSplitFlag` for intra, `max_transform_hierarchy_depth_inter`
    ///   for inter).
    pub max_trafo_depth: u32,
    /// `IntraSplitFlag` (§7.4.9.5) — set when an intra CU is split into
    /// `PART_NxN` partitions; forces a split at `trafoDepth == 0`.
    pub intra_split_flag: bool,
    /// `interSplitFlag` (§7.4.9.8) — `1` exactly when
    /// `max_transform_hierarchy_depth_inter == 0 && CuPredMode ==
    /// MODE_INTER && PartMode != PART_2Nx2N && trafoDepth == 0`. The
    /// caller derives it once for the CU (it can only be non-zero at
    /// `trafoDepth == 0`).
    pub inter_split_flag: bool,
    /// `CuPredMode[ x0 ][ y0 ]` of the coding unit.
    pub cu_pred_mode: CuPredMode,
    /// `ChromaArrayType` (0 = monochrome, 1 = 4:2:0, 2 = 4:2:2,
    /// 3 = 4:4:4).
    pub chroma_array_type: u8,
    /// The non-geometric §7.3.8.10 transform-unit context that is
    /// constant across the tree (prediction modes, PPS / SPS gates,
    /// QP-offset list length, …). The per-node geometry fields
    /// (`log2_trafo_size`, `trafo_depth`, `blk_idx`, the cbf flags) are
    /// overwritten by the recursion before each leaf invocation.
    pub tu_template: TransformUnitParams,
    /// The coding unit's luma top-left `x` (constant across the tree) —
    /// with [`Self::log2_cb_size`], locates each leaf's prediction block
    /// for the per-PB intra-mode selection.
    pub cu_x0: u32,
    /// The coding unit's luma top-left `y`.
    pub cu_y0: u32,
    /// `log2CbSize` of the coding unit.
    pub log2_cb_size: u32,
    /// The §8.4.2-derived `IntraPredModeY` of the CU's prediction
    /// blocks, `[tl, tr, bl, br]`. For `PART_2Nx2N` all four entries
    /// carry the single PB's mode. Consumed by the §7.4.9.11
    /// mode-dependent scan selection of each leaf's `residual_coding()`.
    pub intra_pred_mode_y_corners: [u32; 4],
    /// The §8.4.3-derived `IntraPredModeC` per prediction block,
    /// `[tl, tr, bl, br]` (all equal outside `ChromaArrayType == 3`
    /// `PART_NxN`).
    pub intra_pred_mode_c_corners: [u32; 4],
}

/// One node of the decoded §7.3.8.8 transform tree: either an internal
/// split node carrying its four children, or a leaf carrying the
/// decoded [`TransformUnit`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformTree {
    /// A `split_transform_flag == 1` internal node. The chroma cbf
    /// flags read at this node (before the split) are recorded so the
    /// §7.3.8.8 inheritance gate and the §7.3.8.10 deferred-chroma path
    /// can be reasoned about; the four children are in raster order
    /// (`[tl, tr, bl, br]`).
    Split {
        /// `cbf_cb[ x0 ][ y0 ][ trafoDepth ]` read at this node, or the
        /// §7.4.9.8 inferred value when not present.
        cbf_cb: bool,
        /// The `ChromaArrayType == 2` lower-half companion
        /// `cbf_cb[ x0 ][ y0 + (1 << (log2TrafoSize − 1)) ][ trafoDepth ]`.
        cbf_cb_lower: bool,
        /// `cbf_cr[ x0 ][ y0 ][ trafoDepth ]`.
        cbf_cr: bool,
        /// The `ChromaArrayType == 2` lower-half companion `cbf_cr`.
        cbf_cr_lower: bool,
        /// The four quarter-size children, raster order `[tl, tr, bl, br]`.
        children: Box<[TransformTree; 4]>,
    },
    /// A `split_transform_flag == 0` leaf node, carrying the decoded
    /// §7.3.8.10 transform unit and the `cbf_luma` that gated it.
    Leaf {
        /// `cbf_luma[ x0 ][ y0 ][ trafoDepth ]` (read or inferred to 1).
        cbf_luma: bool,
        /// The decoded transform unit.
        unit: TransformUnit,
    },
}

/// Decode one §7.3.8.8 `transform_tree( )` node (and, recursively, its
/// whole subtree) from the CABAC engine.
///
/// * `x0` / `y0` — the luma top-left sample position of this node.
/// * `x_base` / `y_base` — the parent node's top-left position (equal to
///   `x0` / `y0` at the root).
/// * `log2_trafo_size` — the luma transform-block log2 size of this node.
/// * `trafo_depth` — the recursion depth (0 at the CU root).
/// * `blk_idx` — the sub-block index within the parent (0 at the root).
/// * `parent_cbf_cb` / `parent_cbf_cr` — the parent node's
///   `cbf_cb[ xBase ][ yBase ][ trafoDepth − 1 ]` /
///   `cbf_cr[ … ]`; at the root (`trafoDepth == 0`) they are ignored
///   because the §7.3.8.8 inheritance gate reads the cbf unconditionally.
///
/// The function threads one [`QuantGroupState`] through the subtree so
/// the once-per-quantization-group `delta_qp()` / `chroma_qp_offset()`
/// gates fire exactly once.
#[allow(clippy::too_many_arguments)]
pub fn decode_transform_tree(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &TransformTreeParams,
    qg: &mut QuantGroupState,
    x0: u32,
    y0: u32,
    // xBase / yBase identify the parent node's top-left position. The
    // §7.3.8.10 deferred-chroma leaf reads chroma at (xBase, yBase), but
    // that positional lookup is carried into the [`crate::hevc::engine::transform_unit`]
    // leaf via the inherited `parent_cbf_cb` / `parent_cbf_cr` cbf state
    // rather than raw coordinates, so these are retained for §7.3.8.8
    // signature fidelity and bounds reasoning only.
    _x_base: u32,
    _y_base: u32,
    log2_trafo_size: u32,
    trafo_depth: u32,
    blk_idx: u32,
    parent_cbf_cb: bool,
    parent_cbf_cr: bool,
    // The `ChromaArrayType == 2` lower-half companions of the parent's
    // cbf flags — a `log2TrafoSize == 3` split parent reads both halves
    // and its 4x4 leaves code the parent chroma at `blkIdx == 3` gated
    // per half (§7.3.8.10 `cbf_cb[ xBase ][ yBase + ( tIdx <<
    // log2TrafoSizeC ) ][ trafoDepth ]`).
    parent_cbf_cb_lower: bool,
    parent_cbf_cr_lower: bool,
) -> Result<TransformTree, ResidualCodingError> {
    // §7.3.8.8: split_transform_flag is read only inside the gate
    //   log2TrafoSize <= MaxTbLog2SizeY && log2TrafoSize > MinTbLog2SizeY
    //   && trafoDepth < MaxTrafoDepth && !( IntraSplitFlag && trafoDepth == 0 )
    // otherwise it is inferred (§7.4.9.8).
    let split = if log2_trafo_size <= params.max_tb_log2_size_y
        && log2_trafo_size > params.min_tb_log2_size_y
        && trafo_depth < params.max_trafo_depth
        && !(params.intra_split_flag && trafo_depth == 0)
    {
        let inc = split_transform_flag_ctx_inc(log2_trafo_size) as usize;
        decode_split_transform_flag(engine, &mut ctx.split_transform_flag[inc])? != 0
    } else {
        split_transform_flag_inferred(
            log2_trafo_size,
            params.max_tb_log2_size_y,
            params.intra_split_flag,
            trafo_depth,
            // interSplitFlag is only non-zero at trafoDepth == 0 by its
            // own definition; the caller's value already encodes that.
            params.inter_split_flag && trafo_depth == 0,
        ) != 0
    };

    // §7.3.8.8 chroma-cbf block, gated by
    //   ( log2TrafoSize > 2 && ChromaArrayType != 0 ) || ChromaArrayType == 3
    let chroma_present =
        (log2_trafo_size > 2 && params.chroma_array_type != 0) || params.chroma_array_type == 3;

    // The lower-half companion is read only for ChromaArrayType == 2 when
    // ( !split_transform_flag || log2TrafoSize == 3 ).
    let read_lower = params.chroma_array_type == 2 && (!split || log2_trafo_size == 3);

    let mut cbf_cb_lower = false;
    let mut cbf_cr_lower = false;
    let cbf_cb;
    let cbf_cr;

    if chroma_present {
        // cbf_cb: read when trafoDepth == 0 || cbf_cb[xBase][yBase][td-1].
        if trafo_depth == 0 || parent_cbf_cb {
            let inc = cbf_cb_ctx_inc(trafo_depth) as usize;
            cbf_cb = decode_cbf_cb(engine, &mut ctx.cbf_chroma[inc])? != 0;
            if read_lower {
                let inc = cbf_cb_ctx_inc(trafo_depth) as usize;
                cbf_cb_lower = decode_cbf_cb(engine, &mut ctx.cbf_chroma[inc])? != 0;
            }
        } else {
            cbf_cb = cbf_chroma_inferred() != 0;
        }
        // cbf_cr: symmetric.
        if trafo_depth == 0 || parent_cbf_cr {
            let inc = cbf_cr_ctx_inc(trafo_depth) as usize;
            cbf_cr = decode_cbf_cr(engine, &mut ctx.cbf_chroma[inc])? != 0;
            if read_lower {
                let inc = cbf_cr_ctx_inc(trafo_depth) as usize;
                cbf_cr_lower = decode_cbf_cr(engine, &mut ctx.cbf_chroma[inc])? != 0;
            }
        } else {
            cbf_cr = cbf_chroma_inferred() != 0;
        }
    } else {
        // §7.4.9.8: not-present chroma cbf inherits the parent value at
        // deeper depths (the §7.3.8.10 transform_unit reads
        // cbf_cb[xC][yC][cbfDepthC] with cbfDepthC = trafoDepth − 1 for
        // the log2TrafoSize == 2 leaf), so a 4×4 luma leaf carries the
        // parent's chroma cbf forward — BOTH halves for
        // ChromaArrayType == 2.
        cbf_cb = parent_cbf_cb;
        cbf_cr = parent_cbf_cr;
        cbf_cb_lower = parent_cbf_cb_lower;
        cbf_cr_lower = parent_cbf_cr_lower;
    }

    if split {
        // Recurse into four quarter-size nodes (§7.3.8.8).
        let half = 1u32 << (log2_trafo_size - 1);
        let x1 = x0 + half;
        let y1 = y0 + half;
        let child_log2 = log2_trafo_size - 1;
        let child_depth = trafo_depth + 1;
        // Each child's xBase/yBase is this node's (x0, y0); the inherited
        // parent cbf is this node's (cbf_cb, cbf_cr).
        let mut decode_child = |xc: u32, yc: u32, idx: u32| {
            decode_transform_tree(
                engine,
                ctx,
                params,
                qg,
                xc,
                yc,
                x0,
                y0,
                child_log2,
                child_depth,
                idx,
                cbf_cb,
                cbf_cr,
                cbf_cb_lower,
                cbf_cr_lower,
            )
        };
        let tl = decode_child(x0, y0, 0)?;
        let tr = decode_child(x1, y0, 1)?;
        let bl = decode_child(x0, y1, 2)?;
        let br = decode_child(x1, y1, 3)?;
        Ok(TransformTree::Split {
            cbf_cb,
            cbf_cb_lower,
            cbf_cr,
            cbf_cr_lower,
            children: Box::new([tl, tr, bl, br]),
        })
    } else {
        // Leaf: §7.3.8.8 cbf_luma presence condition.
        //   CuPredMode == MODE_INTRA || trafoDepth != 0 || cbf_cb || cbf_cr
        //   || ( ChromaArrayType == 2 && ( cbf_cb_lower || cbf_cr_lower ) )
        let cbf_luma_present = params.cu_pred_mode == CuPredMode::Intra
            || trafo_depth != 0
            || cbf_cb
            || cbf_cr
            || (params.chroma_array_type == 2 && (cbf_cb_lower || cbf_cr_lower));
        let cbf_luma = if cbf_luma_present {
            let inc = cbf_luma_ctx_inc(trafo_depth) as usize;
            decode_cbf_luma(engine, &mut ctx.cbf_luma[inc])? != 0
        } else {
            cbf_luma_inferred() != 0
        };

        // Assemble the §7.3.8.10 transform_unit params for this leaf:
        // start from the template and overwrite the per-node geometry +
        // cbf state.
        let mut tu_params = params.tu_template;
        tu_params.log2_trafo_size = log2_trafo_size;
        tu_params.trafo_depth = trafo_depth;
        tu_params.blk_idx = blk_idx;
        tu_params.cu_pred_mode = params.cu_pred_mode;
        tu_params.chroma_array_type = params.chroma_array_type;
        tu_params.cbf_luma = cbf_luma;
        tu_params.cbf_cb = cbf_cb;
        tu_params.cbf_cb_lower = cbf_cb_lower;
        tu_params.cbf_cr = cbf_cr;
        tu_params.cbf_cr_lower = cbf_cr_lower;

        // §7.4.9.11 — the mode-dependent scan reads the prediction
        // block THIS leaf lies in: with `IntraSplitFlag` the CU has four
        // PBs at the quarter positions, otherwise one.
        let pb_idx = if params.intra_split_flag && params.log2_cb_size > 0 {
            let half = 1u32 << (params.log2_cb_size - 1);
            (usize::from(y0.wrapping_sub(params.cu_y0) >= half) << 1)
                | usize::from(x0.wrapping_sub(params.cu_x0) >= half)
        } else {
            0
        };
        tu_params.intra_pred_mode_y = params.intra_pred_mode_y_corners[pb_idx];
        tu_params.intra_pred_mode_c = params.intra_pred_mode_c_corners[pb_idx];
        tu_params.intra_chroma_pred_mode =
            params.tu_template.intra_chroma_pred_mode_corners[pb_idx];

        let unit = decode_transform_unit(engine, ctx, &tu_params, qg)?;
        Ok(TransformTree::Leaf { cbf_luma, unit })
    }
}
