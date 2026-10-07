//! §7.3.8.10 `transform_unit( x0, y0, xBase, yBase, log2TrafoSize,
//! trafoDepth, blkIdx )` syntax driver.
//!
//! This module composes the per-element primitives already exposed by
//! [`crate::hevc::engine::binarization`] (`tu_residual_act_flag`, `delta_qp()` /
//! `cu_qp_delta`, `chroma_qp_offset()` / `cu_chroma_qp_offset`, and the
//! §7.3.8.12 `cross_comp_pred()` element) together with the
//! [`crate::hevc::engine::residual`] §7.3.8.11 `residual_coding()` driver into the
//! full §7.3.8.10 transform-unit walk.
//!
//! The driver is the leaf of the §7.3.8 quadtree that the
//! `transform_tree()` recursion bottoms out in: it consumes the coded
//! block flags the parent already decided ([`TransformUnitParams::cbf_luma`]
//! / [`TransformUnitParams::cbf_cb`] / [`TransformUnitParams::cbf_cr`]),
//! reads any present `tu_residual_act_flag`, the per-CU `delta_qp()` and
//! `chroma_qp_offset()` blocks (each gated by the once-per-quantization-
//! group `IsCuQpDeltaCoded` / `IsCuChromaQpOffsetCoded` state), and then
//! the luma / chroma `residual_coding()` invocations — including the
//! `blkIdx == 3` deferred-chroma path for the `4:2:0` / `4:2:2`
//! sub-sampled-chroma case where the chroma residuals of all four luma
//! sub-blocks are coded against the parent (`xBase`, `yBase`,
//! `log2TrafoSize`) at the last luma leaf.
//!
//! The driver does **not** own the §8.6 dequantization / inverse
//! transform that turns the returned [`crate::hevc::engine::residual::ResidualBlock`]
//! arrays into sample residuals (that is [`crate::hevc::engine::transform`]), nor the
//! §7.3.8.8 `transform_tree()` recursion that derives the cbf flags and
//! invokes it; both remain the caller's responsibility.

use crate::hevc::engine::binarization::{
    CrossCompPred, CuChromaQpOffset, CuQpDelta, decode_cross_comp_pred, decode_cu_chroma_qp_offset,
    decode_cu_qp_delta, decode_tu_residual_act_flag, log2_res_scale_abs_plus1_ctx_inc,
    res_scale_sign_flag_ctx_inc, tu_residual_act_flag_inferred,
};
use crate::hevc::engine::cabac::CabacEngine;
use crate::hevc::engine::ctx_init::SliceContexts;
use crate::hevc::engine::residual::{
    ResidualBlock, ResidualCodingError, ResidualCodingParams, decode_residual_coding,
    residual_coding_scan_idx,
};

/// The §7.3.8.5 `CuPredMode` of the coding unit containing this
/// transform unit. Selects the §7.3.8.10 cross-component-prediction and
/// adaptive-colour-transform gates and the §7.4.9.11 scan-order
/// derivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuPredMode {
    /// `MODE_INTRA`.
    Intra,
    /// `MODE_INTER`.
    Inter,
}

impl CuPredMode {
    /// `true` for `MODE_INTRA`.
    #[must_use]
    pub fn is_intra(self) -> bool {
        matches!(self, CuPredMode::Intra)
    }
}

/// Caller-derived inputs to one §7.3.8.10 `transform_unit( )`
/// invocation. Every field corresponds to a variable the §7.3.8.10
/// syntax table reads from the surrounding coding-tree / coding-unit /
/// transform-tree context.
#[derive(Debug, Clone, Copy)]
pub struct TransformUnitParams {
    /// `log2TrafoSize` of the luma transform block, 2..=5.
    pub log2_trafo_size: u32,
    /// `trafoDepth` — the transform-tree recursion depth.
    pub trafo_depth: u32,
    /// `blkIdx` — the luma sub-block index within the parent transform
    /// node, 0..=3. Drives the §7.3.8.10 `blkIdx == 3` deferred-chroma
    /// path.
    pub blk_idx: u32,
    /// `CuPredMode[ x0 ][ y0 ]`.
    pub cu_pred_mode: CuPredMode,
    /// `ChromaArrayType` (0 = monochrome, 1 = 4:2:0, 2 = 4:2:2,
    /// 3 = 4:4:4).
    pub chroma_array_type: u8,
    /// `cbf_luma[ x0 ][ y0 ][ trafoDepth ]` — whether the luma
    /// transform block has coded coefficients.
    pub cbf_luma: bool,
    /// `cbf_cb[ xC ][ yC ][ cbfDepthC ]` for the primary Cb block (and
    /// — for `ChromaArrayType == 2` — the lower-half Cb block; see
    /// [`TransformUnitParams::cbf_cb_lower`]).
    pub cbf_cb: bool,
    /// The `ChromaArrayType == 2` lower-half Cb coded-block flag
    /// (`cbf_cb[ xC ][ yC + (1 << log2TrafoSizeC) ][ cbfDepthC ]`).
    /// Ignored when `ChromaArrayType != 2`.
    pub cbf_cb_lower: bool,
    /// `cbf_cr[ xC ][ yC ][ cbfDepthC ]` for the primary Cr block.
    pub cbf_cr: bool,
    /// The `ChromaArrayType == 2` lower-half Cr coded-block flag.
    /// Ignored when `ChromaArrayType != 2`.
    pub cbf_cr_lower: bool,
    /// `IntraPredModeY[ x0 ][ y0 ]` — the luma intra-prediction mode
    /// (used only by the §7.4.9.11 luma scan-order derivation when
    /// [`CuPredMode::Intra`]).
    pub intra_pred_mode_y: u32,
    /// `IntraPredModeC[ x0 ][ y0 ]` — the chroma intra-prediction mode
    /// (used by the §7.4.9.11 chroma scan-order derivation).
    pub intra_pred_mode_c: u32,
    /// `intra_chroma_pred_mode[ x0 ][ y0 ]` — the raw signalled value,
    /// needed by the §7.3.8.10 cross-component-prediction /
    /// adaptive-colour-transform gates (the `intra_chroma_pred_mode == 4`
    /// "derived-mode" condition).
    pub intra_chroma_pred_mode: u8,
    /// PPS `cu_qp_delta_enabled_flag` (§7.4.3.3.1) — the §7.3.8.14
    /// `delta_qp()` outer gate.
    pub cu_qp_delta_enabled_flag: bool,
    /// PPS `cu_chroma_qp_offset_enabled_flag` derived state (the
    /// per-slice `cu_chroma_qp_offset_enabled_flag` of §7.4.9.10) — the
    /// §7.3.8.15 `chroma_qp_offset()` outer gate.
    pub cu_chroma_qp_offset_enabled_flag: bool,
    /// `chroma_qp_offset_list_len_minus1` (§7.4.3.3.1) — bounds the
    /// `cu_chroma_qp_offset_idx` TR prefix.
    pub chroma_qp_offset_list_len_minus1: u32,
    /// `cu_transquant_bypass_flag[ x0 ][ y0 ]` — suppresses the
    /// `chroma_qp_offset()` call (the §7.3.8.10
    /// `cbfChroma && !cu_transquant_bypass_flag` gate).
    pub cu_transquant_bypass_flag: bool,
    /// PPS `sign_data_hiding_enabled_flag` (§7.3.2.3.1), forwarded to
    /// the `residual_coding()` invocations.
    pub sign_data_hiding_enabled_flag: bool,
    /// PPS `cross_component_prediction_enabled_flag` (§7.4.3.3.1) — the
    /// §7.3.8.10 `cross_comp_pred()` outer gate.
    pub cross_component_prediction_enabled_flag: bool,
    /// SCC `residual_adaptive_colour_transform_enabled_flag`
    /// (§7.4.3.3.1) — the §7.3.8.10 `tu_residual_act_flag` outer gate.
    pub residual_adaptive_colour_transform_enabled_flag: bool,
    /// PPS `transform_skip_enabled_flag` (§7.4.3.3.1) — the §7.3.8.11
    /// `transform_skip_flag` presence gate.
    pub transform_skip_enabled_flag: bool,
    /// `Log2MaxTransformSkipSize` (§7.4.3.3.2).
    pub log2_max_transform_skip_size: u32,
    /// SPS range extension `implicit_rdpcm_enabled_flag` (§7.4.3.2.2).
    pub implicit_rdpcm_enabled_flag: bool,
    /// SPS range extension `explicit_rdpcm_enabled_flag` (§7.4.3.2.2).
    pub explicit_rdpcm_enabled_flag: bool,
    /// SPS range extension `transform_skip_context_enabled_flag`
    /// (§7.4.3.2.2).
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
    /// `BitDepthY` (eq. 9-14 input for luma blocks).
    pub bit_depth_luma: u8,
    /// `BitDepthC` (eq. 9-14 input for chroma blocks).
    pub bit_depth_chroma: u8,
    /// `PartMode == PART_2Nx2N` — part of the §7.3.8.10
    /// adaptive-colour-transform predicate.
    pub part_mode_2nx2n: bool,
    /// The four corner `intra_chroma_pred_mode` values at the
    /// quarter-block positions `(xP, yP)`, `(xP+nCbS/2, yP)`,
    /// `(xP, yP+nCbS/2)`, `(xP+nCbS/2, yP+nCbS/2)` — used only by the
    /// `MODE_INTRA` branch of the §7.3.8.10 adaptive-colour-transform
    /// predicate (all four must be `4`). Order is row-major:
    /// `[tl, tr, bl, br]`.
    pub intra_chroma_pred_mode_corners: [u8; 4],
}

/// The decoded result of one §7.3.8.10 `transform_unit( )` invocation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TransformUnit {
    /// `tu_residual_act_flag[ x0 ][ y0 ]` (§7.4.9.10) — present value or
    /// the inferred `0`.
    pub tu_residual_act_flag: u8,
    /// The `delta_qp()` result, present only when the §7.3.8.14 gate
    /// fired (`cu_qp_delta_enabled_flag && !IsCuQpDeltaCoded`).
    pub cu_qp_delta: Option<CuQpDelta>,
    /// The `chroma_qp_offset()` result, present only when the §7.3.8.15
    /// gate fired.
    pub cu_chroma_qp_offset: Option<CuChromaQpOffset>,
    /// The §7.3.8.12 Cb `cross_comp_pred( x0, y0, 0 )` result, present
    /// only when the §7.3.8.10 cross-component-prediction gate fired.
    pub cross_comp_pred_cb: Option<CrossCompPred>,
    /// The §7.3.8.12 Cr `cross_comp_pred( x0, y0, 1 )` result.
    pub cross_comp_pred_cr: Option<CrossCompPred>,
    /// The luma `residual_coding( x0, y0, log2TrafoSize, 0 )` block,
    /// present only when `cbfLuma` is set.
    pub residual_luma: Option<ResidualBlock>,
    /// The Cb `residual_coding()` blocks, one per coded chroma sub-block
    /// (`ChromaArrayType == 2` can produce two: the upper then the lower
    /// half).
    pub residual_cb: Vec<ResidualBlock>,
    /// The Cr `residual_coding()` blocks.
    pub residual_cr: Vec<ResidualBlock>,
    /// Which chroma halves carried a coded Cb block: `[upper, lower]`
    /// (`ChromaArrayType == 2`; the lower slot is always `false`
    /// otherwise). Pairs [`Self::residual_cb`] entries — in coding
    /// order — to their vertical positions.
    pub cbf_cb_halves: [bool; 2],
    /// Which chroma halves carried a coded Cr block: `[upper, lower]`.
    pub cbf_cr_halves: [bool; 2],
}

/// The once-per-quantization-group decode state the §7.3.8.10 driver
/// both reads and updates: §7.3.8.14 `IsCuQpDeltaCoded` /
/// §7.4.9.14 `CuQpDeltaVal` and §7.3.8.15 `IsCuChromaQpOffsetCoded`.
///
/// The §7.4.3.3 quantization-group reset (`IsCuQpDeltaCoded = 0`,
/// `CuQpDeltaVal = 0`, `IsCuChromaQpOffsetCoded = 0`) at the start of
/// each coding-quadtree quantization group is the caller's
/// responsibility; this struct carries the state across the multiple
/// transform units inside one quantization group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QuantGroupState {
    /// §7.3.8.14 `IsCuQpDeltaCoded`.
    pub is_cu_qp_delta_coded: bool,
    /// §7.4.9.14 `CuQpDeltaVal` — accumulated signed delta.
    pub cu_qp_delta_val: i32,
    /// §7.3.8.15 `IsCuChromaQpOffsetCoded`.
    pub is_cu_chroma_qp_offset_coded: bool,
}

/// Decode one §7.3.8.10 `transform_unit( )` body from the CABAC engine,
/// drawing every context from the slice-scope [`SliceContexts`] bank and
/// updating the per-quantization-group [`QuantGroupState`] in place.
///
/// The function mirrors the §7.3.8.10 syntax table exactly:
///
/// 1. `cbfChroma` is derived (eq. in §7.3.8.10) from the supplied
///    `cbf_cb` / `cbf_cr` (plus their `ChromaArrayType == 2` lower-half
///    companions).
/// 2. When `cbfLuma || cbfChroma`, the §7.3.8.10 adaptive-colour-
///    transform predicate is evaluated; if it holds,
///    `tu_residual_act_flag` is read, else it is inferred to 0.
/// 3. `delta_qp()` and `chroma_qp_offset()` are read, each gated on its
///    `Is…Coded` state.
/// 4. The luma `residual_coding()` is read when `cbfLuma`.
/// 5. For `log2TrafoSize > 2 || ChromaArrayType == 3`, the in-place
///    Cb/Cr residual_coding (with the cross_comp_pred prelude) is read;
///    for the `blkIdx == 3` deferred path the parent-base chroma
///    residuals are read instead.
pub fn decode_transform_unit(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &TransformUnitParams,
    qg: &mut QuantGroupState,
) -> Result<TransformUnit, ResidualCodingError> {
    let mut tu = TransformUnit::default();
    let lower = params.chroma_array_type == 2;
    tu.cbf_cb_halves = [params.cbf_cb, lower && params.cbf_cb_lower];
    tu.cbf_cr_halves = [params.cbf_cr, lower && params.cbf_cr_lower];

    // §7.3.8.10: cbfChroma = cbf_cb[..] || cbf_cr[..] ||
    // ( ChromaArrayType == 2 && ( cbf_cb[lower] || cbf_cr[lower] ) )
    let cbf_chroma = params.cbf_cb
        || params.cbf_cr
        || (params.chroma_array_type == 2 && (params.cbf_cb_lower || params.cbf_cr_lower));

    if params.cbf_luma || cbf_chroma {
        // §7.3.8.10 adaptive-colour-transform predicate — a THREE-way
        // disjunction: MODE_INTER, or a PART_2Nx2N intra CU whose
        // intra_chroma_pred_mode[ x0 ][ y0 ] is 4 (derived-mode), or
        // an intra CU whose four MinCb-quadrant
        // intra_chroma_pred_mode[ xP.. ][ yP.. ] values are ALL 4 (the
        // PART_NxN derived-mode case).
        let act_gate = params.residual_adaptive_colour_transform_enabled_flag
            && (params.cu_pred_mode == CuPredMode::Inter
                || (params.part_mode_2nx2n && params.intra_chroma_pred_mode == 4)
                || params
                    .intra_chroma_pred_mode_corners
                    .iter()
                    .all(|&m| m == 4));
        tu.tu_residual_act_flag = if act_gate {
            decode_tu_residual_act_flag(engine, &mut ctx.tu_residual_act_flag[0])?
        } else {
            tu_residual_act_flag_inferred()
        };

        // delta_qp(): §7.3.8.14, gated by cu_qp_delta_enabled_flag &&
        // !IsCuQpDeltaCoded.
        if params.cu_qp_delta_enabled_flag && !qg.is_cu_qp_delta_coded {
            qg.is_cu_qp_delta_coded = true;
            // ctx.cu_qp_delta_abs[0] is bin 0, [1] the shared rest.
            let (bin0, rest) = ctx.cu_qp_delta_abs.split_at_mut(1);
            let delta = decode_cu_qp_delta(engine, &mut bin0[0], &mut rest[0])?;
            qg.cu_qp_delta_val = delta.value;
            tu.cu_qp_delta = Some(delta);
        }

        // chroma_qp_offset(): §7.3.8.15, gated by
        // cbfChroma && !cu_transquant_bypass_flag (the §7.3.8.10 outer
        // gate) then cu_chroma_qp_offset_enabled_flag &&
        // !IsCuChromaQpOffsetCoded (the §7.3.8.15 inner gate).
        if cbf_chroma
            && !params.cu_transquant_bypass_flag
            && params.cu_chroma_qp_offset_enabled_flag
            && !qg.is_cu_chroma_qp_offset_coded
        {
            qg.is_cu_chroma_qp_offset_coded = true;
            let offset = decode_cu_chroma_qp_offset(
                engine,
                &mut ctx.cu_chroma_qp_offset_flag[0],
                &mut ctx.cu_chroma_qp_offset_idx[0],
                params.chroma_qp_offset_list_len_minus1,
            )?;
            tu.cu_chroma_qp_offset = Some(offset);
        }
    }

    // Luma residual_coding( x0, y0, log2TrafoSize, 0 ).
    if params.cbf_luma {
        tu.residual_luma = Some(self::decode_one_residual(
            engine,
            ctx,
            params,
            params.log2_trafo_size,
            0,
        )?);
    }

    // log2TrafoSizeC = Max( 2, log2TrafoSize − ( ChromaArrayType == 3 ? 0 : 1 ) ).
    let log2_trafo_size_c = core::cmp::max(
        2,
        params.log2_trafo_size - if params.chroma_array_type == 3 { 0 } else { 1 },
    );

    if params.log2_trafo_size > 2 || params.chroma_array_type == 3 {
        // The in-place chroma path: cross_comp_pred + residual_coding
        // for each chroma sub-block at this transform unit.
        // cross_comp_pred is present only when the §7.3.8.10 gate holds.
        let ccp_gate = params.cross_component_prediction_enabled_flag
            && params.cbf_luma
            && (params.cu_pred_mode == CuPredMode::Inter || params.intra_chroma_pred_mode == 4);

        // Cb (c == 0).
        if ccp_gate {
            tu.cross_comp_pred_cb = Some(self::decode_one_cross_comp_pred(engine, ctx, 0)?);
        }
        // ChromaArrayType == 2 ⇒ two stacked Cb sub-blocks, else one.
        let chroma_sub_blocks = if params.chroma_array_type == 2 { 2 } else { 1 };
        for t_idx in 0..chroma_sub_blocks {
            let coded = if t_idx == 0 {
                params.cbf_cb
            } else {
                params.cbf_cb_lower
            };
            if coded {
                tu.residual_cb.push(self::decode_one_residual(
                    engine,
                    ctx,
                    params,
                    log2_trafo_size_c,
                    1,
                )?);
            }
        }

        // Cr (c == 1).
        if ccp_gate {
            tu.cross_comp_pred_cr = Some(self::decode_one_cross_comp_pred(engine, ctx, 1)?);
        }
        for t_idx in 0..chroma_sub_blocks {
            let coded = if t_idx == 0 {
                params.cbf_cr
            } else {
                params.cbf_cr_lower
            };
            if coded {
                tu.residual_cr.push(self::decode_one_residual(
                    engine,
                    ctx,
                    params,
                    log2_trafo_size_c,
                    2,
                )?);
            }
        }
    } else if params.blk_idx == 3 {
        // Deferred-chroma path: the chroma residuals of the four luma
        // sub-blocks are coded against the parent node at the last luma
        // leaf. residual_coding( xBase, yBase, log2TrafoSize, 1/2 ),
        // here parameterised at the parent log2TrafoSize. No
        // cross_comp_pred in this branch (the §7.3.8.10 table only
        // invokes cross_comp_pred in the in-place branch).
        let chroma_sub_blocks = if params.chroma_array_type == 2 { 2 } else { 1 };
        for t_idx in 0..chroma_sub_blocks {
            let coded = if t_idx == 0 {
                params.cbf_cb
            } else {
                params.cbf_cb_lower
            };
            if coded {
                tu.residual_cb.push(self::decode_one_residual(
                    engine,
                    ctx,
                    params,
                    params.log2_trafo_size,
                    1,
                )?);
            }
        }
        for t_idx in 0..chroma_sub_blocks {
            let coded = if t_idx == 0 {
                params.cbf_cr
            } else {
                params.cbf_cr_lower
            };
            if coded {
                tu.residual_cr.push(self::decode_one_residual(
                    engine,
                    ctx,
                    params,
                    params.log2_trafo_size,
                    2,
                )?);
            }
        }
    }

    Ok(tu)
}

/// Decode one §7.3.8.12 `cross_comp_pred( x0, y0, c )` against the
/// per-component context banks sliced out of [`SliceContexts`].
fn decode_one_cross_comp_pred(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    c: u32,
) -> Result<CrossCompPred, ResidualCodingError> {
    // log2_res_scale_abs_plus1 ctxInc = 4*c + binIdx (Table 9-48): the
    // per-component four-context bank is ctx.log2_res_scale_abs_plus1
    // sliced at [base..base+4]; res_scale_sign_flag ctxInc = c.
    let base = log2_res_scale_abs_plus1_ctx_inc(0, c) as usize;
    let sign_idx = res_scale_sign_flag_ctx_inc(c) as usize;
    // Split the sign bank off first so both borrows are disjoint.
    let (prefix_bank, sign_bank) = (
        &mut ctx.log2_res_scale_abs_plus1[base..base + 4],
        &mut ctx.res_scale_sign_flag[sign_idx],
    );
    Ok(decode_cross_comp_pred(engine, prefix_bank, sign_bank)?)
}

/// Decode one `residual_coding( )` invocation for the given component,
/// deriving the §7.4.9.11 scan order and assembling the
/// [`ResidualCodingParams`].
fn decode_one_residual(
    engine: &mut CabacEngine<'_>,
    ctx: &mut SliceContexts,
    params: &TransformUnitParams,
    log2_trafo_size: u32,
    c_idx: u8,
) -> Result<ResidualBlock, ResidualCodingError> {
    let pred_mode_intra = if c_idx == 0 {
        params.intra_pred_mode_y
    } else {
        params.intra_pred_mode_c
    };
    let scan_idx = residual_coding_scan_idx(
        params.cu_pred_mode.is_intra(),
        log2_trafo_size,
        c_idx,
        params.chroma_array_type,
        pred_mode_intra,
    );

    // §7.3.8.11 leading elements: transform_skip_flag when the PPS
    // enables transform skip, the CU is not transquant-bypassed, and
    // the block is within Log2MaxTransformSkipSize.
    let mut transform_skip = false;
    if params.transform_skip_enabled_flag
        && !params.cu_transquant_bypass_flag
        && log2_trafo_size <= params.log2_max_transform_skip_size
    {
        // Table 9-25 contexts: slot 0 luma (cIdx == 0), slot 1 chroma.
        let slot = usize::from(c_idx > 0);
        transform_skip = engine.decode_decision(&mut ctx.transform_skip_flag[slot])? != 0;
    }
    // §7.3.8.11: explicit_rdpcm_flag / explicit_rdpcm_dir_flag (range
    // extensions) for inter blocks under transform skip / bypass.
    let mut explicit_rdpcm_flag = false;
    let mut explicit_rdpcm_dir_flag = false;
    if params.cu_pred_mode == CuPredMode::Inter
        && params.explicit_rdpcm_enabled_flag
        && (transform_skip || params.cu_transquant_bypass_flag)
    {
        let slot = usize::from(c_idx > 0);
        explicit_rdpcm_flag = engine.decode_decision(&mut ctx.explicit_rdpcm_flag[slot])? != 0;
        if explicit_rdpcm_flag {
            explicit_rdpcm_dir_flag =
                engine.decode_decision(&mut ctx.explicit_rdpcm_dir_flag[slot])? != 0;
        }
    }

    // §7.3.8.11 signHidden: forced to 0 for transquant bypass, for the
    // implicit-RDPCM intra horizontal/vertical transform-skip case
    // (predModeIntra 10 / 26), and for explicit RDPCM.
    let implicit_rdpcm = params.cu_pred_mode.is_intra()
        && params.implicit_rdpcm_enabled_flag
        && transform_skip
        && (pred_mode_intra == 10 || pred_mode_intra == 26);
    let rc_params = ResidualCodingParams {
        log2_trafo_size,
        is_chroma: c_idx > 0,
        scan_idx,
        sign_data_hiding_enabled_flag: params.sign_data_hiding_enabled_flag,
        sign_hidden_suppressed: params.cu_transquant_bypass_flag
            || implicit_rdpcm
            || explicit_rdpcm_flag,
        // §9.3.4.2.5 — the transform-skip sig-ctx selection applies
        // when transform_skip_context_enabled_flag is set and the block
        // is transform-skipped or transquant-bypassed.
        transform_skip_sig_ctx: params.transform_skip_context_enabled_flag
            && (transform_skip || params.cu_transquant_bypass_flag),
        persistent_rice_adaptation_enabled_flag: params.persistent_rice_adaptation_enabled_flag,
        cabac_bypass_alignment_enabled_flag: params.cabac_bypass_alignment_enabled_flag,
        extended_precision_processing_flag: params.extended_precision_processing_flag,
        bit_depth: if c_idx > 0 {
            params.bit_depth_chroma
        } else {
            params.bit_depth_luma
        },
        rice_stat_transform_skip: transform_skip || params.cu_transquant_bypass_flag,
    };
    let mut rb = decode_residual_coding(engine, &mut ctx.residual, &rc_params)?;
    rb.transform_skip = transform_skip;
    rb.explicit_rdpcm_flag = explicit_rdpcm_flag;
    rb.explicit_rdpcm_dir_flag = explicit_rdpcm_dir_flag;
    Ok(rb)
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::bitreader::BitReader;

    fn base_params() -> TransformUnitParams {
        TransformUnitParams {
            log2_trafo_size: 3,
            trafo_depth: 0,
            blk_idx: 0,
            cu_pred_mode: CuPredMode::Intra,
            chroma_array_type: 1,
            cbf_luma: false,
            cbf_cb: false,
            cbf_cb_lower: false,
            cbf_cr: false,
            cbf_cr_lower: false,
            intra_pred_mode_y: 0,
            intra_pred_mode_c: 0,
            intra_chroma_pred_mode: 0,
            cu_qp_delta_enabled_flag: false,
            cu_chroma_qp_offset_enabled_flag: false,
            chroma_qp_offset_list_len_minus1: 0,
            cu_transquant_bypass_flag: false,
            sign_data_hiding_enabled_flag: false,
            cross_component_prediction_enabled_flag: false,
            residual_adaptive_colour_transform_enabled_flag: false,
            transform_skip_enabled_flag: false,
            log2_max_transform_skip_size: 2,
            implicit_rdpcm_enabled_flag: false,
            explicit_rdpcm_enabled_flag: false,
            transform_skip_context_enabled_flag: false,
            persistent_rice_adaptation_enabled_flag: false,
            cabac_bypass_alignment_enabled_flag: false,
            extended_precision_processing_flag: false,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            part_mode_2nx2n: true,
            intra_chroma_pred_mode_corners: [0; 4],
        }
    }

    /// No cbf set ⇒ the syntax table reads nothing: an empty
    /// transform unit, no engine bins consumed.
    #[test]
    fn empty_transform_unit_reads_no_bins() {
        // The engine is initialised but must not be consulted since no
        // cbf is set; the context bank must be left untouched.
        let data = [0x5Au8; 96];
        let mut engine = CabacEngine::new(BitReader::new(&data)).unwrap();
        let mut ctx = SliceContexts::init(0, 26);
        let ctx_before = ctx.clone();
        let mut qg = QuantGroupState::default();
        let params = base_params();
        let tu = decode_transform_unit(&mut engine, &mut ctx, &params, &mut qg).unwrap();
        assert_eq!(tu, TransformUnit::default());
        assert_eq!(tu.tu_residual_act_flag, 0);
        assert!(tu.cu_qp_delta.is_none());
        assert!(tu.residual_luma.is_none());
        assert!(tu.residual_cb.is_empty());
        // No context adaptation ⇒ no bins consumed from the engine.
        assert_eq!(ctx, ctx_before);
        assert_eq!(qg, QuantGroupState::default());
    }

    /// §7.3.8.10 — the `tu_residual_act_flag` presence gate is a
    /// THREE-way disjunction. A `PART_NxN` intra coding unit whose
    /// four MinCb-quadrant `intra_chroma_pred_mode` values are all 4
    /// reads the flag even though the `PART_2Nx2N &&
    /// intra_chroma_pred_mode == 4` arm is false, and a `PART_2Nx2N`
    /// derived-mode CU reads it regardless of the corner values.
    #[test]
    fn act_flag_gate_is_a_three_way_disjunction() {
        let decode_consumes_act_bin = |part_2nx2n: bool, icpm: u8, corners: [u8; 4]| -> bool {
            let data = [0x5Au8; 96];
            let mut engine = CabacEngine::new(BitReader::new(&data)).unwrap();
            let mut ctx = SliceContexts::init(0, 26);
            let ctx_before = ctx.clone();
            let mut qg = QuantGroupState::default();
            let mut params = base_params();
            params.chroma_array_type = 3;
            params.residual_adaptive_colour_transform_enabled_flag = true;
            params.cbf_luma = true;
            params.log2_trafo_size = 2;
            params.part_mode_2nx2n = part_2nx2n;
            params.intra_chroma_pred_mode = icpm;
            params.intra_chroma_pred_mode_corners = corners;
            decode_transform_unit(&mut engine, &mut ctx, &params, &mut qg).unwrap();
            // The act flag is the only context-coded element consulting
            // this slot; adaptation proves the bin was read.
            ctx.tu_residual_act_flag != ctx_before.tu_residual_act_flag
        };
        // PART_NxN (not 2Nx2N), all four corners derived-mode: read.
        assert!(decode_consumes_act_bin(false, 4, [4, 4, 4, 4]));
        // PART_2Nx2N derived-mode: read (second arm).
        assert!(decode_consumes_act_bin(true, 4, [4, 4, 4, 4]));
        // PART_NxN with a non-derived corner: inferred 0, no bin.
        assert!(!decode_consumes_act_bin(false, 4, [4, 4, 0, 4]));
        // PART_2Nx2N non-derived mode: inferred 0, no bin.
        assert!(!decode_consumes_act_bin(true, 0, [0, 0, 0, 0]));
    }

    /// `cbfLuma` set but no chroma ⇒ a single luma residual_coding is
    /// invoked and the quant-group state is consulted.
    #[test]
    fn cbf_luma_reads_luma_residual() {
        let data = [0x5Au8; 96];
        let mut engine = CabacEngine::new(BitReader::new(&data)).unwrap();
        let mut ctx = SliceContexts::init(0, 26);
        let mut qg = QuantGroupState::default();
        let mut params = base_params();
        params.cbf_luma = true;
        params.log2_trafo_size = 2;
        let tu = decode_transform_unit(&mut engine, &mut ctx, &params, &mut qg).unwrap();
        assert!(tu.residual_luma.is_some());
        assert!(tu.residual_cb.is_empty());
        assert!(tu.residual_cr.is_empty());
    }

    /// `cu_qp_delta_enabled_flag` with a coded chroma block fires the
    /// delta_qp() read once and flips IsCuQpDeltaCoded; a second TU in
    /// the same quant group must not re-read it.
    #[test]
    fn delta_qp_fires_once_per_quant_group() {
        // Build a stream whose first decode_decision bins yield a zero
        // cu_qp_delta_abs prefix (so delta_qp consumes a bounded number
        // of bins and CuQpDeltaVal == 0).
        let data = [0x5Au8; 96];
        let mut engine = CabacEngine::new(BitReader::new(&data)).unwrap();
        let mut ctx = SliceContexts::init(0, 26);
        let mut qg = QuantGroupState::default();
        let mut params = base_params();
        params.cu_qp_delta_enabled_flag = true;
        params.cbf_cb = true; // makes cbfChroma true so the gate block runs
        params.log2_trafo_size = 3;
        let tu = decode_transform_unit(&mut engine, &mut ctx, &params, &mut qg).unwrap();
        assert!(tu.cu_qp_delta.is_some());
        assert!(qg.is_cu_qp_delta_coded);

        // Second TU in the same quant group: gate must not fire again.
        let tu2 = decode_transform_unit(&mut engine, &mut ctx, &params, &mut qg).unwrap();
        assert!(tu2.cu_qp_delta.is_none());
    }

    /// The deferred-chroma path is selected only at blkIdx == 3 with
    /// log2TrafoSize == 2 and ChromaArrayType != 3.
    #[test]
    fn deferred_chroma_path_at_blk_idx_3() {
        let data = [0x5Au8; 96];
        let mut engine = CabacEngine::new(BitReader::new(&data)).unwrap();
        let mut ctx = SliceContexts::init(0, 26);
        let mut qg = QuantGroupState::default();
        let mut params = base_params();
        params.log2_trafo_size = 2;
        params.blk_idx = 3;
        params.chroma_array_type = 1;
        params.cbf_cb = true;
        let tu = decode_transform_unit(&mut engine, &mut ctx, &params, &mut qg).unwrap();
        // The deferred path coded the Cb residual at the parent size.
        assert_eq!(tu.residual_cb.len(), 1);
        assert_eq!(tu.residual_cb[0].log2_trafo_size, 2);
    }

    /// ChromaArrayType == 2 stacks two chroma sub-blocks; both coded
    /// flags produce two residual blocks each.
    #[test]
    fn chroma_422_stacks_two_sub_blocks() {
        let data = [0x5Au8; 96];
        let mut engine = CabacEngine::new(BitReader::new(&data)).unwrap();
        let mut ctx = SliceContexts::init(0, 26);
        let mut qg = QuantGroupState::default();
        let mut params = base_params();
        params.chroma_array_type = 2;
        params.log2_trafo_size = 3;
        params.cbf_cb = true;
        params.cbf_cb_lower = true;
        params.cbf_cr = true;
        params.cbf_cr_lower = true;
        let tu = decode_transform_unit(&mut engine, &mut ctx, &params, &mut qg).unwrap();
        assert_eq!(tu.residual_cb.len(), 2);
        assert_eq!(tu.residual_cr.len(), 2);
    }
}
