//! §8.5 picture-level inter reconstruction driver.
//!
//! This module is the rung between the §7.3.8 slice-data parse tree (the
//! decoded [`crate::hevc::engine::slice_data::CodingTreeUnit`] structures, with their
//! inter coding units carrying §7.3.8.6 prediction units) and the per-PU
//! §8.5.3.3 motion-compensated prediction + §8.6.2 residual reconstruction
//! already implemented in [`crate::hevc::engine::inter_pred`] and [`crate::hevc::engine::recon`]. It
//! walks a picture's decoded CTUs in tile-scan (decode) order and, for each
//! inter coding unit:
//!
//! 1. §8.5.3.2.1 — resolve every prediction unit's motion vectors,
//!    reference indices and `predFlagLX` from the parsed syntax, gathering
//!    the spatial merge / MVP neighbours from the *current* picture's
//!    motion field (built up by the earlier CUs) and the temporal `Col`
//!    candidate from the collocated picture's motion field
//!    ([`crate::hevc::engine::pu_mv::resolve_cu_motion`]).
//! 2. §8.5.3.3 — for each PU, build the §8.5.3.3.2 reference planes from the
//!    resolved `RefPicListX[ refIdxLX ]` pictures, interpolate + combine,
//!    add the §8.6.2 residual sliced from the CU residual planes
//!    ([`crate::hevc::engine::recon::extract_cu_residual`]) and clip into the target
//!    picture ([`crate::hevc::engine::recon::reconstruct_inter_pu`]).
//!
//! Intra coding units inside a P / B slice are reconstructed by the §8.4
//! intra path; the motion field records them as intra so a later inter CU's
//! §6.4.2 prediction-block availability denies them as motion neighbours.

use crate::hevc::engine::dpb::{DpbEntry, RefPicLists};
use crate::hevc::engine::inter_pred::{PuWeights, WpListWeights};
use crate::hevc::engine::motion::{MotionField, derive_chroma_mv};
use crate::hevc::engine::picture::{Picture, sub_wh_c};
use crate::hevc::engine::pu_mv::{InterCuDesc, PuMotion, PuMvContext, PuRect, resolve_cu_motion};
use crate::hevc::engine::recon::{
    CuResidual, ReconError, ReconParams, ResolvedList, extract_cu_residual,
    reconstruct_inter_pu_weighted,
};
use crate::hevc::engine::slice_data::{CodingUnit, PredictionUnit};

/// The slice's §7.4.7.3-derived weighted-prediction tables, resolved
/// from the parsed `pred_weight_table()` into the per-reference values
/// the §8.5.3.3.4.3 combine reads: `LumaWeightLX[i]`,
/// `luma_offset_lX[i] << WpOffsetBdShiftY` (equations 8-268 / 8-269),
/// `ChromaWeightLX[i][j]` and `ChromaOffsetLX[i][j] << WpOffsetBdShiftC`
/// (equations 8-273 / 8-274).
///
/// Present on an [`InterSliceContext`] exactly when `weightedPredFlag`
/// (§8.5.3.3.4.1: `weighted_pred_flag` for P slices,
/// `weighted_bipred_flag` for B slices) is 1 — `None` selects the
/// §8.5.3.3.4.2 default combine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SliceWpTables {
    /// `luma_log2_weight_denom` (§7.4.7.3).
    pub luma_log2_weight_denom: u8,
    /// `ChromaLog2WeightDenom` (§7.4.7.3).
    pub chroma_log2_weight_denom: u8,
    /// Per-`refIdxL0` weights, length `num_ref_idx_l0_active`.
    pub l0: Vec<WpListWeights>,
    /// Per-`refIdxL1` weights, length `num_ref_idx_l1_active` (empty
    /// for P slices).
    pub l1: Vec<WpListWeights>,
}

impl SliceWpTables {
    /// Resolve one PU's [`PuWeights`] from its reference indices. An
    /// unused list (or an index past the table, which a conforming
    /// stream never produces) falls back to the §7.4.7.3 inferred
    /// values `w = 1 << denom, o = 0` — the identity weighting.
    #[must_use]
    pub fn resolve_pu(&self, motion: &PuMotion) -> PuWeights {
        let identity = |denom: u8| WpListWeights {
            w_luma: 1 << self.luma_log2_weight_denom,
            o_luma: 0,
            w_cb: 1 << denom,
            o_cb: 0,
            w_cr: 1 << denom,
            o_cr: 0,
        };
        let pick = |list: &[WpListWeights], used: bool, idx: i32| {
            if used {
                list.get(idx.max(0) as usize)
                    .copied()
                    .unwrap_or_else(|| identity(self.chroma_log2_weight_denom))
            } else {
                identity(self.chroma_log2_weight_denom)
            }
        };
        PuWeights {
            luma_log2_weight_denom: self.luma_log2_weight_denom,
            chroma_log2_weight_denom: self.chroma_log2_weight_denom,
            l0: pick(&self.l0, motion.pred_flag_l0, motion.ref_idx_l0),
            l1: pick(&self.l1, motion.pred_flag_l1, motion.ref_idx_l1),
        }
    }
}

/// The resolved reference-picture access an inter CU reconstruction needs:
/// `RefPicListX[ refIdx ]` → a borrowed reference [`Picture`] and its POC.
///
/// The picture-level driver binds this to the §8.3.4 [`RefPicLists`] + the
/// [`crate::hevc::engine::dpb::Dpb`] entries; the per-CU reconstruction reads it through
/// the [`PuMvContext`] resolvers (for the candidate derivation) and to
/// fetch each used list's reference planes (for the interpolation).
#[derive(Debug)]
pub struct RefListAccess<'a> {
    /// `RefPicList0` / `RefPicList1` as DPB entry indices.
    pub lists: &'a RefPicLists,
    /// The DPB entries the indices point into.
    pub entries: &'a [DpbEntry],
}

impl<'a> RefListAccess<'a> {
    /// Borrow `RefPicListX[ ref_idx ]`'s reconstructed picture, or `None`
    /// when the list slot is "no reference picture" / out of range.
    #[must_use]
    pub fn ref_pic(&self, list: usize, ref_idx: i32) -> Option<&'a Picture> {
        self.entry(list, ref_idx).map(|e| &e.picture)
    }

    /// `PicOrderCnt( RefPicListX[ ref_idx ] )`, or `i32::MIN` when absent.
    #[must_use]
    pub fn ref_poc(&self, list: usize, ref_idx: i32) -> i32 {
        self.entry(list, ref_idx).map_or(i32::MIN, |e| e.poc)
    }

    /// Borrow `RefPicListX[ ref_idx ]`'s DPB entry, or `None` when the
    /// list slot is "no reference picture", out of range, or the
    /// current picture ([`crate::hevc::engine::dpb::CURR_PIC`] — resolved by the
    /// reconstruction driver, not the DPB).
    #[must_use]
    pub fn entry(&self, list: usize, ref_idx: i32) -> Option<&'a DpbEntry> {
        let idx = self.slot(list, ref_idx)?;
        if idx == crate::hevc::engine::dpb::CURR_PIC {
            return None;
        }
        self.entries.get(idx)
    }

    /// `true` when `RefPicListX[ ref_idx ]` is the CURRENT picture (the
    /// §8.3.4 currPic append — intra block copy).
    #[must_use]
    pub fn is_curr_pic(&self, list: usize, ref_idx: i32) -> bool {
        self.slot(list, ref_idx) == Some(crate::hevc::engine::dpb::CURR_PIC)
    }

    /// The raw `RefPicListX[ ref_idx ]` slot value.
    fn slot(&self, list: usize, ref_idx: i32) -> Option<usize> {
        if ref_idx < 0 {
            return None;
        }
        let slot = if list == 0 {
            self.lists.list0.get(ref_idx as usize)
        } else {
            self.lists.list1.as_ref()?.get(ref_idx as usize)
        };
        *slot?
    }
}

/// §8.5 — reconstruct one inter coding unit into `pic`.
///
/// `motions` are the §8.5.3.2.1-resolved per-PU motions (same order as the
/// CU's §7.3.8.6 prediction units), already written into the picture's
/// motion field by [`resolve_cu_motion`]. `residual` is the CU's
/// per-component residual planes ([`extract_cu_residual`]). `refs` resolves
/// each list's reference picture. Each PU's covering residual is sliced from
/// the CU residual planes and added onto the motion-compensated prediction.
///
/// `wp` carries the slice's §8.5.3.3.4.3 weighted-prediction tables when
/// `weightedPredFlag == 1`; `None` selects the §8.5.3.3.4.2 default
/// combine.
///
/// # Errors
/// Propagates [`ReconError`] from the §8.5.3.3 interpolation / combine.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_inter_cu(
    pic: &mut Picture,
    params: &ReconParams,
    cu: &CodingUnit,
    rects: &[PuRect],
    motions: &[PuMotion],
    residual: &CuResidual,
    refs: &RefListAccess,
    wp: Option<&SliceWpTables>,
) -> Result<(), ReconError> {
    let cat = params.chroma_array_type;
    let (sub_w, sub_h) = if cat != 0 { sub_wh_c(cat) } else { (1, 1) };

    // §8.5.3.1: a prediction from the current picture (intra block
    // copy) reads the current decoded samples before in-loop
    // filtering. The §8.5.3.1 availability constraints guarantee every
    // referenced sample precedes this coding block in z-scan order, so
    // one snapshot at CU entry is exact.
    let needs_curr = refs
        .lists
        .list0
        .contains(&Some(crate::hevc::engine::dpb::CURR_PIC))
        || refs
            .lists
            .list1
            .as_ref()
            .is_some_and(|l| l.contains(&Some(crate::hevc::engine::dpb::CURR_PIC)));
    let needs_curr = needs_curr
        && motions.iter().any(|m| {
            (m.pred_flag_l0 && refs.is_curr_pic(0, m.ref_idx_l0))
                || (m.pred_flag_l1 && refs.is_curr_pic(1, m.ref_idx_l1))
        });
    let snapshot: Option<Picture> = needs_curr.then(|| pic.clone());

    for (rect, motion) in rects.iter().zip(motions.iter()) {
        let l0 = resolve_list(
            params,
            refs,
            snapshot.as_ref(),
            0,
            motion.pred_flag_l0,
            motion.ref_idx_l0,
            motion.mv_l0,
        )?;
        let l1 = resolve_list(
            params,
            refs,
            snapshot.as_ref(),
            1,
            motion.pred_flag_l1,
            motion.ref_idx_l1,
            motion.mv_l1,
        )?;

        // Slice the PU's covering residual out of the CU residual planes.
        let res_luma = residual
            .luma
            .slice_region(rect.x_pb, rect.y_pb, rect.n_pb_w, rect.n_pb_h);
        let (res_cb, res_cr) = if cat != 0 {
            let (cx, cy) = (rect.x_pb / sub_w, rect.y_pb / sub_h);
            let (cw, ch) = (rect.n_pb_w / sub_w, rect.n_pb_h / sub_h);
            (
                residual.cb.as_ref().map(|p| p.slice_region(cx, cy, cw, ch)),
                residual.cr.as_ref().map(|p| p.slice_region(cx, cy, cw, ch)),
            )
        } else {
            (None, None)
        };

        // §8.5.3.3.4.1 — resolve the PU's explicit weights from its
        // reference indices when the slice's weightedPredFlag is 1.
        let pu_weights = wp.map(|t| t.resolve_pu(motion));
        reconstruct_inter_pu_weighted(
            pic,
            params,
            rect.x_pb,
            rect.y_pb,
            rect.n_pb_w,
            rect.n_pb_h,
            l0,
            l1,
            Some(res_luma.as_slice()),
            res_cb.as_deref(),
            res_cr.as_deref(),
            pu_weights.as_ref(),
        )?;
    }
    let _ = cu;
    Ok(())
}

/// Build a [`ResolvedList`] for one reference list, deriving the §8.5.3.2.10
/// chroma MV and fetching `RefPicListX[ refIdx ]`. An unused list (or a
/// list whose reference resolves to "no reference picture") becomes a
/// `pred_flag == false` entry pointing at a fallback picture (never read).
///
/// `curr_snapshot` is the pre-in-loop-filter copy of the current
/// picture (`Some` exactly when the CU references the §8.3.4 currPic —
/// intra block copy).
fn resolve_list<'a, 'b>(
    params: &ReconParams,
    refs: &'b RefListAccess<'a>,
    curr_snapshot: Option<&'b Picture>,
    list: usize,
    pred_flag: bool,
    ref_idx: i32,
    mv_l: [i32; 2],
) -> Result<ResolvedList<'b>, ReconError>
where
    'a: 'b,
{
    let (sw, sh) = if params.chroma_array_type != 0 {
        sub_wh_c(params.chroma_array_type)
    } else {
        (1, 1)
    };
    if pred_flag {
        let ref_pic: Option<&'b Picture> = if refs.is_curr_pic(list, ref_idx) {
            curr_snapshot
        } else {
            refs.ref_pic(list, ref_idx)
        };
        if let Some(ref_pic) = ref_pic {
            let mv_c = derive_chroma_mv(mv_l, sw as i32, sh as i32);
            return Ok(ResolvedList {
                pred_flag: true,
                mv_l,
                mv_c,
                ref_pic,
            });
        }
        // A used list with an unresolvable reference is a malformed stream;
        // surface it rather than silently dropping the prediction.
        return Err(ReconError::InterNotSupported);
    }
    // Unused list: point at any available picture (the prediction skips it).
    let fallback = refs
        .ref_pic(0, 0)
        .or_else(|| refs.ref_pic(1, 0))
        .or(curr_snapshot)
        .ok_or(ReconError::InterNotSupported)?;
    Ok(ResolvedList {
        pred_flag: false,
        mv_l: [0, 0],
        mv_c: [0, 0],
        ref_pic: fallback,
    })
}

/// §8.5.3.2.1 — resolve one inter CU's per-PU motion and write it into
/// `field`, then reconstruct its samples into `pic`.
///
/// This composes [`resolve_cu_motion`] (the candidate derivation reading the
/// in-progress `field`) with [`extract_cu_residual`] + [`reconstruct_inter_cu`].
/// `pus` are the parsed §7.3.8.6 prediction units; `ctx` carries the
/// reference-picture resolvers + slice context; `available` is the §6.4.2
/// prediction-block availability test; `wp` the slice's §8.5.3.3.4.3
/// weighted-prediction tables (`None` for the default combine).
///
/// # Errors
/// Propagates [`ReconError`] from the residual extraction / reconstruction.
#[allow(clippy::too_many_arguments)]
pub fn resolve_and_reconstruct_inter_cu(
    pic: &mut Picture,
    field: &mut MotionField,
    params: &ReconParams,
    cu: &CodingUnit,
    pus: &[PredictionUnit],
    ctx: &PuMvContext,
    available: &dyn Fn(i32, i32) -> bool,
    refs: &RefListAccess,
    qp_y: i32,
    wp: Option<&SliceWpTables>,
) -> Result<(), ReconError> {
    let n_cb_s = 1usize << cu.log2_cb_size;
    let desc = InterCuDesc {
        x0: cu.x0 as usize,
        y0: cu.y0 as usize,
        n_cb_s,
        part_mode: cu.part_mode.into(),
    };
    let motions = resolve_cu_motion(field, desc, pus, ctx, available);
    let rects = crate::hevc::engine::pu_mv::pu_partitions(
        cu.x0 as usize,
        cu.y0 as usize,
        n_cb_s,
        cu.part_mode.into(),
    );

    let residual = extract_cu_residual(
        params,
        cu.transform_tree.as_ref(),
        cu.x0 as usize,
        cu.y0 as usize,
        n_cb_s,
        qp_y,
        cu.cu_transquant_bypass_flag,
    )?;

    // §8.7.2.4 — mark the per-4×4 cells covered by a transform block with a
    // coded luma coefficient so the deblocking boundary-strength `cbf` test
    // (bS = 1 at a transform-block edge with a non-zero coefficient) reads
    // them.
    if let Some(tree) = cu.transform_tree.as_ref() {
        mark_nonzero_luma(field, tree, cu.x0 as usize, cu.y0 as usize, cu.log2_cb_size);
    }

    reconstruct_inter_cu(pic, params, cu, &rects, &motions, &residual, refs, wp)
}

/// Walk an inter CU's transform tree and mark each leaf transform block that
/// carries a non-zero luma coefficient into the motion field
/// ([`MotionField::mark_nonzero_coeff`]).
fn mark_nonzero_luma(
    field: &mut MotionField,
    tree: &crate::hevc::engine::transform_tree::TransformTree,
    x0: usize,
    y0: usize,
    log2_trafo_size: u32,
) {
    use crate::hevc::engine::transform_tree::TransformTree;
    let n = 1usize << log2_trafo_size;
    match tree {
        TransformTree::Leaf { cbf_luma, unit } => {
            let has_coeff = *cbf_luma
                && unit
                    .residual_luma
                    .as_ref()
                    .is_some_and(|rb| rb.levels.iter().any(|&l| l != 0));
            if has_coeff {
                field.mark_nonzero_coeff(x0, y0, n, n);
            }
        }
        TransformTree::Split { children, .. } => {
            let half = n / 2;
            let offsets = [(0, 0), (half, 0), (0, half), (half, half)];
            for (child, (dx, dy)) in children.iter().zip(offsets) {
                mark_nonzero_luma(field, child, x0 + dx, y0 + dy, log2_trafo_size - 1);
            }
        }
    }
}

/// Slice-level inputs constant across one P / B slice's inter
/// reconstruction — every field a §8.5.3.2 / §8.3 input that does not vary
/// per coding unit. The picture-level driver
/// ([`reconstruct_inter_picture`]) binds the [`PuMvContext`] reference
/// resolvers to the [`RefListAccess`] + collocated field internally.
#[derive(Debug, Clone)]
pub struct InterSliceContext {
    /// `PicOrderCntVal` of the current picture.
    pub curr_poc: i32,
    /// `true` for a B slice (enables L1 + the §8.5.3.2.4 combined step).
    pub slice_is_b: bool,
    /// `CtbLog2SizeY`.
    pub ctb_log2_size_y: u32,
    /// `pic_width_in_luma_samples`.
    pub pic_width_luma: u32,
    /// `pic_height_in_luma_samples`.
    pub pic_height_luma: u32,
    /// `MaxNumMergeCand` (§7.4.7.1).
    pub max_num_merge_cand: usize,
    /// `num_ref_idx_l0_active`.
    pub num_ref_idx_l0_active: i32,
    /// `num_ref_idx_l1_active`.
    pub num_ref_idx_l1_active: i32,
    /// `Log2ParMrgLevel` (§7.4.3.3.1).
    pub log2_par_mrg_level: u32,
    /// `slice_temporal_mvp_enabled_flag`.
    pub temporal_mvp_enabled: bool,
    /// `collocated_from_l0_flag`.
    pub collocated_from_l0_flag: bool,
    /// `PicOrderCnt( ColPic )` (§8.5.3.2.9).
    pub col_poc: i32,
    /// `NoBackwardPredFlag` (§8.3.5).
    pub no_backward_pred: bool,
    /// `MinTbLog2SizeY` (the transform-block grid base).
    pub min_tb_log2_size_y: u32,
    /// `Log2MinCuQpDeltaSize` (§7.4.3.3.1) — the §8.6.1
    /// quantization-group size.
    pub log2_min_cu_qp_delta_size: u32,
    /// `entropy_coding_sync_enabled_flag` — the §8.6.1 third bullet
    /// resets `qPY_PREV` to `SliceQpY` at the first quantization group
    /// of each CTB row.
    pub wpp_qp_row_reset: bool,
    /// `slice_loop_filter_across_slices_enabled_flag` (§8.7.2.1 /
    /// §8.7.3.2 — slice-boundary edges are filtered only when set).
    pub filter_across_slices: bool,
    /// `loop_filter_across_tiles_enabled_flag`.
    pub filter_across_tiles: bool,
    /// `slice_deblocking_filter_disabled_flag == 0` — run the §8.7.2
    /// in-loop deblocking pass after reconstruction.
    pub deblock_enabled: bool,
    /// `slice_beta_offset_div2` (§8.7.2.5.3).
    pub beta_offset_div2: i32,
    /// `slice_tc_offset_div2` (§8.7.2.5.3).
    pub tc_offset_div2: i32,
    /// `SliceQpY` — the per-CU QP for the single-quantization-group case
    /// (the deblocking β/tC derivation reads it).
    pub slice_qp_y: i32,
    /// `pps_cb_qp_offset + slice_cb_qp_offset`.
    pub cb_qp_offset: i32,
    /// `pps_cr_qp_offset + slice_cr_qp_offset`.
    pub cr_qp_offset: i32,
    /// `pps_cb_qp_offset` alone — the §8.7.2.5.1/.2 `cQpPicOffset`
    /// input to chroma deblocking, which per the invocation text
    /// excludes the slice-level and CU-level chroma QP adjustments.
    pub pps_cb_qp_offset: i32,
    /// `pps_cr_qp_offset` alone (chroma deblocking `cQpPicOffset`).
    pub pps_cr_qp_offset: i32,
    /// `slice_sao_luma_flag` (§8.7.3.1 luma gate).
    pub slice_sao_luma_flag: bool,
    /// `slice_sao_chroma_flag` (§8.7.3.1 chroma gate).
    pub slice_sao_chroma_flag: bool,
    /// `log2_sao_offset_scale_luma` (§7.4.3.3.2; 0 for 8-bit Main).
    pub log2_sao_offset_scale_luma: u8,
    /// `log2_sao_offset_scale_chroma`.
    pub log2_sao_offset_scale_chroma: u8,
    /// `constrained_intra_pred_flag` (§8.4.4.2.1) — intra prediction
    /// inside this picture may not reference non-intra coding units.
    pub constrained_intra_pred: bool,
    /// The slice's §8.5.3.3.4.3 explicit weighted-prediction tables —
    /// `Some` exactly when `weightedPredFlag` (§8.5.3.3.4.1) is 1.
    pub wp: Option<SliceWpTables>,
    /// `pcm_loop_filter_disabled_flag` (§7.4.3.2.1) — with it set, the
    /// §8.7 loop filters must not modify the reconstructed samples of
    /// `pcm_flag == 1` coding units.
    pub pcm_loop_filter_disabled: bool,
    /// `use_integer_mv_flag` (§7.4.7.1) — the eqs 8-98..8-101 /
    /// 8-124..8-125 integer motion-vector resolution.
    pub use_integer_mv: bool,
    /// `TwoVersionsOfCurrDecPicFlag` (§7.4.3.3.3 eq. 7-40) — gates the
    /// §8.5.3.2.1 eqs 8-102/8-103 8×8 bi→uni reduction.
    pub two_versions_curr_pic: bool,
}

/// One placed coding tree unit for the §8.5 picture-level inter driver.
#[derive(Debug)]
pub struct PlacedInterCtu<'a> {
    /// CTB luma top-left x.
    pub x_ctb: u32,
    /// CTB luma top-left y.
    pub y_ctb: u32,
    /// `SliceAddrRs` of the independent slice segment owning this CTB.
    pub slice_addr_rs: u32,
    /// `slice_loop_filter_across_slices_enabled_flag` of the slice
    /// owning this CTB (§7.4.7.1 — per-slice, not per-picture).
    pub filter_across_slices: bool,
    /// The decoded coding tree unit.
    pub ctu: &'a crate::hevc::engine::slice_data::CodingTreeUnit,
}

/// §8.5 — reconstruct a full P / B picture from its decoded CTUs.
///
/// Walks the placed CTUs in decode order, dispatching each leaf coding unit:
/// an intra CU goes through the §8.4 intra path
/// ([`crate::hevc::engine::recon::reconstruct_intra_cu_ctx`]), an inter CU through
/// [`resolve_and_reconstruct_inter_cu`] (§8.5.3.2 candidate derivation from
/// the in-progress motion field + the collocated `col_field`, then §8.5.3.3
/// motion-compensated reconstruction). The §6.4.2 prediction-block
/// availability the candidate derivation needs is evaluated against the
/// shared [`crate::hevc::engine::recon::ReconCtx`] tiling + the per-cell intra / inter
/// flag of the motion field built up so far.
///
/// Returns the reconstructed picture and its per-PU motion field (the
/// §8.5.3.2.9 collocated arrays a later picture's temporal MVP reads). The
/// returned picture is the full in-loop-filtered output: when
/// `slice.deblock_enabled` the §8.7.2 deblocking pass runs first, then the
/// §8.7.3 SAO pass (a no-op when both slice SAO flags are clear).
///
/// # Errors
/// Propagates [`ReconError`] from the per-CU reconstruction.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_inter_picture(
    pic_width_luma: usize,
    pic_height_luma: usize,
    params: &ReconParams,
    slice: &InterSliceContext,
    tiles: &crate::hevc::engine::availability::TilingParams,
    ctus: &[PlacedInterCtu<'_>],
    refs: &RefListAccess,
    col_field: Option<&MotionField>,
) -> Result<(Picture, MotionField), ReconError> {
    let mut pic = Picture::new(
        pic_width_luma,
        pic_height_luma,
        params.chroma_array_type,
        params.bit_depth_luma,
        params.bit_depth_chroma,
    );
    let mut ctx = crate::hevc::engine::recon::ReconCtx::new(
        pic_width_luma,
        pic_height_luma,
        slice.ctb_log2_size_y,
        slice.min_tb_log2_size_y,
        tiles,
    )?;
    // §8.6.1 QP-derivation state (QpBdOffsetY = 6 * bit_depth_minus8).
    ctx.init_qp_state(
        slice.slice_qp_y,
        slice.log2_min_cu_qp_delta_size,
        6 * (i32::from(params.bit_depth_luma) - 8),
    );
    ctx.set_constrained_intra(slice.constrained_intra_pred);
    let mut field = MotionField::new(pic_width_luma, pic_height_luma);

    let ctb_size = 1usize << slice.ctb_log2_size_y;
    let pic_w_ctbs = pic_width_luma.div_ceil(ctb_size);
    let pic_h_ctbs = pic_height_luma.div_ceil(ctb_size);
    let mut slice_addr_map = vec![0u32; pic_w_ctbs * pic_h_ctbs];
    // Per-CTB slice_loop_filter_across_slices_enabled_flag (§7.4.7.1 —
    // a per-slice value; the §8.7.2.1 / §8.7.3.2 boundary gates consult
    // the owning slice's flag, not a picture-level one).
    let mut filter_across_map = vec![true; pic_w_ctbs * pic_h_ctbs];
    for placed in ctus {
        let rx = (placed.x_ctb as usize) >> slice.ctb_log2_size_y;
        let ry = (placed.y_ctb as usize) >> slice.ctb_log2_size_y;
        slice_addr_map[ry * pic_w_ctbs + rx] = placed.slice_addr_rs;
        filter_across_map[ry * pic_w_ctbs + rx] = placed.filter_across_slices;
    }
    ctx.set_slice_addr_rs(slice_addr_map.clone());

    // §8.5.3.2 reference-picture resolvers, bound to the §8.3.4 ref lists.
    // The CURR_PIC sentinel resolves to the current POC and — per the
    // §8.3.1 "the current decoded picture is marked as used for
    // long-term reference" clause — reads as a long-term reference.
    let ref_poc = |list: usize, ref_idx: i32| {
        if refs.is_curr_pic(list, ref_idx) {
            slice.curr_poc
        } else {
            refs.ref_poc(list, ref_idx)
        }
    };
    let ref_long_term = |list: usize, ref_idx: i32| {
        refs.is_curr_pic(list, ref_idx)
            || refs
                .entry(list, ref_idx)
                .is_some_and(|e| e.marking == crate::hevc::engine::dpb::Marking::LongTerm)
    };
    let ref_short_term = |list: usize, ref_idx: i32| {
        refs.entry(list, ref_idx)
            .is_some_and(|e| e.marking == crate::hevc::engine::dpb::Marking::ShortTerm)
    };
    let col_ref_long_term = |_poc: i32| false;
    let is_curr_pic = |list: usize, ref_idx: i32| refs.is_curr_pic(list, ref_idx);

    let mv_ctx = PuMvContext {
        curr_poc: slice.curr_poc,
        slice_is_b: slice.slice_is_b,
        ctb_log2_size_y: slice.ctb_log2_size_y,
        pic_width_luma: slice.pic_width_luma,
        pic_height_luma: slice.pic_height_luma,
        max_num_merge_cand: slice.max_num_merge_cand,
        num_ref_idx_l0_active: slice.num_ref_idx_l0_active,
        num_ref_idx_l1_active: slice.num_ref_idx_l1_active,
        log2_par_mrg_level: slice.log2_par_mrg_level,
        temporal_mvp_enabled: slice.temporal_mvp_enabled,
        collocated_from_l0_flag: slice.collocated_from_l0_flag,
        col_poc: slice.col_poc,
        no_backward_pred: slice.no_backward_pred,
        ref_poc: &ref_poc,
        ref_long_term: &ref_long_term,
        ref_short_term: &ref_short_term,
        col_field,
        col_ref_long_term: &col_ref_long_term,
        use_integer_mv: slice.use_integer_mv,
        two_versions_curr_pic: slice.two_versions_curr_pic,
        is_curr_pic: &is_curr_pic,
    };

    let mut deblock_cus: Vec<crate::hevc::engine::deblock::DeblockCuDesc> = Vec::new();
    // §8.7.2.5.4 / §8.7.3.1 — per-4×4-cell loop-filter suppression for
    // PCM (`pcm_loop_filter_disabled_flag`) / transquant-bypass CUs.
    let (w4, h4) = (pic_width_luma.div_ceil(4), pic_height_luma.div_ceil(4));
    let mut no_filter_cells = vec![false; w4 * h4];
    // Motion derivation needs an immutable prediction-mode view while it
    // mutates resolved motion cells. Maintain that view once per picture
    // instead of rebuilding a full-grid snapshot for every inter CU.
    let mut intra_cells = vec![true; w4 * h4];
    let mut prev_slice_addr: Option<u32> = None;
    let mut prev_tile: Option<u32> = None;
    for placed in ctus {
        // §8.6.1 step-1 — qPY_PREV resets to SliceQpY at the first
        // quantization group of a slice, of a tile, and (with
        // entropy_coding_sync) of each CTB row of a tile.
        let rx = (placed.x_ctb as usize) >> slice.ctb_log2_size_y;
        let ry = (placed.y_ctb as usize) >> slice.ctb_log2_size_y;
        let rs = (ry * pic_w_ctbs + rx) as u32;
        let tiling = ctx.tiling();
        let tile = tiling.tile_id(tiling.ctb_addr_rs_to_ts(rs));
        let tile_row_start = rx == 0 || tiling.tile_id(tiling.ctb_addr_rs_to_ts(rs - 1)) != tile;
        if prev_slice_addr != Some(placed.slice_addr_rs)
            || prev_tile != Some(tile)
            || (slice.wpp_qp_row_reset && tile_row_start)
        {
            ctx.reset_qp_prev();
        }
        // §7.4.7.1 — CuQpOffsetCb / CuQpOffsetCr reset to 0 at each
        // slice start (unlike qPY_PREV they do NOT reset at tile /
        // WPP-row boundaries).
        if prev_slice_addr != Some(placed.slice_addr_rs) {
            params.cu_qp_offset_c.set((0, 0));
        }
        prev_slice_addr = Some(placed.slice_addr_rs);
        prev_tile = Some(tile);
        reconstruct_inter_quadtree(
            &mut pic,
            &mut ctx,
            &mut field,
            params,
            &mv_ctx,
            refs,
            slice,
            &mut deblock_cus,
            &mut no_filter_cells,
            &mut intra_cells,
            w4,
            &filter_across_map,
            &placed.ctu.quadtree,
        )?;
    }
    let no_filter_map =
        no_filter_cells
            .iter()
            .any(|&b| b)
            .then_some(crate::hevc::engine::deblock::NoFilterMap {
                cells: &no_filter_cells,
                w_cells: w4,
            });

    // §8.7.2 — in-loop deblocking (all vertical edges, then horizontal),
    // ahead of the §8.7.3 SAO pass.
    if slice.deblock_enabled {
        // Issue #189 stage attribution: one scope per picture, since the
        // in-loop filters run as whole-picture passes rather than per block.
        let _profile =
            crate::hevc::engine::profile::scope(crate::hevc::engine::profile::Stage::Deblock);
        let qp_map = ctx
            .qp_cells()
            .map(|(cells, w_cells)| crate::hevc::engine::deblock::QpMap { cells, w_cells });
        crate::hevc::engine::deblock::deblock_picture_full(
            &mut pic,
            &field,
            &deblock_cus,
            qp_map,
            no_filter_map.as_ref(),
        );
    }

    // §8.7.3 — sample-adaptive offset (on the deblocked samples). Resolve
    // each CTB's §7.4.9.3 SAO parameters with left / above merge (denied
    // across slice boundaries), then run the picture-level filter.
    let mut sao_grid = vec![crate::hevc::engine::sao::ResolvedSao::off(); pic_w_ctbs * pic_h_ctbs];
    for placed in ctus {
        let rx = (placed.x_ctb as usize) >> slice.ctb_log2_size_y;
        let ry = (placed.y_ctb as usize) >> slice.ctb_log2_size_y;
        let here = slice_addr_map[ry * pic_w_ctbs + rx];
        if let Some(sao_params) = &placed.ctu.sao {
            let left = (rx > 0 && slice_addr_map[ry * pic_w_ctbs + (rx - 1)] == here)
                .then(|| sao_grid[ry * pic_w_ctbs + (rx - 1)]);
            let above = (ry > 0 && slice_addr_map[(ry - 1) * pic_w_ctbs + rx] == here)
                .then(|| sao_grid[(ry - 1) * pic_w_ctbs + rx]);
            sao_grid[ry * pic_w_ctbs + rx] = crate::hevc::engine::sao::ResolvedSao::resolve(
                sao_params,
                left.as_ref(),
                above.as_ref(),
                slice.log2_sao_offset_scale_luma,
                slice.log2_sao_offset_scale_chroma,
            );
        }
    }
    let sao_boundaries = crate::hevc::engine::sao::SaoBoundaries {
        slice_addr_of_ctb: slice_addr_map.clone(),
        tile_id_of_ctb: (0..(pic_w_ctbs * pic_h_ctbs) as u32)
            .map(|rs| {
                let tiling = ctx.tiling();
                tiling.tile_id(tiling.ctb_addr_rs_to_ts(rs))
            })
            .collect(),
        pic_w_ctbs,
        ctb_log2_size_y: slice.ctb_log2_size_y,
        across_slices: slice.filter_across_slices,
        across_tiles: slice.filter_across_tiles,
        filter_across_of_ctb: Some(filter_across_map.clone()),
        ctb_ts_of_rs: Some(
            (0..(pic_w_ctbs * pic_h_ctbs) as u32)
                .map(|rs| ctx.tiling().ctb_addr_rs_to_ts(rs))
                .collect(),
        ),
    };
    let _sao_profile =
        crate::hevc::engine::profile::scope(crate::hevc::engine::profile::Stage::Sao);
    let filtered = crate::hevc::engine::sao::apply_sao_picture_full(
        pic,
        &sao_grid,
        slice.ctb_log2_size_y,
        params.chroma_array_type,
        slice.slice_sao_luma_flag,
        slice.slice_sao_chroma_flag,
        Some(&sao_boundaries),
        no_filter_map.as_ref(),
    );

    Ok((filtered, field))
}

/// Walk one §7.3.8.4 coding quadtree, dispatching each leaf coding unit to
/// the intra or inter reconstruction path.
#[allow(clippy::too_many_arguments)]
fn reconstruct_inter_quadtree(
    pic: &mut Picture,
    ctx: &mut crate::hevc::engine::recon::ReconCtx,
    field: &mut MotionField,
    params: &ReconParams,
    mv_ctx: &PuMvContext,
    refs: &RefListAccess,
    slice: &InterSliceContext,
    deblock_cus: &mut Vec<crate::hevc::engine::deblock::DeblockCuDesc>,
    no_filter_cells: &mut [bool],
    intra_cells: &mut [bool],
    w4: usize,
    filter_across_of_ctb: &[bool],
    qt: &crate::hevc::engine::slice_data::CodingQuadtree,
) -> Result<(), ReconError> {
    use crate::hevc::engine::slice_data::CodingQuadtree;
    match qt {
        CodingQuadtree::Split(children) => {
            for child in children {
                reconstruct_inter_quadtree(
                    pic,
                    ctx,
                    field,
                    params,
                    mv_ctx,
                    refs,
                    slice,
                    deblock_cus,
                    no_filter_cells,
                    intra_cells,
                    w4,
                    filter_across_of_ctb,
                    child,
                )?;
            }
            Ok(())
        }
        CodingQuadtree::Leaf(cu) => {
            // §8.7.2.5.4 / §8.7.3.1 — mark the CU's cells when the loop
            // filters must leave its reconstructed samples untouched.
            if (cu.pcm_flag && slice.pcm_loop_filter_disabled) || cu.cu_transquant_bypass_flag {
                let n = 1usize << cu.log2_cb_size;
                for j in (0..n).step_by(4) {
                    for i in (0..n).step_by(4) {
                        let (gx, gy) = ((cu.x0 as usize + i) >> 2, (cu.y0 as usize + j) >> 2);
                        if let Some(cell) = no_filter_cells.get_mut(gy * w4 + gx) {
                            *cell = true;
                        }
                    }
                }
            }
            reconstruct_inter_leaf_cu(
                pic,
                ctx,
                field,
                params,
                mv_ctx,
                refs,
                slice,
                deblock_cus,
                intra_cells,
                w4,
                filter_across_of_ctb,
                cu,
            )
        }
    }
}

/// Build the §8.7.2 [`crate::hevc::engine::deblock::DeblockCuDesc`] for one coding unit
/// (its geometry, transform-split topology, partition mode, QP context, and
/// the CB-boundary edge-flag gates) and append it to `deblock_cus`.
#[allow(clippy::too_many_arguments)]
fn collect_deblock_cu(
    cu: &CodingUnit,
    slice: &InterSliceContext,
    chroma_array_type: u8,
    bit_depth_luma: u8,
    bit_depth_chroma: u8,
    qp_y: i32,
    qp_y_p_left: i32,
    qp_y_p_top: i32,
    filter_left: bool,
    filter_top: bool,
    deblock_cus: &mut Vec<crate::hevc::engine::deblock::DeblockCuDesc>,
) {
    let cu_params = crate::hevc::engine::deblock::DeblockCuParams {
        qp_y,
        beta_offset_div2: slice.beta_offset_div2,
        tc_offset_div2: slice.tc_offset_div2,
        cb_qp_offset: slice.pps_cb_qp_offset,
        cr_qp_offset: slice.pps_cr_qp_offset,
        bit_depth_luma,
        bit_depth_chroma,
        chroma_array_type,
    };
    deblock_cus.push(crate::hevc::engine::deblock::DeblockCuDesc {
        cu: crate::hevc::engine::deblock::DeblockCu {
            x_cb: cu.x0 as usize,
            y_cb: cu.y0 as usize,
            log2_cb_size: cu.log2_cb_size,
            params: cu_params,
            qp_y_p_left,
            qp_y_p_top,
        },
        transform_split: crate::hevc::engine::deblock::TransformSplit::from_tree(
            cu.transform_tree.as_ref(),
        ),
        part_mode: cu.part_mode,
        // §8.7.2.1 — the CB-boundary edges are filtered except at the
        // picture's left / top border (the single-slice / single-tile case;
        // a slice / tile boundary with loop-filter-across disabled would
        // additionally clear these, threaded by the caller).
        filter_left,
        filter_top,
    });
}

/// Reconstruct one leaf coding unit (intra → §8.4 path + intra-stamp the
/// motion field; inter → §8.5 path), and collect its deblocking descriptor.
#[allow(clippy::too_many_arguments)]
fn reconstruct_inter_leaf_cu(
    pic: &mut Picture,
    ctx: &mut crate::hevc::engine::recon::ReconCtx,
    field: &mut MotionField,
    params: &ReconParams,
    mv_ctx: &PuMvContext,
    refs: &RefListAccess,
    slice: &InterSliceContext,
    deblock_cus: &mut Vec<crate::hevc::engine::deblock::DeblockCuDesc>,
    intra_cells: &mut [bool],
    w4: usize,
    filter_across_of_ctb: &[bool],
    cu: &CodingUnit,
) -> Result<(), ReconError> {
    use crate::hevc::engine::binarization::CuPredMode;
    let n_cb_s = 1usize << cu.log2_cb_size;
    // §8.6.1 — the CU's QpY (in decode order, before anything reads the
    // QP map for this CU).
    let cu_delta = cu
        .transform_tree
        .as_ref()
        .and_then(crate::hevc::engine::recon::first_tree_cu_qp_delta);
    let qp_y = ctx.derive_cu_qp(
        params,
        cu.x0 as usize,
        cu.y0 as usize,
        cu.log2_cb_size,
        cu_delta,
    );
    if slice.deblock_enabled {
        // Deblocking p-side QPs from the already-stamped neighbour map
        // (§8.7.2.5.3 QpP); the picture boundary rows fall back to the
        // CU's own QP (those edges are not filtered).
        let x0 = cu.x0 as usize;
        let y0 = cu.y0 as usize;
        let qp_p_left = if x0 > 0 {
            ctx.qp_y_at(x0 - 1, y0).unwrap_or(slice.slice_qp_y)
        } else {
            qp_y
        };
        let qp_p_top = if y0 > 0 {
            ctx.qp_y_at(x0, y0 - 1).unwrap_or(slice.slice_qp_y)
        } else {
            qp_y
        };
        // §8.7.2.1 filterLeftCbEdgeFlag / filterTopCbEdgeFlag: a
        // picture-boundary edge is never filtered; a slice / tile
        // boundary edge only when filtering across it is enabled.
        // §8.7.2.1 — the slice-boundary gate consults the CURRENT
        // slice's slice_loop_filter_across_slices_enabled_flag (the
        // slice containing this coding block), a per-slice value.
        let pic_w_ctbs = (slice.pic_width_luma as usize).div_ceil(1 << slice.ctb_log2_size_y);
        let cu_across = {
            let rs = (y0 >> slice.ctb_log2_size_y) * pic_w_ctbs + (x0 >> slice.ctb_log2_size_y);
            filter_across_of_ctb.get(rs).copied().unwrap_or(true)
        };
        let boundary_ok = |x_nb: usize, y_nb: usize| -> bool {
            let same_slice =
                ctx.slice_addr_rs_of_luma(x0, y0) == ctx.slice_addr_rs_of_luma(x_nb, y_nb);
            let same_tile = ctx.tile_id_of_luma(x0, y0) == ctx.tile_id_of_luma(x_nb, y_nb);
            (same_slice || cu_across) && (same_tile || slice.filter_across_tiles)
        };
        let filter_left = x0 != 0 && boundary_ok(x0 - 1, y0);
        let filter_top = y0 != 0 && boundary_ok(x0, y0 - 1);
        collect_deblock_cu(
            cu,
            slice,
            params.chroma_array_type,
            params.bit_depth_luma,
            params.bit_depth_chroma,
            qp_y,
            qp_p_left,
            qp_p_top,
            filter_left,
            filter_top,
            deblock_cus,
        );
    }
    if matches!(cu.cu_pred_mode, CuPredMode::Intra) {
        // §8.4 intra reconstruction; stamp the motion field intra so a
        // later inter CU's §6.4.2 availability denies it as a candidate.
        crate::hevc::engine::recon::reconstruct_intra_cu_ctx(pic, params, ctx, cu)?;
        field.fill_rect(
            cu.x0 as usize,
            cu.y0 as usize,
            n_cb_s,
            n_cb_s,
            crate::hevc::engine::motion::MotionCell {
                is_intra: true,
                ..crate::hevc::engine::motion::MotionCell::default()
            },
        );
        // §8.7.2.4 — mark the intra CU's coded transform blocks (intra
        // neighbours give bS = 2 regardless, but the cbf flag is read for
        // an inter-side q neighbour at the shared edge).
        if let Some(tree) = cu.transform_tree.as_ref() {
            mark_nonzero_luma(field, tree, cu.x0 as usize, cu.y0 as usize, cu.log2_cb_size);
        }
        return Ok(());
    }

    // §6.4.2 prediction-block availability against the shared tiling and the
    // incrementally maintained prediction-mode grid. Motion derivation can
    // read this independent grid while it mutates `field`.
    let x_cb = cu.x0;
    let y_cb = cu.y0;
    let h4 = field.height_4();
    let bx0 = cu.x0 as usize / 4;
    let by0 = cu.y0 as usize / 4;
    let bx1 = (cu.x0 as usize + n_cb_s).min(w4 * 4).div_ceil(4);
    let by1 = (cu.y0 as usize + n_cb_s).min(h4 * 4).div_ceil(4);
    for by in by0..by1 {
        for bx in bx0..bx1 {
            intra_cells[by * w4 + bx] = false;
        }
    }
    let tiling = ctx.tiling();
    let available = |x_nb: i32, y_nb: i32| -> bool {
        let cu_pred_mode = |x: u32, y: u32| -> u8 {
            // §6.4.2 final mask reads CuPredMode[ xNbY ][ yNbY ]. A
            // location covered by the CURRENT (inter) coding block has
            // CuPredMode == MODE_INTER by definition; the z-scan /
            // sameCb steps already guard decode order, so only the
            // earlier-partition region of this CU is ever consulted
            // (e.g. the §8.5.3.2.7 AMVP neighbours of a 2NxN / Nx2N
            // second partition read the first partition's motion). The
            // pre-CU snapshot below would otherwise report the
            // motion-field background (intra) there.
            if (x_cb..x_cb + n_cb_s as u32).contains(&x)
                && (y_cb..y_cb + n_cb_s as u32).contains(&y)
            {
                return 0;
            }
            let (gx, gy) = ((x as usize) / 4, (y as usize) / 4);
            if gx < w4 && gy < h4 && intra_cells[gy * w4 + gx] {
                crate::hevc::engine::availability::MODE_INTRA
            } else {
                0
            }
        };
        tiling.prediction_block_availability(
            x_cb,
            y_cb,
            n_cb_s as u32,
            x_cb,
            y_cb,
            n_cb_s as u32,
            n_cb_s as u32,
            0,
            x_nb,
            y_nb,
            |ctb_rs| ctx.slice_addr_rs_of(ctb_rs),
            cu_pred_mode,
        )
    };

    resolve_and_reconstruct_inter_cu(
        pic,
        field,
        params,
        cu,
        &cu.prediction_units,
        mv_ctx,
        &available,
        refs,
        qp_y,
        slice.wp.as_ref(),
    )
}

/// §8.3 + §8.5 — decode one inter (P / B) picture end to end against the
/// decoded-picture buffer, completing the per-picture reference cycle.
///
/// Ties the §8.3.1 → §8.3.2 → §8.3.4 → §8.3.5 reference derivation
/// ([`crate::hevc::engine::decode::PictureSequenceState::begin_picture`]) to the
/// picture-level inter reconstruction: it resolves `RefPicList0` /
/// `RefPicList1` + `ColPic` into the [`RefListAccess`] + collocated motion
/// field the inter driver reads, runs [`reconstruct_inter_picture`] (recon →
/// deblock → SAO), then inserts the reconstructed picture + its motion field
/// into the DPB ([`crate::hevc::engine::decode::PictureSequenceState::store_picture`]) as a
/// short-term reference for the next picture.
///
/// `header` / `slice_ref` carry the §8.3 inputs; `slice` the §8.5.3.2 /
/// §8.7 slice-constant inputs; `ctus` the decoded CTUs in decode order.
/// Returns the (output) reconstructed picture.
///
/// # Errors
/// [`ReconError::InterNotSupported`] when the picture is an I picture (no
/// reference lists were built — use the intra driver instead) or a used
/// reference resolves to "no reference picture"; otherwise the per-CU
/// reconstruction errors.
#[allow(clippy::too_many_arguments)]
pub fn decode_inter_picture(
    seq: &mut crate::hevc::engine::decode::PictureSequenceState,
    header: &crate::hevc::engine::decode::PictureHeaderInfo,
    slice_ref: &crate::hevc::engine::decode::SliceRefParams,
    pic_width_luma: usize,
    pic_height_luma: usize,
    params: &ReconParams,
    slice: &InterSliceContext,
    tiles: &crate::hevc::engine::availability::TilingParams,
    ctus: &[PlacedInterCtu<'_>],
) -> Result<Picture, ReconError> {
    // §8.3.1 → §8.3.5 — POC, RPS marking, reference lists, ColPic.
    let ref_state = seq.begin_picture(header, slice_ref);
    let lists = ref_state
        .ref_pic_lists
        .clone()
        .ok_or(ReconError::InterNotSupported)?;
    let layer_id = header.layer_id;
    let poc = ref_state.poc;

    // The collocated picture's motion field for the §8.5.3.2.9 temporal MVP.
    let col_field = ref_state
        .col_pic
        .map(|idx| &seq.dpb().entries()[idx].motion);

    let refs = RefListAccess {
        lists: &lists,
        entries: seq.dpb().entries(),
    };

    let (picture, motion) = reconstruct_inter_picture(
        pic_width_luma,
        pic_height_luma,
        params,
        slice,
        tiles,
        ctus,
        &refs,
        col_field,
    )?;

    // §8.3.2 — insert as a short-term reference for the following pictures.
    // (The output picture is returned to the caller before the move.)
    let output = picture.clone();
    seq.store_picture(poc, layer_id, picture, motion);
    Ok(output)
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::binarization::{CuPredMode, PartMode};
    use crate::hevc::engine::dpb::{Marking, RefPicLists};
    use crate::hevc::engine::motion::MotionField;
    use crate::hevc::engine::picture::{Picture, Plane};
    use crate::hevc::engine::pu_mv::PuMvContext;
    use crate::hevc::engine::residual::ResidualBlock;
    use crate::hevc::engine::slice_data::{CodingUnit, PredictionUnit};
    use crate::hevc::engine::transform_tree::TransformTree;
    use crate::hevc::engine::transform_unit::TransformUnit;

    fn p_params() -> ReconParams {
        ReconParams {
            chroma_array_type: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            intra_smoothing_disabled: false,
            strong_intra_smoothing_enabled: true,
            slice_qp_y: 25,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            act_y_qp_offset: 0,
            act_cb_qp_offset: 0,
            act_cr_qp_offset: 0,
            transform_skip_rotation_enabled: false,
            implicit_rdpcm_enabled: false,
            intra_boundary_filtering_disabled: false,
            extended_precision: false,
            scaling: None,
            chroma_qp_offset_list: Vec::new(),
            cu_qp_offset_c: core::cell::Cell::new((0, 0)),
        }
    }

    /// A 32×32 reference picture with a flat luma + chroma value.
    fn flat_ref(luma: i32, chroma: i32) -> Picture {
        let mut p = Picture::new(32, 32, 1, 8, 8);
        for y in 0..32 {
            for x in 0..32 {
                p.set_sample(Plane::Luma, x, y, luma);
            }
        }
        for y in 0..16 {
            for x in 0..16 {
                p.set_sample(Plane::Cb, x, y, chroma);
                p.set_sample(Plane::Cr, x, y, chroma);
            }
        }
        p
    }

    fn dpb_entry(poc: i32, pic: Picture) -> DpbEntry {
        DpbEntry {
            poc,
            layer_id: 0,
            marking: Marking::ShortTerm,
            picture: pic,
            motion: MotionField::new(32, 32),
        }
    }

    /// A merge-mode P prediction unit selecting merge candidate `idx`.
    fn merge_pu(idx: u8) -> PredictionUnit {
        PredictionUnit {
            merge_flag: true,
            merge_idx: Some(idx),
            inter_pred_idc: None,
            ref_idx_l0: None,
            mvd_l0: None,
            mvp_l0_flag: None,
            ref_idx_l1: None,
            mvd_l1: None,
            mvp_l1_flag: None,
        }
    }

    /// A 16×16 inter CU at (0,0), PART_2Nx2N, with one prediction unit and
    /// an optional flat luma DC residual.
    fn inter_cu_16(pu: PredictionUnit, luma_dc: Option<i32>) -> CodingUnit {
        let tree = luma_dc.map(|dc| {
            let mut levels = vec![0i32; 16 * 16];
            levels[0] = dc;
            TransformTree::Leaf {
                cbf_luma: true,
                unit: TransformUnit {
                    residual_luma: Some(ResidualBlock {
                        log2_trafo_size: 4,
                        last_sig_coeff_x: 0,
                        last_sig_coeff_y: 0,
                        levels,
                        transform_skip: false,
                        explicit_rdpcm_flag: false,
                        explicit_rdpcm_dir_flag: false,
                    }),
                    ..Default::default()
                },
            }
        });
        CodingUnit {
            x0: 0,
            y0: 0,
            log2_cb_size: 4,
            cu_pred_mode: CuPredMode::Inter,
            cu_transquant_bypass_flag: false,
            part_mode: PartMode::Part2Nx2N,
            pcm_flag: false,
            pcm: None,
            palette: None,
            prediction_units: vec![pu],
            intra_luma: vec![],
            intra_chroma_pred_mode: vec![],
            rqt_root_cbf: luma_dc.is_some(),
            transform_tree: tree,
        }
    }

    /// Build a single-reference P-slice context: one short-term reference at
    /// POC 0, current picture at POC 4, temporal MVP disabled.
    fn p_ctx<'a>(
        ref_poc: &'a dyn Fn(usize, i32) -> i32,
        long: &'a dyn Fn(usize, i32) -> bool,
        short: &'a dyn Fn(usize, i32) -> bool,
        col_long: &'a dyn Fn(i32) -> bool,
    ) -> PuMvContext<'a> {
        PuMvContext {
            curr_poc: 4,
            slice_is_b: false,
            ctb_log2_size_y: 4,
            pic_width_luma: 32,
            pic_height_luma: 32,
            max_num_merge_cand: 5,
            num_ref_idx_l0_active: 1,
            num_ref_idx_l1_active: 0,
            log2_par_mrg_level: 2,
            temporal_mvp_enabled: false,
            collocated_from_l0_flag: true,
            col_poc: 0,
            no_backward_pred: true,
            ref_poc,
            ref_long_term: long,
            ref_short_term: short,
            col_field: None,
            col_ref_long_term: col_long,
            use_integer_mv: false,
            two_versions_curr_pic: false,
            is_curr_pic: &|_, _| false,
        }
    }

    /// §8.5.3.1 intra block copy: a CU whose L0 reference is the
    /// CURRENT picture (CURR_PIC) copies already-reconstructed samples
    /// of the same picture. The 16×16 CU at (16, 0) signals an AMVP MV
    /// of −16 luma samples (mvd −16 on the eq. 8-98 integer path), so
    /// it reconstructs to a copy of the (0, 0) block — luma and both
    /// chroma planes.
    #[test]
    fn ibc_cu_copies_current_picture_block() {
        let params = p_params();
        let entries: Vec<DpbEntry> = Vec::new();
        let lists = RefPicLists {
            list0: vec![Some(crate::hevc::engine::dpb::CURR_PIC)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };

        let ref_poc = |_l: usize, _r: i32| 4i32;
        let long = |_l: usize, _r: i32| true;
        let short = |_l: usize, _r: i32| false;
        let col_long = |_p: i32| false;
        let is_curr = |l: usize, r: i32| l == 0 && r == 0;
        let mut ctx = p_ctx(&ref_poc, &long, &short, &col_long);
        ctx.is_curr_pic = &is_curr;

        let mvd = |v: i32| crate::hevc::engine::binarization::MvdComponent {
            greater0_flag: u8::from(v != 0),
            greater1_flag: None,
            minus2: None,
            sign_flag: None,
            value: v,
        };
        let pu = PredictionUnit {
            merge_flag: false,
            merge_idx: None,
            inter_pred_idc: Some(crate::hevc::engine::binarization::InterPredIdc::PredL0),
            ref_idx_l0: Some(0),
            mvd_l0: Some([mvd(-16), mvd(0)]),
            mvp_l0_flag: Some(0),
            ref_idx_l1: None,
            mvd_l1: None,
            mvp_l1_flag: None,
        };
        let mut cu = inter_cu_16(pu, None);
        cu.x0 = 16;

        let mut field = MotionField::new(32, 32);
        // Seed the current picture with a distinct pattern; the CU
        // region starts at the 8-bit mid-level.
        let mut pic = Picture::new(32, 32, 1, 8, 8);
        for y in 0..32usize {
            for x in 0..32usize {
                pic.set_sample(Plane::Luma, x, y, ((x * 7 + y * 3) % 200) as i32);
            }
        }
        for y in 0..16usize {
            for x in 0..16usize {
                pic.set_sample(Plane::Cb, x, y, ((x * 5 + y * 11) % 180) as i32);
                pic.set_sample(Plane::Cr, x, y, ((x * 13 + y * 2) % 160) as i32);
            }
        }
        let available = |_x: i32, _y: i32| false;
        resolve_and_reconstruct_inter_cu(
            &mut pic,
            &mut field,
            &params,
            &cu,
            &cu.prediction_units,
            &ctx,
            &available,
            &refs,
            25,
            None,
        )
        .unwrap();

        for y in 0..16usize {
            for x in 0..16usize {
                assert_eq!(
                    pic.sample(Plane::Luma, 16 + x, y),
                    pic.sample(Plane::Luma, x, y),
                    "luma ({x},{y})"
                );
            }
        }
        for y in 0..8usize {
            for x in 0..8usize {
                assert_eq!(
                    pic.sample(Plane::Cb, 8 + x, y),
                    pic.sample(Plane::Cb, x, y),
                    "cb ({x},{y})"
                );
                assert_eq!(
                    pic.sample(Plane::Cr, 8 + x, y),
                    pic.sample(Plane::Cr, x, y),
                    "cr ({x},{y})"
                );
            }
        }
    }

    /// A merge PU with no spatial / temporal neighbours falls through to the
    /// §8.5.3.2.5 zero-MV candidate (mvL0 = 0, refIdxL0 = 0), so the PU
    /// reconstructs to the reference picture's co-located samples.
    #[test]
    fn merge_zero_mv_p_cu_reconstructs_reference() {
        let params = p_params();
        let refpic = flat_ref(100, 120);
        let entries = vec![dpb_entry(0, refpic)];
        let lists = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };

        let ref_poc = |_l: usize, _r: i32| 0i32;
        let long = |_l: usize, _r: i32| false;
        let short = |_l: usize, _r: i32| true;
        let col_long = |_p: i32| false;
        let ctx = p_ctx(&ref_poc, &long, &short, &col_long);

        let cu = inter_cu_16(merge_pu(0), None);
        let mut field = MotionField::new(32, 32);
        let mut pic = Picture::new(32, 32, 1, 8, 8);
        // No neighbours available (single isolated CU at picture origin).
        let available = |_x: i32, _y: i32| false;
        resolve_and_reconstruct_inter_cu(
            &mut pic,
            &mut field,
            &params,
            &cu,
            &cu.prediction_units,
            &ctx,
            &available,
            &refs,
            25,
            None,
        )
        .unwrap();

        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Luma, x, y), 100, "luma ({x},{y})");
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(pic.sample(Plane::Cb, x, y), 120, "cb ({x},{y})");
                assert_eq!(pic.sample(Plane::Cr, x, y), 120, "cr ({x},{y})");
            }
        }
        // eqs 8-80..8-85: the CU's motion is written into the field.
        let cell = field.cell_at(0, 0);
        assert!(!cell.is_intra);
        assert!(cell.pred_flag_l0 && !cell.pred_flag_l1);
        assert_eq!(cell.mv_l0, [0, 0]);
    }

    /// The CU residual is dequantized + inverse-transformed and added onto
    /// the motion-compensated prediction.
    #[test]
    fn merge_zero_mv_p_cu_adds_residual() {
        let params = p_params();
        let refpic = flat_ref(100, 120);
        let entries = vec![dpb_entry(0, refpic)];
        let lists = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };
        let ref_poc = |_l: usize, _r: i32| 0i32;
        let long = |_l: usize, _r: i32| false;
        let short = |_l: usize, _r: i32| true;
        let col_long = |_p: i32| false;
        let ctx = p_ctx(&ref_poc, &long, &short, &col_long);

        // A DC luma residual produces a uniform offset over the 16×16 block.
        let cu = inter_cu_16(merge_pu(0), Some(40));
        let mut field = MotionField::new(32, 32);
        let mut pic = Picture::new(32, 32, 1, 8, 8);
        let available = |_x: i32, _y: i32| false;
        resolve_and_reconstruct_inter_cu(
            &mut pic,
            &mut field,
            &params,
            &cu,
            &cu.prediction_units,
            &ctx,
            &available,
            &refs,
            25,
            None,
        )
        .unwrap();

        // Uniform DC residual ⇒ every luma sample is prediction(100) + r for
        // the same r; assert the block is uniform and offset from 100.
        let v = pic.sample(Plane::Luma, 0, 0);
        assert_ne!(v, 100, "residual shifts the prediction");
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Luma, x, y), v, "uniform at ({x},{y})");
            }
        }
    }

    fn p_slice_ctx() -> InterSliceContext {
        InterSliceContext {
            log2_min_cu_qp_delta_size: 4,
            wpp_qp_row_reset: false,
            filter_across_slices: true,
            filter_across_tiles: true,
            curr_poc: 4,
            constrained_intra_pred: false,
            slice_is_b: false,
            ctb_log2_size_y: 5,
            pic_width_luma: 32,
            pic_height_luma: 32,
            max_num_merge_cand: 5,
            num_ref_idx_l0_active: 1,
            num_ref_idx_l1_active: 0,
            log2_par_mrg_level: 2,
            temporal_mvp_enabled: false,
            collocated_from_l0_flag: true,
            col_poc: 0,
            no_backward_pred: true,
            min_tb_log2_size_y: 2,
            deblock_enabled: false,
            beta_offset_div2: 0,
            tc_offset_div2: 0,
            slice_qp_y: 25,
            cb_qp_offset: 0,
            cr_qp_offset: 0,
            slice_sao_luma_flag: false,
            slice_sao_chroma_flag: false,
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,
            wp: None,
            pcm_loop_filter_disabled: false,
            use_integer_mv: false,
            two_versions_curr_pic: false,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
        }
    }

    /// The §8.5 picture-level driver reconstructs a single-CTU P picture
    /// whose one 32×32 inter merge CU (zero-MV fallback) copies the flat
    /// reference samples, and records the CU's motion into the returned
    /// field.
    #[test]
    fn picture_driver_single_inter_ctu_copies_reference() {
        let params = p_params();
        let refpic = flat_ref(77, 99);
        let entries = vec![dpb_entry(0, refpic)];
        let lists = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };

        // One 32×32 inter merge CU at (0,0) — a single coding tree unit
        // covering the whole picture.
        let mut cu = inter_cu_16(merge_pu(0), None);
        cu.log2_cb_size = 5;
        let ctu = crate::hevc::engine::slice_data::CodingTreeUnit {
            sao: None,
            quadtree: crate::hevc::engine::slice_data::CodingQuadtree::Leaf(Box::new(cu)),
        };
        let placed = vec![PlacedInterCtu {
            x_ctb: 0,
            y_ctb: 0,
            slice_addr_rs: 0,
            filter_across_slices: true,
            ctu: &ctu,
        }];

        let slice = p_slice_ctx();
        let tiles = crate::hevc::engine::availability::TilingParams::single_tile();
        let (pic, field) =
            reconstruct_inter_picture(32, 32, &params, &slice, &tiles, &placed, &refs, None)
                .unwrap();

        for y in 0..32 {
            for x in 0..32 {
                assert_eq!(pic.sample(Plane::Luma, x, y), 77, "luma ({x},{y})");
            }
        }
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Cb, x, y), 99, "cb ({x},{y})");
            }
        }
        let cell = field.cell_at(16, 16);
        assert!(!cell.is_intra && cell.pred_flag_l0);
        assert_eq!(cell.mv_l0, [0, 0]);
    }

    /// A mixed P picture: an intra CU and an inter CU side by side both
    /// reconstruct, and the intra CU is stamped intra in the motion field.
    #[test]
    fn picture_driver_mixed_intra_inter() {
        let params = p_params();
        let refpic = flat_ref(60, 90);
        let entries = vec![dpb_entry(0, refpic)];
        let lists = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };

        // CTU split into four 16×16 quadrants: top-left intra (DC), the
        // other three inter merge.
        let intra_cu = CodingUnit {
            x0: 0,
            y0: 0,
            log2_cb_size: 4,
            cu_pred_mode: CuPredMode::Intra,
            cu_transquant_bypass_flag: false,
            part_mode: PartMode::Part2Nx2N,
            pcm_flag: false,
            pcm: None,
            palette: None,
            prediction_units: vec![],
            intra_luma: vec![crate::hevc::engine::slice_data::IntraLumaMode {
                prev_intra_luma_pred_flag: true,
                mpm_idx: Some(0),
                rem_intra_luma_pred_mode: None,
            }],
            intra_chroma_pred_mode: vec![4],
            rqt_root_cbf: true,
            transform_tree: Some(TransformTree::Leaf {
                cbf_luma: false,
                unit: TransformUnit::default(),
            }),
        };
        let mut inter_tr = inter_cu_16(merge_pu(0), None);
        inter_tr.x0 = 16;
        let mut inter_bl = inter_cu_16(merge_pu(0), None);
        inter_bl.y0 = 16;
        let mut inter_br = inter_cu_16(merge_pu(0), None);
        inter_br.x0 = 16;
        inter_br.y0 = 16;

        let ctu = crate::hevc::engine::slice_data::CodingTreeUnit {
            sao: None,
            quadtree: crate::hevc::engine::slice_data::CodingQuadtree::Split(vec![
                crate::hevc::engine::slice_data::CodingQuadtree::Leaf(Box::new(intra_cu)),
                crate::hevc::engine::slice_data::CodingQuadtree::Leaf(Box::new(inter_tr)),
                crate::hevc::engine::slice_data::CodingQuadtree::Leaf(Box::new(inter_bl)),
                crate::hevc::engine::slice_data::CodingQuadtree::Leaf(Box::new(inter_br)),
            ]),
        };
        let placed = vec![PlacedInterCtu {
            x_ctb: 0,
            y_ctb: 0,
            slice_addr_rs: 0,
            filter_across_slices: true,
            ctu: &ctu,
        }];

        let slice = p_slice_ctx();
        let tiles = crate::hevc::engine::availability::TilingParams::single_tile();
        let (pic, field) =
            reconstruct_inter_picture(32, 32, &params, &slice, &tiles, &placed, &refs, None)
                .unwrap();

        // The three inter quadrants copy the reference value 60.
        assert_eq!(pic.sample(Plane::Luma, 24, 8), 60, "top-right inter");
        assert_eq!(pic.sample(Plane::Luma, 8, 24), 60, "bottom-left inter");
        assert_eq!(pic.sample(Plane::Luma, 24, 24), 60, "bottom-right inter");
        // The intra quadrant is stamped intra; the inter quadrants are not.
        assert!(field.cell_at(0, 0).is_intra, "TL intra-stamped");
        assert!(!field.cell_at(16, 0).is_intra, "TR inter");
        assert!(!field.cell_at(16, 16).is_intra, "BR inter");
    }

    /// With deblocking enabled, the §8.7.2 in-loop pass runs as part of the
    /// picture driver: a strong luma step at the intra/inter CU boundary
    /// (bS = 2) is smoothed, so the boundary samples differ from the
    /// undeblocked reconstruction.
    #[test]
    fn picture_driver_deblock_smooths_cu_boundary() {
        let params = p_params();
        // Reference for the inter CUs: flat 140 — a modest step from the
        // intra DC (128) so the §8.7.2.5.3 dE decision classifies the
        // boundary as a blocking artifact (small gradient) rather than a
        // true edge, and the weak filter engages.
        let refpic = flat_ref(140, 128);
        let entries = vec![dpb_entry(0, refpic)];
        let lists = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };

        // CTU split into four 16×16 quadrants: the two left quadrants intra
        // (DC ⇒ mid-grey 128, no neighbours), the two right inter (copy 200).
        let intra = |x0: u32, y0: u32| CodingUnit {
            x0,
            y0,
            log2_cb_size: 4,
            cu_pred_mode: CuPredMode::Intra,
            cu_transquant_bypass_flag: false,
            part_mode: PartMode::Part2Nx2N,
            pcm_flag: false,
            pcm: None,
            palette: None,
            prediction_units: vec![],
            intra_luma: vec![crate::hevc::engine::slice_data::IntraLumaMode {
                prev_intra_luma_pred_flag: false,
                mpm_idx: None,
                rem_intra_luma_pred_mode: Some(1),
            }],
            intra_chroma_pred_mode: vec![4],
            rqt_root_cbf: false,
            transform_tree: Some(TransformTree::Leaf {
                cbf_luma: false,
                unit: TransformUnit::default(),
            }),
        };
        let inter = |x0: u32, y0: u32| {
            let mut c = inter_cu_16(merge_pu(0), None);
            c.x0 = x0;
            c.y0 = y0;
            c
        };
        let ctu = crate::hevc::engine::slice_data::CodingTreeUnit {
            sao: None,
            quadtree: crate::hevc::engine::slice_data::CodingQuadtree::Split(vec![
                crate::hevc::engine::slice_data::CodingQuadtree::Leaf(Box::new(intra(0, 0))),
                crate::hevc::engine::slice_data::CodingQuadtree::Leaf(Box::new(inter(16, 0))),
                crate::hevc::engine::slice_data::CodingQuadtree::Leaf(Box::new(intra(0, 16))),
                crate::hevc::engine::slice_data::CodingQuadtree::Leaf(Box::new(inter(16, 16))),
            ]),
        };
        let placed = vec![PlacedInterCtu {
            x_ctb: 0,
            y_ctb: 0,
            slice_addr_rs: 0,
            filter_across_slices: true,
            ctu: &ctu,
        }];

        let tiles = crate::hevc::engine::availability::TilingParams::single_tile();
        // Undeblocked baseline.
        let mut undeb = p_slice_ctx();
        undeb.deblock_enabled = false;
        let (plain, _) =
            reconstruct_inter_picture(32, 32, &params, &undeb, &tiles, &placed, &refs, None)
                .unwrap();
        // Deblocked.
        let mut deb = p_slice_ctx();
        deb.deblock_enabled = true;
        let (filtered, _) =
            reconstruct_inter_picture(32, 32, &params, &deb, &tiles, &placed, &refs, None).unwrap();

        // The vertical boundary at x == 16 separates intra (128) from inter
        // (200); the deblock pass adjusts samples on at least one side, so
        // the filtered picture differs from the plain one near the edge.
        let mut changed = false;
        for y in 0..32 {
            for x in 13..19 {
                if filtered.sample(Plane::Luma, x, y) != plain.sample(Plane::Luma, x, y) {
                    changed = true;
                }
            }
        }
        assert!(changed, "deblocking modifies samples at the CU boundary");
        // Far-from-edge interior samples are untouched.
        assert_eq!(filtered.sample(Plane::Luma, 28, 8), 140, "inter interior");
    }

    /// With `slice_sao_luma_flag` set and a band-offset SAO on the CTB, the
    /// §8.7.3 SAO pass runs as part of the picture driver: the inter CU's
    /// flat luma is shifted by the band offset covering its value.
    #[test]
    fn picture_driver_applies_sao() {
        let params = p_params();
        // Inter reference flat 100. Luma 100 is in band 100 >> 3 == 12.
        let refpic = flat_ref(100, 128);
        let entries = vec![dpb_entry(0, refpic)];
        let lists = RefPicLists {
            list0: vec![Some(0)],
            list1: None,
        };
        let refs = RefListAccess {
            lists: &lists,
            entries: &entries,
        };

        let mut cu = inter_cu_16(merge_pu(0), None);
        cu.log2_cb_size = 5;
        // Band-offset SAO on luma: band_position 12 (covers value 100),
        // first offset +5 ⇒ luma 100 → 105.
        let sao = crate::hevc::engine::slice_data::SaoCtbParams {
            merge_left: false,
            merge_up: false,
            components: [
                crate::hevc::engine::slice_data::SaoComponent {
                    sao_type_idx: 1,
                    offset_abs: [5, 0, 0, 0],
                    offset_sign: [0, 0, 0, 0],
                    band_position: 12,
                    eo_class: 0,
                },
                crate::hevc::engine::slice_data::SaoComponent::default(),
                crate::hevc::engine::slice_data::SaoComponent::default(),
            ],
        };
        let ctu = crate::hevc::engine::slice_data::CodingTreeUnit {
            sao: Some(sao),
            quadtree: crate::hevc::engine::slice_data::CodingQuadtree::Leaf(Box::new(cu)),
        };
        let placed = vec![PlacedInterCtu {
            x_ctb: 0,
            y_ctb: 0,
            slice_addr_rs: 0,
            filter_across_slices: true,
            ctu: &ctu,
        }];

        let mut slice = p_slice_ctx();
        slice.slice_sao_luma_flag = true;
        let tiles = crate::hevc::engine::availability::TilingParams::single_tile();
        let (pic, _) =
            reconstruct_inter_picture(32, 32, &params, &slice, &tiles, &placed, &refs, None)
                .unwrap();
        // The reconstructed inter samples (100) fall in SAO band 12 and get
        // the +5 offset.
        assert_eq!(
            pic.sample(Plane::Luma, 8, 8),
            105,
            "SAO band offset applied"
        );
    }

    /// §8.3 + §8.5 end-to-end inter picture cycle: an IDR reference picture
    /// is stored in the DPB, then a P picture (one 32×32 inter merge CU)
    /// decodes against it via `decode_inter_picture` — resolving
    /// RefPicList0[0] → the IDR, copying its samples, and landing in the DPB
    /// as a short-term reference.
    #[test]
    fn decode_inter_picture_full_cycle() {
        use crate::hevc::engine::poc::NalKind;
        use crate::hevc::engine::sps::MaterializedShortTermRefPicSet;

        let params = p_params();
        let mut seq = crate::hevc::engine::decode::PictureSequenceState::new();

        // IDR (POC 0): a flat-110 reference picture stored directly.
        let idr_header = crate::hevc::engine::decode::PictureHeaderInfo {
            nal_kind: NalKind::new(NalKind::IDR_N_LP),
            temporal_id: 0,
            layer_id: 0,
            no_rasl_output: true,
            poc_lsb: 0,
            max_poc_lsb: 256,
            short_term_rps: MaterializedShortTermRefPicSet {
                delta_poc_s0: vec![],
                used_by_curr_pic_s0: vec![],
                delta_poc_s1: vec![],
                used_by_curr_pic_s1: vec![],
            },
            long_term: vec![],
        };
        let i_slice = crate::hevc::engine::decode::SliceRefParams {
            is_inter: false,
            is_b: false,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            num_pic_total_curr: 0,
            temporal_mvp_enabled: false,
            collocated_from_l0_flag: true,
            collocated_ref_idx: 0,
            curr_pic_ref_enabled: false,
        };
        let idr = seq.begin_picture(&idr_header, &i_slice);
        seq.store_picture(idr.poc, 0, flat_ref(110, 128), MotionField::new(32, 32));

        // P picture (POC 1): one short-term-before reference at POC 0.
        let p_header = crate::hevc::engine::decode::PictureHeaderInfo {
            nal_kind: NalKind::new(NalKind::TRAIL_R),
            no_rasl_output: false,
            poc_lsb: 1,
            short_term_rps: MaterializedShortTermRefPicSet {
                delta_poc_s0: vec![-1],
                used_by_curr_pic_s0: vec![true],
                delta_poc_s1: vec![],
                used_by_curr_pic_s1: vec![],
            },
            ..idr_header.clone()
        };
        let p_slice_ref = crate::hevc::engine::decode::SliceRefParams {
            is_inter: true,
            num_pic_total_curr: 1,
            ..i_slice
        };

        let mut cu = inter_cu_16(merge_pu(0), None);
        cu.log2_cb_size = 5;
        let ctu = crate::hevc::engine::slice_data::CodingTreeUnit {
            sao: None,
            quadtree: crate::hevc::engine::slice_data::CodingQuadtree::Leaf(Box::new(cu)),
        };
        let placed = vec![PlacedInterCtu {
            x_ctb: 0,
            y_ctb: 0,
            slice_addr_rs: 0,
            filter_across_slices: true,
            ctu: &ctu,
        }];
        let slice = p_slice_ctx();
        let tiles = crate::hevc::engine::availability::TilingParams::single_tile();

        let out = decode_inter_picture(
            &mut seq,
            &p_header,
            &p_slice_ref,
            32,
            32,
            &params,
            &slice,
            &tiles,
            &placed,
        )
        .unwrap();

        // The P picture copies the IDR reference's flat 110.
        for y in 0..32 {
            for x in 0..32 {
                assert_eq!(out.sample(Plane::Luma, x, y), 110, "P luma ({x},{y})");
            }
        }
        // The DPB now holds two pictures (IDR POC 0 + P POC 1), both
        // short-term references.
        assert_eq!(seq.dpb().entries().len(), 2);
        assert_eq!(seq.dpb().entries()[1].poc, 1);
        assert_eq!(
            seq.dpb().entries()[1].marking,
            crate::hevc::engine::dpb::Marking::ShortTerm
        );
        // The P picture's motion field records the inter CU (for a future
        // picture's temporal MVP).
        assert!(!seq.dpb().entries()[1].motion.cell_at(16, 16).is_intra);
    }
}
