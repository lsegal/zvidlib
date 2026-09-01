//! §8.4 intra sample reconstruction driver.
//!
//! This module is the rung between the §7.3.8 slice-data syntax walk
//! (the [`crate::hevc::engine::slice_data`] CTU/CU/transform-tree structures) and the
//! per-block §8.4.4 intra prediction + §8.6 dequantization / inverse
//! transform primitives already implemented in [`crate::hevc::engine::intra_pred`] and
//! [`crate::hevc::engine::transform`]. It walks a decoded [`crate::hevc::engine::slice_data::CodingTreeUnit`]
//! and writes reconstructed samples into a [`crate::hevc::engine::picture::Picture`]:
//!
//! 1. §8.4.2 — derive `IntraPredModeY` for each luma prediction block
//!    from the signalled `prev_intra_luma_pred_flag` / `mpm_idx` /
//!    `rem_intra_luma_pred_mode` and the neighbour modes; §8.4.3 —
//!    derive `IntraPredModeC`.
//! 2. §8.4.4.1 — for every transform block, gather the §8.4.4.2.1
//!    reference samples from the already-reconstructed picture (the
//!    §6.4.1 availability of left / above neighbours), run §8.4.4.2
//!    prediction.
//! 3. §8.6.2 — dequantize + inverse-transform the coded residual block
//!    (when its coded-block-flag is set), add it to the prediction, and
//!    §8.4.4.1 clip to `[0, (1 << bitDepth) − 1]`, storing the result.
//!
//! The transform-tree recursion mirrors the §8.4.4.1 luma decode order
//! (the residual quadtree drives the transform-block grid) so each block
//! sees its left / above neighbours already reconstructed before it
//! predicts.

use crate::hevc::engine::availability::PictureTiling;
use crate::hevc::engine::binarization::{
    CuPredMode, LumaIntraModeSource, PartMode, derive_intra_pred_mode_c, derive_intra_pred_mode_y,
    intra_luma_cand_mode_list, luma_intra_mode_source_from_flag,
};
use crate::hevc::engine::intra_mode_field::{IntraModeField, Neighbour};
use crate::hevc::engine::intra_pred::{
    Component as IpComponent, INTRA_DC, IntraPredError, IntraPredParams, MarkedReferenceSamples,
    intra_predict_with_substitution,
};
use crate::hevc::engine::picture::{Picture, Plane, clip1, sub_wh_c};
use crate::hevc::engine::profile::{Stage as ProfStage, scope as prof_scope};
use crate::hevc::engine::scaling_list::ScalingFactors;
use crate::hevc::engine::slice_data::{CodingQuadtree, CodingTreeUnit, CodingUnit, IntraLumaMode};
use crate::hevc::engine::transform::{
    BlockParams, Component as TfComponent, PredMode, TransformError, rdpcm_accumulate,
    residual_block,
};
use crate::hevc::engine::transform_tree::TransformTree;
use crate::hevc::engine::transform_unit::TransformUnit;

/// Errors raised while reconstructing samples from a decoded CTU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconError {
    /// A §8.4.4.2 intra-prediction primitive failed.
    IntraPred(IntraPredError),
    /// A §8.6.2 dequantization / inverse-transform primitive failed.
    Transform(TransformError),
    /// A §8.5.3.3 inter-prediction primitive failed.
    InterPred(crate::hevc::engine::inter_pred::InterPredError),
    /// The decoded CTU carried an inter prediction unit, which the intra
    /// reconstruction path does not handle.
    InterNotSupported,
    /// The §6.4.1 picture-tiling geometry needed for neighbour
    /// availability could not be built.
    Tiling(crate::hevc::engine::availability::AvailabilityError),
}

impl core::fmt::Display for ReconError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IntraPred(e) => write!(f, "intra prediction failed: {e}"),
            Self::Transform(e) => write!(f, "inverse transform failed: {e}"),
            Self::InterPred(e) => write!(f, "inter prediction failed: {e}"),
            Self::InterNotSupported => {
                f.write_str("inter prediction is not reconstructed by the intra path")
            }
            Self::Tiling(e) => write!(f, "picture tiling geometry invalid: {e}"),
        }
    }
}

impl std::error::Error for ReconError {}

impl From<IntraPredError> for ReconError {
    fn from(e: IntraPredError) -> Self {
        Self::IntraPred(e)
    }
}

impl From<TransformError> for ReconError {
    fn from(e: TransformError) -> Self {
        Self::Transform(e)
    }
}

/// SPS / PPS / slice-derived state constant across one picture's intra
/// reconstruction. Every field is a §8 input the per-block prediction /
/// dequantization reads.
#[derive(Debug, Clone)]
pub struct ReconParams {
    /// `ChromaArrayType` (0 = monochrome, 1 = 4:2:0, 2 = 4:2:2,
    /// 3 = 4:4:4).
    pub chroma_array_type: u8,
    /// `BitDepthY`.
    pub bit_depth_luma: u8,
    /// `BitDepthC`.
    pub bit_depth_chroma: u8,
    /// `intra_smoothing_disabled_flag` (§8.4.4.2.3 step-1 gate).
    pub intra_smoothing_disabled: bool,
    /// `strong_intra_smoothing_enabled_flag` (§8.4.4.2.3 `biIntFlag`).
    pub strong_intra_smoothing_enabled: bool,
    /// `SliceQpY` (§7.4.7.1) — the slice quantization parameter, the
    /// §8.6.1 luma-QP starting point (the tiny single-CU fixtures carry
    /// `CuQpDeltaVal == 0`, so `QpY == SliceQpY`).
    pub slice_qp_y: i32,
    /// `pps_cb_qp_offset + slice_cb_qp_offset` (§8.6.1 `qPiCb` offset).
    pub cb_qp_offset: i32,
    /// `pps_cr_qp_offset + slice_cr_qp_offset` (§8.6.1 `qPiCr` offset).
    pub cr_qp_offset: i32,
    /// `PpsActQpOffsetY + slice_act_y_qp_offset` — the §8.6.2 eq. 8-291
    /// luma qP offset applied when `tu_residual_act_flag == 1`.
    pub act_y_qp_offset: i32,
    /// `PpsActQpOffsetCb + slice_act_cb_qp_offset` — the §8.6.1
    /// eq. 8-287 `qPiCb` offset base for `tu_residual_act_flag == 1`
    /// transform units (replacing the pps + slice Cb offsets).
    pub act_cb_qp_offset: i32,
    /// `PpsActQpOffsetCr + slice_act_cr_qp_offset` (eq. 8-288).
    pub act_cr_qp_offset: i32,
    /// `transform_skip_rotation_enabled_flag`.
    pub transform_skip_rotation_enabled: bool,
    /// `implicit_rdpcm_enabled_flag` (§7.4.3.2.2) — gates the §8.4.4.1
    /// intra `residualDpcm` condition (transform skip or transquant
    /// bypass, predModeIntra 10 / 26) that invokes the §8.6.5
    /// directional residual modification, and (with transquant bypass)
    /// the §8.4.4.2.6 `disableIntraBoundaryFilter` derivation.
    pub implicit_rdpcm_enabled: bool,
    /// `intra_boundary_filtering_disabled_flag` (§7.4.3.2.3) — the
    /// first §8.4.4.2.6 `disableIntraBoundaryFilter` condition.
    pub intra_boundary_filtering_disabled: bool,
    /// `extended_precision_processing_flag`.
    pub extended_precision: bool,
    /// The active §7.4.5 `ScalingFactor` matrices when
    /// `scaling_list_enabled_flag == 1` (the PPS `scaling_list_data()`
    /// if present, else the SPS body, else the default lists); `None`
    /// when the flag is 0 (the §8.6.3 flat-16 path).
    pub scaling: Option<ScalingFactors>,
    /// `(cb_qp_offset_list[i], cr_qp_offset_list[i])` (§7.4.3.3.2) —
    /// the PPS range extension chroma QP offset lists dereferenced by
    /// `cu_chroma_qp_offset_idx`. Empty when
    /// `chroma_qp_offset_list_enabled_flag == 0`.
    pub chroma_qp_offset_list: Vec<(i32, i32)>,
    /// The live `( CuQpOffsetCb, CuQpOffsetCr )` state (§7.4.9.10):
    /// reset to `(0, 0)` at each slice start (§7.4.7.1), overwritten
    /// whenever a reconstructed transform unit / palette CU carries a
    /// parsed `cu_chroma_qp_offset_flag`, and consumed by the §8.6.1
    /// eqs. 8-285..8-288 chroma QP derivation. Interior-mutable so the
    /// decode-order state threads through the shared `&ReconParams`
    /// without touching every recursion signature (picture
    /// reconstruction is single-threaded).
    pub cu_qp_offset_c: core::cell::Cell<(i32, i32)>,
}

/// §7.4.9.10 — resolve a parsed `cu_chroma_qp_offset` element to the
/// `( CuQpOffsetCb, CuQpOffsetCr )` pair it selects (eqs. 7-87 / 7-88;
/// `(0, 0)` when the flag is 0) and store it in the live state.
fn apply_cu_chroma_qp_offset(
    params: &ReconParams,
    off: &crate::hevc::engine::binarization::CuChromaQpOffset,
) {
    let pair = match off.offset_indices() {
        Some(idx) => params
            .chroma_qp_offset_list
            .get(idx as usize)
            .copied()
            .unwrap_or((0, 0)),
        None => (0, 0),
    };
    params.cu_qp_offset_c.set(pair);
}

/// §8.6.3 `m[ x ][ y ]` source selection: the `ScalingFactor[ sizeId ]
/// [ matrixId ]` matrix for a transform block, or `None` for the flat
/// 16 (scaling lists disabled, or the `transform_skip_flag == 1 &&
/// nTbS > 4` exception).
///
/// `sizeId` is Table 7-3 (`log2(nTbS) − 2`); `matrixId` is Table 7-4
/// (`(CuPredMode == MODE_INTRA ? 0 : 3) + cIdx`).
fn scaling_matrix(
    params: &ReconParams,
    n_tbs: usize,
    pred_mode: PredMode,
    cidx: TfComponent,
    transform_skip: bool,
) -> Option<&crate::hevc::engine::scaling_list::ScalingFactorMatrix> {
    let sf = params.scaling.as_ref()?;
    if transform_skip && n_tbs > 4 {
        return None;
    }
    let size_id = match n_tbs {
        4 => 0usize,
        8 => 1,
        16 => 2,
        _ => 3,
    };
    let matrix_id = match pred_mode {
        PredMode::Intra => 0usize,
        _ => 3,
    } + match cidx {
        TfComponent::Luma => 0usize,
        TfComponent::Cb => 1,
        TfComponent::Cr => 2,
    };
    Some(&sf.factors[size_id][matrix_id])
}

/// `QpBdOffsetY = 6 * bit_depth_luma_minus8` (§7.4.3.2.1, eq. 7-4).
#[inline]
fn qp_bd_offset(bit_depth: u8) -> i32 {
    6 * (i32::from(bit_depth) - 8)
}

/// §8.6.1 — derive `Qp′Y` for a luma transform block.
///
/// `Qp′Y = QpY + QpBdOffsetY`, with `QpY` the slice QP plus any
/// `CuQpDeltaVal` (clipped per the §8.6.1 wrap). The neighbour-prediction
/// of `qPY_PRED` collapses to a single value when every coding unit in
/// the picture shares the slice QP (no `cu_qp_delta`); the recursion
/// threads `cu_qp_delta_val` so the general single-CU-per-QG case is
/// exact.
#[inline]
fn luma_qp(params: &ReconParams, qp_y: i32) -> u32 {
    // §8.6.1 eq. 8-284: Qp′Y = QpY + QpBdOffsetY, with `qp_y` the
    // already-derived (eq. 8-283) luma quantization parameter.
    let qp_bd = qp_bd_offset(params.bit_depth_luma);
    (qp_y + qp_bd) as u32
}

/// §8.6.2 eq. 8-291 — the luma qP of a `tu_residual_act_flag == 1`
/// transform block: `Clip3( 0, 51 + QpBdOffsetY, Qp′Y +
/// PpsActQpOffsetY + slice_act_y_qp_offset )`.
#[inline]
fn luma_qp_act(params: &ReconParams, qp_y: i32) -> u32 {
    let qp_bd = qp_bd_offset(params.bit_depth_luma);
    (qp_y + qp_bd + params.act_y_qp_offset).clamp(0, 51 + qp_bd) as u32
}

/// §8.6.1 eq. 8-283 with `qPY_PRED == SliceQpY` — the single-QG
/// fallback used when no picture-level QP state is initialized (the
/// standalone per-CTU helpers).
#[inline]
fn fallback_qp_y(params: &ReconParams, cu_qp_delta_val: i32) -> i32 {
    let qp_bd = qp_bd_offset(params.bit_depth_luma);
    let modulus = 52 + qp_bd;
    (params.slice_qp_y + cu_qp_delta_val + 52 + 2 * qp_bd).rem_euclid(modulus) - qp_bd
}

/// Table 8-10 — `ChromaArrayType == 1` chroma-QP mapping `QpC = f(qPi)`.
#[inline]
fn qpc_420(qpi: i32) -> i32 {
    match qpi {
        x if x < 30 => x,
        30 => 29,
        31 => 30,
        32 => 31,
        33 => 32,
        34 => 33,
        35 => 33,
        36 => 34,
        37 => 34,
        38 => 35,
        39 => 35,
        40 => 36,
        41 => 36,
        42 => 37,
        43 => 37,
        x => x - 6,
    }
}

/// §8.6.1 — derive `Qp′Cb` / `Qp′Cr` for a chroma transform block.
///
/// `qPiCx = Clip3( −QpBdOffsetC, 57, QpY + cQpOffset )`; for
/// `ChromaArrayType == 1` `qPCx = qPC_table( qPiCx )` (Table 8-10), for
/// the other chroma types `qPCx = Min( qPiCx, 51 )`; then
/// `Qp′Cx = qPCx + QpBdOffsetC` (eq. 8-260).
#[inline]
fn chroma_qp(params: &ReconParams, qp_y: i32, cidx: TfComponent) -> u32 {
    chroma_qp_act(params, qp_y, cidx, false)
}

/// §8.6.1 with the eq. 8-285..8-288 offset selection: for a
/// `tu_residual_act_flag == 1` transform unit the `qPiCx` offset base
/// is `PpsActQpOffsetCx + slice_act_cx_qp_offset` instead of the
/// pps + slice component offsets.
#[inline]
fn chroma_qp_act(params: &ReconParams, qp_y: i32, cidx: TfComponent, act: bool) -> u32 {
    let qp_bd_c = qp_bd_offset(params.bit_depth_chroma);
    // `qp_y` is the already-derived (eq. 8-283) QpY — the §8.6.1
    // chroma input.
    let qpy = qp_y;
    let (cu_off_cb, cu_off_cr) = params.cu_qp_offset_c.get();
    let offset = match (cidx, act) {
        (TfComponent::Cb, false) => params.cb_qp_offset + cu_off_cb,
        (TfComponent::Cr, false) => params.cr_qp_offset + cu_off_cr,
        (TfComponent::Cb, true) => params.act_cb_qp_offset + cu_off_cb,
        (TfComponent::Cr, true) => params.act_cr_qp_offset + cu_off_cr,
        (TfComponent::Luma, _) => 0,
    };
    let qpi = (qpy + offset).clamp(-qp_bd_c, 57);
    let qpc = if params.chroma_array_type == 1 {
        qpc_420(qpi)
    } else {
        qpi.min(51)
    };
    (qpc + qp_bd_c) as u32
}

/// Gather the §8.4.4.2.1 reference-sample array for a transform block at
/// plane position `(xb, yb)` of side `n_tbs` from the already-
/// reconstructed picture, marking each neighbour available per the §6.4.1
/// z-scan availability process (via [`ReconCtx::ref_sample_available`]):
/// a neighbour is available iff it is inside the picture, already decoded
/// in z-scan order, and in the same slice / tile. The unavailable samples
/// are substituted by [`intra_predict_with_substitution`].
fn gather_reference_samples(
    pic: &Picture,
    ctx: &ReconCtx,
    plane: Plane,
    xb: usize,
    yb: usize,
    n_tbs: usize,
) -> MarkedReferenceSamples {
    let (pw, ph) = pic.plane_dims(plane);
    let (sub_w, sub_h) = match plane {
        Plane::Luma => (1, 1),
        Plane::Cb | Plane::Cr => sub_wh_c(pic.chroma_array_type()),
    };
    let avail = |x: i64, y: i64| -> bool {
        x >= 0
            && y >= 0
            && (x as usize) < pw
            && (y as usize) < ph
            && ctx.ref_sample_available(xb, yb, x, y, sub_w, sub_h)
    };
    let read = |x: i64, y: i64| -> (i32, bool) {
        if avail(x, y) {
            (pic.sample(plane, x as usize, y as usize), true)
        } else {
            (0, false)
        }
    };
    // Corner p[−1][−1].
    let corner = read(xb as i64 - 1, yb as i64 - 1);
    // Left column p[−1][0 .. 2*nTbS−1].
    let mut left = Vec::with_capacity(2 * n_tbs);
    for y in 0..(2 * n_tbs) {
        left.push(read(xb as i64 - 1, yb as i64 + y as i64));
    }
    // Top row p[0 .. 2*nTbS−1][−1].
    let mut top = Vec::with_capacity(2 * n_tbs);
    for x in 0..(2 * n_tbs) {
        top.push(read(xb as i64 + x as i64, yb as i64 - 1));
    }
    MarkedReferenceSamples::new(n_tbs, corner, left, top)
        .expect("reference array dimensions match n_tbs")
}

/// §8.6.6 — residual modification for transform blocks using
/// cross-component prediction (`ChromaArrayType == 3` only):
///
/// `r[ x ][ y ] += ( ResScaleVal * ( ( rY[ x ][ y ] << BitDepthC ) >>
/// BitDepthY ) ) >> 3` (eq. 8-324), with `rY` the co-located luma
/// residual of the same transform unit. The intermediate runs in `i64`
/// so the `<< BitDepthC` cannot overflow at extended-precision coeff
/// ranges; the shifts are arithmetic per the §5.8 operator definitions.
fn apply_cross_comp_pred(
    r: &mut [i32],
    r_y: &[i32],
    res_scale_val: i32,
    bit_depth_luma: u8,
    bit_depth_chroma: u8,
) {
    debug_assert_eq!(r.len(), r_y.len());
    let bd_y = i64::from(bit_depth_luma);
    let bd_c = i64::from(bit_depth_chroma);
    let scale = i64::from(res_scale_val);
    for (rv, &ry) in r.iter_mut().zip(r_y.iter()) {
        let scaled = (scale * ((i64::from(ry) << bd_c) >> bd_y)) >> 3;
        *rv = (i64::from(*rv) + scaled) as i32;
    }
}

/// The §8.6.6 inputs for one chroma transform block: the co-located
/// luma residual array `rY` (post-§8.6.5, same `nTbS` — 4:4:4 only)
/// and the §7.4.9.12-derived `ResScaleVal`.
#[derive(Clone, Copy)]
struct CcpInput<'a> {
    luma_residual: &'a [i32],
    res_scale_val: i32,
}

impl<'a> CcpInput<'a> {
    /// Build the §8.6.6 input for one chroma component, when the
    /// bitstream carried a `cross_comp_pred( )` with a non-zero
    /// `ResScaleVal` and the transform unit has a luma residual to
    /// predict from (an all-zero `rY` contributes nothing).
    fn resolve(
        chroma_array_type: u8,
        ccp: Option<&crate::hevc::engine::binarization::CrossCompPred>,
        luma_residual: Option<&'a [i32]>,
    ) -> Option<Self> {
        if chroma_array_type != 3 {
            return None;
        }
        let res_scale_val = ccp?.res_scale_val;
        if res_scale_val == 0 {
            return None;
        }
        Some(Self {
            luma_residual: luma_residual?,
            res_scale_val,
        })
    }
}

/// Predict one intra transform block, add its residual, clip, and store
/// into `pic`. `(xb, yb)` is the plane-coordinate top-left; `pred_mode`
/// is the §8.4.x prediction mode for the plane's component.
///
/// Returns the final residual array (post-§8.6.5/§8.6.6 modification)
/// so a 4:4:4 caller can feed the luma residual into the chroma
/// blocks' §8.6.6 cross-component prediction; `None` when the block
/// carried no residual at all.
#[allow(clippy::too_many_arguments)]
fn reconstruct_intra_block(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &ReconCtx,
    plane: Plane,
    cidx: TfComponent,
    ip_component: IpComponent,
    xb: usize,
    yb: usize,
    n_tbs: usize,
    pred_mode_intra: u8,
    residual: Option<&[i32]>,
    qp: u32,
    transquant_bypass: bool,
    transform_skip: bool,
    ccp: Option<CcpInput<'_>>,
) -> Result<Option<Vec<i32>>, ReconError> {
    // §8.6.2 residual array (zero when the block has no coded coeffs).
    let res: Option<Vec<i32>> = match residual {
        Some(levels) => Some(intra_residual_array(
            params,
            cidx,
            n_tbs,
            pred_mode_intra,
            levels,
            qp,
            transquant_bypass,
            transform_skip,
        )?),
        None => None,
    };

    // §8.4.4.1 step 8: cross-component prediction modifies the chroma
    // residual from the co-located luma residual (after the step-7
    // §8.6.5 modification, before the §8.6.7 add). A block with no
    // coded chroma residual still receives the scaled luma residual.
    let res = match ccp {
        Some(c) => {
            let mut r = res.unwrap_or_else(|| vec![0i32; n_tbs * n_tbs]);
            apply_cross_comp_pred(
                &mut r,
                c.luma_residual,
                c.res_scale_val,
                params.bit_depth_luma,
                params.bit_depth_chroma,
            );
            Some(r)
        }
        None => res,
    };

    predict_add_store(
        pic,
        params,
        ctx,
        plane,
        ip_component,
        xb,
        yb,
        n_tbs,
        pred_mode_intra,
        transquant_bypass,
        res.as_deref(),
    )?;
    Ok(res)
}

/// §8.6.2 + §8.6.5 — dequantize / inverse-transform one intra block's
/// coded levels and apply the §8.4.4.1 implicit-RDPCM residual
/// modification when its condition holds.
#[allow(clippy::too_many_arguments)]
fn intra_residual_array(
    params: &ReconParams,
    cidx: TfComponent,
    n_tbs: usize,
    pred_mode_intra: u8,
    levels: &[i32],
    qp: u32,
    transquant_bypass: bool,
    transform_skip: bool,
) -> Result<Vec<i32>, ReconError> {
    let bit_depth = match cidx {
        TfComponent::Luma => params.bit_depth_luma,
        TfComponent::Cb | TfComponent::Cr => params.bit_depth_chroma,
    };
    let bp = BlockParams {
        n_tbs,
        q_p: qp,
        component: cidx,
        pred_mode: PredMode::Intra,
        bit_depth,
        extended_precision: params.extended_precision,
        transquant_bypass,
        transform_skip,
        transform_skip_rotation_enabled: params.transform_skip_rotation_enabled,
    };
    let m = scaling_matrix(params, n_tbs, PredMode::Intra, cidx, transform_skip);
    let mut r = residual_block(levels, m, bp)?;
    // §8.4.4.1: residualDpcm — implicit RDPCM applies the §8.6.5
    // directional residual modification to intra blocks in
    // transform-skip / transquant-bypass form whose prediction mode is
    // exactly horizontal (10) or vertical (26), with
    // mDir = predModeIntra / 26.
    if params.implicit_rdpcm_enabled
        && (transform_skip || transquant_bypass)
        && (pred_mode_intra == 10 || pred_mode_intra == 26)
    {
        rdpcm_accumulate(&mut r, n_tbs, pred_mode_intra == 26);
    }
    Ok(r)
}

/// §8.4.4.2.1 prediction + §8.6.7 picture construction for one intra
/// transform block: gather references, predict, add `res` (when
/// present), clip, store.
#[allow(clippy::too_many_arguments)]
fn predict_add_store(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &ReconCtx,
    plane: Plane,
    ip_component: IpComponent,
    xb: usize,
    yb: usize,
    n_tbs: usize,
    pred_mode_intra: u8,
    transquant_bypass: bool,
    res: Option<&[i32]>,
) -> Result<(), ReconError> {
    // Issue #189 stage attribution: reference gathering, prediction and the
    // §8.6.7 add are one stage because they are one block's intra work and no
    // decision separates them.
    let _profile = prof_scope(ProfStage::IntraPred);
    let bit_depth = pic.bit_depth(plane);
    let marked = gather_reference_samples(pic, ctx, plane, xb, yb, n_tbs);
    // §8.4.4.2.6: disableIntraBoundaryFilter is 1 when
    // intra_boundary_filtering_disabled_flag is 1, or when
    // implicit_rdpcm_enabled_flag and cu_transquant_bypass_flag are
    // both 1. That derivation gates only the ANGULAR mode-10 / mode-26
    // edge filters; the §8.4.4.2.5 INTRA_DC smoothing (eq. 8-48..8-51)
    // is gated on intra_boundary_filtering_disabled_flag alone, so the
    // implicit-RDPCM term is withheld for the DC mode.
    let disable_boundary_filter = params.intra_boundary_filtering_disabled
        || (pred_mode_intra != INTRA_DC && params.implicit_rdpcm_enabled && transquant_bypass);
    let ip_params = IntraPredParams {
        pred_mode_intra,
        cidx: ip_component,
        bit_depth,
        bit_depth_luma: params.bit_depth_luma,
        intra_smoothing_disabled: params.intra_smoothing_disabled,
        strong_intra_smoothing_enabled: params.strong_intra_smoothing_enabled,
        chroma_array_type_3: params.chroma_array_type == 3,
        disable_boundary_filter,
    };
    let pred = intra_predict_with_substitution(&marked, &ip_params)?;

    // §8.4.4.1 / §8.6.7: recSamples = Clip1( predSamples + resSamples ).
    for y in 0..n_tbs {
        for x in 0..n_tbs {
            let p = pred[y * n_tbs + x];
            let r = res.map_or(0, |r| r[y * n_tbs + x]);
            let v = clip1(p + r, bit_depth);
            pic.set_sample(plane, xb + x, yb + y, v);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// §8.5 inter sample reconstruction
// ---------------------------------------------------------------------------

use crate::hevc::engine::inter_pred::{
    InterPredGeometry, InterPrediction, ListPrediction, RefPlane,
};

/// One reference list's fully-resolved per-PU motion: the
/// §8.5.3.2-derived luma motion vector, the §8.5.3.2.10 chroma motion
/// vector, and the §8.5.3.3.2-selected reference picture.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedList<'a> {
    /// `predFlagLX` — whether this list contributes.
    pub pred_flag: bool,
    /// `mvLX` in quarter-luma-sample units.
    pub mv_l: [i32; 2],
    /// `mvCLX` in eighth-chroma-sample units (§8.5.3.2.10).
    pub mv_c: [i32; 2],
    /// `RefPicListX[refIdxLX]` — the reference picture's samples.
    pub ref_pic: &'a Picture,
}

/// Build the §8.5.3.3.2 [`ListPrediction`] for one reference list,
/// borrowing the reference picture's luma + (when chroma is present)
/// Cb / Cr planes for the lifetime `'a`.
fn build_list_prediction<'a>(
    list: &ResolvedList<'a>,
    chroma_array_type: u8,
) -> Result<ListPrediction<'a>, ReconError> {
    let (lw, lh) = list.ref_pic.plane_dims(Plane::Luma);
    let lp =
        RefPlane::new(list.ref_pic.plane(Plane::Luma), lw, lh).map_err(ReconError::InterPred)?;
    let (cb, cr) = if chroma_array_type != 0 {
        let (cw, ch) = list.ref_pic.plane_dims(Plane::Cb);
        (
            Some(
                RefPlane::new(list.ref_pic.plane(Plane::Cb), cw, ch)
                    .map_err(ReconError::InterPred)?,
            ),
            Some(
                RefPlane::new(list.ref_pic.plane(Plane::Cr), cw, ch)
                    .map_err(ReconError::InterPred)?,
            ),
        )
    } else {
        (None, None)
    };
    Ok(ListPrediction {
        pred_flag: list.pred_flag,
        luma: lp,
        cb,
        cr,
        mv_l: list.mv_l,
        mv_c: list.mv_c,
    })
}

/// §8.5.3.3 — reconstruct one inter prediction unit's motion-compensated
/// prediction into `pic`, then (when a residual block is supplied) add
/// the §8.6.2 residual and §8.6.5 / §8.4.4.1 clip.
///
/// `(x_pb, y_pb)` is the PU's luma top-left; `(n_pb_w, n_pb_h)` its luma
/// size. The L0 / L1 lists carry the resolved motion + reference picture.
/// `residual_luma` / `residual_cb` / `residual_cr` are the optional
/// §8.6.2-output residual arrays (already dequantized + inverse
/// transformed) for the PU's covering transform blocks; pass `None` for a
/// skip / zero-residual PU (the prediction is written directly).
///
/// # Errors
/// Propagates [`ReconError::InterNotSupported`] reuse is avoided here; the
/// §8.5.3.3 interpolation failures surface as [`ReconError`] via the
/// `InterPred` variant carrying the [`crate::hevc::engine::inter_pred::InterPredError`].
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_inter_pu(
    pic: &mut Picture,
    params: &ReconParams,
    x_pb: usize,
    y_pb: usize,
    n_pb_w: usize,
    n_pb_h: usize,
    l0: ResolvedList<'_>,
    l1: ResolvedList<'_>,
    residual_luma: Option<&[i32]>,
    residual_cb: Option<&[i32]>,
    residual_cr: Option<&[i32]>,
) -> Result<(), ReconError> {
    reconstruct_inter_pu_weighted(
        pic,
        params,
        x_pb,
        y_pb,
        n_pb_w,
        n_pb_h,
        l0,
        l1,
        residual_luma,
        residual_cb,
        residual_cr,
        None,
    )
}

/// §8.5.3.3 — as [`reconstruct_inter_pu`], with the §8.5.3.3.4.1
/// weighted-sample-prediction dispatch: `weights == None` uses the
/// §8.5.3.3.4.2 default combine, `weights == Some(..)` the §8.5.3.3.4.3
/// explicit per-reference combine.
///
/// # Errors
/// Same contract as [`reconstruct_inter_pu`].
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_inter_pu_weighted(
    pic: &mut Picture,
    params: &ReconParams,
    x_pb: usize,
    y_pb: usize,
    n_pb_w: usize,
    n_pb_h: usize,
    l0: ResolvedList<'_>,
    l1: ResolvedList<'_>,
    residual_luma: Option<&[i32]>,
    residual_cb: Option<&[i32]>,
    residual_cr: Option<&[i32]>,
    weights: Option<&crate::hevc::engine::inter_pred::PuWeights>,
) -> Result<(), ReconError> {
    // Issue #189 stage attribution: `reconstruct_inter_pu` delegates here, so
    // this one scope covers every inter prediction unit in a picture.
    let _profile = prof_scope(ProfStage::InterPred);
    tmp_pu_hist::record(n_pb_w, n_pb_h, l0.pred_flag, l1.pred_flag);
    let cat = params.chroma_array_type;
    // Build the §8.5.3.3.2 reference planes for each used list.
    let lp0 = build_list_prediction(&l0, cat)?;
    let lp1 = build_list_prediction(&l1, cat)?;
    let geom = InterPredGeometry {
        x_pb: x_pb as i32,
        y_pb: y_pb as i32,
        n_pb_w,
        n_pb_h,
        chroma_array_type: cat,
        bit_depth_luma: params.bit_depth_luma,
        bit_depth_chroma: params.bit_depth_chroma,
    };
    let InterPrediction { luma, cb, cr } = {
        let _profile = prof_scope(ProfStage::InterPredFilter);
        crate::hevc::engine::inter_pred::predict_inter_pu_weighted(&lp0, &lp1, &geom, weights)
            .map_err(ReconError::InterPred)?
    };

    // §8.6.5 / §8.4.4.1: recSamples = Clip1( predSamples + resSamples ).
    let _write_profile = prof_scope(ProfStage::InterPredWrite);
    write_inter_plane(
        pic,
        Plane::Luma,
        x_pb,
        y_pb,
        n_pb_w,
        n_pb_h,
        &luma,
        residual_luma,
    );
    if cat != 0 {
        let (sw, sh) = sub_wh_c(cat);
        let xc = x_pb / sw;
        let yc = y_pb / sh;
        let pcw = n_pb_w / sw;
        let pch = n_pb_h / sh;
        write_inter_plane(pic, Plane::Cb, xc, yc, pcw, pch, &cb, residual_cb);
        write_inter_plane(pic, Plane::Cr, xc, yc, pcw, pch, &cr, residual_cr);
    }
    Ok(())
}

/// Write one motion-compensated prediction plane plus its optional
/// residual into `pic` with the §8.4.4.1 clip.
///
/// This is `recSamples = Clip1( predSamples + resSamples )` and nothing else,
/// but issue #280 measured it at 10% of decode proper on a 1080p decode — a
/// third of everything the `inter_pred` stage was charged with, and a third
/// that no vector kernel touches. What cost that was the shape of the loop
/// rather than the arithmetic: [`Picture::set_sample`] re-resolved the plane
/// and re-derived its stride for every sample, and the `Option` residual was
/// branched on per sample, so each output was a call, a match and a bounds
/// check around one add and one clamp.
///
/// The plane and its stride are now resolved once per prediction unit and the
/// residual branch hoisted out of the row loop, leaving two row slices of
/// known equal length that LLVM vectorizes on its own. The arithmetic is
/// unchanged and still in `i32` in the same order, so every sample written is
/// the one the previous loop wrote.
#[allow(clippy::too_many_arguments)]
fn write_inter_plane(
    pic: &mut Picture,
    plane: Plane,
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    pred: &[i32],
    residual: Option<&[i32]>,
) {
    if w == 0 || h == 0 {
        return;
    }
    let max = clip1(i32::MAX, pic.bit_depth(plane));
    let (buf, stride) = pic.plane_mut(plane);
    for y in 0..h {
        let row = (y0 + y) * stride + x0;
        let dst = &mut buf[row..row + w];
        let pred_row = &pred[y * w..y * w + w];
        match residual {
            Some(residual) => {
                let res_row = &residual[y * w..y * w + w];
                for ((dst, pred), res) in dst.iter_mut().zip(pred_row).zip(res_row) {
                    *dst = (*pred + *res).clamp(0, max);
                }
            }
            None => {
                for (dst, pred) in dst.iter_mut().zip(pred_row) {
                    *dst = (*pred).clamp(0, max);
                }
            }
        }
    }
}

/// Map an `IpComponent` for the plane / cIdx pair.
#[inline]
fn ip_component_of(cidx: TfComponent) -> IpComponent {
    match cidx {
        TfComponent::Luma => IpComponent::Luma,
        TfComponent::Cb => IpComponent::Cb,
        TfComponent::Cr => IpComponent::Cr,
    }
}

/// A per-coding-unit residual buffer for one component, sized to the luma
/// coding block (luma) or its chroma sub-sampling (Cb / Cr). The §8.5
/// inter reconstruction adds this onto the motion-compensated prediction.
///
/// The transform tree (§7.3.8.8) subdivides the coding block independently
/// of the §7.3.8.6 prediction-unit partitioning, so the inter
/// reconstruction first assembles the whole-CU residual (every leaf
/// transform block's §8.6.2 dequant + inverse transform written into its
/// position) and then slices out each prediction unit's covering region.
#[derive(Debug, Clone)]
pub struct CuResidualPlane {
    /// Top-left x of the plane within its component (luma coordinates for
    /// luma, chroma-subsampled coordinates for Cb / Cr).
    pub x0: usize,
    /// Top-left y of the plane.
    pub y0: usize,
    /// Plane width in samples.
    pub width: usize,
    /// Plane height in samples.
    pub height: usize,
    /// Row-major residual samples (`width * height`), zero where no
    /// transform block coded coefficients.
    pub samples: Vec<i32>,
}

impl CuResidualPlane {
    /// A zero residual plane covering `(x0, y0)` of size `width × height`.
    fn zeros(x0: usize, y0: usize, width: usize, height: usize) -> Self {
        Self {
            x0,
            y0,
            width,
            height,
            samples: vec![0; width * height],
        }
    }

    /// Write a `n × n` residual sub-block at component position
    /// `(bx, by)` (the block's top-left in the same coordinate system as
    /// `x0`/`y0`), clipping to the plane bounds.
    fn write_block(&mut self, bx: usize, by: usize, n: usize, block: &[i32]) {
        for y in 0..n {
            let py = by + y;
            if py < self.y0 || py >= self.y0 + self.height {
                continue;
            }
            for x in 0..n {
                let px = bx + x;
                if px < self.x0 || px >= self.x0 + self.width {
                    continue;
                }
                let idx = (py - self.y0) * self.width + (px - self.x0);
                self.samples[idx] = block[y * n + x];
            }
        }
    }

    /// Extract the `w × h` residual covering the prediction region at
    /// component position `(rx, ry)` (row-major), zero-filling positions
    /// outside this plane.
    #[must_use]
    pub fn slice_region(&self, rx: usize, ry: usize, w: usize, h: usize) -> Vec<i32> {
        let mut out = vec![0; w * h];
        if rx >= self.x0
            && ry >= self.y0
            && rx
                .checked_add(w)
                .is_some_and(|end| end <= self.x0 + self.width)
            && ry
                .checked_add(h)
                .is_some_and(|end| end <= self.y0 + self.height)
        {
            let source_x = rx - self.x0;
            let source_y = ry - self.y0;
            for y in 0..h {
                let source = (source_y + y) * self.width + source_x;
                out[y * w..(y + 1) * w].copy_from_slice(&self.samples[source..source + w]);
            }
            return out;
        }
        for y in 0..h {
            let py = ry + y;
            if py < self.y0 || py >= self.y0 + self.height {
                continue;
            }
            for x in 0..w {
                let px = rx + x;
                if px < self.x0 || px >= self.x0 + self.width {
                    continue;
                }
                out[y * w + x] = self.samples[(py - self.y0) * self.width + (px - self.x0)];
            }
        }
        out
    }
}

/// The three per-component residual planes of one inter coding unit.
#[derive(Debug, Clone)]
pub struct CuResidual {
    /// The luma residual plane (luma coordinates).
    pub luma: CuResidualPlane,
    /// The Cb residual plane (chroma-subsampled coordinates), absent for
    /// monochrome.
    pub cb: Option<CuResidualPlane>,
    /// The Cr residual plane.
    pub cr: Option<CuResidualPlane>,
}

impl CuResidual {
    /// `true` when this CU codes no residual at all (skip / `rqt_root_cbf
    /// == 0`); the prediction is written directly.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        let plane_zero = |p: &Option<CuResidualPlane>| match p {
            Some(plane) => plane.samples.iter().all(|&s| s == 0),
            None => true,
        };
        self.luma.samples.iter().all(|&s| s == 0) && plane_zero(&self.cb) && plane_zero(&self.cr)
    }
}

/// §8.5 / §8.6.2 — assemble one inter coding unit's residual planes by
/// walking its §7.3.8.8 transform tree, dequantizing + inverse-transforming
/// each leaf transform block (`MODE_INTER` path) into its CU-relative
/// position.
///
/// `(x_cb, y_cb)` is the CU's luma top-left; `n_cb_s` its luma side.
/// `qp_y` is the §8.6.1-derived `QpY` of the CU (`SliceQpY` for the
/// single-QG case). Returns `CuResidual` with the luma plane and — for a
/// non-monochrome `ChromaArrayType` — the Cb / Cr planes.
///
/// # Errors
/// Propagates [`ReconError::Transform`] from the §8.6.2 inverse transform.
pub fn extract_cu_residual(
    params: &ReconParams,
    tree: Option<&TransformTree>,
    x_cb: usize,
    y_cb: usize,
    n_cb_s: usize,
    qp_y: i32,
    transquant_bypass: bool,
) -> Result<CuResidual, ReconError> {
    let cat = params.chroma_array_type;
    let mut luma = CuResidualPlane::zeros(x_cb, y_cb, n_cb_s, n_cb_s);
    let (mut cb, mut cr) = if cat != 0 {
        let (sw, sh) = sub_wh_c(cat);
        let (cw, ch) = (n_cb_s / sw, n_cb_s / sh);
        let (cx, cy) = (x_cb / sw, y_cb / sh);
        (
            Some(CuResidualPlane::zeros(cx, cy, cw, ch)),
            Some(CuResidualPlane::zeros(cx, cy, cw, ch)),
        )
    } else {
        (None, None)
    };

    if let Some(tree) = tree {
        // The CU log2 size is the depth-0 transform-tree size.
        let log2_cb = n_cb_s.trailing_zeros();
        extract_residual_tree(
            params,
            tree,
            x_cb,
            y_cb,
            log2_cb,
            qp_y,
            transquant_bypass,
            &mut luma,
            cb.as_mut(),
            cr.as_mut(),
        )?;
    }

    Ok(CuResidual { luma, cb, cr })
}

/// Recursive helper for [`extract_cu_residual`]: walk a transform-tree node
/// and write each leaf's residual into the component planes.
//
// `as_deref_mut` here is the standard `Option<&mut T>` reborrow needed to
// reuse the optional chroma planes across the four-child loop; clippy's
// `needless_option_as_deref` misfires on the reborrow pattern.
#[allow(clippy::too_many_arguments, clippy::needless_option_as_deref)]
fn extract_residual_tree(
    params: &ReconParams,
    tree: &TransformTree,
    x0: usize,
    y0: usize,
    log2_trafo_size: u32,
    qp_y: i32,
    transquant_bypass: bool,
    luma: &mut CuResidualPlane,
    mut cb: Option<&mut CuResidualPlane>,
    mut cr: Option<&mut CuResidualPlane>,
) -> Result<(), ReconError> {
    match tree {
        TransformTree::Split { children, .. } => {
            let half = 1usize << (log2_trafo_size - 1);
            let offsets = [(0, 0), (half, 0), (0, half), (half, half)];
            for (child, (dx, dy)) in children.iter().zip(offsets) {
                extract_residual_tree(
                    params,
                    child,
                    x0 + dx,
                    y0 + dy,
                    log2_trafo_size - 1,
                    qp_y,
                    transquant_bypass,
                    luma,
                    cb.as_deref_mut(),
                    cr.as_deref_mut(),
                )?;
            }
            Ok(())
        }
        TransformTree::Leaf { unit, .. } => {
            let n_tbs = 1usize << log2_trafo_size;
            // §7.4.9.10 — thread the CuQpOffsetCb / CuQpOffsetCr state
            // in decode order on the inter residual path too.
            if let Some(off) = &unit.cu_chroma_qp_offset {
                apply_cu_chroma_qp_offset(params, off);
            }
            // §8.5.4.1 step 4 — a tu_residual_act_flag == 1 unit
            // (4:4:4) derives all three co-located residual arrays
            // (ACT-adjusted qP), applies cross-component prediction,
            // then the §8.6.8.2 inverse colour transform, before the
            // arrays land in the CU planes.
            if unit.tu_residual_act_flag == 1 && params.chroma_array_type == 3 {
                let mut r_y = match &unit.residual_luma {
                    Some(rb) => inter_residual_block(
                        params,
                        rb,
                        TfComponent::Luma,
                        luma_qp_act(params, qp_y),
                        transquant_bypass,
                    )?,
                    None => vec![0i32; n_tbs * n_tbs],
                };
                let chroma_res = |blocks: &[crate::hevc::engine::residual::ResidualBlock],
                                  coded: bool,
                                  cidx: TfComponent|
                 -> Result<Vec<i32>, ReconError> {
                    match blocks.first() {
                        Some(rb) if coded => inter_residual_block(
                            params,
                            rb,
                            cidx,
                            chroma_qp_act(params, qp_y, cidx, true),
                            transquant_bypass,
                        ),
                        _ => Ok(vec![0i32; n_tbs * n_tbs]),
                    }
                };
                let mut r_cb =
                    chroma_res(&unit.residual_cb, unit.cbf_cb_halves[0], TfComponent::Cb)?;
                let mut r_cr =
                    chroma_res(&unit.residual_cr, unit.cbf_cr_halves[0], TfComponent::Cr)?;
                for (r, ccp) in [
                    (&mut r_cb, unit.cross_comp_pred_cb.as_ref()),
                    (&mut r_cr, unit.cross_comp_pred_cr.as_ref()),
                ] {
                    if let Some(c) = CcpInput::resolve(params.chroma_array_type, ccp, Some(&r_y)) {
                        apply_cross_comp_pred(
                            r,
                            c.luma_residual,
                            c.res_scale_val,
                            params.bit_depth_luma,
                            params.bit_depth_chroma,
                        );
                    }
                }
                crate::hevc::engine::transform::act_inverse(
                    &mut r_y,
                    &mut r_cb,
                    &mut r_cr,
                    params.bit_depth_luma,
                    params.bit_depth_chroma,
                    params.extended_precision,
                    transquant_bypass,
                );
                luma.write_block(x0, y0, n_tbs, &r_y);
                if let Some(plane) = cb {
                    plane.write_block(x0, y0, n_tbs, &r_cb);
                }
                if let Some(plane) = cr {
                    plane.write_block(x0, y0, n_tbs, &r_cr);
                }
                return Ok(());
            }
            // Luma residual (§8.6.2 with the MODE_INTER path). The
            // array is kept for the §8.5.4.3 step-5 cross-component
            // prediction of this unit's chroma blocks (4:4:4).
            let luma_res: Option<Vec<i32>> = match &unit.residual_luma {
                Some(rb) => {
                    let qp = luma_qp(params, qp_y);
                    let res =
                        inter_residual_block(params, rb, TfComponent::Luma, qp, transquant_bypass)?;
                    luma.write_block(x0, y0, n_tbs, &res);
                    Some(res)
                }
                None => None,
            };
            // Chroma residuals, positioned at the chroma-subsampled
            // coordinates of this luma node. §7.3.8.10 deferred chroma:
            // a 4×4 luma leaf (ChromaArrayType != 3) carries no chroma
            // of its own — the blkIdx == 3 child holds the PARENT
            // node's chroma blocks, positioned at ( xBase, yBase ), the
            // 8×8 parent's top-left (x0 / y0 with the low 3 bits
            // cleared).
            if params.chroma_array_type != 0 {
                let (x_base, y_base) = if log2_trafo_size == 2 && params.chroma_array_type != 3 {
                    (x0 & !7, y0 & !7)
                } else {
                    (x0, y0)
                };
                let (sw, sh) = sub_wh_c(params.chroma_array_type);
                let (xc, yc) = (x_base / sw, y_base / sh);
                let luma_res = luma_res.as_deref();
                let ccp_cb = CcpInput::resolve(
                    params.chroma_array_type,
                    unit.cross_comp_pred_cb.as_ref(),
                    luma_res,
                );
                let ccp_cr = CcpInput::resolve(
                    params.chroma_array_type,
                    unit.cross_comp_pred_cr.as_ref(),
                    luma_res,
                );
                if let Some(plane) = cb {
                    let qp = chroma_qp(params, qp_y, TfComponent::Cb);
                    write_chroma_residual_blocks(
                        params,
                        &unit.residual_cb,
                        unit.cbf_cb_halves,
                        TfComponent::Cb,
                        qp,
                        transquant_bypass,
                        xc,
                        yc,
                        n_tbs,
                        ccp_cb,
                        plane,
                    )?;
                }
                if let Some(plane) = cr {
                    let qp = chroma_qp(params, qp_y, TfComponent::Cr);
                    write_chroma_residual_blocks(
                        params,
                        &unit.residual_cr,
                        unit.cbf_cr_halves,
                        TfComponent::Cr,
                        qp,
                        transquant_bypass,
                        xc,
                        yc,
                        n_tbs,
                        ccp_cr,
                        plane,
                    )?;
                }
            }
            Ok(())
        }
    }
}

/// Write a transform unit's chroma residual blocks into `plane`. The
/// `ChromaArrayType == 2` lower-half companion (when present as a second
/// block) is positioned `1 << log2TrafoSizeC` samples below the first.
///
/// `n_tbs_c` is the §7.3.8.10 chroma transform-block side of this unit
/// (equal to the luma side for 4:4:4) — needed when a `ccp` residual
/// modification must materialize a block whose cbf is clear. `ccp`
/// carries the §8.5.4.3 step-5 cross-component prediction input.
#[allow(clippy::too_many_arguments)]
fn write_chroma_residual_blocks(
    params: &ReconParams,
    blocks: &[crate::hevc::engine::residual::ResidualBlock],
    coded_halves: [bool; 2],
    cidx: TfComponent,
    qp: u32,
    transquant_bypass: bool,
    xc: usize,
    yc: usize,
    n_tbs_c: usize,
    ccp: Option<CcpInput<'_>>,
    plane: &mut CuResidualPlane,
) -> Result<(), ReconError> {
    // §7.3.8.10 codes the halves upper-then-lower, each gated on its
    // own cbf. The stacked pair exists ONLY for ChromaArrayType == 2
    // (§8.5.4.3: blkIdx proceeds over 0..( ChromaArrayType == 2 ? 1 :
    // 0 )) — 4:4:4 also has SubHeightC == 1 but codes a single chroma
    // block per transform unit, so keying on `sub_h` alone would
    // synthesize a phantom second block one TU-height below (visible
    // as a spurious §8.6.6 cross-component application on 4:4:4
    // cbf-clear blocks).
    let mut next_block = 0usize;
    let halves = if params.chroma_array_type == 2 { 2 } else { 1 };
    for (v, &coded) in coded_halves.iter().enumerate().take(halves) {
        let (mut res, n) = if coded {
            let Some(rb) = blocks.get(next_block) else {
                break;
            };
            next_block += 1;
            let n = rb.size();
            (
                inter_residual_block(params, rb, cidx, qp, transquant_bypass)?,
                n,
            )
        } else if ccp.is_some() {
            // §8.5.4.3 step 5 applies to every chroma transform block
            // of the unit; a cbf-clear block starts from zeros.
            (vec![0i32; n_tbs_c * n_tbs_c], n_tbs_c)
        } else {
            continue;
        };
        if let Some(c) = ccp {
            apply_cross_comp_pred(
                &mut res,
                c.luma_residual,
                c.res_scale_val,
                params.bit_depth_luma,
                params.bit_depth_chroma,
            );
        }
        let by = if v == 1 { yc + n } else { yc };
        plane.write_block(xc, by, n, &res);
    }
    Ok(())
}

/// §8.6.2 — dequantize + inverse-transform one residual block in the
/// `MODE_INTER` path (the §8.6.4 `trType` selection always picks DCT-II for
/// inter, vs. the §8.6.4.1 DST-VII gate that only fires for 4×4 luma intra).
fn inter_residual_block(
    params: &ReconParams,
    rb: &crate::hevc::engine::residual::ResidualBlock,
    cidx: TfComponent,
    qp: u32,
    transquant_bypass: bool,
) -> Result<Vec<i32>, ReconError> {
    let n_tbs = rb.size();
    let bit_depth = match cidx {
        TfComponent::Luma => params.bit_depth_luma,
        TfComponent::Cb | TfComponent::Cr => params.bit_depth_chroma,
    };
    let bp = BlockParams {
        n_tbs,
        q_p: qp,
        component: cidx,
        pred_mode: PredMode::Inter,
        bit_depth,
        extended_precision: params.extended_precision,
        transquant_bypass,
        transform_skip: rb.transform_skip,
        transform_skip_rotation_enabled: params.transform_skip_rotation_enabled,
    };
    let m = scaling_matrix(params, n_tbs, PredMode::Inter, cidx, rb.transform_skip);
    let mut r = residual_block(&rb.levels, m, bp)?;
    // §8.5.4.2 step 3 / §8.5.4.3 step 4: when explicit_rdpcm_flag is 1,
    // the §8.6.5 directional residual modification is invoked with
    // mDir = explicit_rdpcm_dir_flag (0 horizontal, 1 vertical).
    if rb.explicit_rdpcm_flag {
        rdpcm_accumulate(&mut r, n_tbs, rb.explicit_rdpcm_dir_flag);
    }
    Ok(r)
}

/// Per-picture intra-reconstruction neighbour state — the §8.4.2
/// `IntraPredModeY` field plus the §6.4.1 picture tiling that resolves
/// neighbour availability.
///
/// One [`ReconCtx`] is shared across every CTU of a slice so the §8.4.2
/// most-probable-mode derivation sees the actual left / above neighbour
/// modes (rather than the flat-single-CU `INTRA_DC` assumption). Build it
/// with [`ReconCtx::new`]; reconstruct each CTU in tile-scan order with
/// [`reconstruct_intra_ctu_ctx`].
#[derive(Debug)]
pub struct ReconCtx {
    field: IntraModeField,
    tiling: PictureTiling,
    /// `SliceAddrRs[ ctbAddrRs ]` — the raster address of the first CTB
    /// of the independent slice segment that owns each CTB. All-zero for a
    /// single-slice picture; the multi-slice driver populates it so the
    /// §6.4.1 z-scan availability denies neighbours across slice
    /// boundaries.
    slice_addr_rs: Vec<u32>,
    /// §8.6.1 per-picture luma-QP derivation state (`None` for the
    /// standalone per-CTU helpers, which fall back to the
    /// `qPY_PRED == SliceQpY` single-QG shortcut).
    qp: Option<QpState>,
    /// `constrained_intra_pred_flag` (§8.4.4.2.1) — when set, reference
    /// samples from non-`MODE_INTRA` coding units are "not available
    /// for intra prediction".
    constrained_intra: bool,
}

/// §8.6.1 quantization-parameter derivation state: the per-4×4 `QpY`
/// map (feeding the `qPY_A` / `qPY_B` neighbour reads and the §8.7.2
/// deblocking QP), the decode-order `qPY_PREV` thread, and the current
/// quantization group.
#[derive(Debug)]
struct QpState {
    /// Per-4×4-cell `QpY`.
    map: Vec<i8>,
    w_cells: usize,
    h_cells: usize,
    /// `Log2MinCuQpDeltaSize`.
    qg_log2: u32,
    /// `CtbLog2SizeY` (the §8.6.1 same-CTB gate on `qPY_A` / `qPY_B`).
    ctb_log2: u32,
    /// `SliceQpY`.
    slice_qp_y: i32,
    /// `QpBdOffsetY`.
    qp_bd_offset_y: i32,
    /// `QpY` of the most recent coding unit (becomes `qPY_PREV` when the
    /// next quantization group starts). `None` before the first QG of
    /// the slice.
    last_cu_qp: Option<i32>,
    /// The current quantization group's grid cell + its `qPY_PRED`.
    cur_qg: Option<(usize, usize)>,
    cur_pred: i32,
    /// `CuQpDeltaVal` accumulated for the current QG (0 until a CU in
    /// the group carries the decoded delta).
    cur_delta: i32,
    /// The most recent CU origin `derive_cu_qp` resolved (idempotence
    /// guard: the intra path may query the same CU twice).
    last_cu: Option<(usize, usize)>,
    last_qp: i32,
}

impl QpState {
    fn cell(&self, x: usize, y: usize) -> usize {
        (y >> 2).min(self.h_cells - 1) * self.w_cells + (x >> 2).min(self.w_cells - 1)
    }

    fn qp_at(&self, x: usize, y: usize) -> i32 {
        i32::from(self.map[self.cell(x, y)])
    }
}

impl ReconCtx {
    /// Build the neighbour context for a `pic_width` × `pic_height` luma
    /// picture with the given `CtbLog2SizeY` / `MinTbLog2SizeY` and tile
    /// layout.
    ///
    /// # Errors
    /// Propagates [`crate::hevc::engine::availability::AvailabilityError`] (wrapped in
    /// [`ReconError::Tiling`]) when the geometry is degenerate.
    pub fn new(
        pic_width_luma: usize,
        pic_height_luma: usize,
        ctb_log2_size_y: u32,
        min_tb_log2_size_y: u32,
        tiles: &crate::hevc::engine::availability::TilingParams,
    ) -> Result<Self, ReconError> {
        let ctb_size = 1usize << ctb_log2_size_y;
        let pic_w_ctbs = pic_width_luma.div_ceil(ctb_size) as u32;
        let pic_h_ctbs = pic_height_luma.div_ceil(ctb_size) as u32;
        let tiling = PictureTiling::new(
            pic_w_ctbs,
            pic_h_ctbs,
            pic_width_luma as u32,
            pic_height_luma as u32,
            ctb_log2_size_y,
            min_tb_log2_size_y,
            tiles,
        )
        .map_err(ReconError::Tiling)?;
        Ok(Self {
            field: IntraModeField::new(pic_width_luma, pic_height_luma, ctb_log2_size_y),
            tiling,
            slice_addr_rs: vec![0u32; (pic_w_ctbs * pic_h_ctbs) as usize],
            qp: None,
            constrained_intra: false,
        })
    }

    /// Enable the §8.4.4.2.1 `constrained_intra_pred_flag` gate:
    /// reference samples whose covering coding unit is not
    /// `MODE_INTRA` are treated as "not available for intra
    /// prediction" (they substitute per §8.4.4.2.2).
    pub fn set_constrained_intra(&mut self, on: bool) {
        self.constrained_intra = on;
    }

    /// Initialize the §8.6.1 per-picture QP-derivation state.
    ///
    /// `slice_qp_y` is `SliceQpY`; `qg_log2` is `Log2MinCuQpDeltaSize`
    /// (§7.4.3.3.1: `CtbLog2SizeY − diff_cu_qp_delta_depth`);
    /// `qp_bd_offset_y` is `QpBdOffsetY`. Must be called before the CU
    /// walk when the picture carries `cu_qp_delta` values; without it
    /// the per-CU derivation falls back to the `qPY_PRED == SliceQpY`
    /// single-QG shortcut.
    pub fn init_qp_state(&mut self, slice_qp_y: i32, qg_log2: u32, qp_bd_offset_y: i32) {
        // The 4×4 cell grid matches the intra-mode field's min-block grid.
        let w_cells = self.field.w_blocks();
        let h_cells = self.field.h_blocks();
        self.qp = Some(QpState {
            map: vec![slice_qp_y as i8; w_cells * h_cells],
            w_cells,
            h_cells,
            qg_log2,
            ctb_log2: self.field.ctb_log2(),
            slice_qp_y,
            qp_bd_offset_y,
            last_cu_qp: None,
            cur_qg: None,
            cur_pred: slice_qp_y,
            cur_delta: 0,
            last_cu: None,
            last_qp: slice_qp_y,
        });
    }

    /// §8.6.1 — derive `QpY` for the coding unit at `(x_cb, y_cb)` of
    /// side `1 << log2_cb_size`, whose transform tree carries
    /// `cu_delta` (`Some` when a `cu_qp_delta` was decoded in it).
    ///
    /// Must be invoked once per coding unit **in decode order**; the
    /// derivation threads `qPY_PREV` across quantization groups and
    /// stamps the per-4×4 QP map. Re-invoking for the same CU origin
    /// returns the cached value. Returns the fallback single-QG value
    /// when [`Self::init_qp_state`] was not called.
    pub fn derive_cu_qp(
        &mut self,
        params: &ReconParams,
        x_cb: usize,
        y_cb: usize,
        log2_cb_size: u32,
        cu_delta: Option<i32>,
    ) -> i32 {
        let Some(_) = self.qp else {
            return fallback_qp_y(params, cu_delta.unwrap_or(0));
        };
        // Idempotence: the picture drivers may resolve the same CU twice
        // (once for the deblock descriptor, once inside the intra path).
        if let Some(qp) = self
            .qp
            .as_ref()
            .and_then(|q| (q.last_cu == Some((x_cb, y_cb))).then_some(q.last_qp))
        {
            return qp;
        }
        // The quantization-group origin (§8.6.1): xQg = xCb − (xCb &
        // ((1 << Log2MinCuQpDeltaSize) − 1)).
        let (qg_log2, cur_qg, ctb_log2) = {
            let q = self.qp.as_ref().unwrap();
            (q.qg_log2, q.cur_qg, q.ctb_log2)
        };
        let x_qg = x_cb & !((1usize << qg_log2) - 1);
        let y_qg = y_cb & !((1usize << qg_log2) - 1);
        let qg = (x_qg >> qg_log2, y_qg >> qg_log2);
        if cur_qg != Some(qg) {
            // New quantization group: derive qPY_PRED (steps 1–4).
            let q = self.qp.as_ref().unwrap();
            // Step 1 — qPY_PREV (first QG in slice ⇒ SliceQpY; the tile
            // / WPP CTB-row resets are follow-ups of those subsystems).
            let qp_prev = q.last_cu_qp.unwrap_or(q.slice_qp_y);
            // Step 2 — qPY_A from (xQg − 1, yQg), same-CTB gated.
            let same_ctb_a = x_qg > 0
                && (x_qg - 1) >> ctb_log2 == x_cb >> ctb_log2
                && y_qg >> ctb_log2 == y_cb >> ctb_log2;
            let avail_a =
                x_qg > 0 && same_ctb_a && self.available(x_cb, y_cb, x_qg as i64 - 1, y_qg as i64);
            let q = self.qp.as_ref().unwrap();
            let qp_a = if avail_a {
                q.qp_at(x_qg - 1, y_qg)
            } else {
                qp_prev
            };
            // Step 3 — qPY_B from (xQg, yQg − 1), same-CTB gated.
            let same_ctb_b = y_qg > 0
                && (y_qg - 1) >> ctb_log2 == y_cb >> ctb_log2
                && x_qg >> ctb_log2 == x_cb >> ctb_log2;
            let avail_b =
                y_qg > 0 && same_ctb_b && self.available(x_cb, y_cb, x_qg as i64, y_qg as i64 - 1);
            let q = self.qp.as_mut().unwrap();
            let qp_b = if avail_b {
                q.qp_at(x_qg, y_qg - 1)
            } else {
                qp_prev
            };
            // Step 4 — qPY_PRED (eq. 8-282).
            q.cur_pred = (qp_a + qp_b + 1) >> 1;
            q.cur_qg = Some(qg);
            q.cur_delta = 0;
        }
        let q = self.qp.as_mut().unwrap();
        // §7.4.9.14: CuQpDeltaVal applies from the CU that decodes it to
        // the end of the quantization group.
        if let Some(delta) = cu_delta {
            q.cur_delta = delta;
        }
        // Eq. 8-283.
        let modulus = 52 + q.qp_bd_offset_y;
        let qp_y = (q.cur_pred + q.cur_delta + 52 + 2 * q.qp_bd_offset_y).rem_euclid(modulus)
            - q.qp_bd_offset_y;
        // Stamp the CU area for later qPY_A / qPY_B reads + deblocking.
        let n = 1usize << log2_cb_size;
        for y in (y_cb..(y_cb + n).min(q.h_cells * 4)).step_by(4) {
            for x in (x_cb..(x_cb + n).min(q.w_cells * 4)).step_by(4) {
                let idx = q.cell(x, y);
                q.map[idx] = qp_y as i8;
            }
        }
        q.last_cu_qp = Some(qp_y);
        q.last_cu = Some((x_cb, y_cb));
        q.last_qp = qp_y;
        qp_y
    }

    /// The `SliceAddrRs` of the CTB containing the luma position.
    #[must_use]
    pub fn slice_addr_rs_of_luma(&self, x_luma: usize, y_luma: usize) -> u32 {
        let ctb_log2 = self.field.ctb_log2();
        let w = self.tiling.pic_width_in_ctbs_y() as usize;
        let rs = (y_luma >> ctb_log2) * w + (x_luma >> ctb_log2);
        self.slice_addr_of(rs as u32)
    }

    /// The §6.5.1 `TileId` of the CTB containing the luma position.
    #[must_use]
    pub fn tile_id_of_luma(&self, x_luma: usize, y_luma: usize) -> u32 {
        let ctb_log2 = self.field.ctb_log2();
        let w = self.tiling.pic_width_in_ctbs_y() as usize;
        let rs = (y_luma >> ctb_log2) * w + (x_luma >> ctb_log2);
        self.tiling
            .tile_id(self.tiling.ctb_addr_rs_to_ts(rs as u32))
    }

    /// The per-4×4-cell §8.6.1 `QpY` map (for the §8.7.2 per-position
    /// `QpQ` / `QpP` reads). `None` when no QP state is initialized.
    #[must_use]
    pub fn qp_cells(&self) -> Option<(&[i8], usize)> {
        self.qp.as_ref().map(|q| (q.map.as_slice(), q.w_cells))
    }

    /// §8.6.1 step-1 — reset `qPY_PREV` to `SliceQpY` (invoked at the
    /// first quantization group of a CTB row when
    /// `entropy_coding_sync_enabled_flag` is set, and at tile starts).
    pub fn reset_qp_prev(&mut self) {
        if let Some(q) = self.qp.as_mut() {
            q.last_cu_qp = None;
        }
    }

    /// The §8.6.1-derived `QpY` recorded at a luma location (for the
    /// deblocking p-side QP). Returns `None` when no QP state is
    /// initialized.
    #[must_use]
    pub fn qp_y_at(&self, x_luma: usize, y_luma: usize) -> Option<i32> {
        self.qp.as_ref().map(|q| q.qp_at(x_luma, y_luma))
    }

    /// Debug: the recon-side §8.4.2 derived mode recorded at a luma
    /// location.
    #[doc(hidden)]
    #[must_use]
    pub fn debug_mode_at(&self, x_luma: usize, y_luma: usize) -> Option<u8> {
        self.field.recorded_mode(x_luma, y_luma)
    }

    /// Set the per-CTB `SliceAddrRs` map (one entry per CTB raster
    /// address, `PicSizeInCtbsY` long). Each entry is the raster address
    /// of the first CTB of the independent slice segment owning that CTB.
    /// Used by the multi-slice driver so the §6.4.1 z-scan availability
    /// denies cross-slice neighbours.
    ///
    /// # Panics
    /// Panics if `map.len()` is not `PicSizeInCtbsY`.
    pub fn set_slice_addr_rs(&mut self, map: Vec<u32>) {
        assert_eq!(
            map.len(),
            self.slice_addr_rs.len(),
            "SliceAddrRs map must be PicSizeInCtbsY long"
        );
        self.slice_addr_rs = map;
    }

    /// `SliceAddrRs[ ctbAddrRs ]` for the CTB raster address.
    #[inline]
    fn slice_addr_of(&self, ctb_rs: u32) -> u32 {
        self.slice_addr_rs
            .get(ctb_rs as usize)
            .copied()
            .unwrap_or(0)
    }

    /// §6.4.1 z-scan availability of the neighbour luma location
    /// `( x_nb, y_nb )` for the current prediction block at
    /// `( x_curr, y_curr )`, consulting the per-CTB `SliceAddrRs` map so
    /// neighbours in a different slice segment are unavailable.
    fn available(&self, x_curr: usize, y_curr: usize, x_nb: i64, y_nb: i64) -> bool {
        self.tiling.z_scan_availability(
            x_curr as u32,
            y_curr as u32,
            x_nb as i32,
            y_nb as i32,
            |ctb_rs| self.slice_addr_of(ctb_rs),
        )
    }

    /// §8.4.4.2.1 reference-sample availability for one neighbour sample of
    /// a transform block. `(x_tb, y_tb)` is the current transform block's
    /// **plane** top-left and `(x_ref, y_ref)` the neighbour's **plane**
    /// coordinates; `(sub_w, sub_h)` is the plane's `(SubWidthC, SubHeightC)`
    /// (`(1, 1)` for luma). The §6.4.1 z-scan availability is evaluated on
    /// the corresponding luma locations.
    fn ref_sample_available(
        &self,
        x_tb: usize,
        y_tb: usize,
        x_ref: i64,
        y_ref: i64,
        sub_w: usize,
        sub_h: usize,
    ) -> bool {
        if x_ref < 0 || y_ref < 0 {
            return false;
        }
        // Map plane coordinates to luma for the §6.4.1 test.
        let x_curr_luma = x_tb * sub_w;
        let y_curr_luma = y_tb * sub_h;
        let x_nb_luma = (x_ref as usize * sub_w) as i64;
        let y_nb_luma = (y_ref as usize * sub_h) as i64;
        if !self.available(x_curr_luma, y_curr_luma, x_nb_luma, y_nb_luma) {
            return false;
        }
        // §8.4.4.2.1: CuPredMode[ xNbY ][ yNbY ] != MODE_INTRA with
        // constrained_intra_pred_flag == 1 ⇒ not available.
        !self.constrained_intra
            || self
                .field
                .is_intra_at(x_nb_luma as usize, y_nb_luma as usize)
    }

    /// Borrow the §6.4.1 / §6.5 picture tiling (the inter driver's §6.4.2
    /// prediction-block availability reads it).
    #[must_use]
    pub fn tiling(&self) -> &PictureTiling {
        &self.tiling
    }

    /// `SliceAddrRs[ ctbAddrRs ]` for the CTB containing the luma location
    /// (the inter driver's §6.4.2 availability uses the same map).
    #[must_use]
    pub fn slice_addr_rs_of(&self, ctb_rs: u32) -> u32 {
        self.slice_addr_of(ctb_rs)
    }

    /// Test-only: the §8.4.2 `IntraPredModeY` recorded at a luma location.
    #[cfg(any())]
    #[must_use]
    pub(crate) fn recorded_mode(&self, x_luma: usize, y_luma: usize) -> Option<u8> {
        self.field.recorded_mode(x_luma, y_luma)
    }
}

/// Reconstruct one decoded coding tree unit's intra samples into `pic`,
/// driving the §8.4.2 most-probable-mode derivation off the shared
/// [`ReconCtx`] neighbour field. CTUs must be reconstructed in tile-scan
/// order so each one's left / above neighbours are already recorded.
///
/// # Errors
/// [`ReconError::InterNotSupported`] if any leaf coding unit is inter;
/// [`ReconError::IntraPred`] / [`ReconError::Transform`] on a primitive
/// failure.
pub fn reconstruct_intra_ctu_ctx(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &mut ReconCtx,
    ctu: &CodingTreeUnit,
) -> Result<(), ReconError> {
    reconstruct_quadtree(pic, params, ctx, &ctu.quadtree)
}

/// Reconstruct one **intra** leaf coding unit into `pic`, recording its
/// §8.4.2 `IntraPredModeY` into the shared [`ReconCtx`] neighbour field.
///
/// This is the per-CU entry the §8.5 picture-level inter driver calls for
/// an intra coding unit embedded in a P / B slice (mixed-mode pictures);
/// the inter coding units of the same picture go through
/// [`crate::hevc::engine::inter_recon::resolve_and_reconstruct_inter_cu`].
///
/// # Errors
/// [`ReconError::IntraPred`] / [`ReconError::Transform`] on a primitive
/// failure; [`ReconError::InterNotSupported`] if `cu` is not intra.
pub fn reconstruct_intra_cu_ctx(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &mut ReconCtx,
    cu: &CodingUnit,
) -> Result<(), ReconError> {
    reconstruct_cu(pic, params, ctx, cu)
}

/// Reconstruct one decoded coding tree unit's intra samples into `pic`.
/// `(ctb_x, ctb_y)` is the CTB's luma top-left position.
///
/// This single-CTU convenience builds a fresh single-CTU [`ReconCtx`]
/// internally — its neighbours are the CTU's own already-reconstructed
/// blocks, so a multi-CTU picture must instead share one [`ReconCtx`]
/// across CTUs via [`reconstruct_intra_ctu_ctx`].
///
/// # Errors
/// [`ReconError::InterNotSupported`] if any leaf coding unit is inter;
/// [`ReconError::IntraPred`] / [`ReconError::Transform`] on a primitive
/// failure.
pub fn reconstruct_intra_ctu(
    pic: &mut Picture,
    params: &ReconParams,
    ctu: &CodingTreeUnit,
) -> Result<(), ReconError> {
    // A single CTB covers the whole synthetic picture for this path; the
    // CTB log2 size is the smallest power of two covering the picture.
    let max_dim = pic.width_luma().max(pic.height_luma()).max(1);
    let ctb_log2 = (usize::BITS - (max_dim - 1).leading_zeros()).max(4);
    let min_tb_log2 = 2;
    let mut ctx = ReconCtx::new(
        pic.width_luma(),
        pic.height_luma(),
        ctb_log2,
        min_tb_log2,
        &crate::hevc::engine::availability::TilingParams::single_tile(),
    )?;
    reconstruct_intra_ctu_ctx(pic, params, &mut ctx, ctu)
}

/// One slice segment's §7.4.7.1 `SliceAddrRs`-derivation inputs: the
/// `slice_segment_address` (raster CTB address of its first CTB) and the
/// `dependent_slice_segment_flag`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceSegmentBoundary {
    /// `slice_segment_address` (raster CTB address).
    pub slice_segment_address: u32,
    /// `dependent_slice_segment_flag`.
    pub dependent: bool,
}

/// §7.4.7.1 — build the per-CTB `SliceAddrRs[ ctbAddrRs ]` map for a
/// picture from its ordered slice segments.
///
/// `segments` are the picture's coded slice segments **in decode
/// (tile-scan-start) order**; each carries its `slice_segment_address`
/// (raster) and `dependent_slice_segment_flag`. An independent segment
/// (`dependent == false`) sets `SliceAddrRs = slice_segment_address`; a
/// dependent segment inherits the active independent segment's
/// `SliceAddrRs`. CTBs are partitioned in tile-scan order, so the run of
/// CTBs owned by a segment starts at `CtbAddrRsToTs[slice_segment_address]`
/// and extends until the next segment's tile-scan start.
///
/// Returns a `PicSizeInCtbsY`-long vector indexed by CTB **raster**
/// address. CTBs before the first segment (which shouldn't happen for a
/// well-formed picture, since the first segment starts at address 0) carry
/// `SliceAddrRs = 0`.
#[must_use]
pub fn build_slice_addr_map(tiling: &PictureTiling, segments: &[SliceSegmentBoundary]) -> Vec<u32> {
    let pic_size = (tiling.pic_width_in_ctbs_y() * tiling.pic_height_in_ctbs_y()) as usize;
    let mut map = vec![0u32; pic_size];

    // Resolve each segment's SliceAddrRs and its tile-scan start address.
    // (slice_addr_rs, ctb_addr_ts_start)
    let mut resolved: Vec<(u32, u32)> = Vec::with_capacity(segments.len());
    let mut active_slice_addr_rs = 0u32;
    for seg in segments {
        let slice_addr_rs = if seg.dependent {
            active_slice_addr_rs
        } else {
            seg.slice_segment_address
        };
        active_slice_addr_rs = slice_addr_rs;
        let ts_start = tiling.ctb_addr_rs_to_ts(seg.slice_segment_address);
        resolved.push((slice_addr_rs, ts_start));
    }
    // Sort by tile-scan start so the run boundaries are monotonic.
    resolved.sort_by_key(|&(_, ts)| ts);

    // Walk every CTB in tile-scan order, assigning the SliceAddrRs of the
    // latest segment whose tile-scan start is <= the CTB's tile-scan addr.
    for ts in 0..pic_size as u32 {
        // The owning segment is the last one with ts_start <= ts.
        let owner = resolved
            .iter()
            .rev()
            .find(|&&(_, ts_start)| ts_start <= ts)
            .map_or(0, |&(addr, _)| addr);
        let rs = tiling.ctb_addr_ts_to_rs(ts);
        map[rs as usize] = owner;
    }
    map
}

/// Picture-level intra decode parameters — the SPS/PPS/slice constants the
/// multi-CTU driver [`reconstruct_intra_picture`] needs beyond the
/// per-block [`ReconParams`].
#[derive(Debug, Clone)]
pub struct IntraPictureParams {
    /// `CtbLog2SizeY`.
    pub ctb_log2_size_y: u32,
    /// `MinTbLog2SizeY` (= `log2_min_luma_transform_block_size_minus2 + 2`).
    pub min_tb_log2_size_y: u32,
    /// The active PPS tile layout (§6.5.1).
    pub tiles: crate::hevc::engine::availability::TilingParams,
    /// `slice_sao_luma_flag` (§8.7.3.1 luma gate).
    pub slice_sao_luma_flag: bool,
    /// `slice_sao_chroma_flag` (§8.7.3.1 chroma gate).
    pub slice_sao_chroma_flag: bool,
    /// `log2_sao_offset_scale_luma` (§7.4.3.3.2; 0 for 8-bit Main).
    pub log2_sao_offset_scale_luma: u8,
    /// `log2_sao_offset_scale_chroma`.
    pub log2_sao_offset_scale_chroma: u8,
}

/// One decoded CTU positioned in the picture for the multi-CTU driver.
#[derive(Debug)]
pub struct PlacedCtu<'a> {
    /// The CTB's luma top-left `( x_ctb, y_ctb )`.
    pub x_ctb: u32,
    /// The CTB's luma top-left `( x_ctb, y_ctb )`.
    pub y_ctb: u32,
    /// `SliceAddrRs` — the raster address of the first CTB of the
    /// independent slice segment that owns this CTB. `0` for a
    /// single-slice picture; the multi-slice driver
    /// ([`reconstruct_intra_multislice_picture`]) sets it to the slice's
    /// `slice_segment_address` so cross-slice neighbours are denied.
    pub slice_addr_rs: u32,
    /// The decoded coding tree unit.
    pub ctu: &'a CodingTreeUnit,
}

/// Reconstruct a full intra picture from its decoded CTUs and apply the
/// §8.7.3 sample-adaptive-offset in-loop filter.
///
/// `ctus` are the picture's decoded coding tree units **in tile-scan
/// (decode) order** — each one's left / above neighbours must already be
/// reconstructed when it is processed, which the tile-scan order
/// guarantees. The driver shares one [`ReconCtx`] across all CTUs so the
/// §8.4.2 most-probable-mode derivation sees the true neighbour modes, then
/// resolves each CTB's [`crate::hevc::engine::sao::ResolvedSao`] (honouring
/// `sao_merge_left_flag` / `sao_merge_up_flag`) and runs
/// [`crate::hevc::engine::sao::apply_sao_picture`].
///
/// The returned picture is the §8.7.3 SAO output (the in-loop deblocking
/// filter, §8.7.2, is applied by the caller via
/// [`crate::hevc::engine::deblock::deblock_picture`] before SAO when deblocking is
/// enabled; this driver covers the recon + SAO stages).
///
/// # Errors
/// Propagates [`ReconError`] from the per-CTU reconstruction (including
/// [`ReconError::Tiling`] for degenerate geometry).
pub fn reconstruct_intra_picture(
    pic_width_luma: usize,
    pic_height_luma: usize,
    params: &ReconParams,
    pic_params: &IntraPictureParams,
    ctus: &[PlacedCtu<'_>],
) -> Result<Picture, ReconError> {
    let mut pic = Picture::new(
        pic_width_luma,
        pic_height_luma,
        params.chroma_array_type,
        params.bit_depth_luma,
        params.bit_depth_chroma,
    );
    let mut ctx = ReconCtx::new(
        pic_width_luma,
        pic_height_luma,
        pic_params.ctb_log2_size_y,
        pic_params.min_tb_log2_size_y,
        &pic_params.tiles,
    )?;

    let ctb_size = 1usize << pic_params.ctb_log2_size_y;
    let pic_w_ctbs = pic_width_luma.div_ceil(ctb_size);
    let pic_h_ctbs = pic_height_luma.div_ceil(ctb_size);

    // Build the per-CTB SliceAddrRs map from the placed CTUs (default 0
    // for any CTB not covered) and feed it to the neighbour context so the
    // §6.4.1 z-scan availability denies cross-slice neighbours.
    let mut slice_addr_map = vec![0u32; pic_w_ctbs * pic_h_ctbs];
    for placed in ctus {
        let rx = (placed.x_ctb as usize) >> pic_params.ctb_log2_size_y;
        let ry = (placed.y_ctb as usize) >> pic_params.ctb_log2_size_y;
        slice_addr_map[ry * pic_w_ctbs + rx] = placed.slice_addr_rs;
    }
    ctx.set_slice_addr_rs(slice_addr_map.clone());

    // §8.7.3.1 resolved-SAO grid (raster order), default all-off so a CTU
    // not present in `ctus` leaves its CTB unmodified.
    let mut sao_grid = vec![crate::hevc::engine::sao::ResolvedSao::off(); pic_w_ctbs * pic_h_ctbs];

    for placed in ctus {
        reconstruct_intra_ctu_ctx(&mut pic, params, &mut ctx, placed.ctu)?;

        // §7.4.9.3 SAO merge: resolve against the already-resolved left /
        // above CTB in the grid, but only when that neighbour is in the
        // SAME slice segment (the merge candidate availability follows the
        // §6.4.1 slice-boundary rule). A neighbour in a different slice is
        // not a merge candidate.
        let rx = (placed.x_ctb as usize) >> pic_params.ctb_log2_size_y;
        let ry = (placed.y_ctb as usize) >> pic_params.ctb_log2_size_y;
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
                pic_params.log2_sao_offset_scale_luma,
                pic_params.log2_sao_offset_scale_chroma,
            );
        }
    }

    // §8.7.3.1 — apply SAO across the whole picture (no-op when both slice
    // SAO flags are clear or every CTB resolved to type 0).
    let filtered = crate::hevc::engine::sao::apply_sao_picture(
        pic,
        &sao_grid,
        pic_params.ctb_log2_size_y,
        params.chroma_array_type,
        pic_params.slice_sao_luma_flag,
        pic_params.slice_sao_chroma_flag,
    );
    Ok(filtered)
}

fn reconstruct_quadtree(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &mut ReconCtx,
    qt: &CodingQuadtree,
) -> Result<(), ReconError> {
    match qt {
        CodingQuadtree::Split(children) => {
            for child in children {
                reconstruct_quadtree(pic, params, ctx, child)?;
            }
            Ok(())
        }
        CodingQuadtree::Leaf(cu) => reconstruct_cu(pic, params, ctx, cu),
    }
}

/// §8.4.2 — derive `IntraPredModeY` for one luma prediction block at
/// `( x_pb, y_pb )` of side `n_pb`, consulting the [`ReconCtx`] neighbour
/// field for the candidate modes, then record it back into the field.
fn derive_and_record_luma_mode(
    ctx: &mut ReconCtx,
    x_pb: usize,
    y_pb: usize,
    n_pb: usize,
    luma_mode: &IntraLumaMode,
    pcm_flag: bool,
) -> u8 {
    // Step 1 / 2 — candidate modes from the left (A) and above (B)
    // neighbours, gated on §6.4.1 z-scan availability.
    let avail_a = ctx.available(x_pb, y_pb, x_pb as i64 - 1, y_pb as i64);
    let avail_b = ctx.available(x_pb, y_pb, x_pb as i64, y_pb as i64 - 1);
    let cand_a = ctx
        .field
        .cand_intra_pred_mode(x_pb, y_pb, Neighbour::Left, avail_a);
    let cand_b = ctx
        .field
        .cand_intra_pred_mode(x_pb, y_pb, Neighbour::Above, avail_b);

    // Step 3 / 4 — candModeList + the prev_intra_luma_pred_flag selection.
    let cand_list = intra_luma_cand_mode_list(cand_a, cand_b);
    let source = luma_intra_mode_source_from_flag(u8::from(luma_mode.prev_intra_luma_pred_flag));
    let field_val = match source {
        LumaIntraModeSource::Mpm => luma_mode.mpm_idx.unwrap_or(0),
        LumaIntraModeSource::Remaining => luma_mode.rem_intra_luma_pred_mode.unwrap_or(0),
    };
    let mode = derive_intra_pred_mode_y(cand_list, source, field_val);
    ctx.field.record_intra_pb(x_pb, y_pb, n_pb, mode, pcm_flag);
    mode
}

/// Reconstruct one leaf coding unit. Only intra CUs are handled; each luma
/// prediction block's `IntraPredModeY` is derived per §8.4.2 from the
/// [`ReconCtx`] neighbour field (most-probable-mode), and chroma
/// `IntraPredModeC` per §8.4.3 from the first PB's luma mode.
/// §8.4.1 — write a PCM coding unit's (already scaled) samples into
/// the picture: `SL[xCb+i][yCb+j] = pcm_sample_luma[nCbS*j + i] <<
/// (BitDepthY − PcmBitDepthY)` (equation 8-12; the shift was applied
/// at parse time) and the chroma analogues.
/// §8.4.4.2.7 — write a palette CU's reconstructed components into
/// the picture. `qp_y` is the §8.6.1-derived QpY of the CU; per-
/// component `Qp′` values (eq. 8-73..8-75) feed the escape
/// dequantization.
fn write_palette_cu(
    pic: &mut Picture,
    params: &ReconParams,
    pal: &crate::hevc::engine::palette::PaletteCu,
    x_cb: usize,
    y_cb: usize,
    qp_y: i32,
    transquant_bypass: bool,
) {
    // §7.4.9.10 — a chroma_qp_offset( ) parsed inside
    // palette_coding( ) updates the slice-wide state exactly like a
    // transform unit's.
    if let Some(off) = &pal.cu_chroma_qp_offset {
        apply_cu_chroma_qp_offset(params, off);
    }
    let qp_luma = luma_qp(params, qp_y) as i32;
    crate::hevc::engine::palette::reconstruct_palette_component(
        pal,
        0,
        1,
        1,
        qp_luma,
        u32::from(params.bit_depth_luma),
        transquant_bypass,
        |x, y, v| pic.set_sample(Plane::Luma, x_cb + x, y_cb + y, v),
    );
    if params.chroma_array_type != 0 {
        let (sub_w, sub_h) = sub_wh_c(params.chroma_array_type);
        let (cx, cy) = (x_cb / sub_w, y_cb / sub_h);
        for (c_idx, plane, comp) in [
            (1usize, Plane::Cb, TfComponent::Cb),
            (2, Plane::Cr, TfComponent::Cr),
        ] {
            let qp_c = chroma_qp(params, qp_y, comp) as i32;
            crate::hevc::engine::palette::reconstruct_palette_component(
                pal,
                c_idx,
                sub_w,
                sub_h,
                qp_c,
                u32::from(params.bit_depth_chroma),
                transquant_bypass,
                |x, y, v| pic.set_sample(plane, cx + x, cy + y, v),
            );
        }
    }
}

fn write_pcm_cu(
    pic: &mut Picture,
    chroma_array_type: u8,
    x_cb: usize,
    y_cb: usize,
    n_cb: usize,
    pcm: &crate::hevc::engine::slice_data::PcmSamples,
) {
    for j in 0..n_cb {
        for i in 0..n_cb {
            pic.set_sample(
                Plane::Luma,
                x_cb + i,
                y_cb + j,
                i32::from(pcm.luma[n_cb * j + i]),
            );
        }
    }
    if chroma_array_type != 0 {
        let (sub_w, sub_h) = sub_wh_c(chroma_array_type);
        let (cw, ch) = (n_cb / sub_w, n_cb / sub_h);
        let (cx, cy) = (x_cb / sub_w, y_cb / sub_h);
        for (plane, samples) in [(Plane::Cb, &pcm.cb), (Plane::Cr, &pcm.cr)] {
            for j in 0..ch {
                for i in 0..cw {
                    pic.set_sample(plane, cx + i, cy + j, i32::from(samples[cw * j + i]));
                }
            }
        }
    }
}

fn reconstruct_cu(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &mut ReconCtx,
    cu: &CodingUnit,
) -> Result<(), ReconError> {
    if matches!(cu.cu_pred_mode, CuPredMode::Inter | CuPredMode::Skip) {
        // Record the inter CU so a later intra block's §8.4.2 neighbour
        // derivation maps it to INTRA_DC, then signal the inter path is
        // unhandled by this driver.
        ctx.field.record_non_intra_cu(
            cu.x0 as usize,
            cu.y0 as usize,
            1usize << cu.log2_cb_size,
            cu.cu_pred_mode,
        );
        return Err(ReconError::InterNotSupported);
    }

    let n_cb = 1usize << cu.log2_cb_size;
    let x_cb = cu.x0 as usize;
    let y_cb = cu.y0 as usize;

    if cu.pcm_flag {
        // §8.4.1 — PCM reconstruction: the parsed (already
        // bit-depth-scaled, equation 8-12) samples ARE the
        // reconstructed picture; stamp the mode field so neighbours
        // see a written block (→ DC).
        ctx.field.record_intra_pb(x_cb, y_cb, n_cb, INTRA_DC, true);
        if let Some(pcm) = cu.pcm.as_ref() {
            write_pcm_cu(pic, params.chroma_array_type, x_cb, y_cb, n_cb, pcm);
        }
        return Ok(());
    }

    if let Some(pal) = cu.palette.as_deref() {
        // §8.4.4.2.7 — palette-mode reconstruction. A palette CU has
        // no IntraPredModeY; neighbours derive INTRA_DC (the PCM
        // convention).
        ctx.field.record_intra_pb(x_cb, y_cb, n_cb, INTRA_DC, true);
        // §8.6.1 QP derivation feeds the escape dequantization; the
        // delta_qp( ) arrives inside palette_coding( ).
        let cu_delta = pal.cu_qp_delta.as_ref().map(|d| d.value);
        let qp_y = ctx.derive_cu_qp(params, x_cb, y_cb, cu.log2_cb_size, cu_delta);
        write_palette_cu(
            pic,
            params,
            pal,
            x_cb,
            y_cb,
            qp_y,
            cu.cu_transquant_bypass_flag,
        );
        return Ok(());
    }

    // §7.4.9.5: PART_NxN intra splits the CU into four nCbS/2 luma PBs in
    // raster order; PART_2Nx2N is one PB covering the CU.
    let is_nxn = cu.part_mode == PartMode::PartNxN;
    let n_pb = if is_nxn { n_cb / 2 } else { n_cb };
    let pb_origins: &[(usize, usize)] = if is_nxn {
        &[(0, 0), (1, 0), (0, 1), (1, 1)]
    } else {
        &[(0, 0)]
    };

    // §8.4.2 — derive (and record) each luma PB's IntraPredModeY. The
    // §8.4.3 chroma mode uses the CU-corner PB (blkIdx 0) for the
    // ChromaArrayType != 3 single-chroma-block case.
    let mut pb_modes = [INTRA_DC; 4];
    for (i, &(qx, qy)) in pb_origins.iter().enumerate() {
        let x_pb = x_cb + qx * n_pb;
        let y_pb = y_cb + qy * n_pb;
        let luma_mode = cu.intra_luma.get(i).copied().unwrap_or(
            crate::hevc::engine::slice_data::IntraLumaMode {
                prev_intra_luma_pred_flag: true,
                mpm_idx: Some(0),
                rem_intra_luma_pred_mode: None,
            },
        );
        pb_modes[i] = derive_and_record_luma_mode(ctx, x_pb, y_pb, n_pb, &luma_mode, false);
    }

    // §8.4.3 — IntraPredModeC. For ChromaArrayType != 3 the CU has one
    // chroma PB whose mode derives from the corner (blkIdx 0) luma PB;
    // for ChromaArrayType == 3 with PART_NxN, §7.3.8.5 signals four
    // intra_chroma_pred_mode elements and each chroma PB derives from
    // its OWN co-located luma PB's IntraPredModeY.
    let per_pb_chroma =
        params.chroma_array_type == 3 && is_nxn && cu.intra_chroma_pred_mode.len() == 4;
    let mut modes_c = [INTRA_DC; 4];
    // ChromaArrayType == 0 (monochrome): §7.3.8.5 signals no
    // intra_chroma_pred_mode and no chroma PB exists — keep the DC
    // placeholders (nothing downstream consumes them without a
    // chroma plane).
    if params.chroma_array_type != 0 {
        for (i, mode_c) in modes_c.iter_mut().enumerate() {
            let (raw, luma_for_c) = if per_pb_chroma {
                (cu.intra_chroma_pred_mode[i], pb_modes[i])
            } else {
                (cu.intra_chroma_pred_mode[0], pb_modes[0])
            };
            *mode_c = derive_intra_pred_mode_c(raw, luma_for_c, params.chroma_array_type == 2);
        }
    }
    let intra_pred_mode_c = modes_c[0];

    // §8.6.1 — derive the CU's QpY (threading qPY_PREV across
    // quantization groups when the picture-level QP state is
    // initialized; single-QG fallback otherwise).
    let cu_delta = cu.transform_tree.as_ref().and_then(first_tree_cu_qp_delta);
    let qp_y = ctx.derive_cu_qp(params, x_cb, y_cb, cu.log2_cb_size, cu_delta);

    // Walk the transform tree, reconstructing each leaf transform block.
    // For PART_NxN the top-level tree is a Split whose four children map to
    // the four luma PBs (each carrying that PB's luma mode); chroma uses
    // the CU-level IntraPredModeC throughout.
    if let Some(tree) = &cu.transform_tree {
        if is_nxn {
            if let TransformTree::Split { children, .. } = tree {
                let half = n_cb / 2;
                // §7.3.8.10: when the four children are 4×4 luma leaves
                // (and ChromaArrayType != 3), the CU's chroma transform
                // blocks are decoded once, deferred to blkIdx == 3 — the
                // children carry no chroma of their own.
                let child_log2 = cu.log2_cb_size - 1;
                let defer_chroma = child_log2 == 2
                    && params.chroma_array_type != 0
                    && params.chroma_array_type != 3;
                let offsets = [(0, 0), (half, 0), (0, half), (half, half)];
                for (i, (child, (dx, dy))) in children.iter().zip(offsets).enumerate() {
                    reconstruct_transform_tree(
                        pic,
                        params,
                        ctx,
                        child,
                        x_cb + dx,
                        y_cb + dy,
                        child_log2,
                        pb_modes[i],
                        modes_c[i],
                        qp_y,
                        cu.cu_transquant_bypass_flag,
                        defer_chroma,
                    )?;
                }
                if defer_chroma {
                    reconstruct_deferred_chroma(
                        pic,
                        params,
                        ctx,
                        children,
                        x_cb,
                        y_cb,
                        cu.log2_cb_size,
                        intra_pred_mode_c,
                        qp_y,
                        cu.cu_transquant_bypass_flag,
                    )?;
                }
                return Ok(());
            }
        }
        reconstruct_transform_tree(
            pic,
            params,
            ctx,
            tree,
            x_cb,
            y_cb,
            cu.log2_cb_size,
            pb_modes[0],
            intra_pred_mode_c,
            qp_y,
            cu.cu_transquant_bypass_flag,
            false,
        )?;
    }
    Ok(())
}

/// The §7.4.9.14 `cu_qp_delta` decoded inside a CU's transform tree
/// (the first leaf carrying one, in z-scan order), if any.
pub(crate) fn first_tree_cu_qp_delta(tree: &TransformTree) -> Option<i32> {
    match tree {
        TransformTree::Leaf { unit, .. } => unit.cu_qp_delta.as_ref().map(|d| d.value),
        TransformTree::Split { children, .. } => children.iter().find_map(first_tree_cu_qp_delta),
    }
}

/// §7.3.8.10 deferred-chroma reconstruction: the chroma transform blocks
/// of an 8×8 luma node whose children are 4×4 leaves are carried by the
/// `blkIdx == 3` child's transform unit and cover the whole node.
#[allow(clippy::too_many_arguments)]
fn reconstruct_deferred_chroma(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &ReconCtx,
    children: &[TransformTree; 4],
    x0: usize,
    y0: usize,
    log2_trafo_size: u32,
    intra_pred_mode_c: u8,
    qp_y: i32,
    transquant_bypass: bool,
) -> Result<(), ReconError> {
    let TransformTree::Leaf { unit, .. } = &children[3] else {
        // A malformed tree (blkIdx 3 split below 4×4) cannot occur; the
        // §7.3.8.8 recursion bottoms out at MinTbLog2SizeY >= 2.
        return Ok(());
    };
    let (sw, sh) = sub_wh_c(params.chroma_array_type);
    let xc = x0 / sw;
    let yc = y0 / sh;
    // Chroma TB side: half the 8×8 node for 4:2:0 / 4:2:2 → 4.
    let n_tbs_c = (1usize << log2_trafo_size) / 2;
    reconstruct_chroma_blocks(
        pic,
        params,
        ctx,
        Plane::Cb,
        TfComponent::Cb,
        &unit.residual_cb,
        unit.cbf_cb_halves,
        xc,
        yc,
        n_tbs_c,
        intra_pred_mode_c,
        qp_y,
        transquant_bypass,
        None,
    )?;
    reconstruct_chroma_blocks(
        pic,
        params,
        ctx,
        Plane::Cr,
        TfComponent::Cr,
        &unit.residual_cr,
        unit.cbf_cr_halves,
        xc,
        yc,
        n_tbs_c,
        intra_pred_mode_c,
        qp_y,
        transquant_bypass,
        None,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_transform_tree(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &ReconCtx,
    tree: &TransformTree,
    x0: usize,
    y0: usize,
    log2_trafo_size: u32,
    intra_pred_mode_y: u8,
    intra_pred_mode_c: u8,
    qp_y: i32,
    transquant_bypass: bool,
    skip_chroma: bool,
) -> Result<(), ReconError> {
    match tree {
        TransformTree::Split { children, .. } => {
            let child_log2 = log2_trafo_size - 1;
            // §7.3.8.10: 4×4 luma children (ChromaArrayType != 3) defer
            // their chroma to blkIdx == 3, covering this whole node.
            let defer_chroma = !skip_chroma
                && child_log2 == 2
                && params.chroma_array_type != 0
                && params.chroma_array_type != 3;
            let half = 1usize << child_log2;
            // raster order [tl, tr, bl, br].
            let offsets = [(0, 0), (half, 0), (0, half), (half, half)];
            for (child, (dx, dy)) in children.iter().zip(offsets) {
                reconstruct_transform_tree(
                    pic,
                    params,
                    ctx,
                    child,
                    x0 + dx,
                    y0 + dy,
                    child_log2,
                    intra_pred_mode_y,
                    intra_pred_mode_c,
                    qp_y,
                    transquant_bypass,
                    skip_chroma || defer_chroma,
                )?;
            }
            if defer_chroma {
                reconstruct_deferred_chroma(
                    pic,
                    params,
                    ctx,
                    children,
                    x0,
                    y0,
                    log2_trafo_size,
                    intra_pred_mode_c,
                    qp_y,
                    transquant_bypass,
                )?;
            }
            Ok(())
        }
        TransformTree::Leaf { unit, .. } => reconstruct_transform_unit(
            pic,
            params,
            ctx,
            unit,
            x0,
            y0,
            log2_trafo_size,
            intra_pred_mode_y,
            intra_pred_mode_c,
            qp_y,
            transquant_bypass,
            skip_chroma,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_transform_unit(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &ReconCtx,
    unit: &TransformUnit,
    x0: usize,
    y0: usize,
    log2_trafo_size: u32,
    intra_pred_mode_y: u8,
    intra_pred_mode_c: u8,
    qp_y: i32,
    transquant_bypass: bool,
    skip_chroma: bool,
) -> Result<(), ReconError> {
    let n_tbs = 1usize << log2_trafo_size;
    let _ = unit.cu_qp_delta.as_ref();
    // §7.4.9.10 — a parsed chroma_qp_offset( ) element updates the
    // slice-wide CuQpOffsetCb / CuQpOffsetCr state in decode order.
    if let Some(off) = &unit.cu_chroma_qp_offset {
        apply_cu_chroma_qp_offset(params, off);
    }

    // §8.6.8 adaptive colour transform (4:4:4 only): the three
    // co-located residual arrays of this unit are jointly modified
    // before the per-component prediction + add.
    if unit.tu_residual_act_flag == 1 && params.chroma_array_type == 3 && !skip_chroma {
        return reconstruct_act_transform_unit(
            pic,
            params,
            ctx,
            unit,
            x0,
            y0,
            n_tbs,
            intra_pred_mode_y,
            intra_pred_mode_c,
            qp_y,
            transquant_bypass,
        );
    }

    // Luma block: `qp_y` is the CU's §8.6.1-derived QpY.
    let luma_qp = luma_qp(params, qp_y);
    let luma_levels = unit.residual_luma.as_ref().map(|rb| rb.levels.as_slice());
    let luma_ts = unit
        .residual_luma
        .as_ref()
        .is_some_and(|rb| rb.transform_skip);
    // (§7.3.8.11: explicit_rdpcm_flag is only signalled for MODE_INTER
    // blocks — an intra unit's residual never carries it.)
    let luma_residual = reconstruct_intra_block(
        pic,
        params,
        ctx,
        Plane::Luma,
        TfComponent::Luma,
        IpComponent::Luma,
        x0,
        y0,
        n_tbs,
        intra_pred_mode_y,
        luma_levels,
        luma_qp,
        transquant_bypass,
        luma_ts,
        None,
    )?;

    // Chroma blocks. For 4:2:0 / 4:2:2 the chroma transform block sits at
    // (x0 >> SubWidthC, y0 >> SubHeightC) and is half the luma side for
    // 4:2:0; the §7.3.8.10 driver collects the chroma residual at the
    // parent node, so a chroma block is reconstructed once per luma node
    // that carries chroma residual (the residual_cb / residual_cr lists).
    if params.chroma_array_type != 0 && !skip_chroma {
        let (sw, sh) = sub_wh_c(params.chroma_array_type);
        let xc = x0 / sw;
        let yc = y0 / sh;
        // §7.3.8.10 log2TrafoSizeC = Max( 2, log2TrafoSize −
        // ( ChromaArrayType == 3 ? 0 : 1 ) ); the log2TrafoSize == 2
        // deferred case is handled by the parent (skip_chroma).
        let n_tbs_c = if params.chroma_array_type == 3 {
            1usize << log2_trafo_size
        } else {
            (1usize << log2_trafo_size) / 2
        };
        // §8.4.4.1 step 8 (4:4:4 only): the chroma residuals of this
        // transform unit are cross-component-predicted from its luma
        // residual, scaled by the §7.4.9.12 ResScaleVal per component.
        let luma_res = luma_residual.as_deref();
        let ccp_cb = CcpInput::resolve(
            params.chroma_array_type,
            unit.cross_comp_pred_cb.as_ref(),
            luma_res,
        );
        let ccp_cr = CcpInput::resolve(
            params.chroma_array_type,
            unit.cross_comp_pred_cr.as_ref(),
            luma_res,
        );
        reconstruct_chroma_blocks(
            pic,
            params,
            ctx,
            Plane::Cb,
            TfComponent::Cb,
            &unit.residual_cb,
            unit.cbf_cb_halves,
            xc,
            yc,
            n_tbs_c,
            intra_pred_mode_c,
            qp_y,
            transquant_bypass,
            ccp_cb,
        )?;
        reconstruct_chroma_blocks(
            pic,
            params,
            ctx,
            Plane::Cr,
            TfComponent::Cr,
            &unit.residual_cr,
            unit.cbf_cr_halves,
            xc,
            yc,
            n_tbs_c,
            intra_pred_mode_c,
            qp_y,
            transquant_bypass,
            ccp_cr,
        )?;
    }
    Ok(())
}

/// §8.4.4.1 with `residual_adaptive_colour_transform_enabled_flag` and
/// `tu_residual_act_flag == 1` (4:4:4): the transform unit's three
/// co-located residual arrays are derived (ACT-adjusted qP per
/// eq. 8-291 / 8-287 / 8-288), cross-component prediction applies
/// first (§8.4.4.1 step 8), then the §8.6.8.2 inverse colour
/// transform, then each component is predicted and stored.
#[allow(clippy::too_many_arguments)]
fn reconstruct_act_transform_unit(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &ReconCtx,
    unit: &TransformUnit,
    x0: usize,
    y0: usize,
    n_tbs: usize,
    intra_pred_mode_y: u8,
    intra_pred_mode_c: u8,
    qp_y: i32,
    transquant_bypass: bool,
) -> Result<(), ReconError> {
    let qp_l = luma_qp_act(params, qp_y);
    let qp_cb = chroma_qp_act(params, qp_y, TfComponent::Cb, true);
    let qp_cr = chroma_qp_act(params, qp_y, TfComponent::Cr, true);

    // The §8.6.8.2 transform mixes the components, so a cbf-clear
    // component still needs a materialized (zero) residual array.
    let mut r_y = match &unit.residual_luma {
        Some(rb) => intra_residual_array(
            params,
            TfComponent::Luma,
            n_tbs,
            intra_pred_mode_y,
            &rb.levels,
            qp_l,
            transquant_bypass,
            rb.transform_skip,
        )?,
        None => vec![0i32; n_tbs * n_tbs],
    };
    let chroma_res = |blocks: &[crate::hevc::engine::residual::ResidualBlock],
                      coded: bool,
                      cidx: TfComponent,
                      qp: u32|
     -> Result<Vec<i32>, ReconError> {
        match blocks.first() {
            Some(rb) if coded => intra_residual_array(
                params,
                cidx,
                n_tbs,
                intra_pred_mode_c,
                &rb.levels,
                qp,
                transquant_bypass,
                rb.transform_skip,
            ),
            _ => Ok(vec![0i32; n_tbs * n_tbs]),
        }
    };
    let mut r_cb = chroma_res(
        &unit.residual_cb,
        unit.cbf_cb_halves[0],
        TfComponent::Cb,
        qp_cb,
    )?;
    let mut r_cr = chroma_res(
        &unit.residual_cr,
        unit.cbf_cr_halves[0],
        TfComponent::Cr,
        qp_cr,
    )?;

    // §8.4.4.1 step 8 — cross-component prediction precedes the
    // colour transform (the §8.4.1 step-2 ordering).
    if let Some(c) = CcpInput::resolve(
        params.chroma_array_type,
        unit.cross_comp_pred_cb.as_ref(),
        Some(&r_y),
    ) {
        apply_cross_comp_pred(
            &mut r_cb,
            c.luma_residual,
            c.res_scale_val,
            params.bit_depth_luma,
            params.bit_depth_chroma,
        );
    }
    if let Some(c) = CcpInput::resolve(
        params.chroma_array_type,
        unit.cross_comp_pred_cr.as_ref(),
        Some(&r_y),
    ) {
        apply_cross_comp_pred(
            &mut r_cr,
            c.luma_residual,
            c.res_scale_val,
            params.bit_depth_luma,
            params.bit_depth_chroma,
        );
    }

    // §8.6.8.2 — the inverse adaptive colour transformation.
    crate::hevc::engine::transform::act_inverse(
        &mut r_y,
        &mut r_cb,
        &mut r_cr,
        params.bit_depth_luma,
        params.bit_depth_chroma,
        params.extended_precision,
        transquant_bypass,
    );

    // Per-component §8.4.4.2.1 prediction + §8.6.7 construction
    // (4:4:4 — the chroma blocks are co-located and equal-sized).
    predict_add_store(
        pic,
        params,
        ctx,
        Plane::Luma,
        IpComponent::Luma,
        x0,
        y0,
        n_tbs,
        intra_pred_mode_y,
        transquant_bypass,
        Some(&r_y),
    )?;
    predict_add_store(
        pic,
        params,
        ctx,
        Plane::Cb,
        ip_component_of(TfComponent::Cb),
        x0,
        y0,
        n_tbs,
        intra_pred_mode_c,
        transquant_bypass,
        Some(&r_cb),
    )?;
    predict_add_store(
        pic,
        params,
        ctx,
        Plane::Cr,
        ip_component_of(TfComponent::Cr),
        x0,
        y0,
        n_tbs,
        intra_pred_mode_c,
        transquant_bypass,
        Some(&r_cr),
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_chroma_blocks(
    pic: &mut Picture,
    params: &ReconParams,
    ctx: &ReconCtx,
    plane: Plane,
    cidx: TfComponent,
    residual_blocks: &[crate::hevc::engine::residual::ResidualBlock],
    coded_halves: [bool; 2],
    xc: usize,
    yc: usize,
    n_tbs_c: usize,
    intra_pred_mode_c: u8,
    qp_y: i32,
    transquant_bypass: bool,
    ccp: Option<CcpInput<'_>>,
) -> Result<(), ReconError> {
    let qp = chroma_qp(params, qp_y, cidx);
    // `ChromaArrayType == 2` stacks two square chroma TBs vertically per
    // luma node (upper then lower); every other type has one. A block
    // with no coded residual (cbf clear) is still intra-PREDICTED — only
    // the residual add is skipped.
    let vertical_blocks = if params.chroma_array_type == 2 { 2 } else { 1 };
    let mut next_block = 0usize;
    for (v, &half_coded) in coded_halves.iter().enumerate().take(vertical_blocks) {
        // §7.3.8.10 codes the halves upper-then-lower, each gated on
        // its own cbf — pair the residual list entries to the coded
        // halves.
        let block = if half_coded {
            let b = residual_blocks.get(next_block);
            next_block += 1;
            b
        } else {
            None
        };
        let levels = block.map(|rb| rb.levels.as_slice());
        let ts = block.is_some_and(|rb| rb.transform_skip);
        reconstruct_intra_block(
            pic,
            params,
            ctx,
            plane,
            cidx,
            ip_component_of(cidx),
            xc,
            yc + v * n_tbs_c,
            n_tbs_c,
            intra_pred_mode_c,
            levels,
            qp,
            transquant_bypass,
            ts,
            ccp,
        )?;
    }
    Ok(())
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::hevc::engine::availability::TilingParams;
    use crate::hevc::engine::binarization::{CuPredMode, PartMode};
    use crate::hevc::engine::residual::ResidualBlock;
    use crate::hevc::engine::slice_data::{
        CodingQuadtree, CodingTreeUnit, CodingUnit, IntraLumaMode,
    };
    use crate::hevc::engine::transform_tree::TransformTree;
    use crate::hevc::engine::transform_unit::TransformUnit;

    /// §8 reconstruction params for a Main-profile 4:2:0 8-bit slice at
    /// SliceQpY = 25 (the tiny-i fixture geometry).
    fn tiny_params() -> ReconParams {
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

    /// A single-DC residual block of side `1 << log2`.
    fn dc_block(log2: u32, dc: i32) -> ResidualBlock {
        let size = 1usize << log2;
        let mut levels = vec![0i32; size * size];
        levels[0] = dc;
        ResidualBlock {
            log2_trafo_size: log2,
            last_sig_coeff_x: 0,
            last_sig_coeff_y: 0,
            levels,
            transform_skip: false,
            explicit_rdpcm_flag: false,
            explicit_rdpcm_dir_flag: false,
        }
    }

    /// Build a flat 16x16 intra CU (PART_2Nx2N, PLANAR luma via mpm_idx 0)
    /// carrying the given luma / Cb / Cr DC residuals.
    fn flat_intra_ctu(luma_dc: i32, cb_dc: Option<i32>, cr_dc: Option<i32>) -> CodingTreeUnit {
        let mut unit = TransformUnit {
            residual_luma: Some(dc_block(4, luma_dc)),
            ..Default::default()
        };
        if let Some(d) = cb_dc {
            unit.residual_cb = vec![dc_block(3, d)];
            unit.cbf_cb_halves = [true, false];
        }
        if let Some(d) = cr_dc {
            unit.residual_cr = vec![dc_block(3, d)];
            unit.cbf_cr_halves = [true, false];
        }
        let tree = TransformTree::Leaf {
            cbf_luma: true,
            unit,
        };
        let cu = CodingUnit {
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
            // prev_intra_luma_pred_flag + mpm_idx 0 ⇒ candModeList[0] =
            // PLANAR for the all-DC neighbour fallback.
            intra_luma: vec![IntraLumaMode {
                prev_intra_luma_pred_flag: true,
                mpm_idx: Some(0),
                rem_intra_luma_pred_mode: None,
            }],
            // intra_chroma_pred_mode 4 ⇒ derived (= luma mode).
            intra_chroma_pred_mode: vec![4],
            rqt_root_cbf: true,
            transform_tree: Some(tree),
        };
        CodingTreeUnit {
            sao: None,
            quadtree: CodingQuadtree::Leaf(Box::new(cu)),
        }
    }

    /// Build a 16x16 transquant-bypass intra CTU whose single luma TB
    /// carries `levels` (row-major raw bypass residual) and whose luma
    /// mode comes from `luma_mode` (an MPM/rem selector against the
    /// no-neighbour candModeList {PLANAR, DC, 26}).
    fn bypass_intra_ctu(levels: Vec<i32>, luma_mode: IntraLumaMode) -> CodingTreeUnit {
        let unit = TransformUnit {
            residual_luma: Some(ResidualBlock {
                log2_trafo_size: 4,
                last_sig_coeff_x: 15,
                last_sig_coeff_y: 15,
                levels,
                transform_skip: false,
                explicit_rdpcm_flag: false,
                explicit_rdpcm_dir_flag: false,
            }),
            ..Default::default()
        };
        let cu = CodingUnit {
            x0: 0,
            y0: 0,
            log2_cb_size: 4,
            cu_pred_mode: CuPredMode::Intra,
            cu_transquant_bypass_flag: true,
            part_mode: PartMode::Part2Nx2N,
            pcm_flag: false,
            pcm: None,
            palette: None,
            prediction_units: vec![],
            intra_luma: vec![luma_mode],
            intra_chroma_pred_mode: vec![4],
            rqt_root_cbf: true,
            transform_tree: Some(TransformTree::Leaf {
                cbf_luma: true,
                unit,
            }),
        };
        CodingTreeUnit {
            sao: None,
            quadtree: CodingQuadtree::Leaf(Box::new(cu)),
        }
    }

    /// §8.4.4.1 implicit RDPCM, vertical: a transquant-bypass TB in
    /// mode 26 (candModeList[2] with no neighbours) whose residual has
    /// row 0 = 5 accumulates down every column (eq. 8-323), so the
    /// whole block reconstructs to 128 + 5.
    #[test]
    fn implicit_rdpcm_vertical_accumulates_bypass_residual() {
        let mut params = tiny_params();
        params.implicit_rdpcm_enabled = true;
        let mut levels = vec![0i32; 256];
        levels[..16].fill(5); // row 0 only
        let ctu = bypass_intra_ctu(
            levels,
            IntraLumaMode {
                prev_intra_luma_pred_flag: true,
                mpm_idx: Some(2), // candModeList[2] = 26 (vertical)
                rem_intra_luma_pred_mode: None,
            },
        );
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Luma, x, y), 133, "luma at ({x},{y})");
            }
        }
    }

    /// §8.4.4.1 implicit RDPCM, horizontal: mode 10
    /// (rem_intra_luma_pred_mode 8 against candModeList {0, 1, 26})
    /// with column 0 = 5 accumulates across every row (eq. 8-322).
    #[test]
    fn implicit_rdpcm_horizontal_accumulates_bypass_residual() {
        let mut params = tiny_params();
        params.implicit_rdpcm_enabled = true;
        let mut levels = vec![0i32; 256];
        for y in 0..16 {
            levels[y * 16] = 5; // column 0 only
        }
        let ctu = bypass_intra_ctu(
            levels,
            IntraLumaMode {
                prev_intra_luma_pred_flag: false,
                mpm_idx: None,
                // 8 → +1 for cand 0, +1 for cand 1 ⇒ IntraPredModeY 10.
                rem_intra_luma_pred_mode: Some(8),
            },
        );
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Luma, x, y), 133, "luma at ({x},{y})");
            }
        }
    }

    /// The implicit-RDPCM condition is exact: a bypass block whose mode
    /// is NOT 10 / 26 (here DC via mpm_idx 1) keeps its raw residual —
    /// row 0 is 128 + 5, every other row stays at the DC prediction.
    #[test]
    fn implicit_rdpcm_skips_non_directional_modes() {
        let mut params = tiny_params();
        params.implicit_rdpcm_enabled = true;
        let mut levels = vec![0i32; 256];
        levels[..16].fill(5);
        let ctu = bypass_intra_ctu(
            levels,
            IntraLumaMode {
                prev_intra_luma_pred_flag: true,
                mpm_idx: Some(1), // candModeList[1] = DC
                rem_intra_luma_pred_mode: None,
            },
        );
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        for x in 0..16 {
            assert_eq!(pic.sample(Plane::Luma, x, 0), 133, "row 0 at x={x}");
            assert_eq!(pic.sample(Plane::Luma, x, 5), 128, "row 5 at x={x}");
        }
    }

    /// §8.5.4.2 step 3: an inter residual block with
    /// `explicit_rdpcm_flag == 1` runs the §8.6.5 modification with
    /// mDir = `explicit_rdpcm_dir_flag` before placement.
    #[test]
    fn explicit_rdpcm_inter_residual_accumulates() {
        let params = tiny_params();
        let mut levels = vec![0i32; 16];
        levels[..4].fill(3); // row 0
        let rb = ResidualBlock {
            log2_trafo_size: 2,
            last_sig_coeff_x: 3,
            last_sig_coeff_y: 3,
            levels,
            transform_skip: false,
            explicit_rdpcm_flag: true,
            explicit_rdpcm_dir_flag: true, // vertical
        };
        let r = inter_residual_block(&params, &rb, TfComponent::Luma, 25, true).unwrap();
        assert!(r.iter().all(|&v| v == 3), "vertical accumulation: {r:?}");

        let mut levels_h = vec![0i32; 16];
        for y in 0..4 {
            levels_h[y * 4] = 2; // column 0
        }
        let rb_h = ResidualBlock {
            explicit_rdpcm_dir_flag: false, // horizontal
            levels: levels_h,
            ..rb
        };
        let r = inter_residual_block(&params, &rb_h, TfComponent::Luma, 25, true).unwrap();
        assert!(r.iter().all(|&v| v == 2), "horizontal accumulation: {r:?}");
    }

    #[test]
    fn flat_intra_luma_reconstructs_to_constant_field() {
        // pred = midlevel 128 (no neighbours); luma DC −67 dequant+IDCT
        // gives a uniform −47 residual, so recSamples = 128 − 47 = 81.
        let params = tiny_params();
        let ctu = flat_intra_ctu(-67, None, None);
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Luma, x, y), 81, "luma at ({x},{y})");
            }
        }
    }

    #[test]
    fn flat_intra_chroma_cb_reconstructs_exactly() {
        // Cb DC −27 at QpC(25)=25 gives a uniform −38 residual ⇒
        // 128 − 38 = 90 (the fixture's expected.yuv Cb plane value).
        let params = tiny_params();
        let ctu = flat_intra_ctu(-67, Some(-27), None);
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(pic.sample(Plane::Cb, x, y), 90, "cb at ({x},{y})");
            }
        }
    }

    #[test]
    fn chroma_residual_produces_uniform_field() {
        // A pure-DC chroma residual reconstructs to a constant plane (the
        // inverse-transform DC basis is flat); validates the chroma
        // predict + dequant + add + clip pipeline end to end.
        let params = tiny_params();
        let ctu = flat_intra_ctu(-67, Some(-27), Some(64));
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        let cr0 = pic.sample(Plane::Cr, 0, 0);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(pic.sample(Plane::Cr, x, y), cr0, "cr uniform at ({x},{y})");
            }
        }
        assert!(cr0 > 128, "positive DC raises the Cr plane");
    }

    #[test]
    fn clip_saturates_out_of_range_reconstruction() {
        // A large negative luma DC drives pred+res below 0; the §8.4.4.1
        // Clip1Y must clamp to 0 (not wrap).
        let params = tiny_params();
        let ctu = flat_intra_ctu(-400, None, None);
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        assert_eq!(pic.sample(Plane::Luma, 0, 0), 0);
    }

    /// Build an 8×8 intra CU at `(x0, y0)` carrying the given luma-mode
    /// signalling and a uniform DC luma residual (no chroma).
    fn intra_cu_8x8(
        x0: u32,
        y0: u32,
        luma: IntraLumaMode,
        luma_dc: i32,
        chroma_pred_mode: u8,
    ) -> CodingUnit {
        let unit = TransformUnit {
            residual_luma: Some(dc_block(3, luma_dc)),
            ..Default::default()
        };
        CodingUnit {
            x0,
            y0,
            log2_cb_size: 3,
            cu_pred_mode: CuPredMode::Intra,
            cu_transquant_bypass_flag: false,
            part_mode: PartMode::Part2Nx2N,
            pcm_flag: false,
            pcm: None,
            palette: None,
            prediction_units: vec![],
            intra_luma: vec![luma],
            intra_chroma_pred_mode: vec![chroma_pred_mode],
            rqt_root_cbf: true,
            transform_tree: Some(TransformTree::Leaf {
                cbf_luma: true,
                unit,
            }),
        }
    }

    /// The §8.4.2 neighbour MPM derivation makes a CU's IntraPredModeY
    /// depend on its already-reconstructed left / above neighbours. A right
    /// CU signalling `mpm_idx == 0` against a left neighbour coded with a
    /// non-DC angular mode picks up `candModeList[0]` = the neighbour's
    /// mode (proved by inspecting the recorded IntraModeField), whereas the
    /// flat-single-CU path would have derived INTRA_DC.
    #[test]
    fn neighbour_mpm_propagates_left_cu_mode_to_right_cu() {
        let params = tiny_params();
        // Left CU (0,0): remaining-mode angular 18. Both neighbours are
        // out-of-picture ⇒ candA == candB == INTRA_DC; the step-4 ordered
        // procedure maps rem_intra_luma_pred_mode 18 (with neither DC nor
        // PLANAR below it) up to IntraPredModeY 20.
        let left = IntraLumaMode {
            prev_intra_luma_pred_flag: false,
            mpm_idx: None,
            rem_intra_luma_pred_mode: Some(18),
        };
        // Right CU (8,0): mpm_idx 0 ⇒ IntraPredModeY = candModeList[0]. Its
        // left neighbour (7,*) is the left CU (mode 20, available in z-scan
        // order); its above neighbour is out-of-picture (DC). candA(20) !=
        // candB(DC) and neither is PLANAR ⇒ candModeList[0] == candA == 20.
        let right = IntraLumaMode {
            prev_intra_luma_pred_flag: true,
            mpm_idx: Some(0),
            rem_intra_luma_pred_mode: None,
        };
        let tl = intra_cu_8x8(0, 0, left, 0, 4);
        let tr = intra_cu_8x8(8, 0, right, 0, 4);
        let bl = intra_cu_8x8(0, 8, left, 0, 4);
        let br = intra_cu_8x8(8, 8, right, 0, 4);
        let ctu = CodingTreeUnit {
            sao: None,
            quadtree: CodingQuadtree::Split(vec![
                CodingQuadtree::Leaf(Box::new(tl)),
                CodingQuadtree::Leaf(Box::new(tr)),
                CodingQuadtree::Leaf(Box::new(bl)),
                CodingQuadtree::Leaf(Box::new(br)),
            ]),
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut ctx = ReconCtx::new(16, 16, 4, 2, &TilingParams::single_tile()).unwrap();
        reconstruct_intra_ctu_ctx(&mut pic, &params, &mut ctx, &ctu).unwrap();

        // The left CU's remaining-mode 18 resolves to IntraPredModeY 20.
        assert_eq!(ctx.recorded_mode(0, 0), Some(20), "left CU mode");
        // The right CU's mpm_idx 0 picked up the left neighbour's mode 20
        // through candModeList[0] — the §8.4.2 propagation under test.
        assert_eq!(
            ctx.recorded_mode(8, 0),
            Some(20),
            "right CU inherited the left neighbour's mode via MPM"
        );
    }

    /// Counter-case: with NO recorded left neighbour (a single isolated
    /// CU), the same `mpm_idx == 0` signalling derives candModeList[0] from
    /// the all-DC fallback ⇒ INTRA_PLANAR (0), not the angular 20 above.
    #[test]
    fn isolated_cu_mpm_zero_is_planar_not_neighbour_mode() {
        let params = tiny_params();
        let right = IntraLumaMode {
            prev_intra_luma_pred_flag: true,
            mpm_idx: Some(0),
            rem_intra_luma_pred_mode: None,
        };
        let cu = intra_cu_8x8(8, 0, right, 0, 4);
        let ctu = CodingTreeUnit {
            sao: None,
            quadtree: CodingQuadtree::Leaf(Box::new(cu)),
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        let mut ctx = ReconCtx::new(16, 16, 4, 2, &TilingParams::single_tile()).unwrap();
        reconstruct_intra_ctu_ctx(&mut pic, &params, &mut ctx, &ctu).unwrap();
        // candA == candB == DC ⇒ candModeList = [PLANAR, DC, 26]; index 0
        // is INTRA_PLANAR.
        assert_eq!(ctx.recorded_mode(8, 0), Some(0));
    }

    #[test]
    fn inter_cu_is_rejected_by_intra_path() {
        let params = tiny_params();
        let mut ctu = flat_intra_ctu(-67, None, None);
        if let CodingQuadtree::Leaf(cu) = &mut ctu.quadtree {
            cu.cu_pred_mode = CuPredMode::Inter;
        }
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        assert_eq!(
            reconstruct_intra_ctu(&mut pic, &params, &ctu),
            Err(ReconError::InterNotSupported)
        );
    }

    /// End-to-end §8.5 inter reconstruction: a uni-L0 P-block with a
    /// full-pel motion vector copies a shifted reference window, then a
    /// uniform residual is added and clipped.
    #[test]
    fn inter_uni_l0_full_pel_reconstructs() {
        let params = tiny_params();
        // Reference picture: luma ramp sample(x,y) == x (mod 256), flat
        // chroma 128.
        let mut refpic = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                refpic.set_sample(Plane::Luma, x, y, x as i32);
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                refpic.set_sample(Plane::Cb, x, y, 128);
                refpic.set_sample(Plane::Cr, x, y, 128);
            }
        }
        let l0 = ResolvedList {
            pred_flag: true,
            mv_l: [8, 0], // +2 full luma samples right.
            mv_c: [8, 0], // 4:2:0 ⇒ mvC = mvL.
            ref_pic: &refpic,
        };
        // Unused L1 points at the same picture but pred_flag is false.
        let l1 = ResolvedList {
            pred_flag: false,
            mv_l: [0, 0],
            mv_c: [0, 0],
            ref_pic: &refpic,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        // A flat +5 luma residual over the 8x8 PU at (0,0).
        let res = vec![5i32; 8 * 8];
        reconstruct_inter_pu(
            &mut pic,
            &params,
            0,
            0,
            8,
            8,
            l0,
            l1,
            Some(&res),
            None,
            None,
        )
        .unwrap();
        // predSamples[xL] reads ref column xPb + 2 + xL = 2 + xL; + 5 res.
        for yl in 0..8 {
            for xl in 0..8 {
                assert_eq!(
                    pic.sample(Plane::Luma, xl, yl),
                    (2 + xl as i32) + 5,
                    "luma ({xl},{yl})"
                );
            }
        }
        // Chroma: flat 128 prediction, no residual ⇒ 128.
        for yc in 0..4 {
            for xc in 0..4 {
                assert_eq!(pic.sample(Plane::Cb, xc, yc), 128);
                assert_eq!(pic.sample(Plane::Cr, xc, yc), 128);
            }
        }
    }

    /// Bi-prediction averages two reference windows; a clip guards the
    /// out-of-range sum.
    #[test]
    fn inter_bi_averages_and_clips() {
        let params = tiny_params();
        let mut a = Picture::new(16, 16, 1, 8, 8);
        let mut b = Picture::new(16, 16, 1, 8, 8);
        for y in 0..16 {
            for x in 0..16 {
                a.set_sample(Plane::Luma, x, y, 40);
                b.set_sample(Plane::Luma, x, y, 200);
            }
        }
        let l0 = ResolvedList {
            pred_flag: true,
            mv_l: [0, 0],
            mv_c: [0, 0],
            ref_pic: &a,
        };
        let l1 = ResolvedList {
            pred_flag: true,
            mv_l: [0, 0],
            mv_c: [0, 0],
            ref_pic: &b,
        };
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_inter_pu(&mut pic, &params, 0, 0, 8, 8, l0, l1, None, None, None).unwrap();
        // (40 + 200) >> 1 == 120.
        for yl in 0..8 {
            for xl in 0..8 {
                assert_eq!(pic.sample(Plane::Luma, xl, yl), 120);
            }
        }
    }

    #[test]
    fn packed_output_matches_planar_layout() {
        let params = tiny_params();
        let ctu = flat_intra_ctu(-67, Some(-27), None);
        let mut pic = Picture::new(16, 16, 1, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        let packed = pic.to_planar_u8().unwrap();
        // 256 luma + 64 cb + 64 cr.
        assert_eq!(packed.len(), 384);
        assert!(packed[..256].iter().all(|&v| v == 81));
        assert!(packed[256..320].iter().all(|&v| v == 90));
    }

    /// Picture-level driver with SAO off: a single flat CTU reconstructs to
    /// the same constant field as the per-CTU path, and the SAO pass (type
    /// 0 / both flags clear) is a no-op.
    #[test]
    fn picture_driver_single_ctu_matches_per_ctu_no_sao() {
        let params = tiny_params();
        let ctu = flat_intra_ctu(-67, Some(-27), Some(64));
        let pic_params = IntraPictureParams {
            ctb_log2_size_y: 4,
            min_tb_log2_size_y: 2,
            tiles: TilingParams::single_tile(),
            slice_sao_luma_flag: false,
            slice_sao_chroma_flag: false,
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,
        };
        let placed = [PlacedCtu {
            x_ctb: 0,
            y_ctb: 0,
            slice_addr_rs: 0,
            ctu: &ctu,
        }];
        let out = reconstruct_intra_picture(16, 16, &params, &pic_params, &placed).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(out.sample(Plane::Luma, x, y), 81);
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out.sample(Plane::Cb, x, y), 90);
            }
        }
    }

    /// Picture-level driver applies the §8.7.3 SAO band offset: a flat luma
    /// field at 81 with a band-offset CTB shifts every covered sample by the
    /// band's offset.
    #[test]
    fn picture_driver_applies_sao_band_offset() {
        use crate::hevc::engine::slice_data::{SaoComponent, SaoCtbParams};
        let params = tiny_params();
        let ctu = flat_intra_ctu(-67, Some(-27), Some(64));
        // SAO band offset on luma: band_position chosen so the 81-valued
        // luma band gets a +7 offset. band_shift = bitDepth-5 = 3, so
        // sample 81 >> 3 == 10 falls in band 10; bandTable maps four
        // consecutive bands from sao_band_position. Pick band_position 10 so
        // band 10 → bandTable index 1 → offset_val[1].
        let mut luma_sao = SaoComponent {
            sao_type_idx: 1, // band offset
            ..Default::default()
        };
        luma_sao.band_position = 10;
        luma_sao.offset_abs = [7, 0, 0, 0];
        luma_sao.offset_sign = [0, 0, 0, 0];
        let ctu_sao = CodingTreeUnit {
            sao: Some(SaoCtbParams {
                merge_left: false,
                merge_up: false,
                components: [luma_sao, SaoComponent::default(), SaoComponent::default()],
            }),
            quadtree: ctu.quadtree.clone(),
        };
        let pic_params = IntraPictureParams {
            ctb_log2_size_y: 4,
            min_tb_log2_size_y: 2,
            tiles: TilingParams::single_tile(),
            slice_sao_luma_flag: true,
            slice_sao_chroma_flag: false,
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,
        };
        let placed = [PlacedCtu {
            x_ctb: 0,
            y_ctb: 0,
            slice_addr_rs: 0,
            ctu: &ctu_sao,
        }];
        let out = reconstruct_intra_picture(16, 16, &params, &pic_params, &placed).unwrap();
        // Luma 81 (band 10, bandTable idx 1) + offset 7 = 88.
        assert_eq!(out.sample(Plane::Luma, 4, 4), 88, "SAO band offset applied");
    }

    /// Multi-slice neighbour isolation: two 16×16 CTUs side by side in a
    /// 32×16 picture. The right CTU's `mpm_idx == 0` inherits the left
    /// CTU's angular mode 20 through the §8.4.2 MPM when both share a slice
    /// (map `[0, 0]`), but falls back to INTRA_PLANAR when the right CTU is
    /// in a different slice (map `[0, 1]`) — the §6.4.1 z-scan availability
    /// denies a neighbour across the slice boundary.
    #[test]
    fn slice_boundary_blocks_neighbour_mpm() {
        let params = tiny_params();
        let left = IntraLumaMode {
            prev_intra_luma_pred_flag: false,
            mpm_idx: None,
            rem_intra_luma_pred_mode: Some(18),
        };
        let right = IntraLumaMode {
            prev_intra_luma_pred_flag: true,
            mpm_idx: Some(0),
            rem_intra_luma_pred_mode: None,
        };
        let make_ctus = || {
            (
                CodingTreeUnit {
                    sao: None,
                    quadtree: CodingQuadtree::Leaf(Box::new(intra_cu_16x16(0, 0, left))),
                },
                CodingTreeUnit {
                    sao: None,
                    quadtree: CodingQuadtree::Leaf(Box::new(intra_cu_16x16(16, 0, right))),
                },
            )
        };

        // Same-slice baseline (map [0, 0]): the right CTU inherits mode 20.
        let (l0, r0) = make_ctus();
        let mut pic = Picture::new(32, 16, 1, 8, 8);
        let mut ctx = ReconCtx::new(32, 16, 4, 2, &TilingParams::single_tile()).unwrap();
        ctx.set_slice_addr_rs(vec![0, 0]);
        reconstruct_intra_ctu_ctx(&mut pic, &params, &mut ctx, &l0).unwrap();
        reconstruct_intra_ctu_ctx(&mut pic, &params, &mut ctx, &r0).unwrap();
        assert_eq!(ctx.recorded_mode(16, 0), Some(20), "same-slice inherits 20");

        // Cross-slice (map [0, 1]): the right CTU's left neighbour (15,*)
        // is in slice 0 ⇒ denied ⇒ MPM falls back to PLANAR.
        let (l1, r1) = make_ctus();
        let mut pic2 = Picture::new(32, 16, 1, 8, 8);
        let mut ctx2 = ReconCtx::new(32, 16, 4, 2, &TilingParams::single_tile()).unwrap();
        ctx2.set_slice_addr_rs(vec![0, 1]);
        reconstruct_intra_ctu_ctx(&mut pic2, &params, &mut ctx2, &l1).unwrap();
        reconstruct_intra_ctu_ctx(&mut pic2, &params, &mut ctx2, &r1).unwrap();
        assert_eq!(ctx2.recorded_mode(0, 0), Some(20), "left CTU mode 20");
        assert_eq!(
            ctx2.recorded_mode(16, 0),
            Some(0),
            "cross-slice right CTU falls back to PLANAR (no cross-slice MPM)"
        );
    }

    /// §7.4.7.1 SliceAddrRs map for the `multi-slice-per-frame` fixture
    /// geometry: a 64×64 picture, 16×16 CTBs ⇒ a 4×4 CTB raster grid, four
    /// row-wise independent slices at addresses 0, 4, 8, 12 (no tiles).
    /// Each CTB's SliceAddrRs is the address of its slice's first CTB.
    #[test]
    fn slice_addr_map_partitions_four_row_slices() {
        let tiling = PictureTiling::new(4, 4, 64, 64, 4, 2, &TilingParams::single_tile()).unwrap();
        let segs = [
            SliceSegmentBoundary {
                slice_segment_address: 0,
                dependent: false,
            },
            SliceSegmentBoundary {
                slice_segment_address: 4,
                dependent: false,
            },
            SliceSegmentBoundary {
                slice_segment_address: 8,
                dependent: false,
            },
            SliceSegmentBoundary {
                slice_segment_address: 12,
                dependent: false,
            },
        ];
        let map = build_slice_addr_map(&tiling, &segs);
        // Row 0 (CTBs 0..3) ⇒ SliceAddrRs 0; row 1 (4..7) ⇒ 4; etc.
        assert_eq!(
            map,
            vec![0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]
        );
    }

    /// A dependent slice segment inherits the SliceAddrRs of the preceding
    /// independent segment.
    #[test]
    fn slice_addr_map_dependent_inherits() {
        // 4×1 CTB row. Independent slice at 0, dependent continuation at 2.
        let tiling = PictureTiling::new(4, 1, 64, 16, 4, 2, &TilingParams::single_tile()).unwrap();
        let segs = [
            SliceSegmentBoundary {
                slice_segment_address: 0,
                dependent: false,
            },
            SliceSegmentBoundary {
                slice_segment_address: 2,
                dependent: true,
            },
        ];
        let map = build_slice_addr_map(&tiling, &segs);
        // The dependent segment inherits SliceAddrRs 0 ⇒ the whole row is
        // one slice (SliceAddrRs 0).
        assert_eq!(map, vec![0, 0, 0, 0]);
    }

    /// Two independent slices split a single CTB row.
    #[test]
    fn slice_addr_map_two_independent_in_a_row() {
        let tiling = PictureTiling::new(4, 1, 64, 16, 4, 2, &TilingParams::single_tile()).unwrap();
        let segs = [
            SliceSegmentBoundary {
                slice_segment_address: 0,
                dependent: false,
            },
            SliceSegmentBoundary {
                slice_segment_address: 2,
                dependent: false,
            },
        ];
        let map = build_slice_addr_map(&tiling, &segs);
        assert_eq!(map, vec![0, 0, 2, 2]);
    }

    /// A 16×16 PART_2Nx2N intra CU at `(x0, y0)` with a uniform DC luma
    /// residual and the given luma-mode signalling (test helper).
    fn intra_cu_16x16(x0: u32, y0: u32, luma: IntraLumaMode) -> CodingUnit {
        let unit = TransformUnit {
            residual_luma: Some(dc_block(4, 0)),
            ..Default::default()
        };
        CodingUnit {
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
            intra_luma: vec![luma],
            intra_chroma_pred_mode: vec![4],
            rqt_root_cbf: true,
            transform_tree: Some(TransformTree::Leaf {
                cbf_luma: true,
                unit,
            }),
        }
    }

    // -----------------------------------------------------------------
    // §8.6.6 cross-component prediction (4:4:4)
    // -----------------------------------------------------------------

    /// 4:4:4 8-bit params for the transquant-bypass CCP fixtures.
    fn params_444() -> ReconParams {
        ReconParams {
            chroma_array_type: 3,
            ..tiny_params()
        }
    }

    /// A §7.4.9.12 `cross_comp_pred()` result with the given
    /// `ResScaleVal` (a signed power of two in −8..=8).
    fn ccp(res_scale_val: i32) -> crate::hevc::engine::binarization::CrossCompPred {
        let mag = res_scale_val.unsigned_abs();
        assert!(mag.is_power_of_two() && mag <= 8);
        crate::hevc::engine::binarization::CrossCompPred {
            log2_res_scale_abs_plus1: mag.trailing_zeros() + 1,
            res_scale_sign_flag: Some(u8::from(res_scale_val < 0)),
            res_scale_val,
        }
    }

    /// A 16×16 transquant-bypass 4:4:4 intra CTU: PLANAR luma (mpm 0,
    /// no neighbours), derived chroma (mode 4), raw bypass residuals.
    fn ccp_intra_ctu(
        luma_levels: Vec<i32>,
        cb_levels: Option<Vec<i32>>,
        ccp_cb: Option<crate::hevc::engine::binarization::CrossCompPred>,
        ccp_cr: Option<crate::hevc::engine::binarization::CrossCompPred>,
    ) -> CodingTreeUnit {
        let raw = |levels: Vec<i32>| ResidualBlock {
            log2_trafo_size: 4,
            last_sig_coeff_x: 15,
            last_sig_coeff_y: 15,
            levels,
            transform_skip: false,
            explicit_rdpcm_flag: false,
            explicit_rdpcm_dir_flag: false,
        };
        let mut unit = TransformUnit {
            residual_luma: Some(raw(luma_levels)),
            cross_comp_pred_cb: ccp_cb,
            cross_comp_pred_cr: ccp_cr,
            ..Default::default()
        };
        if let Some(l) = cb_levels {
            unit.residual_cb = vec![raw(l)];
            unit.cbf_cb_halves = [true, false];
        }
        let cu = CodingUnit {
            x0: 0,
            y0: 0,
            log2_cb_size: 4,
            cu_pred_mode: CuPredMode::Intra,
            cu_transquant_bypass_flag: true,
            part_mode: PartMode::Part2Nx2N,
            pcm_flag: false,
            pcm: None,
            palette: None,
            prediction_units: vec![],
            intra_luma: vec![IntraLumaMode {
                prev_intra_luma_pred_flag: true,
                mpm_idx: Some(0),
                rem_intra_luma_pred_mode: None,
            }],
            intra_chroma_pred_mode: vec![4],
            rqt_root_cbf: true,
            transform_tree: Some(TransformTree::Leaf {
                cbf_luma: true,
                unit,
            }),
        };
        CodingTreeUnit {
            sao: None,
            quadtree: CodingQuadtree::Leaf(Box::new(cu)),
        }
    }

    /// Eq. 8-324 with a cbf-clear chroma block: the scaled luma
    /// residual alone modifies the chroma reconstruction.
    /// rY = 8, ResScaleVal = +1 ⇒ Cb += (8 >> 3) = 1; ResScaleVal = −1
    /// ⇒ Cr += (−8 >> 3) = −1 (arithmetic shift).
    #[test]
    fn ccp_intra_uncoded_chroma_receives_scaled_luma_residual() {
        let params = params_444();
        let ctu = ccp_intra_ctu(vec![8; 256], None, Some(ccp(1)), Some(ccp(-1)));
        let mut pic = Picture::new(16, 16, 3, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Luma, x, y), 136, "luma at ({x},{y})");
                assert_eq!(pic.sample(Plane::Cb, x, y), 129, "cb at ({x},{y})");
                assert_eq!(pic.sample(Plane::Cr, x, y), 127, "cr at ({x},{y})");
            }
        }
    }

    /// Eq. 8-324 on top of a coded chroma residual: r += scaled rY
    /// AFTER the block's own §8.6.2 output. rY = 8, coded Cb = 4,
    /// ResScaleVal = +2 ⇒ Cb residual 4 + ((2·8) >> 3) = 6.
    #[test]
    fn ccp_intra_coded_chroma_adds_scaled_luma_residual() {
        let params = params_444();
        let ctu = ccp_intra_ctu(vec![8; 256], Some(vec![4; 256]), Some(ccp(2)), None);
        let mut pic = Picture::new(16, 16, 3, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Cb, x, y), 134, "cb at ({x},{y})");
                // Cr carried no cross_comp_pred and no residual.
                assert_eq!(pic.sample(Plane::Cr, x, y), 128, "cr at ({x},{y})");
            }
        }
    }

    /// Eq. 8-324 bit-depth alignment: `( rY << BitDepthC ) >> BitDepthY`
    /// rescales the luma residual into the chroma depth before the
    /// ResScaleVal multiply.
    #[test]
    fn ccp_bit_depth_alignment_rescales_luma_residual() {
        // BitDepthY = 10, BitDepthC = 8: rY = 16 → (16 << 8) >> 10 = 4;
        // scale 4 ⇒ += (4·4) >> 3 = 2.
        let mut r = vec![10i32; 4];
        apply_cross_comp_pred(&mut r, &[16, 16, -16, -16], 4, 10, 8);
        assert_eq!(r, vec![12, 12, 8, 8]);
        // Chroma deeper than luma: rY = 3 → (3 << 12) >> 8 = 48;
        // scale −8 ⇒ += (−384) >> 3 = −48.
        let mut r = vec![0i32; 2];
        apply_cross_comp_pred(&mut r, &[3, -3], -8, 8, 12);
        assert_eq!(r, vec![-48, 48]);
    }

    /// §8.5.4.3 step 5 in the inter residual-extraction path: a
    /// cbf-clear Cb block still receives the scaled luma residual, and
    /// a coded Cr block is modified after its own inverse transform.
    #[test]
    fn ccp_inter_extract_modifies_chroma_planes() {
        let params = params_444();
        let raw = |levels: Vec<i32>| ResidualBlock {
            log2_trafo_size: 3,
            last_sig_coeff_x: 7,
            last_sig_coeff_y: 7,
            levels,
            transform_skip: false,
            explicit_rdpcm_flag: false,
            explicit_rdpcm_dir_flag: false,
        };
        let mut unit = TransformUnit {
            residual_luma: Some(raw(vec![16; 64])),
            cross_comp_pred_cb: Some(ccp(2)),
            cross_comp_pred_cr: Some(ccp(-4)),
            ..Default::default()
        };
        unit.residual_cr = vec![raw(vec![5; 64])];
        unit.cbf_cr_halves = [true, false];
        let tree = TransformTree::Leaf {
            cbf_luma: true,
            unit,
        };
        let res = extract_cu_residual(&params, Some(&tree), 0, 0, 8, 25, true).unwrap();
        assert!(res.luma.samples.iter().all(|&v| v == 16), "luma");
        // Cb (uncoded): (2·16) >> 3 = 4.
        assert!(
            res.cb.as_ref().unwrap().samples.iter().all(|&v| v == 4),
            "cb: {:?}",
            &res.cb.as_ref().unwrap().samples[..8]
        );
        // Cr (coded 5): 5 + ((−4·16) >> 3) = −3.
        assert!(
            res.cr.as_ref().unwrap().samples.iter().all(|&v| v == -3),
            "cr: {:?}",
            &res.cr.as_ref().unwrap().samples[..8]
        );
    }

    /// §8.5.4.3: the stacked upper/lower chroma-block pair exists only
    /// for `ChromaArrayType == 2` (`blkIdx` proceeds over
    /// `0..( ChromaArrayType == 2 ? 1 : 0 )`). A 4:4:4 transform unit
    /// codes ONE chroma block — a cbf-clear Cb with cross-component
    /// prediction must modify exactly its own `nTbS` square and never
    /// synthesize a second block one TU-height below (the phantom
    /// write previously corrupted the sibling leaf's region in every
    /// split transform tree).
    #[test]
    fn ccp_444_uncoded_chroma_writes_single_block_not_stacked_halves() {
        let params = params_444();
        let raw = |levels: Vec<i32>| ResidualBlock {
            log2_trafo_size: 3,
            last_sig_coeff_x: 7,
            last_sig_coeff_y: 7,
            levels,
            transform_skip: false,
            explicit_rdpcm_flag: false,
            explicit_rdpcm_dir_flag: false,
        };
        // Top-left 8×8 leaf: luma residual 16, CCP on a cbf-clear Cb
        // (contribution (2·16) >> 3 = 4). The other three leaves are
        // empty (no luma, no chroma, no CCP).
        let ccp_leaf = TransformTree::Leaf {
            cbf_luma: true,
            unit: TransformUnit {
                residual_luma: Some(raw(vec![16; 64])),
                cross_comp_pred_cb: Some(ccp(2)),
                ..Default::default()
            },
        };
        let empty_leaf = || TransformTree::Leaf {
            cbf_luma: false,
            unit: TransformUnit::default(),
        };
        let tree = TransformTree::Split {
            cbf_cb: true,
            cbf_cb_lower: false,
            cbf_cr: false,
            cbf_cr_lower: false,
            children: Box::new([ccp_leaf, empty_leaf(), empty_leaf(), empty_leaf()]),
        };
        let res = extract_cu_residual(&params, Some(&tree), 0, 0, 16, 25, true).unwrap();
        let cb = res.cb.as_ref().unwrap();
        for y in 0..16 {
            for x in 0..16 {
                let expected = if x < 8 && y < 8 { 4 } else { 0 };
                assert_eq!(cb.samples[y * 16 + x], expected, "cb residual at ({x},{y})");
            }
        }
    }

    /// A zero ResScaleVal (`log2_res_scale_abs_plus1 == 0`) is the
    /// signalled no-op: no chroma modification, cbf-clear blocks stay
    /// zero.
    #[test]
    fn ccp_zero_scale_is_noop() {
        let params = params_444();
        let none = crate::hevc::engine::binarization::CrossCompPred {
            log2_res_scale_abs_plus1: 0,
            res_scale_sign_flag: None,
            res_scale_val: 0,
        };
        let ctu = ccp_intra_ctu(vec![8; 256], None, Some(none), Some(none));
        let mut pic = Picture::new(16, 16, 3, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Cb, x, y), 128, "cb at ({x},{y})");
                assert_eq!(pic.sample(Plane::Cr, x, y), 128, "cr at ({x},{y})");
            }
        }
    }

    // -----------------------------------------------------------------
    // §8.6.8 adaptive colour transform (4:4:4)
    // -----------------------------------------------------------------

    /// A 16×16 transquant-bypass 4:4:4 intra CTU whose single TU has
    /// `tu_residual_act_flag == 1` and carries the given raw bypass
    /// residuals per component (all-cbf-set unless a level vec is
    /// `None`).
    fn act_intra_ctu(
        luma_levels: Option<Vec<i32>>,
        cb_levels: Option<Vec<i32>>,
        cr_levels: Option<Vec<i32>>,
    ) -> CodingTreeUnit {
        let raw = |levels: Vec<i32>| ResidualBlock {
            log2_trafo_size: 4,
            last_sig_coeff_x: 15,
            last_sig_coeff_y: 15,
            levels,
            transform_skip: false,
            explicit_rdpcm_flag: false,
            explicit_rdpcm_dir_flag: false,
        };
        let mut unit = TransformUnit {
            tu_residual_act_flag: 1,
            ..Default::default()
        };
        let cbf_luma = luma_levels.is_some();
        if let Some(l) = luma_levels {
            unit.residual_luma = Some(raw(l));
        }
        if let Some(l) = cb_levels {
            unit.residual_cb = vec![raw(l)];
            unit.cbf_cb_halves = [true, false];
        }
        if let Some(l) = cr_levels {
            unit.residual_cr = vec![raw(l)];
            unit.cbf_cr_halves = [true, false];
        }
        let cu = CodingUnit {
            x0: 0,
            y0: 0,
            log2_cb_size: 4,
            cu_pred_mode: CuPredMode::Intra,
            cu_transquant_bypass_flag: true,
            part_mode: PartMode::Part2Nx2N,
            pcm_flag: false,
            pcm: None,
            palette: None,
            prediction_units: vec![],
            intra_luma: vec![IntraLumaMode {
                prev_intra_luma_pred_flag: true,
                mpm_idx: Some(0),
                rem_intra_luma_pred_mode: None,
            }],
            intra_chroma_pred_mode: vec![4],
            rqt_root_cbf: true,
            transform_tree: Some(TransformTree::Leaf { cbf_luma, unit }),
        };
        CodingTreeUnit {
            sao: None,
            quadtree: CodingQuadtree::Leaf(Box::new(cu)),
        }
    }

    /// §8.6.8.2 lossless inverse in the intra path: coded
    /// (Y, Cg, Co) = (4, 3, 14) lifts back to component residuals
    /// (6, −4, 10) — tmp = 4 − 1 = 3, rY = 6, rCb = 3 − 7 = −4,
    /// rCr = 14 − 4 = 10 — on top of the flat 128 prediction.
    #[test]
    fn act_intra_bypass_lifts_residual_triple() {
        let params = params_444();
        let ctu = act_intra_ctu(Some(vec![4; 256]), Some(vec![3; 256]), Some(vec![14; 256]));
        let mut pic = Picture::new(16, 16, 3, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Luma, x, y), 134, "luma at ({x},{y})");
                assert_eq!(pic.sample(Plane::Cb, x, y), 124, "cb at ({x},{y})");
                assert_eq!(pic.sample(Plane::Cr, x, y), 138, "cr at ({x},{y})");
            }
        }
    }

    /// A cbf-clear component still participates in the §8.6.8.2
    /// mixing: chroma-only (Cg = 2) produces tmp = −1 ⇒
    /// (rY, rCb, rCr) = (1, −1, −1).
    #[test]
    fn act_intra_mixes_into_uncoded_components() {
        let params = params_444();
        let ctu = act_intra_ctu(None, Some(vec![2; 256]), None);
        let mut pic = Picture::new(16, 16, 3, 8, 8);
        reconstruct_intra_ctu(&mut pic, &params, &ctu).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(pic.sample(Plane::Luma, x, y), 129, "luma at ({x},{y})");
                assert_eq!(pic.sample(Plane::Cb, x, y), 127, "cb at ({x},{y})");
                assert_eq!(pic.sample(Plane::Cr, x, y), 127, "cr at ({x},{y})");
            }
        }
    }

    /// §8.5.4.1 step 4 in the inter residual-extraction path: the same
    /// (4, 3, 14) → (6, −4, 10) lifting lands in the CU residual
    /// planes.
    #[test]
    fn act_inter_extract_lifts_residual_triple() {
        let params = params_444();
        let raw = |levels: Vec<i32>| ResidualBlock {
            log2_trafo_size: 3,
            last_sig_coeff_x: 7,
            last_sig_coeff_y: 7,
            levels,
            transform_skip: false,
            explicit_rdpcm_flag: false,
            explicit_rdpcm_dir_flag: false,
        };
        let unit = TransformUnit {
            tu_residual_act_flag: 1,
            residual_luma: Some(raw(vec![4; 64])),
            residual_cb: vec![raw(vec![3; 64])],
            residual_cr: vec![raw(vec![14; 64])],
            cbf_cb_halves: [true, false],
            cbf_cr_halves: [true, false],
            ..Default::default()
        };
        let tree = TransformTree::Leaf {
            cbf_luma: true,
            unit,
        };
        let res = extract_cu_residual(&params, Some(&tree), 0, 0, 8, 25, true).unwrap();
        assert!(res.luma.samples.iter().all(|&v| v == 6), "luma");
        assert!(
            res.cb.as_ref().unwrap().samples.iter().all(|&v| v == -4),
            "cb"
        );
        assert!(
            res.cr.as_ref().unwrap().samples.iter().all(|&v| v == 10),
            "cr"
        );
    }

    /// §8.4.4.1 step-8 ordering: cross-component prediction applies
    /// BEFORE the §8.6.8.2 colour transform. Luma coded 8 with
    /// ResScaleVal +8 on Cb: rCb becomes 8 before the lifting ⇒
    /// tmp = 8 − 4 = 4, rY = 12, rCb = 4, rCr = 4 (Cr CCP absent,
    /// rCr = 0 + rCb after eq. 8-339... rCr = 0 → rCb' = 4 ⇒ rCr = 4).
    #[test]
    fn act_applies_after_ccp() {
        let params = params_444();
        let raw = |levels: Vec<i32>| ResidualBlock {
            log2_trafo_size: 3,
            last_sig_coeff_x: 7,
            last_sig_coeff_y: 7,
            levels,
            transform_skip: false,
            explicit_rdpcm_flag: false,
            explicit_rdpcm_dir_flag: false,
        };
        let unit = TransformUnit {
            tu_residual_act_flag: 1,
            residual_luma: Some(raw(vec![8; 64])),
            cross_comp_pred_cb: Some(ccp(8)),
            ..Default::default()
        };
        let tree = TransformTree::Leaf {
            cbf_luma: true,
            unit,
        };
        let res = extract_cu_residual(&params, Some(&tree), 0, 0, 8, 25, true).unwrap();
        assert!(res.luma.samples.iter().all(|&v| v == 12), "luma");
        assert!(
            res.cb.as_ref().unwrap().samples.iter().all(|&v| v == 4),
            "cb"
        );
        assert!(
            res.cr.as_ref().unwrap().samples.iter().all(|&v| v == 4),
            "cr"
        );
    }
}


/// TEMPORARY issue #280 measurement scaffolding: the prediction-unit size and
/// uni/bi mix a real decode actually runs. Removed before the fix lands.
pub mod tmp_pu_hist {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    pub static HIST: Mutex<Option<BTreeMap<(usize, usize, bool), u64>>> = Mutex::new(None);
    pub fn enable() {
        *HIST.lock().unwrap() = Some(BTreeMap::new());
    }
    pub fn record(w: usize, h: usize, p0: bool, p1: bool) {
        if let Ok(mut g) = HIST.lock()
            && let Some(m) = g.as_mut()
        {
            *m.entry((w, h, p0 && p1)).or_insert(0) += 1;
        }
    }
    pub fn dump() {
        let g = HIST.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        let total: u64 = m.values().sum();
        let samples: u64 = m.iter().map(|((w, h, _), n)| (w * h) as u64 * n).sum();
        println!("PU histogram: {total} PUs, {samples} luma samples");
        let mut rows: Vec<_> = m.iter().collect();
        rows.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        for ((w, h, bi), n) in rows {
            let share = (w * h) as f64 * *n as f64 / samples as f64 * 100.0;
            println!(
                "  {w:>2}x{h:<2} {} {n:>7} PUs  {share:>5.1}% of samples",
                if *bi { "bi " } else { "uni" }
            );
        }
    }
}
