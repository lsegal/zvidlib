//! The residual-coding IDR writer — the encoder's lossy access-unit path.
//!
//! [`crate::hevc::engine::encoder::pcm`] writes every coding unit as a
//! `pcm_flag == 1` PCM block, so its output is exactly the source picture and
//! nothing in the encoder ever quantizes. This module is the other writer: the
//! same fixed geometry (`CtbSizeY == MinCbSizeY == 16`, one unsplit
//! `PART_2Nx2N` intra coding unit per CTB, one 16x16 luma transform block and
//! two 8x8 chroma ones), but with `pcm_flag == 0` and a §7.3.8.11
//! `residual_coding( )` body carrying quantized levels.
//!
//! ## Why the reconstruction is produced here
//!
//! A lossy writer cannot reconstruct a picture from its source: the samples a
//! decoder holds are the prediction plus the *dequantized* residual, and the
//! next block's intra prediction reads those samples, not the source's. So
//! this writer reconstructs as it codes, block by block in coding order, and
//! returns the reconstructed picture alongside the access unit. That
//! reconstruction is the decoder's own output by construction: the prediction
//! comes from [`crate::hevc::engine::intra_pred`] and the residual from
//! [`crate::hevc::engine::encoder::quant::reconstruct_residual`], which is the
//! decoder's §8.6.2 process run on the levels that were actually written.
//!
//! ## What is coded
//!
//! Every coding unit carries the intra luma mode
//! [`crate::hevc::engine::encoder::rdo::decide_intra_luma_mode`] picked for it
//! out of all 35 Table 8-1 directions, searched against the same reference
//! samples the block is then coded from, and signalled per §7.3.8.5: a
//! `prev_intra_luma_pred_flag == 1` plus `mpm_idx` when the mode is in the
//! §8.4.2 candidate list, and `rem_intra_luma_pred_mode` otherwise. The
//! candidate list is derived from the left coding unit's mode, since the above
//! neighbour of a CU that fills its CTB always lies in the CTB row above and
//! §8.4.2 reduces it to `INTRA_DC`. Chroma stays at
//! `intra_chroma_pred_mode == 4`, so it is predicted with the luma mode too.
//!
//! ## In-loop filtering
//!
//! The §8.7.2 deblocking filter runs: the PPS carries
//! `pps_deblocking_filter_disabled_flag == 0` and the writer runs the
//! decoder's own §8.7.2 driver over its reconstruction, so what
//! [`write_idr_residual_slice`] returns is still exactly what a decoder holds.
//! It runs as a whole-picture pass after the last coding unit is coded rather
//! than interleaved into coding order, because §8.4.4.2.2 intra prediction
//! reads its neighbouring samples *prior to* the in-loop filter process — the
//! filtered samples are the picture's output and the next picture's reference,
//! never this picture's own prediction input. See
//! [`crate::hevc::engine::encoder::recon::deblock_reconstruction`].
//!
//! SAO stays off (`sample_adaptive_offset_enabled_flag == 0` in the SPS). It
//! is a per-CTB parameter search plus §7.3.8.3 syntax in the slice data, a
//! decision independent of deblocking, and it is deliberately not made here.

use crate::hevc::engine::binarization::{derive_intra_pred_mode_c, intra_luma_cand_mode_list};
use crate::hevc::engine::cabac::init_type;
use crate::hevc::engine::ctx_init::SliceContexts;
use crate::hevc::engine::encoder::bitwriter::BitWriter;
use crate::hevc::engine::encoder::cabac::CabacEncoder;
use crate::hevc::engine::encoder::nal::{annexb, nal_unit};
use crate::hevc::engine::encoder::pcm::{
    PcmEncodeError, level_idc_for, write_pps, write_sps, write_vps,
};
use crate::hevc::engine::encoder::rdo::{
    DistortionBackend, intra_mode_bit_cost, lambda_q8, shortlist_intra_luma_modes,
};
use crate::hevc::engine::encoder::recon::{ReconstructedPicture, deblock_reconstruction};
use crate::hevc::engine::encoder::residual::{
    EngineResidualBinSink, ResidualWriteParams, has_coded_levels, write_residual_coding,
};
use crate::hevc::engine::encoder::transform::{
    ForwardBlockParams, chroma_qp, luma_qp, transform_and_quantize,
};
use crate::hevc::engine::intra_pred::{
    Component as IpComponent, IntraPredParams, MarkedReferenceSamples, ReferenceSamples,
    intra_predict, substitute_reference_samples,
};
use crate::hevc::engine::picture::clip1;
use crate::hevc::engine::scan::ScanIdx;
use crate::hevc::engine::transform::{
    BlockParams, Component as TfComponent, PredMode, residual_block,
};

/// `CtbLog2SizeY` of the writer's fixed geometry, matching the PCM writer's.
const CTB_LOG2: u32 = 4;
/// `CtbSizeY`.
const CTB: usize = 1 << CTB_LOG2;
/// `BitDepthY` / `BitDepthC`.
const BIT_DEPTH: u8 = 8;
/// `ChromaArrayType` — 4:2:0, the only format this writer emits.
const CHROMA_ARRAY_TYPE: u8 = 1;
/// `INTRA_DC` (Table 8-1 mode 1) — the §8.4.2 substitute for an unavailable
/// neighbour's mode.
const INTRA_DC: u8 = 1;
/// How many of the rough-mode-decision's ranked modes are re-scored on their
/// actual quantized reconstruction. Four covers the planar / DC / horizontal /
/// vertical spread the rough pass confuses most often without paying for 35
/// transforms per coding unit.
const MODE_SHORTLIST: usize = 4;
/// Table 9-46 value 4 — `intra_chroma_pred_mode` deriving chroma from the
/// coding unit's own luma mode.
const CHROMA_MODE_DERIVED: u8 = 4;
/// The valid `SliceQpY` range at 8-bit depth (`QpBdOffsetY == 0`).
const QP_RANGE: core::ops::RangeInclusive<i32> = 0..=51;

/// Which intra modes the writer is allowed to code.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ModeSearch {
    /// The full decision: [`shortlist_intra_luma_modes`] over all 35 luma
    /// modes, re-scored on the quantized reconstruction, and the §8.4.3 chroma
    /// mode picked the same way.
    Rdo,
    /// Every coding unit pinned to `INTRA_DC` with chroma derived from it —
    /// the writer's behaviour before the mode search, kept as the baseline the
    /// search is measured against.
    #[cfg(test)]
    DcOnly,
}

/// Whether the writer's reconstruction carries the §8.7.2 in-loop deblocking
/// filter, matching the `pps_deblocking_filter_disabled_flag` of the parameter
/// sets the slice is emitted with.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LoopFilter {
    /// `pps_deblocking_filter_disabled_flag == 0` — what the writer emits.
    Deblock,
    /// The filter neutralized, as the writer emitted before deblocking landed.
    /// Kept as the baseline the filter's gain is measured against.
    #[cfg(test)]
    Off,
}

/// One colour component's geometry inside the picture being coded.
#[derive(Clone, Copy)]
struct Plane<'a> {
    source: &'a [u8],
    width: usize,
    height: usize,
}

/// Encode one 4:2:0 8-bit frame as a self-contained IDR access unit whose
/// coding units carry quantized residual, and return it together with the
/// picture a conforming decoder reconstructs from it.
///
/// `qp` is `SliceQpY`, in 0..=51. At `qp == 0` the quantizer is at its finest
/// step and the reconstruction is close to — but, unlike the PCM writer's,
/// not identical to — the source.
///
/// # Errors
/// [`PcmEncodeError::BadDimensions`] when the dimensions are not nonzero
/// multiples of 16 or `qp` is out of range, and [`PcmEncodeError::PlaneSize`]
/// when a plane buffer has the wrong length.
pub fn encode_idr_residual_au(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
) -> Result<(Vec<u8>, ReconstructedPicture), PcmEncodeError> {
    if width == 0 || height == 0 || width % CTB != 0 || height % CTB != 0 {
        return Err(PcmEncodeError::BadDimensions { width, height });
    }
    if !QP_RANGE.contains(&qp) {
        return Err(PcmEncodeError::BadDimensions { width, height });
    }
    let check = |plane: &'static str, buf: &[u8], expected: usize| {
        if buf.len() == expected {
            Ok(())
        } else {
            Err(PcmEncodeError::PlaneSize {
                plane,
                expected,
                got: buf.len(),
            })
        }
    };
    check("y", y, width * height)?;
    check("cb", cb, width * height / 4)?;
    check("cr", cr, width * height / 4)?;

    let (rbsp, recon, _modes) =
        write_idr_residual_slice(y, cb, cr, width, height, qp, ModeSearch::Rdo, LoopFilter::Deblock);
    let level_idc = level_idc_for(width * height);
    let units = vec![
        nal_unit(32, 0, 0, &write_vps(level_idc)), // VPS_NUT
        nal_unit(33, 0, 0, &write_sps(width, height, level_idc, false, true)), // SPS_NUT
        nal_unit(34, 0, 0, &write_pps(false, true, None)), // PPS_NUT
        nal_unit(20, 0, 0, &rbsp),                 // IDR_N_LP
    ];
    Ok((annexb(&units), recon))
}

/// §7.3.6.1 + §7.3.8.1 — the picture's single I slice segment, every CTB one
/// residual-coded intra coding unit. Returns the slice RBSP, the picture
/// reconstructed alongside it, and the luma intra mode coded for each CTB in
/// coding order.
///
/// `filter` must match the `pps_deblocking_filter_disabled_flag` of the PPS
/// the slice is emitted with, or the returned reconstruction stops being the
/// decoder's.
#[allow(clippy::too_many_arguments)]
fn write_idr_residual_slice(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
    search: ModeSearch,
    filter: LoopFilter,
) -> (Vec<u8>, ReconstructedPicture, Vec<u8>) {
    let mut w = BitWriter::new();
    // ---- slice_segment_header() ----
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(2); // slice_type = I
    w.se(qp - 26); // slice_qp_delta over init_qp_minus26 == 0
    if filter == LoopFilter::Deblock {
        // §7.3.6.1: present because pps_loop_filter_across_slices_enabled_flag
        // is 1 and slice_deblocking_filter_disabled_flag is 0. One slice fills
        // the picture, so the value only has to be legal, not restrictive.
        w.put_bit(1); // slice_loop_filter_across_slices_enabled_flag
    }
    w.rbsp_trailing_bits(); // byte_alignment() before slice data

    // §9.3.2.2 initialization: initType 0 (I slice, equation 9-7) at SliceQpY.
    let mut ctxs = SliceContexts::init(init_type(2, false), qp);
    let mut cabac = CabacEncoder::new();

    let (cw, ch) = (width / 2, height / 2);
    let mut recon = ReconstructedPicture {
        y: vec![0u8; width * height],
        cb: vec![0u8; cw * ch],
        cr: vec![0u8; cw * ch],
        width,
        height,
    };

    // §8.6.1: the luma and Table 8-10 chroma `qP` derivations, with the PPS
    // and slice chroma QP offsets this writer emits (both 0).
    let qp_luma = luma_qp(qp, BIT_DEPTH);
    let qp_chroma = chroma_qp(qp, 0, BIT_DEPTH, CHROMA_ARRAY_TYPE);

    let ctbs_x = width / CTB;
    let ctbs_y = height / CTB;
    let total = ctbs_x * ctbs_y;
    // `IntraPredModeY` of the coding unit coded immediately before this one,
    // which is the left neighbour whenever there is one.
    let mut left_mode = INTRA_DC;
    let mut modes = Vec::with_capacity(total);
    for addr in 0..total {
        let x0 = (addr % ctbs_x) * CTB;
        let y0 = (addr / ctbs_x) * CTB;

        // §8.4.2 step 2: candIntraPredModeB is INTRA_DC for every coding unit
        // here, because one CU fills the CTB and so the above neighbour always
        // lies in the CTB row above. candIntraPredModeA is the left CU's mode,
        // reduced to INTRA_DC at the left picture edge where it is unavailable.
        let cand_a = if x0 > 0 { left_mode } else { INTRA_DC };
        let candidates = intra_luma_cand_mode_list(cand_a, INTRA_DC);

        // The mode search reads the same partially reconstructed neighbours
        // the block is then coded from, so the winning mode's prediction is
        // exactly the one the reconstruction below is built on.
        let luma_plane = Plane {
            source: y,
            width,
            height,
        };
        let (mode, luma_coded) = decide_luma_mode(
            luma_plane, &recon.y, x0, y0, qp, qp_luma, candidates, search,
        );
        left_mode = mode;
        modes.push(mode);

        // ---- coding_unit(): PART_2Nx2N, pcm_flag == 0 ----
        cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1);
        cabac.encode_terminate(&mut w, 0); // pcm_flag = 0
        write_luma_intra_mode(&mut w, &mut cabac, &mut ctxs, mode, candidates);

        let (cx, cy) = (x0 / 2, y0 / 2);
        let chroma_planes = [
            (
                Plane {
                    source: cb,
                    width: cw,
                    height: ch,
                },
                TfComponent::Cb,
            ),
            (
                Plane {
                    source: cr,
                    width: cw,
                    height: ch,
                },
                TfComponent::Cr,
            ),
        ];
        let (signalled_chroma, chroma_coded) = decide_chroma_mode(
            chroma_planes,
            [&recon.cb, &recon.cr],
            cx,
            cy,
            qp,
            qp_chroma,
            mode,
            search,
        );
        // §9.3.3.8 / Table 9-46: value 4 (chroma derived from luma) is the
        // single context-coded 0 bin; 0..=3 is a 1 bin plus two FL bypass bins.
        if signalled_chroma == CHROMA_MODE_DERIVED {
            cabac.encode_decision(&mut w, &mut ctxs.intra_chroma_pred_mode[0], 0);
        } else {
            cabac.encode_decision(&mut w, &mut ctxs.intra_chroma_pred_mode[0], 1);
            cabac.encode_bypass(&mut w, (signalled_chroma >> 1) & 1);
            cabac.encode_bypass(&mut w, signalled_chroma & 1);
        }

        // ---- transform_tree(): split_transform_flag is absent because
        // max_transform_hierarchy_depth_intra == 0, so MaxTrafoDepth == 0
        // and the flag is inferred 0 (one 16x16 luma TB, two 8x8 chroma).
        write_back(&mut recon.y, width, x0, y0, CTB, &luma_coded.samples);
        let luma = luma_coded.levels;
        let [coded_cb, coded_cr] = chroma_coded;
        write_back(&mut recon.cb, cw, cx, cy, CTB / 2, &coded_cb.samples);
        write_back(&mut recon.cr, cw, cx, cy, CTB / 2, &coded_cr.samples);
        let (chroma_cb, chroma_cr) = (coded_cb.levels, coded_cr.levels);

        // §7.3.8.8 order: cbf_cb, cbf_cr (ctxInc = trafoDepth = 0), then
        // cbf_luma (ctxInc = 1 at trafoDepth 0).
        let cbf_cb = u8::from(has_coded_levels(&chroma_cb));
        let cbf_cr = u8::from(has_coded_levels(&chroma_cr));
        let cbf_luma = u8::from(has_coded_levels(&luma));
        cabac.encode_decision(&mut w, &mut ctxs.cbf_chroma[0], cbf_cb);
        cabac.encode_decision(&mut w, &mut ctxs.cbf_chroma[0], cbf_cr);
        cabac.encode_decision(&mut w, &mut ctxs.cbf_luma[1], cbf_luma);

        // ---- transform_unit(): the coded blocks, luma then Cb then Cr.
        for (levels, log2, is_chroma) in [
            (&luma, CTB_LOG2, false),
            (&chroma_cb, CTB_LOG2 - 1, true),
            (&chroma_cr, CTB_LOG2 - 1, true),
        ] {
            if !has_coded_levels(levels) {
                continue;
            }
            write_residual_coding(
                &mut EngineResidualBinSink {
                    writer: &mut w,
                    cabac: &mut cabac,
                    contexts: &mut ctxs.residual,
                },
                &ResidualWriteParams {
                    log2_trafo_size: log2,
                    is_chroma,
                    // §7.4.9.11: the mode-dependent scans need
                    // log2TrafoSize == 2 (or 3 for luma / 4:4:4), so both
                    // block sizes here take the up-right diagonal scan.
                    scan_idx: ScanIdx::Diagonal,
                },
                levels,
            );
        }

        // end_of_slice_segment_flag: 1 only at the picture's last CTB.
        cabac.encode_terminate(&mut w, u8::from(addr == total - 1));
    }
    // The final terminate-1 flush wrote the rbsp_stop_one_bit.
    w.align_zero();

    // §8.7.1 — the in-loop filter stage, after the whole picture is coded.
    // Nothing above may read the filtered samples: every block predicted from
    // the unfiltered reconstruction, which is what §8.4.4.2.2 specifies.
    if filter == LoopFilter::Deblock {
        deblock_reconstruction(&mut recon, qp);
    }
    (w.finish(), recon, modes)
}

/// Pick the luma intra mode for one coding unit and return it with the block
/// it codes to.
///
/// Two passes. [`shortlist_intra_luma_modes`] ranks all 35 Table 8-1 modes on
/// the prediction's SATD, which costs no transform; the shortlist is then
/// re-scored on what each mode actually costs after quantization: the
/// reconstruction's squared error against the source, traded against the
/// mode's own §7.3.8.5 signalling. The residual's rate is deliberately not in
/// that second cost — this writer has no rate control, so its job at a given
/// QP is the closest picture, and a better prediction shrinks the residual on
/// its own. The winner's `CodedBlock` is the one the caller commits, so the
/// mode that is coded is the mode that was measured.
#[allow(clippy::too_many_arguments)]
fn decide_luma_mode(
    plane: Plane<'_>,
    recon: &[u8],
    x0: usize,
    y0: usize,
    qp: i32,
    qp_luma: u32,
    candidates: [u8; 3],
    search: ModeSearch,
) -> (u8, CodedBlock) {
    let refs = reference_samples(recon, plane.width, plane.height, x0, y0, CTB);
    let code = |mode: u8| {
        code_block(
            plane,
            x0,
            y0,
            CTB,
            qp_luma,
            TfComponent::Luma,
            &predict(&refs, mode, TfComponent::Luma),
        )
    };
    #[cfg(test)]
    if search == ModeSearch::DcOnly {
        return (INTRA_DC, code(INTRA_DC));
    }
    let _ = search;

    let shortlist = shortlist_intra_luma_modes(
        &plane.source[y0 * plane.width + x0..],
        plane.width,
        CTB,
        candidates,
        qp,
        DistortionBackend::Dispatched,
        MODE_SHORTLIST,
        |mode| predict(&refs, mode, TfComponent::Luma),
    );
    let lambda = u64::from(lambda_q8(qp));
    let mut best: Option<(u64, u8, CodedBlock)> = None;
    for candidate in &shortlist {
        let coded = code(candidate.mode);
        let bits = u64::from(intra_mode_bit_cost(candidate.mode, candidates));
        let cost = sum_squared_error(plane, x0, y0, CTB, &coded.samples)
            .saturating_add(bits * lambda / 256);
        if best
            .as_ref()
            .is_none_or(|(best_cost, ..)| cost < *best_cost)
        {
            best = Some((cost, candidate.mode, coded));
        }
    }
    let (_, mode, coded) = best.expect("the shortlist is never empty");
    (mode, coded)
}

/// Pick the `intra_chroma_pred_mode` for one coding unit and return it with
/// the Cb and Cr blocks it codes to.
///
/// One syntax element covers both chroma blocks, so the five Table 9-46 values
/// are scored on the pair's joint squared error, against the same
/// signalling-only rate term [`decide_luma_mode`] uses. Value 4 (`IntraPredModeC == IntraPredModeY`) is one of them, so this
/// can only improve on deriving chroma from luma unconditionally — which
/// matters here because the luma mode is chosen on luma alone and the chroma
/// planes need not share its orientation.
#[allow(clippy::too_many_arguments)]
fn decide_chroma_mode(
    planes: [(Plane<'_>, TfComponent); 2],
    recon: [&[u8]; 2],
    cx: usize,
    cy: usize,
    qp: i32,
    qp_chroma: u32,
    luma_mode: u8,
    search: ModeSearch,
) -> (u8, [CodedBlock; 2]) {
    let n_tbs = CTB / 2;
    let refs: Vec<ReferenceSamples> = recon
        .iter()
        .map(|plane| reference_samples(plane, planes[0].0.width, planes[0].0.height, cx, cy, n_tbs))
        .collect();
    let code = |signalled: u8| {
        let mode = derive_intra_pred_mode_c(signalled, luma_mode, false);
        let coded: Vec<CodedBlock> = planes
            .iter()
            .zip(&refs)
            .map(|(&(plane, component), refs)| {
                code_block(
                    plane,
                    cx,
                    cy,
                    n_tbs,
                    qp_chroma,
                    component,
                    &predict(refs, mode, component),
                )
            })
            .collect();
        let [cb, cr]: [CodedBlock; 2] = coded.try_into().ok().expect("two chroma blocks");
        [cb, cr]
    };
    #[cfg(test)]
    if search == ModeSearch::DcOnly {
        return (CHROMA_MODE_DERIVED, code(CHROMA_MODE_DERIVED));
    }
    let _ = search;

    let lambda = u64::from(lambda_q8(qp));
    let mut best: Option<(u64, u8, [CodedBlock; 2])> = None;
    for signalled in 0..=CHROMA_MODE_DERIVED {
        let coded = code(signalled);
        let bits = u64::from(chroma_mode_bit_cost(signalled));
        let distortion = coded
            .iter()
            .zip(&planes)
            .map(|(block, &(plane, _))| sum_squared_error(plane, cx, cy, n_tbs, &block.samples))
            .sum::<u64>();
        let cost = distortion.saturating_add(bits * lambda / 256);
        if best
            .as_ref()
            .is_none_or(|(best_cost, ..)| cost < *best_cost)
        {
            best = Some((cost, signalled, coded));
        }
    }
    let (_, signalled, coded) = best.expect("value 4 is always evaluated");
    (signalled, coded)
}

/// §9.3.3.8 bin count for one `intra_chroma_pred_mode`: the single bin for
/// value 4, or that bin plus the two FL bypass bins for 0..=3.
fn chroma_mode_bit_cost(signalled: u8) -> u32 {
    if signalled == CHROMA_MODE_DERIVED {
        1
    } else {
        3
    }
}

/// One transform block coded against a candidate prediction: the levels the
/// bitstream would carry and the samples a decoder would reconstruct.
struct CodedBlock {
    levels: Vec<i32>,
    /// `n_tbs * n_tbs` reconstructed samples, row-major.
    samples: Vec<u8>,
}

/// Code one transform block from the `prediction` the caller derived for a
/// candidate mode: transform and quantize the residual and reconstruct from
/// the quantized levels, without touching the picture. The caller commits the
/// winning candidate with [`write_back`].
///
/// The reconstruction step deliberately runs the decoder's own §8.6.2 process
/// on the quantized levels rather than the encoder's un-quantized residual, so
/// the committed samples are what a decoder will hold and the next block
/// predicts from the same samples the decoder will.
#[allow(clippy::too_many_arguments)]
fn code_block(
    plane: Plane<'_>,
    x0: usize,
    y0: usize,
    n_tbs: usize,
    q_p: u32,
    component: TfComponent,
    prediction: &[i32],
) -> CodedBlock {
    let mut residual = vec![0i32; n_tbs * n_tbs];
    for row in 0..n_tbs {
        for col in 0..n_tbs {
            let source = i32::from(plane.source[(y0 + row) * plane.width + x0 + col]);
            residual[row * n_tbs + col] = source - prediction[row * n_tbs + col];
        }
    }
    let levels = transform_and_quantize(
        &residual,
        None,
        ForwardBlockParams {
            n_tbs,
            q_p,
            component,
            pred_mode: PredMode::Intra,
            bit_depth: BIT_DEPTH,
            extended_precision: false,
        },
    )
    .expect("encoder-sized transform block");

    // §8.6.6: recSamples = Clip1( predSamples + resSamples ), with the
    // residual reconstructed exactly as the decoder will — and skipped
    // entirely when `cbf == 0`, which is what the decoder infers.
    let reconstructed = if has_coded_levels(&levels) {
        residual_block(
            &levels,
            None,
            BlockParams {
                n_tbs,
                q_p,
                component,
                pred_mode: PredMode::Intra,
                bit_depth: BIT_DEPTH,
                extended_precision: false,
                transquant_bypass: false,
                transform_skip: false,
                transform_skip_rotation_enabled: false,
            },
        )
        .expect("encoder-sized transform block")
    } else {
        vec![0i32; n_tbs * n_tbs]
    };
    let samples = (0..n_tbs * n_tbs)
        .map(|i| clip1(prediction[i] + reconstructed[i], BIT_DEPTH) as u8)
        .collect();
    CodedBlock { levels, samples }
}

/// Commit a coded block's reconstructed samples into the picture the next
/// block predicts from.
fn write_back(recon: &mut [u8], width: usize, x0: usize, y0: usize, n_tbs: usize, samples: &[u8]) {
    for row in 0..n_tbs {
        let start = (y0 + row) * width + x0;
        recon[start..start + n_tbs].copy_from_slice(&samples[row * n_tbs..(row + 1) * n_tbs]);
    }
}

/// Squared error between a coded block's reconstruction and the source it was
/// coded from — the distortion half of the second RDO pass.
fn sum_squared_error(plane: Plane<'_>, x0: usize, y0: usize, n_tbs: usize, samples: &[u8]) -> u64 {
    let mut sse = 0u64;
    for row in 0..n_tbs {
        for col in 0..n_tbs {
            let source = i64::from(plane.source[(y0 + row) * plane.width + x0 + col]);
            let d = source - i64::from(samples[row * n_tbs + col]);
            sse += (d * d) as u64;
        }
    }
    sse
}

/// §8.4.4.2.2 — the substituted reference-sample array for one transform
/// block, read out of the partially reconstructed plane.
///
/// The §6.4.1 z-scan availability for this geometry — one unsplit coding unit
/// per CTB, coded in raster order, one slice, no tiles — is: the left column
/// exists for its first `nTbS` rows when the block is not at the left picture
/// edge (the rest belongs to the not-yet-coded block below-left); the top row
/// exists when the block is not at the top edge, and extends `nTbS` samples to
/// the right as long as those stay inside the picture, because the
/// above-right block was coded on the previous CTB row.
fn reference_samples(
    recon: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    n_tbs: usize,
) -> ReferenceSamples {
    let sample = |x: usize, y: usize| i32::from(recon[y * width + x]);
    let has_left = x0 > 0;
    let has_top = y0 > 0;
    let left: Vec<(i32, bool)> = (0..2 * n_tbs)
        .map(|i| {
            let available = has_left && i < n_tbs && y0 + i < height;
            if available {
                (sample(x0 - 1, y0 + i), true)
            } else {
                (0, false)
            }
        })
        .collect();
    let top: Vec<(i32, bool)> = (0..2 * n_tbs)
        .map(|i| {
            let available = has_top && x0 + i < width && (i < n_tbs || y0 >= n_tbs);
            if available {
                (sample(x0 + i, y0 - 1), true)
            } else {
                (0, false)
            }
        })
        .collect();
    let corner = if has_left && has_top {
        (sample(x0 - 1, y0 - 1), true)
    } else {
        (0, false)
    };
    let marked = MarkedReferenceSamples::new(n_tbs, corner, left, top)
        .expect("encoder-sized reference array");
    substitute_reference_samples(&marked, BIT_DEPTH).expect("encoder-sized reference array")
}

/// §8.4.4.2.1 steps 1 and 2 — the prediction one mode produces from an
/// already-substituted reference array. The SPS this writer emits leaves
/// `intra_smoothing_disabled_flag` and `strong_intra_smoothing_enabled_flag`
/// both 0, which is what these parameters mirror.
fn predict(p: &ReferenceSamples, mode: u8, component: TfComponent) -> Vec<i32> {
    intra_predict(
        p,
        &IntraPredParams {
            pred_mode_intra: mode,
            cidx: match component {
                TfComponent::Luma => IpComponent::Luma,
                TfComponent::Cb => IpComponent::Cb,
                TfComponent::Cr => IpComponent::Cr,
            },
            bit_depth: BIT_DEPTH,
            bit_depth_luma: BIT_DEPTH,
            intra_smoothing_disabled: false,
            strong_intra_smoothing_enabled: false,
            chroma_array_type_3: false,
            disable_boundary_filter: false,
        },
    )
    .expect("Table 8-1 intra mode")
}

/// §7.3.8.5 — write one coding unit's luma intra mode against the §8.4.2
/// `candModeList` the decoder will derive for it.
///
/// A mode in the list costs `prev_intra_luma_pred_flag == 1` plus the TR
/// (`cMax` 2) `mpm_idx` bins; any other costs the flag plus the five FL
/// `rem_intra_luma_pred_mode` bins, whose value is the §8.4.2 step-4
/// re-injection run backwards: the three most-probable modes are removed from
/// the 35-mode space by decrementing past each sorted candidate the mode
/// exceeds, high candidate first.
fn write_luma_intra_mode(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    mode: u8,
    candidates: [u8; 3],
) {
    if let Some(mpm_idx) = candidates.iter().position(|&c| c == mode) {
        cabac.encode_decision(w, &mut ctxs.prev_intra_luma_pred_flag[0], 1);
        // mpm_idx, TR(cMax 2, cRiceParam 0): "0", "10", "11".
        if mpm_idx == 0 {
            cabac.encode_bypass(w, 0);
        } else {
            cabac.encode_bypass(w, 1);
            cabac.encode_bypass(w, u8::from(mpm_idx == 2));
        }
        return;
    }
    cabac.encode_decision(w, &mut ctxs.prev_intra_luma_pred_flag[0], 0);
    let mut sorted = candidates;
    sorted.sort_unstable();
    let mut rem = mode;
    for candidate in sorted.iter().rev() {
        if rem > *candidate {
            rem -= 1;
        }
    }
    debug_assert_eq!(
        crate::hevc::engine::binarization::derive_intra_pred_mode_y(
            candidates,
            crate::hevc::engine::binarization::LumaIntraModeSource::Remaining,
            rem,
        ),
        mode,
        "rem_intra_luma_pred_mode does not decode back to the coded mode"
    );
    // rem_intra_luma_pred_mode, FL(cMax 31): five bypass bins, MSB first.
    for shift in (0..5).rev() {
        cabac.encode_bypass(w, (rem >> shift) & 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A picture with real structure at every scale: a diagonal gradient, a
    /// couple of hard edges, and low-amplitude noise, so the quantizer has
    /// something to throw away at every block size.
    fn picture(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut noise = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 59) as i32) - 8
        };
        let y = (0..width * height)
            .map(|i| {
                let (x, r) = (i % width, i / width);
                let base = (x * 3 + r * 2) % 200 + 20;
                let edge = if x > width / 2 { 40 } else { 0 };
                (base as i32 + edge + noise()).clamp(0, 255) as u8
            })
            .collect();
        let chroma = |offset: i32| -> Vec<u8> {
            (0..(width / 2) * (height / 2))
                .map(|i| {
                    let (x, r) = ((i % (width / 2)) as i32, (i / (width / 2)) as i32);
                    (128 + ((x - r + offset) % 40) - 20).clamp(0, 255) as u8
                })
                .collect()
        };
        (y, chroma(0), chroma(13))
    }

    /// Decode an access unit through the crate's own end-to-end HEVC driver
    /// and return its 4:2:0 planes.
    fn decode(au: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let frames = crate::hevc::engine::sequence::decode_annexb_sequence(au).expect("decode");
        assert_eq!(frames.len(), 1, "one IDR frame");
        let planar = frames[0].picture.to_planar_u8().expect("8-bit");
        let luma = width * height;
        let chroma = luma / 4;
        (
            planar[..luma].to_vec(),
            planar[luma..luma + chroma].to_vec(),
            planar[luma + chroma..].to_vec(),
        )
    }

    #[test]
    fn the_decoded_stream_matches_the_encoders_own_reconstruction() {
        // The property the whole lossy path rests on: whatever the quantizer
        // rounded away, the encoder's reference picture and the decoder's
        // output agree on it sample for sample. A mismatch here means the
        // next picture would predict from something no decoder holds.
        let (width, height) = (64, 48);
        let (y, cb, cr) = picture(width, height);
        for qp in [0i32, 12, 26, 37, 51] {
            let (au, recon) = encode_idr_residual_au(&y, &cb, &cr, width, height, qp).unwrap();
            let (dy, dcb, dcr) = decode(&au, width, height);
            assert_eq!(dy, recon.y, "qp {qp}: luma diverged");
            assert_eq!(dcb, recon.cb, "qp {qp}: Cb diverged");
            assert_eq!(dcr, recon.cr, "qp {qp}: Cr diverged");
        }
    }

    #[test]
    fn the_reconstruction_is_lossy_and_gets_lossier_with_qp() {
        let (width, height) = (64, 48);
        let (y, cb, cr) = picture(width, height);
        let sse = |a: &[u8], b: &[u8]| -> u64 {
            a.iter()
                .zip(b)
                .map(|(&p, &q)| {
                    let d = i64::from(p) - i64::from(q);
                    (d * d) as u64
                })
                .sum()
        };
        let mut previous = 0u64;
        for qp in [12i32, 26, 37, 51] {
            let (_, recon) = encode_idr_residual_au(&y, &cb, &cr, width, height, qp).unwrap();
            let error = sse(&recon.y, &y);
            assert!(
                error > previous,
                "qp {qp} did not cost more distortion than the finer step"
            );
            previous = error;
        }
        assert!(
            previous > 0,
            "the residual writer reproduced the source exactly — it is not lossy"
        );
    }

    #[test]
    fn a_coarser_qp_produces_a_smaller_access_unit_than_the_pcm_writer() {
        // The point of coding residual instead of PCM: at a working QP the
        // access unit must be a fraction of the raw sample payload.
        let (width, height) = (64, 48);
        let (y, cb, cr) = picture(width, height);
        let pcm = crate::hevc::engine::encoder::pcm::encode_idr_pcm_au(&y, &cb, &cr, width, height)
            .unwrap();
        let (lossy, _) = encode_idr_residual_au(&y, &cb, &cr, width, height, 32).unwrap();
        assert!(
            lossy.len() * 2 < pcm.len(),
            "residual AU {} bytes vs PCM {} bytes",
            lossy.len(),
            pcm.len()
        );
    }

    #[test]
    fn a_flat_picture_codes_no_residual_at_all() {
        // Every block predicts exactly, so every cbf is 0 and the access unit
        // carries syntax only — the cbf == 0 inference path.
        let (width, height) = (32, 32);
        let y = vec![120u8; width * height];
        let c = vec![128u8; width * height / 4];
        let (au, recon) = encode_idr_residual_au(&y, &c, &c, width, height, 26).unwrap();
        let (dy, dcb, dcr) = decode(&au, width, height);
        assert_eq!(dy, recon.y);
        assert_eq!(dcb, recon.cb);
        assert_eq!(dcr, recon.cr);
        // The first block has no neighbours, so it predicts 128 and codes the
        // difference; every later block predicts its neighbour exactly.
        assert!(recon.y[width * height - 1].abs_diff(120) <= 1);
    }

    /// The writer's pre-RDO behaviour: the same coding loop with every mode
    /// pinned to `INTRA_DC` and chroma derived from it.
    fn dc_only(
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        width: usize,
        height: usize,
        qp: i32,
    ) -> (Vec<u8>, ReconstructedPicture) {
        let (rbsp, recon, _) =
            write_idr_residual_slice(y, cb, cr, width, height, qp, ModeSearch::DcOnly, LoopFilter::Deblock);
        (rbsp, recon)
    }

    fn psnr_db(source: &[u8], recon: &[u8]) -> f64 {
        let sse: f64 = source
            .iter()
            .zip(recon)
            .map(|(&a, &b)| {
                let d = f64::from(a) - f64::from(b);
                d * d
            })
            .sum();
        if sse == 0.0 {
            return f64::INFINITY;
        }
        10.0 * (255.0f64.powi(2) * source.len() as f64 / sse).log10()
    }

    #[test]
    fn the_writer_codes_more_than_one_intra_mode() {
        // The whole point of the mode search: a picture with structure at
        // several orientations must not come out uniformly DC.
        let (width, height) = (64, 48);
        let (y, cb, cr) = picture(width, height);
        let (_, _, modes) =
            write_idr_residual_slice(&y, &cb, &cr, width, height, 26, ModeSearch::Rdo, LoopFilter::Deblock);
        assert_eq!(modes.len(), (width / CTB) * (height / CTB));
        let distinct: std::collections::BTreeSet<u8> = modes.iter().copied().collect();
        assert!(
            distinct.len() > 1,
            "the writer coded a single mode {distinct:?} — the search is not reaching the \
             bitstream"
        );
        assert!(
            modes.iter().any(|&m| m != INTRA_DC),
            "every coding unit is still DC"
        );
    }

    #[test]
    fn the_mode_search_beats_the_dc_only_writer_on_quality_and_rate() {
        // The gain the changelog records. Both slices come out of the same
        // coding loop at the same QP, so the difference is the mode decision
        // and nothing else: the picture is closer to the source *and* costs
        // fewer bits, which is what a rate-distortion decision is for.
        let (width, height) = (64, 48);
        let (y, cb, cr) = picture(width, height);
        for qp in [12i32, 26, 37] {
            let (searched_slice, searched, _) =
                write_idr_residual_slice(&y, &cb, &cr, width, height, qp, ModeSearch::Rdo, LoopFilter::Deblock);
            let (baseline_slice, baseline, _) =
                write_idr_residual_slice(&y, &cb, &cr, width, height, qp, ModeSearch::DcOnly, LoopFilter::Deblock);
            let luma = (psnr_db(&y, &searched.y), psnr_db(&y, &baseline.y));
            assert!(
                luma.0 > luma.1,
                "qp {qp}: mode search gave {:.2} dB luma, DC-only gave {:.2} dB",
                luma.0,
                luma.1
            );
            let picture_psnr = |recon: &ReconstructedPicture| {
                psnr_db(
                    &[y.clone(), cb.clone(), cr.clone()].concat(),
                    &[recon.y.clone(), recon.cb.clone(), recon.cr.clone()].concat(),
                )
            };
            assert!(
                picture_psnr(&searched) > picture_psnr(&baseline),
                "qp {qp}: whole-picture PSNR {:.2} dB did not beat the DC-only {:.2} dB",
                picture_psnr(&searched),
                picture_psnr(&baseline)
            );
            assert!(
                searched_slice.len() < baseline_slice.len(),
                "qp {qp}: the searched slice is {} bytes against the DC-only {}",
                searched_slice.len(),
                baseline_slice.len()
            );
        }
    }

    /// Smooth, band-limited content: the case blocking artifacts are visible
    /// in and the §8.7.2.5.3 `d < β` decision actually filters. The noisy
    /// [`picture`] above is deliberately the opposite case.
    fn smooth_picture(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y = (0..width * height)
            .map(|i| {
                let (x, r) = ((i % width) as f64, (i / width) as f64);
                (110.0 + 60.0 * ((x / 37.0).sin() + (r / 23.0).cos())).clamp(0.0, 255.0) as u8
            })
            .collect();
        let c = |offset: f64| {
            (0..(width / 2) * (height / 2))
                .map(|i| {
                    let (x, r) = ((i % (width / 2)) as f64, (i / (width / 2)) as f64);
                    (128.0 + 25.0 * ((x + r + offset) / 19.0).sin()).clamp(0.0, 255.0) as u8
                })
                .collect()
        };
        (y, c(0.0), c(7.0))
    }

    #[test]
    fn the_deblocking_filter_improves_round_trip_psnr() {
        // The gain the changelog records. Both reconstructions come out of the
        // same coding loop at the same QP with the same mode decisions, so the
        // only difference between them is the §8.7.2 pass over the finished
        // picture.
        let (width, height) = (64, 48);
        let (y, cb, cr) = smooth_picture(width, height);
        for qp in [20i32, 26, 32, 37, 44, 51] {
            let (_, filtered, _) = write_idr_residual_slice(
                &y, &cb, &cr, width, height, qp, ModeSearch::Rdo, LoopFilter::Deblock,
            );
            let (_, unfiltered, _) = write_idr_residual_slice(
                &y, &cb, &cr, width, height, qp, ModeSearch::Rdo, LoopFilter::Off,
            );
            assert_ne!(
                filtered.y, unfiltered.y,
                "qp {qp}: the filter left every luma sample alone"
            );
            let gain = psnr_db(&y, &filtered.y) - psnr_db(&y, &unfiltered.y);
            assert!(
                gain > 0.3,
                "qp {qp}: deblocking moved luma PSNR by only {gain:+.3} dB"
            );
        }
    }

    #[test]
    fn the_deblocking_filter_does_not_cost_quality_on_noisy_content() {
        // The §8.7.2.5.3 decision declines to filter an edge whose two sides
        // are not flat, so on the noise-carrying picture the filter has almost
        // nothing to do. What it must not do is smear the noise: the gain is
        // allowed to be negligible, not negative.
        let (width, height) = (64, 48);
        let (y, cb, cr) = picture(width, height);
        let source = [y.clone(), cb.clone(), cr.clone()].concat();
        for qp in [12i32, 26, 37, 51] {
            let (_, filtered, _) = write_idr_residual_slice(
                &y, &cb, &cr, width, height, qp, ModeSearch::Rdo, LoopFilter::Deblock,
            );
            let (_, unfiltered, _) = write_idr_residual_slice(
                &y, &cb, &cr, width, height, qp, ModeSearch::Rdo, LoopFilter::Off,
            );
            let whole = |r: &ReconstructedPicture| {
                psnr_db(&source, &[r.y.clone(), r.cb.clone(), r.cr.clone()].concat())
            };
            let gain = whole(&filtered) - whole(&unfiltered);
            assert!(
                gain > -0.05,
                "qp {qp}: deblocking cost {gain:+.3} dB of whole-picture PSNR"
            );
        }
    }

    #[test]
    fn the_access_unit_signals_the_deblocking_filter_as_enabled() {
        // The reconstruction is only the decoder's if the parameter sets ask
        // the decoder for the same filter, so read the flag back off the
        // emitted PPS rather than trusting the writer.
        let (width, height) = (32, 32);
        let (y, cb, cr) = picture(width, height);
        let (au, _) = encode_idr_residual_au(&y, &cb, &cr, width, height, 32).unwrap();
        let pps = crate::hevc::engine::nal::collect_nal_units(&au)
            .expect("the access unit parses")
            .into_iter()
            .find(|nal| nal.header.nal_unit_type == 34)
            .expect("the access unit carries a PPS");
        let parsed =
            crate::hevc::engine::pps::PicParameterSet::parse(&pps.rbsp).expect("PPS parses");
        assert!(parsed.deblocking_filter_control_present_flag);
        assert!(
            !parsed.deblocking.disabled_flag,
            "the writer's PPS still disables deblocking"
        );
    }

    #[test]
    fn rejects_bad_geometry_and_out_of_range_qp() {
        let (y, cb, cr) = picture(32, 32);
        assert!(encode_idr_residual_au(&y, &cb, &cr, 30, 32, 26).is_err());
        assert!(encode_idr_residual_au(&y, &cb, &cr, 32, 32, -1).is_err());
        assert!(encode_idr_residual_au(&y, &cb, &cr, 32, 32, 52).is_err());
        assert!(encode_idr_residual_au(&y, &cb, &y, 32, 32, 26).is_err());
    }
}
