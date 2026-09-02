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
//! §8.7.3 SAO runs behind it, in the §8.7.1 order: the SPS carries
//! `sample_adaptive_offset_enabled_flag == 1`, the writer searches both
//! §8.7.3.2 types per CTB over the *deblocked* reconstruction
//! ([`crate::hevc::engine::encoder::recon::sao_reconstruction`]) and codes the
//! §7.3.8.3 `sao( )` structure it found at the head of each CTB's slice data.
//! Both types, because they reach different error: the four edge-offset
//! classes shape the error around a local edge, while band offset shapes it
//! over a *value range*, which is what a CTB whose reconstruction is uniformly
//! biased across some part of the sample range needs and no edge class can
//! deliver. They are searched together and scored against each other under one
//! `D + lambda * R` comparison, so the type is chosen by what it buys net of
//! what it costs — band offset pays four `sao_offset_sign` bins and five
//! `sao_band_position` bins where edge offset pays two class bins and infers
//! its signs.
//!
//! ## Why SAO is on, and when the writer turns it off again
//!
//! SAO is not free the way deblocking is: deblocking is signalled once in the
//! PPS, while SAO costs a `sao( )` structure on *every* CTB, including every
//! CTB it does nothing to. So the decision is taken twice. Per CTB, a class
//! is signalled only when the squared error it removes clears the §9.3.3 bins
//! it would be coded with, under the same `D + lambda * R` and the same
//! closed-form `rdo::lambda_q8` the mode search uses — inside a picture
//! already committed to the syntax, that is a choice between two codings of
//! one CTB at one QP, which is the trade that multiplier is derived for.
//!
//! Per slice, the whole pass is kept only when the error it actually removed
//! clears the bits the whole grid actually costs — and that one is not the
//! same trade. It is a choice between spending bits on SAO and spending the
//! same bits on a finer quantizer, so what those bits are worth is a property
//! of the picture rather than of the QP, and the closed form misses this
//! picture's own slope by 0.4x to 2.5x over the sweep. So it is measured:
//! [`curve_point`] codes the same picture one QP finer, and
//! [`calibrated_sao_lambda_q8`] reads off the curve through those two points
//! what `sao_bits` more of it is worth. When SAO does not clear that, the
//! reconstruction reverts to the deblocked one and `slice_sao_luma_flag` /
//! `slice_sao_chroma_flag` go out as 0, which §7.3.8.3 reads as "code
//! nothing", leaving the cost at the two header bits.
//!
//! The probe is a second decision pass, so [`keeps_sao`] takes it only where
//! it can still change the answer — outside the band the calibrated
//! multiplier is clamped to, the closed form settles the decision on its own,
//! which over the sweep is half the pictures. Where the probe does run it
//! costs the writer 127% to 148% of a picture; where it does not, the whole
//! SAO stage costs 19% to 25%.
//!
//! Measured against the same writer with SAO off, same QP, same mode
//! decisions, whole-picture PSNR and slice size over the QP 12 to 51 sweep on
//! both test pictures at 64x48 and 128x96, the pass is taken at 36 of those
//! points and buys +0.07 to +0.89 dB for +0.4% to +13.4% of the slice. On
//! smooth content at QP 12 it buys +0.73 dB for +11.4% at 64x48 and +0.89 dB
//! for +13.4% at 128x96. Every accepted point, including those two, sits on
//! or above the SAO-off writer's own rate-distortion curve interpolated in
//! log-rate at the same slice size — from +0.001 dB at the closest to
//! +0.55 dB at the two clear wins — which is what the fixed multiplier could
//! not hold: it declined 30 of these points and accepted two others that sat
//! 0.08 and 0.14 dB below that curve.
//!
//! Band offset is what the finest end of the noise picture's sweep is. With
//! only the edge classes searched the slice-level test declines the pass on
//! that picture at QP 12 at 64x48 outright; searching both types it is taken
//! and buys +0.100 dB for +0.5% of slice. At 128x96 it improves five points
//! the edge-only search already took, each for one to eight bytes more:
//! +0.106 to +0.120 dB at QP 12, +0.184 to +0.190 at QP 13, +0.156 to +0.167
//! at QP 14, +0.110 to +0.118 at QP 17, and +0.100 to +0.127 at QP 19. Every
//! other point of the sweep is byte-identical to the edge-only search, no
//! point regresses, and every accepted point still sits on or above the
//! SAO-off writer's own curve. It is never selected on the smooth picture at
//! any QP — that content's error is edge-shaped, which is what §8.7.3.2 is
//! for — nor at any QP coarse enough that the four signs and five position
//! bins outweigh what a value-range bias is worth.
//!
//! Those band-only bins are charged at 2.5x `lambda_q8`, the coarse end of
//! the departure the calibration measured, because they are rate the closed
//! form has never been checked against. At the closed form itself the search
//! takes band offset at one coarse CTB whose slice then sits 0.003 dB *under*
//! the curve above; at `SAO_LAMBDA_BAND`, the trust bound rather than the
//! measured end, band offset stops being selected anywhere and all of the
//! gains above are given back.
//!
//! ## Why the writer runs two passes
//!
//! §7.3.8.3 puts a CTB's SAO parameters at the head of that CTB's slice data,
//! but §8.7.1 only lets them be searched once the whole picture is coded and
//! deblocked. So the coding loop is split: a decision pass that searches every
//! coding unit and builds the reconstruction, then the filters, then a
//! bitstream pass that codes the recorded decisions with the `sao( )` syntax
//! in front of each. Nothing a decision depends on changes in between —
//! §8.4.4.2.2 intra prediction reads the unfiltered reconstruction, which the
//! decision pass has already built — so the picture that is coded is exactly
//! the picture that was searched.
//!
//! The bitstream pass is also where the slice-level SAO decision is settled
//! without coding anything twice: the slice data is coded once with the grid
//! and once without, the decision compares their sizes, and the one it keeps
//! is appended to the slice header verbatim.

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
    DistortionBackend, intra_mode_bit_cost, lambda_q8, residual_rate_bits,
    shortlist_intra_luma_modes,
};
use crate::hevc::engine::encoder::recon::{
    ReconstructedPicture, SAO_LAMBDA_BAND, SAO_OFFSET_MAX, SaoLambda, SourcePlanes,
    deblock_reconstruction, sao_reconstruction,
};
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
use crate::hevc::engine::sao::ResolvedSao;
use crate::hevc::engine::scan::ScanIdx;
use crate::hevc::engine::transform::{
    BlockParams, Component as TfComponent, PredMode, residual_block,
};

/// `CtbLog2SizeY` of the writer's fixed geometry, matching the PCM writer's.
const CTB_LOG2: u32 = 4;
/// The coarsest `SliceQpY` §7.4.7.1 allows at this bit depth.
const MAX_QP: i32 = 51;
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

/// Which intra modes the writer is allowed to code, and what the decision that
/// picks between them optimizes — the operating point.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ModeSearch {
    /// The fixed-QP operating point: [`shortlist_intra_luma_modes`] over all
    /// 35 luma modes, re-scored on the quantized reconstruction against the
    /// mode's own signalling only, and the §8.4.3 chroma mode picked the same
    /// way. With no bitrate to hit, the writer's job at a given QP is the
    /// closest picture, so the residual's own rate is left out of the cost.
    Rdo,
    /// The rate-constrained operating point: the same search, with the
    /// residual's estimated rate ([`residual_rate_bits`]) added to each
    /// candidate's rate term, so the decision minimizes the full
    /// `D + lambda * R` rather than distortion against signalling alone.
    ///
    /// This is what a rate-controlled encoder needs, and what the public
    /// factory's target-bitrate operating point codes at: with a bitrate to
    /// hit, the bits this decision saves are bits the next picture's QP can
    /// spend somewhere they buy more.
    RateDistortion,
    /// Every coding unit pinned to `INTRA_DC` with chroma derived from it —
    /// the writer's behaviour before the mode search, kept as the baseline the
    /// search is measured against.
    #[cfg(test)]
    DcOnly,
}

impl ModeSearch {
    /// Whether the second-pass cost charges each candidate for the bins its
    /// own residual would code, on top of the mode signalling.
    fn charges_residual_rate(self) -> bool {
        self == ModeSearch::RateDistortion
    }
}

/// Which in-loop filters the writer's reconstruction carries, matching the
/// `pps_deblocking_filter_disabled_flag` and
/// `sample_adaptive_offset_enabled_flag` of the parameter sets and the
/// `slice_sao_*_flag` pair of the slice header the slice is emitted with.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LoopFilter {
    /// §8.7.2 only: `pps_deblocking_filter_disabled_flag == 0` with
    /// `sample_adaptive_offset_enabled_flag == 0`. Kept as the baseline SAO's
    /// gain is measured against.
    Deblock,
    /// §8.7.2 followed by §8.7.3 — what the writer emits at both operating
    /// points.
    DeblockSao,
    /// Both filters neutralized, as the writer emitted before deblocking
    /// landed. Kept as the baseline deblocking's gain is measured against.
    #[cfg(test)]
    Off,
}

impl LoopFilter {
    /// Whether §8.7.2 runs, and the PPS asks a decoder for it.
    fn deblocking(self) -> bool {
        matches!(self, LoopFilter::Deblock | LoopFilter::DeblockSao)
    }

    /// Whether §8.7.3 runs, and the SPS and slice header ask a decoder for it.
    fn sao(self) -> bool {
        matches!(self, LoopFilter::DeblockSao)
    }
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
    encode_idr_residual_au_at(y, cb, cr, width, height, qp, ModeSearch::Rdo)
}

/// [`encode_idr_residual_au`] at the rate-constrained operating point: the
/// intra decision minimizes the full `D + lambda * R`, residual bits included,
/// so the same QP buys fewer bits and slightly more distortion than the
/// fixed-QP writer above — a better trade at equal rate, and the one a bitrate
/// target needs.
///
/// This is the writer the public factory's target-bitrate operating point
/// codes every picture through, with `qp` chosen per picture by
/// [`crate::hevc::engine::encoder::ratecontrol`]. The fixed-QP configuration
/// stays on [`encode_idr_residual_au`]: a caller that names a QP is asking for
/// a picture, not a rate.
///
/// # Errors
/// As [`encode_idr_residual_au`].
pub fn encode_idr_residual_au_rate_constrained(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
) -> Result<(Vec<u8>, ReconstructedPicture), PcmEncodeError> {
    encode_idr_residual_au_at(y, cb, cr, width, height, qp, ModeSearch::RateDistortion)
}

/// The body both operating points share: validate the request, run the coding
/// loop under `search`, and wrap the slice in the parameter sets.
fn encode_idr_residual_au_at(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
    search: ModeSearch,
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

    let filter = LoopFilter::DeblockSao;
    let (rbsp, recon, _modes) =
        write_idr_residual_slice(y, cb, cr, width, height, qp, search, filter);
    let level_idc = level_idc_for(width * height);
    let units = vec![
        nal_unit(32, 0, 0, &write_vps(level_idc)), // VPS_NUT
        nal_unit(
            33,
            0,
            0,
            &write_sps(width, height, level_idc, filter.sao(), true),
        ), // SPS_NUT
        nal_unit(34, 0, 0, &write_pps(false, true, None)), // PPS_NUT
        nal_unit(20, 0, 0, &rbsp),                 // IDR_N_LP
    ];
    Ok((annexb(&units), recon))
}

/// One coding unit's committed decisions, as the decision pass leaves them for
/// the bitstream pass to code.
///
/// The two passes exist because §7.3.8.3 puts each CTB's SAO parameters at the
/// *head* of that CTB's slice data, while §8.7.1 only lets those parameters be
/// searched once the whole picture is coded and deblocked. Nothing a decision
/// depends on changes in between — §8.4.4.2.2 intra prediction reads the
/// unfiltered reconstruction, which the decision pass has already built — so
/// replaying the recorded decisions codes exactly the picture that was
/// searched, at the cost of carrying its levels rather than re-searching it.
struct CtbRecord {
    /// `IntraPredModeY`.
    mode: u8,
    /// The §8.4.2 `candModeList` the mode is signalled against.
    candidates: [u8; 3],
    /// `intra_chroma_pred_mode`, a Table 9-46 value.
    chroma_mode: u8,
    /// The three transform blocks' quantized levels: luma, then Cb, then Cr.
    levels: [Vec<i32>; 3],
}

/// The decision pass: search every coding unit, reconstruct it, and return
/// the committed decisions with the unfiltered reconstruction they built.
///
/// Split out of [`write_idr_residual_slice`] because it is also what a curve
/// point costs: [`curve_point`] runs it at a neighbouring QP to measure the
/// slope the SAO decision is taken against, and the bitstream pass and the
/// in-loop filters are no part of that measurement.
fn run_decision_pass(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
    search: ModeSearch,
) -> (Vec<CtbRecord>, ReconstructedPicture) {
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

    // ---- the decision pass: search every coding unit and reconstruct it ----
    // `IntraPredModeY` of the coding unit coded immediately before this one,
    // which is the left neighbour whenever there is one.
    let mut left_mode = INTRA_DC;
    let mut records: Vec<CtbRecord> = Vec::with_capacity(total);
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
        let (chroma_mode, chroma_coded) = decide_chroma_mode(
            chroma_planes,
            [&recon.cb, &recon.cr],
            cx,
            cy,
            qp,
            qp_chroma,
            mode,
            search,
        );

        write_back(&mut recon.y, width, x0, y0, CTB, &luma_coded.samples);
        let [coded_cb, coded_cr] = chroma_coded;
        write_back(&mut recon.cb, cw, cx, cy, CTB / 2, &coded_cb.samples);
        write_back(&mut recon.cr, cw, cx, cy, CTB / 2, &coded_cr.samples);
        records.push(CtbRecord {
            mode,
            candidates,
            chroma_mode,
            levels: [luma_coded.levels, coded_cb.levels, coded_cr.levels],
        });
    }

    (records, recon)
}

/// §7.3.6.1 + §7.3.8.1 — the picture's single I slice segment, every CTB one
/// residual-coded intra coding unit. Returns the slice RBSP, the picture
/// reconstructed alongside it, and the luma intra mode coded for each CTB in
/// coding order.
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
    let ctbs_x = width / CTB;
    let (records, mut recon) = run_decision_pass(y, cb, cr, width, height, qp, search);

    // §8.7.1 — the in-loop filter stage, after the whole picture is coded.
    // Nothing above may read the filtered samples: every block predicted from
    // the unfiltered reconstruction, which is what §8.4.4.2.2 specifies.
    if filter.deblocking() {
        deblock_reconstruction(&mut recon, qp);
    }
    // §8.7.3 runs behind §8.7.2 and returns the per-CTB parameters the slice
    // data below codes, so the decoder resolves the same grid this
    // reconstruction was filtered with.
    //
    // Then the slice-level half of the decision. Every CTB the search leaves
    // unfiltered still costs its `sao( )` structure — one merge flag, in the
    // best case — so a picture SAO finds nothing in pays for the search's
    // silence on every CTB of it. `slice_sao_luma_flag` /
    // `slice_sao_chroma_flag` are what make that optional: with both 0,
    // §7.3.8.3 codes nothing at all and the whole cost is the two header
    // bits. So the pass is kept only when the error it actually removed
    // clears the bits it actually costs, at what a bit is worth on this
    // picture's own rate-distortion curve; otherwise the reconstruction
    // reverts to the deblocked one and the slice says so.
    //
    // The slice data as it stands without SAO. If the decision keeps the
    // pass, the coded grid's own replaces it — either way what the decision
    // measured is what the access unit carries, byte for byte.
    let mut slice_data = code_slice_data(&records, None, qp, ctbs_x);
    let mut sao_kept = false;
    if filter.sao() {
        let deblocked = recon.clone();
        // The per-CTB search keeps the closed-form multiplier: inside a
        // picture already committed to coding `sao( )` on every CTB, that
        // decision is between two codings of one CTB at one QP, which is the
        // trade `lambda_q8` is derived for and where the picture cancels out
        // of it. The slice-level test below is the one where it does not
        // cancel, and that one is taken against the measured curve.
        let grid = sao_reconstruction(
            &mut recon,
            SourcePlanes {
                y,
                cb,
                cr,
                width,
                height,
            },
            SaoLambda::for_search(lambda_q8(qp)),
        );
        let base = CurvePoint {
            sse: picture_sse(&deblocked, y, cb, cr),
            bits: slice_data.len() as u64 * 8,
        };
        let gain = base.sse.saturating_sub(picture_sse(&recon, y, cb, cr));
        // What the grid costs, read off the coded slice rather than counted
        // beside it: the same slice data with the `sao( )` structures in
        // front of each CTB, against the same slice data without them.
        let coded = code_slice_data(&records, Some(&grid), qp, ctbs_x);
        let sao_bits = (coded.len() as u64 * 8).saturating_sub(base.bits);
        // What SAO is weighed against is one step of the quantizer, so that
        // is the point the probe codes: this same writer one QP finer, or one
        // QP coarser at QP 0, where there is no finer.
        let probe = || {
            let point = |probe_qp| {
                curve_point(
                    SourcePlanes {
                        y,
                        cb,
                        cr,
                        width,
                        height,
                    },
                    probe_qp,
                    search,
                    filter.deblocking(),
                )
            };
            if qp > 0 {
                (point(qp - 1), base)
            } else {
                (base, point(qp + 1))
            }
        };
        if keeps_sao(gain, sao_bits, qp, probe) {
            slice_data = coded;
            sao_kept = true;
        } else {
            recon = deblocked;
        }
    }

    // ---- the bitstream pass: code the recorded decisions ----
    let mut w = BitWriter::new();
    // ---- slice_segment_header() ----
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(2); // slice_type = I
    if filter.sao() {
        // §7.3.6.1, present because the SPS carries
        // sample_adaptive_offset_enabled_flag == 1. Both passes are decided
        // together, so the pair is 1/1 when the picture kept SAO and 0/0 when
        // it did not — and 0/0 suppresses every CTB's sao( ) structure.
        let on = u8::from(sao_kept);
        w.put_bit(on); // slice_sao_luma_flag
        w.put_bit(on); // slice_sao_chroma_flag (ChromaArrayType != 0)
    }
    w.se(qp - 26); // slice_qp_delta over init_qp_minus26 == 0
    if filter.deblocking() || sao_kept {
        // §7.3.6.1: present because pps_loop_filter_across_slices_enabled_flag
        // is 1 and at least one in-loop filter runs on this slice. One slice
        // fills the picture, so the value only has to be legal, not
        // restrictive.
        w.put_bit(1); // slice_loop_filter_across_slices_enabled_flag
    }
    w.rbsp_trailing_bits(); // byte_alignment() before slice data

    let mut rbsp = w.finish();
    rbsp.extend_from_slice(&slice_data);
    let modes = records.iter().map(|record| record.mode).collect();
    (rbsp, recon, modes)
}

/// `cMax` of the Table 9-43 truncated-Rice `sao_offset_abs` binarization —
/// the same §7.4.9.3 bound [`estimate_sao`] clamps its offsets to, so the
/// writer can never be handed a magnitude the binarization cannot carry.
const SAO_OFFSET_ABS_CMAX: u32 = SAO_OFFSET_MAX as u32;

/// Which Table 9-48 context a context-coded §7.3.8.3 bin uses.
#[derive(Clone, Copy)]
enum SaoCtx {
    /// `sao_merge_left_flag` / `sao_merge_up_flag` (Table 9-5, one context).
    MergeFlag,
    /// `sao_type_idx_luma` / `sao_type_idx_chroma` bin 0 (Table 9-6, shared).
    TypeIdx,
}

/// The coder [`code_sao`] puts the bins it derives into.
///
/// There is only one: the §7.3.8.3 structure is walked for the bitstream, and
/// what it costs is read back off the coded slice by [`slice_data_bits`]
/// rather than counted alongside it, so a cost model and the writer have no
/// way to disagree in the first place.
struct SaoWriter<'a> {
    writer: &'a mut BitWriter,
    cabac: &'a mut CabacEncoder,
    contexts: &'a mut SliceContexts,
}

impl SaoWriter<'_> {
    /// One context-coded bin.
    fn ctx(&mut self, ctx: SaoCtx, bin: u8) {
        let model = match ctx {
            SaoCtx::MergeFlag => &mut self.contexts.sao_merge_flag[0],
            SaoCtx::TypeIdx => &mut self.contexts.sao_type_idx[0],
        };
        self.cabac.encode_decision(self.writer, model, bin);
    }

    /// One bypass bin.
    fn bypass(&mut self, bin: u8) {
        self.cabac.encode_bypass(self.writer, bin);
    }
}

/// §7.3.8.3 `sao( rx, ry )` for one CTB, derived from the resolved grid the
/// §8.7.3 pass filtered the reconstruction with.
///
/// A CTB whose resolved parameters are exactly its left (or above)
/// neighbour's codes one merge flag in place of the whole structure. Merging
/// is taken only on exact equality of all three components, so what a decoder
/// resolves out of the bitstream is the grid the encoder filtered with, CTB
/// for CTB — and that covers the common case of two neighbouring CTBs which
/// both left SAO off, which is the whole of what an unfiltered CTB pays.
fn code_sao(sink: &mut SaoWriter<'_>, grid: &[ResolvedSao], addr: usize, ctbs_x: usize) {
    let (rx, ry) = (addr % ctbs_x, addr / ctbs_x);
    let here = grid[addr];
    // The §7.3.8.3 presence conditions: one slice and one tile fill the
    // picture, so a neighbour inside the picture is always a legal merge
    // source.
    let left = (rx > 0).then(|| grid[addr - 1]);
    let above = (ry > 0).then(|| grid[addr - ctbs_x]);
    let merge_left = left == Some(here);
    if left.is_some() {
        sink.ctx(SaoCtx::MergeFlag, u8::from(merge_left));
    }
    let merge_up = !merge_left && above == Some(here);
    if above.is_some() && !merge_left {
        sink.ctx(SaoCtx::MergeFlag, u8::from(merge_up));
    }
    if merge_left || merge_up {
        return;
    }

    for c_idx in 0..3 {
        let component = here.components[c_idx];
        if c_idx < 2 {
            // sao_type_idx_luma / sao_type_idx_chroma, Table 9-43 TR
            // (cMax 2): bin 0 context-coded, bin 1 bypass.
            let applied = u8::from(component.sao_type_idx != 0);
            sink.ctx(SaoCtx::TypeIdx, applied);
            if applied == 1 {
                sink.bypass(u8::from(component.sao_type_idx == 2));
            }
        } else {
            // §7.4.9.3 infers SaoTypeIdx[2] from cIdx 1, so the estimation is
            // not free to give Cr a type of its own.
            debug_assert_eq!(component.sao_type_idx, here.components[1].sao_type_idx);
        }
        if component.sao_type_idx == 0 {
            continue;
        }
        for offset in &component.offset_val[1..5] {
            code_sao_offset_abs(sink, offset.unsigned_abs());
        }
        if component.sao_type_idx == 1 {
            // Band offset: the sign of every nonzero offset, then the band.
            for offset in &component.offset_val[1..5] {
                if *offset != 0 {
                    sink.bypass(u8::from(*offset < 0));
                }
            }
            // sao_band_position, Table 9-43 FL (cMax 31): five bypass bins,
            // MSB first.
            for shift in (0..5).rev() {
                sink.bypass((component.band_position >> shift) & 1);
            }
        } else if c_idx < 2 {
            // sao_eo_class_luma / sao_eo_class_chroma, Table 9-43 FL
            // (cMax 3): two bypass bins, MSB first.
            for shift in (0..2).rev() {
                sink.bypass((component.eo_class >> shift) & 1);
            }
        } else {
            debug_assert_eq!(component.eo_class, here.components[1].eo_class);
        }
    }
}

/// `sao_offset_abs`, Table 9-43 TR with `cMax == 7` and `cRiceParam == 0` —
/// truncated unary, every bin bypass (Table 9-48).
fn code_sao_offset_abs(sink: &mut SaoWriter<'_>, value: u32) {
    let value = value.min(SAO_OFFSET_ABS_CMAX);
    for _ in 0..value {
        sink.bypass(1);
    }
    if value < SAO_OFFSET_ABS_CMAX {
        sink.bypass(0);
    }
}

/// §7.3.8.5 + §7.3.8.8 + §7.3.8.11 — one CTB's coding unit, from the recorded
/// decisions the decision pass left for it.
///
/// Shared by the bitstream pass and by [`slice_data_bits`], so the rate a
/// decision is measured against is the rate the same coder would emit.
fn code_coding_unit(
    w: &mut BitWriter,
    cabac: &mut CabacEncoder,
    ctxs: &mut SliceContexts,
    record: &CtbRecord,
) {
    // ---- coding_unit(): PART_2Nx2N, pcm_flag == 0 ----
    cabac.encode_decision(w, &mut ctxs.part_mode[0], 1);
    cabac.encode_terminate(w, 0); // pcm_flag = 0
    write_luma_intra_mode(w, cabac, ctxs, record.mode, record.candidates);

    // §9.3.3.8 / Table 9-46: value 4 (chroma derived from luma) is the
    // single context-coded 0 bin; 0..=3 is a 1 bin plus two FL bypass bins.
    if record.chroma_mode == CHROMA_MODE_DERIVED {
        cabac.encode_decision(w, &mut ctxs.intra_chroma_pred_mode[0], 0);
    } else {
        cabac.encode_decision(w, &mut ctxs.intra_chroma_pred_mode[0], 1);
        cabac.encode_bypass(w, (record.chroma_mode >> 1) & 1);
        cabac.encode_bypass(w, record.chroma_mode & 1);
    }

    // ---- transform_tree(): split_transform_flag is absent because
    // max_transform_hierarchy_depth_intra == 0, so MaxTrafoDepth == 0
    // and the flag is inferred 0 (one 16x16 luma TB, two 8x8 chroma).
    let [luma, chroma_cb, chroma_cr] = &record.levels;

    // §7.3.8.8 order: cbf_cb, cbf_cr (ctxInc = trafoDepth = 0), then
    // cbf_luma (ctxInc = 1 at trafoDepth 0).
    let cbf_cb = u8::from(has_coded_levels(chroma_cb));
    let cbf_cr = u8::from(has_coded_levels(chroma_cr));
    let cbf_luma = u8::from(has_coded_levels(luma));
    cabac.encode_decision(w, &mut ctxs.cbf_chroma[0], cbf_cb);
    cabac.encode_decision(w, &mut ctxs.cbf_chroma[0], cbf_cr);
    cabac.encode_decision(w, &mut ctxs.cbf_luma[1], cbf_luma);

    // ---- transform_unit(): the coded blocks, luma then Cb then Cr.
    for (levels, log2, is_chroma) in [
        (luma, CTB_LOG2, false),
        (chroma_cb, CTB_LOG2 - 1, true),
        (chroma_cr, CTB_LOG2 - 1, true),
    ] {
        if !has_coded_levels(levels) {
            continue;
        }
        write_residual_coding(
            &mut EngineResidualBinSink {
                writer: w,
                cabac,
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
}

/// §7.3.8.1 `slice_segment_data( )`: every recorded coding unit, each
/// optionally behind the §7.3.8.3 `sao( )` structure of its CTB.
///
/// The slice header ends on `byte_alignment( )` and §9.3.2.2 initializes the
/// CABAC engine from `SliceQpY` and the init type alone, so this stands on
/// its own: what it returns is appended to the header verbatim. It is also
/// what every rate in the SAO decision is measured on — the two curve points
/// [`calibrated_sao_lambda_q8`] interpolates between, and the grid's own cost
/// as the difference between coding it and not — so all of them are the
/// arithmetic coder's own output rather than a bin count standing in for it.
/// A bin count would suit neither: the residual's bins are context-coded and
/// far cheaper than a bit each, while the `sao( )` structure's are nearly all
/// bypass, so a slope measured in bins and spent in bins compares two
/// different currencies.
fn code_slice_data(
    records: &[CtbRecord],
    sao: Option<&[ResolvedSao]>,
    qp: i32,
    ctbs_x: usize,
) -> Vec<u8> {
    let mut w = BitWriter::new();
    let mut ctxs = SliceContexts::init(init_type(2, false), qp);
    let mut cabac = CabacEncoder::new();
    for (addr, record) in records.iter().enumerate() {
        if let Some(grid) = sao {
            code_sao(
                &mut SaoWriter {
                    writer: &mut w,
                    cabac: &mut cabac,
                    contexts: &mut ctxs,
                },
                grid,
                addr,
                ctbs_x,
            );
        }
        code_coding_unit(&mut w, &mut cabac, &mut ctxs, record);
        cabac.encode_terminate(&mut w, u8::from(addr == records.len() - 1));
    }
    w.align_zero();
    w.finish()
}

/// One point on this picture's own rate-distortion curve: the whole-picture
/// squared error of the deblocked reconstruction, and the bins the slice's
/// coding units cost, both at one QP.
#[derive(Clone, Copy)]
struct CurvePoint {
    /// Whole-picture SSE over all three planes, after §8.7.2 and before
    /// §8.7.3 — the distortion the SAO decision is about to move.
    sse: u64,
    /// What the slice's coding units cost the CABAC coder, SAO syntax
    /// excluded — [`slice_data_bits`] at this point's own QP, so two points'
    /// rates are subtractable.
    bits: u64,
}

/// The curve point this writer lands on at `qp`: run the decision pass, run
/// the in-loop filter the base point carries, and measure both coordinates.
///
/// This is the probe [`calibrated_sao_lambda_q8`] takes its slope against. It
/// is the decision pass and the deblocking filter only — no bitstream pass
/// and no §8.7.3 — so it is not a second encode of the picture, and the
/// recursion a probe of a probe would be cannot arise.
fn curve_point(src: SourcePlanes<'_>, qp: i32, search: ModeSearch, deblocking: bool) -> CurvePoint {
    let (records, mut recon) =
        run_decision_pass(src.y, src.cb, src.cr, src.width, src.height, qp, search);
    if deblocking {
        deblock_reconstruction(&mut recon, qp);
    }
    CurvePoint {
        sse: picture_sse(&recon, src.y, src.cb, src.cr),
        bits: code_slice_data(&records, None, qp, src.width / CTB).len() as u64 * 8,
    }
}

/// The slice-level half of the SAO decision: whether a grid that removed
/// `gain` of squared error is worth the `sao_bits` it would be coded with.
///
/// ## What is being decided
///
/// Not `D + lambda * R` against a price, whatever the shape of the arithmetic
/// suggests. The decision is a comparison of two rate-distortion points at
/// the *same* rate: SAO removed `gain` for `sao_bits`, and
/// [`sao_threshold_sse`] says what those same bits would have removed spent
/// on a finer quantizer instead, read off this picture's own curve through
/// the two points `probe` measures. SAO is kept when it is the better of the
/// two, which is exactly the property
/// `every_accepted_sao_point_sits_on_the_writers_own_curve` measures.
///
/// It is stated as a threshold in squared error rather than as a multiplier
/// on the rate because only the threshold form is monotone. The multiplier
/// [`calibrated_sao_lambda_q8`] returns is the threshold divided by the rate
/// being judged, and the threshold is concave, so the multiplier falls as the
/// rate rises and a smaller grid appears to be charged a higher price per
/// bit. #287 read that as the rule not being monotone in the grid: at noise
/// 64x48 QP 12 a full grid gaining 75 for 104 bits was declined at a
/// threshold of 77, while the subset grid gaining 54 for 72 bits was accepted
/// at a threshold of about 54.
///
/// It is not. Those two decisions are consistent, and the threshold form is
/// what makes that visible: the extra 32 bits raised the threshold by 23 and
/// bought 21, so the extra components did *not* pay for themselves against
/// the curve — they paid only against the constant multiplier the per-CTB
/// search prices them with, which is a different instrument measuring a
/// different trade. What a superset grid has to beat is the curve's marginal
/// distortion at the rate being judged, not its average out to that rate, and
/// a superset whose extra components clear that margin is accepted wherever
/// its subset is, because [`sao_threshold_sse`] is nondecreasing in
/// `sao_bits`. `the_slice_level_threshold_is_monotone_in_the_grid_it_judges`
/// asserts that, and it is the only monotonicity the rule can have: the two
/// grids sit at different rates, so no rule that compares a picture against
/// its own curve can accept both a grid and a costlier one that buys less per
/// bit than the curve does there.
///
/// ## What is not being decided
///
/// The comparison is between one measured quantity and one extrapolated one,
/// and [`SAO_ACCEPTANCE_PRECISION_NUM`] is how far the extrapolated one can
/// be wrong. A gain inside that of the threshold does not resolve the
/// comparison in either direction, so it is not taken as clearing it. This is
/// what stops the writer accepting a slice on a margin of a few parts in a
/// thousand of a quantity it can only place to a few parts in a hundred.
///
/// ## When the probe runs
///
/// `probe` is a second decision pass, so it is called only when its answer
/// can change the outcome. The threshold is never outside the band
/// [`SAO_LAMBDA_BAND`] draws either side of the closed-form line, so a gain
/// clearing the band's coarse end already clears whatever the probe would
/// have returned, and one that misses its fine end already misses it. Over
/// the QP 12-51 sweep on the two test pictures that settles half the pictures
/// without a probe, where SAO either found nothing the quantizer had left
/// behind or found far more than a step of it recovers.
fn keeps_sao(
    gain: u64,
    sao_bits: u64,
    qp: i32,
    probe: impl FnOnce() -> (CurvePoint, CurvePoint),
) -> bool {
    if sao_bits == 0 {
        return gain > 0;
    }
    // The band's two ends bound the threshold, so they settle every decision
    // that falls outside them without the probe. They are compared against
    // bare, without the precision margin: what the margin is about is the
    // extrapolation, and neither end of the band is extrapolated.
    let clears = |lambda: u64| gain > sao_bits.saturating_mul(lambda) / 256;
    let fixed = u64::from(lambda_q8(qp));
    if clears(fixed.saturating_mul(SAO_LAMBDA_BAND)) {
        return true;
    }
    if !clears((fixed / SAO_LAMBDA_BAND).max(1)) {
        return false;
    }
    let (fine, coarse) = probe();
    clears_sao_threshold(gain, sao_threshold_sse(fine, coarse, qp, sao_bits))
}

/// Whether `gain` clears `threshold` by more than the acceptance model's own
/// resolution — see [`SAO_ACCEPTANCE_PRECISION_NUM`] for why clearing it at
/// all is not enough.
fn clears_sao_threshold(gain: u64, threshold: f64) -> bool {
    let margin = threshold * SAO_ACCEPTANCE_PRECISION_NUM as f64
        / SAO_ACCEPTANCE_PRECISION_DEN as f64;
    gain as f64 > threshold + margin
}

/// The multiplier the slice-level SAO decision trades distortion against rate
/// with, in the Q8 units [`lambda_q8`] returns: what `sao_bits` more of this
/// picture's own rate-distortion curve is worth, per bit.
///
/// `lambda_q8` is `0.57 * 2 ^ ( ( QP - 12 ) / 3 )`, a closed-form function of
/// the quantizer alone. That is the right instrument for choosing between two
/// codings of the same block at the same QP, where the picture cancels out of
/// the comparison. It is the wrong one here, because this decision is not
/// between two codings of a block but between spending bits on SAO and
/// spending the same bits on a finer quantizer — and what those bits buy that
/// way is a property of the content, not of the QP. Over the QP 12-51 sweep
/// on the two test pictures the closed-form value misses the picture's own
/// slope by 0.4x to 2.5x, in neither direction consistently, which is how
/// #268's decision came to accept two points that sat below the curve.
///
/// So it is read off the curve. `fine` and `coarse` are two points on it —
/// the writer's own coded point and [`curve_point`] one QP away — and the
/// curve between them is taken as a straight line in log-rate against
/// log-distortion, which is the interpolation the acceptance measurement uses
/// and is very nearly straight over a single quantizer step. The distortion
/// that line reaches `sao_bits` above the coded point is what SAO has to
/// beat; what this returns is the per-bit form of it, so the caller's test
/// stays the `D + lambda * R` one.
///
/// Interpolating rather than taking the secant's own slope is what makes the
/// difference: SAO's rate is a fraction of a quantizer step's, the secant is
/// the average slope over the whole step, and a curve this convex is steeper
/// than its average at the near end. Charging SAO the average leaves points
/// just under the curve — measured, four of them over the sweep, the worst
/// 0.08 dB below.
fn calibrated_sao_lambda_q8(fine: CurvePoint, coarse: CurvePoint, qp: i32, sao_bits: u64) -> u32 {
    let fixed = lambda_q8(qp);
    let Some(reachable) = reachable_sse(fine, coarse, qp, sao_bits) else {
        return fixed;
    };
    let fixed = u64::from(fixed);
    let measured = (reachable * 256.0 / sao_bits as f64).round().max(0.0) as u64;
    measured.clamp((fixed / SAO_LAMBDA_BAND).max(1), fixed * SAO_LAMBDA_BAND) as u32
}

/// The squared error this picture's own rate-distortion curve reaches
/// `sao_bits` above the coded point — what those bits would remove if they
/// were spent on a finer quantizer instead of on SAO.
///
/// This is the quantity the slice-level decision is actually about, and
/// [`sao_threshold_sse`] is the decision. `fine` and `coarse` are the two
/// measured points; the curve between them is taken as a straight line in
/// log-rate against log-distortion, which is very nearly straight over a
/// single quantizer step and is the same interpolation
/// [`super::tests::sao_curve_offsets`] measures the accepted points against.
///
/// Returns `None` where the two points describe no curve — the same rate, the
/// same distortion, both moving the same way, or an empty slice — leaving the
/// caller to fall back to the closed form.
fn reachable_sse(fine: CurvePoint, coarse: CurvePoint, qp: i32, sao_bits: u64) -> Option<f64> {
    // The coded point is `coarse` at every QP but 0, where there is no finer
    // point to probe and the writer's own point is `fine` instead. Either way
    // the interpolation runs from the coded point towards the other one.
    let (coded, other) = if qp > 0 {
        (coarse, fine)
    } else {
        (fine, coarse)
    };
    if coded.sse == 0 || coded.bits == 0 || other.sse == 0 || sao_bits == 0 {
        return None;
    }
    let rate_ratio = other.bits as f64 / coded.bits as f64;
    let sse_ratio = other.sse as f64 / coded.sse as f64;
    // A probe that coded the same bits as the writer, or that did not move
    // the reconstruction the way a quantizer step must, describes no curve.
    if (rate_ratio - 1.0).abs() < 1e-9 || (sse_ratio - 1.0).abs() < 1e-9 {
        return None;
    }
    if (rate_ratio > 1.0) != (sse_ratio < 1.0) {
        return None;
    }
    let t = (1.0 + sao_bits as f64 / coded.bits as f64).ln() / rate_ratio.ln();
    let reachable = coded.sse as f64 * (1.0 - sse_ratio.powf(t));
    (reachable.is_finite() && reachable > 0.0).then_some(reachable)
}

/// How far a slice-level SAO decision has to clear the curve before the
/// writer is entitled to call it a decision, as a fraction of the distortion
/// [`reachable_sse`] predicts.
///
/// The acceptance test compares two estimates of the same quantity — the
/// squared error SAO removed, measured exactly, against the squared error a
/// finer quantizer would remove, *extrapolated* from a two-point probe to a
/// rate a fraction of a quantizer step above the coded one. Only the first is
/// measured. `measure_sao_acceptance_precision` quantifies the second against
/// the ladder [`super::tests::sao_curve_offsets`] interpolates, which is this
/// same writer coded at every QP from 0 to 51, and the extrapolation's error
/// there reaches 3.7% of the distortion it predicts over the QP 12-51 sweep
/// on both test pictures at both sizes.
///
/// So a gain that clears the threshold by less than that is not measured to
/// be above the curve; it is inside the model's own resolution, and #287
/// found the writer taking exactly such a decision — accepting a slice at
/// noise 128x96 QP 32 by 0.25% of its own gain, which
/// `every_accepted_sao_point_sits_on_the_writers_own_curve` then measured
/// 0.003 dB *below* the curve. Requiring the margin to exceed the model's
/// resolution is what turns that from a decision taken into a decision
/// recognised as unresolvable, and an unresolvable decision does not spend
/// the bits.
///
/// The bound is the measured 3.7% rounded up to a figure the measurement
/// supports rather than one fitted to a single point; see
/// [`SaoLambda::band_q8`] for what it lets the band-syntax charge do.
const SAO_ACCEPTANCE_PRECISION_NUM: u64 = 1;
/// Denominator of [`SAO_ACCEPTANCE_PRECISION_NUM`] — 1/25 is 4%.
const SAO_ACCEPTANCE_PRECISION_DEN: u64 = 25;

/// The squared error the slice-level SAO decision has to beat, in the picture
/// SSE units `gain` is measured in.
///
/// This is [`reachable_sse`] clamped to the same trust band
/// [`calibrated_sao_lambda_q8`] clamps the per-bit form to, expressed as the
/// two rate lines the band's ends draw rather than as multipliers. Stating it
/// this way is what makes it a threshold rather than a price: `reachable_sse`
/// is increasing in `sao_bits` and both lines are increasing in `sao_bits`,
/// so a clamp between them is increasing too, and a grid can only ever be
/// asked for more distortion by asking for more bits.
///
/// The per-bit form is not, and cannot be. The curve is convex, so the
/// distortion reachable by `sao_bits` grows slower than `sao_bits` does, and
/// dividing an increasing concave threshold by the rate gives a *decreasing*
/// price — which is the sense in which a smaller grid is "charged more per
/// bit". That is what a concave threshold looks like in per-bit units and not
/// a defect in it; see [`keeps_sao`] for what it does and does not imply
/// about supersets.
fn sao_threshold_sse(fine: CurvePoint, coarse: CurvePoint, qp: i32, sao_bits: u64) -> f64 {
    let fixed = u64::from(lambda_q8(qp));
    let line = |lambda: u64| (sao_bits.saturating_mul(lambda) / 256) as f64;
    let (lo, hi) = (
        line((fixed / SAO_LAMBDA_BAND).max(1)),
        line(fixed.saturating_mul(SAO_LAMBDA_BAND)),
    );
    reachable_sse(fine, coarse, qp, sao_bits)
        .unwrap_or_else(|| line(fixed))
        .clamp(lo, hi)
}

/// The sum of squared errors between a reconstruction and the source it was
/// coded from, over all three planes.
fn picture_sse(recon: &ReconstructedPicture, y: &[u8], cb: &[u8], cr: &[u8]) -> u64 {
    let plane = |a: &[u8], b: &[u8]| -> u64 {
        a.iter()
            .zip(b)
            .map(|(&p, &q)| {
                let d = i64::from(p) - i64::from(q);
                (d * d) as u64
            })
            .sum()
    };
    plane(&recon.y, y) + plane(&recon.cb, cb) + plane(&recon.cr, cr)
}

/// Pick the luma intra mode for one coding unit and return it with the block
/// it codes to.
///
/// Two passes. [`shortlist_intra_luma_modes`] ranks all 35 Table 8-1 modes on
/// the prediction's SATD, which costs no transform; the shortlist is then
/// re-scored on what each mode actually costs after quantization: the
/// reconstruction's squared error against the source, traded against the rate
/// the operating point charges. At [`ModeSearch::Rdo`] that rate is the mode's
/// own §7.3.8.5 signalling only — this writer has no bitrate target, so its
/// job at a given QP is the closest picture, and a better prediction shrinks
/// the residual on its own. At [`ModeSearch::RateDistortion`] the residual's
/// own estimated bins join it, which is the cost a rate-controlled encoder
/// has to minimize. The winner's `CodedBlock` is the one the caller commits,
/// so the mode that is coded is the mode that was measured.
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
        let bits = u64::from(intra_mode_bit_cost(candidate.mode, candidates))
            + residual_rate(search, &coded.levels, CTB_LOG2, false);
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
/// are scored on the pair's joint squared error, against the same rate term
/// [`decide_luma_mode`] uses at this operating point — the mode signalling
/// alone, or that plus both blocks' residual bins. Value 4 (`IntraPredModeC == IntraPredModeY`) is one of them, so this
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
        let bits = u64::from(chroma_mode_bit_cost(signalled))
            + coded
                .iter()
                .map(|block| residual_rate(search, &block.levels, CTB_LOG2 - 1, true))
                .sum::<u64>();
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

/// The rate this operating point charges a candidate for its own residual: the
/// bins §7.3.8.11 `residual_coding( )` would emit for `levels`, or nothing at
/// the fixed-QP operating point that does not model residual rate at all.
///
/// The scan is the one [`write_idr_residual_slice`] codes with — §7.4.9.11's
/// mode-dependent scans need a smaller block than either of the two sizes here.
fn residual_rate(search: ModeSearch, levels: &[i32], log2_trafo_size: u32, is_chroma: bool) -> u64 {
    if !search.charges_residual_rate() {
        return 0;
    }
    u64::from(residual_rate_bits(
        levels,
        &ResidualWriteParams {
            log2_trafo_size,
            is_chroma,
            scan_idx: ScanIdx::Diagonal,
        },
    ))
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
        //
        // Both pictures, because the two in-loop filters engage on opposite
        // content: the smooth one is where §8.7.3 actually codes offsets at a
        // fine QP, and a `sao( )` structure the decoder resolves differently
        // than the encoder filtered with would show up here and nowhere else.
        let (width, height) = (64, 48);
        for (name, (y, cb, cr)) in [
            ("noise", picture(width, height)),
            ("smooth", smooth_picture(width, height)),
        ] {
            for qp in [0i32, 12, 26, 37, 51] {
                let (au, recon) = encode_idr_residual_au(&y, &cb, &cr, width, height, qp).unwrap();
                let (dy, dcb, dcr) = decode(&au, width, height);
                assert_eq!(dy, recon.y, "{name} qp {qp}: luma diverged");
                assert_eq!(dcb, recon.cb, "{name} qp {qp}: Cb diverged");
                assert_eq!(dcr, recon.cr, "{name} qp {qp}: Cr diverged");
            }
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
        let (rbsp, recon, _) = write_idr_residual_slice(
            y,
            cb,
            cr,
            width,
            height,
            qp,
            ModeSearch::DcOnly,
            LoopFilter::Deblock,
        );
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
        let (_, _, modes) = write_idr_residual_slice(
            &y,
            &cb,
            &cr,
            width,
            height,
            26,
            ModeSearch::Rdo,
            LoopFilter::Deblock,
        );
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
            let (searched_slice, searched, _) = write_idr_residual_slice(
                &y,
                &cb,
                &cr,
                width,
                height,
                qp,
                ModeSearch::Rdo,
                LoopFilter::Deblock,
            );
            let (baseline_slice, baseline, _) = write_idr_residual_slice(
                &y,
                &cb,
                &cr,
                width,
                height,
                qp,
                ModeSearch::DcOnly,
                LoopFilter::Deblock,
            );
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

    /// The rate-constrained operating point has to buy its bits back: at the
    /// same QP it must code a smaller slice than the fixed-QP decision, and the
    /// point it lands on must sit above the fixed-QP writer's own
    /// rate-distortion curve rather than merely further down it.
    ///
    /// The curve is interpolated in log-rate against PSNR between the two
    /// fixed-QP points that bracket the rate-constrained slice's size, which is
    /// the Bjontegaard construction reduced to the one point being tested.
    #[test]
    fn the_rate_distortion_cost_beats_the_fixed_qp_decision_at_equal_rate() {
        let (width, height) = (64, 48);
        let (y, cb, cr) = picture(width, height);
        let full = |recon: &ReconstructedPicture| {
            psnr_db(
                &[y.clone(), cb.clone(), cr.clone()].concat(),
                &[recon.y.clone(), recon.cb.clone(), recon.cr.clone()].concat(),
            )
        };
        let encode = |qp: i32, search: ModeSearch| {
            let (slice, recon, _) = write_idr_residual_slice(
                &y,
                &cb,
                &cr,
                width,
                height,
                qp,
                search,
                LoopFilter::Deblock,
            );
            (slice.len() as f64, full(&recon))
        };

        // The fixed-QP writer's curve, finest first, to interpolate against.
        let ladder: Vec<(f64, f64)> = [8i32, 12, 18, 22, 26, 32, 37, 42, 47]
            .iter()
            .map(|&qp| encode(qp, ModeSearch::Rdo))
            .collect();

        for qp in [12i32, 18, 26, 32, 37] {
            let (bytes, psnr) = encode(qp, ModeSearch::RateDistortion);
            let (fixed_bytes, _) = encode(qp, ModeSearch::Rdo);
            assert!(
                bytes < fixed_bytes,
                "qp {qp}: charging the residual's rate did not shrink the slice: {bytes} against \
                 {fixed_bytes} bytes"
            );
            // Where the fixed-QP curve is at this slice size.
            let upper = ladder
                .windows(2)
                .find(|pair| pair[1].0 <= bytes && bytes <= pair[0].0)
                .unwrap_or_else(|| panic!("qp {qp}: {bytes} bytes is off the measured ladder"));
            let (bigger, smaller) = (upper[0], upper[1]);
            let t = (bigger.0.ln() - bytes.ln()) / (bigger.0.ln() - smaller.0.ln());
            let interpolated = bigger.1 + t * (smaller.1 - bigger.1);
            assert!(
                psnr > interpolated,
                "qp {qp}: the rate-constrained point ({bytes} bytes, {psnr:.3} dB) is below the \
                 fixed-QP curve's {interpolated:.3} dB at the same rate"
            );
        }
    }

    /// Charging the residual's rate must not cost the fixed-QP operating point
    /// anything: the public writer still takes the closest-picture decision.
    #[test]
    fn the_fixed_qp_operating_point_is_still_the_one_the_public_writer_codes() {
        let (width, height) = (64, 48);
        let (y, cb, cr) = picture(width, height);
        for qp in [12i32, 26, 37] {
            let (au, _) = encode_idr_residual_au(&y, &cb, &cr, width, height, qp).unwrap();
            let (rbsp, _, _) = write_idr_residual_slice(
                &y,
                &cb,
                &cr,
                width,
                height,
                qp,
                ModeSearch::Rdo,
                LoopFilter::DeblockSao,
            );
            assert!(
                au.windows(rbsp.len()).any(|window| window == rbsp),
                "qp {qp}: the public entry point no longer codes the fixed-QP slice"
            );
        }
    }

    /// The rate-constrained writer is still a conforming encoder: whatever its
    /// decision gave up in distortion, the decoder's picture and the encoder's
    /// reference still agree sample for sample.
    #[test]
    fn the_rate_constrained_stream_decodes_to_the_encoders_own_reconstruction() {
        let (width, height) = (64, 48);
        let (y, cb, cr) = picture(width, height);
        for qp in [0i32, 12, 26, 37, 51] {
            let (au, recon) =
                encode_idr_residual_au_rate_constrained(&y, &cb, &cr, width, height, qp).unwrap();
            let (dy, dcb, dcr) = decode(&au, width, height);
            assert_eq!(dy, recon.y, "qp {qp}: luma diverged");
            assert_eq!(dcb, recon.cb, "qp {qp}: Cb diverged");
            assert_eq!(dcr, recon.cr, "qp {qp}: Cr diverged");
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
                &y,
                &cb,
                &cr,
                width,
                height,
                qp,
                ModeSearch::Rdo,
                LoopFilter::Deblock,
            );
            let (_, unfiltered, _) = write_idr_residual_slice(
                &y,
                &cb,
                &cr,
                width,
                height,
                qp,
                ModeSearch::Rdo,
                LoopFilter::Off,
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
                &y,
                &cb,
                &cr,
                width,
                height,
                qp,
                ModeSearch::Rdo,
                LoopFilter::Deblock,
            );
            let (_, unfiltered, _) = write_idr_residual_slice(
                &y,
                &cb,
                &cr,
                width,
                height,
                qp,
                ModeSearch::Rdo,
                LoopFilter::Off,
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

    /// Whole-picture PSNR and slice size at one QP with SAO on and off, from
    /// the same coding loop and the same mode decisions — the measurement the
    /// module documentation and the changelog record.
    fn sao_on_off(
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        width: usize,
        height: usize,
        qp: i32,
    ) -> ((usize, f64), (usize, f64)) {
        let source = [y.to_vec(), cb.to_vec(), cr.to_vec()].concat();
        let encode = |filter| {
            let (slice, recon, _) =
                write_idr_residual_slice(y, cb, cr, width, height, qp, ModeSearch::Rdo, filter);
            let psnr = psnr_db(
                &source,
                &[recon.y.clone(), recon.cb.clone(), recon.cr.clone()].concat(),
            );
            (slice.len(), psnr)
        };
        (encode(LoopFilter::Deblock), encode(LoopFilter::DeblockSao))
    }

    /// The §7.3.8.3 grid the writer would code for one picture at one QP —
    /// the same [`sao_reconstruction`] call the coding loop makes, on the same
    /// deblocked reconstruction and with the same lambda, so what it returns
    /// is what the slice carries.
    fn sao_grid(
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        width: usize,
        height: usize,
        qp: i32,
    ) -> Vec<ResolvedSao> {
        let (_, mut deblocked, _) = write_idr_residual_slice(
            y,
            cb,
            cr,
            width,
            height,
            qp,
            ModeSearch::Rdo,
            LoopFilter::Deblock,
        );
        sao_reconstruction(
            &mut deblocked,
            SourcePlanes {
                y,
                cb,
                cr,
                width,
                height,
            },
            SaoLambda::for_search(lambda_q8(qp)),
        )
    }

    #[test]
    fn the_search_picks_band_offset_where_it_beats_every_edge_class() {
        // The band path of §7.3.8.3 — four `sao_offset_sign` bins and a
        // five-bin `sao_band_position` — is only ever written when the search
        // picks `SaoTypeIdx == 1`, so pin a picture and a QP where it does.
        // The noise-carrying picture at a fine QP is the case: its error is
        // spread over a value range rather than shaped around a local edge,
        // which is exactly what edge offset cannot reach.
        let (width, height) = (64, 48);
        let (y, cb, cr) = picture(width, height);
        let grid = sao_grid(&y, &cb, &cr, width, height, 12);
        let band = grid
            .iter()
            .flat_map(|cell| cell.components.iter())
            .filter(|c| c.sao_type_idx == 1)
            .count();
        assert!(
            band > 0,
            "the search never chose band offset, so the writer's band path is still dead code"
        );

        // And it is the writer's own decision, not just the search's: the
        // slice-level test kept the pass, so those `sao( )` structures reached
        // the bitstream and a decoder resolved them back into the same
        // picture.
        let ((off_bytes, off_psnr), (on_bytes, on_psnr)) =
            sao_on_off(&y, &cb, &cr, width, height, 12);
        assert!(
            on_bytes > off_bytes && on_psnr > off_psnr,
            "the slice-level test declined the pass band offset was chosen in: \
             {off_bytes} -> {on_bytes} bytes, {off_psnr:.3} -> {on_psnr:.3} dB"
        );
        let (au, recon) = encode_idr_residual_au(&y, &cb, &cr, width, height, 12).unwrap();
        let (dy, dcb, dcr) = decode(&au, width, height);
        assert_eq!(dy, recon.y, "luma diverged over a band-offset slice");
        assert_eq!(dcb, recon.cb, "Cb diverged over a band-offset slice");
        assert_eq!(dcr, recon.cr, "Cr diverged over a band-offset slice");
    }

    #[test]
    fn band_offset_earns_its_rate_where_the_search_takes_it() {
        // The measurement the changelog records: band offset is worth more
        // than the four signs and five position bins it costs beyond edge
        // offset. Both numbers come from the same coding loop at the same QP,
        // so the only difference is which types the search was allowed to
        // consider — here, that the noise picture at QP 12 gains 0.10 dB of
        // whole-picture PSNR for half a percent of slice.
        let (width, height) = (64, 48);
        let (y, cb, cr) = picture(width, height);
        let (_, (on_bytes, on_psnr)) = sao_on_off(&y, &cb, &cr, width, height, 12);
        assert!(
            on_psnr > 51.10,
            "the noise picture at QP 12 reconstructed at {on_psnr:.3} dB, below what band offset \
             was measured to buy"
        );
        assert!(
            on_bytes < 1960,
            "the band-offset slice cost {on_bytes} bytes, above what it was measured to cost"
        );
    }

    /// Where every SAO point the writer accepts sits relative to the SAO-off
    /// writer's own rate-distortion curve, interpolated in log-rate at the
    /// same slice size — the same construction
    /// [`the_rate_distortion_cost_beats_the_fixed_qp_decision_at_equal_rate`]
    /// uses, applied to the other decision.
    ///
    /// Returns one entry per QP at which the slice-level test accepted SAO:
    /// the QP, the accepted slice size, and how far above the curve it landed
    /// in dB.
    fn sao_curve_offsets(
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        width: usize,
        height: usize,
        sweep: &[i32],
    ) -> Vec<(i32, usize, f64)> {
        let source = [y.to_vec(), cb.to_vec(), cr.to_vec()].concat();
        let encode = |qp: i32, filter| {
            let (slice, recon, _) =
                write_idr_residual_slice(y, cb, cr, width, height, qp, ModeSearch::Rdo, filter);
            let psnr = psnr_db(
                &source,
                &[recon.y.clone(), recon.cb.clone(), recon.cr.clone()].concat(),
            );
            (slice.len() as f64, psnr)
        };
        // The SAO-off writer's curve, finest first — the ladder
        // [`the_rate_distortion_cost_beats_the_fixed_qp_decision_at_equal_rate`]
        // interpolates against, widened at both ends to bracket the whole
        // sweep. Its rungs are spaced widely on purpose: adjacent QPs differ
        // by a handful of bytes on pictures this size, so a chord between
        // them measures the byte rounding rather than the curve.
        let ladder: Vec<(f64, f64)> = (0..=51).map(|qp| encode(qp, LoopFilter::Deblock)).collect();
        let mut out = Vec::new();
        for &qp in sweep {
            let (_, off_psnr) = encode(qp, LoopFilter::Deblock);
            let (bytes, psnr) = encode(qp, LoopFilter::DeblockSao);
            if psnr <= off_psnr {
                // The slice-level test declined the pass: the reconstruction
                // reverted, and the slice is the deblocked one plus the two
                // `slice_sao_*_flag` bits of header.
                continue;
            }
            let pair = ladder
                .windows(2)
                .find(|pair| pair[1].0 <= bytes && bytes <= pair[0].0)
                .unwrap_or_else(|| panic!("qp {qp}: {bytes} bytes is off the measured ladder"));
            let (bigger, smaller) = (pair[0], pair[1]);
            let t = (bigger.0.ln() - bytes.ln()) / (bigger.0.ln() - smaller.0.ln());
            let interpolated = bigger.1 + t * (smaller.1 - bigger.1);
            out.push((qp, bytes as usize, psnr - interpolated));
        }
        out
    }

    #[test]
    #[ignore = "measurement: what the SAO stage costs the writer in wall clock"]
    fn sao_decision_cost() {
        // Interleaved rounds, elementwise minimum, because a shared host
        // makes any single timed round an upper bound on somebody else's
        // load rather than a measurement of this code.
        let (width, height) = (128usize, 96usize);
        for (name, (y, cb, cr)) in [
            ("noise", picture(width, height)),
            ("smooth", smooth_picture(width, height)),
        ] {
            for qp in [12i32, 26, 37] {
                let run = |filter| {
                    let start = std::time::Instant::now();
                    let out = write_idr_residual_slice(
                        &y,
                        &cb,
                        &cr,
                        width,
                        height,
                        qp,
                        ModeSearch::Rdo,
                        filter,
                    );
                    std::hint::black_box(&out);
                    start.elapsed()
                };
                let mut off = std::time::Duration::MAX;
                let mut on = std::time::Duration::MAX;
                for _ in 0..7 {
                    off = off.min(run(LoopFilter::Deblock));
                    on = on.min(run(LoopFilter::DeblockSao));
                }
                println!(
                    "COST {name} qp {qp}: SAO off {:.2} ms, on {:.2} ms ({:+.0}%)",
                    off.as_secs_f64() * 1e3,
                    on.as_secs_f64() * 1e3,
                    (on.as_secs_f64() / off.as_secs_f64() - 1.0) * 100.0
                );
            }
        }
    }

    /// The QP 12-51 sweep on both test pictures at both sizes, which is what
    /// #287's three numbers were read off: where each accepted SAO point sits
    /// against the SAO-off writer's own curve, how many components the search
    /// gave band offset, and what the pass bought.
    ///
    /// Re-run it against a candidate band-syntax charge to see the bracket
    /// [`SaoLambda::band_q8`] records. Below 1.5x `lambda_q8` the `CURVE` line
    /// for noise 128x96 QP 32 goes to -0.003 dB; at 4x the `SWEEP` lines lose
    /// their band components entirely.
    #[test]
    #[ignore = "measurement: the QP 12-51 SAO sweep on both pictures at both sizes"]
    fn sao_sweep() {
        for (name, width, height, smooth) in [
            ("smooth 64x48", 64usize, 48usize, true),
            ("smooth 128x96", 128, 96, true),
            ("noise 64x48", 64, 48, false),
            ("noise 128x96", 128, 96, false),
        ] {
            let (y, cb, cr) = if smooth {
                smooth_picture(width, height)
            } else {
                picture(width, height)
            };
            for qp in 12..=51 {
                let ((off_bytes, off_psnr), (on_bytes, on_psnr)) =
                    sao_on_off(&y, &cb, &cr, width, height, qp);
                let band = sao_grid(&y, &cb, &cr, width, height, qp)
                    .iter()
                    .flat_map(|cell| cell.components.iter())
                    .filter(|c| c.sao_type_idx == 1)
                    .count();
                println!(
                    "SWEEP {name} qp {qp}: {off_bytes} -> {on_bytes} bytes, \
                     {off_psnr:.3} -> {on_psnr:.3} dB ({:+.3}), band components {band}",
                    on_psnr - off_psnr
                );
            }
            let sweep: Vec<i32> = (12..=51).collect();
            for (qp, bytes, delta) in sao_curve_offsets(&y, &cb, &cr, width, height, &sweep) {
                println!("CURVE {name} qp {qp}: {bytes} bytes, {delta:+.3} dB off the curve");
            }
        }
    }

    /// A point on a picture's curve, for the calibration's own unit tests.
    fn curve(bits: u64, sse: u64) -> CurvePoint {
        CurvePoint { sse, bits }
    }

    #[test]
    fn the_calibrated_multiplier_charges_more_than_the_secant_it_is_read_off() {
        // The whole reason the calibration interpolates instead of taking the
        // secant's slope: SAO spends a fraction of a quantizer step's rate,
        // and a convex curve is steeper near the coded point than the step's
        // average. Charging the average is what left points under the curve.
        let coded = curve(8_000, 100_000);
        let finer = curve(10_000, 80_000);
        let sao_bits = 400;
        let secant = 256 * (coded.sse - finer.sse) / (finer.bits - coded.bits);
        let calibrated = u64::from(calibrated_sao_lambda_q8(finer, coded, 26, sao_bits));
        assert!(
            calibrated > secant,
            "the interpolated multiplier {calibrated} does not charge more than the secant's \
             {secant}"
        );
        // ...and the closer to the coded point SAO stays, the steeper the
        // stretch of curve it is charged for.
        let nearer = u64::from(calibrated_sao_lambda_q8(finer, coded, 26, sao_bits / 4));
        assert!(
            nearer > calibrated,
            "a smaller SAO rate was charged {nearer}, no more than the larger one's {calibrated}"
        );
    }

    #[test]
    fn a_probe_that_describes_no_curve_falls_back_to_the_closed_form() {
        // Every degenerate two-point measurement: a probe that coded the same
        // bits, one that reconstructed the same picture, one that moved both
        // the wrong way, and an empty slice.
        let coded = curve(8_000, 100_000);
        let fixed = lambda_q8(26);
        for (name, probe) in [
            ("same rate", curve(8_000, 80_000)),
            ("same distortion", curve(10_000, 100_000)),
            ("more bits and more error", curve(10_000, 120_000)),
            ("no slice at all", curve(0, 0)),
        ] {
            assert_eq!(
                calibrated_sao_lambda_q8(probe, coded, 26, 400),
                fixed,
                "{name}: the calibration invented a slope out of it"
            );
        }
        assert_eq!(
            calibrated_sao_lambda_q8(curve(10_000, 80_000), coded, 26, 0),
            fixed,
            "a grid that costs nothing has no per-bit worth to calibrate"
        );
    }

    #[test]
    fn the_calibrated_multiplier_stays_inside_its_band() {
        let fixed = u64::from(lambda_q8(26));
        // A picture whose probe says one step of the quantizer buys almost
        // nothing, and one whose probe says it buys nearly everything.
        let flat = calibrated_sao_lambda_q8(curve(10_000, 99_999), curve(8_000, 100_000), 26, 400);
        let steep = calibrated_sao_lambda_q8(curve(10_000, 1), curve(8_000, 100_000), 26, 400);
        assert_eq!(u64::from(flat), (fixed / SAO_LAMBDA_BAND).max(1));
        assert_eq!(u64::from(steep), fixed * SAO_LAMBDA_BAND);
    }

    #[test]
    fn the_slice_level_test_only_probes_when_the_band_leaves_it_open() {
        // The probe is a second decision pass, so it may only run where its
        // answer can still change the decision — which is exactly where the
        // gain falls inside the band the calibrated multiplier is clamped to.
        let qp = 26;
        let fixed = u64::from(lambda_q8(qp));
        let sao_bits = 256;
        let never = || panic!("the probe ran where the band had already settled the decision");
        assert!(
            keeps_sao(
                sao_bits * fixed * SAO_LAMBDA_BAND / 256 + 1,
                sao_bits,
                qp,
                never
            ),
            "a gain clearing the band's coarse end was not kept"
        );
        assert!(
            !keeps_sao(
                sao_bits * (fixed / SAO_LAMBDA_BAND) / 256,
                sao_bits,
                qp,
                never
            ),
            "a gain missing the band's fine end was not declined"
        );
        // Between them the probe decides, and its answer is used.
        let inside = sao_bits * fixed / 256;
        let probed = std::cell::Cell::new(false);
        let probe = || {
            probed.set(true);
            (curve(10_000, 80_000), curve(8_000, 100_000))
        };
        keeps_sao(inside, sao_bits, qp, probe);
        assert!(
            probed.get(),
            "the probe was skipped where the band left it open"
        );
    }

    #[test]
    fn every_accepted_sao_point_sits_on_the_writers_own_curve() {
        // The property the calibrated multiplier exists for, and the one the
        // fixed lambda could not hold: wherever the writer decides SAO is
        // worth its syntax, the point it lands on is at least as good as
        // spending the same bits on a finer quantizer instead. Both pictures
        // at both sizes over the whole sweep, because the fixed lambda's two
        // failures were on one picture at one QP and a decision that is only
        // right where it was measured is not a decision.
        for (name, width, height, smooth) in [
            ("smooth 64x48", 64usize, 48usize, true),
            ("smooth 128x96", 128, 96, true),
            ("noise 64x48", 64, 48, false),
            ("noise 128x96", 128, 96, false),
        ] {
            let (y, cb, cr) = if smooth {
                smooth_picture(width, height)
            } else {
                picture(width, height)
            };
            let sweep: Vec<i32> = (12..=51).collect();
            let accepted = sao_curve_offsets(&y, &cb, &cr, width, height, &sweep);
            assert!(
                !accepted.is_empty(),
                "{name}: the writer accepted SAO nowhere in the sweep, so the \
                 measurement asserts nothing"
            );
            for (qp, bytes, delta) in accepted {
                assert!(
                    delta >= 0.0,
                    "{name} qp {qp}: the accepted SAO point ({bytes} bytes) sits {delta:+.3} dB \
                     below the SAO-off writer's own curve at the same rate"
                );
            }
        }
    }

    #[test]
    fn the_sao_pass_is_worth_taking_where_the_writer_takes_it() {
        // The gain the changelog records, at the two operating points the
        // decision actually accepts: a fine QP on smooth content, where the
        // quantizer leaves ringing along every edge for edge offset to pull
        // back, and a coarse one on the noise-carrying picture, where §8.7.2
        // declined most edges and left the error for §8.7.3.
        for (name, (y, cb, cr), qp, floor) in [
            ("smooth 64x48", smooth_picture(64, 48), 12i32, 0.7f64),
            ("smooth 128x96", smooth_picture(128, 96), 12, 0.85),
            ("noise 64x48", picture(64, 48), 37, 0.2),
            ("noise 128x96", picture(128, 96), 37, 0.3),
        ] {
            let (width, height) = if name.ends_with("128x96") {
                (128, 96)
            } else {
                (64, 48)
            };
            let ((off_bytes, off_psnr), (on_bytes, on_psnr)) =
                sao_on_off(&y, &cb, &cr, width, height, qp);
            let gain = on_psnr - off_psnr;
            assert!(
                gain > floor,
                "{name} qp {qp}: SAO moved whole-picture PSNR by only {gain:+.3} dB"
            );
            assert!(
                on_bytes > off_bytes,
                "{name} qp {qp}: SAO coded {on_bytes} bytes against {off_bytes} without paying \
                 for its own syntax — the sao( ) structures are not reaching the slice"
            );
        }
    }

    #[test]
    fn the_sao_pass_is_declined_when_it_would_not_clear_its_own_syntax() {
        // The other half of the decision, and the reason SAO can be left on
        // unconditionally. Every CTB pays a sao( ) structure whether or not
        // the search found anything for it, so a picture the search finds
        // nothing in must come out as the deblocked one did, to the byte:
        // `slice_sao_luma_flag == 0` suppresses §7.3.8.3 entirely.
        let (width, height) = (64, 48);
        let (y, cb, cr) = smooth_picture(width, height);
        let mut declined = 0;
        for qp in [26i32, 37, 44, 51] {
            let ((off_bytes, off_psnr), (on_bytes, on_psnr)) =
                sao_on_off(&y, &cb, &cr, width, height, qp);
            if on_bytes == off_bytes {
                declined += 1;
                assert_eq!(
                    on_psnr, off_psnr,
                    "qp {qp}: the slice is the deblocked one's size but not its picture"
                );
            }
        }
        assert!(
            declined > 0,
            "the slice-level decision never declined SAO — the rate test is not reached"
        );
    }

    #[test]
    fn the_sao_pass_never_costs_the_writer_quality() {
        // Whatever the decision does, it may not make the picture worse: the
        // per-CTB search only signals offsets that reduce that CTB's squared
        // error, and the slice-level test reverts the whole pass when the
        // total does not clear its rate.
        for (name, (y, cb, cr)) in [
            ("noise", picture(64, 48)),
            ("smooth", smooth_picture(64, 48)),
        ] {
            for qp in [12i32, 20, 26, 32, 37, 44, 51] {
                let ((off_bytes, off_psnr), (on_bytes, on_psnr)) =
                    sao_on_off(&y, &cb, &cr, 64, 48, qp);
                assert!(
                    on_psnr >= off_psnr,
                    "{name} qp {qp}: SAO cost {:+.3} dB of whole-picture PSNR",
                    on_psnr - off_psnr
                );
                assert!(
                    on_bytes as f64 <= off_bytes as f64 * 1.15,
                    "{name} qp {qp}: SAO cost {on_bytes} bytes against {off_bytes}, more than the \
                     measured worst case"
                );
            }
        }
    }

    #[test]
    fn the_access_unit_signals_sao_as_enabled() {
        // The reconstruction is only the decoder's if the parameter sets ask
        // the decoder for the same filter, so read the flag back off the
        // emitted SPS rather than trusting the writer.
        let (width, height) = (64, 48);
        let (y, cb, cr) = smooth_picture(width, height);
        let (au, _) = encode_idr_residual_au(&y, &cb, &cr, width, height, 12).unwrap();
        let sps = crate::hevc::engine::nal::collect_nal_units(&au)
            .expect("the access unit parses")
            .into_iter()
            .find(|nal| nal.header.nal_unit_type == 33)
            .expect("the access unit carries an SPS");
        let parsed =
            crate::hevc::engine::sps::SeqParameterSet::parse(&sps.rbsp).expect("SPS parses");
        assert!(
            parsed.sample_adaptive_offset_enabled_flag,
            "the writer's SPS still disables SAO"
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
