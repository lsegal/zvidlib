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
//! In-loop filters are neutralized exactly as the PCM writer does it: SAO off
//! in the SPS and deblocking disabled in the PPS, so the reconstruction below
//! is the decoder's without a filter pass. Enabling them is a separate
//! decision from mode search and is deliberately not made here.

use crate::hevc::engine::binarization::intra_luma_cand_mode_list;
use crate::hevc::engine::cabac::init_type;
use crate::hevc::engine::ctx_init::SliceContexts;
use crate::hevc::engine::encoder::bitwriter::BitWriter;
use crate::hevc::engine::encoder::cabac::CabacEncoder;
use crate::hevc::engine::encoder::nal::{annexb, nal_unit};
use crate::hevc::engine::encoder::pcm::{
    PcmEncodeError, level_idc_for, write_pps, write_sps, write_vps,
};
use crate::hevc::engine::encoder::recon::ReconstructedPicture;
use crate::hevc::engine::encoder::rdo::{DistortionBackend, decide_intra_luma_mode};
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
/// The valid `SliceQpY` range at 8-bit depth (`QpBdOffsetY == 0`).
const QP_RANGE: core::ops::RangeInclusive<i32> = 0..=51;

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

    let (rbsp, recon) = write_idr_residual_slice(y, cb, cr, width, height, qp);
    let level_idc = level_idc_for(width * height);
    let units = vec![
        nal_unit(32, 0, 0, &write_vps(level_idc)), // VPS_NUT
        nal_unit(33, 0, 0, &write_sps(width, height, level_idc, false, true)), // SPS_NUT
        nal_unit(34, 0, 0, &write_pps(false, false, None)), // PPS_NUT
        nal_unit(20, 0, 0, &rbsp),                 // IDR_N_LP
    ];
    Ok((annexb(&units), recon))
}

/// §7.3.6.1 + §7.3.8.1 — the picture's single I slice segment, every CTB one
/// residual-coded intra coding unit. Returns the slice RBSP and the picture
/// reconstructed alongside it.
fn write_idr_residual_slice(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
) -> (Vec<u8>, ReconstructedPicture) {
    let mut w = BitWriter::new();
    // ---- slice_segment_header() ----
    w.put_bit(1); // first_slice_segment_in_pic_flag
    w.put_bit(0); // no_output_of_prior_pics_flag (IRAP NAL)
    w.ue(0); // slice_pic_parameter_set_id
    w.ue(2); // slice_type = I
    w.se(qp - 26); // slice_qp_delta over init_qp_minus26 == 0
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
        let luma_refs = reference_samples(&recon.y, width, height, x0, y0, CTB);
        let mode = decide_intra_luma_mode(
            &y[y0 * width + x0..],
            width,
            CTB,
            candidates,
            qp,
            DistortionBackend::Dispatched,
            |mode| predict(&luma_refs, mode, TfComponent::Luma),
        )
        .mode;
        left_mode = mode;

        // ---- coding_unit(): PART_2Nx2N, pcm_flag == 0 ----
        cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1);
        cabac.encode_terminate(&mut w, 0); // pcm_flag = 0
        write_luma_intra_mode(&mut w, &mut cabac, &mut ctxs, mode, candidates);
        // intra_chroma_pred_mode = 4 (derived from luma): a single 0 bin, so
        // IntraPredModeC is the luma mode chosen above.
        cabac.encode_decision(&mut w, &mut ctxs.intra_chroma_pred_mode[0], 0);

        // ---- transform_tree(): split_transform_flag is absent because
        // max_transform_hierarchy_depth_intra == 0, so MaxTrafoDepth == 0
        // and the flag is inferred 0 (one 16x16 luma TB, two 8x8 chroma).
        let luma = code_block(
            Plane {
                source: y,
                width,
                height,
            },
            &mut recon.y,
            x0,
            y0,
            CTB,
            qp_luma,
            TfComponent::Luma,
            &predict(&luma_refs, mode, TfComponent::Luma),
        );
        let (cx, cy) = (x0 / 2, y0 / 2);
        let chroma_refs = |plane: &[u8]| reference_samples(plane, cw, ch, cx, cy, CTB / 2);
        let cb_pred = predict(&chroma_refs(&recon.cb), mode, TfComponent::Cb);
        let chroma_cb = code_block(
            Plane {
                source: cb,
                width: cw,
                height: ch,
            },
            &mut recon.cb,
            cx,
            cy,
            CTB / 2,
            qp_chroma,
            TfComponent::Cb,
            &cb_pred,
        );
        let cr_pred = predict(&chroma_refs(&recon.cr), mode, TfComponent::Cr);
        let chroma_cr = code_block(
            Plane {
                source: cr,
                width: cw,
                height: ch,
            },
            &mut recon.cr,
            cx,
            cy,
            CTB / 2,
            qp_chroma,
            TfComponent::Cr,
            &cr_pred,
        );

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
    (w.finish(), recon)
}

/// Code one transform block from the `prediction` the caller derived for the
/// mode it coded: transform and quantize the residual, write the
/// reconstruction back, and return the quantized levels the caller will code.
///
/// The reconstruction step deliberately runs the decoder's own §8.6.2 process
/// on the quantized levels rather than the encoder's un-quantized residual, so
/// `recon` holds what a decoder will hold and the next block predicts from the
/// same samples the decoder will.
#[allow(clippy::too_many_arguments)]
fn code_block(
    plane: Plane<'_>,
    recon: &mut [u8],
    x0: usize,
    y0: usize,
    n_tbs: usize,
    q_p: u32,
    component: TfComponent,
    prediction: &[i32],
) -> Vec<i32> {
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
    for row in 0..n_tbs {
        for col in 0..n_tbs {
            let i = row * n_tbs + col;
            let sample = clip1(prediction[i] + reconstructed[i], BIT_DEPTH);
            recon[(y0 + row) * plane.width + x0 + col] = sample as u8;
        }
    }
    levels
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

    #[test]
    fn rejects_bad_geometry_and_out_of_range_qp() {
        let (y, cb, cr) = picture(32, 32);
        assert!(encode_idr_residual_au(&y, &cb, &cr, 30, 32, 26).is_err());
        assert!(encode_idr_residual_au(&y, &cb, &cr, 32, 32, -1).is_err());
        assert!(encode_idr_residual_au(&y, &cb, &cr, 32, 32, 52).is_err());
        assert!(encode_idr_residual_au(&y, &cb, &y, 32, 32, 26).is_err());
    }
}
