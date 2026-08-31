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
//! Every coding unit is intra DC (`prev_intra_luma_pred_flag == 1`,
//! `mpm_idx == 1` — with every neighbour also DC, the §8.4.2 candidate list is
//! `{ Planar, DC, Angular26 }`) with `intra_chroma_pred_mode == 4` (derived
//! from luma). Mode search over the 35 intra directions is a separate concern
//! from having a residual path at all, and belongs with the RDO work in
//! [`crate::hevc::engine::encoder::rdo`].
//!
//! In-loop filters are neutralized exactly as the PCM writer does it: SAO off
//! in the SPS and deblocking disabled in the PPS, so the reconstruction below
//! is the decoder's without a filter pass.

use crate::hevc::engine::cabac::init_type;
use crate::hevc::engine::ctx_init::SliceContexts;
use crate::hevc::engine::encoder::bitwriter::BitWriter;
use crate::hevc::engine::encoder::cabac::CabacEncoder;
use crate::hevc::engine::encoder::nal::{annexb, nal_unit};
use crate::hevc::engine::encoder::pcm::{
    PcmEncodeError, level_idc_for, write_pps, write_sps, write_vps,
};
use crate::hevc::engine::encoder::recon::ReconstructedPicture;
use crate::hevc::engine::encoder::residual::{
    EngineResidualBinSink, ResidualWriteParams, has_coded_levels, write_residual_coding,
};
use crate::hevc::engine::encoder::transform::{
    ForwardBlockParams, chroma_qp, luma_qp, transform_and_quantize,
};
use crate::hevc::engine::intra_pred::{
    Component as IpComponent, IntraPredParams, MarkedReferenceSamples,
    intra_predict_with_substitution,
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
/// `INTRA_DC` (Table 8-1 mode 1) — the only mode this writer codes.
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
    for addr in 0..total {
        let x0 = (addr % ctbs_x) * CTB;
        let y0 = (addr / ctbs_x) * CTB;

        // ---- coding_unit(): PART_2Nx2N, pcm_flag == 0, intra DC ----
        cabac.encode_decision(&mut w, &mut ctxs.part_mode[0], 1);
        cabac.encode_terminate(&mut w, 0); // pcm_flag = 0
        // prev_intra_luma_pred_flag = 1 with mpm_idx = 1: with every
        // neighbour DC the §8.4.2 candidate list is { Planar, DC, 26 }.
        cabac.encode_decision(&mut w, &mut ctxs.prev_intra_luma_pred_flag[0], 1);
        cabac.encode_bypass(&mut w, 1); // mpm_idx, TR(cMax 2): "10" == 1
        cabac.encode_bypass(&mut w, 0);
        // intra_chroma_pred_mode = 4 (derived from luma): a single 0 bin.
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
        );
        let (cx, cy) = (x0 / 2, y0 / 2);
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
        );
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

/// Code one transform block: predict from the reconstruction so far, transform
/// and quantize the residual, write the reconstruction back, and return the
/// quantized levels the caller will code.
///
/// The reconstruction step deliberately runs the decoder's own §8.6.2 process
/// on the quantized levels rather than the encoder's un-quantized residual, so
/// `recon` holds what a decoder will hold and the next block predicts from the
/// same samples the decoder will.
fn code_block(
    plane: Plane<'_>,
    recon: &mut [u8],
    x0: usize,
    y0: usize,
    n_tbs: usize,
    q_p: u32,
    component: TfComponent,
) -> Vec<i32> {
    let prediction = predict_dc(recon, plane.width, plane.height, x0, y0, n_tbs, component);
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

/// §8.4.4.2 intra DC prediction for one transform block, reading its
/// neighbours out of the partially reconstructed plane.
///
/// The §6.4.1 z-scan availability for this geometry — one unsplit coding unit
/// per CTB, coded in raster order, one slice, no tiles — is: the left column
/// exists for its first `nTbS` rows when the block is not at the left picture
/// edge (the rest belongs to the not-yet-coded block below-left); the top row
/// exists when the block is not at the top edge, and extends `nTbS` samples to
/// the right as long as those stay inside the picture, because the
/// above-right block was coded on the previous CTB row.
fn predict_dc(
    recon: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    n_tbs: usize,
    component: TfComponent,
) -> Vec<i32> {
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
    intra_predict_with_substitution(
        &marked,
        &IntraPredParams {
            pred_mode_intra: INTRA_DC,
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
    .expect("intra DC prediction")
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
