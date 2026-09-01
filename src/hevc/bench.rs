//! Benchmark-only access to the HEVC encoder's individual pipeline stages.
//!
//! `crate::hevc` is a private module, and criterion benchmarks are a separate
//! crate, so the per-stage encoder groups in `benches/hevc_encode.rs` cannot
//! reach `rdo::decide_picture`, the CABAC encoding engine, or the PCM
//! bitstream writer through the public API. The public
//! [`crate::native_hevc_video_encoder_factory`] runs all of them at once, which
//! is exactly what a per-stage breakdown must avoid.
//!
//! This module is that access, and nothing more: thin wrappers that own their
//! inputs, return plain bytes, and add no logic the benchmark could accidentally
//! measure instead of the encoder. It is `#[doc(hidden)]` and explicitly not
//! part of the stable API, matching the `crate::hevc::engine` convention of
//! exposing internals for tests and fuzzing without promising them.
//!
//! Each wrapper returns the bytes that identify its result, because
//! `benches/support/isa.rs` compares those bytes across instruction sets before
//! timing anything: a stage whose return value did not depend on the kernels
//! under test would silently disarm that guard.

use crate::hevc::engine::cabac::ContextModel;
use crate::hevc::engine::encoder::bitwriter::BitWriter;
use crate::hevc::engine::encoder::cabac::CabacEncoder;
use crate::hevc::engine::encoder::lossy::encode_idr_residual_au;
use crate::hevc::engine::encoder::pcm::encode_idr_pcm_au;
use crate::hevc::engine::encoder::rdo::{DecisionConfig, PictureDecision, decide_picture};
use crate::hevc::engine::encoder::recon::{
    ReconConfig, ReconstructedPicture, SourcePlanes, reconstruct_picture,
};
use crate::hevc::engine::encoder::transform::{self as fwd_transform, ForwardBlockParams};
use crate::hevc::engine::transform::{Component, PredMode};

/// Bit depth the encoder benchmarks run at — the only depth the PCM
/// writer and the synthetic 8-bit inputs use.
const BENCH_BIT_DEPTH: u8 = 8;

/// The fixed predictor [`fwd_transform_quant_picture`] subtracts, the
/// mid-point of the 8-bit range.
const BENCH_PREDICTOR: i32 = 128;

/// Runs the encoder's mode-search / RDO stage over one luma picture.
///
/// This is the stage that reaches `hevc_rdcost`, the crate's only
/// encoder-side SIMD dispatch family, through its SAD and SATD distortion
/// metrics.
///
/// `reference_y`, when present, enables the coarse whole-pel inter search;
/// passing `None` measures the intra-only decision cost. Returns the search's
/// decisions serialized in traversal order, so the bit-exactness guard covers
/// every partition's motion vector and cost rather than only the picture total.
#[must_use]
pub fn rdo_decide_picture(
    y: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    reference_y: Option<&[u8]>,
    qp: i32,
    search_radius: i32,
) -> Vec<u8> {
    let decision = decide_picture(
        y,
        stride,
        width,
        height,
        reference_y,
        DecisionConfig {
            qp,
            search_radius,
            ..DecisionConfig::default()
        },
    );
    let mut out = Vec::with_capacity(decision.blocks.len() * 32 + 16);
    out.extend_from_slice(&decision.rd_cost.to_le_bytes());
    out.extend_from_slice(&(decision.pcm_blocks as u64).to_le_bytes());
    for block in &decision.blocks {
        out.extend_from_slice(&(block.x as u32).to_le_bytes());
        out.extend_from_slice(&(block.y as u32).to_le_bytes());
        out.extend_from_slice(&(block.size as u32).to_le_bytes());
        out.extend_from_slice(&block.rd_cost.to_le_bytes());
        out.extend_from_slice(&block.pcm_cost.to_le_bytes());
        for partition in &block.partitions {
            out.extend_from_slice(&(partition.x as u32).to_le_bytes());
            out.extend_from_slice(&(partition.y as u32).to_le_bytes());
            out.extend_from_slice(&(partition.w as u32).to_le_bytes());
            out.extend_from_slice(&(partition.h as u32).to_le_bytes());
            out.extend_from_slice(&partition.mv_x.to_le_bytes());
            out.extend_from_slice(&partition.mv_y.to_le_bytes());
            out.extend_from_slice(&partition.sad.to_le_bytes());
            out.extend_from_slice(&partition.satd.to_le_bytes());
            out.extend_from_slice(&partition.bit_cost.to_le_bytes());
            out.extend_from_slice(&partition.rd_cost.to_le_bytes());
        }
    }
    out
}

/// Writes one IDR access unit for a YUV420 picture: parameter sets, slice
/// header, and the CABAC-coded CU syntax carrying the PCM samples.
///
/// This is the encoder's entropy-coding and bitwriting stage as it actually
/// runs today. The returned Annex B access unit is the stage's own output, so
/// timing it measures the writer and the bit-exactness guard covers the
/// bitstream itself.
///
/// # Panics
///
/// Panics if the picture does not satisfy the PCM writer's requirements
/// (dimensions divisible by 16 and correctly sized planes), which a benchmark
/// input always does.
#[must_use]
pub fn write_idr_pcm_access_unit(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    encode_idr_pcm_au(y, cb, cr, width, height).expect("benchmark pictures are writable as PCM")
}

/// Drives the §9.3.5 CABAC arithmetic encoder over a deterministic bin
/// sequence, isolated from any picture data.
///
/// `bins` alternates context-coded and bypass bins so both `encode_decision`
/// (with its context-model state transition) and `encode_bypass` are exercised;
/// `contexts` sets how many distinct context models the sequence cycles
/// through, which is what makes the state transitions non-degenerate. The
/// returned bytes are the arithmetic codeword, so this measures the CABAC
/// engine plus the bit writer it writes through.
#[must_use]
pub fn cabac_encode_bins(bins: &[u8], contexts: usize) -> Vec<u8> {
    assert!(contexts > 0, "the bin sequence needs at least one context");
    // §9.3.2.2 initialization at a representative slice QP. Spreading the
    // `initValue` entries keeps the models from all starting in the same state,
    // which would make the state transitions degenerate.
    let mut models: Vec<ContextModel> = (0..contexts)
        .map(|index| ContextModel::init(154 - (index % 32) as u8 * 2, 26))
        .collect();
    let mut writer = BitWriter::new();
    let mut cabac = CabacEncoder::new();
    for (index, &bin) in bins.iter().enumerate() {
        if index % 2 == 0 {
            cabac.encode_decision(&mut writer, &mut models[index % contexts], bin & 1);
        } else {
            cabac.encode_bypass(&mut writer, bin & 1);
        }
    }
    cabac.encode_terminate(&mut writer, 1);
    writer.finish()
}

/// Writes a deterministic syntax-element sequence through the raw bit writer,
/// with no arithmetic coding on top.
///
/// The parameter sets and slice headers of every access unit go through these
/// fixed-length, `ue(v)` and `se(v)` writers, so measuring them separately
/// separates raw bitwriting cost from the CABAC engine's.
///
/// Each value is reduced into the range the corresponding syntax elements
/// actually occupy, so the codeword lengths — which are what the writer's cost
/// scales with — stay representative instead of being dominated by pathological
/// 32-bit Exp-Golomb codes.
#[must_use]
pub fn bitwriter_write_syntax(values: &[u32]) -> Vec<u8> {
    let mut writer = BitWriter::new();
    for (index, &value) in values.iter().enumerate() {
        match index % 3 {
            0 => writer.put_bits(value & 0xffff, 16),
            1 => writer.ue(value % 4096),
            _ => writer.se((value % 2048) as i32 - 1024),
        }
    }
    writer.rbsp_trailing_bits();
    writer.finish()
}

/// Converts one RGBA8 frame to the YUV420 planes the encoder's later stages
/// consume.
///
/// The public encoder takes RGBA8 input, so this conversion runs once per frame
/// ahead of mode search and bitstream writing. It is not one of the stages the
/// SIMD override reaches, and measuring it separately is what makes the
/// whole-frame number decomposable into stages that add up.
///
/// # Panics
///
/// Panics unless `frame` is an RGBA8 frame, which a benchmark input always is.
#[must_use]
pub fn rgba_to_yuv420_planes(frame: &crate::VideoFrame) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    super::encoder::rgba_to_yuv420(frame, crate::Orientation::TopLeft)
        .expect("benchmark frames are RGBA8")
}

/// Runs the encoder's forward transform and quantization stage over one
/// luma picture, once per transform-block size.
///
/// Every 4x4, 8x8, 16x16 and 32x32 block that fits in the picture is
/// transformed and quantized, so one iteration covers all four §8.6.4.2
/// matrices and the 4x4 DST-VII the intra path selects. The residual is
/// the source sample minus a fixed mid-level predictor rather than a real
/// prediction: this group measures the transform and quantization
/// kernels, and a mode-dependent predictor would fold mode-search cost
/// into that number.
///
/// The returned bytes are the quantized levels themselves, so the
/// bit-exactness guard in `benches/support/isa.rs` covers every level the
/// vector kernels produced.
///
/// # Panics
///
/// Panics if `y` is smaller than `height * stride`.
#[must_use]
pub fn fwd_transform_quant_picture(
    y: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    qp: i32,
) -> Vec<u8> {
    assert!(
        y.len() >= (height - 1) * stride + width,
        "the luma plane is smaller than the requested picture"
    );
    let q_p = fwd_transform::luma_qp(qp, BENCH_BIT_DEPTH);
    let mut out = Vec::new();
    let mut residual = vec![0i32; 32 * 32];
    for log2 in 2u32..=5 {
        let n_tbs = 1usize << log2;
        let params = ForwardBlockParams {
            n_tbs,
            q_p,
            component: Component::Luma,
            pred_mode: PredMode::Intra,
            bit_depth: BENCH_BIT_DEPTH,
            extended_precision: false,
        };
        let block = &mut residual[..n_tbs * n_tbs];
        for by in (0..=height.saturating_sub(n_tbs)).step_by(n_tbs) {
            for bx in (0..=width.saturating_sub(n_tbs)).step_by(n_tbs) {
                for row in 0..n_tbs {
                    let src = &y[(by + row) * stride + bx..][..n_tbs];
                    for (dst, &sample) in block[row * n_tbs..][..n_tbs].iter_mut().zip(src) {
                        *dst = i32::from(sample) - BENCH_PREDICTOR;
                    }
                }
                let levels = fwd_transform::transform_and_quantize(block, None, params)
                    .expect("benchmark blocks are legal transform blocks");
                for level in levels {
                    out.extend_from_slice(&(level as i16).to_le_bytes());
                }
            }
        }
    }
    out
}

/// Writes one IDR access unit whose coding units carry *quantized residual*
/// rather than raw PCM samples: intra prediction, forward transform,
/// quantization, the decoder's own reconstruction, and the §7.3.8.11
/// `residual_coding( )` entropy coding of the levels.
///
/// This is the lossy write path end to end, and the counterpart to
/// [`write_idr_pcm_access_unit`]: comparing the two separates what coding a
/// quantized residual costs from what writing a bitstream costs at all. The
/// returned Annex B access unit is the stage's own output, so the
/// bit-exactness guard covers the bitstream itself.
///
/// # Panics
///
/// Panics if the picture does not satisfy the writer's requirements
/// (dimensions divisible by 16, correctly sized planes, `qp` in 0..=51),
/// which a benchmark input always does.
#[must_use]
pub fn write_idr_residual_access_unit(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    qp: i32,
) -> Vec<u8> {
    encode_idr_residual_au(y, cb, cr, width, height, qp)
        .expect("benchmark pictures are writable as residual")
        .0
}

/// A picture-reconstruction workload with its mode-search plan already built.
///
/// The reconstruction stage consumes the decisions the mode search produced,
/// and mode search costs an order of magnitude more than everything else in
/// the encoder. Building the plan here, once, keeps it out of the timed loop
/// so `hevc_encode_*_reconstruct` measures reconstruction and the in-loop
/// filters rather than re-measuring `hevc_encode_*_rdo_inter`.
///
/// Opaque on purpose: the decision plan and the reconstructed-picture types
/// are crate-internal, and this surface promises nothing about them.
pub struct ReconstructWorkload {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    width: usize,
    height: usize,
    reference: Option<ReconstructedPicture>,
    decision: PictureDecision,
}

/// Builds a [`ReconstructWorkload`] over one 4:2:0 8-bit picture.
///
/// `reference` is the previous picture's `(y, cb, cr)` planes, which stand in
/// for the previous *reconstruction*: passing them enables the inter
/// prediction path through the reconstruction loop, and `None` measures the
/// intra path. `qp` and `search_radius` configure the mode search that runs
/// here, in setup.
///
/// # Panics
///
/// Panics if the planes do not describe a 4:2:0 picture of `width * height`
/// with whole 16-sample CTBs.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn plan_reconstruct(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    reference: Option<(&[u8], &[u8], &[u8])>,
    qp: i32,
    search_radius: i32,
) -> ReconstructWorkload {
    let reference = reference.map(|(ry, rcb, rcr)| ReconstructedPicture {
        y: ry.to_vec(),
        cb: rcb.to_vec(),
        cr: rcr.to_vec(),
        width,
        height,
    });
    let decision = decide_picture(
        y,
        width,
        width,
        height,
        reference.as_ref().map(|r| r.y.as_slice()),
        DecisionConfig {
            qp,
            search_radius,
            ..DecisionConfig::default()
        },
    );
    ReconstructWorkload {
        y: y.to_vec(),
        cb: cb.to_vec(),
        cr: cr.to_vec(),
        width,
        height,
        reference,
        decision,
    }
}

/// Runs the encoder's reconstruction and in-loop filter stage over a planned
/// picture, returning the reconstructed 4:2:0 planes.
///
/// This is the stage that reaches the decoder's already-vectorized §8.7.2
/// deblocking and §8.7.3 SAO kernels from the encode side: `deblocking` and
/// `sao` select the loop-filter shape of the access unit being modelled, and
/// with both set the reconstruction runs the same filters a decoder would.
/// The returned planes are the stage's own output, so the bit-exactness guard
/// covers every filtered sample.
#[must_use]
pub fn reconstruct_encoded_picture(
    workload: &ReconstructWorkload,
    deblocking: bool,
    sao: bool,
) -> Vec<u8> {
    reconstruct_encoded_picture_quantized(workload, deblocking, sao, false)
}

/// [`reconstruct_encoded_picture`] with control over whether the residual is
/// round-tripped through the forward transform and quantizer.
///
/// With `quantized` set, every transform block of every partition goes through
/// §8.6.4 / §8.6.3 and back through the decoder's §8.6.2 reconstruction, which
/// is what the reconstruction stage costs once the writer stops coding PCM.
/// Measuring both says how much of the stage is prediction and filtering and
/// how much is the transform round trip.
#[must_use]
pub fn reconstruct_encoded_picture_quantized(
    workload: &ReconstructWorkload,
    deblocking: bool,
    sao: bool,
    quantized: bool,
) -> Vec<u8> {
    let reconstructed = reconstruct_picture(
        SourcePlanes {
            y: &workload.y,
            cb: &workload.cb,
            cr: &workload.cr,
            width: workload.width,
            height: workload.height,
        },
        workload.reference.as_ref(),
        &workload.decision,
        ReconConfig {
            deblocking,
            sao_luma: sao,
            sao_chroma: sao,
            // The filters are being measured, so model an access unit that
            // does not suppress them on its PCM coding units
            // (`pcm_loop_filter_disabled_flag == 0`, which
            // `PcmAuOptions::pcm_loop_filter_disabled == false` writes).
            pcm_loop_filter_disabled: !(deblocking || sao),
            quantized_residual: quantized,
            ..ReconConfig::default()
        },
    );
    let mut out = reconstructed.y;
    out.extend_from_slice(&reconstructed.cb);
    out.extend_from_slice(&reconstructed.cr);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{self, SimdIsa, test_lock};

    /// A picture whose dimensions the PCM writer and the mode search accept
    /// (both require multiples of 16), filled with a moving gradient plus
    /// low-amplitude noise so no stage sees a degenerate best case.
    fn picture(width: usize, height: usize, phase: i32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ phase as u64;
        let mut noise = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 58) as i32
        };
        let y = (0..height)
            .flat_map(|row| (0..width).map(move |col| (col, row)))
            .map(|(col, row)| (((col as i32 + row as i32 + phase) / 2 + noise()) & 0xff) as u8)
            .collect();
        let chroma = |offset: i32| {
            (0..height / 2)
                .flat_map(|row| (0..width / 2).map(move |col| (col, row)))
                .map(|(col, row)| {
                    (128 + ((col as i32 - row as i32 + phase + offset) % 24) - 12) as u8
                })
                .collect::<Vec<u8>>()
        };
        (y, chroma(0), chroma(7))
    }

    /// The guard the benchmark's `bench_across_isas` relies on: every wrapper
    /// must return the same bytes under every instruction set the host can run.
    ///
    /// The encoder's SIMD dispatch families are the mode search's distortion
    /// metrics and the RGBA8 to YUV420 input conversion. The distortion metrics are
    /// where a divergence would surface as a *different mode decision* rather than a
    /// different picture — which a bit-exactness check on decoded pixels would never
    /// see.
    #[test]
    fn every_stage_wrapper_is_bit_exact_across_instruction_sets() {
        let _guard = test_lock();
        let (y, cb, cr) = picture(64, 32, 0);
        let (reference, _, _) = picture(64, 32, 3);
        let bins: Vec<u8> = (0..2048)
            .map(|i| ((i * 2_654_435_761u64) >> 13) as u8)
            .collect();
        let values: Vec<u32> = (0..1024)
            .map(|i: u32| i.wrapping_mul(2_654_435_761))
            .collect();

        simd::set_override(Some(SimdIsa::Scalar));
        let expected = (
            rdo_decide_picture(&y, 64, 64, 32, None, 26, 4),
            rdo_decide_picture(&y, 64, 64, 32, Some(&reference), 26, 4),
            write_idr_pcm_access_unit(&y, &cb, &cr, 64, 32),
            cabac_encode_bins(&bins, 64),
            bitwriter_write_syntax(&values),
        );
        for isa in simd::available() {
            simd::set_override(Some(isa));
            let name = isa.name();
            assert_eq!(
                rdo_decide_picture(&y, 64, 64, 32, None, 26, 4),
                expected.0,
                "{name}: intra mode search diverged from scalar"
            );
            assert_eq!(
                rdo_decide_picture(&y, 64, 64, 32, Some(&reference), 26, 4),
                expected.1,
                "{name}: inter mode search diverged from scalar"
            );
            assert_eq!(
                write_idr_pcm_access_unit(&y, &cb, &cr, 64, 32),
                expected.2,
                "{name}: the written access unit diverged from scalar"
            );
            assert_eq!(
                cabac_encode_bins(&bins, 64),
                expected.3,
                "{name}: the CABAC codeword diverged from scalar"
            );
            assert_eq!(
                bitwriter_write_syntax(&values),
                expected.4,
                "{name}: the written syntax elements diverged from scalar"
            );
        }
        simd::set_override(None);
    }

    /// The bit-exactness guard above is only meaningful if the bytes each
    /// wrapper returns actually depend on its input. A wrapper that returned a
    /// constant would pass every comparison in the benchmark harness while
    /// measuring nothing, so pin that each one is input-sensitive.
    #[test]
    fn every_stage_wrapper_returns_bytes_that_depend_on_its_input() {
        let _guard = test_lock();
        simd::set_override(None);
        let (y, cb, cr) = picture(64, 32, 0);
        let (other, other_cb, other_cr) = picture(64, 32, 11);

        assert_ne!(
            rdo_decide_picture(&y, 64, 64, 32, None, 26, 4),
            rdo_decide_picture(&other, 64, 64, 32, None, 26, 4),
            "the serialized mode-search decisions ignore the picture"
        );
        assert_ne!(
            rdo_decide_picture(&y, 64, 64, 32, None, 26, 4),
            rdo_decide_picture(&y, 64, 64, 32, Some(&other), 26, 4),
            "the serialized mode-search decisions ignore the reference picture"
        );
        assert_ne!(
            write_idr_pcm_access_unit(&y, &cb, &cr, 64, 32),
            write_idr_pcm_access_unit(&other, &other_cb, &other_cr, 64, 32),
            "the written access unit ignores the picture"
        );
        assert_ne!(
            cabac_encode_bins(&[1, 0, 1, 1, 0, 0, 1, 0], 4),
            cabac_encode_bins(&[0, 1, 0, 0, 1, 1, 0, 1], 4),
            "the CABAC codeword ignores the bins"
        );
        assert_ne!(
            bitwriter_write_syntax(&[1, 2, 3, 4]),
            bitwriter_write_syntax(&[9, 8, 7, 6]),
            "the written syntax elements ignore their values"
        );
    }

    /// The RGBA8 conversion wrapper produces the plane sizes the later stages
    /// index into, so a benchmark cannot silently feed them a short buffer.
    #[test]
    fn the_rgba_conversion_wrapper_produces_correctly_sized_yuv420_planes() {
        let limits = crate::Limits::default();
        let dimensions = crate::VideoDimensions::new(64, 32, &limits).unwrap();
        let frame = crate::VideoFrame::new(
            dimensions,
            crate::PixelFormat::Rgba8,
            crate::ColorRange::Limited,
            vec![crate::Plane {
                data: (0..64 * 32 * 4).map(|i| (i % 251) as u8).collect(),
                stride: 64 * 4,
            }],
            &limits,
        )
        .unwrap();
        let (y, cb, cr) = rgba_to_yuv420_planes(&frame);
        assert_eq!(y.len(), 64 * 32);
        assert_eq!(cb.len(), 64 * 32 / 4);
        assert_eq!(cr.len(), 64 * 32 / 4);
        // Limited-range luma stays inside the spec's 16..=235 window.
        assert!(y.iter().all(|&sample| (16..=235).contains(&sample)));
    }
}
