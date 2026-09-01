//! Encoder-side reconstruction and in-loop filtering — the encode-side
//! counterpart to the decoder's `engine/recon.rs`, `engine/deblock.rs` and
//! `engine/sao.rs`.
//!
//! An encoder cannot predict from the pictures it was handed. It has to
//! predict from the pictures a decoder will actually hold, which is what this
//! module produces: for every coded block it forms the prediction, adds the
//! coded residual back onto it (§8.6.6), and then runs the §8.7.2 deblocking
//! filter and the §8.7.3 sample-adaptive offset over the whole picture through
//! the decoder's own kernels. The result is the reference picture the next
//! frame's mode search in [`crate::hevc::engine::encoder::rdo`] predicts from.
//!
//! ## Lossless and lossy reconstruction
//!
//! [`ReconConfig::quantized_residual`] selects which writer this
//! reconstruction models. Cleared, the residual is coded exactly, matching the
//! PCM writer in [`crate::hevc::engine::encoder::pcm`] whose coding units are
//! all `pcm_flag == 1` blocks: the reconstruction is then bit-identical to the
//! source. Set, the residual is round-tripped through the §8.6.4 forward
//! transform and §8.6.3 quantization in
//! [`crate::hevc::engine::encoder::transform`] and back through the decoder's
//! own scaling and inverse transform, which is what the residual writer in
//! [`crate::hevc::engine::encoder::lossy`] codes — and the reconstructed
//! reference then genuinely diverges from the source, so the mode search in
//! [`crate::hevc::engine::encoder::rdo`] predicts from a lossy picture the way
//! a real encoder's does.
//!
//! Either way the loop here is the real one — predict, add the coded residual,
//! clip — rather than a plane copy. That is also what makes the in-loop filter
//! stage below meaningful, since those filters run on the reconstruction, not
//! on the source.
//!
//! ## What the in-loop filters do here
//!
//! [`ReconConfig`] mirrors the loop-filter shape of the access unit the writer
//! emits, so the encoder's reconstruction is always the picture a conforming
//! decoder derives from that access unit — never an approximation of it. With
//! the shipped [`crate::hevc::engine::encoder::pcm::PcmAuOptions`] defaults
//! (deblocking disabled in the PPS, SAO off in the SPS,
//! `pcm_loop_filter_disabled_flag == 1`) the filters are correctly *not*
//! applied and the encode stays lossless; with an access unit that enables
//! them, the §8.7.2 and §8.7.3 drivers run over the reconstruction exactly as
//! the decoder's do, including the §8.7.2.5.4 / §8.7.3.1 PCM suppression map.

use crate::hevc::engine::binarization::PartMode;
use crate::hevc::engine::deblock::{DeblockCu, DeblockCuDesc, DeblockCuParams, NoFilterMap};
use crate::hevc::engine::encoder::rdo::PictureDecision;
use crate::hevc::engine::encoder::transform::{
    ForwardBlockParams, chroma_qp, luma_qp, transform_and_quantize,
};
use crate::hevc::engine::motion::{MotionCell, MotionField};
use crate::hevc::engine::picture::{Picture, Plane, clip1};
use crate::hevc::engine::sao::{ResolvedSao, ResolvedSaoComponent};
use crate::hevc::engine::transform::{
    BlockParams, Component as TfComponent, PredMode, residual_block,
};

/// `CtbLog2SizeY` of the encoder's fixed geometry (16-sample CTBs), matching
/// the PCM writer's `CTB_LOG2`.
const CTB_LOG2: u32 = 4;
/// `CtbSizeY`.
const CTB: usize = 1 << CTB_LOG2;
/// `ChromaArrayType` — 4:2:0, the only format the encoder writes.
const CHROMA_ARRAY_TYPE: u8 = 1;
/// `BitDepthY` / `BitDepthC`.
const BIT_DEPTH: u8 = 8;
/// The prediction value of an intra block with no reconstructed neighbours
/// available, matching the RDO search's own neutral reference.
const NEUTRAL_LUMA: i32 = 128;

/// The loop-filter shape of the access unit being written, as the
/// reconstruction has to model it.
///
/// These are the decoder-visible flags, not encoder preferences: they must
/// match what [`crate::hevc::engine::encoder::pcm`] writes into the SPS / PPS /
/// slice header, or the encoder's reference picture stops being the decoder's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReconConfig {
    /// `slice_deblocking_filter_disabled_flag == 0` — run §8.7.2.
    pub deblocking: bool,
    /// `slice_sao_luma_flag` — run the §8.7.3 luma pass.
    pub sao_luma: bool,
    /// `slice_sao_chroma_flag` — run the §8.7.3 chroma passes.
    pub sao_chroma: bool,
    /// `pcm_loop_filter_disabled_flag` (§7.4.3.2.1). Every coding unit this
    /// writer emits is a PCM block, so when this is set the loop filters must
    /// leave every reconstructed sample alone.
    pub pcm_loop_filter_disabled: bool,
    /// `SliceQpY`, the q-side QP the §8.7.2.5.3 β / t<sub>C</sub> derivation
    /// reads, and the `qP` the residual round trip quantizes at when
    /// [`Self::quantized_residual`] is set.
    pub qp: i32,
    /// Model a writer that codes *quantized* residual rather than the exact
    /// one a PCM block carries: every transform block of every partition goes
    /// through forward transform, quantization, and the decoder's own §8.6.2
    /// reconstruction at [`Self::qp`]. Cleared, the residual is exact and the
    /// reconstruction is the source picture.
    pub quantized_residual: bool,
}

impl Default for ReconConfig {
    /// The shipped PCM writer's shape: both filters neutralized, so the
    /// reconstruction is the source picture bit for bit.
    fn default() -> Self {
        Self {
            deblocking: false,
            sao_luma: false,
            sao_chroma: false,
            pcm_loop_filter_disabled: true,
            qp: 26,
            quantized_residual: false,
        }
    }
}

/// One reconstructed picture in the encoder's own planar 4:2:0 8-bit form —
/// the same layout the source planes and the RDO search use, so a
/// reconstruction can be fed straight back in as the next picture's reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReconstructedPicture {
    /// Reconstructed luma, `width * height`.
    pub y: Vec<u8>,
    /// Reconstructed Cb, `(width / 2) * (height / 2)`.
    pub cb: Vec<u8>,
    /// Reconstructed Cr.
    pub cr: Vec<u8>,
    /// Luma width.
    pub width: usize,
    /// Luma height.
    pub height: usize,
}

/// The source picture a frame is coded from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SourcePlanes<'a> {
    /// Source luma.
    pub y: &'a [u8],
    /// Source Cb.
    pub cb: &'a [u8],
    /// Source Cr.
    pub cr: &'a [u8],
    /// Luma width (also the luma stride).
    pub width: usize,
    /// Luma height.
    pub height: usize,
}

/// Reconstruct one coded picture and run the in-loop filters over it.
///
/// `decision` is the plan the mode search produced for this picture (its
/// partitions supply the prediction geometry and motion vectors); `reference`
/// is the previous *reconstructed* picture, or `None` for an intra picture.
///
/// # Panics
/// Panics if the source planes do not match a 4:2:0 picture of
/// `src.width * src.height`, or if the dimensions are not whole CTBs.
pub(crate) fn reconstruct_picture(
    src: SourcePlanes<'_>,
    reference: Option<&ReconstructedPicture>,
    decision: &PictureDecision,
    cfg: ReconConfig,
) -> ReconstructedPicture {
    let (width, height) = (src.width, src.height);
    assert!(width > 0 && height > 0);
    assert!(width % CTB == 0 && height % CTB == 0);
    assert_eq!(src.y.len(), width * height);
    let (cw, ch) = (width / 2, height / 2);
    assert_eq!(src.cb.len(), cw * ch);
    assert_eq!(src.cr.len(), cw * ch);
    let reference = reference.filter(|r| r.width == width && r.height == height);

    let mut pic = Picture::new(width, height, CHROMA_ARRAY_TYPE, BIT_DEPTH, BIT_DEPTH);
    let mut field = MotionField::new(width, height);

    for block in &decision.blocks {
        // A block whose PCM cost won the search is coded as an intra PCM
        // block, which is what this writer emits today; anything else is
        // coded from its searched partitions.
        let coded_as_pcm = reference.is_none() || block.rd_cost >= block.pcm_cost;
        for partition in &block.partitions {
            let (mv_x, mv_y) = if coded_as_pcm {
                (0, 0)
            } else {
                (partition.mv_x, partition.mv_y)
            };
            let predict_from = if coded_as_pcm { None } else { reference };
            reconstruct_partition(
                &mut pic,
                src,
                predict_from,
                partition.x,
                partition.y,
                partition.w,
                partition.h,
                mv_x,
                mv_y,
                cfg.quantized_residual.then_some(cfg.qp),
            );
            field.fill_rect(
                partition.x,
                partition.y,
                partition.w,
                partition.h,
                MotionCell {
                    is_intra: coded_as_pcm,
                    // The §8.7.2.4 `cbf` test. An exactly-coded residual
                    // always carries levels; a quantized one is assumed to,
                    // which only ever over-filters a boundary.
                    has_nonzero_coeff: true,
                    pred_flag_l0: !coded_as_pcm,
                    pred_flag_l1: false,
                    ref_poc_l0: if coded_as_pcm { i32::MIN } else { 0 },
                    ref_poc_l1: i32::MIN,
                    mv_l0: [mv_x * 4, mv_y * 4],
                    mv_l1: [0, 0],
                },
            );
        }
    }

    // §8.7.2.5.4 / §8.7.3.1 — every coding unit is a PCM block, so
    // `pcm_loop_filter_disabled_flag` suppresses the filters picture-wide.
    let (w4, h4) = (width.div_ceil(4), height.div_ceil(4));
    let no_filter_cells = vec![cfg.pcm_loop_filter_disabled; w4 * h4];
    let no_filter = cfg.pcm_loop_filter_disabled.then_some(NoFilterMap {
        cells: &no_filter_cells,
        w_cells: w4,
    });

    if cfg.deblocking {
        let cus = deblock_descriptors(width, height, cfg.qp);
        crate::hevc::engine::deblock::deblock_picture_full(
            &mut pic,
            &field,
            &cus,
            None,
            no_filter.as_ref(),
        );
    }

    if cfg.sao_luma || cfg.sao_chroma {
        // No §7.3.8.3 syntax is written for this reconstruction, so the
        // decision is distortion-only.
        let grid = estimate_sao(&pic, src, cfg.sao_luma, cfg.sao_chroma, 0);
        pic = crate::hevc::engine::sao::apply_sao_picture_full(
            pic,
            &grid,
            CTB_LOG2,
            CHROMA_ARRAY_TYPE,
            cfg.sao_luma,
            cfg.sao_chroma,
            None,
            no_filter.as_ref(),
        );
    }

    ReconstructedPicture {
        y: plane_to_u8(&pic, Plane::Luma),
        cb: plane_to_u8(&pic, Plane::Cb),
        cr: plane_to_u8(&pic, Plane::Cr),
        width,
        height,
    }
}

/// Reconstruct one prediction partition: form the prediction, code the
/// residual against it, and add the coded residual back (§8.6.6
/// `recSamples = Clip1( predSamples + resSamples )`).
///
/// With `quant` set to `None` the residual is `source − prediction` coded
/// exactly, as a PCM block carries it, and the reconstruction is the source.
/// With a `qP`, the residual is instead round-tripped through the §8.6.4
/// forward transform and §8.6.3 quantization and back through the decoder's
/// own §8.6.2 reconstruction — which is what makes the reference picture
/// differ from the source, and the entire reason it is kept separately at all.
#[allow(clippy::too_many_arguments)]
fn reconstruct_partition(
    pic: &mut Picture,
    src: SourcePlanes<'_>,
    reference: Option<&ReconstructedPicture>,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mv_x: i32,
    mv_y: i32,
    quant: Option<i32>,
) {
    reconstruct_component(
        pic,
        Plane::Luma,
        src.y,
        src.width,
        reference.map(|r| (r.y.as_slice(), r.width, r.height)),
        x,
        y,
        w,
        h,
        mv_x,
        mv_y,
        // §8.6.1 eq. 8-284 — the luma `qP` from `SliceQpY`.
        quant.map(|qp| luma_qp(qp, BIT_DEPTH)),
        TfComponent::Luma,
    );
    // 4:2:0 — the chroma partition is the luma one halved, and the motion
    // vector with it (§8.5.3.3.3.3 whole-sample part; this search is
    // whole-pel, so the halved vector needs no fractional interpolation).
    let (cw, chh) = (src.width / 2, src.height / 2);
    for (plane, source, reference_plane, component) in [
        (
            Plane::Cb,
            src.cb,
            reference.map(|r| r.cb.as_slice()),
            TfComponent::Cb,
        ),
        (
            Plane::Cr,
            src.cr,
            reference.map(|r| r.cr.as_slice()),
            TfComponent::Cr,
        ),
    ] {
        reconstruct_component(
            pic,
            plane,
            source,
            cw,
            reference_plane.map(|p| (p, cw, chh)),
            x / 2,
            y / 2,
            (w / 2).max(1),
            (h / 2).max(1),
            mv_x / 2,
            mv_y / 2,
            // §8.6.1 with the writer's zero chroma QP offsets: the chroma
            // blocks quantize at the Table 8-10 mapping of the luma QP.
            quant.map(|qp| chroma_qp(qp, 0, BIT_DEPTH, CHROMA_ARRAY_TYPE)),
            component,
        );
    }
}

/// [`reconstruct_partition`] for one colour component.
#[allow(clippy::too_many_arguments)]
fn reconstruct_component(
    pic: &mut Picture,
    plane: Plane,
    source: &[u8],
    src_stride: usize,
    reference: Option<(&[u8], usize, usize)>,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mv_x: i32,
    mv_y: i32,
    quant: Option<u32>,
    component: TfComponent,
) {
    let predict = |sx: usize, sy: usize| -> i32 {
        match reference {
            // §8.5.3.3.2 reference-sample clamping: a vector that leaves the
            // picture reads the edge sample.
            Some((reference, rw, rh)) => {
                let rx = (sx as i32 + mv_x).clamp(0, rw as i32 - 1) as usize;
                let ry = (sy as i32 + mv_y).clamp(0, rh as i32 - 1) as usize;
                i32::from(reference[ry * rw + rx])
            }
            None => NEUTRAL_LUMA,
        }
    };
    let Some(q_p) = quant else {
        // The residual is coded exactly, so the reconstruction is the source.
        for row in 0..h {
            for col in 0..w {
                let (sx, sy) = (x + col, y + row);
                let predicted = predict(sx, sy);
                let residual = i32::from(source[sy * src_stride + sx]) - predicted;
                pic.set_sample(plane, sx, sy, clip1(predicted + residual, BIT_DEPTH));
            }
        }
        return;
    };

    // §7.3.8.8: the residual of a partition is coded as square transform
    // blocks. This geometry's partitions are whole or halved CTBs, so tiling
    // by the largest legal square that fits is the transform tree a writer
    // would build for them.
    let n_tbs = transform_block_size(w, h);
    let mut prediction = vec![0i32; n_tbs * n_tbs];
    let mut residual = vec![0i32; n_tbs * n_tbs];
    for by in (0..h).step_by(n_tbs) {
        for bx in (0..w).step_by(n_tbs) {
            for row in 0..n_tbs {
                for col in 0..n_tbs {
                    let (sx, sy) = (x + bx + col, y + by + row);
                    let predicted = predict(sx, sy);
                    prediction[row * n_tbs + col] = predicted;
                    residual[row * n_tbs + col] =
                        i32::from(source[sy * src_stride + sx]) - predicted;
                }
            }
            // Forward transform → quantize → the decoder's own §8.6.2
            // reconstruction of the levels that survived.
            let pred_mode = if reference.is_some() {
                PredMode::Inter
            } else {
                PredMode::Intra
            };
            let levels = transform_and_quantize(
                &residual,
                None,
                ForwardBlockParams {
                    n_tbs,
                    q_p,
                    component,
                    pred_mode,
                    bit_depth: BIT_DEPTH,
                    extended_precision: false,
                },
            )
            .expect("encoder-sized transform block");
            // An all-zero level block codes `cbf == 0` and carries no
            // residual at all, which is what a decoder reconstructs.
            let coded = if levels.iter().any(|&l| l != 0) {
                residual_block(
                    &levels,
                    None,
                    BlockParams {
                        n_tbs,
                        q_p,
                        component,
                        pred_mode,
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
                    pic.set_sample(
                        plane,
                        x + bx + col,
                        y + by + row,
                        clip1(prediction[i] + coded[i], BIT_DEPTH),
                    );
                }
            }
        }
    }
}

/// The largest legal square transform block that tiles a `w` x `h` partition:
/// the largest power of two no greater than either side, clamped to the
/// §7.4.3.2.1 4..=32 transform-block range.
fn transform_block_size(w: usize, h: usize) -> usize {
    let side = w.min(h);
    let log2 = (usize::BITS - 1 - side.leading_zeros()).clamp(2, 5);
    1usize << log2
}

/// One §8.7.2 descriptor per coding unit. Every CTB is one unsplit,
/// untransformed `PART_2Nx2N` coding unit in this encoder's geometry, and the
/// picture is one slice with no tiles, so only the picture boundaries are
/// excluded from filtering (§8.7.2.1 `filterLeftCbEdgeFlag` /
/// `filterTopCbEdgeFlag`).
fn deblock_descriptors(width: usize, height: usize, qp: i32) -> Vec<DeblockCuDesc> {
    let params = DeblockCuParams {
        qp_y: qp,
        beta_offset_div2: 0,
        tc_offset_div2: 0,
        cb_qp_offset: 0,
        cr_qp_offset: 0,
        bit_depth_luma: BIT_DEPTH,
        bit_depth_chroma: BIT_DEPTH,
        chroma_array_type: CHROMA_ARRAY_TYPE,
    };
    let mut cus = Vec::with_capacity((width / CTB) * (height / CTB));
    for y_cb in (0..height).step_by(CTB) {
        for x_cb in (0..width).step_by(CTB) {
            cus.push(DeblockCuDesc {
                cu: DeblockCu {
                    x_cb,
                    y_cb,
                    log2_cb_size: CTB_LOG2,
                    params,
                    qp_y_p_left: qp,
                    qp_y_p_top: qp,
                },
                transform_split: crate::hevc::engine::deblock::TransformSplit::leaf(),
                part_mode: PartMode::Part2Nx2N,
                filter_left: x_cb > 0,
                filter_top: y_cb > 0,
            });
        }
    }
    cus
}

/// §8.7.2 — run the in-loop deblocking filter over a finished
/// reconstruction, in place.
///
/// The residual writer in [`crate::hevc::engine::encoder::lossy`] builds its
/// reconstruction block by block as it codes, because §8.4.4.2.2 intra
/// prediction reads the neighbouring samples *prior to* the in-loop filter
/// process. Deblocking is therefore a whole-picture pass run once the last
/// coding unit of the picture is coded — the §8.7.1 ordering the decoder
/// itself uses — and not something interleaved into coding order.
///
/// Every coding unit the writer emits is one 16×16 intra `PART_2Nx2N` block
/// with an unsplit transform tree, coded at `SliceQpY`, so the descriptors are
/// exactly the ones [`deblock_descriptors`] builds. Because every unit is
/// intra, the §8.7.2.4 boundary strength is 2 at every filtered edge and the
/// `has_nonzero_coeff` / motion fields of the [`MotionField`] are never read —
/// its all-intra default is the whole of what this picture's field says.
pub(crate) fn deblock_reconstruction(recon: &mut ReconstructedPicture, qp: i32) {
    let (width, height) = (recon.width, recon.height);
    let mut pic = Picture::new(width, height, CHROMA_ARRAY_TYPE, BIT_DEPTH, BIT_DEPTH);
    for (plane, samples) in [
        (Plane::Luma, &recon.y),
        (Plane::Cb, &recon.cb),
        (Plane::Cr, &recon.cr),
    ] {
        let (buf, _) = pic.plane_mut(plane);
        for (dst, &src) in buf.iter_mut().zip(samples) {
            *dst = i32::from(src);
        }
    }
    let field = MotionField::new(width, height);
    let cus = deblock_descriptors(width, height, qp);
    crate::hevc::engine::deblock::deblock_picture(&mut pic, &field, &cus);
    recon.y = plane_to_u8(&pic, Plane::Luma);
    recon.cb = plane_to_u8(&pic, Plane::Cb);
    recon.cr = plane_to_u8(&pic, Plane::Cr);
}

/// The largest magnitude a `sao_offset_abs` can carry at 8-bit depth
/// (§7.4.9.3: `(1 << (Min(bitDepth, 10) − 5)) − 1`).
const SAO_OFFSET_MAX: i32 = 7;

/// §8.7.3 — estimate SAO parameters for a finished, already-deblocked
/// reconstruction, apply them in place, and return the per-CTB grid the
/// writer has to code as §7.3.8.3 syntax.
///
/// The counterpart of [`deblock_reconstruction`] for the second in-loop
/// filter, and run after it: §8.7.1 orders SAO behind deblocking, and the
/// parameter search is only meaningful against the samples SAO will actually
/// see. Like deblocking it is a whole-picture pass, because §8.4.4.2.2 intra
/// prediction reads the *unfiltered* neighbours — the grid returned here
/// describes the picture's output, never this picture's own prediction input.
///
/// `lambda_q8` is passed through to [`estimate_sao`], which is what makes the
/// per-CTB decision charge for the syntax the caller then writes.
pub(crate) fn sao_reconstruction(
    recon: &mut ReconstructedPicture,
    src: SourcePlanes<'_>,
    lambda_q8: u32,
) -> Vec<ResolvedSao> {
    let (width, height) = (recon.width, recon.height);
    let mut pic = Picture::new(width, height, CHROMA_ARRAY_TYPE, BIT_DEPTH, BIT_DEPTH);
    for (plane, samples) in [
        (Plane::Luma, &recon.y),
        (Plane::Cb, &recon.cb),
        (Plane::Cr, &recon.cr),
    ] {
        let (buf, _) = pic.plane_mut(plane);
        for (dst, &src) in buf.iter_mut().zip(samples) {
            *dst = i32::from(src);
        }
    }
    let grid = estimate_sao(&pic, src, true, true, lambda_q8);
    let pic = crate::hevc::engine::sao::apply_sao_picture_full(
        pic,
        &grid,
        CTB_LOG2,
        CHROMA_ARRAY_TYPE,
        true,
        true,
        None,
        None,
    );
    recon.y = plane_to_u8(&pic, Plane::Luma);
    recon.cb = plane_to_u8(&pic, Plane::Cb);
    recon.cr = plane_to_u8(&pic, Plane::Cr);
    grid
}

/// Encoder-side §7.3.8.3 SAO parameter estimation.
///
/// For each CTB this picks the edge-offset class whose per-category mean error
/// between the source and the deblocked reconstruction reduces the sum of
/// squared errors the most, and leaves SAO off for that CTB when no class
/// earns its own syntax. The offsets are the per-category mean errors clamped
/// to the signalable range with the §7.4.9.3 inferred edge-offset signs
/// (categories 1 and 2 positive, 3 and 4 negative), which is the standard
/// least-squares choice for edge offset and the reason this stage is worth
/// measuring: it reads every reconstructed sample of the picture once per
/// candidate class.
///
/// The two chroma components are decided *together*, on the summed gain of one
/// shared class, because §7.4.9.3 infers `SaoTypeIdx[2]` and `SaoEoClass[2]`
/// from cIdx 1: a Cb and a Cr that picked different classes are not a
/// bitstream. Each still carries its own four offsets, which is the whole of
/// what the syntax gives cIdx 2 of its own.
///
/// `lambda_q8` is the §9 rate-distortion multiplier in 1/256 units, matching
/// [`crate::hevc::engine::encoder::rdo::lambda_q8`]. SAO costs per-CTB syntax
/// on every CTB it is enabled for, so a class is taken only when its SSE
/// reduction clears `lambda * bins`, the bins being the §9.3.3 binarization's
/// own count for the parameters that would be coded. At `lambda_q8 == 0` the
/// test degrades to "any reduction at all", which is what a caller that codes
/// no syntax for the decision wants.
fn estimate_sao(
    pic: &Picture,
    src: SourcePlanes<'_>,
    sao_luma: bool,
    sao_chroma: bool,
    lambda_q8: u32,
) -> Vec<ResolvedSao> {
    let w_ctbs = src.width.div_ceil(CTB);
    let h_ctbs = src.height.div_ceil(CTB);
    let mut grid = vec![ResolvedSao::off(); w_ctbs * h_ctbs];
    for ry in 0..h_ctbs {
        for rx in 0..w_ctbs {
            let cell = &mut grid[ry * w_ctbs + rx];
            if sao_luma {
                cell.components[0] = best_luma_edge_offset(pic, src, rx, ry, lambda_q8);
            }
            if sao_chroma {
                let [cb, cr] = best_chroma_edge_offset(pic, src, rx, ry, lambda_q8);
                cell.components[1] = cb;
                cell.components[2] = cr;
            }
        }
    }
    grid
}

/// The CTB at `(rx, ry)` as a half-open rectangle in one plane's own sample
/// grid, clipped to the plane.
fn ctb_rect(pic: &Picture, plane: Plane, rx: usize, ry: usize) -> (usize, usize, usize, usize) {
    let (pw, ph) = pic.plane_dims(plane);
    let step = if plane == Plane::Luma { CTB } else { CTB / 2 };
    let (x0, y0) = (rx * step, ry * step);
    (x0, y0, (x0 + step).min(pw), (y0 + step).min(ph))
}

/// The §9.3.3 bin count of the four `sao_offset_abs` values of one component —
/// Table 9-43 TR with `cMax == 7` and `cRiceParam == 0`, i.e. truncated unary.
fn offset_abs_bins(offsets: &[i32; 5]) -> u64 {
    offsets[1..5]
        .iter()
        .map(|&o| {
            let v = u64::from(o.unsigned_abs()).min(SAO_OFFSET_MAX as u64);
            if v < SAO_OFFSET_MAX as u64 { v + 1 } else { v }
        })
        .sum()
}

/// Whether an SSE reduction of `gain` is worth the `bins` of §7.3.8.3 syntax
/// it has to be signalled with, under the same `D + lambda * R` cost the mode
/// decision uses.
fn clears_its_rate(gain: i64, bins: u64, lambda_q8: u32) -> bool {
    gain > (bins * u64::from(lambda_q8) / 256) as i64
}

/// The best §8.7.3.2 edge-offset component for the luma of one CTB, or an off
/// component when no class earns the `sao_type_idx` + `sao_eo_class_luma` +
/// four `sao_offset_abs` it would be coded with.
fn best_luma_edge_offset(
    pic: &Picture,
    src: SourcePlanes<'_>,
    rx: usize,
    ry: usize,
    lambda_q8: u32,
) -> ResolvedSaoComponent {
    let rect = ctb_rect(pic, Plane::Luma, rx, ry);
    let mut best = ResolvedSaoComponent::off();
    let mut best_gain = 0i64;
    for eo_class in 0..4u8 {
        let (gain, offsets) = class_offsets(pic, Plane::Luma, src.y, src.width, rect, eo_class);
        // One `sao_type_idx_luma` bin beyond the "off" bin, two
        // `sao_eo_class_luma` bins, and the offsets' own.
        let bins = 1 + 2 + offset_abs_bins(&offsets);
        if gain > best_gain && clears_its_rate(gain, bins, lambda_q8) {
            best_gain = gain;
            best = ResolvedSaoComponent {
                sao_type_idx: 2,
                offset_val: offsets,
                band_position: 0,
                eo_class,
            };
        }
    }
    best
}

/// The best §8.7.3.2 edge-offset components for the Cb and Cr of one CTB,
/// sharing the one class the §7.4.9.3 inference leaves them.
fn best_chroma_edge_offset(
    pic: &Picture,
    src: SourcePlanes<'_>,
    rx: usize,
    ry: usize,
    lambda_q8: u32,
) -> [ResolvedSaoComponent; 2] {
    let (cb_rect, cr_rect) = (
        ctb_rect(pic, Plane::Cb, rx, ry),
        ctb_rect(pic, Plane::Cr, rx, ry),
    );
    let chroma_stride = src.width / 2;
    let mut best = [ResolvedSaoComponent::off(); 2];
    let mut best_gain = 0i64;
    for eo_class in 0..4u8 {
        let (cb_gain, cb_offsets) =
            class_offsets(pic, Plane::Cb, src.cb, chroma_stride, cb_rect, eo_class);
        let (cr_gain, cr_offsets) =
            class_offsets(pic, Plane::Cr, src.cr, chroma_stride, cr_rect, eo_class);
        let gain = cb_gain + cr_gain;
        // One `sao_type_idx_chroma` bin beyond the "off" bin, two
        // `sao_eo_class_chroma` bins, and both components' offsets — cIdx 2
        // codes neither a type nor a class of its own.
        let bins = 1 + 2 + offset_abs_bins(&cb_offsets) + offset_abs_bins(&cr_offsets);
        if gain > best_gain && clears_its_rate(gain, bins, lambda_q8) {
            best_gain = gain;
            best = [
                ResolvedSaoComponent {
                    sao_type_idx: 2,
                    offset_val: cb_offsets,
                    band_position: 0,
                    eo_class,
                },
                ResolvedSaoComponent {
                    sao_type_idx: 2,
                    offset_val: cr_offsets,
                    band_position: 0,
                    eo_class,
                },
            ];
        }
    }
    best
}

/// The §7.4.9.3 offsets one edge-offset class would take on one CTB of one
/// plane, and the SSE reduction they buy.
fn class_offsets(
    pic: &Picture,
    plane: Plane,
    source: &[u8],
    src_stride: usize,
    rect: (usize, usize, usize, usize),
    eo_class: u8,
) -> (i64, [i32; 5]) {
    let (x0, y0, x1, y1) = rect;
    let (pw, ph) = pic.plane_dims(plane);
    let samples = pic.plane(plane);
    let (h0, v0, h1, v1) = crate::hevc::engine::sao::eo_pos(eo_class);
    // Per §8.7.3.2 category (1..4), the summed and counted error.
    let mut sums = [0i64; 5];
    let mut counts = [0i64; 5];
    for y in y0..y1 {
        for x in x0..x1 {
            let Some(category) = edge_category(samples, pw, ph, x, y, h0, v0, h1, v1) else {
                continue;
            };
            let error = i64::from(source[y * src_stride + x]) - i64::from(samples[y * pw + x]);
            sums[category] += error;
            counts[category] += 1;
        }
    }
    let mut offsets = [0i32; 5];
    let mut gain = 0i64;
    for category in 1..5 {
        if counts[category] == 0 {
            continue;
        }
        let mean = div_round(sums[category], counts[category]);
        // §7.4.9.3 infers the sign per category, so a mean that points the
        // other way is not signalable and the offset stays 0.
        let offset = if category <= 2 {
            mean.clamp(0, SAO_OFFSET_MAX)
        } else {
            mean.clamp(-SAO_OFFSET_MAX, 0)
        };
        offsets[category] = offset;
        // The SSE reduction of adding a constant `o` to `n` samples whose
        // summed error is `s`: 2*o*s − n*o^2.
        gain += 2 * i64::from(offset) * sums[category]
            - counts[category] * i64::from(offset) * i64::from(offset);
    }
    (gain, offsets)
}


/// §8.7.3.2 `edgeIdx` remapped to the `SaoOffsetVal` category index
/// (1..4), or `None` for category 0 (no offset) and for samples whose
/// neighbours leave the plane.
#[allow(clippy::too_many_arguments)]
fn edge_category(
    samples: &[i32],
    pw: usize,
    ph: usize,
    x: usize,
    y: usize,
    h0: i32,
    v0: i32,
    h1: i32,
    v1: i32,
) -> Option<usize> {
    let at = |dx: i32, dy: i32| -> Option<i32> {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        (nx >= 0 && ny >= 0 && (nx as usize) < pw && (ny as usize) < ph)
            .then(|| samples[ny as usize * pw + nx as usize])
    };
    let here = samples[y * pw + x];
    let a = at(h0, v0)?;
    let b = at(h1, v1)?;
    let edge_idx = 2 + (here - a).signum() + (here - b).signum();
    // §8.7.3.2: edgeIdx 0/1/2/3/4 maps to categories 1, 2, 0, 3, 4.
    match edge_idx {
        0 => Some(1),
        1 => Some(2),
        3 => Some(3),
        4 => Some(4),
        _ => None,
    }
}

/// Round-half-away-from-zero integer division.
fn div_round(numerator: i64, denominator: i64) -> i32 {
    debug_assert!(denominator > 0);
    let half = denominator / 2;
    let rounded = if numerator >= 0 {
        (numerator + half) / denominator
    } else {
        (numerator - half) / denominator
    };
    rounded as i32
}

/// One plane of a reconstructed [`Picture`] as 8-bit samples.
fn plane_to_u8(pic: &Picture, plane: Plane) -> Vec<u8> {
    pic.plane(plane)
        .iter()
        .map(|&v| v.clamp(0, 255) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hevc::engine::encoder::rdo::{DecisionConfig, decide_picture};

    const W: usize = 64;
    const H: usize = 32;

    /// A deterministic picture with flat coding blocks separated by small
    /// steps.
    ///
    /// The §8.7.2.5.3 decision only filters an edge whose two sides are flat
    /// enough (`d < β`), so a picture of hard high-contrast edges would be
    /// left entirely alone and would say nothing about whether the filters
    /// ran. Flat blocks with a modest step across each coding-block boundary
    /// are what the deblocking filter is for.
    fn source(shift: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; W * H];
        for row in 0..H {
            for col in 0..W {
                let block = ((col + shift * CTB) / CTB + row / CTB) % 4;
                y[row * W + col] = (96 + block * 9) as u8;
            }
        }
        let mut cb = vec![0u8; (W / 2) * (H / 2)];
        let mut cr = vec![0u8; (W / 2) * (H / 2)];
        for row in 0..H / 2 {
            for col in 0..W / 2 {
                let block = ((col + shift * CTB / 2) / (CTB / 2) + row / (CTB / 2)) % 4;
                cb[row * (W / 2) + col] = (110 + block * 7) as u8;
                cr[row * (W / 2) + col] = (140 - block * 7) as u8;
            }
        }
        (y, cb, cr)
    }

    fn planes<'a>(p: &'a (Vec<u8>, Vec<u8>, Vec<u8>)) -> SourcePlanes<'a> {
        SourcePlanes {
            y: &p.0,
            cb: &p.1,
            cr: &p.2,
            width: W,
            height: H,
        }
    }

    fn plan(src: &(Vec<u8>, Vec<u8>, Vec<u8>), reference: Option<&[u8]>) -> PictureDecision {
        decide_picture(&src.0, W, W, H, reference, DecisionConfig::default())
    }

    fn sse(a: &[u8], b: &[u8]) -> u64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = i64::from(x) - i64::from(y);
                (d * d) as u64
            })
            .sum()
    }

    #[test]
    fn reconstruction_is_the_source_while_the_writer_codes_pcm_losslessly() {
        // The shipped access unit neutralizes both loop filters, so the
        // reconstruction a decoder derives is the source picture bit for bit —
        // both for the intra picture and for one coded against a reference.
        let src = source(0);
        let intra = reconstruct_picture(
            planes(&src),
            None,
            &plan(&src, None),
            ReconConfig::default(),
        );
        assert_eq!(intra.y, src.0);
        assert_eq!(intra.cb, src.1);
        assert_eq!(intra.cr, src.2);

        let next = source(3);
        let inter = reconstruct_picture(
            planes(&next),
            Some(&intra),
            &plan(&next, Some(&intra.y)),
            ReconConfig::default(),
        );
        assert_eq!(inter.y, next.0);
        assert_eq!(inter.cb, next.1);
        assert_eq!(inter.cr, next.2);
    }

    #[test]
    fn pcm_suppression_keeps_enabled_filters_from_touching_the_reconstruction() {
        // §8.7.2.5.4 / §8.7.3.1: every coding unit here is a PCM block, so
        // `pcm_loop_filter_disabled_flag == 1` must leave every sample alone
        // even with both filters enabled in the PPS / slice header.
        let src = source(0);
        let suppressed = reconstruct_picture(
            planes(&src),
            None,
            &plan(&src, None),
            ReconConfig {
                deblocking: true,
                sao_luma: true,
                sao_chroma: true,
                ..ReconConfig::default()
            },
        );
        assert_eq!(suppressed.y, src.0);
        assert_eq!(suppressed.cb, src.1);
        assert_eq!(suppressed.cr, src.2);
    }

    #[test]
    fn enabled_in_loop_filters_modify_the_reconstruction() {
        let src = source(0);
        let filtered = reconstruct_picture(
            planes(&src),
            None,
            &plan(&src, None),
            ReconConfig {
                deblocking: true,
                sao_luma: true,
                sao_chroma: true,
                pcm_loop_filter_disabled: false,
                ..ReconConfig::default()
            },
        );
        assert_ne!(
            filtered.y, src.0,
            "an enabled deblocking + SAO pass must reach the reconstructed luma"
        );
        assert_eq!(filtered.y.len(), src.0.len());
        assert_eq!(filtered.cb.len(), src.1.len());
    }

    #[test]
    fn sao_estimation_does_not_increase_distortion_against_the_source() {
        // The estimator picks per-CTB edge offsets by the SSE reduction they
        // buy, and refuses a class that buys none, so enabling SAO on top of
        // the deblocked reconstruction can only move it towards the source.
        let src = source(0);
        let deblocked = reconstruct_picture(
            planes(&src),
            None,
            &plan(&src, None),
            ReconConfig {
                deblocking: true,
                pcm_loop_filter_disabled: false,
                ..ReconConfig::default()
            },
        );
        let with_sao = reconstruct_picture(
            planes(&src),
            None,
            &plan(&src, None),
            ReconConfig {
                deblocking: true,
                sao_luma: true,
                sao_chroma: true,
                pcm_loop_filter_disabled: false,
                ..ReconConfig::default()
            },
        );
        assert!(
            sse(&with_sao.y, &src.0) < sse(&deblocked.y, &src.0),
            "SAO estimation found no offsets on a deblocked picture that needs them"
        );
        assert!(sse(&with_sao.cb, &src.1) <= sse(&deblocked.cb, &src.1));
    }

    #[test]
    fn inter_reconstruction_predicts_from_the_reference_not_the_source() {
        // A reference that is not the source picture still reconstructs to the
        // source, because the residual is coded against whatever prediction
        // the reference produced — which is exactly the property that keeps
        // the encoder's reference and the decoder's in step.
        let src = source(0);
        let mut reference = reconstruct_picture(
            planes(&src),
            None,
            &plan(&src, None),
            ReconConfig::default(),
        );
        for sample in &mut reference.y {
            *sample = sample.saturating_add(9);
        }
        let next = source(2);
        let out = reconstruct_picture(
            planes(&next),
            Some(&reference),
            &plan(&next, Some(&reference.y)),
            ReconConfig::default(),
        );
        assert_eq!(out.y, next.0);
    }

    #[test]
    fn a_quantized_residual_makes_the_reconstruction_diverge_from_the_source() {
        // The property the lossy path exists for: once the residual is coded
        // through the transform and quantizer, the encoder's reference is no
        // longer the picture it was handed, and it drifts further as the step
        // coarsens.
        let src = source(0);
        let lossless = reconstruct_picture(
            planes(&src),
            None,
            &plan(&src, None),
            ReconConfig::default(),
        );
        assert_eq!(lossless.y, src.0, "the exact-residual path must stay exact");

        // The blocks in `source` are flat, so a fine step still reproduces
        // their DC-only residual exactly; the distortion is required to be
        // monotone in qP and strictly non-zero by the coarsest step.
        let mut previous = 0u64;
        for qp in [20i32, 34, 47] {
            let lossy = reconstruct_picture(
                planes(&src),
                None,
                &plan(&src, None),
                ReconConfig {
                    quantized_residual: true,
                    qp,
                    ..ReconConfig::default()
                },
            );
            let error = sse(&lossy.y, &src.0);
            assert!(
                error >= previous,
                "qP {qp} cost less distortion than the finer step"
            );
            previous = error;
            assert_eq!(lossy.y.len(), src.0.len());
            assert_eq!(lossy.cb.len(), src.1.len());
        }
        assert!(
            previous > 0,
            "a quantized reconstruction reproduced the source exactly"
        );
    }

    #[test]
    fn a_quantized_inter_reconstruction_predicts_from_the_lossy_reference() {
        // The reconstruction of an inter picture coded against a lossy
        // reference must differ from its source too — the drift the RDO
        // search is supposed to see accumulates picture over picture.
        let first = source(0);
        let cfg = ReconConfig {
            quantized_residual: true,
            qp: 34,
            ..ReconConfig::default()
        };
        let reference = reconstruct_picture(planes(&first), None, &plan(&first, None), cfg);
        let next = source(2);
        let out = reconstruct_picture(
            planes(&next),
            Some(&reference),
            &plan(&next, Some(&reference.y)),
            cfg,
        );
        assert_ne!(out.y, next.0, "the inter reconstruction stayed lossless");
        assert_eq!(out.y.len(), next.0.len());
    }

    #[test]
    fn partitions_tile_into_the_largest_legal_square_transform_block() {
        assert_eq!(transform_block_size(16, 16), 16);
        assert_eq!(transform_block_size(16, 8), 8);
        assert_eq!(transform_block_size(8, 4), 4);
        // Below the 4x4 minimum and above the 32x32 maximum both clamp.
        assert_eq!(transform_block_size(2, 2), 4);
        assert_eq!(transform_block_size(64, 64), 32);
    }

    #[test]
    fn every_ctb_gets_a_deblocking_descriptor_with_picture_edges_excluded() {
        let cus = deblock_descriptors(W, H, 26);
        assert_eq!(cus.len(), (W / CTB) * (H / CTB));
        assert!(!cus[0].filter_left && !cus[0].filter_top);
        assert!(cus[1].filter_left && !cus[1].filter_top);
        let second_row = &cus[W / CTB];
        assert!(!second_row.filter_left && second_row.filter_top);
    }
}
