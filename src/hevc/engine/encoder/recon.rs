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
use crate::hevc::engine::encoder::rdo::{BlockDecision, PictureDecision};
use crate::hevc::engine::encoder::recon_simd::{self, BandStats, EdgeStats};
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
    /// Run the band-offset half of the §8.7.3 parameter search.
    ///
    /// Always set outside the benchmark harness: clearing it leaves the
    /// per-CTB decision to the four edge-offset classes alone, which is a
    /// conformant but worse encode. It exists because the *share* of a
    /// reconstruction the band search costs is otherwise unmeasurable from
    /// outside — #382 needed the difference between the whole-picture group
    /// with the search running and with it skipped, and an estimate was
    /// exactly what would not settle the question. See
    /// [`crate::hevc::engine::encoder::recon_simd::band_offset_row`].
    pub sao_band_search: bool,
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
            sao_band_search: true,
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
        let predict_from = if coded_as_pcm { None } else { reference };
        reconstruct_block(
            &mut pic,
            src,
            predict_from,
            block,
            cfg.quantized_residual.then_some(cfg.qp),
        );
        for partition in &block.partitions {
            let (mv_x, mv_y) = if coded_as_pcm {
                (0, 0)
            } else {
                (partition.mv_x, partition.mv_y)
            };
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
        let grid = estimate_sao(
            &pic,
            src,
            cfg.sao_luma,
            cfg.sao_chroma,
            0,
            cfg.sao_band_search,
        );
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

/// Reconstruct one coding unit: form each prediction partition's prediction,
/// code the residual against it over the *coding unit's* transform tree, and
/// add the coded residual back (§8.6.6
/// `recSamples = Clip1( predSamples + resSamples )`).
///
/// §7.3.8.8 hangs the transform tree off the coding unit, not off the
/// prediction unit, which is why a 16x16 coding unit keeps an 8x8 chroma
/// transform block however its prediction is partitioned. The prediction
/// geometry is still each partition's own — its motion vector predicts only
/// its own samples — but it is gathered into one coding-unit-sized buffer that
/// the residual coding then reads whole.
///
/// With `quant` set to `None` the residual is `source − prediction` coded
/// exactly, as a PCM block carries it, and the reconstruction is the source.
/// With a `qP`, the residual is instead round-tripped through the §8.6.4
/// forward transform and §8.6.3 quantization and back through the decoder's
/// own §8.6.2 reconstruction — which is what makes the reference picture
/// differ from the source, and the entire reason it is kept separately at all.
fn reconstruct_block(
    pic: &mut Picture,
    src: SourcePlanes<'_>,
    reference: Option<&ReconstructedPicture>,
    block: &BlockDecision,
    quant: Option<i32>,
) {
    let pred_mode = if reference.is_some() {
        PredMode::Inter
    } else {
        PredMode::Intra
    };
    let size = block.size;
    let mut prediction = vec![0u8; size * size];
    gather_block_prediction(
        &mut prediction,
        size,
        reference.map(|r| (r.y.as_slice(), r.width, r.height)),
        block,
        false,
    );
    reconstruct_component(
        pic,
        Plane::Luma,
        src.y,
        src.width,
        &prediction,
        block.x,
        block.y,
        size,
        // §8.6.1 eq. 8-284 — the luma `qP` from `SliceQpY`.
        quant.map(|qp| luma_qp(qp, BIT_DEPTH)),
        TfComponent::Luma,
        pred_mode,
    );
    // 4:2:0 — the chroma coding block is the luma one halved, and each
    // partition's motion vector with it (§8.5.3.3.3.3 whole-sample part; this
    // search is whole-pel, so the halved vector needs no fractional
    // interpolation).
    let (cw, chh) = (src.width / 2, src.height / 2);
    let c_size = size / 2;
    prediction.truncate(c_size * c_size);
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
        gather_block_prediction(
            &mut prediction,
            c_size,
            reference_plane.map(|p| (p, cw, chh)),
            block,
            true,
        );
        reconstruct_component(
            pic,
            plane,
            source,
            cw,
            &prediction,
            block.x / 2,
            block.y / 2,
            c_size,
            // §8.6.1 with the writer's zero chroma QP offsets: the chroma
            // blocks quantize at the Table 8-10 mapping of the luma QP.
            quant.map(|qp| chroma_qp(qp, 0, BIT_DEPTH, CHROMA_ARRAY_TYPE)),
            component,
            pred_mode,
        );
    }
}

/// Gathers one coding unit's §8.5.3.3.2 prediction, one prediction partition
/// at a time, into the `size` x `size` buffer [`reconstruct_component`] codes
/// the residual against.
///
/// `chroma` halves the coding block, each partition inside it, and each
/// partition's motion vector, which is the 4:2:0 geometry. The partitions of a
/// coding unit tile it exactly, so every position of `prediction` is written.
fn gather_block_prediction(
    prediction: &mut [u8],
    size: usize,
    reference: Option<(&[u8], usize, usize)>,
    block: &BlockDecision,
    chroma: bool,
) {
    let scale = |v: usize| if chroma { v / 2 } else { v };
    let (x0, y0) = (scale(block.x), scale(block.y));
    for partition in &block.partitions {
        let (px, py) = (scale(partition.x), scale(partition.y));
        let (pw, ph) = (scale(partition.w).max(1), scale(partition.h).max(1));
        let (mv_x, mv_y) = if chroma {
            (partition.mv_x / 2, partition.mv_y / 2)
        } else {
            (partition.mv_x, partition.mv_y)
        };
        for row in 0..ph {
            let start = (py - y0 + row) * size + (px - x0);
            gather_prediction_row(
                &mut prediction[start..start + pw],
                reference,
                px,
                py + row,
                mv_x,
                mv_y,
            );
        }
    }
}

/// [`reconstruct_block`] for one colour component, over the square `size` x
/// `size` coding block whose prediction the caller has already gathered.
#[allow(clippy::too_many_arguments)]
fn reconstruct_component(
    pic: &mut Picture,
    plane: Plane,
    source: &[u8],
    src_stride: usize,
    prediction: &[u8],
    x: usize,
    y: usize,
    size: usize,
    quant: Option<u32>,
    component: TfComponent,
    pred_mode: PredMode,
) {
    let (w, h) = (size, size);
    debug_assert_eq!(prediction.len(), w * h);
    let Some(q_p) = quant else {
        // The residual is coded exactly, so the reconstruction is the source,
        // and the §8.6.6 add-and-clip of a whole row is a straight-line run
        // the vector kernel can take whole.
        let (dst, dst_stride) = pic.plane_mut(plane);
        for row in 0..h {
            let sy = y + row;
            let src_row = &source[sy * src_stride + x..sy * src_stride + x + w];
            let dst_row = &mut dst[sy * dst_stride + x..sy * dst_stride + x + w];
            recon_simd::reconstruct_row(dst_row, src_row, &prediction[row * w..row * w + w]);
        }
        return;
    };

    // §7.3.8.8: the residual of a *coding unit* is coded as square transform
    // blocks. The coding block is square and no smaller than the 4x4 floor in
    // either component, so the largest legal square tiles it exactly and no
    // block is ever clipped or zero-padded.
    let n_tbs = transform_block_size(w, h);
    debug_assert_eq!(w % n_tbs, 0);
    let mut pred_block = vec![0i32; n_tbs * n_tbs];
    let mut residual = vec![0i32; n_tbs * n_tbs];
    for by in (0..h).step_by(n_tbs) {
        for bx in (0..w).step_by(n_tbs) {
            for row in 0..n_tbs {
                let sy = y + by + row;
                for col in 0..n_tbs {
                    let predicted = i32::from(prediction[(by + row) * w + bx + col]);
                    pred_block[row * n_tbs + col] = predicted;
                    residual[row * n_tbs + col] =
                        i32::from(source[sy * src_stride + x + bx + col]) - predicted;
                }
            }
            // Forward transform → quantize → the decoder's own §8.6.2
            // reconstruction of the levels that survived.
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
            // The §8.6.6 add-and-clip of a transform block is `n_tbs`
            // straight-line runs over two `i32` operands, so each row goes
            // through the vector kernel whole rather than one `set_sample` at
            // a time.
            let (dst, dst_stride) = pic.plane_mut(plane);
            for row in 0..n_tbs {
                let start = (y + by + row) * dst_stride + x + bx;
                recon_simd::add_clip_row(
                    &mut dst[start..start + n_tbs],
                    &pred_block[row * n_tbs..(row + 1) * n_tbs],
                    &coded[row * n_tbs..(row + 1) * n_tbs],
                );
            }
        }
    }
}

/// Gathers the §8.5.3.3.2 prediction of one row of a partition contiguously,
/// so the add-and-clip that consumes it is a straight-line run.
///
/// The gather is a plain copy whenever the motion vector keeps the row inside
/// the reference, which is the common case; only a row that hangs off an edge
/// falls back to the per-sample reference-sample clamp. `prediction.len()` is
/// the run length.
fn gather_prediction_row(
    prediction: &mut [u8],
    reference: Option<(&[u8], usize, usize)>,
    x: usize,
    sy: usize,
    mv_x: i32,
    mv_y: i32,
) {
    let w = prediction.len();
    match reference {
        Some((reference, rw, rh)) => {
            let ry = (sy as i32 + mv_y).clamp(0, rh as i32 - 1) as usize;
            let base = ry * rw;
            let left = x as i32 + mv_x;
            if left >= 0 && left + w as i32 <= rw as i32 {
                let start = base + left as usize;
                prediction.copy_from_slice(&reference[start..start + w]);
            } else {
                for (col, sample) in prediction.iter_mut().enumerate() {
                    let rx = (left + col as i32).clamp(0, rw as i32 - 1) as usize;
                    *sample = reference[base + rx];
                }
            }
        }
        None => prediction.fill(NEUTRAL_LUMA as u8),
    }
}

/// The largest legal square transform block for a `w` x `h` coding block: the
/// largest power of two no greater than either side, clamped to the
/// §7.4.3.2.1 4..=32 transform-block range.
///
/// A coding block is square and at least 8x8 in luma, so both it and its 4:2:0
/// chroma halves are whole multiples of the result and the tiling is exact.
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
/// (§7.4.9.3: `(1 << (Min(bitDepth, 10) − 5)) − 1`), which is also the `cMax`
/// of its Table 9-43 truncated-Rice binarization — `pub(crate)` so the writer
/// codes against the same bound the search clamps to.
pub(crate) const SAO_OFFSET_MAX: i32 = 7;

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
    let pic = as_picture(recon);
    let grid = estimate_sao(&pic, src, true, true, lambda_q8, true);
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

/// A reconstruction's three planes as the [`Picture`] the SAO stage reads.
fn as_picture(recon: &ReconstructedPicture) -> Picture {
    let mut pic = Picture::new(
        recon.width,
        recon.height,
        CHROMA_ARRAY_TYPE,
        BIT_DEPTH,
        BIT_DEPTH,
    );
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
    pic
}

/// Encoder-side §7.3.8.3 SAO parameter estimation.
///
/// For each CTB this searches both §8.7.3.2 types against each other and
/// against leaving SAO off, and takes whichever wins on `D + lambda * R`:
///
/// * the four **edge-offset** classes, whose offsets are the per-category mean
///   errors between the source and the deblocked reconstruction, clamped to
///   the signalable range with the §7.4.9.3 inferred signs (categories 1 and
///   2 positive, 3 and 4 negative);
/// * the 32 **band-offset** positions, whose four consecutive bands take their
///   own per-band mean errors, clamped to the same magnitude but signalled
///   with a `sao_offset_sign` each, so an offset may point either way.
///
/// Both are the least-squares choice for their type, which is the reason this
/// stage is worth measuring: it reads every reconstructed sample of the
/// picture once per candidate class, plus once more to bin the samples the 32
/// band positions are all scored from.
///
/// The two types are compared under one score rather than a winner being
/// picked per type first, because they do not cost the same: band offset pays
/// four signs and five `sao_band_position` bins where edge offset pays two
/// class bins, so a band candidate has to buy more error to be worth the same.
///
/// The two chroma components are decided *together*, on the summed gain of one
/// shared type, because §7.4.9.3 infers `SaoTypeIdx[2]` and `SaoEoClass[2]`
/// from cIdx 1: a Cb and a Cr that picked different types, or different edge
/// classes, are not a bitstream. Each still carries its own four offsets and,
/// under band offset, its own `sao_band_position` — no position is inferred —
/// which is the whole of what the syntax gives cIdx 2 of its own.
///
/// `lambda` is the pair of §9 rate-distortion multipliers the candidates are
/// priced with — see [`SaoLambda`]. SAO costs per-CTB syntax
/// on every CTB it is enabled for, so a candidate is taken only when its SSE
/// reduction clears `lambda * bins`, the bins being the §9.3.3 binarization's
/// own count for the parameters that would be coded. At a zero `lambda` the
/// test degrades to "any reduction at all", which is what a caller that codes
/// no syntax for the decision wants.
fn estimate_sao(
    pic: &Picture,
    src: SourcePlanes<'_>,
    sao_luma: bool,
    sao_chroma: bool,
    lambda_q8: u32,
    band_search: bool,
) -> Vec<ResolvedSao> {
    let w_ctbs = src.width.div_ceil(CTB);
    let h_ctbs = src.height.div_ceil(CTB);
    let chroma_stride = src.width / 2;
    let mut grid = vec![ResolvedSao::off(); w_ctbs * h_ctbs];
    for ry in 0..h_ctbs {
        for rx in 0..w_ctbs {
            let addr = ry * w_ctbs + rx;
            let luma = sao_luma.then(|| {
                PlaneStats::gather(pic, Plane::Luma, src.y, src.width, rx, ry, band_search)
            });
            let chroma = sao_chroma.then(|| {
                [
                    PlaneStats::gather(pic, Plane::Cb, src.cb, chroma_stride, rx, ry, band_search),
                    PlaneStats::gather(pic, Plane::Cr, src.cr, chroma_stride, rx, ry, band_search),
                ]
            });
            // The structure this CTB would code for itself. Its luma and its
            // chroma pair are chosen independently because the bins
            // [`cell_bins`] counts for the two are separable, so the best
            // pair is the pair of bests.
            let mut standalone = ResolvedSao::off();
            if let Some(stats) = &luma {
                standalone.components[0] = best_luma_component(stats, lambda_q8, band_search);
            }
            if let Some([cb, cr]) = &chroma {
                let [best_cb, best_cr] = best_chroma_components(cb, cr, lambda_q8, band_search);
                standalone.components[1] = best_cb;
                standalone.components[2] = best_cr;
            }
            let left = (rx > 0).then(|| grid[addr - 1]);
            let above = (ry > 0).then(|| grid[addr - w_ctbs]);
            let score = |cell: &ResolvedSao| {
                let (mode, band) = coding_bins(cell, left, above);
                let gain = cell_gain(cell, luma.as_ref(), chroma.as_ref());
                rd_score(gain, mode + band, lambda_q8)
            };
            // "Off" first and the two merges before the structure, so that a
            // tie is broken towards the cell that codes the least. A merge
            // scoring exactly what the standalone structure scores is the
            // same trade for this CTB and a cheaper neighbour for the next
            // one, which is a preference the score itself cannot express
            // because it only sees this CTB.
            let mut best = ResolvedSao::off();
            let mut best_score = score(&best);
            for candidate in [left, above, Some(standalone)].into_iter().flatten() {
                let candidate_score = score(&candidate);
                if candidate_score > best_score {
                    best_score = candidate_score;
                    best = candidate;
                }
            }
            grid[addr] = best;
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
/// The `D + lambda * R` score of one SAO candidate: the SSE reduction it buys
/// less the §7.3.8.3 syntax it has to be signalled with, in the same units
/// the mode decision uses.
///
/// Every bin is priced at the one closed-form multiplier, because `bins` is
/// what [`coding_bins`] says `code_sao` will really write for the candidate
/// rather than what its structure would cost standalone. There is no second
/// multiplier: what a band component used to be charged beyond its own
/// syntax was the merge it forfeits, and the search now counts that merge
/// instead of pricing a constant in its place.
///
/// Positive means the candidate is worth coding at all; the largest score
/// wins, which is what lets edge offset, band offset, the two merges and
/// leaving SAO off be compared in one decision rather than several.
fn rd_score(gain: i64, bins: u64, lambda_q8: u32) -> i64 {
    gain - (bins * u64::from(lambda_q8) / 256) as i64
}

/// The widest the slice-level SAO decision's calibrated multiplier is allowed
/// to depart from `lambda_q8`, as a factor either way. It bounds how far a
/// two-point measurement of one picture is trusted over the closed form —
/// and, because it is a bound known before the measurement is taken, it is
/// also what lets `keeps_sao` skip the measurement whenever the decision is
/// already outside the band.
pub(super) const SAO_LAMBDA_BAND: u64 = 4;

/// The §7.3.8.3 bins one band-offset component costs that edge offset does
/// not: the five fixed-length `sao_band_position` bins and a
/// `sao_offset_sign` for every nonzero offset.
///
/// This is what makes band offset a different bet from edge offset at the
/// same gain — it pays five position bins and up to four sign bins where edge
/// offset pays two class bins and infers its signs — and it is the part of a
/// band candidate's rate [`SaoLambda::band_q8`] prices. The offsets' own
/// truncated-Rice bins are counted by [`offset_abs_bins`] exactly as edge
/// offset's are, and priced the same way, because they are the same kind of
/// bit.
pub(crate) fn band_syntax_bins(offsets: &[i32; 5]) -> u64 {
    5 + offsets[1..5].iter().filter(|&&o| o != 0).count() as u64
}

/// The §7.3.8.3 bins one cell's own `sao( )` structure costs, split into the
/// structure's own bins and the band-offset bins [`band_syntax_bins`] counts.
///
/// Absolute rather than marginal. The per-component searches
/// [`best_luma_component`] and [`best_chroma_components`] still score
/// marginally over leaving that component off, because there the alternative
/// really is this same CTB coding its own structure and the "off" bin is
/// common to both sides; [`coding_bins`] is where that stops being true and
/// the cell-level decision is taken on absolute bins.
pub(crate) fn cell_bins(cell: &ResolvedSao) -> (u64, u64) {
    let (mut mode, mut band) = (0u64, 0u64);
    // `sao_type_idx_luma`, then `sao_type_idx_chroma` for the pair — §7.4.9.3
    // infers cIdx 2's type from cIdx 1, so Cr codes none of its own.
    for (component, partner) in [
        (&cell.components[0], None),
        (&cell.components[1], Some(&cell.components[2])),
    ] {
        // Table 9-43 TR (cMax 2): bin 0 always, bin 1 only when it is on.
        mode += 1;
        if component.sao_type_idx == 0 {
            continue;
        }
        mode += 1;
        for c in std::iter::once(component).chain(partner) {
            mode += offset_abs_bins(&c.offset_val);
            if component.sao_type_idx == 1 {
                // Band offset: each component signals its own signs and its
                // own five-bin `sao_band_position`.
                band += band_syntax_bins(&c.offset_val);
            }
        }
        if component.sao_type_idx == 2 {
            // One `sao_eo_class` for the component or the pair.
            mode += 2;
        }
    }
    (mode, band)
}

/// What the writer will really spend on `cell` at this position, given what
/// its left and above neighbours resolved to: the §7.3.8.3 merge flags
/// `code_sao` codes for it, and the structure it codes only when neither
/// merge fires.
/// This is the quantity the per-CTB search is scored on, and what a fitted
/// band-syntax charge used to stand in for. `code_sao` merges on exact
/// equality of all three components, so a cell equal to a decided neighbour
/// costs one flag in place of its whole structure. Edge offset picks one of
/// four classes and lets §7.4.9.3 infer its signs, so two neighbouring CTBs
/// of one kind of content land on identical parameters routinely; band offset
/// picks one of 32 positions and four signed offsets per component, so two
/// neighbours essentially never do. A band component therefore costs its own
/// syntax *and* the merge it forfeits — for itself, and for whichever
/// neighbours would have merged with it — and none of that is a bin it codes.
///
/// This is the quantity the per-CTB search is scored on:
/// [`estimate_sao`] walks the grid in the raster order `code_sao` writes it
/// in, so the left and above cells are decided by the time this one is, and
/// every candidate — including the two merges themselves — is priced at what
/// the writer will really spend on it.
pub(crate) fn coding_bins(
    cell: &ResolvedSao,
    left: Option<ResolvedSao>,
    above: Option<ResolvedSao>,
) -> (u64, u64) {
    let mut mode = 0u64;
    let merge_left = left == Some(*cell);
    if left.is_some() {
        mode += 1;
    }
    let merge_up = !merge_left && above == Some(*cell);
    if above.is_some() && !merge_left {
        mode += 1;
    }
    if merge_left || merge_up {
        return (mode, 0);
    }
    let (structure_mode, band) = cell_bins(cell);
    (mode + structure_mode, band)
}

/// The bins `code_sao` will write for a whole grid, split the same way
/// [`cell_bins`] splits them.
pub(crate) fn grid_bins(grid: &[ResolvedSao], ctbs_x: usize) -> (u64, u64) {
    let (mut mode, mut band) = (0u64, 0u64);
    for (addr, cell) in grid.iter().enumerate() {
        let (rx, ry) = (addr % ctbs_x, addr / ctbs_x);
        let left = (rx > 0).then(|| grid[addr - 1]);
        let above = (ry > 0).then(|| grid[addr - ctbs_x]);
        let (m, b) = coding_bins(cell, left, above);
        mode += m;
        band += b;
    }
    (mode, band)
}

/// The per-band summed and counted error of one CTB of one plane, indexed by
/// the §8.7.3.2 band index `sample >> (bitDepth − 5)`.
///
/// Gathered once per CTB and reused by all 32 candidate band positions, since
/// a position only selects which four of these 32 bands are offset.
fn band_stats(
    pic: &Picture,
    plane: Plane,
    source: &[u8],
    src_stride: usize,
    rect: (usize, usize, usize, usize),
) -> ([i64; 32], [i64; 32]) {
    let (x0, y0, x1, y1) = rect;
    let (pw, _) = pic.plane_dims(plane);
    let samples = pic.plane(plane);
    // The kernel's band shift is fixed to this module's 8-bit geometry.
    debug_assert_eq!(BIT_DEPTH, 8, "band classification assumes 8-bit samples");
    let mut stats = BandStats::default();
    // One dispatched call for the whole CTB rather than one per row: #382
    // measured the per-row shape's `#[target_feature]` call overhead as the
    // reason the isolated kernel win did not reach this group. See
    // [`recon_simd::band_offset_rect`].
    recon_simd::band_offset_rect(
        &samples[y0 * pw + x0..],
        pw,
        &source[y0 * src_stride + x0..],
        src_stride,
        x1 - x0,
        y1 - y0,
        &mut stats,
    );
    (stats.sums, stats.counts)
}

/// The §7.4.9.3 offsets the four consecutive bands starting at
/// `band_position` would take, and the SSE reduction they buy.
///
/// Unlike edge offset, whose signs §7.4.9.3 infers from the category, band
/// offset codes a `sao_offset_sign` per nonzero offset — so the offset is the
/// band's mean error in whichever direction it points, clamped only to the
/// signalable magnitude.
fn band_offsets(sums: &[i64; 32], counts: &[i64; 32], band_position: u8) -> (i64, [i32; 5]) {
    let mut offsets = [0i32; 5];
    let mut gain = 0i64;
    for k in 0..4usize {
        // §8.7.3.2 equation 8-414: the four bands wrap the 32-band range.
        let band = (usize::from(band_position) + k) & 31;
        if counts[band] == 0 {
            continue;
        }
        let offset = div_round(sums[band], counts[band]).clamp(-SAO_OFFSET_MAX, SAO_OFFSET_MAX);
        offsets[k + 1] = offset;
        gain += offset_gain(sums[band], counts[band], offset);
    }
    (gain, offsets)
}

/// Every statistic one CTB of one plane can be scored on: the summed and
/// counted `source − reconstruction` error of each §8.7.3.2 edge category
/// under each of the four classes, and of each of the 32 bands.
///
/// Gathered whole, rather than alongside the winner as the search finds it,
/// because the search now scores candidates it did not generate. The two
/// §7.3.8.3 merges take whatever the left or above neighbour resolved to —
/// any class, any band position, and offsets chosen for a different CTB's
/// error — so a search that kept only its own candidates' statistics could
/// not say what a merge buys here.
struct PlaneStats {
    /// Per edge-offset class, the §8.7.3.2 per-category sums and counts
    /// [`edge_offsets`] chooses from.
    edge: [([i64; 5], [i64; 5]); 4],
    /// Per §8.7.3.2 band index, the same sums and counts. All zero when the
    /// caller cleared the band search, which is also when no cell in the grid
    /// can carry a band component for a merge to copy.
    band: ([i64; 32], [i64; 32]),
}

impl PlaneStats {
    /// Gather one CTB of one plane.
    fn gather(
        pic: &Picture,
        plane: Plane,
        source: &[u8],
        src_stride: usize,
        rx: usize,
        ry: usize,
        band_search: bool,
    ) -> Self {
        let rect = ctb_rect(pic, plane, rx, ry);
        Self {
            edge: std::array::from_fn(|eo_class| {
                edge_stats(pic, plane, source, src_stride, rect, eo_class as u8)
            }),
            band: if band_search {
                band_stats(pic, plane, source, src_stride, rect)
            } else {
                ([0; 32], [0; 32])
            },
        }
    }

    /// The SSE reduction `component` buys on this CTB, whether or not this
    /// CTB's own search would have proposed it — which is exactly what
    /// pricing a merge against a structure needs.
    fn gain(&self, component: &ResolvedSaoComponent) -> i64 {
        match component.sao_type_idx {
            // Band offset: the four consecutive bands §8.7.3.2 equation 8-414
            // wraps around the 32-band range, each with its own signed offset.
            1 => (0..4)
                .map(|k| {
                    let (sums, counts) = &self.band;
                    let band = (usize::from(component.band_position) + k) & 31;
                    offset_gain(sums[band], counts[band], component.offset_val[k + 1])
                })
                .sum(),
            // Edge offset: one §8.7.3.2 category per signalled offset, under
            // the class the component carries.
            2 => {
                let (sums, counts) = &self.edge[usize::from(component.eo_class)];
                (1..5)
                    .map(|category| {
                        offset_gain(
                            sums[category],
                            counts[category],
                            component.offset_val[category],
                        )
                    })
                    .sum()
            }
            _ => 0,
        }
    }
}

/// The SSE reduction of adding a constant `offset` to `count` samples whose
/// summed `source − reconstruction` error is `sum`: `2*o*s − n*o^2`.
fn offset_gain(sum: i64, count: i64, offset: i32) -> i64 {
    2 * i64::from(offset) * sum - count * i64::from(offset) * i64::from(offset)
}

/// What a whole candidate cell buys on this CTB, summed over the components
/// the caller is deciding.
fn cell_gain(
    cell: &ResolvedSao,
    luma: Option<&PlaneStats>,
    chroma: Option<&[PlaneStats; 2]>,
) -> i64 {
    let mut gain = luma.map_or(0, |stats| stats.gain(&cell.components[0]));
    if let Some([cb, cr]) = chroma {
        gain += cb.gain(&cell.components[1]) + cr.gain(&cell.components[2]);
    }
    gain
}

/// The best SAO component for the luma of one CTB, or an off component when
/// nothing earns the syntax it would be coded with.
///
/// The four §8.7.3.2 edge-offset classes and the 32 band positions are scored
/// against each other under one [`rd_score`], so the type is chosen by what it
/// buys net of what it costs rather than by picking a winner per type first.
/// This is the structure the CTB would code for itself; whether it is worth
/// coding at all against a merge is [`estimate_sao`]'s decision, not this
/// one's.
fn best_luma_component(
    stats: &PlaneStats,
    lambda_q8: u32,
    band_search: bool,
) -> ResolvedSaoComponent {
    let mut best = ResolvedSaoComponent::off();
    let mut best_score = 0i64;
    for eo_class in 0..4u8 {
        let (sums, counts) = &stats.edge[usize::from(eo_class)];
        let (gain, offsets) = edge_offsets(sums, counts);
        // One `sao_type_idx_luma` bin beyond the "off" bin, two
        // `sao_eo_class_luma` bins, and the offsets' own.
        let score = rd_score(gain, 1 + 2 + offset_abs_bins(&offsets), lambda_q8);
        if score > best_score {
            best_score = score;
            best = ResolvedSaoComponent {
                sao_type_idx: 2,
                offset_val: offsets,
                band_position: 0,
                eo_class,
            };
        }
    }
    if !band_search {
        return best;
    }
    let (sums, counts) = &stats.band;
    for band_position in 0..32u8 {
        let (gain, offsets) = band_offsets(sums, counts, band_position);
        // One `sao_type_idx_luma` bin beyond the "off" bin, then the band
        // path's own signs, position and offsets.
        let score = rd_score(
            gain,
            1 + offset_abs_bins(&offsets) + band_syntax_bins(&offsets),
            lambda_q8,
        );
        if score > best_score {
            best_score = score;
            best = ResolvedSaoComponent {
                sao_type_idx: 1,
                offset_val: offsets,
                band_position,
                eo_class: 0,
            };
        }
    }
    best
}

/// The best SAO components for the Cb and Cr of one CTB, sharing the one type
/// the §7.4.9.3 inference leaves them.
///
/// `SaoTypeIdx[2]` and `SaoEoClass[2]` are inferred from cIdx 1, so the two
/// components are decided together on the summed gain of one shared type —
/// and, for edge offset, one shared class. Band offset infers no position, so
/// each component still picks the four bands that suit it, exactly as each
/// picks its own four offsets.
fn best_chroma_components(
    cb_stats: &PlaneStats,
    cr_stats: &PlaneStats,
    lambda_q8: u32,
    band_search: bool,
) -> [ResolvedSaoComponent; 2] {
    let mut best = [ResolvedSaoComponent::off(); 2];
    let mut best_score = 0i64;
    for eo_class in 0..4u8 {
        let class = usize::from(eo_class);
        let (cb_gain, cb_offsets) = edge_offsets(&cb_stats.edge[class].0, &cb_stats.edge[class].1);
        let (cr_gain, cr_offsets) = edge_offsets(&cr_stats.edge[class].0, &cr_stats.edge[class].1);
        // One `sao_type_idx_chroma` bin beyond the "off" bin, two
        // `sao_eo_class_chroma` bins, and both components' offsets — cIdx 2
        // codes neither a type nor a class of its own.
        let bins = 1 + 2 + offset_abs_bins(&cb_offsets) + offset_abs_bins(&cr_offsets);
        let score = rd_score(cb_gain + cr_gain, bins, lambda_q8);
        if score > best_score {
            best_score = score;
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
    if !band_search {
        return best;
    }
    let cb = best_band_component(cb_stats, lambda_q8);
    let cr = best_band_component(cr_stats, lambda_q8);
    // The shared `sao_type_idx_chroma` bin is paid once for the pair.
    let bins = 1
        + offset_abs_bins(&cb.1)
        + offset_abs_bins(&cr.1)
        + band_syntax_bins(&cb.1)
        + band_syntax_bins(&cr.1);
    let score = rd_score(cb.0 + cr.0, bins, lambda_q8);
    if score > best_score {
        best = [
            ResolvedSaoComponent {
                sao_type_idx: 1,
                offset_val: cb.1,
                band_position: cb.2,
                eo_class: 0,
            },
            ResolvedSaoComponent {
                sao_type_idx: 1,
                offset_val: cr.1,
                band_position: cr.2,
                eo_class: 0,
            },
        ];
    }
    best
}

/// The band position of one chroma component that scores best on its own
/// share of the band path's rate, with the gain and offsets it buys.
///
/// The pair's shared `sao_type_idx_chroma` bin is not charged here, because it
/// is paid once for both components by the caller.
fn best_band_component(stats: &PlaneStats, lambda_q8: u32) -> (i64, [i32; 5], u8) {
    let (sums, counts) = &stats.band;
    let mut best = (0i64, [0i32; 5], 0u8);
    let mut best_score = i64::MIN;
    for band_position in 0..32u8 {
        let (gain, offsets) = band_offsets(sums, counts, band_position);
        let score = rd_score(
            gain,
            offset_abs_bins(&offsets) + band_syntax_bins(&offsets),
            lambda_q8,
        );
        if score > best_score {
            best_score = score;
            best = (gain, offsets, band_position);
        }
    }
    best
}

/// The §8.7.3.2 per-category summed and counted `source − reconstruction`
/// error of one CTB of one plane, under one edge-offset class.
fn edge_stats(
    pic: &Picture,
    plane: Plane,
    source: &[u8],
    src_stride: usize,
    rect: (usize, usize, usize, usize),
    eo_class: u8,
) -> ([i64; 5], [i64; 5]) {
    let (x0, y0, x1, y1) = rect;
    let (pw, ph) = pic.plane_dims(plane);
    let samples = pic.plane(plane);
    let (h0, v0, h1, v1) = crate::hevc::engine::sao::eo_pos(eo_class);
    // Per §8.7.3.2 category (1..4), the summed and counted error. The
    // neighbour bounds test is hoisted out of the sample loop and turned into
    // a row range: a sample is classifiable exactly when both of its
    // neighbours are inside the plane, and for a fixed class that is a
    // whole-row condition vertically and a contiguous run horizontally.
    // Everything left in the run is a straight-line accumulation the vector
    // kernel can take whole.
    let mut stats = EdgeStats::default();
    for y in y0..y1 {
        let (ay, by) = (y as i32 + v0, y as i32 + v1);
        if ay < 0 || by < 0 || ay >= ph as i32 || by >= ph as i32 {
            continue;
        }
        let lo = (x0 as i32).max(-h0).max(-h1);
        let hi = (x1 as i32).min(pw as i32 - h0).min(pw as i32 - h1);
        if hi <= lo {
            continue;
        }
        let (lo, run) = (lo as usize, (hi - lo) as usize);
        let a_start = (ay as usize) * pw + (lo as i32 + h0) as usize;
        let b_start = (by as usize) * pw + (lo as i32 + h1) as usize;
        recon_simd::edge_offset_row(
            &samples[y * pw + lo..y * pw + lo + run],
            &samples[a_start..a_start + run],
            &samples[b_start..b_start + run],
            &source[y * src_stride + lo..y * src_stride + lo + run],
            &mut stats,
        );
    }
    (stats.sums, stats.counts)
}

/// The §7.4.9.3 offsets one edge-offset class would take from those
/// statistics, and the SSE reduction they buy.
fn edge_offsets(sums: &[i64; 5], counts: &[i64; 5]) -> (i64, [i32; 5]) {
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
        gain += offset_gain(sums[category], counts[category], offset);
    }
    (gain, offsets)
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
    use crate::hevc::engine::encoder::rdo::{
        BlockDecision, DecisionConfig, PartitionDecision, decide_picture,
    };

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
    fn the_search_takes_band_offset_where_the_bias_is_a_value_range() {
        // The case §8.7.3.2 edge offset cannot reach: two flat regions, one of
        // which the "quantizer" biased 3 low. Every interior sample of a flat
        // region is edge category 0, so no edge class has anything to offset —
        // but every one of those samples sits in the same value band, which is
        // precisely what band offset selects on.
        let mut pic = Picture::new(W, H, CHROMA_ARRAY_TYPE, BIT_DEPTH, BIT_DEPTH);
        let (buf, _) = pic.plane_mut(Plane::Luma);
        for row in 0..H {
            for col in 0..W {
                buf[row * W + col] = if col < W / 2 { 100 } else { 200 };
            }
        }
        for plane in [Plane::Cb, Plane::Cr] {
            let (buf, _) = pic.plane_mut(plane);
            buf.fill(128);
        }
        let y: Vec<u8> = (0..W * H)
            .map(|i| if i % W < W / 2 { 103 } else { 200 })
            .collect();
        let chroma = vec![128u8; (W / 2) * (H / 2)];
        let src = SourcePlanes {
            y: &y,
            cb: &chroma,
            cr: &chroma,
            width: W,
            height: H,
        };

        let grid = estimate_sao(&pic, src, true, true, 32, true);
        // The first CTB lies wholly inside the biased region.
        let luma = grid[0].components[0];
        assert_eq!(
            luma.sao_type_idx, 1,
            "the search picked type {} where only band offset can reach the error",
            luma.sao_type_idx
        );
        // Band 12 is where sample 100 lives at 8-bit depth (100 >> 3), and the
        // four signalled bands wrap the 32-band range from `band_position`.
        let position = usize::from(luma.band_position);
        let k = (12 + 32 - position) % 32;
        assert!(
            k < 4,
            "band 12 is outside the four bands at position {position}"
        );
        assert_eq!(
            luma.offset_val[k + 1],
            3,
            "the band carrying the bias was offset by {} rather than its mean error",
            luma.offset_val[k + 1]
        );
    }

    /// The band-search stub is only a measurement instrument if it really
    /// removes the band half and nothing else. Both halves of that are
    /// asserted here: with the flag set the same picture takes band offset
    /// somewhere, and with it cleared no component in the whole grid does —
    /// so the difference the #382 benchmark subtracts is the band search's
    /// own cost and not a compiler-elided arm or a search that never reached
    /// the band candidates in the first place.
    #[test]
    fn clearing_the_band_search_leaves_no_band_offset_component() {
        let _guard = crate::simd::test_lock();
        let src = source(0);
        let decision = plan(&src, None);
        let cfg = ReconConfig {
            deblocking: true,
            sao_luma: true,
            sao_chroma: true,
            pcm_loop_filter_disabled: false,
            quantized_residual: true,
            qp: 37,
            ..ReconConfig::default()
        };

        let searched = reconstruct_picture(planes(&src), None, &decision, cfg);
        let stubbed = reconstruct_picture(
            planes(&src),
            None,
            &decision,
            ReconConfig {
                sao_band_search: false,
                ..cfg
            },
        );
        assert_ne!(
            searched.y, stubbed.y,
            "clearing the band search changed no reconstructed sample, so the              arm the benchmark subtracts costs nothing to remove"
        );

        // Read the grids directly, since the reconstructed planes only show
        // that *something* moved.
        let mut pic = Picture::new(
            searched.width,
            searched.height,
            CHROMA_ARRAY_TYPE,
            BIT_DEPTH,
            BIT_DEPTH,
        );
        for (plane, samples) in [
            (Plane::Luma, &searched.y),
            (Plane::Cb, &searched.cb),
            (Plane::Cr, &searched.cr),
        ] {
            let (buf, _) = pic.plane_mut(plane);
            for (dst, &s) in buf.iter_mut().zip(samples) {
                *dst = i32::from(s);
            }
        }
        let source_planes = planes(&src);
        let lambda = 0;
        let with = estimate_sao(&pic, source_planes, true, true, lambda, true);
        let without = estimate_sao(&pic, source_planes, true, true, lambda, false);
        assert!(
            with.iter()
                .any(|cell| cell.components.iter().any(|c| c.sao_type_idx == 1)),
            "no CTB of this picture takes band offset even with the search on"
        );
        assert!(
            without
                .iter()
                .all(|cell| cell.components.iter().all(|c| c.sao_type_idx != 1)),
            "a band-offset component survived with the band search cleared"
        );
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
    fn the_whole_reconstruction_is_identical_on_every_instruction_set() {
        // The §8.6.6 loop and the SAO parameter search both dispatch through
        // `hevc_recon`, so pinning each available instruction set in turn has
        // to leave the reconstructed picture — and the SAO decisions that
        // shaped it — bit for bit the same.
        let _guard = crate::simd::test_lock();
        let cfg = ReconConfig {
            deblocking: true,
            sao_luma: true,
            sao_chroma: true,
            pcm_loop_filter_disabled: false,
            ..ReconConfig::default()
        };
        let src = source(0);
        let next = source(3);
        crate::simd::set_override(Some(crate::simd::SimdIsa::Scalar));
        let reference = reconstruct_picture(planes(&src), None, &plan(&src, None), cfg);
        let expected_inter = reconstruct_picture(
            planes(&next),
            Some(&reference),
            &plan(&next, Some(&reference.y)),
            cfg,
        );
        for isa in crate::simd::available() {
            crate::simd::set_override(Some(isa));
            let intra = reconstruct_picture(planes(&src), None, &plan(&src, None), cfg);
            assert_eq!(intra, reference, "{} intra reconstruction", isa.name());
            let inter = reconstruct_picture(
                planes(&next),
                Some(&reference),
                &plan(&next, Some(&reference.y)),
                cfg,
            );
            assert_eq!(inter, expected_inter, "{} inter reconstruction", isa.name());
        }
        crate::simd::set_override(None);
    }

    #[test]
    fn the_quantized_reconstruction_is_identical_on_every_instruction_set_at_every_qp() {
        // The lossy writer's §8.6.6 add-and-clip dispatches through
        // `hevc_recon` too, over `i32` operands rather than the byte runs the
        // exact-residual path carries. Every QP the round-trip tests cover has
        // to reconstruct byte for byte the same on every instruction set, both
        // intra and predicting from a lossy reference.
        let _guard = crate::simd::test_lock();
        let src = source(0);
        let next = source(3);
        for qp in [12i32, 20, 26, 34, 37, 47, 51] {
            let cfg = ReconConfig {
                deblocking: true,
                sao_luma: true,
                sao_chroma: true,
                pcm_loop_filter_disabled: false,
                quantized_residual: true,
                sao_band_search: true,
                qp,
            };
            crate::simd::set_override(Some(crate::simd::SimdIsa::Scalar));
            let reference = reconstruct_picture(planes(&src), None, &plan(&src, None), cfg);
            let expected_inter = reconstruct_picture(
                planes(&next),
                Some(&reference),
                &plan(&next, Some(&reference.y)),
                cfg,
            );
            for isa in crate::simd::available() {
                crate::simd::set_override(Some(isa));
                let intra = reconstruct_picture(planes(&src), None, &plan(&src, None), cfg);
                assert_eq!(intra, reference, "{} intra at qp {qp}", isa.name());
                let inter = reconstruct_picture(
                    planes(&next),
                    Some(&reference),
                    &plan(&next, Some(&reference.y)),
                    cfg,
                );
                assert_eq!(inter, expected_inter, "{} inter at qp {qp}", isa.name());
            }
        }
        crate::simd::set_override(None);
    }

    /// A whole-picture decision whose every CTB is one coding unit split into
    /// `shapes`, given as `(dx, dy, w, h)` offsets inside the CTB, all with a
    /// zero motion vector.
    fn decision_of(shapes: &[(usize, usize, usize, usize)]) -> PictureDecision {
        let mut blocks = Vec::new();
        for y0 in (0..H).step_by(CTB) {
            for x0 in (0..W).step_by(CTB) {
                blocks.push(BlockDecision {
                    x: x0,
                    y: y0,
                    size: CTB,
                    partitions: shapes
                        .iter()
                        .map(|&(dx, dy, w, h)| PartitionDecision {
                            x: x0 + dx,
                            y: y0 + dy,
                            w,
                            h,
                            mv_x: 0,
                            mv_y: 0,
                            sad: 0,
                            satd: 0,
                            bit_cost: 0,
                            rd_cost: 0,
                        })
                        .collect(),
                    rd_cost: 0,
                    pcm_cost: u64::MAX,
                });
            }
        }
        PictureDecision {
            blocks,
            rd_cost: 0,
            pcm_blocks: 0,
        }
    }

    /// The 16x4-over-16x12 split, whose 4:2:0 chroma halves are 8x2 and 8x6 —
    /// neither a whole number of the 4x4 blocks §7.3.8.8 bottoms out at.
    const ASYMMETRIC: &[(usize, usize, usize, usize)] = &[(0, 0, CTB, 4), (0, 4, CTB, CTB - 4)];

    #[test]
    fn an_asymmetric_partitions_chroma_half_reconstructs_inside_the_plane() {
        // The residual is coded over the coding unit's own transform tree, so
        // an 8x2 chroma prediction partition never becomes a transform block
        // and nothing indexes past the end of the chroma plane.
        let src = source(0);
        let decision = decision_of(ASYMMETRIC);
        let cfg = ReconConfig {
            quantized_residual: true,
            qp: 32,
            ..ReconConfig::default()
        };
        let _guard = crate::simd::test_lock();
        crate::simd::set_override(Some(crate::simd::SimdIsa::Scalar));
        let expected = reconstruct_picture(planes(&src), None, &decision, cfg);
        assert_eq!(expected.y.len(), src.0.len());
        assert_eq!(expected.cb.len(), src.1.len());
        assert_eq!(expected.cr.len(), src.2.len());
        // Every transform block is now a whole one, but the add-and-clip still
        // goes through the vector kernel, so the result is pinned across
        // instruction sets.
        for isa in crate::simd::available() {
            crate::simd::set_override(Some(isa));
            let lossy = reconstruct_picture(planes(&src), None, &decision, cfg);
            assert_eq!(lossy, expected, "{}", isa.name());
        }
        crate::simd::set_override(None);
    }

    #[test]
    fn the_transform_tree_is_the_coding_units_however_the_prediction_is_partitioned() {
        // §7.3.8.8 hangs the transform tree off the coding unit, so a 16x16 CU
        // codes one 16x16 luma and two 8x8 chroma transform blocks no matter
        // how its prediction is split. These partitionings all predict the
        // same samples — every motion vector is zero and there is no reference
        // — so if the residual is coded over the CU they must reconstruct
        // identically. Tiling each *partition* instead did not: the asymmetric
        // shapes fell to 8x8 and 4x4 luma blocks, and their chroma halves to
        // zero-padded partial ones.
        let src = source(0);
        let cfg = ReconConfig {
            quantized_residual: true,
            qp: 32,
            ..ReconConfig::default()
        };
        let whole = reconstruct_picture(planes(&src), None, &decision_of(&[(0, 0, CTB, CTB)]), cfg);
        for shapes in [
            ASYMMETRIC,
            &[(0, 0, CTB, CTB - 4), (0, CTB - 4, CTB, 4)],
            &[(0, 0, 4, CTB), (4, 0, CTB - 4, CTB)],
            &[(0, 0, CTB / 2, CTB), (CTB / 2, 0, CTB / 2, CTB)],
            &[
                (0, 0, CTB / 2, CTB / 2),
                (CTB / 2, 0, CTB / 2, CTB / 2),
                (0, CTB / 2, CTB / 2, CTB / 2),
                (CTB / 2, CTB / 2, CTB / 2, CTB / 2),
            ],
        ] {
            let split = reconstruct_picture(planes(&src), None, &decision_of(shapes), cfg);
            assert_eq!(
                split, whole,
                "{shapes:?} did not reconstruct through the coding unit's transform tree"
            );
        }
    }

    /// A resolved cell carrying one luma band component with the given
    /// position, for the coded-rate tests below.
    fn band_cell(band_position: u8, offsets: [i32; 5]) -> ResolvedSao {
        let mut cell = ResolvedSao::off();
        cell.components[0] = ResolvedSaoComponent {
            sao_type_idx: 1,
            offset_val: offsets,
            band_position,
            eo_class: 0,
        };
        cell
    }

    /// The same, carrying one luma edge component of the given class.
    fn edge_cell(eo_class: u8, offsets: [i32; 5]) -> ResolvedSao {
        let mut cell = ResolvedSao::off();
        cell.components[0] = ResolvedSaoComponent {
            sao_type_idx: 2,
            offset_val: offsets,
            band_position: 0,
            eo_class,
        };
        cell
    }

    #[test]
    fn a_cell_equal_to_a_neighbour_costs_one_merge_flag_instead_of_its_structure() {
        // §7.3.8.3's merge, which is the whole of what `coding_bins` exists
        // to price: `code_sao` codes `sao_merge_left_flag` and returns,
        // whatever the structure would have been.
        let cell = band_cell(9, [0, 3, -2, 0, 1]);
        let (structure_mode, structure_band) = cell_bins(&cell);
        assert!(
            structure_mode + structure_band > 10,
            "a band cell's own structure is not the many bins the merge elides"
        );

        // First CTB of the picture: no neighbour exists, so no flag is coded
        // and the structure is unavoidable.
        assert_eq!(
            coding_bins(&cell, None, None),
            (structure_mode, structure_band)
        );
        // Equal to the left neighbour: one flag, and nothing else at all.
        assert_eq!(coding_bins(&cell, Some(cell), None), (1, 0));
        // Equal to the above neighbour but not the left one: the left flag is
        // coded as 0 first, then the up flag as 1.
        let other = edge_cell(1, [0, 1, 1, -1, -1]);
        assert_eq!(coding_bins(&cell, Some(other), Some(cell)), (2, 0));
    }

    #[test]
    fn a_cell_that_matches_neither_neighbour_pays_the_flags_it_declines() {
        // The other side of the same decision, and the one the search was
        // charging nothing for: a cell that cannot merge pays two declined
        // flags on top of its whole structure.
        let cell = band_cell(9, [0, 3, -2, 0, 1]);
        let other = edge_cell(1, [0, 1, 1, -1, -1]);
        let (mode, band) = cell_bins(&cell);
        assert_eq!(
            coding_bins(&cell, Some(other), Some(other)),
            (mode + 2, band)
        );
        // At the left edge of the picture there is only the up flag, and in
        // the top row only the left one.
        assert_eq!(coding_bins(&cell, None, Some(other)), (mode + 1, band));
        assert_eq!(coding_bins(&cell, Some(other), None), (mode + 1, band));
    }

    #[test]
    fn a_band_cell_is_charged_the_syntax_an_edge_cell_does_not_code() {
        // `cell_bins` splits the §7.3.8.3 bins the way `SaoLambda` prices
        // them, so what lands in the band half has to be exactly the five
        // `sao_band_position` bins and the sign of each nonzero offset.
        let offsets = [0, 3, -2, 0, 1];
        let (band_mode, band_band) = cell_bins(&band_cell(9, offsets));
        let (edge_mode, edge_band) = cell_bins(&edge_cell(1, offsets));
        assert_eq!(edge_band, 0, "an edge cell codes no band syntax");
        assert_eq!(band_band, 5 + 3, "five position bins and three signs");
        // Against edge offset's two `sao_eo_class_luma` bins, which are
        // `mode_q8` bins like the offsets' own.
        assert_eq!(band_mode + 2, edge_mode);
    }

    #[test]
    fn an_off_cell_still_costs_the_type_bin_every_component_pays() {
        // Why `cell_bins` is absolute where the search's own scoring is
        // marginal: leaving SAO off is not free once a merge is on the table,
        // because a merged CTB does not code these two bins either.
        assert_eq!(cell_bins(&ResolvedSao::off()), (2, 0));
        assert_eq!(
            coding_bins(&ResolvedSao::off(), Some(ResolvedSao::off()), None),
            (1, 0)
        );
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
    fn coding_blocks_tile_into_the_largest_legal_square_transform_block() {
        // Every coding block this encoder codes, and every 4:2:0 chroma half
        // of one, is a whole number of its own transform blocks.
        for size in [8usize, 16, 32, 64] {
            let n_tbs = transform_block_size(size, size);
            assert_eq!(size % n_tbs, 0, "luma {size}");
            let c_size = size / 2;
            assert_eq!(
                c_size % transform_block_size(c_size, c_size),
                0,
                "chroma {c_size}"
            );
        }
        assert_eq!(transform_block_size(16, 16), 16);
        assert_eq!(transform_block_size(8, 8), 8);
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
