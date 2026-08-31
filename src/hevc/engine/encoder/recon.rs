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
//! ## What "reconstruction" means for the current writer
//!
//! The bitstream writer in [`crate::hevc::engine::encoder::pcm`] still codes
//! every coding unit as a `pcm_flag == 1` PCM block, so the residual it codes
//! is exact and the reconstruction of a block is bit-identical to its source
//! samples. The loop here is nevertheless the real one — predict, add residual,
//! clip — rather than a plane copy, because that is the *only* difference
//! between this and a lossy writer: quantize the residual before
//! [`reconstruct_partition`] adds it back and the reconstructed reference
//! starts diverging from the source with no other change to this module. It is
//! also what makes the in-loop filter stage below meaningful, since those
//! filters run on the reconstruction, not on the source.
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
use crate::hevc::engine::encoder::recon_simd::{self, EdgeStats};
use crate::hevc::engine::motion::{MotionCell, MotionField};
use crate::hevc::engine::picture::{Picture, Plane};
use crate::hevc::engine::sao::{ResolvedSao, ResolvedSaoComponent};

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
    /// reads.
    pub qp: i32,
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
            );
            field.fill_rect(
                partition.x,
                partition.y,
                partition.w,
                partition.h,
                MotionCell {
                    is_intra: coded_as_pcm,
                    // The residual is coded exactly, so every block that
                    // differs from its prediction carries coded levels — the
                    // §8.7.2.4 `cbf` test.
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
        let grid = estimate_sao(&pic, src, cfg);
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
/// residual against it, and add the residual back (§8.6.6
/// `recSamples = Clip1( predSamples + resSamples )`).
///
/// The residual is `source − prediction` because this writer codes it
/// losslessly. A writer that quantizes would round-trip the residual through
/// its forward and inverse transform here; nothing else about the loop would
/// change, and the reconstruction would then differ from `src`, which is the
/// entire reason the reference picture is kept separately at all.
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
    );
    // 4:2:0 — the chroma partition is the luma one halved, and the motion
    // vector with it (§8.5.3.3.3.3 whole-sample part; this search is
    // whole-pel, so the halved vector needs no fractional interpolation).
    let (cw, chh) = (src.width / 2, src.height / 2);
    for (plane, source, reference_plane) in [
        (Plane::Cb, src.cb, reference.map(|r| r.cb.as_slice())),
        (Plane::Cr, src.cr, reference.map(|r| r.cr.as_slice())),
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
) {
    // The prediction of a row is gathered once, contiguously, so the §8.6.6
    // add-and-clip that follows is a straight-line run the vector kernel can
    // take whole. The gather is a plain copy whenever the motion vector keeps
    // the row inside the reference, which is the common case; only a row that
    // hangs off an edge falls back to the per-sample §8.5.3.3.2 clamp.
    let mut prediction = vec![0u8; w];
    let (dst, dst_stride) = pic.plane_mut(plane);
    for row in 0..h {
        let sy = y + row;
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
        let src_row = &source[sy * src_stride + x..sy * src_stride + x + w];
        let dst_row = &mut dst[sy * dst_stride + x..sy * dst_stride + x + w];
        recon_simd::reconstruct_row(dst_row, src_row, &prediction);
    }
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

/// The largest magnitude a `sao_offset_abs` can carry at 8-bit depth
/// (§7.4.9.3: `(1 << (Min(bitDepth, 10) − 5)) − 1`).
const SAO_OFFSET_MAX: i32 = 7;

/// Encoder-side §7.3.8.3 SAO parameter estimation.
///
/// For each CTB and each component this picks the edge-offset class whose
/// per-category mean error between the source and the deblocked reconstruction
/// reduces the sum of squared errors the most, and leaves SAO off for that CTB
/// when no class does. The offsets are the per-category mean errors clamped to
/// the signalable range with the §7.4.9.3 inferred edge-offset signs
/// (categories 1 and 2 positive, 3 and 4 negative), which is the standard
/// least-squares choice for edge offset and the reason this stage is worth
/// measuring: it reads every reconstructed sample of the picture once per
/// candidate class.
fn estimate_sao(pic: &Picture, src: SourcePlanes<'_>, cfg: ReconConfig) -> Vec<ResolvedSao> {
    let w_ctbs = src.width.div_ceil(CTB);
    let h_ctbs = src.height.div_ceil(CTB);
    let mut grid = vec![ResolvedSao::off(); w_ctbs * h_ctbs];
    let components: [(Plane, &[u8], bool); 3] = [
        (Plane::Luma, src.y, cfg.sao_luma),
        (Plane::Cb, src.cb, cfg.sao_chroma),
        (Plane::Cr, src.cr, cfg.sao_chroma),
    ];
    for (c_idx, (plane, source, enabled)) in components.into_iter().enumerate() {
        if !enabled {
            continue;
        }
        let (pw, ph) = pic.plane_dims(plane);
        let step = if plane == Plane::Luma { CTB } else { CTB / 2 };
        for ry in 0..h_ctbs {
            for rx in 0..w_ctbs {
                let (x0, y0) = (rx * step, ry * step);
                let (x1, y1) = ((x0 + step).min(pw), (y0 + step).min(ph));
                grid[ry * w_ctbs + rx].components[c_idx] =
                    best_edge_offset(pic, plane, source, pw, x0, y0, x1, y1);
            }
        }
    }
    grid
}

/// The best §8.7.3.2 edge-offset component for one CTB of one plane, or an
/// off component when none of the four classes reduces the distortion.
#[allow(clippy::too_many_arguments)]
fn best_edge_offset(
    pic: &Picture,
    plane: Plane,
    source: &[u8],
    src_stride: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> ResolvedSaoComponent {
    let (pw, ph) = pic.plane_dims(plane);
    let samples = pic.plane(plane);
    let mut best = ResolvedSaoComponent::off();
    let mut best_gain = 0i64;
    for eo_class in 0..4u8 {
        let (h0, v0, h1, v1) = crate::hevc::engine::sao::eo_pos(eo_class);
        // Per §8.7.3.2 category (1..4), the summed and counted error. The
        // neighbour bounds test is hoisted out of the sample loop and turned
        // into a row range: a sample is classifiable exactly when both of its
        // neighbours are inside the plane, and for a fixed class that is a
        // whole-row condition vertically and a contiguous run horizontally.
        // Everything left in the run is a straight-line accumulation the
        // vector kernel can take whole.
        let mut stats = EdgeStats::default();
        for y in y0..y1 {
            let (ay, by) = (y as i32 + v0, y as i32 + v1);
            if ay < 0 || by < 0 || ay >= ph as i32 || by >= ph as i32 {
                continue;
            }
            let lo = (x0 as i32).max(-h0).max(-h1).max(0);
            let hi = (x1 as i32).min(pw as i32 - h0).min(pw as i32 - h1);
            if hi <= lo {
                continue;
            }
            let (lo, run) = (lo as usize, (hi - lo) as usize);
            let here = &samples[y * pw + lo..y * pw + lo + run];
            let a_start = (ay as usize) * pw + (lo as i32 + h0) as usize;
            let b_start = (by as usize) * pw + (lo as i32 + h1) as usize;
            recon_simd::edge_offset_row(
                here,
                &samples[a_start..a_start + run],
                &samples[b_start..b_start + run],
                &source[y * src_stride + lo..y * src_stride + lo + run],
                &mut stats,
            );
        }
        let (sums, counts) = (stats.sums, stats.counts);
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
        if gain > best_gain {
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
    fn every_ctb_gets_a_deblocking_descriptor_with_picture_edges_excluded() {
        let cus = deblock_descriptors(W, H, 26);
        assert_eq!(cus.len(), (W / CTB) * (H / CTB));
        assert!(!cus[0].filter_left && !cus[0].filter_top);
        assert!(cus[1].filter_left && !cus[1].filter_top);
        let second_row = &cus[W / CTB];
        assert!(!second_row.filter_left && second_row.filter_top);
    }
}
