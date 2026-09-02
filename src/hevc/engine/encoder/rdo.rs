//! Encoder-side rate-distortion candidate search.
//!
//! This module is deliberately independent of the current PCM bitstream writer:
//! it builds the block, partition, and motion decisions that a compressed HEVC
//! writer can consume, while the bootstrap encoder can already run the same
//! deterministic search to exercise the SIMD distortion kernels end to end.

use crate::hevc::engine::binarization::INTRA_PRED_MODE_MAX;
use crate::hevc::engine::encoder::rdcost;
use crate::hevc::engine::encoder::residual::{
    ResidualBinSink, ResidualWriteParams, has_coded_levels, write_residual_coding,
};
use crate::hevc::engine::residual::ResidualElement;

const CTB: usize = 16;
const NEUTRAL_LUMA: u8 = 128;

/// Distortion backend used by the RDO search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DistortionBackend {
    /// Runtime-dispatched backend selected by [`rdcost`].
    Dispatched,
    /// Portable reference backend, used by bitstream identity tests.
    Scalar,
}

/// Per-picture decision-search controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecisionConfig {
    /// Luma QP used to derive the RDO lambda.
    pub qp: i32,
    /// Whole-pel search radius for the coarse inter stage.
    pub search_radius: i32,
    /// Distortion implementation.
    pub backend: DistortionBackend,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            qp: 26,
            search_radius: 4,
            backend: DistortionBackend::Dispatched,
        }
    }
}

/// One evaluated prediction partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PartitionDecision {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub mv_x: i32,
    pub mv_y: i32,
    pub sad: u32,
    pub satd: u32,
    pub bit_cost: u32,
    pub rd_cost: u64,
}

/// The selected candidate for one coding block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BlockDecision {
    pub x: usize,
    pub y: usize,
    pub size: usize,
    pub partitions: Vec<PartitionDecision>,
    pub rd_cost: u64,
    pub pcm_cost: u64,
}

/// A deterministic picture-level decision plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PictureDecision {
    pub blocks: Vec<BlockDecision>,
    pub rd_cost: u64,
    pub pcm_blocks: usize,
}

/// Searches the current encoder-side candidate space.
///
/// The current writer can emit only PCM CUs, so the returned plan keeps `pcm_*`
/// accounting alongside the best predictive candidate. Running this before
/// writing the access unit wires real frame data through [`rdcost::sad`] and
/// [`rdcost::satd`] without changing the established lossless PCM bitstream.
pub(crate) fn decide_picture(
    y: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    reference_y: Option<&[u8]>,
    cfg: DecisionConfig,
) -> PictureDecision {
    assert!(width > 0 && height > 0);
    assert!(width % CTB == 0 && height % CTB == 0);
    assert!(stride >= width);
    assert!(y.len() >= (height - 1) * stride + width);
    if let Some(reference) = reference_y {
        assert!(reference.len() >= (height - 1) * stride + width);
    }

    let mut blocks = Vec::with_capacity((width / CTB) * (height / CTB));
    let mut rd_cost = 0u64;
    for by in (0..height).step_by(CTB) {
        for bx in (0..width).step_by(CTB) {
            let block = decide_block(y, stride, width, height, reference_y, bx, by, cfg);
            rd_cost = rd_cost.saturating_add(block.rd_cost);
            blocks.push(block);
        }
    }
    PictureDecision {
        pcm_blocks: blocks.len(),
        blocks,
        rd_cost,
    }
}

#[allow(clippy::too_many_arguments)]
fn decide_block(
    y: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    reference_y: Option<&[u8]>,
    bx: usize,
    by: usize,
    cfg: DecisionConfig,
) -> BlockDecision {
    let lambda = lambda_q8(cfg.qp);
    let mut best_partitions = Vec::new();
    let mut best_cost = u64::MAX;

    for partitions in candidate_partitions(CTB) {
        let mut current = Vec::with_capacity(partitions.len());
        let mut current_cost = split_bit_cost(partitions.len()) * u64::from(lambda) / 256;
        for &(ox, oy, pw, ph) in *partitions {
            let decision = decide_partition(
                y,
                stride,
                width,
                height,
                reference_y,
                bx + ox,
                by + oy,
                pw,
                ph,
                cfg,
                lambda,
            );
            current_cost = current_cost.saturating_add(decision.rd_cost);
            current.push(decision);
        }
        if current_cost < best_cost {
            best_cost = current_cost;
            best_partitions = current;
        }
    }

    let pcm_cost = pcm_block_bit_cost(CTB) * u64::from(lambda) / 256;
    BlockDecision {
        x: bx,
        y: by,
        size: CTB,
        partitions: best_partitions,
        rd_cost: best_cost.min(pcm_cost),
        pcm_cost,
    }
}

#[allow(clippy::too_many_arguments)]
fn decide_partition(
    y: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    reference_y: Option<&[u8]>,
    x: usize,
    y0: usize,
    w: usize,
    h: usize,
    cfg: DecisionConfig,
    lambda: u32,
) -> PartitionDecision {
    let src = &y[y0 * stride + x..];
    let (mv_x, mv_y, pred, pred_stride, sad) =
        motion_search(src, stride, width, height, reference_y, x, y0, w, h, cfg);
    let satd = metric_satd(src, stride, pred, pred_stride, w, h, cfg.backend);
    let bit_cost = motion_bit_cost(mv_x, mv_y, w, h);
    let rd_cost = u64::from(satd) + u64::from(bit_cost) * u64::from(lambda) / 256;
    PartitionDecision {
        x,
        y: y0,
        w,
        h,
        mv_x,
        mv_y,
        sad,
        satd,
        bit_cost,
        rd_cost,
    }
}

#[allow(clippy::too_many_arguments)]
fn motion_search<'a>(
    src: &[u8],
    stride: usize,
    width: usize,
    height: usize,
    reference_y: Option<&'a [u8]>,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    cfg: DecisionConfig,
) -> (i32, i32, &'a [u8], usize, u32) {
    let Some(reference) = reference_y else {
        let neutral = neutral_block();
        let sad = metric_sad(src, stride, neutral, CTB, w, h, cfg.backend);
        return (0, 0, neutral, CTB, sad);
    };

    // The whole-pel stage scores `(2 * radius + 1)^2` predictions of one source block, which
    // is where this search spends its time and the only place it has vector width to give:
    // the blocks are at most 16 wide, so a per-candidate SAD leaves a 256-bit register doing
    // a 128-bit job (#370). Candidates are gathered into fixed batches and scored together by
    // `rdcost::sad_batch`, which packs several of them into one `_mm256_sad_epu8`. The scan
    // order and the tie-break below are unchanged, and the batched SADs are the same values
    // the per-candidate call returns, so the winning motion vector - and the mode decision
    // built on it - is bit-identical to what the unbatched search picked.
    let radius = cfg.search_radius.max(0);
    let mut best = (0, 0, u32::MAX);
    let mut offsets = [0usize; SAD_BATCH];
    let mut mvs = [(0i32, 0i32); SAD_BATCH];
    let mut sads = [0u32; SAD_BATCH];
    let mut pending = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let rx = x as i32 + dx;
            let ry = y as i32 + dy;
            if rx < 0 || ry < 0 || rx as usize + w > width || ry as usize + h > height {
                continue;
            }
            offsets[pending] = ry as usize * stride + rx as usize;
            mvs[pending] = (dx, dy);
            pending += 1;
            if pending == SAD_BATCH {
                metric_sad_batch(
                    src,
                    stride,
                    reference,
                    &offsets,
                    stride,
                    w,
                    h,
                    &mut sads,
                    cfg.backend,
                );
                take_best(&mut best, &mvs, &sads);
                pending = 0;
            }
        }
    }
    if pending > 0 {
        metric_sad_batch(
            src,
            stride,
            reference,
            &offsets[..pending],
            stride,
            w,
            h,
            &mut sads[..pending],
            cfg.backend,
        );
        take_best(&mut best, &mvs[..pending], &sads[..pending]);
    }

    let mut refined = best;
    for (dx, dy) in [
        (best.0, best.1),
        (best.0 - 1, best.1),
        (best.0 + 1, best.1),
        (best.0, best.1 - 1),
        (best.0, best.1 + 1),
    ] {
        let rx = x as i32 + dx;
        let ry = y as i32 + dy;
        if rx < 0 || ry < 0 || rx as usize + w > width || ry as usize + h > height {
            continue;
        }
        let pred = &reference[ry as usize * stride + rx as usize..];
        let satd = metric_satd(src, stride, pred, stride, w, h, cfg.backend);
        if satd < refined.2
            || (satd == refined.2 && mv_order(dx, dy) < mv_order(refined.0, refined.1))
        {
            refined = (dx, dy, satd);
        }
    }

    let pred =
        &reference[(y as i32 + refined.1) as usize * stride + (x as i32 + refined.0) as usize..];
    let sad = metric_sad(src, stride, pred, stride, w, h, cfg.backend);
    (refined.0, refined.1, pred, stride, sad)
}

/// Candidates gathered before one batched SAD call.
///
/// A multiple of both AVX2 candidate-group sizes (two 16-wide candidates per vector, four
/// narrower ones), so a full batch never leaves a group-sized remainder on the slower
/// per-candidate path, and small enough to stay on the stack next to the search.
const SAD_BATCH: usize = 16;

/// Folds one batch of scored candidates into the running best, keeping the unbatched
/// search's tie-break: a lower SAD wins, and an equal SAD goes to the smaller `mv_order`.
fn take_best(best: &mut (i32, i32, u32), mvs: &[(i32, i32)], sads: &[u32]) {
    for (&(dx, dy), &sad) in mvs.iter().zip(sads) {
        if sad < best.2 || (sad == best.2 && mv_order(dx, dy) < mv_order(best.0, best.1)) {
            *best = (dx, dy, sad);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn metric_sad_batch(
    src: &[u8],
    src_stride: usize,
    plane: &[u8],
    offsets: &[usize],
    pred_stride: usize,
    w: usize,
    h: usize,
    out: &mut [u32],
    backend: DistortionBackend,
) {
    match backend {
        DistortionBackend::Dispatched => rdcost::sad_batch(
            src, src_stride, plane, offsets, pred_stride, w, h, out,
        ),
        DistortionBackend::Scalar => {
            for (slot, &offset) in out.iter_mut().zip(offsets) {
                *slot = rdcost::sad_scalar(
                    src,
                    src_stride,
                    &plane[offset..],
                    pred_stride,
                    w,
                    h,
                );
            }
        }
    }
}

fn metric_sad(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    w: usize,
    h: usize,
    backend: DistortionBackend,
) -> u32 {
    match backend {
        DistortionBackend::Dispatched => rdcost::sad(src, src_stride, pred, pred_stride, w, h),
        DistortionBackend::Scalar => rdcost::sad_scalar(src, src_stride, pred, pred_stride, w, h),
    }
}

fn metric_satd(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    w: usize,
    h: usize,
    backend: DistortionBackend,
) -> u32 {
    match backend {
        DistortionBackend::Dispatched => rdcost::satd(src, src_stride, pred, pred_stride, w, h),
        DistortionBackend::Scalar => rdcost::satd_scalar(src, src_stride, pred, pred_stride, w, h),
    }
}

fn candidate_partitions(size: usize) -> &'static [&'static [(usize, usize, usize, usize)]] {
    const SHAPES_4: &[&[(usize, usize, usize, usize)]] = &[&[(0, 0, 4, 4)]];
    const SHAPES_8: &[&[(usize, usize, usize, usize)]] = &[
        &[(0, 0, 8, 8)],
        &[(0, 0, 8, 4), (0, 4, 8, 4)],
        &[(0, 0, 4, 8), (4, 0, 4, 8)],
        &[(0, 0, 4, 4), (4, 0, 4, 4), (0, 4, 4, 4), (4, 4, 4, 4)],
    ];
    const SHAPES_16: &[&[(usize, usize, usize, usize)]] = &[
        &[(0, 0, 16, 16)],
        &[(0, 0, 16, 8), (0, 8, 16, 8)],
        &[(0, 0, 8, 16), (8, 0, 8, 16)],
        &[(0, 0, 16, 4), (0, 4, 16, 12)],
        &[(0, 0, 16, 12), (0, 12, 16, 4)],
        &[(0, 0, 4, 16), (4, 0, 12, 16)],
        &[(0, 0, 12, 16), (12, 0, 4, 16)],
        &[(0, 0, 8, 8), (8, 0, 8, 8), (0, 8, 8, 8), (8, 8, 8, 8)],
    ];
    const SHAPES_32: &[&[(usize, usize, usize, usize)]] = &[
        &[(0, 0, 32, 32)],
        &[(0, 0, 32, 16), (0, 16, 32, 16)],
        &[(0, 0, 16, 32), (16, 0, 16, 32)],
        &[(0, 0, 32, 8), (0, 8, 32, 24)],
        &[(0, 0, 32, 24), (0, 24, 32, 8)],
        &[(0, 0, 8, 32), (8, 0, 24, 32)],
        &[(0, 0, 24, 32), (24, 0, 8, 32)],
        &[
            (0, 0, 16, 16),
            (16, 0, 16, 16),
            (0, 16, 16, 16),
            (16, 16, 16, 16),
        ],
    ];
    const SHAPES_64: &[&[(usize, usize, usize, usize)]] = &[
        &[(0, 0, 64, 64)],
        &[(0, 0, 64, 32), (0, 32, 64, 32)],
        &[(0, 0, 32, 64), (32, 0, 32, 64)],
        &[(0, 0, 64, 16), (0, 16, 64, 48)],
        &[(0, 0, 64, 48), (0, 48, 64, 16)],
        &[(0, 0, 16, 64), (16, 0, 48, 64)],
        &[(0, 0, 48, 64), (48, 0, 16, 64)],
        &[
            (0, 0, 32, 32),
            (32, 0, 32, 32),
            (0, 32, 32, 32),
            (32, 32, 32, 32),
        ],
    ];
    match size {
        4 => SHAPES_4,
        8 => SHAPES_8,
        16 => SHAPES_16,
        32 => SHAPES_32,
        64 => SHAPES_64,
        _ => &[],
    }
}

/// The intra luma mode chosen for one block, with the cost that chose it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IntraModeDecision {
    /// The selected Table 8-1 `predModeIntra` (0..=34).
    pub mode: u8,
    /// The winning mode's SATD against the source block.
    pub satd: u32,
    /// `satd + lambda * signalling bits`, the value that selected it.
    pub rd_cost: u64,
}

/// Rough-mode-decision pass over the 35 Table 8-1 intra luma modes for one
/// square block: returns the `shortlist` cheapest, best first.
///
/// `predict` returns the §8.4.4.2.1 prediction for a mode, so the caller
/// supplies the reference samples it will actually code against — for the
/// residual writer that is the partially reconstructed picture, which is what
/// makes the decision consistent with the reconstruction the writer returns.
///
/// Distortion is the prediction's SATD, which needs no transform per mode, and
/// the rate term is the §7.3.8.5 luma mode signalling: a most-probable mode
/// costs `prev_intra_luma_pred_flag` plus its `mpm_idx` bins, and any other
/// mode costs the flag plus the five `rem_intra_luma_pred_mode` bins.
/// `candidates` is the §8.4.2 `candModeList`. Ties go to the lower mode index,
/// so the search is deterministic.
///
/// Ranking on the prediction alone is only an approximation of what the block
/// costs once it is quantized, so a caller that can afford it re-scores this
/// shortlist on the reconstruction it will actually code, keeping
/// [`intra_mode_bit_cost`] as the rate half of that second pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn shortlist_intra_luma_modes(
    source: &[u8],
    source_stride: usize,
    n_tbs: usize,
    candidates: [u8; 3],
    qp: i32,
    backend: DistortionBackend,
    shortlist: usize,
    mut predict: impl FnMut(u8) -> Vec<i32>,
) -> Vec<IntraModeDecision> {
    // The rough pass measures a sum of absolute differences, not squared
    // error, so it trades against the square root of the SSD-domain lambda.
    let lambda = lambda_satd_q8(qp);
    let mut pred = vec![0u8; n_tbs * n_tbs];
    let mut ranked = Vec::with_capacity(usize::from(INTRA_PRED_MODE_MAX) + 1);
    for mode in 0..=INTRA_PRED_MODE_MAX {
        let samples = predict(mode);
        for (dst, &src) in pred.iter_mut().zip(samples.iter()) {
            *dst = src as u8;
        }
        let satd = metric_satd(source, source_stride, &pred, n_tbs, n_tbs, n_tbs, backend);
        let bits = u64::from(intra_mode_bit_cost(mode, candidates));
        ranked.push(IntraModeDecision {
            mode,
            satd,
            rd_cost: u64::from(satd).saturating_add(bits * u64::from(lambda) / 256),
        });
    }
    // `sort_by_key` is stable, so equal costs keep the lower mode index.
    ranked.sort_by_key(|decision| decision.rd_cost);
    ranked.truncate(shortlist.max(1));
    ranked
}

/// A [`ResidualBinSink`] that writes nothing and counts what it was asked to
/// write — the rate half of a full rate-distortion cost, without an
/// arithmetic coder or a bitstream.
struct BinCounter {
    bins: u32,
}

impl ResidualBinSink for BinCounter {
    fn decision(&mut self, _element: ResidualElement, _ctx_inc: u32, _bin: u8) {
        self.bins += 1;
    }

    fn bypass(&mut self, _bin: u8) {
        self.bins += 1;
    }
}

/// Estimated rate, in bits, of the residual one transform block would code:
/// the number of CABAC bins §7.3.8.11 `residual_coding( )` emits for `levels`,
/// counted by running the real writer's walk against a sink that only tallies.
///
/// Counting bins rather than modelling each one's arithmetic-coded length
/// charges a context-coded bin a full bit, which overstates the well-modelled
/// ones — a `sig_coeff_flag` in a context that has settled costs a fraction of
/// a bit. What it preserves is the ordering: every bin the writer emits is
/// counted exactly once, in the same walk, so two candidate residuals for the
/// same block size and component are compared on the same scale. That is the
/// property a mode decision needs, and it is what the levels themselves cost,
/// as opposed to the mode signalling [`intra_mode_bit_cost`] covers.
///
/// A block whose levels are all zero codes no `residual_coding( )` at all —
/// the decoder infers it from `cbf == 0` — so its residual rate is zero. The
/// `cbf` bin itself is not counted here: it is coded for every block either
/// way, so it cannot separate two candidates.
///
/// # Panics
/// Panics through [`write_residual_coding`] if `levels` does not match
/// `params.log2_trafo_size`.
#[must_use]
pub(crate) fn residual_rate_bits(levels: &[i32], params: &ResidualWriteParams) -> u32 {
    if !has_coded_levels(levels) {
        return 0;
    }
    let mut counter = BinCounter { bins: 0 };
    write_residual_coding(&mut counter, params, levels);
    counter.bins
}

/// §7.3.8.5 bin count for signalling one luma intra mode: the
/// `prev_intra_luma_pred_flag` bin plus either the TR (`cMax` 2) `mpm_idx`
/// bins or the five FL `rem_intra_luma_pred_mode` bins.
pub(crate) fn intra_mode_bit_cost(mode: u8, candidates: [u8; 3]) -> u32 {
    match candidates.iter().position(|&c| c == mode) {
        Some(0) => 2,
        Some(_) => 3,
        None => 6,
    }
}

/// The rough pass's lambda: [`lambda_q8`] taken into the SATD domain, where
/// distortion is a first-order metric rather than a squared one.
pub(crate) fn lambda_satd_q8(qp: i32) -> u32 {
    let lambda = f64::from(lambda_q8(qp)) / 256.0;
    (lambda.sqrt() * 256.0).round().max(1.0) as u32
}

/// The lambda the searches above trade distortion against rate with,
/// `0.57 * 2 ^ ( ( QP - 12 ) / 3 )` in Q8 fixed point.
pub(crate) fn lambda_q8(qp: i32) -> u32 {
    let qp = qp.clamp(0, 51);
    let q = 2f64.powf((qp as f64 - 12.0) / 3.0);
    (0.57 * q * 256.0).round().max(1.0) as u32
}

fn split_bit_cost(partitions: usize) -> u64 {
    match partitions {
        1 => 1,
        2 => 5,
        4 => 9,
        _ => 13,
    }
}

fn motion_bit_cost(mv_x: i32, mv_y: i32, w: usize, h: usize) -> u32 {
    let mv = exp_golomb_bits(mv_x.unsigned_abs()) + exp_golomb_bits(mv_y.unsigned_abs());
    let sign = u32::from(mv_x != 0) + u32::from(mv_y != 0);
    let residual_class = ((w * h) / 16).max(1) as u32;
    2 + mv + sign + residual_class
}

fn pcm_block_bit_cost(size: usize) -> u64 {
    let luma = size * size;
    let chroma = luma / 2;
    (luma + chroma) as u64 * 8 + 4
}

fn exp_golomb_bits(v: u32) -> u32 {
    let code_num = v + 1;
    2 * (32 - code_num.leading_zeros()) - 1
}

fn mv_order(x: i32, y: i32) -> (i32, i32, i32) {
    (x.abs() + y.abs(), y.abs(), x.abs())
}

fn neutral_block() -> &'static [u8] {
    static BLOCK: [u8; CTB * CTB] = [NEUTRAL_LUMA; CTB * CTB];
    &BLOCK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(width: usize, height: usize, dx: usize, dy: usize) -> Vec<u8> {
        let mut out = vec![0; width * height];
        for y in 0..height {
            for x in 0..width {
                out[y * width + x] = (16 + ((x + dx) * 3 + (y + dy) * 5) % 220) as u8;
            }
        }
        out
    }

    #[test]
    fn decision_search_uses_matching_scalar_and_dispatched_costs() {
        let src = plane(32, 32, 0, 0);
        let reference = plane(32, 32, 1, 0);
        let dispatched = decide_picture(
            &src,
            32,
            32,
            32,
            Some(&reference),
            DecisionConfig::default(),
        );
        let scalar = decide_picture(
            &src,
            32,
            32,
            32,
            Some(&reference),
            DecisionConfig {
                backend: DistortionBackend::Scalar,
                ..DecisionConfig::default()
            },
        );
        assert_eq!(dispatched, scalar);
        assert!(
            dispatched
                .blocks
                .iter()
                .all(|block| !block.partitions.is_empty())
        );
        assert!(dispatched.blocks.iter().any(|block| {
            block
                .partitions
                .iter()
                .any(|partition| partition.sad > 0 && partition.satd > 0)
        }));
    }

    /// The batched whole-pel search has to pick the same motion vectors the per-candidate
    /// search picked, including where several candidates tie and the `mv_order` tie-break
    /// decides. The reference here is an unbatched scalar search written out in full - the
    /// whole-pel stage and the SATD refinement that consumes its winner - and `motion_search`
    /// is held against it on every instruction set the host offers.
    #[test]
    fn the_batched_motion_search_picks_what_the_per_candidate_search_picks() {
        const N: usize = 48;
        const SHAPES: &[(usize, usize, usize, usize)] = &[
            (0, 0, 16, 16),
            (0, 0, 8, 8),
            (0, 0, 4, 4),
            (0, 0, 16, 8),
            (0, 0, 4, 16),
            (4, 4, 12, 12),
        ];
        let _guard = crate::simd::test_lock();
        let src = plane(N, N, 0, 0);
        let reference = plane(N, N, 2, 1);
        let cfg = DecisionConfig::default();

        let inside = |rx: i32, ry: i32, w: usize, h: usize| {
            rx >= 0 && ry >= 0 && rx as usize + w <= N && ry as usize + h <= N
        };
        let mut expected = Vec::new();
        for by in (0..N).step_by(CTB) {
            for bx in (0..N).step_by(CTB) {
                for &(ox, oy, w, h) in SHAPES {
                    let (x, y) = (bx + ox, by + oy);
                    let block = &src[y * N + x..];
                    let mut best = (0i32, 0i32, u32::MAX);
                    for dy in -cfg.search_radius..=cfg.search_radius {
                        for dx in -cfg.search_radius..=cfg.search_radius {
                            let (rx, ry) = (x as i32 + dx, y as i32 + dy);
                            if !inside(rx, ry, w, h) {
                                continue;
                            }
                            let sad = rdcost::sad_scalar(
                                block,
                                N,
                                &reference[ry as usize * N + rx as usize..],
                                N,
                                w,
                                h,
                            );
                            if sad < best.2
                                || (sad == best.2 && mv_order(dx, dy) < mv_order(best.0, best.1))
                            {
                                best = (dx, dy, sad);
                            }
                        }
                    }
                    let mut refined = best;
                    for (dx, dy) in [
                        (best.0, best.1),
                        (best.0 - 1, best.1),
                        (best.0 + 1, best.1),
                        (best.0, best.1 - 1),
                        (best.0, best.1 + 1),
                    ] {
                        let (rx, ry) = (x as i32 + dx, y as i32 + dy);
                        if !inside(rx, ry, w, h) {
                            continue;
                        }
                        let satd = rdcost::satd_scalar(
                            block,
                            N,
                            &reference[ry as usize * N + rx as usize..],
                            N,
                            w,
                            h,
                        );
                        if satd < refined.2
                            || (satd == refined.2
                                && mv_order(dx, dy) < mv_order(refined.0, refined.1))
                        {
                            refined = (dx, dy, satd);
                        }
                    }
                    let sad = rdcost::sad_scalar(
                        block,
                        N,
                        &reference[(y as i32 + refined.1) as usize * N
                            + (x as i32 + refined.0) as usize..],
                        N,
                        w,
                        h,
                    );
                    expected.push((refined.0, refined.1, sad));
                }
            }
        }

        for isa in crate::simd::available() {
            crate::simd::set_override(Some(isa));
            let mut got = Vec::new();
            for by in (0..N).step_by(CTB) {
                for bx in (0..N).step_by(CTB) {
                    for &(ox, oy, w, h) in SHAPES {
                        let (x, y) = (bx + ox, by + oy);
                        let (mv_x, mv_y, _, _, sad) = motion_search(
                            &src[y * N + x..],
                            N,
                            N,
                            N,
                            Some(&reference),
                            x,
                            y,
                            w,
                            h,
                            cfg,
                        );
                        got.push((mv_x, mv_y, sad));
                    }
                }
            }
            assert_eq!(got, expected, "batched whole-pel search on {isa:?}");
        }
        crate::simd::set_override(None);
    }

    /// The residual rate estimate has to order two candidate residuals the
    /// same way the bitstream does, which is the only property a mode decision
    /// asks of it.
    #[test]
    fn the_residual_rate_estimate_orders_blocks_the_way_the_writer_does() {
        use crate::hevc::engine::cabac::init_type;
        use crate::hevc::engine::ctx_init::SliceContexts;
        use crate::hevc::engine::encoder::bitwriter::BitWriter;
        use crate::hevc::engine::encoder::cabac::CabacEncoder;
        use crate::hevc::engine::encoder::residual::EngineResidualBinSink;
        use crate::hevc::engine::scan::ScanIdx;

        let params = ResidualWriteParams {
            log2_trafo_size: 4,
            is_chroma: false,
            scan_idx: ScanIdx::Diagonal,
        };
        // A block the decoder infers from `cbf == 0` codes no
        // `residual_coding( )` at all, so it costs nothing to carry.
        assert_eq!(residual_rate_bits(&[0i32; 256], &params), 0);

        // Progressively more expensive residuals: one DC level, the same
        // level pushed further from the DC corner, a spread of levels, and
        // large levels that spill into `coeff_abs_level_remaining`.
        let block = |fill: &dyn Fn(usize, usize) -> i32| -> Vec<i32> {
            (0..256).map(|i| fill(i % 16, i / 16)).collect()
        };
        let candidates = [
            block(&|x, y| i32::from(x == 0 && y == 0)),
            block(&|x, y| i32::from(x == 3 && y == 2)),
            block(&|x, y| if x < 4 && y < 4 { 1 } else { 0 }),
            block(&|x, y| if x < 4 && y < 4 { 40 } else { 0 }),
        ];

        // What each one really costs: the arithmetic-coded length the writer
        // emits from a freshly initialized I-slice context bank.
        let coded_bits = |levels: &[i32]| -> usize {
            let mut w = BitWriter::new();
            let mut cabac = CabacEncoder::new();
            let mut ctxs = SliceContexts::init(init_type(2, false), 26);
            write_residual_coding(
                &mut EngineResidualBinSink {
                    writer: &mut w,
                    cabac: &mut cabac,
                    contexts: &mut ctxs.residual,
                },
                &params,
                levels,
            );
            w.finish().len() * 8
        };

        let estimates: Vec<u32> = candidates
            .iter()
            .map(|levels| residual_rate_bits(levels, &params))
            .collect();
        let coded: Vec<usize> = candidates.iter().map(|levels| coded_bits(levels)).collect();
        for i in 1..candidates.len() {
            assert!(
                estimates[i] > estimates[i - 1],
                "candidate {i} estimated {} bins against candidate {}'s {}",
                estimates[i],
                i - 1,
                estimates[i - 1]
            );
            assert!(
                coded[i] >= coded[i - 1],
                "the estimate ordered candidates {} and {i} the writer does not: {:?} vs {:?}",
                i - 1,
                estimates,
                coded
            );
        }
    }

    #[test]
    fn candidate_table_covers_hevc_block_and_partition_shapes() {
        for size in [4usize, 8, 16, 32, 64] {
            let shapes = candidate_partitions(size);
            assert!(!shapes.is_empty(), "{size}x{size} candidates");
            for shape in shapes {
                let area: usize = shape.iter().map(|&(_, _, w, h)| w * h).sum();
                assert_eq!(area, size * size, "{size}x{size} partition area");
                for &(x, y, w, h) in *shape {
                    assert!(w % 4 == 0 && h % 4 == 0);
                    assert!(x + w <= size && y + h <= size);
                }
            }
        }
        assert!(
            candidate_partitions(64)
                .iter()
                .flat_map(|shape| shape.iter())
                .any(|&(_, _, w, h)| w != h),
            "asymmetric partitions are included"
        );
    }

    /// End-to-end encoder-side RDO throughput over a 1080p-sized luma picture.
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn bench_picture_decision_search() {
        let width = 1920;
        let height = 1088;
        let src = plane(width, height, 0, 0);
        let reference = plane(width, height, 2, 1);
        let run = |backend| {
            let start = std::time::Instant::now();
            let decision = decide_picture(
                &src,
                width,
                width,
                height,
                Some(&reference),
                DecisionConfig {
                    backend,
                    ..DecisionConfig::default()
                },
            );
            (start.elapsed(), decision.rd_cost)
        };
        let (scalar_elapsed, scalar_cost) = run(DistortionBackend::Scalar);
        let (simd_elapsed, simd_cost) = run(DistortionBackend::Dispatched);
        assert_eq!(scalar_cost, simd_cost);
        println!(
            "hevc encoder RDO search: scalar={:.3}s dispatched={:.3}s speedup={:.2}x",
            scalar_elapsed.as_secs_f64(),
            simd_elapsed.as_secs_f64(),
            scalar_elapsed.as_secs_f64() / simd_elapsed.as_secs_f64()
        );
    }
}
