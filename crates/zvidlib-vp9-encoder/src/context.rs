//! The adaptive probability state a VP9 decoder carries between frames: the
//! frame context a frame codes with, the symbol counts it gathers, and the
//! backward adaptation that merges those counts into the context the next
//! frame starts from (VP9 section 8.4, libvpx's `vp9_adapt_coef_probs`,
//! `vp9_adapt_mode_probs` and `vp9_adapt_mv_probs`).
//!
//! The context holds only the probabilities this encoder's coding tools read.
//! The rest (compound references, switchable filters) is never coded, counted
//! or updated, so it keeps its default value in the decoder too.

use super::bitwriter::{BitCost, BoolSink, bit_cost};
use super::frame::{
    INTER_MODE_TREE, INTRA_MODE_TREE, MV_CLASS_TREE, MV_FP_TREE, MV_JOINT_TREE, PARTITION_TREE,
    write_magnitude,
};
use super::tables::{
    IF_UV_MODE_PROBS, IF_Y_MODE_PROBS, INTER_MODE_PROBS, INTRA_INTER_PROBS, PARTITION_PROBS,
    SINGLE_REF_PROBS, SKIP_PROBS, TX_PROBS_8X8, TX_PROBS_16X16, TX_PROBS_32X32,
};
use zvidlib_vp9_decoder::vp9_dec::tables as shared;

/// One transform size's coefficient model probabilities:
/// `[plane type 2][reference 2][band 6][context 6][node 3]`.
pub(super) type CoefProbs = [[[[[u8; 3]; 6]; 6]; 2]; 2];

/// One transform size's token counts, laid out like [`CoefProbs`]: ZERO, ONE,
/// larger and end-of-block tokens per context.
type CoefCounts = [[[[[u32; 4]; 6]; 6]; 2]; 2];

/// One transform size's end-of-block node counts per context.
type EobCounts = [[[[u32; 6]; 6]; 2]; 2];

/// One transform size's counts of the tokens past ONE per context:
/// TWO, THREE, FOUR and the six categories.
type LargerTokenCounts = [[[[[u32; 9]; 6]; 6]; 2]; 2];

/// The smallest magnitude of each token past ONE, as [`LargerTokenCounts`]
/// orders them.
pub(super) const LARGER_TOKEN_MAGNITUDES: [u32; 9] = [2, 3, 4, 5, 7, 11, 19, 35, 67];

/// The probability a frame signals whether it updates each probability with
/// (`DIFF_UPDATE_PROB`).
pub(super) const DIFF_UPDATE_PROB: u8 = 252;

/// The probabilities of one motion vector component.
#[derive(Clone, Copy)]
pub(super) struct MvComponentProbs {
    pub(super) sign: u8,
    pub(super) classes: [u8; 10],
    pub(super) class0: u8,
    pub(super) bits: [u8; 10],
    pub(super) class0_fp: [[u8; 3]; 2],
    pub(super) fp: [u8; 3],
    /// The eighth-sample bit of class 0 and of the larger classes.
    pub(super) class0_hp: u8,
    pub(super) hp: u8,
}

/// Row (vertical) then column (horizontal) component defaults.
const DEFAULT_MV_COMPONENT_PROBS: [MvComponentProbs; 2] = [
    MvComponentProbs {
        sign: 128,
        classes: [224, 144, 192, 168, 192, 176, 192, 198, 198, 245],
        class0: 216,
        bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
        class0_fp: [[128, 128, 64], [96, 112, 64]],
        fp: [64, 96, 64],
        class0_hp: 160,
        hp: 128,
    },
    MvComponentProbs {
        sign: 128,
        classes: [216, 128, 176, 160, 176, 176, 192, 198, 198, 208],
        class0: 208,
        bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
        class0_fp: [[128, 128, 64], [96, 112, 64]],
        fp: [64, 96, 64],
        class0_hp: 160,
        hp: 128,
    },
];

/// The probabilities a frame codes its partitions, modes, transform sizes,
/// motion vectors and coefficient tokens with. Key frames read their
/// partitions and modes from fixed tables instead.
#[derive(Clone)]
pub(super) struct FrameContext {
    /// Indexed by transform size, 4x4 to 32x32.
    pub(super) coef: [CoefProbs; 4],
    /// Transform size probabilities for blocks whose largest transform is
    /// 8x8 (`[context 2][node 1]`), 16x16 (`[context 2][node 2]`) and 32x32
    /// (`[context 2][node 3]`).
    pub(super) tx_8x8: [u8; 2],
    pub(super) tx_16x16: [u8; 4],
    pub(super) tx_32x32: [u8; 6],
    pub(super) skip: [u8; 3],
    /// `[mode context 7][node 3]`.
    pub(super) inter_mode: [u8; 21],
    pub(super) intra_inter: [u8; 4],
    /// `[context 5][bit 2]`.
    pub(super) single_ref: [u8; 10],
    /// `[block size group 4][node 9]`.
    pub(super) y_mode: [u8; 36],
    /// `[luma mode 10][node 9]`.
    pub(super) uv_mode: [u8; 90],
    /// `[context 16][node 3]`.
    pub(super) partition: [u8; 48],
    pub(super) mv_joints: [u8; 3],
    pub(super) mv: [MvComponentProbs; 2],
}

impl Default for FrameContext {
    fn default() -> Self {
        Self {
            coef: [
                shared::DEFAULT_COEF_PROBS_4X4,
                shared::DEFAULT_COEF_PROBS_8X8,
                shared::DEFAULT_COEF_PROBS_16X16,
                shared::DEFAULT_COEF_PROBS_32X32,
            ],
            tx_8x8: TX_PROBS_8X8,
            tx_16x16: TX_PROBS_16X16,
            tx_32x32: TX_PROBS_32X32,
            skip: SKIP_PROBS,
            inter_mode: INTER_MODE_PROBS,
            intra_inter: INTRA_INTER_PROBS,
            single_ref: SINGLE_REF_PROBS,
            y_mode: IF_Y_MODE_PROBS,
            uv_mode: IF_UV_MODE_PROBS,
            partition: PARTITION_PROBS,
            mv_joints: [32, 64, 96],
            mv: DEFAULT_MV_COMPONENT_PROBS,
        }
    }
}

/// One motion vector component's symbol counts.
#[derive(Clone, Copy, Default)]
pub(super) struct MvComponentCounts {
    pub(super) sign: [u32; 2],
    pub(super) classes: [u32; 11],
    pub(super) class0: [u32; 2],
    pub(super) bits: [[u32; 2]; 10],
    pub(super) class0_fp: [[u32; 4]; 2],
    pub(super) fp: [u32; 4],
    pub(super) class0_hp: [u32; 2],
    pub(super) hp: [u32; 2],
}

/// The symbols one frame coded, counted as the decoder counts them while it
/// reads the frame.
#[derive(Clone, Default)]
pub(super) struct FrameCounts {
    /// Per transform size and coefficient context: ZERO, ONE, larger and
    /// end-of-block tokens.
    pub(super) coef: [CoefCounts; 4],
    /// Per transform size and coefficient context: how often the
    /// end-of-block node was read.
    pub(super) eob_branch: [EobCounts; 4],
    /// Per transform size and coefficient context: the tokens past ONE,
    /// which the decoder does not count. Their probabilities are modelled
    /// from the third node's, so the encoder weighs updates of that node by
    /// every one of them.
    pub(super) larger_tokens: [LargerTokenCounts; 4],
    /// `[largest transform - 1][context][transform size]`.
    pub(super) tx: [[[u32; 4]; 2]; 3],
    pub(super) skip: [[u32; 2]; 3],
    /// `[mode context][mode - NEARESTMV]`.
    pub(super) inter_mode: [[u32; 4]; 7],
    pub(super) intra_inter: [[u32; 2]; 4],
    /// The first single-reference bit; the second is never coded because
    /// every inter block predicts from `LAST_FRAME`.
    pub(super) single_ref: [[u32; 2]; 5],
    pub(super) y_mode: [[u32; 10]; 4],
    pub(super) uv_mode: [[u32; 10]; 10],
    pub(super) partition: [[u32; 4]; 16],
    pub(super) mv_joints: [u32; 4],
    pub(super) mv: [MvComponentCounts; 2],
}

impl FrameContext {
    /// The context a frame that coded with `self` and gathered `counts` saves
    /// for the next frame. Intra frames adapt only the coefficient
    /// probabilities; the frame after a key frame adapts them faster.
    /// Transform size probabilities adapt only when the frame selected
    /// transform sizes per block (`TX_MODE_SELECT`), and the eighth-sample
    /// bits only when it allowed high-precision vectors.
    pub(super) fn adapted(
        &self,
        counts: &FrameCounts,
        intra: bool,
        after_key: bool,
        tx_select: bool,
        allow_high_precision_mv: bool,
    ) -> Self {
        // COEF_MAX_UPDATE_FACTOR(_KEY, _AFTER_KEY) and COEF_COUNT_SAT.
        let update_factor = if !intra && after_key { 128 } else { 112 };
        let mut next = self.clone();
        let contexts = next.coef.iter_mut().flatten().flatten().flatten().flatten();
        let token_counts = counts.coef.iter().flatten().flatten().flatten().flatten();
        let eob_counts = counts
            .eob_branch
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .flatten();
        for ((probs, &[zero, one, more, end]), &eob_branch) in
            contexts.zip(token_counts).zip(eob_counts)
        {
            let branches = [[end, eob_branch - end], [zero, one + more], [one, more]];
            for (probability, [left, right]) in probs.iter_mut().zip(branches) {
                *probability = merge(*probability, left, right, 24, update_factor);
            }
        }
        if intra {
            return next;
        }

        if tx_select {
            for context in 0..2 {
                let [c0, c1, ..] = counts.tx[0][context];
                next.tx_8x8[context] = merge_mode(next.tx_8x8[context], [c0, c1]);
                let [c0, c1, c2, _] = counts.tx[1][context];
                let probs = &mut next.tx_16x16[context * 2..context * 2 + 2];
                for (probability, branch) in probs.iter_mut().zip([[c0, c1 + c2], [c1, c2]]) {
                    *probability = merge_mode(*probability, branch);
                }
                let [c0, c1, c2, c3] = counts.tx[2][context];
                let probs = &mut next.tx_32x32[context * 3..context * 3 + 3];
                let branches = [[c0, c1 + c2 + c3], [c1, c2 + c3], [c2, c3]];
                for (probability, branch) in probs.iter_mut().zip(branches) {
                    *probability = merge_mode(*probability, branch);
                }
            }
        }
        for (probability, &counts) in next.intra_inter.iter_mut().zip(&counts.intra_inter) {
            *probability = merge_mode(*probability, counts);
        }
        for (context, &counts) in counts.single_ref.iter().enumerate() {
            next.single_ref[context * 2] = merge_mode(next.single_ref[context * 2], counts);
        }
        for (probs, counts) in next.inter_mode.chunks_mut(3).zip(&counts.inter_mode) {
            merge_tree(&INTER_MODE_TREE, probs, counts);
        }
        for (probs, counts) in next.y_mode.chunks_mut(9).zip(&counts.y_mode) {
            merge_tree(&INTRA_MODE_TREE, probs, counts);
        }
        for (probs, counts) in next.uv_mode.chunks_mut(9).zip(&counts.uv_mode) {
            merge_tree(&INTRA_MODE_TREE, probs, counts);
        }
        for (probs, counts) in next.partition.chunks_mut(3).zip(&counts.partition) {
            merge_tree(&PARTITION_TREE, probs, counts);
        }
        for (probability, &counts) in next.skip.iter_mut().zip(&counts.skip) {
            *probability = merge_mode(*probability, counts);
        }

        merge_tree(&MV_JOINT_TREE, &mut next.mv_joints, &counts.mv_joints);
        for (probs, counts) in next.mv.iter_mut().zip(&counts.mv) {
            probs.sign = merge_mode(probs.sign, counts.sign);
            merge_tree(&MV_CLASS_TREE, &mut probs.classes, &counts.classes);
            probs.class0 = merge_mode(probs.class0, counts.class0);
            for (probability, &counts) in probs.bits.iter_mut().zip(&counts.bits) {
                *probability = merge_mode(*probability, counts);
            }
            for (fp, counts) in probs.class0_fp.iter_mut().zip(&counts.class0_fp) {
                merge_tree(&MV_FP_TREE, fp, counts);
            }
            merge_tree(&MV_FP_TREE, &mut probs.fp, &counts.fp);
            if allow_high_precision_mv {
                probs.class0_hp = merge_mode(probs.class0_hp, counts.class0_hp);
                probs.hp = merge_mode(probs.hp, counts.hp);
            }
        }
        next
    }
}

impl FrameContext {
    /// The coefficient probabilities a frame whose tokens `counts` counted
    /// should code with, as forward updates of `self`'s signalled in its
    /// compressed header, for the first `tx_sizes` transform sizes (libvpx's
    /// `update_coef_probs_common`, two-loop search). Each probability moves
    /// to the value between its own and the counts' that saves the most
    /// bits once its update is paid for, and a transform size takes its
    /// updates only when together they save more than declining every one
    /// costs; otherwise it keeps `self`'s probabilities.
    pub(super) fn coef_updates(&self, counts: &FrameCounts, tx_sizes: usize) -> [CoefProbs; 4] {
        let keep = bit_cost(false, DIFF_UPDATE_PROB);
        let update = bit_cost(true, DIFF_UPDATE_PROB);
        let mut coded = self.coef;
        for (tx_size, coded) in coded.iter_mut().enumerate().take(tx_sizes) {
            let mut probs = *coded;
            let mut savings = 0.0;
            for (plane_type, reference, band, context) in coef_contexts() {
                let [zero, one, more, end] =
                    counts.coef[tx_size][plane_type][reference][band][context];
                let eob_branch = counts.eob_branch[tx_size][plane_type][reference][band][context];
                let larger = &counts.larger_tokens[tx_size][plane_type][reference][band][context];
                let model = &mut probs[plane_type][reference][band][context];
                let branches = [[end, eob_branch - end], [zero, one + more], [one, more]];
                for (node, [left, right]) in branches.into_iter().enumerate() {
                    let cost = |probability: u8| {
                        if node < 2 {
                            return f64::from(left) * bit_cost(false, probability)
                                + f64::from(right) * bit_cost(true, probability);
                        }
                        // The pivot: ONE against every larger token, each of
                        // which also pays its path through the model.
                        let mut bits = f64::from(left) * bit_cost(false, probability);
                        for (&count, &magnitude) in larger.iter().zip(&LARGER_TOKEN_MAGNITUDES) {
                            if count != 0 {
                                let mut sink = BitCost::default();
                                write_magnitude(&mut sink, magnitude, probability);
                                bits += f64::from(count) * sink.0;
                            }
                        }
                        bits
                    };
                    let old = model[node];
                    let target = observed(left, right).unwrap_or(old);
                    let old_bits = cost(old);
                    let mut best = (old, 0.0);
                    let step: i16 = if target > old { -1 } else { 1 };
                    let mut candidate = i16::from(target);
                    while candidate != i16::from(old) {
                        let probability = candidate as u8;
                        let saving = old_bits
                            - cost(probability)
                            - (diff_update_bits(probability, old) + update - keep);
                        if saving > best.1 {
                            best = (probability, saving);
                        }
                        candidate += step;
                    }
                    if best.0 == old {
                        savings -= keep;
                    } else {
                        model[node] = best.0;
                        savings += best.1 - keep;
                    }
                }
            }
            if savings > 0.0 {
                *coded = probs;
            }
        }
        coded
    }
}

/// Every coefficient context a frame codes, as `(plane type, reference,
/// band, context)`: the first band has only three.
pub(super) fn coef_contexts() -> impl Iterator<Item = (usize, usize, usize, usize)> {
    (0..2).flat_map(|plane_type| {
        (0..2).flat_map(move |reference| {
            (0..6).flat_map(move |band| {
                (0..if band == 0 { 3 } else { 6 })
                    .map(move |context| (plane_type, reference, band, context))
            })
        })
    })
}

/// The probability of a zero bit that `left` zeros and `right` ones imply
/// (`get_binary_prob`), or `None` without symbols.
fn observed(left: u32, right: u32) -> Option<u8> {
    let count = u64::from(left) + u64::from(right);
    (count != 0).then(|| ((u64::from(left) * 256 + (count >> 1)) / count).clamp(1, 255) as u8)
}

/// Writes the forward update of `old` to `new` in a compressed header, or
/// that it does not change (`vp9_cond_prob_diff_update`).
pub(super) fn write_diff_update<S: BoolSink>(sink: &mut S, old: u8, new: u8) {
    sink.write(new != old, DIFF_UPDATE_PROB);
    if new != old {
        write_term_subexp(sink, remap_prob(new, old));
    }
}

/// What updating `old` to `new` codes its difference in, in bits, beyond
/// the update flag.
fn diff_update_bits(new: u8, old: u8) -> f64 {
    let mut sink = BitCost::default();
    write_term_subexp(&mut sink, remap_prob(new, old));
    sink.0
}

/// The index of `new` among the probabilities an update of `old` can code,
/// ordered by their distance from `old` (libvpx's `remap_prob`), inverting the
/// decoder's `inv_remap_prob`.
fn remap_prob(new: u8, old: u8) -> u32 {
    fn recenter_nonneg(v: i32, m: i32) -> i32 {
        if v > m << 1 {
            v
        } else if v >= m {
            (v - m) << 1
        } else {
            ((m - v) << 1) - 1
        }
    }
    let (v, m) = (i32::from(new) - 1, i32::from(old) - 1);
    let index = if m << 1 <= 255 {
        recenter_nonneg(v, m)
    } else {
        recenter_nonneg(254 - v, 254 - m)
    } - 1;
    u32::from(MAP_TABLE[index as usize])
}

/// `map_table`: the inverse of the decoder's `inv_map_table`, which lists
/// every 13th distance from 7 first so that they code in fewer bits.
const MAP_TABLE: [u8; 254] = {
    let mut table = [0_u8; 254];
    let mut delta = 0;
    while delta < 20 {
        table[6 + 13 * delta] = delta as u8;
        delta += 1;
    }
    let mut distance = 1;
    while distance <= 254 {
        if distance < 7 || (distance - 7) % 13 != 0 {
            table[distance - 1] = delta as u8;
            delta += 1;
        }
        distance += 1;
    }
    table
};

/// Writes a remapped difference with the terminated subexponential code
/// `decode_term_subexp` reads, in equiprobable bits.
fn write_term_subexp<S: BoolSink>(sink: &mut S, value: u32) {
    let mut literal = |value: u32, bits: u32| {
        for bit in (0..bits).rev() {
            sink.write((value >> bit) & 1 != 0, 128);
        }
    };
    match value {
        0..16 => literal(value, 1 + 4),
        16..32 => literal(0b10 << 4 | (value - 16), 2 + 4),
        32..64 => literal(0b110 << 5 | (value - 32), 3 + 5),
        _ => {
            literal(0b111, 3);
            // `decode_uniform`: seven bits below 65, then eight.
            let value = value - 64;
            const SHORT: u32 = (1 << 8) - 191;
            if value < SHORT {
                literal(value, 7);
            } else {
                literal(value + SHORT, 8);
            }
        }
    }
}

/// Moves `previous` toward the probability `left / (left + right)` of a zero
/// bit, by at most `max_update_factor / 256` once `max_count` symbols were
/// seen (libvpx's `merge_probs`).
fn merge(previous: u8, left: u32, right: u32, max_count: u32, max_update_factor: u32) -> u8 {
    let count = left + right;
    if count == 0 {
        return previous;
    }
    let observed =
        ((u64::from(left) * 256 + u64::from(count >> 1)) / u64::from(count)).clamp(1, 255) as i32;
    let factor = (max_update_factor * count.min(max_count) / max_count) as i32;
    let previous = i32::from(previous);
    (previous + (((observed - previous) * factor + 128) >> 8)) as u8
}

/// `mode_mv_merge_probs`: MODE_MV_COUNT_SAT 20, MODE_MV_MAX_UPDATE_FACTOR 128.
fn merge_mode(previous: u8, [left, right]: [u32; 2]) -> u8 {
    merge(previous, left, right, 20, 128)
}

/// Adapts every node of a tree from the counts of its leaves
/// (`vpx_tree_merge_probs`).
fn merge_tree(tree: &[i8], probs: &mut [u8], counts: &[u32]) {
    fn walk(tree: &[i8], probs: &mut [u8], counts: &[u32], node: usize) -> u32 {
        let mut branch = [0_u32; 2];
        for (side, total) in branch.iter_mut().enumerate() {
            let entry = tree[node + side];
            *total = if entry <= 0 {
                counts[usize::from(entry.unsigned_abs())]
            } else {
                walk(tree, probs, counts, entry as usize)
            };
        }
        probs[node >> 1] = merge_mode(probs[node >> 1], branch);
        branch[0] + branch[1]
    }
    walk(tree, probs, counts, 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merging_follows_libvpx_rounding() {
        // No symbols keeps the probability.
        assert_eq!(merge_mode(200, [0, 0]), 200);
        // Saturated counts move halfway: factor 128.
        assert_eq!(merge_mode(128, [40, 0]), 192);
        // Ten of twenty symbols give factor 64, toward 255.
        assert_eq!(merge_mode(100, [10, 0]), 139);
        // Moving down rounds toward negative infinity as libvpx's shift does:
        // 200 + ((1 - 200) * 128 + 128) >> 8.
        assert_eq!(merge_mode(200, [0, 20]), 101);
        // Coefficient merging saturates at 24 symbols, with factor 112:
        // 128 + ((1 - 128) * 112 + 128) >> 8.
        assert_eq!(merge(128, 0, 48, 24, 112), 72);
    }

    #[test]
    fn tree_merging_sums_the_leaves_below_each_node() {
        // PARTITION_TREE: NONE | (HORZ | (VERT | SPLIT)).
        let mut probs = [128_u8; 3];
        merge_tree(&PARTITION_TREE, &mut probs, &[0, 0, 0, 20]);
        // Every node saw only its right branch: 128 + ((1 - 128) * 128 + 128) >> 8.
        assert_eq!(probs, [65, 65, 65]);
        let mut probs = [128_u8; 3];
        merge_tree(&PARTITION_TREE, &mut probs, &[20, 0, 0, 0]);
        // Only the root saw symbols.
        assert_eq!(probs, [192, 128, 128]);
    }

    #[test]
    fn intra_frames_adapt_only_coefficients() {
        let mut counts = FrameCounts::default();
        counts.coef[1][0][0][0][0] = [10, 0, 0, 10];
        counts.eob_branch[1][0][0][0][0] = 20;
        counts.skip[0] = [0, 20];
        counts.tx[2][0] = [20, 0, 0, 0];
        let context = FrameContext::default();
        let intra = context.adapted(&counts, true, false, true, false);
        assert_ne!(intra.coef[1][0][0][0][0], context.coef[1][0][0][0][0]);
        assert_eq!(intra.skip, context.skip);
        assert_eq!(intra.tx_32x32, context.tx_32x32);
        let inter = context.adapted(&counts, false, false, true, false);
        assert_eq!(inter.coef[1][0][0][0][0], intra.coef[1][0][0][0][0]);
        assert_ne!(inter.skip, context.skip);
        assert_ne!(inter.tx_32x32, context.tx_32x32);
        // Transform sizes adapt only when blocks select them.
        let only_4x4 = context.adapted(&counts, false, false, false, false);
        assert_eq!(only_4x4.tx_32x32, context.tx_32x32);
        // After a key frame, coefficients adapt with the larger factor.
        let after_key = context.adapted(&counts, false, true, true, false);
        assert_ne!(after_key.coef[1][0][0][0][0], inter.coef[1][0][0][0][0]);
    }

    #[test]
    fn eighth_sample_bits_adapt_only_when_the_frame_allows_them() {
        let mut counts = FrameCounts::default();
        counts.mv[0].class0_hp = [0, 20];
        counts.mv[1].hp = [20, 0];
        let context = FrameContext::default();
        let low = context.adapted(&counts, false, false, true, false);
        assert_eq!(low.mv[0].class0_hp, context.mv[0].class0_hp);
        assert_eq!(low.mv[1].hp, context.mv[1].hp);
        let high = context.adapted(&counts, false, false, true, true);
        assert!(high.mv[0].class0_hp < context.mv[0].class0_hp);
        assert!(high.mv[1].hp > context.mv[1].hp);
    }
}
