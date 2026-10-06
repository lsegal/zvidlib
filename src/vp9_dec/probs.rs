//! Probability contexts (section 7.2 of the VP9 specification): the four
//! saved frame contexts, the symbol counts a frame accumulates, and the
//! backward adaptation that merges the two once a frame is decoded
//! (section 8.4). The merge rules and their rounding are libvpx's
//! (`vpx_dsp/prob.h`, `vp9_entropy.c`, `vp9_entropymode.c` and
//! `vp9_entropymv.c`).

use super::tables::*;

/// `vp9_intra_mode_tree`.
pub(super) const INTRA_MODE_TREE: [i8; 18] = [
    -DC_PRED, 2, -TM_PRED, 4, -V_PRED, 6, 8, 12, -H_PRED, 10, -D135_PRED, -D117_PRED, -D45_PRED,
    14, -D63_PRED, 16, -D153_PRED, -D207_PRED,
];
/// `vp9_inter_mode_tree`, with symbols relative to `NEARESTMV`.
pub(super) const INTER_MODE_TREE: [i8; 6] = [-2, 2, 0, 4, -1, -3];
/// `vp9_partition_tree`.
pub(super) const PARTITION_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
/// `vp9_switchable_interp_tree`: regular, smooth, sharp.
pub(super) const SWITCHABLE_INTERP_TREE: [i8; 4] = [0, 2, -1, -2];
/// `vp9_segment_tree`.
pub(super) const SEGMENT_TREE: [i8; 14] = [2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7];
/// `vp9_mv_joint_tree`.
pub(super) const MV_JOINT_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
/// `vp9_mv_class_tree`.
pub(super) const MV_CLASS_TREE: [i8; 20] = [
    0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10,
];
/// `vp9_mv_class0_tree`.
pub(super) const MV_CLASS0_TREE: [i8; 2] = [0, -1];
/// `vp9_mv_fp_tree`.
pub(super) const MV_FP_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];

pub(super) const DC_PRED: i8 = 0;
pub(super) const V_PRED: i8 = 1;
pub(super) const H_PRED: i8 = 2;
pub(super) const D45_PRED: i8 = 3;
pub(super) const D135_PRED: i8 = 4;
pub(super) const D117_PRED: i8 = 5;
pub(super) const D153_PRED: i8 = 6;
pub(super) const D207_PRED: i8 = 7;
pub(super) const D63_PRED: i8 = 8;
pub(super) const TM_PRED: i8 = 9;

/// The coefficient probabilities of one transform size:
/// `[plane type][ref][band][context][node]` (`vp9_coeff_probs_model`).
pub(super) type CoefProbs = [[[[[u8; 3]; 6]; 6]; 2]; 2];

/// The coefficient counts of one transform size, over the four model tokens
/// (zero, one, more than one, end of block): `[plane type][ref][band]
/// [context][token]` (`vp9_coeff_count_model`).
pub(super) type CoefCounts = [[[[[u32; 4]; 6]; 6]; 2]; 2];

/// The motion vector probabilities of one component (`nmv_component`).
#[derive(Clone, Copy, Debug)]
pub(super) struct MvComponentProbs {
    pub(super) sign: u8,
    pub(super) classes: [u8; 10],
    pub(super) class0: [u8; 1],
    pub(super) bits: [u8; 10],
    pub(super) class0_fp: [[u8; 3]; 2],
    pub(super) fp: [u8; 3],
    pub(super) class0_hp: u8,
    pub(super) hp: u8,
}

/// One frame context (`FRAME_CONTEXT`).
#[derive(Clone, Debug)]
pub(super) struct FrameContext {
    pub(super) y_mode: [[u8; 9]; 4],
    pub(super) uv_mode: [[u8; 9]; 10],
    pub(super) partition: [[u8; 3]; 16],
    /// Indexed by transform size.
    pub(super) coef: [CoefProbs; 4],
    pub(super) switchable_interp: [[u8; 2]; 4],
    pub(super) inter_mode: [[u8; 3]; 7],
    pub(super) intra_inter: [u8; 4],
    pub(super) comp_inter: [u8; 5],
    pub(super) single_ref: [[u8; 2]; 5],
    pub(super) comp_ref: [u8; 5],
    pub(super) tx8: [[u8; 1]; 2],
    pub(super) tx16: [[u8; 2]; 2],
    pub(super) tx32: [[u8; 3]; 2],
    pub(super) skip: [u8; 3],
    pub(super) mv_joints: [u8; 3],
    pub(super) mv: [MvComponentProbs; 2],
}

impl Default for FrameContext {
    /// The defaults `vp9_setup_past_independence` installs.
    fn default() -> Self {
        let component = |classes: [u8; 10], class0: u8| MvComponentProbs {
            sign: 128,
            classes,
            class0: [class0],
            bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
            class0_fp: [[128, 128, 64], [96, 112, 64]],
            fp: [64, 96, 64],
            class0_hp: 160,
            hp: 128,
        };
        Self {
            y_mode: DEFAULT_IF_Y_PROBS,
            uv_mode: DEFAULT_IF_UV_PROBS,
            partition: DEFAULT_PARTITION_PROBS,
            coef: [
                DEFAULT_COEF_PROBS_4X4,
                DEFAULT_COEF_PROBS_8X8,
                DEFAULT_COEF_PROBS_16X16,
                DEFAULT_COEF_PROBS_32X32,
            ],
            switchable_interp: [[235, 162], [36, 255], [34, 3], [149, 144]],
            inter_mode: DEFAULT_INTER_MODE_PROBS,
            intra_inter: [9, 102, 187, 225],
            comp_inter: [239, 183, 119, 96, 41],
            single_ref: [[33, 16], [77, 74], [142, 142], [172, 170], [238, 247]],
            comp_ref: [50, 126, 123, 221, 226],
            tx8: [[100], [66]],
            tx16: [[20, 152], [15, 101]],
            tx32: [[3, 136, 37], [5, 52, 13]],
            skip: [192, 128, 64],
            mv_joints: [32, 64, 96],
            mv: [
                component([224, 144, 192, 168, 192, 176, 192, 198, 198, 245], 216),
                component([216, 128, 176, 160, 176, 176, 192, 198, 198, 208], 208),
            ],
        }
    }
}

/// The counts of one motion vector component (`nmv_component_counts`).
#[derive(Clone, Copy, Debug, Default)]
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

/// The symbol counts of one frame (`FRAME_COUNTS`).
#[derive(Clone, Debug, Default)]
pub(super) struct FrameCounts {
    pub(super) y_mode: [[u32; 10]; 4],
    pub(super) uv_mode: [[u32; 10]; 10],
    pub(super) partition: [[u32; 4]; 16],
    /// Indexed by transform size.
    pub(super) coef: [CoefCounts; 4],
    pub(super) eob_branch: [[[[[u32; 6]; 6]; 2]; 2]; 4],
    pub(super) switchable_interp: [[u32; 3]; 4],
    pub(super) inter_mode: [[u32; 4]; 7],
    pub(super) intra_inter: [[u32; 2]; 4],
    pub(super) comp_inter: [[u32; 2]; 5],
    pub(super) single_ref: [[[u32; 2]; 2]; 5],
    pub(super) comp_ref: [[u32; 2]; 5],
    pub(super) tx8: [[u32; 2]; 2],
    pub(super) tx16: [[u32; 3]; 2],
    pub(super) tx32: [[u32; 4]; 2],
    pub(super) skip: [[u32; 2]; 3],
    pub(super) mv_joints: [u32; 4],
    pub(super) mv: [MvComponentCounts; 2],
}

/// `get_prob`.
fn get_prob(numerator: u32, denominator: u32) -> u8 {
    let p = ((u64::from(numerator) * 256 + u64::from(denominator >> 1)) / u64::from(denominator))
        as i64;
    p.clamp(1, 255) as u8
}

/// `get_binary_prob`.
fn get_binary_prob(n0: u32, n1: u32) -> u8 {
    let denominator = n0 + n1;
    if denominator == 0 {
        128
    } else {
        get_prob(n0, denominator)
    }
}

/// `weighted_prob`.
fn weighted_prob(prob1: u8, prob2: u8, factor: u32) -> u8 {
    ((u32::from(prob1) * (256 - factor) + u32::from(prob2) * factor + 128) >> 8) as u8
}

/// `merge_probs`, used for coefficient probabilities.
fn merge_probs(pre: u8, counts: [u32; 2], count_sat: u32, max_update_factor: u32) -> u8 {
    let prob = get_binary_prob(counts[0], counts[1]);
    let count = (counts[0] + counts[1]).min(count_sat);
    let factor = max_update_factor * count / count_sat;
    weighted_prob(pre, prob, factor)
}

const COUNT_TO_UPDATE_FACTOR: [u32; 21] = [
    0, 6, 12, 19, 25, 32, 38, 44, 51, 57, 64, 70, 76, 83, 89, 96, 102, 108, 115, 121, 128,
];

/// `mode_mv_merge_probs`.
fn mode_mv_merge(pre: u8, counts: [u32; 2]) -> u8 {
    let denominator = counts[0] + counts[1];
    if denominator == 0 {
        pre
    } else {
        let factor = COUNT_TO_UPDATE_FACTOR[denominator.min(20) as usize];
        weighted_prob(pre, get_prob(counts[0], denominator), factor)
    }
}

/// `vpx_tree_merge_probs`.
fn tree_merge(tree: &[i8], pre: &[u8], counts: &[u32], probs: &mut [u8]) {
    fn walk(index: usize, tree: &[i8], pre: &[u8], counts: &[u32], probs: &mut [u8]) -> u32 {
        let left = tree[index];
        let left_count = if left <= 0 {
            counts[(-left) as usize]
        } else {
            walk(left as usize, tree, pre, counts, probs)
        };
        let right = tree[index + 1];
        let right_count = if right <= 0 {
            counts[(-right) as usize]
        } else {
            walk(right as usize, tree, pre, counts, probs)
        };
        probs[index >> 1] = mode_mv_merge(pre[index >> 1], [left_count, right_count]);
        left_count + right_count
    }
    walk(0, tree, pre, counts, probs);
}

/// `vp9_adapt_coef_probs`.
pub(super) fn adapt_coef_probs(
    fc: &mut FrameContext,
    pre: &FrameContext,
    counts: &FrameCounts,
    frame_is_intra_only: bool,
    last_frame_was_key: bool,
) {
    let update_factor = if frame_is_intra_only {
        112
    } else if last_frame_was_key {
        128
    } else {
        112
    };
    let count_sat = 24;
    for tx in 0..4 {
        for plane in 0..2 {
            for reference in 0..2 {
                for band in 0..6 {
                    let contexts = if band == 0 { 3 } else { 6 };
                    for context in 0..contexts {
                        let c = counts.coef[tx][plane][reference][band][context];
                        let eob = counts.eob_branch[tx][plane][reference][band][context];
                        let (n0, n1, n2, neob) = (c[0], c[1], c[2], c[3]);
                        let branches = [[neob, eob.wrapping_sub(neob)], [n0, n1 + n2], [n1, n2]];
                        for node in 0..3 {
                            fc.coef[tx][plane][reference][band][context][node] = merge_probs(
                                pre.coef[tx][plane][reference][band][context][node],
                                branches[node],
                                count_sat,
                                update_factor,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// `vp9_adapt_mode_probs`.
pub(super) fn adapt_mode_probs(
    fc: &mut FrameContext,
    pre: &FrameContext,
    counts: &FrameCounts,
    switchable_filter: bool,
    tx_mode_select: bool,
) {
    for i in 0..4 {
        fc.intra_inter[i] = mode_mv_merge(pre.intra_inter[i], counts.intra_inter[i]);
    }
    for i in 0..5 {
        fc.comp_inter[i] = mode_mv_merge(pre.comp_inter[i], counts.comp_inter[i]);
        fc.comp_ref[i] = mode_mv_merge(pre.comp_ref[i], counts.comp_ref[i]);
        for j in 0..2 {
            fc.single_ref[i][j] = mode_mv_merge(pre.single_ref[i][j], counts.single_ref[i][j]);
        }
    }
    for i in 0..7 {
        // The inter mode tree's symbols are offsets from NEARESTMV, but the
        // counts are indexed by the same offsets, so they line up.
        tree_merge(
            &INTER_MODE_TREE,
            &pre.inter_mode[i],
            &counts.inter_mode[i],
            &mut fc.inter_mode[i],
        );
    }
    for i in 0..4 {
        tree_merge(
            &INTRA_MODE_TREE,
            &pre.y_mode[i],
            &counts.y_mode[i],
            &mut fc.y_mode[i],
        );
    }
    for i in 0..10 {
        tree_merge(
            &INTRA_MODE_TREE,
            &pre.uv_mode[i],
            &counts.uv_mode[i],
            &mut fc.uv_mode[i],
        );
    }
    for i in 0..16 {
        tree_merge(
            &PARTITION_TREE,
            &pre.partition[i],
            &counts.partition[i],
            &mut fc.partition[i],
        );
    }
    if switchable_filter {
        for i in 0..4 {
            tree_merge(
                &SWITCHABLE_INTERP_TREE,
                &pre.switchable_interp[i],
                &counts.switchable_interp[i],
                &mut fc.switchable_interp[i],
            );
        }
    }
    if tx_mode_select {
        for i in 0..2 {
            let c8 = counts.tx8[i];
            fc.tx8[i][0] = mode_mv_merge(pre.tx8[i][0], [c8[0], c8[1]]);
            let c16 = counts.tx16[i];
            let branches16 = [[c16[0], c16[1] + c16[2]], [c16[1], c16[2]]];
            for j in 0..2 {
                fc.tx16[i][j] = mode_mv_merge(pre.tx16[i][j], branches16[j]);
            }
            let c32 = counts.tx32[i];
            let branches32 = [
                [c32[0], c32[1] + c32[2] + c32[3]],
                [c32[1], c32[2] + c32[3]],
                [c32[2], c32[3]],
            ];
            for j in 0..3 {
                fc.tx32[i][j] = mode_mv_merge(pre.tx32[i][j], branches32[j]);
            }
        }
    }
    for i in 0..3 {
        fc.skip[i] = mode_mv_merge(pre.skip[i], counts.skip[i]);
    }
}

/// `vp9_adapt_mv_probs`.
pub(super) fn adapt_mv_probs(
    fc: &mut FrameContext,
    pre: &FrameContext,
    counts: &FrameCounts,
    allow_high_precision_mv: bool,
) {
    tree_merge(
        &MV_JOINT_TREE,
        &pre.mv_joints,
        &counts.mv_joints,
        &mut fc.mv_joints,
    );
    for i in 0..2 {
        let component = &mut fc.mv[i];
        let pre_component = &pre.mv[i];
        let c = &counts.mv[i];
        component.sign = mode_mv_merge(pre_component.sign, c.sign);
        tree_merge(
            &MV_CLASS_TREE,
            &pre_component.classes,
            &c.classes,
            &mut component.classes,
        );
        tree_merge(
            &MV_CLASS0_TREE,
            &pre_component.class0,
            &c.class0,
            &mut component.class0,
        );
        for j in 0..10 {
            component.bits[j] = mode_mv_merge(pre_component.bits[j], c.bits[j]);
        }
        for j in 0..2 {
            tree_merge(
                &MV_FP_TREE,
                &pre_component.class0_fp[j],
                &c.class0_fp[j],
                &mut component.class0_fp[j],
            );
        }
        tree_merge(&MV_FP_TREE, &pre_component.fp, &c.fp, &mut component.fp);
        if allow_high_precision_mv {
            component.class0_hp = mode_mv_merge(pre_component.class0_hp, c.class0_hp);
            component.hp = mode_mv_merge(pre_component.hp, c.hp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_keeps_the_previous_probability_without_counts() {
        assert_eq!(mode_mv_merge(77, [0, 0]), 77);
        assert_eq!(merge_probs(77, [0, 0], 24, 112), 77);
    }

    #[test]
    fn merge_moves_toward_the_observed_distribution() {
        // 20 or more observations use the full update factor of 128.
        assert_eq!(
            mode_mv_merge(128, [40, 0]),
            ((128 * 128 + 255 * 128 + 128) >> 8) as u8
        );
        assert_eq!(get_prob(0, 10), 1);
        assert_eq!(get_prob(10, 10), 255);
    }

    #[test]
    fn tree_merge_visits_every_node() {
        let mut probs = [0u8; 3];
        tree_merge(
            &PARTITION_TREE,
            &[128, 128, 128],
            &[30, 0, 0, 0],
            &mut probs,
        );
        assert!(probs[0] > 128);
        assert_eq!(probs[1], 128);
        assert_eq!(probs[2], 128);
    }
}
