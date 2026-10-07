//! Encodes one VP9 frame: headers, partitioning, mode decision, motion search
//! and coefficient tokens.
//!
//! The coding tools are a deliberately small, fully specified subset of VP9
//! profile 0:
//!
//! - frames send no forward probability updates. By default they adapt
//!   backwards instead: each frame codes with the probabilities the frames
//!   before it adapted to (see [`super::context`]), and inter frames also take
//!   the previous frame's motion vectors as candidates. Error-resilient frames
//!   code with the default probabilities and need no state from earlier
//!   frames other than the reconstructed reference picture;
//! - each 64x64 superblock is either coded whole or split into four, down to
//!   8x8 blocks, and each block picks its transform size from 4x4 up to the
//!   largest that fits (`tx_mode = TX_MODE_SELECT`), both by rate and
//!   distortion;
//! - the loop filter runs at the per-frame level that brings the
//!   reconstruction closest to the source, with sharpness 0 and no mode or
//!   reference deltas;
//! - key frames choose among the DC, V, H and TM intra modes per block;
//! - inter frames predict from the previous frame (`LAST_FRAME`) with
//!   whole-sample motion vectors found by a diamond search, started from the
//!   neighbours' vectors and the frame's motion estimated from its
//!   projections, coded as `ZEROMV`, `NEARESTMV`, `NEARMV` or `NEWMV`, or fall back to intra.
//!
//! Each superblock is searched first, with every candidate's rate counted
//! from the same probabilities and contexts the bitstream codes it with, and
//! then written. The search leaves out what rarely pays: a block coded
//! without residual is not split further, intra prediction is not tried
//! where motion compensation alone codes a block, and the luma intra mode
//! is chosen with the largest transform before the smaller ones are tried.
//! Beyond those, it stops searching wherever the result is already decided:
//! a split's quadrants once they cost more than the block whole, and a
//! candidate once its cost so far reaches the best one's. It also codes each
//! transform block's residual once per superblock, however many partitions
//! and candidates share it.
//! The encoder keeps a reconstruction that matches the decoding process
//! exactly: blocks predict from it unfiltered within the frame, and later
//! frames predict from it after the loop filter.

use super::bitwriter::{BitCost, BitWriter, BoolEncoder, BoolSink, bit_cost};
use super::context::{CoefProbs, FrameContext, FrameCounts, MvComponentCounts};
use super::dsp::{
    IntraMode, ReferencePlane, TransformScratch, TxType, forward_transform, inverse_transform_add,
    predict_inter, predict_intra,
};
use super::simd::{self, Quantizer};
use super::tables::{
    AC_QLOOKUP, CAT6_PROBS, DC_QLOOKUP, KF_PARTITION_PROBS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS,
    PARETO8_FULL,
};
use crate::vp9_dec::loopfilter::{self, FilterPlane, LoopFilterMask, MaskBlock};
use crate::vp9_dec::tables as shared;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

pub(super) const INTRA_MODE_TREE: [i8; 18] = [
    0, 2, -9, 4, -1, 6, 8, 12, -2, 10, -4, -5, -3, 14, -8, 16, -6, -7,
];
pub(super) const PARTITION_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
/// Leaves are `mode - NEARESTMV`: NEARESTMV 0, NEARMV 1, ZEROMV 2, NEWMV 3.
pub(super) const INTER_MODE_TREE: [i8; 6] = [-2, 2, 0, 4, -1, -3];
pub(super) const MV_JOINT_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
pub(super) const MV_CLASS_TREE: [i8; 20] = [
    0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10,
];
pub(super) const MV_FP_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];

const CAT_PROBS: [&[u8]; 5] = [
    &[159],
    &[165, 145],
    &[173, 148, 140],
    &[176, 155, 140, 135],
    &[180, 157, 141, 134, 130],
];

/// `mode_2_counter` from libvpx, indexed by VP9 mode number.
const MODE_TO_COUNTER: [u8; 14] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 3, 1];
const COUNTER_TO_CONTEXT: [u8; 19] = [2, 3, 4, 1, 3, 9, 0, 9, 9, 5, 5, 9, 5, 9, 9, 9, 9, 9, 6];

/// `mv_ref_blocks` for the square block sizes, 8x8 to 64x64, as `(row,
/// column)` offsets in 8x8 units.
const MV_REF_SEARCH: [[(isize, isize); 8]; 4] = [
    [
        (-1, 0),
        (0, -1),
        (-1, -1),
        (-2, 0),
        (0, -2),
        (-2, -1),
        (-1, -2),
        (-2, -2),
    ],
    [
        (-1, 0),
        (0, -1),
        (-1, 1),
        (1, -1),
        (-1, -1),
        (-3, 0),
        (0, -3),
        (-3, -3),
    ],
    [
        (-1, 1),
        (1, -1),
        (-1, 2),
        (2, -1),
        (-1, -1),
        (-3, 0),
        (0, -3),
        (-3, -3),
    ],
    [
        (-1, 3),
        (3, -1),
        (-1, 4),
        (4, -1),
        (-1, -1),
        (-1, 0),
        (0, -1),
        (-1, 6),
    ],
];

/// `partition_context_lookup` for the square block sizes, 8x8 to 64x64; the
/// above and left values agree.
const PARTITION_CONTEXT: [u8; 4] = [14, 12, 8, 0];

/// The border, in samples, around the copies the loop filter runs on.
const FILTER_BORDER: usize = 8;

const NEARESTMV: u8 = 10;
const NEARMV: u8 = 11;
const ZEROMV: u8 = 12;
const NEWMV: u8 = 13;

/// Which coding tools the frame encoder searches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CodingTools {
    /// The largest block coded whole, as log2 of its width in 8x8 units:
    /// 0 for 8x8 up to 3 for 64x64.
    pub(super) largest_block: usize,
    /// Whether blocks choose among transform sizes (`TX_MODE_SELECT`) or
    /// every transform is 4x4 (`ONLY_4X4`).
    pub(super) larger_transforms: bool,
}

impl CodingTools {
    /// Every partition and transform size.
    pub(super) const ALL: Self = Self {
        largest_block: 3,
        larger_transforms: true,
    };
}

/// Sizes derived once per stream from the frame dimensions.
#[derive(Clone, Copy, Debug)]
pub(super) struct Geometry {
    pub(super) width: usize,
    pub(super) height: usize,
    /// The decoded area: the frame size rounded up to whole 8x8 blocks.
    pub(super) aligned_width: usize,
    pub(super) aligned_height: usize,
    pub(super) mi_cols: usize,
    pub(super) mi_rows: usize,
}

impl Geometry {
    pub(super) fn new(width: usize, height: usize) -> Self {
        let aligned_width = width.div_ceil(8) * 8;
        let aligned_height = height.div_ceil(8) * 8;
        Self {
            width,
            height,
            aligned_width,
            aligned_height,
            mi_cols: aligned_width / 8,
            mi_rows: aligned_height / 8,
        }
    }

    pub(super) fn chroma_width(&self) -> usize {
        self.width.div_ceil(2)
    }

    pub(super) fn chroma_height(&self) -> usize {
        self.height.div_ceil(2)
    }
}

/// A 4:2:0 picture covering the whole 8-aligned decoded area.
#[derive(Clone)]
pub(super) struct Picture {
    pub(super) planes: [Vec<u8>; 3],
    pub(super) strides: [usize; 3],
}

impl Picture {
    pub(super) fn new(geometry: &Geometry) -> Self {
        let luma = geometry.aligned_width * geometry.aligned_height;
        Self {
            planes: [vec![0; luma], vec![0; luma / 4], vec![0; luma / 4]],
            strides: [
                geometry.aligned_width,
                geometry.aligned_width / 2,
                geometry.aligned_width / 2,
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct Mv {
    row: i32,
    col: i32,
}

/// The mode information later blocks, and the next frame's motion vector
/// candidate search, read as context.
#[derive(Clone, Copy, Default)]
pub(super) struct ModeInfo {
    is_inter: bool,
    /// The VP9 mode number: an intra mode, or `NEARESTMV..=NEWMV`.
    mode: u8,
    mv: Mv,
    skip: bool,
    tx_size: u8,
}

/// The blocks above and to the left of the one being coded, if any.
#[derive(Clone, Copy)]
struct Neighbors {
    above: Option<ModeInfo>,
    left: Option<ModeInfo>,
}

/// One transform block's quantized levels, in raster order.
#[derive(Clone)]
struct TxBlock {
    levels: Vec<i32>,
    /// One past the last nonzero level in scan order.
    eob: usize,
    tx_type: TxType,
}

/// Everything decided for one block.
#[derive(Clone)]
struct BlockChoice {
    info: ModeInfo,
    uv_mode: IntraMode,
    /// For `NEWMV`, the reference the motion vector is coded against.
    best_mv: Mv,
    /// Each plane's transform blocks in raster order.
    blocks: [Vec<TxBlock>; 3],
    /// The reconstruction of each plane, row by row.
    pixels: [Vec<u8>; 3],
    /// Distortion plus `lambda` times the bits of everything but the
    /// partition symbol.
    cost: f64,
}

/// The partition decided for a square of the frame.
enum Node {
    /// The square lies outside the frame.
    Outside,
    Whole(Box<BlockChoice>),
    Split(Box<[Node; 4]>),
}

/// How a plane's transform blocks are predicted.
#[derive(Clone, Copy)]
enum Prediction {
    /// Each transform block from its reconstructed neighbours, in turn.
    Intra(IntraMode),
    /// From the reference frame along a motion vector, already in place.
    Inter(Mv),
}

/// Identifies a transform block's residual. Coding one depends only on
/// where it is, its transform, whether it is inter, and its prediction, so
/// blocks that agree on all of these quantize and reconstruct identically.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum ResidualKey {
    /// An inter prediction depends only on the position and the motion
    /// vector.
    Inter {
        plane: u8,
        tx_size: u8,
        x: u16,
        y: u16,
        mv: Mv,
    },
    /// An intra prediction depends on the reconstruction around the block,
    /// so it is identified by a hash of its samples, which are compared in
    /// full on a match.
    Intra {
        plane: u8,
        tx_size: u8,
        tx_type: u8,
        x: u16,
        y: u16,
        prediction: u64,
    },
}

/// FxHash's mix of one word into `hash`.
fn fx_mix(hash: u64, word: u64) -> u64 {
    (hash.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95)
}

/// A [`ResidualKey`] hasher: the cache is looked up for every transform block
/// the search codes, and the keys are small integers, which the default
/// SipHash spent more on than the lookups saved.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = fx_mix(self.0, u64::from(byte));
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.0 = fx_mix(self.0, u64::from(value));
    }

    fn write_u16(&mut self, value: u16) {
        self.0 = fx_mix(self.0, u64::from(value));
    }

    fn write_u32(&mut self, value: u32) {
        self.0 = fx_mix(self.0, u64::from(value));
    }

    fn write_i32(&mut self, value: i32) {
        self.0 = fx_mix(self.0, u64::from(value as u32));
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = fx_mix(self.0, value);
    }

    fn write_usize(&mut self, value: usize) {
        self.0 = fx_mix(self.0, value as u64);
    }

    fn write_isize(&mut self, value: isize) {
        self.0 = fx_mix(self.0, value as u64);
    }
}

/// What coding one transform block's residual produced, for every candidate
/// in the superblock that codes the same one. Only its token bits depend on
/// the block's surroundings, through the nonzero context.
struct CachedResidual {
    /// An intra block's prediction, to tell blocks apart whose hashes agree.
    prediction: Vec<u8>,
    prediction_error: u64,
    eob: usize,
    levels: Vec<i32>,
    /// The reconstruction with the levels coded, and its squared error.
    reconstructed: Vec<u8>,
    error: u64,
    bits: [Option<f64>; 3],
}

/// The result of coding one plane of a block.
#[derive(Clone)]
struct PlaneCoding {
    blocks: Vec<TxBlock>,
    error: u64,
    bits: f64,
}

/// The cost a candidate has to stay under to be chosen, and what the planes
/// coded before this one and its symbols already add to it.
///
/// A candidate's cost is `error + lambda * bits` over its planes and symbols,
/// and coding more transform blocks only adds to both, so a plane whose cost
/// so far reaches `best` cannot be chosen and is not coded further. The sum
/// is formed as the candidate's full cost is, so the comparison is exact.
///
/// `ceiling` bounds the whole block's cost instead, for an intra candidate
/// competing with an inter one: whatever else it codes, an intra block costs
/// at least its luma error and luma symbols, and its luma tokens too once a
/// luma transform block has levels, since the block is then not skipped. It
/// cannot be chosen once those exceed the inter block's cost.
#[derive(Clone, Copy)]
struct Bound {
    best: f64,
    error: u64,
    bits: f64,
    symbol_bits: f64,
    ceiling: f64,
}

impl Bound {
    /// A bound on `best` alone.
    fn under(best: f64, symbol_bits: f64) -> Self {
        Self {
            best,
            error: 0,
            bits: 0.0,
            symbol_bits,
            ceiling: f64::INFINITY,
        }
    }

    /// Whether a plane coded so far as `coding`, with levels in some
    /// transform block if `coded`, can no longer cost under `best` and under
    /// `ceiling`.
    fn reached(&self, lambda: f64, coding: &PlaneCoding, coded: bool) -> (bool, bool) {
        let error = (self.error + coding.error) as f64;
        let bits = self.bits + coding.bits;
        let floor_bits = if coded { bits } else { 0.0 };
        (
            error + lambda * (bits + self.symbol_bits) >= self.best,
            error + lambda * (floor_bits + self.symbol_bits) > self.ceiling,
        )
    }
}

/// The best luma mode and transform size found so far.
struct LumaCandidate {
    cost: f64,
    tx_size: u8,
    mode: IntraMode,
    pixels: Vec<u8>,
    coding: PlaneCoding,
}

/// The best chroma mode found so far: its cost, mode, both planes' coding and
/// their reconstruction.
type ChromaCandidate = (f64, IntraMode, [PlaneCoding; 2], [Vec<u8>; 2]);

/// Working memory for coding one transform block, kept between blocks
/// because the search codes every transform block of every candidate it
/// tries.
struct Scratch {
    residual: [i32; 32 * 32],
    coefficients: [f64; 32 * 32],
    levels: [i32; 32 * 32],
    dequantized: [i32; 32 * 32],
    prediction: [u8; 32 * 32],
    transform: Box<TransformScratch>,
}

impl Scratch {
    fn new() -> Box<Self> {
        Box::new(Self {
            residual: [0; 32 * 32],
            coefficients: [0.0; 32 * 32],
            levels: [0; 32 * 32],
            dequantized: [0; 32 * 32],
            prediction: [0; 32 * 32],
            transform: TransformScratch::new(),
        })
    }
}

/// The above and left coding contexts a block reads and writes.
struct ContextSnapshot {
    above_nonzero: [Vec<bool>; 3],
    left_nonzero: [[bool; 16]; 3],
    above_partition: Vec<u8>,
    left_partition: [u8; 8],
}

/// Everything coding a block changes, so that another candidate can replace
/// it.
struct BlockSnapshot {
    pixels: [Vec<u8>; 3],
    mode_info: Vec<ModeInfo>,
    contexts: ContextSnapshot,
}

/// The tile's boolean encoder, counting each symbol as the decoder will
/// when it reads it.
struct TileWriter {
    encoder: BoolEncoder,
    counts: FrameCounts,
}

impl BoolSink for TileWriter {
    fn write(&mut self, bit: bool, probability: u8) {
        self.encoder.write(bit, probability);
    }

    fn counts(&mut self) -> Option<&mut FrameCounts> {
        Some(&mut self.counts)
    }
}

/// A coded frame and the state the frames after it depend on.
pub(super) struct EncodedFrame {
    pub(super) data: Vec<u8>,
    pub(super) reconstruction: Picture,
    /// The symbols the frame coded, for backward adaptation.
    pub(super) counts: FrameCounts,
    /// Every 8x8 block's mode and motion vector, in raster order.
    pub(super) mode_info: Vec<ModeInfo>,
}

pub(super) struct FrameEncoder<'a> {
    geometry: Geometry,
    source: &'a Picture,
    reference: Option<&'a Picture>,
    recon: Picture,
    tools: CodingTools,
    base_q_idx: u8,
    error_resilient: bool,
    /// The probabilities the frame codes with.
    context: &'a FrameContext,
    /// The previous frame's modes, when this frame takes its motion vectors as
    /// candidates (`UsePrevFrameMvs`).
    previous_mode_info: Option<&'a [ModeInfo]>,
    /// The loop filter level the frame header signals; zero turns it off.
    filter_level: u8,
    /// Whether [`Self::encode`] chooses a loop filter level at all; tests
    /// clear it to compare against an unfiltered encode.
    loop_filter: bool,
    /// The frame's motion estimated from its projections, which every
    /// block's motion search also starts from.
    projected_mvs: Vec<Mv>,
    /// Every block written, as `(mi_row, mi_col, bsl)`, which the loop
    /// filter's edge masks are built from.
    coded_blocks: Vec<(usize, usize, usize)>,
    /// The quantizer steps every plane uses: the header codes no deltas.
    dc_q: i32,
    ac_q: i32,
    lambda: f64,
    mode_info: Vec<ModeInfo>,
    above_nonzero: [Vec<bool>; 3],
    left_nonzero: [[bool; 16]; 3],
    above_partition: Vec<u8>,
    left_partition: [u8; 8],
    scratch: Box<Scratch>,
    /// Whether a candidate was ruled out by a [`Bound`]'s ceiling since
    /// [`Self::choose_block`] last cleared it.
    over_ceiling: bool,
    /// The residuals coded in the current superblock, which the search codes
    /// again for every partition and candidate that shares one.
    residual_cache: HashMap<ResidualKey, CachedResidual, BuildHasherDefault<KeyHasher>>,
    /// Token costs under `context`'s coefficient probabilities.
    token_costs: Box<[TokenCosts; 4]>,
}

impl<'a> FrameEncoder<'a> {
    /// Prepares to encode `source`, as a key frame when `reference` is `None`
    /// and as an inter frame predicted from `reference` otherwise, coding with
    /// the probabilities in `context`.
    ///
    /// An error-resilient frame must code with the default context and has no
    /// `previous_mode_info`. Otherwise `previous_mode_info` is the previous
    /// frame's [`EncodedFrame::mode_info`], which an inter frame's motion
    /// vector candidates include.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        geometry: Geometry,
        source: &'a Picture,
        reference: Option<&'a Picture>,
        base_q_idx: u8,
        tools: CodingTools,
        error_resilient: bool,
        context: &'a FrameContext,
        previous_mode_info: Option<&'a [ModeInfo]>,
    ) -> Self {
        let q = usize::from(base_q_idx);
        let ac = AC_QLOOKUP[q];
        Self {
            geometry,
            source,
            reference,
            recon: Picture::new(&geometry),
            tools,
            base_q_idx,
            error_resilient,
            context,
            previous_mode_info: previous_mode_info.filter(|_| reference.is_some()),
            filter_level: 0,
            loop_filter: true,
            projected_mvs: Vec::new(),
            coded_blocks: Vec::new(),
            dc_q: DC_QLOOKUP[q],
            ac_q: ac,
            // Distortion is a pixel-domain squared error and rate is in bits.
            // The transform's coefficients are eight times orthonormal, so the
            // effective step is `ac / 8`, and lambda is about a nineteenth of
            // its square: enough to drop coefficients, skip blocks and merge
            // partitions where they cost more than they restore, without
            // trading away the quality the quantizer would otherwise keep.
            lambda: f64::from(ac * ac) / 1200.0,
            mode_info: vec![ModeInfo::default(); geometry.mi_cols * geometry.mi_rows],
            above_nonzero: [
                vec![false; geometry.mi_cols * 2],
                vec![false; geometry.mi_cols],
                vec![false; geometry.mi_cols],
            ],
            left_nonzero: [[false; 16]; 3],
            above_partition: vec![0; geometry.mi_cols],
            left_partition: [0; 8],
            scratch: Scratch::new(),
            over_ceiling: false,
            residual_cache: HashMap::default(),
            token_costs: token_costs(&context.coef),
        }
    }

    fn is_key(&self) -> bool {
        self.reference.is_none()
    }

    /// Leaves the loop filter off, as the encoder did before it chose a level.
    #[cfg(test)]
    pub(super) fn without_loop_filter(mut self) -> Self {
        self.loop_filter = false;
        self
    }

    /// Encodes the frame; `full_range` is the colour range the key frame
    /// header signals.
    pub(super) fn encode(mut self, full_range: bool) -> EncodedFrame {
        let mut writer = TileWriter {
            encoder: BoolEncoder::new(),
            counts: FrameCounts::default(),
        };
        let sb_rows = self.geometry.mi_rows.div_ceil(8);
        let sb_cols = self.geometry.mi_cols.div_ceil(8);
        if let Some(reference) = self.reference {
            self.projected_mvs = self.projected_motion(reference);
        }
        for sb_row in 0..sb_rows {
            self.left_nonzero = [[false; 16]; 3];
            self.left_partition = [0; 8];
            for sb_col in 0..sb_cols {
                let (mi_row, mi_col) = (sb_row * 8, sb_col * 8);
                // The search leaves the reconstruction and mode information
                // of the chosen partition in place, and the contexts are
                // replayed as the superblock is written.
                let contexts = self.save_contexts(mi_col, 3);
                self.residual_cache.clear();
                let (node, _) = self.search_partition(mi_row, mi_col, 3, f64::INFINITY);
                self.restore_contexts(mi_col, 3, &contexts);
                self.write_partition(&mut writer, &node, mi_row, mi_col, 3);
            }
        }
        if self.loop_filter {
            self.apply_loop_filter();
        }
        let key = self.is_key();
        let geometry = self.geometry;
        let tile = writer.encoder.finish();
        let compressed = compressed_header(key, self.tools.larger_transforms);
        let mut data = uncompressed_header(
            &geometry,
            key,
            self.error_resilient,
            self.base_q_idx,
            self.filter_level,
            full_range,
            compressed.len(),
        );
        data.extend_from_slice(&compressed);
        data.extend_from_slice(&tile);
        EncodedFrame {
            data,
            reconstruction: self.recon,
            counts: writer.counts,
            mode_info: self.mode_info,
        }
    }

    /// Chooses the loop filter level whose output is closest to the source
    /// and filters the reconstruction with it, as the decoder will.
    ///
    /// This is libvpx's `search_filter_level`: start from the level libvpx's
    /// `LPF_PICK_FROM_Q` guesses for the quantizer, then step towards lower
    /// error, halving the step each time neither neighbour improves. Like
    /// libvpx it biases the search towards lower levels, because a level
    /// that only just lowers this frame's error over-smooths the reference
    /// later frames predict from and makes them cost more.
    fn apply_loop_filter(&mut self) {
        let mut guess = (i64::from(self.ac_q) * 20_723 + 1_015_158 + (1 << 17)) >> 18;
        if self.is_key() {
            guess -= 4;
        }
        let mut errors = [None; 64];
        let mut error = |level: u8| {
            *errors[usize::from(level)].get_or_insert_with(|| self.filtered_error(level))
        };

        let mut middle = guess.clamp(0, 63) as u8;
        let mut best = middle;
        let mut best_error = error(middle);
        let mut step = if middle < 16 { 4 } else { middle / 4 };
        let mut direction = 0_i8;
        while step > 0 {
            let low = middle.saturating_sub(step);
            let high = (middle + step).min(63);
            let bias = (best_error >> (15 - middle / 8)) * u64::from(step);
            if direction <= 0 && low != middle {
                let low_error = error(low);
                if low_error.saturating_sub(bias) < best_error {
                    best_error = best_error.min(low_error);
                    best = low;
                }
            }
            if direction >= 0 && high != middle {
                let high_error = error(high);
                if high_error < best_error.saturating_sub(bias) {
                    best_error = high_error;
                    best = high;
                }
            }
            if best == middle {
                step /= 2;
                direction = 0;
            } else {
                direction = if best < middle { -1 } else { 1 };
                middle = best;
            }
        }
        self.filter_level = best;
        self.recon = self.filtered(best);
    }

    /// The reconstruction after the loop filter at `level`.
    fn filtered(&self, level: u8) -> Picture {
        let mut picture = self.recon.clone();
        if level == 0 {
            return picture;
        }
        for (plane, (data, padded_stride)) in self.filter_copy(level).iter().enumerate() {
            let stride = picture.strides[plane];
            for (row, output) in picture.planes[plane].chunks_exact_mut(stride).enumerate() {
                let start = (row + FILTER_BORDER) * padded_stride + FILTER_BORDER;
                output.copy_from_slice(&data[start..start + stride]);
            }
        }
        picture
    }

    /// The squared error from the source of the reconstruction after the
    /// loop filter at `level`, measured on the filter's own copies.
    fn filtered_error(&self, level: u8) -> u64 {
        if level == 0 {
            return self.source_error(core::array::from_fn(|plane| {
                (&self.recon.planes[plane][..], self.recon.strides[plane])
            }));
        }
        let padded = self.filter_copy(level);
        self.source_error(
            padded
                .each_ref()
                .map(|(data, stride)| (&data[FILTER_BORDER * stride + FILTER_BORDER..], *stride)),
        )
    }

    /// Copies of the reconstruction's planes after the loop filter at
    /// `level`, with their strides. The filter works a superblock at a time
    /// and may touch samples past the decoded area, so it runs on copies laid
    /// out as the decoder's planes are: whole superblocks with a border of
    /// [`FILTER_BORDER`] samples.
    fn filter_copy(&self, level: u8) -> [(Vec<u8>, usize); 3] {
        let Geometry {
            mi_rows, mi_cols, ..
        } = self.geometry;
        let sb_cols = mi_cols.div_ceil(8);
        let mut masks = vec![LoopFilterMask::default(); sb_cols * mi_rows.div_ceil(8)];
        // One mask entry per coded block, as `decode_block` builds them.
        for &(mi_row, mi_col, bsl) in &self.coded_blocks {
            let info = self.mode_info[mi_row * mi_cols + mi_col];
            let size = 1 << bsl;
            loopfilter::build_mask(
                &mut masks[(mi_row >> 3) * sb_cols + (mi_col >> 3)],
                &MaskBlock {
                    // `BLOCK_8X8` and every square size above it.
                    sb_type: 3 + 3 * bsl as u8,
                    tx_size: info.tx_size,
                    skip_inter: info.skip && info.is_inter,
                    filter_level: level,
                },
                mi_row,
                mi_col,
                size,
                size,
            );
        }
        let recon = &self.recon;
        let mut padded = [0, 1, 2].map(|plane| {
            let superblock = if plane == 0 { 64 } else { 32 };
            let stride = recon.strides[plane];
            let rows = recon.planes[plane].len() / stride;
            let padded_stride = stride.div_ceil(superblock) * superblock + 2 * FILTER_BORDER;
            let padded_rows = rows.div_ceil(superblock) * superblock + 2 * FILTER_BORDER;
            let mut data = vec![0; padded_stride * padded_rows];
            for row in 0..rows {
                let start = (row + FILTER_BORDER) * padded_stride + FILTER_BORDER;
                data[start..start + stride]
                    .copy_from_slice(&recon.planes[plane][row * stride..(row + 1) * stride]);
            }
            (data, padded_stride)
        });
        let mut planes = padded.each_mut().map(|(data, stride)| FilterPlane {
            data,
            stride: *stride,
            origin: FILTER_BORDER * *stride + FILTER_BORDER,
        });
        loopfilter::filter_frame(&mut planes, &mut masks, mi_rows, mi_cols, 0);
        padded
    }

    /// The squared error from the source over the visible area of every
    /// plane of a picture, given as each plane's samples from its top-left
    /// one, with its stride.
    fn source_error(&self, planes: [(&[u8], usize); 3]) -> u64 {
        let geometry = &self.geometry;
        let mut error = 0;
        for (plane, width, height) in [
            (0, geometry.width, geometry.height),
            (1, geometry.chroma_width(), geometry.chroma_height()),
            (2, geometry.chroma_width(), geometry.chroma_height()),
        ] {
            let (samples, stride) = planes[plane];
            error += simd::sse(
                samples,
                stride,
                &self.source.planes[plane],
                self.source.strides[plane],
                width,
                height,
            );
        }
        error
    }

    /// Whether a block of `bsl` at `(mi_row, mi_col)` lies wholly inside the
    /// decoded area, the only blocks the encoder codes whole.
    fn fits(&self, mi_row: usize, mi_col: usize, bsl: usize) -> bool {
        let size = 1 << bsl;
        mi_row + size <= self.geometry.mi_rows && mi_col + size <= self.geometry.mi_cols
    }

    /// Chooses how to code the square of `bsl` at `(mi_row, mi_col)`, leaving
    /// its reconstruction, mode information and contexts in place; returns
    /// the decision with its cost.
    ///
    /// Only a cost under `budget` can change the caller's decision, so once
    /// the cost is known to reach it the search stops and returns an infinite
    /// cost; the caller then discards the decision. Costs are never negative,
    /// so the quadrants of a split stop being searched as soon as those coded
    /// so far reach the cost of the whole block or the budget: the decisions
    /// are those of the full search.
    fn search_partition(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        budget: f64,
    ) -> (Node, f64) {
        if mi_row >= self.geometry.mi_rows || mi_col >= self.geometry.mi_cols {
            return (Node::Outside, 0.0);
        }
        let whole = bsl <= self.tools.largest_block && self.fits(mi_row, mi_col, bsl);
        let mut best = None;
        let before = (whole && bsl > 0).then(|| self.save_block(mi_row, mi_col, bsl));
        if whole {
            let bits = cost(|sink| self.partition_symbol(sink, mi_row, mi_col, bsl, false));
            // Coding the block whole is only searched for a cost under the
            // budget.
            let ceiling = budget - self.lambda * bits;
            if let Some(choice) = self.choose_block(mi_row, mi_col, bsl, ceiling) {
                let total = choice.cost + self.lambda * bits;
                self.commit(mi_row, mi_col, bsl, &choice);
                // A block its prediction alone codes well enough is not split.
                let settled = choice.info.skip;
                best = Some((Node::Whole(Box::new(choice)), total));
                if settled {
                    return best.expect("just coded whole");
                }
            }
        }
        if bsl == 0 {
            // An 8x8 block inside the frame is coded whole, so without a whole
            // coding under the budget it cannot be.
            return best.unwrap_or((Node::Outside, f64::INFINITY));
        }
        let after_whole = best.is_some().then(|| self.save_block(mi_row, mi_col, bsl));
        if let Some(before) = &before {
            self.restore_block(mi_row, mi_col, bsl, before);
        }
        let bound = best
            .as_ref()
            .map_or(budget, |&(_, whole_cost)| whole_cost.min(budget));
        let mut total =
            self.lambda * cost(|sink| self.partition_symbol(sink, mi_row, mi_col, bsl, true));
        let half = 1 << (bsl - 1);
        let origins = [
            (mi_row, mi_col),
            (mi_row, mi_col + half),
            (mi_row + half, mi_col),
            (mi_row + half, mi_col + half),
        ];
        let mut quadrants = [const { Node::Outside }; 4];
        for (quadrant, (row, col)) in quadrants.iter_mut().zip(origins) {
            if total >= bound {
                break;
            }
            let (node, cost) = self.search_partition(row, col, bsl - 1, bound - total);
            total += cost;
            *quadrant = node;
        }
        let over_budget = |cost: f64| if cost < budget { cost } else { f64::INFINITY };
        match best {
            Some((node, whole_cost)) if whole_cost <= total => {
                let after = after_whole.expect("saved after coding the block whole");
                self.restore_block(mi_row, mi_col, bsl, &after);
                (node, over_budget(whole_cost))
            }
            _ => (Node::Split(Box::new(quadrants)), over_budget(total)),
        }
    }

    /// Writes a searched partition, replaying its effect on the contexts.
    fn write_partition(
        &mut self,
        writer: &mut TileWriter,
        node: &Node,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
    ) {
        match node {
            Node::Outside => {}
            Node::Whole(choice) => {
                self.coded_blocks.push((mi_row, mi_col, bsl));
                self.partition_symbol(writer, mi_row, mi_col, bsl, false);
                self.write_mode_info(writer, mi_row, mi_col, bsl, choice);
                self.write_tokens(writer, mi_row, mi_col, bsl, choice);
                self.update_contexts(mi_row, mi_col, bsl, choice);
            }
            Node::Split(quadrants) => {
                self.partition_symbol(writer, mi_row, mi_col, bsl, true);
                let half = 1 << (bsl - 1);
                let origins = [
                    (mi_row, mi_col),
                    (mi_row, mi_col + half),
                    (mi_row + half, mi_col),
                    (mi_row + half, mi_col + half),
                ];
                for (quadrant, (row, col)) in quadrants.iter().zip(origins) {
                    self.write_partition(writer, quadrant, row, col, bsl - 1);
                }
            }
        }
    }

    /// The part of a block inside the decoded area of `plane`, as `(x, y,
    /// width, height)` in samples.
    fn plane_region(
        &self,
        plane: usize,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
    ) -> (usize, usize, usize, usize) {
        let ss = usize::from(plane > 0);
        let size = (8 << bsl) >> ss;
        let (x, y) = ((mi_col * 8) >> ss, (mi_row * 8) >> ss);
        let width = size.min((self.geometry.aligned_width >> ss) - x);
        let height = size.min((self.geometry.aligned_height >> ss) - y);
        (x, y, width, height)
    }

    fn block_pixels(&self, plane: usize, mi_row: usize, mi_col: usize, bsl: usize) -> Vec<u8> {
        let (x, y, width, height) = self.plane_region(plane, mi_row, mi_col, bsl);
        let stride = self.recon.strides[plane];
        let mut pixels = Vec::with_capacity(width * height);
        for row in y..y + height {
            pixels.extend_from_slice(&self.recon.planes[plane][row * stride + x..][..width]);
        }
        pixels
    }

    fn load_pixels(
        &mut self,
        plane: usize,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        pixels: &[u8],
    ) {
        let (x, y, width, height) = self.plane_region(plane, mi_row, mi_col, bsl);
        let stride = self.recon.strides[plane];
        for row in 0..height {
            self.recon.planes[plane][(y + row) * stride + x..][..width]
                .copy_from_slice(&pixels[row * width..][..width]);
        }
    }

    /// The above context entries of a block in `plane`, in 4x4 units.
    fn nonzero_span(&self, plane: usize, mi_col: usize, bsl: usize) -> std::ops::Range<usize> {
        let ss = usize::from(plane > 0);
        let start = (mi_col * 2) >> ss;
        start..(start + ((2 << bsl) >> ss)).min(self.above_nonzero[plane].len())
    }

    fn partition_span(&self, mi_col: usize, bsl: usize) -> std::ops::Range<usize> {
        mi_col..(mi_col + (1 << bsl)).min(self.geometry.mi_cols)
    }

    fn save_contexts(&self, mi_col: usize, bsl: usize) -> ContextSnapshot {
        ContextSnapshot {
            above_nonzero: core::array::from_fn(|plane| {
                self.above_nonzero[plane][self.nonzero_span(plane, mi_col, bsl)].to_vec()
            }),
            left_nonzero: self.left_nonzero,
            above_partition: self.above_partition[self.partition_span(mi_col, bsl)].to_vec(),
            left_partition: self.left_partition,
        }
    }

    fn restore_contexts(&mut self, mi_col: usize, bsl: usize, snapshot: &ContextSnapshot) {
        for plane in 0..3 {
            let span = self.nonzero_span(plane, mi_col, bsl);
            self.above_nonzero[plane][span].copy_from_slice(&snapshot.above_nonzero[plane]);
        }
        self.left_nonzero = snapshot.left_nonzero;
        let span = self.partition_span(mi_col, bsl);
        self.above_partition[span].copy_from_slice(&snapshot.above_partition);
        self.left_partition = snapshot.left_partition;
    }

    fn save_block(&self, mi_row: usize, mi_col: usize, bsl: usize) -> BlockSnapshot {
        let mut mode_info = Vec::new();
        let rows = mi_row..(mi_row + (1 << bsl)).min(self.geometry.mi_rows);
        for row in rows {
            let start = row * self.geometry.mi_cols;
            let span = self.partition_span(mi_col, bsl);
            mode_info.extend_from_slice(&self.mode_info[start + span.start..start + span.end]);
        }
        BlockSnapshot {
            pixels: core::array::from_fn(|plane| self.block_pixels(plane, mi_row, mi_col, bsl)),
            mode_info,
            contexts: self.save_contexts(mi_col, bsl),
        }
    }

    fn restore_block(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        snapshot: &BlockSnapshot,
    ) {
        for plane in 0..3 {
            self.load_pixels(plane, mi_row, mi_col, bsl, &snapshot.pixels[plane]);
        }
        let span = self.partition_span(mi_col, bsl);
        let rows = mi_row..(mi_row + (1 << bsl)).min(self.geometry.mi_rows);
        for (index, row) in rows.enumerate() {
            let start = row * self.geometry.mi_cols;
            self.mode_info[start + span.start..start + span.end]
                .copy_from_slice(&snapshot.mode_info[index * span.len()..][..span.len()]);
        }
        self.restore_contexts(mi_col, bsl, &snapshot.contexts);
    }

    /// Puts a chosen block in place: its reconstruction, mode information
    /// and contexts.
    fn commit(&mut self, mi_row: usize, mi_col: usize, bsl: usize, choice: &BlockChoice) {
        for plane in 0..3 {
            self.load_pixels(plane, mi_row, mi_col, bsl, &choice.pixels[plane]);
        }
        let size = 1 << bsl;
        for row in mi_row..mi_row + size {
            let start = row * self.geometry.mi_cols + mi_col;
            self.mode_info[start..start + size].fill(choice.info);
        }
        self.update_contexts(mi_row, mi_col, bsl, choice);
    }

    /// Updates the nonzero and partition contexts as decoding the block does.
    fn update_contexts(&mut self, mi_row: usize, mi_col: usize, bsl: usize, choice: &BlockChoice) {
        for plane in 0..3 {
            let ss = usize::from(plane > 0);
            let n4 = (2 << bsl) >> ss;
            let (x4, y4) = ((mi_col * 2) >> ss, ((mi_row & 7) * 2) >> ss);
            if choice.info.skip {
                self.above_nonzero[plane][x4..x4 + n4].fill(false);
                self.left_nonzero[plane][y4..y4 + n4].fill(false);
                continue;
            }
            let step = 1 << plane_tx_size(choice.info.tx_size, bsl, plane);
            let mut blocks = choice.blocks[plane].iter();
            for row in (0..n4).step_by(step) {
                for col in (0..n4).step_by(step) {
                    let nonzero = blocks.next().expect("one block per transform").eob > 0;
                    self.above_nonzero[plane][x4 + col..x4 + col + step].fill(nonzero);
                    self.left_nonzero[plane][y4 + row..y4 + row + step].fill(nonzero);
                }
            }
        }
        let size = 1 << bsl;
        self.above_partition[mi_col..mi_col + size].fill(PARTITION_CONTEXT[bsl]);
        self.left_partition[mi_row & 7..(mi_row & 7) + size].fill(PARTITION_CONTEXT[bsl]);
    }

    fn neighbors(&self, mi_row: usize, mi_col: usize) -> Neighbors {
        let cols = self.geometry.mi_cols;
        Neighbors {
            above: (mi_row > 0).then(|| self.mode_info[(mi_row - 1) * cols + mi_col]),
            left: (mi_col > 0).then(|| self.mode_info[mi_row * cols + mi_col - 1]),
        }
    }

    /// The largest transform a block of `bsl` may use.
    fn max_tx_size(&self, bsl: usize) -> u8 {
        if self.tools.larger_transforms {
            (bsl as u8 + 1).min(3)
        } else {
            0
        }
    }

    /// The best coding of the block as a whole, or `None` if none costs
    /// less than `ceiling`.
    fn choose_block(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        ceiling: f64,
    ) -> Option<BlockChoice> {
        let neighbors = self.neighbors(mi_row, mi_col);
        let inter = self.choose_inter(mi_row, mi_col, bsl, neighbors, ceiling);
        // Intra prediction is not tried where motion compensation alone
        // already codes the block.
        if let Some(inter) = &inter
            && inter.info.skip
        {
            return Some(inter.clone());
        }
        let intra_ceiling = inter.as_ref().map_or(ceiling, |inter| inter.cost);
        self.over_ceiling = false;
        let mut intra = self.choose_intra(mi_row, mi_col, bsl, neighbors, intra_ceiling);
        // Ruling out a candidate by the ceiling can leave another one chosen
        // in its place, so where intra prediction then wins it is searched
        // again in full. It rarely wins against an inter candidate.
        if self.over_ceiling
            && intra
                .as_ref()
                .is_some_and(|intra| intra.cost <= intra_ceiling)
        {
            intra = self.choose_intra(mi_row, mi_col, bsl, neighbors, f64::INFINITY);
        }
        match (inter, intra) {
            (Some(inter), Some(intra)) if inter.cost < intra.cost => Some(inter),
            (_, Some(intra)) if intra.cost < ceiling => Some(intra),
            (inter, _) => inter,
        }
    }

    /// Codes one plane of a block with `tx_size` transforms (the plane's own
    /// size), predicting each transform block first for an intra
    /// `prediction`; an inter prediction is already in place. Returns `None`,
    /// leaving the plane part coded, once the candidate's cost reaches
    /// `bound`.
    #[allow(clippy::too_many_arguments)]
    fn code_plane(
        &mut self,
        plane: usize,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        tx_size: usize,
        prediction: Prediction,
        bound: Option<Bound>,
    ) -> Option<PlaneCoding> {
        let (intra, inter) = match prediction {
            Prediction::Intra(mode) => (Some(mode), None),
            Prediction::Inter(mv) => (None, Some(mv)),
        };
        let ss = usize::from(plane > 0);
        let n4 = (2 << bsl) >> ss;
        let step = 1 << tx_size;
        let (x0, y0) = ((mi_col * 8) >> ss, (mi_row * 8) >> ss);
        let (x4, y4) = ((mi_col * 2) >> ss, ((mi_row & 7) * 2) >> ss);
        let mut above = [false; 16];
        let mut left = [false; 16];
        above[..n4].copy_from_slice(&self.above_nonzero[plane][x4..x4 + n4]);
        left[..n4].copy_from_slice(&self.left_nonzero[plane][y4..y4 + n4]);
        let tx_type = match intra {
            Some(mode) if plane == 0 && tx_size < 3 => mode.tx_type(),
            _ => TxType::DctDct,
        };
        let mut coding = PlaneCoding {
            blocks: Vec::with_capacity((n4 / step) * (n4 / step)),
            error: 0,
            bits: 0.0,
        };
        let mut coded = false;
        for row in (0..n4).step_by(step) {
            for col in (0..n4).step_by(step) {
                let (x, y) = (x0 + col * 4, y0 + row * 4);
                if let Some(mode) = intra {
                    let stride = self.recon.strides[plane];
                    predict_intra(
                        &mut self.recon.planes[plane],
                        stride,
                        x,
                        y,
                        4 << tx_size,
                        mode,
                        y > y0 || mi_row > 0,
                        x > x0 || mi_col > 0,
                    );
                }
                let context = usize::from(above[col..col + step].contains(&true))
                    + usize::from(left[row..row + step].contains(&true));
                let (block, error, bits) =
                    self.code_residual(plane, x, y, tx_size, tx_type, inter, context);
                above[col..col + step].fill(block.eob > 0);
                left[row..row + step].fill(block.eob > 0);
                coded |= block.eob > 0;
                coding.blocks.push(block);
                coding.error += error;
                coding.bits += bits;
                if let Some(bound) = bound {
                    let (beaten, over_ceiling) = bound.reached(self.lambda, &coding, coded);
                    self.over_ceiling |= over_ceiling;
                    if beaten || over_ceiling {
                        return None;
                    }
                }
            }
        }
        Some(coding)
    }

    /// Quantizes, dequantizes and reconstructs one transform block in place;
    /// returns its levels, squared error and token bits. A block whose
    /// levels cost more than the distortion they remove is coded empty.
    ///
    /// The block reuses the coding of the same residual earlier in the
    /// superblock: an inter one, predicted along `inter`, from 8x8 up, and an
    /// intra one from 16x16 up. Smaller transforms cost less than looking
    /// them up.
    #[allow(clippy::too_many_arguments)]
    fn code_residual(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tx_size: usize,
        tx_type: TxType,
        inter: Option<Mv>,
        context: usize,
    ) -> (TxBlock, u64, f64) {
        let n = 4 << tx_size;
        let stride = self.recon.strides[plane];
        let plane_type = usize::from(plane > 0);
        let reference = usize::from(inter.is_some());
        let empty_bits = bit_cost(
            false,
            self.context.coef[tx_size][plane_type][reference][0][context][0],
        );
        // An empty block's levels are never read.
        let empty = TxBlock {
            levels: Vec::new(),
            eob: 0,
            tx_type,
        };
        let key = match inter {
            Some(mv) if tx_size > 0 => Some(ResidualKey::Inter {
                plane: plane as u8,
                tx_size: tx_size as u8,
                x: x as u16,
                y: y as u16,
                mv,
            }),
            // Hashing a prediction only pays for itself on the larger
            // transforms.
            None if tx_size >= 2 => Some(ResidualKey::Intra {
                plane: plane as u8,
                tx_size: tx_size as u8,
                tx_type: tx_type as u8,
                x: x as u16,
                y: y as u16,
                prediction: block_rows(&self.recon, plane, x, y, n).fold(0, |hash, row| {
                    row.chunks_exact(8).fold(hash, |hash, word| {
                        fx_mix(
                            hash,
                            u64::from_le_bytes(word.try_into().expect("eight samples")),
                        )
                    })
                }),
            }),
            _ => None,
        };
        if let Some(key) = &key
            && let Some(cached) = self.residual_cache.get_mut(key)
            && (cached.prediction.is_empty()
                || block_rows(&self.recon, plane, x, y, n)
                    .zip(cached.prediction.chunks_exact(n))
                    .all(|(row, cached)| row == cached))
        {
            let CachedResidual {
                prediction: _,
                prediction_error,
                eob,
                levels,
                reconstructed,
                error,
                bits,
            } = cached;
            if *eob == 0 {
                return (empty, *prediction_error, empty_bits);
            }
            let bits = *bits[context].get_or_insert_with(|| {
                coefficient_bits(
                    &self.token_costs[tx_size],
                    levels,
                    *eob,
                    tx_size,
                    tx_type,
                    plane_type,
                    reference,
                    context,
                )
            });
            if *prediction_error as f64 + self.lambda * empty_bits
                <= *error as f64 + self.lambda * bits
            {
                return (empty, *prediction_error, empty_bits);
            }
            for (row, pixels) in reconstructed.chunks_exact(n).enumerate() {
                let start = (y + row) * stride + x;
                self.recon.planes[plane][start..start + n].copy_from_slice(pixels);
            }
            let block = TxBlock {
                levels: levels.clone(),
                eob: *eob,
                tx_type,
            };
            return (block, *error, bits);
        }
        let Scratch {
            residual,
            coefficients,
            levels,
            dequantized,
            prediction,
            transform,
        } = &mut *self.scratch;
        let residual = &mut residual[..n * n];
        let start = y * stride + x;
        let prediction_error = simd::residual(
            &self.source.planes[plane][start..],
            stride,
            &self.recon.planes[plane][start..],
            stride,
            n,
            residual,
        );
        // An intra block's prediction, kept with its coding.
        let mut prediction_copy = match key {
            Some(ResidualKey::Intra { .. }) => {
                let mut copy = Vec::with_capacity(n * n);
                for row in block_rows(&self.recon, plane, x, y, n) {
                    copy.extend_from_slice(row);
                }
                copy
            }
            _ => Vec::new(),
        };
        // Remembers an empty coding of this residual.
        let mut cache_empty = |encoder: &mut Self| {
            if let Some(key) = key {
                encoder.residual_cache.insert(
                    key,
                    CachedResidual {
                        prediction: core::mem::take(&mut prediction_copy),
                        prediction_error,
                        eob: 0,
                        levels: Vec::new(),
                        reconstructed: Vec::new(),
                        error: prediction_error,
                        bits: [None; 3],
                    },
                );
            }
        };
        if prediction_error == 0 {
            cache_empty(self);
            return (empty, 0, empty_bits);
        }

        let coefficients = &mut coefficients[..n * n];
        forward_transform(residual, tx_size, tx_type, coefficients, transform);
        // A smaller rounding offset for inter residuals, as libvpx uses.
        let rounding = if inter.is_some() { 0.25 } else { 0.375 };
        let levels = &mut levels[..n * n];
        let dequantized = &mut dequantized[..n * n];
        let quantizer = Quantizer {
            dc_q: self.dc_q,
            ac_q: self.ac_q,
            half_step: tx_size == 3,
            rounding,
        };
        simd::quantize(coefficients, &quantizer, levels, dequantized);
        let (scan, _) = scan_order(tx_size, tx_type);
        let eob = scan
            .iter()
            .rposition(|&position| levels[position as usize] != 0)
            .map_or(0, |last| last + 1);
        if eob == 0 {
            cache_empty(self);
            return (empty, prediction_error, empty_bits);
        }
        let bits = coefficient_bits(
            &self.token_costs[tx_size],
            levels,
            eob,
            tx_size,
            tx_type,
            plane_type,
            reference,
            context,
        );
        // Restored if the residual is not worth its bits.
        let prediction = &mut prediction[..n * n];
        for row in 0..n {
            let start = (y + row) * stride + x;
            prediction[row * n..row * n + n]
                .copy_from_slice(&self.recon.planes[plane][start..start + n]);
        }
        let start = y * stride + x;
        inverse_transform_add(
            dequantized,
            tx_size,
            tx_type,
            eob,
            &mut self.recon.planes[plane][start..],
            stride,
        );
        let error = simd::sse(
            &self.source.planes[plane][start..],
            stride,
            &self.recon.planes[plane][start..],
            stride,
            n,
            n,
        );
        if let Some(key) = key {
            let mut cached_bits = [None; 3];
            cached_bits[context] = Some(bits);
            let mut reconstructed = Vec::with_capacity(n * n);
            for row in 0..n {
                let start = (y + row) * stride + x;
                reconstructed.extend_from_slice(&self.recon.planes[plane][start..start + n]);
            }
            self.residual_cache.insert(
                key,
                CachedResidual {
                    prediction: prediction_copy,
                    prediction_error,
                    eob,
                    levels: levels.to_vec(),
                    reconstructed,
                    error,
                    bits: cached_bits,
                },
            );
        }
        if prediction_error as f64 + self.lambda * empty_bits <= error as f64 + self.lambda * bits {
            for row in 0..n {
                let start = (y + row) * stride + x;
                self.recon.planes[plane][start..start + n]
                    .copy_from_slice(&prediction[row * n..row * n + n]);
            }
            return (empty, prediction_error, empty_bits);
        }
        let block = TxBlock {
            levels: levels.to_vec(),
            eob,
            tx_type,
        };
        (block, error, bits)
    }

    /// The best intra coding of the block, or `None` if it cannot cost
    /// `ceiling` or less.
    fn choose_intra(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        neighbors: Neighbors,
        ceiling: f64,
    ) -> Option<BlockChoice> {
        // The luma mode is chosen with the largest transform, whose prediction
        // reads only the block's own edges, and the smaller transforms are
        // then tried with that mode.
        let max_tx_size = self.max_tx_size(bsl);
        let mut best_luma: Option<LumaCandidate> = None;
        let try_luma = |encoder: &mut Self,
                        best_luma: &mut Option<LumaCandidate>,
                        tx_size: u8,
                        mode: IntraMode| {
            let symbol_bits = cost(|sink| {
                encoder.tx_size_symbol(sink, neighbors, bsl, tx_size);
                encoder.y_mode_symbol(sink, neighbors, bsl, mode);
            });
            // The mode is chosen with the largest transform before the
            // ceiling applies: ruling out the best mode there would try the
            // smaller transforms with another one.
            let bound = Bound {
                ceiling: if tx_size < max_tx_size {
                    ceiling
                } else {
                    f64::INFINITY
                },
                ..Bound::under(
                    best_luma.as_ref().map_or(f64::INFINITY, |best| best.cost),
                    symbol_bits,
                )
            };
            let Some(coding) = encoder.code_plane(
                0,
                mi_row,
                mi_col,
                bsl,
                usize::from(tx_size),
                Prediction::Intra(mode),
                Some(bound),
            ) else {
                return;
            };
            let total = coding.error as f64 + encoder.lambda * (coding.bits + symbol_bits);
            if best_luma.as_ref().is_none_or(|best| total < best.cost) {
                *best_luma = Some(LumaCandidate {
                    cost: total,
                    tx_size,
                    mode,
                    pixels: encoder.block_pixels(0, mi_row, mi_col, bsl),
                    coding,
                });
            }
        };
        let luma_order =
            self.intra_order(&[0], mi_row, mi_col, bsl, max_tx_size, |encoder, mode| {
                cost(|sink| encoder.y_mode_symbol(sink, neighbors, bsl, mode))
            });
        for mode in luma_order {
            try_luma(self, &mut best_luma, max_tx_size, mode);
        }
        let y_mode = best_luma.as_ref()?.mode;
        for tx_size in (0..max_tx_size).rev() {
            try_luma(self, &mut best_luma, tx_size, y_mode);
        }
        let LumaCandidate {
            tx_size,
            pixels: luma_pixels,
            coding: luma,
            ..
        } = best_luma?;
        let luma_symbol_bits = cost(|sink| {
            self.tx_size_symbol(sink, neighbors, bsl, tx_size);
            self.y_mode_symbol(sink, neighbors, bsl, y_mode);
        });
        let luma_coded = luma.blocks.iter().any(|block| block.eob > 0);
        let luma_floor_bits = if luma_coded { luma.bits } else { 0.0 };
        if luma.error as f64 + self.lambda * (luma_floor_bits + luma_symbol_bits) > ceiling {
            return None;
        }

        let uv_tx_size = plane_tx_size(tx_size, bsl, 1);
        let mut best_uv: Option<ChromaCandidate> = None;
        let chroma_order =
            self.intra_order(&[1, 2], mi_row, mi_col, bsl, tx_size, |encoder, mode| {
                cost(|sink| encoder.uv_mode_symbol(sink, y_mode, mode))
            });
        for mode in chroma_order {
            let symbol_bits = cost(|sink| self.uv_mode_symbol(sink, y_mode, mode));
            let mut bound = best_uv
                .as_ref()
                .map(|best| Bound::under(best.0, symbol_bits));
            let intra = Prediction::Intra(mode);
            let Some(u) = self.code_plane(1, mi_row, mi_col, bsl, uv_tx_size, intra, bound) else {
                continue;
            };
            if let Some(bound) = &mut bound {
                bound.error += u.error;
                bound.bits += u.bits;
            }
            let Some(v) = self.code_plane(2, mi_row, mi_col, bsl, uv_tx_size, intra, bound) else {
                continue;
            };
            let codings = [u, v];
            let bits = codings.iter().map(|coding| coding.bits).sum::<f64>() + symbol_bits;
            let error: u64 = codings.iter().map(|coding| coding.error).sum();
            let total = error as f64 + self.lambda * bits;
            if best_uv.as_ref().is_none_or(|best| total < best.0) {
                let pixels = [1, 2].map(|plane| self.block_pixels(plane, mi_row, mi_col, bsl));
                best_uv = Some((total, mode, codings, pixels));
            }
        }
        let (_, uv_mode, [u, v], [u_pixels, v_pixels]) =
            best_uv.expect("at least one chroma mode is evaluated");

        let skip = [&luma, &u, &v]
            .iter()
            .all(|coding| coding.blocks.iter().all(|block| block.eob == 0));
        let token_bits = if skip {
            0.0
        } else {
            luma.bits + u.bits + v.bits
        };
        let header_bits = cost(|sink| {
            self.skip_symbol(sink, neighbors, skip);
            if !self.is_key() {
                self.intra_inter_symbol(sink, neighbors, false);
            }
            self.tx_size_symbol(sink, neighbors, bsl, tx_size);
            self.y_mode_symbol(sink, neighbors, bsl, y_mode);
            self.uv_mode_symbol(sink, y_mode, uv_mode);
        });
        let error = luma.error + u.error + v.error;
        Some(BlockChoice {
            info: ModeInfo {
                is_inter: false,
                mode: y_mode as u8,
                mv: Mv::default(),
                skip,
                tx_size,
            },
            uv_mode,
            best_mv: Mv::default(),
            blocks: [luma.blocks, u.blocks, v.blocks],
            pixels: [luma_pixels, u_pixels, v_pixels],
            cost: error as f64 + self.lambda * (header_bits + token_bits),
        })
    }

    /// The intra modes in the order to try them in `planes`, best predicting
    /// first: by the squared error and mode `bits` of predicting the block's
    /// first transform block of `tx_size` (the luma size), which reads only
    /// the block's own edges.
    ///
    /// Every mode is still tried; the order only lets a candidate that cannot
    /// beat the best so far stop sooner.
    fn intra_order(
        &mut self,
        planes: &[usize],
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        tx_size: u8,
        bits: impl Fn(&Self, IntraMode) -> f64,
    ) -> [IntraMode; 4] {
        let mut ranked = IntraMode::ALL.map(|mode| {
            let mut error = 0_u64;
            for &plane in planes {
                let ss = usize::from(plane > 0);
                let size = 4 << plane_tx_size(tx_size, bsl, plane);
                let (x, y) = ((mi_col * 8) >> ss, (mi_row * 8) >> ss);
                let stride = self.recon.strides[plane];
                // Coding the plane predicts it again, so the reconstruction
                // holds this prediction meanwhile.
                predict_intra(
                    &mut self.recon.planes[plane],
                    stride,
                    x,
                    y,
                    size,
                    mode,
                    mi_row > 0,
                    mi_col > 0,
                );
                for row in y..y + size {
                    let range = row * stride + x..row * stride + x + size;
                    error += self.recon.planes[plane][range.clone()]
                        .iter()
                        .zip(&self.source.planes[plane][range])
                        .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
                        .sum::<u64>();
                }
            }
            (error as f64 + self.lambda * bits(self, mode), mode)
        });
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        ranked.map(|(_, mode)| mode)
    }

    /// The `NEARESTMV` and `NEARMV` candidates and the inter mode context, as
    /// `find_mv_refs` derives them for a block predicting from `LAST_FRAME`.
    fn mv_references(&self, mi_row: usize, mi_col: usize, bsl: usize) -> ([Mv; 2], usize) {
        let mut list = [Mv::default(); 2];
        let mut count = 0;
        let mut counter = 0_usize;
        let mut done = false;
        for (index, &(row_offset, col_offset)) in MV_REF_SEARCH[bsl].iter().enumerate() {
            let row = mi_row as isize + row_offset;
            let col = mi_col as isize + col_offset;
            if row < 0
                || col < 0
                || row >= self.geometry.mi_rows as isize
                || col >= self.geometry.mi_cols as isize
            {
                continue;
            }
            let candidate = self.mode_info[row as usize * self.geometry.mi_cols + col as usize];
            if index < 2 {
                counter += usize::from(MODE_TO_COUNTER[usize::from(candidate.mode)]);
            }
            if done || !candidate.is_inter {
                continue;
            }
            if count == 0 {
                list[0] = candidate.mv;
                count = 1;
            } else if candidate.mv != list[0] {
                list[1] = candidate.mv;
                done = true;
            }
        }
        // Then the co-located block of the previous frame.
        if !done
            && let Some(previous) = self.previous_mode_info
            && let candidate = previous[mi_row * self.geometry.mi_cols + mi_col]
            && candidate.is_inter
        {
            if count == 0 {
                list[0] = candidate.mv;
            } else if candidate.mv != list[0] {
                list[1] = candidate.mv;
            }
        }
        // Every inter block predicts from LAST_FRAME, so the searches over
        // other reference frames find nothing. Clamp as `clamp_mv_ref` does.
        let border = 16 * 8;
        let size = 1_i32 << bsl;
        let to_left = -((mi_col * 64) as i32);
        let to_right = (self.geometry.mi_cols as i32 - size - mi_col as i32) * 64;
        let to_top = -((mi_row * 64) as i32);
        let to_bottom = (self.geometry.mi_rows as i32 - size - mi_row as i32) * 64;
        for mv in &mut list {
            mv.col = mv.col.clamp(to_left - border, to_right + border);
            mv.row = mv.row.clamp(to_top - border, to_bottom + border);
        }
        (list, usize::from(COUNTER_TO_CONTEXT[counter]))
    }

    fn luma_sad(
        &self,
        reference: &Picture,
        mi_row: usize,
        mi_col: usize,
        size: usize,
        mv: Mv,
        limit: u32,
    ) -> u32 {
        let (dy, dx) = (mv.row / 8, mv.col / 8);
        let x0 = (mi_col * 8) as isize + dx as isize;
        let y0 = (mi_row * 8) as isize + dy as isize;
        let stride = self.source.strides[0];
        let max_x = self.geometry.width as isize - 1;
        let max_y = self.geometry.height as isize - 1;
        let last = size as isize - 1;
        let inside = x0 >= 0 && y0 >= 0 && x0 + last <= max_x && y0 + last <= max_y;
        if inside {
            return simd::sad(
                &self.source.planes[0][mi_row * 8 * stride + mi_col * 8..],
                stride,
                &reference.planes[0][y0 as usize * stride + x0 as usize..],
                stride,
                size,
                limit,
            );
        }
        let mut sad = 0_u32;
        for row in 0..size {
            let source_row =
                &self.source.planes[0][(mi_row * 8 + row) * stride + mi_col * 8..][..size];
            let y = (y0 + row as isize).clamp(0, max_y) as usize;
            for (column, &source) in source_row.iter().enumerate() {
                let x = (x0 + column as isize).clamp(0, max_x) as usize;
                sad += u32::from(source.abs_diff(reference.planes[0][y * stride + x]));
            }
            if sad >= limit {
                return sad;
            }
        }
        sad
    }

    /// Estimates the frame's whole-sample motion from its luma projections,
    /// as libvpx's `vp9_int_pro_motion_estimation` does for a block: the
    /// column sums give the horizontal motion and the row sums the vertical,
    /// each by the offset within the search range that matches the
    /// reference's sums best. Content can leave one direction's sums nearly
    /// flat, and its estimate noise, so each direction is also offered on its
    /// own.
    ///
    /// A diamond search from the neighbours' vectors alone can settle in a
    /// local minimum on fine texture. Once a frame's first blocks miss the
    /// motion and fall back to intra, the blocks after them have no vectors
    /// to start from either, and a panning frame can code nearly all intra:
    /// small changes to the reference, such as the loop filter's, then swing
    /// the size of the whole sequence (issue #583).
    fn projected_motion(&self, reference: &Picture) -> Vec<Mv> {
        const RANGE: usize = 64;
        let stride = self.source.strides[0];
        let (width, height) = (self.geometry.width, self.geometry.height);
        let source = |x: usize, y: usize| u32::from(self.source.planes[0][y * stride + x]);
        // The reference sample at an offset, extended past the edges as
        // motion compensation extends it.
        let reference = |x: isize, y: isize| {
            let x = x.clamp(0, width as isize - 1) as usize;
            let y = y.clamp(0, height as isize - 1) as usize;
            u32::from(reference.planes[0][y * stride + x])
        };
        let best_offset = |projection: &[u32], shifted: &[u32]| {
            (0..=2 * RANGE)
                .min_by_key(|&offset| {
                    let sad: u64 = projection
                        .iter()
                        .zip(&shifted[offset..])
                        .map(|(&a, &b)| u64::from(a.abs_diff(b)))
                        .sum();
                    // Prefer the smaller offset among equals.
                    (sad, offset.abs_diff(RANGE))
                })
                .expect("the range is not empty") as i32
                - RANGE as i32
        };
        let columns: Vec<u32> = (0..width)
            .map(|x| (0..height).map(|y| source(x, y)).sum())
            .collect();
        let reference_columns: Vec<u32> = (-(RANGE as isize)..(width + RANGE) as isize)
            .map(|x| (0..height).map(|y| reference(x, y as isize)).sum())
            .collect();
        let rows: Vec<u32> = (0..height)
            .map(|y| (0..width).map(|x| source(x, y)).sum())
            .collect();
        let reference_rows: Vec<u32> = (-(RANGE as isize)..(height + RANGE) as isize)
            .map(|y| (0..width).map(|x| reference(x as isize, y)).sum())
            .collect();
        let row = best_offset(&rows, &reference_rows) * 8;
        let col = best_offset(&columns, &reference_columns) * 8;
        vec![Mv { row, col }, Mv { row: 0, col }, Mv { row, col: 0 }]
    }

    /// Finds the whole-sample motion vector with the smallest luma SAD,
    /// starting from `starts` and the frame's projected motion.
    fn search_motion(
        &self,
        reference: &Picture,
        mi_row: usize,
        mi_col: usize,
        size: usize,
        starts: [Mv; 2],
    ) -> Mv {
        // Kept within 64 samples of the block so every vector stays in the
        // cheap motion vector classes.
        let range = 64 * 8;
        let mut best_mv = Mv::default();
        let mut best_sad = self.luma_sad(reference, mi_row, mi_col, size, best_mv, u32::MAX);
        for start in starts.into_iter().chain(self.projected_mvs.iter().copied()) {
            let start = Mv {
                row: start.row / 8 * 8,
                col: start.col / 8 * 8,
            };
            if start.row.abs() <= range && start.col.abs() <= range {
                let sad = self.luma_sad(reference, mi_row, mi_col, size, start, best_sad);
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = start;
                }
            }
        }
        for step in [8, 4, 2, 1] {
            let delta = step * 8;
            for _ in 0..16 {
                let mut improved = false;
                let center = best_mv;
                for (row, col) in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ] {
                    let candidate = Mv {
                        row: center.row + row * delta,
                        col: center.col + col * delta,
                    };
                    if candidate.row.abs() > range || candidate.col.abs() > range {
                        continue;
                    }
                    let sad = self.luma_sad(reference, mi_row, mi_col, size, candidate, best_sad);
                    if sad < best_sad {
                        best_sad = sad;
                        best_mv = candidate;
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
        }
        best_mv
    }

    /// The best inter coding of the block, or `None` in a key frame or if
    /// none costs less than `ceiling`.
    fn choose_inter(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        neighbors: Neighbors,
        ceiling: f64,
    ) -> Option<BlockChoice> {
        let reference = self.reference?;
        let (candidates, mode_context) = self.mv_references(mi_row, mi_col, bsl);
        let [nearest, near] = candidates;
        let size = 8 << bsl;
        let searched = self.search_motion(reference, mi_row, mi_col, size, candidates);
        // Also try the reference vector that predicts best, if it predicts
        // nearly as well as the searched one: it codes in fewer bits.
        let searched_sad = self.luma_sad(reference, mi_row, mi_col, size, searched, u32::MAX);
        let mut vectors = vec![searched];
        if let Some((alternative, sad)) = [Mv::default(), nearest, near]
            .into_iter()
            .filter(|&mv| mv != searched)
            .map(|mv| {
                (
                    mv,
                    self.luma_sad(reference, mi_row, mi_col, size, mv, u32::MAX),
                )
            })
            .min_by_key(|&(_, sad)| sad)
            && sad <= searched_sad + searched_sad / 8
        {
            vectors.push(alternative);
        }

        let max_tx_size = self.max_tx_size(bsl);
        let mut best: Option<BlockChoice> = None;
        for mv in vectors {
            // The cheapest mode that codes this vector.
            let (mode, mode_bits) = [ZEROMV, NEARESTMV, NEARMV, NEWMV]
                .into_iter()
                .filter(|&mode| match mode {
                    ZEROMV => mv == Mv::default(),
                    NEARESTMV => mv == nearest,
                    NEARMV => mv == near,
                    _ => true,
                })
                .map(|mode| {
                    let bits = cost(|sink| {
                        self.intra_inter_symbol(sink, neighbors, true);
                        self.inter_symbols(sink, neighbors, mode_context, mode, mv, nearest);
                    });
                    (mode, bits)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("NEWMV codes any vector");

            // Predict every plane, then try each transform size on the residual.
            let geometry = self.geometry;
            let mut prediction_error = 0_u64;
            for plane in 0..3 {
                let (size, width, height, scale) = if plane == 0 {
                    (8 << bsl, geometry.width, geometry.height, 2)
                } else {
                    (
                        4 << bsl,
                        geometry.chroma_width(),
                        geometry.chroma_height(),
                        1,
                    )
                };
                let reference_plane = ReferencePlane {
                    pixels: &reference.planes[plane],
                    stride: reference.strides[plane],
                    width,
                    height,
                };
                let mut prediction = vec![0_u8; size * size];
                let ss = usize::from(plane > 0);
                let (x, y) = ((mi_col * 8) >> ss, (mi_row * 8) >> ss);
                predict_inter(
                    &reference_plane,
                    x,
                    y,
                    size,
                    mv.row * scale,
                    mv.col * scale,
                    &mut prediction,
                );
                let stride = self.recon.strides[plane];
                let start = y * stride + x;
                prediction_error += simd::sse(
                    &self.source.planes[plane][start..],
                    stride,
                    &prediction,
                    size,
                    size,
                    size,
                );
                for (row, predicted) in prediction.chunks_exact(size).enumerate() {
                    let start = start + row * stride;
                    self.recon.planes[plane][start..start + size].copy_from_slice(predicted);
                }
            }
            let predicted: [Vec<u8>; 3] =
                core::array::from_fn(|plane| self.block_pixels(plane, mi_row, mi_col, bsl));
            let info = ModeInfo {
                is_inter: true,
                mode,
                mv,
                skip: true,
                tx_size: max_tx_size,
            };

            let skip_bits = cost(|sink| self.skip_symbol(sink, neighbors, true)) + mode_bits;
            let skip_cost = prediction_error as f64 + self.lambda * skip_bits;
            if best.as_ref().is_none_or(|best| skip_cost < best.cost) {
                best = Some(BlockChoice {
                    info,
                    uv_mode: IntraMode::Dc,
                    best_mv: nearest,
                    blocks: [Vec::new(), Vec::new(), Vec::new()],
                    pixels: predicted.clone(),
                    cost: skip_cost,
                });
            }

            // Chroma caps its transform at its own block size, so several luma
            // sizes can share one chroma coding; it is coded once.
            let mut chroma: Option<(usize, [PlaneCoding; 2], [Vec<u8>; 2])> = None;
            for tx_size in 0..=max_tx_size {
                if tx_size > 0 {
                    self.load_pixels(0, mi_row, mi_col, bsl, &predicted[0]);
                }
                let uv_tx_size = plane_tx_size(tx_size, bsl, 1);
                let header_bits = cost(|sink| {
                    self.skip_symbol(sink, neighbors, false);
                    self.tx_size_symbol(sink, neighbors, bsl, tx_size);
                }) + mode_bits;
                let to_beat = best.as_ref().map_or(ceiling, |best| best.cost.min(ceiling));
                let bound = Some(Bound::under(to_beat, header_bits));
                let tx = usize::from(tx_size);
                let inter = Prediction::Inter(mv);
                let Some(luma) = self.code_plane(0, mi_row, mi_col, bsl, tx, inter, bound) else {
                    continue;
                };
                let [u, v] = match &chroma {
                    Some((size, codings, pixels)) if *size == uv_tx_size => {
                        for (plane, pixels) in [1, 2].into_iter().zip(pixels) {
                            self.load_pixels(plane, mi_row, mi_col, bsl, pixels);
                        }
                        codings.clone()
                    }
                    _ => {
                        if tx_size > 0 {
                            for plane in [1, 2] {
                                self.load_pixels(plane, mi_row, mi_col, bsl, &predicted[plane]);
                            }
                        }
                        // Not bounded: the coding is kept for other sizes.
                        let codings = [1, 2].map(|plane| {
                            self.code_plane(plane, mi_row, mi_col, bsl, uv_tx_size, inter, None)
                                .expect("an unbounded plane is coded whole")
                        });
                        let pixels =
                            [1, 2].map(|plane| self.block_pixels(plane, mi_row, mi_col, bsl));
                        chroma = Some((uv_tx_size, codings.clone(), pixels));
                        codings
                    }
                };
                let codings = [luma, u, v];
                if codings
                    .iter()
                    .all(|coding| coding.blocks.iter().all(|block| block.eob == 0))
                {
                    // Coded empty, this is the skipped block already tried.
                    continue;
                }
                let token_bits: f64 = codings.iter().map(|coding| coding.bits).sum();
                let error: u64 = codings.iter().map(|coding| coding.error).sum();
                let total = error as f64 + self.lambda * (header_bits + token_bits);
                if best.as_ref().is_none_or(|best| total < best.cost) {
                    let [y, u, v] = codings;
                    best = Some(BlockChoice {
                        info: ModeInfo {
                            skip: false,
                            tx_size,
                            ..info
                        },
                        uv_mode: IntraMode::Dc,
                        best_mv: nearest,
                        blocks: [y.blocks, u.blocks, v.blocks],
                        pixels: core::array::from_fn(|plane| {
                            self.block_pixels(plane, mi_row, mi_col, bsl)
                        }),
                        cost: total,
                    });
                }
            }
        }
        best.filter(|best| best.cost < ceiling)
    }

    /// Writes the partition symbol of the square of `bsl` at `(mi_row,
    /// mi_col)`. At the frame's bottom and right edges a square that does
    /// not fit is always split, which the bitstream then codes in one bit or
    /// none.
    fn partition_symbol<S: BoolSink>(
        &self,
        sink: &mut S,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        split: bool,
    ) {
        let half = (1 << bsl) >> 1;
        let has_rows = mi_row + half < self.geometry.mi_rows;
        let has_cols = mi_col + half < self.geometry.mi_cols;
        let above = (self.above_partition[mi_col] >> bsl) & 1;
        let left = (self.left_partition[mi_row & 7] >> bsl) & 1;
        let context = usize::from(left * 2 + above) + bsl * 4;
        let table = if self.is_key() {
            &KF_PARTITION_PROBS
        } else {
            &self.context.partition
        };
        let probs = &table[context * 3..context * 3 + 3];
        // The decoder counts every partition, including the splits the frame
        // edges imply.
        if let Some(counts) = sink.counts() {
            counts.partition[context][if split { 3 } else { 0 }] += 1;
        }
        if bsl == 0 {
            debug_assert!(!split, "8x8 blocks are never split");
            sink.tree(&PARTITION_TREE, probs, 0);
            return;
        }
        match (has_rows, has_cols) {
            (true, true) => sink.tree(&PARTITION_TREE, probs, if split { 3 } else { 0 }),
            (false, true) => sink.write(true, probs[1]),
            (true, false) => sink.write(true, probs[2]),
            (false, false) => {}
        }
        debug_assert!(
            split || (has_rows && has_cols),
            "a partial square must split"
        );
    }

    fn skip_symbol<S: BoolSink>(&self, sink: &mut S, neighbors: Neighbors, skip: bool) {
        let context = usize::from(neighbors.above.is_some_and(|info| info.skip))
            + usize::from(neighbors.left.is_some_and(|info| info.skip));
        sink.write(skip, self.context.skip[context]);
        if let Some(counts) = sink.counts() {
            counts.skip[context][usize::from(skip)] += 1;
        }
    }

    fn intra_inter_symbol<S: BoolSink>(&self, sink: &mut S, neighbors: Neighbors, inter: bool) {
        let context = intra_inter_context(neighbors);
        sink.write(inter, self.context.intra_inter[context]);
        if let Some(counts) = sink.counts() {
            counts.intra_inter[context][usize::from(inter)] += 1;
        }
    }

    /// Writes the transform size as `read_tx_size` reads it with
    /// `TX_MODE_SELECT`; nothing is coded with `ONLY_4X4`.
    fn tx_size_symbol<S: BoolSink>(
        &self,
        sink: &mut S,
        neighbors: Neighbors,
        bsl: usize,
        tx_size: u8,
    ) {
        if !self.tools.larger_transforms {
            return;
        }
        let max = self.max_tx_size(bsl);
        let side = |info: Option<ModeInfo>| match info {
            Some(info) if !info.skip => info.tx_size,
            _ => max,
        };
        let (mut above, mut left) = (side(neighbors.above), side(neighbors.left));
        if neighbors.left.is_none() {
            left = above;
        }
        if neighbors.above.is_none() {
            above = left;
        }
        let context = usize::from(above + left > max);
        let probs: &[u8] = match max {
            1 => &self.context.tx_8x8[context..context + 1],
            2 => &self.context.tx_16x16[context * 2..context * 2 + 2],
            _ => &self.context.tx_32x32[context * 3..context * 3 + 3],
        };
        if let Some(counts) = sink.counts() {
            counts.tx[usize::from(max) - 1][context][usize::from(tx_size)] += 1;
        }
        sink.write(tx_size != 0, probs[0]);
        if tx_size != 0 && max >= 2 {
            sink.write(tx_size != 1, probs[1]);
            if tx_size != 1 && max >= 3 {
                sink.write(tx_size != 2, probs[2]);
            }
        }
    }

    fn y_mode_symbol<S: BoolSink>(
        &self,
        sink: &mut S,
        neighbors: Neighbors,
        bsl: usize,
        mode: IntraMode,
    ) {
        let probs = if self.is_key() {
            let above = neighbors.above.map_or(0, |info| usize::from(info.mode));
            let left = neighbors.left.map_or(0, |info| usize::from(info.mode));
            let index = (above * 10 + left) * 9;
            &KF_Y_MODE_PROBS[index..index + 9]
        } else {
            // `size_group_lookup` of the square block sizes.
            let group = (bsl + 1).min(3);
            if let Some(counts) = sink.counts() {
                counts.y_mode[group][mode as usize] += 1;
            }
            &self.context.y_mode[group * 9..group * 9 + 9]
        };
        sink.tree(&INTRA_MODE_TREE, probs, mode as u8);
    }

    fn uv_mode_symbol<S: BoolSink>(&self, sink: &mut S, y_mode: IntraMode, mode: IntraMode) {
        let index = (y_mode as usize) * 9;
        let table = if self.is_key() {
            &KF_UV_MODE_PROBS
        } else {
            if let Some(counts) = sink.counts() {
                counts.uv_mode[y_mode as usize][mode as usize] += 1;
            }
            &self.context.uv_mode
        };
        sink.tree(&INTRA_MODE_TREE, &table[index..index + 9], mode as u8);
    }

    /// The reference frame, inter mode and any new motion vector of an inter
    /// block.
    fn inter_symbols<S: BoolSink>(
        &self,
        sink: &mut S,
        neighbors: Neighbors,
        mode_context: usize,
        mode: u8,
        mv: Mv,
        nearest: Mv,
    ) {
        // A single LAST_FRAME reference: the first single_ref bit is zero.
        let single_ref = single_ref_context(neighbors);
        sink.write(false, self.context.single_ref[single_ref * 2]);
        sink.tree(
            &INTER_MODE_TREE,
            &self.context.inter_mode[mode_context * 3..mode_context * 3 + 3],
            mode - NEARESTMV,
        );
        if let Some(counts) = sink.counts() {
            counts.single_ref[single_ref][0] += 1;
            counts.inter_mode[mode_context][usize::from(mode - NEARESTMV)] += 1;
        }
        if mode == NEWMV {
            write_mv(
                sink,
                self.context,
                Mv {
                    row: mv.row - nearest.row,
                    col: mv.col - nearest.col,
                },
            );
        }
    }

    fn write_mode_info(
        &self,
        writer: &mut TileWriter,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        choice: &BlockChoice,
    ) {
        let neighbors = self.neighbors(mi_row, mi_col);
        let info = choice.info;
        self.skip_symbol(writer, neighbors, info.skip);
        if !self.is_key() {
            self.intra_inter_symbol(writer, neighbors, info.is_inter);
        }
        if !info.is_inter || !info.skip {
            self.tx_size_symbol(writer, neighbors, bsl, info.tx_size);
        }
        if !info.is_inter {
            let y_mode = intra_mode(info.mode);
            self.y_mode_symbol(writer, neighbors, bsl, y_mode);
            self.uv_mode_symbol(writer, y_mode, choice.uv_mode);
            return;
        }
        let (candidates, mode_context) = self.mv_references(mi_row, mi_col, bsl);
        debug_assert_eq!(
            candidates[0], choice.best_mv,
            "NEARESTMV changed since the search"
        );
        debug_assert!(match info.mode {
            NEARESTMV => info.mv == candidates[0],
            NEARMV => info.mv == candidates[1],
            ZEROMV => info.mv == Mv::default(),
            _ => true,
        });
        self.inter_symbols(
            writer,
            neighbors,
            mode_context,
            info.mode,
            info.mv,
            choice.best_mv,
        );
    }

    fn write_tokens(
        &self,
        writer: &mut TileWriter,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        choice: &BlockChoice,
    ) {
        if choice.info.skip {
            return;
        }
        // The contexts start from the state before the block, and evolve
        // within it as `update_contexts` will leave them.
        for plane in 0..3 {
            let ss = usize::from(plane > 0);
            let n4 = (2 << bsl) >> ss;
            let (x4, y4) = ((mi_col * 2) >> ss, ((mi_row & 7) * 2) >> ss);
            let mut above = [false; 16];
            let mut left = [false; 16];
            above[..n4].copy_from_slice(&self.above_nonzero[plane][x4..x4 + n4]);
            left[..n4].copy_from_slice(&self.left_nonzero[plane][y4..y4 + n4]);
            let tx_size = plane_tx_size(choice.info.tx_size, bsl, plane);
            let step = 1 << tx_size;
            let mut blocks = choice.blocks[plane].iter();
            for row in (0..n4).step_by(step) {
                for col in (0..n4).step_by(step) {
                    let block = blocks.next().expect("one block per transform");
                    let context = usize::from(above[col..col + step].contains(&true))
                        + usize::from(left[row..row + step].contains(&true));
                    write_coefficients(
                        writer,
                        &self.context.coef[tx_size],
                        &block.levels,
                        block.eob,
                        tx_size,
                        block.tx_type,
                        usize::from(plane > 0) * 2 + usize::from(choice.info.is_inter),
                        context,
                    );
                    above[col..col + step].fill(block.eob > 0);
                    left[row..row + step].fill(block.eob > 0);
                }
            }
        }
    }
}

/// The rows of the `n` x `n` block of `plane` whose top-left sample is at
/// `(x, y)`.
fn block_rows(
    picture: &Picture,
    plane: usize,
    x: usize,
    y: usize,
    n: usize,
) -> impl Iterator<Item = &[u8]> {
    let stride = picture.strides[plane];
    (0..n).map(move |row| &picture.planes[plane][(y + row) * stride + x..][..n])
}

/// The total bits of the symbols `write` codes.
fn cost(write: impl FnOnce(&mut BitCost)) -> f64 {
    let mut counter = BitCost::default();
    write(&mut counter);
    counter.0
}

/// A plane's transform size: chroma blocks are half the luma size, and
/// `uv_txsize_lookup` caps their transform at that.
fn plane_tx_size(tx_size: u8, bsl: usize, plane: usize) -> usize {
    if plane == 0 {
        usize::from(tx_size)
    } else {
        usize::from(tx_size).min(bsl)
    }
}

fn intra_mode(mode: u8) -> IntraMode {
    IntraMode::ALL
        .into_iter()
        .find(|&candidate| candidate as u8 == mode)
        .expect("the encoder only chooses its own intra modes")
}

fn intra_inter_context(neighbors: Neighbors) -> usize {
    match (neighbors.above, neighbors.left) {
        (Some(above), Some(left)) => {
            let (above_intra, left_intra) = (!above.is_inter, !left.is_inter);
            if above_intra && left_intra {
                3
            } else {
                usize::from(above_intra || left_intra)
            }
        }
        (Some(edge), None) | (None, Some(edge)) => 2 * usize::from(!edge.is_inter),
        (None, None) => 0,
    }
}

/// `vp9_get_pred_context_single_ref_p1` when every inter block uses LAST_FRAME.
fn single_ref_context(neighbors: Neighbors) -> usize {
    if neighbors.above.is_some_and(|info| info.is_inter)
        || neighbors.left.is_some_and(|info| info.is_inter)
    {
        4
    } else {
        2
    }
}

/// `vp9_scan_orders[tx_size][tx_type]`: the scan and its neighbour pairs.
fn scan_order(tx_size: usize, tx_type: TxType) -> (&'static [i16], &'static [i16]) {
    match (tx_size, tx_type) {
        (0, TxType::AdstDct) => (&shared::ROW_SCAN_4X4, &shared::ROW_SCAN_4X4_NEIGHBORS),
        (0, TxType::DctAdst) => (&shared::COL_SCAN_4X4, &shared::COL_SCAN_4X4_NEIGHBORS),
        (0, _) => (
            &shared::DEFAULT_SCAN_4X4,
            &shared::DEFAULT_SCAN_4X4_NEIGHBORS,
        ),
        (1, TxType::AdstDct) => (&shared::ROW_SCAN_8X8, &shared::ROW_SCAN_8X8_NEIGHBORS),
        (1, TxType::DctAdst) => (&shared::COL_SCAN_8X8, &shared::COL_SCAN_8X8_NEIGHBORS),
        (1, _) => (
            &shared::DEFAULT_SCAN_8X8,
            &shared::DEFAULT_SCAN_8X8_NEIGHBORS,
        ),
        (2, TxType::AdstDct) => (&shared::ROW_SCAN_16X16, &shared::ROW_SCAN_16X16_NEIGHBORS),
        (2, TxType::DctAdst) => (&shared::COL_SCAN_16X16, &shared::COL_SCAN_16X16_NEIGHBORS),
        (2, _) => (
            &shared::DEFAULT_SCAN_16X16,
            &shared::DEFAULT_SCAN_16X16_NEIGHBORS,
        ),
        _ => (
            &shared::DEFAULT_SCAN_32X32,
            &shared::DEFAULT_SCAN_32X32_NEIGHBORS,
        ),
    }
}

/// The token cache energy class of a coefficient magnitude.
fn energy_class(magnitude: u32) -> u8 {
    match magnitude {
        0 => 0,
        1 => 1,
        2 => 2,
        3 | 4 => 3,
        5..=10 => 4,
        _ => 5,
    }
}

/// Writes one transform block's tokens as `decode_coefs` reads them, and
/// counts them as it does, from `levels` in raster order with `eob` the end
/// of block in scan order. `block_type` is the plane type (luma or chroma)
/// times two plus the reference type (intra or inter).
#[allow(clippy::too_many_arguments)]
fn write_coefficients<S: BoolSink>(
    sink: &mut S,
    coef_probs: &CoefProbs,
    levels: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: TxType,
    block_type: usize,
    mut context: usize,
) {
    let (scan, neighbors) = scan_order(tx_size, tx_type);
    let bands: &[u8] = if tx_size == 0 {
        &shared::COEFBAND_TRANS_4X4
    } else {
        &shared::COEFBAND_TRANS_8X8PLUS
    };
    let probs = &coef_probs[block_type / 2][block_type % 2];
    let mut cache = [0_u8; 1024];
    let mut previous_zero = false;
    for c in 0..eob {
        let band = usize::from(bands[c]);
        let model = &probs[band][context];
        if !previous_zero {
            sink.write(true, model[0]);
        }
        let position = scan[c] as usize;
        let level = levels[position];
        let magnitude = level.unsigned_abs();
        if let Some(counts) = sink.counts() {
            let (pt, reference) = (block_type / 2, block_type % 2);
            if !previous_zero {
                counts.eob_branch[tx_size][pt][reference][band][context] += 1;
            }
            // ZERO_TOKEN, ONE_TOKEN or TWO_TOKEN (any larger magnitude).
            counts.coef[tx_size][pt][reference][band][context][magnitude.min(2) as usize] += 1;
        }
        let probs = model;
        if magnitude == 0 {
            sink.write(false, probs[1]);
            previous_zero = true;
        } else {
            sink.write(true, probs[1]);
            write_magnitude(sink, magnitude, probs[2]);
            sink.write(level < 0, 128);
            previous_zero = false;
        }
        cache[position] = energy_class(magnitude);
        let next = c + 1;
        context = (1
            + usize::from(cache[neighbors[next * 2] as usize])
            + usize::from(cache[neighbors[next * 2 + 1] as usize]))
            >> 1;
    }
    if eob < scan.len() {
        let band = usize::from(bands[eob]);
        sink.write(false, probs[band][context][0]);
        if let Some(counts) = sink.counts() {
            let (pt, reference) = (block_type / 2, block_type % 2);
            counts.eob_branch[tx_size][pt][reference][band][context] += 1;
            counts.coef[tx_size][pt][reference][band][context][3] += 1; // EOB_MODEL_TOKEN
        }
    }
}

/// Magnitudes up to this have their token costs precomputed per context.
const COSTED_MAGNITUDES: usize = 16;

/// What each token costs in one coefficient context, in bits, as
/// [`write_coefficients`] codes it.
struct ContextCosts {
    /// The "more coefficients" bit before a token that follows a nonzero one.
    more: f64,
    /// The end of block.
    end: f64,
    zero: f64,
    /// A nonzero level of each magnitude from 1, with its sign.
    nonzero: [f64; COSTED_MAGNITUDES],
    /// The context's probabilities, for magnitudes past `nonzero`.
    probs: [u8; 3],
}

impl ContextCosts {
    fn new(probs: &[u8; 3]) -> Self {
        // The first band has only three contexts; the rest of its table is
        // zeros, which no block reads.
        if probs.contains(&0) {
            return Self {
                more: f64::INFINITY,
                end: f64::INFINITY,
                zero: f64::INFINITY,
                nonzero: [f64::INFINITY; COSTED_MAGNITUDES],
                probs: *probs,
            };
        }
        Self {
            more: bit_cost(true, probs[0]),
            end: bit_cost(false, probs[0]),
            zero: bit_cost(false, probs[1]),
            nonzero: core::array::from_fn(|index| nonzero_cost(index as u32 + 1, probs)),
            probs: *probs,
        }
    }
}

/// What a nonzero level of `magnitude` costs with its sign, in bits.
fn nonzero_cost(magnitude: u32, probs: &[u8; 3]) -> f64 {
    cost(|sink| {
        sink.write(true, probs[1]);
        write_magnitude(sink, magnitude, probs[2]);
        sink.write(false, 128);
    })
}

/// The token costs of every coefficient context of a transform size, by
/// plane type, reference, band and context.
type TokenCosts = [[[[ContextCosts; 6]; 6]; 2]; 2];

/// The token costs of every transform size under a frame's coefficient
/// probabilities.
fn token_costs(coef: &[CoefProbs; 4]) -> Box<[TokenCosts; 4]> {
    Box::new(core::array::from_fn(|tx_size| {
        core::array::from_fn(|plane_type| {
            core::array::from_fn(|reference| {
                core::array::from_fn(|band| {
                    core::array::from_fn(|context| {
                        ContextCosts::new(&coef[tx_size][plane_type][reference][band][context])
                    })
                })
            })
        })
    }))
}

/// The bits [`write_coefficients`] codes one transform block's tokens in,
/// from the costs of whole tokens: the search costs every transform block it
/// tries, and summing bit by bit dominated it.
#[allow(clippy::too_many_arguments)]
fn coefficient_bits(
    costs: &TokenCosts,
    levels: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: TxType,
    plane_type: usize,
    reference: usize,
    context: usize,
) -> f64 {
    let bits = match tx_size {
        0 => token_bits::<16>,
        1 => token_bits::<64>,
        2 => token_bits::<256>,
        _ => token_bits::<1024>,
    };
    bits(
        costs, levels, eob, tx_size, tx_type, plane_type, reference, context,
    )
}

/// [`coefficient_bits`] for a transform of `SIZE` coefficients, following
/// [`write_coefficients`] token by token.
#[allow(clippy::too_many_arguments)]
fn token_bits<const SIZE: usize>(
    costs: &TokenCosts,
    levels: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: TxType,
    plane_type: usize,
    reference: usize,
    mut context: usize,
) -> f64 {
    let (scan, neighbors) = scan_order(tx_size, tx_type);
    let bands: &[u8] = if tx_size == 0 {
        &shared::COEFBAND_TRANS_4X4
    } else {
        &shared::COEFBAND_TRANS_8X8PLUS
    };
    let costs = &costs[plane_type][reference];
    let mut bits = 0.0;
    let mut cache = [0_u8; SIZE];
    let mut previous_zero = false;
    for c in 0..eob {
        let band = usize::from(bands[c]);
        let token = &costs[band][context];
        if !previous_zero {
            bits += token.more;
        }
        let position = scan[c] as usize;
        let magnitude = levels[position].unsigned_abs();
        if magnitude == 0 {
            bits += token.zero;
            previous_zero = true;
        } else {
            bits += match token.nonzero.get(magnitude as usize - 1) {
                Some(&cost) => cost,
                None => nonzero_cost(magnitude, &token.probs),
            };
            previous_zero = false;
        }
        cache[position] = energy_class(magnitude);
        let next = c + 1;
        context = (1
            + usize::from(cache[neighbors[next * 2] as usize])
            + usize::from(cache[neighbors[next * 2 + 1] as usize]))
            >> 1;
    }
    if eob < scan.len() {
        bits += costs[usize::from(bands[eob])][context].end;
    }
    bits
}

fn write_magnitude<S: BoolSink>(sink: &mut S, magnitude: u32, pivot: u8) {
    if magnitude == 1 {
        sink.write(false, pivot);
        return;
    }
    sink.write(true, pivot);
    let pareto = &PARETO8_FULL[(usize::from(pivot) - 1) * 8..][..8];
    let extra = |sink: &mut S, value: u32, probs: &[u8]| {
        for (bit, &probability) in probs.iter().enumerate() {
            sink.write((value >> (probs.len() - 1 - bit)) & 1 != 0, probability);
        }
    };
    if magnitude <= 4 {
        sink.write(false, pareto[0]);
        if magnitude == 2 {
            sink.write(false, pareto[1]);
        } else {
            sink.write(true, pareto[1]);
            sink.write(magnitude == 4, pareto[2]);
        }
        return;
    }
    sink.write(true, pareto[0]);
    if magnitude <= 10 {
        sink.write(false, pareto[3]);
        if magnitude <= 6 {
            sink.write(false, pareto[4]);
            extra(sink, magnitude - 5, CAT_PROBS[0]);
        } else {
            sink.write(true, pareto[4]);
            extra(sink, magnitude - 7, CAT_PROBS[1]);
        }
        return;
    }
    sink.write(true, pareto[3]);
    if magnitude <= 34 {
        sink.write(false, pareto[5]);
        if magnitude <= 18 {
            sink.write(false, pareto[6]);
            extra(sink, magnitude - 11, CAT_PROBS[2]);
        } else {
            sink.write(true, pareto[6]);
            extra(sink, magnitude - 19, CAT_PROBS[3]);
        }
    } else {
        sink.write(true, pareto[5]);
        if magnitude <= 66 {
            sink.write(false, pareto[7]);
            extra(sink, magnitude - 35, CAT_PROBS[4]);
        } else {
            sink.write(true, pareto[7]);
            extra(sink, magnitude - 67, &CAT6_PROBS);
        }
    }
}

/// The class and in-class offset of a motion vector magnitude minus one.
fn mv_class(z: u32) -> (usize, u32) {
    let class = if z >= 2 * 4096 {
        10
    } else {
        (z >> 3).checked_ilog2().map_or(0, |log| log as usize)
    };
    let base = if class == 0 { 0 } else { 2 << (class + 2) };
    (class, z - base)
}

/// Writes a motion vector difference, counting it as `vp9_inc_mv` does.
fn write_mv<S: BoolSink>(sink: &mut S, context: &FrameContext, difference: Mv) {
    let joint = usize::from(difference.row != 0) * 2 + usize::from(difference.col != 0);
    sink.tree(&MV_JOINT_TREE, &context.mv_joints, joint as u8);
    if let Some(counts) = sink.counts() {
        counts.mv_joints[joint] += 1;
    }
    for (component, value) in [(0, difference.row), (1, difference.col)] {
        if value == 0 {
            continue;
        }
        let probs = &context.mv[component];
        sink.write(value < 0, probs.sign);
        let (class, offset) = mv_class(value.unsigned_abs() - 1);
        sink.tree(&MV_CLASS_TREE, &probs.classes, class as u8);
        let integer = offset >> 3;
        if class == 0 {
            sink.write(integer != 0, probs.class0);
        } else {
            for bit in 0..class {
                sink.write((integer >> bit) & 1 != 0, probs.bits[bit]);
            }
        }
        let fraction = ((offset >> 1) & 3) as u8;
        let fp = if class == 0 {
            &probs.class0_fp[integer as usize]
        } else {
            &probs.fp
        };
        sink.tree(&MV_FP_TREE, fp, fraction);
        // Without high-precision vectors the eighth-sample bit is implied,
        // and its probabilities are never adapted.
        if let Some(counts) = sink.counts() {
            let MvComponentCounts {
                sign,
                classes,
                class0,
                bits,
                class0_fp,
                fp: fp_counts,
            } = &mut counts.mv[component];
            sign[usize::from(value < 0)] += 1;
            classes[class] += 1;
            if class == 0 {
                class0[integer as usize] += 1;
                class0_fp[integer as usize][usize::from(fraction)] += 1;
            } else {
                for (bit, bit_counts) in bits.iter_mut().enumerate().take(class) {
                    bit_counts[((integer >> bit) & 1) as usize] += 1;
                }
                fp_counts[usize::from(fraction)] += 1;
            }
        }
    }
}

/// The compressed header: the transform mode and no probability updates.
fn compressed_header(key: bool, larger_transforms: bool) -> Vec<u8> {
    const NO_UPDATE: u8 = 252;
    let mut writer = BoolEncoder::new();
    if larger_transforms {
        // tx_mode = TX_MODE_SELECT: ALLOW_32X32 (3) and the select bit, then
        // no update of the 8x8, 16x16 and 32x32 transform size probabilities.
        writer.literal(3, 2);
        writer.bit(true);
        for _ in 0..2 + 2 * 2 + 2 * 3 {
            writer.write(false, NO_UPDATE);
        }
    } else {
        // tx_mode = ONLY_4X4.
        writer.literal(0, 2);
    }
    // No update of the coefficient probabilities of each transform size.
    let tx_sizes = if larger_transforms { 4 } else { 1 };
    for _ in 0..tx_sizes {
        writer.bit(false);
    }
    for _ in 0..3 {
        writer.write(false, NO_UPDATE); // skip
    }
    if !key {
        let updates = 7 * 3 // inter mode
            + 4 // intra/inter
            + 5 * 2 // single reference
            + 4 * 9 // luma mode
            + 16 * 3 // partition
            + 3 // motion vector joints
            + 2 * (1 + 10 + 1 + 10) // sign, classes, class0, bits
            + 2 * (2 * 3 + 3); // class0_fp, fp
        for _ in 0..updates {
            writer.write(false, NO_UPDATE);
        }
    }
    writer.finish()
}

fn uncompressed_header(
    geometry: &Geometry,
    key: bool,
    error_resilient: bool,
    base_q_idx: u8,
    filter_level: u8,
    full_range: bool,
    compressed_size: usize,
) -> Vec<u8> {
    let mut writer = BitWriter::default();
    writer.literal(2, 2); // frame_marker
    writer.literal(0, 2); // profile 0
    writer.bit(false); // show_existing_frame
    writer.bit(!key); // frame_type
    writer.bit(true); // show_frame
    writer.bit(error_resilient); // error_resilient_mode
    if key {
        writer.literal(0x49_83_42, 24); // frame sync code
        writer.literal(1, 3); // color_space = CS_BT_601
        writer.bit(full_range);
        writer.literal(geometry.width as u32 - 1, 16);
        writer.literal(geometry.height as u32 - 1, 16);
        writer.bit(false); // render_and_frame_size_different
    } else {
        if !error_resilient {
            writer.literal(0, 2); // reset_frame_context
        }
        writer.literal(1, 8); // refresh_frame_flags: slot 0 only
        for _ in 0..3 {
            writer.literal(0, 3); // ref_frame_idx: every reference is slot 0
            writer.bit(false); // ref_frame_sign_bias
        }
        writer.bit(true); // found_ref: the size of LAST_FRAME
        writer.bit(false); // render_and_frame_size_different
        writer.bit(false); // allow_high_precision_mv
        writer.bit(false); // is_filter_switchable
        writer.literal(1, 2); // raw_interpolation_filter = EIGHTTAP (regular)
    }
    if !error_resilient {
        // Adapt the probabilities backwards after every frame and keep them
        // for the next one.
        writer.bit(true); // refresh_frame_context
        writer.bit(false); // frame_parallel_decoding_mode
    }
    writer.literal(0, 2); // frame_context_idx
    writer.literal(u32::from(filter_level), 6); // loop_filter_level
    writer.literal(0, 3); // loop_filter_sharpness
    writer.bit(false); // loop_filter_delta_enabled
    writer.literal(u32::from(base_q_idx), 8);
    writer.bit(false); // delta_q_y_dc
    writer.bit(false); // delta_q_uv_dc
    writer.bit(false); // delta_q_uv_ac
    writer.bit(false); // segmentation_enabled
    let sb_cols = geometry.mi_cols.div_ceil(8);
    let mut max_log2 = 1;
    while (sb_cols >> max_log2) >= 4 {
        max_log2 += 1;
    }
    // A single tile column: the minimum is zero for frames up to 4096 wide.
    if max_log2 - 1 > 0 {
        writer.bit(false); // increment_tile_cols_log2
    }
    writer.bit(false); // tile_rows_log2
    writer.literal(compressed_size as u32, 16);
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mv_classes_follow_the_specification_boundaries() {
        assert_eq!(mv_class(0), (0, 0));
        assert_eq!(mv_class(15), (0, 15));
        assert_eq!(mv_class(16), (1, 0));
        assert_eq!(mv_class(31), (1, 15));
        assert_eq!(mv_class(32), (2, 0));
        assert_eq!(mv_class(8191), (9, 8191 - 4096));
        assert_eq!(mv_class(8192), (10, 0));
    }

    #[test]
    fn energy_classes_match_the_token_cache() {
        let expected = [0, 1, 2, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5];
        for (magnitude, &class) in expected.iter().enumerate() {
            assert_eq!(
                energy_class(magnitude as u32),
                class,
                "magnitude {magnitude}"
            );
        }
    }

    #[test]
    fn token_costs_match_costing_every_bit() {
        // `coefficient_bits` sums whole tokens where `write_coefficients`
        // sums bits, so the two agree up to the order of the additions. The
        // default probabilities, and a set skewed as adaptation skews them.
        let mut skewed = FrameContext::default().coef;
        for (index, probs) in skewed
            .as_flattened_mut()
            .as_flattened_mut()
            .as_flattened_mut()
            .as_flattened_mut()
            .iter_mut()
            .enumerate()
        {
            for prob in probs.iter_mut().filter(|prob| **prob != 0) {
                *prob = (u32::from(*prob) * 7 + index as u32 * 13).clamp(1, 255) as u8 % 255 + 1;
            }
        }
        let mut state = 0x2545_f491_u32;
        for coef in [FrameContext::default().coef, skewed] {
            let costs = token_costs(&coef);
            for tx_size in [0, 1, 2, 3] {
                let n = 4 << tx_size;
                for tx_type in [
                    TxType::DctDct,
                    TxType::AdstDct,
                    TxType::DctAdst,
                    TxType::AdstAdst,
                ] {
                    if tx_size == 3 && tx_type != TxType::DctDct {
                        continue;
                    }
                    for (plane_type, reference, context) in
                        [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 0)]
                    {
                        let levels: Vec<i32> = (0..n * n)
                            .map(|index| {
                                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                                // Mostly small levels, some past the costed
                                // magnitudes, thinning out with frequency.
                                let magnitude = match (state >> 16) % 16 {
                                    0..=7 => 0,
                                    8..=12 => ((state >> 8) % 4) as i32,
                                    13 | 14 => ((state >> 8) % 40) as i32,
                                    _ => ((state >> 8) % 3000) as i32,
                                } >> (index / n).min(4);
                                if state & 1 == 0 {
                                    magnitude
                                } else {
                                    -magnitude
                                }
                            })
                            .collect();
                        let (scan, _) = scan_order(tx_size, tx_type);
                        let last = scan
                            .iter()
                            .rposition(|&position| levels[position as usize] != 0)
                            .map_or(0, |last| last + 1);
                        for eob in [last, last / 2, 1] {
                            let mut counter = BitCost::default();
                            write_coefficients(
                                &mut counter,
                                &coef[tx_size],
                                &levels,
                                eob,
                                tx_size,
                                tx_type,
                                plane_type * 2 + reference,
                                context,
                            );
                            let bits = coefficient_bits(
                                &costs[tx_size],
                                &levels,
                                eob,
                                tx_size,
                                tx_type,
                                plane_type,
                                reference,
                                context,
                            );
                            assert!(
                                (bits - counter.0).abs() <= 1e-9 * counter.0.max(1.0),
                                "{n}x{n} {tx_type:?} eob {eob}: {bits} bits against {}",
                                counter.0
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn counted_token_bits_match_the_written_length() {
        // A long run of symbols costs what the boolean coder spends on them,
        // to within its few bytes of flush.
        let mut state = 0x9e37_79b9_u32;
        let mut blocks = Vec::new();
        for tx_size in [0, 1, 2, 3] {
            let n = 4 << tx_size;
            for _ in 0..40 {
                let levels: Vec<i32> = (0..n * n)
                    .map(|index| {
                        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                        let magnitude = ((state >> 16) % 7) as i32 >> (index / n).min(3);
                        if state & 1 == 0 {
                            magnitude
                        } else {
                            -magnitude
                        }
                    })
                    .collect();
                let (scan, _) = scan_order(tx_size, TxType::DctDct);
                let eob = scan
                    .iter()
                    .rposition(|&position| levels[position as usize] != 0)
                    .map_or(0, |last| last + 1);
                blocks.push((tx_size, levels, eob));
            }
        }
        let mut writer = BoolEncoder::new();
        let mut counter = BitCost::default();
        for (tx_size, levels, eob) in &blocks {
            for context in 0..3 {
                let probs = &FrameContext::default().coef[*tx_size];
                write_coefficients(
                    &mut writer,
                    probs,
                    levels,
                    *eob,
                    *tx_size,
                    TxType::DctDct,
                    1,
                    context,
                );
                write_coefficients(
                    &mut counter,
                    probs,
                    levels,
                    *eob,
                    *tx_size,
                    TxType::DctDct,
                    1,
                    context,
                );
            }
        }
        let written = writer.finish().len() as f64 * 8.0;
        assert!(
            (written - counter.0).abs() < written * 0.01 + 64.0,
            "counted {:.0} bits, wrote {written}",
            counter.0
        );
    }
}
