//! Encodes one VP9 frame: headers, partitioning, mode decision, motion search
//! and coefficient tokens.
//!
//! The coding tools are a deliberately small, fully specified subset of VP9
//! profile 0:
//!
//! - every frame is error resilient, so it is coded with the default
//!   probabilities, sends no probability updates, and needs no state from
//!   earlier frames other than the reconstructed reference picture;
//! - each 64x64 superblock is either coded whole or split into four, down to
//!   8x8 blocks, and each block picks its transform size from 4x4 up to the
//!   largest that fits (`tx_mode = TX_MODE_SELECT`), both by rate and
//!   distortion; the loop filter is off (`filter_level = 0`);
//! - key frames choose among the DC, V, H and TM intra modes per block;
//! - inter frames predict from the previous frame (`LAST_FRAME`) with
//!   whole-sample motion vectors found by a diamond search, coded as
//!   `ZEROMV`, `NEARESTMV`, `NEARMV` or `NEWMV`, or fall back to intra.
//!
//! Each superblock is searched first, with every candidate's rate counted
//! from the same probabilities and contexts the bitstream codes it with, and
//! then written. The encoder keeps a reconstruction that matches the decoding
//! process exactly, and predicts every later block and frame from it.

use super::bitwriter::{BitCost, BitWriter, BoolEncoder, BoolSink, bit_cost};
use super::dsp::{
    IntraMode, ReferencePlane, TxType, forward_transform, inverse_transform_add, predict_inter,
    predict_intra,
};
use super::tables::{
    AC_QLOOKUP, CAT6_PROBS, DC_QLOOKUP, IF_UV_MODE_PROBS, IF_Y_MODE_PROBS, INTER_MODE_PROBS,
    INTRA_INTER_PROBS, KF_PARTITION_PROBS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS, PARETO8_FULL,
    PARTITION_PROBS, SINGLE_REF_PROBS, SKIP_PROBS, TX_PROBS_8X8, TX_PROBS_16X16, TX_PROBS_32X32,
};
use crate::vp9_dec::tables as shared;

const INTRA_MODE_TREE: [i8; 18] = [
    0, 2, -9, 4, -1, 6, 8, 12, -2, 10, -4, -5, -3, 14, -8, 16, -6, -7,
];
const PARTITION_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
/// Leaves are `mode - NEARESTMV`: NEARESTMV 0, NEARMV 1, ZEROMV 2, NEWMV 3.
const INTER_MODE_TREE: [i8; 6] = [-2, 2, 0, 4, -1, -3];
const MV_JOINT_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
const MV_CLASS_TREE: [i8; 20] = [
    0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10,
];
const MV_FP_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];

const MV_JOINT_PROBS: [u8; 3] = [32, 64, 96];

/// The default probabilities of one motion vector component.
struct MvComponentProbs {
    sign: u8,
    classes: [u8; 10],
    class0: u8,
    bits: [u8; 10],
    class0_fp: [[u8; 3]; 2],
    fp: [u8; 3],
}

/// Row (vertical) then column (horizontal) component defaults.
const MV_COMPONENT_PROBS: [MvComponentProbs; 2] = [
    MvComponentProbs {
        sign: 128,
        classes: [224, 144, 192, 168, 192, 176, 192, 198, 198, 245],
        class0: 216,
        bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
        class0_fp: [[128, 128, 64], [96, 112, 64]],
        fp: [64, 96, 64],
    },
    MvComponentProbs {
        sign: 128,
        classes: [216, 128, 176, 160, 176, 176, 192, 198, 198, 208],
        class0: 208,
        bits: [136, 140, 148, 160, 176, 192, 224, 234, 234, 240],
        class0_fp: [[128, 128, 64], [96, 112, 64]],
        fp: [64, 96, 64],
    },
];

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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Mv {
    row: i32,
    col: i32,
}

/// The mode information later blocks read as context.
#[derive(Clone, Copy, Default)]
struct ModeInfo {
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

/// The result of coding one plane of a block.
struct PlaneCoding {
    blocks: Vec<TxBlock>,
    error: u64,
    bits: f64,
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

pub(super) struct FrameEncoder<'a> {
    geometry: Geometry,
    source: &'a Picture,
    reference: Option<&'a Picture>,
    recon: Picture,
    tools: CodingTools,
    base_q_idx: u8,
    /// The quantizer steps every plane uses: the header codes no deltas.
    dc_q: i32,
    ac_q: i32,
    lambda: f64,
    mode_info: Vec<ModeInfo>,
    above_nonzero: [Vec<bool>; 3],
    left_nonzero: [[bool; 16]; 3],
    above_partition: Vec<u8>,
    left_partition: [u8; 8],
}

impl<'a> FrameEncoder<'a> {
    /// Prepares to encode `source`, as a key frame when `reference` is `None`
    /// and as an inter frame predicted from `reference` otherwise.
    pub(super) fn new(
        geometry: Geometry,
        source: &'a Picture,
        reference: Option<&'a Picture>,
        base_q_idx: u8,
        tools: CodingTools,
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
            dc_q: DC_QLOOKUP[q],
            ac_q: ac,
            // Distortion is a pixel-domain squared error and rate is in bits.
            // The transform's coefficients are eight times orthonormal, so the
            // effective step is `ac / 8`.
            lambda: f64::from(ac * ac) / 96.0,
            mode_info: vec![ModeInfo::default(); geometry.mi_cols * geometry.mi_rows],
            above_nonzero: [
                vec![false; geometry.mi_cols * 2],
                vec![false; geometry.mi_cols],
                vec![false; geometry.mi_cols],
            ],
            left_nonzero: [[false; 16]; 3],
            above_partition: vec![0; geometry.mi_cols],
            left_partition: [0; 8],
        }
    }

    fn is_key(&self) -> bool {
        self.reference.is_none()
    }

    /// Encodes the frame and returns its bytes with the reconstruction;
    /// `full_range` is the colour range the key frame header signals.
    pub(super) fn encode(mut self, full_range: bool) -> (Vec<u8>, Picture) {
        let mut writer = BoolEncoder::new();
        let sb_rows = self.geometry.mi_rows.div_ceil(8);
        let sb_cols = self.geometry.mi_cols.div_ceil(8);
        for sb_row in 0..sb_rows {
            self.left_nonzero = [[false; 16]; 3];
            self.left_partition = [0; 8];
            for sb_col in 0..sb_cols {
                let (mi_row, mi_col) = (sb_row * 8, sb_col * 8);
                // The search leaves the reconstruction and mode information
                // of the chosen partition in place, and the contexts are
                // replayed as the superblock is written.
                let contexts = self.save_contexts(mi_col, 3);
                let (node, _) = self.search_partition(mi_row, mi_col, 3);
                self.restore_contexts(mi_col, 3, &contexts);
                self.write_partition(&mut writer, &node, mi_row, mi_col, 3);
            }
        }
        let key = self.is_key();
        let geometry = self.geometry;
        let tile = writer.finish();
        let compressed = compressed_header(key, self.tools.larger_transforms);
        let mut frame = uncompressed_header(
            &geometry,
            key,
            self.base_q_idx,
            full_range,
            compressed.len(),
        );
        frame.extend_from_slice(&compressed);
        frame.extend_from_slice(&tile);
        (frame, self.recon)
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
    fn search_partition(&mut self, mi_row: usize, mi_col: usize, bsl: usize) -> (Node, f64) {
        if mi_row >= self.geometry.mi_rows || mi_col >= self.geometry.mi_cols {
            return (Node::Outside, 0.0);
        }
        let whole = bsl <= self.tools.largest_block && self.fits(mi_row, mi_col, bsl);
        let mut best = None;
        let before = (whole && bsl > 0).then(|| self.save_block(mi_row, mi_col, bsl));
        if whole {
            let bits = cost(|sink| self.partition_symbol(sink, mi_row, mi_col, bsl, false));
            let choice = self.choose_block(mi_row, mi_col, bsl);
            let total = choice.cost + self.lambda * bits;
            self.commit(mi_row, mi_col, bsl, &choice);
            best = Some((Node::Whole(Box::new(choice)), total));
        }
        if bsl == 0 {
            return best.expect("an 8x8 block inside the frame is coded whole");
        }
        let after_whole = best.is_some().then(|| self.save_block(mi_row, mi_col, bsl));
        if let Some(before) = &before {
            self.restore_block(mi_row, mi_col, bsl, before);
        }
        let mut total =
            self.lambda * cost(|sink| self.partition_symbol(sink, mi_row, mi_col, bsl, true));
        let half = 1 << (bsl - 1);
        let quadrants = [
            (mi_row, mi_col),
            (mi_row, mi_col + half),
            (mi_row + half, mi_col),
            (mi_row + half, mi_col + half),
        ]
        .map(|(row, col)| {
            let (node, cost) = self.search_partition(row, col, bsl - 1);
            total += cost;
            node
        });
        match best {
            Some((node, whole_cost)) if whole_cost <= total => {
                let after = after_whole.expect("saved after coding the block whole");
                self.restore_block(mi_row, mi_col, bsl, &after);
                (node, whole_cost)
            }
            _ => (Node::Split(Box::new(quadrants)), total),
        }
    }

    /// Writes a searched partition, replaying its effect on the contexts.
    fn write_partition(
        &mut self,
        writer: &mut BoolEncoder,
        node: &Node,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
    ) {
        match node {
            Node::Outside => {}
            Node::Whole(choice) => {
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

    fn load_pixels(&mut self, plane: usize, mi_row: usize, mi_col: usize, bsl: usize, pixels: &[u8]) {
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

    fn restore_block(&mut self, mi_row: usize, mi_col: usize, bsl: usize, snapshot: &BlockSnapshot) {
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

    fn choose_block(&mut self, mi_row: usize, mi_col: usize, bsl: usize) -> BlockChoice {
        let neighbors = self.neighbors(mi_row, mi_col);
        let mut choice = self.choose_intra(mi_row, mi_col, bsl, neighbors);
        if !self.is_key()
            && let Some(inter) = self.choose_inter(mi_row, mi_col, bsl, neighbors)
            && inter.cost < choice.cost
        {
            choice = inter;
        }
        choice
    }

    /// Codes one plane of a block with `tx_size` transforms (the plane's own
    /// size), predicting each transform block first with `intra` when given;
    /// an inter prediction is already in place.
    fn code_plane(
        &mut self,
        plane: usize,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        tx_size: usize,
        intra: Option<IntraMode>,
    ) -> PlaneCoding {
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
                    self.code_residual(plane, x, y, tx_size, tx_type, intra.is_none(), context);
                above[col..col + step].fill(block.eob > 0);
                left[row..row + step].fill(block.eob > 0);
                coding.blocks.push(block);
                coding.error += error;
                coding.bits += bits;
            }
        }
        coding
    }

    /// Quantizes, dequantizes and reconstructs one transform block in place;
    /// returns its levels, squared error and token bits. A block whose
    /// levels cost more than the distortion they remove is coded empty.
    #[allow(clippy::too_many_arguments)]
    fn code_residual(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tx_size: usize,
        tx_type: TxType,
        inter: bool,
        context: usize,
    ) -> (TxBlock, u64, f64) {
        let n = 4 << tx_size;
        let stride = self.recon.strides[plane];
        let mut residual = vec![0_i32; n * n];
        let mut prediction = vec![0_u8; n * n];
        let mut prediction_error = 0_u64;
        for row in 0..n {
            let start = (y + row) * stride + x;
            let source = &self.source.planes[plane][start..start + n];
            let predicted = &self.recon.planes[plane][start..start + n];
            prediction[row * n..row * n + n].copy_from_slice(predicted);
            for column in 0..n {
                let difference = i32::from(source[column]) - i32::from(predicted[column]);
                residual[row * n + column] = difference;
                prediction_error += (difference * difference) as u64;
            }
        }
        let plane_type = usize::from(plane > 0);
        let reference = usize::from(inter);
        let empty_bits = bit_cost(
            false,
            coefficient_probs(tx_size, plane_type, reference, 0, context)[0],
        );
        let empty = TxBlock {
            levels: vec![0; n * n],
            eob: 0,
            tx_type,
        };
        if prediction_error == 0 {
            return (empty, 0, empty_bits);
        }

        let mut coefficients = vec![0.0; n * n];
        forward_transform(&residual, tx_size, tx_type, &mut coefficients);
        // A smaller rounding offset for inter residuals, as libvpx uses.
        let rounding = if inter { 0.25 } else { 0.375 };
        let mut levels = vec![0_i32; n * n];
        let mut dequantized = vec![0_i32; n * n];
        for (index, &coefficient) in coefficients.iter().enumerate() {
            let step = if index == 0 { self.dc_q } else { self.ac_q };
            // 32x32 levels dequantize to half the step.
            let (effective, limit) = if tx_size == 3 {
                (f64::from(step) / 2.0, 65535 / step)
            } else {
                (f64::from(step), 32767 / step)
            };
            // Keep every dequantized value inside the 16-bit range the decoder
            // stores coefficients in.
            let level = ((coefficient.abs() / effective + rounding).floor() as i32).min(limit);
            let value = (level * step) >> u32::from(tx_size == 3);
            levels[index] = if coefficient < 0.0 { -level } else { level };
            dequantized[index] = if coefficient < 0.0 { -value } else { value };
        }
        let (scan, _) = scan_order(tx_size, tx_type);
        let eob = scan
            .iter()
            .rposition(|&position| levels[position as usize] != 0)
            .map_or(0, |last| last + 1);
        if eob == 0 {
            return (empty, prediction_error, empty_bits);
        }
        let mut counter = BitCost::default();
        write_coefficients(
            &mut counter,
            &levels,
            eob,
            tx_size,
            tx_type,
            plane_type,
            reference,
            context,
        );
        let start = y * stride + x;
        inverse_transform_add(
            &dequantized,
            tx_size,
            tx_type,
            eob,
            &mut self.recon.planes[plane][start..],
            stride,
        );
        let mut error = 0_u64;
        for row in 0..n {
            let start = (y + row) * stride + x;
            let source = &self.source.planes[plane][start..start + n];
            let reconstructed = &self.recon.planes[plane][start..start + n];
            for (&source, &reconstructed) in source.iter().zip(reconstructed) {
                let difference = i32::from(source) - i32::from(reconstructed);
                error += (difference * difference) as u64;
            }
        }
        if prediction_error as f64 + self.lambda * empty_bits
            <= error as f64 + self.lambda * counter.0
        {
            for row in 0..n {
                let start = (y + row) * stride + x;
                self.recon.planes[plane][start..start + n]
                    .copy_from_slice(&prediction[row * n..row * n + n]);
            }
            return (empty, prediction_error, empty_bits);
        }
        let block = TxBlock {
            levels,
            eob,
            tx_type,
        };
        (block, error, counter.0)
    }

    fn choose_intra(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        neighbors: Neighbors,
    ) -> BlockChoice {
        let mut best: Option<BlockChoice> = None;
        for tx_size in 0..=self.max_tx_size(bsl) {
            let mut best_luma: Option<(f64, IntraMode, PlaneCoding, Vec<u8>)> = None;
            for mode in IntraMode::ALL {
                let coding = self.code_plane(0, mi_row, mi_col, bsl, usize::from(tx_size), Some(mode));
                let bits = coding.bits + cost(|sink| self.y_mode_symbol(sink, neighbors, bsl, mode));
                let total = coding.error as f64 + self.lambda * bits;
                if best_luma.as_ref().is_none_or(|best| total < best.0) {
                    let pixels = self.block_pixels(0, mi_row, mi_col, bsl);
                    best_luma = Some((total, mode, coding, pixels));
                }
            }
            let (_, y_mode, luma, luma_pixels) =
                best_luma.expect("at least one intra mode is evaluated");

            let uv_tx_size = plane_tx_size(tx_size, bsl, 1);
            let mut best_uv: Option<(f64, IntraMode, [PlaneCoding; 2], [Vec<u8>; 2])> = None;
            for mode in IntraMode::ALL {
                let codings = [1, 2].map(|plane| {
                    self.code_plane(plane, mi_row, mi_col, bsl, uv_tx_size, Some(mode))
                });
                let bits = codings.iter().map(|coding| coding.bits).sum::<f64>()
                    + cost(|sink| self.uv_mode_symbol(sink, y_mode, mode));
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
            let info = ModeInfo {
                is_inter: false,
                mode: y_mode as u8,
                mv: Mv::default(),
                skip,
                tx_size,
            };
            let token_bits = if skip { 0.0 } else { luma.bits + u.bits + v.bits };
            let header_bits = cost(|sink| {
                self.skip_symbol(sink, neighbors, skip);
                if !self.is_key() {
                    sink.write(false, INTRA_INTER_PROBS[intra_inter_context(neighbors)]);
                }
                self.tx_size_symbol(sink, neighbors, bsl, tx_size);
                self.y_mode_symbol(sink, neighbors, bsl, y_mode);
                self.uv_mode_symbol(sink, y_mode, uv_mode);
            });
            let error = luma.error + u.error + v.error;
            let total = error as f64 + self.lambda * (header_bits + token_bits);
            if best.as_ref().is_none_or(|best| total < best.cost) {
                best = Some(BlockChoice {
                    info,
                    uv_mode,
                    best_mv: Mv::default(),
                    blocks: [luma.blocks, u.blocks, v.blocks],
                    pixels: [luma_pixels, u_pixels, v_pixels],
                    cost: total,
                });
            }
        }
        best.expect("at least one transform size is evaluated")
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
        // Every inter block predicts from LAST_FRAME, so the search over other
        // reference frames finds nothing. Clamp as `clamp_mv_ref` does.
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
        let mut sad = 0_u32;
        for row in 0..size {
            let source_row =
                &self.source.planes[0][(mi_row * 8 + row) * stride + mi_col * 8..][..size];
            let y = (y0 + row as isize).clamp(0, max_y) as usize;
            for (column, &source) in source_row.iter().enumerate() {
                let x = if inside {
                    (x0 + column as isize) as usize
                } else {
                    (x0 + column as isize).clamp(0, max_x) as usize
                };
                sad += u32::from(source.abs_diff(reference.planes[0][y * stride + x]));
            }
            if sad >= limit {
                return sad;
            }
        }
        sad
    }

    /// Finds the whole-sample motion vector with the smallest luma SAD.
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
        for start in starts {
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

    fn choose_inter(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        neighbors: Neighbors,
    ) -> Option<BlockChoice> {
        let reference = self.reference?;
        let (candidates, mode_context) = self.mv_references(mi_row, mi_col, bsl);
        let [nearest, near] = candidates;
        let size = 8 << bsl;
        let searched = self.search_motion(reference, mi_row, mi_col, size, candidates);
        // Also try the cheapest-to-code vector nearest the searched one.
        let mut vectors = vec![searched];
        if let Some(alternative) = [Mv::default(), nearest, near]
            .into_iter()
            .filter(|&mv| mv != searched)
            .min_by_key(|&mv| self.luma_sad(reference, mi_row, mi_col, size, mv, u32::MAX))
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
                        sink.write(true, INTRA_INTER_PROBS[intra_inter_context(neighbors)]);
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
                    (4 << bsl, geometry.chroma_width(), geometry.chroma_height(), 1)
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
                for row in 0..size {
                    let start = (y + row) * stride + x;
                    let source = &self.source.planes[plane][start..start + size];
                    let predicted = &prediction[row * size..row * size + size];
                    for (&source, &predicted) in source.iter().zip(predicted) {
                        let difference = i32::from(source) - i32::from(predicted);
                        prediction_error += (difference * difference) as u64;
                    }
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

            for tx_size in 0..=max_tx_size {
                if tx_size > 0 {
                    for (plane, pixels) in predicted.iter().enumerate() {
                        self.load_pixels(plane, mi_row, mi_col, bsl, pixels);
                    }
                }
                let uv_tx_size = plane_tx_size(tx_size, bsl, 1);
                let codings = [0, 1, 2].map(|plane| {
                    let plane_tx = if plane == 0 { usize::from(tx_size) } else { uv_tx_size };
                    self.code_plane(plane, mi_row, mi_col, bsl, plane_tx, None)
                });
                if codings
                    .iter()
                    .all(|coding| coding.blocks.iter().all(|block| block.eob == 0))
                {
                    // Coded empty, this is the skipped block already tried.
                    continue;
                }
                let header_bits = cost(|sink| {
                    self.skip_symbol(sink, neighbors, false);
                    self.tx_size_symbol(sink, neighbors, bsl, tx_size);
                }) + mode_bits;
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
        best
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
            &PARTITION_PROBS
        };
        let probs = &table[context * 3..context * 3 + 3];
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
        debug_assert!(split || (has_rows && has_cols), "a partial square must split");
    }

    fn skip_symbol<S: BoolSink>(&self, sink: &mut S, neighbors: Neighbors, skip: bool) {
        let context = usize::from(neighbors.above.is_some_and(|info| info.skip))
            + usize::from(neighbors.left.is_some_and(|info| info.skip));
        sink.write(skip, SKIP_PROBS[context]);
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
            1 => &TX_PROBS_8X8[context..context + 1],
            2 => &TX_PROBS_16X16[context * 2..context * 2 + 2],
            _ => &TX_PROBS_32X32[context * 3..context * 3 + 3],
        };
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
            &IF_Y_MODE_PROBS[group * 9..group * 9 + 9]
        };
        sink.tree(&INTRA_MODE_TREE, probs, mode as u8);
    }

    fn uv_mode_symbol<S: BoolSink>(&self, sink: &mut S, y_mode: IntraMode, mode: IntraMode) {
        let index = (y_mode as usize) * 9;
        let table = if self.is_key() {
            &KF_UV_MODE_PROBS
        } else {
            &IF_UV_MODE_PROBS
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
        sink.write(false, SINGLE_REF_PROBS[single_ref_context(neighbors) * 2]);
        sink.tree(
            &INTER_MODE_TREE,
            &INTER_MODE_PROBS[mode_context * 3..mode_context * 3 + 3],
            mode - NEARESTMV,
        );
        if mode == NEWMV {
            write_mv(
                sink,
                Mv {
                    row: mv.row - nearest.row,
                    col: mv.col - nearest.col,
                },
            );
        }
    }

    fn write_mode_info(
        &self,
        writer: &mut BoolEncoder,
        mi_row: usize,
        mi_col: usize,
        bsl: usize,
        choice: &BlockChoice,
    ) {
        let neighbors = self.neighbors(mi_row, mi_col);
        let info = choice.info;
        self.skip_symbol(writer, neighbors, info.skip);
        if !self.is_key() {
            writer.write(
                info.is_inter,
                INTRA_INTER_PROBS[intra_inter_context(neighbors)],
            );
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
        debug_assert_eq!(candidates[0], choice.best_mv, "NEARESTMV changed since the search");
        debug_assert!(match info.mode {
            NEARESTMV => info.mv == candidates[0],
            NEARMV => info.mv == candidates[1],
            ZEROMV => info.mv == Mv::default(),
            _ => true,
        });
        self.inter_symbols(writer, neighbors, mode_context, info.mode, info.mv, choice.best_mv);
    }

    fn write_tokens(
        &self,
        writer: &mut BoolEncoder,
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
                        &block.levels,
                        block.eob,
                        tx_size,
                        block.tx_type,
                        usize::from(plane > 0),
                        usize::from(choice.info.is_inter),
                        context,
                    );
                    above[col..col + step].fill(block.eob > 0);
                    left[row..row + step].fill(block.eob > 0);
                }
            }
        }
    }
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
        (0, _) => (&shared::DEFAULT_SCAN_4X4, &shared::DEFAULT_SCAN_4X4_NEIGHBORS),
        (1, TxType::AdstDct) => (&shared::ROW_SCAN_8X8, &shared::ROW_SCAN_8X8_NEIGHBORS),
        (1, TxType::DctAdst) => (&shared::COL_SCAN_8X8, &shared::COL_SCAN_8X8_NEIGHBORS),
        (1, _) => (&shared::DEFAULT_SCAN_8X8, &shared::DEFAULT_SCAN_8X8_NEIGHBORS),
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

/// The default coefficient model probabilities of one context.
fn coefficient_probs(
    tx_size: usize,
    plane_type: usize,
    reference: usize,
    band: usize,
    context: usize,
) -> &'static [u8; 3] {
    let table = match tx_size {
        0 => &shared::DEFAULT_COEF_PROBS_4X4,
        1 => &shared::DEFAULT_COEF_PROBS_8X8,
        2 => &shared::DEFAULT_COEF_PROBS_16X16,
        _ => &shared::DEFAULT_COEF_PROBS_32X32,
    };
    &table[plane_type][reference][band][context]
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

/// Writes one transform block's tokens as `decode_coefs` reads them, from
/// `levels` in raster order with `eob` the end of block in scan order.
#[allow(clippy::too_many_arguments)]
fn write_coefficients<S: BoolSink>(
    sink: &mut S,
    levels: &[i32],
    eob: usize,
    tx_size: usize,
    tx_type: TxType,
    plane_type: usize,
    reference: usize,
    mut context: usize,
) {
    let (scan, neighbors) = scan_order(tx_size, tx_type);
    let bands: &[u8] = if tx_size == 0 {
        &shared::COEFBAND_TRANS_4X4
    } else {
        &shared::COEFBAND_TRANS_8X8PLUS
    };
    let mut cache = [0_u8; 1024];
    let mut previous_zero = false;
    for c in 0..eob {
        let probs = coefficient_probs(tx_size, plane_type, reference, usize::from(bands[c]), context);
        if !previous_zero {
            sink.write(true, probs[0]);
        }
        let position = scan[c] as usize;
        let level = levels[position];
        let magnitude = level.unsigned_abs();
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
        let probs =
            coefficient_probs(tx_size, plane_type, reference, usize::from(bands[eob]), context);
        sink.write(false, probs[0]);
    }
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

fn write_mv<S: BoolSink>(sink: &mut S, difference: Mv) {
    let joint = usize::from(difference.row != 0) * 2 + usize::from(difference.col != 0);
    sink.tree(&MV_JOINT_TREE, &MV_JOINT_PROBS, joint as u8);
    for (component, value) in [(0, difference.row), (1, difference.col)] {
        if value == 0 {
            continue;
        }
        let probs = &MV_COMPONENT_PROBS[component];
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
        // Without high-precision vectors the eighth-sample bit is implied.
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
    base_q_idx: u8,
    full_range: bool,
    compressed_size: usize,
) -> Vec<u8> {
    let mut writer = BitWriter::default();
    writer.literal(2, 2); // frame_marker
    writer.literal(0, 2); // profile 0
    writer.bit(false); // show_existing_frame
    writer.bit(!key); // frame_type
    writer.bit(true); // show_frame
    writer.bit(true); // error_resilient_mode
    if key {
        writer.literal(0x49_83_42, 24); // frame sync code
        writer.literal(1, 3); // color_space = CS_BT_601
        writer.bit(full_range);
        writer.literal(geometry.width as u32 - 1, 16);
        writer.literal(geometry.height as u32 - 1, 16);
        writer.bit(false); // render_and_frame_size_different
    } else {
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
    writer.literal(0, 2); // frame_context_idx
    writer.literal(0, 6); // loop_filter_level
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
                        if state & 1 == 0 { magnitude } else { -magnitude }
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
                write_coefficients(&mut writer, levels, *eob, *tx_size, TxType::DctDct, 0, 1, context);
                write_coefficients(&mut counter, levels, *eob, *tx_size, TxType::DctDct, 0, 1, context);
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
