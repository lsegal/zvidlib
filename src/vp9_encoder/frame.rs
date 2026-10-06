//! Encodes one VP9 frame: headers, partitioning, mode decision, motion search
//! and coefficient tokens.
//!
//! The coding tools are a deliberately small, fully specified subset of VP9
//! profile 0:
//!
//! - every frame is error resilient, so it is coded with the default
//!   probabilities, sends no probability updates, and needs no state from
//!   earlier frames other than the reconstructed reference picture;
//! - every superblock is split down to 8x8 blocks and every transform is 4x4
//!   (`tx_mode = ONLY_4X4`);
//! - the loop filter runs at the per-frame level that brings the
//!   reconstruction closest to the source, with sharpness 0 and no mode or
//!   reference deltas;
//! - key frames choose among the DC, V, H and TM intra modes per block;
//! - inter frames predict from the previous frame (`LAST_FRAME`) with motion
//!   vectors found by a whole-sample search and refined to quarter samples,
//!   or to eighth samples where `allow_high_precision_mv` lets the block code
//!   them, through the regular 8-tap filters the header selects. They are
//!   coded as `ZEROMV`, `NEARESTMV`, `NEARMV` or `NEWMV`, or fall back to
//!   intra.
//!
//! The encoder keeps a reconstruction that matches the decoding process
//! exactly: blocks predict from it unfiltered within the frame, and later
//! frames predict from it after the loop filter.

use super::bitwriter::{BitWriter, BoolEncoder};
use super::dsp::{
    IntraMode, ReferencePlane, TxType, forward_transform, inverse_transform_add, predict_inter,
    predict_intra_4x4,
};
use crate::vp9_dec::loopfilter::{self, FilterPlane, LoopFilterMask, MaskBlock};

use super::tables::{
    AC_QLOOKUP, CAT6_PROBS, COEF_PROBS_4X4, COL_SCAN_4X4, COL_SCAN_4X4_NEIGHBORS, DC_QLOOKUP,
    DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X4_NEIGHBORS, IF_UV_MODE_PROBS, IF_Y_MODE_PROBS,
    INTER_MODE_PROBS, INTRA_INTER_PROBS, KF_PARTITION_PROBS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS,
    PARETO8_FULL, PARTITION_PROBS, ROW_SCAN_4X4, ROW_SCAN_4X4_NEIGHBORS, SINGLE_REF_PROBS,
    SKIP_PROBS,
};

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
    class0_hp: u8,
    hp: u8,
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

const COEF_BAND_4X4: [usize; 16] = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5];
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

/// `mv_ref_blocks[BLOCK_8X8]` as `(row, column)` offsets in 8x8 units.
const MV_REF_SEARCH: [(isize, isize); 8] = [
    (-1, 0),
    (0, -1),
    (-1, -1),
    (-2, 0),
    (0, -2),
    (-2, -1),
    (-1, -2),
    (-2, -2),
];

/// `BLOCK_8X8` and `TX_4X4`, the only block and transform sizes coded.
const BLOCK_8X8: u8 = 3;
const TX_4X4: u8 = 0;

const NEARESTMV: u8 = 10;
const NEARMV: u8 = 11;
const ZEROMV: u8 = 12;
const NEWMV: u8 = 13;

/// libvpx enables eighth-sample vectors below this quantizer index, where the
/// residual is fine enough for the extra precision to pay for its bit.
const HIGH_PRECISION_MV_QTHRESH: u8 = 200;
/// `COMPANDED_MVREF_THRESH`: a block codes eighth-sample vectors only when
/// both components of its reference vector are shorter than this many samples.
const COMPANDED_MVREF_THRESH: i32 = 8;

/// Whether a vector coded against `reference` may use eighth samples
/// (`use_mv_hp`).
fn use_mv_hp(reference: Mv) -> bool {
    (reference.row.abs() >> 3) < COMPANDED_MVREF_THRESH
        && (reference.col.abs() >> 3) < COMPANDED_MVREF_THRESH
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
}

/// Everything decided for one 8x8 block.
#[derive(Clone)]
struct BlockChoice {
    info: ModeInfo,
    uv_mode: IntraMode,
    /// For `NEWMV`, the reference the motion vector is coded against.
    best_mv: Mv,
    /// Quantized levels in raster order: four luma blocks, then U and V.
    levels: [[i32; 16]; 6],
    tx_types: [TxType; 6],
    /// The reconstruction: 8x8 luma, then 4x4 U and V.
    luma: [u8; 64],
    chroma: [[u8; 16]; 2],
    cost: f64,
}

/// The best chroma mode for a block and what coding it costs and produces.
struct ChromaChoice {
    cost: f64,
    mode: IntraMode,
    levels: [[i32; 16]; 2],
    pixels: [[u8; 16]; 2],
}

pub(super) struct FrameEncoder<'a> {
    geometry: Geometry,
    source: &'a Picture,
    reference: Option<&'a Picture>,
    recon: Picture,
    base_q_idx: u8,
    /// The loop filter level the frame header signals; zero turns it off.
    filter_level: u8,
    /// Whether [`Self::encode`] chooses a loop filter level at all; tests
    /// clear it to compare against an unfiltered encode.
    loop_filter: bool,
    /// The quantizer steps every plane uses: the header codes no deltas.
    dc_q: i32,
    ac_q: i32,
    lambda: f64,
    /// Whether inter blocks may code eighth-sample motion vectors.
    allow_high_precision_mv: bool,
    /// The finest motion search step in eighth samples: 1 or 2, or 8 to
    /// search whole samples only.
    subpel_step: i32,
    mode_info: Vec<ModeInfo>,
    above_nonzero: [Vec<bool>; 3],
    left_nonzero: [[bool; 16]; 3],
    above_partition: Vec<u8>,
    left_partition: [u8; 8],
    writer: BoolEncoder,
}

impl<'a> FrameEncoder<'a> {
    /// Prepares to encode `source`, as a key frame when `reference` is `None`
    /// and as an inter frame predicted from `reference` otherwise.
    pub(super) fn new(
        geometry: Geometry,
        source: &'a Picture,
        reference: Option<&'a Picture>,
        base_q_idx: u8,
    ) -> Self {
        let q = usize::from(base_q_idx);
        let ac = AC_QLOOKUP[q];
        Self {
            geometry,
            source,
            reference,
            recon: Picture::new(&geometry),
            base_q_idx,
            filter_level: 0,
            loop_filter: true,
            dc_q: DC_QLOOKUP[q],
            ac_q: ac,
            // Distortion is a pixel-domain squared error and rate is in bits.
            // The transform's coefficients are eight times orthonormal, so the
            // effective step is `ac / 8`.
            lambda: f64::from(ac * ac) / 96.0,
            allow_high_precision_mv: reference.is_some() && base_q_idx < HIGH_PRECISION_MV_QTHRESH,
            subpel_step: 1,
            mode_info: vec![ModeInfo::default(); geometry.mi_cols * geometry.mi_rows],
            above_nonzero: [
                vec![false; geometry.mi_cols * 2],
                vec![false; geometry.mi_cols],
                vec![false; geometry.mi_cols],
            ],
            left_nonzero: [[false; 16]; 3],
            above_partition: vec![0; geometry.mi_cols],
            left_partition: [0; 8],
            writer: BoolEncoder::new(),
        }
    }

    /// Restricts the motion search to whole samples, so tests can measure
    /// what sub-sample refinement gains.
    #[cfg(test)]
    pub(super) fn whole_sample_motion(mut self) -> Self {
        self.allow_high_precision_mv = false;
        self.subpel_step = 8;
        self
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

    /// Encodes the frame and returns its bytes with the reconstruction;
    /// `full_range` is the colour range the key frame header signals.
    pub(super) fn encode(mut self, full_range: bool) -> (Vec<u8>, Picture) {
        let sb_rows = self.geometry.mi_rows.div_ceil(8);
        let sb_cols = self.geometry.mi_cols.div_ceil(8);
        for sb_row in 0..sb_rows {
            self.left_nonzero = [[false; 16]; 3];
            self.left_partition = [0; 8];
            for sb_col in 0..sb_cols {
                self.encode_partition(sb_row * 8, sb_col * 8, 3);
            }
        }
        if self.loop_filter {
            self.apply_loop_filter();
        }
        let key = self.is_key();
        let geometry = self.geometry;
        let tile = std::mem::replace(&mut self.writer, BoolEncoder::new()).finish();
        let compressed = compressed_header(key, self.allow_high_precision_mv);
        let mut frame = uncompressed_header(
            &geometry,
            key,
            self.allow_high_precision_mv,
            self.base_q_idx,
            self.filter_level,
            full_range,
            compressed.len(),
        );
        frame.extend_from_slice(&compressed);
        frame.extend_from_slice(&tile);
        (frame, self.recon)
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
            *errors[usize::from(level)]
                .get_or_insert_with(|| self.source_error(&self.filtered(level)))
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
        let Geometry {
            mi_rows, mi_cols, ..
        } = self.geometry;
        let sb_cols = mi_cols.div_ceil(8);
        let mut masks = vec![LoopFilterMask::default(); sb_cols * mi_rows.div_ceil(8)];
        for mi_row in 0..mi_rows {
            for mi_col in 0..mi_cols {
                let info = self.mode_info[mi_row * mi_cols + mi_col];
                loopfilter::build_mask(
                    &mut masks[(mi_row >> 3) * sb_cols + (mi_col >> 3)],
                    &MaskBlock {
                        sb_type: BLOCK_8X8,
                        tx_size: TX_4X4,
                        skip_inter: info.skip && info.is_inter,
                        filter_level: level,
                    },
                    mi_row,
                    mi_col,
                    1,
                    1,
                );
            }
        }
        // The filter works a superblock at a time and may touch samples past
        // the decoded area, so it runs on copies laid out as the decoder's
        // planes are: whole superblocks with an 8-sample border.
        const BORDER: usize = 8;
        let mut padded = [0, 1, 2].map(|plane| {
            let superblock = if plane == 0 { 64 } else { 32 };
            let stride = picture.strides[plane];
            let rows = picture.planes[plane].len() / stride;
            let padded_stride = stride.div_ceil(superblock) * superblock + 2 * BORDER;
            let padded_rows = rows.div_ceil(superblock) * superblock + 2 * BORDER;
            let mut data = vec![0; padded_stride * padded_rows];
            for row in 0..rows {
                let start = (row + BORDER) * padded_stride + BORDER;
                data[start..start + stride]
                    .copy_from_slice(&picture.planes[plane][row * stride..(row + 1) * stride]);
            }
            (data, padded_stride)
        });
        let mut planes = padded.each_mut().map(|(data, stride)| FilterPlane {
            data,
            stride: *stride,
            origin: BORDER * *stride + BORDER,
        });
        loopfilter::filter_frame(&mut planes, &mut masks, mi_rows, mi_cols, 0);
        for (plane, (data, padded_stride)) in padded.iter().enumerate() {
            let stride = picture.strides[plane];
            for (row, output) in picture.planes[plane].chunks_exact_mut(stride).enumerate() {
                let start = (row + BORDER) * padded_stride + BORDER;
                output.copy_from_slice(&data[start..start + stride]);
            }
        }
        picture
    }

    /// The squared error between `picture` and the source over the visible
    /// area of every plane.
    fn source_error(&self, picture: &Picture) -> u64 {
        let geometry = &self.geometry;
        let mut error = 0;
        for (plane, width, height) in [
            (0, geometry.width, geometry.height),
            (1, geometry.chroma_width(), geometry.chroma_height()),
            (2, geometry.chroma_width(), geometry.chroma_height()),
        ] {
            let stride = picture.strides[plane];
            for row in 0..height {
                let range = row * stride..row * stride + width;
                error += picture.planes[plane][range.clone()]
                    .iter()
                    .zip(&self.source.planes[plane][range])
                    .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
                    .sum::<u64>();
            }
        }
        error
    }

    fn encode_partition(&mut self, mi_row: usize, mi_col: usize, bsl: usize) {
        if mi_row >= self.geometry.mi_rows || mi_col >= self.geometry.mi_cols {
            return;
        }
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
            self.writer.tree(&PARTITION_TREE, probs, 0);
            self.encode_block(mi_row, mi_col);
            // `partition_context_lookup[BLOCK_8X8]`.
            self.above_partition[mi_col] = 14;
            self.left_partition[mi_row & 7] = 14;
            return;
        }
        match (has_rows, has_cols) {
            (true, true) => self.writer.tree(&PARTITION_TREE, probs, 3),
            (false, true) => self.writer.write(true, probs[1]),
            (true, false) => self.writer.write(true, probs[2]),
            (false, false) => {}
        }
        self.encode_partition(mi_row, mi_col, bsl - 1);
        self.encode_partition(mi_row, mi_col + half, bsl - 1);
        self.encode_partition(mi_row + half, mi_col, bsl - 1);
        self.encode_partition(mi_row + half, mi_col + half, bsl - 1);
    }

    fn above_info(&self, mi_row: usize, mi_col: usize) -> Option<ModeInfo> {
        (mi_row > 0).then(|| self.mode_info[(mi_row - 1) * self.geometry.mi_cols + mi_col])
    }

    fn left_info(&self, mi_row: usize, mi_col: usize) -> Option<ModeInfo> {
        (mi_col > 0).then(|| self.mode_info[mi_row * self.geometry.mi_cols + mi_col - 1])
    }

    fn encode_block(&mut self, mi_row: usize, mi_col: usize) {
        let mut choice = self.choose_intra(mi_row, mi_col);
        if !self.is_key()
            && let Some(inter) = self.choose_inter(mi_row, mi_col)
            && inter.cost < choice.cost
        {
            choice = inter;
        }
        let skip = choice
            .levels
            .iter()
            .all(|block| block.iter().all(|&level| level == 0));
        choice.info.skip = skip;
        self.store_reconstruction(mi_row, mi_col, &choice);
        self.write_mode_info(mi_row, mi_col, &choice);
        self.mode_info[mi_row * self.geometry.mi_cols + mi_col] = choice.info;
        self.write_tokens(mi_row, mi_col, &choice);
    }

    fn store_reconstruction(&mut self, mi_row: usize, mi_col: usize, choice: &BlockChoice) {
        let stride = self.recon.strides[0];
        for row in 0..8 {
            let start = (mi_row * 8 + row) * stride + mi_col * 8;
            self.recon.planes[0][start..start + 8]
                .copy_from_slice(&choice.luma[row * 8..row * 8 + 8]);
        }
        for plane in 1..3 {
            let stride = self.recon.strides[plane];
            for row in 0..4 {
                let start = (mi_row * 4 + row) * stride + mi_col * 4;
                self.recon.planes[plane][start..start + 4]
                    .copy_from_slice(&choice.chroma[plane - 1][row * 4..row * 4 + 4]);
            }
        }
    }

    /// Quantizes, dequantizes and reconstructs one 4x4 block in place; returns
    /// its levels and squared error.
    fn code_residual(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tx_type: TxType,
        intra: bool,
    ) -> ([i32; 16], u64) {
        let stride = self.recon.strides[plane];
        let mut residual = [0_i32; 16];
        for row in 0..4 {
            for column in 0..4 {
                let index = (y + row) * stride + x + column;
                residual[row * 4 + column] = i32::from(self.source.planes[plane][index])
                    - i32::from(self.recon.planes[plane][index]);
            }
        }
        let coefficients = forward_transform(&residual, tx_type);
        // A smaller rounding offset for inter residuals, as libvpx uses.
        let rounding = if intra { 0.375 } else { 0.25 };
        let mut levels = [0_i32; 16];
        let mut dequantized = [0_i32; 16];
        for (index, &coefficient) in coefficients.iter().enumerate() {
            let step = if index == 0 { self.dc_q } else { self.ac_q };
            let magnitude = f64::from(coefficient.unsigned_abs()) / f64::from(step) + rounding;
            // Keep every dequantized value inside the 16-bit range the decoder
            // stores coefficients in.
            let level = (magnitude.floor() as i32).min(32767 / step);
            levels[index] = if coefficient < 0 { -level } else { level };
            dequantized[index] = levels[index] * step;
        }
        let start = y * stride + x;
        inverse_transform_add(
            &dequantized,
            tx_type,
            &mut self.recon.planes[plane][start..],
            stride,
        );
        let mut error = 0_u64;
        for row in 0..4 {
            for column in 0..4 {
                let index = (y + row) * stride + x + column;
                let difference = i32::from(self.source.planes[plane][index])
                    - i32::from(self.recon.planes[plane][index]);
                error += (difference * difference) as u64;
            }
        }
        (levels, error)
    }

    fn choose_intra(&mut self, mi_row: usize, mi_col: usize) -> BlockChoice {
        let mut best: Option<BlockChoice> = None;
        let key = self.is_key();
        let above_mode = self
            .above_info(mi_row, mi_col)
            .map_or(0, |info| if info.is_inter { 0 } else { info.mode });
        let left_mode = self
            .left_info(mi_row, mi_col)
            .map_or(0, |info| if info.is_inter { 0 } else { info.mode });
        for mode in IntraMode::ALL {
            let mut levels = [[0_i32; 16]; 6];
            let mut error = 0_u64;
            let mut bits = 0.0;
            let tx_type = mode.tx_type();
            for (sub, sub_levels) in levels.iter_mut().take(4).enumerate() {
                let (sub_row, sub_col) = (sub / 2, sub % 2);
                let x = mi_col * 8 + sub_col * 4;
                let y = mi_row * 8 + sub_row * 4;
                let stride = self.recon.strides[0];
                predict_intra_4x4(
                    &mut self.recon.planes[0],
                    stride,
                    x,
                    y,
                    mode,
                    sub_row > 0 || mi_row > 0,
                    sub_col > 0 || mi_col > 0,
                );
                let (block, block_error) = self.code_residual(0, x, y, tx_type, true);
                bits += estimate_token_bits(&block, tx_type);
                *sub_levels = block;
                error += block_error;
            }
            let probs = if key {
                let index = (usize::from(above_mode) * 10 + usize::from(left_mode)) * 9;
                &KF_Y_MODE_PROBS[index..index + 9]
            } else {
                &IF_Y_MODE_PROBS[9..18]
            };
            bits += tree_bits(&INTRA_MODE_TREE, probs, mode as u8);
            let cost = error as f64 + self.lambda * bits;
            if best.as_ref().is_none_or(|best| cost < best.cost) {
                let mut luma = [0_u8; 64];
                let stride = self.recon.strides[0];
                for row in 0..8 {
                    let start = (mi_row * 8 + row) * stride + mi_col * 8;
                    luma[row * 8..row * 8 + 8]
                        .copy_from_slice(&self.recon.planes[0][start..start + 8]);
                }
                best = Some(BlockChoice {
                    info: ModeInfo {
                        is_inter: false,
                        mode: mode as u8,
                        mv: Mv::default(),
                        skip: false,
                    },
                    uv_mode: IntraMode::Dc,
                    best_mv: Mv::default(),
                    levels,
                    tx_types: [
                        tx_type,
                        tx_type,
                        tx_type,
                        tx_type,
                        TxType::DctDct,
                        TxType::DctDct,
                    ],
                    luma,
                    chroma: [[0; 16]; 2],
                    cost,
                });
            }
        }
        let mut best = best.expect("at least one intra mode is evaluated");
        // Later luma blocks predict from this block's chosen reconstruction.
        let stride = self.recon.strides[0];
        for row in 0..8 {
            let start = (mi_row * 8 + row) * stride + mi_col * 8;
            self.recon.planes[0][start..start + 8]
                .copy_from_slice(&best.luma[row * 8..row * 8 + 8]);
        }

        let mut best_uv: Option<ChromaChoice> = None;
        for mode in IntraMode::ALL {
            let mut error = 0_u64;
            let mut bits = 0.0;
            let mut levels = [[0_i32; 16]; 2];
            let mut pixels = [[0_u8; 16]; 2];
            for plane in 1..3 {
                let stride = self.recon.strides[plane];
                let (x, y) = (mi_col * 4, mi_row * 4);
                predict_intra_4x4(
                    &mut self.recon.planes[plane],
                    stride,
                    x,
                    y,
                    mode,
                    mi_row > 0,
                    mi_col > 0,
                );
                let (block, block_error) = self.code_residual(plane, x, y, TxType::DctDct, true);
                bits += estimate_token_bits(&block, TxType::DctDct);
                levels[plane - 1] = block;
                error += block_error;
                for row in 0..4 {
                    let start = (y + row) * stride + x;
                    pixels[plane - 1][row * 4..row * 4 + 4]
                        .copy_from_slice(&self.recon.planes[plane][start..start + 4]);
                }
            }
            let y_mode = usize::from(best.info.mode);
            let probs = if key {
                &KF_UV_MODE_PROBS[y_mode * 9..y_mode * 9 + 9]
            } else {
                &IF_UV_MODE_PROBS[y_mode * 9..y_mode * 9 + 9]
            };
            bits += tree_bits(&INTRA_MODE_TREE, probs, mode as u8);
            let cost = error as f64 + self.lambda * bits;
            if best_uv.as_ref().is_none_or(|best| cost < best.cost) {
                best_uv = Some(ChromaChoice {
                    cost,
                    mode,
                    levels,
                    pixels,
                });
            }
        }
        let ChromaChoice {
            cost: uv_cost,
            mode: uv_mode,
            levels: uv_levels,
            pixels: uv_pixels,
        } = best_uv.expect("at least one chroma mode is evaluated");
        best.uv_mode = uv_mode;
        best.levels[4] = uv_levels[0];
        best.levels[5] = uv_levels[1];
        best.chroma = uv_pixels;
        best.cost += uv_cost;
        if !key {
            let context = intra_inter_context(
                self.above_info(mi_row, mi_col),
                self.left_info(mi_row, mi_col),
            );
            best.cost += self.lambda * bool_bits(false, INTRA_INTER_PROBS[context]);
        }
        best
    }

    /// The `NEARESTMV` and `NEARMV` candidates and the inter mode context, as
    /// `find_mv_refs` derives them for an 8x8 block predicting from `LAST_FRAME`.
    fn mv_references(&self, mi_row: usize, mi_col: usize) -> ([Mv; 2], usize) {
        let mut list = [Mv::default(); 2];
        let mut count = 0;
        let mut counter = 0_usize;
        let mut done = false;
        for (index, &(row_offset, col_offset)) in MV_REF_SEARCH.iter().enumerate() {
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
        let to_left = -((mi_col * 64) as i32);
        let to_right = ((self.geometry.mi_cols - 1 - mi_col) * 64) as i32;
        let to_top = -((mi_row * 64) as i32);
        let to_bottom = ((self.geometry.mi_rows - 1 - mi_row) * 64) as i32;
        for mv in &mut list {
            mv.col = mv.col.clamp(to_left - border, to_right + border);
            mv.row = mv.row.clamp(to_top - border, to_bottom + border);
            // `lower_mv_precision`: a candidate the block cannot code in
            // eighth samples rounds its odd components toward zero.
            if !(self.allow_high_precision_mv && use_mv_hp(*mv)) {
                for component in [&mut mv.row, &mut mv.col] {
                    if *component & 1 != 0 {
                        *component -= component.signum();
                    }
                }
            }
        }
        (list, usize::from(COUNTER_TO_CONTEXT[counter]))
    }

    fn luma_sad(
        &self,
        reference: &Picture,
        mi_row: usize,
        mi_col: usize,
        mv: Mv,
        limit: u32,
    ) -> u32 {
        let (dy, dx) = (mv.row / 8, mv.col / 8);
        let x0 = (mi_col * 8) as isize + dx as isize;
        let y0 = (mi_row * 8) as isize + dy as isize;
        let stride = self.source.strides[0];
        let max_x = self.geometry.width as isize - 1;
        let max_y = self.geometry.height as isize - 1;
        let inside = x0 >= 0 && y0 >= 0 && x0 + 7 <= max_x && y0 + 7 <= max_y;
        let mut sad = 0_u32;
        for row in 0..8 {
            let source_row =
                &self.source.planes[0][(mi_row * 8 + row) * stride + mi_col * 8..][..8];
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

    /// The squared error of the 8x8 luma prediction `mv` builds.
    fn luma_prediction_error(
        &self,
        reference: &Picture,
        mi_row: usize,
        mi_col: usize,
        mv: Mv,
    ) -> u64 {
        let reference_plane = ReferencePlane {
            pixels: &reference.planes[0],
            stride: reference.strides[0],
            width: self.geometry.width,
            height: self.geometry.height,
        };
        let mut prediction = [0_u8; 64];
        let (x, y) = (mi_col * 8, mi_row * 8);
        predict_inter(
            &reference_plane,
            x,
            y,
            8,
            mv.row * 2,
            mv.col * 2,
            &mut prediction,
        );
        let stride = self.source.strides[0];
        let mut error = 0_u64;
        for row in 0..8 {
            let source = &self.source.planes[0][(y + row) * stride + x..][..8];
            for (&source, &predicted) in source.iter().zip(&prediction[row * 8..row * 8 + 8]) {
                let difference = i32::from(source) - i32::from(predicted);
                error += (difference * difference) as u64;
            }
        }
        error
    }

    fn choose_inter(&mut self, mi_row: usize, mi_col: usize) -> Option<BlockChoice> {
        let reference = self.reference?;
        let (candidates, mode_context) = self.mv_references(mi_row, mi_col);
        let [nearest, near] = candidates;

        // Whole-sample search, kept within 64 samples of the block so every
        // vector stays in the cheap motion vector classes. A diamond search
        // alone settles on false minima in detailed texture, so it starts from
        // the best of an exhaustive search of the four samples around zero and
        // the sample around each candidate.
        let range = 64 * 8;
        // The best vector so far and its luma SAD.
        let mut best = (Mv::default(), u32::MAX);
        let consider = |best: &mut (Mv, u32), candidate: Mv| -> bool {
            if candidate.row.abs() > range || candidate.col.abs() > range {
                return false;
            }
            let sad = self.luma_sad(reference, mi_row, mi_col, candidate, best.1);
            if sad < best.1 {
                *best = (candidate, sad);
                true
            } else {
                false
            }
        };
        // Zero and the candidates themselves come first, so that in flat
        // areas, where many vectors tie, the block keeps one that is cheap to
        // code.
        for (center, radius) in [
            (Mv::default(), 0),
            (nearest, 0),
            (near, 0),
            (nearest, 1),
            (near, 1),
            (Mv::default(), 4),
        ] {
            let center = Mv {
                row: center.row / 8 * 8,
                col: center.col / 8 * 8,
            };
            for row in -radius..=radius {
                for col in -radius..=radius {
                    consider(
                        &mut best,
                        Mv {
                            row: center.row + row * 8,
                            col: center.col + col * 8,
                        },
                    );
                }
            }
        }
        for step in [8, 4, 2, 1] {
            let delta = step * 8;
            for _ in 0..16 {
                let center = best.0;
                let mut improved = false;
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
                    improved |= consider(
                        &mut best,
                        Mv {
                            row: center.row + row * delta,
                            col: center.col + col * delta,
                        },
                    );
                }
                if !improved {
                    break;
                }
            }
        }

        // Sub-sample refinement around the whole-sample vector, measured on
        // the 8-tap prediction itself and including the rate of coding it:
        // half, then quarter, then eighth samples.
        let usehp = self.allow_high_precision_mv && use_mv_hp(nearest);
        let bits_for = |mv: Mv| motion_bits(mv, candidates, mode_context, usehp);
        let whole_mv = best.0;
        let mut best_mv = whole_mv;
        if self.subpel_step < 8 {
            let finest = if usehp {
                self.subpel_step
            } else {
                self.subpel_step.max(2)
            };
            // The prediction error overstates the distortion left once the
            // residual is coded, so rate weighs a quarter of what it does in
            // the mode decision.
            let cost = |mv: Mv| {
                bits_for(mv).map(|bits| {
                    self.luma_prediction_error(reference, mi_row, mi_col, mv) as f64
                        + 0.25 * self.lambda * bits
                })
            };
            let mut best_cost = cost(best_mv).unwrap_or(f64::INFINITY);
            let mut step = 4;
            while step >= finest {
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
                        row: center.row + row * step,
                        col: center.col + col * step,
                    };
                    if candidate.row.abs() > range || candidate.col.abs() > range {
                        continue;
                    }
                    if let Some(candidate_cost) = cost(candidate)
                        && candidate_cost < best_cost
                    {
                        best_cost = candidate_cost;
                        best_mv = candidate;
                    }
                }
                step /= 2;
            }
        }

        // Prediction error is only a proxy for what the residual costs to
        // code, so the refined vector has to beat the whole-sample one on the
        // coded rate and distortion. It must not code worse, either: where
        // the motion is whole samples, a fractional vector only fits the
        // reference's quantization noise, and trading distortion for its rate
        // loses quality.
        let (mut choice, whole_error) = self.code_inter(
            reference,
            mi_row,
            mi_col,
            whole_mv,
            candidates,
            mode_context,
            usehp,
        );
        if best_mv != whole_mv {
            let (refined, refined_error) = self.code_inter(
                reference,
                mi_row,
                mi_col,
                best_mv,
                candidates,
                mode_context,
                usehp,
            );
            if refined.cost < choice.cost && refined_error <= whole_error {
                choice = refined;
            }
        }
        Some(choice)
    }

    /// Predicts the block with `mv` and codes its residual; returns the
    /// choice and its squared error.
    #[allow(clippy::too_many_arguments)]
    fn code_inter(
        &mut self,
        reference: &Picture,
        mi_row: usize,
        mi_col: usize,
        mv: Mv,
        candidates: [Mv; 2],
        mode_context: usize,
        usehp: bool,
    ) -> (BlockChoice, u64) {
        let [nearest, near] = candidates;
        let mode = inter_mode(mv, nearest, near);
        let mut bits = motion_bits(mv, candidates, mode_context, usehp)
            .expect("the chosen motion vector is codable");
        let above = self.above_info(mi_row, mi_col);
        let left = self.left_info(mi_row, mi_col);
        bits += bool_bits(true, INTRA_INTER_PROBS[intra_inter_context(above, left)]);
        bits += bool_bits(false, SINGLE_REF_PROBS[single_ref_context(above, left) * 2]);

        // Prediction, then the residual of each 4x4 block.
        let geometry = self.geometry;
        let mut levels = [[0_i32; 16]; 6];
        let mut error = 0_u64;
        for plane in 0..3 {
            let (size, width, height, scale) = if plane == 0 {
                (8, geometry.width, geometry.height, 2)
            } else {
                (4, geometry.chroma_width(), geometry.chroma_height(), 1)
            };
            let reference_plane = ReferencePlane {
                pixels: &reference.planes[plane],
                stride: reference.strides[plane],
                width,
                height,
            };
            let mut prediction = [0_u8; 64];
            let (x, y) = (mi_col * size, mi_row * size);
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
                self.recon.planes[plane][start..start + size]
                    .copy_from_slice(&prediction[row * size..row * size + size]);
            }
            let blocks = if plane == 0 { 4 } else { 1 };
            for sub in 0..blocks {
                let (sub_x, sub_y) = (x + (sub % 2) * 4, y + (sub / 2) * 4);
                let (block, block_error) =
                    self.code_residual(plane, sub_x, sub_y, TxType::DctDct, false);
                bits += estimate_token_bits(&block, TxType::DctDct);
                levels[if plane == 0 { sub } else { 3 + plane }] = block;
                error += block_error;
            }
        }
        let mut luma = [0_u8; 64];
        let mut chroma = [[0_u8; 16]; 2];
        for row in 0..8 {
            let start = (mi_row * 8 + row) * self.recon.strides[0] + mi_col * 8;
            luma[row * 8..row * 8 + 8].copy_from_slice(&self.recon.planes[0][start..start + 8]);
        }
        for plane in 1..3 {
            for row in 0..4 {
                let start = (mi_row * 4 + row) * self.recon.strides[plane] + mi_col * 4;
                chroma[plane - 1][row * 4..row * 4 + 4]
                    .copy_from_slice(&self.recon.planes[plane][start..start + 4]);
            }
        }
        let choice = BlockChoice {
            info: ModeInfo {
                is_inter: true,
                mode,
                mv,
                skip: false,
            },
            uv_mode: IntraMode::Dc,
            best_mv: nearest,
            levels,
            tx_types: [TxType::DctDct; 6],
            luma,
            chroma,
            cost: error as f64 + self.lambda * bits,
        };
        (choice, error)
    }

    fn write_mode_info(&mut self, mi_row: usize, mi_col: usize, choice: &BlockChoice) {
        let above = self.above_info(mi_row, mi_col);
        let left = self.left_info(mi_row, mi_col);
        let skip_context = usize::from(above.is_some_and(|info| info.skip))
            + usize::from(left.is_some_and(|info| info.skip));
        self.writer
            .write(choice.info.skip, SKIP_PROBS[skip_context]);
        let y_mode = choice.info.mode;
        if self.is_key() {
            let above_mode = above.map_or(0, |info| usize::from(info.mode));
            let left_mode = left.map_or(0, |info| usize::from(info.mode));
            let index = (above_mode * 10 + left_mode) * 9;
            self.writer
                .tree(&INTRA_MODE_TREE, &KF_Y_MODE_PROBS[index..index + 9], y_mode);
            let index = usize::from(y_mode) * 9;
            self.writer.tree(
                &INTRA_MODE_TREE,
                &KF_UV_MODE_PROBS[index..index + 9],
                choice.uv_mode as u8,
            );
            return;
        }
        self.writer.write(
            choice.info.is_inter,
            INTRA_INTER_PROBS[intra_inter_context(above, left)],
        );
        if !choice.info.is_inter {
            // `size_group_lookup[BLOCK_8X8]` is 1.
            self.writer
                .tree(&INTRA_MODE_TREE, &IF_Y_MODE_PROBS[9..18], y_mode);
            let index = usize::from(y_mode) * 9;
            self.writer.tree(
                &INTRA_MODE_TREE,
                &IF_UV_MODE_PROBS[index..index + 9],
                choice.uv_mode as u8,
            );
            return;
        }
        // A single LAST_FRAME reference: the first single_ref bit is zero.
        self.writer
            .write(false, SINGLE_REF_PROBS[single_ref_context(above, left) * 2]);
        let (_, mode_context) = self.mv_references(mi_row, mi_col);
        self.writer.tree(
            &INTER_MODE_TREE,
            &INTER_MODE_PROBS[mode_context * 3..mode_context * 3 + 3],
            y_mode - NEARESTMV,
        );
        if y_mode == NEWMV {
            let difference = Mv {
                row: choice.info.mv.row - choice.best_mv.row,
                col: choice.info.mv.col - choice.best_mv.col,
            };
            let usehp = self.allow_high_precision_mv && use_mv_hp(choice.best_mv);
            write_mv(&mut self.writer, difference, usehp);
        }
    }

    fn write_tokens(&mut self, mi_row: usize, mi_col: usize, choice: &BlockChoice) {
        let is_inter = choice.info.is_inter;
        for (index, levels) in choice.levels.iter().enumerate() {
            let (plane, x4, y4) = if index < 4 {
                (0, mi_col * 2 + index % 2, (mi_row & 7) * 2 + index / 2)
            } else {
                (index - 3, mi_col, mi_row & 7)
            };
            if choice.info.skip {
                self.above_nonzero[plane][x4] = false;
                self.left_nonzero[plane][y4] = false;
                continue;
            }
            let context = usize::from(self.above_nonzero[plane][x4])
                + usize::from(self.left_nonzero[plane][y4]);
            let nonzero = write_coefficients(
                &mut self.writer,
                levels,
                choice.tx_types[index],
                usize::from(plane > 0),
                usize::from(is_inter),
                context,
            );
            self.above_nonzero[plane][x4] = nonzero;
            self.left_nonzero[plane][y4] = nonzero;
        }
    }
}

fn intra_inter_context(above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
    match (above, left) {
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
fn single_ref_context(above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
    if above.is_some_and(|info| info.is_inter) || left.is_some_and(|info| info.is_inter) {
        4
    } else {
        2
    }
}

fn scan_for(tx_type: TxType) -> (&'static [usize; 16], &'static [usize; 34]) {
    match tx_type {
        TxType::DctDct | TxType::AdstAdst => (&DEFAULT_SCAN_4X4, &DEFAULT_SCAN_4X4_NEIGHBORS),
        TxType::AdstDct => (&ROW_SCAN_4X4, &ROW_SCAN_4X4_NEIGHBORS),
        TxType::DctAdst => (&COL_SCAN_4X4, &COL_SCAN_4X4_NEIGHBORS),
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

/// Writes one 4x4 block's tokens as `decode_coefs` reads them; returns
/// whether any coefficient was nonzero.
fn write_coefficients(
    writer: &mut BoolEncoder,
    levels: &[i32; 16],
    tx_type: TxType,
    plane_type: usize,
    reference: usize,
    mut context: usize,
) -> bool {
    let (scan, neighbors) = scan_for(tx_type);
    let end = scan
        .iter()
        .rposition(|&position| levels[position] != 0)
        .map_or(0, |last| last + 1);
    let probs_for = |band: usize, context: usize| -> [u8; 3] {
        let index = ((((plane_type * 2 + reference) * 6 + band) * 6) + context) * 3;
        [
            COEF_PROBS_4X4[index],
            COEF_PROBS_4X4[index + 1],
            COEF_PROBS_4X4[index + 2],
        ]
    };
    let mut cache = [0_u8; 16];
    let mut previous_zero = false;
    for c in 0..end {
        let probs = probs_for(COEF_BAND_4X4[c], context);
        if !previous_zero {
            writer.write(true, probs[0]);
        }
        let level = levels[scan[c]];
        let magnitude = level.unsigned_abs();
        if magnitude == 0 {
            writer.write(false, probs[1]);
            previous_zero = true;
        } else {
            writer.write(true, probs[1]);
            write_magnitude(writer, magnitude, probs[2]);
            writer.write(level < 0, 128);
            previous_zero = false;
        }
        cache[scan[c]] = energy_class(magnitude);
        let next = c + 1;
        context = (1
            + usize::from(cache[neighbors[next * 2]])
            + usize::from(cache[neighbors[next * 2 + 1]]))
            >> 1;
    }
    if end < 16 {
        let probs = probs_for(COEF_BAND_4X4[end], context);
        writer.write(false, probs[0]);
    }
    end > 0
}

fn write_magnitude(writer: &mut BoolEncoder, magnitude: u32, pivot: u8) {
    if magnitude == 1 {
        writer.write(false, pivot);
        return;
    }
    writer.write(true, pivot);
    let pareto = &PARETO8_FULL[(usize::from(pivot) - 1) * 8..][..8];
    let extra = |writer: &mut BoolEncoder, value: u32, probs: &[u8]| {
        for (bit, &probability) in probs.iter().enumerate() {
            writer.write((value >> (probs.len() - 1 - bit)) & 1 != 0, probability);
        }
    };
    if magnitude <= 4 {
        writer.write(false, pareto[0]);
        if magnitude == 2 {
            writer.write(false, pareto[1]);
        } else {
            writer.write(true, pareto[1]);
            writer.write(magnitude == 4, pareto[2]);
        }
        return;
    }
    writer.write(true, pareto[0]);
    if magnitude <= 10 {
        writer.write(false, pareto[3]);
        if magnitude <= 6 {
            writer.write(false, pareto[4]);
            extra(writer, magnitude - 5, CAT_PROBS[0]);
        } else {
            writer.write(true, pareto[4]);
            extra(writer, magnitude - 7, CAT_PROBS[1]);
        }
        return;
    }
    writer.write(true, pareto[3]);
    if magnitude <= 34 {
        writer.write(false, pareto[5]);
        if magnitude <= 18 {
            writer.write(false, pareto[6]);
            extra(writer, magnitude - 11, CAT_PROBS[2]);
        } else {
            writer.write(true, pareto[6]);
            extra(writer, magnitude - 19, CAT_PROBS[3]);
        }
    } else {
        writer.write(true, pareto[5]);
        if magnitude <= 66 {
            writer.write(false, pareto[7]);
            extra(writer, magnitude - 35, CAT_PROBS[4]);
        } else {
            writer.write(true, pareto[7]);
            extra(writer, magnitude - 67, &CAT6_PROBS);
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

/// The cost in bits of coding `mv` as an inter mode and, for `NEWMV`, a
/// difference from the nearest candidate, or `None` when the difference has an
/// eighth-sample component the block cannot code.
fn motion_bits(mv: Mv, [nearest, near]: [Mv; 2], mode_context: usize, usehp: bool) -> Option<f64> {
    let mode = inter_mode(mv, nearest, near);
    let mut bits = tree_bits(
        &INTER_MODE_TREE,
        &INTER_MODE_PROBS[mode_context * 3..mode_context * 3 + 3],
        mode - NEARESTMV,
    );
    if mode == NEWMV {
        let difference = Mv {
            row: mv.row - nearest.row,
            col: mv.col - nearest.col,
        };
        if !usehp && (difference.row | difference.col) & 1 != 0 {
            return None;
        }
        bits += mv_bits(difference, usehp);
    }
    Some(bits)
}

/// The inter mode that codes `mv` given the block's reference candidates.
fn inter_mode(mv: Mv, nearest: Mv, near: Mv) -> u8 {
    if mv == Mv::default() {
        ZEROMV
    } else if mv == nearest {
        NEARESTMV
    } else if mv == near {
        NEARMV
    } else {
        NEWMV
    }
}

/// Writes a motion vector difference as `read_mv` reads it; `usehp` is whether
/// the block codes eighth samples, and otherwise every component is even.
fn write_mv(writer: &mut BoolEncoder, difference: Mv, usehp: bool) {
    let joint = usize::from(difference.row != 0) * 2 + usize::from(difference.col != 0);
    writer.tree(&MV_JOINT_TREE, &MV_JOINT_PROBS, joint as u8);
    for (component, value) in [(0, difference.row), (1, difference.col)] {
        if value == 0 {
            continue;
        }
        let probs = &MV_COMPONENT_PROBS[component];
        writer.write(value < 0, probs.sign);
        let (class, offset) = mv_class(value.unsigned_abs() - 1);
        writer.tree(&MV_CLASS_TREE, &probs.classes, class as u8);
        let integer = offset >> 3;
        if class == 0 {
            writer.write(integer != 0, probs.class0);
        } else {
            for bit in 0..class {
                writer.write((integer >> bit) & 1 != 0, probs.bits[bit]);
            }
        }
        let fraction = ((offset >> 1) & 3) as u8;
        let fp = if class == 0 {
            &probs.class0_fp[integer as usize]
        } else {
            &probs.fp
        };
        writer.tree(&MV_FP_TREE, fp, fraction);
        // Without high-precision vectors the eighth-sample bit is implied.
        if usehp {
            let hp = if class == 0 {
                probs.class0_hp
            } else {
                probs.hp
            };
            writer.write(offset & 1 != 0, hp);
        }
    }
}

fn bool_bits(bit: bool, probability: u8) -> f64 {
    let p = f64::from(probability) / 256.0;
    -(if bit { 1.0 - p } else { p }).log2()
}

fn tree_bits(tree: &[i8], probs: &[u8], value: u8) -> f64 {
    fn walk(tree: &[i8], probs: &[u8], node: usize, value: i16) -> Option<f64> {
        for bit in [false, true] {
            let entry = tree[node + usize::from(bit)];
            let cost = bool_bits(bit, probs[node >> 1]);
            if entry <= 0 {
                if -i16::from(entry) == value {
                    return Some(cost);
                }
            } else if let Some(rest) = walk(tree, probs, entry as usize, value) {
                return Some(cost + rest);
            }
        }
        None
    }
    walk(tree, probs, 0, i16::from(value)).unwrap_or(0.0)
}

/// The exact cost in bits of what [`write_mv`] writes.
fn mv_bits(difference: Mv, usehp: bool) -> f64 {
    let joint = usize::from(difference.row != 0) * 2 + usize::from(difference.col != 0);
    let mut bits = tree_bits(&MV_JOINT_TREE, &MV_JOINT_PROBS, joint as u8);
    for (component, value) in [(0, difference.row), (1, difference.col)] {
        if value == 0 {
            continue;
        }
        let probs = &MV_COMPONENT_PROBS[component];
        bits += bool_bits(value < 0, probs.sign);
        let (class, offset) = mv_class(value.unsigned_abs() - 1);
        bits += tree_bits(&MV_CLASS_TREE, &probs.classes, class as u8);
        let integer = offset >> 3;
        if class == 0 {
            bits += bool_bits(integer != 0, probs.class0);
        } else {
            for bit in 0..class {
                bits += bool_bits((integer >> bit) & 1 != 0, probs.bits[bit]);
            }
        }
        let fp = if class == 0 {
            &probs.class0_fp[integer as usize]
        } else {
            &probs.fp
        };
        bits += tree_bits(&MV_FP_TREE, fp, ((offset >> 1) & 3) as u8);
        if usehp {
            let hp = if class == 0 {
                probs.class0_hp
            } else {
                probs.hp
            };
            bits += bool_bits(offset & 1 != 0, hp);
        }
    }
    bits
}

/// A rough rate estimate for mode decision; the exact cost depends on
/// contexts that are not known until the block is written.
fn estimate_token_bits(levels: &[i32; 16], tx_type: TxType) -> f64 {
    let (scan, _) = scan_for(tx_type);
    let Some(last) = scan.iter().rposition(|&position| levels[position] != 0) else {
        return 0.5;
    };
    let mut bits = 1.0;
    for &position in &scan[..=last] {
        let magnitude = levels[position].unsigned_abs();
        bits += if magnitude == 0 {
            1.0
        } else {
            2.5 + 2.0 * f64::from(magnitude).log2()
        };
    }
    bits
}

/// The compressed header: ONLY_4X4 transforms and no probability updates.
fn compressed_header(key: bool, allow_high_precision_mv: bool) -> Vec<u8> {
    const NO_UPDATE: u8 = 252;
    let mut writer = BoolEncoder::new();
    // tx_mode = ONLY_4X4, then no update of the 4x4 coefficient probabilities.
    writer.literal(0, 2);
    writer.bit(false);
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
            + 2 * (2 * 3 + 3) // class0_fp, fp
            + if allow_high_precision_mv { 2 * 2 } else { 0 }; // class0_hp, hp
        for _ in 0..updates {
            writer.write(false, NO_UPDATE);
        }
    }
    writer.finish()
}

fn uncompressed_header(
    geometry: &Geometry,
    key: bool,
    allow_high_precision_mv: bool,
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
        writer.bit(allow_high_precision_mv);
        writer.bit(false); // is_filter_switchable
        writer.literal(1, 2); // raw_interpolation_filter = EIGHTTAP (regular)
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
}
