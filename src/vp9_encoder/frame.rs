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
//! - every superblock is split down to 8x8 blocks, every transform is 4x4
//!   (`tx_mode = ONLY_4X4`), and the loop filter is off (`filter_level = 0`);
//! - key frames choose among the DC, V, H and TM intra modes per block;
//! - inter frames predict from the previous frame (`LAST_FRAME`) with
//!   whole-sample motion vectors found by a diamond search, coded as
//!   `ZEROMV`, `NEARESTMV`, `NEARMV` or `NEWMV`, or fall back to intra.
//!
//! The encoder keeps a reconstruction that matches the decoding process
//! exactly, and predicts every later block and frame from it.

use super::bitwriter::{BitWriter, BoolEncoder};
use super::context::{FrameContext, FrameCounts, MvComponentCounts};
use super::dsp::{
    IntraMode, ReferencePlane, TxType, forward_transform, inverse_transform_add, predict_inter,
    predict_intra_4x4,
};
use super::tables::{
    AC_QLOOKUP, CAT6_PROBS, COL_SCAN_4X4, COL_SCAN_4X4_NEIGHBORS, DC_QLOOKUP, DEFAULT_SCAN_4X4,
    DEFAULT_SCAN_4X4_NEIGHBORS, KF_PARTITION_PROBS, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS,
    PARETO8_FULL, ROW_SCAN_4X4, ROW_SCAN_4X4_NEIGHBORS,
};

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

const NEARESTMV: u8 = 10;
const NEARMV: u8 = 11;
const ZEROMV: u8 = 12;
const NEWMV: u8 = 13;

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

/// The mode information later blocks, and the next frame's motion vector
/// candidate search, read as context.
#[derive(Clone, Copy, Default)]
pub(super) struct ModeInfo {
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
    base_q_idx: u8,
    error_resilient: bool,
    /// The probabilities the frame codes with.
    context: &'a FrameContext,
    counts: FrameCounts,
    /// The previous frame's modes, when this frame takes its motion vectors as
    /// candidates (`UsePrevFrameMvs`).
    previous_mode_info: Option<&'a [ModeInfo]>,
    /// The quantizer steps every plane uses: the header codes no deltas.
    dc_q: i32,
    ac_q: i32,
    lambda: f64,
    mode_info: Vec<ModeInfo>,
    above_nonzero: [Vec<bool>; 3],
    left_nonzero: [[bool; 16]; 3],
    above_partition: Vec<u8>,
    left_partition: [u8; 8],
    writer: BoolEncoder,
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
    pub(super) fn new(
        geometry: Geometry,
        source: &'a Picture,
        reference: Option<&'a Picture>,
        base_q_idx: u8,
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
            base_q_idx,
            error_resilient,
            context,
            counts: FrameCounts::default(),
            previous_mode_info: previous_mode_info.filter(|_| reference.is_some()),
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
            writer: BoolEncoder::new(),
        }
    }

    fn is_key(&self) -> bool {
        self.reference.is_none()
    }

    /// Encodes the frame; `full_range` is the colour range the key frame
    /// header signals.
    pub(super) fn encode(mut self, full_range: bool) -> EncodedFrame {
        let sb_rows = self.geometry.mi_rows.div_ceil(8);
        let sb_cols = self.geometry.mi_cols.div_ceil(8);
        for sb_row in 0..sb_rows {
            self.left_nonzero = [[false; 16]; 3];
            self.left_partition = [0; 8];
            for sb_col in 0..sb_cols {
                self.encode_partition(sb_row * 8, sb_col * 8, 3);
            }
        }
        let key = self.is_key();
        let geometry = self.geometry;
        let tile = std::mem::replace(&mut self.writer, BoolEncoder::new()).finish();
        let compressed = compressed_header(key);
        let mut data = uncompressed_header(
            &geometry,
            key,
            self.error_resilient,
            self.base_q_idx,
            full_range,
            compressed.len(),
        );
        data.extend_from_slice(&compressed);
        data.extend_from_slice(&tile);
        EncodedFrame {
            data,
            reconstruction: self.recon,
            counts: self.counts,
            mode_info: self.mode_info,
        }
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
            &self.context.partition
        };
        let probs = &table[context * 3..context * 3 + 3];
        // The decoder counts every partition, including the implied splits
        // at the frame edges: NONE at 8x8 and SPLIT above.
        self.counts.partition[context][if bsl == 0 { 0 } else { 3 }] += 1;
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
                &self.context.y_mode[9..18]
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
                &self.context.uv_mode[y_mode * 9..y_mode * 9 + 9]
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
            best.cost += self.lambda * bool_bits(false, self.context.intra_inter[context]);
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
        let to_left = -((mi_col * 64) as i32);
        let to_right = ((self.geometry.mi_cols - 1 - mi_col) * 64) as i32;
        let to_top = -((mi_row * 64) as i32);
        let to_bottom = ((self.geometry.mi_rows - 1 - mi_row) * 64) as i32;
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

    fn choose_inter(&mut self, mi_row: usize, mi_col: usize) -> Option<BlockChoice> {
        let reference = self.reference?;
        let (candidates, mode_context) = self.mv_references(mi_row, mi_col);
        let [nearest, near] = candidates;

        // Whole-sample search, kept within 64 samples of the block so every
        // vector stays in the cheap motion vector classes.
        let range = 64 * 8;
        let mut best_mv = Mv::default();
        let mut best_sad = self.luma_sad(reference, mi_row, mi_col, best_mv, u32::MAX);
        for start in [nearest, near] {
            let start = Mv {
                row: start.row / 8 * 8,
                col: start.col / 8 * 8,
            };
            if start.row.abs() <= range && start.col.abs() <= range {
                let sad = self.luma_sad(reference, mi_row, mi_col, start, best_sad);
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
                    let sad = self.luma_sad(reference, mi_row, mi_col, candidate, best_sad);
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

        let mode = if best_mv == Mv::default() {
            ZEROMV
        } else if best_mv == nearest {
            NEARESTMV
        } else if best_mv == near {
            NEARMV
        } else {
            NEWMV
        };
        let mut bits = tree_bits(
            &INTER_MODE_TREE,
            &self.context.inter_mode[mode_context * 3..mode_context * 3 + 3],
            mode - NEARESTMV,
        );
        if mode == NEWMV {
            bits += mv_bits(Mv {
                row: best_mv.row - nearest.row,
                col: best_mv.col - nearest.col,
            });
        }
        let above = self.above_info(mi_row, mi_col);
        let left = self.left_info(mi_row, mi_col);
        bits += bool_bits(
            true,
            self.context.intra_inter[intra_inter_context(above, left)],
        );
        bits += bool_bits(
            false,
            self.context.single_ref[single_ref_context(above, left) * 2],
        );

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
                best_mv.row * scale,
                best_mv.col * scale,
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
        Some(BlockChoice {
            info: ModeInfo {
                is_inter: true,
                mode,
                mv: best_mv,
                skip: false,
            },
            uv_mode: IntraMode::Dc,
            best_mv: nearest,
            levels,
            tx_types: [TxType::DctDct; 6],
            luma,
            chroma,
            cost: error as f64 + self.lambda * bits,
        })
    }

    fn write_mode_info(&mut self, mi_row: usize, mi_col: usize, choice: &BlockChoice) {
        let above = self.above_info(mi_row, mi_col);
        let left = self.left_info(mi_row, mi_col);
        let skip_context = usize::from(above.is_some_and(|info| info.skip))
            + usize::from(left.is_some_and(|info| info.skip));
        self.writer
            .write(choice.info.skip, self.context.skip[skip_context]);
        self.counts.skip[skip_context][usize::from(choice.info.skip)] += 1;
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
        let intra_inter = intra_inter_context(above, left);
        self.writer.write(
            choice.info.is_inter,
            self.context.intra_inter[intra_inter],
        );
        self.counts.intra_inter[intra_inter][usize::from(choice.info.is_inter)] += 1;
        if !choice.info.is_inter {
            // `size_group_lookup[BLOCK_8X8]` is 1.
            self.writer
                .tree(&INTRA_MODE_TREE, &self.context.y_mode[9..18], y_mode);
            self.counts.y_mode[1][usize::from(y_mode)] += 1;
            let index = usize::from(y_mode) * 9;
            self.writer.tree(
                &INTRA_MODE_TREE,
                &self.context.uv_mode[index..index + 9],
                choice.uv_mode as u8,
            );
            self.counts.uv_mode[usize::from(y_mode)][choice.uv_mode as usize] += 1;
            return;
        }
        // A single LAST_FRAME reference: the first single_ref bit is zero.
        let single_ref = single_ref_context(above, left);
        self.writer
            .write(false, self.context.single_ref[single_ref * 2]);
        self.counts.single_ref[single_ref][0] += 1;
        let (_, mode_context) = self.mv_references(mi_row, mi_col);
        self.writer.tree(
            &INTER_MODE_TREE,
            &self.context.inter_mode[mode_context * 3..mode_context * 3 + 3],
            y_mode - NEARESTMV,
        );
        self.counts.inter_mode[mode_context][usize::from(y_mode - NEARESTMV)] += 1;
        if y_mode == NEWMV {
            let difference = Mv {
                row: choice.info.mv.row - choice.best_mv.row,
                col: choice.info.mv.col - choice.best_mv.col,
            };
            write_mv(&mut self.writer, self.context, &mut self.counts, difference);
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
                &self.context.coef,
                &mut self.counts,
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

/// Writes one 4x4 block's tokens as `decode_coefs` reads them, counting them
/// as it does; returns whether any coefficient was nonzero.
fn write_coefficients(
    writer: &mut BoolEncoder,
    coef_probs: &[u8],
    counts: &mut FrameCounts,
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
    let context_index =
        |band: usize, context: usize| ((plane_type * 2 + reference) * 6 + band) * 6 + context;
    let mut cache = [0_u8; 16];
    let mut previous_zero = false;
    for c in 0..end {
        let index = context_index(COEF_BAND_4X4[c], context);
        let probs = &coef_probs[index * 3..index * 3 + 3];
        if !previous_zero {
            writer.write(true, probs[0]);
            counts.eob_branch[index] += 1;
        }
        let level = levels[scan[c]];
        let magnitude = level.unsigned_abs();
        // ZERO_TOKEN, ONE_TOKEN or TWO_TOKEN (any larger magnitude).
        counts.coef[index][magnitude.min(2) as usize] += 1;
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
        let index = context_index(COEF_BAND_4X4[end], context);
        writer.write(false, coef_probs[index * 3]);
        counts.eob_branch[index] += 1;
        counts.coef[index][3] += 1; // EOB_MODEL_TOKEN
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

/// Writes a motion vector difference and counts it as `vp9_inc_mv` does.
fn write_mv(
    writer: &mut BoolEncoder,
    context: &FrameContext,
    counts: &mut FrameCounts,
    difference: Mv,
) {
    let joint = usize::from(difference.row != 0) * 2 + usize::from(difference.col != 0);
    writer.tree(&MV_JOINT_TREE, &context.mv_joints, joint as u8);
    counts.mv_joints[joint] += 1;
    for (component, value) in [(0, difference.row), (1, difference.col)] {
        if value == 0 {
            continue;
        }
        let probs = &context.mv[component];
        let MvComponentCounts {
            sign,
            classes,
            class0,
            bits,
            class0_fp,
            fp: fp_counts,
        } = &mut counts.mv[component];
        writer.write(value < 0, probs.sign);
        sign[usize::from(value < 0)] += 1;
        let (class, offset) = mv_class(value.unsigned_abs() - 1);
        writer.tree(&MV_CLASS_TREE, &probs.classes, class as u8);
        classes[class] += 1;
        let integer = offset >> 3;
        if class == 0 {
            writer.write(integer != 0, probs.class0);
            class0[integer as usize] += 1;
        } else {
            for (bit, bit_counts) in bits.iter_mut().enumerate().take(class) {
                let value = (integer >> bit) & 1;
                writer.write(value != 0, probs.bits[bit]);
                bit_counts[value as usize] += 1;
            }
        }
        let fraction = ((offset >> 1) & 3) as u8;
        let (fp, fraction_counts) = if class == 0 {
            (
                &probs.class0_fp[integer as usize],
                &mut class0_fp[integer as usize],
            )
        } else {
            (&probs.fp, fp_counts)
        };
        writer.tree(&MV_FP_TREE, fp, fraction);
        fraction_counts[usize::from(fraction)] += 1;
        // Without high-precision vectors the eighth-sample bit is implied,
        // and its probabilities are never adapted.
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

fn mv_bits(difference: Mv) -> f64 {
    let component = |value: i32| -> f64 {
        if value == 0 {
            0.0
        } else {
            let (class, _) = mv_class(value.unsigned_abs() - 1);
            3.0 + 1.5 * class as f64
        }
    };
    2.0 + component(difference.row) + component(difference.col)
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
fn compressed_header(key: bool) -> Vec<u8> {
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
}
