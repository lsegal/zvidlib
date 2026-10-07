//! Tile decoding (sections 6.4 and 8.4 of the VP9 specification): the
//! partition tree, mode info and motion vector prediction, coefficient
//! tokens, and the per-block prediction and reconstruction that follow
//! them. Every function mirrors the libvpx decoder function named in its
//! documentation (`vp9_decodeframe.c`, `vp9_decodemv.c` and
//! `vp9_detokenize.c`), including the context and counting side effects
//! later blocks and the backward adaptation depend on.

use std::sync::Arc;

use super::loopfilter::{self, LoopFilterMask, MaskBlock};
use super::probs::*;
use super::recon::{self, IntraEdges};
use super::tables::*;
use super::{
    COMPOUND_REFERENCE, Frame, FrameHeader, REFERENCE_MODE_SELECT, SEG_LVL_ALT_LF, SEG_LVL_ALT_Q,
    SEG_LVL_REF_FRAME, SEG_LVL_SKIP, SWITCHABLE, Scale, Segmentation, TX_MODE_SELECT, malformed,
};
use crate::Result;
use zvidlib_vp9_syntax::bits::BoolDecoder;

const BLOCK_4X4: u8 = 0;
const BLOCK_8X8: u8 = 3;
const BLOCK_64X64: u8 = 12;
const BLOCK_INVALID: u8 = 13;

const PARTITION_HORZ: u8 = 1;
const PARTITION_VERT: u8 = 2;
const PARTITION_SPLIT: u8 = 3;

const NEARESTMV: u8 = 10;
const NEARMV: u8 = 11;
const ZEROMV: u8 = 12;
const NEWMV: u8 = 13;

const INTRA_FRAME: i8 = 0;
const LAST_FRAME: i8 = 1;
const GOLDEN_FRAME: i8 = 2;
const ALTREF_FRAME: i8 = 3;
const NONE_FRAME: i8 = -1;

/// A motion vector in 1/8 pixel units (`MV`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Mv {
    pub row: i16,
    pub col: i16,
}

impl Mv {
    const ZERO: Self = Self { row: 0, col: 0 };
    /// `invalid_mv` (0x80008000), which libvpx copies into the unused
    /// second vector of single-reference sub-8x8 blocks.
    const INVALID: Self = Self {
        row: i16::MIN,
        col: i16::MIN,
    };

    fn negated(self) -> Self {
        Self {
            row: self.row.wrapping_neg(),
            col: self.col.wrapping_neg(),
        }
    }
}

/// The mode info of one block (`MODE_INFO`), copied to every 8x8 position
/// the block covers.
#[derive(Clone, Copy, Debug)]
pub struct ModeInfo {
    sb_type: u8,
    mode: u8,
    uv_mode: u8,
    sub_modes: [u8; 4],
    ref_frame: [i8; 2],
    mv: [Mv; 2],
    sub_mvs: [[Mv; 2]; 4],
    interp_filter: u8,
    tx_size: u8,
    skip: bool,
    segment_id: u8,
    seg_id_predicted: bool,
}

impl Default for ModeInfo {
    fn default() -> Self {
        Self {
            sb_type: BLOCK_4X4,
            mode: 0,
            uv_mode: 0,
            sub_modes: [0; 4],
            ref_frame: [INTRA_FRAME, NONE_FRAME],
            mv: [Mv::ZERO; 2],
            sub_mvs: [[Mv::ZERO; 2]; 4],
            interp_filter: 3,
            tx_size: 0,
            skip: false,
            segment_id: 0,
            seg_id_predicted: false,
        }
    }
}

impl ModeInfo {
    fn is_inter(&self) -> bool {
        self.ref_frame[0] > INTRA_FRAME
    }

    fn has_second_ref(&self) -> bool {
        self.ref_frame[1] > INTRA_FRAME
    }

    /// `get_y_mode`.
    fn y_mode(&self, block: usize) -> u8 {
        if self.sb_type < BLOCK_8X8 {
            self.sub_modes[block]
        } else {
            self.mode
        }
    }
}

/// The motion of one 8x8 position, kept for the next frame's motion
/// vector prediction (`MV_REF`).
#[derive(Clone, Copy, Debug, Default)]
pub struct MvRef {
    ref_frame: [i8; 2],
    mv: [Mv; 2],
}

/// The decoding state of one tile column.
struct Tile<'a> {
    bd: BoolDecoder<'a>,
    mi_col_start: usize,
    mi_col_end: usize,
    left_context: [[u8; 16]; 3],
    left_partition: [u8; 8],
}

/// The neighbourhood of the block being decoded (the parts of libvpx's
/// `MACROBLOCKD` set by `set_offsets`).
#[derive(Clone, Copy)]
struct BlockContext {
    mi_row: usize,
    mi_col: usize,
    sb_type: u8,
    /// Size in 8x8 units, before clipping to the frame.
    bw: usize,
    bh: usize,
    /// `n4_wl` of the luma plane: log2 of the width in 4x4 units.
    bwl: usize,
    x_mis: usize,
    y_mis: usize,
    mb_to_left_edge: i32,
    mb_to_right_edge: i32,
    mb_to_top_edge: i32,
    mb_to_bottom_edge: i32,
    above: Option<ModeInfo>,
    left: Option<ModeInfo>,
    tile_start: usize,
    tile_end: usize,
    bmode_blocks_wl: usize,
    bmode_blocks_hl: usize,
}

/// Decodes the tiles of one frame into its buffer.
pub struct FrameDecoder<'a> {
    header: &'a FrameHeader,
    fc: &'a FrameContext,
    seg: &'a Segmentation,
    refs: &'a [Option<(Arc<Frame>, Scale)>; 3],
    prev_mvs: Option<&'a [MvRef]>,
    last_seg_map: &'a [u8],
    cur_seg_map: &'a mut Vec<u8>,
    frame: &'a mut Frame,
    mi_rows: usize,
    mi_cols: usize,
    pub counts: Option<Box<FrameCounts>>,
    pub cur_mvs: Vec<MvRef>,
    mi: Vec<ModeInfo>,
    above_context: [Vec<u8>; 3],
    above_partition: Vec<u8>,
    masks: Vec<LoopFilterMask>,
    lf_levels: [[[u8; 2]; 4]; 8],
    y_dequant: [[i32; 2]; 8],
    uv_dequant: [[i32; 2]; 8],
    coefficients: Vec<i32>,
    /// `token_cache` of `decode_coefs`. Coefficient contexts only read the
    /// entries of positions earlier in the scan, so it is never cleared.
    token_cache: Vec<u8>,
    mc_buffer: Vec<u8>,
    /// The horizontal pass of the two-dimensional convolution.
    convolve_temp: Box<[u8; 64 * 135]>,
}

impl<'a> FrameDecoder<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        header: &'a FrameHeader,
        fc: &'a FrameContext,
        seg: &'a Segmentation,
        lf_mode_ref_delta_enabled: bool,
        lf_ref_deltas: [i8; 4],
        lf_mode_deltas: [i8; 2],
        refs: &'a [Option<(Arc<Frame>, Scale)>; 3],
        prev_mvs: Option<&'a [MvRef]>,
        last_seg_map: &'a [u8],
        cur_seg_map: &'a mut Vec<u8>,
        frame: &'a mut Frame,
        mi_rows: usize,
        mi_cols: usize,
        counting: bool,
    ) -> Result<Self> {
        let aligned_cols = (mi_cols + 7) & !7;
        let sb_rows = mi_rows.div_ceil(8);
        let sb_cols = mi_cols.div_ceil(8);

        let mut y_dequant = [[0i32; 2]; 8];
        let mut uv_dequant = [[0i32; 2]; 8];
        for segment in 0..8u8 {
            // vp9_get_qindex
            let base = i32::from(header.base_qindex);
            let qindex = if seg.feature_active(segment, SEG_LVL_ALT_Q) {
                let data = seg.data(segment, SEG_LVL_ALT_Q);
                if seg.abs_delta { data } else { base + data }.clamp(0, 255)
            } else {
                base
            };
            let dc = |delta: i32| i32::from(DC_QLOOKUP[(qindex + delta).clamp(0, 255) as usize]);
            let ac = |delta: i32| i32::from(AC_QLOOKUP[(qindex + delta).clamp(0, 255) as usize]);
            y_dequant[usize::from(segment)] = [dc(header.y_dc_delta_q), ac(0)];
            uv_dequant[usize::from(segment)] = [dc(header.uv_dc_delta_q), ac(header.uv_ac_delta_q)];
        }

        let segment_levels = std::array::from_fn(|segment| {
            seg.feature_active(segment as u8, SEG_LVL_ALT_LF)
                .then(|| (seg.abs_delta, seg.data(segment as u8, SEG_LVL_ALT_LF)))
        });
        let lf_levels = loopfilter::filter_levels(
            header.filter_level,
            segment_levels,
            lf_mode_ref_delta_enabled,
            lf_ref_deltas,
            lf_mode_deltas,
        );

        Ok(Self {
            header,
            fc,
            seg,
            refs,
            prev_mvs,
            last_seg_map,
            cur_seg_map,
            frame,
            mi_rows,
            mi_cols,
            counts: counting.then(Box::default),
            cur_mvs: vec![MvRef::default(); mi_rows * mi_cols],
            mi: vec![ModeInfo::default(); mi_rows * mi_cols],
            above_context: [
                vec![0; aligned_cols * 2],
                vec![0; aligned_cols],
                vec![0; aligned_cols],
            ],
            above_partition: vec![0; aligned_cols],
            masks: vec![LoopFilterMask::default(); sb_rows * sb_cols],
            lf_levels,
            y_dequant,
            uv_dequant,
            coefficients: vec![0; 32 * 32],
            token_cache: vec![0; 32 * 32],
            mc_buffer: Vec::new(),
            convolve_temp: Box::new([0; 64 * 135]),
        })
    }

    /// `decode_tiles`: decodes every tile and applies the loop filter,
    /// returning the offset in `data` just past the last tile's data.
    pub fn decode_tiles(&mut self, data: &'a [u8]) -> Result<usize> {
        let tile_cols = 1usize << self.header.log2_tile_cols;
        let tile_rows = 1usize << self.header.log2_tile_rows;

        // get_tile_buffers
        let mut buffers = Vec::with_capacity(tile_rows * tile_cols);
        let mut offset = 0usize;
        for row in 0..tile_rows {
            for col in 0..tile_cols {
                let is_last = row == tile_rows - 1 && col == tile_cols - 1;
                let size = if is_last {
                    data.len() - offset
                } else {
                    if data.len() - offset < 4 {
                        return Err(malformed("VP9 tile size is truncated"));
                    }
                    let size =
                        u32::from_be_bytes(data[offset..offset + 4].try_into().expect("4 bytes"))
                            as usize;
                    offset += 4;
                    if size > data.len() - offset {
                        return Err(malformed("VP9 tile size exceeds the frame data"));
                    }
                    size
                };
                buffers.push((offset, size));
                offset += size;
            }
        }

        let mut tiles = Vec::with_capacity(tile_rows * tile_cols);
        for (index, &(start, size)) in buffers.iter().enumerate() {
            if size == 0 {
                return Err(malformed("VP9 tile is empty"));
            }
            let col = index % tile_cols;
            tiles.push(Tile {
                bd: BoolDecoder::new(&data[start..start + size])?,
                mi_col_start: tile_offset(col, self.mi_cols, self.header.log2_tile_cols),
                mi_col_end: tile_offset(col + 1, self.mi_cols, self.header.log2_tile_cols),
                left_context: [[0; 16]; 3],
                left_partition: [0; 8],
            });
        }

        for tile_row in 0..tile_rows {
            let row_start = tile_offset(tile_row, self.mi_rows, self.header.log2_tile_rows);
            let row_end = tile_offset(tile_row + 1, self.mi_rows, self.header.log2_tile_rows);
            for mi_row in (row_start..row_end).step_by(8) {
                for tile_col in 0..tile_cols {
                    let tile = &mut tiles[tile_row * tile_cols + tile_col];
                    tile.left_context = [[0; 16]; 3];
                    tile.left_partition = [0; 8];
                    for mi_col in (tile.mi_col_start..tile.mi_col_end).step_by(8) {
                        self.decode_partition(tile, mi_row, mi_col, BLOCK_64X64, 4)?;
                    }
                    if tile.bd.has_error() {
                        return Err(malformed("VP9 tile data is truncated"));
                    }
                }
            }
        }

        if self.header.filter_level != 0 {
            let [y, u, v] = &mut self.frame.planes;
            let mut planes = [
                loopfilter::FilterPlane {
                    data: &mut y.data,
                    stride: y.stride,
                    origin: y.origin,
                },
                loopfilter::FilterPlane {
                    data: &mut u.data,
                    stride: u.stride,
                    origin: u.origin,
                },
                loopfilter::FilterPlane {
                    data: &mut v.data,
                    stride: v.stride,
                    origin: v.origin,
                },
            ];
            loopfilter::filter_frame(
                &mut planes,
                &mut self.masks,
                self.mi_rows,
                self.mi_cols,
                self.header.sharpness,
            );
        }

        let (last_start, _) = *buffers.last().expect("at least one tile");
        let last = tiles.last().expect("at least one tile");
        Ok(last_start + last.bd.find_end())
    }

    fn count(&mut self) -> Option<&mut FrameCounts> {
        self.counts.as_deref_mut()
    }

    /// `decode_partition`.
    fn decode_partition(
        &mut self,
        tile: &mut Tile,
        mi_row: usize,
        mi_col: usize,
        bsize: u8,
        n4x4_l2: usize,
    ) -> Result<()> {
        if mi_row >= self.mi_rows || mi_col >= self.mi_cols {
            return Ok(());
        }
        let n8x8_l2 = n4x4_l2 - 1;
        let num_8x8_wh = 1usize << n8x8_l2;
        let hbs = num_8x8_wh >> 1;
        let has_rows = mi_row + hbs < self.mi_rows;
        let has_cols = mi_col + hbs < self.mi_cols;

        // read_partition
        let above = (self.above_partition[mi_col] >> n8x8_l2) & 1;
        let left = (tile.left_partition[mi_row & 7] >> n8x8_l2) & 1;
        let ctx = usize::from(left * 2 + above) + n8x8_l2 * 4;
        let probs = if self.header.is_intra_only() {
            &KF_PARTITION_PROBS[ctx]
        } else {
            &self.fc.partition[ctx]
        };
        let partition = if has_rows && has_cols {
            tile.bd.tree(&PARTITION_TREE, probs)
        } else if !has_rows && has_cols {
            if tile.bd.read(probs[1]) {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows && !has_cols {
            if tile.bd.read(probs[2]) {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        };
        if let Some(counts) = self.count() {
            counts.partition[ctx][usize::from(partition)] += 1;
        }

        let subsize = SUBSIZE_LOOKUP[usize::from(partition)][usize::from(bsize)];
        if subsize == BLOCK_INVALID {
            return Err(malformed("VP9 partition produces an invalid block size"));
        }
        if hbs == 0 {
            let wl = 1 >> usize::from(partition & PARTITION_VERT != 0);
            let hl = 1 >> usize::from(partition & PARTITION_HORZ != 0);
            self.decode_block(tile, mi_row, mi_col, subsize, 1, 1, wl, hl)?;
        } else {
            match partition {
                0 => self.decode_block(tile, mi_row, mi_col, subsize, n4x4_l2, n4x4_l2, 1, 1)?,
                PARTITION_HORZ => {
                    self.decode_block(tile, mi_row, mi_col, subsize, n4x4_l2, n8x8_l2, 1, 1)?;
                    if has_rows {
                        self.decode_block(
                            tile,
                            mi_row + hbs,
                            mi_col,
                            subsize,
                            n4x4_l2,
                            n8x8_l2,
                            1,
                            1,
                        )?;
                    }
                }
                PARTITION_VERT => {
                    self.decode_block(tile, mi_row, mi_col, subsize, n8x8_l2, n4x4_l2, 1, 1)?;
                    if has_cols {
                        self.decode_block(
                            tile,
                            mi_row,
                            mi_col + hbs,
                            subsize,
                            n8x8_l2,
                            n4x4_l2,
                            1,
                            1,
                        )?;
                    }
                }
                _ => {
                    self.decode_partition(tile, mi_row, mi_col, subsize, n8x8_l2)?;
                    self.decode_partition(tile, mi_row, mi_col + hbs, subsize, n8x8_l2)?;
                    self.decode_partition(tile, mi_row + hbs, mi_col, subsize, n8x8_l2)?;
                    self.decode_partition(tile, mi_row + hbs, mi_col + hbs, subsize, n8x8_l2)?;
                }
            }
        }

        if bsize >= BLOCK_8X8 && (bsize == BLOCK_8X8 || partition != PARTITION_SPLIT) {
            let (above_value, left_value) = PARTITION_CONTEXT_LOOKUP[usize::from(subsize)];
            self.above_partition[mi_col..mi_col + num_8x8_wh].fill(above_value);
            let left_start = mi_row & 7;
            tile.left_partition[left_start..left_start + num_8x8_wh].fill(left_value);
        }
        Ok(())
    }

    /// `decode_block`.
    #[allow(clippy::too_many_arguments)]
    fn decode_block(
        &mut self,
        tile: &mut Tile,
        mi_row: usize,
        mi_col: usize,
        bsize: u8,
        bwl: usize,
        bhl: usize,
        bmode_blocks_wl: usize,
        bmode_blocks_hl: usize,
    ) -> Result<()> {
        let bw = 1usize << (bwl - 1);
        let bh = 1usize << (bhl - 1);
        let x_mis = bw.min(self.mi_cols - mi_col);
        let y_mis = bh.min(self.mi_rows - mi_row);
        let ctx = BlockContext {
            mi_row,
            mi_col,
            sb_type: bsize,
            bw,
            bh,
            bwl,
            x_mis,
            y_mis,
            mb_to_left_edge: -((mi_col * 64) as i32),
            mb_to_right_edge: (self.mi_cols as i32 - bw as i32 - mi_col as i32) * 64,
            mb_to_top_edge: -((mi_row * 64) as i32),
            mb_to_bottom_edge: (self.mi_rows as i32 - bh as i32 - mi_row as i32) * 64,
            above: (mi_row != 0).then(|| self.mi[(mi_row - 1) * self.mi_cols + mi_col]),
            left: (mi_col > tile.mi_col_start).then(|| self.mi[mi_row * self.mi_cols + mi_col - 1]),
            tile_start: tile.mi_col_start,
            tile_end: tile.mi_col_end,
            bmode_blocks_wl,
            bmode_blocks_hl,
        };
        if bsize >= BLOCK_8X8 && SS_SIZE_LOOKUP[usize::from(bsize)][1][1] == BLOCK_INVALID {
            return Err(malformed("VP9 block size is invalid for 4:2:0"));
        }

        let mut mi = self.read_mode_info(tile, &ctx)?;

        if mi.skip {
            self.reset_skip_context(tile, &ctx);
        }
        if !mi.is_inter() {
            for plane in 0..3 {
                let ss = usize::from(plane > 0);
                let tx_size = self.plane_tx_size(&mi, plane);
                let step = 1usize << tx_size;
                let (max_wide, max_high) = max_blocks(&ctx, ss);
                let mut row = 0;
                while row < max_high {
                    let mut col = 0;
                    while col < max_wide {
                        self.predict_and_reconstruct_intra(
                            tile, &ctx, &mi, plane, row, col, tx_size,
                        )?;
                        col += step;
                    }
                    row += step;
                }
            }
        } else {
            self.predict_inter(&ctx, &mi)?;
            if !mi.skip {
                let mut eob_total = 0;
                for plane in 0..3 {
                    let ss = usize::from(plane > 0);
                    let tx_size = self.plane_tx_size(&mi, plane);
                    let step = 1usize << tx_size;
                    let (max_wide, max_high) = max_blocks(&ctx, ss);
                    let mut row = 0;
                    while row < max_high {
                        let mut col = 0;
                        while col < max_wide {
                            eob_total +=
                                self.reconstruct_inter(tile, &ctx, &mi, plane, row, col, tx_size);
                            col += step;
                        }
                        row += step;
                    }
                }
                if bsize >= BLOCK_8X8 && eob_total == 0 {
                    mi.skip = true;
                }
            }
        }

        for y in 0..y_mis {
            let start = (mi_row + y) * self.mi_cols + mi_col;
            self.mi[start..start + x_mis].fill(mi);
        }

        if self.header.filter_level != 0 {
            let mode_class = usize::from(mi.is_inter() && mi.mode != ZEROMV);
            let filter_level = self.lf_levels[usize::from(mi.segment_id)]
                [mi.ref_frame[0].max(0) as usize][mode_class];
            let sb_cols = self.mi_cols.div_ceil(8);
            loopfilter::build_mask(
                &mut self.masks[(mi_row >> 3) * sb_cols + (mi_col >> 3)],
                &MaskBlock {
                    sb_type: mi.sb_type,
                    tx_size: mi.tx_size,
                    skip_inter: mi.skip && mi.is_inter(),
                    filter_level,
                },
                mi_row,
                mi_col,
                bw,
                bh,
            );
        }
        Ok(())
    }

    fn plane_tx_size(&self, mi: &ModeInfo, plane: usize) -> u8 {
        if plane == 0 {
            mi.tx_size
        } else {
            UV_TXSIZE_LOOKUP[usize::from(mi.sb_type)][usize::from(mi.tx_size)][1][1]
        }
    }

    /// `dec_reset_skip_context`.
    fn reset_skip_context(&mut self, tile: &mut Tile, ctx: &BlockContext) {
        for plane in 0..3 {
            let ss = usize::from(plane > 0);
            let n4_w = (ctx.bw << 1) >> ss;
            let n4_h = (ctx.bh << 1) >> ss;
            let above = (ctx.mi_col * 2) >> ss;
            self.above_context[plane][above..above + n4_w].fill(0);
            let left = ((ctx.mi_row * 2) & 15) >> ss;
            tile.left_context[plane][left..left + n4_h].fill(0);
        }
    }

    // --- Mode info (vp9_decodemv.c) --------------------------------------

    /// `vp9_read_mode_info`.
    fn read_mode_info(&mut self, tile: &mut Tile, ctx: &BlockContext) -> Result<ModeInfo> {
        let mut mi = ModeInfo {
            sb_type: ctx.sb_type,
            ..ModeInfo::default()
        };
        if self.header.is_intra_only() {
            self.read_intra_frame_mode_info(tile, ctx, &mut mi);
        } else {
            self.read_inter_frame_mode_info(tile, ctx, &mut mi)?;
            let mv_ref = MvRef {
                ref_frame: mi.ref_frame,
                mv: mi.mv,
            };
            for y in 0..ctx.y_mis {
                let start = (ctx.mi_row + y) * self.mi_cols + ctx.mi_col;
                self.cur_mvs[start..start + ctx.x_mis].fill(mv_ref);
            }
        }
        Ok(mi)
    }

    fn read_intra_frame_mode_info(
        &mut self,
        tile: &mut Tile,
        ctx: &BlockContext,
        mi: &mut ModeInfo,
    ) {
        mi.segment_id = self.read_intra_segment_id(tile, ctx);
        mi.skip = self.read_skip(tile, ctx, mi.segment_id);
        mi.tx_size = self.read_tx_size(tile, ctx, true);
        mi.ref_frame = [INTRA_FRAME, NONE_FRAME];
        let kf_mode = |tile: &mut Tile, mi: &ModeInfo, block: usize| {
            let above = above_block_mode(mi, ctx.above.as_ref(), block);
            let left = left_block_mode(mi, ctx.left.as_ref(), block);
            tile.bd.tree(
                &INTRA_MODE_TREE,
                &KF_Y_MODE_PROB[usize::from(above)][usize::from(left)],
            )
        };
        match mi.sb_type {
            0 => {
                for block in 0..4 {
                    mi.sub_modes[block] = kf_mode(tile, mi, block);
                }
                mi.mode = mi.sub_modes[3];
            }
            1 => {
                let first = kf_mode(tile, mi, 0);
                mi.sub_modes[0] = first;
                mi.sub_modes[2] = first;
                let second = kf_mode(tile, mi, 1);
                mi.sub_modes[1] = second;
                mi.sub_modes[3] = second;
                mi.mode = second;
            }
            2 => {
                let first = kf_mode(tile, mi, 0);
                mi.sub_modes[0] = first;
                mi.sub_modes[1] = first;
                let second = kf_mode(tile, mi, 2);
                mi.sub_modes[2] = second;
                mi.sub_modes[3] = second;
                mi.mode = second;
            }
            _ => mi.mode = kf_mode(tile, mi, 0),
        }
        mi.uv_mode = tile
            .bd
            .tree(&INTRA_MODE_TREE, &KF_UV_MODE_PROB[usize::from(mi.mode)]);
    }

    fn read_inter_frame_mode_info(
        &mut self,
        tile: &mut Tile,
        ctx: &BlockContext,
        mi: &mut ModeInfo,
    ) -> Result<()> {
        mi.segment_id = self.read_inter_segment_id(tile, ctx, mi);
        mi.skip = self.read_skip(tile, ctx, mi.segment_id);
        let inter = self.read_is_inter_block(tile, ctx, mi.segment_id);
        mi.tx_size = self.read_tx_size(tile, ctx, !mi.skip || !inter);
        if inter {
            self.read_inter_block_mode_info(tile, ctx, mi)
        } else {
            self.read_intra_block_mode_info(tile, mi);
            Ok(())
        }
    }

    /// `read_intra_segment_id`.
    fn read_intra_segment_id(&mut self, tile: &mut Tile, ctx: &BlockContext) -> u8 {
        if !self.seg.enabled {
            return 0;
        }
        if !self.seg.update_map {
            self.copy_segment_id(ctx);
            return 0;
        }
        let segment_id = tile.bd.tree(&SEGMENT_TREE, &self.seg.tree_probs);
        self.set_segment_id(ctx, segment_id);
        segment_id
    }

    /// `read_inter_segment_id`.
    fn read_inter_segment_id(
        &mut self,
        tile: &mut Tile,
        ctx: &BlockContext,
        mi: &mut ModeInfo,
    ) -> u8 {
        if !self.seg.enabled {
            return 0;
        }
        let mut predicted = u8::MAX;
        for y in 0..ctx.y_mis {
            for x in 0..ctx.x_mis {
                let index = (ctx.mi_row + y) * self.mi_cols + ctx.mi_col + x;
                predicted = predicted.min(self.last_seg_map[index]);
            }
        }
        if !self.seg.update_map {
            self.copy_segment_id(ctx);
            return predicted;
        }
        let segment_id = if self.seg.temporal_update {
            let context = usize::from(ctx.above.is_some_and(|m| m.seg_id_predicted))
                + usize::from(ctx.left.is_some_and(|m| m.seg_id_predicted));
            mi.seg_id_predicted = tile.bd.read(self.seg.pred_probs[context]);
            if mi.seg_id_predicted {
                predicted
            } else {
                tile.bd.tree(&SEGMENT_TREE, &self.seg.tree_probs)
            }
        } else {
            tile.bd.tree(&SEGMENT_TREE, &self.seg.tree_probs)
        };
        self.set_segment_id(ctx, segment_id);
        segment_id
    }

    fn copy_segment_id(&mut self, ctx: &BlockContext) {
        for y in 0..ctx.y_mis {
            let start = (ctx.mi_row + y) * self.mi_cols + ctx.mi_col;
            self.cur_seg_map[start..start + ctx.x_mis]
                .copy_from_slice(&self.last_seg_map[start..start + ctx.x_mis]);
        }
    }

    fn set_segment_id(&mut self, ctx: &BlockContext, segment_id: u8) {
        for y in 0..ctx.y_mis {
            let start = (ctx.mi_row + y) * self.mi_cols + ctx.mi_col;
            self.cur_seg_map[start..start + ctx.x_mis].fill(segment_id);
        }
    }

    /// `read_skip`.
    fn read_skip(&mut self, tile: &mut Tile, ctx: &BlockContext, segment_id: u8) -> bool {
        if self.seg.feature_active(segment_id, SEG_LVL_SKIP) {
            return true;
        }
        let context = usize::from(ctx.above.is_some_and(|m| m.skip))
            + usize::from(ctx.left.is_some_and(|m| m.skip));
        let skip = tile.bd.read(self.fc.skip[context]);
        if let Some(counts) = self.count() {
            counts.skip[context][usize::from(skip)] += 1;
        }
        skip
    }

    /// `read_tx_size`.
    fn read_tx_size(&mut self, tile: &mut Tile, ctx: &BlockContext, allow_select: bool) -> u8 {
        let max_tx_size = MAX_TXSIZE_LOOKUP[usize::from(ctx.sb_type)];
        if !(allow_select && self.header.tx_mode == TX_MODE_SELECT && ctx.sb_type >= BLOCK_8X8) {
            return max_tx_size.min(TX_MODE_TO_BIGGEST_TX_SIZE[usize::from(self.header.tx_mode)]);
        }
        // get_tx_size_context
        let mut above_ctx = match ctx.above {
            Some(m) if !m.skip => m.tx_size,
            _ => max_tx_size,
        };
        let mut left_ctx = match ctx.left {
            Some(m) if !m.skip => m.tx_size,
            _ => max_tx_size,
        };
        if ctx.left.is_none() {
            left_ctx = above_ctx;
        }
        if ctx.above.is_none() {
            above_ctx = left_ctx;
        }
        let context = usize::from(above_ctx + left_ctx > max_tx_size);
        let probs: &[u8] = match max_tx_size {
            1 => &self.fc.tx8[context],
            2 => &self.fc.tx16[context],
            _ => &self.fc.tx32[context],
        };
        let mut tx_size = u8::from(tile.bd.read(probs[0]));
        if tx_size != 0 && max_tx_size >= 2 {
            tx_size += u8::from(tile.bd.read(probs[1]));
            if tx_size != 1 && max_tx_size >= 3 {
                tx_size += u8::from(tile.bd.read(probs[2]));
            }
        }
        if let Some(counts) = self.count() {
            match max_tx_size {
                1 => counts.tx8[context][usize::from(tx_size)] += 1,
                2 => counts.tx16[context][usize::from(tx_size)] += 1,
                _ => counts.tx32[context][usize::from(tx_size)] += 1,
            }
        }
        tx_size
    }

    /// `read_is_inter_block`.
    fn read_is_inter_block(&mut self, tile: &mut Tile, ctx: &BlockContext, segment_id: u8) -> bool {
        if self.seg.feature_active(segment_id, SEG_LVL_REF_FRAME) {
            return self.seg.data(segment_id, SEG_LVL_REF_FRAME) != i32::from(INTRA_FRAME);
        }
        // get_intra_inter_context
        let context = match (ctx.above, ctx.left) {
            (Some(above), Some(left)) => {
                let above_intra = !above.is_inter();
                let left_intra = !left.is_inter();
                if above_intra && left_intra {
                    3
                } else {
                    usize::from(above_intra || left_intra)
                }
            }
            (Some(edge), None) | (None, Some(edge)) => 2 * usize::from(!edge.is_inter()),
            (None, None) => 0,
        };
        let inter = tile.bd.read(self.fc.intra_inter[context]);
        if let Some(counts) = self.count() {
            counts.intra_inter[context][usize::from(inter)] += 1;
        }
        inter
    }

    /// `read_intra_block_mode_info`.
    fn read_intra_block_mode_info(&mut self, tile: &mut Tile, mi: &mut ModeInfo) {
        let mut read_y = |this: &mut Self, size_group: usize| {
            let mode = tile.bd.tree(&INTRA_MODE_TREE, &this.fc.y_mode[size_group]);
            if let Some(counts) = this.count() {
                counts.y_mode[size_group][usize::from(mode)] += 1;
            }
            mode
        };
        match mi.sb_type {
            0 => {
                for block in 0..4 {
                    mi.sub_modes[block] = read_y(self, 0);
                }
                mi.mode = mi.sub_modes[3];
            }
            1 => {
                let first = read_y(self, 0);
                mi.sub_modes[0] = first;
                mi.sub_modes[2] = first;
                let second = read_y(self, 0);
                mi.sub_modes[1] = second;
                mi.sub_modes[3] = second;
                mi.mode = second;
            }
            2 => {
                let first = read_y(self, 0);
                mi.sub_modes[0] = first;
                mi.sub_modes[1] = first;
                let second = read_y(self, 0);
                mi.sub_modes[2] = second;
                mi.sub_modes[3] = second;
                mi.mode = second;
            }
            sb_type => {
                mi.mode = read_y(self, usize::from(SIZE_GROUP_LOOKUP[usize::from(sb_type)]));
            }
        }
        mi.uv_mode = tile
            .bd
            .tree(&INTRA_MODE_TREE, &self.fc.uv_mode[usize::from(mi.mode)]);
        if let Some(counts) = self.count() {
            counts.uv_mode[usize::from(mi.mode)][usize::from(mi.uv_mode)] += 1;
        }
        mi.interp_filter = 3;
        mi.ref_frame = [INTRA_FRAME, NONE_FRAME];
    }

    /// `read_ref_frames`.
    fn read_ref_frames(&mut self, tile: &mut Tile, ctx: &BlockContext, segment_id: u8) -> [i8; 2] {
        if self.seg.feature_active(segment_id, SEG_LVL_REF_FRAME) {
            return [
                self.seg.data(segment_id, SEG_LVL_REF_FRAME) as i8,
                NONE_FRAME,
            ];
        }
        let header = self.header;
        let mode = if header.reference_mode == REFERENCE_MODE_SELECT {
            let context = self.reference_mode_context(ctx);
            let compound = tile.bd.read(self.fc.comp_inter[context]);
            if let Some(counts) = self.count() {
                counts.comp_inter[context][usize::from(compound)] += 1;
            }
            u8::from(compound)
        } else {
            header.reference_mode
        };
        if mode == COMPOUND_REFERENCE {
            let index = usize::from(header.ref_frame_sign_bias[header.comp_fixed_ref as usize]);
            let context = self.comp_ref_context(ctx);
            let bit = tile.bd.read(self.fc.comp_ref[context]);
            if let Some(counts) = self.count() {
                counts.comp_ref[context][usize::from(bit)] += 1;
            }
            let mut refs = [0i8; 2];
            refs[index] = header.comp_fixed_ref;
            refs[1 - index] = header.comp_var_ref[usize::from(bit)];
            refs
        } else {
            let context0 = single_ref_p1_context(ctx);
            let bit0 = tile.bd.read(self.fc.single_ref[context0][0]);
            if let Some(counts) = self.count() {
                counts.single_ref[context0][0][usize::from(bit0)] += 1;
            }
            let first = if bit0 {
                let context1 = single_ref_p2_context(ctx);
                let bit1 = tile.bd.read(self.fc.single_ref[context1][1]);
                if let Some(counts) = self.count() {
                    counts.single_ref[context1][1][usize::from(bit1)] += 1;
                }
                if bit1 { ALTREF_FRAME } else { GOLDEN_FRAME }
            } else {
                LAST_FRAME
            };
            [first, NONE_FRAME]
        }
    }

    /// `vp9_get_reference_mode_context`.
    fn reference_mode_context(&self, ctx: &BlockContext) -> usize {
        let fixed = self.header.comp_fixed_ref;
        match (ctx.above, ctx.left) {
            (Some(above), Some(left)) => {
                if !above.has_second_ref() && !left.has_second_ref() {
                    usize::from((above.ref_frame[0] == fixed) ^ (left.ref_frame[0] == fixed))
                } else if !above.has_second_ref() {
                    2 + usize::from(above.ref_frame[0] == fixed || !above.is_inter())
                } else if !left.has_second_ref() {
                    2 + usize::from(left.ref_frame[0] == fixed || !left.is_inter())
                } else {
                    4
                }
            }
            (Some(edge), None) | (None, Some(edge)) => {
                if !edge.has_second_ref() {
                    usize::from(edge.ref_frame[0] == fixed)
                } else {
                    3
                }
            }
            (None, None) => 1,
        }
    }

    /// `vp9_get_pred_context_comp_ref_p`.
    fn comp_ref_context(&self, ctx: &BlockContext) -> usize {
        let header = self.header;
        let fix_ref_idx = usize::from(header.ref_frame_sign_bias[header.comp_fixed_ref as usize]);
        let var_ref_idx = 1 - fix_ref_idx;
        let var1 = header.comp_var_ref[1];
        let var0 = header.comp_var_ref[0];
        match (ctx.above, ctx.left) {
            (Some(above), Some(left)) => {
                let above_intra = !above.is_inter();
                let left_intra = !left.is_inter();
                if above_intra && left_intra {
                    2
                } else if above_intra || left_intra {
                    let edge = if above_intra { left } else { above };
                    if !edge.has_second_ref() {
                        1 + 2 * usize::from(edge.ref_frame[0] != var1)
                    } else {
                        1 + 2 * usize::from(edge.ref_frame[var_ref_idx] != var1)
                    }
                } else {
                    let l_sg = !left.has_second_ref();
                    let a_sg = !above.has_second_ref();
                    let vrfa = if a_sg {
                        above.ref_frame[0]
                    } else {
                        above.ref_frame[var_ref_idx]
                    };
                    let vrfl = if l_sg {
                        left.ref_frame[0]
                    } else {
                        left.ref_frame[var_ref_idx]
                    };
                    if vrfa == vrfl && var1 == vrfa {
                        0
                    } else if l_sg && a_sg {
                        if (vrfa == header.comp_fixed_ref && vrfl == var0)
                            || (vrfl == header.comp_fixed_ref && vrfa == var0)
                        {
                            4
                        } else if vrfa == vrfl {
                            3
                        } else {
                            1
                        }
                    } else if l_sg || a_sg {
                        let vrfc = if l_sg { vrfa } else { vrfl };
                        let rfs = if a_sg { vrfa } else { vrfl };
                        if vrfc == var1 && rfs != var1 {
                            1
                        } else if rfs == var1 && vrfc != var1 {
                            2
                        } else {
                            4
                        }
                    } else if vrfa == vrfl {
                        4
                    } else {
                        2
                    }
                }
            }
            (Some(edge), None) | (None, Some(edge)) => {
                if !edge.is_inter() {
                    2
                } else if edge.has_second_ref() {
                    4 * usize::from(edge.ref_frame[var_ref_idx] != var1)
                } else {
                    3 * usize::from(edge.ref_frame[0] != var1)
                }
            }
            (None, None) => 2,
        }
    }

    /// `read_inter_block_mode_info`.
    fn read_inter_block_mode_info(
        &mut self,
        tile: &mut Tile,
        ctx: &BlockContext,
        mi: &mut ModeInfo,
    ) -> Result<()> {
        let allow_hp = self.header.allow_high_precision_mv;
        mi.ref_frame = self.read_ref_frames(tile, ctx, mi.segment_id);
        let is_compound = mi.has_second_ref();
        let search = &MV_REF_BLOCKS[usize::from(ctx.sb_type)];
        let inter_mode_ctx = self.mode_context(ctx, search);

        if self.seg.feature_active(mi.segment_id, SEG_LVL_SKIP) {
            mi.mode = ZEROMV;
            if ctx.sb_type < BLOCK_8X8 {
                return Err(malformed(
                    "VP9 segment skip feature is used on a block smaller than 8x8",
                ));
            }
        } else if ctx.sb_type >= BLOCK_8X8 {
            mi.mode = self.read_inter_mode(tile, inter_mode_ctx);
        }

        mi.interp_filter = if self.header.interp_filter == SWITCHABLE {
            // get_pred_context_switchable_interp
            let left_type = ctx.left.map_or(3, |m| m.interp_filter);
            let above_type = ctx.above.map_or(3, |m| m.interp_filter);
            let context = if left_type == above_type {
                left_type
            } else if left_type == 3 {
                above_type
            } else if above_type == 3 {
                left_type
            } else {
                3
            };
            let filter = tile.bd.tree(
                &SWITCHABLE_INTERP_TREE,
                &self.fc.switchable_interp[usize::from(context)],
            );
            if let Some(counts) = self.count() {
                counts.switchable_interp[usize::from(context)][usize::from(filter)] += 1;
            }
            filter
        } else {
            self.header.interp_filter
        };

        let mut best_ref_mvs = [Mv::ZERO; 2];
        if ctx.sb_type < BLOCK_8X8 {
            let num_4x4_w = 1 << ctx.bmode_blocks_wl;
            let num_4x4_h = 1 << ctx.bmode_blocks_hl;
            let mut got_mv_refs_for_new = false;
            let mut best_sub8x8 = [Mv::ZERO, Mv::INVALID];
            let mut b_mode = 0;
            let mut idy = 0;
            while idy < 2 {
                let mut idx = 0;
                while idx < 2 {
                    let j = idy * 2 + idx;
                    b_mode = self.read_inter_mode(tile, inter_mode_ctx);
                    if b_mode == NEARESTMV || b_mode == NEARMV {
                        for reference in 0..1 + usize::from(is_compound) {
                            best_sub8x8[reference] = self
                                .append_sub8x8_mvs_for_idx(ctx, mi, search, b_mode, j, reference);
                        }
                    } else if b_mode == NEWMV && !got_mv_refs_for_new {
                        for reference in 0..1 + usize::from(is_compound) {
                            let (mut list, _) =
                                self.find_mv_refs(ctx, NEWMV, mi.ref_frame[reference], search, -1);
                            lower_mv_precision(&mut list[0], allow_hp);
                            best_ref_mvs[reference] = list[0];
                            got_mv_refs_for_new = true;
                        }
                    }
                    let mut block_mvs = mi.sub_mvs[j];
                    if !self.assign_mv(
                        tile,
                        b_mode,
                        &mut block_mvs,
                        &best_ref_mvs,
                        &best_sub8x8,
                        is_compound,
                        allow_hp,
                    ) {
                        return Err(malformed("VP9 motion vector is out of range"));
                    }
                    mi.sub_mvs[j] = block_mvs;
                    if num_4x4_h == 2 {
                        mi.sub_mvs[j + 2] = block_mvs;
                    }
                    if num_4x4_w == 2 {
                        mi.sub_mvs[j + 1] = block_mvs;
                    }
                    idx += num_4x4_w;
                }
                idy += num_4x4_h;
            }
            mi.mode = b_mode;
            mi.mv = mi.sub_mvs[3];
        } else {
            if mi.mode != ZEROMV {
                for reference in 0..1 + usize::from(is_compound) {
                    let (mut list, count) =
                        self.find_mv_refs(ctx, mi.mode, mi.ref_frame[reference], search, -1);
                    lower_mv_precision(&mut list[count - 1], allow_hp);
                    best_ref_mvs[reference] = list[count - 1];
                }
            }
            let mut mvs = mi.mv;
            let nearest = best_ref_mvs;
            if !self.assign_mv(
                tile,
                mi.mode,
                &mut mvs,
                &best_ref_mvs,
                &nearest,
                is_compound,
                allow_hp,
            ) {
                return Err(malformed("VP9 motion vector is out of range"));
            }
            mi.mv = mvs;
        }
        Ok(())
    }

    fn read_inter_mode(&mut self, tile: &mut Tile, context: usize) -> u8 {
        let offset = tile.bd.tree(&INTER_MODE_TREE, &self.fc.inter_mode[context]);
        if let Some(counts) = self.count() {
            counts.inter_mode[context][usize::from(offset)] += 1;
        }
        NEARESTMV + offset
    }

    /// `get_mode_context`.
    fn mode_context(&self, ctx: &BlockContext, search: &[(i32, i32); 8]) -> usize {
        let mut counter = 0;
        for &(row, col) in &search[..2] {
            if let Some(candidate) = self.candidate(ctx, row, col) {
                counter += MODE_2_COUNTER[usize::from(candidate.mode)];
            }
        }
        COUNTER_TO_CONTEXT[counter]
    }

    /// The mode info at a search offset, if `is_inside` the tile.
    fn candidate(&self, ctx: &BlockContext, row: i32, col: i32) -> Option<ModeInfo> {
        let r = ctx.mi_row as i32 + row;
        let c = ctx.mi_col as i32 + col;
        if r < 0
            || c < ctx.tile_start as i32
            || r >= self.mi_rows as i32
            || c >= ctx.tile_end as i32
        {
            return None;
        }
        Some(self.mi[r as usize * self.mi_cols + c as usize])
    }

    /// `dec_find_mv_refs`, returning the candidate list and how many of
    /// its entries are meaningful.
    fn find_mv_refs(
        &self,
        ctx: &BlockContext,
        mode: u8,
        ref_frame: i8,
        search: &[(i32, i32); 8],
        block: i32,
    ) -> ([Mv; 2], usize) {
        let sign_bias = &self.header.ref_frame_sign_bias;
        let early_break = mode != NEARMV;
        let mut list = [Mv::ZERO; 2];
        let mut count = 0usize;
        let mut different_ref_found = false;
        let prev = self
            .prev_mvs
            .map(|mvs| mvs[ctx.mi_row * self.mi_cols + ctx.mi_col]);

        // ADD_MV_REF_LIST_EB: returns whether the search is done.
        let add = |mv: Mv, list: &mut [Mv; 2], count: &mut usize| -> bool {
            if *count > 0 {
                if mv != list[0] {
                    list[*count] = mv;
                    *count += 1;
                    return true;
                }
                false
            } else {
                list[0] = mv;
                *count = 1;
                early_break
            }
        };
        let scale = |candidate: &ModeInfo, which: usize| -> Mv {
            let mv = candidate.mv[which];
            if sign_bias[candidate.ref_frame[which] as usize] != sign_bias[ref_frame as usize] {
                mv.negated()
            } else {
                mv
            }
        };

        let done = 'search: {
            let mut i = 0;
            if block >= 0 {
                while i < 2 {
                    let (row, col) = search[i];
                    if let Some(candidate) = self.candidate(ctx, row, col) {
                        different_ref_found = true;
                        let which = if candidate.ref_frame[0] == ref_frame {
                            Some(0)
                        } else if candidate.ref_frame[1] == ref_frame {
                            Some(1)
                        } else {
                            None
                        };
                        if let Some(which) = which {
                            let mv = sub_block_mv(&candidate, which, col, block);
                            if add(mv, &mut list, &mut count) {
                                break 'search true;
                            }
                        }
                    }
                    i += 1;
                }
            }
            while i < 8 {
                let (row, col) = search[i];
                if let Some(candidate) = self.candidate(ctx, row, col) {
                    different_ref_found = true;
                    if candidate.ref_frame[0] == ref_frame {
                        if add(candidate.mv[0], &mut list, &mut count) {
                            break 'search true;
                        }
                    } else if candidate.ref_frame[1] == ref_frame
                        && add(candidate.mv[1], &mut list, &mut count)
                    {
                        break 'search true;
                    }
                }
                i += 1;
            }
            if let Some(prev) = prev {
                if prev.ref_frame[0] == ref_frame {
                    if add(prev.mv[0], &mut list, &mut count) {
                        break 'search true;
                    }
                } else if prev.ref_frame[1] == ref_frame && add(prev.mv[1], &mut list, &mut count) {
                    break 'search true;
                }
            }
            if different_ref_found {
                for &(row, col) in search {
                    if let Some(candidate) = self.candidate(ctx, row, col) {
                        if candidate.is_inter() {
                            if candidate.ref_frame[0] != ref_frame
                                && add(scale(&candidate, 0), &mut list, &mut count)
                            {
                                break 'search true;
                            }
                            if candidate.has_second_ref()
                                && candidate.ref_frame[1] != ref_frame
                                && candidate.mv[1] != candidate.mv[0]
                                && add(scale(&candidate, 1), &mut list, &mut count)
                            {
                                break 'search true;
                            }
                        }
                    }
                }
            }
            if let Some(prev) = prev {
                if prev.ref_frame[0] != ref_frame && prev.ref_frame[0] > INTRA_FRAME {
                    let mut mv = prev.mv[0];
                    if sign_bias[prev.ref_frame[0] as usize] != sign_bias[ref_frame as usize] {
                        mv = mv.negated();
                    }
                    if add(mv, &mut list, &mut count) {
                        break 'search true;
                    }
                }
                if prev.ref_frame[1] > INTRA_FRAME
                    && prev.ref_frame[1] != ref_frame
                    && prev.mv[1] != prev.mv[0]
                {
                    let mut mv = prev.mv[1];
                    if sign_bias[prev.ref_frame[1] as usize] != sign_bias[ref_frame as usize] {
                        mv = mv.negated();
                    }
                    if add(mv, &mut list, &mut count) {
                        break 'search true;
                    }
                }
            }
            false
        };
        if !done {
            count = if mode == NEARMV { 2 } else { 1 };
        }
        // clamp_mv_ref
        for mv in &mut list[..count] {
            clamp_mv(
                mv,
                ctx.mb_to_left_edge - MV_BORDER,
                ctx.mb_to_right_edge + MV_BORDER,
                ctx.mb_to_top_edge - MV_BORDER,
                ctx.mb_to_bottom_edge + MV_BORDER,
            );
        }
        (list, count)
    }

    /// `append_sub8x8_mvs_for_idx`.
    fn append_sub8x8_mvs_for_idx(
        &self,
        ctx: &BlockContext,
        mi: &ModeInfo,
        search: &[(i32, i32); 8],
        b_mode: u8,
        block: usize,
        reference: usize,
    ) -> Mv {
        let ref_frame = mi.ref_frame[reference];
        let sub = |index: usize| mi.sub_mvs[index][reference];
        match block {
            0 => {
                let (list, count) = self.find_mv_refs(ctx, b_mode, ref_frame, search, 0);
                list[count - 1]
            }
            1 | 2 => {
                if b_mode == NEARESTMV {
                    sub(0)
                } else {
                    let (list, _) = self.find_mv_refs(ctx, b_mode, ref_frame, search, block as i32);
                    list.into_iter()
                        .find(|&mv| mv != sub(0))
                        .unwrap_or(Mv::ZERO)
                }
            }
            _ => {
                if b_mode == NEARESTMV {
                    sub(2)
                } else if sub(2) != sub(1) {
                    sub(1)
                } else if sub(2) != sub(0) {
                    sub(0)
                } else {
                    let (list, _) = self.find_mv_refs(ctx, b_mode, ref_frame, search, block as i32);
                    list.into_iter()
                        .find(|&mv| mv != sub(2))
                        .unwrap_or(Mv::ZERO)
                }
            }
        }
    }

    /// `assign_mv`.
    #[allow(clippy::too_many_arguments)]
    fn assign_mv(
        &mut self,
        tile: &mut Tile,
        mode: u8,
        mvs: &mut [Mv; 2],
        ref_mvs: &[Mv; 2],
        near_nearest: &[Mv; 2],
        is_compound: bool,
        allow_hp: bool,
    ) -> bool {
        match mode {
            NEWMV => {
                let mut valid = true;
                for i in 0..1 + usize::from(is_compound) {
                    mvs[i] = self.read_mv(tile, ref_mvs[i], allow_hp);
                    valid &= is_mv_valid(mvs[i]);
                }
                valid
            }
            NEARMV | NEARESTMV => {
                *mvs = *near_nearest;
                true
            }
            ZEROMV => {
                *mvs = [Mv::ZERO; 2];
                true
            }
            _ => false,
        }
    }

    /// `read_mv`.
    fn read_mv(&mut self, tile: &mut Tile, reference: Mv, allow_hp: bool) -> Mv {
        let joint = tile.bd.tree(&MV_JOINT_TREE, &self.fc.mv_joints);
        let use_hp = allow_hp && use_mv_hp(reference);
        let mut diff = [0i32; 2];
        if joint == 2 || joint == 3 {
            diff[0] = self.read_mv_component(tile, 0, use_hp);
        }
        if joint == 1 || joint == 3 {
            diff[1] = self.read_mv_component(tile, 1, use_hp);
        }
        if let Some(counts) = self.count() {
            // vp9_inc_mv
            counts.mv_joints[usize::from(joint)] += 1;
            for (component, &value) in diff.iter().enumerate() {
                if value != 0 {
                    increment_mv_component(&mut counts.mv[component], value);
                }
            }
        }
        Mv {
            row: (i32::from(reference.row) + diff[0]) as i16,
            col: (i32::from(reference.col) + diff[1]) as i16,
        }
    }

    /// `read_mv_component`.
    fn read_mv_component(&mut self, tile: &mut Tile, component: usize, use_hp: bool) -> i32 {
        let probs = &self.fc.mv[component];
        let bd = &mut tile.bd;
        let sign = bd.read(probs.sign);
        let mv_class = bd.tree(&MV_CLASS_TREE, &probs.classes);
        let class0 = mv_class == 0;
        let (d, mut magnitude) = if class0 {
            (u32::from(bd.read(probs.class0[0])), 0)
        } else {
            let bits = u32::from(mv_class);
            let mut d = 0;
            for i in 0..bits {
                d |= u32::from(bd.read(probs.bits[i as usize])) << i;
            }
            (d, 2 << (u32::from(mv_class) + 2))
        };
        let fr = u32::from(bd.tree(
            &MV_FP_TREE,
            if class0 {
                &probs.class0_fp[d as usize]
            } else {
                &probs.fp
            },
        ));
        let hp = if use_hp {
            u32::from(bd.read(if class0 { probs.class0_hp } else { probs.hp }))
        } else {
            1
        };
        magnitude += ((d << 3) | (fr << 1) | hp) + 1;
        if sign {
            -(magnitude as i32)
        } else {
            magnitude as i32
        }
    }

    // --- Residual (vp9_detokenize.c) -------------------------------------

    /// `vp9_decode_block_tokens`: decodes one transform block's
    /// coefficients into `self.coefficients`, returning the end of block.
    #[allow(clippy::too_many_arguments)]
    fn decode_block_tokens(
        &mut self,
        tile: &mut Tile,
        ctx: &BlockContext,
        mi: &ModeInfo,
        plane: usize,
        scan: Scan,
        col: usize,
        row: usize,
        tx_size: u8,
    ) -> usize {
        let ss = usize::from(plane > 0);
        let above_index = ((ctx.mi_col * 2) >> ss) + col;
        let left_index = (((ctx.mi_row * 2) & 15) >> ss) + row;
        let n = 1usize << tx_size;
        let above_any = self.above_context[plane][above_index..above_index + n]
            .iter()
            .any(|&v| v != 0);
        let left_any = tile.left_context[plane][left_index..left_index + n]
            .iter()
            .any(|&v| v != 0);
        let context = usize::from(above_any) + usize::from(left_any);
        let dequant = if plane == 0 {
            self.y_dequant[usize::from(mi.segment_id)]
        } else {
            self.uv_dequant[usize::from(mi.segment_id)]
        };
        let eob = self.decode_coefs(
            tile,
            plane > 0,
            mi.is_inter(),
            tx_size,
            dequant,
            context,
            scan,
        );

        // The context entries past the frame edge are cleared
        // (`get_ctx_shift`).
        let (max_wide, max_high) = max_blocks(ctx, ss);
        let flag = u8::from(eob > 0);
        for i in 0..n {
            let inside_wide = ctx.mb_to_right_edge >= 0 || col + i < max_wide;
            self.above_context[plane][above_index + i] = if inside_wide { flag } else { 0 };
            let inside_high = ctx.mb_to_bottom_edge >= 0 || row + i < max_high;
            tile.left_context[plane][left_index + i] = if inside_high { flag } else { 0 };
        }
        eob
    }

    /// `decode_coefs`.
    #[allow(clippy::too_many_arguments)]
    fn decode_coefs(
        &mut self,
        tile: &mut Tile,
        uv: bool,
        inter: bool,
        tx_size: u8,
        dequant: [i32; 2],
        mut context: usize,
        scan: Scan,
    ) -> usize {
        let plane_type = usize::from(uv);
        let reference = usize::from(inter);
        let tx = usize::from(tx_size);
        let max_eob = 16usize << (tx << 1);
        let band_translate: &[u8] = if tx_size == 0 {
            &COEFBAND_TRANS_4X4
        } else {
            &COEFBAND_TRANS_8X8PLUS
        };
        let dq_shift = u32::from(tx_size == 3);
        let probs = &self.fc.coef[tx][plane_type][reference];
        let mut counts = self.counts.as_deref_mut();
        let token_cache = &mut self.token_cache[..];
        let bd = &mut tile.bd;
        let mut dqv = dequant[0];
        let mut c = 0usize;
        while c < max_eob {
            let mut band = usize::from(band_translate[c]);
            let mut prob = &probs[band][context];
            if let Some(counts) = counts.as_deref_mut() {
                counts.eob_branch[tx][plane_type][reference][band][context] += 1;
            }
            if !bd.read(prob[0]) {
                if let Some(counts) = counts.as_deref_mut() {
                    counts.coef[tx][plane_type][reference][band][context][3] += 1;
                }
                break;
            }
            while !bd.read(prob[1]) {
                if let Some(counts) = counts.as_deref_mut() {
                    counts.coef[tx][plane_type][reference][band][context][0] += 1;
                }
                dqv = dequant[1];
                token_cache[scan.scan[c] as usize] = 0;
                c += 1;
                if c >= max_eob {
                    return c;
                }
                context = coef_context(scan.neighbors, token_cache, c);
                band = usize::from(band_translate[c]);
                prob = &probs[band][context];
            }
            let value = if bd.read(prob[1 + 1]) {
                let p = &PARETO8_FULL[usize::from(prob[2]) - 1];
                if let Some(counts) = counts.as_deref_mut() {
                    counts.coef[tx][plane_type][reference][band][context][2] += 1;
                }
                if bd.read(p[0]) {
                    let val = if bd.read(p[3]) {
                        token_cache[scan.scan[c] as usize] = 5;
                        if bd.read(p[5]) {
                            if bd.read(p[7]) {
                                67 + read_coeff(bd, &CAT6_PROB)
                            } else {
                                35 + read_coeff(bd, &[180, 157, 141, 134, 130])
                            }
                        } else if bd.read(p[6]) {
                            19 + read_coeff(bd, &[176, 155, 140, 135])
                        } else {
                            11 + read_coeff(bd, &[173, 148, 140])
                        }
                    } else {
                        token_cache[scan.scan[c] as usize] = 4;
                        if bd.read(p[4]) {
                            7 + read_coeff(bd, &[165, 145])
                        } else {
                            5 + read_coeff(bd, &[159])
                        }
                    };
                    (val * dqv) >> dq_shift
                } else if bd.read(p[1]) {
                    token_cache[scan.scan[c] as usize] = 3;
                    ((3 + i32::from(bd.read(p[2]))) * dqv) >> dq_shift
                } else {
                    token_cache[scan.scan[c] as usize] = 2;
                    (2 * dqv) >> dq_shift
                }
            } else {
                if let Some(counts) = counts.as_deref_mut() {
                    counts.coef[tx][plane_type][reference][band][context][1] += 1;
                }
                token_cache[scan.scan[c] as usize] = 1;
                dqv >> dq_shift
            };
            let position = scan.scan[c] as usize;
            self.coefficients[position] = if bd.read(128) { -value } else { value };
            c += 1;
            context = coef_context(scan.neighbors, token_cache, c);
            dqv = dequant[1];
        }
        c
    }

    /// Clears the coefficients a block's tokens set.
    fn clear_coefficients(&mut self, scan: Scan, eob: usize) {
        for &position in &scan.scan[..eob] {
            self.coefficients[position as usize] = 0;
        }
    }

    // --- Reconstruction (vp9_decodeframe.c) ------------------------------

    /// `predict_and_reconstruct_intra_block`.
    #[allow(clippy::too_many_arguments)]
    fn predict_and_reconstruct_intra(
        &mut self,
        tile: &mut Tile,
        ctx: &BlockContext,
        mi: &ModeInfo,
        plane: usize,
        row: usize,
        col: usize,
        tx_size: u8,
    ) -> Result<()> {
        let ss = usize::from(plane > 0);
        let mut mode = if plane == 0 { mi.mode } else { mi.uv_mode };
        if mi.sb_type < BLOCK_8X8 && plane == 0 {
            mode = mi.sub_modes[(row << 1) + col];
        }
        let x = ((ctx.mi_col * 8) >> ss) + 4 * col;
        let y = ((ctx.mi_row * 8) >> ss) + 4 * row;
        let bs = 4usize << tx_size;
        // vp9_predict_intra_block
        let n4_wl = ctx.bwl - ss;
        let have_top = row > 0 || ctx.above.is_some();
        let have_left = col > 0 || ctx.left.is_some();
        let have_right = col + (1 << tx_size) < (1 << n4_wl);
        let plane_buffer = &mut self.frame.planes[plane];
        let edges = build_intra_edges(
            plane_buffer,
            x,
            y,
            bs,
            mode,
            have_top,
            have_left,
            have_right,
        );
        let stride = plane_buffer.stride;
        let dest_start = plane_buffer.index(x, y);
        recon::predict_intra(
            &mut plane_buffer.data[dest_start..],
            stride,
            bs,
            mode,
            &edges,
            have_left,
            have_top,
        );

        if !mi.skip {
            let lossless = self.header.lossless;
            let tx_type = if plane > 0 || lossless {
                recon::DCT_DCT
            } else {
                INTRA_MODE_TO_TX_TYPE[usize::from(mode)]
            };
            let scan = scan_order(tx_size, tx_type);
            let eob = self.decode_block_tokens(tile, ctx, mi, plane, scan, col, row, tx_size);
            if eob > 0 {
                let plane_buffer = &mut self.frame.planes[plane];
                recon::inverse_transform_add(
                    &self.coefficients,
                    &mut plane_buffer.data[dest_start..],
                    stride,
                    tx_size,
                    tx_type,
                    eob,
                    lossless,
                );
                self.clear_coefficients(scan, eob);
            }
        }
        Ok(())
    }

    /// `reconstruct_inter_block`: decodes and adds one transform block's
    /// residual, returning its end of block.
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_inter(
        &mut self,
        tile: &mut Tile,
        ctx: &BlockContext,
        mi: &ModeInfo,
        plane: usize,
        row: usize,
        col: usize,
        tx_size: u8,
    ) -> usize {
        let scan = scan_order(tx_size, recon::DCT_DCT);
        let eob = self.decode_block_tokens(tile, ctx, mi, plane, scan, col, row, tx_size);
        if eob > 0 {
            let ss = usize::from(plane > 0);
            let x = ((ctx.mi_col * 8) >> ss) + 4 * col;
            let y = ((ctx.mi_row * 8) >> ss) + 4 * row;
            let plane_buffer = &mut self.frame.planes[plane];
            let stride = plane_buffer.stride;
            let start = plane_buffer.index(x, y);
            recon::inverse_transform_add(
                &self.coefficients,
                &mut plane_buffer.data[start..],
                stride,
                tx_size,
                recon::DCT_DCT,
                eob,
                self.header.lossless,
            );
            self.clear_coefficients(scan, eob);
        }
        eob
    }

    /// `dec_build_inter_predictors_sb`.
    fn predict_inter(&mut self, ctx: &BlockContext, mi: &ModeInfo) -> Result<()> {
        let kernel = &FILTER_KERNELS[usize::from(mi.interp_filter.min(3))];
        let refs = self.refs;
        for reference in 0..1 + usize::from(mi.has_second_ref()) {
            let frame_index = mi.ref_frame[reference];
            let Some((ref_frame, scale)) = usize::try_from(frame_index - 1)
                .ok()
                .and_then(|index| refs.get(index))
                .and_then(Option::as_ref)
            else {
                return Err(malformed(
                    "VP9 block uses a reference frame the frame does not have",
                ));
            };
            if !scale.is_valid() {
                return Err(malformed("VP9 reference frame has invalid dimensions"));
            }
            let is_scaled = scale.is_scaled();
            for plane in 0..3 {
                let ss = usize::from(plane > 0);
                let n4w = (ctx.bw << 1) >> ss;
                let n4h = (ctx.bh << 1) >> ss;
                let (bw, bh) = (4 * n4w, 4 * n4h);
                if ctx.sb_type < BLOCK_8X8 {
                    let mut block = 0;
                    for y in 0..n4h {
                        for x in 0..n4w {
                            let mv = average_split_mvs(plane, mi, reference, block);
                            block += 1;
                            self.build_inter_predictor(
                                ctx,
                                plane,
                                bw,
                                bh,
                                4 * x,
                                4 * y,
                                4,
                                4,
                                kernel,
                                scale,
                                mv,
                                ref_frame,
                                is_scaled,
                                reference,
                            );
                        }
                    }
                } else {
                    self.build_inter_predictor(
                        ctx,
                        plane,
                        bw,
                        bh,
                        0,
                        0,
                        bw,
                        bh,
                        kernel,
                        scale,
                        mi.mv[reference],
                        ref_frame,
                        is_scaled,
                        reference,
                    );
                }
            }
        }
        Ok(())
    }

    /// `dec_build_inter_predictors`.
    #[allow(clippy::too_many_arguments)]
    fn build_inter_predictor(
        &mut self,
        ctx: &BlockContext,
        plane: usize,
        bw: usize,
        bh: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        kernel: &recon::Kernel,
        scale: &Scale,
        mv: Mv,
        ref_frame: &Frame,
        is_scaled: bool,
        reference: usize,
    ) {
        let ss = i32::from(plane > 0);
        let reference_plane = &ref_frame.planes[plane];
        let frame_width = reference_plane.crop_width as i32;
        let frame_height = reference_plane.crop_height as i32;
        let mi_x = (ctx.mi_col * 8) as i32;
        let mi_y = (ctx.mi_row * 8) as i32;
        let (x, y, w, h) = (x as i32, y as i32, w as i32, h as i32);

        let (mut x0, mut y0, mut x0_16, mut y0_16, scaled_row, scaled_col, xs, ys);
        if is_scaled {
            let mv_q4 = clamp_mv_to_umv_border(ctx, mv, bw as i32, bh as i32, ss);
            let x_start = (-ctx.mb_to_left_edge) >> (3 + ss);
            let y_start = (-ctx.mb_to_top_edge) >> (3 + ss);
            x0_16 = scale.scale_x((x_start + x) << 4);
            y0_16 = scale.scale_y((y_start + y) << 4);
            x0 = scale.scale_x(x_start + x);
            y0 = scale.scale_y(y_start + y);
            // vp9_scale_mv
            let x_off_q4 = scale.scale_x((mi_x + x) << 4) & 15;
            let y_off_q4 = scale.scale_y((mi_y + y) << 4) & 15;
            scaled_row = scale.scale_y(mv_q4.0) + y_off_q4;
            scaled_col = scale.scale_x(mv_q4.1) + x_off_q4;
            xs = scale.x_step_q4;
            ys = scale.y_step_q4;
        } else {
            x0 = ((-ctx.mb_to_left_edge) >> (3 + ss)) + x;
            y0 = ((-ctx.mb_to_top_edge) >> (3 + ss)) + y;
            x0_16 = x0 << 4;
            y0_16 = y0 << 4;
            scaled_row = i32::from(mv.row) * (1 << (1 - ss));
            scaled_col = i32::from(mv.col) * (1 << (1 - ss));
            xs = 16;
            ys = 16;
        }
        let subpel_x = scaled_col & 15;
        let subpel_y = scaled_row & 15;
        x0 += scaled_col >> 4;
        y0 += scaled_row >> 4;
        x0_16 += scaled_col;
        y0_16 += scaled_row;

        let dest_plane_stride = self.frame.planes[plane].stride;
        let dest_start = self.frame.planes[plane].index(
            ((ctx.mi_col * 8) >> ss) + x as usize,
            ((ctx.mi_row * 8) >> ss) + y as usize,
        );
        let average = reference == 1;

        if is_scaled
            || scaled_col != 0
            || scaled_row != 0
            || frame_width & 7 != 0
            || frame_height & 7 != 0
        {
            let mut y1 = ((y0_16 + (h - 1) * ys) >> 4) + 1;
            let mut x1 = ((x0_16 + (w - 1) * xs) >> 4) + 1;
            let mut fx0 = x0;
            let mut fy0 = y0;
            let mut x_pad = 0;
            let mut y_pad = 0;
            if subpel_x != 0 || scale.x_step_q4 != 16 {
                fx0 -= 3;
                x1 += 4;
                x_pad = 1;
            }
            if subpel_y != 0 || scale.y_step_q4 != 16 {
                fy0 -= 3;
                y1 += 4;
                y_pad = 1;
            }
            if fx0 < 0
                || fx0 > frame_width - 1
                || x1 < 0
                || x1 > frame_width - 1
                || fy0 < 0
                || fy0 > frame_height - 1
                || y1 < 0
                || y1 > frame_height - 1
            {
                // extend_and_predict: copy the referenced area with its
                // edges replicated, keeping three columns and rows of
                // margin plus the filter's reach on every side.
                let b_w = (x1 - fx0 + 1) as usize;
                let b_h = (y1 - fy0 + 1) as usize;
                let window_w = b_w + 16;
                let window_h = b_h + 16;
                self.mc_buffer.resize(window_w * window_h, 0);
                for wy in 0..window_h {
                    let sy = (fy0 + wy as i32 - 8).clamp(0, frame_height - 1) as usize;
                    let row_start = reference_plane.index(0, sy);
                    for wx in 0..window_w {
                        let sx = (fx0 + wx as i32 - 8).clamp(0, frame_width - 1) as usize;
                        self.mc_buffer[wy * window_w + wx] = reference_plane.data[row_start + sx];
                    }
                }
                let origin = (8 + 3 * y_pad) * window_w + 8 + 3 * x_pad;
                let dest = &mut self.frame.planes[plane].data[dest_start..];
                recon::convolve(
                    &self.mc_buffer,
                    origin,
                    window_w,
                    dest,
                    dest_plane_stride,
                    w as usize,
                    h as usize,
                    kernel,
                    subpel_x,
                    xs,
                    subpel_y,
                    ys,
                    average,
                    &mut self.convolve_temp,
                );
                return;
            }
        }
        let origin = (reference_plane.origin as isize
            + y0 as isize * reference_plane.stride as isize
            + x0 as isize) as usize;
        let dest = &mut self.frame.planes[plane].data[dest_start..];
        recon::convolve(
            &reference_plane.data,
            origin,
            reference_plane.stride,
            dest,
            dest_plane_stride,
            w as usize,
            h as usize,
            kernel,
            subpel_x,
            xs,
            subpel_y,
            ys,
            average,
            &mut self.convolve_temp,
        );
    }
}

/// `max_blocks_wide`/`max_blocks_high` of a plane, in 4x4 units: the part
/// of the block inside the frame.
fn max_blocks(ctx: &BlockContext, ss: usize) -> (usize, usize) {
    let n4_w = ((ctx.bw << 1) >> ss) as i32;
    let n4_h = ((ctx.bh << 1) >> ss) as i32;
    let wide = n4_w
        + if ctx.mb_to_right_edge >= 0 {
            0
        } else {
            ctx.mb_to_right_edge >> (5 + ss)
        };
    let high = n4_h
        + if ctx.mb_to_bottom_edge >= 0 {
            0
        } else {
            ctx.mb_to_bottom_edge >> (5 + ss)
        };
    (wide.max(0) as usize, high.max(0) as usize)
}

/// `get_tile_offset`.
fn tile_offset(index: usize, mis: usize, log2: u32) -> usize {
    let sb_cols = (mis + 7) >> 3;
    let offset = ((index * sb_cols) >> log2) << 3;
    offset.min(mis)
}

/// A scan order and its neighbour table (`ScanOrder`).
#[derive(Clone, Copy)]
struct Scan {
    scan: &'static [i16],
    neighbors: &'static [i16],
}

/// `vp9_scan_orders[tx_size][tx_type]`.
fn scan_order(tx_size: u8, tx_type: u8) -> Scan {
    let (scan, neighbors): (&'static [i16], &'static [i16]) = match (tx_size, tx_type) {
        (0, 1) => (&ROW_SCAN_4X4, &ROW_SCAN_4X4_NEIGHBORS),
        (0, 2) => (&COL_SCAN_4X4, &COL_SCAN_4X4_NEIGHBORS),
        (0, _) => (&DEFAULT_SCAN_4X4, &DEFAULT_SCAN_4X4_NEIGHBORS),
        (1, 1) => (&ROW_SCAN_8X8, &ROW_SCAN_8X8_NEIGHBORS),
        (1, 2) => (&COL_SCAN_8X8, &COL_SCAN_8X8_NEIGHBORS),
        (1, _) => (&DEFAULT_SCAN_8X8, &DEFAULT_SCAN_8X8_NEIGHBORS),
        (2, 1) => (&ROW_SCAN_16X16, &ROW_SCAN_16X16_NEIGHBORS),
        (2, 2) => (&COL_SCAN_16X16, &COL_SCAN_16X16_NEIGHBORS),
        (2, _) => (&DEFAULT_SCAN_16X16, &DEFAULT_SCAN_16X16_NEIGHBORS),
        _ => (&DEFAULT_SCAN_32X32, &DEFAULT_SCAN_32X32_NEIGHBORS),
    };
    Scan { scan, neighbors }
}

/// `get_coef_context`.
#[inline]
fn coef_context(neighbors: &[i16], token_cache: &[u8], c: usize) -> usize {
    (1 + usize::from(token_cache[neighbors[2 * c] as usize])
        + usize::from(token_cache[neighbors[2 * c + 1] as usize]))
        >> 1
}

/// `read_coeff`: the extra bits of a category token, most significant
/// first.
fn read_coeff(bd: &mut BoolDecoder, probs: &[u8]) -> i32 {
    let mut value = 0;
    for &probability in probs {
        value = (value << 1) | i32::from(bd.read(probability));
    }
    value
}

/// `intra_mode_to_tx_type_lookup`.
const INTRA_MODE_TO_TX_TYPE: [u8; 10] = [0, 1, 2, 0, 3, 1, 2, 2, 1, 3];

/// `partition_context_lookup`: (above, left).
const PARTITION_CONTEXT_LOOKUP: [(u8, u8); 13] = [
    (15, 15),
    (15, 14),
    (14, 15),
    (14, 14),
    (14, 12),
    (12, 14),
    (12, 12),
    (12, 8),
    (8, 12),
    (8, 8),
    (8, 0),
    (0, 8),
    (0, 0),
];

/// `mv_ref_blocks`: (row, col) offsets of the candidate blocks.
const MV_REF_BLOCKS: [[(i32, i32); 8]; 13] = {
    const SMALL: [(i32, i32); 8] = [
        (-1, 0),
        (0, -1),
        (-1, -1),
        (-2, 0),
        (0, -2),
        (-2, -1),
        (-1, -2),
        (-2, -2),
    ];
    [
        SMALL,
        SMALL,
        SMALL,
        SMALL,
        [
            (0, -1),
            (-1, 0),
            (1, -1),
            (-1, -1),
            (0, -2),
            (-2, 0),
            (-2, -1),
            (-1, -2),
        ],
        [
            (-1, 0),
            (0, -1),
            (-1, 1),
            (-1, -1),
            (-2, 0),
            (0, -2),
            (-1, -2),
            (-2, -1),
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
            (0, -1),
            (-1, 0),
            (2, -1),
            (-1, -1),
            (-1, 1),
            (0, -3),
            (-3, 0),
            (-3, -3),
        ],
        [
            (-1, 0),
            (0, -1),
            (-1, 2),
            (-1, -1),
            (1, -1),
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
            (0, -1),
            (-1, 0),
            (4, -1),
            (-1, 2),
            (-1, -1),
            (0, -3),
            (-3, 0),
            (2, -1),
        ],
        [
            (-1, 0),
            (0, -1),
            (-1, 4),
            (2, -1),
            (-1, -1),
            (-3, 0),
            (0, -3),
            (-1, 2),
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
    ]
};

/// `mode_2_counter`.
const MODE_2_COUNTER: [usize; 14] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 3, 1];
/// `counter_to_context` (9 marks impossible counts).
const COUNTER_TO_CONTEXT: [usize; 19] = [2, 3, 4, 1, 3, 9, 0, 9, 9, 5, 5, 9, 5, 9, 9, 9, 9, 9, 6];

const MV_BORDER: i32 = 16 << 3;

fn clamp_mv(mv: &mut Mv, min_col: i32, max_col: i32, min_row: i32, max_row: i32) {
    mv.col = i32::from(mv.col).clamp(min_col, max_col) as i16;
    mv.row = i32::from(mv.row).clamp(min_row, max_row) as i16;
}

/// `clamp_mv_to_umv_border_sb`, returning the vector in 1/16 pixel units
/// of the plane as (row, col).
fn clamp_mv_to_umv_border(ctx: &BlockContext, mv: Mv, bw: i32, bh: i32, ss: i32) -> (i32, i32) {
    let spel_left = (4 + bw) << 4;
    let spel_right = spel_left - 16;
    let spel_top = (4 + bh) << 4;
    let spel_bottom = spel_top - 16;
    let row = i32::from((i32::from(mv.row) * (1 << (1 - ss))) as i16);
    let col = i32::from((i32::from(mv.col) * (1 << (1 - ss))) as i16);
    let factor = 1 << (1 - ss);
    (
        row.clamp(
            ctx.mb_to_top_edge * factor - spel_top,
            ctx.mb_to_bottom_edge * factor + spel_bottom,
        ),
        col.clamp(
            ctx.mb_to_left_edge * factor - spel_left,
            ctx.mb_to_right_edge * factor + spel_right,
        ),
    )
}

/// `average_split_mvs` for 4:2:0: chroma averages the four luma vectors.
fn average_split_mvs(plane: usize, mi: &ModeInfo, reference: usize, block: usize) -> Mv {
    if plane == 0 {
        return mi.sub_mvs[block][reference];
    }
    let round_q4 = |value: i32| (if value < 0 { value - 2 } else { value + 2 }) / 4;
    let sum =
        |f: fn(&Mv) -> i16| -> i32 { mi.sub_mvs.iter().map(|m| i32::from(f(&m[reference]))).sum() };
    Mv {
        row: round_q4(sum(|m| m.row)) as i16,
        col: round_q4(sum(|m| m.col)) as i16,
    }
}

/// `get_sub_block_mv`.
fn sub_block_mv(candidate: &ModeInfo, which: usize, search_col: i32, block: i32) -> Mv {
    const IDX_N_COLUMN_TO_SUBBLOCK: [[usize; 2]; 4] = [[1, 2], [1, 3], [3, 2], [3, 3]];
    if block >= 0 && candidate.sb_type < BLOCK_8X8 {
        let index = IDX_N_COLUMN_TO_SUBBLOCK[block as usize][usize::from(search_col == 0)];
        candidate.sub_mvs[index][which]
    } else {
        candidate.mv[which]
    }
}

fn use_mv_hp(mv: Mv) -> bool {
    i32::from(mv.row).abs() < 64 && i32::from(mv.col).abs() < 64
}

fn lower_mv_precision(mv: &mut Mv, allow_hp: bool) {
    if !(allow_hp && use_mv_hp(*mv)) {
        if mv.row & 1 != 0 {
            mv.row += if mv.row > 0 { -1 } else { 1 };
        }
        if mv.col & 1 != 0 {
            mv.col += if mv.col > 0 { -1 } else { 1 };
        }
    }
}

fn is_mv_valid(mv: Mv) -> bool {
    const LOW: i32 = -(1 << 14);
    const HIGH: i32 = (1 << 14) - 1;
    let (row, col) = (i32::from(mv.row), i32::from(mv.col));
    row > LOW && row < HIGH && col > LOW && col < HIGH
}

/// `inc_mv_component`.
fn increment_mv_component(counts: &mut MvComponentCounts, value: i32) {
    let sign = value < 0;
    counts.sign[usize::from(sign)] += 1;
    let z = value.unsigned_abs() as i32 - 1;
    let class = if z >= 2 * 4096 {
        10
    } else {
        log_in_base_2((z >> 3) as u32)
    };
    counts.classes[class] += 1;
    let base = if class == 0 { 0 } else { 2 << (class + 2) };
    let offset = z - base;
    let d = offset >> 3;
    let f = ((offset >> 1) & 3) as usize;
    let e = (offset & 1) as usize;
    if class == 0 {
        counts.class0[d as usize] += 1;
        counts.class0_fp[d as usize][f] += 1;
        counts.class0_hp[e] += 1;
    } else {
        for i in 0..class {
            counts.bits[i][((d >> i) & 1) as usize] += 1;
        }
        counts.fp[f] += 1;
        counts.hp[e] += 1;
    }
}

/// `log_in_base_2` for the motion vector classes: floor(log2(n)) capped
/// at 10, with 0 for 0.
fn log_in_base_2(n: u32) -> usize {
    if n == 0 {
        0
    } else {
        (31 - n.leading_zeros()).min(10) as usize
    }
}

/// `vp9_above_block_mode`.
fn above_block_mode(mi: &ModeInfo, above: Option<&ModeInfo>, block: usize) -> u8 {
    if block == 0 || block == 1 {
        match above {
            Some(above) if !above.is_inter() => above.y_mode(block + 2),
            _ => 0,
        }
    } else {
        mi.sub_modes[block - 2]
    }
}

/// `vp9_left_block_mode`.
fn left_block_mode(mi: &ModeInfo, left: Option<&ModeInfo>, block: usize) -> u8 {
    if block == 0 || block == 2 {
        match left {
            Some(left) if !left.is_inter() => left.y_mode(block + 1),
            _ => 0,
        }
    } else {
        mi.sub_modes[block - 1]
    }
}

/// `vp9_get_pred_context_single_ref_p1`.
fn single_ref_p1_context(ctx: &BlockContext) -> usize {
    match (ctx.above, ctx.left) {
        (Some(above), Some(left)) => {
            let above_intra = !above.is_inter();
            let left_intra = !left.is_inter();
            if above_intra && left_intra {
                2
            } else if above_intra || left_intra {
                let edge = if above_intra { left } else { above };
                if !edge.has_second_ref() {
                    4 * usize::from(edge.ref_frame[0] == LAST_FRAME)
                } else {
                    1 + usize::from(
                        edge.ref_frame[0] == LAST_FRAME || edge.ref_frame[1] == LAST_FRAME,
                    )
                }
            } else {
                let above_has_second = above.has_second_ref();
                let left_has_second = left.has_second_ref();
                let [above0, above1] = above.ref_frame;
                let [left0, left1] = left.ref_frame;
                if above_has_second && left_has_second {
                    1 + usize::from(
                        above0 == LAST_FRAME
                            || above1 == LAST_FRAME
                            || left0 == LAST_FRAME
                            || left1 == LAST_FRAME,
                    )
                } else if above_has_second || left_has_second {
                    let rfs = if !above_has_second { above0 } else { left0 };
                    let crf1 = if above_has_second { above0 } else { left0 };
                    let crf2 = if above_has_second { above1 } else { left1 };
                    if rfs == LAST_FRAME {
                        3 + usize::from(crf1 == LAST_FRAME || crf2 == LAST_FRAME)
                    } else {
                        usize::from(crf1 == LAST_FRAME || crf2 == LAST_FRAME)
                    }
                } else {
                    2 * usize::from(above0 == LAST_FRAME) + 2 * usize::from(left0 == LAST_FRAME)
                }
            }
        }
        (Some(edge), None) | (None, Some(edge)) => {
            if !edge.is_inter() {
                2
            } else if !edge.has_second_ref() {
                4 * usize::from(edge.ref_frame[0] == LAST_FRAME)
            } else {
                1 + usize::from(edge.ref_frame[0] == LAST_FRAME || edge.ref_frame[1] == LAST_FRAME)
            }
        }
        (None, None) => 2,
    }
}

/// `vp9_get_pred_context_single_ref_p2`.
fn single_ref_p2_context(ctx: &BlockContext) -> usize {
    match (ctx.above, ctx.left) {
        (Some(above), Some(left)) => {
            let above_intra = !above.is_inter();
            let left_intra = !left.is_inter();
            if above_intra && left_intra {
                2
            } else if above_intra || left_intra {
                let edge = if above_intra { left } else { above };
                if !edge.has_second_ref() {
                    if edge.ref_frame[0] == LAST_FRAME {
                        3
                    } else {
                        4 * usize::from(edge.ref_frame[0] == GOLDEN_FRAME)
                    }
                } else {
                    1 + 2 * usize::from(
                        edge.ref_frame[0] == GOLDEN_FRAME || edge.ref_frame[1] == GOLDEN_FRAME,
                    )
                }
            } else {
                let above_has_second = above.has_second_ref();
                let left_has_second = left.has_second_ref();
                let [above0, above1] = above.ref_frame;
                let [left0, left1] = left.ref_frame;
                if above_has_second && left_has_second {
                    if above0 == left0 && above1 == left1 {
                        3 * usize::from(
                            above0 == GOLDEN_FRAME
                                || above1 == GOLDEN_FRAME
                                || left0 == GOLDEN_FRAME
                                || left1 == GOLDEN_FRAME,
                        )
                    } else {
                        2
                    }
                } else if above_has_second || left_has_second {
                    let rfs = if !above_has_second { above0 } else { left0 };
                    let crf1 = if above_has_second { above0 } else { left0 };
                    let crf2 = if above_has_second { above1 } else { left1 };
                    if rfs == GOLDEN_FRAME {
                        3 + usize::from(crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME)
                    } else if rfs == ALTREF_FRAME {
                        usize::from(crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME)
                    } else {
                        1 + 2 * usize::from(crf1 == GOLDEN_FRAME || crf2 == GOLDEN_FRAME)
                    }
                } else if above0 == LAST_FRAME && left0 == LAST_FRAME {
                    3
                } else if above0 == LAST_FRAME || left0 == LAST_FRAME {
                    let edge0 = if above0 == LAST_FRAME { left0 } else { above0 };
                    4 * usize::from(edge0 == GOLDEN_FRAME)
                } else {
                    2 * usize::from(above0 == GOLDEN_FRAME) + 2 * usize::from(left0 == GOLDEN_FRAME)
                }
            }
        }
        (Some(edge), None) | (None, Some(edge)) => {
            if !edge.is_inter() || (edge.ref_frame[0] == LAST_FRAME && !edge.has_second_ref()) {
                2
            } else if !edge.has_second_ref() {
                4 * usize::from(edge.ref_frame[0] == GOLDEN_FRAME)
            } else {
                3 * usize::from(
                    edge.ref_frame[0] == GOLDEN_FRAME || edge.ref_frame[1] == GOLDEN_FRAME,
                )
            }
        }
        (None, None) => 2,
    }
}

const NEED_LEFT: u8 = 1 << 1;
const NEED_ABOVE: u8 = 1 << 2;
const NEED_ABOVERIGHT: u8 = 1 << 3;

/// `extend_modes`.
const EXTEND_MODES: [u8; 10] = [
    NEED_ABOVE | NEED_LEFT,
    NEED_ABOVE,
    NEED_LEFT,
    NEED_ABOVERIGHT,
    NEED_LEFT | NEED_ABOVE,
    NEED_LEFT | NEED_ABOVE,
    NEED_LEFT | NEED_ABOVE,
    NEED_LEFT,
    NEED_ABOVERIGHT,
    NEED_LEFT | NEED_ABOVE,
];

/// The edge gathering of `build_intra_predictors`: pixels past the frame's
/// 8-aligned edge repeat the last one inside it, and missing edges are 127
/// (above) or 129 (left).
#[allow(clippy::too_many_arguments)]
fn build_intra_edges(
    plane: &super::Plane,
    x0: usize,
    y0: usize,
    bs: usize,
    mode: u8,
    have_top: bool,
    have_left: bool,
    have_right: bool,
) -> IntraEdges {
    let mut edges = IntraEdges {
        above: [0; 65],
        left: [0; 32],
    };
    let extend = EXTEND_MODES[usize::from(mode)];
    if extend & NEED_LEFT != 0 {
        if have_left {
            let available = bs.min(plane.height.saturating_sub(y0)).max(1);
            for i in 0..bs {
                let row = y0 + i.min(available - 1);
                edges.left[i] = plane.data[plane.index(x0, row) - 1];
            }
        } else {
            edges.left[..bs].fill(129);
        }
    }
    let above_count = if extend & NEED_ABOVERIGHT != 0 {
        Some(if have_right && bs == 4 { 2 * bs } else { bs })
    } else if extend & NEED_ABOVE != 0 {
        Some(bs)
    } else {
        None
    };
    if let Some(available) = above_count {
        let total = if extend & NEED_ABOVERIGHT != 0 {
            2 * bs
        } else {
            bs
        };
        if have_top {
            let row_start = plane.index(x0, y0 - 1);
            let count = available.min(plane.width.saturating_sub(x0)).max(1);
            edges.above[1..=count].copy_from_slice(&plane.data[row_start..row_start + count]);
            let last = edges.above[count];
            edges.above[count + 1..=total].fill(last);
            edges.above[0] = if have_left {
                plane.data[row_start - 1]
            } else {
                129
            };
        } else {
            edges.above[..=total].fill(127);
        }
    }
    edges
}
