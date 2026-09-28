//! Tile, partition and block mode info syntax (specification sections 5.11
//! and 6.10), together with the CDF selection of section 8.3.2.

use std::sync::Arc;

use super::RefSlot;
use super::cdf::CdfContext;
use super::consts::*;
use super::coverage;
use super::frame::FrameState;
use super::header::{FrameHeader, SequenceHeader};
use super::symbol::SymbolDecoder;
use super::tables::*;

/// All the state the specification keeps in globals while decoding a tile.
pub(crate) struct TileDecoder<'a> {
    pub(crate) seq: &'a SequenceHeader,
    pub(crate) fh: &'a FrameHeader,
    pub(crate) refs: &'a [Option<Arc<RefSlot>>; NUM_REF_FRAMES],
    pub(crate) f: &'a mut FrameState,
    pub(crate) cdf: &'a mut CdfContext,
    pub(crate) sd: SymbolDecoder<'a>,
    /// `TileIntraFrameYModeCdf`, which restarts from its default in every tile.
    pub(crate) intra_frame_y_mode_cdf: [[[u16; 14]; 5]; 5],

    pub(crate) mi_row_start: usize,
    pub(crate) mi_row_end: usize,
    pub(crate) mi_col_start: usize,
    pub(crate) mi_col_end: usize,
    pub(crate) current_q_index: i32,
    pub(crate) blocks_decoded: u64,

    // Above/left entropy contexts.
    pub(crate) above_level_context: [Vec<u8>; 3],
    pub(crate) above_dc_context: [Vec<u8>; 3],
    pub(crate) above_seg_pred_context: Vec<u8>,
    pub(crate) left_level_context: [Vec<u8>; 3],
    pub(crate) left_dc_context: [Vec<u8>; 3],
    pub(crate) left_seg_pred_context: Vec<u8>,
    pub(crate) delta_lf: [i32; FRAME_LF_COUNT],
    pub(crate) ref_sgr_xqd: [[i32; 2]; 3],
    pub(crate) ref_lr_wiener: [[[i32; 3]; 2]; 3],
    pub(crate) read_deltas: bool,
    /// `BlockDecoded[ plane ][ y ][ x ]` for y, x in -1..=32, stored at +1.
    pub(crate) block_decoded: [[[bool; 34]; 34]; 3],

    // Block state.
    pub(crate) mi_row: usize,
    pub(crate) mi_col: usize,
    pub(crate) mi_size: usize,
    pub(crate) has_chroma: bool,
    pub(crate) avail_u: bool,
    pub(crate) avail_l: bool,
    pub(crate) avail_u_chroma: bool,
    pub(crate) avail_l_chroma: bool,
    pub(crate) skip: bool,
    pub(crate) skip_mode: bool,
    pub(crate) segment_id: usize,
    pub(crate) lossless: bool,
    pub(crate) is_inter: bool,
    pub(crate) use_intrabc: bool,
    pub(crate) y_mode: u8,
    pub(crate) uv_mode: u8,
    pub(crate) angle_delta_y: i32,
    pub(crate) angle_delta_uv: i32,
    pub(crate) cfl_alpha_u: i32,
    pub(crate) cfl_alpha_v: i32,
    pub(crate) use_filter_intra: bool,
    pub(crate) filter_intra_mode: usize,
    pub(crate) palette_size_y: usize,
    pub(crate) palette_size_uv: usize,
    pub(crate) palette_colors_y: [u16; PALETTE_COLORS],
    pub(crate) palette_colors_u: [u16; PALETTE_COLORS],
    pub(crate) palette_colors_v: [u16; PALETTE_COLORS],
    pub(crate) color_map_y: Box<[[u8; 64]; 64]>,
    pub(crate) color_map_uv: Box<[[u8; 64]; 64]>,
    pub(crate) ref_frame: [i8; 2],
    pub(crate) mv: [[i32; 2]; 2],
    pub(crate) pred_mv: [[i32; 2]; 2],
    pub(crate) ref_mv_idx: usize,
    pub(crate) interintra: bool,
    pub(crate) interintra_mode: u8,
    pub(crate) wedge_interintra: bool,
    pub(crate) wedge_index: usize,
    pub(crate) wedge_sign: usize,
    pub(crate) mask_type: bool,
    pub(crate) motion_mode: u8,
    pub(crate) compound_type: u8,
    pub(crate) comp_group_idx: u8,
    pub(crate) compound_idx: u8,
    pub(crate) interp_filter: [u8; 2],
    pub(crate) tx_size: usize,
    pub(crate) max_luma_w: usize,
    pub(crate) max_luma_h: usize,
    pub(crate) left_ref_frame: [i8; 2],
    pub(crate) above_ref_frame: [i8; 2],
    pub(crate) left_intra: bool,
    pub(crate) above_intra: bool,
    pub(crate) left_single: bool,
    pub(crate) above_single: bool,

    // Motion vector prediction state.
    pub(crate) num_mv_found: usize,
    pub(crate) new_mv_count: usize,
    pub(crate) ref_stack_mv: [[[i32; 2]; 2]; MAX_REF_MV_STACK_SIZE + 1],
    pub(crate) weight_stack: [u32; MAX_REF_MV_STACK_SIZE + 1],
    pub(crate) global_mvs: [[i32; 2]; 2],
    pub(crate) drl_ctx_stack: [usize; MAX_REF_MV_STACK_SIZE + 1],
    pub(crate) new_mv_context: usize,
    pub(crate) ref_mv_context: usize,
    pub(crate) zero_mv_context: usize,
    pub(crate) found_match: bool,
    pub(crate) close_matches: usize,
    pub(crate) total_matches: usize,
    pub(crate) ref_id_count: [usize; 2],
    pub(crate) ref_diff_count: [usize; 2],
    pub(crate) ref_id_mvs: [[[i32; 2]; 2]; 2],
    pub(crate) ref_diff_mvs: [[[i32; 2]; 2]; 2],
    pub(crate) num_samples: usize,
    pub(crate) num_samples_scanned: usize,
    pub(crate) cand_list: [[i32; 4]; LEAST_SQUARES_SAMPLES_MAX],
    pub(crate) local_warp_params: [i32; 6],
    pub(crate) local_valid: bool,

    // Residual state.
    pub(crate) quant: Box<[i32; 1024]>,
    pub(crate) plane_tx_type: u8,
    pub(crate) scratch: super::predict::PredScratch,
}

#[inline(always)]
pub(crate) fn bw4(size: usize) -> usize {
    NUM_4X4_BLOCKS_WIDE[size] as usize
}

#[inline(always)]
pub(crate) fn bh4(size: usize) -> usize {
    NUM_4X4_BLOCKS_HIGH[size] as usize
}

fn neg_deinterleave(diff: i32, r: i32, max: i32) -> i32 {
    if r == 0 {
        return diff;
    }
    if r >= max - 1 {
        return max - diff - 1;
    }
    if 2 * r < max {
        if diff <= 2 * r {
            if diff & 1 != 0 {
                return r + ((diff + 1) >> 1);
            }
            return r - (diff >> 1);
        }
        diff
    } else {
        if diff <= 2 * (max - r - 1) {
            if diff & 1 != 0 {
                return r + ((diff + 1) >> 1);
            }
            return r - (diff >> 1);
        }
        max - (diff + 1)
    }
}

pub(crate) fn is_directional_mode(mode: u8) -> bool {
    (V_PRED..=D67_PRED).contains(&mode)
}

fn check_backward(ref_frame: i8) -> bool {
    (BWDREF_FRAME..=ALTREF_FRAME).contains(&ref_frame)
}

fn ref_count_ctx(counts0: usize, counts1: usize) -> usize {
    match counts0.cmp(&counts1) {
        std::cmp::Ordering::Less => 0,
        std::cmp::Ordering::Equal => 1,
        std::cmp::Ordering::Greater => 2,
    }
}

impl<'a> TileDecoder<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        seq: &'a SequenceHeader,
        fh: &'a FrameHeader,
        refs: &'a [Option<Arc<RefSlot>>; NUM_REF_FRAMES],
        f: &'a mut FrameState,
        cdf: &'a mut CdfContext,
        sd: SymbolDecoder<'a>,
        tile_row: usize,
        tile_col: usize,
    ) -> Self {
        let ti = &fh.tile_info;
        let cols = fh.mi_cols + 64;
        let rows = fh.mi_rows + 64;
        Self {
            seq,
            fh,
            refs,
            f,
            cdf,
            sd,
            intra_frame_y_mode_cdf: DEFAULT_INTRA_FRAME_Y_MODE_CDF,
            mi_row_start: ti.mi_row_starts[tile_row],
            mi_row_end: ti.mi_row_starts[tile_row + 1],
            mi_col_start: ti.mi_col_starts[tile_col],
            mi_col_end: ti.mi_col_starts[tile_col + 1],
            current_q_index: i32::from(fh.base_q_idx),
            blocks_decoded: 0,
            above_level_context: [vec![0; cols], vec![0; cols], vec![0; cols]],
            above_dc_context: [vec![0; cols], vec![0; cols], vec![0; cols]],
            above_seg_pred_context: vec![0; cols],
            left_level_context: [vec![0; rows], vec![0; rows], vec![0; rows]],
            left_dc_context: [vec![0; rows], vec![0; rows], vec![0; rows]],
            left_seg_pred_context: vec![0; rows],
            delta_lf: [0; FRAME_LF_COUNT],
            ref_sgr_xqd: [[0; 2]; 3],
            ref_lr_wiener: [[[0; 3]; 2]; 3],
            read_deltas: false,
            block_decoded: [[[false; 34]; 34]; 3],
            mi_row: 0,
            mi_col: 0,
            mi_size: 0,
            has_chroma: false,
            avail_u: false,
            avail_l: false,
            avail_u_chroma: false,
            avail_l_chroma: false,
            skip: false,
            skip_mode: false,
            segment_id: 0,
            lossless: false,
            is_inter: false,
            use_intrabc: false,
            y_mode: 0,
            uv_mode: 0,
            angle_delta_y: 0,
            angle_delta_uv: 0,
            cfl_alpha_u: 0,
            cfl_alpha_v: 0,
            use_filter_intra: false,
            filter_intra_mode: 0,
            palette_size_y: 0,
            palette_size_uv: 0,
            palette_colors_y: [0; PALETTE_COLORS],
            palette_colors_u: [0; PALETTE_COLORS],
            palette_colors_v: [0; PALETTE_COLORS],
            color_map_y: Box::new([[0; 64]; 64]),
            color_map_uv: Box::new([[0; 64]; 64]),
            ref_frame: [INTRA_FRAME, NONE],
            mv: [[0; 2]; 2],
            pred_mv: [[0; 2]; 2],
            ref_mv_idx: 0,
            interintra: false,
            interintra_mode: 0,
            wedge_interintra: false,
            wedge_index: 0,
            wedge_sign: 0,
            mask_type: false,
            motion_mode: SIMPLE,
            compound_type: COMPOUND_AVERAGE,
            comp_group_idx: 0,
            compound_idx: 0,
            interp_filter: [0; 2],
            tx_size: 0,
            max_luma_w: 0,
            max_luma_h: 0,
            left_ref_frame: [0; 2],
            above_ref_frame: [0; 2],
            left_intra: false,
            above_intra: false,
            left_single: false,
            above_single: false,
            num_mv_found: 0,
            new_mv_count: 0,
            ref_stack_mv: [[[0; 2]; 2]; MAX_REF_MV_STACK_SIZE + 1],
            weight_stack: [0; MAX_REF_MV_STACK_SIZE + 1],
            global_mvs: [[0; 2]; 2],
            drl_ctx_stack: [0; MAX_REF_MV_STACK_SIZE + 1],
            new_mv_context: 0,
            ref_mv_context: 0,
            zero_mv_context: 0,
            found_match: false,
            close_matches: 0,
            total_matches: 0,
            ref_id_count: [0; 2],
            ref_diff_count: [0; 2],
            ref_id_mvs: [[[0; 2]; 2]; 2],
            ref_diff_mvs: [[[0; 2]; 2]; 2],
            num_samples: 0,
            num_samples_scanned: 0,
            cand_list: [[0; 4]; LEAST_SQUARES_SAMPLES_MAX],
            local_warp_params: [0; 6],
            local_valid: false,
            quant: Box::new([0; 1024]),
            plane_tx_type: 0,
            scratch: super::predict::PredScratch::new(),
        }
    }

    #[inline(always)]
    pub(crate) fn sub_x(&self, plane: usize) -> usize {
        if plane > 0 {
            usize::from(self.seq.subsampling_x)
        } else {
            0
        }
    }

    #[inline(always)]
    pub(crate) fn sub_y(&self, plane: usize) -> usize {
        if plane > 0 {
            usize::from(self.seq.subsampling_y)
        } else {
            0
        }
    }

    #[inline(always)]
    pub(crate) fn mi_idx(&self, row: usize, col: usize) -> usize {
        row * self.fh.mi_cols + col
    }

    /// `is_inside( candidateR, candidateC )`.
    #[inline(always)]
    pub(crate) fn is_inside(&self, r: isize, c: isize) -> bool {
        c >= self.mi_col_start as isize
            && c < self.mi_col_end as isize
            && r >= self.mi_row_start as isize
            && r < self.mi_row_end as isize
    }

    fn sb_size(&self) -> usize {
        if self.seq.use_128x128_superblock {
            BLOCK_128X128
        } else {
            BLOCK_64X64
        }
    }

    #[inline(always)]
    pub(crate) fn block_decoded(&self, plane: usize, y: isize, x: isize) -> bool {
        let (y, x) = (y + 1, x + 1);
        if !(0..34).contains(&y) || !(0..34).contains(&x) {
            return false;
        }
        self.block_decoded[plane][y as usize][x as usize]
    }

    /// `decode_tile( )`.
    pub(crate) fn decode_tile(&mut self) -> crate::Result<()> {
        // clear_above_context( ): the context arrays start zeroed.
        for i in 0..FRAME_LF_COUNT {
            self.delta_lf[i] = 0;
        }
        for plane in 0..self.seq.num_planes {
            for pass in 0..2 {
                self.ref_sgr_xqd[plane][pass] = i32::from(SGRPROJ_XQD_MID[pass]);
                for i in 0..3 {
                    self.ref_lr_wiener[plane][pass][i] = i32::from(WIENER_TAPS_MID[i]);
                }
            }
        }
        let sb_size = self.sb_size();
        let sb_size4 = bw4(sb_size);
        let mut r = self.mi_row_start;
        while r < self.mi_row_end {
            // clear_left_context( )
            for plane in 0..3 {
                self.left_level_context[plane].fill(0);
                self.left_dc_context[plane].fill(0);
            }
            self.left_seg_pred_context.fill(0);
            let mut c = self.mi_col_start;
            while c < self.mi_col_end {
                self.read_deltas = self.fh.delta_q_present;
                self.clear_cdef(r, c);
                self.clear_block_decoded_flags(r, c, sb_size4);
                self.read_lr(r, c, sb_size);
                self.decode_partition(r, c, sb_size)?;
                c += sb_size4;
            }
            r += sb_size4;
        }
        if self.sd.overran() {
            return Err(super::malformed(
                "AV1 tile data ends before its last symbol",
            ));
        }
        Ok(())
    }

    fn clear_block_decoded_flags(&mut self, r: usize, c: usize, sb_size4: usize) {
        for plane in 0..self.seq.num_planes {
            let sub_x = self.sub_x(plane);
            let sub_y = self.sub_y(plane);
            let sb_width4 = ((self.mi_col_end - c) >> sub_x) as isize;
            let sb_height4 = ((self.mi_row_end - r) >> sub_y) as isize;
            let decoded = &mut self.block_decoded[plane];
            for y in -1..=((sb_size4 >> sub_y) as isize) {
                for x in -1..=((sb_size4 >> sub_x) as isize) {
                    let value = if y < 0 && x < sb_width4 {
                        true
                    } else {
                        x < 0 && y < sb_height4
                    };
                    decoded[(y + 1) as usize][(x + 1) as usize] = value;
                }
            }
            decoded[(sb_size4 >> sub_y) + 1][0] = false;
        }
    }

    fn decode_partition(&mut self, r: usize, c: usize, b_size: usize) -> crate::Result<()> {
        if r >= self.fh.mi_rows || c >= self.fh.mi_cols {
            return Ok(());
        }
        self.avail_u = self.is_inside(r as isize - 1, c as isize);
        self.avail_l = self.is_inside(r as isize, c as isize - 1);
        let num4x4 = bw4(b_size);
        let half_block4x4 = num4x4 >> 1;
        let quarter_block4x4 = half_block4x4 >> 1;
        let has_rows = (r + half_block4x4) < self.fh.mi_rows;
        let has_cols = (c + half_block4x4) < self.fh.mi_cols;
        let partition = if b_size < BLOCK_8X8 {
            PARTITION_NONE
        } else if has_rows && has_cols {
            let ctx = self.partition_ctx(r, c, b_size);
            let bsl = MI_WIDTH_LOG2[b_size];
            match bsl {
                1 => self.sd.symbol(&mut self.cdf.partition_w8[ctx]),
                2 => self.sd.symbol(&mut self.cdf.partition_w16[ctx]),
                3 => self.sd.symbol(&mut self.cdf.partition_w32[ctx]),
                4 => self.sd.symbol(&mut self.cdf.partition_w64[ctx]),
                _ => self.sd.symbol(&mut self.cdf.partition_w128[ctx]),
            }
        } else if has_cols {
            let cdf = self.partition_cdf(r, c, b_size);
            let mut psum = 0u32;
            for p in [
                PARTITION_VERT,
                PARTITION_SPLIT,
                PARTITION_HORZ_A,
                PARTITION_VERT_A,
                PARTITION_VERT_B,
            ] {
                psum += cdf_prob(&cdf, p);
            }
            if b_size != BLOCK_128X128 {
                psum += cdf_prob(&cdf, PARTITION_VERT_4);
            }
            let split_or_horz = self.bool_from_psum(psum);
            if split_or_horz {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows {
            let cdf = self.partition_cdf(r, c, b_size);
            let mut psum = 0u32;
            for p in [
                PARTITION_HORZ,
                PARTITION_SPLIT,
                PARTITION_HORZ_A,
                PARTITION_HORZ_B,
                PARTITION_VERT_A,
            ] {
                psum += cdf_prob(&cdf, p);
            }
            if b_size != BLOCK_128X128 {
                psum += cdf_prob(&cdf, PARTITION_HORZ_4);
            }
            let split_or_vert = self.bool_from_psum(psum);
            if split_or_vert {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        };
        let sub_size = PARTITION_SUBSIZE[partition][b_size] as usize;
        let split_size = PARTITION_SUBSIZE[PARTITION_SPLIT][b_size] as usize;
        if sub_size >= BLOCK_SIZES {
            return Err(super::malformed(
                "AV1 partition produces an invalid block size",
            ));
        }
        let (h, q) = (half_block4x4, quarter_block4x4);
        match partition {
            PARTITION_NONE => self.decode_block(r, c, sub_size)?,
            PARTITION_HORZ => {
                self.decode_block(r, c, sub_size)?;
                if has_rows {
                    self.decode_block(r + h, c, sub_size)?;
                }
            }
            PARTITION_VERT => {
                self.decode_block(r, c, sub_size)?;
                if has_cols {
                    self.decode_block(r, c + h, sub_size)?;
                }
            }
            PARTITION_SPLIT => {
                self.decode_partition(r, c, sub_size)?;
                self.decode_partition(r, c + h, sub_size)?;
                self.decode_partition(r + h, c, sub_size)?;
                self.decode_partition(r + h, c + h, sub_size)?;
            }
            PARTITION_HORZ_A => {
                self.decode_block(r, c, split_size)?;
                self.decode_block(r, c + h, split_size)?;
                self.decode_block(r + h, c, sub_size)?;
            }
            PARTITION_HORZ_B => {
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r + h, c, split_size)?;
                self.decode_block(r + h, c + h, split_size)?;
            }
            PARTITION_VERT_A => {
                self.decode_block(r, c, split_size)?;
                self.decode_block(r + h, c, split_size)?;
                self.decode_block(r, c + h, sub_size)?;
            }
            PARTITION_VERT_B => {
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r, c + h, split_size)?;
                self.decode_block(r + h, c + h, split_size)?;
            }
            PARTITION_HORZ_4 => {
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r + q, c, sub_size)?;
                self.decode_block(r + 2 * q, c, sub_size)?;
                if r + 3 * q < self.fh.mi_rows {
                    self.decode_block(r + 3 * q, c, sub_size)?;
                }
            }
            _ => {
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r, c + q, sub_size)?;
                self.decode_block(r, c + 2 * q, sub_size)?;
                if c + 3 * q < self.fh.mi_cols {
                    self.decode_block(r, c + 3 * q, sub_size)?;
                }
            }
        }
        Ok(())
    }

    fn partition_ctx(&self, r: usize, c: usize, b_size: usize) -> usize {
        let bsl = MI_WIDTH_LOG2[b_size];
        let above =
            self.avail_u && MI_WIDTH_LOG2[self.f.mi.mi_sizes[self.mi_idx(r - 1, c)] as usize] < bsl;
        let left = self.avail_l
            && MI_HEIGHT_LOG2[self.f.mi.mi_sizes[self.mi_idx(r, c - 1)] as usize] < bsl;
        usize::from(left) * 2 + usize::from(above)
    }

    /// The partition CDF (without adaptation) as an 11-entry array.
    fn partition_cdf(&self, r: usize, c: usize, b_size: usize) -> [u16; 11] {
        let ctx = self.partition_ctx(r, c, b_size);
        let mut out = [32768u16; 11];
        // Copy only the cumulative values, never the trailing counter.
        match MI_WIDTH_LOG2[b_size] {
            1 => out[..4].copy_from_slice(&self.cdf.partition_w8[ctx][..4]),
            2 => out[..10].copy_from_slice(&self.cdf.partition_w16[ctx][..10]),
            3 => out[..10].copy_from_slice(&self.cdf.partition_w32[ctx][..10]),
            4 => out[..10].copy_from_slice(&self.cdf.partition_w64[ctx][..10]),
            _ => out[..8].copy_from_slice(&self.cdf.partition_w128[ctx][..8]),
        }
        out
    }

    fn bool_from_psum(&mut self, psum: u32) -> bool {
        // The derived CDF is rebuilt for every use, so adapting it is moot.
        let mut cdf = [((1u32 << 15) - psum) as u16, 1 << 15, 0];
        self.sd.symbol(&mut cdf) == 1
    }

    fn decode_block(&mut self, r: usize, c: usize, sub_size: usize) -> crate::Result<()> {
        self.blocks_decoded += 1;
        self.mi_row = r;
        self.mi_col = c;
        self.mi_size = sub_size;
        let bw4 = bw4(sub_size);
        let bh4 = bh4(sub_size);
        let ss_x = self.seq.subsampling_x;
        let ss_y = self.seq.subsampling_y;
        self.has_chroma =
            if (bh4 == 1 && ss_y && (r & 1) == 0) || (bw4 == 1 && ss_x && (c & 1) == 0) {
                false
            } else {
                self.seq.num_planes > 1
            };
        self.avail_u = self.is_inside(r as isize - 1, c as isize);
        self.avail_l = self.is_inside(r as isize, c as isize - 1);
        self.avail_u_chroma = self.avail_u;
        self.avail_l_chroma = self.avail_l;
        if self.has_chroma {
            if ss_y && bh4 == 1 {
                self.avail_u_chroma = self.is_inside(r as isize - 2, c as isize);
            }
            if ss_x && bw4 == 1 {
                self.avail_l_chroma = self.is_inside(r as isize, c as isize - 2);
            }
        } else {
            self.avail_u_chroma = false;
            self.avail_l_chroma = false;
        }
        // Block-level syntax defaults.
        self.use_filter_intra = false;
        self.interintra = false;
        self.motion_mode = SIMPLE;
        self.compound_type = COMPOUND_AVERAGE;
        self.comp_group_idx = 0;
        self.compound_idx = 1;
        self.interp_filter = [0; 2];
        self.mv = [[0; 2]; 2];
        self.palette_size_y = 0;
        self.palette_size_uv = 0;
        self.angle_delta_y = 0;
        self.angle_delta_uv = 0;
        self.cfl_alpha_u = 0;
        self.cfl_alpha_v = 0;
        self.uv_mode = DC_PRED;

        self.mode_info();
        self.note_tools();
        self.palette_tokens();
        self.read_block_tx_size();
        if self.skip {
            self.reset_block_context(bw4, bh4);
        }
        let is_compound = self.ref_frame[1] > INTRA_FRAME;
        let mi_rows = self.fh.mi_rows;
        let mi_cols = self.fh.mi_cols;
        for y in 0..bh4 {
            if r + y >= mi_rows {
                break;
            }
            for x in 0..bw4 {
                if c + x >= mi_cols {
                    break;
                }
                let i = self.mi_idx(r + y, c + x);
                let mi = &mut self.f.mi;
                mi.y_modes[i] = self.y_mode;
                if self.ref_frame[0] == INTRA_FRAME && self.has_chroma {
                    mi.uv_modes[i] = self.uv_mode;
                }
                mi.ref_frames[i] = self.ref_frame;
                mi.ref_frames_written[i] = true;
                if self.is_inter {
                    if !self.use_intrabc {
                        mi.comp_group_idxs[i] = self.comp_group_idx;
                        mi.compound_idxs[i] = self.compound_idx;
                    }
                    mi.interp_filters[i] = self.interp_filter;
                    mi.mvs[i][0] = self.mv[0];
                    if is_compound {
                        mi.mvs[i][1] = self.mv[1];
                    }
                }
            }
        }
        self.compute_prediction()?;
        self.residual()?;
        for y in 0..bh4 {
            if r + y >= mi_rows {
                break;
            }
            for x in 0..bw4 {
                if c + x >= mi_cols {
                    break;
                }
                let i = self.mi_idx(r + y, c + x);
                let mi = &mut self.f.mi;
                mi.is_inters[i] = self.is_inter;
                mi.skip_modes[i] = self.skip_mode;
                mi.skips[i] = self.skip;
                mi.tx_sizes[i] = self.tx_size as u8;
                mi.mi_sizes[i] = self.mi_size as u8;
                mi.segment_ids[i] = self.segment_id as u8;
                mi.palette_sizes[0][i] = self.palette_size_y as u8;
                mi.palette_sizes[1][i] = self.palette_size_uv as u8;
                mi.palette_colors[0][i] = self.palette_colors_y;
                mi.palette_colors[1][i] = self.palette_colors_u;
                for k in 0..FRAME_LF_COUNT {
                    mi.delta_lfs[i][k] = self.delta_lf[k] as i8;
                }
            }
        }
        Ok(())
    }

    /// Records the block-level tools this block uses (see `coverage`).
    fn note_tools(&self) {
        let mut tools = 0;
        if self.palette_size_y > 0 || self.palette_size_uv > 0 {
            tools |= coverage::PALETTE;
        }
        if self.use_intrabc {
            tools |= coverage::INTRA_BLOCK_COPY;
        }
        if self.use_filter_intra {
            tools |= coverage::FILTER_INTRA;
        }
        if self.ref_frame[1] > INTRA_FRAME {
            tools |= match self.compound_type {
                COMPOUND_WEDGE => coverage::WEDGE_COMPOUND,
                COMPOUND_DIFFWTD => coverage::DIFF_WEIGHTED_COMPOUND,
                COMPOUND_DISTANCE => coverage::DISTANCE_WEIGHTED_COMPOUND,
                _ => 0,
            };
        }
        if self.is_inter && self.interp_filter[0] != self.interp_filter[1] {
            tools |= coverage::DUAL_FILTER;
        }
        coverage::note(tools);
    }

    fn reset_block_context(&mut self, bw4: usize, bh4: usize) {
        for plane in 0..1 + 2 * usize::from(self.has_chroma) {
            let sub_x = self.sub_x(plane);
            let sub_y = self.sub_y(plane);
            for i in (self.mi_col >> sub_x)..((self.mi_col + bw4) >> sub_x) {
                self.above_level_context[plane][i] = 0;
                self.above_dc_context[plane][i] = 0;
            }
            for i in (self.mi_row >> sub_y)..((self.mi_row + bh4) >> sub_y) {
                self.left_level_context[plane][i] = 0;
                self.left_dc_context[plane][i] = 0;
            }
        }
    }

    fn mode_info(&mut self) {
        if self.fh.frame_is_intra {
            self.intra_frame_mode_info();
        } else {
            self.inter_frame_mode_info();
        }
    }

    fn intra_frame_mode_info(&mut self) {
        self.skip = false;
        if self.fh.segmentation.seg_id_pre_skip {
            self.intra_segment_id();
        }
        self.skip_mode = false;
        self.read_skip();
        if !self.fh.segmentation.seg_id_pre_skip {
            self.intra_segment_id();
        }
        self.read_cdef();
        self.read_delta_qindex();
        self.read_delta_lf();
        self.read_deltas = false;
        self.ref_frame = [INTRA_FRAME, NONE];
        self.use_intrabc = if self.fh.allow_intrabc {
            self.sd.symbol(&mut self.cdf.intrabc) == 1
        } else {
            false
        };
        if self.use_intrabc {
            self.is_inter = true;
            self.y_mode = DC_PRED;
            self.uv_mode = DC_PRED;
            self.motion_mode = SIMPLE;
            self.compound_type = COMPOUND_AVERAGE;
            self.palette_size_y = 0;
            self.palette_size_uv = 0;
            self.interp_filter = [BILINEAR, BILINEAR];
            self.find_mv_stack(false);
            self.assign_mv(false);
        } else {
            self.is_inter = false;
            let above = if self.avail_u {
                self.f.mi.y_modes[self.mi_idx(self.mi_row - 1, self.mi_col)]
            } else {
                DC_PRED
            };
            let left = if self.avail_l {
                self.f.mi.y_modes[self.mi_idx(self.mi_row, self.mi_col - 1)]
            } else {
                DC_PRED
            };
            let above_ctx = INTRA_MODE_CONTEXT[above as usize] as usize;
            let left_ctx = INTRA_MODE_CONTEXT[left as usize] as usize;
            self.y_mode = self.read_intra_frame_y_mode(above_ctx, left_ctx) as u8;
            self.intra_angle_info_y();
            if self.has_chroma {
                self.read_uv_mode();
                if self.uv_mode == UV_CFL_PRED {
                    self.read_cfl_alphas();
                }
                self.intra_angle_info_uv();
            }
            self.palette_size_y = 0;
            self.palette_size_uv = 0;
            if self.mi_size >= BLOCK_8X8
                && block_width(self.mi_size) <= 64
                && block_height(self.mi_size) <= 64
                && self.fh.allow_screen_content_tools
            {
                self.palette_mode_info();
            }
            self.filter_intra_mode_info();
        }
    }

    fn intra_segment_id(&mut self) {
        if self.fh.segmentation.enabled {
            self.read_segment_id();
        } else {
            self.segment_id = 0;
        }
        self.lossless = self.fh.lossless_array[self.segment_id];
    }

    fn read_segment_id(&mut self) {
        let (r, c) = (self.mi_row, self.mi_col);
        let prev_ul: i32 = if self.avail_u && self.avail_l {
            i32::from(self.f.mi.segment_ids[self.mi_idx(r - 1, c - 1)])
        } else {
            -1
        };
        let prev_u: i32 = if self.avail_u {
            i32::from(self.f.mi.segment_ids[self.mi_idx(r - 1, c)])
        } else {
            -1
        };
        let prev_l: i32 = if self.avail_l {
            i32::from(self.f.mi.segment_ids[self.mi_idx(r, c - 1)])
        } else {
            -1
        };
        let pred = if prev_u == -1 {
            if prev_l == -1 { 0 } else { prev_l }
        } else if prev_l == -1 || prev_ul == prev_u {
            prev_u
        } else {
            prev_l
        };
        if self.skip {
            self.segment_id = pred as usize;
        } else {
            let ctx = if prev_ul < 0 {
                0
            } else if prev_ul == prev_u && prev_ul == prev_l {
                2
            } else if prev_ul == prev_u || prev_ul == prev_l || prev_u == prev_l {
                1
            } else {
                0
            };
            let coded = self.sd.symbol(&mut self.cdf.segment_id[ctx]) as i32;
            let max = self.fh.segmentation.last_active_seg_id as i32 + 1;
            self.segment_id = neg_deinterleave(coded, pred, max).clamp(0, 7) as usize;
        }
    }

    fn read_skip_mode(&mut self) {
        let seg = &self.fh.segmentation;
        if seg.feature_active(self.segment_id, SEG_LVL_SKIP)
            || seg.feature_active(self.segment_id, SEG_LVL_REF_FRAME)
            || seg.feature_active(self.segment_id, SEG_LVL_GLOBALMV)
            || !self.fh.skip_mode_present
            || block_width(self.mi_size) < 8
            || block_height(self.mi_size) < 8
        {
            self.skip_mode = false;
        } else {
            let mut ctx = 0;
            if self.avail_u {
                ctx += usize::from(self.f.mi.skip_modes[self.mi_idx(self.mi_row - 1, self.mi_col)]);
            }
            if self.avail_l {
                ctx += usize::from(self.f.mi.skip_modes[self.mi_idx(self.mi_row, self.mi_col - 1)]);
            }
            self.skip_mode = self.sd.symbol(&mut self.cdf.skip_mode[ctx]) == 1;
        }
    }

    fn read_skip(&mut self) {
        if self.fh.segmentation.seg_id_pre_skip
            && self
                .fh
                .segmentation
                .feature_active(self.segment_id, SEG_LVL_SKIP)
        {
            self.skip = true;
        } else {
            let mut ctx = 0;
            if self.avail_u {
                ctx += usize::from(self.f.mi.skips[self.mi_idx(self.mi_row - 1, self.mi_col)]);
            }
            if self.avail_l {
                ctx += usize::from(self.f.mi.skips[self.mi_idx(self.mi_row, self.mi_col - 1)]);
            }
            self.skip = self.sd.symbol(&mut self.cdf.skip[ctx]) == 1;
        }
    }

    fn read_delta_qindex(&mut self) {
        if self.mi_size == self.sb_size() && self.skip {
            return;
        }
        if self.read_deltas {
            let mut delta_q_abs = self.sd.symbol(&mut self.cdf.delta_q) as u32;
            if delta_q_abs == DELTA_Q_SMALL {
                let delta_q_rem_bits = self.sd.literal(3) + 1;
                let delta_q_abs_bits = self.sd.literal(delta_q_rem_bits);
                delta_q_abs = delta_q_abs_bits + (1 << delta_q_rem_bits) + 1;
            }
            if delta_q_abs != 0 {
                let sign = self.sd.literal(1);
                let reduced = if sign != 0 {
                    -(delta_q_abs as i32)
                } else {
                    delta_q_abs as i32
                };
                self.current_q_index =
                    (self.current_q_index + (reduced << self.fh.delta_q_res)).clamp(1, 255);
            }
        }
    }

    fn read_delta_lf(&mut self) {
        if self.mi_size == self.sb_size() && self.skip {
            return;
        }
        if self.read_deltas && self.fh.delta_lf_present {
            let mut frame_lf_count = 1;
            if self.fh.delta_lf_multi {
                frame_lf_count = if self.seq.num_planes > 1 {
                    FRAME_LF_COUNT
                } else {
                    FRAME_LF_COUNT - 2
                };
            }
            for i in 0..frame_lf_count {
                let delta_lf_abs = if self.fh.delta_lf_multi {
                    self.sd.symbol(&mut self.cdf.delta_lf_multi[i]) as u32
                } else {
                    self.sd.symbol(&mut self.cdf.delta_lf) as u32
                };
                let delta_lf_abs = if delta_lf_abs == DELTA_LF_SMALL {
                    let n = self.sd.literal(3) + 1;
                    let bits = self.sd.literal(n);
                    bits + (1 << n) + 1
                } else {
                    delta_lf_abs
                };
                if delta_lf_abs != 0 {
                    let sign = self.sd.literal(1);
                    let reduced = if sign != 0 {
                        -(delta_lf_abs as i32)
                    } else {
                        delta_lf_abs as i32
                    };
                    self.delta_lf[i] = (self.delta_lf[i] + (reduced << self.fh.delta_lf_res))
                        .clamp(-MAX_LOOP_FILTER, MAX_LOOP_FILTER);
                }
            }
        }
    }

    fn read_tx_size(&mut self, allow_select: bool) {
        if self.lossless {
            self.tx_size = TX_4X4;
            return;
        }
        let max_rect_tx_size = MAX_TX_SIZE_RECT[self.mi_size] as usize;
        let max_tx_depth = MAX_TX_DEPTH[self.mi_size];
        self.tx_size = max_rect_tx_size;
        if self.mi_size > BLOCK_4X4 && allow_select && self.fh.tx_mode == TX_MODE_SELECT {
            let ctx = self.tx_depth_ctx(max_rect_tx_size);
            let tx_depth = match max_tx_depth {
                4 => self.sd.symbol(&mut self.cdf.tx_64x64[ctx]),
                3 => self.sd.symbol(&mut self.cdf.tx_32x32[ctx]),
                2 => self.sd.symbol(&mut self.cdf.tx_16x16[ctx]),
                _ => self.sd.symbol(&mut self.cdf.tx_8x8[ctx]),
            };
            for _ in 0..tx_depth {
                self.tx_size = SPLIT_TX_SIZE[self.tx_size] as usize;
            }
        }
    }

    fn tx_depth_ctx(&self, max_rect_tx_size: usize) -> usize {
        let max_tx_width = TX_WIDTH[max_rect_tx_size] as usize;
        let max_tx_height = TX_HEIGHT[max_rect_tx_size] as usize;
        let (r, c) = (self.mi_row, self.mi_col);
        let above_w = if self.avail_u && self.f.mi.is_inters[self.mi_idx(r - 1, c)] {
            block_width(self.f.mi.mi_sizes[self.mi_idx(r - 1, c)] as usize)
        } else if self.avail_u {
            self.get_above_tx_width(r, c)
        } else {
            0
        };
        let left_h = if self.avail_l && self.f.mi.is_inters[self.mi_idx(r, c - 1)] {
            block_height(self.f.mi.mi_sizes[self.mi_idx(r, c - 1)] as usize)
        } else if self.avail_l {
            self.get_left_tx_height(r, c)
        } else {
            0
        };
        usize::from(above_w >= max_tx_width) + usize::from(left_h >= max_tx_height)
    }

    fn get_above_tx_width(&self, row: usize, col: usize) -> usize {
        if row == self.mi_row {
            if !self.avail_u {
                return 64;
            }
            let i = self.mi_idx(row - 1, col);
            if self.f.mi.skips[i] && self.f.mi.is_inters[i] {
                return block_width(self.f.mi.mi_sizes[i] as usize);
            }
        }
        TX_WIDTH[self.f.mi.inter_tx_sizes[self.mi_idx(row - 1, col)] as usize] as usize
    }

    fn get_left_tx_height(&self, row: usize, col: usize) -> usize {
        if col == self.mi_col {
            if !self.avail_l {
                return 64;
            }
            let i = self.mi_idx(row, col - 1);
            if self.f.mi.skips[i] && self.f.mi.is_inters[i] {
                return block_height(self.f.mi.mi_sizes[i] as usize);
            }
        }
        TX_HEIGHT[self.f.mi.inter_tx_sizes[self.mi_idx(row, col - 1)] as usize] as usize
    }

    fn read_block_tx_size(&mut self) {
        let bw4 = bw4(self.mi_size);
        let bh4 = bh4(self.mi_size);
        if self.fh.tx_mode == TX_MODE_SELECT
            && self.mi_size > BLOCK_4X4
            && self.is_inter
            && !self.skip
            && !self.lossless
        {
            let max_tx_sz = MAX_TX_SIZE_RECT[self.mi_size] as usize;
            let tx_w4 = TX_WIDTH[max_tx_sz] as usize / MI_SIZE;
            let tx_h4 = TX_HEIGHT[max_tx_sz] as usize / MI_SIZE;
            let mut row = self.mi_row;
            while row < self.mi_row + bh4 {
                let mut col = self.mi_col;
                while col < self.mi_col + bw4 {
                    self.read_var_tx_size(row, col, max_tx_sz, 0);
                    col += tx_w4;
                }
                row += tx_h4;
            }
        } else {
            self.read_tx_size(!self.skip || !self.is_inter);
            for row in self.mi_row..(self.mi_row + bh4).min(self.fh.mi_rows) {
                for col in self.mi_col..(self.mi_col + bw4).min(self.fh.mi_cols) {
                    let i = self.mi_idx(row, col);
                    self.f.mi.inter_tx_sizes[i] = self.tx_size as u8;
                }
            }
        }
    }

    fn read_var_tx_size(&mut self, row: usize, col: usize, tx_sz: usize, depth: usize) {
        if row >= self.fh.mi_rows || col >= self.fh.mi_cols {
            return;
        }
        let txfm_split = if tx_sz == TX_4X4 || depth == MAX_VARTX_DEPTH {
            false
        } else {
            let ctx = self.txfm_split_ctx(row, col, tx_sz);
            self.sd.symbol(&mut self.cdf.txfm_split[ctx]) == 1
        };
        let w4 = TX_WIDTH[tx_sz] as usize / MI_SIZE;
        let h4 = TX_HEIGHT[tx_sz] as usize / MI_SIZE;
        if txfm_split {
            let sub_tx_sz = SPLIT_TX_SIZE[tx_sz] as usize;
            let step_w = TX_WIDTH[sub_tx_sz] as usize / MI_SIZE;
            let step_h = TX_HEIGHT[sub_tx_sz] as usize / MI_SIZE;
            let mut i = 0;
            while i < h4 {
                let mut j = 0;
                while j < w4 {
                    self.read_var_tx_size(row + i, col + j, sub_tx_sz, depth + 1);
                    j += step_w;
                }
                i += step_h;
            }
        } else {
            for i in 0..h4 {
                if row + i >= self.fh.mi_rows {
                    break;
                }
                for j in 0..w4 {
                    if col + j >= self.fh.mi_cols {
                        break;
                    }
                    let idx = self.mi_idx(row + i, col + j);
                    self.f.mi.inter_tx_sizes[idx] = tx_sz as u8;
                }
            }
            self.tx_size = tx_sz;
        }
    }

    fn txfm_split_ctx(&self, row: usize, col: usize, tx_sz: usize) -> usize {
        let above = usize::from(self.get_above_tx_width(row, col) < TX_WIDTH[tx_sz] as usize);
        let left = usize::from(self.get_left_tx_height(row, col) < TX_HEIGHT[tx_sz] as usize);
        let size = 64.min(block_width(self.mi_size).max(block_height(self.mi_size)));
        let max_tx_sz = find_tx_size(size, size);
        let tx_sz_sqr_up = TX_SIZE_SQR_UP[tx_sz] as usize;
        usize::from(tx_sz_sqr_up != max_tx_sz) * 3 + (TX_SIZES - 1 - max_tx_sz) * 6 + above + left
    }

    fn inter_frame_mode_info(&mut self) {
        self.use_intrabc = false;
        let (r, c) = (self.mi_row, self.mi_col);
        self.left_ref_frame[0] = if self.avail_l {
            self.f.mi.ref_frames[self.mi_idx(r, c - 1)][0]
        } else {
            INTRA_FRAME
        };
        self.above_ref_frame[0] = if self.avail_u {
            self.f.mi.ref_frames[self.mi_idx(r - 1, c)][0]
        } else {
            INTRA_FRAME
        };
        self.left_ref_frame[1] = if self.avail_l {
            self.f.mi.ref_frames[self.mi_idx(r, c - 1)][1]
        } else {
            NONE
        };
        self.above_ref_frame[1] = if self.avail_u {
            self.f.mi.ref_frames[self.mi_idx(r - 1, c)][1]
        } else {
            NONE
        };
        self.left_intra = self.left_ref_frame[0] <= INTRA_FRAME;
        self.above_intra = self.above_ref_frame[0] <= INTRA_FRAME;
        self.left_single = self.left_ref_frame[1] <= INTRA_FRAME;
        self.above_single = self.above_ref_frame[1] <= INTRA_FRAME;
        self.skip = false;
        self.inter_segment_id(true);
        self.read_skip_mode();
        if self.skip_mode {
            self.skip = true;
        } else {
            self.read_skip();
        }
        if !self.fh.segmentation.seg_id_pre_skip {
            self.inter_segment_id(false);
        }
        self.lossless = self.fh.lossless_array[self.segment_id];
        self.read_cdef();
        self.read_delta_qindex();
        self.read_delta_lf();
        self.read_deltas = false;
        self.read_is_inter();
        if self.is_inter {
            self.inter_block_mode_info();
        } else {
            self.intra_block_mode_info();
        }
    }

    fn inter_segment_id(&mut self, pre_skip: bool) {
        if self.fh.segmentation.enabled {
            let predicted_segment_id = self.get_segment_id();
            if self.fh.segmentation.update_map {
                if pre_skip && !self.fh.segmentation.seg_id_pre_skip {
                    self.segment_id = 0;
                    return;
                }
                if !pre_skip && self.skip {
                    let seg_id_predicted = 0;
                    for i in 0..bw4(self.mi_size) {
                        self.above_seg_pred_context[self.mi_col + i] = seg_id_predicted;
                    }
                    for i in 0..bh4(self.mi_size) {
                        self.left_seg_pred_context[self.mi_row + i] = seg_id_predicted;
                    }
                    self.read_segment_id();
                    return;
                }
                if self.fh.segmentation.temporal_update {
                    let ctx = usize::from(self.left_seg_pred_context[self.mi_row])
                        + usize::from(self.above_seg_pred_context[self.mi_col]);
                    let seg_id_predicted =
                        self.sd.symbol(&mut self.cdf.segment_id_predicted[ctx]) as u8;
                    if seg_id_predicted != 0 {
                        self.segment_id = predicted_segment_id;
                    } else {
                        self.read_segment_id();
                    }
                    for i in 0..bw4(self.mi_size) {
                        self.above_seg_pred_context[self.mi_col + i] = seg_id_predicted;
                    }
                    for i in 0..bh4(self.mi_size) {
                        self.left_seg_pred_context[self.mi_row + i] = seg_id_predicted;
                    }
                } else {
                    self.read_segment_id();
                }
            } else {
                self.segment_id = predicted_segment_id;
            }
        } else {
            self.segment_id = 0;
        }
    }

    fn read_is_inter(&mut self) {
        let seg = &self.fh.segmentation;
        if self.skip_mode {
            self.is_inter = true;
        } else if seg.feature_active(self.segment_id, SEG_LVL_REF_FRAME) {
            self.is_inter =
                seg.feature_data[self.segment_id][SEG_LVL_REF_FRAME] != i16::from(INTRA_FRAME);
        } else if seg.feature_active(self.segment_id, SEG_LVL_GLOBALMV) {
            self.is_inter = true;
        } else {
            let ctx = if self.avail_u && self.avail_l {
                if self.left_intra && self.above_intra {
                    3
                } else {
                    usize::from(self.left_intra || self.above_intra)
                }
            } else if self.avail_u || self.avail_l {
                2 * usize::from(if self.avail_u {
                    self.above_intra
                } else {
                    self.left_intra
                })
            } else {
                0
            };
            self.is_inter = self.sd.symbol(&mut self.cdf.is_inter[ctx]) == 1;
        }
    }

    fn get_segment_id(&self) -> usize {
        let bw4 = bw4(self.mi_size);
        let bh4 = bh4(self.mi_size);
        let x_mis = (self.fh.mi_cols - self.mi_col).min(bw4);
        let y_mis = (self.fh.mi_rows - self.mi_row).min(bh4);
        let mut seg = 7u8;
        for y in 0..y_mis {
            for x in 0..x_mis {
                seg =
                    seg.min(self.f.prev_segment_ids[self.mi_idx(self.mi_row + y, self.mi_col + x)]);
            }
        }
        seg as usize
    }

    fn intra_block_mode_info(&mut self) {
        self.ref_frame = [INTRA_FRAME, NONE];
        let ctx = SIZE_GROUP[self.mi_size] as usize;
        self.y_mode = self.sd.symbol(&mut self.cdf.y_mode[ctx]) as u8;
        self.intra_angle_info_y();
        if self.has_chroma {
            self.read_uv_mode();
            if self.uv_mode == UV_CFL_PRED {
                self.read_cfl_alphas();
            }
            self.intra_angle_info_uv();
        }
        self.palette_size_y = 0;
        self.palette_size_uv = 0;
        if self.mi_size >= BLOCK_8X8
            && block_width(self.mi_size) <= 64
            && block_height(self.mi_size) <= 64
            && self.fh.allow_screen_content_tools
        {
            self.palette_mode_info();
        }
        self.filter_intra_mode_info();
    }

    fn read_uv_mode(&mut self) {
        let y = self.y_mode as usize;
        let cfl_allowed =
            if self.lossless && self.get_plane_residual_size(self.mi_size, 1) == BLOCK_4X4 {
                true
            } else {
                !self.lossless && block_width(self.mi_size).max(block_height(self.mi_size)) <= 32
            };
        self.uv_mode = if cfl_allowed {
            self.sd.symbol(&mut self.cdf.uv_mode_cfl_allowed[y]) as u8
        } else {
            self.sd.symbol(&mut self.cdf.uv_mode_cfl_not_allowed[y]) as u8
        };
    }

    fn read_intra_frame_y_mode(&mut self, above_ctx: usize, left_ctx: usize) -> usize {
        self.sd
            .symbol(&mut self.intra_frame_y_mode_cdf[above_ctx][left_ctx])
    }

    fn intra_angle_info_y(&mut self) {
        self.angle_delta_y = 0;
        if self.mi_size >= BLOCK_8X8 && is_directional_mode(self.y_mode) {
            let v = self
                .sd
                .symbol(&mut self.cdf.angle_delta[(self.y_mode - V_PRED) as usize]);
            self.angle_delta_y = v as i32 - MAX_ANGLE_DELTA;
        }
    }

    fn intra_angle_info_uv(&mut self) {
        self.angle_delta_uv = 0;
        if self.mi_size >= BLOCK_8X8 && is_directional_mode(self.uv_mode) {
            let v = self
                .sd
                .symbol(&mut self.cdf.angle_delta[(self.uv_mode - V_PRED) as usize]);
            self.angle_delta_uv = v as i32 - MAX_ANGLE_DELTA;
        }
    }

    fn read_cfl_alphas(&mut self) {
        let signs = self.sd.symbol(&mut self.cdf.cfl_sign);
        let sign_u = (signs + 1) / 3;
        let sign_v = (signs + 1) % 3;
        if sign_u != CFL_SIGN_ZERO {
            let ctx = (sign_u - 1) * 3 + sign_v;
            let a = self.sd.symbol(&mut self.cdf.cfl_alpha[ctx]) as i32 + 1;
            self.cfl_alpha_u = if sign_u == CFL_SIGN_NEG { -a } else { a };
        } else {
            self.cfl_alpha_u = 0;
        }
        if sign_v != CFL_SIGN_ZERO {
            let ctx = (sign_v - 1) * 3 + sign_u;
            let a = self.sd.symbol(&mut self.cdf.cfl_alpha[ctx]) as i32 + 1;
            self.cfl_alpha_v = if sign_v == CFL_SIGN_NEG { -a } else { a };
        } else {
            self.cfl_alpha_v = 0;
        }
    }

    fn filter_intra_mode_info(&mut self) {
        self.use_filter_intra = false;
        if self.seq.enable_filter_intra
            && self.y_mode == DC_PRED
            && self.palette_size_y == 0
            && block_width(self.mi_size).max(block_height(self.mi_size)) <= 32
        {
            self.use_filter_intra = self.sd.symbol(&mut self.cdf.filter_intra[self.mi_size]) == 1;
            if self.use_filter_intra {
                self.filter_intra_mode = self.sd.symbol(&mut self.cdf.filter_intra_mode);
            }
        }
    }

    fn inter_block_mode_info(&mut self) {
        self.palette_size_y = 0;
        self.palette_size_uv = 0;
        self.read_ref_frames();
        let is_compound = self.ref_frame[1] > INTRA_FRAME;
        self.find_mv_stack(is_compound);
        let seg = self.fh.segmentation;
        if self.skip_mode {
            self.y_mode = NEAREST_NEARESTMV;
        } else if seg.feature_active(self.segment_id, SEG_LVL_SKIP)
            || seg.feature_active(self.segment_id, SEG_LVL_GLOBALMV)
        {
            self.y_mode = GLOBALMV;
        } else if is_compound {
            let ctx = COMPOUND_MODE_CTX_MAP[self.ref_mv_context >> 1][self.new_mv_context.min(4)]
                as usize;
            let compound_mode = self.sd.symbol(&mut self.cdf.compound_mode[ctx]) as u8;
            self.y_mode = NEAREST_NEARESTMV + compound_mode;
        } else {
            let new_mv = self.sd.symbol(&mut self.cdf.new_mv[self.new_mv_context]);
            if new_mv == 0 {
                self.y_mode = NEWMV;
            } else {
                let zero_mv = self.sd.symbol(&mut self.cdf.zero_mv[self.zero_mv_context]);
                if zero_mv == 0 {
                    self.y_mode = GLOBALMV;
                } else {
                    let ref_mv = self.sd.symbol(&mut self.cdf.ref_mv[self.ref_mv_context]);
                    self.y_mode = if ref_mv == 0 { NEARESTMV } else { NEARMV };
                }
            }
        }
        self.ref_mv_idx = 0;
        if self.y_mode == NEWMV || self.y_mode == NEW_NEWMV {
            for idx in 0..2 {
                if self.num_mv_found > idx + 1 {
                    let ctx = self.drl_ctx_stack[idx];
                    let drl_mode = self.sd.symbol(&mut self.cdf.drl_mode[ctx]);
                    if drl_mode == 0 {
                        self.ref_mv_idx = idx;
                        break;
                    }
                    self.ref_mv_idx = idx + 1;
                }
            }
        } else if self.has_nearmv() {
            self.ref_mv_idx = 1;
            for idx in 1..3 {
                if self.num_mv_found > idx + 1 {
                    let ctx = self.drl_ctx_stack[idx];
                    let drl_mode = self.sd.symbol(&mut self.cdf.drl_mode[ctx]);
                    if drl_mode == 0 {
                        self.ref_mv_idx = idx;
                        break;
                    }
                    self.ref_mv_idx = idx + 1;
                }
            }
        }
        self.assign_mv(is_compound);
        self.read_interintra_mode(is_compound);
        self.read_motion_mode(is_compound);
        self.read_compound_type(is_compound);
        if self.fh.interpolation_filter == SWITCHABLE {
            let dirs = if self.seq.enable_dual_filter { 2 } else { 1 };
            for dir in 0..dirs {
                if self.needs_interp_filter() {
                    let ctx = self.interp_filter_ctx(dir);
                    self.interp_filter[dir] =
                        self.sd.symbol(&mut self.cdf.interp_filter[ctx]) as u8;
                } else {
                    self.interp_filter[dir] = EIGHTTAP;
                }
            }
            if !self.seq.enable_dual_filter {
                self.interp_filter[1] = self.interp_filter[0];
            }
        } else {
            self.interp_filter = [self.fh.interpolation_filter; 2];
        }
    }

    fn has_nearmv(&self) -> bool {
        matches!(self.y_mode, NEARMV | NEAR_NEARMV | NEAR_NEWMV | NEW_NEARMV)
    }

    fn needs_interp_filter(&self) -> bool {
        let large = block_width(self.mi_size).min(block_height(self.mi_size)) >= 8;
        if self.skip_mode || self.motion_mode == LOCALWARP {
            false
        } else if large && self.y_mode == GLOBALMV {
            self.fh.gm_type[self.ref_frame[0] as usize] == TRANSLATION
        } else if large && self.y_mode == GLOBAL_GLOBALMV {
            self.fh.gm_type[self.ref_frame[0] as usize] == TRANSLATION
                || self.fh.gm_type[self.ref_frame[1] as usize] == TRANSLATION
        } else {
            true
        }
    }

    fn interp_filter_ctx(&self, dir: usize) -> usize {
        let mut ctx = ((dir & 1) * 2 + usize::from(self.ref_frame[1] > INTRA_FRAME)) * 4;
        let mut left_type = 3;
        let mut above_type = 3;
        if self.avail_l {
            let i = self.mi_idx(self.mi_row, self.mi_col - 1);
            let rf = self.f.mi.ref_frames[i];
            if rf[0] == self.ref_frame[0] || rf[1] == self.ref_frame[0] {
                left_type = self.f.mi.interp_filters[i][dir] as usize;
            }
        }
        if self.avail_u {
            let i = self.mi_idx(self.mi_row - 1, self.mi_col);
            let rf = self.f.mi.ref_frames[i];
            if rf[0] == self.ref_frame[0] || rf[1] == self.ref_frame[0] {
                above_type = self.f.mi.interp_filters[i][dir] as usize;
            }
        }
        if left_type == above_type {
            ctx += left_type;
        } else if left_type == 3 {
            ctx += above_type;
        } else if above_type == 3 {
            ctx += left_type;
        } else {
            ctx += 3;
        }
        ctx
    }

    fn count_refs(&self, frame_type: i8) -> usize {
        let mut c = 0;
        if self.avail_u {
            if self.above_ref_frame[0] == frame_type {
                c += 1;
            }
            if self.above_ref_frame[1] == frame_type {
                c += 1;
            }
        }
        if self.avail_l {
            if self.left_ref_frame[0] == frame_type {
                c += 1;
            }
            if self.left_ref_frame[1] == frame_type {
                c += 1;
            }
        }
        c
    }

    fn comp_ref_ctx(&self) -> usize {
        let last12 = self.count_refs(LAST_FRAME) + self.count_refs(LAST2_FRAME);
        let last3_gold = self.count_refs(LAST3_FRAME) + self.count_refs(GOLDEN_FRAME);
        ref_count_ctx(last12, last3_gold)
    }

    fn comp_ref_p1_ctx(&self) -> usize {
        ref_count_ctx(self.count_refs(LAST_FRAME), self.count_refs(LAST2_FRAME))
    }

    fn comp_ref_p2_ctx(&self) -> usize {
        ref_count_ctx(self.count_refs(LAST3_FRAME), self.count_refs(GOLDEN_FRAME))
    }

    fn comp_bwdref_ctx(&self) -> usize {
        let brfarf2 = self.count_refs(BWDREF_FRAME) + self.count_refs(ALTREF2_FRAME);
        ref_count_ctx(brfarf2, self.count_refs(ALTREF_FRAME))
    }

    fn comp_bwdref_p1_ctx(&self) -> usize {
        ref_count_ctx(
            self.count_refs(BWDREF_FRAME),
            self.count_refs(ALTREF2_FRAME),
        )
    }

    fn single_ref_p1_ctx(&self) -> usize {
        let fwd = self.count_refs(LAST_FRAME)
            + self.count_refs(LAST2_FRAME)
            + self.count_refs(LAST3_FRAME)
            + self.count_refs(GOLDEN_FRAME);
        let bwd = self.count_refs(BWDREF_FRAME)
            + self.count_refs(ALTREF2_FRAME)
            + self.count_refs(ALTREF_FRAME);
        ref_count_ctx(fwd, bwd)
    }

    fn uni_comp_ref_p1_ctx(&self) -> usize {
        let last2 = self.count_refs(LAST2_FRAME);
        let last3_gold = self.count_refs(LAST3_FRAME) + self.count_refs(GOLDEN_FRAME);
        ref_count_ctx(last2, last3_gold)
    }

    fn comp_mode_ctx(&self) -> usize {
        if self.avail_u && self.avail_l {
            if self.above_single && self.left_single {
                usize::from(
                    check_backward(self.above_ref_frame[0])
                        ^ check_backward(self.left_ref_frame[0]),
                )
            } else if self.above_single {
                2 + usize::from(check_backward(self.above_ref_frame[0]) || self.above_intra)
            } else if self.left_single {
                2 + usize::from(check_backward(self.left_ref_frame[0]) || self.left_intra)
            } else {
                4
            }
        } else if self.avail_u {
            if self.above_single {
                usize::from(check_backward(self.above_ref_frame[0]))
            } else {
                3
            }
        } else if self.avail_l {
            if self.left_single {
                usize::from(check_backward(self.left_ref_frame[0]))
            } else {
                3
            }
        } else {
            1
        }
    }

    fn comp_ref_type_ctx(&self) -> usize {
        let above0 = self.above_ref_frame[0];
        let above1 = self.above_ref_frame[1];
        let left0 = self.left_ref_frame[0];
        let left1 = self.left_ref_frame[1];
        let is_samedir = |a: i8, b: i8| (a >= BWDREF_FRAME) == (b >= BWDREF_FRAME);
        let above_comp_inter = self.avail_u && !self.above_intra && !self.above_single;
        let left_comp_inter = self.avail_l && !self.left_intra && !self.left_single;
        let above_uni_comp = above_comp_inter && is_samedir(above0, above1);
        let left_uni_comp = left_comp_inter && is_samedir(left0, left1);
        if self.avail_u && !self.above_intra && self.avail_l && !self.left_intra {
            let samedir = usize::from(is_samedir(above0, left0));
            if !above_comp_inter && !left_comp_inter {
                1 + 2 * samedir
            } else if !above_comp_inter {
                if !left_uni_comp { 1 } else { 3 + samedir }
            } else if !left_comp_inter {
                if !above_uni_comp { 1 } else { 3 + samedir }
            } else if !above_uni_comp && !left_uni_comp {
                0
            } else if !above_uni_comp || !left_uni_comp {
                2
            } else {
                3 + usize::from((above0 == BWDREF_FRAME) == (left0 == BWDREF_FRAME))
            }
        } else if self.avail_u && self.avail_l {
            if above_comp_inter {
                1 + 2 * usize::from(above_uni_comp)
            } else if left_comp_inter {
                1 + 2 * usize::from(left_uni_comp)
            } else {
                2
            }
        } else if above_comp_inter {
            4 * usize::from(above_uni_comp)
        } else if left_comp_inter {
            4 * usize::from(left_uni_comp)
        } else {
            2
        }
    }

    fn read_ref_frames(&mut self) {
        let seg = self.fh.segmentation;
        if self.skip_mode {
            self.ref_frame = self.fh.skip_mode_frame;
        } else if seg.feature_active(self.segment_id, SEG_LVL_REF_FRAME) {
            self.ref_frame = [
                seg.feature_data[self.segment_id][SEG_LVL_REF_FRAME] as i8,
                NONE,
            ];
        } else if seg.feature_active(self.segment_id, SEG_LVL_SKIP)
            || seg.feature_active(self.segment_id, SEG_LVL_GLOBALMV)
        {
            self.ref_frame = [LAST_FRAME, NONE];
        } else {
            let bw4 = bw4(self.mi_size);
            let bh4 = bh4(self.mi_size);
            let comp_mode = if self.fh.reference_select && bw4.min(bh4) >= 2 {
                let ctx = self.comp_mode_ctx();
                self.sd.symbol(&mut self.cdf.comp_mode[ctx])
            } else {
                SINGLE_REFERENCE
            };
            if comp_mode == COMPOUND_REFERENCE {
                let ctx = self.comp_ref_type_ctx();
                let comp_ref_type = self.sd.symbol(&mut self.cdf.comp_ref_type[ctx]);
                if comp_ref_type == UNIDIR_COMP_REFERENCE {
                    let ctx = self.single_ref_p1_ctx();
                    let uni_comp_ref = self.sd.symbol(&mut self.cdf.uni_comp_ref[ctx][0]);
                    if uni_comp_ref != 0 {
                        self.ref_frame = [BWDREF_FRAME, ALTREF_FRAME];
                    } else {
                        let ctx = self.uni_comp_ref_p1_ctx();
                        let p1 = self.sd.symbol(&mut self.cdf.uni_comp_ref[ctx][1]);
                        if p1 != 0 {
                            let ctx = self.comp_ref_p2_ctx();
                            let p2 = self.sd.symbol(&mut self.cdf.uni_comp_ref[ctx][2]);
                            self.ref_frame = if p2 != 0 {
                                [LAST_FRAME, GOLDEN_FRAME]
                            } else {
                                [LAST_FRAME, LAST3_FRAME]
                            };
                        } else {
                            self.ref_frame = [LAST_FRAME, LAST2_FRAME];
                        }
                    }
                } else {
                    let ctx = self.comp_ref_ctx();
                    let comp_ref = self.sd.symbol(&mut self.cdf.comp_ref[ctx][0]);
                    if comp_ref == 0 {
                        let ctx = self.comp_ref_p1_ctx();
                        let p1 = self.sd.symbol(&mut self.cdf.comp_ref[ctx][1]);
                        self.ref_frame[0] = if p1 != 0 { LAST2_FRAME } else { LAST_FRAME };
                    } else {
                        let ctx = self.comp_ref_p2_ctx();
                        let p2 = self.sd.symbol(&mut self.cdf.comp_ref[ctx][2]);
                        self.ref_frame[0] = if p2 != 0 { GOLDEN_FRAME } else { LAST3_FRAME };
                    }
                    let ctx = self.comp_bwdref_ctx();
                    let comp_bwdref = self.sd.symbol(&mut self.cdf.comp_bwd_ref[ctx][0]);
                    if comp_bwdref == 0 {
                        let ctx = self.comp_bwdref_p1_ctx();
                        let p1 = self.sd.symbol(&mut self.cdf.comp_bwd_ref[ctx][1]);
                        self.ref_frame[1] = if p1 != 0 { ALTREF2_FRAME } else { BWDREF_FRAME };
                    } else {
                        self.ref_frame[1] = ALTREF_FRAME;
                    }
                }
            } else {
                let ctx = self.single_ref_p1_ctx();
                let p1 = self.sd.symbol(&mut self.cdf.single_ref[ctx][0]);
                if p1 != 0 {
                    let ctx = self.comp_bwdref_ctx();
                    let p2 = self.sd.symbol(&mut self.cdf.single_ref[ctx][1]);
                    if p2 == 0 {
                        let ctx = self.comp_bwdref_p1_ctx();
                        let p6 = self.sd.symbol(&mut self.cdf.single_ref[ctx][5]);
                        self.ref_frame[0] = if p6 != 0 { ALTREF2_FRAME } else { BWDREF_FRAME };
                    } else {
                        self.ref_frame[0] = ALTREF_FRAME;
                    }
                } else {
                    let ctx = self.comp_ref_ctx();
                    let p3 = self.sd.symbol(&mut self.cdf.single_ref[ctx][2]);
                    if p3 != 0 {
                        let ctx = self.comp_ref_p2_ctx();
                        let p5 = self.sd.symbol(&mut self.cdf.single_ref[ctx][4]);
                        self.ref_frame[0] = if p5 != 0 { GOLDEN_FRAME } else { LAST3_FRAME };
                    } else {
                        let ctx = self.comp_ref_p1_ctx();
                        let p4 = self.sd.symbol(&mut self.cdf.single_ref[ctx][3]);
                        self.ref_frame[0] = if p4 != 0 { LAST2_FRAME } else { LAST_FRAME };
                    }
                }
                self.ref_frame[1] = NONE;
            }
        }
    }

    pub(crate) fn get_mode(&self, ref_list: usize) -> u8 {
        let y = self.y_mode;
        if ref_list == 0 {
            if y < NEAREST_NEARESTMV {
                y
            } else if y == NEW_NEWMV || y == NEW_NEARESTMV || y == NEW_NEARMV {
                NEWMV
            } else if y == NEAREST_NEARESTMV || y == NEAREST_NEWMV {
                NEARESTMV
            } else if y == NEAR_NEARMV || y == NEAR_NEWMV {
                NEARMV
            } else {
                GLOBALMV
            }
        } else if y == NEW_NEWMV || y == NEAREST_NEWMV || y == NEAR_NEWMV {
            NEWMV
        } else if y == NEAREST_NEARESTMV || y == NEW_NEARESTMV {
            NEARESTMV
        } else if y == NEAR_NEARMV || y == NEW_NEARMV {
            NEARMV
        } else {
            GLOBALMV
        }
    }

    fn assign_mv(&mut self, is_compound: bool) {
        for i in 0..1 + usize::from(is_compound) {
            let comp_mode = if self.use_intrabc {
                NEWMV
            } else {
                self.get_mode(i)
            };
            if self.use_intrabc {
                self.pred_mv[0] = self.ref_stack_mv[0][0];
                if self.pred_mv[0][0] == 0 && self.pred_mv[0][1] == 0 {
                    self.pred_mv[0] = self.ref_stack_mv[1][0];
                }
                if self.pred_mv[0][0] == 0 && self.pred_mv[0][1] == 0 {
                    let sb_size4 = bh4(self.sb_size()) as i32;
                    if (self.mi_row as i32) - sb_size4 < self.mi_row_start as i32 {
                        self.pred_mv[0][0] = 0;
                        self.pred_mv[0][1] =
                            -(sb_size4 * MI_SIZE as i32 + INTRABC_DELAY_PIXELS) * 8;
                    } else {
                        self.pred_mv[0][0] = -(sb_size4 * MI_SIZE as i32 * 8);
                        self.pred_mv[0][1] = 0;
                    }
                }
            } else if comp_mode == GLOBALMV {
                self.pred_mv[i] = self.global_mvs[i];
            } else {
                let mut pos = if comp_mode == NEARESTMV {
                    0
                } else {
                    self.ref_mv_idx
                };
                if comp_mode == NEWMV && self.num_mv_found <= 1 {
                    pos = 0;
                }
                self.pred_mv[i] = self.ref_stack_mv[pos][i];
            }
            if comp_mode == NEWMV {
                self.read_mv(i);
            } else {
                self.mv[i] = self.pred_mv[i];
            }
        }
    }

    fn read_motion_mode(&mut self, is_compound: bool) {
        if self.skip_mode || !self.fh.is_motion_mode_switchable {
            self.motion_mode = SIMPLE;
            return;
        }
        if block_width(self.mi_size).min(block_height(self.mi_size)) < 8 {
            self.motion_mode = SIMPLE;
            return;
        }
        if !self.fh.force_integer_mv
            && (self.y_mode == GLOBALMV || self.y_mode == GLOBAL_GLOBALMV)
            && self.fh.gm_type[self.ref_frame[0] as usize] > TRANSLATION
        {
            self.motion_mode = SIMPLE;
            return;
        }
        if is_compound || self.ref_frame[1] == INTRA_FRAME || !self.has_overlappable_candidates() {
            self.motion_mode = SIMPLE;
            return;
        }
        self.find_warp_samples();
        if self.fh.force_integer_mv
            || self.num_samples == 0
            || !self.fh.allow_warped_motion
            || self.is_scaled(self.ref_frame[0])
        {
            let use_obmc = self.sd.symbol(&mut self.cdf.use_obmc[self.mi_size]);
            self.motion_mode = if use_obmc != 0 { OBMC } else { SIMPLE };
        } else {
            self.motion_mode = self.sd.symbol(&mut self.cdf.motion_mode[self.mi_size]) as u8;
        }
    }

    pub(crate) fn is_scaled(&self, ref_frame: i8) -> bool {
        let slot = self.ref_slot(ref_frame);
        let fw = self.fh.frame_width as i64;
        let fhh = self.fh.frame_height as i64;
        let x_scale = (((slot.upscaled_width as i64) << REF_SCALE_SHIFT) + fw / 2) / fw;
        let y_scale = (((slot.frame_height as i64) << REF_SCALE_SHIFT) + fhh / 2) / fhh;
        let no_scale = 1i64 << REF_SCALE_SHIFT;
        x_scale != no_scale || y_scale != no_scale
    }

    pub(crate) fn ref_slot(&self, ref_frame: i8) -> &RefSlot {
        let idx = self.fh.ref_frame_idx[(ref_frame - LAST_FRAME) as usize];
        self.refs[idx]
            .as_deref()
            .expect("reference slots are validated by the frame header")
    }

    fn read_interintra_mode(&mut self, is_compound: bool) {
        if !self.skip_mode
            && self.seq.enable_interintra_compound
            && !is_compound
            && self.mi_size >= BLOCK_8X8
            && self.mi_size <= BLOCK_32X32
        {
            let ctx = SIZE_GROUP[self.mi_size] as usize - 1;
            self.interintra = self.sd.symbol(&mut self.cdf.inter_intra[ctx]) == 1;
            if self.interintra {
                self.interintra_mode = self.sd.symbol(&mut self.cdf.inter_intra_mode[ctx]) as u8;
                self.ref_frame[1] = INTRA_FRAME;
                self.angle_delta_y = 0;
                self.angle_delta_uv = 0;
                self.use_filter_intra = false;
                self.wedge_interintra = self
                    .sd
                    .symbol(&mut self.cdf.wedge_inter_intra[self.mi_size])
                    == 1;
                if self.wedge_interintra {
                    self.wedge_index = self.sd.symbol(&mut self.cdf.wedge_index[self.mi_size]);
                    self.wedge_sign = 0;
                }
            }
        } else {
            self.interintra = false;
        }
    }

    fn read_compound_type(&mut self, is_compound: bool) {
        self.comp_group_idx = 0;
        self.compound_idx = 1;
        if self.skip_mode {
            self.compound_type = COMPOUND_AVERAGE;
            return;
        }
        if is_compound {
            let n = WEDGE_BITS[self.mi_size];
            if self.seq.enable_masked_compound {
                let ctx = self.comp_group_idx_ctx();
                self.comp_group_idx = self.sd.symbol(&mut self.cdf.comp_group_idx[ctx]) as u8;
            }
            if self.comp_group_idx == 0 {
                if self.seq.enable_jnt_comp {
                    let ctx = self.compound_idx_ctx();
                    self.compound_idx = self.sd.symbol(&mut self.cdf.compound_idx[ctx]) as u8;
                    self.compound_type = if self.compound_idx != 0 {
                        COMPOUND_AVERAGE
                    } else {
                        COMPOUND_DISTANCE
                    };
                } else {
                    self.compound_type = COMPOUND_AVERAGE;
                }
            } else if n == 0 {
                self.compound_type = COMPOUND_DIFFWTD;
            } else {
                self.compound_type =
                    self.sd.symbol(&mut self.cdf.compound_type[self.mi_size]) as u8;
            }
            if self.compound_type == COMPOUND_WEDGE {
                self.wedge_index = self.sd.symbol(&mut self.cdf.wedge_index[self.mi_size]);
                self.wedge_sign = self.sd.literal(1) as usize;
            } else if self.compound_type == COMPOUND_DIFFWTD {
                self.mask_type = self.sd.literal(1) == 1;
            }
        } else if self.interintra {
            self.compound_type = if self.wedge_interintra {
                COMPOUND_WEDGE
            } else {
                COMPOUND_INTRA
            };
        } else {
            self.compound_type = COMPOUND_AVERAGE;
        }
    }

    fn comp_group_idx_ctx(&self) -> usize {
        let mut ctx = 0;
        if self.avail_u {
            if !self.above_single {
                ctx +=
                    self.f.mi.comp_group_idxs[self.mi_idx(self.mi_row - 1, self.mi_col)] as usize;
            } else if self.above_ref_frame[0] == ALTREF_FRAME {
                ctx += 3;
            }
        }
        if self.avail_l {
            if !self.left_single {
                ctx +=
                    self.f.mi.comp_group_idxs[self.mi_idx(self.mi_row, self.mi_col - 1)] as usize;
            } else if self.left_ref_frame[0] == ALTREF_FRAME {
                ctx += 3;
            }
        }
        ctx.min(5)
    }

    fn compound_idx_ctx(&self) -> usize {
        let fwd = self
            .fh
            .get_relative_dist(
                self.seq,
                self.fh.order_hints[self.ref_frame[0] as usize],
                self.fh.order_hint,
            )
            .abs();
        let bck = self
            .fh
            .get_relative_dist(
                self.seq,
                self.fh.order_hints[self.ref_frame[1] as usize],
                self.fh.order_hint,
            )
            .abs();
        let mut ctx = if fwd == bck { 3 } else { 0 };
        if self.avail_u {
            if !self.above_single {
                ctx += self.f.mi.compound_idxs[self.mi_idx(self.mi_row - 1, self.mi_col)] as usize;
            } else if self.above_ref_frame[0] == ALTREF_FRAME {
                ctx += 1;
            }
        }
        if self.avail_l {
            if !self.left_single {
                ctx += self.f.mi.compound_idxs[self.mi_idx(self.mi_row, self.mi_col - 1)] as usize;
            } else if self.left_ref_frame[0] == ALTREF_FRAME {
                ctx += 1;
            }
        }
        ctx
    }

    fn read_mv(&mut self, r: usize) {
        let mut diff_mv = [0i32; 2];
        let mv_ctx = if self.use_intrabc {
            MV_INTRABC_CONTEXT
        } else {
            0
        };
        let mv_joint = self.sd.symbol(&mut self.cdf.mv_joint[mv_ctx]);
        if mv_joint == MV_JOINT_HZVNZ || mv_joint == MV_JOINT_HNZVNZ {
            diff_mv[0] = self.read_mv_component(mv_ctx, 0);
        }
        if mv_joint == MV_JOINT_HNZVZ || mv_joint == MV_JOINT_HNZVNZ {
            diff_mv[1] = self.read_mv_component(mv_ctx, 1);
        }
        self.mv[r][0] = self.pred_mv[r][0] + diff_mv[0];
        self.mv[r][1] = self.pred_mv[r][1] + diff_mv[1];
    }

    fn read_mv_component(&mut self, ctx: usize, comp: usize) -> i32 {
        let mv_sign = self.sd.symbol(&mut self.cdf.mv_sign[ctx][comp]);
        let mv_class = self.sd.symbol(&mut self.cdf.mv_class[ctx][comp]);
        let mag = if mv_class == 0 {
            let mv_class0_bit = self.sd.symbol(&mut self.cdf.mv_class0_bit[ctx][comp]) as i32;
            let fr = if self.fh.force_integer_mv {
                3
            } else {
                self.sd
                    .symbol(&mut self.cdf.mv_class0_fr[ctx][comp][mv_class0_bit as usize])
                    as i32
            };
            let hp = if self.fh.allow_high_precision_mv {
                self.sd.symbol(&mut self.cdf.mv_class0_hp[ctx][comp]) as i32
            } else {
                1
            };
            ((mv_class0_bit << 3) | (fr << 1) | hp) + 1
        } else {
            let mut d = 0i32;
            for i in 0..mv_class {
                let bit = self.sd.symbol(&mut self.cdf.mv_bit[ctx][comp][i]) as i32;
                d |= bit << i;
            }
            let fr = if self.fh.force_integer_mv {
                3
            } else {
                self.sd.symbol(&mut self.cdf.mv_fr[ctx][comp]) as i32
            };
            let hp = if self.fh.allow_high_precision_mv {
                self.sd.symbol(&mut self.cdf.mv_hp[ctx][comp]) as i32
            } else {
                1
            };
            let m = CLASS0_SIZE << (mv_class + 2);
            m + ((d << 3) | (fr << 1) | hp) + 1
        };
        if mv_sign != 0 { -mag } else { mag }
    }

    pub(crate) fn get_plane_residual_size(&self, subsize: usize, plane: usize) -> usize {
        SUBSAMPLED_SIZE[subsize][self.sub_x(plane)][self.sub_y(plane)] as usize
    }

    fn clear_cdef(&mut self, r: usize, c: usize) {
        self.set_cdef_idx(r, c, -1);
        if self.seq.use_128x128_superblock {
            let cdef_size4 = bw4(BLOCK_64X64);
            self.set_cdef_idx(r, c + cdef_size4, -1);
            self.set_cdef_idx(r + cdef_size4, c, -1);
            self.set_cdef_idx(r + cdef_size4, c + cdef_size4, -1);
        }
    }

    fn set_cdef_idx(&mut self, r: usize, c: usize, value: i8) {
        let (row, col) = (r >> 4, c >> 4);
        let stride = self.f.cdef_stride;
        if col < stride {
            if let Some(entry) = self.f.cdef_idx.get_mut(row * stride + col) {
                *entry = value;
            }
        }
    }

    fn get_cdef_idx(&self, r: usize, c: usize) -> i8 {
        let (row, col) = (r >> 4, c >> 4);
        let stride = self.f.cdef_stride;
        if col >= stride {
            return -1;
        }
        self.f
            .cdef_idx
            .get(row * stride + col)
            .copied()
            .unwrap_or(-1)
    }

    fn read_cdef(&mut self) {
        if self.skip || self.fh.coded_lossless || !self.seq.enable_cdef || self.fh.allow_intrabc {
            return;
        }
        let cdef_size4 = bw4(BLOCK_64X64);
        let cdef_mask4 = !(cdef_size4 - 1);
        let r = self.mi_row & cdef_mask4;
        let c = self.mi_col & cdef_mask4;
        if self.get_cdef_idx(r, c) == -1 {
            let value = self.sd.literal(u32::from(self.fh.cdef.bits)) as i8;
            self.set_cdef_idx(r, c, value);
            let w4 = bw4(self.mi_size);
            let h4 = bh4(self.mi_size);
            let mut i = r;
            while i < r + h4 {
                let mut j = c;
                while j < c + w4 {
                    self.set_cdef_idx(i, j, value);
                    j += cdef_size4;
                }
                i += cdef_size4;
            }
        }
    }

    fn read_lr(&mut self, r: usize, c: usize, b_size: usize) {
        if self.fh.allow_intrabc {
            return;
        }
        let w = bw4(b_size);
        let h = bh4(b_size);
        for plane in 0..self.seq.num_planes {
            if self.fh.frame_restoration_type[plane] != RESTORE_NONE {
                let sub_x = self.sub_x(plane);
                let sub_y = self.sub_y(plane);
                let unit_size = self.fh.loop_restoration_size[plane];
                let unit_rows = self.f.lr[plane].unit_rows;
                let unit_cols = self.f.lr[plane].unit_cols;
                let unit_row_start = (r * (MI_SIZE >> sub_y)).div_ceil(unit_size);
                let unit_row_end =
                    unit_rows.min(((r + h) * (MI_SIZE >> sub_y)).div_ceil(unit_size));
                let (numerator, denominator) = if self.fh.use_superres {
                    (
                        (MI_SIZE >> sub_x) * self.fh.superres_denom,
                        unit_size * SUPERRES_NUM,
                    )
                } else {
                    (MI_SIZE >> sub_x, unit_size)
                };
                let unit_col_start = (c * numerator).div_ceil(denominator);
                let unit_col_end = unit_cols.min(((c + w) * numerator).div_ceil(denominator));
                for unit_row in unit_row_start..unit_row_end {
                    for unit_col in unit_col_start..unit_col_end {
                        self.read_lr_unit(plane, unit_row, unit_col);
                    }
                }
            }
        }
    }

    fn read_lr_unit(&mut self, plane: usize, unit_row: usize, unit_col: usize) {
        let restoration_type = match self.fh.frame_restoration_type[plane] {
            RESTORE_WIENER => {
                if self.sd.symbol(&mut self.cdf.use_wiener) != 0 {
                    RESTORE_WIENER
                } else {
                    RESTORE_NONE
                }
            }
            RESTORE_SGRPROJ => {
                if self.sd.symbol(&mut self.cdf.use_sgrproj) != 0 {
                    RESTORE_SGRPROJ
                } else {
                    RESTORE_NONE
                }
            }
            _ => self.sd.symbol(&mut self.cdf.restoration_type) as u8,
        };
        let unit = unit_row * self.f.lr[plane].unit_cols + unit_col;
        self.f.lr[plane].lr_type[unit] = restoration_type;
        if restoration_type == RESTORE_WIENER {
            for pass in 0..2 {
                let first_coeff = if plane > 0 {
                    self.f.lr[plane].wiener[unit][pass][0] = 0;
                    1
                } else {
                    0
                };
                for j in first_coeff..3 {
                    let min = i32::from(WIENER_TAPS_MIN[j]);
                    let max = i32::from(WIENER_TAPS_MAX[j]);
                    let k = u32::from(WIENER_TAPS_K[j]);
                    let v = self.decode_signed_subexp_with_ref_bool(
                        min,
                        max + 1,
                        k,
                        self.ref_lr_wiener[plane][pass][j],
                    );
                    self.f.lr[plane].wiener[unit][pass][j] = v;
                    self.ref_lr_wiener[plane][pass][j] = v;
                }
            }
        } else if restoration_type == RESTORE_SGRPROJ {
            let lr_sgr_set = self.sd.literal(SGRPROJ_PARAMS_BITS) as usize;
            self.f.lr[plane].sgr_set[unit] = lr_sgr_set as u8;
            for i in 0..2 {
                let radius = SGR_PARAMS[lr_sgr_set][i * 2];
                let min = i32::from(SGRPROJ_XQD_MIN[i]);
                let max = i32::from(SGRPROJ_XQD_MAX[i]);
                let v = if radius != 0 {
                    self.decode_signed_subexp_with_ref_bool(
                        min,
                        max + 1,
                        SGRPROJ_PRJ_SUBEXP_K,
                        self.ref_sgr_xqd[plane][i],
                    )
                } else if i == 1 {
                    ((1 << SGRPROJ_PRJ_BITS) - self.ref_sgr_xqd[plane][0]).clamp(min, max)
                } else {
                    0
                };
                self.f.lr[plane].sgr_xqd[unit][i] = v;
                self.ref_sgr_xqd[plane][i] = v;
            }
        }
    }

    fn decode_signed_subexp_with_ref_bool(&mut self, low: i32, high: i32, k: u32, r: i32) -> i32 {
        let mx = high - low;
        let r = r - low;
        let v = self.decode_subexp_bool(mx, k);
        let x = if (r << 1) <= mx {
            inverse_recenter(r, v)
        } else {
            mx - 1 - inverse_recenter(mx - 1 - r, v)
        };
        x + low
    }

    fn decode_subexp_bool(&mut self, num_syms: i32, k: u32) -> i32 {
        let mut i = 0u32;
        let mut mk = 0i32;
        loop {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1i32 << b2;
            if num_syms <= mk + 3 * a {
                return self.sd.ns((num_syms - mk) as u32) as i32 + mk;
            }
            if self.sd.literal(1) != 0 {
                i += 1;
                mk += a;
            } else {
                return self.sd.literal(b2) as i32 + mk;
            }
        }
    }
}

fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// `find_tx_size( w, h )`.
pub(crate) fn find_tx_size(w: usize, h: usize) -> usize {
    (0..TX_SIZES_ALL)
        .find(|&t| TX_WIDTH[t] as usize == w && TX_HEIGHT[t] as usize == h)
        .unwrap_or(TX_4X4)
}

/// The probability mass `cdf` assigns to symbol `p` (for p >= 1).
fn cdf_prob(cdf: &[u16; 11], p: usize) -> u32 {
    u32::from(cdf[p]) - u32::from(cdf[p - 1])
}
