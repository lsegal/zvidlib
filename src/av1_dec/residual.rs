//! Residual, transform block, coefficient and palette syntax (specification
//! sections 5.11.34 to 5.11.50).

use super::consts::*;
use super::tables::*;
use super::tile::{TileDecoder, bh4, bw4, find_tx_size};
use crate::Result;

fn get_tx_class(tx_type: u8) -> usize {
    match tx_type {
        V_DCT | V_ADST | V_FLIPADST => TX_CLASS_VERT,
        H_DCT | H_ADST | H_FLIPADST => TX_CLASS_HORIZ,
        _ => TX_CLASS_2D,
    }
}

fn get_mrow_scan(tx_sz: usize) -> &'static [u16] {
    match tx_sz {
        0 => &MROW_SCAN_4X4,
        5 => &MROW_SCAN_4X8,
        6 => &MROW_SCAN_8X4,
        1 => &MROW_SCAN_8X8,
        7 => &MROW_SCAN_8X16,
        8 => &MROW_SCAN_16X8,
        2 => &MROW_SCAN_16X16,
        13 => &MROW_SCAN_4X16,
        _ => &MROW_SCAN_16X4,
    }
}

fn get_mcol_scan(tx_sz: usize) -> &'static [u16] {
    match tx_sz {
        0 => &MCOL_SCAN_4X4,
        5 => &MCOL_SCAN_4X8,
        6 => &MCOL_SCAN_8X4,
        1 => &MCOL_SCAN_8X8,
        7 => &MCOL_SCAN_8X16,
        8 => &MCOL_SCAN_16X8,
        2 => &MCOL_SCAN_16X16,
        13 => &MCOL_SCAN_4X16,
        _ => &MCOL_SCAN_16X4,
    }
}

fn get_default_scan(tx_sz: usize) -> &'static [u16] {
    match tx_sz {
        0 => &DEFAULT_SCAN_4X4,
        5 => &DEFAULT_SCAN_4X8,
        6 => &DEFAULT_SCAN_8X4,
        1 => &DEFAULT_SCAN_8X8,
        7 => &DEFAULT_SCAN_8X16,
        8 => &DEFAULT_SCAN_16X8,
        2 => &DEFAULT_SCAN_16X16,
        9 => &DEFAULT_SCAN_16X32,
        10 => &DEFAULT_SCAN_32X16,
        13 => &DEFAULT_SCAN_4X16,
        14 => &DEFAULT_SCAN_16X4,
        15 => &DEFAULT_SCAN_8X32,
        16 => &DEFAULT_SCAN_32X8,
        _ => &DEFAULT_SCAN_32X32,
    }
}

impl TileDecoder<'_> {
    /// `get_scan( txSz )` for the current `PlaneTxType`.
    fn get_scan(&self, tx_sz: usize) -> &'static [u16] {
        if tx_sz == TX_16X64 {
            return &DEFAULT_SCAN_16X32;
        }
        if tx_sz == TX_64X16 {
            return &DEFAULT_SCAN_32X16;
        }
        if TX_SIZE_SQR_UP[tx_sz] as usize == TX_64X64 {
            return &DEFAULT_SCAN_32X32;
        }
        if self.plane_tx_type == IDTX {
            return get_default_scan(tx_sz);
        }
        let prefer_row = matches!(self.plane_tx_type, V_DCT | V_ADST | V_FLIPADST);
        let prefer_col = matches!(self.plane_tx_type, H_DCT | H_ADST | H_FLIPADST);
        if prefer_row {
            get_mrow_scan(tx_sz)
        } else if prefer_col {
            get_mcol_scan(tx_sz)
        } else {
            get_default_scan(tx_sz)
        }
    }

    /// `get_tx_size( plane, txSz )`.
    pub(crate) fn get_tx_size(&self, plane: usize, tx_sz: usize) -> usize {
        if plane == 0 {
            return tx_sz;
        }
        let uv_tx = MAX_TX_SIZE_RECT[self.get_plane_residual_size(self.mi_size, plane)] as usize;
        if TX_WIDTH[uv_tx] == 64 || TX_HEIGHT[uv_tx] == 64 {
            if TX_WIDTH[uv_tx] == 16 {
                return 9; // TX_16X32
            }
            if TX_HEIGHT[uv_tx] == 16 {
                return 10; // TX_32X16
            }
            return TX_32X32;
        }
        uv_tx
    }

    /// `residual( )`.
    pub(crate) fn residual(&mut self) -> Result<()> {
        let bw = block_width(self.mi_size);
        let bh = block_height(self.mi_size);
        let width_chunks = (bw >> 6).max(1);
        let height_chunks = (bh >> 6).max(1);
        let mi_size_chunk = if width_chunks > 1 || height_chunks > 1 {
            BLOCK_64X64
        } else {
            self.mi_size
        };
        for chunk_y in 0..height_chunks {
            for chunk_x in 0..width_chunks {
                let mi_row_chunk = self.mi_row + (chunk_y << 4);
                let mi_col_chunk = self.mi_col + (chunk_x << 4);
                for plane in 0..1 + usize::from(self.has_chroma) * 2 {
                    let tx_sz = if self.lossless {
                        TX_4X4
                    } else {
                        self.get_tx_size(plane, self.tx_size)
                    };
                    let step_x = TX_WIDTH[tx_sz] as usize >> 2;
                    let step_y = TX_HEIGHT[tx_sz] as usize >> 2;
                    let plane_sz = self.get_plane_residual_size(mi_size_chunk, plane);
                    let num4x4_w = bw4(plane_sz);
                    let num4x4_h = bh4(plane_sz);
                    let sub_x = self.sub_x(plane);
                    let sub_y = self.sub_y(plane);
                    let base_x = (mi_col_chunk >> sub_x) * MI_SIZE;
                    let base_y = (mi_row_chunk >> sub_y) * MI_SIZE;
                    if self.is_inter && !self.lossless && plane == 0 {
                        self.transform_tree(base_x, base_y, num4x4_w * 4, num4x4_h * 4)?;
                    } else {
                        let base_x_block = (self.mi_col >> sub_x) * MI_SIZE;
                        let base_y_block = (self.mi_row >> sub_y) * MI_SIZE;
                        let mut y = 0;
                        while y < num4x4_h {
                            let mut x = 0;
                            while x < num4x4_w {
                                self.transform_block(
                                    plane,
                                    base_x_block,
                                    base_y_block,
                                    tx_sz,
                                    x + ((chunk_x << 4) >> sub_x),
                                    y + ((chunk_y << 4) >> sub_y),
                                )?;
                                x += step_x;
                            }
                            y += step_y;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn transform_tree(&mut self, start_x: usize, start_y: usize, w: usize, h: usize) -> Result<()> {
        let max_x = self.fh.mi_cols * MI_SIZE;
        let max_y = self.fh.mi_rows * MI_SIZE;
        if start_x >= max_x || start_y >= max_y {
            return Ok(());
        }
        let row = start_y >> MI_SIZE_LOG2;
        let col = start_x >> MI_SIZE_LOG2;
        let luma_tx_sz = self.f.mi.inter_tx_sizes[self.mi_idx(row, col)] as usize;
        let luma_w = TX_WIDTH[luma_tx_sz] as usize;
        let luma_h = TX_HEIGHT[luma_tx_sz] as usize;
        if w <= luma_w && h <= luma_h {
            let tx_sz = find_tx_size(w, h);
            self.transform_block(0, start_x, start_y, tx_sz, 0, 0)?;
        } else if w > h {
            self.transform_tree(start_x, start_y, w / 2, h)?;
            self.transform_tree(start_x + w / 2, start_y, w / 2, h)?;
        } else if w < h {
            self.transform_tree(start_x, start_y, w, h / 2)?;
            self.transform_tree(start_x, start_y + h / 2, w, h / 2)?;
        } else {
            self.transform_tree(start_x, start_y, w / 2, h / 2)?;
            self.transform_tree(start_x + w / 2, start_y, w / 2, h / 2)?;
            self.transform_tree(start_x, start_y + h / 2, w / 2, h / 2)?;
            self.transform_tree(start_x + w / 2, start_y + h / 2, w / 2, h / 2)?;
        }
        Ok(())
    }

    fn transform_block(
        &mut self,
        plane: usize,
        base_x: usize,
        base_y: usize,
        tx_sz: usize,
        x: usize,
        y: usize,
    ) -> Result<()> {
        let start_x = base_x + 4 * x;
        let start_y = base_y + 4 * y;
        let sub_x = self.sub_x(plane);
        let sub_y = self.sub_y(plane);
        let row = (start_y << sub_y) >> MI_SIZE_LOG2;
        let col = (start_x << sub_x) >> MI_SIZE_LOG2;
        let sb_mask = if self.seq.use_128x128_superblock {
            31
        } else {
            15
        };
        let sub_block_mi_row = (row & sb_mask) as isize;
        let sub_block_mi_col = (col & sb_mask) as isize;
        let step_x = TX_WIDTH[tx_sz] as usize >> MI_SIZE_LOG2;
        let step_y = TX_HEIGHT[tx_sz] as usize >> MI_SIZE_LOG2;
        let max_x = (self.fh.mi_cols * MI_SIZE) >> sub_x;
        let max_y = (self.fh.mi_rows * MI_SIZE) >> sub_y;
        if start_x >= max_x || start_y >= max_y {
            return Ok(());
        }
        if !self.is_inter {
            if (plane == 0 && self.palette_size_y > 0) || (plane != 0 && self.palette_size_uv > 0) {
                self.predict_palette(plane, start_x, start_y, x, y, tx_sz);
            } else {
                let is_cfl = plane > 0 && self.uv_mode == UV_CFL_PRED;
                let mode = if plane == 0 {
                    self.y_mode
                } else if is_cfl {
                    DC_PRED
                } else {
                    self.uv_mode
                };
                let log2w = TX_WIDTH_LOG2[tx_sz] as usize;
                let log2h = TX_HEIGHT_LOG2[tx_sz] as usize;
                let row_up = (sub_block_mi_row >> sub_y) - 1;
                let col_left = (sub_block_mi_col >> sub_x) - 1;
                let have_above_right = self.block_decoded(
                    plane,
                    row_up,
                    (sub_block_mi_col >> sub_x) + step_x as isize,
                );
                let have_below_left = self.block_decoded(
                    plane,
                    (sub_block_mi_row >> sub_y) + step_y as isize,
                    col_left,
                );
                let have_left = (if plane == 0 {
                    self.avail_l
                } else {
                    self.avail_l_chroma
                }) || x > 0;
                let have_above = (if plane == 0 {
                    self.avail_u
                } else {
                    self.avail_u_chroma
                }) || y > 0;
                self.predict_intra(
                    plane,
                    start_x,
                    start_y,
                    have_left,
                    have_above,
                    have_above_right,
                    have_below_left,
                    mode,
                    log2w,
                    log2h,
                );
                if is_cfl {
                    self.predict_chroma_from_luma(plane, start_x, start_y, tx_sz);
                }
            }
            if plane == 0 {
                self.max_luma_w = start_x + step_x * 4;
                self.max_luma_h = start_y + step_y * 4;
            }
        }
        if !self.skip {
            let eob = self.coeffs(plane, start_x, start_y, tx_sz);
            if eob > 0 {
                self.reconstruct(plane, start_x, start_y, tx_sz);
            }
        }
        for i in 0..step_y {
            for j in 0..step_x {
                let lr = (row >> sub_y) + i;
                let lc = (col >> sub_x) + j;
                let stride = self.f.lf_tx_stride[plane];
                if lc < stride {
                    if let Some(v) = self.f.lf_tx_sizes[plane].get_mut(lr * stride + lc) {
                        *v = tx_sz as u8;
                    }
                }
                let by = (sub_block_mi_row >> sub_y) as usize + i + 1;
                let bx = (sub_block_mi_col >> sub_x) as usize + j + 1;
                if by < 34 && bx < 34 {
                    self.block_decoded[plane][by][bx] = true;
                }
            }
        }
        Ok(())
    }

    fn get_tx_set(&self, tx_sz: usize) -> usize {
        let tx_sz_sqr = TX_SIZE_SQR[tx_sz] as usize;
        let tx_sz_sqr_up = TX_SIZE_SQR_UP[tx_sz] as usize;
        if tx_sz_sqr_up > TX_32X32 {
            return TX_SET_DCTONLY;
        }
        if self.is_inter {
            if self.fh.reduced_tx_set || tx_sz_sqr_up == TX_32X32 {
                TX_SET_INTER_3
            } else if tx_sz_sqr == TX_16X16 {
                TX_SET_INTER_2
            } else {
                TX_SET_INTER_1
            }
        } else if tx_sz_sqr_up == TX_32X32 {
            TX_SET_DCTONLY
        } else if self.fh.reduced_tx_set || tx_sz_sqr == TX_16X16 {
            TX_SET_INTRA_2
        } else {
            TX_SET_INTRA_1
        }
    }

    fn is_tx_type_in_set(&self, tx_set: usize, tx_type: u8) -> bool {
        if self.is_inter {
            TX_TYPE_IN_SET_INTER[tx_set][tx_type as usize] != 0
        } else {
            TX_TYPE_IN_SET_INTRA[tx_set][tx_type as usize] != 0
        }
    }

    fn compute_tx_type(&self, plane: usize, tx_sz: usize, block_x: usize, block_y: usize) -> u8 {
        let tx_sz_sqr_up = TX_SIZE_SQR_UP[tx_sz] as usize;
        if self.lossless || tx_sz_sqr_up > TX_32X32 {
            return DCT_DCT;
        }
        let tx_set = self.get_tx_set(tx_sz);
        if plane == 0 {
            return self.f.mi.tx_types[self.mi_idx(block_y, block_x)];
        }
        let tx_type = if self.is_inter {
            let x4 = self
                .mi_col
                .max(block_x << usize::from(self.seq.subsampling_x));
            let y4 = self
                .mi_row
                .max(block_y << usize::from(self.seq.subsampling_y));
            self.f.mi.tx_types[self.mi_idx(y4, x4)]
        } else {
            MODE_TO_TXFM[self.uv_mode as usize]
        };
        if !self.is_tx_type_in_set(tx_set, tx_type) {
            return DCT_DCT;
        }
        tx_type
    }

    fn set_tx_types(&mut self, x4: usize, y4: usize, tx_sz: usize, tx_type: u8) {
        let w4 = TX_WIDTH[tx_sz] as usize >> 2;
        let h4 = TX_HEIGHT[tx_sz] as usize >> 2;
        for j in 0..h4 {
            if y4 + j >= self.fh.mi_rows {
                break;
            }
            for i in 0..w4 {
                if x4 + i >= self.fh.mi_cols {
                    break;
                }
                let idx = self.mi_idx(y4 + j, x4 + i);
                self.f.mi.tx_types[idx] = tx_type;
            }
        }
    }

    fn transform_type(&mut self, x4: usize, y4: usize, tx_sz: usize) {
        let set = self.get_tx_set(tx_sz);
        let qidx = if self.fh.segmentation.enabled {
            self.fh
                .get_qindex(true, self.segment_id, self.current_q_index)
        } else {
            i32::from(self.fh.base_q_idx)
        };
        let tx_type = if set > 0 && qidx > 0 {
            let sqr = TX_SIZE_SQR[tx_sz] as usize;
            if self.is_inter {
                match set {
                    TX_SET_INTER_1 => {
                        let v = self.sd.symbol(&mut self.cdf.inter_tx_type_set1[sqr]);
                        TX_TYPE_INTER_INV_SET1[v]
                    }
                    TX_SET_INTER_2 => {
                        let v = self.sd.symbol(&mut self.cdf.inter_tx_type_set2);
                        TX_TYPE_INTER_INV_SET2[v]
                    }
                    _ => {
                        let v = self.sd.symbol(&mut self.cdf.inter_tx_type_set3[sqr]);
                        TX_TYPE_INTER_INV_SET3[v]
                    }
                }
            } else {
                let intra_dir = if self.use_filter_intra {
                    FILTER_INTRA_MODE_TO_INTRA_DIR[self.filter_intra_mode] as usize
                } else {
                    self.y_mode as usize
                };
                if set == TX_SET_INTRA_1 {
                    let v = self
                        .sd
                        .symbol(&mut self.cdf.intra_tx_type_set1[sqr][intra_dir]);
                    TX_TYPE_INTRA_INV_SET1[v]
                } else {
                    let v = self
                        .sd
                        .symbol(&mut self.cdf.intra_tx_type_set2[sqr][intra_dir]);
                    TX_TYPE_INTRA_INV_SET2[v]
                }
            }
        } else {
            DCT_DCT
        };
        self.set_tx_types(x4, y4, tx_sz, tx_type);
    }

    /// `coeffs( plane, startX, startY, txSz )`, returning `eob`.
    fn coeffs(&mut self, plane: usize, start_x: usize, start_y: usize, tx_sz: usize) -> usize {
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let w4 = TX_WIDTH[tx_sz] as usize >> 2;
        let h4 = TX_HEIGHT[tx_sz] as usize >> 2;
        let tx_sz_ctx = (TX_SIZE_SQR[tx_sz] as usize + TX_SIZE_SQR_UP[tx_sz] as usize + 1) >> 1;
        let ptype = usize::from(plane > 0);
        let seg_eob: usize = if tx_sz == TX_16X64 || tx_sz == TX_64X16 {
            512
        } else {
            1024.min(TX_WIDTH[tx_sz] as usize * TX_HEIGHT[tx_sz] as usize)
        };
        self.quant[..seg_eob].fill(0);
        let mut eob = 0usize;
        let mut cul_level = 0u32;
        let mut dc_category = 0u8;
        let all_zero_ctx = self.all_zero_ctx(plane, tx_sz, x4, y4, w4, h4);
        let all_zero = self
            .sd
            .symbol(&mut self.cdf.txb_skip[tx_sz_ctx][all_zero_ctx])
            == 1;
        if all_zero {
            if plane == 0 {
                self.set_tx_types(x4, y4, tx_sz, DCT_DCT);
            }
        } else {
            if plane == 0 {
                self.transform_type(x4, y4, tx_sz);
            }
            self.plane_tx_type = self.compute_tx_type(plane, tx_sz, x4, y4);
            let tx_class = get_tx_class(self.plane_tx_type);
            let scan = self.get_scan(tx_sz);
            let eob_multisize = (TX_WIDTH_LOG2[tx_sz] as usize).min(5)
                + (TX_HEIGHT_LOG2[tx_sz] as usize).min(5)
                - 4;
            let eob_ctx = usize::from(tx_class != TX_CLASS_2D);
            let eob_pt = 1 + match eob_multisize {
                0 => self.sd.symbol(&mut self.cdf.eob_pt_16[ptype][eob_ctx]),
                1 => self.sd.symbol(&mut self.cdf.eob_pt_32[ptype][eob_ctx]),
                2 => self.sd.symbol(&mut self.cdf.eob_pt_64[ptype][eob_ctx]),
                3 => self.sd.symbol(&mut self.cdf.eob_pt_128[ptype][eob_ctx]),
                4 => self.sd.symbol(&mut self.cdf.eob_pt_256[ptype][eob_ctx]),
                5 => self.sd.symbol(&mut self.cdf.eob_pt_512[ptype]),
                _ => self.sd.symbol(&mut self.cdf.eob_pt_1024[ptype]),
            };
            eob = if eob_pt < 2 {
                eob_pt
            } else {
                (1 << (eob_pt - 2)) + 1
            };
            let eob_shift = eob_pt as isize - 3;
            if eob_shift >= 0 {
                let eob_extra = self
                    .sd
                    .symbol(&mut self.cdf.eob_extra[tx_sz_ctx][ptype][eob_pt - 3]);
                if eob_extra != 0 {
                    eob += 1 << eob_shift;
                }
                let limit = (eob_pt as isize - 2).max(0) as usize;
                for i in 1..limit {
                    let eob_shift = limit - 1 - i;
                    if self.sd.literal(1) != 0 {
                        eob += 1 << eob_shift;
                    }
                }
            }
            let eob = eob.min(seg_eob);
            let adj_tx_sz = ADJUSTED_TX_SIZE[tx_sz] as usize;
            let bwl = TX_WIDTH_LOG2[adj_tx_sz] as usize;
            let width = 1usize << bwl;
            let height = TX_HEIGHT[adj_tx_sz] as usize;
            for c in (0..eob).rev() {
                let pos = scan[c] as usize;
                let mut level;
                if c == eob - 1 {
                    let ctx = self.coeff_base_eob_ctx(c, bwl, height);
                    level = self
                        .sd
                        .symbol(&mut self.cdf.coeff_base_eob[tx_sz_ctx][ptype][ctx])
                        as i32
                        + 1;
                } else {
                    let ctx = self.coeff_base_ctx(tx_sz, tx_class, pos, bwl, width, height);
                    level = self
                        .sd
                        .symbol(&mut self.cdf.coeff_base[tx_sz_ctx][ptype][ctx])
                        as i32;
                }
                if level > NUM_BASE_LEVELS {
                    let br_ctx = self.coeff_br_ctx(tx_class, pos, bwl, height);
                    let cdf_idx = tx_sz_ctx.min(TX_32X32);
                    for _ in 0..(COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)) {
                        let coeff_br = self
                            .sd
                            .symbol(&mut self.cdf.coeff_br[cdf_idx][ptype][br_ctx])
                            as i32;
                        level += coeff_br;
                        if coeff_br < BR_CDF_SIZE - 1 {
                            break;
                        }
                    }
                }
                self.quant[pos] = level;
            }
            for c in 0..eob {
                let pos = scan[c] as usize;
                let sign = if self.quant[pos] != 0 {
                    if c == 0 {
                        let ctx = self.dc_sign_ctx(plane, x4, y4, w4, h4);
                        self.sd.symbol(&mut self.cdf.dc_sign[ptype][ctx]) as u32
                    } else {
                        self.sd.literal(1)
                    }
                } else {
                    0
                };
                if self.quant[pos] > NUM_BASE_LEVELS + COEFF_BASE_RANGE {
                    let mut length = 0;
                    loop {
                        length += 1;
                        let golomb_length_bit = self.sd.literal(1);
                        if golomb_length_bit != 0 {
                            break;
                        }
                        if length >= 32 {
                            break;
                        }
                    }
                    let mut x: u32 = 1;
                    for _ in (0..length - 1).rev() {
                        let golomb_data_bit = self.sd.literal(1);
                        x = (x << 1) | golomb_data_bit;
                    }
                    self.quant[pos] = (x as i64 + (COEFF_BASE_RANGE + NUM_BASE_LEVELS) as i64)
                        .min(i32::MAX as i64) as i32;
                }
                if pos == 0 && self.quant[pos] > 0 {
                    dc_category = if sign != 0 { 1 } else { 2 };
                }
                self.quant[pos] &= 0xFFFFF;
                cul_level += self.quant[pos] as u32;
                if sign != 0 {
                    self.quant[pos] = -self.quant[pos];
                }
            }
            cul_level = cul_level.min(63);
            let _ = eob;
        }
        let eob = if all_zero { 0 } else { eob.min(seg_eob) };
        for i in 0..w4 {
            self.above_level_context[plane][x4 + i] = cul_level as u8;
            self.above_dc_context[plane][x4 + i] = dc_category;
        }
        for i in 0..h4 {
            self.left_level_context[plane][y4 + i] = cul_level as u8;
            self.left_dc_context[plane][y4 + i] = dc_category;
        }
        eob
    }

    fn all_zero_ctx(
        &self,
        plane: usize,
        tx_sz: usize,
        x4: usize,
        y4: usize,
        w4: usize,
        h4: usize,
    ) -> usize {
        let mut max_x4 = self.fh.mi_cols;
        let mut max_y4 = self.fh.mi_rows;
        if plane > 0 {
            max_x4 >>= usize::from(self.seq.subsampling_x);
            max_y4 >>= usize::from(self.seq.subsampling_y);
        }
        let w = TX_WIDTH[tx_sz] as usize;
        let h = TX_HEIGHT[tx_sz] as usize;
        let bsize = self.get_plane_residual_size(self.mi_size, plane);
        let bw = block_width(bsize);
        let bh = block_height(bsize);
        if plane == 0 {
            let mut top = 0u32;
            let mut left = 0u32;
            for k in 0..w4 {
                if x4 + k < max_x4 {
                    top = top.max(u32::from(self.above_level_context[plane][x4 + k]));
                }
            }
            for k in 0..h4 {
                if y4 + k < max_y4 {
                    left = left.max(u32::from(self.left_level_context[plane][y4 + k]));
                }
            }
            let top = top.min(255);
            let left = left.min(255);
            if bw == w && bh == h {
                0
            } else if top == 0 && left == 0 {
                1
            } else if top == 0 || left == 0 {
                2 + usize::from(top.max(left) > 3)
            } else if top.max(left) <= 3 {
                4
            } else if top.min(left) <= 3 {
                5
            } else {
                6
            }
        } else {
            let mut above = 0u8;
            let mut left = 0u8;
            for i in 0..w4 {
                if x4 + i < max_x4 {
                    above |= self.above_level_context[plane][x4 + i];
                    above |= self.above_dc_context[plane][x4 + i];
                }
            }
            for i in 0..h4 {
                if y4 + i < max_y4 {
                    left |= self.left_level_context[plane][y4 + i];
                    left |= self.left_dc_context[plane][y4 + i];
                }
            }
            let mut ctx = usize::from(above != 0) + usize::from(left != 0);
            ctx += 7;
            if bw * bh > w * h {
                ctx += 3;
            }
            ctx
        }
    }

    fn coeff_base_eob_ctx(&self, c: usize, bwl: usize, height: usize) -> usize {
        if c == 0 {
            return 0;
        }
        if c <= (height << bwl) / 8 {
            return 1;
        }
        if c <= (height << bwl) / 4 {
            return 2;
        }
        3
    }

    fn coeff_base_ctx(
        &self,
        tx_sz: usize,
        tx_class: usize,
        pos: usize,
        bwl: usize,
        width: usize,
        height: usize,
    ) -> usize {
        let row = pos >> bwl;
        let col = pos - (row << bwl);
        let mut mag = 0i32;
        for idx in 0..5 {
            let ref_row = row + SIG_REF_DIFF_OFFSET[tx_class][idx][0] as usize;
            let ref_col = col + SIG_REF_DIFF_OFFSET[tx_class][idx][1] as usize;
            if ref_row < height && ref_col < width {
                mag += self.quant[(ref_row << bwl) + ref_col].abs().min(3);
            }
        }
        let ctx = ((mag + 1) >> 1).min(4) as usize;
        if tx_class == TX_CLASS_2D {
            if row == 0 && col == 0 {
                return 0;
            }
            return ctx + COEFF_BASE_CTX_OFFSET[tx_sz][row.min(4)][col.min(4)] as usize;
        }
        let idx = if tx_class == TX_CLASS_VERT { row } else { col };
        ctx + COEFF_BASE_POS_CTX_OFFSET[idx.min(2)] as usize
    }

    fn coeff_br_ctx(&self, tx_class: usize, pos: usize, bwl: usize, txh: usize) -> usize {
        let txw = 1usize << bwl;
        let row = pos >> bwl;
        let col = pos - (row << bwl);
        let mut mag = 0i32;
        for idx in 0..3 {
            let ref_row = row + MAG_REF_OFFSET_WITH_TX_CLASS[tx_class][idx][0] as usize;
            let ref_col = col + MAG_REF_OFFSET_WITH_TX_CLASS[tx_class][idx][1] as usize;
            if ref_row < txh && ref_col < txw {
                mag +=
                    self.quant[ref_row * txw + ref_col].min(COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1);
            }
        }
        let mag = ((mag + 1) >> 1).min(6) as usize;
        if pos == 0 {
            mag
        } else if tx_class == TX_CLASS_2D {
            if row < 2 && col < 2 {
                mag + 7
            } else {
                mag + 14
            }
        } else if tx_class == TX_CLASS_HORIZ {
            if col == 0 { mag + 7 } else { mag + 14 }
        } else if row == 0 {
            mag + 7
        } else {
            mag + 14
        }
    }

    fn dc_sign_ctx(&self, plane: usize, x4: usize, y4: usize, w4: usize, h4: usize) -> usize {
        let mut max_x4 = self.fh.mi_cols;
        let mut max_y4 = self.fh.mi_rows;
        if plane > 0 {
            max_x4 >>= usize::from(self.seq.subsampling_x);
            max_y4 >>= usize::from(self.seq.subsampling_y);
        }
        let mut dc_sign = 0i32;
        for k in 0..w4 {
            if x4 + k < max_x4 {
                match self.above_dc_context[plane][x4 + k] {
                    1 => dc_sign -= 1,
                    2 => dc_sign += 1,
                    _ => {}
                }
            }
        }
        for k in 0..h4 {
            if y4 + k < max_y4 {
                match self.left_dc_context[plane][y4 + k] {
                    1 => dc_sign -= 1,
                    2 => dc_sign += 1,
                    _ => {}
                }
            }
        }
        if dc_sign < 0 {
            1
        } else if dc_sign > 0 {
            2
        } else {
            0
        }
    }

    /// `palette_mode_info( )`.
    pub(crate) fn palette_mode_info(&mut self) {
        let bsize_ctx =
            MI_WIDTH_LOG2[self.mi_size] as usize + MI_HEIGHT_LOG2[self.mi_size] as usize - 2;
        let bit_depth = u32::from(self.seq.bit_depth);
        if self.y_mode == DC_PRED {
            let mut ctx = 0;
            if self.avail_u
                && self.f.mi.palette_sizes[0][self.mi_idx(self.mi_row - 1, self.mi_col)] > 0
            {
                ctx += 1;
            }
            if self.avail_l
                && self.f.mi.palette_sizes[0][self.mi_idx(self.mi_row, self.mi_col - 1)] > 0
            {
                ctx += 1;
            }
            let has_palette_y = self.sd.symbol(&mut self.cdf.palette_y_mode[bsize_ctx][ctx]) == 1;
            if has_palette_y {
                self.palette_size_y = self.sd.symbol(&mut self.cdf.palette_y_size[bsize_ctx]) + 2;
                let (cache, cache_n) = self.get_palette_cache(0);
                let mut idx = 0;
                let mut i = 0;
                while i < cache_n && idx < self.palette_size_y {
                    if self.sd.literal(1) != 0 {
                        self.palette_colors_y[idx] = cache[i];
                        idx += 1;
                    }
                    i += 1;
                }
                if idx < self.palette_size_y {
                    self.palette_colors_y[idx] = self.sd.literal(bit_depth) as u16;
                    idx += 1;
                }
                let mut palette_bits = 0;
                if idx < self.palette_size_y {
                    let min_bits = bit_depth - 3;
                    palette_bits = min_bits + self.sd.literal(2);
                }
                while idx < self.palette_size_y {
                    let delta = self.sd.literal(palette_bits) as i32 + 1;
                    let v = (i32::from(self.palette_colors_y[idx - 1]) + delta)
                        .clamp(0, (1 << bit_depth) - 1);
                    self.palette_colors_y[idx] = v as u16;
                    let range = (1i32 << bit_depth) - v - 1;
                    palette_bits = palette_bits.min(ceil_log2(range.max(0) as u32));
                    idx += 1;
                }
                self.palette_colors_y[..self.palette_size_y].sort_unstable();
            }
        }
        if self.has_chroma && self.uv_mode == DC_PRED {
            let ctx = usize::from(self.palette_size_y > 0);
            let has_palette_uv = self.sd.symbol(&mut self.cdf.palette_uv_mode[ctx]) == 1;
            if has_palette_uv {
                self.palette_size_uv = self.sd.symbol(&mut self.cdf.palette_uv_size[bsize_ctx]) + 2;
                let (cache, cache_n) = self.get_palette_cache(1);
                let mut idx = 0;
                let mut i = 0;
                while i < cache_n && idx < self.palette_size_uv {
                    if self.sd.literal(1) != 0 {
                        self.palette_colors_u[idx] = cache[i];
                        idx += 1;
                    }
                    i += 1;
                }
                if idx < self.palette_size_uv {
                    self.palette_colors_u[idx] = self.sd.literal(bit_depth) as u16;
                    idx += 1;
                }
                let mut palette_bits = 0;
                if idx < self.palette_size_uv {
                    let min_bits = bit_depth - 3;
                    palette_bits = min_bits + self.sd.literal(2);
                }
                while idx < self.palette_size_uv {
                    let delta = self.sd.literal(palette_bits) as i32;
                    let v = (i32::from(self.palette_colors_u[idx - 1]) + delta)
                        .clamp(0, (1 << bit_depth) - 1);
                    self.palette_colors_u[idx] = v as u16;
                    let range = (1i32 << bit_depth) - v;
                    palette_bits = palette_bits.min(ceil_log2(range.max(0) as u32));
                    idx += 1;
                }
                self.palette_colors_u[..self.palette_size_uv].sort_unstable();
                if self.sd.literal(1) != 0 {
                    let min_bits = bit_depth - 4;
                    let max_val = 1i32 << bit_depth;
                    let palette_bits = min_bits + self.sd.literal(2);
                    self.palette_colors_v[0] = self.sd.literal(bit_depth) as u16;
                    for idx in 1..self.palette_size_uv {
                        let mut delta = self.sd.literal(palette_bits) as i32;
                        if delta != 0 && self.sd.literal(1) != 0 {
                            delta = -delta;
                        }
                        let mut val = i32::from(self.palette_colors_v[idx - 1]) + delta;
                        if val < 0 {
                            val += max_val;
                        }
                        if val >= max_val {
                            val -= max_val;
                        }
                        self.palette_colors_v[idx] = val.clamp(0, (1 << bit_depth) - 1) as u16;
                    }
                } else {
                    for idx in 0..self.palette_size_uv {
                        self.palette_colors_v[idx] = self.sd.literal(bit_depth) as u16;
                    }
                }
            }
        }
    }

    fn get_palette_cache(&self, plane: usize) -> ([u16; 2 * PALETTE_COLORS], usize) {
        let mut cache = [0u16; 2 * PALETTE_COLORS];
        let above_n = if (self.mi_row * MI_SIZE) % 64 != 0 && self.avail_u {
            self.f.mi.palette_sizes[plane][self.mi_idx(self.mi_row - 1, self.mi_col)] as usize
        } else {
            0
        };
        let left_n = if self.avail_l {
            self.f.mi.palette_sizes[plane][self.mi_idx(self.mi_row, self.mi_col - 1)] as usize
        } else {
            0
        };
        let above_colors = if above_n > 0 {
            self.f.mi.palette_colors[plane][self.mi_idx(self.mi_row - 1, self.mi_col)]
        } else {
            [0; PALETTE_COLORS]
        };
        let left_colors = if left_n > 0 {
            self.f.mi.palette_colors[plane][self.mi_idx(self.mi_row, self.mi_col - 1)]
        } else {
            [0; PALETTE_COLORS]
        };
        let mut above_idx = 0;
        let mut left_idx = 0;
        let mut n = 0;
        while above_idx < above_n && left_idx < left_n {
            let above_c = above_colors[above_idx];
            let left_c = left_colors[left_idx];
            if left_c < above_c {
                if n == 0 || left_c != cache[n - 1] {
                    cache[n] = left_c;
                    n += 1;
                }
                left_idx += 1;
            } else {
                if n == 0 || above_c != cache[n - 1] {
                    cache[n] = above_c;
                    n += 1;
                }
                above_idx += 1;
                if left_c == above_c {
                    left_idx += 1;
                }
            }
        }
        while above_idx < above_n {
            let val = above_colors[above_idx];
            above_idx += 1;
            if n == 0 || val != cache[n - 1] {
                cache[n] = val;
                n += 1;
            }
        }
        while left_idx < left_n {
            let val = left_colors[left_idx];
            left_idx += 1;
            if n == 0 || val != cache[n - 1] {
                cache[n] = val;
                n += 1;
            }
        }
        (cache, n)
    }

    /// `palette_tokens( )`.
    pub(crate) fn palette_tokens(&mut self) {
        let mut block_height = block_height(self.mi_size);
        let mut block_width = block_width(self.mi_size);
        let mut onscreen_height = block_height.min((self.fh.mi_rows - self.mi_row) * MI_SIZE);
        let mut onscreen_width = block_width.min((self.fh.mi_cols - self.mi_col) * MI_SIZE);
        if self.palette_size_y > 0 {
            let n = self.palette_size_y;
            self.color_map_y[0][0] = self.sd.ns(n as u32) as u8;
            self.read_color_map(
                false,
                n,
                onscreen_width,
                onscreen_height,
                block_width,
                block_height,
            );
        }
        if self.palette_size_uv > 0 {
            let n = self.palette_size_uv;
            self.color_map_uv[0][0] = self.sd.ns(n as u32) as u8;
            block_height >>= usize::from(self.seq.subsampling_y);
            block_width >>= usize::from(self.seq.subsampling_x);
            onscreen_height >>= usize::from(self.seq.subsampling_y);
            onscreen_width >>= usize::from(self.seq.subsampling_x);
            if block_width < 4 {
                block_width += 2;
                onscreen_width += 2;
            }
            if block_height < 4 {
                block_height += 2;
                onscreen_height += 2;
            }
            self.read_color_map(
                true,
                n,
                onscreen_width,
                onscreen_height,
                block_width,
                block_height,
            );
        }
    }

    fn read_color_map(
        &mut self,
        uv: bool,
        n: usize,
        onscreen_width: usize,
        onscreen_height: usize,
        block_width: usize,
        block_height: usize,
    ) {
        for i in 1..onscreen_height + onscreen_width - 1 {
            let j_start = i.min(onscreen_width - 1) as isize;
            let j_end = (i as isize - onscreen_height as isize + 1).max(0);
            let mut j = j_start;
            while j >= j_end {
                let (r, c) = (i - j as usize, j as usize);
                let (color_order, ctx) = {
                    let map = if uv {
                        &*self.color_map_uv
                    } else {
                        &*self.color_map_y
                    };
                    palette_color_context(map, r, c, n)
                };
                let symbol = if uv {
                    self.read_palette_color_idx_uv(n, ctx)
                } else {
                    self.read_palette_color_idx_y(n, ctx)
                };
                let map = if uv {
                    &mut *self.color_map_uv
                } else {
                    &mut *self.color_map_y
                };
                map[r][c] = color_order[symbol];
                j -= 1;
            }
        }
        let map = if uv {
            &mut *self.color_map_uv
        } else {
            &mut *self.color_map_y
        };
        for i in 0..onscreen_height {
            for j in onscreen_width..block_width {
                map[i][j] = map[i][onscreen_width - 1];
            }
        }
        for i in onscreen_height..block_height {
            for j in 0..block_width {
                map[i][j] = map[onscreen_height - 1][j];
            }
        }
    }

    fn read_palette_color_idx_y(&mut self, n: usize, ctx: usize) -> usize {
        match n {
            2 => self.sd.symbol(&mut self.cdf.palette_size_2_y_color[ctx]),
            3 => self.sd.symbol(&mut self.cdf.palette_size_3_y_color[ctx]),
            4 => self.sd.symbol(&mut self.cdf.palette_size_4_y_color[ctx]),
            5 => self.sd.symbol(&mut self.cdf.palette_size_5_y_color[ctx]),
            6 => self.sd.symbol(&mut self.cdf.palette_size_6_y_color[ctx]),
            7 => self.sd.symbol(&mut self.cdf.palette_size_7_y_color[ctx]),
            _ => self.sd.symbol(&mut self.cdf.palette_size_8_y_color[ctx]),
        }
    }

    fn read_palette_color_idx_uv(&mut self, n: usize, ctx: usize) -> usize {
        match n {
            2 => self.sd.symbol(&mut self.cdf.palette_size_2_uv_color[ctx]),
            3 => self.sd.symbol(&mut self.cdf.palette_size_3_uv_color[ctx]),
            4 => self.sd.symbol(&mut self.cdf.palette_size_4_uv_color[ctx]),
            5 => self.sd.symbol(&mut self.cdf.palette_size_5_uv_color[ctx]),
            6 => self.sd.symbol(&mut self.cdf.palette_size_6_uv_color[ctx]),
            7 => self.sd.symbol(&mut self.cdf.palette_size_7_uv_color[ctx]),
            _ => self.sd.symbol(&mut self.cdf.palette_size_8_uv_color[ctx]),
        }
    }
}

/// `get_palette_color_context( )`, returning `ColorOrder` and the CDF context.
fn palette_color_context(
    color_map: &[[u8; 64]; 64],
    r: usize,
    c: usize,
    n: usize,
) -> ([u8; PALETTE_COLORS], usize) {
    let mut scores = [0u32; PALETTE_COLORS];
    let mut color_order = [0u8; PALETTE_COLORS];
    for (i, order) in color_order.iter_mut().enumerate() {
        *order = i as u8;
    }
    if c > 0 {
        scores[color_map[r][c - 1] as usize] += 2;
    }
    if r > 0 && c > 0 {
        scores[color_map[r - 1][c - 1] as usize] += 1;
    }
    if r > 0 {
        scores[color_map[r - 1][c] as usize] += 2;
    }
    for i in 0..PALETTE_NUM_NEIGHBORS {
        let mut max_score = scores[i];
        let mut max_idx = i;
        for j in i + 1..n {
            if scores[j] > max_score {
                max_score = scores[j];
                max_idx = j;
            }
        }
        if max_idx != i {
            let max_score = scores[max_idx];
            let max_color_order = color_order[max_idx];
            let mut k = max_idx;
            while k > i {
                scores[k] = scores[k - 1];
                color_order[k] = color_order[k - 1];
                k -= 1;
            }
            scores[i] = max_score;
            color_order[i] = max_color_order;
        }
    }
    let mut hash = 0usize;
    for i in 0..PALETTE_NUM_NEIGHBORS {
        hash += scores[i] as usize * PALETTE_COLOR_HASH_MULTIPLIERS[i] as usize;
    }
    let ctx = PALETTE_COLOR_CONTEXT[hash.min(8)].max(0) as usize;
    (color_order, ctx)
}

fn ceil_log2(x: u32) -> u32 {
    if x < 2 {
        return 0;
    }
    let mut i = 1;
    let mut p = 2;
    while p < x {
        i += 1;
        p <<= 1;
    }
    i
}
