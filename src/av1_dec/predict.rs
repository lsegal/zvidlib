//! The prediction processes (specification section 7.11): intra, inter
//! (including warped, OBMC and masked compound prediction), palette and
//! chroma-from-luma prediction.

use std::sync::OnceLock;

use super::consts::*;
use super::recon::ReconScratch;
use super::tables::*;
use super::tile::{TileDecoder, bh4, bw4, is_directional_mode};
use crate::Result;

const EDGE_OFFSET: usize = 16;
const EDGE_LEN: usize = EDGE_OFFSET + 2 * 256 + 16;

/// Reusable per-tile prediction buffers.
pub(crate) struct PredScratch {
    pub(crate) recon: ReconScratch,
    preds: [Vec<i32>; 2],
    mask: Vec<i32>,
    obmc_pred: Vec<i32>,
    intermediate: Vec<i32>,
    above_row: Vec<i32>,
    left_col: Vec<i32>,
    pred: Vec<i32>,
}

impl PredScratch {
    pub(crate) fn new() -> Self {
        Self {
            recon: ReconScratch::new(),
            preds: [vec![0; 128 * 128], vec![0; 128 * 128]],
            mask: vec![0; 128 * 128],
            obmc_pred: vec![0; 128 * 128],
            intermediate: vec![0; (128 * 2 + 16 + 8) * 128],
            above_row: vec![0; EDGE_LEN],
            left_col: vec![0; EDGE_LEN],
            pred: vec![0; 64 * 64],
        }
    }
}

#[inline(always)]
fn round2(x: i64, n: u32) -> i64 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

#[inline(always)]
fn round2signed(x: i64, n: u32) -> i64 {
    if x >= 0 { round2(x, n) } else { -round2(-x, n) }
}

fn floor_log2(x: u64) -> u32 {
    63 - x.leading_zeros()
}

fn sm_weights(log2: usize) -> &'static [u8] {
    match log2 {
        2 => &SM_WEIGHTS_TX_4X4,
        3 => &SM_WEIGHTS_TX_8X8,
        4 => &SM_WEIGHTS_TX_16X16,
        5 => &SM_WEIGHTS_TX_32X32,
        _ => &SM_WEIGHTS_TX_64X64,
    }
}

/// `resolve_divisor( d )`, returning `(divShift, divFactor)`.
fn resolve_divisor(d: i64) -> (i32, i64) {
    let n = floor_log2(d.unsigned_abs()) as i32;
    let e = d.abs() - (1i64 << n);
    let f = if n > DIV_LUT_BITS {
        round2(e, (n - DIV_LUT_BITS) as u32)
    } else {
        e << (DIV_LUT_BITS - n)
    };
    let div_shift = n + DIV_LUT_PREC_BITS;
    let div_factor = if d < 0 {
        -i64::from(DIV_LUT[f as usize])
    } else {
        i64::from(DIV_LUT[f as usize])
    };
    (div_shift, div_factor)
}

/// `setup_shear( warpParams )`, returning `(warpValid, alpha, beta, gamma, delta)`.
pub(crate) fn setup_shear(warp_params: &[i32; 6]) -> (bool, i32, i32, i32, i32) {
    let alpha0 = (warp_params[2] - (1 << WARPEDMODEL_PREC_BITS)).clamp(-32768, 32767);
    let beta0 = warp_params[3].clamp(-32768, 32767);
    let (div_shift, div_factor) = resolve_divisor(i64::from(warp_params[2]));
    let v = i64::from(warp_params[4]) << WARPEDMODEL_PREC_BITS;
    let gamma0 = round2signed(v * div_factor, div_shift as u32).clamp(-32768, 32767) as i32;
    let w = i64::from(warp_params[3]) * i64::from(warp_params[4]);
    let delta0 = (i64::from(warp_params[5])
        - round2signed(w * div_factor, div_shift as u32)
        - (1 << WARPEDMODEL_PREC_BITS))
        .clamp(-32768, 32767) as i32;
    let reduce = |x: i32| {
        (round2signed(i64::from(x), WARP_PARAM_REDUCE_BITS as u32) << WARP_PARAM_REDUCE_BITS) as i32
    };
    let alpha = reduce(alpha0);
    let beta = reduce(beta0);
    let gamma = reduce(gamma0);
    let delta = reduce(delta0);
    let mut valid = true;
    if 4 * alpha.abs() + 7 * beta.abs() >= (1 << WARPEDMODEL_PREC_BITS) {
        valid = false;
    }
    if 4 * gamma.abs() + 4 * delta.abs() >= (1 << WARPEDMODEL_PREC_BITS) {
        valid = false;
    }
    (valid, alpha, beta, gamma, delta)
}

/// `WedgeMasks[ bsize ][ flipSign ][ wedge ]`, flattened per block size.
struct WedgeMasks {
    masks: Vec<Option<Vec<u8>>>,
}

fn wedge_masks() -> &'static WedgeMasks {
    static MASKS: OnceLock<WedgeMasks> = OnceLock::new();
    MASKS.get_or_init(|| {
        const WEDGE_VERTICAL: usize = 1;
        const WEDGE_OBLIQUE27: usize = 2;
        const WEDGE_OBLIQUE63: usize = 3;
        const WEDGE_OBLIQUE117: usize = 4;
        const WEDGE_OBLIQUE153: usize = 5;
        const WEDGE_HORIZONTAL: usize = 0;
        let n = MASK_MASTER_SIZE;
        let mut master = vec![[[0u8; MASK_MASTER_SIZE]; MASK_MASTER_SIZE]; 6];
        for j in 0..n {
            let mut shift = (MASK_MASTER_SIZE / 4) as isize;
            let mut i = 0;
            while i < n {
                master[WEDGE_OBLIQUE63][i][j] = WEDGE_MASTER_OBLIQUE_EVEN
                    [(j as isize - shift).clamp(0, n as isize - 1) as usize];
                shift -= 1;
                master[WEDGE_OBLIQUE63][i + 1][j] = WEDGE_MASTER_OBLIQUE_ODD
                    [(j as isize - shift).clamp(0, n as isize - 1) as usize];
                master[WEDGE_VERTICAL][i][j] = WEDGE_MASTER_VERTICAL[j];
                master[WEDGE_VERTICAL][i + 1][j] = WEDGE_MASTER_VERTICAL[j];
                i += 2;
            }
        }
        for i in 0..n {
            for j in 0..n {
                let msk = master[WEDGE_OBLIQUE63][i][j];
                master[WEDGE_OBLIQUE27][j][i] = msk;
                master[WEDGE_OBLIQUE117][i][n - 1 - j] = 64 - msk;
                master[WEDGE_OBLIQUE153][n - 1 - j][i] = 64 - msk;
                master[WEDGE_HORIZONTAL][j][i] = master[WEDGE_VERTICAL][i][j];
            }
        }
        let mut masks = vec![None; BLOCK_SIZES];
        for (bsize, slot) in masks.iter_mut().enumerate().skip(BLOCK_8X8) {
            if WEDGE_BITS[bsize] == 0 {
                continue;
            }
            let w = block_width(bsize);
            let h = block_height(bsize);
            let mut out = vec![0u8; 2 * 16 * w * h];
            let shape = if bh4(bsize) > bw4(bsize) {
                0
            } else if bh4(bsize) < bw4(bsize) {
                1
            } else {
                2
            };
            for wedge in 0..16 {
                let code = WEDGE_CODEBOOK[shape][wedge];
                let dir = code[0] as usize;
                let xoff = MASK_MASTER_SIZE / 2 - ((code[1] as usize * w) >> 3);
                let yoff = MASK_MASTER_SIZE / 2 - ((code[2] as usize * h) >> 3);
                let mut sum = 0usize;
                for i in 0..w {
                    sum += master[dir][yoff][xoff + i] as usize;
                }
                for i in 1..h {
                    sum += master[dir][yoff + i][xoff] as usize;
                }
                let avg = (sum + (w + h - 1) / 2) / (w + h - 1);
                let flip_sign = usize::from(avg < 32);
                for i in 0..h {
                    for j in 0..w {
                        let m = master[dir][yoff + i][xoff + j];
                        out[((flip_sign * 16 + wedge) * h + i) * w + j] = m;
                        out[(((1 - flip_sign) * 16 + wedge) * h + i) * w + j] = 64 - m;
                    }
                }
            }
            *slot = Some(out);
        }
        WedgeMasks { masks }
    })
}

impl TileDecoder<'_> {
    /// `compute_prediction( )`.
    pub(crate) fn compute_prediction(&mut self) -> Result<()> {
        let sb_mask = if self.seq.use_128x128_superblock {
            31
        } else {
            15
        };
        let sub_block_mi_row = (self.mi_row & sb_mask) as isize;
        let sub_block_mi_col = (self.mi_col & sb_mask) as isize;
        for plane in 0..1 + usize::from(self.has_chroma) * 2 {
            let plane_sz = self.get_plane_residual_size(self.mi_size, plane);
            let num4x4_w = bw4(plane_sz);
            let num4x4_h = bh4(plane_sz);
            let log2w = MI_SIZE_LOG2 + MI_WIDTH_LOG2[plane_sz] as usize;
            let log2h = MI_SIZE_LOG2 + MI_HEIGHT_LOG2[plane_sz] as usize;
            let sub_x = self.sub_x(plane);
            let sub_y = self.sub_y(plane);
            let base_x = (self.mi_col >> sub_x) * MI_SIZE;
            let base_y = (self.mi_row >> sub_y) * MI_SIZE;
            let mut cand_row = (self.mi_row >> sub_y) << sub_y;
            let mut cand_col = (self.mi_col >> sub_x) << sub_x;
            let is_inter_intra = self.is_inter && self.ref_frame[1] == INTRA_FRAME;
            if is_inter_intra {
                let mode = match self.interintra_mode {
                    II_DC_PRED => DC_PRED,
                    II_V_PRED => V_PRED,
                    II_H_PRED => H_PRED,
                    _ => SMOOTH_PRED,
                };
                let have_above_right = self.block_decoded(
                    plane,
                    (sub_block_mi_row >> sub_y) - 1,
                    (sub_block_mi_col >> sub_x) + num4x4_w as isize,
                );
                let have_below_left = self.block_decoded(
                    plane,
                    (sub_block_mi_row >> sub_y) + num4x4_h as isize,
                    (sub_block_mi_col >> sub_x) - 1,
                );
                self.predict_intra(
                    plane,
                    base_x,
                    base_y,
                    if plane == 0 {
                        self.avail_l
                    } else {
                        self.avail_l_chroma
                    },
                    if plane == 0 {
                        self.avail_u
                    } else {
                        self.avail_u_chroma
                    },
                    have_above_right,
                    have_below_left,
                    mode,
                    log2w,
                    log2h,
                );
            }
            if self.is_inter {
                let mut pred_w = block_width(self.mi_size) >> sub_x;
                let mut pred_h = block_height(self.mi_size) >> sub_y;
                let mut some_use_intra = false;
                for r in 0..(num4x4_h << sub_y) {
                    for c in 0..(num4x4_w << sub_x) {
                        let row = cand_row + r;
                        let col = cand_col + c;
                        if row < self.fh.mi_rows
                            && col < self.fh.mi_cols
                            && self.f.mi.ref_frames[self.mi_idx(row, col)][0] == INTRA_FRAME
                        {
                            some_use_intra = true;
                        }
                    }
                }
                if some_use_intra {
                    pred_w = num4x4_w * 4;
                    pred_h = num4x4_h * 4;
                    cand_row = self.mi_row;
                    cand_col = self.mi_col;
                }
                let mut r = 0;
                let mut y = 0;
                while y < num4x4_h * 4 {
                    let mut c = 0;
                    let mut x = 0;
                    while x < num4x4_w * 4 {
                        self.predict_inter(
                            plane,
                            base_x + x,
                            base_y + y,
                            pred_w,
                            pred_h,
                            cand_row + r,
                            cand_col + c,
                        )?;
                        x += pred_w;
                        c += 1;
                    }
                    y += pred_h;
                    r += 1;
                }
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Intra prediction (section 7.11.2)
    // ---------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict_intra(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        have_left: bool,
        have_above: bool,
        have_above_right: bool,
        have_below_left: bool,
        mode: u8,
        log2w: usize,
        log2h: usize,
    ) {
        let w = 1usize << log2w;
        let h = 1usize << log2h;
        let sub_x = self.sub_x(plane);
        let sub_y = self.sub_y(plane);
        let max_x = ((self.fh.mi_cols * MI_SIZE) >> sub_x) as isize - 1;
        let max_y = ((self.fh.mi_rows * MI_SIZE) >> sub_y) as isize - 1;
        let bit_depth = u32::from(self.seq.bit_depth);
        let mut above_row = std::mem::take(&mut self.scratch.above_row);
        let mut left_col = std::mem::take(&mut self.scratch.left_col);
        {
            let frame = &self.f.curr.planes[plane];
            let base = |v: i32| v;
            let o = EDGE_OFFSET;
            for i in 0..w + h {
                above_row[o + i] = if !have_above && have_left {
                    base(i32::from(frame.get(x - 1, y)))
                } else if !have_above && !have_left {
                    (1 << (bit_depth - 1)) - 1
                } else {
                    let above_limit = max_x
                        .min(x as isize + if have_above_right { 2 * w } else { w } as isize - 1);
                    let xx = above_limit.min((x + i) as isize) as usize;
                    i32::from(frame.get(xx, y - 1))
                };
                left_col[o + i] = if !have_left && have_above {
                    i32::from(frame.get(x, y - 1))
                } else if !have_left && !have_above {
                    (1 << (bit_depth - 1)) + 1
                } else {
                    let left_limit = max_y
                        .min(y as isize + if have_below_left { 2 * h } else { h } as isize - 1);
                    let yy = left_limit.min((y + i) as isize) as usize;
                    i32::from(frame.get(x - 1, yy))
                };
            }
            let corner = if have_above && have_left {
                i32::from(frame.get(x - 1, y - 1))
            } else if have_above {
                i32::from(frame.get(x, y - 1))
            } else if have_left {
                i32::from(frame.get(x - 1, y))
            } else {
                1 << (bit_depth - 1)
            };
            above_row[o - 1] = corner;
            left_col[o - 1] = corner;
        }
        let mut pred = std::mem::take(&mut self.scratch.pred);
        if plane == 0 && self.use_filter_intra {
            self.recursive_intra_prediction(&above_row, &left_col, &mut pred, w, h);
        } else if is_directional_mode(mode) {
            self.directional_intra_prediction(
                plane,
                x,
                y,
                have_left,
                have_above,
                mode,
                w,
                h,
                max_x,
                max_y,
                &mut above_row,
                &mut left_col,
                &mut pred,
            );
        } else if mode == SMOOTH_PRED || mode == SMOOTH_V_PRED || mode == SMOOTH_H_PRED {
            smooth_intra_prediction(mode, log2w, log2h, &above_row, &left_col, &mut pred);
        } else if mode == DC_PRED {
            dc_intra_prediction(
                have_left, have_above, log2w, log2h, &above_row, &left_col, &mut pred, bit_depth,
            );
        } else {
            paeth_intra_prediction(w, h, &above_row, &left_col, &mut pred);
        }
        let frame = &mut self.f.curr.planes[plane];
        for i in 0..h {
            if y + i >= frame.height {
                break;
            }
            let row = frame.row_mut(y + i);
            for j in 0..w {
                if x + j < row.len() {
                    row[x + j] = pred[i * w + j] as u16;
                }
            }
        }
        self.scratch.above_row = above_row;
        self.scratch.left_col = left_col;
        self.scratch.pred = pred;
    }

    fn recursive_intra_prediction(
        &self,
        above_row: &[i32],
        left_col: &[i32],
        pred: &mut [i32],
        w: usize,
        h: usize,
    ) {
        let o = EDGE_OFFSET as isize;
        let w4 = w >> 2;
        let h2 = h >> 1;
        let max = (1i64 << self.seq.bit_depth) - 1;
        for i2 in 0..h2 {
            for j4 in 0..w4 {
                let mut p = [0i32; 7];
                for (i, pv) in p.iter_mut().enumerate() {
                    *pv = if i < 5 {
                        if i2 == 0 {
                            above_row[(o + ((j4 << 2) + i) as isize - 1) as usize]
                        } else if j4 == 0 && i == 0 {
                            left_col[(o + ((i2 << 1) as isize) - 1) as usize]
                        } else {
                            pred[((i2 << 1) - 1) * w + (j4 << 2) + i - 1]
                        }
                    } else if j4 == 0 {
                        left_col[(o + ((i2 << 1) + i - 5) as isize) as usize]
                    } else {
                        pred[((i2 << 1) + i - 5) * w + (j4 << 2) - 1]
                    };
                }
                for i1 in 0..2 {
                    for j1 in 0..4 {
                        let mut pr = 0i64;
                        for (i, pv) in p.iter().enumerate() {
                            pr += i64::from(
                                INTRA_FILTER_TAPS[self.filter_intra_mode][(i1 << 2) + j1][i],
                            ) * i64::from(*pv);
                        }
                        pred[((i2 << 1) + i1) * w + (j4 << 2) + j1] =
                            round2signed(pr, INTRA_FILTER_SCALE_BITS as u32).clamp(0, max) as i32;
                    }
                }
            }
        }
    }

    fn intra_filter_type(&self, plane: usize) -> bool {
        let mut above_smooth = false;
        let mut left_smooth = false;
        let avail_u = if plane == 0 {
            self.avail_u
        } else {
            self.avail_u_chroma
        };
        let avail_l = if plane == 0 {
            self.avail_l
        } else {
            self.avail_l_chroma
        };
        if avail_u {
            let mut r = self.mi_row as isize - 1;
            let mut c = self.mi_col as isize;
            if plane > 0 {
                if self.seq.subsampling_x && (self.mi_col & 1) == 0 {
                    c += 1;
                }
                if self.seq.subsampling_y && (self.mi_row & 1) == 1 {
                    r -= 1;
                }
            }
            above_smooth = self.is_smooth(r as usize, c as usize, plane);
        }
        if avail_l {
            let mut r = self.mi_row as isize;
            let mut c = self.mi_col as isize - 1;
            if plane > 0 {
                if self.seq.subsampling_x && (self.mi_col & 1) == 1 {
                    c -= 1;
                }
                if self.seq.subsampling_y && (self.mi_row & 1) == 0 {
                    r += 1;
                }
            }
            left_smooth = self.is_smooth(r as usize, c as usize, plane);
        }
        above_smooth || left_smooth
    }

    fn is_smooth(&self, row: usize, col: usize, plane: usize) -> bool {
        let row = row.min(self.fh.mi_rows - 1);
        let col = col.min(self.fh.mi_cols - 1);
        let i = self.mi_idx(row, col);
        let mode = if plane == 0 {
            self.f.mi.y_modes[i]
        } else {
            if self.f.mi.ref_frames[i][0] > INTRA_FRAME {
                return false;
            }
            self.f.mi.uv_modes[i]
        };
        mode == SMOOTH_PRED || mode == SMOOTH_V_PRED || mode == SMOOTH_H_PRED
    }

    #[allow(clippy::too_many_arguments)]
    fn directional_intra_prediction(
        &self,
        plane: usize,
        x: usize,
        y: usize,
        have_left: bool,
        have_above: bool,
        mode: u8,
        w: usize,
        h: usize,
        max_x: isize,
        max_y: isize,
        above_row: &mut [i32],
        left_col: &mut [i32],
        pred: &mut [i32],
    ) {
        let o = EDGE_OFFSET as isize;
        let angle_delta = if plane == 0 {
            self.angle_delta_y
        } else {
            self.angle_delta_uv
        };
        let p_angle = i32::from(MODE_TO_ANGLE[mode as usize]) + angle_delta * ANGLE_STEP;
        let mut upsample_above = 0u32;
        let mut upsample_left = 0u32;
        let bit_depth = u32::from(self.seq.bit_depth);
        if self.seq.enable_intra_edge_filter {
            if p_angle != 90 && p_angle != 180 {
                if p_angle > 90 && p_angle < 180 && (w + h) >= 24 {
                    let s = left_col[o as usize] * 5
                        + above_row[(o - 1) as usize] * 6
                        + above_row[o as usize] * 5;
                    let v = round2(i64::from(s), 4) as i32;
                    left_col[(o - 1) as usize] = v;
                    above_row[(o - 1) as usize] = v;
                }
                let filter_type = self.intra_filter_type(plane);
                if have_above {
                    let strength = intra_edge_filter_strength(w, h, filter_type, p_angle - 90);
                    let num_px = w.min((max_x - x as isize + 1) as usize)
                        + if p_angle < 90 { h } else { 0 }
                        + 1;
                    intra_edge_filter(above_row, num_px, strength);
                }
                if have_left {
                    let strength = intra_edge_filter_strength(w, h, filter_type, p_angle - 180);
                    let num_px = h.min((max_y - y as isize + 1) as usize)
                        + if p_angle > 180 { w } else { 0 }
                        + 1;
                    intra_edge_filter(left_col, num_px, strength);
                }
            }
            let filter_type = self.intra_filter_type(plane);
            if intra_edge_upsample(w, h, filter_type, p_angle - 90) {
                upsample_above = 1;
                let num_px = w + if p_angle < 90 { h } else { 0 };
                intra_edge_upsample_process(above_row, num_px, bit_depth);
            }
            if intra_edge_upsample(w, h, filter_type, p_angle - 180) {
                upsample_left = 1;
                let num_px = h + if p_angle > 180 { w } else { 0 };
                intra_edge_upsample_process(left_col, num_px, bit_depth);
            }
        }
        let dx = if p_angle < 90 {
            i32::from(DR_INTRA_DERIVATIVE[p_angle as usize])
        } else if p_angle > 90 && p_angle < 180 {
            i32::from(DR_INTRA_DERIVATIVE[(180 - p_angle) as usize])
        } else {
            0
        };
        let dy = if p_angle > 90 && p_angle < 180 {
            i32::from(DR_INTRA_DERIVATIVE[(p_angle - 90) as usize])
        } else if p_angle > 180 {
            i32::from(DR_INTRA_DERIVATIVE[(270 - p_angle) as usize])
        } else {
            0
        };
        let at = |buf: &[i32], idx: i32| buf[(o + idx as isize) as usize];
        for i in 0..h {
            for j in 0..w {
                let v = if p_angle < 90 {
                    let idx = (i as i32 + 1) * dx;
                    let base = (idx >> (6 - upsample_above)) + ((j as i32) << upsample_above);
                    let shift = ((idx << upsample_above) >> 1) & 0x1F;
                    let max_base_x = ((w + h - 1) as i32) << upsample_above;
                    if base < max_base_x {
                        round2(
                            i64::from(
                                at(above_row, base) * (32 - shift)
                                    + at(above_row, base + 1) * shift,
                            ),
                            5,
                        ) as i32
                    } else {
                        at(above_row, max_base_x)
                    }
                } else if p_angle > 90 && p_angle < 180 {
                    let idx = ((j as i32) << 6) - (i as i32 + 1) * dx;
                    let base = idx >> (6 - upsample_above);
                    if base >= -(1 << upsample_above) {
                        let shift = ((idx << upsample_above) >> 1) & 0x1F;
                        round2(
                            i64::from(
                                at(above_row, base) * (32 - shift)
                                    + at(above_row, base + 1) * shift,
                            ),
                            5,
                        ) as i32
                    } else {
                        let idx = ((i as i32) << 6) - (j as i32 + 1) * dy;
                        let base = idx >> (6 - upsample_left);
                        let shift = ((idx << upsample_left) >> 1) & 0x1F;
                        round2(
                            i64::from(
                                at(left_col, base) * (32 - shift) + at(left_col, base + 1) * shift,
                            ),
                            5,
                        ) as i32
                    }
                } else if p_angle > 180 {
                    let idx = (j as i32 + 1) * dy;
                    let base = (idx >> (6 - upsample_left)) + ((i as i32) << upsample_left);
                    let shift = ((idx << upsample_left) >> 1) & 0x1F;
                    round2(
                        i64::from(
                            at(left_col, base) * (32 - shift) + at(left_col, base + 1) * shift,
                        ),
                        5,
                    ) as i32
                } else if p_angle == 90 {
                    at(above_row, j as i32)
                } else {
                    at(left_col, i as i32)
                };
                pred[i * w + j] = v;
            }
        }
    }

    // ---------------------------------------------------------------------
    // Palette and chroma-from-luma (sections 7.11.4 and 7.11.5)
    // ---------------------------------------------------------------------

    pub(crate) fn predict_palette(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        x: usize,
        y: usize,
        tx_sz: usize,
    ) {
        let w = TX_WIDTH[tx_sz] as usize;
        let h = TX_HEIGHT[tx_sz] as usize;
        let palette = match plane {
            0 => self.palette_colors_y,
            1 => self.palette_colors_u,
            _ => self.palette_colors_v,
        };
        let map = if plane == 0 {
            &*self.color_map_y
        } else {
            &*self.color_map_uv
        };
        let frame = &mut self.f.curr.planes[plane];
        for i in 0..h {
            if start_y + i >= frame.height {
                break;
            }
            for j in 0..w {
                if start_x + j >= frame.stride {
                    break;
                }
                let idx = map[(y * 4 + i).min(63)][(x * 4 + j).min(63)] as usize;
                frame.set(
                    start_x + j,
                    start_y + i,
                    palette[idx.min(PALETTE_COLORS - 1)],
                );
            }
        }
    }

    pub(crate) fn predict_chroma_from_luma(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: usize,
    ) {
        let w = TX_WIDTH[tx_sz] as usize;
        let h = TX_HEIGHT[tx_sz] as usize;
        let sub_x = usize::from(self.seq.subsampling_x);
        let sub_y = usize::from(self.seq.subsampling_y);
        let alpha = if plane == 1 {
            self.cfl_alpha_u
        } else {
            self.cfl_alpha_v
        };
        let mut l = std::mem::take(&mut self.scratch.pred);
        let mut luma_avg: i64 = 0;
        {
            let luma = &self.f.curr.planes[0];
            for i in 0..h {
                let luma_y =
                    ((start_y + i) << sub_y).min(self.max_luma_h.saturating_sub(1 << sub_y));
                for j in 0..w {
                    let luma_x =
                        ((start_x + j) << sub_x).min(self.max_luma_w.saturating_sub(1 << sub_x));
                    let mut t = 0i32;
                    for dy in 0..=sub_y {
                        for dx in 0..=sub_x {
                            t += i32::from(luma.get(luma_x + dx, luma_y + dy));
                        }
                    }
                    let v = t << (3 - sub_x - sub_y);
                    l[i * w + j] = v;
                    luma_avg += i64::from(v);
                }
            }
        }
        let luma_avg = round2(
            luma_avg,
            (TX_WIDTH_LOG2[tx_sz] + TX_HEIGHT_LOG2[tx_sz]) as u32,
        );
        let max = (1i64 << self.seq.bit_depth) - 1;
        let frame = &mut self.f.curr.planes[plane];
        for i in 0..h {
            for j in 0..w {
                if start_x + j >= frame.stride || start_y + i >= frame.height {
                    continue;
                }
                let dc = i64::from(frame.get(start_x + j, start_y + i));
                let scaled_luma =
                    round2signed(i64::from(alpha) * (i64::from(l[i * w + j]) - luma_avg), 6);
                frame.set(
                    start_x + j,
                    start_y + i,
                    (dc + scaled_luma).clamp(0, max) as u16,
                );
            }
        }
        self.scratch.pred = l;
    }

    // ---------------------------------------------------------------------
    // Inter prediction (section 7.11.3)
    // ---------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn predict_inter(
        &mut self,
        plane: usize,
        base_x: usize,
        base_y: usize,
        w: usize,
        h: usize,
        cand_row: usize,
        cand_col: usize,
    ) -> Result<()> {
        let cand_row = cand_row.min(self.fh.mi_rows - 1);
        let cand_col = cand_col.min(self.fh.mi_cols - 1);
        let cand_idx = self.mi_idx(cand_row, cand_col);
        let cand_refs = self.f.mi.ref_frames[cand_idx];
        let cand_mvs = self.f.mi.mvs[cand_idx];
        let cand_filters = self.f.mi.interp_filters[cand_idx];
        let is_compound = cand_refs[1] > INTRA_FRAME;
        let bit_depth = u32::from(self.seq.bit_depth);
        let mut inter_round0 = 3u32;
        let mut inter_round1 = if is_compound { 7u32 } else { 11 };
        if bit_depth == 12 {
            inter_round0 += 2;
            if !is_compound {
                inter_round1 -= 2;
            }
        }
        let inter_post_round = 2 * FILTER_BITS as u32 - (inter_round0 + inter_round1);
        if plane == 0 && self.motion_mode == LOCALWARP {
            self.warp_estimation();
            if self.local_valid {
                self.local_valid = setup_shear(&self.local_warp_params).0;
            }
        }
        let mut preds = std::mem::take(&mut self.scratch.preds);
        for ref_list in 0..1 + usize::from(is_compound) {
            let ref_frame = cand_refs[ref_list];
            let mut global_valid = false;
            if (self.y_mode == GLOBALMV || self.y_mode == GLOBAL_GLOBALMV)
                && self.fh.gm_type[ref_frame.max(0) as usize] > TRANSLATION
            {
                global_valid = setup_shear(&self.fh.gm_params[ref_frame as usize]).0;
            }
            let use_warp = if w < 8 || h < 8 || self.fh.force_integer_mv {
                0
            } else if self.motion_mode == LOCALWARP && self.local_valid {
                1
            } else if (self.y_mode == GLOBALMV || self.y_mode == GLOBAL_GLOBALMV)
                && self.fh.gm_type[ref_frame.max(0) as usize] > TRANSLATION
                && !self.is_scaled(ref_frame)
                && global_valid
            {
                2
            } else {
                0
            };
            let mv = cand_mvs[ref_list];
            if use_warp != 0 {
                let warp_params = if use_warp == 1 {
                    self.local_warp_params
                } else {
                    self.fh.gm_params[ref_frame as usize]
                };
                for i8 in 0..=((h - 1) >> 3) {
                    for j8 in 0..=((w - 1) >> 3) {
                        self.block_warp(
                            &warp_params,
                            plane,
                            ref_frame,
                            base_x,
                            base_y,
                            i8,
                            j8,
                            w,
                            h,
                            inter_round0,
                            inter_round1,
                            &mut preds[ref_list],
                        );
                    }
                }
            } else {
                let (ref_upscaled_width, ref_frame_height) = if self.use_intrabc {
                    (self.fh.frame_width, self.fh.frame_height)
                } else {
                    let slot = self.ref_slot(ref_frame);
                    (slot.upscaled_width, slot.frame_height)
                };
                let (start_x, start_y, step_x, step_y) = self.motion_vector_scaling(
                    plane,
                    ref_upscaled_width,
                    ref_frame_height,
                    base_x,
                    base_y,
                    mv,
                );
                let (last_w, last_h) = if self.use_intrabc {
                    (self.fh.mi_cols * MI_SIZE, self.fh.mi_rows * MI_SIZE)
                } else {
                    (ref_upscaled_width, ref_frame_height)
                };
                self.block_inter_prediction(
                    plane,
                    if self.use_intrabc {
                        None
                    } else {
                        Some(ref_frame)
                    },
                    last_w,
                    last_h,
                    start_x,
                    start_y,
                    step_x,
                    step_y,
                    w,
                    h,
                    cand_filters,
                    inter_round0,
                    inter_round1,
                    &mut preds[ref_list],
                    w,
                );
            }
        }
        let max = (1i64 << bit_depth) - 1;
        if self.compound_type == COMPOUND_WEDGE && plane == 0 {
            self.wedge_mask(w, h);
        } else if self.compound_type == COMPOUND_INTRA {
            self.intra_variant_mask(w, h);
        } else if self.compound_type == COMPOUND_DIFFWTD && plane == 0 {
            let mask = &mut self.scratch.mask;
            for i in 0..h {
                for j in 0..w {
                    let diff = (preds[0][i * w + j] - preds[1][i * w + j]).abs();
                    let diff = round2(i64::from(diff), (bit_depth - 8) + inter_post_round) as i32;
                    let m = (38 + diff / 16).clamp(0, 64);
                    mask[i * w + j] = if self.mask_type { 64 - m } else { m };
                }
            }
        }
        let is_inter_intra = self.is_inter && self.ref_frame[1] == INTRA_FRAME;
        let frame_height = self.f.curr.planes[plane].height;
        let frame_width = self.f.curr.planes[plane].stride;
        if !is_compound && !is_inter_intra {
            let frame = &mut self.f.curr.planes[plane];
            for i in 0..h {
                if base_y + i >= frame_height {
                    break;
                }
                for j in 0..w {
                    if base_x + j >= frame_width {
                        break;
                    }
                    frame.set(
                        base_x + j,
                        base_y + i,
                        i64::from(preds[0][i * w + j]).clamp(0, max) as u16,
                    );
                }
            }
        } else if self.compound_type == COMPOUND_AVERAGE {
            let frame = &mut self.f.curr.planes[plane];
            for i in 0..h {
                if base_y + i >= frame_height {
                    break;
                }
                for j in 0..w {
                    if base_x + j >= frame_width {
                        break;
                    }
                    let v = round2(
                        i64::from(preds[0][i * w + j] + preds[1][i * w + j]),
                        1 + inter_post_round,
                    );
                    frame.set(base_x + j, base_y + i, v.clamp(0, max) as u16);
                }
            }
        } else if self.compound_type == COMPOUND_DISTANCE {
            let (fwd, bck) = self.distance_weights(cand_refs);
            let frame = &mut self.f.curr.planes[plane];
            for i in 0..h {
                if base_y + i >= frame_height {
                    break;
                }
                for j in 0..w {
                    if base_x + j >= frame_width {
                        break;
                    }
                    let v = round2(
                        fwd * i64::from(preds[0][i * w + j]) + bck * i64::from(preds[1][i * w + j]),
                        4 + inter_post_round,
                    );
                    frame.set(base_x + j, base_y + i, v.clamp(0, max) as u16);
                }
            }
        } else {
            self.mask_blend(&preds, plane, base_x, base_y, w, h, inter_post_round);
        }
        self.scratch.preds = preds;
        if self.motion_mode == OBMC {
            self.overlapped_motion_compensation(plane, w, h, inter_round0, inter_round1)?;
        }
        Ok(())
    }

    /// The motion vector scaling process, returning `(startX, startY, stepX, stepY)`.
    fn motion_vector_scaling(
        &self,
        plane: usize,
        ref_upscaled_width: usize,
        ref_frame_height: usize,
        x: usize,
        y: usize,
        mv: [i32; 2],
    ) -> (i64, i64, i64, i64) {
        let fw = self.fh.frame_width as i64;
        let fhh = self.fh.frame_height as i64;
        let x_scale = (((ref_upscaled_width as i64) << REF_SCALE_SHIFT) + fw / 2) / fw;
        let y_scale = (((ref_frame_height as i64) << REF_SCALE_SHIFT) + fhh / 2) / fhh;
        let sub_x = self.sub_x(plane) as u32;
        let sub_y = self.sub_y(plane) as u32;
        let half_sample = 1i64 << (SUBPEL_BITS - 1);
        let orig_x = ((x as i64) << SUBPEL_BITS) + ((2 * i64::from(mv[1])) >> sub_x) + half_sample;
        let orig_y = ((y as i64) << SUBPEL_BITS) + ((2 * i64::from(mv[0])) >> sub_y) + half_sample;
        let base_x = orig_x * x_scale - (half_sample << REF_SCALE_SHIFT);
        let base_y = orig_y * y_scale - (half_sample << REF_SCALE_SHIFT);
        let off = (1i64 << (SCALE_SUBPEL_BITS - SUBPEL_BITS)) / 2;
        let shift = (REF_SCALE_SHIFT + SUBPEL_BITS - SCALE_SUBPEL_BITS) as u32;
        let start_x = round2signed(base_x, shift) + off;
        let start_y = round2signed(base_y, shift) + off;
        let step_x = round2signed(x_scale, (REF_SCALE_SHIFT - SCALE_SUBPEL_BITS) as u32);
        let step_y = round2signed(y_scale, (REF_SCALE_SHIFT - SCALE_SUBPEL_BITS) as u32);
        (start_x, start_y, step_x, step_y)
    }

    /// The block inter prediction process. `reference` of `None` predicts
    /// from the current (pre-filter) frame, as intra block copy does.
    #[allow(clippy::too_many_arguments)]
    fn block_inter_prediction(
        &mut self,
        plane: usize,
        reference: Option<i8>,
        ref_upscaled_width: usize,
        ref_frame_height: usize,
        x: i64,
        y: i64,
        x_step: i64,
        y_step: i64,
        w: usize,
        h: usize,
        interp_filters: [u8; 2],
        inter_round0: u32,
        inter_round1: u32,
        pred: &mut [i32],
        pred_stride: usize,
    ) {
        let sub_x = self.sub_x(plane);
        let sub_y = self.sub_y(plane);
        let last_x = ((ref_upscaled_width + sub_x) >> sub_x) as i64 - 1;
        let last_y = ((ref_frame_height + sub_y) >> sub_y) as i64 - 1;
        let intermediate_height = ((((h as i64 - 1) * y_step + (1 << SCALE_SUBPEL_BITS) - 1)
            >> SCALE_SUBPEL_BITS)
            + 8) as usize;
        let mut filter_x = interp_filters[1] as usize;
        if w <= 4 {
            if filter_x == EIGHTTAP as usize || filter_x == EIGHTTAP_SHARP as usize {
                filter_x = 4;
            } else if filter_x == EIGHTTAP_SMOOTH as usize {
                filter_x = 5;
            }
        }
        let mut filter_y = interp_filters[0] as usize;
        if h <= 4 {
            if filter_y == EIGHTTAP as usize || filter_y == EIGHTTAP_SHARP as usize {
                filter_y = 4;
            } else if filter_y == EIGHTTAP_SMOOTH as usize {
                filter_y = 5;
            }
        }
        let mut intermediate = std::mem::take(&mut self.scratch.intermediate);
        if intermediate.len() < intermediate_height * w {
            intermediate.resize(intermediate_height * w, 0);
        }
        {
            let ref_plane = match reference {
                Some(r) => &self.ref_slot(r).frame.planes[plane],
                None => &self.f.curr.planes[plane],
            };
            let rnd0 = if inter_round0 == 0 {
                0
            } else {
                1i32 << (inter_round0 - 1)
            };
            for r in 0..intermediate_height {
                let ry = ((y >> 10) + r as i64 - 3).clamp(0, last_y) as usize;
                let row = ref_plane.row(ry);
                for c in 0..w {
                    let p = x + x_step * c as i64;
                    let filter =
                        &SUBPEL_FILTERS[filter_x][((p >> 6) & SUBPEL_MASK as i64) as usize];
                    let px = p >> 10;
                    let mut s = 0i32;
                    for t in 0..8 {
                        let sx = (px + t as i64 - 3).clamp(0, last_x) as usize;
                        s += i32::from(filter[t]) * i32::from(row[sx]);
                    }
                    intermediate[r * w + c] = (s + rnd0) >> inter_round0;
                }
            }
        }
        let rnd1 = 1i64 << (inter_round1 - 1);
        for r in 0..h {
            let p = (y & 1023) + y_step * r as i64;
            let filter = &SUBPEL_FILTERS[filter_y][((p >> 6) & SUBPEL_MASK as i64) as usize];
            let base = (p >> 10) as usize;
            for c in 0..w {
                let mut s = 0i64;
                for t in 0..8 {
                    s += i64::from(filter[t]) * i64::from(intermediate[(base + t) * w + c]);
                }
                pred[r * pred_stride + c] = ((s + rnd1) >> inter_round1) as i32;
            }
        }
        self.scratch.intermediate = intermediate;
    }

    #[allow(clippy::too_many_arguments)]
    fn block_warp(
        &self,
        warp_params: &[i32; 6],
        plane: usize,
        ref_frame: i8,
        x: usize,
        y: usize,
        i8: usize,
        j8: usize,
        w: usize,
        h: usize,
        inter_round0: u32,
        inter_round1: u32,
        pred: &mut [i32],
    ) {
        let slot = self.ref_slot(ref_frame);
        let reference = &slot.frame.planes[plane];
        let sub_x = self.sub_x(plane);
        let sub_y = self.sub_y(plane);
        let last_x = ((slot.upscaled_width + sub_x) >> sub_x) as i64 - 1;
        let last_y = ((slot.frame_height + sub_y) >> sub_y) as i64 - 1;
        let src_x = ((x + j8 * 8 + 4) << sub_x) as i64;
        let src_y = ((y + i8 * 8 + 4) << sub_y) as i64;
        let dst_x = i64::from(warp_params[2]) * src_x
            + i64::from(warp_params[3]) * src_y
            + i64::from(warp_params[0]);
        let dst_y = i64::from(warp_params[4]) * src_x
            + i64::from(warp_params[5]) * src_y
            + i64::from(warp_params[1]);
        let (_, alpha, beta, gamma, delta) = setup_shear(warp_params);
        let x4 = dst_x >> sub_x;
        let y4 = dst_y >> sub_y;
        let ix4 = x4 >> WARPEDMODEL_PREC_BITS;
        let sx4 = x4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);
        let iy4 = y4 >> WARPEDMODEL_PREC_BITS;
        let sy4 = y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);
        let mut intermediate = [[0i64; 8]; 15];
        for i1 in -7i64..8 {
            let row = reference.row((iy4 + i1).clamp(0, last_y) as usize);
            for i2 in -4i64..4 {
                let sx = sx4 + i64::from(alpha) * i2 + i64::from(beta) * i1;
                let offs = (round2(sx, WARPEDDIFF_PREC_BITS as u32)
                    + i64::from(WARPEDPIXEL_PREC_SHIFTS)) as usize;
                let filter = &WARPED_FILTERS[offs.min(192)];
                let mut s = 0i64;
                for i3 in 0..8i64 {
                    let sample = row[(ix4 + i2 - 3 + i3).clamp(0, last_x) as usize];
                    s += i64::from(filter[i3 as usize]) * i64::from(sample);
                }
                intermediate[(i1 + 7) as usize][(i2 + 4) as usize] = round2(s, inter_round0);
            }
        }
        let limit_i = 4.min(h as i64 - i8 as i64 * 8 - 4);
        let limit_j = 4.min(w as i64 - j8 as i64 * 8 - 4);
        for i1 in -4i64..limit_i {
            for i2 in -4i64..limit_j {
                let sy = sy4 + i64::from(gamma) * i2 + i64::from(delta) * i1;
                let offs = (round2(sy, WARPEDDIFF_PREC_BITS as u32)
                    + i64::from(WARPEDPIXEL_PREC_SHIFTS)) as usize;
                let filter = &WARPED_FILTERS[offs.min(192)];
                let mut s = 0i64;
                for i3 in 0..8i64 {
                    s += i64::from(filter[i3 as usize])
                        * intermediate[(i1 + i3 + 4) as usize][(i2 + 4) as usize];
                }
                let py = (i8 as i64 * 8 + i1 + 4) as usize;
                let px = (j8 as i64 * 8 + i2 + 4) as usize;
                pred[py * w + px] = round2(s, inter_round1) as i32;
            }
        }
    }

    /// The warp estimation process.
    fn warp_estimation(&mut self) {
        let mut a = [[0i64; 2]; 2];
        let mut bx = [0i64; 2];
        let mut by = [0i64; 2];
        let w4 = bw4(self.mi_size) as i64;
        let h4 = bh4(self.mi_size) as i64;
        let mid_y = self.mi_row as i64 * 4 + h4 * 2 - 1;
        let mid_x = self.mi_col as i64 * 4 + w4 * 2 - 1;
        let suy = mid_y * 8;
        let sux = mid_x * 8;
        let duy = suy + i64::from(self.mv[0][0]);
        let dux = sux + i64::from(self.mv[0][1]);
        let ls_product = |a: i64, b: i64| ((a * b) >> 2) + (a + b);
        for i in 0..self.num_samples {
            let c = self.cand_list[i];
            let sy = i64::from(c[0]) - suy;
            let sx = i64::from(c[1]) - sux;
            let dy = i64::from(c[2]) - duy;
            let dx = i64::from(c[3]) - dux;
            if (sx - dx).abs() < i64::from(LS_MV_MAX) && (sy - dy).abs() < i64::from(LS_MV_MAX) {
                a[0][0] += ls_product(sx, sx) + 8;
                a[0][1] += ls_product(sx, sy) + 4;
                a[1][1] += ls_product(sy, sy) + 8;
                bx[0] += ls_product(sx, dx) + 8;
                bx[1] += ls_product(sy, dx) + 4;
                by[0] += ls_product(sx, dy) + 4;
                by[1] += ls_product(sy, dy) + 8;
            }
        }
        let det = a[0][0] * a[1][1] - a[0][1] * a[0][1];
        self.local_valid = det != 0;
        if det == 0 {
            return;
        }
        let (mut div_shift, mut div_factor) = resolve_divisor(det);
        div_shift -= WARPEDMODEL_PREC_BITS;
        if div_shift < 0 {
            div_factor <<= -div_shift;
            div_shift = 0;
        }
        let div_shift = div_shift as u32;
        let nondiag = |v: i128| -> i32 {
            let r = round2signed_i128(v * i128::from(div_factor), div_shift);
            r.clamp(
                i128::from(-WARPEDMODEL_NONDIAGAFFINE_CLAMP + 1),
                i128::from(WARPEDMODEL_NONDIAGAFFINE_CLAMP - 1),
            ) as i32
        };
        let diag = |v: i128| -> i32 {
            let r = round2signed_i128(v * i128::from(div_factor), div_shift);
            r.clamp(
                i128::from((1 << WARPEDMODEL_PREC_BITS) - WARPEDMODEL_NONDIAGAFFINE_CLAMP + 1),
                i128::from((1 << WARPEDMODEL_PREC_BITS) + WARPEDMODEL_NONDIAGAFFINE_CLAMP - 1),
            ) as i32
        };
        let a = a.map(|row| row.map(i128::from));
        let bx = bx.map(i128::from);
        let by = by.map(i128::from);
        let p = &mut self.local_warp_params;
        p[2] = diag(a[1][1] * bx[0] - a[0][1] * bx[1]);
        p[3] = nondiag(-a[0][1] * bx[0] + a[0][0] * bx[1]);
        p[4] = nondiag(a[1][1] * by[0] - a[0][1] * by[1]);
        p[5] = diag(-a[0][1] * by[0] + a[0][0] * by[1]);
        let mvx = i64::from(self.mv[0][1]);
        let mvy = i64::from(self.mv[0][0]);
        let vx = mvx * (1 << (WARPEDMODEL_PREC_BITS - 3))
            - (mid_x * (i64::from(p[2]) - (1 << WARPEDMODEL_PREC_BITS)) + mid_y * i64::from(p[3]));
        let vy = mvy * (1 << (WARPEDMODEL_PREC_BITS - 3))
            - (mid_x * i64::from(p[4]) + mid_y * (i64::from(p[5]) - (1 << WARPEDMODEL_PREC_BITS)));
        p[0] = vx.clamp(
            -i64::from(WARPEDMODEL_TRANS_CLAMP),
            i64::from(WARPEDMODEL_TRANS_CLAMP) - 1,
        ) as i32;
        p[1] = vy.clamp(
            -i64::from(WARPEDMODEL_TRANS_CLAMP),
            i64::from(WARPEDMODEL_TRANS_CLAMP) - 1,
        ) as i32;
    }

    fn wedge_mask(&mut self, w: usize, h: usize) {
        let masks = wedge_masks();
        let Some(table) = masks.masks[self.mi_size].as_ref() else {
            return;
        };
        let bw = block_width(self.mi_size);
        let bh = block_height(self.mi_size);
        let offset = (self.wedge_sign * 16 + self.wedge_index) * bh * bw;
        for i in 0..h.min(bh) {
            for j in 0..w.min(bw) {
                self.scratch.mask[i * w + j] = i32::from(table[offset + i * bw + j]);
            }
        }
    }

    fn intra_variant_mask(&mut self, w: usize, h: usize) {
        let size_scale = MAX_SB_SIZE / h.max(w);
        for i in 0..h {
            for j in 0..w {
                self.scratch.mask[i * w + j] = match self.interintra_mode {
                    II_V_PRED => i32::from(II_WEIGHTS_1D[i * size_scale]),
                    II_H_PRED => i32::from(II_WEIGHTS_1D[j * size_scale]),
                    II_SMOOTH_PRED => i32::from(II_WEIGHTS_1D[i.min(j) * size_scale]),
                    _ => 32,
                };
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mask_blend(
        &mut self,
        preds: &[Vec<i32>; 2],
        plane: usize,
        dst_x: usize,
        dst_y: usize,
        w: usize,
        h: usize,
        inter_post_round: u32,
    ) {
        let sub_x = self.sub_x(plane);
        let sub_y = self.sub_y(plane);
        let max = (1i64 << self.seq.bit_depth) - 1;
        // The luma mask was prepared with the luma block width.
        let mask_w = if (sub_x == 0 && sub_y == 0) || (self.interintra && !self.wedge_interintra) {
            w
        } else {
            w << sub_x
        };
        let mask = &self.scratch.mask;
        let frame = &mut self.f.curr.planes[plane];
        for yy in 0..h {
            for xx in 0..w {
                let m = if (sub_x == 0 && sub_y == 0) || (self.interintra && !self.wedge_interintra)
                {
                    mask[yy * mask_w + xx]
                } else if sub_x != 0 && sub_y == 0 {
                    round2(
                        i64::from(mask[yy * mask_w + 2 * xx] + mask[yy * mask_w + 2 * xx + 1]),
                        1,
                    ) as i32
                } else {
                    round2(
                        i64::from(
                            mask[(2 * yy) * mask_w + 2 * xx]
                                + mask[(2 * yy) * mask_w + 2 * xx + 1]
                                + mask[(2 * yy + 1) * mask_w + 2 * xx]
                                + mask[(2 * yy + 1) * mask_w + 2 * xx + 1],
                        ),
                        2,
                    ) as i32
                };
                if dst_x + xx >= frame.stride || dst_y + yy >= frame.height {
                    continue;
                }
                if self.interintra {
                    let pred0 =
                        round2(i64::from(preds[0][yy * w + xx]), inter_post_round).clamp(0, max);
                    let pred1 = i64::from(frame.get(dst_x + xx, dst_y + yy));
                    let v = round2(i64::from(m) * pred1 + (64 - i64::from(m)) * pred0, 6);
                    frame.set(dst_x + xx, dst_y + yy, v as u16);
                } else {
                    let pred0 = i64::from(preds[0][yy * w + xx]);
                    let pred1 = i64::from(preds[1][yy * w + xx]);
                    let v = round2(
                        i64::from(m) * pred0 + (64 - i64::from(m)) * pred1,
                        6 + inter_post_round,
                    );
                    frame.set(dst_x + xx, dst_y + yy, v.clamp(0, max) as u16);
                }
            }
        }
    }

    fn distance_weights(&self, cand_refs: [i8; 2]) -> (i64, i64) {
        let mut dist = [0i32; 2];
        for (ref_list, d) in dist.iter_mut().enumerate() {
            let h = self.fh.order_hints[cand_refs[ref_list] as usize];
            *d = self
                .fh
                .get_relative_dist(self.seq, h, self.fh.order_hint)
                .abs()
                .clamp(0, MAX_FRAME_DISTANCE);
        }
        let d0 = dist[1];
        let d1 = dist[0];
        let order = usize::from(d0 <= d1);
        if d0 == 0 || d1 == 0 {
            return (
                i64::from(QUANT_DIST_LOOKUP[3][order]),
                i64::from(QUANT_DIST_LOOKUP[3][1 - order]),
            );
        }
        let mut i = 0;
        while i < 3 {
            let c0 = i32::from(QUANT_DIST_WEIGHT[i][order]);
            let c1 = i32::from(QUANT_DIST_WEIGHT[i][1 - order]);
            if order == 1 {
                if d0 * c0 > d1 * c1 {
                    break;
                }
            } else if d0 * c0 < d1 * c1 {
                break;
            }
            i += 1;
        }
        (
            i64::from(QUANT_DIST_LOOKUP[i][order]),
            i64::from(QUANT_DIST_LOOKUP[i][1 - order]),
        )
    }

    fn overlapped_motion_compensation(
        &mut self,
        plane: usize,
        w: usize,
        h: usize,
        inter_round0: u32,
        inter_round1: u32,
    ) -> Result<()> {
        let sub_x = self.sub_x(plane);
        let sub_y = self.sub_y(plane);
        let max = (1i64 << self.seq.bit_depth) - 1;
        if self.avail_u && self.get_plane_residual_size(self.mi_size, plane) >= BLOCK_8X8 {
            let w4 = bw4(self.mi_size);
            let mut x4 = self.mi_col;
            let y4 = self.mi_row;
            let mut n_count = 0;
            let n_limit = 4.min(MI_WIDTH_LOG2[self.mi_size] as usize);
            while n_count < n_limit && x4 < self.fh.mi_cols.min(self.mi_col + w4) {
                let cand_row = self.mi_row - 1;
                let cand_col = (x4 | 1).min(self.fh.mi_cols - 1);
                let cand_sz = self.f.mi.mi_sizes[self.mi_idx(cand_row, cand_col)] as usize;
                let step4 = bw4(cand_sz).clamp(2, 16);
                if self.f.mi.ref_frames[self.mi_idx(cand_row, cand_col)][0] > INTRA_FRAME {
                    n_count += 1;
                    let pred_w = w.min((step4 * MI_SIZE) >> sub_x);
                    let pred_h = (h >> 1).min(32 >> sub_y);
                    let mask = obmc_mask(pred_h);
                    self.predict_overlap(
                        plane,
                        cand_row,
                        cand_col,
                        x4,
                        y4,
                        pred_w,
                        pred_h,
                        0,
                        mask,
                        inter_round0,
                        inter_round1,
                        max,
                    );
                }
                x4 += step4;
            }
        }
        if self.avail_l {
            let h4 = bh4(self.mi_size);
            let x4 = self.mi_col;
            let mut y4 = self.mi_row;
            let mut n_count = 0;
            let n_limit = 4.min(MI_HEIGHT_LOG2[self.mi_size] as usize);
            while n_count < n_limit && y4 < self.fh.mi_rows.min(self.mi_row + h4) {
                let cand_col = self.mi_col - 1;
                let cand_row = (y4 | 1).min(self.fh.mi_rows - 1);
                let cand_sz = self.f.mi.mi_sizes[self.mi_idx(cand_row, cand_col)] as usize;
                let step4 = bh4(cand_sz).clamp(2, 16);
                if self.f.mi.ref_frames[self.mi_idx(cand_row, cand_col)][0] > INTRA_FRAME {
                    n_count += 1;
                    let pred_w = (w >> 1).min(32 >> sub_x);
                    let pred_h = h.min((step4 * MI_SIZE) >> sub_y);
                    let mask = obmc_mask(pred_w);
                    self.predict_overlap(
                        plane,
                        cand_row,
                        cand_col,
                        x4,
                        y4,
                        pred_w,
                        pred_h,
                        1,
                        mask,
                        inter_round0,
                        inter_round1,
                        max,
                    );
                }
                y4 += step4;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_overlap(
        &mut self,
        plane: usize,
        cand_row: usize,
        cand_col: usize,
        x4: usize,
        y4: usize,
        pred_w: usize,
        pred_h: usize,
        pass: usize,
        mask: &'static [u8],
        inter_round0: u32,
        inter_round1: u32,
        max: i64,
    ) {
        let cand_idx = self.mi_idx(cand_row, cand_col);
        let mv = self.f.mi.mvs[cand_idx][0];
        let ref_frame = self.f.mi.ref_frames[cand_idx][0];
        let filters = self.f.mi.interp_filters[cand_idx];
        let sub_x = self.sub_x(plane);
        let sub_y = self.sub_y(plane);
        let pred_x = (x4 * 4) >> sub_x;
        let pred_y = (y4 * 4) >> sub_y;
        let (upscaled_width, frame_height) = {
            let slot = self.ref_slot(ref_frame);
            (slot.upscaled_width, slot.frame_height)
        };
        let (start_x, start_y, step_x, step_y) =
            self.motion_vector_scaling(plane, upscaled_width, frame_height, pred_x, pred_y, mv);
        let mut obmc_pred = std::mem::take(&mut self.scratch.obmc_pred);
        self.block_inter_prediction(
            plane,
            Some(ref_frame),
            upscaled_width,
            frame_height,
            start_x,
            start_y,
            step_x,
            step_y,
            pred_w,
            pred_h,
            filters,
            inter_round0,
            inter_round1,
            &mut obmc_pred,
            pred_w,
        );
        let frame = &mut self.f.curr.planes[plane];
        for i in 0..pred_h {
            if pred_y + i >= frame.height {
                break;
            }
            for j in 0..pred_w {
                if pred_x + j >= frame.stride {
                    break;
                }
                let obmc = i64::from(obmc_pred[i * pred_w + j]).clamp(0, max);
                let m = i64::from(if pass == 0 { mask[i] } else { mask[j] });
                let cur = i64::from(frame.get(pred_x + j, pred_y + i));
                frame.set(
                    pred_x + j,
                    pred_y + i,
                    round2(m * cur + (64 - m) * obmc, 6) as u16,
                );
            }
        }
        self.scratch.obmc_pred = obmc_pred;
    }
}

fn round2signed_i128(x: i128, n: u32) -> i128 {
    let r = |v: i128| {
        if n == 0 {
            v
        } else {
            (v + (1i128 << (n - 1))) >> n
        }
    };
    if x >= 0 { r(x) } else { -r(-x) }
}

fn obmc_mask(length: usize) -> &'static [u8] {
    match length {
        2 => &OBMC_MASK_2,
        4 => &OBMC_MASK_4,
        8 => &OBMC_MASK_8,
        16 => &OBMC_MASK_16,
        _ => &OBMC_MASK_32,
    }
}

fn intra_edge_filter_strength(w: usize, h: usize, filter_type: bool, delta: i32) -> usize {
    let d = delta.abs();
    let blk_wh = w + h;
    let mut strength = 0;
    if !filter_type {
        if blk_wh <= 8 {
            if d >= 56 {
                strength = 1;
            }
        } else if blk_wh <= 12 {
            if d >= 40 {
                strength = 1;
            }
        } else if blk_wh <= 16 {
            if d >= 40 {
                strength = 1;
            }
        } else if blk_wh <= 24 {
            if d >= 8 {
                strength = 1;
            }
            if d >= 16 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else if blk_wh <= 32 {
            strength = 1;
            if d >= 4 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else {
            strength = 3;
        }
    } else if blk_wh <= 8 {
        if d >= 40 {
            strength = 1;
        }
        if d >= 64 {
            strength = 2;
        }
    } else if blk_wh <= 16 {
        if d >= 20 {
            strength = 1;
        }
        if d >= 48 {
            strength = 2;
        }
    } else if blk_wh <= 24 {
        if d >= 4 {
            strength = 3;
        }
    } else {
        strength = 3;
    }
    strength
}

fn intra_edge_upsample(w: usize, h: usize, filter_type: bool, delta: i32) -> bool {
    let d = delta.abs();
    let blk_wh = w + h;
    if d <= 0 || d >= 40 {
        false
    } else if !filter_type {
        blk_wh <= 16
    } else {
        blk_wh <= 8
    }
}

/// The intra edge filter process over `buf[-1 ..]` (stored at `EDGE_OFFSET`).
fn intra_edge_filter(buf: &mut [i32], sz: usize, strength: usize) {
    if strength == 0 {
        return;
    }
    let o = EDGE_OFFSET;
    let mut edge = [0i32; 2 * 129 + 8];
    for i in 0..sz {
        edge[i] = buf[o + i - 1];
    }
    for i in 1..sz {
        let mut s = 0i32;
        for j in 0..5 {
            let k = (i as isize - 2 + j as isize).clamp(0, sz as isize - 1) as usize;
            s += i32::from(INTRA_EDGE_KERNEL[strength - 1][j]) * edge[k];
        }
        buf[o + i - 1] = (s + 8) >> 4;
    }
}

/// The intra edge upsample process over `buf` (stored at `EDGE_OFFSET`).
fn intra_edge_upsample_process(buf: &mut [i32], num_px: usize, bit_depth: u32) {
    let o = EDGE_OFFSET as isize;
    let at = |i: isize| (o + i) as usize;
    let mut dup = [0i32; 2 * 64 + 8];
    dup[0] = buf[at(-1)];
    for i in -1..num_px as isize {
        dup[(i + 2) as usize] = buf[at(i)];
    }
    dup[num_px + 2] = buf[at(num_px as isize - 1)];
    let max = (1i32 << bit_depth) - 1;
    buf[at(-2)] = dup[0];
    for i in 0..num_px {
        let s = -dup[i] + 9 * dup[i + 1] + 9 * dup[i + 2] - dup[i + 3];
        let s = (round2(i64::from(s), 4) as i32).clamp(0, max);
        buf[at(2 * i as isize - 1)] = s;
        buf[at(2 * i as isize)] = dup[i + 2];
    }
}

fn smooth_intra_prediction(
    mode: u8,
    log2w: usize,
    log2h: usize,
    above_row: &[i32],
    left_col: &[i32],
    pred: &mut [i32],
) {
    let o = EDGE_OFFSET;
    let w = 1usize << log2w;
    let h = 1usize << log2h;
    if mode == SMOOTH_PRED {
        let wx = sm_weights(log2w);
        let wy = sm_weights(log2h);
        for i in 0..h {
            for j in 0..w {
                let s = i32::from(wy[i]) * above_row[o + j]
                    + (256 - i32::from(wy[i])) * left_col[o + h - 1]
                    + i32::from(wx[j]) * left_col[o + i]
                    + (256 - i32::from(wx[j])) * above_row[o + w - 1];
                pred[i * w + j] = round2(i64::from(s), 9) as i32;
            }
        }
    } else if mode == SMOOTH_V_PRED {
        let wy = sm_weights(log2h);
        for i in 0..h {
            for j in 0..w {
                let s = i32::from(wy[i]) * above_row[o + j]
                    + (256 - i32::from(wy[i])) * left_col[o + h - 1];
                pred[i * w + j] = round2(i64::from(s), 8) as i32;
            }
        }
    } else {
        let wx = sm_weights(log2w);
        for i in 0..h {
            for j in 0..w {
                let s = i32::from(wx[j]) * left_col[o + i]
                    + (256 - i32::from(wx[j])) * above_row[o + w - 1];
                pred[i * w + j] = round2(i64::from(s), 8) as i32;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dc_intra_prediction(
    have_left: bool,
    have_above: bool,
    log2w: usize,
    log2h: usize,
    above_row: &[i32],
    left_col: &[i32],
    pred: &mut [i32],
    bit_depth: u32,
) {
    let o = EDGE_OFFSET;
    let w = 1usize << log2w;
    let h = 1usize << log2h;
    let max = (1i32 << bit_depth) - 1;
    let avg = if have_left && have_above {
        let mut sum: i32 = 0;
        for k in 0..h {
            sum += left_col[o + k];
        }
        for k in 0..w {
            sum += above_row[o + k];
        }
        sum += ((w + h) >> 1) as i32;
        sum / (w + h) as i32
    } else if have_left {
        let mut sum: i32 = 0;
        for k in 0..h {
            sum += left_col[o + k];
        }
        ((sum + (h >> 1) as i32) >> log2h).clamp(0, max)
    } else if have_above {
        let mut sum: i32 = 0;
        for k in 0..w {
            sum += above_row[o + k];
        }
        ((sum + (w >> 1) as i32) >> log2w).clamp(0, max)
    } else {
        1 << (bit_depth - 1)
    };
    pred[..w * h].fill(avg);
}

fn paeth_intra_prediction(
    w: usize,
    h: usize,
    above_row: &[i32],
    left_col: &[i32],
    pred: &mut [i32],
) {
    let o = EDGE_OFFSET;
    let top_left = above_row[o - 1];
    for i in 0..h {
        for j in 0..w {
            let base = above_row[o + j] + left_col[o + i] - top_left;
            let p_left = (base - left_col[o + i]).abs();
            let p_top = (base - above_row[o + j]).abs();
            let p_top_left = (base - top_left).abs();
            pred[i * w + j] = if p_left <= p_top && p_left <= p_top_left {
                left_col[o + i]
            } else if p_top <= p_top_left {
                above_row[o + j]
            } else {
                top_left
            };
        }
    }
}
