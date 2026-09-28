//! Motion vector prediction (specification section 7.10): the reference MV
//! stack, overlappable candidate detection and warp sample collection.

use super::consts::*;
use super::tile::{TileDecoder, bh4, bw4};

const INVALID_MV: i32 = -1 << 15;

fn has_newmv(mode: u8) -> bool {
    matches!(
        mode,
        NEWMV | NEW_NEWMV | NEAR_NEWMV | NEW_NEARMV | NEAREST_NEWMV | NEW_NEARESTMV
    )
}

fn round2signed(x: i64, n: u32) -> i64 {
    if n == 0 {
        return x;
    }
    if x >= 0 {
        (x + (1 << (n - 1))) >> n
    } else {
        -((-x + (1 << (n - 1))) >> n)
    }
}

impl TileDecoder<'_> {
    /// `find_mv_stack( isCompound )`.
    pub(crate) fn find_mv_stack(&mut self, is_compound: bool) {
        let bw4 = bw4(self.mi_size);
        let bh4 = bh4(self.mi_size);
        self.num_mv_found = 0;
        self.new_mv_count = 0;
        self.global_mvs[0] = self.setup_global_mv(0);
        if is_compound {
            self.global_mvs[1] = self.setup_global_mv(1);
        }
        self.found_match = false;
        self.scan_row(-1, is_compound);
        let mut found_above_match = self.found_match;
        self.found_match = false;
        self.scan_col(-1, is_compound);
        let mut found_left_match = self.found_match;
        self.found_match = false;
        if bw4.max(bh4) <= 16 {
            self.scan_point(-1, bw4 as isize, is_compound);
        }
        if self.found_match {
            found_above_match = true;
        }
        self.close_matches = usize::from(found_above_match) + usize::from(found_left_match);
        let num_nearest = self.num_mv_found;
        let num_new = self.new_mv_count;
        if num_nearest > 0 {
            for idx in 0..num_nearest {
                self.weight_stack[idx] += REF_CAT_LEVEL;
            }
        }
        self.zero_mv_context = 0;
        if self.fh.use_ref_frame_mvs {
            self.temporal_scan(is_compound);
        }
        self.scan_point(-1, -1, is_compound);
        if self.found_match {
            found_above_match = true;
        }
        self.found_match = false;
        self.scan_row(-3, is_compound);
        if self.found_match {
            found_above_match = true;
        }
        self.found_match = false;
        self.scan_col(-3, is_compound);
        if self.found_match {
            found_left_match = true;
        }
        self.found_match = false;
        if bh4 > 1 {
            self.scan_row(-5, is_compound);
        }
        if self.found_match {
            found_above_match = true;
        }
        self.found_match = false;
        if bw4 > 1 {
            self.scan_col(-5, is_compound);
        }
        if self.found_match {
            found_left_match = true;
        }
        self.total_matches = usize::from(found_above_match) + usize::from(found_left_match);
        self.sort_stack(0, num_nearest, is_compound);
        self.sort_stack(num_nearest, self.num_mv_found, is_compound);
        if self.num_mv_found < 2 {
            self.extra_search(is_compound);
        }
        self.context_and_clamping(is_compound, num_new);
    }

    fn setup_global_mv(&self, ref_list: usize) -> [i32; 2] {
        let r = self.ref_frame[ref_list];
        let mut mv = [0i32; 2];
        let typ = if r != INTRA_FRAME {
            self.fh.gm_type[r as usize]
        } else {
            IDENTITY
        };
        if r == INTRA_FRAME || typ == IDENTITY {
            mv = [0, 0];
        } else if typ == TRANSLATION {
            let gm = &self.fh.gm_params[r as usize];
            mv[0] = gm[0] >> (WARPEDMODEL_PREC_BITS - 3);
            mv[1] = gm[1] >> (WARPEDMODEL_PREC_BITS - 3);
        } else {
            let gm = &self.fh.gm_params[r as usize];
            let bw = block_width(self.mi_size) as i64;
            let bh = block_height(self.mi_size) as i64;
            let x = (self.mi_col * MI_SIZE) as i64 + bw / 2 - 1;
            let y = (self.mi_row * MI_SIZE) as i64 + bh / 2 - 1;
            let xc = (i64::from(gm[2]) - (1 << WARPEDMODEL_PREC_BITS)) * x
                + i64::from(gm[3]) * y
                + i64::from(gm[0]);
            let yc = i64::from(gm[4]) * x
                + (i64::from(gm[5]) - (1 << WARPEDMODEL_PREC_BITS)) * y
                + i64::from(gm[1]);
            if self.fh.allow_high_precision_mv {
                mv[0] = round2signed(yc, (WARPEDMODEL_PREC_BITS - 3) as u32) as i32;
                mv[1] = round2signed(xc, (WARPEDMODEL_PREC_BITS - 3) as u32) as i32;
            } else {
                mv[0] = round2signed(yc, (WARPEDMODEL_PREC_BITS - 2) as u32) as i32 * 2;
                mv[1] = round2signed(xc, (WARPEDMODEL_PREC_BITS - 2) as u32) as i32 * 2;
            }
        }
        self.lower_mv_precision(&mut mv);
        mv
    }

    pub(crate) fn lower_mv_precision(&self, mv: &mut [i32; 2]) {
        if self.fh.allow_high_precision_mv {
            return;
        }
        for c in mv.iter_mut() {
            if self.fh.force_integer_mv {
                let a = c.abs();
                let a_int = (a + 3) >> 3;
                if *c > 0 {
                    *c = a_int << 3;
                } else {
                    *c = -(a_int << 3);
                }
            } else if *c & 1 != 0 {
                if *c > 0 {
                    *c -= 1;
                } else {
                    *c += 1;
                }
            }
        }
    }

    fn scan_row(&mut self, delta_row: isize, is_compound: bool) {
        let bw4 = bw4(self.mi_size);
        let end4 = bw4.min(self.fh.mi_cols - self.mi_col).min(16);
        let mut delta_col: isize = 0;
        let use_step16 = bw4 >= 16;
        let mut delta_row = delta_row;
        if delta_row.abs() > 1 {
            delta_row += (self.mi_row & 1) as isize;
            delta_col = 1 - (self.mi_col & 1) as isize;
        }
        let mut i = 0usize;
        while i < end4 {
            let mv_row = self.mi_row as isize + delta_row;
            let mv_col = self.mi_col as isize + delta_col + i as isize;
            if !self.is_inside(mv_row, mv_col) {
                break;
            }
            let cand_size =
                self.f.mi.mi_sizes[self.mi_idx(mv_row as usize, mv_col as usize)] as usize;
            let mut len = bw4.min(super::tile::bw4(cand_size));
            if delta_row.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            let weight = len as u32 * 2;
            self.add_ref_mv_candidate(mv_row as usize, mv_col as usize, is_compound, weight);
            i += len;
        }
    }

    fn scan_col(&mut self, delta_col: isize, is_compound: bool) {
        let bh4 = bh4(self.mi_size);
        let end4 = bh4.min(self.fh.mi_rows - self.mi_row).min(16);
        let mut delta_row: isize = 0;
        let use_step16 = bh4 >= 16;
        let mut delta_col = delta_col;
        if delta_col.abs() > 1 {
            delta_row = 1 - (self.mi_row & 1) as isize;
            delta_col += (self.mi_col & 1) as isize;
        }
        let mut i = 0usize;
        while i < end4 {
            let mv_row = self.mi_row as isize + delta_row + i as isize;
            let mv_col = self.mi_col as isize + delta_col;
            if !self.is_inside(mv_row, mv_col) {
                break;
            }
            let cand_size =
                self.f.mi.mi_sizes[self.mi_idx(mv_row as usize, mv_col as usize)] as usize;
            let mut len = bh4.min(super::tile::bh4(cand_size));
            if delta_col.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            let weight = len as u32 * 2;
            self.add_ref_mv_candidate(mv_row as usize, mv_col as usize, is_compound, weight);
            i += len;
        }
    }

    fn scan_point(&mut self, delta_row: isize, delta_col: isize, is_compound: bool) {
        let mv_row = self.mi_row as isize + delta_row;
        let mv_col = self.mi_col as isize + delta_col;
        let weight = 4;
        if self.is_inside(mv_row, mv_col)
            && mv_row < self.fh.mi_rows as isize
            && mv_col < self.fh.mi_cols as isize
            && self.f.mi.ref_frames_written[self.mi_idx(mv_row as usize, mv_col as usize)]
        {
            self.add_ref_mv_candidate(mv_row as usize, mv_col as usize, is_compound, weight);
        }
    }

    fn temporal_scan(&mut self, is_compound: bool) {
        let bw4 = bw4(self.mi_size);
        let bh4 = bh4(self.mi_size);
        let step_w4 = if bw4 >= 16 { 4 } else { 2 };
        let step_h4 = if bh4 >= 16 { 4 } else { 2 };
        let mut delta_row = 0;
        while delta_row < bh4.min(16) {
            let mut delta_col = 0;
            while delta_col < bw4.min(16) {
                self.add_tpl_ref_mv(delta_row as isize, delta_col as isize, is_compound);
                delta_col += step_w4;
            }
            delta_row += step_h4;
        }
        let allow_extension = bh4 >= super::tile::bh4(BLOCK_8X8)
            && bh4 < super::tile::bh4(BLOCK_64X64)
            && bw4 >= super::tile::bw4(BLOCK_8X8)
            && bw4 < super::tile::bw4(BLOCK_64X64);
        if allow_extension {
            let positions = [
                (bh4 as isize, -2isize),
                (bh4 as isize, bw4 as isize),
                (bh4 as isize - 2, bw4 as isize),
            ];
            for (delta_row, delta_col) in positions {
                if self.check_sb_border(delta_row, delta_col) {
                    self.add_tpl_ref_mv(delta_row, delta_col, is_compound);
                }
            }
        }
    }

    fn check_sb_border(&self, delta_row: isize, delta_col: isize) -> bool {
        let row = (self.mi_row & 15) as isize + delta_row;
        let col = (self.mi_col & 15) as isize + delta_col;
        (0..16).contains(&row) && (0..16).contains(&col)
    }

    fn add_tpl_ref_mv(&mut self, delta_row: isize, delta_col: isize, is_compound: bool) {
        let mv_row = (self.mi_row as isize + delta_row) | 1;
        let mv_col = (self.mi_col as isize + delta_col) | 1;
        if !self.is_inside(mv_row, mv_col) {
            return;
        }
        let x8 = (mv_col >> 1) as usize;
        let y8 = (mv_row >> 1) as usize;
        let w8 = self.fh.mi_cols >> 1;
        let h8 = self.fh.mi_rows >> 1;
        if x8 >= w8 || y8 >= h8 {
            return;
        }
        let field_idx = y8 * w8 + x8;
        if delta_row == 0 && delta_col == 0 {
            self.zero_mv_context = 1;
        }
        if !is_compound {
            let mut cand_mv = self.f.motion_field_mvs[self.ref_frame[0] as usize][field_idx];
            if cand_mv[0] == INVALID_MV {
                return;
            }
            self.lower_mv_precision(&mut cand_mv);
            if delta_row == 0 && delta_col == 0 {
                self.zero_mv_context = usize::from(
                    (cand_mv[0] - self.global_mvs[0][0]).abs() >= 16
                        || (cand_mv[1] - self.global_mvs[0][1]).abs() >= 16,
                );
            }
            let mut idx = 0;
            while idx < self.num_mv_found {
                if cand_mv == self.ref_stack_mv[idx][0] {
                    break;
                }
                idx += 1;
            }
            if idx < self.num_mv_found {
                self.weight_stack[idx] += 2;
            } else if self.num_mv_found < MAX_REF_MV_STACK_SIZE {
                self.ref_stack_mv[self.num_mv_found][0] = cand_mv;
                self.weight_stack[self.num_mv_found] = 2;
                self.num_mv_found += 1;
            }
        } else {
            let mut cand_mv0 = self.f.motion_field_mvs[self.ref_frame[0] as usize][field_idx];
            if cand_mv0[0] == INVALID_MV {
                return;
            }
            let mut cand_mv1 = self.f.motion_field_mvs[self.ref_frame[1] as usize][field_idx];
            if cand_mv1[0] == INVALID_MV {
                return;
            }
            self.lower_mv_precision(&mut cand_mv0);
            self.lower_mv_precision(&mut cand_mv1);
            if delta_row == 0 && delta_col == 0 {
                self.zero_mv_context = usize::from(
                    (cand_mv0[0] - self.global_mvs[0][0]).abs() >= 16
                        || (cand_mv0[1] - self.global_mvs[0][1]).abs() >= 16
                        || (cand_mv1[0] - self.global_mvs[1][0]).abs() >= 16
                        || (cand_mv1[1] - self.global_mvs[1][1]).abs() >= 16,
                );
            }
            let mut idx = 0;
            while idx < self.num_mv_found {
                if cand_mv0 == self.ref_stack_mv[idx][0] && cand_mv1 == self.ref_stack_mv[idx][1] {
                    break;
                }
                idx += 1;
            }
            if idx < self.num_mv_found {
                self.weight_stack[idx] += 2;
            } else if self.num_mv_found < MAX_REF_MV_STACK_SIZE {
                self.ref_stack_mv[self.num_mv_found][0] = cand_mv0;
                self.ref_stack_mv[self.num_mv_found][1] = cand_mv1;
                self.weight_stack[self.num_mv_found] = 2;
                self.num_mv_found += 1;
            }
        }
    }

    fn add_ref_mv_candidate(
        &mut self,
        mv_row: usize,
        mv_col: usize,
        is_compound: bool,
        weight: u32,
    ) {
        let i = self.mi_idx(mv_row, mv_col);
        if !self.f.mi.is_inters[i] {
            return;
        }
        let ref_frames = self.f.mi.ref_frames[i];
        if !is_compound {
            for cand_list in 0..2 {
                if ref_frames[cand_list] == self.ref_frame[0] {
                    self.search_stack(mv_row, mv_col, cand_list, weight);
                }
            }
        } else if ref_frames[0] == self.ref_frame[0] && ref_frames[1] == self.ref_frame[1] {
            self.compound_search_stack(mv_row, mv_col, weight);
        }
    }

    fn search_stack(&mut self, mv_row: usize, mv_col: usize, cand_list: usize, weight: u32) {
        let i = self.mi_idx(mv_row, mv_col);
        let cand_mode = self.f.mi.y_modes[i];
        let cand_size = self.f.mi.mi_sizes[i] as usize;
        let large = block_width(cand_size).min(block_height(cand_size)) >= 8;
        let mut cand_mv = if (cand_mode == GLOBALMV || cand_mode == GLOBAL_GLOBALMV)
            && self.fh.gm_type[self.ref_frame[0] as usize] > TRANSLATION
            && large
        {
            self.global_mvs[0]
        } else {
            self.f.mi.mvs[i][cand_list]
        };
        self.lower_mv_precision(&mut cand_mv);
        if has_newmv(cand_mode) {
            self.new_mv_count += 1;
        }
        self.found_match = true;
        let mut idx = 0;
        while idx < self.num_mv_found {
            if cand_mv == self.ref_stack_mv[idx][0] {
                break;
            }
            idx += 1;
        }
        if idx < self.num_mv_found {
            self.weight_stack[idx] += weight;
        } else if self.num_mv_found < MAX_REF_MV_STACK_SIZE {
            self.ref_stack_mv[self.num_mv_found][0] = cand_mv;
            self.weight_stack[self.num_mv_found] = weight;
            self.num_mv_found += 1;
        }
    }

    fn compound_search_stack(&mut self, mv_row: usize, mv_col: usize, weight: u32) {
        let i = self.mi_idx(mv_row, mv_col);
        let mut cand_mvs = self.f.mi.mvs[i];
        let cand_mode = self.f.mi.y_modes[i];
        let cand_size = self.f.mi.mi_sizes[i] as usize;
        if cand_mode == GLOBAL_GLOBALMV {
            for ref_list in 0..2 {
                if self.fh.gm_type[self.ref_frame[ref_list] as usize] > TRANSLATION {
                    cand_mvs[ref_list] = self.global_mvs[ref_list];
                }
            }
        }
        let _ = cand_size;
        for mv in cand_mvs.iter_mut() {
            self.lower_mv_precision(mv);
        }
        self.found_match = true;
        let mut idx = 0;
        while idx < self.num_mv_found {
            if cand_mvs[0] == self.ref_stack_mv[idx][0] && cand_mvs[1] == self.ref_stack_mv[idx][1]
            {
                break;
            }
            idx += 1;
        }
        if idx < self.num_mv_found {
            self.weight_stack[idx] += weight;
        } else if self.num_mv_found < MAX_REF_MV_STACK_SIZE {
            self.ref_stack_mv[self.num_mv_found] = cand_mvs;
            self.weight_stack[self.num_mv_found] = weight;
            self.num_mv_found += 1;
        }
        if has_newmv(cand_mode) {
            self.new_mv_count += 1;
        }
    }

    fn sort_stack(&mut self, start: usize, end: usize, is_compound: bool) {
        let mut end = end;
        while end > start {
            let mut new_end = start;
            for idx in start + 1..end {
                if self.weight_stack[idx - 1] < self.weight_stack[idx] {
                    self.weight_stack.swap(idx - 1, idx);
                    let lists = 1 + usize::from(is_compound);
                    for list in 0..lists {
                        let tmp = self.ref_stack_mv[idx - 1][list];
                        self.ref_stack_mv[idx - 1][list] = self.ref_stack_mv[idx][list];
                        self.ref_stack_mv[idx][list] = tmp;
                    }
                    new_end = idx;
                }
            }
            end = new_end;
        }
    }

    fn extra_search(&mut self, is_compound: bool) {
        self.ref_id_count = [0; 2];
        self.ref_diff_count = [0; 2];
        let mut w4 = 16.min(bw4(self.mi_size));
        let mut h4 = 16.min(bh4(self.mi_size));
        w4 = w4.min(self.fh.mi_cols - self.mi_col);
        h4 = h4.min(self.fh.mi_rows - self.mi_row);
        let num4x4 = w4.min(h4);
        for pass in 0..2 {
            let mut idx = 0;
            while idx < num4x4 && self.num_mv_found < 2 {
                let (mv_row, mv_col) = if pass == 0 {
                    (self.mi_row as isize - 1, (self.mi_col + idx) as isize)
                } else {
                    ((self.mi_row + idx) as isize, self.mi_col as isize - 1)
                };
                if !self.is_inside(mv_row, mv_col) {
                    break;
                }
                self.add_extra_mv_candidate(mv_row as usize, mv_col as usize, is_compound);
                let cand_size =
                    self.f.mi.mi_sizes[self.mi_idx(mv_row as usize, mv_col as usize)] as usize;
                if pass == 0 {
                    idx += bw4(cand_size);
                } else {
                    idx += bh4(cand_size);
                }
            }
        }
        if is_compound {
            let mut combined_mvs = [[[0i32; 2]; 2]; 2];
            for list in 0..2 {
                let mut comp_count = 0;
                for idx in 0..self.ref_id_count[list] {
                    combined_mvs[comp_count][list] = self.ref_id_mvs[list][idx];
                    comp_count += 1;
                }
                let mut idx = 0;
                while idx < self.ref_diff_count[list] && comp_count < 2 {
                    combined_mvs[comp_count][list] = self.ref_diff_mvs[list][idx];
                    comp_count += 1;
                    idx += 1;
                }
                while comp_count < 2 {
                    combined_mvs[comp_count][list] = self.global_mvs[list];
                    comp_count += 1;
                }
            }
            if self.num_mv_found == 1 {
                if combined_mvs[0][0] == self.ref_stack_mv[0][0]
                    && combined_mvs[0][1] == self.ref_stack_mv[0][1]
                {
                    self.ref_stack_mv[self.num_mv_found] = combined_mvs[1];
                } else {
                    self.ref_stack_mv[self.num_mv_found] = combined_mvs[0];
                }
                self.weight_stack[self.num_mv_found] = 2;
                self.num_mv_found += 1;
            } else {
                for combined in combined_mvs {
                    self.ref_stack_mv[self.num_mv_found] = combined;
                    self.weight_stack[self.num_mv_found] = 2;
                    self.num_mv_found += 1;
                }
            }
        } else {
            for idx in self.num_mv_found..2 {
                self.ref_stack_mv[idx][0] = self.global_mvs[0];
            }
        }
    }

    fn add_extra_mv_candidate(&mut self, mv_row: usize, mv_col: usize, is_compound: bool) {
        let i = self.mi_idx(mv_row, mv_col);
        if is_compound {
            for cand_list in 0..2 {
                let cand_ref = self.f.mi.ref_frames[i][cand_list];
                if cand_ref > INTRA_FRAME {
                    for list in 0..2 {
                        let mut cand_mv = self.f.mi.mvs[i][cand_list];
                        if cand_ref == self.ref_frame[list] && self.ref_id_count[list] < 2 {
                            self.ref_id_mvs[list][self.ref_id_count[list]] = cand_mv;
                            self.ref_id_count[list] += 1;
                        } else if self.ref_diff_count[list] < 2 {
                            if self.fh.ref_frame_sign_bias[cand_ref as usize]
                                != self.fh.ref_frame_sign_bias[self.ref_frame[list] as usize]
                            {
                                cand_mv[0] *= -1;
                                cand_mv[1] *= -1;
                            }
                            self.ref_diff_mvs[list][self.ref_diff_count[list]] = cand_mv;
                            self.ref_diff_count[list] += 1;
                        }
                    }
                }
            }
        } else {
            for cand_list in 0..2 {
                let cand_ref = self.f.mi.ref_frames[i][cand_list];
                if cand_ref > INTRA_FRAME {
                    let mut cand_mv = self.f.mi.mvs[i][cand_list];
                    if self.fh.ref_frame_sign_bias[cand_ref as usize]
                        != self.fh.ref_frame_sign_bias[self.ref_frame[0] as usize]
                    {
                        cand_mv[0] *= -1;
                        cand_mv[1] *= -1;
                    }
                    let mut idx = 0;
                    while idx < self.num_mv_found {
                        if cand_mv == self.ref_stack_mv[idx][0] {
                            break;
                        }
                        idx += 1;
                    }
                    if idx == self.num_mv_found {
                        self.ref_stack_mv[idx][0] = cand_mv;
                        self.weight_stack[idx] = 2;
                        self.num_mv_found += 1;
                    }
                }
            }
        }
    }

    fn context_and_clamping(&mut self, is_compound: bool, num_new: usize) {
        let bw = block_width(self.mi_size) as i32;
        let bh = block_height(self.mi_size) as i32;
        let num_lists = if is_compound { 2 } else { 1 };
        for idx in 0..self.num_mv_found {
            let mut z = 0;
            if idx + 1 < self.num_mv_found {
                let w0 = self.weight_stack[idx];
                let w1 = self.weight_stack[idx + 1];
                if w0 >= REF_CAT_LEVEL {
                    if w1 < REF_CAT_LEVEL {
                        z = 1;
                    }
                } else {
                    z = 2;
                }
            }
            self.drl_ctx_stack[idx] = z;
        }
        for list in 0..num_lists {
            for idx in 0..self.num_mv_found {
                let mut ref_mv = self.ref_stack_mv[idx][list];
                ref_mv[0] = self.clamp_mv_row(ref_mv[0], MV_BORDER + bh * 8);
                ref_mv[1] = self.clamp_mv_col(ref_mv[1], MV_BORDER + bw * 8);
                self.ref_stack_mv[idx][list] = ref_mv;
            }
        }
        if self.close_matches == 0 {
            self.new_mv_context = self.total_matches.min(1);
            self.ref_mv_context = self.total_matches;
        } else if self.close_matches == 1 {
            self.new_mv_context = 3 - num_new.min(1);
            self.ref_mv_context = 2 + self.total_matches;
        } else {
            self.new_mv_context = 5 - num_new.min(1);
            self.ref_mv_context = 5;
        }
    }

    pub(crate) fn clamp_mv_row(&self, mvec: i32, border: i32) -> i32 {
        let bh4 = bh4(self.mi_size) as i32;
        let mb_to_top_edge = -((self.mi_row as i32 * MI_SIZE as i32) * 8);
        let mb_to_bottom_edge =
            ((self.fh.mi_rows as i32 - bh4 - self.mi_row as i32) * MI_SIZE as i32) * 8;
        mvec.clamp(mb_to_top_edge - border, mb_to_bottom_edge + border)
    }

    pub(crate) fn clamp_mv_col(&self, mvec: i32, border: i32) -> i32 {
        let bw4 = bw4(self.mi_size) as i32;
        let mb_to_left_edge = -((self.mi_col as i32 * MI_SIZE as i32) * 8);
        let mb_to_right_edge =
            ((self.fh.mi_cols as i32 - bw4 - self.mi_col as i32) * MI_SIZE as i32) * 8;
        mvec.clamp(mb_to_left_edge - border, mb_to_right_edge + border)
    }

    /// `has_overlappable_candidates( )`.
    pub(crate) fn has_overlappable_candidates(&self) -> bool {
        if self.avail_u {
            let w4 = bw4(self.mi_size);
            let mut x4 = self.mi_col;
            while x4 < self.fh.mi_cols.min(self.mi_col + w4) {
                let col = (x4 | 1).min(self.fh.mi_cols - 1);
                if self.f.mi.ref_frames[self.mi_idx(self.mi_row - 1, col)][0] > INTRA_FRAME {
                    return true;
                }
                x4 += 2;
            }
        }
        if self.avail_l {
            let h4 = bh4(self.mi_size);
            let mut y4 = self.mi_row;
            while y4 < self.fh.mi_rows.min(self.mi_row + h4) {
                let row = (y4 | 1).min(self.fh.mi_rows - 1);
                if self.f.mi.ref_frames[self.mi_idx(row, self.mi_col - 1)][0] > INTRA_FRAME {
                    return true;
                }
                y4 += 2;
            }
        }
        false
    }

    /// `find_warp_samples( )`.
    pub(crate) fn find_warp_samples(&mut self) {
        self.num_samples = 0;
        self.num_samples_scanned = 0;
        let w4 = bw4(self.mi_size);
        let h4 = bh4(self.mi_size);
        let mut do_top_left = true;
        let mut do_top_right = true;
        if self.avail_u {
            let src_size = self.f.mi.mi_sizes[self.mi_idx(self.mi_row - 1, self.mi_col)] as usize;
            let src_w = bw4(src_size);
            if w4 <= src_w {
                let col_offset = -((self.mi_col & (src_w - 1)) as isize);
                if col_offset < 0 {
                    do_top_left = false;
                }
                if col_offset + src_w as isize > w4 as isize {
                    do_top_right = false;
                }
                self.add_sample(-1, 0);
            } else {
                let mut i = 0;
                while i < w4.min(self.fh.mi_cols - self.mi_col) {
                    let src_size =
                        self.f.mi.mi_sizes[self.mi_idx(self.mi_row - 1, self.mi_col + i)] as usize;
                    let src_w = bw4(src_size);
                    let mi_step = w4.min(src_w);
                    self.add_sample(-1, i as isize);
                    i += mi_step;
                }
            }
        }
        if self.avail_l {
            let src_size = self.f.mi.mi_sizes[self.mi_idx(self.mi_row, self.mi_col - 1)] as usize;
            let src_h = bh4(src_size);
            if h4 <= src_h {
                let row_offset = -((self.mi_row & (src_h - 1)) as isize);
                if row_offset < 0 {
                    do_top_left = false;
                }
                self.add_sample(0, -1);
            } else {
                let mut i = 0;
                while i < h4.min(self.fh.mi_rows - self.mi_row) {
                    let src_size =
                        self.f.mi.mi_sizes[self.mi_idx(self.mi_row + i, self.mi_col - 1)] as usize;
                    let src_h = bh4(src_size);
                    let mi_step = h4.min(src_h);
                    self.add_sample(i as isize, -1);
                    i += mi_step;
                }
            }
        }
        if do_top_left {
            self.add_sample(-1, -1);
        }
        if do_top_right && w4.max(h4) <= 16 {
            self.add_sample(-1, w4 as isize);
        }
        if self.num_samples == 0 && self.num_samples_scanned > 0 {
            self.num_samples = 1;
        }
    }

    fn add_sample(&mut self, delta_row: isize, delta_col: isize) {
        if self.num_samples_scanned >= LEAST_SQUARES_SAMPLES_MAX {
            return;
        }
        let mv_row = self.mi_row as isize + delta_row;
        let mv_col = self.mi_col as isize + delta_col;
        if !self.is_inside(mv_row, mv_col)
            || mv_row >= self.fh.mi_rows as isize
            || mv_col >= self.fh.mi_cols as isize
        {
            return;
        }
        let i = self.mi_idx(mv_row as usize, mv_col as usize);
        if !self.f.mi.ref_frames_written[i] {
            return;
        }
        if self.f.mi.ref_frames[i][0] != self.ref_frame[0] {
            return;
        }
        if self.f.mi.ref_frames[i][1] != NONE {
            return;
        }
        let cand_sz = self.f.mi.mi_sizes[i] as usize;
        let cand_w4 = bw4(cand_sz);
        let cand_h4 = bh4(cand_sz);
        let cand_row = (mv_row as usize) & !(cand_h4 - 1);
        let cand_col = (mv_col as usize) & !(cand_w4 - 1);
        let mid_y = (cand_row * 4 + cand_h4 * 2) as i32 - 1;
        let mid_x = (cand_col * 4 + cand_w4 * 2) as i32 - 1;
        let threshold =
            (block_width(self.mi_size).max(block_height(self.mi_size)) as i32).clamp(16, 112);
        let cand_mv = self.f.mi.mvs[self.mi_idx(cand_row, cand_col)][0];
        let mv_diff_row = (cand_mv[0] - self.mv[0][0]).abs();
        let mv_diff_col = (cand_mv[1] - self.mv[0][1]).abs();
        let valid = mv_diff_row + mv_diff_col <= threshold;
        let cand = [
            mid_y * 8,
            mid_x * 8,
            mid_y * 8 + cand_mv[0],
            mid_x * 8 + cand_mv[1],
        ];
        self.num_samples_scanned += 1;
        if !valid && self.num_samples_scanned > 1 {
            return;
        }
        self.cand_list[self.num_samples] = cand;
        if valid {
            self.num_samples += 1;
        }
    }
}
