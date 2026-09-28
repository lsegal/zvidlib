//! In-loop filtering (specification sections 7.14 to 7.17): the deblocking
//! loop filter, CDEF, super-resolution upscaling and loop restoration.

use super::consts::*;
use super::frame::{FrameBuf, FrameState, PlaneBuf};
use super::header::{FrameHeader, SequenceHeader};
use super::tables::*;

#[inline(always)]
fn round2(x: i64, n: u32) -> i64 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

struct Ctx<'a> {
    seq: &'a SequenceHeader,
    fh: &'a FrameHeader,
    f: &'a FrameState,
}

impl Ctx<'_> {
    fn mi(&self, row: usize, col: usize) -> usize {
        row * self.fh.mi_cols + col
    }

    fn sub(&self, plane: usize) -> (usize, usize) {
        if plane == 0 {
            (0, 0)
        } else {
            (
                usize::from(self.seq.subsampling_x),
                usize::from(self.seq.subsampling_y),
            )
        }
    }

    fn lf_tx_size(&self, plane: usize, row: usize, col: usize) -> usize {
        let stride = self.f.lf_tx_stride[plane];
        self.f.lf_tx_sizes[plane]
            .get(row * stride + col)
            .copied()
            .unwrap_or(0) as usize
    }

    /// The adaptive filter strength process, returning `(lvl, limit, blimit, thresh)`.
    fn filter_strength(
        &self,
        row: usize,
        col: usize,
        plane: usize,
        pass: usize,
    ) -> (i32, i32, i32, i32) {
        let i = self.mi(row, col);
        let segment = self.f.mi.segment_ids[i] as usize;
        let rf = self.f.mi.ref_frames[i][0];
        let mode = self.f.mi.y_modes[i];
        let mode_type =
            usize::from(mode >= NEARESTMV && mode != GLOBALMV && mode != GLOBAL_GLOBALMV);
        let delta_lf = if !self.fh.delta_lf_multi {
            i32::from(self.f.mi.delta_lfs[i][0])
        } else {
            i32::from(self.f.mi.delta_lfs[i][if plane == 0 { pass } else { plane + 1 }])
        };
        let li = if plane == 0 { pass } else { plane + 1 };
        let lf = &self.fh.loop_filter;
        let base_filter_level = (delta_lf + i32::from(lf.level[li])).clamp(0, MAX_LOOP_FILTER);
        let mut lvl_seg = base_filter_level;
        let feature = SEG_LVL_ALT_LF_Y_V + li;
        if self.fh.segmentation.feature_active(segment, feature) {
            lvl_seg = (i32::from(self.fh.segmentation.feature_data[segment][feature]) + lvl_seg)
                .clamp(0, MAX_LOOP_FILTER);
        }
        if lf.delta_enabled {
            let n_shift = lvl_seg >> 5;
            if rf == INTRA_FRAME {
                lvl_seg += i32::from(lf.ref_deltas[INTRA_FRAME as usize]) << n_shift;
            } else if rf > INTRA_FRAME {
                lvl_seg += (i32::from(lf.ref_deltas[rf as usize]) << n_shift)
                    + (i32::from(lf.mode_deltas[mode_type]) << n_shift);
            }
            lvl_seg = lvl_seg.clamp(0, MAX_LOOP_FILTER);
        }
        let lvl = lvl_seg;
        let sharpness = i32::from(lf.sharpness);
        let shift = if sharpness > 4 {
            2
        } else if sharpness > 0 {
            1
        } else {
            0
        };
        let limit = if sharpness > 0 {
            (lvl >> shift).clamp(1, 9 - sharpness)
        } else {
            (lvl >> shift).max(1)
        };
        let blimit = 2 * (lvl + 2) + limit;
        let thresh = lvl >> 4;
        (lvl, limit, blimit, thresh)
    }
}

/// The loop filter process (section 7.14), in place on `curr`.
pub(crate) fn loop_filter(seq: &SequenceHeader, fh: &FrameHeader, f: &mut FrameState) {
    let mut planes = std::mem::take(&mut f.curr.planes);
    {
        let ctx = Ctx { seq, fh, f };
        for plane in 0..seq.num_planes {
            if plane == 0 || fh.loop_filter.level[1 + plane] != 0 {
                for pass in 0..2 {
                    let row_step = if plane == 0 {
                        1
                    } else {
                        1 << usize::from(seq.subsampling_y)
                    };
                    let col_step = if plane == 0 {
                        1
                    } else {
                        1 << usize::from(seq.subsampling_x)
                    };
                    let mut row = 0;
                    while row < fh.mi_rows {
                        let mut col = 0;
                        while col < fh.mi_cols {
                            edge_loop_filter(&ctx, &mut planes[plane], plane, pass, row, col);
                            col += col_step;
                        }
                        row += row_step;
                    }
                }
            }
        }
    }
    f.curr.planes = planes;
}

fn edge_loop_filter(
    ctx: &Ctx<'_>,
    buf: &mut PlaneBuf,
    plane: usize,
    pass: usize,
    row: usize,
    col: usize,
) {
    let (sub_x, sub_y) = ctx.sub(plane);
    let (dx, dy) = if pass == 0 { (1isize, 0isize) } else { (0, 1) };
    let x = col * MI_SIZE;
    let y = row * MI_SIZE;
    let row = row | sub_y;
    let col = col | sub_x;
    let on_screen = !(x >= ctx.fh.frame_width
        || y >= ctx.fh.frame_height
        || (pass == 0 && x == 0)
        || (pass == 1 && y == 0));
    if !on_screen {
        return;
    }
    let xp = x >> sub_x;
    let yp = y >> sub_y;
    let prev_row = row - ((dy as usize) << sub_y);
    let prev_col = col - ((dx as usize) << sub_x);
    let row_c = row.min(ctx.fh.mi_rows - 1);
    let col_c = col.min(ctx.fh.mi_cols - 1);
    let mi_size = ctx.f.mi.mi_sizes[ctx.mi(row_c, col_c)] as usize;
    let tx_sz = ctx.lf_tx_size(plane, row >> sub_y, col >> sub_x);
    let plane_size = SUBSAMPLED_SIZE[mi_size][sub_x][sub_y] as usize;
    let skip = ctx.f.mi.skips[ctx.mi(row_c, col_c)];
    let is_intra = ctx.f.mi.ref_frames[ctx.mi(row_c, col_c)][0] <= INTRA_FRAME;
    let prev_tx_sz = ctx.lf_tx_size(plane, prev_row >> sub_y, prev_col >> sub_x);
    let is_block_edge = if pass == 0 {
        xp % block_width(plane_size.min(BLOCK_SIZES - 1)) == 0
    } else {
        yp % block_height(plane_size.min(BLOCK_SIZES - 1)) == 0
    };
    let is_tx_edge = if pass == 0 {
        xp % TX_WIDTH[tx_sz] as usize == 0
    } else {
        yp % TX_HEIGHT[tx_sz] as usize == 0
    };
    let apply_filter = if !is_tx_edge {
        false
    } else {
        is_block_edge || !skip || is_intra
    };
    let base_size = if pass == 0 {
        TX_WIDTH[prev_tx_sz].min(TX_WIDTH[tx_sz]) as usize
    } else {
        TX_HEIGHT[prev_tx_sz].min(TX_HEIGHT[tx_sz]) as usize
    };
    let filter_size = if plane == 0 {
        16.min(base_size)
    } else {
        8.min(base_size)
    };
    let (mut lvl, mut limit, mut blimit, mut thresh) =
        ctx.filter_strength(row_c, col_c, plane, pass);
    if lvl == 0 {
        (lvl, limit, blimit, thresh) = ctx.filter_strength(
            prev_row.min(ctx.fh.mi_rows - 1),
            prev_col.min(ctx.fh.mi_cols - 1),
            plane,
            pass,
        );
    }
    if !apply_filter || lvl == 0 {
        return;
    }
    let bit_depth = u32::from(ctx.seq.bit_depth);
    for i in 0..MI_SIZE {
        let sx = xp as isize + dy * i as isize;
        let sy = yp as isize + dx * i as isize;
        if sx < 0 || sy < 0 || sx as usize >= buf.stride || sy as usize >= buf.height {
            continue;
        }
        sample_filter(
            buf,
            sx,
            sy,
            plane,
            limit,
            blimit,
            thresh,
            dx,
            dy,
            filter_size,
            bit_depth,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_filter(
    buf: &mut PlaneBuf,
    x: isize,
    y: isize,
    plane: usize,
    limit: i32,
    blimit: i32,
    thresh: i32,
    dx: isize,
    dy: isize,
    filter_size: usize,
    bit_depth: u32,
) {
    let get = |buf: &PlaneBuf, k: isize| -> i32 {
        let xx = (x + dx * k).clamp(0, buf.stride as isize - 1) as usize;
        let yy = (y + dy * k).clamp(0, buf.height as isize - 1) as usize;
        i32::from(buf.get(xx, yy))
    };
    let q0 = get(buf, 0);
    let q1 = get(buf, 1);
    let q2 = get(buf, 2);
    let q3 = get(buf, 3);
    let p0 = get(buf, -1);
    let p1 = get(buf, -2);
    let p2 = get(buf, -3);
    let p3 = get(buf, -4);
    let shift = bit_depth - 8;
    let thresh_bd = thresh << shift;
    let hev_mask = (p1 - p0).abs() > thresh_bd || (q1 - q0).abs() > thresh_bd;
    let filter_len = if filter_size == 4 {
        4
    } else if plane != 0 {
        6
    } else if filter_size == 8 {
        8
    } else {
        16
    };
    let limit_bd = limit << shift;
    let blimit_bd = blimit << shift;
    let mut mask = false;
    mask |= (p1 - p0).abs() > limit_bd;
    mask |= (q1 - q0).abs() > limit_bd;
    mask |= (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit_bd;
    if filter_len >= 6 {
        mask |= (p2 - p1).abs() > limit_bd;
        mask |= (q2 - q1).abs() > limit_bd;
    }
    if filter_len >= 8 {
        mask |= (p3 - p2).abs() > limit_bd;
        mask |= (q3 - q2).abs() > limit_bd;
    }
    let filter_mask = !mask;
    if !filter_mask {
        return;
    }
    let threshold_bd = 1 << shift;
    let mut flat_mask = false;
    if filter_size >= 8 {
        let mut m = false;
        m |= (p1 - p0).abs() > threshold_bd;
        m |= (q1 - q0).abs() > threshold_bd;
        m |= (p2 - p0).abs() > threshold_bd;
        m |= (q2 - q0).abs() > threshold_bd;
        if filter_len >= 8 {
            m |= (p3 - p0).abs() > threshold_bd;
            m |= (q3 - q0).abs() > threshold_bd;
        }
        flat_mask = !m;
    }
    let mut flat_mask2 = false;
    if filter_size >= 16 {
        let q4 = get(buf, 4);
        let q5 = get(buf, 5);
        let q6 = get(buf, 6);
        let p4 = get(buf, -5);
        let p5 = get(buf, -6);
        let p6 = get(buf, -7);
        let mut m = false;
        m |= (p6 - p0).abs() > threshold_bd;
        m |= (q6 - q0).abs() > threshold_bd;
        m |= (p5 - p0).abs() > threshold_bd;
        m |= (q5 - q0).abs() > threshold_bd;
        m |= (p4 - p0).abs() > threshold_bd;
        m |= (q4 - q0).abs() > threshold_bd;
        flat_mask2 = !m;
    }
    if filter_size == 4 || !flat_mask {
        narrow_filter(buf, x, y, dx, dy, hev_mask, bit_depth);
    } else if filter_size == 8 || !flat_mask2 {
        wide_filter(buf, x, y, dx, dy, plane, 3);
    } else {
        wide_filter(buf, x, y, dx, dy, plane, 4);
    }
}

fn narrow_filter(
    buf: &mut PlaneBuf,
    x: isize,
    y: isize,
    dx: isize,
    dy: isize,
    hev_mask: bool,
    bit_depth: u32,
) {
    let at = |k: isize| ((x + dx * k) as usize, (y + dy * k) as usize);
    let get = |buf: &PlaneBuf, k: isize| {
        let (xx, yy) = at(k);
        i32::from(buf.get(xx, yy))
    };
    let lo = -(1i32 << (bit_depth - 1));
    let hi = (1i32 << (bit_depth - 1)) - 1;
    let c = |v: i32| v.clamp(lo, hi);
    let off = 0x80 << (bit_depth - 8);
    let q0 = get(buf, 0);
    let q1 = get(buf, 1);
    let p0 = get(buf, -1);
    let p1 = get(buf, -2);
    let ps1 = p1 - off;
    let ps0 = p0 - off;
    let qs0 = q0 - off;
    let qs1 = q1 - off;
    let mut filter = if hev_mask { c(ps1 - qs1) } else { 0 };
    filter = c(filter + 3 * (qs0 - ps0));
    let filter1 = c(filter + 4) >> 3;
    let filter2 = c(filter + 3) >> 3;
    let oq0 = c(qs0 - filter1) + off;
    let op0 = c(ps0 + filter2) + off;
    let (xx, yy) = at(0);
    buf.set(xx, yy, oq0 as u16);
    let (xx, yy) = at(-1);
    buf.set(xx, yy, op0 as u16);
    if !hev_mask {
        let filter = round2(i64::from(filter1), 1) as i32;
        let oq1 = c(qs1 - filter) + off;
        let op1 = c(ps1 + filter) + off;
        let (xx, yy) = at(1);
        buf.set(xx, yy, oq1 as u16);
        let (xx, yy) = at(-2);
        buf.set(xx, yy, op1 as u16);
    }
}

fn wide_filter(
    buf: &mut PlaneBuf,
    x: isize,
    y: isize,
    dx: isize,
    dy: isize,
    plane: usize,
    log2_size: u32,
) {
    let n: isize = if log2_size == 4 {
        6
    } else if plane == 0 {
        3
    } else {
        2
    };
    let n2: isize = if log2_size == 3 && plane == 0 { 0 } else { 1 };
    let get = |k: isize| {
        let xx = (x + dx * k) as usize;
        let yy = (y + dy * k) as usize;
        i64::from(buf.get(xx, yy))
    };
    let mut out = [0i64; 12];
    for i in -n..n {
        let mut t = 0i64;
        for j in -n..=n {
            let p = (i + j).clamp(-(n + 1), n);
            let tap = if j.abs() <= n2 { 2 } else { 1 };
            t += get(p) * tap;
        }
        out[(i + n) as usize] = round2(t, log2_size);
    }
    for i in -n..n {
        let xx = (x + dx * i) as usize;
        let yy = (y + dy * i) as usize;
        buf.set(xx, yy, out[(i + n) as usize] as u16);
    }
}

/// The CDEF process (section 7.15), producing `CdefFrame` from `CurrFrame`.
pub(crate) fn cdef(seq: &SequenceHeader, fh: &FrameHeader, f: &FrameState) -> FrameBuf {
    let mut cdef_frame = f.curr.clone();
    let ctx = Ctx { seq, fh, f };
    let step4 = 2;
    let cdef_size4 = 16;
    let cdef_mask4 = !(cdef_size4 - 1);
    let mut r = 0;
    while r < fh.mi_rows {
        let mut c = 0;
        while c < fh.mi_cols {
            let base_r = r & cdef_mask4;
            let base_c = c & cdef_mask4;
            let idx = f.cdef_idx[(base_r >> 4) * f.cdef_stride + (base_c >> 4)];
            cdef_block(&ctx, &mut cdef_frame, r, c, idx);
            c += step4;
        }
        r += step4;
    }
    cdef_frame
}

fn cdef_block(ctx: &Ctx<'_>, cdef_frame: &mut FrameBuf, r: usize, c: usize, idx: i8) {
    if idx == -1 {
        return;
    }
    let idx = idx as usize;
    let coeff_shift = u32::from(ctx.seq.bit_depth) - 8;
    let mi = &ctx.f.mi;
    let skip_at = |rr: usize, cc: usize| {
        if rr < ctx.fh.mi_rows && cc < ctx.fh.mi_cols {
            mi.skips[ctx.mi(rr, cc)]
        } else {
            true
        }
    };
    let skip = skip_at(r, c) && skip_at(r + 1, c) && skip_at(r, c + 1) && skip_at(r + 1, c + 1);
    if skip {
        return;
    }
    let (y_dir, var) = cdef_direction(ctx, r, c);
    let pri_str = i32::from(ctx.fh.cdef.y_pri_strength[idx]) << coeff_shift;
    let sec_str = i32::from(ctx.fh.cdef.y_sec_strength[idx]) << coeff_shift;
    let dir = if pri_str == 0 { 0 } else { y_dir };
    let var_str = if (var >> 6) != 0 {
        (63 - ((var >> 6) as u64).leading_zeros()).min(12) as i32
    } else {
        0
    };
    let pri_str = if var != 0 {
        (pri_str * (4 + var_str) + 8) >> 4
    } else {
        0
    };
    let damping = i32::from(ctx.fh.cdef.damping) + coeff_shift as i32;
    cdef_filter(ctx, cdef_frame, 0, r, c, pri_str, sec_str, damping, dir);
    if ctx.seq.num_planes == 1 {
        return;
    }
    let pri_str = i32::from(ctx.fh.cdef.uv_pri_strength[idx]) << coeff_shift;
    let sec_str = i32::from(ctx.fh.cdef.uv_sec_strength[idx]) << coeff_shift;
    let dir = if pri_str == 0 {
        0
    } else {
        CDEF_UV_DIR[usize::from(ctx.seq.subsampling_x)][usize::from(ctx.seq.subsampling_y)][y_dir]
            as usize
    };
    let damping = i32::from(ctx.fh.cdef.damping) + coeff_shift as i32 - 1;
    cdef_filter(ctx, cdef_frame, 1, r, c, pri_str, sec_str, damping, dir);
    cdef_filter(ctx, cdef_frame, 2, r, c, pri_str, sec_str, damping, dir);
}

fn cdef_direction(ctx: &Ctx<'_>, r: usize, c: usize) -> (usize, i32) {
    let mut cost = [0i64; 8];
    let mut partial = [[0i64; 15]; 8];
    let mut best_cost = 0i64;
    let mut y_dir = 0usize;
    let x0 = c << MI_SIZE_LOG2;
    let y0 = r << MI_SIZE_LOG2;
    let luma = &ctx.f.curr.planes[0];
    let shift = u32::from(ctx.seq.bit_depth) - 8;
    for i in 0..8 {
        for j in 0..8 {
            let x = (i64::from(luma.get(x0 + j, y0 + i)) >> shift) - 128;
            partial[0][i + j] += x;
            partial[1][i + j / 2] += x;
            partial[2][i] += x;
            partial[3][3 + i - j / 2] += x;
            partial[4][7 + i - j] += x;
            partial[5][3 - i / 2 + j] += x;
            partial[6][j] += x;
            partial[7][i / 2 + j] += x;
        }
    }
    let div = |i: usize| i64::from(DIV_TABLE[i]);
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= div(8);
    cost[6] *= div(8);
    for i in 0..7 {
        cost[0] +=
            (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i]) * div(i + 1);
        cost[4] +=
            (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i]) * div(i + 1);
    }
    cost[0] += partial[0][7] * partial[0][7] * div(8);
    cost[4] += partial[4][7] * partial[4][7] * div(8);
    let mut i = 1;
    while i < 8 {
        for j in 0..5 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= div(8);
        for j in 0..3 {
            cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j])
                * div(2 * j + 2);
        }
        i += 2;
    }
    for (i, &c) in cost.iter().enumerate() {
        if c > best_cost {
            best_cost = c;
            y_dir = i;
        }
    }
    let var = ((best_cost - cost[(y_dir + 4) & 7]) >> 10) as i32;
    (y_dir, var)
}

fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let damping_adj = (damping - (31 - (threshold as u32).leading_zeros()) as i32).max(0);
    let sign = if diff < 0 { -1 } else { 1 };
    sign * (threshold - (diff.abs() >> damping_adj)).clamp(0, diff.abs())
}

#[allow(clippy::too_many_arguments)]
fn cdef_filter(
    ctx: &Ctx<'_>,
    cdef_frame: &mut FrameBuf,
    plane: usize,
    r: usize,
    c: usize,
    pri_str: i32,
    sec_str: i32,
    damping: i32,
    dir: usize,
) {
    let coeff_shift = u32::from(ctx.seq.bit_depth) - 8;
    let (sub_x, sub_y) = ctx.sub(plane);
    let x0 = (c * MI_SIZE) >> sub_x;
    let y0 = (r * MI_SIZE) >> sub_y;
    let w = 8 >> sub_x;
    let h = 8 >> sub_y;
    let curr = &ctx.f.curr.planes[plane];
    let mi_rows = ctx.fh.mi_rows as isize;
    let mi_cols = ctx.fh.mi_cols as isize;
    let get_at = |i: usize, j: usize, dir: usize, k: usize, sign: isize| -> Option<i32> {
        let y = y0 as isize + i as isize + sign * CDEF_DIRECTIONS[dir][k][0] as isize;
        let x = x0 as isize + j as isize + sign * CDEF_DIRECTIONS[dir][k][1] as isize;
        let candidate_r = (y << sub_y) >> MI_SIZE_LOG2;
        let candidate_c = (x << sub_x) >> MI_SIZE_LOG2;
        if candidate_c >= 0 && candidate_c < mi_cols && candidate_r >= 0 && candidate_r < mi_rows {
            Some(i32::from(curr.get(x as usize, y as usize)))
        } else {
            None
        }
    };
    let pri_taps = &CDEF_PRI_TAPS[((pri_str >> coeff_shift) & 1) as usize];
    let sec_taps = &CDEF_SEC_TAPS[((pri_str >> coeff_shift) & 1) as usize];
    for i in 0..h {
        for j in 0..w {
            let mut sum = 0i32;
            let x = i32::from(curr.get(x0 + j, y0 + i));
            let mut max = x;
            let mut min = x;
            for k in 0..2 {
                for sign in [-1isize, 1] {
                    if let Some(p) = get_at(i, j, dir, k, sign) {
                        sum += i32::from(pri_taps[k]) * constrain(p - x, pri_str, damping);
                        max = max.max(p);
                        min = min.min(p);
                    }
                    for dir_off in [-2isize, 2] {
                        let d = ((dir as isize + dir_off) & 7) as usize;
                        if let Some(s) = get_at(i, j, d, k, sign) {
                            sum += i32::from(sec_taps[k]) * constrain(s - x, sec_str, damping);
                            max = max.max(s);
                            min = min.min(s);
                        }
                    }
                }
            }
            let v = (x + ((8 + sum - i32::from(sum < 0)) >> 4)).clamp(min, max);
            cdef_frame.planes[plane].set(x0 + j, y0 + i, v as u16);
        }
    }
}

/// The upscaling process (section 7.16).
pub(crate) fn upscale(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    input: &FrameBuf,
    alloc: impl Fn(usize, usize, usize) -> PlaneBuf,
) -> FrameBuf {
    if !fh.use_superres {
        return input.clone();
    }
    let mut planes = Vec::with_capacity(seq.num_planes);
    for plane in 0..seq.num_planes {
        let (sub_x, sub_y) = if plane > 0 {
            (
                usize::from(seq.subsampling_x),
                usize::from(seq.subsampling_y),
            )
        } else {
            (0, 0)
        };
        let downscaled_plane_w = round2(fh.frame_width as i64, sub_x as u32);
        let upscaled_plane_w = round2(fh.upscaled_width as i64, sub_x as u32);
        let plane_h = round2(fh.frame_height as i64, sub_y as u32) as usize;
        let step_x = ((downscaled_plane_w << SUPERRES_SCALE_BITS) + (upscaled_plane_w / 2))
            / upscaled_plane_w;
        let err = upscaled_plane_w * step_x - (downscaled_plane_w << SUPERRES_SCALE_BITS);
        let initial_subpel_x = ((-((upscaled_plane_w - downscaled_plane_w)
            << (SUPERRES_SCALE_BITS - 1))
            + upscaled_plane_w / 2)
            / upscaled_plane_w
            + (1 << (SUPERRES_EXTRA_BITS - 1))
            - err / 2)
            & i64::from(SUPERRES_SCALE_MASK);
        let mi_w = (fh.mi_cols >> sub_x) as i64;
        let max_x = mi_w * MI_SIZE as i64 - 1;
        let src = &input.planes[plane];
        let mut out = alloc(plane, upscaled_plane_w as usize, src.height);
        let max = (1i64 << seq.bit_depth) - 1;
        for y in 0..plane_h.min(src.height) {
            for x in 0..upscaled_plane_w as usize {
                let src_x = -(1i64 << SUPERRES_SCALE_BITS) + initial_subpel_x + x as i64 * step_x;
                let src_x_px = src_x >> SUPERRES_SCALE_BITS;
                let src_x_subpel =
                    ((src_x & i64::from(SUPERRES_SCALE_MASK)) >> SUPERRES_EXTRA_BITS) as usize;
                let mut sum = 0i64;
                for k in 0..SUPERRES_FILTER_TAPS as i64 {
                    let sample_x =
                        (src_x_px + (k - SUPERRES_FILTER_OFFSET as i64)).clamp(0, max_x) as usize;
                    let px = i64::from(src.get(sample_x.min(src.stride - 1), y));
                    sum += px * i64::from(UPSCALE_FILTER[src_x_subpel][k as usize]);
                }
                out.set(x, y, round2(sum, FILTER_BITS as u32).clamp(0, max) as u16);
            }
        }
        planes.push(out);
    }
    FrameBuf { planes }
}

/// The loop restoration process (section 7.17), producing `LrFrame`.
pub(crate) fn loop_restoration(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    f: &FrameState,
    upscaled_curr: &FrameBuf,
    upscaled_cdef: &FrameBuf,
) -> FrameBuf {
    let mut lr_frame = upscaled_cdef.clone();
    if !fh.uses_lr {
        return lr_frame;
    }
    let mut y = 0;
    while y < fh.frame_height {
        let mut x = 0;
        while x < fh.upscaled_width {
            for plane in 0..seq.num_planes {
                if fh.frame_restoration_type[plane] != RESTORE_NONE {
                    let row = y >> MI_SIZE_LOG2;
                    let col = x >> MI_SIZE_LOG2;
                    loop_restore_block(
                        seq,
                        fh,
                        f,
                        upscaled_curr,
                        upscaled_cdef,
                        &mut lr_frame,
                        plane,
                        row,
                        col,
                    );
                }
            }
            x += MI_SIZE;
        }
        y += MI_SIZE;
    }
    lr_frame
}

fn count_units_in_frame(unit_size: usize, frame_size: usize) -> usize {
    ((frame_size + (unit_size >> 1)) / unit_size).max(1)
}

pub(crate) fn lr_unit_counts(
    fh: &FrameHeader,
    plane: usize,
    sub_x: usize,
    sub_y: usize,
) -> (usize, usize) {
    let unit_size = fh.loop_restoration_size[plane];
    if unit_size == 0 {
        return (0, 0);
    }
    let rows = count_units_in_frame(
        unit_size,
        round2(fh.frame_height as i64, sub_y as u32) as usize,
    );
    let cols = count_units_in_frame(
        unit_size,
        round2(fh.upscaled_width as i64, sub_x as u32) as usize,
    );
    (rows, cols)
}

struct StripeSource<'a> {
    curr: &'a PlaneBuf,
    cdef: &'a PlaneBuf,
    stripe_start_y: isize,
    stripe_end_y: isize,
    plane_end_x: isize,
    plane_end_y: isize,
}

impl StripeSource<'_> {
    #[inline(always)]
    fn get(&self, x: isize, y: isize) -> i32 {
        let x = x.min(self.plane_end_x).max(0) as usize;
        let y = y.min(self.plane_end_y).max(0);
        if y < self.stripe_start_y {
            let y = (self.stripe_start_y - 2).max(y) as usize;
            i32::from(self.curr.get(x, y))
        } else if y > self.stripe_end_y {
            let y = (self.stripe_end_y + 2).min(y) as usize;
            i32::from(self.curr.get(x, y))
        } else {
            i32::from(self.cdef.get(x, y as usize))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn loop_restore_block(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    f: &FrameState,
    upscaled_curr: &FrameBuf,
    upscaled_cdef: &FrameBuf,
    lr_frame: &mut FrameBuf,
    plane: usize,
    row: usize,
    col: usize,
) {
    let luma_y = row * MI_SIZE;
    let stripe_num = (luma_y + 8) / 64;
    let (sub_x, sub_y) = if plane > 0 {
        (
            usize::from(seq.subsampling_x),
            usize::from(seq.subsampling_y),
        )
    } else {
        (0, 0)
    };
    let stripe_start_y = (-8 + stripe_num as isize * 64) >> sub_y;
    let stripe_end_y = stripe_start_y + (64 >> sub_y) - 1;
    let unit_size = fh.loop_restoration_size[plane];
    let lr = &f.lr[plane];
    let unit_rows = lr.unit_rows;
    let unit_cols = lr.unit_cols;
    let unit_row = (unit_rows - 1).min(((row * MI_SIZE + 8) >> sub_y) / unit_size);
    let unit_col = (unit_cols - 1).min(((col * MI_SIZE) >> sub_x) / unit_size);
    let plane_end_x = round2(fh.upscaled_width as i64, sub_x as u32) as isize - 1;
    let plane_end_y = round2(fh.frame_height as i64, sub_y as u32) as isize - 1;
    let x = ((col * MI_SIZE) >> sub_x) as isize;
    let y = ((row * MI_SIZE) >> sub_y) as isize;
    if x > plane_end_x || y > plane_end_y {
        return;
    }
    let w = ((MI_SIZE >> sub_x) as isize).min(plane_end_x - x + 1) as usize;
    let h = ((MI_SIZE >> sub_y) as isize).min(plane_end_y - y + 1) as usize;
    let unit = unit_row * unit_cols + unit_col;
    let r_type = lr.lr_type[unit];
    let src = StripeSource {
        curr: &upscaled_curr.planes[plane],
        cdef: &upscaled_cdef.planes[plane],
        stripe_start_y,
        stripe_end_y,
        plane_end_x,
        plane_end_y,
    };
    let bit_depth = u32::from(seq.bit_depth);
    let (x, y) = (x as usize, y as usize);
    if r_type == RESTORE_WIENER {
        wiener_filter(
            &src,
            &lr.wiener[unit],
            &mut lr_frame.planes[plane],
            x,
            y,
            w,
            h,
            bit_depth,
        );
    } else if r_type == RESTORE_SGRPROJ {
        self_guided_filter(
            &src,
            lr.sgr_set[unit] as usize,
            lr.sgr_xqd[unit],
            &mut lr_frame.planes[plane],
            x,
            y,
            w,
            h,
            bit_depth,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn wiener_filter(
    src: &StripeSource<'_>,
    coeffs: &[[i32; 3]; 2],
    out: &mut PlaneBuf,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    bit_depth: u32,
) {
    let inter_round0 = if bit_depth == 12 { 5 } else { 3 };
    let inter_round1 = if bit_depth == 12 { 9 } else { 11 };
    let coefficients = |c: &[i32; 3]| {
        let mut filter = [0i32; 7];
        filter[3] = 128;
        for i in 0..3 {
            filter[i] = c[i];
            filter[6 - i] = c[i];
            filter[3] -= 2 * c[i];
        }
        filter
    };
    let vfilter = coefficients(&coeffs[0]);
    let hfilter = coefficients(&coeffs[1]);
    let offset = 1i32 << (bit_depth + FILTER_BITS as u32 - inter_round0 - 1);
    let limit = (1i32 << (bit_depth + 1 + FILTER_BITS as u32 - inter_round0)) - 1;
    let mut intermediate = [[0i32; 4]; 10];
    for r in 0..h + 6 {
        for c in 0..w {
            let mut s = 0i32;
            for t in 0..7 {
                s += hfilter[t] * src.get((x + c + t) as isize - 3, (y + r) as isize - 3);
            }
            let v = round2(i64::from(s), inter_round0) as i32;
            intermediate[r][c] = v.clamp(-offset, limit - offset);
        }
    }
    let max = (1i64 << bit_depth) - 1;
    for r in 0..h {
        for c in 0..w {
            let mut s = 0i64;
            for t in 0..7 {
                s += i64::from(vfilter[t]) * i64::from(intermediate[r + t][c]);
            }
            let v = round2(s, inter_round1);
            out.set(x + c, y + r, v.clamp(0, max) as u16);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn self_guided_filter(
    src: &StripeSource<'_>,
    set: usize,
    xqd: [i32; 2],
    out: &mut PlaneBuf,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    bit_depth: u32,
) {
    let flt0 = box_filter(src, x, y, w, h, set, 0, bit_depth);
    let flt1 = box_filter(src, x, y, w, h, set, 1, bit_depth);
    let w0 = xqd[0];
    let w1 = xqd[1];
    let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
    let r0 = SGR_PARAMS[set][0];
    let r1 = SGR_PARAMS[set][2];
    let max = (1i64 << bit_depth) - 1;
    for i in 0..h {
        for j in 0..w {
            let u = i64::from(src.cdef.get(x + j, y + i)) << SGRPROJ_RST_BITS;
            let mut v = i64::from(w1) * u;
            v += if r0 != 0 {
                i64::from(w0) * flt0[i][j]
            } else {
                i64::from(w0) * u
            };
            v += if r1 != 0 {
                i64::from(w2) * flt1[i][j]
            } else {
                i64::from(w2) * u
            };
            let s = round2(v, (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS) as u32);
            out.set(x + j, y + i, s.clamp(0, max) as u16);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn box_filter(
    src: &StripeSource<'_>,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    set: usize,
    pass: usize,
    bit_depth: u32,
) -> [[i64; 4]; 4] {
    let mut f = [[0i64; 4]; 4];
    let r = SGR_PARAMS[set][pass * 2] as isize;
    if r == 0 {
        return f;
    }
    let eps = i64::from(SGR_PARAMS[set][pass * 2 + 1]);
    let mut a_arr = [[0i64; 6]; 6];
    let mut b_arr = [[0i64; 6]; 6];
    let n = (2 * r + 1) * (2 * r + 1);
    let n = n as i64;
    let n2e = n * n * eps;
    let s = ((1i64 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e;
    let one_over_n = ((1i64 << SGRPROJ_RECIP_BITS) + (n / 2)) / n;
    for i in -1..(h as isize + 1) {
        for j in -1..(w as isize + 1) {
            let mut a = 0i64;
            let mut b = 0i64;
            for dy in -r..=r {
                for dx in -r..=r {
                    let c = i64::from(src.get(x as isize + j + dx, y as isize + i + dy));
                    a += c * c;
                    b += c;
                }
            }
            let a = round2(a, 2 * (bit_depth - 8));
            let d = round2(b, bit_depth - 8);
            let p = (a * n - d * d).max(0);
            let z = round2(p * s, SGRPROJ_MTABLE_BITS as u32);
            let a2 = if z >= 255 {
                256
            } else if z == 0 {
                1
            } else {
                ((z << SGRPROJ_SGR_BITS) + (z / 2)) / (z + 1)
            };
            let b2 = ((1i64 << SGRPROJ_SGR_BITS) - a2) * b * one_over_n;
            a_arr[(i + 1) as usize][(j + 1) as usize] = a2;
            b_arr[(i + 1) as usize][(j + 1) as usize] = round2(b2, SGRPROJ_RECIP_BITS as u32);
        }
    }
    for i in 0..h {
        let shift = if pass == 0 && (i & 1) == 1 { 4 } else { 5 };
        for j in 0..w {
            let mut a = 0i64;
            let mut b = 0i64;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let weight = if pass == 0 {
                        if ((i as isize + dy) & 1) != 0 {
                            if dx == 0 { 6 } else { 5 }
                        } else {
                            0
                        }
                    } else if dx == 0 || dy == 0 {
                        4
                    } else {
                        3
                    };
                    a += weight
                        * a_arr[(i as isize + dy + 1) as usize][(j as isize + dx + 1) as usize];
                    b += weight
                        * b_arr[(i as isize + dy + 1) as usize][(j as isize + dx + 1) as usize];
                }
            }
            let v = a * i64::from(src.cdef.get(x + j, y + i)) + b;
            f[i][j] = round2(v, (SGRPROJ_SGR_BITS + shift - SGRPROJ_RST_BITS) as u32);
        }
    }
    f
}
