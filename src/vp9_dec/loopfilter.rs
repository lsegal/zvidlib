//! The loop filter (section 8.8 of the VP9 specification), in the form of
//! libvpx's 4:2:0 implementation: per-superblock edge masks built as each
//! block is decoded (`vp9_build_mask`), adjusted at the frame edges
//! (`vp9_adjust_mask`), then applied a superblock at a time, vertical
//! edges before horizontal ones and luma before chroma
//! (`vp9_filter_block_plane_ss00`/`_ss11`). The order matters because
//! neighbouring edges' filters overlap, so it is kept exactly.

use super::tables::UV_TXSIZE_LOOKUP;

/// The edge masks of one 64x64 superblock (`LOOP_FILTER_MASK`): one bit per
/// 8x8 luma block (row-major, low bit first) or per 8x8 chroma block.
#[derive(Clone, Copy, Debug)]
pub(super) struct LoopFilterMask {
    left_y: [u64; 4],
    above_y: [u64; 4],
    int_4x4_y: u64,
    left_uv: [u16; 4],
    above_uv: [u16; 4],
    int_4x4_uv: u16,
    lfl_y: [u8; 64],
}

impl Default for LoopFilterMask {
    fn default() -> Self {
        Self {
            left_y: [0; 4],
            above_y: [0; 4],
            int_4x4_y: 0,
            left_uv: [0; 4],
            above_uv: [0; 4],
            int_4x4_uv: 0,
            lfl_y: [0; 64],
        }
    }
}

const LEFT_64X64_TXFORM_MASK: [u64; 4] = [
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0x5555_5555_5555_5555,
    0x1111_1111_1111_1111,
];
const ABOVE_64X64_TXFORM_MASK: [u64; 4] = [
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0x00ff_00ff_00ff_00ff,
    0x0000_00ff_0000_00ff,
];
const LEFT_PREDICTION_MASK: [u64; 13] = [
    0x1,
    0x1,
    0x1,
    0x1,
    0x101,
    0x1,
    0x101,
    0x0101_0101,
    0x101,
    0x0101_0101,
    0x0101_0101_0101_0101,
    0x0101_0101,
    0x0101_0101_0101_0101,
];
const ABOVE_PREDICTION_MASK: [u64; 13] = [
    0x1, 0x1, 0x1, 0x1, 0x1, 0x3, 0x3, 0x3, 0xf, 0xf, 0xf, 0xff, 0xff,
];
const SIZE_MASK: [u64; 13] = [
    0x1,
    0x1,
    0x1,
    0x1,
    0x101,
    0x3,
    0x303,
    0x0303_0303,
    0x0f0f,
    0x0f0f_0f0f,
    0x0f0f_0f0f_0f0f_0f0f,
    0xffff_ffff,
    0xffff_ffff_ffff_ffff,
];
const LEFT_BORDER: u64 = 0x1111_1111_1111_1111;
const ABOVE_BORDER: u64 = 0x0000_00ff_0000_00ff;
const LEFT_64X64_TXFORM_MASK_UV: [u16; 4] = [0xffff, 0xffff, 0x5555, 0x1111];
const ABOVE_64X64_TXFORM_MASK_UV: [u16; 4] = [0xffff, 0xffff, 0x0f0f, 0x000f];
const LEFT_PREDICTION_MASK_UV: [u16; 13] = [
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x11, 0x1, 0x11, 0x1111, 0x11, 0x1111,
];
const ABOVE_PREDICTION_MASK_UV: [u16; 13] = [
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x3, 0x3, 0x3, 0xf, 0xf,
];
const SIZE_MASK_UV: [u16; 13] = [
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x11, 0x3, 0x33, 0x3333, 0xff, 0xffff,
];
const LEFT_BORDER_UV: u16 = 0x1111;
const ABOVE_BORDER_UV: u16 = 0x000f;

/// The facts about one decoded block the masks are built from.
pub(super) struct MaskBlock {
    pub(super) sb_type: u8,
    pub(super) tx_size: u8,
    pub(super) skip_inter: bool,
    pub(super) filter_level: u8,
}

/// `vp9_build_mask`: adds one block, of `bw`x`bh` 8x8 units at
/// (`mi_row`, `mi_col`), to its superblock's masks.
pub(super) fn build_mask(
    lfm: &mut LoopFilterMask,
    block: &MaskBlock,
    mi_row: usize,
    mi_col: usize,
    bw: usize,
    bh: usize,
) {
    let block_size = usize::from(block.sb_type);
    let tx_size_y = usize::from(block.tx_size);
    let tx_size_uv = usize::from(UV_TXSIZE_LOOKUP[block_size][tx_size_y][1][1]);
    let row_in_sb = mi_row & 7;
    let col_in_sb = mi_col & 7;
    let shift_y = col_in_sb + (row_in_sb << 3);
    let shift_uv = (col_in_sb >> 1) + ((row_in_sb >> 1) << 2);
    let build_uv = row_in_sb & 1 == 0 && col_in_sb & 1 == 0;

    if block.filter_level == 0 {
        return;
    }
    for row in 0..bh {
        let index = shift_y + row * 8;
        lfm.lfl_y[index..index + bw].fill(block.filter_level);
    }

    lfm.above_y[tx_size_y] |= ABOVE_PREDICTION_MASK[block_size] << shift_y;
    lfm.left_y[tx_size_y] |= LEFT_PREDICTION_MASK[block_size] << shift_y;
    if build_uv {
        lfm.above_uv[tx_size_uv] |= ABOVE_PREDICTION_MASK_UV[block_size] << shift_uv;
        lfm.left_uv[tx_size_uv] |= LEFT_PREDICTION_MASK_UV[block_size] << shift_uv;
    }

    // A block with no coefficients and no intra prediction filters only
    // its outer edges.
    if block.skip_inter {
        return;
    }

    lfm.above_y[tx_size_y] |=
        (SIZE_MASK[block_size] & ABOVE_64X64_TXFORM_MASK[tx_size_y]) << shift_y;
    lfm.left_y[tx_size_y] |= (SIZE_MASK[block_size] & LEFT_64X64_TXFORM_MASK[tx_size_y]) << shift_y;
    if build_uv {
        lfm.above_uv[tx_size_uv] |=
            (SIZE_MASK_UV[block_size] & ABOVE_64X64_TXFORM_MASK_UV[tx_size_uv]) << shift_uv;
        lfm.left_uv[tx_size_uv] |=
            (SIZE_MASK_UV[block_size] & LEFT_64X64_TXFORM_MASK_UV[tx_size_uv]) << shift_uv;
    }
    if tx_size_y == 0 {
        lfm.int_4x4_y |= SIZE_MASK[block_size] << shift_y;
    }
    if build_uv && tx_size_uv == 0 {
        lfm.int_4x4_uv |= SIZE_MASK_UV[block_size] << shift_uv;
    }
}

/// `vp9_adjust_mask`.
fn adjust_mask(
    lfm: &mut LoopFilterMask,
    mi_row: usize,
    mi_col: usize,
    mi_rows: usize,
    mi_cols: usize,
) {
    lfm.left_y[2] |= lfm.left_y[3];
    lfm.above_y[2] |= lfm.above_y[3];
    lfm.left_uv[2] |= lfm.left_uv[3];
    lfm.above_uv[2] |= lfm.above_uv[3];

    lfm.left_y[1] |= lfm.left_y[0] & LEFT_BORDER;
    lfm.left_y[0] &= !LEFT_BORDER;
    lfm.above_y[1] |= lfm.above_y[0] & ABOVE_BORDER;
    lfm.above_y[0] &= !ABOVE_BORDER;
    lfm.left_uv[1] |= lfm.left_uv[0] & LEFT_BORDER_UV;
    lfm.left_uv[0] &= !LEFT_BORDER_UV;
    lfm.above_uv[1] |= lfm.above_uv[0] & ABOVE_BORDER_UV;
    lfm.above_uv[0] &= !ABOVE_BORDER_UV;

    if mi_row + 8 > mi_rows {
        let rows = (mi_rows - mi_row) as u64;
        let mask_y = (1u64 << (rows << 3)) - 1;
        let mask_uv = ((1u32 << (((rows + 1) >> 1) << 2)) - 1) as u16;
        for i in 0..3 {
            lfm.left_y[i] &= mask_y;
            lfm.above_y[i] &= mask_y;
            lfm.left_uv[i] &= mask_uv;
            lfm.above_uv[i] &= mask_uv;
        }
        lfm.int_4x4_y &= mask_y;
        lfm.int_4x4_uv &= mask_uv;
        if rows == 1 {
            lfm.above_uv[1] |= lfm.above_uv[2];
            lfm.above_uv[2] = 0;
        }
        if rows == 5 {
            lfm.above_uv[1] |= lfm.above_uv[2] & 0xff00;
            lfm.above_uv[2] &= !(lfm.above_uv[2] & 0xff00);
        }
    }

    if mi_col + 8 > mi_cols {
        let columns = (mi_cols - mi_col) as u64;
        let mask_y = ((1u64 << columns) - 1) * 0x0101_0101_0101_0101;
        let mask_uv = (((1u32 << ((columns + 1) >> 1)) - 1) * 0x1111) as u16;
        let mask_uv_int = (((1u32 << (columns >> 1)) - 1) * 0x1111) as u16;
        for i in 0..3 {
            lfm.left_y[i] &= mask_y;
            lfm.above_y[i] &= mask_y;
            lfm.left_uv[i] &= mask_uv;
            lfm.above_uv[i] &= mask_uv;
        }
        lfm.int_4x4_y &= mask_y;
        lfm.int_4x4_uv &= mask_uv_int;
        if columns == 1 {
            lfm.left_uv[1] |= lfm.left_uv[2];
            lfm.left_uv[2] = 0;
        }
        if columns == 5 {
            lfm.left_uv[1] |= lfm.left_uv[2] & 0xcccc;
            lfm.left_uv[2] &= !(lfm.left_uv[2] & 0xcccc);
        }
    }

    if mi_col == 0 {
        for i in 0..3 {
            lfm.left_y[i] &= 0xfefe_fefe_fefe_fefe;
            lfm.left_uv[i] &= 0xeeee;
        }
    }
}

/// The thresholds of one filter level (`loop_filter_thresh`).
#[derive(Clone, Copy, Debug, Default)]
struct Thresholds {
    mblim: u8,
    lim: u8,
    hev_thr: u8,
}

/// `update_sharpness` and the `hev_thr` setup of `vp9_loop_filter_init`.
fn thresholds(sharpness: u8) -> [Thresholds; 64] {
    let mut table = [Thresholds::default(); 64];
    for (level, entry) in table.iter_mut().enumerate() {
        let level = level as i32;
        let sharpness = i32::from(sharpness);
        let mut inside = level >> ((sharpness > 0) as i32 + (sharpness > 4) as i32);
        if sharpness > 0 && inside > 9 - sharpness {
            inside = 9 - sharpness;
        }
        inside = inside.max(1);
        *entry = Thresholds {
            mblim: (2 * (level + 2) + inside) as u8,
            lim: inside as u8,
            hev_thr: (level >> 4) as u8,
        };
    }
    table
}

/// A view of one plane for the filters: `data[offset]` is the pixel the
/// current edge starts at.
struct Pixels<'a> {
    data: &'a mut [u8],
    stride: usize,
}

#[inline]
fn signed_char_clamp(value: i32) -> i32 {
    value.clamp(-128, 127)
}

#[inline]
fn filter_mask(limit: u8, blimit: u8, p: [u8; 4], q: [u8; 4]) -> bool {
    let limit = i32::from(limit);
    let d = |a: u8, b: u8| (i32::from(a) - i32::from(b)).abs();
    !(d(p[3], p[2]) > limit
        || d(p[2], p[1]) > limit
        || d(p[1], p[0]) > limit
        || d(q[1], q[0]) > limit
        || d(q[2], q[1]) > limit
        || d(q[3], q[2]) > limit
        || d(p[0], q[0]) * 2 + d(p[1], q[1]) / 2 > i32::from(blimit))
}

#[inline]
fn flat_mask4(p: [u8; 4], q: [u8; 4]) -> bool {
    let d = |a: u8, b: u8| (i32::from(a) - i32::from(b)).abs();
    !(d(p[1], p[0]) > 1
        || d(q[1], q[0]) > 1
        || d(p[2], p[0]) > 1
        || d(q[2], q[0]) > 1
        || d(p[3], p[0]) > 1
        || d(q[3], q[0]) > 1)
}

impl Pixels<'_> {
    #[inline]
    fn get(&self, index: isize) -> u8 {
        self.data[index as usize]
    }

    #[inline]
    fn set(&mut self, index: isize, value: u8) {
        self.data[index as usize] = value;
    }

    /// `filter4`, on the four pixels at `base + k * step` for k in -2..2.
    #[inline]
    fn filter4(&mut self, base: isize, step: isize, mask: bool, thresh: u8) {
        let p1 = self.get(base - 2 * step);
        let p0 = self.get(base - step);
        let q0 = self.get(base);
        let q1 = self.get(base + step);
        let ps1 = i32::from(p1 as i8 ^ -128);
        let ps0 = i32::from(p0 as i8 ^ -128);
        let qs0 = i32::from(q0 as i8 ^ -128);
        let qs1 = i32::from(q1 as i8 ^ -128);
        let d = |a: u8, b: u8| (i32::from(a) - i32::from(b)).abs();
        let hev = d(p1, p0) > i32::from(thresh) || d(q1, q0) > i32::from(thresh);
        let mut filter = if hev { signed_char_clamp(ps1 - qs1) } else { 0 };
        filter = if mask {
            signed_char_clamp(filter + 3 * (qs0 - ps0))
        } else {
            0
        };
        let filter1 = signed_char_clamp(filter + 4) >> 3;
        let filter2 = signed_char_clamp(filter + 3) >> 3;
        let to_pixel = |value: i32| (signed_char_clamp(value) as i8 ^ -128) as u8;
        self.set(base, to_pixel(qs0 - filter1));
        self.set(base - step, to_pixel(ps0 + filter2));
        let outer = if hev { 0 } else { (filter1 + 1) >> 1 };
        self.set(base + step, to_pixel(qs1 - outer));
        self.set(base - 2 * step, to_pixel(ps1 + outer));
    }

    fn taps4(&self, base: isize, step: isize) -> ([u8; 4], [u8; 4]) {
        let p = [
            self.get(base - step),
            self.get(base - 2 * step),
            self.get(base - 3 * step),
            self.get(base - 4 * step),
        ];
        let q = [
            self.get(base),
            self.get(base + step),
            self.get(base + 2 * step),
            self.get(base + 3 * step),
        ];
        (p, q)
    }

    /// The 4-tap filter along `count` pixels of an edge. `step` crosses
    /// the edge and `along` moves along it.
    fn lpf4(&mut self, start: isize, step: isize, along: isize, count: usize, t: Thresholds) {
        for i in 0..count as isize {
            let base = start + i * along;
            let (p, q) = self.taps4(base, step);
            let mask = filter_mask(t.lim, t.mblim, p, q);
            self.filter4(base, step, mask, t.hev_thr);
        }
    }

    /// `filter8` along `count` pixels of an edge.
    fn lpf8(&mut self, start: isize, step: isize, along: isize, count: usize, t: Thresholds) {
        for i in 0..count as isize {
            let base = start + i * along;
            let (p, q) = self.taps4(base, step);
            let mask = filter_mask(t.lim, t.mblim, p, q);
            if mask && flat_mask4(p, q) {
                self.filter7(base, step, p, q);
            } else {
                self.filter4(base, step, mask, t.hev_thr);
            }
        }
    }

    fn filter7(&mut self, base: isize, step: isize, p: [u8; 4], q: [u8; 4]) {
        let [p0, p1, p2, p3] = p.map(u32::from);
        let [q0, q1, q2, q3] = q.map(u32::from);
        let r = |sum: u32| ((sum + 4) >> 3) as u8;
        self.set(base - 3 * step, r(p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0));
        self.set(base - 2 * step, r(p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1));
        self.set(base - step, r(p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2));
        self.set(base, r(p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3));
        self.set(base + step, r(p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3));
        self.set(base + 2 * step, r(p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3));
    }

    /// `filter16` along `count` pixels of an edge.
    fn lpf16(&mut self, start: isize, step: isize, along: isize, count: usize, t: Thresholds) {
        for i in 0..count as isize {
            let base = start + i * along;
            let (p, q) = self.taps4(base, step);
            let mask = filter_mask(t.lim, t.mblim, p, q);
            let flat = flat_mask4(p, q);
            if !(mask && flat) {
                self.filter4(base, step, mask, t.hev_thr);
                continue;
            }
            // p7..p0 then q0..q7.
            let mut wide = [0u32; 16];
            for k in 0..8isize {
                wide[(7 - k) as usize] = u32::from(self.get(base - (k + 1) * step));
                wide[(8 + k) as usize] = u32::from(self.get(base + k * step));
            }
            let d = |a: u32, b: u32| (a as i32 - b as i32).abs();
            // flat_mask5: p4..p7 against p0 and q4..q7 against q0.
            let flat2 =
                (4..8).all(|k| d(wide[7 - k], wide[7]) <= 1 && d(wide[8 + k], wide[8]) <= 1);
            if !flat2 {
                self.filter7(base, step, p, q);
                continue;
            }
            // The 15-tap filter [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            // with taps past either end repeating p7 or q7.
            for out in 1..15isize {
                let mut sum = wide[out as usize];
                for k in -7..=7isize {
                    sum += wide[(out + k).clamp(0, 15) as usize];
                }
                self.set(base + (out - 8) * step, ((sum + 8) >> 4) as u8);
            }
        }
    }
}

/// `filter_selectively_vert_row2`: the vertical edges of two rows of 8x8
/// blocks (`lfl_forward` apart in the level array).
#[allow(clippy::too_many_arguments)]
fn filter_selectively_vert_row2(
    pixels: &mut Pixels,
    start: isize,
    subsampled: bool,
    mut mask_16x16: u32,
    mut mask_8x8: u32,
    mut mask_4x4: u32,
    mut mask_4x4_int: u32,
    thresholds: &[Thresholds; 64],
    lfl: &[u8],
) {
    let dual_mask_cutoff = if subsampled { 0xff } else { 0xffff };
    let lfl_forward = if subsampled { 4 } else { 8 };
    let dual_one = 1 | (1 << lfl_forward);
    let pitch = pixels.stride as isize;
    let mut s0 = start;
    let mut lfl_index = 0;
    let mut mask = (mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int) & dual_mask_cutoff;
    while mask != 0 {
        if mask & dual_one != 0 {
            let lfis = [
                thresholds[usize::from(lfl[lfl_index])],
                thresholds[usize::from(lfl[lfl_index + lfl_forward])],
            ];
            let s = [s0, s0 + 8 * pitch];
            if mask_16x16 & dual_one != 0 {
                if mask_16x16 & dual_one == dual_one {
                    pixels.lpf16(s[0], 1, pitch, 16, lfis[0]);
                } else {
                    let which = usize::from(mask_16x16 & 1 == 0);
                    pixels.lpf16(s[which], 1, pitch, 8, lfis[which]);
                }
            }
            if mask_8x8 & dual_one != 0 {
                if mask_8x8 & dual_one == dual_one {
                    pixels.lpf8(s[0], 1, pitch, 8, lfis[0]);
                    pixels.lpf8(s[1], 1, pitch, 8, lfis[1]);
                } else {
                    let which = usize::from(mask_8x8 & 1 == 0);
                    pixels.lpf8(s[which], 1, pitch, 8, lfis[which]);
                }
            }
            if mask_4x4 & dual_one != 0 {
                if mask_4x4 & dual_one == dual_one {
                    pixels.lpf4(s[0], 1, pitch, 8, lfis[0]);
                    pixels.lpf4(s[1], 1, pitch, 8, lfis[1]);
                } else {
                    let which = usize::from(mask_4x4 & 1 == 0);
                    pixels.lpf4(s[which], 1, pitch, 8, lfis[which]);
                }
            }
            if mask_4x4_int & dual_one != 0 {
                if mask_4x4_int & dual_one == dual_one {
                    pixels.lpf4(s[0] + 4, 1, pitch, 8, lfis[0]);
                    pixels.lpf4(s[1] + 4, 1, pitch, 8, lfis[1]);
                } else {
                    let which = usize::from(mask_4x4_int & 1 == 0);
                    pixels.lpf4(s[which] + 4, 1, pitch, 8, lfis[which]);
                }
            }
        }
        s0 += 8;
        lfl_index += 1;
        mask_16x16 >>= 1;
        mask_8x8 >>= 1;
        mask_4x4 >>= 1;
        mask_4x4_int >>= 1;
        mask = (mask & !dual_one) >> 1;
    }
}

/// `filter_selectively_horiz`: the horizontal edges of one row of 8x8
/// blocks.
#[allow(clippy::too_many_arguments)]
fn filter_selectively_horiz(
    pixels: &mut Pixels,
    start: isize,
    mut mask_16x16: u32,
    mut mask_8x8: u32,
    mut mask_4x4: u32,
    mut mask_4x4_int: u32,
    thresholds: &[Thresholds; 64],
    lfl: &[u8],
) {
    let pitch = pixels.stride as isize;
    let mut s = start;
    let mut lfl_index = 0;
    let mut mask = mask_16x16 | mask_8x8 | mask_4x4 | mask_4x4_int;
    while mask != 0 {
        let mut count = 1;
        if mask & 1 != 0 {
            let lfi = thresholds[usize::from(lfl[lfl_index])];
            if mask_16x16 & 1 != 0 {
                if mask_16x16 & 3 == 3 {
                    pixels.lpf16(s, pitch, 1, 16, lfi);
                    count = 2;
                } else {
                    pixels.lpf16(s, pitch, 1, 8, lfi);
                }
            } else if mask_8x8 & 1 != 0 || mask_4x4 & 1 != 0 {
                let eight = mask_8x8 & 1 != 0;
                let pair = if eight {
                    mask_8x8 & 3 == 3
                } else {
                    mask_4x4 & 3 == 3
                };
                if pair {
                    let lfin = thresholds[usize::from(lfl[lfl_index + 1])];
                    if eight {
                        pixels.lpf8(s, pitch, 1, 8, lfi);
                        pixels.lpf8(s + 8, pitch, 1, 8, lfin);
                    } else {
                        pixels.lpf4(s, pitch, 1, 8, lfi);
                        pixels.lpf4(s + 8, pitch, 1, 8, lfin);
                    }
                    if mask_4x4_int & 3 == 3 {
                        pixels.lpf4(s + 4 * pitch, pitch, 1, 8, lfi);
                        pixels.lpf4(s + 8 + 4 * pitch, pitch, 1, 8, lfin);
                    } else if mask_4x4_int & 1 != 0 {
                        pixels.lpf4(s + 4 * pitch, pitch, 1, 8, lfi);
                    } else if mask_4x4_int & 2 != 0 {
                        pixels.lpf4(s + 8 + 4 * pitch, pitch, 1, 8, lfin);
                    }
                    count = 2;
                } else {
                    if eight {
                        pixels.lpf8(s, pitch, 1, 8, lfi);
                    } else {
                        pixels.lpf4(s, pitch, 1, 8, lfi);
                    }
                    if mask_4x4_int & 1 != 0 {
                        pixels.lpf4(s + 4 * pitch, pitch, 1, 8, lfi);
                    }
                }
            } else {
                pixels.lpf4(s + 4 * pitch, pitch, 1, 8, lfi);
            }
        }
        s += 8 * count;
        lfl_index += count as usize;
        mask_16x16 >>= count;
        mask_8x8 >>= count;
        mask_4x4 >>= count;
        mask_4x4_int >>= count;
        mask >>= count;
    }
}

/// One plane of the frame being filtered.
pub(super) struct FilterPlane<'a> {
    pub(super) data: &'a mut [u8],
    pub(super) stride: usize,
    /// The index of the plane's top-left pixel in `data`.
    pub(super) origin: usize,
}

/// Filters the whole frame (`loop_filter_rows` over every superblock row
/// with the 4:2:0 path).
pub(super) fn filter_frame(
    planes: &mut [FilterPlane; 3],
    masks: &mut [LoopFilterMask],
    mi_rows: usize,
    mi_cols: usize,
    sharpness: u8,
) {
    let thresholds = thresholds(sharpness);
    let sb_cols = mi_cols.div_ceil(8);
    for mi_row in (0..mi_rows).step_by(8) {
        for mi_col in (0..mi_cols).step_by(8) {
            let lfm = &mut masks[(mi_row / 8) * sb_cols + mi_col / 8];
            adjust_mask(lfm, mi_row, mi_col, mi_rows, mi_cols);
            let lfm = *lfm;
            for (index, plane) in planes.iter_mut().enumerate() {
                let subsampled = index > 0;
                let (x, y) = if subsampled {
                    (mi_col * 4, mi_row * 4)
                } else {
                    (mi_col * 8, mi_row * 8)
                };
                let start = (plane.origin + y * plane.stride + x) as isize;
                let mut pixels = Pixels {
                    data: plane.data,
                    stride: plane.stride,
                };
                if subsampled {
                    filter_plane_ss11(&mut pixels, start, &lfm, mi_row, mi_rows, &thresholds);
                } else {
                    filter_plane_ss00(&mut pixels, start, &lfm, mi_row, mi_rows, &thresholds);
                }
            }
        }
    }
}

/// `vp9_filter_block_plane_ss00`.
fn filter_plane_ss00(
    pixels: &mut Pixels,
    start: isize,
    lfm: &LoopFilterMask,
    mi_row: usize,
    mi_rows: usize,
    thresholds: &[Thresholds; 64],
) {
    let pitch = pixels.stride as isize;
    let mut mask_16x16 = lfm.left_y[2];
    let mut mask_8x8 = lfm.left_y[1];
    let mut mask_4x4 = lfm.left_y[0];
    let mut mask_4x4_int = lfm.int_4x4_y;
    let mut row_start = start;
    let mut r = 0;
    while r < 8 && mi_row + r < mi_rows {
        filter_selectively_vert_row2(
            pixels,
            row_start,
            false,
            mask_16x16 as u32,
            mask_8x8 as u32,
            mask_4x4 as u32,
            mask_4x4_int as u32,
            thresholds,
            &lfm.lfl_y[r << 3..],
        );
        row_start += 16 * pitch;
        mask_16x16 >>= 16;
        mask_8x8 >>= 16;
        mask_4x4 >>= 16;
        mask_4x4_int >>= 16;
        r += 2;
    }

    let mut mask_16x16 = lfm.above_y[2];
    let mut mask_8x8 = lfm.above_y[1];
    let mut mask_4x4 = lfm.above_y[0];
    let mut mask_4x4_int = lfm.int_4x4_y;
    let mut row_start = start;
    let mut r = 0;
    while r < 8 && mi_row + r < mi_rows {
        let (m16, m8, m4) = if mi_row + r == 0 {
            (0, 0, 0)
        } else {
            (mask_16x16 & 0xff, mask_8x8 & 0xff, mask_4x4 & 0xff)
        };
        filter_selectively_horiz(
            pixels,
            row_start,
            m16 as u32,
            m8 as u32,
            m4 as u32,
            (mask_4x4_int & 0xff) as u32,
            thresholds,
            &lfm.lfl_y[r << 3..],
        );
        row_start += 8 * pitch;
        mask_16x16 >>= 8;
        mask_8x8 >>= 8;
        mask_4x4 >>= 8;
        mask_4x4_int >>= 8;
        r += 1;
    }
}

/// `vp9_filter_block_plane_ss11`.
fn filter_plane_ss11(
    pixels: &mut Pixels,
    start: isize,
    lfm: &LoopFilterMask,
    mi_row: usize,
    mi_rows: usize,
    thresholds: &[Thresholds; 64],
) {
    let pitch = pixels.stride as isize;
    let mut lfl_uv = [0u8; 16];
    let mut mask_16x16 = lfm.left_uv[2];
    let mut mask_8x8 = lfm.left_uv[1];
    let mut mask_4x4 = lfm.left_uv[0];
    let mut mask_4x4_int = lfm.int_4x4_uv;
    let mut row_start = start;
    let mut r = 0;
    while r < 8 && mi_row + r < mi_rows {
        for c in 0..4 {
            lfl_uv[(r << 1) + c] = lfm.lfl_y[(r << 3) + (c << 1)];
            lfl_uv[((r + 2) << 1) + c] = lfm.lfl_y[((r + 2) << 3) + (c << 1)];
        }
        filter_selectively_vert_row2(
            pixels,
            row_start,
            true,
            u32::from(mask_16x16),
            u32::from(mask_8x8),
            u32::from(mask_4x4),
            u32::from(mask_4x4_int),
            thresholds,
            &lfl_uv[r << 1..],
        );
        row_start += 16 * pitch;
        mask_16x16 >>= 8;
        mask_8x8 >>= 8;
        mask_4x4 >>= 8;
        mask_4x4_int >>= 8;
        r += 4;
    }

    let mut mask_16x16 = lfm.above_uv[2];
    let mut mask_8x8 = lfm.above_uv[1];
    let mut mask_4x4 = lfm.above_uv[0];
    let mut mask_4x4_int = lfm.int_4x4_uv;
    let mut row_start = start;
    let mut r = 0;
    while r < 8 && mi_row + r < mi_rows {
        let skip_border_4x4_r = mi_row + r == mi_rows - 1;
        let mask_4x4_int_r = if skip_border_4x4_r {
            0
        } else {
            mask_4x4_int & 0xf
        };
        let (m16, m8, m4) = if mi_row + r == 0 {
            (0, 0, 0)
        } else {
            (mask_16x16 & 0xf, mask_8x8 & 0xf, mask_4x4 & 0xf)
        };
        filter_selectively_horiz(
            pixels,
            row_start,
            u32::from(m16),
            u32::from(m8),
            u32::from(m4),
            u32::from(mask_4x4_int_r),
            thresholds,
            &lfl_uv[r << 1..],
        );
        row_start += 8 * pitch;
        mask_16x16 >>= 4;
        mask_8x8 >>= 4;
        mask_4x4 >>= 4;
        mask_4x4_int >>= 4;
        r += 2;
    }
}

/// The filter level of every segment, reference frame and mode class
/// (`vp9_loop_filter_frame_init`'s `lfi->lvl`), indexed
/// `[segment][reference frame][mode is not ZEROMV]`.
pub(super) fn filter_levels(
    default_level: u8,
    segment_levels: [Option<(bool, i32)>; 8],
    mode_ref_delta_enabled: bool,
    ref_deltas: [i8; 4],
    mode_deltas: [i8; 2],
) -> [[[u8; 2]; 4]; 8] {
    let scale = 1 << (default_level >> 5);
    let mut levels = [[[0u8; 2]; 4]; 8];
    for (segment, entry) in levels.iter_mut().enumerate() {
        let mut level = i32::from(default_level);
        if let Some((absolute, data)) = segment_levels[segment] {
            level = if absolute { data } else { level + data }.clamp(0, 63);
        }
        if !mode_ref_delta_enabled {
            *entry = [[level as u8; 2]; 4];
            continue;
        }
        let intra = level + i32::from(ref_deltas[0]) * scale;
        entry[0][0] = intra.clamp(0, 63) as u8;
        for reference in 1..4 {
            for mode in 0..2 {
                let inter = level
                    + i32::from(ref_deltas[reference]) * scale
                    + i32::from(mode_deltas[mode]) * scale;
                entry[reference][mode] = inter.clamp(0, 63) as u8;
            }
        }
    }
    levels
}
