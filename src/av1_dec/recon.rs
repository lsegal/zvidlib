//! Dequantization, the inverse transforms and reconstruction (specification
//! sections 7.12 and 7.13).

use super::consts::*;
use super::tables::*;
use super::tile::TileDecoder;

#[inline(always)]
fn round2(x: i64, n: u32) -> i64 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

fn brev(num_bits: u32, x: usize) -> usize {
    let mut t = 0;
    for i in 0..num_bits {
        let bit = (x >> i) & 1;
        t += bit << (num_bits - 1 - i);
    }
    t
}

#[inline(always)]
fn cos128(angle: i32) -> i64 {
    let angle2 = angle & 255;
    let v = if angle2 <= 64 {
        COS128_LOOKUP[angle2 as usize] as i64
    } else if angle2 <= 128 {
        -(COS128_LOOKUP[(128 - angle2) as usize] as i64)
    } else if angle2 <= 192 {
        -(COS128_LOOKUP[(angle2 - 128) as usize] as i64)
    } else {
        COS128_LOOKUP[(256 - angle2) as usize] as i64
    };
    v
}

#[inline(always)]
fn sin128(angle: i32) -> i64 {
    cos128(angle - 64)
}

/// The 1D transform working array `T` with the butterfly helpers.
struct Tx<'a> {
    t: &'a mut [i64],
}

impl Tx<'_> {
    /// `B( a, b, angle, flip, r )`.
    #[inline(always)]
    fn b(&mut self, a: usize, b: usize, angle: i32, flip: bool) {
        let x = self.t[a] * cos128(angle) - self.t[b] * sin128(angle);
        let y = self.t[a] * sin128(angle) + self.t[b] * cos128(angle);
        self.t[a] = round2(x, 12);
        self.t[b] = round2(y, 12);
        if flip {
            self.t.swap(a, b);
        }
    }

    /// `H( a, b, flip, r )`.
    #[inline(always)]
    fn h(&mut self, a: usize, b: usize, flip: bool, r: u32) {
        let (a, b) = if flip { (b, a) } else { (a, b) };
        let x = self.t[a];
        let y = self.t[b];
        let lo = -(1i64 << (r - 1));
        let hi = (1i64 << (r - 1)) - 1;
        self.t[a] = (x + y).clamp(lo, hi);
        self.t[b] = (x - y).clamp(lo, hi);
    }

    fn dct_permutation(&mut self, n: u32) {
        let len = 1usize << n;
        let mut copy = [0i64; 64];
        copy[..len].copy_from_slice(&self.t[..len]);
        for i in 0..len {
            self.t[i] = copy[brev(n, i)];
        }
    }

    /// The inverse DCT process (section 7.13.2.3).
    fn inverse_dct(&mut self, n: u32, r: u32) {
        self.dct_permutation(n);
        if n == 6 {
            for i in 0..16 {
                self.b(32 + i, 63 - i, 63 - 4 * brev(4, i) as i32, false);
            }
        }
        if n >= 5 {
            for i in 0..8 {
                self.b(16 + i, 31 - i, 6 + ((brev(3, 7 - i) as i32) << 3), false);
            }
        }
        if n == 6 {
            for i in 0..16 {
                self.h(32 + i * 2, 33 + i * 2, i & 1 != 0, r);
            }
        }
        if n >= 4 {
            for i in 0..4 {
                self.b(8 + i, 15 - i, 12 + ((brev(2, 3 - i) as i32) << 4), false);
            }
        }
        if n >= 5 {
            for i in 0..8 {
                self.h(16 + 2 * i, 17 + 2 * i, i & 1 != 0, r);
            }
        }
        if n == 6 {
            for i in 0..4 {
                for j in 0..2 {
                    self.b(
                        62 - i * 4 - j,
                        33 + i * 4 + j,
                        60 - 16 * brev(2, i) as i32 + 64 * j as i32,
                        true,
                    );
                }
            }
        }
        if n >= 3 {
            for i in 0..2 {
                self.b(4 + i, 7 - i, 56 - 32 * i as i32, false);
            }
        }
        if n >= 4 {
            for i in 0..4 {
                self.h(8 + 2 * i, 9 + 2 * i, i & 1 != 0, r);
            }
        }
        if n >= 5 {
            for i in 0..2 {
                for j in 0..2 {
                    self.b(
                        30 - 4 * i - j,
                        17 + 4 * i + j,
                        24 + ((j as i32) << 6) + ((1 - i as i32) << 5),
                        true,
                    );
                }
            }
        }
        if n == 6 {
            for i in 0..8 {
                for j in 0..2 {
                    self.h(32 + i * 4 + j, 35 + i * 4 - j, i & 1 != 0, r);
                }
            }
        }
        for i in 0..2 {
            self.b(2 * i, 2 * i + 1, 32 + 16 * i as i32, i == 0);
        }
        if n >= 3 {
            for i in 0..2 {
                self.h(4 + 2 * i, 5 + 2 * i, i != 0, r);
            }
        }
        if n >= 4 {
            for i in 0..2 {
                self.b(14 - i, 9 + i, 48 + 64 * i as i32, true);
            }
        }
        if n >= 5 {
            for i in 0..4 {
                for j in 0..2 {
                    self.h(16 + 4 * i + j, 19 + 4 * i - j, i & 1 != 0, r);
                }
            }
        }
        if n == 6 {
            for i in 0..2 {
                for j in 0..4 {
                    self.b(
                        61 - i * 8 - j,
                        34 + i * 8 + j,
                        56 - i as i32 * 32 + (j as i32 >> 1) * 64,
                        true,
                    );
                }
            }
        }
        for i in 0..2 {
            self.h(i, 3 - i, false, r);
        }
        if n >= 3 {
            self.b(6, 5, 32, true);
        }
        if n >= 4 {
            for i in 0..2 {
                for j in 0..2 {
                    self.h(8 + 4 * i + j, 11 + 4 * i - j, i != 0, r);
                }
            }
        }
        if n >= 5 {
            for i in 0..4 {
                self.b(29 - i, 18 + i, 48 + (i as i32 >> 1) * 64, true);
            }
        }
        if n == 6 {
            for i in 0..4 {
                for j in 0..4 {
                    self.h(32 + 8 * i + j, 39 + 8 * i - j, i & 1 != 0, r);
                }
            }
        }
        if n >= 3 {
            for i in 0..4 {
                self.h(i, 7 - i, false, r);
            }
        }
        if n >= 4 {
            for i in 0..2 {
                self.b(13 - i, 10 + i, 32, true);
            }
        }
        if n >= 5 {
            for i in 0..2 {
                for j in 0..4 {
                    self.h(16 + i * 8 + j, 23 + i * 8 - j, i != 0, r);
                }
            }
        }
        if n == 6 {
            for i in 0..8 {
                self.b(59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true);
            }
        }
        if n >= 4 {
            for i in 0..8 {
                self.h(i, 15 - i, false, r);
            }
        }
        if n >= 5 {
            for i in 0..4 {
                self.b(27 - i, 20 + i, 32, true);
            }
        }
        if n == 6 {
            for i in 0..8 {
                self.h(32 + i, 47 - i, false, r);
                self.h(48 + i, 63 - i, true, r);
            }
        }
        if n >= 5 {
            for i in 0..16 {
                self.h(i, 31 - i, false, r);
            }
        }
        if n == 6 {
            for i in 0..8 {
                self.b(55 - i, 40 + i, 32, true);
            }
        }
        if n == 6 {
            for i in 0..32 {
                self.h(i, 63 - i, false, r);
            }
        }
    }

    fn adst_input_permutation(&mut self, n: u32) {
        let n0 = 1usize << n;
        let mut copy = [0i64; 16];
        copy[..n0].copy_from_slice(&self.t[..n0]);
        for i in 0..n0 {
            let idx = if i & 1 != 0 { i - 1 } else { n0 - i - 1 };
            self.t[i] = copy[idx];
        }
    }

    fn adst_output_permutation(&mut self, n: u32) {
        let n0 = 1usize << n;
        let mut copy = [0i64; 16];
        copy[..n0].copy_from_slice(&self.t[..n0]);
        for i in 0..n0 {
            let a = (i >> 3) & 1;
            let b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
            let c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
            let d = (i & 1) ^ ((i >> 1) & 1);
            let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
            self.t[i] = if i & 1 != 0 { -copy[idx] } else { copy[idx] };
        }
    }

    fn inverse_adst4(&mut self) {
        const SINPI_1_9: i64 = 1321;
        const SINPI_2_9: i64 = 2482;
        const SINPI_3_9: i64 = 3344;
        const SINPI_4_9: i64 = 3803;
        let t = &mut self.t;
        let mut s = [0i64; 7];
        s[0] = SINPI_1_9 * t[0];
        s[1] = SINPI_2_9 * t[0];
        s[2] = SINPI_3_9 * t[1];
        s[3] = SINPI_4_9 * t[2];
        s[4] = SINPI_1_9 * t[2];
        s[5] = SINPI_2_9 * t[3];
        s[6] = SINPI_4_9 * t[3];
        let a7 = t[0] - t[2];
        let b7 = a7 + t[3];
        s[0] += s[3];
        s[1] -= s[4];
        s[3] = s[2];
        s[2] = SINPI_3_9 * b7;
        s[0] += s[5];
        s[1] -= s[6];
        let x0 = s[0] + s[3];
        let x1 = s[1] + s[3];
        let x2 = s[2];
        let x3 = s[0] + s[1] - s[3];
        t[0] = round2(x0, 12);
        t[1] = round2(x1, 12);
        t[2] = round2(x2, 12);
        t[3] = round2(x3, 12);
    }

    fn inverse_adst8(&mut self, r: u32) {
        self.adst_input_permutation(3);
        for i in 0..4 {
            self.b(2 * i, 2 * i + 1, 60 - 16 * i as i32, true);
        }
        for i in 0..4 {
            self.h(i, 4 + i, false, r);
        }
        for i in 0..2 {
            self.b(4 + 3 * i, 5 + i, 48 - 32 * i as i32, true);
        }
        for i in 0..2 {
            for j in 0..2 {
                self.h(4 * j + i, 2 + 4 * j + i, false, r);
            }
        }
        for i in 0..2 {
            self.b(2 + 4 * i, 3 + 4 * i, 32, true);
        }
        self.adst_output_permutation(3);
    }

    fn inverse_adst16(&mut self, r: u32) {
        self.adst_input_permutation(4);
        for i in 0..8 {
            self.b(2 * i, 2 * i + 1, 62 - 8 * i as i32, true);
        }
        for i in 0..8 {
            self.h(i, 8 + i, false, r);
        }
        for i in 0..2 {
            self.b(8 + 2 * i, 9 + 2 * i, 56 - 32 * i as i32, true);
            self.b(13 + 2 * i, 12 + 2 * i, 8 + 32 * i as i32, true);
        }
        for i in 0..4 {
            for j in 0..2 {
                self.h(8 * j + i, 4 + 8 * j + i, false, r);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                self.b(4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i as i32, true);
            }
        }
        for i in 0..2 {
            for j in 0..4 {
                self.h(4 * j + i, 2 + 4 * j + i, false, r);
            }
        }
        for i in 0..4 {
            self.b(2 + 4 * i, 3 + 4 * i, 32, true);
        }
        self.adst_output_permutation(4);
    }

    fn inverse_adst(&mut self, n: u32, r: u32) {
        match n {
            2 => self.inverse_adst4(),
            3 => self.inverse_adst8(r),
            _ => self.inverse_adst16(r),
        }
    }

    fn inverse_identity(&mut self, n: u32) {
        let len = 1usize << n;
        for v in self.t[..len].iter_mut() {
            *v = match n {
                2 => round2(*v * 5793, 12),
                3 => *v * 2,
                4 => round2(*v * 11586, 12),
                _ => *v * 4,
            };
        }
    }

    fn inverse_wht(&mut self, shift: u32) {
        let t = &mut self.t;
        let mut a = t[0] >> shift;
        let mut c = t[1] >> shift;
        let mut d = t[2] >> shift;
        let mut b = t[3] >> shift;
        a += c;
        d -= b;
        let e = (a - d) >> 1;
        b = e - b;
        c = e - c;
        a -= b;
        d += c;
        t[0] = a;
        t[1] = b;
        t[2] = c;
        t[3] = d;
    }
}

fn row_uses_dct(tx_type: u8) -> bool {
    matches!(tx_type, DCT_DCT | ADST_DCT | FLIPADST_DCT | H_DCT)
}

fn row_uses_adst(tx_type: u8) -> bool {
    matches!(
        tx_type,
        DCT_ADST
            | ADST_ADST
            | DCT_FLIPADST
            | FLIPADST_FLIPADST
            | ADST_FLIPADST
            | FLIPADST_ADST
            | H_ADST
            | H_FLIPADST
    )
}

fn col_uses_dct(tx_type: u8) -> bool {
    matches!(tx_type, DCT_DCT | DCT_ADST | DCT_FLIPADST | V_DCT)
}

fn col_uses_adst(tx_type: u8) -> bool {
    matches!(
        tx_type,
        ADST_DCT
            | ADST_ADST
            | FLIPADST_DCT
            | FLIPADST_FLIPADST
            | ADST_FLIPADST
            | FLIPADST_ADST
            | V_ADST
            | V_FLIPADST
    )
}

/// Scratch buffers for one transform block.
pub(crate) struct ReconScratch {
    dequant: Vec<i64>,
    residual: Vec<i64>,
}

impl ReconScratch {
    pub(crate) fn new() -> Self {
        Self {
            dequant: vec![0; 64 * 64],
            residual: vec![0; 64 * 64],
        }
    }
}

/// The 2D inverse transform (section 7.13.3) of `dequant` into `residual`.
fn inverse_transform_2d(
    dequant: &[i64],
    residual: &mut [i64],
    tx_sz: usize,
    tx_type: u8,
    lossless: bool,
    bit_depth: u32,
) {
    let log2w = TX_WIDTH_LOG2[tx_sz] as u32;
    let log2h = TX_HEIGHT_LOG2[tx_sz] as u32;
    let w = 1usize << log2w;
    let h = 1usize << log2h;
    let row_shift = if lossless {
        0
    } else {
        TRANSFORM_ROW_SHIFT[tx_sz] as u32
    };
    let col_shift = if lossless { 0 } else { 4 };
    let row_clamp_range = bit_depth + 8;
    let col_clamp_range = (bit_depth + 6).max(16);
    let mut t = [0i64; 64];
    for i in 0..h {
        for j in 0..w {
            t[j] = if i < 32 && j < 32 {
                dequant[i * 64 + j]
            } else {
                0
            };
        }
        if log2w.abs_diff(log2h) == 1 {
            for v in t[..w].iter_mut() {
                *v = round2(*v * 2896, 12);
            }
        }
        let mut tx = Tx { t: &mut t };
        if lossless {
            tx.inverse_wht(2);
        } else if row_uses_dct(tx_type) {
            tx.inverse_dct(log2w, row_clamp_range);
        } else if row_uses_adst(tx_type) {
            tx.inverse_adst(log2w, row_clamp_range);
        } else {
            tx.inverse_identity(log2w);
        }
        let lo = -(1i64 << (col_clamp_range - 1));
        let hi = (1i64 << (col_clamp_range - 1)) - 1;
        for j in 0..w {
            residual[i * 64 + j] = round2(t[j], row_shift).clamp(lo, hi);
        }
    }
    for j in 0..w {
        for i in 0..h {
            t[i] = residual[i * 64 + j];
        }
        let mut tx = Tx { t: &mut t };
        if lossless {
            tx.inverse_wht(0);
        } else if col_uses_dct(tx_type) {
            tx.inverse_dct(log2h, col_clamp_range);
        } else if col_uses_adst(tx_type) {
            tx.inverse_adst(log2h, col_clamp_range);
        } else {
            tx.inverse_identity(log2h);
        }
        for i in 0..h {
            residual[i * 64 + j] = round2(t[i], col_shift);
        }
    }
}

impl TileDecoder<'_> {
    fn dc_q(&self, b: i32) -> i64 {
        DC_QLOOKUP[((self.seq.bit_depth as usize) - 8) >> 1][b.clamp(0, 255) as usize] as i64
    }

    fn ac_q(&self, b: i32) -> i64 {
        AC_QLOOKUP[((self.seq.bit_depth as usize) - 8) >> 1][b.clamp(0, 255) as usize] as i64
    }

    fn get_dc_quant(&self, plane: usize) -> i64 {
        let q = self
            .fh
            .get_qindex(false, self.segment_id, self.current_q_index);
        match plane {
            0 => self.dc_q(q + self.fh.delta_q_y_dc),
            1 => self.dc_q(q + self.fh.delta_q_u_dc),
            _ => self.dc_q(q + self.fh.delta_q_v_dc),
        }
    }

    fn get_ac_quant(&self, plane: usize) -> i64 {
        let q = self
            .fh
            .get_qindex(false, self.segment_id, self.current_q_index);
        match plane {
            0 => self.ac_q(q),
            1 => self.ac_q(q + self.fh.delta_q_u_ac),
            _ => self.ac_q(q + self.fh.delta_q_v_ac),
        }
    }

    /// The reconstruct process (section 7.12.3).
    pub(crate) fn reconstruct(&mut self, plane: usize, x: usize, y: usize, tx_sz: usize) {
        let dq_denom: i64 = match tx_sz {
            3 | 9 | 10 | 17 | 18 => 2,
            4 | 11 | 12 => 4,
            _ => 1,
        };
        let log2w = TX_WIDTH_LOG2[tx_sz] as usize;
        let log2h = TX_HEIGHT_LOG2[tx_sz] as usize;
        let w = 1usize << log2w;
        let h = 1usize << log2h;
        let tw = w.min(32);
        let th = h.min(32);
        let tx_type = self.plane_tx_type;
        let flip_ud = matches!(
            tx_type,
            FLIPADST_DCT | FLIPADST_ADST | V_FLIPADST | FLIPADST_FLIPADST
        );
        let flip_lr = matches!(
            tx_type,
            DCT_FLIPADST | ADST_FLIPADST | H_FLIPADST | FLIPADST_FLIPADST
        );
        let bit_depth = u32::from(self.seq.bit_depth);
        let dc_quant = self.get_dc_quant(plane);
        let ac_quant = self.get_ac_quant(plane);
        let qm_level = if self.fh.using_qmatrix && tx_type < IDTX {
            let level = self.fh.seg_qm_level[plane][self.segment_id];
            if level < 15 {
                Some(level as usize)
            } else {
                None
            }
        } else {
            None
        };
        let max = (1i64 << (7 + bit_depth)) - 1;
        let min = -(1i64 << (7 + bit_depth));
        let scratch = &mut self.scratch.recon;
        for i in 0..th {
            for j in 0..tw {
                let q = if i == 0 && j == 0 { dc_quant } else { ac_quant };
                let q2 = match qm_level {
                    Some(level) => round2(
                        q * i64::from(super::qm::weight(
                            level,
                            usize::from(plane > 0),
                            QM_OFFSET[tx_sz] as usize + i * tw + j,
                        )),
                        5,
                    ),
                    None => q,
                };
                let dq = i64::from(self.quant[i * tw + j]) * q2;
                let sign = if dq < 0 { -1 } else { 1 };
                let dq2 = sign * ((dq.abs() & 0xFFFFFF) / dq_denom);
                scratch.dequant[i * 64 + j] = dq2.clamp(min, max);
            }
        }
        // Positions outside the coded 32x32 quadrant are zero.
        for i in 0..h.min(64) {
            for j in 0..w.min(64) {
                if i >= th || j >= tw {
                    scratch.dequant[i * 64 + j] = 0;
                }
            }
        }
        inverse_transform_2d(
            &scratch.dequant,
            &mut scratch.residual,
            tx_sz,
            tx_type,
            self.lossless,
            bit_depth,
        );
        let max_value = (1i64 << bit_depth) - 1;
        let frame = &mut self.f.curr.planes[plane];
        for i in 0..h {
            let yy = if flip_ud { h - i - 1 } else { i };
            if y + yy >= frame.height {
                continue;
            }
            for j in 0..w {
                let xx = if flip_lr { w - j - 1 } else { j };
                if x + xx >= frame.stride {
                    continue;
                }
                let current = i64::from(frame.get(x + xx, y + yy));
                let v = (current + scratch.residual[i * 64 + j]).clamp(0, max_value);
                frame.set(x + xx, y + yy, v as u16);
            }
        }
    }
}
