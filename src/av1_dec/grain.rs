//! Film grain synthesis (specification section 7.18.3), applied to an output
//! picture after decoding.

use super::header::FilmGrainParams;
use super::tables::GAUSSIAN_SEQUENCE;

#[inline(always)]
fn round2(x: i32, n: u32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

struct Rng(u16);

impl Rng {
    fn get(&mut self, bits: u32) -> i32 {
        let r = self.0;
        let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        let r = (r >> 1) | (bit << 15);
        self.0 = r;
        i32::from((r >> (16 - bits)) & ((1 << bits) - 1))
    }
}

/// Output planes of a picture, each `width x height` with no padding.
pub(crate) struct OutPlanes<'a> {
    pub(crate) planes: &'a mut [Vec<u16>],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) sub_x: usize,
    pub(crate) sub_y: usize,
    pub(crate) mono: bool,
    pub(crate) bit_depth: u32,
    pub(crate) matrix_identity: bool,
}

/// The film grain synthesis process.
pub(crate) fn apply_film_grain(p: &FilmGrainParams, out: &mut OutPlanes<'_>) {
    let bit_depth = out.bit_depth;
    let grain_center = 128i32 << (bit_depth - 8);
    let grain_min = -grain_center;
    let grain_max = (256i32 << (bit_depth - 8)) - 1 - grain_center;
    let sub_x = out.sub_x;
    let sub_y = out.sub_y;
    let num_planes = if out.mono { 1 } else { 3 };

    // Generate grain.
    let mut rng = Rng(p.grain_seed);
    let shift = 12 - bit_depth + u32::from(p.grain_scale_shift);
    let mut luma_grain = vec![[0i32; 82]; 73];
    for row in luma_grain.iter_mut() {
        for v in row.iter_mut() {
            let g = if p.num_y_points > 0 {
                i32::from(GAUSSIAN_SEQUENCE[rng.get(11) as usize])
            } else {
                0
            };
            *v = round2(g, shift);
        }
    }
    let ar_shift = u32::from(p.ar_coeff_shift_minus_6) + 6;
    let lag = p.ar_coeff_lag as isize;
    for y in 3..73usize {
        for x in 3..82 - 3usize {
            let mut s = 0i32;
            let mut pos = 0;
            'outer: for delta_row in -lag..=0 {
                for delta_col in -lag..=lag {
                    if delta_row == 0 && delta_col == 0 {
                        break 'outer;
                    }
                    let c = i32::from(p.ar_coeffs_y_plus_128[pos]) - 128;
                    s += luma_grain[(y as isize + delta_row) as usize]
                        [(x as isize + delta_col) as usize]
                        * c;
                    pos += 1;
                }
            }
            luma_grain[y][x] = (luma_grain[y][x] + round2(s, ar_shift)).clamp(grain_min, grain_max);
        }
    }
    let chroma_w = if sub_x != 0 { 44 } else { 82 };
    let chroma_h = if sub_y != 0 { 38 } else { 73 };
    let mut cb_grain = vec![vec![0i32; chroma_w]; chroma_h];
    let mut cr_grain = vec![vec![0i32; chroma_w]; chroma_h];
    if !out.mono {
        let mut rng = Rng(p.grain_seed ^ 0xb524);
        for row in cb_grain.iter_mut() {
            for v in row.iter_mut() {
                let g = if p.num_cb_points > 0 || p.chroma_scaling_from_luma {
                    i32::from(GAUSSIAN_SEQUENCE[rng.get(11) as usize])
                } else {
                    0
                };
                *v = round2(g, shift);
            }
        }
        let mut rng = Rng(p.grain_seed ^ 0x49d8);
        for row in cr_grain.iter_mut() {
            for v in row.iter_mut() {
                let g = if p.num_cr_points > 0 || p.chroma_scaling_from_luma {
                    i32::from(GAUSSIAN_SEQUENCE[rng.get(11) as usize])
                } else {
                    0
                };
                *v = round2(g, shift);
            }
        }
        for y in 3..chroma_h {
            for x in 3..chroma_w - 3 {
                let mut s0 = 0i32;
                let mut s1 = 0i32;
                let mut pos = 0;
                'outer2: for delta_row in -lag..=0 {
                    for delta_col in -lag..=lag {
                        let c0 = i32::from(p.ar_coeffs_cb_plus_128[pos]) - 128;
                        let c1 = i32::from(p.ar_coeffs_cr_plus_128[pos]) - 128;
                        if delta_row == 0 && delta_col == 0 {
                            if p.num_y_points > 0 {
                                let mut luma = 0i32;
                                let luma_x = ((x - 3) << sub_x) + 3;
                                let luma_y = ((y - 3) << sub_y) + 3;
                                for i in 0..=sub_y {
                                    for j in 0..=sub_x {
                                        luma += luma_grain[luma_y + i][luma_x + j];
                                    }
                                }
                                let luma = round2(luma, (sub_x + sub_y) as u32);
                                s0 += luma * c0;
                                s1 += luma * c1;
                            }
                            break 'outer2;
                        }
                        let yy = (y as isize + delta_row) as usize;
                        let xx = (x as isize + delta_col) as usize;
                        s0 += cb_grain[yy][xx] * c0;
                        s1 += cr_grain[yy][xx] * c1;
                        pos += 1;
                    }
                }
                cb_grain[y][x] =
                    (cb_grain[y][x] + round2(s0, ar_shift)).clamp(grain_min, grain_max);
                cr_grain[y][x] =
                    (cr_grain[y][x] + round2(s1, ar_shift)).clamp(grain_min, grain_max);
            }
        }
    }

    // Scaling lookup initialization.
    let mut scaling_lut = [[0i32; 256]; 3];
    for plane in 0..num_planes {
        let (num_points, xs, ys): (usize, &[u8], &[u8]) =
            if plane == 0 || p.chroma_scaling_from_luma {
                (p.num_y_points, &p.point_y_value, &p.point_y_scaling)
            } else if plane == 1 {
                (p.num_cb_points, &p.point_cb_value, &p.point_cb_scaling)
            } else {
                (p.num_cr_points, &p.point_cr_value, &p.point_cr_scaling)
            };
        if num_points == 0 {
            continue;
        }
        let lut = &mut scaling_lut[plane];
        for x in 0..xs[0] as usize {
            lut[x] = i32::from(ys[0]);
        }
        for i in 0..num_points - 1 {
            let delta_y = i32::from(ys[i + 1]) - i32::from(ys[i]);
            let delta_x = i32::from(xs[i + 1]) - i32::from(xs[i]);
            if delta_x <= 0 {
                continue;
            }
            let delta = delta_y * ((65536 + (delta_x >> 1)) / delta_x);
            for x in 0..delta_x {
                let v = i32::from(ys[i]) + ((x * delta + 32768) >> 16);
                let idx = xs[i] as usize + x as usize;
                if idx < 256 {
                    lut[idx] = v;
                }
            }
        }
        for x in xs[num_points - 1] as usize..256 {
            lut[x] = i32::from(ys[num_points - 1]);
        }
    }
    let scale_lut = |plane: usize, index: i32| -> i32 {
        let shift = bit_depth - 8;
        let x = (index >> shift) as usize;
        let rem = index - ((x as i32) << shift);
        if bit_depth == 8 || x == 255 {
            scaling_lut[plane][x.min(255)]
        } else {
            let start = scaling_lut[plane][x];
            let end = scaling_lut[plane][x + 1];
            start + round2((end - start) * rem, shift)
        }
    };

    // Add noise: build noise stripes, then the noise image, then blend.
    let w = out.width;
    let h = out.height;
    let stripe_count = h.div_ceil(2) / 16 + 2;
    let stripe_w = w + 64;
    let mut noise_stripe: Vec<[Vec<i32>; 3]> = (0..stripe_count)
        .map(|_| {
            [
                vec![0i32; 34 * stripe_w],
                vec![0i32; 34 * stripe_w],
                vec![0i32; 34 * stripe_w],
            ]
        })
        .collect();
    let mut luma_num = 0usize;
    let mut y = 0usize;
    while y < h.div_ceil(2) {
        let mut rng = Rng(p.grain_seed);
        rng.0 ^= (((luma_num * 37 + 178) & 255) << 8) as u16;
        rng.0 ^= ((luma_num * 173 + 105) & 255) as u16;
        let mut x = 0usize;
        while x < w.div_ceil(2) {
            let rand = rng.get(8);
            let offset_x = (rand >> 4) as usize;
            let offset_y = (rand & 15) as usize;
            for plane in 0..num_planes {
                let psx = if plane > 0 { sub_x } else { 0 };
                let psy = if plane > 0 { sub_y } else { 0 };
                let plane_offset_x = if psx != 0 {
                    6 + offset_x
                } else {
                    9 + offset_x * 2
                };
                let plane_offset_y = if psy != 0 {
                    6 + offset_y
                } else {
                    9 + offset_y * 2
                };
                for i in 0..(34 >> psy) {
                    for j in 0..(34 >> psx) {
                        let mut g = match plane {
                            0 => luma_grain[plane_offset_y + i][plane_offset_x + j],
                            1 => cb_grain[plane_offset_y + i][plane_offset_x + j],
                            _ => cr_grain[plane_offset_y + i][plane_offset_x + j],
                        };
                        let stripe = &mut noise_stripe[luma_num][plane];
                        if psx == 0 {
                            let col = x * 2 + j;
                            if col >= stripe_w {
                                continue;
                            }
                            if j < 2 && p.overlap_flag && x > 0 {
                                let old = stripe[i * stripe_w + col];
                                g = if j == 0 {
                                    old * 27 + g * 17
                                } else {
                                    old * 17 + g * 27
                                };
                                g = round2(g, 5).clamp(grain_min, grain_max);
                            }
                            stripe[i * stripe_w + col] = g;
                        } else {
                            let col = x + j;
                            if col >= stripe_w {
                                continue;
                            }
                            if j == 0 && p.overlap_flag && x > 0 {
                                let old = stripe[i * stripe_w + col];
                                g = old * 23 + g * 22;
                                g = round2(g, 5).clamp(grain_min, grain_max);
                            }
                            stripe[i * stripe_w + col] = g;
                        }
                    }
                }
            }
            x += 16;
        }
        luma_num += 1;
        y += 16;
    }
    let mut noise_image: Vec<Vec<i32>> = Vec::with_capacity(num_planes);
    for plane in 0..num_planes {
        let psx = if plane > 0 { sub_x } else { 0 };
        let psy = if plane > 0 { sub_y } else { 0 };
        let pw = (w + psx) >> psx;
        let ph = (h + psy) >> psy;
        let mut img = vec![0i32; pw * ph];
        for y in 0..ph {
            let luma_num = y >> (5 - psy);
            let i = y - (luma_num << (5 - psy));
            for x in 0..pw {
                let mut g = noise_stripe[luma_num][plane][i * stripe_w + x];
                if psy == 0 {
                    if i < 2 && luma_num > 0 && p.overlap_flag {
                        let old = noise_stripe[luma_num - 1][plane][(i + 32) * stripe_w + x];
                        g = if i == 0 {
                            old * 27 + g * 17
                        } else {
                            old * 17 + g * 27
                        };
                        g = round2(g, 5).clamp(grain_min, grain_max);
                    }
                } else if i < 1 && luma_num > 0 && p.overlap_flag {
                    let old = noise_stripe[luma_num - 1][plane][(i + 16) * stripe_w + x];
                    g = old * 23 + g * 22;
                    g = round2(g, 5).clamp(grain_min, grain_max);
                }
                img[y * pw + x] = g;
            }
        }
        noise_image.push(img);
    }
    let (min_value, max_luma, max_chroma) = if p.clip_to_restricted_range {
        let min_value = 16 << (bit_depth - 8);
        let max_luma = 235 << (bit_depth - 8);
        let max_chroma = if out.matrix_identity {
            max_luma
        } else {
            240 << (bit_depth - 8)
        };
        (min_value, max_luma, max_chroma)
    } else {
        let max = (256 << (bit_depth - 8)) - 1;
        (0, max, max)
    };
    let scaling_shift = u32::from(p.grain_scaling_minus_8) + 8;
    let pixel_max = (1i32 << bit_depth) - 1;
    if !out.mono {
        let cw = (w + sub_x) >> sub_x;
        let ch = (h + sub_y) >> sub_y;
        for y in 0..ch {
            for x in 0..cw {
                let luma_x = x << sub_x;
                let luma_y = y << sub_y;
                let luma_next_x = (luma_x + 1).min(w - 1);
                let average_luma = if sub_x != 0 {
                    round2(
                        i32::from(out.planes[0][luma_y * w + luma_x])
                            + i32::from(out.planes[0][luma_y * w + luma_next_x]),
                        1,
                    )
                } else {
                    i32::from(out.planes[0][luma_y * w + luma_x])
                };
                if p.num_cb_points > 0 || p.chroma_scaling_from_luma {
                    let orig = i32::from(out.planes[1][y * cw + x]);
                    let merged = if p.chroma_scaling_from_luma {
                        average_luma
                    } else {
                        let combined = average_luma * (i32::from(p.cb_luma_mult) - 128)
                            + orig * (i32::from(p.cb_mult) - 128);
                        ((combined >> 6) + ((i32::from(p.cb_offset) - 256) << (bit_depth - 8)))
                            .clamp(0, pixel_max)
                    };
                    let noise = round2(
                        scale_lut(1, merged) * noise_image[1][y * cw + x],
                        scaling_shift,
                    );
                    out.planes[1][y * cw + x] = (orig + noise).clamp(min_value, max_chroma) as u16;
                }
                if p.num_cr_points > 0 || p.chroma_scaling_from_luma {
                    let orig = i32::from(out.planes[2][y * cw + x]);
                    let merged = if p.chroma_scaling_from_luma {
                        average_luma
                    } else {
                        let combined = average_luma * (i32::from(p.cr_luma_mult) - 128)
                            + orig * (i32::from(p.cr_mult) - 128);
                        ((combined >> 6) + ((i32::from(p.cr_offset) - 256) << (bit_depth - 8)))
                            .clamp(0, pixel_max)
                    };
                    let noise = round2(
                        scale_lut(2, merged) * noise_image[2][y * cw + x],
                        scaling_shift,
                    );
                    out.planes[2][y * cw + x] = (orig + noise).clamp(min_value, max_chroma) as u16;
                }
            }
        }
    }
    if p.num_y_points > 0 {
        for y in 0..h {
            for x in 0..w {
                let orig = i32::from(out.planes[0][y * w + x]);
                let noise = round2(
                    scale_lut(0, orig) * noise_image[0][y * w + x],
                    scaling_shift,
                );
                out.planes[0][y * w + x] = (orig + noise).clamp(min_value, max_luma) as u16;
            }
        }
    }
}
