//! The 8-tap sub-pixel convolution of inter prediction
//! (`vp9_dec::recon::convolve`, libvpx's `vpx_convolve8_c` family), with
//! one output pixel per lane.
//!
//! The regular, smooth, sharp and bilinear filters are all tables of the
//! same 8-tap shape, so one kernel covers them, and compound prediction's
//! rounded average is folded into its store. Every pixel of an output row
//! shares one vertical phase, so the vertical pass is vectorized for scaled
//! references too; a scaled horizontal step gives each pixel of a row its
//! own phase, and that pass alone stays scalar.

// Loops over vectors index rather than iterate: an iterator adapter or an
// `array::from_fn` closure over vector values is a separate function the
// inliner can leave outside the `#[target_feature]` wrapper, compiled at the
// baseline instruction set (#341).
#![allow(clippy::needless_range_loop)]

use crate::av1_simd::vector::I32x;
pub(crate) use crate::vp9_dec::recon::Kernel;

/// The eight taps of one phase, one splatted vector each.
#[inline(always)]
unsafe fn splat_taps<V: I32x>(taps: &[i16; 8]) -> [V; 8] {
    unsafe {
        let mut splatted = [V::zero(); 8];
        for k in 0..8 {
            splatted[k] = V::splat(i32::from(taps[k]));
        }
        splatted
    }
}

/// `ROUND_POWER_OF_TWO(sum, 7)` of eight taps applied to the vectors at
/// `src[at + k * pitch..]`, clamped to a pixel.
#[inline(always)]
unsafe fn filter<V: I32x>(src: &[u8], at: usize, pitch: usize, taps: &[V; 8]) -> V {
    unsafe {
        let mut sum = V::splat(64);
        for k in 0..8 {
            sum = sum.add(V::load_u8(&src[at + k * pitch..]).mul(taps[k]));
        }
        sum.sra::<7>().clamp(V::zero(), V::splat(255))
    }
}

/// Stores `value` (already a pixel) at `dest[at..]`, or its rounded average
/// with what is there for the second prediction of a compound block.
#[inline(always)]
unsafe fn store<V: I32x>(dest: &mut [u8], at: usize, value: V, average: bool) {
    unsafe {
        let value = if average {
            V::load_u8(&dest[at..])
                .add(value)
                .add(V::splat(1))
                .sra::<1>()
        } else {
            value
        };
        value.store_u8_clamped(&mut dest[at..]);
    }
}

/// The vector form of `vp9_dec::recon::convolve`. Returns `false`, having
/// written nothing, when `w` is not a whole number of vectors.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn convolve<V: I32x>(
    src: &[u8],
    origin: usize,
    src_stride: usize,
    dest: &mut [u8],
    stride: usize,
    w: usize,
    h: usize,
    kernel: &Kernel,
    x_frac: i32,
    x_step: i32,
    y_frac: i32,
    y_step: i32,
    average: bool,
    temp: &mut [u8; 64 * 135],
) -> bool {
    unsafe {
        let lanes = V::LANES;
        if w % lanes != 0 {
            return false;
        }
        let whole_x = x_frac == 0 && x_step == 16;
        let whole_y = y_frac == 0 && y_step == 16;

        if whole_x && whole_y {
            for y in 0..h {
                let line = origin + y * src_stride;
                if average {
                    for x in (0..w).step_by(lanes) {
                        store(dest, y * stride + x, V::load_u8(&src[line + x..]), true);
                    }
                } else {
                    dest[y * stride..][..w].copy_from_slice(&src[line..line + w]);
                }
            }
            return true;
        }
        if whole_y {
            // A scaled step gives every pixel its own phase.
            if x_step != 16 {
                return false;
            }
            let taps = splat_taps::<V>(&kernel[(x_frac & 15) as usize]);
            let offset = (x_frac >> 4) as usize;
            for y in 0..h {
                let line = origin + y * src_stride - 3 + offset;
                for x in (0..w).step_by(lanes) {
                    let value = filter(src, line + x, 1, &taps);
                    store(dest, y * stride + x, value, average);
                }
            }
            return true;
        }
        if whole_x {
            let top = origin - 3 * src_stride;
            let mut y_q4 = y_frac;
            for y in 0..h {
                let base = top + (y_q4 >> 4) as usize * src_stride;
                let taps = splat_taps::<V>(&kernel[(y_q4 & 15) as usize]);
                for x in (0..w).step_by(lanes) {
                    let value = filter(src, base + x, src_stride, &taps);
                    store(dest, y * stride + x, value, average);
                }
                y_q4 += y_step;
            }
            return true;
        }

        let intermediate_height = ((((h as i32 - 1) * y_step + y_frac) >> 4) + 8) as usize;
        let top = origin - 3 * src_stride - 3;
        if x_step == 16 {
            let taps = splat_taps::<V>(&kernel[(x_frac & 15) as usize]);
            let offset = (x_frac >> 4) as usize;
            for row in 0..intermediate_height {
                let line = top + row * src_stride + offset;
                for x in (0..w).step_by(lanes) {
                    filter(src, line + x, 1, &taps).store_u8_clamped(&mut temp[row * 64 + x..]);
                }
            }
        } else {
            // The scalar reference's horizontal pass, unchanged.
            for row in 0..intermediate_height {
                let line = &src[top + row * src_stride..];
                let mut x_q4 = x_frac;
                for x in 0..w {
                    let base = (x_q4 >> 4) as usize;
                    let taps = &kernel[(x_q4 & 15) as usize];
                    let mut sum = 0i32;
                    for k in 0..8 {
                        sum += i32::from(line[base + k]) * i32::from(taps[k]);
                    }
                    temp[row * 64 + x] = ((sum + 64) >> 7).clamp(0, 255) as u8;
                    x_q4 += x_step;
                }
            }
        }
        let mut y_q4 = y_frac;
        for y in 0..h {
            let base = (y_q4 >> 4) as usize * 64;
            let taps = splat_taps::<V>(&kernel[(y_q4 & 15) as usize]);
            for x in (0..w).step_by(lanes) {
                let value = filter(&temp[..], base + x, 64, &taps);
                store(dest, y * stride + x, value, average);
            }
            y_q4 += y_step;
        }
        true
    }
}
