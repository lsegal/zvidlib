//! Reconstruction kernels: the two-dimensional inverse transforms (section
//! 8.7.2 of the VP9 specification), the intra predictors (section 8.5.1)
//! and the sub-pixel convolution of inter prediction (section 8.5.2.3).
//!
//! The kernels reproduce libvpx's C implementations, including the
//! shortcuts its decoder takes on the end-of-block position, so that every
//! intermediate value rounds the way the reference decoder's does.

use super::idct1d::{iadst4, iadst8, iadst16, idct4, idct8, idct16, idct32};

pub(super) const DCT_DCT: u8 = 0;
pub(super) const ADST_DCT: u8 = 1;
pub(super) const DCT_ADST: u8 = 2;

#[inline]
fn clip_pixel_add(dest: u8, residual: i32) -> u8 {
    (i32::from(dest) + residual).clamp(0, 255) as u8
}

#[inline]
fn round_power_of_two(value: i32, bits: u32) -> i32 {
    (value + (1 << (bits - 1))) >> bits
}

type Transform1d = fn(&[i32], &mut [i32]);

/// Applies a separable inverse transform of size `n` to `input` and adds
/// the result to `dest`. Only the first `rows` rows of `input` are
/// transformed; libvpx skips the rest when the end-of-block position
/// guarantees they are zero.
#[allow(clippy::too_many_arguments)]
fn inverse_2d(
    input: &[i32],
    dest: &mut [u8],
    stride: usize,
    n: usize,
    rows: usize,
    row_transform: Transform1d,
    column_transform: Transform1d,
    shift: u32,
    skip_zero_rows: bool,
) {
    match n {
        4 => inverse_2d_n::<4>(
            input,
            dest,
            stride,
            rows,
            row_transform,
            column_transform,
            shift,
            skip_zero_rows,
        ),
        8 => inverse_2d_n::<8>(
            input,
            dest,
            stride,
            rows,
            row_transform,
            column_transform,
            shift,
            skip_zero_rows,
        ),
        16 => inverse_2d_n::<16>(
            input,
            dest,
            stride,
            rows,
            row_transform,
            column_transform,
            shift,
            skip_zero_rows,
        ),
        _ => inverse_2d_n::<32>(
            input,
            dest,
            stride,
            rows,
            row_transform,
            column_transform,
            shift,
            skip_zero_rows,
        ),
    }
}

/// [`inverse_2d`] for one size, so the intermediate is only as large as the
/// block.
#[allow(clippy::too_many_arguments)]
fn inverse_2d_n<const N: usize>(
    input: &[i32],
    dest: &mut [u8],
    stride: usize,
    rows: usize,
    row_transform: Transform1d,
    column_transform: Transform1d,
    shift: u32,
    skip_zero_rows: bool,
) {
    let mut out = [[0i32; N]; N];
    for (row, output) in out.iter_mut().enumerate().take(rows) {
        let input_row = &input[row * N..row * N + N];
        if skip_zero_rows && input_row.iter().fold(0i16, |acc, &v| acc | v as i16) == 0 {
            continue;
        }
        row_transform(input_row, output);
    }
    let mut column = [0i32; N];
    let mut result = [0i32; N];
    for x in 0..N {
        for y in 0..N {
            column[y] = out[y][x];
        }
        column_transform(&column, &mut result);
        for y in 0..N {
            let pixel = &mut dest[y * stride + x];
            *pixel = clip_pixel_add(*pixel, round_power_of_two(result[y], shift));
        }
    }
}

/// The DC-only shortcut every DCT size takes when `eob == 1`.
fn idct_dc_only(input: &[i32], dest: &mut [u8], stride: usize, n: usize, shift: u32) {
    let round = |x: i64| (x + (1 << 13)) >> 14;
    let out = round(i64::from(input[0] as i16) * 11585) as i32;
    let out = round(i64::from(out) * 11585) as i32;
    let a1 = round_power_of_two(out, shift);
    for y in 0..n {
        for pixel in &mut dest[y * stride..y * stride + n] {
            *pixel = clip_pixel_add(*pixel, a1);
        }
    }
}

/// The 4x4 inverse Walsh-Hadamard transform of lossless frames
/// (`vpx_iwht4x4_16_add` and its DC-only form).
fn iwht4x4_add(input: &[i32], dest: &mut [u8], stride: usize, eob: usize) {
    if eob <= 1 {
        let mut a1 = input[0] >> 2;
        let e1 = a1 >> 1;
        a1 -= e1;
        let tmp = [a1, e1, e1, e1];
        for x in 0..4 {
            let e1 = tmp[x] >> 1;
            let a1 = tmp[x] - e1;
            dest[x] = clip_pixel_add(dest[x], a1);
            for y in 1..4 {
                dest[y * stride + x] = clip_pixel_add(dest[y * stride + x], e1);
            }
        }
        return;
    }
    let mut output = [0i32; 16];
    for row in 0..4 {
        let ip = &input[row * 4..row * 4 + 4];
        let mut a1 = ip[0] >> 2;
        let mut c1 = ip[1] >> 2;
        let mut d1 = ip[2] >> 2;
        let mut b1 = ip[3] >> 2;
        a1 += c1;
        d1 -= b1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;
        output[row * 4..row * 4 + 4].copy_from_slice(&[a1, b1, c1, d1]);
    }
    for x in 0..4 {
        let mut a1 = output[x];
        let mut c1 = output[4 + x];
        let mut d1 = output[8 + x];
        let mut b1 = output[12 + x];
        a1 += c1;
        d1 -= b1;
        let e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= b1;
        d1 += c1;
        for (y, value) in [a1, b1, c1, d1].into_iter().enumerate() {
            dest[y * stride + x] = clip_pixel_add(dest[y * stride + x], value);
        }
    }
}

/// Inverse transforms one transform block's dequantized coefficients and
/// adds them to the prediction in `dest`, choosing the same kernels as
/// libvpx's `inverse_transform_block_inter`/`_intra` for the given
/// end-of-block position.
pub(super) fn inverse_transform_add(
    coefficients: &[i32],
    dest: &mut [u8],
    stride: usize,
    tx_size: u8,
    tx_type: u8,
    eob: usize,
    lossless: bool,
) {
    if lossless {
        iwht4x4_add(coefficients, dest, stride, eob);
        return;
    }
    let (n, shift): (usize, u32) = match tx_size {
        0 => (4, 4),
        1 => (8, 5),
        2 => (16, 6),
        _ => (32, 6),
    };
    if tx_type == DCT_DCT || tx_size == 3 {
        if eob == 1 {
            idct_dc_only(coefficients, dest, stride, n, shift);
            return;
        }
        let (transform, rows): (Transform1d, usize) = match tx_size {
            0 => (idct4, 4),
            1 => (idct8, if eob <= 12 { 4 } else { 8 }),
            2 => (
                idct16,
                if eob <= 10 {
                    4
                } else if eob <= 38 {
                    8
                } else {
                    16
                },
            ),
            _ => (
                idct32,
                if eob <= 34 {
                    8
                } else if eob <= 135 {
                    16
                } else {
                    32
                },
            ),
        };
        let skip_zero_rows = tx_size == 3 && rows == 32;
        inverse_2d(
            coefficients,
            dest,
            stride,
            n,
            rows,
            transform,
            transform,
            shift,
            skip_zero_rows,
        );
        return;
    }
    let (dct, adst): (Transform1d, Transform1d) = match tx_size {
        0 => (idct4, iadst4),
        1 => (idct8, iadst8),
        _ => (idct16, iadst16),
    };
    // `tx_type` names the vertical (column) transform first.
    let (column_transform, row_transform) = match tx_type {
        ADST_DCT => (adst, dct),
        DCT_ADST => (dct, adst),
        _ => (adst, adst),
    };
    inverse_2d(
        coefficients,
        dest,
        stride,
        n,
        n,
        row_transform,
        column_transform,
        shift,
        false,
    );
}

/// The prediction mode numbering of the bitstream.
pub(super) const DC_PRED: u8 = 0;
pub(super) const V_PRED: u8 = 1;
pub(super) const H_PRED: u8 = 2;
pub(super) const D45_PRED: u8 = 3;
pub(super) const D135_PRED: u8 = 4;
pub(super) const D117_PRED: u8 = 5;
pub(super) const D207_PRED: u8 = 7;
pub(super) const D63_PRED: u8 = 8;
pub(super) const TM_PRED: u8 = 9;

#[inline]
fn avg2(a: u8, b: u8) -> u8 {
    ((u32::from(a) + u32::from(b) + 1) >> 1) as u8
}

#[inline]
fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((u32::from(a) + 2 * u32::from(b) + u32::from(c) + 2) >> 2) as u8
}

/// Edge pixels for one intra prediction: `above[0]` is the above-left
/// pixel and `above[1..=2 * bs]` the row above, `left[..bs]` the column to
/// the left.
pub(super) struct IntraEdges {
    pub(super) above: [u8; 65],
    pub(super) left: [u8; 32],
}

/// Fills a `bs`x`bs` block of `dest` with the intra prediction `mode`
/// (libvpx's `vpx_*_predictor_NxN`, with the DC variant chosen by edge
/// availability as `build_intra_predictors` does).
pub(super) fn predict_intra(
    dest: &mut [u8],
    stride: usize,
    bs: usize,
    mode: u8,
    edges: &IntraEdges,
    have_left: bool,
    have_above: bool,
) {
    let above = |i: isize| edges.above[(i + 1) as usize];
    let left = &edges.left;
    match mode {
        DC_PRED => {
            let value = match (have_left, have_above) {
                (false, false) => 128,
                (true, false) => {
                    let sum: u32 = left[..bs].iter().map(|&v| u32::from(v)).sum();
                    (sum + (bs as u32 >> 1)) / bs as u32
                }
                (false, true) => {
                    let sum: u32 = edges.above[1..=bs].iter().map(|&v| u32::from(v)).sum();
                    (sum + (bs as u32 >> 1)) / bs as u32
                }
                (true, true) => {
                    let sum: u32 = left[..bs]
                        .iter()
                        .chain(&edges.above[1..=bs])
                        .map(|&v| u32::from(v))
                        .sum();
                    let count = 2 * bs as u32;
                    (sum + (count >> 1)) / count
                }
            } as u8;
            for y in 0..bs {
                dest[y * stride..y * stride + bs].fill(value);
            }
        }
        V_PRED => {
            for y in 0..bs {
                dest[y * stride..y * stride + bs].copy_from_slice(&edges.above[1..=bs]);
            }
        }
        H_PRED => {
            for y in 0..bs {
                dest[y * stride..y * stride + bs].fill(left[y]);
            }
        }
        TM_PRED => {
            let top_left = i32::from(above(-1));
            for y in 0..bs {
                for x in 0..bs {
                    dest[y * stride + x] = (i32::from(left[y]) + i32::from(above(x as isize))
                        - top_left)
                        .clamp(0, 255) as u8;
                }
            }
        }
        D45_PRED => {
            // The specification's form, which equals libvpx's 4x4 kernel and,
            // because larger blocks never see the above-right pixels (they
            // are replicated), its generic one.
            for y in 0..bs {
                for x in 0..bs {
                    let i = (x + y) as isize;
                    dest[y * stride + x] = if x + y + 2 < 2 * bs {
                        avg3(above(i), above(i + 1), above(i + 2))
                    } else {
                        above(2 * bs as isize - 1)
                    };
                }
            }
        }
        D63_PRED => {
            for y in 0..bs {
                for x in 0..bs {
                    let i = (y / 2 + x) as isize;
                    dest[y * stride + x] = if y & 1 == 1 {
                        avg3(above(i), above(i + 1), above(i + 2))
                    } else {
                        avg2(above(i), above(i + 1))
                    };
                }
            }
        }
        D207_PRED => {
            // libvpx's `d207_predictor`.
            let mut block = [[0u8; 32]; 32];
            for r in 0..bs - 1 {
                block[r][0] = avg2(left[r], left[r + 1]);
            }
            block[bs - 1][0] = left[bs - 1];
            for r in 0..bs - 2 {
                block[r][1] = avg3(left[r], left[r + 1], left[r + 2]);
            }
            block[bs - 2][1] = avg3(left[bs - 2], left[bs - 1], left[bs - 1]);
            block[bs - 1][1] = left[bs - 1];
            for c in 2..bs {
                block[bs - 1][c] = left[bs - 1];
            }
            for r in (0..bs - 1).rev() {
                for c in 2..bs {
                    block[r][c] = block[r + 1][c - 2];
                }
            }
            for y in 0..bs {
                dest[y * stride..y * stride + bs].copy_from_slice(&block[y][..bs]);
            }
        }
        D117_PRED => {
            // libvpx's `d117_predictor`.
            let mut block = [[0u8; 32]; 32];
            for c in 0..bs {
                block[0][c] = avg2(above(c as isize - 1), above(c as isize));
            }
            block[1][0] = avg3(left[0], above(-1), above(0));
            for c in 1..bs {
                block[1][c] = avg3(
                    above(c as isize - 2),
                    above(c as isize - 1),
                    above(c as isize),
                );
            }
            block[2][0] = avg3(above(-1), left[0], left[1]);
            for r in 3..bs {
                block[r][0] = avg3(left[r - 3], left[r - 2], left[r - 1]);
            }
            for r in 2..bs {
                for c in 1..bs {
                    block[r][c] = block[r - 2][c - 1];
                }
            }
            for y in 0..bs {
                dest[y * stride..y * stride + bs].copy_from_slice(&block[y][..bs]);
            }
        }
        D135_PRED => {
            // libvpx's `d135_predictor`.
            let mut border = [0u8; 63];
            for i in 0..bs - 2 {
                border[i] = avg3(left[bs - 3 - i], left[bs - 2 - i], left[bs - 1 - i]);
            }
            border[bs - 2] = avg3(above(-1), left[0], left[1]);
            border[bs - 1] = avg3(left[0], above(-1), above(0));
            border[bs] = avg3(above(-1), above(0), above(1));
            for i in 0..bs - 2 {
                border[bs + 1 + i] = avg3(
                    above(i as isize),
                    above(i as isize + 1),
                    above(i as isize + 2),
                );
            }
            for y in 0..bs {
                dest[y * stride..y * stride + bs]
                    .copy_from_slice(&border[bs - 1 - y..2 * bs - 1 - y]);
            }
        }
        _ => {
            // D153: libvpx's `d153_predictor`.
            let mut block = [[0u8; 32]; 32];
            block[0][0] = avg2(above(-1), left[0]);
            for r in 1..bs {
                block[r][0] = avg2(left[r - 1], left[r]);
            }
            block[0][1] = avg3(left[0], above(-1), above(0));
            block[1][1] = avg3(above(-1), left[0], left[1]);
            for r in 2..bs {
                block[r][1] = avg3(left[r - 2], left[r - 1], left[r]);
            }
            for c in 0..bs - 2 {
                block[0][c + 2] = avg3(
                    above(c as isize - 1),
                    above(c as isize),
                    above(c as isize + 1),
                );
            }
            for r in 1..bs {
                for c in 0..bs - 2 {
                    block[r][c + 2] = block[r - 1][c];
                }
            }
            for y in 0..bs {
                dest[y * stride..y * stride + bs].copy_from_slice(&block[y][..bs]);
            }
        }
    }
}

/// One libvpx `InterpKernel` table: 16 sub-pixel phases of 8 taps.
pub(super) type Kernel = [[i16; 8]; 16];

/// The separable 8-tap convolution of `vpx_convolve8_c` and its scaled
/// variants, writing `w`x`h` pixels to `dest`. `src[origin]` is the
/// integer position of the first output pixel, and the window must reach
/// three pixels left of and above it and four right of and below the last
/// tap. The horizontal pass produces `(((h - 1) * y_step + y_frac) >> 4) + 8`
/// rows starting three rows above the origin, the intermediate libvpx clips
/// to 8 bits before the vertical pass. Positions are in 1/16 pixel.
///
/// A phase-zero tap set is the identity, which is why libvpx's copy and
/// one-dimensional convolutions give what this would; whole-pixel
/// directions take those shortcuts here too.
#[allow(clippy::too_many_arguments)]
pub(super) fn convolve(
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
) {
    #[inline]
    fn store(pixel: &mut u8, value: u8, average: bool) {
        *pixel = if average {
            ((u32::from(*pixel) + u32::from(value) + 1) >> 1) as u8
        } else {
            value
        };
    }
    #[inline]
    fn filter(pixels: &[u8], taps: &[i16; 8]) -> u8 {
        let mut sum = 0i32;
        for k in 0..8 {
            sum += i32::from(pixels[k]) * i32::from(taps[k]);
        }
        ((sum + 64) >> 7).clamp(0, 255) as u8
    }
    let whole_x = x_frac == 0 && x_step == 16;
    let whole_y = y_frac == 0 && y_step == 16;

    if whole_x && whole_y {
        for y in 0..h {
            let line = &src[origin + y * src_stride..][..w];
            let out = &mut dest[y * stride..][..w];
            if average {
                for (pixel, &value) in out.iter_mut().zip(line) {
                    store(pixel, value, true);
                }
            } else {
                out.copy_from_slice(line);
            }
        }
        return;
    }
    if whole_y {
        for y in 0..h {
            let line = &src[origin + y * src_stride - 3..];
            let out = &mut dest[y * stride..][..w];
            let mut x_q4 = x_frac;
            for pixel in out {
                let base = (x_q4 >> 4) as usize;
                store(
                    pixel,
                    filter(&line[base..base + 8], &kernel[(x_q4 & 15) as usize]),
                    average,
                );
                x_q4 += x_step;
            }
        }
        return;
    }
    if whole_x {
        let top = origin - 3 * src_stride;
        let mut y_q4 = y_frac;
        for y in 0..h {
            let base = top + (y_q4 >> 4) as usize * src_stride;
            let taps = &kernel[(y_q4 & 15) as usize];
            let out = &mut dest[y * stride..y * stride + w];
            for (x, pixel) in out.iter_mut().enumerate() {
                let mut sum = 0i32;
                for (k, &tap) in taps.iter().enumerate() {
                    sum += i32::from(src[base + k * src_stride + x]) * i32::from(tap);
                }
                store(pixel, ((sum + 64) >> 7).clamp(0, 255) as u8, average);
            }
            y_q4 += y_step;
        }
        return;
    }

    let intermediate_height = ((((h as i32 - 1) * y_step + y_frac) >> 4) + 8) as usize;
    let top = origin - 3 * src_stride - 3;
    for row in 0..intermediate_height {
        let line = &src[top + row * src_stride..];
        let mut x_q4 = x_frac;
        for x in 0..w {
            let base = (x_q4 >> 4) as usize;
            temp[row * 64 + x] = filter(&line[base..base + 8], &kernel[(x_q4 & 15) as usize]);
            x_q4 += x_step;
        }
    }
    let mut y_q4 = y_frac;
    for y in 0..h {
        let base = (y_q4 >> 4) as usize;
        let taps = &kernel[(y_q4 & 15) as usize];
        let out = &mut dest[y * stride..y * stride + w];
        for (x, pixel) in out.iter_mut().enumerate() {
            let mut sum = 0i32;
            for (k, &tap) in taps.iter().enumerate() {
                sum += i32::from(temp[(base + k) * 64 + x]) * i32::from(tap);
            }
            store(pixel, ((sum + 64) >> 7).clamp(0, 255) as u8, average);
        }
        y_q4 += y_step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_only_matches_the_full_transform() {
        for tx_size in 0..4u8 {
            let n = 4usize << tx_size;
            let mut coefficients = vec![0i32; n * n];
            coefficients[0] = 517;
            let mut shortcut = vec![100u8; n * n];
            inverse_transform_add(&coefficients, &mut shortcut, n, tx_size, DCT_DCT, 1, false);
            let mut full = vec![100u8; n * n];
            inverse_transform_add(&coefficients, &mut full, n, tx_size, DCT_DCT, 1024, false);
            assert_eq!(shortcut, full, "tx size {tx_size}");
        }
    }

    #[test]
    fn lossless_dc_matches_the_full_walsh_hadamard_transform() {
        let mut coefficients = [0i32; 16];
        coefficients[0] = 37;
        let mut shortcut = [50u8; 16];
        iwht4x4_add(&coefficients, &mut shortcut, 4, 1);
        let mut full = [50u8; 16];
        iwht4x4_add(&coefficients, &mut full, 4, 16);
        assert_eq!(shortcut, full);
    }

    #[test]
    fn integer_positions_copy_the_source() {
        let kernel = super::super::tables::FILTER_KERNELS[0];
        let src: Vec<u8> = (0..16 * 16).map(|i| (i * 7 % 256) as u8).collect();
        let origin = 3 * 16 + 3;
        let mut dest = [0u8; 16];
        convolve(
            &src,
            origin,
            16,
            &mut dest,
            4,
            4,
            4,
            &kernel,
            0,
            16,
            0,
            16,
            false,
            &mut [0; 64 * 135],
        );
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(dest[y * 4 + x], src[origin + y * 16 + x]);
            }
        }
    }
}
