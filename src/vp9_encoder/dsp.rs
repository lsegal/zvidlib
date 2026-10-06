//! Pixel kernels the encoder shares with every VP9 decoder: transforms, intra
//! prediction and 8-tap motion compensation.
//!
//! The inverse transforms, the predictors and the interpolation must match
//! the decoding process bit for bit, because the encoder predicts later blocks
//! and frames from its own reconstruction; the inverse transforms are the
//! native decoder's own. The forward transforms only have to be a good
//! approximation of the inverse: the 4x4 one follows libvpx's `vp9_fht4x4_c`,
//! and the larger ones are exact floating-point transforms.

use super::tables::SUBPEL_FILTERS_REGULAR;
use std::sync::OnceLock;

const COSPI_8_64: i64 = 15137;
const COSPI_16_64: i64 = 11585;
const COSPI_24_64: i64 = 6270;
const SINPI_1_9: i64 = 5283;
const SINPI_2_9: i64 = 9929;
const SINPI_3_9: i64 = 13377;
const SINPI_4_9: i64 = 15212;

/// The transform types; the first named 1-D transform is the vertical one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TxType {
    DctDct = 0,
    AdstDct = 1,
    DctAdst = 2,
    AdstAdst = 3,
}

impl TxType {
    fn vertical_adst(self) -> bool {
        matches!(self, Self::AdstDct | Self::AdstAdst)
    }

    fn horizontal_adst(self) -> bool {
        matches!(self, Self::DctAdst | Self::AdstAdst)
    }
}

fn round_shift(value: i64) -> i64 {
    (value + (1 << 13)) >> 14
}

fn fdct4(input: [i64; 4]) -> [i64; 4] {
    let step0 = input[0] + input[3];
    let step1 = input[1] + input[2];
    let step2 = input[1] - input[2];
    let step3 = input[0] - input[3];
    [
        round_shift((step0 + step1) * COSPI_16_64),
        round_shift(step2 * COSPI_24_64 + step3 * COSPI_8_64),
        round_shift((step0 - step1) * COSPI_16_64),
        round_shift(-step2 * COSPI_8_64 + step3 * COSPI_24_64),
    ]
}

fn fadst4(input: [i64; 4]) -> [i64; 4] {
    let [x0, x1, x2, x3] = input;
    if x0 | x1 | x2 | x3 == 0 {
        return [0; 4];
    }
    let s0 = SINPI_1_9 * x0;
    let s1 = SINPI_4_9 * x0;
    let s2 = SINPI_2_9 * x1;
    let s3 = SINPI_1_9 * x1;
    let s4 = SINPI_3_9 * x2;
    let s5 = SINPI_4_9 * x3;
    let s6 = SINPI_2_9 * x3;
    let s7 = x0 + x1 - x3;
    let x0 = s0 + s2 + s5;
    let x1 = SINPI_3_9 * s7;
    let x2 = s1 - s3 + s6;
    let x3 = s4;
    [
        round_shift(x0 + x3),
        round_shift(x1),
        round_shift(x2 - x3),
        round_shift(x2 - x0 + x3),
    ]
}

/// Forward 4x4 transform of a residual block, in raster order.
fn forward_transform_4x4(residual: &[i32], tx_type: TxType) -> [i32; 16] {
    let mut columns = [0_i64; 16];
    for column in 0..4 {
        let mut input = [0_i64; 4];
        for (row, value) in input.iter_mut().enumerate() {
            *value = i64::from(residual[row * 4 + column]) * 16;
        }
        if column == 0 && input[0] != 0 {
            input[0] += 1;
        }
        let output = if tx_type.vertical_adst() {
            fadst4(input)
        } else {
            fdct4(input)
        };
        for (row, value) in output.into_iter().enumerate() {
            columns[row * 4 + column] = value;
        }
    }
    let mut coefficients = [0_i32; 16];
    for row in 0..4 {
        let mut input = [0_i64; 4];
        input.copy_from_slice(&columns[row * 4..row * 4 + 4]);
        let output = if tx_type.horizontal_adst() {
            fadst4(input)
        } else {
            fdct4(input)
        };
        for (column, value) in output.into_iter().enumerate() {
            coefficients[row * 4 + column] = ((value + 1) >> 2) as i32;
        }
    }
    coefficients
}

/// The 1-D inverse transforms of the 8x8, 16x16 and 32x32 sizes as `n` x `n`
/// matrices: an inverse maps coefficients `x` to samples
/// `y[i] = sum(basis[i * n + k] * x[k])`. Each is `sqrt(n / 2)` times
/// orthonormal.
struct Bases {
    dct: [Vec<f64>; 3],
    adst: [Vec<f64>; 2],
}

fn bases() -> &'static Bases {
    static BASES: OnceLock<Bases> = OnceLock::new();
    BASES.get_or_init(|| {
        use std::f64::consts::{FRAC_1_SQRT_2, PI};
        let dct = |n: usize| -> Vec<f64> {
            (0..n * n)
                .map(|index| {
                    let (i, k) = (index / n, index % n);
                    let scale = if k == 0 { FRAC_1_SQRT_2 } else { 1.0 };
                    scale * (PI * ((2 * i + 1) * k) as f64 / (2 * n) as f64).cos()
                })
                .collect()
        };
        let adst = |n: usize| -> Vec<f64> {
            (0..n * n)
                .map(|index| {
                    let (i, k) = (index / n, index % n);
                    (PI * ((2 * i + 1) * (2 * k + 1)) as f64 / (4 * n) as f64).sin()
                })
                .collect()
        };
        Bases {
            dct: [dct(8), dct(16), dct(32)],
            adst: [adst(8), adst(16)],
        }
    })
}

/// Forward transform of an `n` x `n` residual block (`n = 4 << tx_size`), in
/// raster order, into `output`.
///
/// The coefficients are on the scale the inverse transform of that size
/// expects, so that dequantized levels reconstruct the residual: eight times
/// orthonormal, or four times for 32x32, whose dequantization halves the
/// level. 32x32 blocks are always DCT_DCT.
pub(super) fn forward_transform(
    residual: &[i32],
    tx_size: usize,
    tx_type: TxType,
    output: &mut [f64],
) {
    if tx_size == 0 {
        let coefficients = forward_transform_4x4(residual, tx_type);
        for (output, coefficient) in output.iter_mut().zip(coefficients) {
            *output = f64::from(coefficient);
        }
        return;
    }
    let n = 4 << tx_size;
    let bases = bases();
    let basis = |adst: bool| {
        if adst && tx_size < 3 {
            &bases.adst[tx_size - 1]
        } else {
            &bases.dct[tx_size - 1]
        }
    };
    let vertical = basis(tx_type.vertical_adst());
    let horizontal = basis(tx_type.horizontal_adst());
    // The inverse is `vertical * coefficients * horizontal^T`, shifted right
    // by 5 bits for 8x8 and 6 bits above, so the forward transform is
    // `vertical^T * residual * horizontal` scaled by that shift over (n / 2)^2.
    let shift = if tx_size == 1 { 32.0 } else { 64.0 };
    let half = (n / 2) as f64;
    let scale = shift / (half * half);
    // Both products accumulate whole rows, which the compiler vectorizes.
    let mut columns = vec![0.0; n * n];
    for (i, samples) in residual.chunks_exact(n).enumerate() {
        let samples: Vec<f64> = samples.iter().map(|&value| f64::from(value)).collect();
        for k in 0..n {
            let weight = vertical[i * n + k] * scale;
            for (column, &sample) in columns[k * n..][..n].iter_mut().zip(&samples) {
                *column += weight * sample;
            }
        }
    }
    output[..n * n].fill(0.0);
    for (k, row) in output.chunks_exact_mut(n).take(n).enumerate() {
        for j in 0..n {
            let weight = columns[k * n + j];
            for (output, &basis) in row.iter_mut().zip(&horizontal[j * n..][..n]) {
                *output += weight * basis;
            }
        }
    }
}

/// Adds the inverse transform of dequantized `coefficients` (raster order,
/// with `eob` the end of block in scan order) to an `n` x `n` block of
/// `pixels` with row stride `stride`, exactly as the decoder does.
pub(super) fn inverse_transform_add(
    coefficients: &[i32],
    tx_size: usize,
    tx_type: TxType,
    eob: usize,
    pixels: &mut [u8],
    stride: usize,
) {
    crate::vp9_dec::recon::inverse_transform_add(
        coefficients,
        pixels,
        stride,
        tx_size as u8,
        tx_type as u8,
        eob,
        false,
    );
}

/// The intra modes this encoder chooses from, with their VP9 mode numbers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum IntraMode {
    Dc = 0,
    V = 1,
    H = 2,
    Tm = 9,
}

impl IntraMode {
    pub(super) const ALL: [Self; 4] = [Self::Dc, Self::V, Self::H, Self::Tm];

    /// The luma transform type VP9 pairs with this mode below 32x32
    /// (`intra_mode_to_tx_type_lookup`).
    pub(super) fn tx_type(self) -> TxType {
        match self {
            Self::Dc => TxType::DctDct,
            Self::V => TxType::AdstDct,
            Self::H => TxType::DctAdst,
            Self::Tm => TxType::AdstAdst,
        }
    }
}

/// Predicts a `size` x `size` block in place from the reconstructed pixels
/// around it.
///
/// `x`, `y` locate the block in a plane of row stride `stride`. The encoder
/// only codes blocks that lie wholly inside the 8-aligned decoded area, and
/// none of these modes reads the samples above and to the right, so none of
/// the frame-edge extension of `build_intra_predictors` applies.
#[allow(clippy::too_many_arguments)]
pub(super) fn predict_intra(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    size: usize,
    mode: IntraMode,
    have_above: bool,
    have_left: bool,
) {
    let mut above = [127_u8; 32];
    let mut above_left = 127_u8;
    let mut left = [129_u8; 32];
    let (above, left) = (&mut above[..size], &mut left[..size]);
    if have_above {
        let start = (y - 1) * stride + x;
        above.copy_from_slice(&plane[start..start + size]);
        above_left = if have_left { plane[start - 1] } else { 129 };
    }
    if have_left {
        for (row, value) in left.iter_mut().enumerate() {
            *value = plane[(y + row) * stride + x - 1];
        }
    }
    let log2 = size.trailing_zeros();
    for row in 0..size {
        let output = &mut plane[(y + row) * stride + x..][..size];
        match mode {
            IntraMode::Dc => {
                let above_sum: u32 = above.iter().map(|&value| u32::from(value)).sum();
                let left_sum: u32 = left.iter().map(|&value| u32::from(value)).sum();
                let half = (size / 2) as u32;
                let dc = match (have_above, have_left) {
                    (true, true) => (above_sum + left_sum + size as u32) >> (log2 + 1),
                    (true, false) => (above_sum + half) >> log2,
                    (false, true) => (left_sum + half) >> log2,
                    (false, false) => 128,
                } as u8;
                output.fill(dc);
            }
            IntraMode::V => output.copy_from_slice(above),
            IntraMode::H => output.fill(left[row]),
            IntraMode::Tm => {
                for (column, value) in output.iter_mut().enumerate() {
                    *value = (i32::from(left[row]) + i32::from(above[column])
                        - i32::from(above_left))
                    .clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// A reference plane read with VP9's edge clamping: samples outside the
/// visible `width` x `height` repeat the nearest edge sample.
pub(super) struct ReferencePlane<'a> {
    pub(super) pixels: &'a [u8],
    pub(super) stride: usize,
    pub(super) width: usize,
    pub(super) height: usize,
}

impl ReferencePlane<'_> {
    fn sample(&self, x: isize, y: isize) -> u8 {
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.pixels[y * self.stride + x]
    }
}

/// Motion-compensates a `size` x `size` block whose top-left sample is at `(x, y)`
/// with a motion vector in sixteenth-sample units of this plane, using the
/// regular 8-tap filter the frame header selects.
///
/// Like libvpx's `vpx_convolve8_c`, the horizontal pass is rounded and clipped
/// to 8 bits before the vertical pass, and a whole-sample component filters
/// with the identity kernel.
pub(super) fn predict_inter(
    reference: &ReferencePlane<'_>,
    x: usize,
    y: usize,
    size: usize,
    mv_row_q4: i32,
    mv_col_q4: i32,
    output: &mut [u8],
) {
    let x0 = x as isize + (mv_col_q4 >> 4) as isize;
    let y0 = y as isize + (mv_row_q4 >> 4) as isize;
    let filter_x = &SUBPEL_FILTERS_REGULAR[(mv_col_q4 & 15) as usize * 8..][..8];
    let filter_y = &SUBPEL_FILTERS_REGULAR[(mv_row_q4 & 15) as usize * 8..][..8];
    let (w, h) = (size, size);
    let mut intermediate = vec![0_u8; (h + 7) * w];
    for row in 0..h + 7 {
        let source_y = y0 + row as isize - 3;
        for column in 0..w {
            let source_x = x0 + column as isize - 3;
            let sum: i32 = (0..8)
                .map(|tap| {
                    filter_x[tap] * i32::from(reference.sample(source_x + tap as isize, source_y))
                })
                .sum();
            intermediate[row * w + column] = ((sum + 64) >> 7).clamp(0, 255) as u8;
        }
    }
    for row in 0..h {
        for column in 0..w {
            let sum: i32 = (0..8)
                .map(|tap| filter_y[tap] * i32::from(intermediate[(row + tap) * w + column]))
                .sum();
            output[row * w + column] = ((sum + 64) >> 7).clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vp9_dec::idct1d;

    const TX_TYPES: [TxType; 4] = [
        TxType::DctDct,
        TxType::AdstDct,
        TxType::DctAdst,
        TxType::AdstAdst,
    ];

    #[test]
    fn forward_bases_are_the_decoders_inverse_transforms() {
        // Feeding each 1-D inverse a scaled impulse reads one basis column back.
        type Inverse = fn(&[i32], &mut [i32]);
        let bases = bases();
        let cases: [(Inverse, &Vec<f64>, usize); 5] = [
            (idct1d::idct8, &bases.dct[0], 8),
            (idct1d::idct16, &bases.dct[1], 16),
            (idct1d::idct32, &bases.dct[2], 32),
            (idct1d::iadst8, &bases.adst[0], 8),
            (idct1d::iadst16, &bases.adst[1], 16),
        ];
        const SCALE: f64 = 4096.0;
        for (inverse, basis, n) in cases {
            for k in 0..n {
                let mut input = vec![0; n];
                input[k] = SCALE as i32;
                let mut output = vec![0; n];
                inverse(&input, &mut output);
                for i in 0..n {
                    let expected = basis[i * n + k] * SCALE;
                    let error = (f64::from(output[i]) - expected).abs();
                    assert!(
                        error <= 3.0,
                        "{n}-point basis {k} sample {i} off by {error}"
                    );
                }
            }
        }
    }

    #[test]
    fn forward_then_inverse_reconstructs_every_size_and_type() {
        for tx_size in 0..4 {
            let n = 4 << tx_size;
            let residual: Vec<i32> = (0..n * n)
                .map(|index| (index as i32 * 37 % 61) - 30)
                .collect();
            for tx_type in TX_TYPES {
                let tx_type = if tx_size == 3 {
                    TxType::DctDct
                } else {
                    tx_type
                };
                let mut coefficients = vec![0.0; n * n];
                forward_transform(&residual, tx_size, tx_type, &mut coefficients);
                let rounded: Vec<i32> = coefficients
                    .iter()
                    .map(|&value| value.round() as i32)
                    .collect();
                let mut pixels = vec![128_u8; n * n];
                inverse_transform_add(&rounded, tx_size, tx_type, n * n, &mut pixels, n);
                for (index, &pixel) in pixels.iter().enumerate() {
                    let error = (i32::from(pixel) - 128 - residual[index]).abs();
                    assert!(
                        error <= 1,
                        "{n}x{n} {tx_type:?} sample {index} off by {error}"
                    );
                }
            }
        }
    }

    #[test]
    fn dc_only_coefficients_reconstruct_a_flat_block() {
        for tx_size in 0..4 {
            let n = 4 << tx_size;
            for value in [-60, -1, 1, 17, 90] {
                let residual = vec![value; n * n];
                let mut coefficients = vec![0.0; n * n];
                forward_transform(&residual, tx_size, TxType::DctDct, &mut coefficients);
                let mut dc = vec![0; n * n];
                dc[0] = coefficients[0].round() as i32;
                assert!(
                    coefficients[1..].iter().all(|value| value.abs() < 1.0),
                    "{n}x{n} DC {value} leaks into AC coefficients"
                );
                let mut pixels = vec![128_u8; n * n];
                inverse_transform_add(&dc, tx_size, TxType::DctDct, 1, &mut pixels, n);
                let expected = 128 + value;
                assert!(
                    pixels
                        .iter()
                        .all(|&pixel| (i32::from(pixel) - expected).abs() <= 1),
                    "{n}x{n} DC {value}"
                );
            }
        }
    }

    #[test]
    fn intra_predictors_scale_to_every_transform_size() {
        let stride = 40;
        let mut plane: Vec<u8> = (0..stride * 40)
            .map(|index| (index * 7 % 251) as u8)
            .collect();
        for size in [4, 8, 16, 32] {
            let (x, y) = (4, 4);
            predict_intra(&mut plane, stride, x, y, size, IntraMode::V, true, true);
            for row in 0..size {
                assert_eq!(
                    plane[(y + row) * stride + x..][..size],
                    plane[(y - 1) * stride + x..][..size].to_vec()[..]
                );
            }
            predict_intra(&mut plane, stride, x, y, size, IntraMode::Dc, false, false);
            assert!(
                plane[y * stride + x..][..size]
                    .iter()
                    .all(|&value| value == 128)
            );
            let mut copy = plane.clone();
            predict_intra(&mut copy, stride, x, y, size, IntraMode::Dc, true, false);
            let sum: u32 = plane[(y - 1) * stride + x..][..size]
                .iter()
                .map(|&value| u32::from(value))
                .sum();
            let expected = ((sum + size as u32 / 2) / size as u32) as u8;
            assert_eq!(copy[(y + size - 1) * stride + x + size - 1], expected);
        }
    }

    #[test]
    fn whole_sample_motion_copies_and_clamps_at_the_edges() {
        let pixels: Vec<u8> = (0..64).map(|value| value as u8 * 3).collect();
        let reference = ReferencePlane {
            pixels: &pixels,
            stride: 8,
            width: 8,
            height: 8,
        };
        let mut output = [0_u8; 16];
        predict_inter(&reference, 4, 4, 4, -16 * 6, 16 * 2, &mut output);
        // Rows -2..=1 read rows 0, 0, 0, 1 and columns 6..=9 read 6, 7, 7, 7.
        assert_eq!(&output[..4], &[18, 21, 21, 21]);
        assert_eq!(&output[8..12], &[18, 21, 21, 21]);
        assert_eq!(&output[12..], &[42, 45, 45, 45]);
    }
}
