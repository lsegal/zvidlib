//! Pixel kernels the encoder shares with every VP9 decoder: 4x4 transforms,
//! intra prediction and 8-tap motion compensation.
//!
//! The inverse transform, the predictors and the interpolation must match the
//! decoding process bit for bit, because the encoder predicts later blocks and
//! frames from its own reconstruction. The forward transform only has to be a
//! good approximation of the inverse; it follows libvpx's `vp9_fht4x4_c`.

use super::tables::SUBPEL_FILTERS_REGULAR;

const COSPI_8_64: i64 = 15137;
const COSPI_16_64: i64 = 11585;
const COSPI_24_64: i64 = 6270;
const SINPI_1_9: i64 = 5283;
const SINPI_2_9: i64 = 9929;
const SINPI_3_9: i64 = 13377;
const SINPI_4_9: i64 = 15212;

/// The 4x4 transform types; the first named 1-D transform is the vertical one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TxType {
    DctDct,
    AdstDct,
    DctAdst,
    AdstAdst,
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

fn idct4(input: [i32; 4]) -> [i32; 4] {
    let [x0, x1, x2, x3] = input.map(|value| i64::from(value as i16));
    let step0 = round_shift((x0 + x2) * COSPI_16_64) as i16;
    let step1 = round_shift((x0 - x2) * COSPI_16_64) as i16;
    let step2 = round_shift(x1 * COSPI_24_64 - x3 * COSPI_8_64) as i16;
    let step3 = round_shift(x1 * COSPI_8_64 + x3 * COSPI_24_64) as i16;
    let [s0, s1, s2, s3] = [step0, step1, step2, step3].map(i32::from);
    [s0 + s3, s1 + s2, s1 - s2, s0 - s3]
}

fn iadst4(input: [i32; 4]) -> [i32; 4] {
    let [x0, x1, x2, x3] = input.map(i64::from);
    if x0 | x1 | x2 | x3 == 0 {
        return [0; 4];
    }
    let s0 = SINPI_1_9 * x0 + SINPI_4_9 * x2 + SINPI_2_9 * x3;
    let s1 = SINPI_2_9 * x0 - SINPI_1_9 * x2 - SINPI_4_9 * x3;
    let s2 = SINPI_3_9 * (x0 - x2 + x3);
    let s3 = SINPI_3_9 * x1;
    [
        round_shift(s0 + s3) as i32,
        round_shift(s1 + s3) as i32,
        round_shift(s2) as i32,
        round_shift(s0 + s1 - s3) as i32,
    ]
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
pub(super) fn forward_transform(residual: &[i32; 16], tx_type: TxType) -> [i32; 16] {
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

/// Adds the inverse transform of dequantized `coefficients` to a 4x4 block of
/// `pixels` (row stride `stride`), exactly as `vp9_iht4x4_16_add_c` does.
pub(super) fn inverse_transform_add(
    coefficients: &[i32; 16],
    tx_type: TxType,
    pixels: &mut [u8],
    stride: usize,
) {
    let mut rows = [0_i32; 16];
    for row in 0..4 {
        let mut input = [0_i32; 4];
        input.copy_from_slice(&coefficients[row * 4..row * 4 + 4]);
        let output = if tx_type.horizontal_adst() {
            iadst4(input)
        } else {
            idct4(input)
        };
        rows[row * 4..row * 4 + 4].copy_from_slice(&output);
    }
    for column in 0..4 {
        let input = [0, 1, 2, 3].map(|row| rows[row * 4 + column]);
        let output = if tx_type.vertical_adst() {
            iadst4(input)
        } else {
            idct4(input)
        };
        for (row, value) in output.into_iter().enumerate() {
            let pixel = &mut pixels[row * stride + column];
            *pixel = (i32::from(*pixel) + ((value + 8) >> 4)).clamp(0, 255) as u8;
        }
    }
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

    /// The luma transform type VP9 pairs with this mode
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

/// Predicts a 4x4 block in place from the reconstructed pixels around it.
///
/// `x`, `y` locate the block in a plane of row stride `stride`. The block's
/// neighbours always lie inside the 8-aligned decoded area, so none of the
/// frame-edge extension of `build_intra_predictors` applies.
pub(super) fn predict_intra_4x4(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    mode: IntraMode,
    have_above: bool,
    have_left: bool,
) {
    let mut above = [127_u8; 4];
    let mut above_left = 127_u8;
    let mut left = [129_u8; 4];
    if have_above {
        let start = (y - 1) * stride + x;
        above.copy_from_slice(&plane[start..start + 4]);
        above_left = if have_left { plane[start - 1] } else { 129 };
    }
    if have_left {
        for (row, value) in left.iter_mut().enumerate() {
            *value = plane[(y + row) * stride + x - 1];
        }
    }
    let mut block = [0_u8; 16];
    match mode {
        IntraMode::Dc => {
            let above_sum: u32 = above.iter().map(|&value| u32::from(value)).sum();
            let left_sum: u32 = left.iter().map(|&value| u32::from(value)).sum();
            let dc = match (have_above, have_left) {
                (true, true) => (above_sum + left_sum + 4) >> 3,
                (true, false) => (above_sum + 2) >> 2,
                (false, true) => (left_sum + 2) >> 2,
                (false, false) => 128,
            } as u8;
            block = [dc; 16];
        }
        IntraMode::V => {
            for row in 0..4 {
                block[row * 4..row * 4 + 4].copy_from_slice(&above);
            }
        }
        IntraMode::H => {
            for row in 0..4 {
                block[row * 4..row * 4 + 4].fill(left[row]);
            }
        }
        IntraMode::Tm => {
            for row in 0..4 {
                for column in 0..4 {
                    block[row * 4 + column] = (i32::from(left[row]) + i32::from(above[column])
                        - i32::from(above_left))
                    .clamp(0, 255) as u8;
                }
            }
        }
    }
    for row in 0..4 {
        let start = (y + row) * stride + x;
        plane[start..start + 4].copy_from_slice(&block[row * 4..row * 4 + 4]);
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

    #[test]
    fn dc_only_inverse_matches_the_libvpx_shortcut() {
        // `vpx_idct4x4_1_add_c` is what libvpx runs for an end-of-block of 1;
        // the full transform must agree with it for every DC value.
        for dc in [-4000, -517, -1, 1, 64, 333, 4095] {
            let mut coefficients = [0; 16];
            coefficients[0] = dc;
            let mut pixels = [128_u8; 16];
            inverse_transform_add(&coefficients, TxType::DctDct, &mut pixels, 4);
            let out = round_shift(i64::from(dc) * COSPI_16_64);
            let out = round_shift(out * COSPI_16_64);
            let expected = (128 + ((out + 8) >> 4)).clamp(0, 255) as u8;
            assert_eq!(pixels, [expected; 16], "dc {dc}");
        }
    }

    #[test]
    fn forward_then_inverse_reconstructs_each_transform_type() {
        let residual: [i32; 16] = core::array::from_fn(|index| (index as i32 * 37 % 61) - 30);
        for tx_type in [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
        ] {
            let coefficients = forward_transform(&residual, tx_type);
            let mut pixels = [128_u8; 16];
            inverse_transform_add(&coefficients, tx_type, &mut pixels, 4);
            for (index, &pixel) in pixels.iter().enumerate() {
                let error = (i32::from(pixel) - 128 - residual[index]).abs();
                assert!(error <= 1, "{tx_type:?} sample {index} off by {error}");
            }
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
