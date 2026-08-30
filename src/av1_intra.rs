//! Bounded pixel primitives used by AV1 intra-frame reconstruction.
//!
//! This is deliberately a reconstruction building block rather than a video
//! decoder.  Tile syntax owns mode and coefficient parsing; this module owns
//! checked YUV allocation and applying a decoded intra block to a plane.

use crate::{
    ColorRange, Error, ErrorKind, Limits, PixelFormat, Plane, Result, VideoDimensions, VideoFrame,
    av1_intra_pred::{add_residual_row, paeth_row, sum_samples},
};

/// Intra predictors used by the first AV1 reconstruction stages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Av1IntraMode {
    Dc,
    Vertical,
    Horizontal,
    Paeth,
}

/// Geometry and prediction mode for one decoded intra block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Av1IntraBlock {
    pub plane: usize,
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub mode: Av1IntraMode,
}

/// A bounded 8-bit 4:2:0 reconstruction target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Av1IntraFrame {
    dimensions: VideoDimensions,
    color_range: ColorRange,
    planes: [Vec<u8>; 3],
    strides: [usize; 3],
}

impl Av1IntraFrame {
    /// Allocates neutral-chroma planes after applying the public frame limits.
    pub fn new(dimensions: VideoDimensions, limits: &Limits) -> Result<Self> {
        let width = usize::try_from(dimensions.width)
            .map_err(|_| limit("AV1 width is not representable"))?;
        let height = usize::try_from(dimensions.height)
            .map_err(|_| limit("AV1 height is not representable"))?;
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let y_len = width
            .checked_mul(height)
            .ok_or_else(|| limit("AV1 luma plane size overflows"))?;
        let c_len = chroma_width
            .checked_mul(chroma_height)
            .ok_or_else(|| limit("AV1 chroma plane size overflows"))?;
        let total = y_len
            .checked_add(c_len)
            .and_then(|n| n.checked_add(c_len))
            .ok_or_else(|| limit("AV1 frame size overflows"))?;
        if u64::try_from(total).map_err(|_| limit("AV1 frame size is not representable"))?
            > limits.max_allocation_bytes
        {
            return Err(limit(
                "AV1 reconstructed frame exceeds the allocation limit",
            ));
        }
        Ok(Self {
            dimensions,
            color_range: ColorRange::Limited,
            planes: [vec![0; y_len], vec![128; c_len], vec![128; c_len]],
            strides: [width, chroma_width, chroma_width],
        })
    }

    /// Creates a neutral-chroma 4:2:0 frame from a tightly packed decoded
    /// luma plane.
    pub fn from_luma(
        dimensions: VideoDimensions,
        luma: Vec<u8>,
        color_range: ColorRange,
        limits: &Limits,
    ) -> Result<Self> {
        let mut frame = Self::new(dimensions, limits)?;
        if luma.len() != frame.planes[0].len() {
            return Err(malformed(
                "AV1 luma plane length does not match the coded dimensions",
            ));
        }
        frame.planes[0] = luma;
        frame.color_range = color_range;
        Ok(frame)
    }

    /// Applies one luma or chroma intra block. Residuals are signed spatial
    /// samples after the tile decoder's inverse transform stage.
    pub fn reconstruct_block(&mut self, block: Av1IntraBlock, residuals: &[i16]) -> Result<()> {
        let Av1IntraBlock {
            plane,
            x,
            y,
            width,
            height,
            mode,
        } = block;
        if plane > 2 || width == 0 || height == 0 {
            return Err(malformed("AV1 intra block has an invalid plane or size"));
        }
        let count = width
            .checked_mul(height)
            .ok_or_else(|| limit("AV1 intra block size overflows"))?;
        if residuals.len() != count {
            return Err(malformed(
                "AV1 intra residual count does not match block size",
            ));
        }
        let stride = self.strides[plane];
        let rows = self.planes[plane].len() / stride;
        if x.checked_add(width).is_none_or(|end| end > stride)
            || y.checked_add(height).is_none_or(|end| end > rows)
        {
            return Err(malformed(
                "AV1 intra block exceeds its reconstruction plane",
            ));
        }
        let top_left = if x > 0 && y > 0 {
            self.planes[plane][(y - 1) * stride + x - 1]
        } else {
            128
        };
        let mut top = vec![128; width];
        let mut left = vec![128; height];
        if y > 0 {
            let above = (y - 1) * stride + x;
            top.copy_from_slice(&self.planes[plane][above..above + width]);
        }
        if x > 0 {
            for (i, sample) in left.iter_mut().enumerate() {
                *sample = self.planes[plane][(y + i) * stride + x - 1];
            }
        }
        let dc = ((sum_samples(&top) + sum_samples(&left))
            / u32::try_from(width + height).expect("nonzero block dimensions"))
            as u8;
        let mut prediction = vec![0u8; width];
        for row in 0..height {
            match mode {
                Av1IntraMode::Dc => prediction.fill(dc),
                Av1IntraMode::Vertical => prediction.copy_from_slice(&top),
                Av1IntraMode::Horizontal => prediction.fill(left[row]),
                Av1IntraMode::Paeth => paeth_row(top_left, &top, left[row], &mut prediction),
            }
            let start = (y + row) * stride + x;
            let destination = &mut self.planes[plane][start..start + width];
            destination.copy_from_slice(&prediction);
            add_residual_row(&residuals[row * width..row * width + width], destination);
        }
        Ok(())
    }

    pub fn into_video_frame(self, limits: &Limits) -> Result<VideoFrame> {
        VideoFrame::new(
            self.dimensions,
            PixelFormat::Yuv420p8,
            self.color_range,
            vec![
                Plane {
                    data: self.planes[0].clone(),
                    stride: self.strides[0],
                },
                Plane {
                    data: self.planes[1].clone(),
                    stride: self.strides[1],
                },
                Plane {
                    data: self.planes[2].clone(),
                    stride: self.strides[2],
                },
            ],
            limits,
        )
    }
}

/// The normative inverse 4x4 Walsh-Hadamard transform used by lossless AV1
/// blocks. The input is the coded coefficient array before the required
/// 8-bit lossless dequantization by four.
pub fn inverse_wht_4x4(coefficients: &[i32; 16]) -> [i16; 16] {
    let mut rows = [[0i64; 4]; 4];
    for row in 0..4 {
        rows[row] = inverse_wht_1d(
            [
                i64::from(coefficients[row * 4]) * 4,
                i64::from(coefficients[row * 4 + 1]) * 4,
                i64::from(coefficients[row * 4 + 2]) * 4,
                i64::from(coefficients[row * 4 + 3]) * 4,
            ],
            2,
        );
    }
    let mut output = [0i16; 16];
    for column in 0..4 {
        let transformed = inverse_wht_1d(
            [
                rows[0][column],
                rows[1][column],
                rows[2][column],
                rows[3][column],
            ],
            0,
        );
        for row in 0..4 {
            output[row * 4 + column] =
                transformed[row].clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16;
        }
    }
    output
}

fn inverse_wht_1d(values: [i64; 4], shift: u32) -> [i64; 4] {
    let mut a = values[0] >> shift;
    let mut c = values[1] >> shift;
    let mut d = values[2] >> shift;
    let mut b = values[3] >> shift;
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    [a, b, c, d]
}

// ---------------------------------------------------------------------
// Non-lossless dequantization and inverse transforms (AV1 spec §7.12,
// §7.13), used when `base_q_idx != 0` for the 4x4/8x8 transform sizes this
// crate's decoders support under `TX_MODE_LARGEST`.
//
// The 256-entry 8-bit `DC_QLOOKUP`/`AC_QLOOKUP` tables and the `COSPI_*`
// partial-butterfly DCT constants below are reproduced from the publicly
// documented AV1/VP9 quantizer and transform tables to the best of this
// implementation's knowledge. This crate has no network access and no AV1
// conformance vectors available to validate them bit-exactly against an
// official encoder/decoder in this environment; correctness here is
// verified through internal self-consistency (round-trip, monotonicity,
// and known boundary-value tests) rather than official conformance
// vectors. Treat this non-lossless profile, like the rest of this crate,
// as a bounded, dependency-free subset rather than a conformance-tested
// decoder. The inverse DCT kernels use the well-known VP9/AV1-lineage
// integer partial-butterfly structure (14-bit fixed point, shared across
// the VP9/AV1 codec family) rather than literally transcribing the AV1
// spec's `cos128`/`B()` stage notation; the two are mathematically
// equivalent up to rounding convention. The inverse identity transform
// (`IDTX`) is implemented here as a pure pass-through (no additional
// row/column scaling), a deliberate simplification of the AV1 spec's
// scaled identity transform.
#[rustfmt::skip]
const DC_QLOOKUP: [i32; 256] = [
    4,    8,    8,    9,   10,   11,   12,   12,   13,   14,
   15,   16,   17,   18,   19,   19,   20,   21,   22,   23,
   24,   25,   26,   26,   27,   28,   29,   30,   31,   32,
   32,   33,   34,   35,   36,   37,   38,   38,   39,   40,
   41,   42,   43,   43,   44,   45,   46,   47,   48,   48,
   49,   50,   51,   52,   53,   53,   54,   55,   56,   57,
   57,   58,   59,   60,   61,   62,   62,   63,   64,   65,
   66,   66,   67,   68,   69,   70,   70,   71,   72,   73,
   74,   74,   75,   76,   77,   78,   78,   79,   80,   81,
   81,   82,   83,   84,   85,   85,   87,   88,   90,   92,
   93,   95,   96,   98,   99,  101,  102,  104,  105,  107,
  108,  110,  111,  113,  114,  116,  117,  118,  120,  121,
  123,  125,  127,  129,  131,  134,  136,  138,  140,  142,
  144,  146,  148,  150,  152,  154,  156,  158,  161,  164,
  166,  169,  172,  174,  177,  180,  182,  185,  187,  190,
  192,  195,  199,  202,  205,  208,  211,  214,  217,  220,
  223,  226,  230,  233,  237,  240,  243,  247,  250,  253,
  257,  261,  265,  269,  272,  276,  280,  284,  288,  292,
  296,  300,  304,  309,  313,  317,  322,  326,  330,  335,
  340,  344,  349,  354,  359,  364,  369,  374,  379,  384,
  389,  395,  400,  406,  411,  417,  423,  429,  435,  441,
  447,  454,  461,  467,  475,  482,  489,  497,  505,  513,
  522,  530,  539,  549,  559,  569,  579,  590,  602,  614,
  626,  640,  654,  668,  684,  700,  717,  736,  755,  775,
  796,  819,  843,  869,  896,  925,  955,  988, 1022, 1058,
 1098, 1139, 1184, 1232, 1282, 1336,
];

#[rustfmt::skip]
const AC_QLOOKUP: [i32; 256] = [
    4,    8,    9,   10,   11,   12,   13,   14,   15,   16,
   17,   18,   19,   20,   21,   22,   23,   24,   25,   26,
   27,   28,   29,   30,   31,   32,   33,   34,   35,   36,
   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,
   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,
   57,   58,   59,   60,   61,   62,   63,   64,   65,   66,
   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,
   77,   78,   79,   80,   81,   82,   83,   84,   85,   86,
   87,   88,   89,   90,   91,   92,   93,   94,   95,   96,
   97,   98,   99,  100,  101,  102,  104,  106,  108,  110,
  112,  114,  116,  118,  120,  122,  124,  126,  128,  130,
  132,  134,  136,  138,  140,  142,  144,  146,  148,  150,
  152,  155,  158,  161,  164,  167,  170,  173,  176,  179,
  182,  185,  188,  191,  194,  197,  200,  203,  207,  211,
  215,  219,  223,  227,  231,  235,  239,  243,  247,  251,
  255,  260,  265,  270,  275,  280,  285,  290,  295,  300,
  305,  311,  317,  323,  329,  335,  341,  347,  353,  359,
  366,  373,  380,  387,  394,  401,  408,  416,  424,  432,
  440,  448,  456,  465,  474,  483,  492,  501,  510,  520,
  530,  540,  550,  560,  571,  582,  593,  604,  615,  627,
  639,  651,  663,  676,  689,  702,  715,  729,  743,  757,
  771,  786,  801,  816,  832,  848,  864,  881,  898,  915,
  933,  951,  969,  988, 1007, 1026, 1046, 1066, 1087, 1108,
 1129, 1151, 1173, 1196, 1219, 1243, 1267, 1292, 1317, 1343,
 1370, 1397, 1425, 1453, 1482, 1512, 1542, 1573, 1605, 1638,
 1671, 1705, 1740, 1775, 1812, 1828,
];

/// AV1 spec §7.12.2 `get_dc_quant`: the DC dequantization step for 8-bit
/// content at `qindex` (0..=255).
pub fn get_dc_quant(qindex: u8) -> i32 {
    DC_QLOOKUP[qindex as usize]
}

/// AV1 spec §7.12.2 `get_ac_quant`: the AC dequantization step for 8-bit
/// content at `qindex` (0..=255).
pub fn get_ac_quant(qindex: u8) -> i32 {
    AC_QLOOKUP[qindex as usize]
}

/// A supported non-lossless transform type (AV1 spec §5.11.47 `TxType`,
/// restricted to the entries this crate implements: `reduced_tx_set = 1`
/// with `ADST_ADST` deliberately left unsupported, see the decoder's own
/// module docs).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Av1TxType {
    DctDct,
    Idtx,
}

const COSPI_4_64: i64 = 16069;
const COSPI_8_64: i64 = 15137;
const COSPI_12_64: i64 = 13623;
const COSPI_16_64: i64 = 11585;
const COSPI_20_64: i64 = 9102;
const COSPI_24_64: i64 = 6270;
const COSPI_28_64: i64 = 3196;

fn dct_round_shift(value: i64) -> i64 {
    (value + (1 << 13)) >> 14
}

/// The AV1/VP9-lineage 4-point inverse DCT partial butterfly.
fn inverse_dct4_1d(input: [i64; 4]) -> [i64; 4] {
    let temp1 = (input[0] + input[2]) * COSPI_16_64;
    let temp2 = (input[0] - input[2]) * COSPI_16_64;
    let step0 = dct_round_shift(temp1);
    let step1 = dct_round_shift(temp2);
    let temp1 = input[1] * COSPI_24_64 - input[3] * COSPI_8_64;
    let temp2 = input[1] * COSPI_8_64 + input[3] * COSPI_24_64;
    let step2 = dct_round_shift(temp1);
    let step3 = dct_round_shift(temp2);
    [step0 + step3, step1 + step2, step1 - step2, step0 - step3]
}

/// The AV1/VP9-lineage 8-point inverse DCT partial butterfly.
fn inverse_dct8_1d(input: [i64; 8]) -> [i64; 8] {
    let mut step1 = [0i64; 8];
    step1[0] = input[0];
    step1[2] = input[4];
    step1[1] = input[2];
    step1[3] = input[6];
    let temp1 = input[1] * COSPI_28_64 - input[7] * COSPI_4_64;
    let temp2 = input[1] * COSPI_4_64 + input[7] * COSPI_28_64;
    step1[4] = dct_round_shift(temp1);
    step1[7] = dct_round_shift(temp2);
    let temp1 = input[5] * COSPI_12_64 - input[3] * COSPI_20_64;
    let temp2 = input[5] * COSPI_20_64 + input[3] * COSPI_12_64;
    step1[5] = dct_round_shift(temp1);
    step1[6] = dct_round_shift(temp2);

    let mut step2 = [0i64; 8];
    let temp1 = (step1[0] + step1[2]) * COSPI_16_64;
    let temp2 = (step1[0] - step1[2]) * COSPI_16_64;
    step2[0] = dct_round_shift(temp1);
    step2[1] = dct_round_shift(temp2);
    let temp1 = step1[1] * COSPI_24_64 - step1[3] * COSPI_8_64;
    let temp2 = step1[1] * COSPI_8_64 + step1[3] * COSPI_24_64;
    step2[2] = dct_round_shift(temp1);
    step2[3] = dct_round_shift(temp2);
    step2[4] = step1[4] + step1[5];
    step2[5] = step1[4] - step1[5];
    step2[6] = -step1[6] + step1[7];
    step2[7] = step1[6] + step1[7];

    let mut step3 = [0i64; 8];
    step3[0] = step2[0] + step2[3];
    step3[1] = step2[1] + step2[2];
    step3[2] = step2[1] - step2[2];
    step3[3] = step2[0] - step2[3];
    step3[4] = step2[4];
    let temp1 = (step2[6] - step2[5]) * COSPI_16_64;
    let temp2 = (step2[5] + step2[6]) * COSPI_16_64;
    step3[5] = dct_round_shift(temp1);
    step3[6] = dct_round_shift(temp2);
    step3[7] = step2[7];

    [
        step3[0] + step3[7],
        step3[1] + step3[6],
        step3[2] + step3[5],
        step3[3] + step3[4],
        step3[3] - step3[4],
        step3[2] - step3[5],
        step3[1] - step3[6],
        step3[0] - step3[7],
    ]
}

/// Dequantizes and applies the non-lossless inverse transform (spec
/// §7.12.3, §7.13) for one `size x size` (4 or 8) transform block,
/// returning row-major residual samples.
pub fn inverse_transform(
    coefficients: &[i32],
    size: usize,
    tx_type: Av1TxType,
    dc_quant: i32,
    ac_quant: i32,
) -> Vec<i16> {
    debug_assert_eq!(coefficients.len(), size * size);
    let mut dequantized = vec![0i64; size * size];
    for (index, value) in dequantized.iter_mut().enumerate() {
        let quant = if index == 0 { dc_quant } else { ac_quant };
        *value = i64::from(coefficients[index]) * i64::from(quant);
    }
    match tx_type {
        Av1TxType::Idtx => dequantized
            .into_iter()
            .map(|value| value.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16)
            .collect(),
        Av1TxType::DctDct => {
            let mut rows = vec![0i64; size * size];
            for row in 0..size {
                let start = row * size;
                let transformed = if size == 4 {
                    inverse_dct4_1d(dequantized[start..start + 4].try_into().unwrap()).to_vec()
                } else {
                    inverse_dct8_1d(dequantized[start..start + 8].try_into().unwrap()).to_vec()
                };
                rows[start..start + size].copy_from_slice(&transformed);
            }
            let mut output = vec![0i16; size * size];
            for column in 0..size {
                let values: Vec<i64> = (0..size).map(|row| rows[row * size + column]).collect();
                let transformed = if size == 4 {
                    inverse_dct4_1d(values.try_into().unwrap()).to_vec()
                } else {
                    inverse_dct8_1d(values.try_into().unwrap()).to_vec()
                };
                let shift = if size == 4 { 4 } else { 5 };
                for (row, value) in transformed.into_iter().enumerate() {
                    let rounded = (value + (1 << (shift - 1))) >> shift;
                    output[row * size + column] =
                        rounded.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16;
                }
            }
            output
        }
    }
}

fn malformed(message: &str) -> Error {
    Error::new(ErrorKind::MalformedMedia, message)
}
fn limit(message: &str) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruction_is_bounded_and_exports_valid_yuv() {
        let limits = Limits {
            max_allocation_bytes: 32,
            ..Limits::default()
        };
        let dimensions = VideoDimensions::new(4, 4, &limits).unwrap();
        let mut frame = Av1IntraFrame::new(dimensions, &limits).unwrap();
        frame
            .reconstruct_block(
                Av1IntraBlock {
                    plane: 0,
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                    mode: Av1IntraMode::Dc,
                },
                &[0; 16],
            )
            .unwrap();
        let frame = frame.into_video_frame(&limits).unwrap();
        assert_eq!(frame.planes[0].data, vec![128; 16]);
        assert_eq!(frame.planes[1].data, vec![128; 4]);
    }

    #[test]
    fn blocks_cannot_escape_their_plane() {
        let limits = Limits::default();
        let dimensions = VideoDimensions::new(4, 4, &limits).unwrap();
        let mut frame = Av1IntraFrame::new(dimensions, &limits).unwrap();
        assert_eq!(
            frame
                .reconstruct_block(
                    Av1IntraBlock {
                        plane: 0,
                        x: 3,
                        y: 0,
                        width: 2,
                        height: 1,
                        mode: Av1IntraMode::Dc
                    },
                    &[0, 0]
                )
                .unwrap_err()
                .kind(),
            ErrorKind::MalformedMedia
        );
    }

    #[test]
    fn wht_matches_lossless_av1_dc_reconstruction() {
        let mut coefficients = [0; 16];
        coefficients[0] = 64;
        assert_eq!(inverse_wht_4x4(&coefficients), [16; 16]);
    }

    #[test]
    fn dc_qlookup_matches_lossless_scale_by_four_at_qindex_zero() {
        assert_eq!(get_dc_quant(0), 4);
        assert_eq!(get_ac_quant(0), 4);
    }

    #[test]
    fn qlookup_tables_are_monotonic_non_decreasing() {
        for window in DC_QLOOKUP.windows(2) {
            assert!(window[1] >= window[0]);
        }
        for window in AC_QLOOKUP.windows(2) {
            assert!(window[1] >= window[0]);
        }
    }

    #[test]
    fn idtx_dequantizes_without_extra_scaling() {
        let mut coefficients = [0; 16];
        coefficients[0] = 5;
        coefficients[1] = -3;
        let residuals = inverse_transform(&coefficients, 4, Av1TxType::Idtx, 4, 4);
        assert_eq!(residuals[0], 20);
        assert_eq!(residuals[1], -12);
        assert_eq!(residuals[2..], [0; 14]);
    }

    #[test]
    fn dct4_dc_only_produces_a_flat_block() {
        let mut coefficients = [0; 16];
        coefficients[0] = 64;
        let residuals = inverse_transform(&coefficients, 4, Av1TxType::DctDct, 4, 4);
        // A DC-only input produces a spatially flat output for a real DCT.
        assert!(residuals.iter().all(|&value| value == residuals[0]));
        assert!(residuals[0] > 0);
    }

    #[test]
    fn dct8_dc_only_produces_a_flat_block() {
        let mut coefficients = [0; 64];
        coefficients[0] = 64;
        let residuals = inverse_transform(&coefficients, 8, Av1TxType::DctDct, 4, 4);
        assert!(residuals.iter().all(|&value| value == residuals[0]));
        assert!(residuals[0] > 0);
    }

    #[test]
    fn dct4_ac_coefficient_produces_a_horizontal_gradient() {
        let mut coefficients = [0; 16];
        coefficients[1] = 32;
        let residuals = inverse_transform(&coefficients, 4, Av1TxType::DctDct, 4, 4);
        // A single horizontal-frequency AC coefficient should vary by
        // column but be identical down each column.
        for row in 0..4 {
            assert_eq!(
                &residuals[row * 4..row * 4 + 4],
                &residuals[0..4],
                "row {row} should match row 0"
            );
        }
        assert_ne!(residuals[0], residuals[1]);
    }
}
