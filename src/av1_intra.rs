//! Bounded pixel primitives used by AV1 intra-frame reconstruction.
//!
//! This is deliberately a reconstruction building block rather than a video
//! decoder.  Tile syntax owns mode and coefficient parsing; this module owns
//! checked YUV allocation and applying a decoded intra block to a plane.

use crate::{
    ColorRange, Error, ErrorKind, Limits, PixelFormat, Plane, Result, VideoDimensions, VideoFrame,
    av1_intra_pred::{
        SmoothMode, add_residual_row, directional_row, paeth_row, smooth_row, sum_samples,
    },
};

/// Intra predictors used by the first AV1 reconstruction stages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Av1IntraMode {
    Dc,
    Vertical,
    Horizontal,
    Paeth,
    Smooth,
    SmoothVertical,
    SmoothHorizontal,
    D45,
    D63,
    D67,
    D113,
    D135,
    D157,
    D203,
    Directional { angle: i16, filter_edges: bool },
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
        let dc = ((sum_samples(&top)
            + sum_samples(&left)
            + u32::try_from(width + height).expect("nonzero block dimensions") / 2)
            / u32::try_from(width + height).expect("nonzero block dimensions"))
            as u8;
        let mut prediction = vec![0u8; width];
        for row in 0..height {
            match mode {
                Av1IntraMode::Dc => prediction.fill(dc),
                Av1IntraMode::Vertical => prediction.copy_from_slice(&top),
                Av1IntraMode::Horizontal => prediction.fill(left[row]),
                Av1IntraMode::Paeth => paeth_row(top_left, &top, left[row], &mut prediction),
                Av1IntraMode::Smooth => {
                    smooth_row(SmoothMode::Smooth, &top, &left, row, &mut prediction)
                }
                Av1IntraMode::SmoothVertical => smooth_row(
                    SmoothMode::SmoothVertical,
                    &top,
                    &left,
                    row,
                    &mut prediction,
                ),
                Av1IntraMode::SmoothHorizontal => smooth_row(
                    SmoothMode::SmoothHorizontal,
                    &top,
                    &left,
                    row,
                    &mut prediction,
                ),
                Av1IntraMode::D45 => directional_row(45, &top, &left, row, true, &mut prediction),
                Av1IntraMode::D63 => directional_row(63, &top, &left, row, true, &mut prediction),
                Av1IntraMode::D67 => directional_row(67, &top, &left, row, true, &mut prediction),
                Av1IntraMode::D113 => directional_row(113, &top, &left, row, true, &mut prediction),
                Av1IntraMode::D135 => directional_row(135, &top, &left, row, true, &mut prediction),
                Av1IntraMode::D157 => directional_row(157, &top, &left, row, true, &mut prediction),
                Av1IntraMode::D203 => directional_row(203, &top, &left, row, true, &mut prediction),
                Av1IntraMode::Directional {
                    angle,
                    filter_edges,
                } => directional_row(angle, &top, &left, row, filter_edges, &mut prediction),
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
/// restricted to the entries this crate implements: the identity transform
/// plus every combination of the DCT, ADST, and flipped-ADST kernels).
///
/// The decoder only ever signals [`Av1TxType::DctDct`] and
/// [`Av1TxType::Idtx`] today; the remaining entries are reachable through
/// [`inverse_transform`] and are covered by the transform tests and
/// benchmark.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Av1TxType {
    DctDct,
    Idtx,
    AdstDct,
    DctAdst,
    AdstAdst,
    FlipadstDct,
    DctFlipadst,
    FlipadstFlipadst,
    AdstFlipadst,
    FlipadstAdst,
}

/// One of the two separable 1-D kernels an [`Av1TxType`] applies.
///
/// The spec names each transform type after its vertical and horizontal
/// kernels; `FLIPADST` is the ADST kernel plus a reversal of the finished
/// block along that axis, so it is represented here as [`Tx1d::Adst`] with a
/// flip flag rather than as a separate kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tx1d {
    /// Inverse DCT. Defined for 4, 8, 16, and 32 points.
    Dct,
    /// Inverse ADST. Defined for 4, 8, and 16 points; a 32-point block always
    /// uses the DCT.
    Adst,
}

impl Av1TxType {
    /// The vertical kernel, the horizontal kernel, and whether the finished
    /// block is reversed left-to-right and/or top-to-bottom.
    ///
    /// [`Av1TxType::Idtx`] has no butterfly pass at all and reports the DCT
    /// pair; [`inverse_transform`] handles it before consulting this.
    #[must_use]
    pub fn kernels(self) -> (Tx1d, Tx1d, bool, bool) {
        use Tx1d::{Adst, Dct};
        match self {
            Av1TxType::DctDct | Av1TxType::Idtx => (Dct, Dct, false, false),
            Av1TxType::AdstDct => (Adst, Dct, false, false),
            Av1TxType::DctAdst => (Dct, Adst, false, false),
            Av1TxType::AdstAdst => (Adst, Adst, false, false),
            Av1TxType::FlipadstDct => (Adst, Dct, false, true),
            Av1TxType::DctFlipadst => (Dct, Adst, true, false),
            Av1TxType::FlipadstFlipadst => (Adst, Adst, true, true),
            Av1TxType::AdstFlipadst => (Adst, Adst, true, false),
            Av1TxType::FlipadstAdst => (Adst, Adst, false, true),
        }
    }
}

const COSPI_1_64: i64 = 16364;
const COSPI_2_64: i64 = 16305;
const COSPI_3_64: i64 = 16207;
const COSPI_4_64: i64 = 16069;
const COSPI_5_64: i64 = 15893;
const COSPI_6_64: i64 = 15679;
const COSPI_7_64: i64 = 15426;
const COSPI_8_64: i64 = 15137;
const COSPI_9_64: i64 = 14811;
const COSPI_10_64: i64 = 14449;
const COSPI_11_64: i64 = 14053;
const COSPI_12_64: i64 = 13623;
const COSPI_13_64: i64 = 13160;
const COSPI_14_64: i64 = 12665;
const COSPI_15_64: i64 = 12140;
const COSPI_16_64: i64 = 11585;
const COSPI_17_64: i64 = 11003;
const COSPI_18_64: i64 = 10394;
const COSPI_19_64: i64 = 9760;
const COSPI_20_64: i64 = 9102;
const COSPI_21_64: i64 = 8423;
const COSPI_22_64: i64 = 7723;
const COSPI_23_64: i64 = 7005;
const COSPI_24_64: i64 = 6270;
const COSPI_25_64: i64 = 5520;
const COSPI_26_64: i64 = 4756;
const COSPI_27_64: i64 = 3981;
const COSPI_28_64: i64 = 3196;
const COSPI_29_64: i64 = 2404;
const COSPI_30_64: i64 = 1606;
const COSPI_31_64: i64 = 804;
const SINPI_1_9: i64 = 5283;
const SINPI_2_9: i64 = 9929;
const SINPI_3_9: i64 = 13377;
const SINPI_4_9: i64 = 15212;

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

/// The AV1/VP9-lineage 16-point inverse DCT partial butterfly.
fn inverse_dct16_1d(input: [i64; 16]) -> [i64; 16] {
    let mut output = [0i64; 16];
    // stage 1
    let step1_0 = input[0];
    let step1_1 = input[8];
    let step1_2 = input[4];
    let step1_3 = input[12];
    let step1_4 = input[2];
    let step1_5 = input[10];
    let step1_6 = input[6];
    let step1_7 = input[14];
    let step1_8 = input[1];
    let step1_9 = input[9];
    let step1_10 = input[5];
    let step1_11 = input[13];
    let step1_12 = input[3];
    let step1_13 = input[11];
    let step1_14 = input[7];
    let step1_15 = input[15];
    // stage 2
    let step2_0 = step1_0;
    let step2_1 = step1_1;
    let step2_2 = step1_2;
    let step2_3 = step1_3;
    let step2_4 = step1_4;
    let step2_5 = step1_5;
    let step2_6 = step1_6;
    let step2_7 = step1_7;
    let temp_1 = (step1_8 * COSPI_30_64) - (step1_15 * COSPI_2_64);
    let temp_2 = (step1_8 * COSPI_2_64) + (step1_15 * COSPI_30_64);
    let step2_8 = dct_round_shift(temp_1);
    let step2_15 = dct_round_shift(temp_2);
    let temp_1_2 = (step1_9 * COSPI_14_64) - (step1_14 * COSPI_18_64);
    let temp_2_2 = (step1_9 * COSPI_18_64) + (step1_14 * COSPI_14_64);
    let step2_9 = dct_round_shift(temp_1_2);
    let step2_14 = dct_round_shift(temp_2_2);
    let temp_1_3 = (step1_10 * COSPI_22_64) - (step1_13 * COSPI_10_64);
    let temp_2_3 = (step1_10 * COSPI_10_64) + (step1_13 * COSPI_22_64);
    let step2_10 = dct_round_shift(temp_1_3);
    let step2_13 = dct_round_shift(temp_2_3);
    let temp_1_4 = (step1_11 * COSPI_6_64) - (step1_12 * COSPI_26_64);
    let temp_2_4 = (step1_11 * COSPI_26_64) + (step1_12 * COSPI_6_64);
    let step2_11 = dct_round_shift(temp_1_4);
    let step2_12 = dct_round_shift(temp_2_4);
    // stage 3
    let step1_0_2 = step2_0;
    let step1_1_2 = step2_1;
    let step1_2_2 = step2_2;
    let step1_3_2 = step2_3;
    let temp_1_5 = (step2_4 * COSPI_28_64) - (step2_7 * COSPI_4_64);
    let temp_2_5 = (step2_4 * COSPI_4_64) + (step2_7 * COSPI_28_64);
    let step1_4_2 = dct_round_shift(temp_1_5);
    let step1_7_2 = dct_round_shift(temp_2_5);
    let temp_1_6 = (step2_5 * COSPI_12_64) - (step2_6 * COSPI_20_64);
    let temp_2_6 = (step2_5 * COSPI_20_64) + (step2_6 * COSPI_12_64);
    let step1_5_2 = dct_round_shift(temp_1_6);
    let step1_6_2 = dct_round_shift(temp_2_6);
    let step1_8_2 = step2_8 + step2_9;
    let step1_9_2 = step2_8 - step2_9;
    let step1_10_2 = -step2_10 + step2_11;
    let step1_11_2 = step2_10 + step2_11;
    let step1_12_2 = step2_12 + step2_13;
    let step1_13_2 = step2_12 - step2_13;
    let step1_14_2 = -step2_14 + step2_15;
    let step1_15_2 = step2_14 + step2_15;
    // stage 4
    let temp_1_7 = (step1_0_2 + step1_1_2) * COSPI_16_64;
    let temp_2_7 = (step1_0_2 - step1_1_2) * COSPI_16_64;
    let step2_0_2 = dct_round_shift(temp_1_7);
    let step2_1_2 = dct_round_shift(temp_2_7);
    let temp_1_8 = (step1_2_2 * COSPI_24_64) - (step1_3_2 * COSPI_8_64);
    let temp_2_8 = (step1_2_2 * COSPI_8_64) + (step1_3_2 * COSPI_24_64);
    let step2_2_2 = dct_round_shift(temp_1_8);
    let step2_3_2 = dct_round_shift(temp_2_8);
    let step2_4_2 = step1_4_2 + step1_5_2;
    let step2_5_2 = step1_4_2 - step1_5_2;
    let step2_6_2 = -step1_6_2 + step1_7_2;
    let step2_7_2 = step1_6_2 + step1_7_2;
    let step2_8_2 = step1_8_2;
    let step2_15_2 = step1_15_2;
    let temp_1_9 = (-step1_9_2 * COSPI_8_64) + (step1_14_2 * COSPI_24_64);
    let temp_2_9 = (step1_9_2 * COSPI_24_64) + (step1_14_2 * COSPI_8_64);
    let step2_9_2 = dct_round_shift(temp_1_9);
    let step2_14_2 = dct_round_shift(temp_2_9);
    let temp_1_10 = (-step1_10_2 * COSPI_24_64) - (step1_13_2 * COSPI_8_64);
    let temp_2_10 = (-step1_10_2 * COSPI_8_64) + (step1_13_2 * COSPI_24_64);
    let step2_10_2 = dct_round_shift(temp_1_10);
    let step2_13_2 = dct_round_shift(temp_2_10);
    let step2_11_2 = step1_11_2;
    let step2_12_2 = step1_12_2;
    // stage 5
    let step1_0_3 = step2_0_2 + step2_3_2;
    let step1_1_3 = step2_1_2 + step2_2_2;
    let step1_2_3 = step2_1_2 - step2_2_2;
    let step1_3_3 = step2_0_2 - step2_3_2;
    let step1_4_3 = step2_4_2;
    let temp_1_11 = (step2_6_2 - step2_5_2) * COSPI_16_64;
    let temp_2_11 = (step2_5_2 + step2_6_2) * COSPI_16_64;
    let step1_5_3 = dct_round_shift(temp_1_11);
    let step1_6_3 = dct_round_shift(temp_2_11);
    let step1_7_3 = step2_7_2;
    let step1_8_3 = step2_8_2 + step2_11_2;
    let step1_9_3 = step2_9_2 + step2_10_2;
    let step1_10_3 = step2_9_2 - step2_10_2;
    let step1_11_3 = step2_8_2 - step2_11_2;
    let step1_12_3 = -step2_12_2 + step2_15_2;
    let step1_13_3 = -step2_13_2 + step2_14_2;
    let step1_14_3 = step2_13_2 + step2_14_2;
    let step1_15_3 = step2_12_2 + step2_15_2;
    // stage 6
    let step2_0_3 = step1_0_3 + step1_7_3;
    let step2_1_3 = step1_1_3 + step1_6_3;
    let step2_2_3 = step1_2_3 + step1_5_3;
    let step2_3_3 = step1_3_3 + step1_4_3;
    let step2_4_3 = step1_3_3 - step1_4_3;
    let step2_5_3 = step1_2_3 - step1_5_3;
    let step2_6_3 = step1_1_3 - step1_6_3;
    let step2_7_3 = step1_0_3 - step1_7_3;
    let step2_8_3 = step1_8_3;
    let step2_9_3 = step1_9_3;
    let temp_1_12 = (-step1_10_3 + step1_13_3) * COSPI_16_64;
    let temp_2_12 = (step1_10_3 + step1_13_3) * COSPI_16_64;
    let step2_10_3 = dct_round_shift(temp_1_12);
    let step2_13_3 = dct_round_shift(temp_2_12);
    let temp_1_13 = (-step1_11_3 + step1_12_3) * COSPI_16_64;
    let temp_2_13 = (step1_11_3 + step1_12_3) * COSPI_16_64;
    let step2_11_3 = dct_round_shift(temp_1_13);
    let step2_12_3 = dct_round_shift(temp_2_13);
    let step2_14_3 = step1_14_3;
    let step2_15_3 = step1_15_3;
    // stage 7
    output[0] = step2_0_3 + step2_15_3;
    output[1] = step2_1_3 + step2_14_3;
    output[2] = step2_2_3 + step2_13_3;
    output[3] = step2_3_3 + step2_12_3;
    output[4] = step2_4_3 + step2_11_3;
    output[5] = step2_5_3 + step2_10_3;
    output[6] = step2_6_3 + step2_9_3;
    output[7] = step2_7_3 + step2_8_3;
    output[8] = step2_7_3 - step2_8_3;
    output[9] = step2_6_3 - step2_9_3;
    output[10] = step2_5_3 - step2_10_3;
    output[11] = step2_4_3 - step2_11_3;
    output[12] = step2_3_3 - step2_12_3;
    output[13] = step2_2_3 - step2_13_3;
    output[14] = step2_1_3 - step2_14_3;
    output[15] = step2_0_3 - step2_15_3;
    output
}

/// The AV1/VP9-lineage 32-point inverse DCT partial butterfly.
fn inverse_dct32_1d(input: [i64; 32]) -> [i64; 32] {
    let mut output = [0i64; 32];
    // stage 1
    let step1_0 = input[0];
    let step1_1 = input[16];
    let step1_2 = input[8];
    let step1_3 = input[24];
    let step1_4 = input[4];
    let step1_5 = input[20];
    let step1_6 = input[12];
    let step1_7 = input[28];
    let step1_8 = input[2];
    let step1_9 = input[18];
    let step1_10 = input[10];
    let step1_11 = input[26];
    let step1_12 = input[6];
    let step1_13 = input[22];
    let step1_14 = input[14];
    let step1_15 = input[30];
    let temp_1 = (input[1] * COSPI_31_64) - (input[31] * COSPI_1_64);
    let temp_2 = (input[1] * COSPI_1_64) + (input[31] * COSPI_31_64);
    let step1_16 = dct_round_shift(temp_1);
    let step1_31 = dct_round_shift(temp_2);
    let temp_1_2 = (input[17] * COSPI_15_64) - (input[15] * COSPI_17_64);
    let temp_2_2 = (input[17] * COSPI_17_64) + (input[15] * COSPI_15_64);
    let step1_17 = dct_round_shift(temp_1_2);
    let step1_30 = dct_round_shift(temp_2_2);
    let temp_1_3 = (input[9] * COSPI_23_64) - (input[23] * COSPI_9_64);
    let temp_2_3 = (input[9] * COSPI_9_64) + (input[23] * COSPI_23_64);
    let step1_18 = dct_round_shift(temp_1_3);
    let step1_29 = dct_round_shift(temp_2_3);
    let temp_1_4 = (input[25] * COSPI_7_64) - (input[7] * COSPI_25_64);
    let temp_2_4 = (input[25] * COSPI_25_64) + (input[7] * COSPI_7_64);
    let step1_19 = dct_round_shift(temp_1_4);
    let step1_28 = dct_round_shift(temp_2_4);
    let temp_1_5 = (input[5] * COSPI_27_64) - (input[27] * COSPI_5_64);
    let temp_2_5 = (input[5] * COSPI_5_64) + (input[27] * COSPI_27_64);
    let step1_20 = dct_round_shift(temp_1_5);
    let step1_27 = dct_round_shift(temp_2_5);
    let temp_1_6 = (input[21] * COSPI_11_64) - (input[11] * COSPI_21_64);
    let temp_2_6 = (input[21] * COSPI_21_64) + (input[11] * COSPI_11_64);
    let step1_21 = dct_round_shift(temp_1_6);
    let step1_26 = dct_round_shift(temp_2_6);
    let temp_1_7 = (input[13] * COSPI_19_64) - (input[19] * COSPI_13_64);
    let temp_2_7 = (input[13] * COSPI_13_64) + (input[19] * COSPI_19_64);
    let step1_22 = dct_round_shift(temp_1_7);
    let step1_25 = dct_round_shift(temp_2_7);
    let temp_1_8 = (input[29] * COSPI_3_64) - (input[3] * COSPI_29_64);
    let temp_2_8 = (input[29] * COSPI_29_64) + (input[3] * COSPI_3_64);
    let step1_23 = dct_round_shift(temp_1_8);
    let step1_24 = dct_round_shift(temp_2_8);
    // stage 2
    let step2_0 = step1_0;
    let step2_1 = step1_1;
    let step2_2 = step1_2;
    let step2_3 = step1_3;
    let step2_4 = step1_4;
    let step2_5 = step1_5;
    let step2_6 = step1_6;
    let step2_7 = step1_7;
    let temp_1_9 = (step1_8 * COSPI_30_64) - (step1_15 * COSPI_2_64);
    let temp_2_9 = (step1_8 * COSPI_2_64) + (step1_15 * COSPI_30_64);
    let step2_8 = dct_round_shift(temp_1_9);
    let step2_15 = dct_round_shift(temp_2_9);
    let temp_1_10 = (step1_9 * COSPI_14_64) - (step1_14 * COSPI_18_64);
    let temp_2_10 = (step1_9 * COSPI_18_64) + (step1_14 * COSPI_14_64);
    let step2_9 = dct_round_shift(temp_1_10);
    let step2_14 = dct_round_shift(temp_2_10);
    let temp_1_11 = (step1_10 * COSPI_22_64) - (step1_13 * COSPI_10_64);
    let temp_2_11 = (step1_10 * COSPI_10_64) + (step1_13 * COSPI_22_64);
    let step2_10 = dct_round_shift(temp_1_11);
    let step2_13 = dct_round_shift(temp_2_11);
    let temp_1_12 = (step1_11 * COSPI_6_64) - (step1_12 * COSPI_26_64);
    let temp_2_12 = (step1_11 * COSPI_26_64) + (step1_12 * COSPI_6_64);
    let step2_11 = dct_round_shift(temp_1_12);
    let step2_12 = dct_round_shift(temp_2_12);
    let step2_16 = step1_16 + step1_17;
    let step2_17 = step1_16 - step1_17;
    let step2_18 = -step1_18 + step1_19;
    let step2_19 = step1_18 + step1_19;
    let step2_20 = step1_20 + step1_21;
    let step2_21 = step1_20 - step1_21;
    let step2_22 = -step1_22 + step1_23;
    let step2_23 = step1_22 + step1_23;
    let step2_24 = step1_24 + step1_25;
    let step2_25 = step1_24 - step1_25;
    let step2_26 = -step1_26 + step1_27;
    let step2_27 = step1_26 + step1_27;
    let step2_28 = step1_28 + step1_29;
    let step2_29 = step1_28 - step1_29;
    let step2_30 = -step1_30 + step1_31;
    let step2_31 = step1_30 + step1_31;
    // stage 3
    let step1_0_2 = step2_0;
    let step1_1_2 = step2_1;
    let step1_2_2 = step2_2;
    let step1_3_2 = step2_3;
    let temp_1_13 = (step2_4 * COSPI_28_64) - (step2_7 * COSPI_4_64);
    let temp_2_13 = (step2_4 * COSPI_4_64) + (step2_7 * COSPI_28_64);
    let step1_4_2 = dct_round_shift(temp_1_13);
    let step1_7_2 = dct_round_shift(temp_2_13);
    let temp_1_14 = (step2_5 * COSPI_12_64) - (step2_6 * COSPI_20_64);
    let temp_2_14 = (step2_5 * COSPI_20_64) + (step2_6 * COSPI_12_64);
    let step1_5_2 = dct_round_shift(temp_1_14);
    let step1_6_2 = dct_round_shift(temp_2_14);
    let step1_8_2 = step2_8 + step2_9;
    let step1_9_2 = step2_8 - step2_9;
    let step1_10_2 = -step2_10 + step2_11;
    let step1_11_2 = step2_10 + step2_11;
    let step1_12_2 = step2_12 + step2_13;
    let step1_13_2 = step2_12 - step2_13;
    let step1_14_2 = -step2_14 + step2_15;
    let step1_15_2 = step2_14 + step2_15;
    let step1_16_2 = step2_16;
    let step1_31_2 = step2_31;
    let temp_1_15 = (-step2_17 * COSPI_4_64) + (step2_30 * COSPI_28_64);
    let temp_2_15 = (step2_17 * COSPI_28_64) + (step2_30 * COSPI_4_64);
    let step1_17_2 = dct_round_shift(temp_1_15);
    let step1_30_2 = dct_round_shift(temp_2_15);
    let temp_1_16 = (-step2_18 * COSPI_28_64) - (step2_29 * COSPI_4_64);
    let temp_2_16 = (-step2_18 * COSPI_4_64) + (step2_29 * COSPI_28_64);
    let step1_18_2 = dct_round_shift(temp_1_16);
    let step1_29_2 = dct_round_shift(temp_2_16);
    let step1_19_2 = step2_19;
    let step1_20_2 = step2_20;
    let temp_1_17 = (-step2_21 * COSPI_20_64) + (step2_26 * COSPI_12_64);
    let temp_2_17 = (step2_21 * COSPI_12_64) + (step2_26 * COSPI_20_64);
    let step1_21_2 = dct_round_shift(temp_1_17);
    let step1_26_2 = dct_round_shift(temp_2_17);
    let temp_1_18 = (-step2_22 * COSPI_12_64) - (step2_25 * COSPI_20_64);
    let temp_2_18 = (-step2_22 * COSPI_20_64) + (step2_25 * COSPI_12_64);
    let step1_22_2 = dct_round_shift(temp_1_18);
    let step1_25_2 = dct_round_shift(temp_2_18);
    let step1_23_2 = step2_23;
    let step1_24_2 = step2_24;
    let step1_27_2 = step2_27;
    let step1_28_2 = step2_28;
    // stage 4
    let temp_1_19 = (step1_0_2 + step1_1_2) * COSPI_16_64;
    let temp_2_19 = (step1_0_2 - step1_1_2) * COSPI_16_64;
    let step2_0_2 = dct_round_shift(temp_1_19);
    let step2_1_2 = dct_round_shift(temp_2_19);
    let temp_1_20 = (step1_2_2 * COSPI_24_64) - (step1_3_2 * COSPI_8_64);
    let temp_2_20 = (step1_2_2 * COSPI_8_64) + (step1_3_2 * COSPI_24_64);
    let step2_2_2 = dct_round_shift(temp_1_20);
    let step2_3_2 = dct_round_shift(temp_2_20);
    let step2_4_2 = step1_4_2 + step1_5_2;
    let step2_5_2 = step1_4_2 - step1_5_2;
    let step2_6_2 = -step1_6_2 + step1_7_2;
    let step2_7_2 = step1_6_2 + step1_7_2;
    let step2_8_2 = step1_8_2;
    let step2_15_2 = step1_15_2;
    let temp_1_21 = (-step1_9_2 * COSPI_8_64) + (step1_14_2 * COSPI_24_64);
    let temp_2_21 = (step1_9_2 * COSPI_24_64) + (step1_14_2 * COSPI_8_64);
    let step2_9_2 = dct_round_shift(temp_1_21);
    let step2_14_2 = dct_round_shift(temp_2_21);
    let temp_1_22 = (-step1_10_2 * COSPI_24_64) - (step1_13_2 * COSPI_8_64);
    let temp_2_22 = (-step1_10_2 * COSPI_8_64) + (step1_13_2 * COSPI_24_64);
    let step2_10_2 = dct_round_shift(temp_1_22);
    let step2_13_2 = dct_round_shift(temp_2_22);
    let step2_11_2 = step1_11_2;
    let step2_12_2 = step1_12_2;
    let step2_16_2 = step1_16_2 + step1_19_2;
    let step2_17_2 = step1_17_2 + step1_18_2;
    let step2_18_2 = step1_17_2 - step1_18_2;
    let step2_19_2 = step1_16_2 - step1_19_2;
    let step2_20_2 = -step1_20_2 + step1_23_2;
    let step2_21_2 = -step1_21_2 + step1_22_2;
    let step2_22_2 = step1_21_2 + step1_22_2;
    let step2_23_2 = step1_20_2 + step1_23_2;
    let step2_24_2 = step1_24_2 + step1_27_2;
    let step2_25_2 = step1_25_2 + step1_26_2;
    let step2_26_2 = step1_25_2 - step1_26_2;
    let step2_27_2 = step1_24_2 - step1_27_2;
    let step2_28_2 = -step1_28_2 + step1_31_2;
    let step2_29_2 = -step1_29_2 + step1_30_2;
    let step2_30_2 = step1_29_2 + step1_30_2;
    let step2_31_2 = step1_28_2 + step1_31_2;
    // stage 5
    let step1_0_3 = step2_0_2 + step2_3_2;
    let step1_1_3 = step2_1_2 + step2_2_2;
    let step1_2_3 = step2_1_2 - step2_2_2;
    let step1_3_3 = step2_0_2 - step2_3_2;
    let step1_4_3 = step2_4_2;
    let temp_1_23 = (step2_6_2 - step2_5_2) * COSPI_16_64;
    let temp_2_23 = (step2_5_2 + step2_6_2) * COSPI_16_64;
    let step1_5_3 = dct_round_shift(temp_1_23);
    let step1_6_3 = dct_round_shift(temp_2_23);
    let step1_7_3 = step2_7_2;
    let step1_8_3 = step2_8_2 + step2_11_2;
    let step1_9_3 = step2_9_2 + step2_10_2;
    let step1_10_3 = step2_9_2 - step2_10_2;
    let step1_11_3 = step2_8_2 - step2_11_2;
    let step1_12_3 = -step2_12_2 + step2_15_2;
    let step1_13_3 = -step2_13_2 + step2_14_2;
    let step1_14_3 = step2_13_2 + step2_14_2;
    let step1_15_3 = step2_12_2 + step2_15_2;
    let step1_16_3 = step2_16_2;
    let step1_17_3 = step2_17_2;
    let temp_1_24 = (-step2_18_2 * COSPI_8_64) + (step2_29_2 * COSPI_24_64);
    let temp_2_24 = (step2_18_2 * COSPI_24_64) + (step2_29_2 * COSPI_8_64);
    let step1_18_3 = dct_round_shift(temp_1_24);
    let step1_29_3 = dct_round_shift(temp_2_24);
    let temp_1_25 = (-step2_19_2 * COSPI_8_64) + (step2_28_2 * COSPI_24_64);
    let temp_2_25 = (step2_19_2 * COSPI_24_64) + (step2_28_2 * COSPI_8_64);
    let step1_19_3 = dct_round_shift(temp_1_25);
    let step1_28_3 = dct_round_shift(temp_2_25);
    let temp_1_26 = (-step2_20_2 * COSPI_24_64) - (step2_27_2 * COSPI_8_64);
    let temp_2_26 = (-step2_20_2 * COSPI_8_64) + (step2_27_2 * COSPI_24_64);
    let step1_20_3 = dct_round_shift(temp_1_26);
    let step1_27_3 = dct_round_shift(temp_2_26);
    let temp_1_27 = (-step2_21_2 * COSPI_24_64) - (step2_26_2 * COSPI_8_64);
    let temp_2_27 = (-step2_21_2 * COSPI_8_64) + (step2_26_2 * COSPI_24_64);
    let step1_21_3 = dct_round_shift(temp_1_27);
    let step1_26_3 = dct_round_shift(temp_2_27);
    let step1_22_3 = step2_22_2;
    let step1_23_3 = step2_23_2;
    let step1_24_3 = step2_24_2;
    let step1_25_3 = step2_25_2;
    let step1_30_3 = step2_30_2;
    let step1_31_3 = step2_31_2;
    // stage 6
    let step2_0_3 = step1_0_3 + step1_7_3;
    let step2_1_3 = step1_1_3 + step1_6_3;
    let step2_2_3 = step1_2_3 + step1_5_3;
    let step2_3_3 = step1_3_3 + step1_4_3;
    let step2_4_3 = step1_3_3 - step1_4_3;
    let step2_5_3 = step1_2_3 - step1_5_3;
    let step2_6_3 = step1_1_3 - step1_6_3;
    let step2_7_3 = step1_0_3 - step1_7_3;
    let step2_8_3 = step1_8_3;
    let step2_9_3 = step1_9_3;
    let temp_1_28 = (-step1_10_3 + step1_13_3) * COSPI_16_64;
    let temp_2_28 = (step1_10_3 + step1_13_3) * COSPI_16_64;
    let step2_10_3 = dct_round_shift(temp_1_28);
    let step2_13_3 = dct_round_shift(temp_2_28);
    let temp_1_29 = (-step1_11_3 + step1_12_3) * COSPI_16_64;
    let temp_2_29 = (step1_11_3 + step1_12_3) * COSPI_16_64;
    let step2_11_3 = dct_round_shift(temp_1_29);
    let step2_12_3 = dct_round_shift(temp_2_29);
    let step2_14_3 = step1_14_3;
    let step2_15_3 = step1_15_3;
    let step2_16_3 = step1_16_3 + step1_23_3;
    let step2_17_3 = step1_17_3 + step1_22_3;
    let step2_18_3 = step1_18_3 + step1_21_3;
    let step2_19_3 = step1_19_3 + step1_20_3;
    let step2_20_3 = step1_19_3 - step1_20_3;
    let step2_21_3 = step1_18_3 - step1_21_3;
    let step2_22_3 = step1_17_3 - step1_22_3;
    let step2_23_3 = step1_16_3 - step1_23_3;
    let step2_24_3 = -step1_24_3 + step1_31_3;
    let step2_25_3 = -step1_25_3 + step1_30_3;
    let step2_26_3 = -step1_26_3 + step1_29_3;
    let step2_27_3 = -step1_27_3 + step1_28_3;
    let step2_28_3 = step1_27_3 + step1_28_3;
    let step2_29_3 = step1_26_3 + step1_29_3;
    let step2_30_3 = step1_25_3 + step1_30_3;
    let step2_31_3 = step1_24_3 + step1_31_3;
    // stage 7
    let step1_0_4 = step2_0_3 + step2_15_3;
    let step1_1_4 = step2_1_3 + step2_14_3;
    let step1_2_4 = step2_2_3 + step2_13_3;
    let step1_3_4 = step2_3_3 + step2_12_3;
    let step1_4_4 = step2_4_3 + step2_11_3;
    let step1_5_4 = step2_5_3 + step2_10_3;
    let step1_6_4 = step2_6_3 + step2_9_3;
    let step1_7_4 = step2_7_3 + step2_8_3;
    let step1_8_4 = step2_7_3 - step2_8_3;
    let step1_9_4 = step2_6_3 - step2_9_3;
    let step1_10_4 = step2_5_3 - step2_10_3;
    let step1_11_4 = step2_4_3 - step2_11_3;
    let step1_12_4 = step2_3_3 - step2_12_3;
    let step1_13_4 = step2_2_3 - step2_13_3;
    let step1_14_4 = step2_1_3 - step2_14_3;
    let step1_15_4 = step2_0_3 - step2_15_3;
    let step1_16_4 = step2_16_3;
    let step1_17_4 = step2_17_3;
    let step1_18_4 = step2_18_3;
    let step1_19_4 = step2_19_3;
    let temp_1_30 = (-step2_20_3 + step2_27_3) * COSPI_16_64;
    let temp_2_30 = (step2_20_3 + step2_27_3) * COSPI_16_64;
    let step1_20_4 = dct_round_shift(temp_1_30);
    let step1_27_4 = dct_round_shift(temp_2_30);
    let temp_1_31 = (-step2_21_3 + step2_26_3) * COSPI_16_64;
    let temp_2_31 = (step2_21_3 + step2_26_3) * COSPI_16_64;
    let step1_21_4 = dct_round_shift(temp_1_31);
    let step1_26_4 = dct_round_shift(temp_2_31);
    let temp_1_32 = (-step2_22_3 + step2_25_3) * COSPI_16_64;
    let temp_2_32 = (step2_22_3 + step2_25_3) * COSPI_16_64;
    let step1_22_4 = dct_round_shift(temp_1_32);
    let step1_25_4 = dct_round_shift(temp_2_32);
    let temp_1_33 = (-step2_23_3 + step2_24_3) * COSPI_16_64;
    let temp_2_33 = (step2_23_3 + step2_24_3) * COSPI_16_64;
    let step1_23_4 = dct_round_shift(temp_1_33);
    let step1_24_4 = dct_round_shift(temp_2_33);
    let step1_28_4 = step2_28_3;
    let step1_29_4 = step2_29_3;
    let step1_30_4 = step2_30_3;
    let step1_31_4 = step2_31_3;
    // final stage
    output[0] = step1_0_4 + step1_31_4;
    output[1] = step1_1_4 + step1_30_4;
    output[2] = step1_2_4 + step1_29_4;
    output[3] = step1_3_4 + step1_28_4;
    output[4] = step1_4_4 + step1_27_4;
    output[5] = step1_5_4 + step1_26_4;
    output[6] = step1_6_4 + step1_25_4;
    output[7] = step1_7_4 + step1_24_4;
    output[8] = step1_8_4 + step1_23_4;
    output[9] = step1_9_4 + step1_22_4;
    output[10] = step1_10_4 + step1_21_4;
    output[11] = step1_11_4 + step1_20_4;
    output[12] = step1_12_4 + step1_19_4;
    output[13] = step1_13_4 + step1_18_4;
    output[14] = step1_14_4 + step1_17_4;
    output[15] = step1_15_4 + step1_16_4;
    output[16] = step1_15_4 - step1_16_4;
    output[17] = step1_14_4 - step1_17_4;
    output[18] = step1_13_4 - step1_18_4;
    output[19] = step1_12_4 - step1_19_4;
    output[20] = step1_11_4 - step1_20_4;
    output[21] = step1_10_4 - step1_21_4;
    output[22] = step1_9_4 - step1_22_4;
    output[23] = step1_8_4 - step1_23_4;
    output[24] = step1_7_4 - step1_24_4;
    output[25] = step1_6_4 - step1_25_4;
    output[26] = step1_5_4 - step1_26_4;
    output[27] = step1_4_4 - step1_27_4;
    output[28] = step1_3_4 - step1_28_4;
    output[29] = step1_2_4 - step1_29_4;
    output[30] = step1_1_4 - step1_30_4;
    output[31] = step1_0_4 - step1_31_4;
    output
}

/// The AV1/VP9-lineage 4-point inverse ADST partial butterfly.
fn inverse_adst4_1d(input: [i64; 4]) -> [i64; 4] {
    let mut output = [0i64; 4];
    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];
    let x3 = input[3];
    // 32-bit result is enough for the following multiplications.
    let s0 = SINPI_1_9 * x0;
    let s1 = SINPI_2_9 * x0;
    let s2 = SINPI_3_9 * x1;
    let s3 = SINPI_4_9 * x2;
    let s4 = SINPI_1_9 * x2;
    let s5 = SINPI_2_9 * x3;
    let s6 = SINPI_4_9 * x3;
    let s7 = (x0 - x2) + x3;
    let s0_2 = (s0 + s3) + s5;
    let s1_2 = (s1 - s4) - s6;
    let s3_2 = s2;
    let s2_2 = SINPI_3_9 * s7;
    // 1-D transform scaling factor is sqrt(2).
    // The overall dynamic range is 14b (input) + 14b (multiplication scaling)
    // + 1b (addition) = 29b.
    // Hence the output bit depth is 15b.
    output[0] = dct_round_shift(s0_2 + s3_2);
    output[1] = dct_round_shift(s1_2 + s3_2);
    output[2] = dct_round_shift(s2_2);
    output[3] = dct_round_shift((s0_2 + s1_2) - s3_2);
    output
}

/// The AV1/VP9-lineage 8-point inverse ADST partial butterfly.
fn inverse_adst8_1d(input: [i64; 8]) -> [i64; 8] {
    let mut output = [0i64; 8];
    let x0 = input[7];
    let x1 = input[0];
    let x2 = input[5];
    let x3 = input[2];
    let x4 = input[3];
    let x5 = input[4];
    let x6 = input[1];
    let x7 = input[6];
    // stage 1
    let s0 = (COSPI_2_64 * x0) + (COSPI_30_64 * x1);
    let s1 = (COSPI_30_64 * x0) - (COSPI_2_64 * x1);
    let s2 = (COSPI_10_64 * x2) + (COSPI_22_64 * x3);
    let s3 = (COSPI_22_64 * x2) - (COSPI_10_64 * x3);
    let s4 = (COSPI_18_64 * x4) + (COSPI_14_64 * x5);
    let s5 = (COSPI_14_64 * x4) - (COSPI_18_64 * x5);
    let s6 = (COSPI_26_64 * x6) + (COSPI_6_64 * x7);
    let s7 = (COSPI_6_64 * x6) - (COSPI_26_64 * x7);
    let x0_2 = dct_round_shift(s0 + s4);
    let x1_2 = dct_round_shift(s1 + s5);
    let x2_2 = dct_round_shift(s2 + s6);
    let x3_2 = dct_round_shift(s3 + s7);
    let x4_2 = dct_round_shift(s0 - s4);
    let x5_2 = dct_round_shift(s1 - s5);
    let x6_2 = dct_round_shift(s2 - s6);
    let x7_2 = dct_round_shift(s3 - s7);
    // stage 2
    let s0_2 = x0_2;
    let s1_2 = x1_2;
    let s2_2 = x2_2;
    let s3_2 = x3_2;
    let s4_2 = (COSPI_8_64 * x4_2) + (COSPI_24_64 * x5_2);
    let s5_2 = (COSPI_24_64 * x4_2) - (COSPI_8_64 * x5_2);
    let s6_2 = (-COSPI_24_64 * x6_2) + (COSPI_8_64 * x7_2);
    let s7_2 = (COSPI_8_64 * x6_2) + (COSPI_24_64 * x7_2);
    let x0_3 = s0_2 + s2_2;
    let x1_3 = s1_2 + s3_2;
    let x2_3 = s0_2 - s2_2;
    let x3_3 = s1_2 - s3_2;
    let x4_3 = dct_round_shift(s4_2 + s6_2);
    let x5_3 = dct_round_shift(s5_2 + s7_2);
    let x6_3 = dct_round_shift(s4_2 - s6_2);
    let x7_3 = dct_round_shift(s5_2 - s7_2);
    // stage 3
    let s2_3 = COSPI_16_64 * (x2_3 + x3_3);
    let s3_3 = COSPI_16_64 * (x2_3 - x3_3);
    let s6_3 = COSPI_16_64 * (x6_3 + x7_3);
    let s7_3 = COSPI_16_64 * (x6_3 - x7_3);
    let x2_4 = dct_round_shift(s2_3);
    let x3_4 = dct_round_shift(s3_3);
    let x6_4 = dct_round_shift(s6_3);
    let x7_4 = dct_round_shift(s7_3);
    output[0] = x0_3;
    output[1] = -x4_3;
    output[2] = x6_4;
    output[3] = -x2_4;
    output[4] = x3_4;
    output[5] = -x7_4;
    output[6] = x5_3;
    output[7] = -x1_3;
    output
}

/// The AV1/VP9-lineage 16-point inverse ADST partial butterfly.
fn inverse_adst16_1d(input: [i64; 16]) -> [i64; 16] {
    let mut output = [0i64; 16];
    let x0 = input[15];
    let x1 = input[0];
    let x2 = input[13];
    let x3 = input[2];
    let x4 = input[11];
    let x5 = input[4];
    let x6 = input[9];
    let x7 = input[6];
    let x8 = input[7];
    let x9 = input[8];
    let x10 = input[5];
    let x11 = input[10];
    let x12 = input[3];
    let x13 = input[12];
    let x14 = input[1];
    let x15 = input[14];
    // stage 1
    let s0 = (x0 * COSPI_1_64) + (x1 * COSPI_31_64);
    let s1 = (x0 * COSPI_31_64) - (x1 * COSPI_1_64);
    let s2 = (x2 * COSPI_5_64) + (x3 * COSPI_27_64);
    let s3 = (x2 * COSPI_27_64) - (x3 * COSPI_5_64);
    let s4 = (x4 * COSPI_9_64) + (x5 * COSPI_23_64);
    let s5 = (x4 * COSPI_23_64) - (x5 * COSPI_9_64);
    let s6 = (x6 * COSPI_13_64) + (x7 * COSPI_19_64);
    let s7 = (x6 * COSPI_19_64) - (x7 * COSPI_13_64);
    let s8 = (x8 * COSPI_17_64) + (x9 * COSPI_15_64);
    let s9 = (x8 * COSPI_15_64) - (x9 * COSPI_17_64);
    let s10 = (x10 * COSPI_21_64) + (x11 * COSPI_11_64);
    let s11 = (x10 * COSPI_11_64) - (x11 * COSPI_21_64);
    let s12 = (x12 * COSPI_25_64) + (x13 * COSPI_7_64);
    let s13 = (x12 * COSPI_7_64) - (x13 * COSPI_25_64);
    let s14 = (x14 * COSPI_29_64) + (x15 * COSPI_3_64);
    let s15 = (x14 * COSPI_3_64) - (x15 * COSPI_29_64);
    let x0_2 = dct_round_shift(s0 + s8);
    let x1_2 = dct_round_shift(s1 + s9);
    let x2_2 = dct_round_shift(s2 + s10);
    let x3_2 = dct_round_shift(s3 + s11);
    let x4_2 = dct_round_shift(s4 + s12);
    let x5_2 = dct_round_shift(s5 + s13);
    let x6_2 = dct_round_shift(s6 + s14);
    let x7_2 = dct_round_shift(s7 + s15);
    let x8_2 = dct_round_shift(s0 - s8);
    let x9_2 = dct_round_shift(s1 - s9);
    let x10_2 = dct_round_shift(s2 - s10);
    let x11_2 = dct_round_shift(s3 - s11);
    let x12_2 = dct_round_shift(s4 - s12);
    let x13_2 = dct_round_shift(s5 - s13);
    let x14_2 = dct_round_shift(s6 - s14);
    let x15_2 = dct_round_shift(s7 - s15);
    // stage 2
    let s0_2 = x0_2;
    let s1_2 = x1_2;
    let s2_2 = x2_2;
    let s3_2 = x3_2;
    let s4_2 = x4_2;
    let s5_2 = x5_2;
    let s6_2 = x6_2;
    let s7_2 = x7_2;
    let s8_2 = (x8_2 * COSPI_4_64) + (x9_2 * COSPI_28_64);
    let s9_2 = (x8_2 * COSPI_28_64) - (x9_2 * COSPI_4_64);
    let s10_2 = (x10_2 * COSPI_20_64) + (x11_2 * COSPI_12_64);
    let s11_2 = (x10_2 * COSPI_12_64) - (x11_2 * COSPI_20_64);
    let s12_2 = (-x12_2 * COSPI_28_64) + (x13_2 * COSPI_4_64);
    let s13_2 = (x12_2 * COSPI_4_64) + (x13_2 * COSPI_28_64);
    let s14_2 = (-x14_2 * COSPI_12_64) + (x15_2 * COSPI_20_64);
    let s15_2 = (x14_2 * COSPI_20_64) + (x15_2 * COSPI_12_64);
    let x0_3 = s0_2 + s4_2;
    let x1_3 = s1_2 + s5_2;
    let x2_3 = s2_2 + s6_2;
    let x3_3 = s3_2 + s7_2;
    let x4_3 = s0_2 - s4_2;
    let x5_3 = s1_2 - s5_2;
    let x6_3 = s2_2 - s6_2;
    let x7_3 = s3_2 - s7_2;
    let x8_3 = dct_round_shift(s8_2 + s12_2);
    let x9_3 = dct_round_shift(s9_2 + s13_2);
    let x10_3 = dct_round_shift(s10_2 + s14_2);
    let x11_3 = dct_round_shift(s11_2 + s15_2);
    let x12_3 = dct_round_shift(s8_2 - s12_2);
    let x13_3 = dct_round_shift(s9_2 - s13_2);
    let x14_3 = dct_round_shift(s10_2 - s14_2);
    let x15_3 = dct_round_shift(s11_2 - s15_2);
    // stage 3
    let s0_3 = x0_3;
    let s1_3 = x1_3;
    let s2_3 = x2_3;
    let s3_3 = x3_3;
    let s4_3 = (x4_3 * COSPI_8_64) + (x5_3 * COSPI_24_64);
    let s5_3 = (x4_3 * COSPI_24_64) - (x5_3 * COSPI_8_64);
    let s6_3 = (-x6_3 * COSPI_24_64) + (x7_3 * COSPI_8_64);
    let s7_3 = (x6_3 * COSPI_8_64) + (x7_3 * COSPI_24_64);
    let s8_3 = x8_3;
    let s9_3 = x9_3;
    let s10_3 = x10_3;
    let s11_3 = x11_3;
    let s12_3 = (x12_3 * COSPI_8_64) + (x13_3 * COSPI_24_64);
    let s13_3 = (x12_3 * COSPI_24_64) - (x13_3 * COSPI_8_64);
    let s14_3 = (-x14_3 * COSPI_24_64) + (x15_3 * COSPI_8_64);
    let s15_3 = (x14_3 * COSPI_8_64) + (x15_3 * COSPI_24_64);
    let x0_4 = s0_3 + s2_3;
    let x1_4 = s1_3 + s3_3;
    let x2_4 = s0_3 - s2_3;
    let x3_4 = s1_3 - s3_3;
    let x4_4 = dct_round_shift(s4_3 + s6_3);
    let x5_4 = dct_round_shift(s5_3 + s7_3);
    let x6_4 = dct_round_shift(s4_3 - s6_3);
    let x7_4 = dct_round_shift(s5_3 - s7_3);
    let x8_4 = s8_3 + s10_3;
    let x9_4 = s9_3 + s11_3;
    let x10_4 = s8_3 - s10_3;
    let x11_4 = s9_3 - s11_3;
    let x12_4 = dct_round_shift(s12_3 + s14_3);
    let x13_4 = dct_round_shift(s13_3 + s15_3);
    let x14_4 = dct_round_shift(s12_3 - s14_3);
    let x15_4 = dct_round_shift(s13_3 - s15_3);
    // stage 4
    let s2_4 = -COSPI_16_64 * (x2_4 + x3_4);
    let s3_4 = COSPI_16_64 * (x2_4 - x3_4);
    let s6_4 = COSPI_16_64 * (x6_4 + x7_4);
    let s7_4 = COSPI_16_64 * (-x6_4 + x7_4);
    let s10_4 = COSPI_16_64 * (x10_4 + x11_4);
    let s11_4 = COSPI_16_64 * (-x10_4 + x11_4);
    let s14_4 = -COSPI_16_64 * (x14_4 + x15_4);
    let s15_4 = COSPI_16_64 * (x14_4 - x15_4);
    let x2_5 = dct_round_shift(s2_4);
    let x3_5 = dct_round_shift(s3_4);
    let x6_5 = dct_round_shift(s6_4);
    let x7_5 = dct_round_shift(s7_4);
    let x10_5 = dct_round_shift(s10_4);
    let x11_5 = dct_round_shift(s11_4);
    let x14_5 = dct_round_shift(s14_4);
    let x15_5 = dct_round_shift(s15_4);
    output[0] = x0_4;
    output[1] = -x8_4;
    output[2] = x12_4;
    output[3] = -x4_4;
    output[4] = x6_5;
    output[5] = x14_5;
    output[6] = x10_5;
    output[7] = x2_5;
    output[8] = x3_5;
    output[9] = x11_5;
    output[10] = x15_5;
    output[11] = x7_5;
    output[12] = x5_4;
    output[13] = -x13_4;
    output[14] = x9_4;
    output[15] = -x1_4;
    output
}

/// One separable 1-D pass. Falls back to the identity for a size the AV1
/// transform set does not define, which [`inverse_transform`] rejects first.
pub(crate) fn inverse_transform_1d(kind: Tx1d, values: &[i64]) -> Vec<i64> {
    match (kind, values.len()) {
        (Tx1d::Dct, 4) => inverse_dct4_1d(values.try_into().unwrap()).to_vec(),
        (Tx1d::Dct, 8) => inverse_dct8_1d(values.try_into().unwrap()).to_vec(),
        (Tx1d::Dct, 16) => inverse_dct16_1d(values.try_into().unwrap()).to_vec(),
        (Tx1d::Dct, 32) => inverse_dct32_1d(values.try_into().unwrap()).to_vec(),
        (Tx1d::Adst, 4) => inverse_adst4_1d(values.try_into().unwrap()).to_vec(),
        (Tx1d::Adst, 8) => inverse_adst8_1d(values.try_into().unwrap()).to_vec(),
        (Tx1d::Adst, 16) => inverse_adst16_1d(values.try_into().unwrap()).to_vec(),
        _ => values.to_vec(),
    }
}

/// Downshift applied to the column pass output for a `size x size` block.
pub(crate) fn transform_shift(size: usize) -> u32 {
    match size {
        4 => 4,
        8 => 5,
        _ => 6,
    }
}

/// Dequantizes and applies the non-lossless inverse transform (spec
/// §7.12.3, §7.13) for one `size x size` transform block, returning
/// row-major residual samples.
///
/// `size` must be 4, 8, 16, or 32. The ADST kernels are only defined for 4,
/// 8, and 16 points, so a 32-point block runs the DCT along both axes
/// regardless of `tx_type`. An unsupported `size` leaves the dequantized
/// coefficients untransformed rather than panicking.
pub fn inverse_transform(
    coefficients: &[i32],
    size: usize,
    tx_type: Av1TxType,
    dc_quant: i32,
    ac_quant: i32,
) -> Vec<i16> {
    debug_assert_eq!(coefficients.len(), size * size);
    debug_assert!(matches!(size, 4 | 8 | 16 | 32));
    let mut dequantized = vec![0i64; size * size];
    for (index, value) in dequantized.iter_mut().enumerate() {
        let quant = if index == 0 { dc_quant } else { ac_quant };
        *value = i64::from(coefficients[index]) * i64::from(quant);
    }
    if tx_type == Av1TxType::Idtx {
        // The 2-D identity is the one type with no butterfly pass and no
        // output downshift, so it keeps its own short path.
        return dequantized
            .into_iter()
            .map(|value| value.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16)
            .collect();
    }

    let (mut column, mut row, lr_flip, ud_flip) = tx_type.kernels();
    if size == 32 {
        column = Tx1d::Dct;
        row = Tx1d::Dct;
    }

    // The vectorized kernels work in 32-bit lanes and decline blocks whose
    // magnitudes could overflow one, so a coefficient the guard rejects
    // simply falls through to the scalar butterflies below.
    let narrow: Option<Vec<i32>> = dequantized
        .iter()
        .map(|&value| i32::try_from(value).ok())
        .collect();
    if let Some(values) = narrow {
        let mut output = vec![0i16; size * size];
        if crate::av1_simd::inverse_transform(
            crate::av1_simd::active_isa(),
            &values,
            size,
            column,
            row,
            lr_flip,
            ud_flip,
            &mut output,
        ) {
            return output;
        }
    }

    let mut rows = vec![0i64; size * size];
    for index in 0..size {
        let start = index * size;
        let transformed = inverse_transform_1d(row, &dequantized[start..start + size]);
        rows[start..start + size].copy_from_slice(&transformed);
    }
    let shift = transform_shift(size);
    let mut output = vec![0i16; size * size];
    for column_index in 0..size {
        let values: Vec<i64> = (0..size).map(|r| rows[r * size + column_index]).collect();
        let transformed = inverse_transform_1d(column, &values);
        let target_column = if lr_flip {
            size - 1 - column_index
        } else {
            column_index
        };
        for (row_index, value) in transformed.into_iter().enumerate() {
            let rounded = (value + (1 << (shift - 1))) >> shift;
            let target_row = if ud_flip {
                size - 1 - row_index
            } else {
                row_index
            };
            output[target_row * size + target_column] =
                rounded.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16;
        }
    }
    output
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
