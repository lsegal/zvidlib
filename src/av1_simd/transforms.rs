//! Vectorized AV1 transform butterfly passes.
//!
//! Covers the lossless 4x4 Walsh-Hadamard transform in both directions
//! ([`crate::av1_encoder`]) and the whole non-lossless inverse transform set
//! used by [`crate::av1_intra::inverse_transform`]: the 4-, 8-, 16-, 32-, and
//! 64-point inverse DCT and the 4-, 8-, and 16-point inverse ADST, in every
//! vertical/horizontal combination including the flipped-ADST output
//! reversals.
//!
//! Both scalar references compute in `i64`. These kernels compute in `i32`
//! lanes, which is only bit-exact while no intermediate overflows, so each
//! dispatcher first checks its input against a magnitude bound (see
//! [`WHT_INPUT_LIMIT`] and [`input_limit`]) and falls back to the scalar path
//! otherwise. Real bitstream coefficients sit orders of magnitude below those
//! bounds; the guard exists so a hostile or malformed stream cannot make the
//! two paths disagree.
//!
//! # Accumulator width
//!
//! The obvious `i32` transcription of `dct_round_shift(a * K + b * L)` needs
//! the *products* to fit a lane, which caps the usable input at about 2^15
//! and pushed every wide-dynamic-range block onto the scalar path. Instead,
//! [`dot_rs14`] splits each operand as `x = (x >> 14) * 2^14 + (x & 0x3fff)`
//! and accumulates the two halves separately, so
//!
//! ```text
//! (sum(x_i * k_i) + 2^13) >> 14  ==  sum((x_i >> 14) * k_i)
//!                                    + ((sum((x_i & 0x3fff) * k_i) + 2^13) >> 14)
//! ```
//!
//! holds exactly with no product ever leaving 32 bits. Only the *result* of
//! each butterfly stage has to fit a lane now, which is what [`input_limit`]
//! bounds, so the vectorized range is roughly three orders of magnitude wider
//! than before and covers every coefficient a conformant stream can carry.
//!
//! Data flow is the standard separable shape: each pass gathers four rows
//! into one lane apiece with 4x4 transposes, runs the butterfly, and writes
//! its output transposed, so two passes return the block to row-major order
//! with no separate transpose step.

use super::vector::{I32x, Transpose4};
use crate::av1_encoder::transform::{
    FADST4, FADST8, FADST16, FDCT4, FDCT8, FDCT16, FDCT32, staged_limit as forward_staged_limit,
};
use crate::av1_intra::{Tx1d, identity_scale};

/// Largest input magnitude for which the `i32` Walsh-Hadamard butterflies
/// cannot overflow. The transform grows a value by at most a factor of 4 per
/// pass over two passes plus the `* 4` dequant, so `2^18` leaves 2^26 headroom.
pub(crate) const WHT_INPUT_LIMIT: i32 = 1 << 18;

/// Upper bound on how much one 1-D pass can grow a value at each transform
/// size: `max_n sum_k |basis(n, k)|`, rounded up. The DCT and ADST kernels of
/// a given size have close enough gains to share one bound.
fn pass_gain(size: usize) -> i32 {
    match size {
        4 => 4,
        8 => 7,
        16 => 12,
        32 => 22,
        // The 64-point gain is `40.95`, the same `2N/pi` trend the smaller
        // sizes follow, with the same ~7% margin the 32-point bound carries.
        _ => 44,
    }
}

/// Largest dequantized coefficient magnitude for which every `i32`
/// intermediate of both passes stays in range.
///
/// Two passes multiply magnitudes by at most `pass_gain(size)` each, and the
/// spare factor of four covers the rounding slack [`dot_rs14`] adds to its
/// high half. Compare the old product-limited bound of 30000: this is 2^25 at
/// 4 points, still 2^20 at 32 points, and 2^18 (277309) at 64.
pub(crate) fn input_limit(size: usize) -> i32 {
    let gain = pass_gain(size);
    (i32::MAX / 4) / (gain * gain)
}

/// Largest magnitude the row pass may leave behind for the column pass to
/// stay in range. Checked at run time so the bound above is verified rather
/// than merely argued.
pub(crate) fn staged_limit(size: usize) -> i32 {
    (i32::MAX / 4) / pass_gain(size)
}

const COSPI_1_64: i32 = 16364;
const COSPI_2_64: i32 = 16305;
const COSPI_3_64: i32 = 16207;
const COSPI_4_64: i32 = 16069;
const COSPI_5_64: i32 = 15893;
const COSPI_6_64: i32 = 15679;
const COSPI_7_64: i32 = 15426;
const COSPI_8_64: i32 = 15137;
const COSPI_9_64: i32 = 14811;
const COSPI_10_64: i32 = 14449;
const COSPI_11_64: i32 = 14053;
const COSPI_12_64: i32 = 13623;
const COSPI_13_64: i32 = 13160;
const COSPI_14_64: i32 = 12665;
const COSPI_15_64: i32 = 12140;
const COSPI_16_64: i32 = 11585;
const COSPI_17_64: i32 = 11003;
const COSPI_18_64: i32 = 10394;
const COSPI_19_64: i32 = 9760;
const COSPI_20_64: i32 = 9102;
const COSPI_21_64: i32 = 8423;
const COSPI_22_64: i32 = 7723;
const COSPI_23_64: i32 = 7005;
const COSPI_24_64: i32 = 6270;
const COSPI_25_64: i32 = 5520;
const COSPI_26_64: i32 = 4756;
const COSPI_27_64: i32 = 3981;
const COSPI_28_64: i32 = 3196;
const COSPI_29_64: i32 = 2404;
const COSPI_30_64: i32 = 1606;
const COSPI_31_64: i32 = 804;
const SINPI_1_9: i32 = 5283;
const SINPI_2_9: i32 = 9929;
const SINPI_3_9: i32 = 13377;
const SINPI_4_9: i32 = 15212;

/// True when every value is small enough for the `i32` kernels to stay exact.
pub(crate) fn within_limit(values: &[i32], limit: i32) -> bool {
    let limit = limit as u32;
    values.iter().all(|&v| v.unsigned_abs() <= limit)
}

// ---------------------------------------------------------------------
// Butterflies (one lane per independent row/column)
// ---------------------------------------------------------------------

#[inline(always)]
unsafe fn round_shift14<V: I32x>(value: V) -> V {
    unsafe { value.add(V::splat(1 << 13)).sra::<14>() }
}

/// Bit-exact vector form of `av1_intra::dct_round_shift(sum(x_i * k_i))`.
///
/// See the accumulator-width note in the module docs: splitting each operand
/// around bit 14 keeps every product inside a lane no matter how large the
/// operands are, so only the result has to fit. At most seven terms ever
/// reach here, bounding the low half by `7 * 2^28` and leaving it in range.
#[inline(always)]
unsafe fn dot_rs14<V: I32x, const N: usize>(terms: [(V, i32); N]) -> V {
    unsafe {
        let mask = V::splat(0x3fff);
        let mut high = V::zero();
        let mut low = V::zero();
        for (value, coefficient) in terms {
            let scale = V::splat(coefficient);
            high = high.add(value.sra::<14>().mul(scale));
            low = low.add(value.and(mask).mul(scale));
        }
        high.add(round_shift14(low))
    }
}

/// Vector form of `av1_intra::inverse_dct8_1d`.
#[inline(always)]
unsafe fn dct8<V: I32x>(input: [V; 8]) -> [V; 8] {
    unsafe {
        let mut output = [V::zero(); 8];
        // stage 1
        let step1_0 = input[0];
        let step1_2 = input[4];
        let step1_1 = input[2];
        let step1_3 = input[6];
        let step1_4 = dot_rs14([(input[1], COSPI_28_64), (input[7], -COSPI_4_64)]);
        let step1_7 = dot_rs14([(input[1], COSPI_4_64), (input[7], COSPI_28_64)]);
        let step1_5 = dot_rs14([(input[5], COSPI_12_64), (input[3], -COSPI_20_64)]);
        let step1_6 = dot_rs14([(input[5], COSPI_20_64), (input[3], COSPI_12_64)]);
        // stage 2
        let step2_0 = dot_rs14([(step1_0, COSPI_16_64), (step1_2, COSPI_16_64)]);
        let step2_1 = dot_rs14([(step1_0, COSPI_16_64), (step1_2, -COSPI_16_64)]);
        let step2_2 = dot_rs14([(step1_1, COSPI_24_64), (step1_3, -COSPI_8_64)]);
        let step2_3 = dot_rs14([(step1_1, COSPI_8_64), (step1_3, COSPI_24_64)]);
        let step2_4 = step1_4.add(step1_5);
        let step2_5 = step1_4.sub(step1_5);
        let step2_6 = V::zero().sub(step1_6).add(step1_7);
        let step2_7 = step1_6.add(step1_7);
        // stage 3
        let step1_0_2 = step2_0.add(step2_3);
        let step1_1_2 = step2_1.add(step2_2);
        let step1_2_2 = step2_1.sub(step2_2);
        let step1_3_2 = step2_0.sub(step2_3);
        let step1_4_2 = step2_4;
        let step1_5_2 = dot_rs14([(step2_6, COSPI_16_64), (step2_5, -COSPI_16_64)]);
        let step1_6_2 = dot_rs14([(step2_5, COSPI_16_64), (step2_6, COSPI_16_64)]);
        let step1_7_2 = step2_7;
        // stage 4
        output[0] = step1_0_2.add(step1_7_2);
        output[1] = step1_1_2.add(step1_6_2);
        output[2] = step1_2_2.add(step1_5_2);
        output[3] = step1_3_2.add(step1_4_2);
        output[4] = step1_3_2.sub(step1_4_2);
        output[5] = step1_2_2.sub(step1_5_2);
        output[6] = step1_1_2.sub(step1_6_2);
        output[7] = step1_0_2.sub(step1_7_2);
        output
    }
}

/// Vector form of `av1_intra::inverse_dct4_1d`.
#[inline(always)]
unsafe fn dct4<V: I32x>(input: [V; 4]) -> [V; 4] {
    unsafe {
        let mut output = [V::zero(); 4];
        // stage 1
        let step_0 = dot_rs14([(input[0], COSPI_16_64), (input[2], COSPI_16_64)]);
        let step_1 = dot_rs14([(input[0], COSPI_16_64), (input[2], -COSPI_16_64)]);
        let step_2 = dot_rs14([(input[1], COSPI_24_64), (input[3], -COSPI_8_64)]);
        let step_3 = dot_rs14([(input[1], COSPI_8_64), (input[3], COSPI_24_64)]);
        // stage 2
        output[0] = step_0.add(step_3);
        output[1] = step_1.add(step_2);
        output[2] = step_1.sub(step_2);
        output[3] = step_0.sub(step_3);
        output
    }
}

/// Vector form of `av1_intra::inverse_dct16_1d`.
#[inline(always)]
unsafe fn dct16<V: I32x>(input: [V; 16]) -> [V; 16] {
    unsafe {
        let mut output = [V::zero(); 16];
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
        let step2_8 = dot_rs14([(step1_8, COSPI_30_64), (step1_15, -COSPI_2_64)]);
        let step2_15 = dot_rs14([(step1_8, COSPI_2_64), (step1_15, COSPI_30_64)]);
        let step2_9 = dot_rs14([(step1_9, COSPI_14_64), (step1_14, -COSPI_18_64)]);
        let step2_14 = dot_rs14([(step1_9, COSPI_18_64), (step1_14, COSPI_14_64)]);
        let step2_10 = dot_rs14([(step1_10, COSPI_22_64), (step1_13, -COSPI_10_64)]);
        let step2_13 = dot_rs14([(step1_10, COSPI_10_64), (step1_13, COSPI_22_64)]);
        let step2_11 = dot_rs14([(step1_11, COSPI_6_64), (step1_12, -COSPI_26_64)]);
        let step2_12 = dot_rs14([(step1_11, COSPI_26_64), (step1_12, COSPI_6_64)]);
        // stage 3
        let step1_0_2 = step2_0;
        let step1_1_2 = step2_1;
        let step1_2_2 = step2_2;
        let step1_3_2 = step2_3;
        let step1_4_2 = dot_rs14([(step2_4, COSPI_28_64), (step2_7, -COSPI_4_64)]);
        let step1_7_2 = dot_rs14([(step2_4, COSPI_4_64), (step2_7, COSPI_28_64)]);
        let step1_5_2 = dot_rs14([(step2_5, COSPI_12_64), (step2_6, -COSPI_20_64)]);
        let step1_6_2 = dot_rs14([(step2_5, COSPI_20_64), (step2_6, COSPI_12_64)]);
        let step1_8_2 = step2_8.add(step2_9);
        let step1_9_2 = step2_8.sub(step2_9);
        let step1_10_2 = V::zero().sub(step2_10).add(step2_11);
        let step1_11_2 = step2_10.add(step2_11);
        let step1_12_2 = step2_12.add(step2_13);
        let step1_13_2 = step2_12.sub(step2_13);
        let step1_14_2 = V::zero().sub(step2_14).add(step2_15);
        let step1_15_2 = step2_14.add(step2_15);
        // stage 4
        let step2_0_2 = dot_rs14([(step1_0_2, COSPI_16_64), (step1_1_2, COSPI_16_64)]);
        let step2_1_2 = dot_rs14([(step1_0_2, COSPI_16_64), (step1_1_2, -COSPI_16_64)]);
        let step2_2_2 = dot_rs14([(step1_2_2, COSPI_24_64), (step1_3_2, -COSPI_8_64)]);
        let step2_3_2 = dot_rs14([(step1_2_2, COSPI_8_64), (step1_3_2, COSPI_24_64)]);
        let step2_4_2 = step1_4_2.add(step1_5_2);
        let step2_5_2 = step1_4_2.sub(step1_5_2);
        let step2_6_2 = V::zero().sub(step1_6_2).add(step1_7_2);
        let step2_7_2 = step1_6_2.add(step1_7_2);
        let step2_8_2 = step1_8_2;
        let step2_15_2 = step1_15_2;
        let step2_9_2 = dot_rs14([(step1_9_2, -COSPI_8_64), (step1_14_2, COSPI_24_64)]);
        let step2_14_2 = dot_rs14([(step1_9_2, COSPI_24_64), (step1_14_2, COSPI_8_64)]);
        let step2_10_2 = dot_rs14([(step1_10_2, -COSPI_24_64), (step1_13_2, -COSPI_8_64)]);
        let step2_13_2 = dot_rs14([(step1_10_2, -COSPI_8_64), (step1_13_2, COSPI_24_64)]);
        let step2_11_2 = step1_11_2;
        let step2_12_2 = step1_12_2;
        // stage 5
        let step1_0_3 = step2_0_2.add(step2_3_2);
        let step1_1_3 = step2_1_2.add(step2_2_2);
        let step1_2_3 = step2_1_2.sub(step2_2_2);
        let step1_3_3 = step2_0_2.sub(step2_3_2);
        let step1_4_3 = step2_4_2;
        let step1_5_3 = dot_rs14([(step2_6_2, COSPI_16_64), (step2_5_2, -COSPI_16_64)]);
        let step1_6_3 = dot_rs14([(step2_5_2, COSPI_16_64), (step2_6_2, COSPI_16_64)]);
        let step1_7_3 = step2_7_2;
        let step1_8_3 = step2_8_2.add(step2_11_2);
        let step1_9_3 = step2_9_2.add(step2_10_2);
        let step1_10_3 = step2_9_2.sub(step2_10_2);
        let step1_11_3 = step2_8_2.sub(step2_11_2);
        let step1_12_3 = V::zero().sub(step2_12_2).add(step2_15_2);
        let step1_13_3 = V::zero().sub(step2_13_2).add(step2_14_2);
        let step1_14_3 = step2_13_2.add(step2_14_2);
        let step1_15_3 = step2_12_2.add(step2_15_2);
        // stage 6
        let step2_0_3 = step1_0_3.add(step1_7_3);
        let step2_1_3 = step1_1_3.add(step1_6_3);
        let step2_2_3 = step1_2_3.add(step1_5_3);
        let step2_3_3 = step1_3_3.add(step1_4_3);
        let step2_4_3 = step1_3_3.sub(step1_4_3);
        let step2_5_3 = step1_2_3.sub(step1_5_3);
        let step2_6_3 = step1_1_3.sub(step1_6_3);
        let step2_7_3 = step1_0_3.sub(step1_7_3);
        let step2_8_3 = step1_8_3;
        let step2_9_3 = step1_9_3;
        let step2_10_3 = dot_rs14([(step1_10_3, -COSPI_16_64), (step1_13_3, COSPI_16_64)]);
        let step2_13_3 = dot_rs14([(step1_10_3, COSPI_16_64), (step1_13_3, COSPI_16_64)]);
        let step2_11_3 = dot_rs14([(step1_11_3, -COSPI_16_64), (step1_12_3, COSPI_16_64)]);
        let step2_12_3 = dot_rs14([(step1_11_3, COSPI_16_64), (step1_12_3, COSPI_16_64)]);
        let step2_14_3 = step1_14_3;
        let step2_15_3 = step1_15_3;
        // stage 7
        output[0] = step2_0_3.add(step2_15_3);
        output[1] = step2_1_3.add(step2_14_3);
        output[2] = step2_2_3.add(step2_13_3);
        output[3] = step2_3_3.add(step2_12_3);
        output[4] = step2_4_3.add(step2_11_3);
        output[5] = step2_5_3.add(step2_10_3);
        output[6] = step2_6_3.add(step2_9_3);
        output[7] = step2_7_3.add(step2_8_3);
        output[8] = step2_7_3.sub(step2_8_3);
        output[9] = step2_6_3.sub(step2_9_3);
        output[10] = step2_5_3.sub(step2_10_3);
        output[11] = step2_4_3.sub(step2_11_3);
        output[12] = step2_3_3.sub(step2_12_3);
        output[13] = step2_2_3.sub(step2_13_3);
        output[14] = step2_1_3.sub(step2_14_3);
        output[15] = step2_0_3.sub(step2_15_3);
        output
    }
}

/// Vector form of `av1_intra::inverse_dct32_1d`.
#[inline(always)]
unsafe fn dct32<V: I32x>(input: [V; 32]) -> [V; 32] {
    unsafe {
        let mut output = [V::zero(); 32];
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
        let step1_16 = dot_rs14([(input[1], COSPI_31_64), (input[31], -COSPI_1_64)]);
        let step1_31 = dot_rs14([(input[1], COSPI_1_64), (input[31], COSPI_31_64)]);
        let step1_17 = dot_rs14([(input[17], COSPI_15_64), (input[15], -COSPI_17_64)]);
        let step1_30 = dot_rs14([(input[17], COSPI_17_64), (input[15], COSPI_15_64)]);
        let step1_18 = dot_rs14([(input[9], COSPI_23_64), (input[23], -COSPI_9_64)]);
        let step1_29 = dot_rs14([(input[9], COSPI_9_64), (input[23], COSPI_23_64)]);
        let step1_19 = dot_rs14([(input[25], COSPI_7_64), (input[7], -COSPI_25_64)]);
        let step1_28 = dot_rs14([(input[25], COSPI_25_64), (input[7], COSPI_7_64)]);
        let step1_20 = dot_rs14([(input[5], COSPI_27_64), (input[27], -COSPI_5_64)]);
        let step1_27 = dot_rs14([(input[5], COSPI_5_64), (input[27], COSPI_27_64)]);
        let step1_21 = dot_rs14([(input[21], COSPI_11_64), (input[11], -COSPI_21_64)]);
        let step1_26 = dot_rs14([(input[21], COSPI_21_64), (input[11], COSPI_11_64)]);
        let step1_22 = dot_rs14([(input[13], COSPI_19_64), (input[19], -COSPI_13_64)]);
        let step1_25 = dot_rs14([(input[13], COSPI_13_64), (input[19], COSPI_19_64)]);
        let step1_23 = dot_rs14([(input[29], COSPI_3_64), (input[3], -COSPI_29_64)]);
        let step1_24 = dot_rs14([(input[29], COSPI_29_64), (input[3], COSPI_3_64)]);
        // stage 2
        let step2_0 = step1_0;
        let step2_1 = step1_1;
        let step2_2 = step1_2;
        let step2_3 = step1_3;
        let step2_4 = step1_4;
        let step2_5 = step1_5;
        let step2_6 = step1_6;
        let step2_7 = step1_7;
        let step2_8 = dot_rs14([(step1_8, COSPI_30_64), (step1_15, -COSPI_2_64)]);
        let step2_15 = dot_rs14([(step1_8, COSPI_2_64), (step1_15, COSPI_30_64)]);
        let step2_9 = dot_rs14([(step1_9, COSPI_14_64), (step1_14, -COSPI_18_64)]);
        let step2_14 = dot_rs14([(step1_9, COSPI_18_64), (step1_14, COSPI_14_64)]);
        let step2_10 = dot_rs14([(step1_10, COSPI_22_64), (step1_13, -COSPI_10_64)]);
        let step2_13 = dot_rs14([(step1_10, COSPI_10_64), (step1_13, COSPI_22_64)]);
        let step2_11 = dot_rs14([(step1_11, COSPI_6_64), (step1_12, -COSPI_26_64)]);
        let step2_12 = dot_rs14([(step1_11, COSPI_26_64), (step1_12, COSPI_6_64)]);
        let step2_16 = step1_16.add(step1_17);
        let step2_17 = step1_16.sub(step1_17);
        let step2_18 = V::zero().sub(step1_18).add(step1_19);
        let step2_19 = step1_18.add(step1_19);
        let step2_20 = step1_20.add(step1_21);
        let step2_21 = step1_20.sub(step1_21);
        let step2_22 = V::zero().sub(step1_22).add(step1_23);
        let step2_23 = step1_22.add(step1_23);
        let step2_24 = step1_24.add(step1_25);
        let step2_25 = step1_24.sub(step1_25);
        let step2_26 = V::zero().sub(step1_26).add(step1_27);
        let step2_27 = step1_26.add(step1_27);
        let step2_28 = step1_28.add(step1_29);
        let step2_29 = step1_28.sub(step1_29);
        let step2_30 = V::zero().sub(step1_30).add(step1_31);
        let step2_31 = step1_30.add(step1_31);
        // stage 3
        let step1_0_2 = step2_0;
        let step1_1_2 = step2_1;
        let step1_2_2 = step2_2;
        let step1_3_2 = step2_3;
        let step1_4_2 = dot_rs14([(step2_4, COSPI_28_64), (step2_7, -COSPI_4_64)]);
        let step1_7_2 = dot_rs14([(step2_4, COSPI_4_64), (step2_7, COSPI_28_64)]);
        let step1_5_2 = dot_rs14([(step2_5, COSPI_12_64), (step2_6, -COSPI_20_64)]);
        let step1_6_2 = dot_rs14([(step2_5, COSPI_20_64), (step2_6, COSPI_12_64)]);
        let step1_8_2 = step2_8.add(step2_9);
        let step1_9_2 = step2_8.sub(step2_9);
        let step1_10_2 = V::zero().sub(step2_10).add(step2_11);
        let step1_11_2 = step2_10.add(step2_11);
        let step1_12_2 = step2_12.add(step2_13);
        let step1_13_2 = step2_12.sub(step2_13);
        let step1_14_2 = V::zero().sub(step2_14).add(step2_15);
        let step1_15_2 = step2_14.add(step2_15);
        let step1_16_2 = step2_16;
        let step1_31_2 = step2_31;
        let step1_17_2 = dot_rs14([(step2_17, -COSPI_4_64), (step2_30, COSPI_28_64)]);
        let step1_30_2 = dot_rs14([(step2_17, COSPI_28_64), (step2_30, COSPI_4_64)]);
        let step1_18_2 = dot_rs14([(step2_18, -COSPI_28_64), (step2_29, -COSPI_4_64)]);
        let step1_29_2 = dot_rs14([(step2_18, -COSPI_4_64), (step2_29, COSPI_28_64)]);
        let step1_19_2 = step2_19;
        let step1_20_2 = step2_20;
        let step1_21_2 = dot_rs14([(step2_21, -COSPI_20_64), (step2_26, COSPI_12_64)]);
        let step1_26_2 = dot_rs14([(step2_21, COSPI_12_64), (step2_26, COSPI_20_64)]);
        let step1_22_2 = dot_rs14([(step2_22, -COSPI_12_64), (step2_25, -COSPI_20_64)]);
        let step1_25_2 = dot_rs14([(step2_22, -COSPI_20_64), (step2_25, COSPI_12_64)]);
        let step1_23_2 = step2_23;
        let step1_24_2 = step2_24;
        let step1_27_2 = step2_27;
        let step1_28_2 = step2_28;
        // stage 4
        let step2_0_2 = dot_rs14([(step1_0_2, COSPI_16_64), (step1_1_2, COSPI_16_64)]);
        let step2_1_2 = dot_rs14([(step1_0_2, COSPI_16_64), (step1_1_2, -COSPI_16_64)]);
        let step2_2_2 = dot_rs14([(step1_2_2, COSPI_24_64), (step1_3_2, -COSPI_8_64)]);
        let step2_3_2 = dot_rs14([(step1_2_2, COSPI_8_64), (step1_3_2, COSPI_24_64)]);
        let step2_4_2 = step1_4_2.add(step1_5_2);
        let step2_5_2 = step1_4_2.sub(step1_5_2);
        let step2_6_2 = V::zero().sub(step1_6_2).add(step1_7_2);
        let step2_7_2 = step1_6_2.add(step1_7_2);
        let step2_8_2 = step1_8_2;
        let step2_15_2 = step1_15_2;
        let step2_9_2 = dot_rs14([(step1_9_2, -COSPI_8_64), (step1_14_2, COSPI_24_64)]);
        let step2_14_2 = dot_rs14([(step1_9_2, COSPI_24_64), (step1_14_2, COSPI_8_64)]);
        let step2_10_2 = dot_rs14([(step1_10_2, -COSPI_24_64), (step1_13_2, -COSPI_8_64)]);
        let step2_13_2 = dot_rs14([(step1_10_2, -COSPI_8_64), (step1_13_2, COSPI_24_64)]);
        let step2_11_2 = step1_11_2;
        let step2_12_2 = step1_12_2;
        let step2_16_2 = step1_16_2.add(step1_19_2);
        let step2_17_2 = step1_17_2.add(step1_18_2);
        let step2_18_2 = step1_17_2.sub(step1_18_2);
        let step2_19_2 = step1_16_2.sub(step1_19_2);
        let step2_20_2 = V::zero().sub(step1_20_2).add(step1_23_2);
        let step2_21_2 = V::zero().sub(step1_21_2).add(step1_22_2);
        let step2_22_2 = step1_21_2.add(step1_22_2);
        let step2_23_2 = step1_20_2.add(step1_23_2);
        let step2_24_2 = step1_24_2.add(step1_27_2);
        let step2_25_2 = step1_25_2.add(step1_26_2);
        let step2_26_2 = step1_25_2.sub(step1_26_2);
        let step2_27_2 = step1_24_2.sub(step1_27_2);
        let step2_28_2 = V::zero().sub(step1_28_2).add(step1_31_2);
        let step2_29_2 = V::zero().sub(step1_29_2).add(step1_30_2);
        let step2_30_2 = step1_29_2.add(step1_30_2);
        let step2_31_2 = step1_28_2.add(step1_31_2);
        // stage 5
        let step1_0_3 = step2_0_2.add(step2_3_2);
        let step1_1_3 = step2_1_2.add(step2_2_2);
        let step1_2_3 = step2_1_2.sub(step2_2_2);
        let step1_3_3 = step2_0_2.sub(step2_3_2);
        let step1_4_3 = step2_4_2;
        let step1_5_3 = dot_rs14([(step2_6_2, COSPI_16_64), (step2_5_2, -COSPI_16_64)]);
        let step1_6_3 = dot_rs14([(step2_5_2, COSPI_16_64), (step2_6_2, COSPI_16_64)]);
        let step1_7_3 = step2_7_2;
        let step1_8_3 = step2_8_2.add(step2_11_2);
        let step1_9_3 = step2_9_2.add(step2_10_2);
        let step1_10_3 = step2_9_2.sub(step2_10_2);
        let step1_11_3 = step2_8_2.sub(step2_11_2);
        let step1_12_3 = V::zero().sub(step2_12_2).add(step2_15_2);
        let step1_13_3 = V::zero().sub(step2_13_2).add(step2_14_2);
        let step1_14_3 = step2_13_2.add(step2_14_2);
        let step1_15_3 = step2_12_2.add(step2_15_2);
        let step1_16_3 = step2_16_2;
        let step1_17_3 = step2_17_2;
        let step1_18_3 = dot_rs14([(step2_18_2, -COSPI_8_64), (step2_29_2, COSPI_24_64)]);
        let step1_29_3 = dot_rs14([(step2_18_2, COSPI_24_64), (step2_29_2, COSPI_8_64)]);
        let step1_19_3 = dot_rs14([(step2_19_2, -COSPI_8_64), (step2_28_2, COSPI_24_64)]);
        let step1_28_3 = dot_rs14([(step2_19_2, COSPI_24_64), (step2_28_2, COSPI_8_64)]);
        let step1_20_3 = dot_rs14([(step2_20_2, -COSPI_24_64), (step2_27_2, -COSPI_8_64)]);
        let step1_27_3 = dot_rs14([(step2_20_2, -COSPI_8_64), (step2_27_2, COSPI_24_64)]);
        let step1_21_3 = dot_rs14([(step2_21_2, -COSPI_24_64), (step2_26_2, -COSPI_8_64)]);
        let step1_26_3 = dot_rs14([(step2_21_2, -COSPI_8_64), (step2_26_2, COSPI_24_64)]);
        let step1_22_3 = step2_22_2;
        let step1_23_3 = step2_23_2;
        let step1_24_3 = step2_24_2;
        let step1_25_3 = step2_25_2;
        let step1_30_3 = step2_30_2;
        let step1_31_3 = step2_31_2;
        // stage 6
        let step2_0_3 = step1_0_3.add(step1_7_3);
        let step2_1_3 = step1_1_3.add(step1_6_3);
        let step2_2_3 = step1_2_3.add(step1_5_3);
        let step2_3_3 = step1_3_3.add(step1_4_3);
        let step2_4_3 = step1_3_3.sub(step1_4_3);
        let step2_5_3 = step1_2_3.sub(step1_5_3);
        let step2_6_3 = step1_1_3.sub(step1_6_3);
        let step2_7_3 = step1_0_3.sub(step1_7_3);
        let step2_8_3 = step1_8_3;
        let step2_9_3 = step1_9_3;
        let step2_10_3 = dot_rs14([(step1_10_3, -COSPI_16_64), (step1_13_3, COSPI_16_64)]);
        let step2_13_3 = dot_rs14([(step1_10_3, COSPI_16_64), (step1_13_3, COSPI_16_64)]);
        let step2_11_3 = dot_rs14([(step1_11_3, -COSPI_16_64), (step1_12_3, COSPI_16_64)]);
        let step2_12_3 = dot_rs14([(step1_11_3, COSPI_16_64), (step1_12_3, COSPI_16_64)]);
        let step2_14_3 = step1_14_3;
        let step2_15_3 = step1_15_3;
        let step2_16_3 = step1_16_3.add(step1_23_3);
        let step2_17_3 = step1_17_3.add(step1_22_3);
        let step2_18_3 = step1_18_3.add(step1_21_3);
        let step2_19_3 = step1_19_3.add(step1_20_3);
        let step2_20_3 = step1_19_3.sub(step1_20_3);
        let step2_21_3 = step1_18_3.sub(step1_21_3);
        let step2_22_3 = step1_17_3.sub(step1_22_3);
        let step2_23_3 = step1_16_3.sub(step1_23_3);
        let step2_24_3 = V::zero().sub(step1_24_3).add(step1_31_3);
        let step2_25_3 = V::zero().sub(step1_25_3).add(step1_30_3);
        let step2_26_3 = V::zero().sub(step1_26_3).add(step1_29_3);
        let step2_27_3 = V::zero().sub(step1_27_3).add(step1_28_3);
        let step2_28_3 = step1_27_3.add(step1_28_3);
        let step2_29_3 = step1_26_3.add(step1_29_3);
        let step2_30_3 = step1_25_3.add(step1_30_3);
        let step2_31_3 = step1_24_3.add(step1_31_3);
        // stage 7
        let step1_0_4 = step2_0_3.add(step2_15_3);
        let step1_1_4 = step2_1_3.add(step2_14_3);
        let step1_2_4 = step2_2_3.add(step2_13_3);
        let step1_3_4 = step2_3_3.add(step2_12_3);
        let step1_4_4 = step2_4_3.add(step2_11_3);
        let step1_5_4 = step2_5_3.add(step2_10_3);
        let step1_6_4 = step2_6_3.add(step2_9_3);
        let step1_7_4 = step2_7_3.add(step2_8_3);
        let step1_8_4 = step2_7_3.sub(step2_8_3);
        let step1_9_4 = step2_6_3.sub(step2_9_3);
        let step1_10_4 = step2_5_3.sub(step2_10_3);
        let step1_11_4 = step2_4_3.sub(step2_11_3);
        let step1_12_4 = step2_3_3.sub(step2_12_3);
        let step1_13_4 = step2_2_3.sub(step2_13_3);
        let step1_14_4 = step2_1_3.sub(step2_14_3);
        let step1_15_4 = step2_0_3.sub(step2_15_3);
        let step1_16_4 = step2_16_3;
        let step1_17_4 = step2_17_3;
        let step1_18_4 = step2_18_3;
        let step1_19_4 = step2_19_3;
        let step1_20_4 = dot_rs14([(step2_20_3, -COSPI_16_64), (step2_27_3, COSPI_16_64)]);
        let step1_27_4 = dot_rs14([(step2_20_3, COSPI_16_64), (step2_27_3, COSPI_16_64)]);
        let step1_21_4 = dot_rs14([(step2_21_3, -COSPI_16_64), (step2_26_3, COSPI_16_64)]);
        let step1_26_4 = dot_rs14([(step2_21_3, COSPI_16_64), (step2_26_3, COSPI_16_64)]);
        let step1_22_4 = dot_rs14([(step2_22_3, -COSPI_16_64), (step2_25_3, COSPI_16_64)]);
        let step1_25_4 = dot_rs14([(step2_22_3, COSPI_16_64), (step2_25_3, COSPI_16_64)]);
        let step1_23_4 = dot_rs14([(step2_23_3, -COSPI_16_64), (step2_24_3, COSPI_16_64)]);
        let step1_24_4 = dot_rs14([(step2_23_3, COSPI_16_64), (step2_24_3, COSPI_16_64)]);
        let step1_28_4 = step2_28_3;
        let step1_29_4 = step2_29_3;
        let step1_30_4 = step2_30_3;
        let step1_31_4 = step2_31_3;
        // final stage
        output[0] = step1_0_4.add(step1_31_4);
        output[1] = step1_1_4.add(step1_30_4);
        output[2] = step1_2_4.add(step1_29_4);
        output[3] = step1_3_4.add(step1_28_4);
        output[4] = step1_4_4.add(step1_27_4);
        output[5] = step1_5_4.add(step1_26_4);
        output[6] = step1_6_4.add(step1_25_4);
        output[7] = step1_7_4.add(step1_24_4);
        output[8] = step1_8_4.add(step1_23_4);
        output[9] = step1_9_4.add(step1_22_4);
        output[10] = step1_10_4.add(step1_21_4);
        output[11] = step1_11_4.add(step1_20_4);
        output[12] = step1_12_4.add(step1_19_4);
        output[13] = step1_13_4.add(step1_18_4);
        output[14] = step1_14_4.add(step1_17_4);
        output[15] = step1_15_4.add(step1_16_4);
        output[16] = step1_15_4.sub(step1_16_4);
        output[17] = step1_14_4.sub(step1_17_4);
        output[18] = step1_13_4.sub(step1_18_4);
        output[19] = step1_12_4.sub(step1_19_4);
        output[20] = step1_11_4.sub(step1_20_4);
        output[21] = step1_10_4.sub(step1_21_4);
        output[22] = step1_9_4.sub(step1_22_4);
        output[23] = step1_8_4.sub(step1_23_4);
        output[24] = step1_7_4.sub(step1_24_4);
        output[25] = step1_6_4.sub(step1_25_4);
        output[26] = step1_5_4.sub(step1_26_4);
        output[27] = step1_4_4.sub(step1_27_4);
        output[28] = step1_3_4.sub(step1_28_4);
        output[29] = step1_2_4.sub(step1_29_4);
        output[30] = step1_1_4.sub(step1_30_4);
        output[31] = step1_0_4.sub(step1_31_4);
        output
    }
}

/// Vector form of `av1_intra::inverse_adst4_1d`.
#[inline(always)]
unsafe fn adst4<V: I32x>(input: [V; 4]) -> [V; 4] {
    unsafe {
        let mut output = [V::zero(); 4];
        let x0 = input[0];
        let x1 = input[1];
        let x2 = input[2];
        let x3 = input[3];
        // 32-bit result is enough for the following multiplications.
        let s7 = x0.sub(x2).add(x3);
        // 1-D transform scaling factor is sqrt(2).
        // The overall dynamic range is 14b (input) + 14b (multiplication scaling)
        // + 1b (addition) = 29b.
        // Hence the output bit depth is 15b.
        output[0] = dot_rs14([
            (x0, SINPI_1_9),
            (x2, SINPI_4_9),
            (x3, SINPI_2_9),
            (x1, SINPI_3_9),
        ]);
        output[1] = dot_rs14([
            (x0, SINPI_2_9),
            (x2, -SINPI_1_9),
            (x3, -SINPI_4_9),
            (x1, SINPI_3_9),
        ]);
        output[2] = dot_rs14([(s7, SINPI_3_9)]);
        output[3] = dot_rs14([
            (x0, SINPI_1_9),
            (x2, SINPI_4_9),
            (x3, SINPI_2_9),
            (x0, SINPI_2_9),
            (x2, -SINPI_1_9),
            (x3, -SINPI_4_9),
            (x1, -SINPI_3_9),
        ]);
        output
    }
}

/// Vector form of `av1_intra::inverse_adst8_1d`.
#[inline(always)]
unsafe fn adst8<V: I32x>(input: [V; 8]) -> [V; 8] {
    unsafe {
        let mut output = [V::zero(); 8];
        let x0 = input[7];
        let x1 = input[0];
        let x2 = input[5];
        let x3 = input[2];
        let x4 = input[3];
        let x5 = input[4];
        let x6 = input[1];
        let x7 = input[6];
        // stage 1
        let x0_2 = dot_rs14([
            (x0, COSPI_2_64),
            (x1, COSPI_30_64),
            (x4, COSPI_18_64),
            (x5, COSPI_14_64),
        ]);
        let x1_2 = dot_rs14([
            (x0, COSPI_30_64),
            (x1, -COSPI_2_64),
            (x4, COSPI_14_64),
            (x5, -COSPI_18_64),
        ]);
        let x2_2 = dot_rs14([
            (x2, COSPI_10_64),
            (x3, COSPI_22_64),
            (x6, COSPI_26_64),
            (x7, COSPI_6_64),
        ]);
        let x3_2 = dot_rs14([
            (x2, COSPI_22_64),
            (x3, -COSPI_10_64),
            (x6, COSPI_6_64),
            (x7, -COSPI_26_64),
        ]);
        let x4_2 = dot_rs14([
            (x0, COSPI_2_64),
            (x1, COSPI_30_64),
            (x4, -COSPI_18_64),
            (x5, -COSPI_14_64),
        ]);
        let x5_2 = dot_rs14([
            (x0, COSPI_30_64),
            (x1, -COSPI_2_64),
            (x4, -COSPI_14_64),
            (x5, COSPI_18_64),
        ]);
        let x6_2 = dot_rs14([
            (x2, COSPI_10_64),
            (x3, COSPI_22_64),
            (x6, -COSPI_26_64),
            (x7, -COSPI_6_64),
        ]);
        let x7_2 = dot_rs14([
            (x2, COSPI_22_64),
            (x3, -COSPI_10_64),
            (x6, -COSPI_6_64),
            (x7, COSPI_26_64),
        ]);
        // stage 2
        let s0 = x0_2;
        let s1 = x1_2;
        let s2 = x2_2;
        let s3 = x3_2;
        let x0_3 = s0.add(s2);
        let x1_3 = s1.add(s3);
        let x2_3 = s0.sub(s2);
        let x3_3 = s1.sub(s3);
        let x4_3 = dot_rs14([
            (x4_2, COSPI_8_64),
            (x5_2, COSPI_24_64),
            (x6_2, -COSPI_24_64),
            (x7_2, COSPI_8_64),
        ]);
        let x5_3 = dot_rs14([
            (x4_2, COSPI_24_64),
            (x5_2, -COSPI_8_64),
            (x6_2, COSPI_8_64),
            (x7_2, COSPI_24_64),
        ]);
        let x6_3 = dot_rs14([
            (x4_2, COSPI_8_64),
            (x5_2, COSPI_24_64),
            (x6_2, COSPI_24_64),
            (x7_2, -COSPI_8_64),
        ]);
        let x7_3 = dot_rs14([
            (x4_2, COSPI_24_64),
            (x5_2, -COSPI_8_64),
            (x6_2, -COSPI_8_64),
            (x7_2, -COSPI_24_64),
        ]);
        // stage 3
        let x2_4 = dot_rs14([(x2_3, COSPI_16_64), (x3_3, COSPI_16_64)]);
        let x3_4 = dot_rs14([(x2_3, COSPI_16_64), (x3_3, -COSPI_16_64)]);
        let x6_4 = dot_rs14([(x6_3, COSPI_16_64), (x7_3, COSPI_16_64)]);
        let x7_4 = dot_rs14([(x6_3, COSPI_16_64), (x7_3, -COSPI_16_64)]);
        output[0] = x0_3;
        output[1] = V::zero().sub(x4_3);
        output[2] = x6_4;
        output[3] = V::zero().sub(x2_4);
        output[4] = x3_4;
        output[5] = V::zero().sub(x7_4);
        output[6] = x5_3;
        output[7] = V::zero().sub(x1_3);
        output
    }
}

/// Vector form of `av1_intra::inverse_adst16_1d`.
#[inline(always)]
unsafe fn adst16<V: I32x>(input: [V; 16]) -> [V; 16] {
    unsafe {
        let mut output = [V::zero(); 16];
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
        let x0_2 = dot_rs14([
            (x0, COSPI_1_64),
            (x1, COSPI_31_64),
            (x8, COSPI_17_64),
            (x9, COSPI_15_64),
        ]);
        let x1_2 = dot_rs14([
            (x0, COSPI_31_64),
            (x1, -COSPI_1_64),
            (x8, COSPI_15_64),
            (x9, -COSPI_17_64),
        ]);
        let x2_2 = dot_rs14([
            (x2, COSPI_5_64),
            (x3, COSPI_27_64),
            (x10, COSPI_21_64),
            (x11, COSPI_11_64),
        ]);
        let x3_2 = dot_rs14([
            (x2, COSPI_27_64),
            (x3, -COSPI_5_64),
            (x10, COSPI_11_64),
            (x11, -COSPI_21_64),
        ]);
        let x4_2 = dot_rs14([
            (x4, COSPI_9_64),
            (x5, COSPI_23_64),
            (x12, COSPI_25_64),
            (x13, COSPI_7_64),
        ]);
        let x5_2 = dot_rs14([
            (x4, COSPI_23_64),
            (x5, -COSPI_9_64),
            (x12, COSPI_7_64),
            (x13, -COSPI_25_64),
        ]);
        let x6_2 = dot_rs14([
            (x6, COSPI_13_64),
            (x7, COSPI_19_64),
            (x14, COSPI_29_64),
            (x15, COSPI_3_64),
        ]);
        let x7_2 = dot_rs14([
            (x6, COSPI_19_64),
            (x7, -COSPI_13_64),
            (x14, COSPI_3_64),
            (x15, -COSPI_29_64),
        ]);
        let x8_2 = dot_rs14([
            (x0, COSPI_1_64),
            (x1, COSPI_31_64),
            (x8, -COSPI_17_64),
            (x9, -COSPI_15_64),
        ]);
        let x9_2 = dot_rs14([
            (x0, COSPI_31_64),
            (x1, -COSPI_1_64),
            (x8, -COSPI_15_64),
            (x9, COSPI_17_64),
        ]);
        let x10_2 = dot_rs14([
            (x2, COSPI_5_64),
            (x3, COSPI_27_64),
            (x10, -COSPI_21_64),
            (x11, -COSPI_11_64),
        ]);
        let x11_2 = dot_rs14([
            (x2, COSPI_27_64),
            (x3, -COSPI_5_64),
            (x10, -COSPI_11_64),
            (x11, COSPI_21_64),
        ]);
        let x12_2 = dot_rs14([
            (x4, COSPI_9_64),
            (x5, COSPI_23_64),
            (x12, -COSPI_25_64),
            (x13, -COSPI_7_64),
        ]);
        let x13_2 = dot_rs14([
            (x4, COSPI_23_64),
            (x5, -COSPI_9_64),
            (x12, -COSPI_7_64),
            (x13, COSPI_25_64),
        ]);
        let x14_2 = dot_rs14([
            (x6, COSPI_13_64),
            (x7, COSPI_19_64),
            (x14, -COSPI_29_64),
            (x15, -COSPI_3_64),
        ]);
        let x15_2 = dot_rs14([
            (x6, COSPI_19_64),
            (x7, -COSPI_13_64),
            (x14, -COSPI_3_64),
            (x15, COSPI_29_64),
        ]);
        // stage 2
        let s0 = x0_2;
        let s1 = x1_2;
        let s2 = x2_2;
        let s3 = x3_2;
        let s4 = x4_2;
        let s5 = x5_2;
        let s6 = x6_2;
        let s7 = x7_2;
        let x0_3 = s0.add(s4);
        let x1_3 = s1.add(s5);
        let x2_3 = s2.add(s6);
        let x3_3 = s3.add(s7);
        let x4_3 = s0.sub(s4);
        let x5_3 = s1.sub(s5);
        let x6_3 = s2.sub(s6);
        let x7_3 = s3.sub(s7);
        let x8_3 = dot_rs14([
            (x8_2, COSPI_4_64),
            (x9_2, COSPI_28_64),
            (x12_2, -COSPI_28_64),
            (x13_2, COSPI_4_64),
        ]);
        let x9_3 = dot_rs14([
            (x8_2, COSPI_28_64),
            (x9_2, -COSPI_4_64),
            (x12_2, COSPI_4_64),
            (x13_2, COSPI_28_64),
        ]);
        let x10_3 = dot_rs14([
            (x10_2, COSPI_20_64),
            (x11_2, COSPI_12_64),
            (x14_2, -COSPI_12_64),
            (x15_2, COSPI_20_64),
        ]);
        let x11_3 = dot_rs14([
            (x10_2, COSPI_12_64),
            (x11_2, -COSPI_20_64),
            (x14_2, COSPI_20_64),
            (x15_2, COSPI_12_64),
        ]);
        let x12_3 = dot_rs14([
            (x8_2, COSPI_4_64),
            (x9_2, COSPI_28_64),
            (x12_2, COSPI_28_64),
            (x13_2, -COSPI_4_64),
        ]);
        let x13_3 = dot_rs14([
            (x8_2, COSPI_28_64),
            (x9_2, -COSPI_4_64),
            (x12_2, -COSPI_4_64),
            (x13_2, -COSPI_28_64),
        ]);
        let x14_3 = dot_rs14([
            (x10_2, COSPI_20_64),
            (x11_2, COSPI_12_64),
            (x14_2, COSPI_12_64),
            (x15_2, -COSPI_20_64),
        ]);
        let x15_3 = dot_rs14([
            (x10_2, COSPI_12_64),
            (x11_2, -COSPI_20_64),
            (x14_2, -COSPI_20_64),
            (x15_2, -COSPI_12_64),
        ]);
        // stage 3
        let s0_2 = x0_3;
        let s1_2 = x1_3;
        let s2_2 = x2_3;
        let s3_2 = x3_3;
        let s8 = x8_3;
        let s9 = x9_3;
        let s10 = x10_3;
        let s11 = x11_3;
        let x0_4 = s0_2.add(s2_2);
        let x1_4 = s1_2.add(s3_2);
        let x2_4 = s0_2.sub(s2_2);
        let x3_4 = s1_2.sub(s3_2);
        let x4_4 = dot_rs14([
            (x4_3, COSPI_8_64),
            (x5_3, COSPI_24_64),
            (x6_3, -COSPI_24_64),
            (x7_3, COSPI_8_64),
        ]);
        let x5_4 = dot_rs14([
            (x4_3, COSPI_24_64),
            (x5_3, -COSPI_8_64),
            (x6_3, COSPI_8_64),
            (x7_3, COSPI_24_64),
        ]);
        let x6_4 = dot_rs14([
            (x4_3, COSPI_8_64),
            (x5_3, COSPI_24_64),
            (x6_3, COSPI_24_64),
            (x7_3, -COSPI_8_64),
        ]);
        let x7_4 = dot_rs14([
            (x4_3, COSPI_24_64),
            (x5_3, -COSPI_8_64),
            (x6_3, -COSPI_8_64),
            (x7_3, -COSPI_24_64),
        ]);
        let x8_4 = s8.add(s10);
        let x9_4 = s9.add(s11);
        let x10_4 = s8.sub(s10);
        let x11_4 = s9.sub(s11);
        let x12_4 = dot_rs14([
            (x12_3, COSPI_8_64),
            (x13_3, COSPI_24_64),
            (x14_3, -COSPI_24_64),
            (x15_3, COSPI_8_64),
        ]);
        let x13_4 = dot_rs14([
            (x12_3, COSPI_24_64),
            (x13_3, -COSPI_8_64),
            (x14_3, COSPI_8_64),
            (x15_3, COSPI_24_64),
        ]);
        let x14_4 = dot_rs14([
            (x12_3, COSPI_8_64),
            (x13_3, COSPI_24_64),
            (x14_3, COSPI_24_64),
            (x15_3, -COSPI_8_64),
        ]);
        let x15_4 = dot_rs14([
            (x12_3, COSPI_24_64),
            (x13_3, -COSPI_8_64),
            (x14_3, -COSPI_8_64),
            (x15_3, -COSPI_24_64),
        ]);
        // stage 4
        let x2_5 = dot_rs14([(x2_4, -COSPI_16_64), (x3_4, -COSPI_16_64)]);
        let x3_5 = dot_rs14([(x2_4, COSPI_16_64), (x3_4, -COSPI_16_64)]);
        let x6_5 = dot_rs14([(x6_4, COSPI_16_64), (x7_4, COSPI_16_64)]);
        let x7_5 = dot_rs14([(x6_4, -COSPI_16_64), (x7_4, COSPI_16_64)]);
        let x10_5 = dot_rs14([(x10_4, COSPI_16_64), (x11_4, COSPI_16_64)]);
        let x11_5 = dot_rs14([(x10_4, -COSPI_16_64), (x11_4, COSPI_16_64)]);
        let x14_5 = dot_rs14([(x14_4, -COSPI_16_64), (x15_4, -COSPI_16_64)]);
        let x15_5 = dot_rs14([(x14_4, COSPI_16_64), (x15_4, -COSPI_16_64)]);
        output[0] = x0_4;
        output[1] = V::zero().sub(x8_4);
        output[2] = x12_4;
        output[3] = V::zero().sub(x4_4);
        output[4] = x6_5;
        output[5] = x14_5;
        output[6] = x10_5;
        output[7] = x2_5;
        output[8] = x3_5;
        output[9] = x11_5;
        output[10] = x15_5;
        output[11] = x7_5;
        output[12] = x5_4;
        output[13] = V::zero().sub(x13_4);
        output[14] = x9_4;
        output[15] = V::zero().sub(x1_4);
        output
    }
}

/// Vector form of `av1_encoder::wht::iwht_1d`.
#[inline(always)]
unsafe fn iwht<V: I32x, const SHIFT: i32>(t: [V; 4]) -> [V; 4] {
    unsafe {
        let mut a = t[0].sra::<SHIFT>();
        let mut c = t[1].sra::<SHIFT>();
        let mut d = t[2].sra::<SHIFT>();
        let mut b = t[3].sra::<SHIFT>();
        a = a.add(c);
        d = d.sub(b);
        let e = a.sub(d).sra::<1>();
        b = e.sub(b);
        c = e.sub(c);
        a = a.sub(b);
        d = d.add(c);
        [a, b, c, d]
    }
}

/// Vector form of `av1_encoder::wht::iwht_1d_inverse`.
#[inline(always)]
unsafe fn iwht_inverse<V: I32x>(o: [V; 4]) -> [V; 4] {
    unsafe {
        let a1 = o[0].add(o[1]);
        let d1 = o[3].sub(o[2]);
        let e = a1.sub(d1).sra::<1>();
        let in3 = e.sub(o[1]);
        let in1 = e.sub(o[2]);
        let in0 = a1.sub(in1);
        let in2 = d1.add(in3);
        [in0, in1, in2, in3]
    }
}

// ---------------------------------------------------------------------
// Block drivers
// ---------------------------------------------------------------------

#[inline(always)]
unsafe fn load_rows4<V: I32x + Transpose4>(block: &[i32]) -> [V; 4] {
    unsafe {
        [
            V::load(&block[0..]),
            V::load(&block[4..]),
            V::load(&block[8..]),
            V::load(&block[12..]),
        ]
    }
}

#[inline(always)]
unsafe fn store_rows4<V: I32x>(rows: [V; 4], block: &mut [i32]) {
    unsafe {
        for (index, row) in rows.into_iter().enumerate() {
            row.store(&mut block[index * 4..]);
        }
    }
}

/// Bit-exact vector form of `av1_encoder::wht::iwht4x4`.
///
/// # Safety
/// The caller must have verified `V`'s instruction set is available, and every
/// entry of `quant` must satisfy `|q| <= WHT_INPUT_LIMIT`.
pub(crate) unsafe fn iwht4x4<V: I32x + Transpose4>(quant: &[i32; 16]) -> [i32; 16] {
    unsafe {
        let rows = load_rows4::<V>(quant);
        // Lane j of `columns[k]` is element k of row j: one lane per row, so the
        // row pass below transforms all four rows at once.
        let columns = V::transpose4(rows);
        let four = V::splat(4);
        let scaled = [
            columns[0].mul(four),
            columns[1].mul(four),
            columns[2].mul(four),
            columns[3].mul(four),
        ];
        let row_pass = iwht::<V, 2>(scaled);
        // Transposing back leaves one lane per column, which is what the column
        // pass needs.
        let staged = V::transpose4(row_pass);
        let column_pass = iwht::<V, 0>(staged);
        let mut out = [0i32; 16];
        store_rows4(column_pass, &mut out);
        out
    }
}

/// Bit-exact vector form of `av1_encoder::wht::fwht4x4`.
///
/// # Safety
/// The caller must have verified `V`'s instruction set is available, and every
/// entry of `residual` must satisfy `|r| <= WHT_INPUT_LIMIT`.
pub(crate) unsafe fn fwht4x4<V: I32x + Transpose4>(residual: &[i32; 16]) -> [i32; 16] {
    unsafe {
        // The forward transform undoes the decoder's column pass first, and the
        // loaded rows already carry one lane per column.
        let rows = load_rows4::<V>(residual);
        let column_pass = iwht_inverse(rows);
        let staged = V::transpose4(column_pass);
        let row_pass = iwht_inverse(staged);
        let out_rows = V::transpose4(row_pass);
        let mut out = [0i32; 16];
        store_rows4(out_rows, &mut out);
        out
    }
}

#[inline(always)]
unsafe fn store_i16_clamped<V: I32x>(row: V, out: &mut [i16]) {
    unsafe {
        let mut scratch = [0i32; super::vector::MAX_LANES];
        row.clamp(V::splat(i16::MIN as i32), V::splat(i16::MAX as i32))
            .store(&mut scratch);
        for (slot, &value) in out.iter_mut().zip(scratch.iter()) {
            *slot = value as i16;
        }
    }
}

/// [`crate::av1_intra::COSPI_128`], narrowed to the lane width so the scalar
/// and vector 64-point kernels cannot drift apart.
const COSPI_128: [i32; 65] = {
    let source = crate::av1_intra::COSPI_128;
    let mut table = [0i32; 65];
    let mut index = 0;
    while index < 65 {
        table[index] = source[index] as i32;
        index += 1;
    }
    table
};

/// Vector form of `av1_intra::dct_iv2_1d`.
#[inline(always)]
unsafe fn dct_iv2<V: I32x>(input: [V; 2]) -> [V; 2] {
    unsafe {
        [
            dot_rs14([(input[0], COSPI_128[16]), (input[1], COSPI_128[48])]),
            dot_rs14([(input[0], COSPI_128[48]), (input[1], -COSPI_128[16])]),
        ]
    }
}

/// Vector form of the `av1_intra::dct_iv_kernel!` recursion, which documents
/// the even/odd split and the rotation angles.
macro_rules! dct_iv_kernel {
    ($name:ident, $m:literal, $half:ident) => {
        #[doc = concat!("Vector form of `av1_intra::dct_iv", stringify!($m), "_1d`.")]
        #[inline(always)]
        unsafe fn $name<V: I32x>(input: [V; $m]) -> [V; $m] {
            unsafe {
                const HALF: usize = $m / 2;
                const STEP: usize = 128 / (4 * $m);
                let mut p = [V::zero(); HALF];
                let mut q = [V::zero(); HALF];
                for j in 0..HALF {
                    let index = (2 * j + 1) * STEP;
                    let (cosine, sine) = (COSPI_128[index], COSPI_128[64 - index]);
                    let (first, second) = (input[j], input[$m - 1 - j]);
                    p[j] = dot_rs14([(first, cosine), (second, -sine)]);
                    let rotated = dot_rs14([(first, sine), (second, cosine)]);
                    q[j] = if j % 2 == 0 {
                        rotated
                    } else {
                        V::zero().sub(rotated)
                    };
                }
                let p = $half(p);
                let q = $half(q);
                let mut output = [V::zero(); $m];
                for m in 0..HALF {
                    let odd = q[HALF - 1 - m];
                    output[2 * m] = p[m].add(odd);
                    output[2 * m + 1] = p[m].sub(odd);
                }
                output
            }
        }
    };
}

dct_iv_kernel!(dct_iv4, 4, dct_iv2);
dct_iv_kernel!(dct_iv8, 8, dct_iv4);
dct_iv_kernel!(dct_iv16, 16, dct_iv8);
dct_iv_kernel!(dct_iv32, 32, dct_iv16);

/// Vector form of `av1_intra::inverse_dct64_1d`.
///
/// The same even/odd split the scalar kernel documents: the even-indexed
/// coefficients are a 32-point inverse DCT and the odd-indexed ones a 32-point
/// DCT-IV, whose sum and difference give the two output halves.
#[inline(always)]
unsafe fn dct64<V: I32x>(input: [V; 64]) -> [V; 64] {
    unsafe {
        let mut even = [V::zero(); 32];
        let mut odd = [V::zero(); 32];
        for j in 0..32 {
            even[j] = input[2 * j];
            odd[j] = input[2 * j + 1];
        }
        let even = dct32(even);
        let odd = dct_iv32(odd);

        let mut output = [V::zero(); 64];
        for n in 0..32 {
            output[n] = even[n].add(odd[n]);
            output[63 - n] = even[n].sub(odd[n]);
        }
        output
    }
}

/// Runs one separable pass of an `N x N` block.
///
/// Reads `$src` row-major, transforms each row with the kernel `$kind`
/// selects, and writes the result *transposed* into `$dst`. Two passes
/// therefore leave the block back in row-major order without a separate
/// transpose step: pass one turns rows into columns, pass two turns them
/// back.
macro_rules! separable_pass {
    ($v:ty, $n:literal, $kind:expr, $src:expr, $dst:expr, $dct:ident, $adst:ident) => {{
        for group in 0..$n / 4 {
            // Lane j of `lanes[k]` is element k of row `4 * group + j`, so the
            // butterfly below transforms four rows at once.
            let mut lanes = [<$v>::zero(); $n];
            for quad in 0..$n / 4 {
                let tile = <$v>::transpose4([
                    <$v>::load(&$src[(4 * group) * $n + 4 * quad..]),
                    <$v>::load(&$src[(4 * group + 1) * $n + 4 * quad..]),
                    <$v>::load(&$src[(4 * group + 2) * $n + 4 * quad..]),
                    <$v>::load(&$src[(4 * group + 3) * $n + 4 * quad..]),
                ]);
                lanes[4 * quad..4 * quad + 4].copy_from_slice(&tile);
            }
            let transformed = match $kind {
                Tx1d::Dct => $dct(lanes),
                Tx1d::Adst => $adst(lanes),
                Tx1d::Identity => identity_pass::<V, $n>(lanes),
            };
            // `transformed[k]` lane j is element k of row `4 * group + j`,
            // which is exactly column `4 * group + j` of output row k.
            for (k, value) in transformed.into_iter().enumerate() {
                value.store(&mut $dst[k * $n + 4 * group..]);
            }
        }
    }};
}

/// Vector form of the identity 1-D pass both `av1_intra::inverse_transform_1d`
/// and `av1_encoder::transform::forward_1d` apply for a half-identity type:
/// every element scaled by `av1_intra::identity_scale` at its size, in the
/// same 14-bit fixed point the butterflies use, and in the same
/// split-accumulator form so a wide input cannot overflow a lane.
#[inline(always)]
unsafe fn identity_pass<V: I32x, const N: usize>(input: [V; N]) -> [V; N] {
    unsafe {
        let scale = identity_scale(N) as i32;
        let mut output = [V::zero(); N];
        for (slot, value) in output.iter_mut().zip(input) {
            *slot = dot_rs14([(value, scale)]);
        }
        output
    }
}

/// Defines the `N x N` inverse transform driver for one block size.
///
/// `$adst` is the ADST kernel at that size, or the DCT kernel again at 32 and
/// 64 points where AV1 defines no ADST (the dispatcher normalizes the request
/// before calling in, so that arm is unreachable).
macro_rules! inverse_transform_driver {
    ($name:ident, $n:literal, $shift:literal, $dct:ident, $adst:ident) => {
        /// Bit-exact vector form of `av1_intra::inverse_transform` for one
        #[doc = concat!(stringify!($n), "x", stringify!($n), " block.")]
        ///
        /// # Safety
        /// The caller must have verified `V`'s instruction set is available
        /// and that `dequantized` is within `input_limit`.
        pub(crate) unsafe fn $name<V: I32x + Transpose4>(
            dequantized: &[i32],
            column: Tx1d,
            row: Tx1d,
            lr_flip: bool,
            ud_flip: bool,
            out: &mut [i16],
        ) -> bool {
            unsafe {
                let mut staged = [0i32; $n * $n];
                separable_pass!(V, $n, row, dequantized, staged, $dct, $adst);
                if !within_limit(&staged, staged_limit($n)) {
                    return false;
                }
                let mut finished = [0i32; $n * $n];
                separable_pass!(V, $n, column, staged, finished, $dct, $adst);

                let bias = V::splat(1 << ($shift - 1));
                let mut scratch = [0i16; 4];
                for source_row in 0..$n {
                    let target_row = if ud_flip {
                        $n - 1 - source_row
                    } else {
                        source_row
                    };
                    for chunk in (0..$n).step_by(4) {
                        let rounded = V::load(&finished[source_row * $n + chunk..])
                            .add(bias)
                            .sra::<$shift>();
                        store_i16_clamped(rounded, &mut scratch);
                        for (offset, &value) in scratch.iter().enumerate() {
                            let source_column = chunk + offset;
                            let target_column = if lr_flip {
                                $n - 1 - source_column
                            } else {
                                source_column
                            };
                            out[target_row * $n + target_column] = value;
                        }
                    }
                }
                true
            }
        }
    };
}

inverse_transform_driver!(inverse_transform4, 4, 4, dct4, adst4);
inverse_transform_driver!(inverse_transform8, 8, 5, dct8, adst8);
inverse_transform_driver!(inverse_transform16, 16, 6, dct16, adst16);
inverse_transform_driver!(inverse_transform32, 32, 6, dct32, dct32);
inverse_transform_driver!(inverse_transform64, 64, 6, dct64, dct64);

// ---------------------------------------------------------------------
// Forward transforms (issue #140)
// ---------------------------------------------------------------------

/// Bit-exact vector form of one row of `av1_encoder::transform::forward_1d`:
/// `dct_round_shift(sum_n basis[n] * input[n])`.
///
/// Same split-around-bit-14 accumulator as [`dot_rs14`], but a forward basis
/// row has up to 32 terms rather than the butterflies' seven, which would
/// overflow the low half. Folding the low accumulator's completed high bits
/// into `high` every four terms is exact - `x == (x >> 14) * 2^14 + (x &
/// 0x3fff)` holds for negative `x` too - and keeps the low half under `2^30`.
#[inline(always)]
unsafe fn basis_row_rs14<V: I32x, const N: usize>(input: &[V; N], basis: &[i32]) -> V {
    unsafe {
        let mask = V::splat(0x3fff);
        let mut high = V::zero();
        let mut low = V::zero();
        for (index, (&value, &coefficient)) in input.iter().zip(basis.iter()).enumerate() {
            let scale = V::splat(coefficient);
            high = high.add(value.sra::<14>().mul(scale));
            low = low.add(value.and(mask).mul(scale));
            if index % 4 == 3 {
                high = high.add(low.sra::<14>());
                low = low.and(mask);
            }
        }
        high.add(round_shift14(low))
    }
}

/// Defines the vector form of one forward 1-D kernel from its basis table.
macro_rules! forward_kernel {
    ($name:ident, $n:literal, $basis:ident) => {
        /// Vector form of the matching `av1_encoder::transform` forward pass.
        #[inline(always)]
        unsafe fn $name<V: I32x>(input: [V; $n]) -> [V; $n] {
            unsafe {
                let mut output = [V::zero(); $n];
                for (k, slot) in output.iter_mut().enumerate() {
                    *slot = basis_row_rs14(&input, &$basis[k * $n..(k + 1) * $n]);
                }
                output
            }
        }
    };
}

forward_kernel!(fdct4, 4, FDCT4);
forward_kernel!(fdct8, 8, FDCT8);
forward_kernel!(fdct16, 16, FDCT16);
forward_kernel!(fdct32, 32, FDCT32);
forward_kernel!(fadst4, 4, FADST4);
forward_kernel!(fadst8, 8, FADST8);
forward_kernel!(fadst16, 16, FADST16);

/// Defines the `N x N` forward transform driver for one block size.
///
/// `$pre` and `$mid` are that size's input left shift and inter-pass rounding
/// right shift (see `av1_encoder::transform`); `$adst` is the DCT kernel again
/// at 32 points, where AV1 defines no ADST and the dispatcher has already
/// normalized the request.
macro_rules! forward_transform_driver {
    ($name:ident, $n:literal, $pre:literal, $mid:literal, $dct:ident, $adst:ident) => {
        /// Bit-exact vector form of `av1_encoder::transform::forward_transform`
        /// for one
        #[doc = concat!(stringify!($n), "x", stringify!($n), " block.")]
        ///
        /// # Safety
        /// The caller must have verified `V`'s instruction set is available
        /// and that `residual` is within `input_limit`.
        pub(crate) unsafe fn $name<V: I32x + Transpose4>(
            residual: &[i32],
            column: Tx1d,
            row: Tx1d,
            lr_flip: bool,
            ud_flip: bool,
            out: &mut [i32],
        ) -> bool {
            unsafe {
                // `FLIPADST` reverses the inverse transform's output along an
                // axis, so the forward transform reverses its input instead.
                let mut source = [0i32; $n * $n];
                for target_row in 0..$n {
                    let source_row = if ud_flip {
                        $n - 1 - target_row
                    } else {
                        target_row
                    };
                    for target_column in 0..$n {
                        let source_column = if lr_flip {
                            $n - 1 - target_column
                        } else {
                            target_column
                        };
                        source[target_row * $n + target_column] =
                            residual[source_row * $n + source_column] << $pre;
                    }
                }

                let mut staged = [0i32; $n * $n];
                separable_pass!(V, $n, row, source, staged, $dct, $adst);
                if !within_limit(&staged, forward_staged_limit($n)) {
                    return false;
                }
                for value in &mut staged {
                    *value = (*value + ((1 << $mid) >> 1)) >> $mid;
                }
                separable_pass!(V, $n, column, staged, out, $dct, $adst);
                true
            }
        }
    };
}

forward_transform_driver!(forward_transform4, 4, 2, 0, fdct4, fadst4);
forward_transform_driver!(forward_transform8, 8, 1, 0, fdct8, fadst8);
forward_transform_driver!(forward_transform16, 16, 0, 0, fdct16, fadst16);
forward_transform_driver!(forward_transform32, 32, 0, 2, fdct32, fdct32);
