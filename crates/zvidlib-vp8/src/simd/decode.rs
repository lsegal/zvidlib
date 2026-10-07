//! The `vp8_decode` dispatch site (issue #568): the decoder-path kernels the
//! `vp8_recon` site does not cover - the ten 4x4 subblock intra predictors and
//! the DC-only inverse DCT.
//!
//! The DC-only inverse DCT adds one rounded value to a 4x4 block, which LLVM
//! already does well one block at a time, so the vector kernel takes a whole
//! row of a macroblock's blocks - four luma or two chroma, 16 or 8 samples
//! wide - in one 16- or 8-byte load and store per sample row.
//!
//! A subblock is too small for 32-bit lanes to pay: an `I32x` version of the
//! predictors measured 0.65x of the scalar code. These work on the block's 16
//! bytes at once instead. Every directional mode's samples are `avg2` or
//! `avg3` of neighbouring edge samples, so the kernel computes both rows over
//! the whole edge in three instructions and gathers each mode's 16 samples
//! out of them with one table lookup (`pshufb` on x86_64, `tbl` on aarch64).
//! TM is computed in 16-bit lanes and narrowed with saturation, which is its
//! clamp.
//!
//! The entry points are `#[inline(never)]` and `#[target_feature]`, and the
//! kernels `#[inline(always)]`, so a kernel the inliner left standing would
//! show up as out-of-line intrinsic calls to
//! `.github/scripts/check_simd_target_features.py` (#341).

use crate::tables::{
    B_DC_PRED, B_HD_PRED, B_HE_PRED, B_HU_PRED, B_LD_PRED, B_RD_PRED, B_TM_PRED, B_VE_PRED,
    B_VL_PRED, B_VR_PRED,
};
use zvidlib_core::simd::SimdIsa;

/// The instruction set the `vp8_decode` kernels use.
#[must_use]
pub fn decode_isa() -> SimdIsa {
    zvidlib_core::simd::active()
}

/// Where a subblock mode takes each of its 16 samples from (raster order).
/// `A3 + j` is `avg3` of edge samples `j`, `j + 1` and `j + 2`, `A2 + j` is
/// `avg2` of edge samples `j` and `j + 1`, and `E + j` is edge sample `j`
/// itself, the edge running `l3, l3, l2, l1, l0, p, a0, .., a7` and then
/// repeating `a7`.
const A3: u8 = 0;
const A2: u8 = 16;
const E: u8 = 32;
#[rustfmt::skip]
const SUBBLOCK_SOURCES: [[u8; 16]; 8] = [
    // B_VE_PRED
    [A3 + 5, A3 + 6, A3 + 7, A3 + 8, A3 + 5, A3 + 6, A3 + 7, A3 + 8,
     A3 + 5, A3 + 6, A3 + 7, A3 + 8, A3 + 5, A3 + 6, A3 + 7, A3 + 8],
    // B_HE_PRED
    [A3 + 3, A3 + 3, A3 + 3, A3 + 3, A3 + 2, A3 + 2, A3 + 2, A3 + 2,
     A3 + 1, A3 + 1, A3 + 1, A3 + 1, A3, A3, A3, A3],
    // B_LD_PRED
    [A3 + 6, A3 + 7, A3 + 8, A3 + 9, A3 + 7, A3 + 8, A3 + 9, A3 + 10,
     A3 + 8, A3 + 9, A3 + 10, A3 + 11, A3 + 9, A3 + 10, A3 + 11, A3 + 12],
    // B_RD_PRED
    [A3 + 4, A3 + 5, A3 + 6, A3 + 7, A3 + 3, A3 + 4, A3 + 5, A3 + 6,
     A3 + 2, A3 + 3, A3 + 4, A3 + 5, A3 + 1, A3 + 2, A3 + 3, A3 + 4],
    // B_VR_PRED
    [A2 + 5, A2 + 6, A2 + 7, A2 + 8, A3 + 4, A3 + 5, A3 + 6, A3 + 7,
     A3 + 3, A2 + 5, A2 + 6, A2 + 7, A3 + 2, A3 + 4, A3 + 5, A3 + 6],
    // B_VL_PRED
    [A2 + 6, A2 + 7, A2 + 8, A2 + 9, A3 + 6, A3 + 7, A3 + 8, A3 + 9,
     A2 + 7, A2 + 8, A2 + 9, A3 + 10, A3 + 7, A3 + 8, A3 + 9, A3 + 11],
    // B_HD_PRED
    [A2 + 4, A3 + 4, A3 + 5, A3 + 6, A2 + 3, A3 + 3, A2 + 4, A3 + 4,
     A2 + 2, A3 + 2, A2 + 3, A3 + 3, A2 + 1, A3 + 1, A2 + 2, A3 + 2],
    // B_HU_PRED
    [A2 + 3, A3 + 2, A2 + 2, A3 + 1, A2 + 2, A3 + 1, A2 + 1, A3,
     A2 + 1, A3, E, E, E, E, E, E],
];

/// The 16 bytes of a whole 4x4 subblock in one vector.
trait Bytes16: Copy {
    /// The subblock's edge as `SUBBLOCK_SOURCES` indexes it, and the same edge
    /// moved down one and two samples: byte `j` of the three is edge sample
    /// `j`, `j + 1` and `j + 2`. Built in registers rather than staged through
    /// memory, which a 16-byte reload of byte stores would stall on.
    unsafe fn edge(above: &[u8; 9], left: &[u8; 4]) -> [Self; 3];
    /// Writes byte `4 * r + c` to `plane[offset + r * stride + c]`.
    unsafe fn store_rows(self, plane: &mut [u8], offset: usize, stride: usize);
    /// `(x + y + 1) >> 1` per byte.
    unsafe fn avg2(x: Self, y: Self) -> Self;
    /// `(x + 2 * y + z + 2) >> 2` per byte, which is exactly the rounded
    /// average of `y` and the truncated average of `x` and `z`.
    unsafe fn avg3(x: Self, y: Self, z: Self) -> Self;
    /// Byte `i` of the result is byte `SUBBLOCK_SOURCES[mode][i]` of `rows`
    /// laid end to end.
    unsafe fn lookup(rows: [Self; 3], mode: usize) -> Self;
    /// TM prediction: `clamp255(left[r] + above[1 + c] - above[0])`.
    unsafe fn tm(above: &[u8; 9], left: &[u8; 4]) -> Self;
    /// Adds `residuals[i]` to every sample of the `i`-th 4x4 block of the
    /// `blocks` (2 or 4) side by side at `offset`, saturating.
    unsafe fn add_dc_row(
        residuals: [i16; 4],
        blocks: usize,
        plane: &mut [u8],
        offset: usize,
        stride: usize,
    );
}

/// Writes `words` back as the four rows of the block at `offset`.
#[inline(always)]
fn write_rows(words: [u32; 4], plane: &mut [u8], offset: usize, stride: usize) {
    for (row, word) in words.into_iter().enumerate() {
        let at = offset + row * stride;
        plane[at..at + 4].copy_from_slice(&word.to_le_bytes());
    }
}

/// [`crate::predict::predict_subblock_scalar`].
#[inline(always)]
unsafe fn subblock_kernel<B: Bytes16>(
    mode: u8,
    above: &[u8; 9],
    left: &[u8; 4],
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) {
    unsafe {
        match mode {
            B_DC_PRED => {
                let sum: u32 = above[1..5]
                    .iter()
                    .chain(left)
                    .map(|&value| u32::from(value))
                    .sum();
                let value = (sum + 4) >> 3;
                write_rows([value * 0x0101_0101; 4], plane, offset, stride);
            }
            B_TM_PRED => B::tm(above, left).store_rows(plane, offset, stride),
            B_VE_PRED | B_HE_PRED | B_LD_PRED | B_RD_PRED | B_VR_PRED | B_VL_PRED | B_HD_PRED
            | B_HU_PRED => {
                let [x, y, z] = B::edge(above, left);
                let rows = [B::avg3(x, y, z), B::avg2(x, y), x];
                B::lookup(rows, usize::from(mode - B_VE_PRED)).store_rows(plane, offset, stride);
            }
            _ => unreachable!("subblock intra modes are 0..=9"),
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Bytes16, SUBBLOCK_SOURCES, write_rows};
    use core::arch::x86_64::*;

    /// `pshufb` masks taking each mode's samples out of each of the three
    /// rows, `0x80` (zero) where a sample comes from another row.
    const SHUFFLES: [[[u8; 16]; 3]; 8] = {
        let mut masks = [[[0x80u8; 16]; 3]; 8];
        let mut mode = 0;
        while mode < 8 {
            let mut index = 0;
            while index < 16 {
                let source = SUBBLOCK_SOURCES[mode][index];
                masks[mode][(source / 16) as usize][index] = source % 16;
                index += 1;
            }
            mode += 1;
        }
        masks
    };

    #[derive(Clone, Copy)]
    pub struct X86Bytes(__m128i);

    impl Bytes16 for X86Bytes {
        #[inline(always)]
        unsafe fn edge(above: &[u8; 9], left: &[u8; 4]) -> [Self; 3] {
            // `l3, l3, l2, l1, l0, p` in bytes 0 to 5, `a0..a7` in 6 to 13,
            // and `a7` again in 14, which `B_LD_PRED`'s last `avg3` reads.
            // Byte 15 and the bytes the shifts move in are never looked up.
            let corner =
                u64::from_le_bytes([left[3], left[3], left[2], left[1], left[0], above[0], 0, 0]);
            let top = u64::from_le_bytes(above[1..9].try_into().unwrap());
            unsafe {
                let x = _mm_set_epi64x((top >> 16) as i64, (corner | (top << 48)) as i64);
                let x = _mm_insert_epi8::<14>(x, i32::from(above[8]));
                [
                    Self(x),
                    Self(_mm_srli_si128::<1>(x)),
                    Self(_mm_srli_si128::<2>(x)),
                ]
            }
        }
        #[inline(always)]
        unsafe fn store_rows(self, plane: &mut [u8], offset: usize, stride: usize) {
            let words = unsafe {
                [
                    _mm_cvtsi128_si32(self.0) as u32,
                    _mm_extract_epi32::<1>(self.0) as u32,
                    _mm_extract_epi32::<2>(self.0) as u32,
                    _mm_extract_epi32::<3>(self.0) as u32,
                ]
            };
            write_rows(words, plane, offset, stride);
        }
        #[inline(always)]
        unsafe fn avg2(x: Self, y: Self) -> Self {
            unsafe { Self(_mm_avg_epu8(x.0, y.0)) }
        }
        #[inline(always)]
        unsafe fn avg3(x: Self, y: Self, z: Self) -> Self {
            unsafe {
                // `pavgb` rounds up; taking the odd bit back off makes it the
                // truncated average.
                let odd = _mm_and_si128(_mm_xor_si128(x.0, z.0), _mm_set1_epi8(1));
                let floor = _mm_sub_epi8(_mm_avg_epu8(x.0, z.0), odd);
                Self(_mm_avg_epu8(floor, y.0))
            }
        }
        #[inline(always)]
        unsafe fn lookup(rows: [Self; 3], mode: usize) -> Self {
            unsafe {
                let mut picked = _mm_setzero_si128();
                for (row, mask) in rows.iter().zip(&SHUFFLES[mode]) {
                    let mask = _mm_loadu_si128(mask.as_ptr().cast());
                    picked = _mm_or_si128(picked, _mm_shuffle_epi8(row.0, mask));
                }
                Self(picked)
            }
        }
        #[inline(always)]
        unsafe fn tm(above: &[u8; 9], left: &[u8; 4]) -> Self {
            let [a0, a1, a2, a3] = [1, 2, 3, 4].map(|index| i16::from(above[index]));
            let [l0, l1, l2, l3] = left.map(i16::from);
            unsafe {
                let tops = _mm_setr_epi16(a0, a1, a2, a3, a0, a1, a2, a3);
                let base = _mm_sub_epi16(tops, _mm_set1_epi16(i16::from(above[0])));
                let rows01 = _mm_add_epi16(base, _mm_setr_epi16(l0, l0, l0, l0, l1, l1, l1, l1));
                let rows23 = _mm_add_epi16(base, _mm_setr_epi16(l2, l2, l2, l2, l3, l3, l3, l3));
                Self(_mm_packus_epi16(rows01, rows23))
            }
        }
        #[inline(always)]
        unsafe fn add_dc_row(
            residuals: [i16; 4],
            blocks: usize,
            plane: &mut [u8],
            offset: usize,
            stride: usize,
        ) {
            let [r0, r1, r2, r3] = residuals;
            unsafe {
                let low = _mm_setr_epi16(r0, r0, r0, r0, r1, r1, r1, r1);
                let high = _mm_setr_epi16(r2, r2, r2, r2, r3, r3, r3, r3);
                for row in 0..4 {
                    let at = offset + row * stride;
                    if blocks == 4 {
                        let line = &mut plane[at..at + 16];
                        let samples = _mm_loadu_si128(line.as_ptr().cast());
                        let sums = _mm_packus_epi16(
                            _mm_adds_epi16(_mm_cvtepu8_epi16(samples), low),
                            _mm_adds_epi16(_mm_cvtepu8_epi16(_mm_srli_si128::<8>(samples)), high),
                        );
                        _mm_storeu_si128(line.as_mut_ptr().cast(), sums);
                    } else {
                        let line = &mut plane[at..at + 8];
                        let samples = _mm_loadl_epi64(line.as_ptr().cast());
                        let sums = _mm_adds_epi16(_mm_cvtepu8_epi16(samples), low);
                        _mm_storel_epi64(line.as_mut_ptr().cast(), _mm_packus_epi16(sums, sums));
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{Bytes16, SUBBLOCK_SOURCES, write_rows};
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct NeonBytes(uint8x16_t);

    impl Bytes16 for NeonBytes {
        #[inline(always)]
        unsafe fn edge(above: &[u8; 9], left: &[u8; 4]) -> [Self; 3] {
            // `l3, l3, l2, l1, l0, p` in bytes 0 to 5, `a0..a7` in 6 to 13,
            // and `a7` again in 14, which `B_LD_PRED`'s last `avg3` reads.
            // Byte 15 and the bytes the rotations wrap around are never
            // looked up.
            let corner =
                u64::from_le_bytes([left[3], left[3], left[2], left[1], left[0], above[0], 0, 0]);
            let top = u64::from_le_bytes(above[1..9].try_into().unwrap());
            unsafe {
                let x = vcombine_u8(vcreate_u8(corner | (top << 48)), vcreate_u8(top >> 16));
                let x = vsetq_lane_u8::<14>(above[8], x);
                [
                    Self(x),
                    Self(vextq_u8::<1>(x, x)),
                    Self(vextq_u8::<2>(x, x)),
                ]
            }
        }
        #[inline(always)]
        unsafe fn store_rows(self, plane: &mut [u8], offset: usize, stride: usize) {
            let words = unsafe {
                let words = vreinterpretq_u32_u8(self.0);
                [
                    vgetq_lane_u32::<0>(words),
                    vgetq_lane_u32::<1>(words),
                    vgetq_lane_u32::<2>(words),
                    vgetq_lane_u32::<3>(words),
                ]
            };
            write_rows(words, plane, offset, stride);
        }
        #[inline(always)]
        unsafe fn avg2(x: Self, y: Self) -> Self {
            unsafe { Self(vrhaddq_u8(x.0, y.0)) }
        }
        #[inline(always)]
        unsafe fn avg3(x: Self, y: Self, z: Self) -> Self {
            unsafe { Self(vrhaddq_u8(vhaddq_u8(x.0, z.0), y.0)) }
        }
        #[inline(always)]
        unsafe fn lookup(rows: [Self; 3], mode: usize) -> Self {
            unsafe {
                let table = uint8x16x3_t(rows[0].0, rows[1].0, rows[2].0);
                Self(vqtbl3q_u8(table, vld1q_u8(SUBBLOCK_SOURCES[mode].as_ptr())))
            }
        }
        #[inline(always)]
        unsafe fn tm(above: &[u8; 9], left: &[u8; 4]) -> Self {
            let tops = [
                above[1], above[2], above[3], above[4], above[1], above[2], above[3], above[4],
            ];
            let sides: [[i16; 8]; 2] = [0, 2].map(|first| {
                let (a, b) = (i16::from(left[first]), i16::from(left[first + 1]));
                [a, a, a, a, b, b, b, b]
            });
            unsafe {
                let tops = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(tops.as_ptr())));
                let base = vsubq_s16(tops, vdupq_n_s16(i16::from(above[0])));
                let low = vqmovun_s16(vaddq_s16(base, vld1q_s16(sides[0].as_ptr())));
                let high = vqmovun_s16(vaddq_s16(base, vld1q_s16(sides[1].as_ptr())));
                Self(vcombine_u8(low, high))
            }
        }
        #[inline(always)]
        unsafe fn add_dc_row(
            residuals: [i16; 4],
            blocks: usize,
            plane: &mut [u8],
            offset: usize,
            stride: usize,
        ) {
            let [r0, r1, r2, r3] = residuals;
            unsafe {
                let low = vcombine_s16(vdup_n_s16(r0), vdup_n_s16(r1));
                let high = vcombine_s16(vdup_n_s16(r2), vdup_n_s16(r3));
                for row in 0..4 {
                    let at = offset + row * stride;
                    if blocks == 4 {
                        let line = &mut plane[at..at + 16];
                        let samples = vld1q_u8(line.as_ptr());
                        let first = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(samples)));
                        let second = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(samples)));
                        let sums = vcombine_u8(
                            vqmovun_s16(vqaddq_s16(first, low)),
                            vqmovun_s16(vqaddq_s16(second, high)),
                        );
                        vst1q_u8(line.as_mut_ptr(), sums);
                    } else {
                        let line = &mut plane[at..at + 8];
                        let samples = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(line.as_ptr())));
                        vst1_u8(line.as_mut_ptr(), vqmovun_s16(vqaddq_s16(samples, low)));
                    }
                }
            }
        }
    }
}

/// [`crate::predict::idct_dc_add_row_scalar`]: the DC-only inverse DCT
/// of the 2 or 4 blocks side by side at `offset`, one row of a macroblock's
/// blocks at a time.
#[inline(always)]
unsafe fn idct_dc_add_row_kernel<B: Bytes16>(
    dcs: &[i16],
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) {
    // `(dc + 4) >> 3` of an `i16` fits in an `i16`, and the saturating add
    // and narrowing are the scalar `clamp255`.
    let mut residuals = [0i16; 4];
    for (residual, &dc) in residuals.iter_mut().zip(dcs) {
        *residual = ((i32::from(dc) + 4) >> 3) as i16;
    }
    unsafe { B::add_dc_row(residuals, dcs.len(), plane, offset, stride) }
}

macro_rules! entry_points {
    ($kernel:ident: [$sse:ident, $avx:ident, $neon:ident]($($arg:ident : $ty:ty),* $(,)?)) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse4.1")]
        #[inline(never)]
        unsafe fn $sse($($arg: $ty),*) {
            unsafe { $kernel::<x86::X86Bytes>($($arg),*) }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2")]
        #[inline(never)]
        unsafe fn $avx($($arg: $ty),*) {
            unsafe { $kernel::<x86::X86Bytes>($($arg),*) }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline(never)]
        unsafe fn $neon($($arg: $ty),*) {
            unsafe { $kernel::<arm::NeonBytes>($($arg),*) }
        }
    };
}

entry_points!(subblock_kernel: [subblock_sse41, subblock_avx2, subblock_neon](
    mode: u8, above: &[u8; 9], left: &[u8; 4], plane: &mut [u8], offset: usize, stride: usize,
));
entry_points!(idct_dc_add_row_kernel: [dc_row_sse41, dc_row_avx2, dc_row_neon](
    dcs: &[i16], plane: &mut [u8], offset: usize, stride: usize,
));

/// Runs the entry point for `isa`, evaluating to `true`, or to `false` when
/// `isa` has no kernel here. The instruction set comes from
/// [`decode_isa`], which only ever reports one this host can execute.
macro_rules! dispatch {
    ($isa:expr, [$sse:ident, $avx:ident, $neon:ident]($($arg:expr),* $(,)?)) => {
        match $isa {
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Sse41 => {
                unsafe { $sse($($arg),*) };
                true
            }
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx2 => {
                unsafe { $avx($($arg),*) };
                true
            }
            #[cfg(target_arch = "aarch64")]
            SimdIsa::Neon => {
                unsafe { $neon($($arg),*) };
                true
            }
            _ => false,
        }
    };
}

/// The vector `crate::predict::predict_subblock_scalar`, or `false`
/// when the active instruction set has none.
pub fn subblock(
    mode: u8,
    above: &[u8; 9],
    left: &[u8; 4],
    plane: &mut [u8],
    offset: usize,
    stride: usize,
) -> bool {
    dispatch!(
        decode_isa(),
        [subblock_sse41, subblock_avx2, subblock_neon](mode, above, left, plane, offset, stride)
    )
}

/// The vector `crate::predict::idct_dc_add_row_scalar` over `dcs`, 2
/// or 4 blocks, or `false` when the active instruction set has none.
pub fn idct_dc_add_row(dcs: &[i16], plane: &mut [u8], offset: usize, stride: usize) -> bool {
    assert!(matches!(dcs.len(), 2 | 4), "a row of 2 or 4 blocks");
    dispatch!(
        decode_isa(),
        [dc_row_sse41, dc_row_avx2, dc_row_neon](dcs, plane, offset, stride)
    )
}
