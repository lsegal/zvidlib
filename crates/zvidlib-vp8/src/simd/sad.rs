//! The 16x16 sum of absolute differences that dominates VP8 motion search.
//!
//! Unlike the other VP8 kernels these are not written over `I32x`: a SAD is a
//! byte operation, and every instruction set has a byte instruction for it
//! (`psadbw` on x86_64, `uabd`/`uadalp` on aarch64) that a 32-bit lane
//! abstraction would only get in the way of. The intrinsics are written
//! directly inside each `#[target_feature]` function, so there is no separate
//! kernel body for the inliner to leave behind.
//!
//! Callers check that both slices hold a whole 16x16 block at their stride.

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use core::arch::x86_64::*;

    /// # Safety
    ///
    /// SSE4.1 must be available, and each slice must hold `15 * stride + 16`
    /// bytes.
    #[target_feature(enable = "sse4.1")]
    pub(in crate::simd) unsafe fn sad16_sse41(
        source: &[u8],
        source_stride: usize,
        prediction: &[u8],
        prediction_stride: usize,
    ) -> u32 {
        unsafe {
            let mut sum = _mm_setzero_si128();
            for row in 0..16 {
                let a = _mm_loadu_si128(source[row * source_stride..][..16].as_ptr().cast());
                let b =
                    _mm_loadu_si128(prediction[row * prediction_stride..][..16].as_ptr().cast());
                sum = _mm_add_epi64(sum, _mm_sad_epu8(a, b));
            }
            (_mm_cvtsi128_si64(sum) + _mm_extract_epi64::<1>(sum)) as u32
        }
    }

    /// Two rows a 256-bit `vpsadbw`.
    ///
    /// # Safety
    ///
    /// AVX2 must be available, and each slice must hold `15 * stride + 16`
    /// bytes.
    #[target_feature(enable = "avx2")]
    pub(in crate::simd) unsafe fn sad16_avx2(
        source: &[u8],
        source_stride: usize,
        prediction: &[u8],
        prediction_stride: usize,
    ) -> u32 {
        unsafe {
            let rows = |data: &[u8], stride: usize, row: usize| {
                let low = _mm_loadu_si128(data[row * stride..][..16].as_ptr().cast());
                let high = _mm_loadu_si128(data[(row + 1) * stride..][..16].as_ptr().cast());
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(low), high)
            };
            let mut sum = _mm256_setzero_si256();
            for row in (0..16).step_by(2) {
                let a = rows(source, source_stride, row);
                let b = rows(prediction, prediction_stride, row);
                sum = _mm256_add_epi64(sum, _mm256_sad_epu8(a, b));
            }
            let sum = _mm_add_epi64(
                _mm256_castsi256_si128(sum),
                _mm256_extracti128_si256::<1>(sum),
            );
            (_mm_cvtsi128_si64(sum) + _mm_extract_epi64::<1>(sum)) as u32
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub mod arm {
    use core::arch::aarch64::*;

    /// # Safety
    ///
    /// NEON must be available, and each slice must hold `15 * stride + 16`
    /// bytes.
    #[target_feature(enable = "neon")]
    pub(in crate::simd) unsafe fn sad16_neon(
        source: &[u8],
        source_stride: usize,
        prediction: &[u8],
        prediction_stride: usize,
    ) -> u32 {
        unsafe {
            // Sixteen rows of two byte differences each is at most 8160 per
            // 16-bit lane, so the pairwise accumulation cannot overflow.
            let mut sum = vdupq_n_u16(0);
            for row in 0..16 {
                let a = vld1q_u8(source[row * source_stride..][..16].as_ptr());
                let b = vld1q_u8(prediction[row * prediction_stride..][..16].as_ptr());
                sum = vpadalq_u8(sum, vabdq_u8(a, b));
            }
            vaddlvq_u16(sum)
        }
    }
}
