//! A minimal, width-agnostic `f32` SIMD abstraction for the Vorbis kernels.
//!
//! Each Vorbis kernel is written once, generic over [`F32x`], and instantiated
//! per instruction set behind a `#[target_feature]` wrapper in [`super`], the
//! same arrangement `crate::av1_simd::vector` uses for the AV1 kernels.
//!
//! The trait's methods are not themselves `#[target_feature]`: they are
//! `#[inline(always)]` and only ever reached from a wrapper that already
//! carries the feature, so LLVM inlines them with the wrapper's feature set.
//! Calling one from a context that has not verified CPU support is undefined
//! behaviour, which is why every method is `unsafe`.
//!
//! Only operations that IEEE 754 defines exactly lane by lane are offered -
//! add, subtract, multiply, sign flip, ordered compare, select and lane
//! permutes - and in particular no fused multiply-add, so a kernel built from
//! them rounds every lane exactly as the scalar reference rounds that element.

#![allow(clippy::missing_safety_doc)]

/// Elementwise `f32` vector operations.
///
/// # Safety
///
/// Every method may execute instructions that are not part of the target's
/// baseline ISA. A caller must only reach these through a wrapper that has
/// verified the corresponding CPU feature is present.
pub(crate) trait F32x: Copy {
    /// Number of `f32` lanes this vector holds.
    const LANES: usize;

    unsafe fn splat(value: f32) -> Self;
    /// Loads `LANES` lanes from the front of `src` (`src.len() >= LANES`).
    unsafe fn load(src: &[f32]) -> Self;
    /// Stores `LANES` lanes to the front of `dst` (`dst.len() >= LANES`).
    unsafe fn store(self, dst: &mut [f32]);

    unsafe fn add(self, other: Self) -> Self;
    unsafe fn sub(self, other: Self) -> Self;
    unsafe fn mul(self, other: Self) -> Self;
    /// Flips every lane's sign bit, which is exactly what scalar `-x` does.
    unsafe fn neg(self) -> Self;
    /// Lane mask: all ones where `self > other`, zero elsewhere (and zero
    /// wherever either side is NaN, as the scalar `>` is false there).
    unsafe fn gt(self, other: Self) -> Self;
    /// Lane mask: all ones where `self < other`, zero elsewhere.
    unsafe fn lt(self, other: Self) -> Self;
    /// `mask ? a : b` per lane, where `mask` lanes are all-ones or all-zero.
    unsafe fn select(mask: Self, a: Self, b: Self) -> Self;

    /// `if self < low { low } else { self }` per lane, the first half of
    /// [`f32::clamp`]; NaN passes through.
    #[inline(always)]
    unsafe fn raise_to(self, low: Self) -> Self {
        unsafe { Self::select(self.lt(low), low, self) }
    }
    /// `if self > high { high } else { self }` per lane, the second half of
    /// [`f32::clamp`]; NaN passes through.
    #[inline(always)]
    unsafe fn lower_to(self, high: Self) -> Self {
        unsafe { Self::select(self.gt(high), high, self) }
    }

    /// The lanes in reverse order.
    unsafe fn reverse(self) -> Self;
    /// Splits the `2 * LANES` consecutive values `[self, next]` into their
    /// even-indexed and odd-indexed halves, each in ascending order.
    unsafe fn deinterleave(self, next: Self) -> (Self, Self);
    /// The inverse of [`F32x::deinterleave`]: `self[0], odd[0], self[1],
    /// odd[1], ...` as two consecutive vectors.
    unsafe fn interleave(self, odd: Self) -> (Self, Self);
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::F32x;
    use core::arch::x86_64::*;

    /// Four lanes in an SSE register; `select` needs SSE4.1's `blendvps`.
    #[derive(Clone, Copy)]
    pub(crate) struct Sse4(__m128);

    impl F32x for Sse4 {
        const LANES: usize = 4;

        #[inline(always)]
        unsafe fn splat(value: f32) -> Self {
            unsafe { Self(_mm_set1_ps(value)) }
        }
        #[inline(always)]
        unsafe fn load(src: &[f32]) -> Self {
            debug_assert!(src.len() >= Self::LANES);
            unsafe { Self(_mm_loadu_ps(src.as_ptr())) }
        }
        #[inline(always)]
        unsafe fn store(self, dst: &mut [f32]) {
            debug_assert!(dst.len() >= Self::LANES);
            unsafe { _mm_storeu_ps(dst.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        unsafe fn add(self, other: Self) -> Self {
            unsafe { Self(_mm_add_ps(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn sub(self, other: Self) -> Self {
            unsafe { Self(_mm_sub_ps(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn mul(self, other: Self) -> Self {
            unsafe { Self(_mm_mul_ps(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn neg(self) -> Self {
            unsafe { Self(_mm_xor_ps(self.0, _mm_set1_ps(-0.0))) }
        }
        #[inline(always)]
        unsafe fn gt(self, other: Self) -> Self {
            unsafe { Self(_mm_cmpgt_ps(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn lt(self, other: Self) -> Self {
            unsafe { Self(_mm_cmplt_ps(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn select(mask: Self, a: Self, b: Self) -> Self {
            unsafe { Self(_mm_blendv_ps(b.0, a.0, mask.0)) }
        }
        // `maxps(a, b)` is exactly `a > b ? a : b` and `minps(a, b)` is
        // `a < b ? a : b`, returning `b` when either is NaN, so with the bound
        // first they are the two halves of `f32::clamp` in one instruction.
        #[inline(always)]
        unsafe fn raise_to(self, low: Self) -> Self {
            unsafe { Self(_mm_max_ps(low.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn lower_to(self, high: Self) -> Self {
            unsafe { Self(_mm_min_ps(high.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn reverse(self) -> Self {
            unsafe { Self(_mm_shuffle_ps::<0b00_01_10_11>(self.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn deinterleave(self, next: Self) -> (Self, Self) {
            unsafe {
                (
                    Self(_mm_shuffle_ps::<0b10_00_10_00>(self.0, next.0)),
                    Self(_mm_shuffle_ps::<0b11_01_11_01>(self.0, next.0)),
                )
            }
        }
        #[inline(always)]
        unsafe fn interleave(self, odd: Self) -> (Self, Self) {
            unsafe {
                (
                    Self(_mm_unpacklo_ps(self.0, odd.0)),
                    Self(_mm_unpackhi_ps(self.0, odd.0)),
                )
            }
        }
    }

    /// Eight lanes in an AVX register. The lane permutes cross the two 128-bit
    /// halves, which is what needs AVX2 rather than AVX.
    #[derive(Clone, Copy)]
    pub(crate) struct Avx2(__m256);

    impl F32x for Avx2 {
        const LANES: usize = 8;

        #[inline(always)]
        unsafe fn splat(value: f32) -> Self {
            unsafe { Self(_mm256_set1_ps(value)) }
        }
        #[inline(always)]
        unsafe fn load(src: &[f32]) -> Self {
            debug_assert!(src.len() >= Self::LANES);
            unsafe { Self(_mm256_loadu_ps(src.as_ptr())) }
        }
        #[inline(always)]
        unsafe fn store(self, dst: &mut [f32]) {
            debug_assert!(dst.len() >= Self::LANES);
            unsafe { _mm256_storeu_ps(dst.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        unsafe fn add(self, other: Self) -> Self {
            unsafe { Self(_mm256_add_ps(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn sub(self, other: Self) -> Self {
            unsafe { Self(_mm256_sub_ps(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn mul(self, other: Self) -> Self {
            unsafe { Self(_mm256_mul_ps(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn neg(self) -> Self {
            unsafe { Self(_mm256_xor_ps(self.0, _mm256_set1_ps(-0.0))) }
        }
        #[inline(always)]
        unsafe fn gt(self, other: Self) -> Self {
            unsafe { Self(_mm256_cmp_ps::<_CMP_GT_OQ>(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn lt(self, other: Self) -> Self {
            unsafe { Self(_mm256_cmp_ps::<_CMP_LT_OQ>(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn select(mask: Self, a: Self, b: Self) -> Self {
            unsafe { Self(_mm256_blendv_ps(b.0, a.0, mask.0)) }
        }
        // As for `Sse4`: exactly the two halves of `f32::clamp`.
        #[inline(always)]
        unsafe fn raise_to(self, low: Self) -> Self {
            unsafe { Self(_mm256_max_ps(low.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn lower_to(self, high: Self) -> Self {
            unsafe { Self(_mm256_min_ps(high.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn reverse(self) -> Self {
            unsafe {
                Self(_mm256_permutevar8x32_ps(
                    self.0,
                    _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0),
                ))
            }
        }
        #[inline(always)]
        unsafe fn deinterleave(self, next: Self) -> (Self, Self) {
            unsafe {
                // `shuffle_ps` works within each 128-bit half, leaving
                // `[a0 a2 b0 b2 | a4 a6 b4 b6]`; the 64-bit permute puts the
                // halves of `self` ahead of the halves of `next`.
                let even = _mm256_shuffle_ps::<0b10_00_10_00>(self.0, next.0);
                let odd = _mm256_shuffle_ps::<0b11_01_11_01>(self.0, next.0);
                (
                    Self(_mm256_castpd_ps(_mm256_permute4x64_pd::<0b11_01_10_00>(
                        _mm256_castps_pd(even),
                    ))),
                    Self(_mm256_castpd_ps(_mm256_permute4x64_pd::<0b11_01_10_00>(
                        _mm256_castps_pd(odd),
                    ))),
                )
            }
        }
        #[inline(always)]
        unsafe fn interleave(self, odd: Self) -> (Self, Self) {
            unsafe {
                // `unpack*_ps` interleave within each 128-bit half; the
                // cross-half permutes reassemble them in order.
                let low = _mm256_unpacklo_ps(self.0, odd.0);
                let high = _mm256_unpackhi_ps(self.0, odd.0);
                (
                    Self(_mm256_permute2f128_ps::<0x20>(low, high)),
                    Self(_mm256_permute2f128_ps::<0x31>(low, high)),
                )
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use x86::{Avx2, Sse4};

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::F32x;
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub(crate) struct Neon(float32x4_t);

    impl F32x for Neon {
        const LANES: usize = 4;

        #[inline(always)]
        unsafe fn splat(value: f32) -> Self {
            unsafe { Self(vdupq_n_f32(value)) }
        }
        #[inline(always)]
        unsafe fn load(src: &[f32]) -> Self {
            debug_assert!(src.len() >= Self::LANES);
            unsafe { Self(vld1q_f32(src.as_ptr())) }
        }
        #[inline(always)]
        unsafe fn store(self, dst: &mut [f32]) {
            debug_assert!(dst.len() >= Self::LANES);
            unsafe { vst1q_f32(dst.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        unsafe fn add(self, other: Self) -> Self {
            unsafe { Self(vaddq_f32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn sub(self, other: Self) -> Self {
            unsafe { Self(vsubq_f32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn mul(self, other: Self) -> Self {
            unsafe { Self(vmulq_f32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn neg(self) -> Self {
            unsafe { Self(vnegq_f32(self.0)) }
        }
        #[inline(always)]
        unsafe fn gt(self, other: Self) -> Self {
            unsafe { Self(vreinterpretq_f32_u32(vcgtq_f32(self.0, other.0))) }
        }
        #[inline(always)]
        unsafe fn lt(self, other: Self) -> Self {
            unsafe { Self(vreinterpretq_f32_u32(vcltq_f32(self.0, other.0))) }
        }
        #[inline(always)]
        unsafe fn select(mask: Self, a: Self, b: Self) -> Self {
            unsafe { Self(vbslq_f32(vreinterpretq_u32_f32(mask.0), a.0, b.0)) }
        }
        // `raise_to`/`lower_to` keep the compare-and-select defaults: NEON's
        // `fmax`/`fmin` return a NaN when either operand is one, not the
        // second operand, so they are not `f32::clamp`.
        #[inline(always)]
        unsafe fn reverse(self) -> Self {
            unsafe {
                let pairs = vrev64q_f32(self.0);
                Self(vextq_f32::<2>(pairs, pairs))
            }
        }
        #[inline(always)]
        unsafe fn deinterleave(self, next: Self) -> (Self, Self) {
            unsafe {
                (
                    Self(vuzp1q_f32(self.0, next.0)),
                    Self(vuzp2q_f32(self.0, next.0)),
                )
            }
        }
        #[inline(always)]
        unsafe fn interleave(self, odd: Self) -> (Self, Self) {
            unsafe {
                (
                    Self(vzip1q_f32(self.0, odd.0)),
                    Self(vzip2q_f32(self.0, odd.0)),
                )
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) use arm::Neon;
