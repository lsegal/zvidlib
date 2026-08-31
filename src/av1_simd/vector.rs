//! A minimal, width-agnostic signed 32-bit integer SIMD abstraction.
//!
//! Every vectorized AV1 kernel in this module tree is written **once**, generic
//! over [`I32x`], and then instantiated once per instruction set behind a
//! `#[target_feature]` wrapper in the kernel modules. That keeps a single
//! readable copy of each algorithm (which is what makes bit-exactness against
//! the scalar reference reviewable) while still emitting real SSE4.1, AVX2, and
//! NEON code.
//!
//! The trait's methods are deliberately *not* themselves annotated with
//! `#[target_feature]`: they are `#[inline(always)]` and are only ever reached
//! from a wrapper function that already carries the required feature, so LLVM
//! inlines them with the wrapper's feature set. Calling one from a context that
//! has not verified CPU support is undefined behaviour, which is why every
//! method is `unsafe` and every dispatcher in [`super`] checks support first.

#![allow(clippy::missing_safety_doc)]

/// The widest lane count any [`I32x`] implementation in this module uses.
/// Scratch buffers are sized to this so a single stack array serves all ISAs.
pub(crate) const MAX_LANES: usize = 8;

/// Elementwise signed 32-bit integer vector operations.
///
/// # Safety
///
/// Every method may execute instructions that are not part of the target's
/// baseline ISA. A caller must only reach these through a wrapper that has
/// verified the corresponding CPU feature is present.
pub(crate) trait I32x: Copy {
    /// Number of `i32` lanes this vector holds.
    const LANES: usize;

    unsafe fn splat(value: i32) -> Self;
    /// Loads `LANES` lanes starting at the front of `src` (`src.len() >= LANES`).
    unsafe fn load(src: &[i32]) -> Self;
    /// Stores `LANES` lanes to the front of `dst` (`dst.len() >= LANES`).
    unsafe fn store(self, dst: &mut [i32]);
    /// Zero-extends `LANES` bytes from the front of `src` (`src.len() >= LANES`).
    unsafe fn load_u8(src: &[u8]) -> Self;
    /// Clamps to `0..=255` and stores all `LANES` lanes as consecutive bytes at
    /// the front of `dst` (`dst.len() >= LANES`).
    ///
    /// Every implementation narrows 32-bit lanes to bytes with its own
    /// saturating pack instructions (`packus`, `vqmovun`/`vqmovn`) and writes
    /// the result as one machine word, so the whole-vector store never goes
    /// through a stack scratch buffer. This is the store the deblocking,
    /// CDEF and Wiener kernels take for all but a plane's trailing positions.
    unsafe fn store_u8_clamped(self, dst: &mut [u8]);

    unsafe fn add(self, other: Self) -> Self;
    unsafe fn sub(self, other: Self) -> Self;
    unsafe fn mul(self, other: Self) -> Self;
    unsafe fn and(self, other: Self) -> Self;
    unsafe fn or(self, other: Self) -> Self;
    /// `(!self) & other`.
    unsafe fn andnot(self, other: Self) -> Self;
    unsafe fn min(self, other: Self) -> Self;
    unsafe fn max(self, other: Self) -> Self;
    unsafe fn abs(self) -> Self;
    /// Arithmetic right shift by a compile-time constant in `0..32`.
    unsafe fn sra<const N: i32>(self) -> Self;
    /// Logical (zero-filling) shift right by a constant, used to pull the
    /// individual bytes out of a packed 32-bit row word.
    unsafe fn srl<const N: i32>(self) -> Self;
    /// Shift left by a constant, the inverse of [`I32x::srl`] used to pack
    /// filtered bytes back into a row word.
    unsafe fn sll<const N: i32>(self) -> Self;
    /// Arithmetic right shift by a runtime-uniform amount in `0..32`.
    unsafe fn sra_var(self, bits: i32) -> Self;
    /// Lane mask: all ones where `self > other`, zero elsewhere.
    unsafe fn gt(self, other: Self) -> Self;
    /// Sum of all lanes.
    unsafe fn hsum(self) -> i32;
    /// Lane-wise `self / divisor`, truncating toward zero.
    ///
    /// Only valid for non-negative lanes with `1 <= divisor` and
    /// `self <= 255 * divisor + divisor / 2`, which is the range the wide
    /// deblocking taper in [`super::filters`] produces (a weighted average of
    /// 8-bit samples, so the quotient never exceeds 255). Within that range the
    /// `f32` round trip these implementations use is exact: `f32` represents
    /// both operands exactly, IEEE division is correctly rounded, and a
    /// quotient of at most `255.5` has an absolute rounding error under
    /// `2^-16`, far below the `1/divisor >= 1/255` distance from a
    /// non-integral quotient to the next integer.
    unsafe fn div_small_nonneg(self, divisor: i32) -> Self;

    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self::splat(0) }
    }

    /// `mask ? a : b`, where `mask` lanes are all-ones or all-zero.
    #[inline(always)]
    unsafe fn select(mask: Self, a: Self, b: Self) -> Self {
        unsafe { mask.and(a).or(mask.andnot(b)) }
    }

    /// Lane mask: all ones where `self <= other`.
    #[inline(always)]
    unsafe fn le(self, other: Self) -> Self {
        unsafe { self.gt(other).andnot(Self::splat(-1)) }
    }

    /// `-1`, `0`, or `1` per lane, matching `i32::signum`.
    #[inline(always)]
    unsafe fn signum(self) -> Self {
        unsafe {
            let zero = Self::zero();
            zero.gt(self).sub(self.gt(zero))
        }
    }

    #[inline(always)]
    unsafe fn clamp(self, low: Self, high: Self) -> Self {
        unsafe { self.max(low).min(high) }
    }

    /// True when any lane of a comparison mask is set.
    #[inline(always)]
    unsafe fn any(self) -> bool {
        unsafe { self.hsum() != 0 }
    }

    /// Stores the first `count` lanes (`count <= LANES`) to `dst`, leaving the
    /// remaining lanes unwritten. This is how the kernels finish a partial row
    /// or column at the right or bottom edge of a plane without spilling into
    /// samples the caller did not ask for.
    #[inline(always)]
    unsafe fn store_masked(self, dst: &mut [i32], count: usize) {
        unsafe {
            // Whole vectors are the common case (only a plane's trailing
            // positions are partial), and they store natively.
            if count == Self::LANES {
                self.store(dst);
                return;
            }
            let mut scratch = [0i32; MAX_LANES];
            self.store(&mut scratch);
            dst[..count].copy_from_slice(&scratch[..count]);
        }
    }

    /// Clamps to `0..=255` and stores the first `count` lanes as bytes.
    ///
    /// A whole vector - every position but a plane's trailing ones - goes
    /// straight to [`I32x::store_u8_clamped`]'s native saturating pack. Only a
    /// partial tail stages through a stack buffer, because masking a packed
    /// byte store differs enough between the three instruction sets (and
    /// AVX2's `packus` crosses its 128-bit halves) not to be worth writing
    /// three times for the last few positions of a row.
    #[inline(always)]
    unsafe fn store_u8_clamped_masked(self, dst: &mut [u8], count: usize) {
        unsafe {
            if count == Self::LANES {
                self.store_u8_clamped(dst);
                return;
            }
            let mut scratch = [0i32; MAX_LANES];
            self.clamp(Self::zero(), Self::splat(255))
                .store(&mut scratch);
            for (out, &value) in dst.iter_mut().zip(scratch.iter()).take(count) {
                *out = value as u8;
            }
        }
    }

    /// Loads one little-endian 32-bit word per lane, lane `i` taking the four
    /// bytes at `src[base + i * stride..]`.
    ///
    /// This is how a kernel whose filter window is four consecutive bytes of
    /// each of `LANES` rows - the narrow vertical-edge deblocking filter -
    /// reads its window: one word per row instead of one byte per tap per row,
    /// with [`I32x::srl`] separating the taps afterwards.
    #[inline(always)]
    unsafe fn load_u32_rows(src: &[u8], base: usize, stride: usize) -> Self {
        let mut words = [0i32; MAX_LANES];
        for (lane, slot) in words.iter_mut().enumerate().take(Self::LANES) {
            let at = base + lane * stride;
            *slot = i32::from_le_bytes([src[at], src[at + 1], src[at + 2], src[at + 3]]);
        }
        unsafe { Self::load(&words) }
    }

    /// Writes each lane back as the little-endian 32-bit word at
    /// `dst[base + i * stride..]`, the inverse of [`I32x::load_u32_rows`].
    #[inline(always)]
    unsafe fn store_u32_rows(self, dst: &mut [u8], base: usize, stride: usize) {
        let mut words = [0i32; MAX_LANES];
        unsafe { self.store(&mut words) };
        for (lane, &word) in words.iter().enumerate().take(Self::LANES) {
            let at = base + lane * stride;
            dst[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
    }
}

/// A 4-lane vector that can transpose a 4x4 block of lanes.
///
/// The separable 4- and 8-point transforms need to swap rows and columns
/// between their two passes; only the 128-bit implementations provide it,
/// because a 4x4 or 8x8 coefficient block has no useful 256-bit shape (AVX2
/// hosts run the transforms through the SSE4.1 path and spend their wider
/// registers on the pixel filters instead).
pub(crate) trait Transpose4: I32x {
    /// Returns `out` such that `out[j]` lane `i` equals `rows[i]` lane `j`.
    unsafe fn transpose4(rows: [Self; 4]) -> [Self; 4];
}

// ---------------------------------------------------------------------
// x86_64: SSE4.1 (128-bit, 4 lanes) and AVX2 (256-bit, 8 lanes)
// ---------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{I32x, Transpose4};
    use core::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub(crate) struct Sse4(__m128i);

    impl I32x for Sse4 {
        const LANES: usize = 4;

        #[inline(always)]
        unsafe fn splat(value: i32) -> Self {
            unsafe { Self(_mm_set1_epi32(value)) }
        }
        #[inline(always)]
        unsafe fn load(src: &[i32]) -> Self {
            unsafe { Self(_mm_loadu_si128(src.as_ptr().cast())) }
        }
        #[inline(always)]
        unsafe fn store(self, dst: &mut [i32]) {
            unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast(), self.0) }
        }
        #[inline(always)]
        unsafe fn load_u8(src: &[u8]) -> Self {
            unsafe {
                let bytes = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
                Self(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(bytes as i32)))
            }
        }
        #[inline(always)]
        unsafe fn add(self, other: Self) -> Self {
            unsafe { Self(_mm_add_epi32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn sub(self, other: Self) -> Self {
            unsafe { Self(_mm_sub_epi32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn mul(self, other: Self) -> Self {
            unsafe { Self(_mm_mullo_epi32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn and(self, other: Self) -> Self {
            unsafe { Self(_mm_and_si128(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn or(self, other: Self) -> Self {
            unsafe { Self(_mm_or_si128(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn andnot(self, other: Self) -> Self {
            unsafe { Self(_mm_andnot_si128(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn min(self, other: Self) -> Self {
            unsafe { Self(_mm_min_epi32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn max(self, other: Self) -> Self {
            unsafe { Self(_mm_max_epi32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn abs(self) -> Self {
            unsafe { Self(_mm_abs_epi32(self.0)) }
        }
        #[inline(always)]
        unsafe fn sra<const N: i32>(self) -> Self {
            unsafe { Self(_mm_srai_epi32::<N>(self.0)) }
        }
        #[inline(always)]
        unsafe fn srl<const N: i32>(self) -> Self {
            unsafe { Self(_mm_srli_epi32::<N>(self.0)) }
        }
        #[inline(always)]
        unsafe fn sll<const N: i32>(self) -> Self {
            unsafe { Self(_mm_slli_epi32::<N>(self.0)) }
        }
        #[inline(always)]
        unsafe fn load_u32_rows(src: &[u8], base: usize, stride: usize) -> Self {
            unsafe {
                let word = |lane: usize| {
                    let at = base + lane * stride;
                    i32::from_le_bytes([src[at], src[at + 1], src[at + 2], src[at + 3]])
                };
                Self(_mm_set_epi32(word(3), word(2), word(1), word(0)))
            }
        }
        #[inline(always)]
        unsafe fn store_u32_rows(self, dst: &mut [u8], base: usize, stride: usize) {
            unsafe {
                let words = [
                    _mm_extract_epi32::<0>(self.0) as u32,
                    _mm_extract_epi32::<1>(self.0) as u32,
                    _mm_extract_epi32::<2>(self.0) as u32,
                    _mm_extract_epi32::<3>(self.0) as u32,
                ];
                for (lane, word) in words.into_iter().enumerate() {
                    let at = base + lane * stride;
                    dst[at..at + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
        }
        #[inline(always)]
        unsafe fn store_u8_clamped(self, dst: &mut [u8]) {
            unsafe {
                // `packus_epi16` reads its inputs as *signed* 16-bit, so the
                // range clamp has to happen before the first pack rather than
                // being left to it.
                let clamped = _mm_min_epi32(
                    _mm_max_epi32(self.0, _mm_setzero_si128()),
                    _mm_set1_epi32(255),
                );
                let halves = _mm_packus_epi32(clamped, clamped);
                let bytes = _mm_packus_epi16(halves, halves);
                let word = _mm_cvtsi128_si32(bytes) as u32;
                dst[..4].copy_from_slice(&word.to_le_bytes());
            }
        }
        #[inline(always)]
        unsafe fn sra_var(self, bits: i32) -> Self {
            unsafe { Self(_mm_sra_epi32(self.0, _mm_cvtsi32_si128(bits))) }
        }
        #[inline(always)]
        unsafe fn gt(self, other: Self) -> Self {
            unsafe { Self(_mm_cmpgt_epi32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn hsum(self) -> i32 {
            unsafe {
                let pairs = _mm_hadd_epi32(self.0, self.0);
                _mm_cvtsi128_si32(_mm_hadd_epi32(pairs, pairs))
            }
        }
        #[inline(always)]
        unsafe fn div_small_nonneg(self, divisor: i32) -> Self {
            unsafe {
                let quotient = _mm_div_ps(_mm_cvtepi32_ps(self.0), _mm_set1_ps(divisor as f32));
                Self(_mm_cvttps_epi32(quotient))
            }
        }
    }

    impl Transpose4 for Sse4 {
        #[inline(always)]
        unsafe fn transpose4(rows: [Self; 4]) -> [Self; 4] {
            unsafe {
                let a = _mm_unpacklo_epi32(rows[0].0, rows[1].0);
                let b = _mm_unpackhi_epi32(rows[0].0, rows[1].0);
                let c = _mm_unpacklo_epi32(rows[2].0, rows[3].0);
                let d = _mm_unpackhi_epi32(rows[2].0, rows[3].0);
                [
                    Self(_mm_unpacklo_epi64(a, c)),
                    Self(_mm_unpackhi_epi64(a, c)),
                    Self(_mm_unpacklo_epi64(b, d)),
                    Self(_mm_unpackhi_epi64(b, d)),
                ]
            }
        }
    }

    #[derive(Clone, Copy)]
    pub(crate) struct Avx2(__m256i);

    impl I32x for Avx2 {
        const LANES: usize = 8;

        #[inline(always)]
        unsafe fn splat(value: i32) -> Self {
            unsafe { Self(_mm256_set1_epi32(value)) }
        }
        #[inline(always)]
        unsafe fn load(src: &[i32]) -> Self {
            unsafe { Self(_mm256_loadu_si256(src.as_ptr().cast())) }
        }
        #[inline(always)]
        unsafe fn store(self, dst: &mut [i32]) {
            unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), self.0) }
        }
        #[inline(always)]
        unsafe fn load_u8(src: &[u8]) -> Self {
            unsafe {
                let lo = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
                let hi = u32::from_le_bytes([src[4], src[5], src[6], src[7]]);
                let packed = _mm_set_epi32(0, 0, hi as i32, lo as i32);
                Self(_mm256_cvtepu8_epi32(packed))
            }
        }
        #[inline(always)]
        unsafe fn add(self, other: Self) -> Self {
            unsafe { Self(_mm256_add_epi32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn sub(self, other: Self) -> Self {
            unsafe { Self(_mm256_sub_epi32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn mul(self, other: Self) -> Self {
            unsafe { Self(_mm256_mullo_epi32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn and(self, other: Self) -> Self {
            unsafe { Self(_mm256_and_si256(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn or(self, other: Self) -> Self {
            unsafe { Self(_mm256_or_si256(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn andnot(self, other: Self) -> Self {
            unsafe { Self(_mm256_andnot_si256(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn min(self, other: Self) -> Self {
            unsafe { Self(_mm256_min_epi32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn max(self, other: Self) -> Self {
            unsafe { Self(_mm256_max_epi32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn abs(self) -> Self {
            unsafe { Self(_mm256_abs_epi32(self.0)) }
        }
        #[inline(always)]
        unsafe fn sra<const N: i32>(self) -> Self {
            unsafe { Self(_mm256_srai_epi32::<N>(self.0)) }
        }
        #[inline(always)]
        unsafe fn srl<const N: i32>(self) -> Self {
            unsafe { Self(_mm256_srli_epi32::<N>(self.0)) }
        }
        #[inline(always)]
        unsafe fn sll<const N: i32>(self) -> Self {
            unsafe { Self(_mm256_slli_epi32::<N>(self.0)) }
        }
        #[inline(always)]
        unsafe fn load_u32_rows(src: &[u8], base: usize, stride: usize) -> Self {
            unsafe {
                let word = |lane: usize| {
                    let at = base + lane * stride;
                    i32::from_le_bytes([src[at], src[at + 1], src[at + 2], src[at + 3]])
                };
                let lo = _mm_set_epi32(word(3), word(2), word(1), word(0));
                let hi = _mm_set_epi32(word(7), word(6), word(5), word(4));
                Self(_mm256_set_m128i(hi, lo))
            }
        }
        #[inline(always)]
        unsafe fn store_u32_rows(self, dst: &mut [u8], base: usize, stride: usize) {
            unsafe {
                let halves = [
                    _mm256_castsi256_si128(self.0),
                    _mm256_extracti128_si256::<1>(self.0),
                ];
                for (half, lanes) in halves.into_iter().enumerate() {
                    let words = [
                        _mm_extract_epi32::<0>(lanes) as u32,
                        _mm_extract_epi32::<1>(lanes) as u32,
                        _mm_extract_epi32::<2>(lanes) as u32,
                        _mm_extract_epi32::<3>(lanes) as u32,
                    ];
                    for (lane, word) in words.into_iter().enumerate() {
                        let at = base + (half * 4 + lane) * stride;
                        dst[at..at + 4].copy_from_slice(&word.to_le_bytes());
                    }
                }
            }
        }
        #[inline(always)]
        unsafe fn store_u8_clamped(self, dst: &mut [u8]) {
            unsafe {
                let clamped = _mm256_min_epi32(
                    _mm256_max_epi32(self.0, _mm256_setzero_si256()),
                    _mm256_set1_epi32(255),
                );
                // Packing the two 128-bit halves against each other rather than
                // `clamped` against itself is what keeps the eight bytes in
                // lane order despite `packus` working within each half.
                let lo = _mm256_castsi256_si128(clamped);
                let hi = _mm256_extracti128_si256::<1>(clamped);
                let halves = _mm_packus_epi32(lo, hi);
                let bytes = _mm_packus_epi16(halves, halves);
                let word = _mm_cvtsi128_si64(bytes) as u64;
                dst[..8].copy_from_slice(&word.to_le_bytes());
            }
        }
        #[inline(always)]
        unsafe fn sra_var(self, bits: i32) -> Self {
            unsafe { Self(_mm256_sra_epi32(self.0, _mm_cvtsi32_si128(bits))) }
        }
        #[inline(always)]
        unsafe fn gt(self, other: Self) -> Self {
            unsafe { Self(_mm256_cmpgt_epi32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn hsum(self) -> i32 {
            unsafe {
                let lo = _mm256_castsi256_si128(self.0);
                let hi = _mm256_extracti128_si256::<1>(self.0);
                let sum = _mm_add_epi32(lo, hi);
                let pairs = _mm_hadd_epi32(sum, sum);
                _mm_cvtsi128_si32(_mm_hadd_epi32(pairs, pairs))
            }
        }
        #[inline(always)]
        unsafe fn div_small_nonneg(self, divisor: i32) -> Self {
            unsafe {
                let quotient =
                    _mm256_div_ps(_mm256_cvtepi32_ps(self.0), _mm256_set1_ps(divisor as f32));
                Self(_mm256_cvttps_epi32(quotient))
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use x86::{Avx2, Sse4};

// ---------------------------------------------------------------------
// aarch64: NEON (128-bit, 4 lanes)
// ---------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{I32x, Transpose4};
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub(crate) struct Neon(int32x4_t);

    impl I32x for Neon {
        const LANES: usize = 4;

        #[inline(always)]
        unsafe fn splat(value: i32) -> Self {
            unsafe { Self(vdupq_n_s32(value)) }
        }
        #[inline(always)]
        unsafe fn load(src: &[i32]) -> Self {
            unsafe { Self(vld1q_s32(src.as_ptr())) }
        }
        #[inline(always)]
        unsafe fn store(self, dst: &mut [i32]) {
            unsafe { vst1q_s32(dst.as_mut_ptr(), self.0) }
        }
        #[inline(always)]
        unsafe fn load_u8(src: &[u8]) -> Self {
            unsafe {
                let bytes = u32::from_le_bytes([src[0], src[1], src[2], src[3]]);
                let lanes = vreinterpret_u8_u32(vdup_n_u32(bytes));
                let widened = vmovl_u16(vget_low_u16(vmovl_u8(lanes)));
                Self(vreinterpretq_s32_u32(widened))
            }
        }
        #[inline(always)]
        unsafe fn add(self, other: Self) -> Self {
            unsafe { Self(vaddq_s32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn sub(self, other: Self) -> Self {
            unsafe { Self(vsubq_s32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn mul(self, other: Self) -> Self {
            unsafe { Self(vmulq_s32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn and(self, other: Self) -> Self {
            unsafe { Self(vandq_s32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn or(self, other: Self) -> Self {
            unsafe { Self(vorrq_s32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn andnot(self, other: Self) -> Self {
            // `vbicq` computes `a & !b`.
            unsafe { Self(vbicq_s32(other.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn min(self, other: Self) -> Self {
            unsafe { Self(vminq_s32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn max(self, other: Self) -> Self {
            unsafe { Self(vmaxq_s32(self.0, other.0)) }
        }
        #[inline(always)]
        unsafe fn abs(self) -> Self {
            unsafe { Self(vabsq_s32(self.0)) }
        }
        #[inline(always)]
        unsafe fn sra<const N: i32>(self) -> Self {
            unsafe {
                if N == 0 {
                    self
                } else {
                    Self(vshlq_s32(self.0, vdupq_n_s32(-N)))
                }
            }
        }
        #[inline(always)]
        unsafe fn srl<const N: i32>(self) -> Self {
            unsafe {
                let lanes = vreinterpretq_u32_s32(self.0);
                Self(vreinterpretq_s32_u32(vshrq_n_u32::<N>(lanes)))
            }
        }
        #[inline(always)]
        unsafe fn sll<const N: i32>(self) -> Self {
            unsafe { Self(vshlq_n_s32::<N>(self.0)) }
        }
        #[inline(always)]
        unsafe fn load_u32_rows(src: &[u8], base: usize, stride: usize) -> Self {
            unsafe {
                let word = |lane: usize| {
                    let at = base + lane * stride;
                    u32::from_le_bytes([src[at], src[at + 1], src[at + 2], src[at + 3]])
                };
                let words = vdupq_n_u32(word(0));
                let words = vsetq_lane_u32::<1>(word(1), words);
                let words = vsetq_lane_u32::<2>(word(2), words);
                let words = vsetq_lane_u32::<3>(word(3), words);
                Self(vreinterpretq_s32_u32(words))
            }
        }
        #[inline(always)]
        unsafe fn store_u32_rows(self, dst: &mut [u8], base: usize, stride: usize) {
            unsafe {
                let lanes = vreinterpretq_u32_s32(self.0);
                let words = [
                    vgetq_lane_u32::<0>(lanes),
                    vgetq_lane_u32::<1>(lanes),
                    vgetq_lane_u32::<2>(lanes),
                    vgetq_lane_u32::<3>(lanes),
                ];
                for (lane, word) in words.into_iter().enumerate() {
                    let at = base + lane * stride;
                    dst[at..at + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
        }
        #[inline(always)]
        unsafe fn store_u8_clamped(self, dst: &mut [u8]) {
            unsafe {
                // `vqmovun_s32` saturates a negative lane to 0 and anything
                // above 65535 to 65535, and `vqmovn_u16` then saturates to
                // 255, so the pair performs the `0..=255` clamp itself.
                let halves = vqmovun_s32(self.0);
                let bytes = vqmovn_u16(vcombine_u16(halves, halves));
                let word = vget_lane_u32::<0>(vreinterpret_u32_u8(bytes));
                dst[..4].copy_from_slice(&word.to_le_bytes());
            }
        }
        #[inline(always)]
        unsafe fn sra_var(self, bits: i32) -> Self {
            unsafe { Self(vshlq_s32(self.0, vdupq_n_s32(-bits))) }
        }
        #[inline(always)]
        unsafe fn gt(self, other: Self) -> Self {
            unsafe { Self(vreinterpretq_s32_u32(vcgtq_s32(self.0, other.0))) }
        }
        #[inline(always)]
        unsafe fn hsum(self) -> i32 {
            unsafe { vaddvq_s32(self.0) }
        }
        #[inline(always)]
        unsafe fn div_small_nonneg(self, divisor: i32) -> Self {
            unsafe {
                let quotient = vdivq_f32(vcvtq_f32_s32(self.0), vdupq_n_f32(divisor as f32));
                // `vcvtq_s32_f32` rounds toward zero, which is the floor the
                // scalar reference's non-negative integer division performs.
                Self(vcvtq_s32_f32(quotient))
            }
        }
    }

    impl Transpose4 for Neon {
        #[inline(always)]
        unsafe fn transpose4(rows: [Self; 4]) -> [Self; 4] {
            unsafe {
                let a = vtrn1q_s32(rows[0].0, rows[1].0);
                let b = vtrn2q_s32(rows[0].0, rows[1].0);
                let c = vtrn1q_s32(rows[2].0, rows[3].0);
                let d = vtrn2q_s32(rows[2].0, rows[3].0);
                let (a, b) = (vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b));
                let (c, d) = (vreinterpretq_s64_s32(c), vreinterpretq_s64_s32(d));
                [
                    Self(vreinterpretq_s32_s64(vtrn1q_s64(a, c))),
                    Self(vreinterpretq_s32_s64(vtrn1q_s64(b, d))),
                    Self(vreinterpretq_s32_s64(vtrn2q_s64(a, c))),
                    Self(vreinterpretq_s32_s64(vtrn2q_s64(b, d))),
                ]
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) use arm::Neon;
