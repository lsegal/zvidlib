//! The four-lane `f32`/`i32` vocabulary the Vorbis encoder kernels are written
//! in, implemented once for x86_64 (SSE4.1, also compiled into the AVX2 entry
//! points) and once for aarch64 NEON.
//!
//! Every operation is a single IEEE-754 operation per lane, or a pure lane
//! permutation, so a kernel written in this vocabulary computes exactly what
//! the scalar code it mirrors computes as long as it issues the same
//! operations in the same order. There is deliberately no fused multiply-add:
//! the scalar reference never contracts, so neither may the kernels.
//!
//! Every method is `#[inline(always)]`. The kernels are generic over these
//! types and are only compiled with their instruction set enabled when they
//! are inlined into the `#[target_feature]` entry points in [`super`]; an
//! out-of-line copy would be built at the baseline instruction set (see the
//! note on `simd_entry_points!` there and issue #336).

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Four `f32` lanes. Lane 0 is the lowest address of a load.
///
/// # Safety
///
/// Every method is `unsafe` because it executes instructions of the
/// implementing instruction set; callers must have established that the host
/// supports it. Pointer arguments must be valid for the bytes they name.
pub(super) trait F32x4: Copy {
    /// The integer vector of the same width.
    type I: I32x4;

    unsafe fn splat(v: f32) -> Self;
    unsafe fn from_array(a: [f32; 4]) -> Self;
    /// Four consecutive floats from `p` (no alignment requirement).
    unsafe fn load(p: *const f32) -> Self;
    unsafe fn store(self, p: *mut f32);
    /// `(lo[0], lo[1], hi[0], hi[1])`.
    unsafe fn load_pairs(lo: *const f32, hi: *const f32) -> Self;

    unsafe fn add(self, b: Self) -> Self;
    unsafe fn sub(self, b: Self) -> Self;
    unsafe fn mul(self, b: Self) -> Self;
    unsafe fn div(self, b: Self) -> Self;
    /// Flips the sign bit of every lane, as scalar `-x` does.
    unsafe fn neg(self) -> Self;
    /// Flips the sign bit of lanes 1 and 3.
    unsafe fn neg_odd(self) -> Self;

    /// `(a0, a0, a2, a2)`.
    unsafe fn dup_even(self) -> Self;
    /// `(a1, a1, a3, a3)`.
    unsafe fn dup_odd(self) -> Self;
    /// `(a1, a0, a3, a2)`.
    unsafe fn swap_pairs(self) -> Self;
    /// `(a2, a3, a0, a1)`.
    unsafe fn swap_halves(self) -> Self;
    /// `(a3, a2, a1, a0)`.
    unsafe fn reverse(self) -> Self;
    /// `((a0, a2, b0, b2), (a1, a3, b1, b3))`.
    unsafe fn uzp(self, b: Self) -> (Self, Self);
    /// `((a0, b0, a1, b1), (a2, b2, a3, b3))`.
    unsafe fn zip(self, b: Self) -> (Self, Self);
    /// `(a0, b0, a2, b2)`.
    unsafe fn trn1(self, b: Self) -> Self;
    /// `(a1, b1, a3, b3)`.
    unsafe fn trn2(self, b: Self) -> Self;
    /// `(a0, a1, b0, b1)`.
    unsafe fn low_halves(self, b: Self) -> Self;
    /// `(a2, a3, b2, b3)`.
    unsafe fn high_halves(self, b: Self) -> Self;
    /// `(a0, b1, a2, b3)`.
    unsafe fn blend_odd(self, b: Self) -> Self;

    /// Per lane, `if self < b { x } else { y }` (false for NaN, as in scalar).
    unsafe fn select_lt(self, b: Self, x: Self, y: Self) -> Self;
    /// Per lane, all ones where `self >= b` (false for NaN, as in scalar).
    unsafe fn ge_mask(self, b: Self) -> Self::I;

    /// scales.h `todB`: `(bits & 0x7fffffff) as f32 * 7.17711438e-7 - 764.6161886`.
    unsafe fn todb(self) -> Self;
    /// `((self as f64) + c) as f32`, the libvorbis `float + double` promotion.
    unsafe fn add_f64(self, c: f64) -> Self;
    /// `(self as i32).clamp(0, hi)`, Rust's saturating cast included (NaN is
    /// 0). `hi` must be a non-negative integer-valued float.
    unsafe fn trunc_clamp(self, hi: f32) -> Self::I;
    /// `(((self as f64) + c) as i32).clamp(0, hi)` with the same saturation.
    /// `hi` must be a non-negative integer-valued float.
    unsafe fn add_f64_trunc_clamp(self, c: f64, hi: f64) -> Self::I;
}

/// Four `i32` lanes, with wrapping arithmetic.
pub(super) trait I32x4: Copy {
    unsafe fn splat(v: i32) -> Self;
    unsafe fn from_array(a: [i32; 4]) -> Self;
    unsafe fn to_array(self) -> [i32; 4];
    unsafe fn add(self, b: Self) -> Self;
    unsafe fn sub(self, b: Self) -> Self;
    unsafe fn mul(self, b: Self) -> Self;
    unsafe fn and(self, b: Self) -> Self;
    /// `self & !b`.
    unsafe fn and_not(self, b: Self) -> Self;
    /// All ones where the lane is zero.
    unsafe fn eq_zero(self) -> Self;
    /// Wrapping sum of the four lanes.
    unsafe fn sum(self) -> i32;
}

#[cfg(target_arch = "x86_64")]
pub(super) use x86::Sse;

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;

    /// SSE4.1 lanes. The AVX2 entry points use this type too and get the same
    /// instructions VEX-encoded: every kernel shape here is four lanes wide.
    #[derive(Clone, Copy)]
    pub(in super::super) struct Sse(__m128);

    #[derive(Clone, Copy)]
    pub(in super::super) struct SseI(__m128i);

    impl F32x4 for Sse {
        type I = SseI;

        #[inline(always)]
        unsafe fn splat(v: f32) -> Self {
            unsafe { Sse(_mm_set1_ps(v)) }
        }
        #[inline(always)]
        unsafe fn from_array(a: [f32; 4]) -> Self {
            unsafe { Sse(_mm_loadu_ps(a.as_ptr())) }
        }
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            unsafe { Sse(_mm_loadu_ps(p)) }
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut f32) {
            unsafe { _mm_storeu_ps(p, self.0) }
        }
        #[inline(always)]
        unsafe fn load_pairs(lo: *const f32, hi: *const f32) -> Self {
            unsafe {
                let lo = _mm_load_sd(lo.cast::<f64>());
                Sse(_mm_castpd_ps(_mm_loadh_pd(lo, hi.cast::<f64>())))
            }
        }

        #[inline(always)]
        unsafe fn add(self, b: Self) -> Self {
            unsafe { Sse(_mm_add_ps(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn sub(self, b: Self) -> Self {
            unsafe { Sse(_mm_sub_ps(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn mul(self, b: Self) -> Self {
            unsafe { Sse(_mm_mul_ps(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn div(self, b: Self) -> Self {
            unsafe { Sse(_mm_div_ps(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn neg(self) -> Self {
            unsafe { Sse(_mm_xor_ps(self.0, _mm_set1_ps(-0.0))) }
        }
        #[inline(always)]
        unsafe fn neg_odd(self) -> Self {
            unsafe { Sse(_mm_xor_ps(self.0, _mm_setr_ps(0.0, -0.0, 0.0, -0.0))) }
        }

        #[inline(always)]
        unsafe fn dup_even(self) -> Self {
            unsafe { Sse(_mm_moveldup_ps(self.0)) }
        }
        #[inline(always)]
        unsafe fn dup_odd(self) -> Self {
            unsafe { Sse(_mm_movehdup_ps(self.0)) }
        }
        #[inline(always)]
        unsafe fn swap_pairs(self) -> Self {
            unsafe { Sse(_mm_shuffle_ps::<0b10_11_00_01>(self.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn swap_halves(self) -> Self {
            unsafe { Sse(_mm_shuffle_ps::<0b01_00_11_10>(self.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn reverse(self) -> Self {
            unsafe { Sse(_mm_shuffle_ps::<0b00_01_10_11>(self.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn uzp(self, b: Self) -> (Self, Self) {
            unsafe {
                (
                    Sse(_mm_shuffle_ps::<0b10_00_10_00>(self.0, b.0)),
                    Sse(_mm_shuffle_ps::<0b11_01_11_01>(self.0, b.0)),
                )
            }
        }
        #[inline(always)]
        unsafe fn zip(self, b: Self) -> (Self, Self) {
            unsafe {
                (
                    Sse(_mm_unpacklo_ps(self.0, b.0)),
                    Sse(_mm_unpackhi_ps(self.0, b.0)),
                )
            }
        }
        #[inline(always)]
        unsafe fn trn1(self, b: Self) -> Self {
            unsafe {
                let even = _mm_shuffle_ps::<0b10_00_10_00>(self.0, b.0);
                Sse(_mm_shuffle_ps::<0b11_01_10_00>(even, even))
            }
        }
        #[inline(always)]
        unsafe fn trn2(self, b: Self) -> Self {
            unsafe {
                let odd = _mm_shuffle_ps::<0b11_01_11_01>(self.0, b.0);
                Sse(_mm_shuffle_ps::<0b11_01_10_00>(odd, odd))
            }
        }
        #[inline(always)]
        unsafe fn low_halves(self, b: Self) -> Self {
            unsafe { Sse(_mm_movelh_ps(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn high_halves(self, b: Self) -> Self {
            unsafe { Sse(_mm_movehl_ps(b.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn blend_odd(self, b: Self) -> Self {
            unsafe { Sse(_mm_blend_ps::<0b1010>(self.0, b.0)) }
        }

        #[inline(always)]
        unsafe fn select_lt(self, b: Self, x: Self, y: Self) -> Self {
            unsafe { Sse(_mm_blendv_ps(y.0, x.0, _mm_cmplt_ps(self.0, b.0))) }
        }
        #[inline(always)]
        unsafe fn ge_mask(self, b: Self) -> SseI {
            unsafe { SseI(_mm_castps_si128(_mm_cmpge_ps(self.0, b.0))) }
        }

        #[inline(always)]
        unsafe fn todb(self) -> Self {
            unsafe {
                let bits = _mm_and_si128(_mm_castps_si128(self.0), _mm_set1_epi32(0x7fff_ffff));
                let f = _mm_cvtepi32_ps(bits);
                Sse(_mm_sub_ps(
                    _mm_mul_ps(f, _mm_set1_ps(7.177_114_4e-7)),
                    _mm_set1_ps(764.616_2),
                ))
            }
        }
        #[inline(always)]
        unsafe fn add_f64(self, c: f64) -> Self {
            unsafe {
                let c = _mm_set1_pd(c);
                let lo = _mm_add_pd(_mm_cvtps_pd(self.0), c);
                let hi = _mm_add_pd(_mm_cvtps_pd(_mm_movehl_ps(self.0, self.0)), c);
                Sse(_mm_movelh_ps(_mm_cvtpd_ps(lo), _mm_cvtpd_ps(hi)))
            }
        }
        #[inline(always)]
        unsafe fn trunc_clamp(self, hi: f32) -> SseI {
            unsafe {
                // `minps` returns its second operand when either is NaN, so a NaN
                // survives the `min` and is then replaced by zero in the `max`.
                let clamped = _mm_max_ps(_mm_min_ps(_mm_set1_ps(hi), self.0), _mm_setzero_ps());
                SseI(_mm_cvttps_epi32(clamped))
            }
        }
        #[inline(always)]
        unsafe fn add_f64_trunc_clamp(self, c: f64, hi: f64) -> SseI {
            unsafe {
                let c = _mm_set1_pd(c);
                let hi = _mm_set1_pd(hi);
                let zero = _mm_setzero_pd();
                let lo = _mm_add_pd(_mm_cvtps_pd(self.0), c);
                let up = _mm_add_pd(_mm_cvtps_pd(_mm_movehl_ps(self.0, self.0)), c);
                let lo = _mm_cvttpd_epi32(_mm_max_pd(_mm_min_pd(hi, lo), zero));
                let up = _mm_cvttpd_epi32(_mm_max_pd(_mm_min_pd(hi, up), zero));
                SseI(_mm_unpacklo_epi64(lo, up))
            }
        }
    }

    impl I32x4 for SseI {
        #[inline(always)]
        unsafe fn splat(v: i32) -> Self {
            unsafe { SseI(_mm_set1_epi32(v)) }
        }
        #[inline(always)]
        unsafe fn from_array(a: [i32; 4]) -> Self {
            unsafe { SseI(_mm_loadu_si128(a.as_ptr().cast())) }
        }
        #[inline(always)]
        unsafe fn to_array(self) -> [i32; 4] {
            unsafe {
                let mut a = [0i32; 4];
                _mm_storeu_si128(a.as_mut_ptr().cast(), self.0);
                a
            }
        }
        #[inline(always)]
        unsafe fn add(self, b: Self) -> Self {
            unsafe { SseI(_mm_add_epi32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn sub(self, b: Self) -> Self {
            unsafe { SseI(_mm_sub_epi32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn mul(self, b: Self) -> Self {
            unsafe { SseI(_mm_mullo_epi32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn and(self, b: Self) -> Self {
            unsafe { SseI(_mm_and_si128(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn and_not(self, b: Self) -> Self {
            unsafe { SseI(_mm_andnot_si128(b.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn eq_zero(self) -> Self {
            unsafe { SseI(_mm_cmpeq_epi32(self.0, _mm_setzero_si128())) }
        }
        #[inline(always)]
        unsafe fn sum(self) -> i32 {
            unsafe {
                let a = self.to_array();
                a[0].wrapping_add(a[1])
                    .wrapping_add(a[2])
                    .wrapping_add(a[3])
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) use arm::Neon;

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::*;

    #[derive(Clone, Copy)]
    pub(in super::super) struct Neon(float32x4_t);

    #[derive(Clone, Copy)]
    pub(in super::super) struct NeonI(int32x4_t);

    /// All ones in lanes 1 and 3.
    #[inline(always)]
    unsafe fn odd_mask() -> uint32x4_t {
        unsafe {
            let lanes = [0u32, u32::MAX, 0, u32::MAX];
            vld1q_u32(lanes.as_ptr())
        }
    }

    impl F32x4 for Neon {
        type I = NeonI;

        #[inline(always)]
        unsafe fn splat(v: f32) -> Self {
            unsafe { Neon(vdupq_n_f32(v)) }
        }
        #[inline(always)]
        unsafe fn from_array(a: [f32; 4]) -> Self {
            unsafe { Neon(vld1q_f32(a.as_ptr())) }
        }
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            unsafe { Neon(vld1q_f32(p)) }
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut f32) {
            unsafe { vst1q_f32(p, self.0) }
        }
        #[inline(always)]
        unsafe fn load_pairs(lo: *const f32, hi: *const f32) -> Self {
            unsafe { Neon(vcombine_f32(vld1_f32(lo), vld1_f32(hi))) }
        }

        #[inline(always)]
        unsafe fn add(self, b: Self) -> Self {
            unsafe { Neon(vaddq_f32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn sub(self, b: Self) -> Self {
            unsafe { Neon(vsubq_f32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn mul(self, b: Self) -> Self {
            unsafe { Neon(vmulq_f32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn div(self, b: Self) -> Self {
            unsafe { Neon(vdivq_f32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn neg(self) -> Self {
            unsafe { Neon(vnegq_f32(self.0)) }
        }
        #[inline(always)]
        unsafe fn neg_odd(self) -> Self {
            unsafe {
                let sign = vandq_u32(odd_mask(), vdupq_n_u32(0x8000_0000));
                Neon(vreinterpretq_f32_u32(veorq_u32(
                    vreinterpretq_u32_f32(self.0),
                    sign,
                )))
            }
        }

        #[inline(always)]
        unsafe fn dup_even(self) -> Self {
            unsafe { Neon(vtrn1q_f32(self.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn dup_odd(self) -> Self {
            unsafe { Neon(vtrn2q_f32(self.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn swap_pairs(self) -> Self {
            unsafe { Neon(vrev64q_f32(self.0)) }
        }
        #[inline(always)]
        unsafe fn swap_halves(self) -> Self {
            unsafe { Neon(vextq_f32::<2>(self.0, self.0)) }
        }
        #[inline(always)]
        unsafe fn reverse(self) -> Self {
            unsafe {
                let pairs = vrev64q_f32(self.0);
                Neon(vextq_f32::<2>(pairs, pairs))
            }
        }
        #[inline(always)]
        unsafe fn uzp(self, b: Self) -> (Self, Self) {
            unsafe { (Neon(vuzp1q_f32(self.0, b.0)), Neon(vuzp2q_f32(self.0, b.0))) }
        }
        #[inline(always)]
        unsafe fn zip(self, b: Self) -> (Self, Self) {
            unsafe { (Neon(vzip1q_f32(self.0, b.0)), Neon(vzip2q_f32(self.0, b.0))) }
        }
        #[inline(always)]
        unsafe fn trn1(self, b: Self) -> Self {
            unsafe { Neon(vtrn1q_f32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn trn2(self, b: Self) -> Self {
            unsafe { Neon(vtrn2q_f32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn low_halves(self, b: Self) -> Self {
            unsafe { Neon(vcombine_f32(vget_low_f32(self.0), vget_low_f32(b.0))) }
        }
        #[inline(always)]
        unsafe fn high_halves(self, b: Self) -> Self {
            unsafe { Neon(vcombine_f32(vget_high_f32(self.0), vget_high_f32(b.0))) }
        }
        #[inline(always)]
        unsafe fn blend_odd(self, b: Self) -> Self {
            unsafe { Neon(vbslq_f32(odd_mask(), b.0, self.0)) }
        }

        #[inline(always)]
        unsafe fn select_lt(self, b: Self, x: Self, y: Self) -> Self {
            unsafe { Neon(vbslq_f32(vcltq_f32(self.0, b.0), x.0, y.0)) }
        }
        #[inline(always)]
        unsafe fn ge_mask(self, b: Self) -> NeonI {
            unsafe { NeonI(vreinterpretq_s32_u32(vcgeq_f32(self.0, b.0))) }
        }

        #[inline(always)]
        unsafe fn todb(self) -> Self {
            unsafe {
                let bits = vandq_u32(vreinterpretq_u32_f32(self.0), vdupq_n_u32(0x7fff_ffff));
                let f = vcvtq_f32_s32(vreinterpretq_s32_u32(bits));
                Neon(vsubq_f32(
                    vmulq_f32(f, vdupq_n_f32(7.177_114_4e-7)),
                    vdupq_n_f32(764.616_2),
                ))
            }
        }
        #[inline(always)]
        unsafe fn add_f64(self, c: f64) -> Self {
            unsafe {
                let c = vdupq_n_f64(c);
                let lo = vaddq_f64(vcvt_f64_f32(vget_low_f32(self.0)), c);
                let hi = vaddq_f64(vcvt_high_f64_f32(self.0), c);
                Neon(vcvt_high_f32_f64(vcvt_f32_f64(lo), hi))
            }
        }
        #[inline(always)]
        unsafe fn trunc_clamp(self, hi: f32) -> NeonI {
            unsafe {
                // `fcvtzs` saturates and maps NaN to zero, exactly as Rust's `as`.
                let t = vcvtq_s32_f32(self.0);
                NeonI(vmaxq_s32(
                    vminq_s32(t, vdupq_n_s32(hi as i32)),
                    vdupq_n_s32(0),
                ))
            }
        }
        #[inline(always)]
        unsafe fn add_f64_trunc_clamp(self, c: f64, hi: f64) -> NeonI {
            unsafe {
                let c = vdupq_n_f64(c);
                let hi = vdupq_n_s64(hi as i64);
                let zero = vdupq_n_s64(0);
                let lo = vcvtq_s64_f64(vaddq_f64(vcvt_f64_f32(vget_low_f32(self.0)), c));
                let up = vcvtq_s64_f64(vaddq_f64(vcvt_high_f64_f32(self.0), c));
                // Saturating to `i64` and clamping there is the same as Rust's
                // saturation to `i32` followed by the clamp, since `hi` is small.
                let lo = vbslq_s64(vcgtq_s64(lo, hi), hi, lo);
                let lo = vbslq_s64(vcltq_s64(lo, zero), zero, lo);
                let up = vbslq_s64(vcgtq_s64(up, hi), hi, up);
                let up = vbslq_s64(vcltq_s64(up, zero), zero, up);
                NeonI(vcombine_s32(vmovn_s64(lo), vmovn_s64(up)))
            }
        }
    }

    impl I32x4 for NeonI {
        #[inline(always)]
        unsafe fn splat(v: i32) -> Self {
            unsafe { NeonI(vdupq_n_s32(v)) }
        }
        #[inline(always)]
        unsafe fn from_array(a: [i32; 4]) -> Self {
            unsafe { NeonI(vld1q_s32(a.as_ptr())) }
        }
        #[inline(always)]
        unsafe fn to_array(self) -> [i32; 4] {
            unsafe {
                let mut a = [0i32; 4];
                vst1q_s32(a.as_mut_ptr(), self.0);
                a
            }
        }
        #[inline(always)]
        unsafe fn add(self, b: Self) -> Self {
            unsafe { NeonI(vaddq_s32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn sub(self, b: Self) -> Self {
            unsafe { NeonI(vsubq_s32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn mul(self, b: Self) -> Self {
            unsafe { NeonI(vmulq_s32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn and(self, b: Self) -> Self {
            unsafe { NeonI(vandq_s32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn and_not(self, b: Self) -> Self {
            unsafe { NeonI(vbicq_s32(self.0, b.0)) }
        }
        #[inline(always)]
        unsafe fn eq_zero(self) -> Self {
            unsafe { NeonI(vreinterpretq_s32_u32(vceqzq_s32(self.0))) }
        }
        #[inline(always)]
        unsafe fn sum(self) -> i32 {
            unsafe { vaddvq_s32(self.0) }
        }
    }
}
