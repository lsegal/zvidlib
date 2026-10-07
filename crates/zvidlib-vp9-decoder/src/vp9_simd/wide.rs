//! [`W`], a vector of `i32` lanes with ordinary arithmetic operators, so the
//! vector inverse transforms in [`super::idct1d`] can read exactly like the
//! scalar libvpx transliteration they reproduce.
//!
//! A `W` can only be made from an [`I32x`] value, and every way of making
//! one of those is `unsafe` and only reached from a `#[target_feature]`
//! wrapper that verified the instruction set. Holding a `W` therefore proves
//! the instructions its operators execute are available, which is what makes
//! the operators themselves safe to call.

use core::ops::{Add, Mul, Neg, Sub};

use zvidlib_core::simd::vector::I32x;

/// One `i32` per lane, with wrapping arithmetic.
#[derive(Clone, Copy)]
pub(crate) struct W<V>(pub(crate) V);

impl<V: I32x> W<V> {
    #[inline(always)]
    pub(crate) unsafe fn zero() -> Self {
        unsafe { Self(V::zero()) }
    }

    /// Each lane truncated to 16 bits and sign-extended back, as the
    /// scalar code's `as i16` does.
    #[inline(always)]
    pub(crate) fn t16(self) -> Self {
        // SAFETY: see the module documentation.
        unsafe { Self(self.0.sll::<16>().sra::<16>()) }
    }
}

impl<V: I32x> Add for W<V> {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: see the module documentation.
        unsafe { Self(self.0.add(other.0)) }
    }
}

impl<V: I32x> Sub for W<V> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: see the module documentation.
        unsafe { Self(self.0.sub(other.0)) }
    }
}

impl<V: I32x> Neg for W<V> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        // SAFETY: see the module documentation.
        unsafe { Self(V::zero().sub(self.0)) }
    }
}

/// Multiplication by one of the transform constants, which all fit in 16
/// bits. They are `i64` only because the scalar transforms they are shared
/// with compute in `i64`.
impl<V: I32x> Mul<i64> for W<V> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, constant: i64) -> Self {
        // SAFETY: see the module documentation.
        unsafe { Self(self.0.mul(V::splat(constant as i32))) }
    }
}

impl<V: I32x> Mul<W<V>> for i64 {
    type Output = W<V>;
    #[inline(always)]
    fn mul(self, vector: W<V>) -> W<V> {
        vector * self
    }
}

/// The scalar code's `wraplow`, a wrap to 32 bits. Every lane already is 32
/// bits, and the input limits keep the value the scalar code wraps inside
/// that range anyway.
#[inline(always)]
pub(crate) fn wraplow<V: I32x>(x: W<V>) -> W<V> {
    x
}

/// The scalar code's `wrap32`; see [`wraplow`].
#[inline(always)]
pub(crate) fn wrap32<V: I32x>(x: W<V>) -> W<V> {
    x
}

/// `ROUND_POWER_OF_TWO(x, 14)`, the rounding of every transform constant
/// product.
#[inline(always)]
pub(crate) fn round_shift<V: I32x>(x: W<V>) -> W<V> {
    // SAFETY: see the module documentation.
    unsafe { W(x.0.add(V::splat(1 << 13)).sra::<14>()) }
}
