//! Runtime-dispatched SIMD kernel for the encoder-side quantization loop.
//!
//! Quantization is the exact inverse of the decoder's §8.6.3 scaling
//! loop ([`crate::hevc::engine::transform_simd::dequant_block`]): where
//! the decoder multiplies a level by `levelScale[ qP % 6 ]` and shifts
//! left by `qP / 6`, the encoder multiplies a transform coefficient by
//! the reciprocal step `quantScale[ qP % 6 ]` and shifts right by
//! `qbits`. The two run over the same `(nTbS)x(nTbS)` block shape and
//! the same per-position scaling factors, so this kernel mirrors the
//! dequantization one: one flat pass over the block, 64-bit
//! intermediates, and a final clip into the coefficient range.
//!
//! It is a separate module — and a separate [`crate::simd`] dispatch
//! site — rather than another entry point in `transform_simd` because
//! it is encoder-only work: the decoder never runs it, and a profile
//! that pins one instruction set for the encode path should be able to
//! see it move independently of the decode-side kernels.
//!
//! Every backend is bit-exact with [`Backend::Scalar`]: the vector code
//! performs the same products, in the same widths, in the same order,
//! and the rounding offset and clip are applied at the same points.

use core::sync::atomic::{AtomicU8, Ordering};

pub use crate::hevc::engine::transform_simd::Backend;

/// Candidate backends in descending preference order, matching the
/// decode-side kernel's list. SSE4.1 is enough here — the clip runs on
/// narrowed 32-bit lanes, so the 64-bit `pcmpgtq` compare the
/// dequantization clip needs is not required — but SSE4.2 is still
/// preferred so a host that has it reports the same backend for both
/// transform-domain sites.
const PRIORITY: [Backend; 4] = [Backend::Avx2, Backend::Neon, Backend::Sse42, Backend::Sse41];

static DETECTED: AtomicU8 = AtomicU8::new(0);

/// Maps the crate-wide SIMD override, if any, onto this module's
/// [`Backend`], exactly as the decode-side transform kernels do.
fn overridden_backend() -> Option<Backend> {
    use crate::simd::SimdIsa;
    Some(match crate::simd::override_isa()? {
        SimdIsa::Scalar => Backend::Scalar,
        SimdIsa::Sse41 => {
            if Backend::Sse42.supported() {
                Backend::Sse42
            } else {
                Backend::Sse41
            }
        }
        SimdIsa::Avx2 => Backend::Avx2,
        SimdIsa::Neon => Backend::Neon,
    })
}

/// The best quantization backend this host supports.
///
/// The CPU feature probe runs once and is cached. A
/// [`crate::simd::set_override`] override wins over the cache and is
/// re-read on every call, so an instruction set pinned after the first
/// probe still reaches this kernel.
#[must_use]
pub fn detected() -> Backend {
    if let Some(backend) = overridden_backend() {
        return backend;
    }
    let cached = DETECTED.load(Ordering::Relaxed);
    if cached != 0 {
        return PRIORITY
            .get(cached as usize - 1)
            .copied()
            .unwrap_or(Backend::Scalar);
    }
    let mut chosen = Backend::Scalar;
    let mut index = PRIORITY.len();
    for (i, candidate) in PRIORITY.iter().enumerate() {
        if candidate.supported() {
            chosen = *candidate;
            index = i;
            break;
        }
    }
    DETECTED.store(index as u8 + 1, Ordering::Relaxed);
    chosen
}

/// Every backend the host can actually run, scalar first.
#[must_use]
pub fn supported_backends() -> Vec<Backend> {
    let mut backends = vec![Backend::Scalar];
    backends.extend(PRIORITY.iter().copied().filter(|b| b.supported()));
    backends
}

/// Fixed inputs of the quantization equation that are constant across a
/// transform block.
#[derive(Debug, Clone, Copy)]
pub struct QuantParams {
    /// `quantScale[ qP % 6 ]`, the reciprocal of the decoder's
    /// `levelScale[ qP % 6 ]`, used for every position when no scaling
    /// list is in force.
    pub quant_scale: i32,
    /// The right shift that undoes the decoder's `<< ( qP / 6 )` and
    /// `>> bdShift`: `24 + qP / 6 − bdShift`. Always in `1..=62` for the
    /// 8..=16-bit depths the codec accepts.
    pub qbits: u32,
    /// The rounding offset added before the shift, as a fraction of
    /// `1 << qbits`: `171 / 512` for intra blocks and `85 / 512` for
    /// inter ones, the deadzone the reference encoder uses.
    pub round_add: i64,
    /// `coeffMin` from §7.4.5 equations 7-27..7-30.
    pub coeff_min: i32,
    /// `coeffMax` from §7.4.5 equations 7-27..7-30.
    pub coeff_max: i32,
}

/// Quantizes a whole transform block:
/// `level[ i ] = sign( c[ i ] ) * Min( coeffMax, ( |c[ i ]| * f[ i ] +
/// roundAdd ) >> qbits )`.
///
/// `factors` supplies the per-position quantization factor `f[ i ]` —
/// the reciprocal of the decoder's `m[ x ][ y ] * levelScale`, derived
/// once per block from the active scaling list — or is `None` for the
/// flat default, in which case [`QuantParams::quant_scale`] is used at
/// every position. The magnitude product is formed in 64 bits on every
/// backend.
///
/// Every value in `coeffs` must already lie inside
/// `[ coeffMin, coeffMax ]`, which the forward transform guarantees;
/// the vector paths take the 32-bit absolute value, so an `i32::MIN`
/// input would not match the scalar reference.
///
/// # Panics
/// Panics if `out`, `coeffs`, or a supplied `factors` disagree in length.
pub fn quant_block(
    backend: Backend,
    out: &mut [i32],
    coeffs: &[i32],
    factors: Option<&[i32]>,
    params: QuantParams,
) {
    assert_eq!(out.len(), coeffs.len(), "quant operand length mismatch");
    if let Some(f) = factors {
        assert_eq!(f.len(), coeffs.len(), "quant factor length mismatch");
    }
    // The vector paths shift a 64-bit lane, which is only defined for a
    // shift inside the word. The derived `qbits` never leaves 1..=62 for
    // the dimensioned bit depths, but stay exact if it ever does.
    if !(1..=62).contains(&params.qbits) {
        quant_block_scalar(out, coeffs, factors, params);
        return;
    }
    match backend {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => {
            // SAFETY: `Backend::Avx2` is only produced after
            // `is_x86_feature_detected!("avx2")` succeeded on this host.
            unsafe { x86::quant_block_avx2(out, coeffs, factors, params) }
        }
        #[cfg(target_arch = "x86_64")]
        Backend::Sse41 | Backend::Sse42 => {
            // SAFETY: as above, for SSE4.1 (SSE4.2 implies SSE4.1).
            unsafe { x86::quant_block_sse41(out, coeffs, factors, params) }
        }
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => {
            // SAFETY: NEON is architecturally guaranteed on aarch64.
            unsafe { aarch64::quant_block_neon(out, coeffs, factors, params) }
        }
        _ => quant_block_scalar(out, coeffs, factors, params),
    }
}

/// Portable reference for [`quant_block`].
fn quant_block_scalar(
    out: &mut [i32],
    coeffs: &[i32],
    factors: Option<&[i32]>,
    params: QuantParams,
) {
    for (i, (o, &c)) in out.iter_mut().zip(coeffs.iter()).enumerate() {
        let factor = i64::from(factors.map_or(params.quant_scale, |f| f[i]));
        let magnitude = (i64::from(c).abs() * factor + params.round_add) >> params.qbits;
        let level = magnitude.min(i64::from(params.coeff_max)) as i32;
        *o = if c < 0 { -level } else { level };
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::QuantParams;
    use core::arch::x86_64::*;

    /// AVX2 [`super::quant_block`]: four coefficients per iteration,
    /// widened to 64-bit lanes for the magnitude product, narrowed back
    /// for the clip and the sign restore.
    ///
    /// # Safety
    /// The host must support AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn quant_block_avx2(
        out: &mut [i32],
        coeffs: &[i32],
        factors: Option<&[i32]>,
        params: QuantParams,
    ) {
        unsafe {
            let shr = _mm_cvtsi32_si128(params.qbits as i32);
            let add = _mm256_set1_epi64x(params.round_add);
            let scale = _mm_set1_epi32(params.quant_scale);
            let max = _mm_set1_epi32(params.coeff_max);
            // Collects the low half of each 64-bit lane back into four
            // contiguous i32 results.
            let gather = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);

            let mut i = 0;
            while i + 4 <= out.len() {
                let coeff = _mm_loadu_si128(coeffs.as_ptr().add(i).cast());
                let magnitude = _mm256_cvtepi32_epi64(_mm_abs_epi32(coeff));
                let factor32 = match factors {
                    Some(f) => _mm_loadu_si128(f.as_ptr().add(i).cast()),
                    None => scale,
                };
                let factor = _mm256_cvtepi32_epi64(factor32);
                // Both operands are non-negative, so the shift is the
                // same whether it is arithmetic or logical.
                let scaled = _mm256_srl_epi64(
                    _mm256_add_epi64(_mm256_mul_epi32(magnitude, factor), add),
                    shr,
                );
                let packed = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(scaled, gather));
                let clipped = _mm_min_epi32(packed, max);
                // sign(c) applied as (v ^ mask) − mask with mask = c >> 31.
                let mask = _mm_srai_epi32(coeff, 31);
                let signed = _mm_sub_epi32(_mm_xor_si128(clipped, mask), mask);
                _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), signed);
                i += 4;
            }
            if i < out.len() {
                let tail = factors.map(|f| &f[i..]);
                super::quant_block_scalar(&mut out[i..], &coeffs[i..], tail, params);
            }
        }
    }

    /// SSE4.1 [`super::quant_block`]: the same four-coefficient step as
    /// the AVX2 kernel, with the 64-bit half of the work split across
    /// two 128-bit registers.
    ///
    /// # Safety
    /// The host must support SSE4.1.
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn quant_block_sse41(
        out: &mut [i32],
        coeffs: &[i32],
        factors: Option<&[i32]>,
        params: QuantParams,
    ) {
        unsafe {
            let shr = _mm_cvtsi32_si128(params.qbits as i32);
            let add = _mm_set1_epi64x(params.round_add);
            let scale = _mm_set1_epi32(params.quant_scale);
            let max = _mm_set1_epi32(params.coeff_max);

            let mut i = 0;
            while i + 4 <= out.len() {
                let coeff = _mm_loadu_si128(coeffs.as_ptr().add(i).cast());
                let magnitude = _mm_abs_epi32(coeff);
                let factor32 = match factors {
                    Some(f) => _mm_loadu_si128(f.as_ptr().add(i).cast()),
                    None => scale,
                };
                let step = |m: __m128i, f: __m128i| {
                    _mm_srl_epi64(
                        _mm_add_epi64(
                            _mm_mul_epi32(_mm_cvtepi32_epi64(m), _mm_cvtepi32_epi64(f)),
                            add,
                        ),
                        shr,
                    )
                };
                let low = step(magnitude, factor32);
                let high = step(_mm_srli_si128(magnitude, 8), _mm_srli_si128(factor32, 8));
                // Take the low i32 of each 64-bit lane: [low0, low1, high0, high1].
                let packed = _mm_castps_si128(_mm_shuffle_ps(
                    _mm_castsi128_ps(low),
                    _mm_castsi128_ps(high),
                    0b10_00_10_00,
                ));
                let clipped = _mm_min_epi32(packed, max);
                let mask = _mm_srai_epi32(coeff, 31);
                let signed = _mm_sub_epi32(_mm_xor_si128(clipped, mask), mask);
                _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), signed);
                i += 4;
            }
            if i < out.len() {
                let tail = factors.map(|f| &f[i..]);
                super::quant_block_scalar(&mut out[i..], &coeffs[i..], tail, params);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::QuantParams;
    use core::arch::aarch64::*;

    /// NEON [`super::quant_block`]: four coefficients per iteration,
    /// `vmull_s32` widening the magnitude product to 64 bits.
    ///
    /// # Safety
    /// NEON is architecturally guaranteed on aarch64.
    #[target_feature(enable = "neon")]
    pub unsafe fn quant_block_neon(
        out: &mut [i32],
        coeffs: &[i32],
        factors: Option<&[i32]>,
        params: QuantParams,
    ) {
        unsafe {
            let shr = vdupq_n_s64(-i64::from(params.qbits));
            let add = vdupq_n_s64(params.round_add);
            let scale = vdupq_n_s32(params.quant_scale);
            let max = vdupq_n_s32(params.coeff_max);

            let step = |m: int32x2_t, f: int32x2_t| {
                // Both operands are non-negative, so the arithmetic
                // shift matches the scalar reference's logical one.
                vmovn_s64(vshlq_s64(vaddq_s64(vmull_s32(m, f), add), shr))
            };

            let mut i = 0;
            while i + 4 <= out.len() {
                let coeff = vld1q_s32(coeffs.as_ptr().add(i));
                let magnitude = vabsq_s32(coeff);
                let factor = match factors {
                    Some(f) => vld1q_s32(f.as_ptr().add(i)),
                    None => scale,
                };
                let packed = vcombine_s32(
                    step(vget_low_s32(magnitude), vget_low_s32(factor)),
                    step(vget_high_s32(magnitude), vget_high_s32(factor)),
                );
                let clipped = vminq_s32(packed, max);
                let mask = vshrq_n_s32(coeff, 31);
                let signed = vsubq_s32(veorq_s32(clipped, mask), mask);
                vst1q_s32(out.as_mut_ptr().add(i), signed);
                i += 4;
            }
            if i < out.len() {
                let tail = factors.map(|f| &f[i..]);
                super::quant_block_scalar(&mut out[i..], &coeffs[i..], tail, params);
            }
        }
    }
}
