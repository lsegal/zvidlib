//! Runtime-dispatched kernels for the decoder's 8-bit 4:2:0 YUV-to-RGBA output
//! conversion.
//!
//! This is the pass every decoded picture takes on its way out of
//! [`super::picture_to_rgba`]: the fixed BT.601/709 limited-range integer
//! matrix, applied per sample, with the chroma planes read at half resolution
//! in both directions. It is not decoding, but it is on the path of both
//! whole-frame benchmark groups, and the issue #189 stage attribution recorded
//! in `benches/README.md` measured it at a third of everything those groups
//! time — the single largest item in a whole-frame decode, and until now the
//! largest one with no vector kernel at all.
//!
//! # The kernel
//!
//! One output pixel is
//!
//! ```text
//! yScaled = 8·Y - 128      uScaled = 8·Cb - 1024      vScaled = 8·Cr - 1024
//! yTerm   = (yScaled · 9539) >> 16
//! R = Clip3(0, 255, yTerm + (vScaled ·  13075) >> 16)
//! G = Clip3(0, 255, yTerm + (uScaled ·  -3209) >> 16 + (vScaled · -6660) >> 16)
//! B = Clip3(0, 255, yTerm + (uScaled ·  16525) >> 16)
//! A = 255
//! ```
//!
//! which is embarrassingly parallel per sample: a constant integer matrix, an
//! arithmetic right shift, and a clip. The vector backends compute four
//! (SSE4.1, NEON) or eight (AVX2) pixels at a time in `i32` lanes and write the
//! four components of each pixel as one little-endian `u32`
//! `R | G<<8 | B<<16 | 0xFF00_0000`, which turns the RGBA interleave into two
//! shifts and two ORs instead of a byte-level pack.
//!
//! # Bit-exactness
//!
//! Every backend is **bit-exact** with [`convert_row_scalar`], which is the
//! reference the shipped scalar path uses and which
//! `canonical_conversion_uses_decoder_integer_rounding` in [`super`] pins. The
//! arithmetic is plain `i32` throughout, in the same order, and the `>> 16` is
//! an arithmetic shift in every backend, so the negative `Cb`/`Cr` products
//! round toward negative infinity exactly as the scalar reference does.
//!
//! The scalar reference reaches for `saturating_mul` / `saturating_add` where
//! the vector kernels wrap. That is not a divergence: a decoded 8-bit picture's
//! samples are clipped to `0..=255` by reconstruction, so `yScaled` stays in
//! `-128..=1912` and the chroma terms in `-1024..=1016`, and the widest product
//! the matrix can form is `1912 · 16525 ≈ 3.2e7` — three orders of magnitude
//! inside `i32`. No saturation is reachable, so saturating and wrapping
//! arithmetic agree on every input this kernel is given.
//!
//! # Dispatch
//!
//! [`detected_isa`] resolves the backend once per process, and consults
//! [`crate::simd::override_isa`] ahead of that cache on every call, so
//! `simd::set_override` reaches this kernel the way it reaches the engine's.
//! The site is reported as `hevc_color_convert` by
//! [`crate::simd::active_by_site`].

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use std::sync::OnceLock;

/// The instruction-set backend the conversion runs on.
///
/// Only the variants the target architecture can execute are compiled in;
/// [`Isa::Scalar`] is always available and is the bit-exactness reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Isa {
    /// Portable scalar fallback — the bit-exactness reference.
    Scalar,
    /// x86_64 SSE4.1 (`pmulld` / `pminsd` / `pmaxsd`), 4 pixels per step.
    #[cfg(target_arch = "x86_64")]
    Sse41,
    /// x86_64 AVX2, 8 pixels per step.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// AArch64 NEON, 4 pixels per step.
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// Detects the widest backend the running CPU supports, once per process.
///
/// A [`crate::simd::set_override`] override is consulted ahead of the cache on
/// every call, so pinning an instruction set still reaches this kernel after
/// detection has resolved.
#[must_use]
pub fn detected_isa() -> Isa {
    if let Some(isa) = overridden_isa() {
        return isa;
    }
    static ISA: OnceLock<Isa> = OnceLock::new();
    *ISA.get_or_init(detect)
}

/// Maps the crate-wide SIMD override, if any, onto this module's [`Isa`].
#[inline]
fn overridden_isa() -> Option<Isa> {
    use crate::simd::SimdIsa;
    Some(match crate::simd::override_isa()? {
        SimdIsa::Scalar => Isa::Scalar,
        #[cfg(target_arch = "x86_64")]
        SimdIsa::Sse41 => Isa::Sse41,
        #[cfg(target_arch = "x86_64")]
        SimdIsa::Avx2 => Isa::Avx2,
        #[cfg(target_arch = "aarch64")]
        SimdIsa::Neon => Isa::Neon,
        #[allow(unreachable_patterns)]
        _ => Isa::Scalar,
    })
}

fn detect() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return Isa::Avx2;
        }
        if is_x86_feature_detected!("sse4.1") {
            return Isa::Sse41;
        }
        Isa::Scalar
    }
    #[cfg(target_arch = "aarch64")]
    {
        Isa::Neon
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Isa::Scalar
    }
}

/// Luma-to-RGB matrix coefficients, in the decoder's Q16 fixed point.
const Y_COEFF: i32 = 9_539;
const V_TO_R: i32 = 13_075;
const U_TO_G: i32 = -3_209;
const V_TO_G: i32 = -6_660;
const U_TO_B: i32 = 16_525;

/// The `A = 255` byte, pre-placed in the high lane of the packed pixel.
const OPAQUE: i32 = 0xFF00_0000_u32 as i32;

/// Converts a whole 8-bit 4:2:0 picture to RGBA on the detected backend.
///
/// `luma`, `cb` and `cr` are the picture's row-major planes with the given
/// strides in samples; `rgba` is the destination with `rgba_stride` bytes per
/// row. Every plane must be large enough for `width` x `height` luma samples
/// and the corresponding half-resolution chroma.
///
/// # Panics
/// Panics if any plane or the destination is too small for the dimensions.
#[allow(clippy::too_many_arguments)]
pub fn convert_yuv420_to_rgba(
    luma: &[i32],
    luma_stride: usize,
    cb: &[i32],
    cr: &[i32],
    chroma_stride: usize,
    width: usize,
    height: usize,
    rgba: &mut [u8],
    rgba_stride: usize,
) {
    convert_yuv420_to_rgba_with(
        detected_isa(),
        luma,
        luma_stride,
        cb,
        cr,
        chroma_stride,
        width,
        height,
        rgba,
        rgba_stride,
    );
}

/// [`convert_yuv420_to_rgba`] on an explicitly chosen backend.
///
/// An [`Isa`] the running CPU cannot execute is not reachable — the variants
/// are compiled per architecture and [`crate::simd::set_override`] refuses to
/// pin an unavailable one — so every arm here is safe to call.
///
/// # Panics
/// Panics if any plane or the destination is too small for the dimensions.
#[allow(clippy::too_many_arguments)]
pub fn convert_yuv420_to_rgba_with(
    isa: Isa,
    luma: &[i32],
    luma_stride: usize,
    cb: &[i32],
    cr: &[i32],
    chroma_stride: usize,
    width: usize,
    height: usize,
    rgba: &mut [u8],
    rgba_stride: usize,
) {
    let chroma_width = width.div_ceil(2);
    for y in 0..height {
        let luma_row = &luma[y * luma_stride..y * luma_stride + width];
        let chroma_base = (y / 2) * chroma_stride;
        let cb_row = &cb[chroma_base..chroma_base + chroma_width];
        let cr_row = &cr[chroma_base..chroma_base + chroma_width];
        let out_row = &mut rgba[y * rgba_stride..y * rgba_stride + width * 4];
        convert_row_with(isa, luma_row, cb_row, cr_row, out_row);
    }
}

/// One output row, on `isa`.
fn convert_row_with(isa: Isa, luma: &[i32], cb: &[i32], cr: &[i32], out: &mut [u8]) {
    match isa {
        Isa::Scalar => convert_row_scalar(luma, cb, cr, out, 0),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `Isa::Sse41` is only reachable on a CPU `detect` or
        // `crate::simd::available` confirmed supports SSE4.1.
        Isa::Sse41 => unsafe { convert_row_sse41(luma, cb, cr, out) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above, for AVX2.
        Isa::Avx2 => unsafe { convert_row_avx2(luma, cb, cr, out) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is architecturally mandatory on aarch64.
        Isa::Neon => unsafe { convert_row_neon(luma, cb, cr, out) },
    }
}

/// The scalar reference, converting the row from `start` onward.
///
/// The vector backends call it with the first index their whole-vector loop did
/// not cover, so the tail of an odd-width row runs the same arithmetic that
/// defines the result.
fn convert_row_scalar(luma: &[i32], cb: &[i32], cr: &[i32], out: &mut [u8], start: usize) {
    for x in start..luma.len() {
        let y_scaled = luma[x] * 8 - 128;
        let u_scaled = cb[x / 2] * 8 - 1024;
        let v_scaled = cr[x / 2] * 8 - 1024;
        let y_term = multiply_high(y_scaled, Y_COEFF);
        let at = x * 4;
        out[at] = clip_u8(y_term.saturating_add(multiply_high(v_scaled, V_TO_R)));
        out[at + 1] = clip_u8(
            y_term
                .saturating_add(multiply_high(u_scaled, U_TO_G))
                .saturating_add(multiply_high(v_scaled, V_TO_G)),
        );
        out[at + 2] = clip_u8(y_term.saturating_add(multiply_high(u_scaled, U_TO_B)));
        out[at + 3] = 255;
    }
}

/// The decoder's Q16 fixed-point multiply: a full-width product, then an
/// arithmetic shift back down.
pub fn multiply_high(left: i32, right: i32) -> i32 {
    left.saturating_mul(right) >> 16
}

/// The §C.3 style clip to the 8-bit output range.
pub fn clip_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

/// Four pixels per step in `i32` lanes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_row_neon(luma: &[i32], cb: &[i32], cr: &[i32], out: &mut [u8]) {
    /// `[a, b] -> [a, a, b, b]`: the 4:2:0 horizontal chroma upsample.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn duplicate(pair: int32x2_t) -> int32x4_t {
        let both = vcombine_s32(pair, pair);
        vzip1q_s32(both, both)
    }
    /// `(value · coefficient) >> 16`, lane-wise.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn multiply_high_lanes(value: int32x4_t, coefficient: i32) -> int32x4_t {
        vshrq_n_s32(vmulq_s32(value, vdupq_n_s32(coefficient)), 16)
    }

    let width = luma.len();
    let zero = vdupq_n_s32(0);
    let max = vdupq_n_s32(255);
    let mut x = 0;
    while x + 4 <= width {
        unsafe {
            let y_scaled = vsubq_s32(
                vshlq_n_s32(vld1q_s32(luma.as_ptr().add(x)), 3),
                vdupq_n_s32(128),
            );
            let u_scaled = vsubq_s32(
                vshlq_n_s32(duplicate(vld1_s32(cb.as_ptr().add(x / 2))), 3),
                vdupq_n_s32(1024),
            );
            let v_scaled = vsubq_s32(
                vshlq_n_s32(duplicate(vld1_s32(cr.as_ptr().add(x / 2))), 3),
                vdupq_n_s32(1024),
            );
            let y_term = multiply_high_lanes(y_scaled, Y_COEFF);
            let clip = |value| vminq_s32(vmaxq_s32(value, zero), max);
            let red = clip(vaddq_s32(y_term, multiply_high_lanes(v_scaled, V_TO_R)));
            let green = clip(vaddq_s32(
                vaddq_s32(y_term, multiply_high_lanes(u_scaled, U_TO_G)),
                multiply_high_lanes(v_scaled, V_TO_G),
            ));
            let blue = clip(vaddq_s32(y_term, multiply_high_lanes(u_scaled, U_TO_B)));
            let packed = vorrq_s32(
                vorrq_s32(red, vshlq_n_s32(green, 8)),
                vorrq_s32(vshlq_n_s32(blue, 16), vdupq_n_s32(OPAQUE)),
            );
            vst1q_u8(out.as_mut_ptr().add(x * 4), vreinterpretq_u8_s32(packed));
        }
        x += 4;
    }
    convert_row_scalar(luma, cb, cr, out, x);
}

/// Four pixels per step in `i32` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn convert_row_sse41(luma: &[i32], cb: &[i32], cr: &[i32], out: &mut [u8]) {
    /// `[a, b] -> [a, a, b, b]`: the 4:2:0 horizontal chroma upsample.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn duplicate(pair: *const i32) -> __m128i {
        unsafe { _mm_shuffle_epi32(_mm_loadl_epi64(pair.cast()), 0b01_01_00_00) }
    }
    /// `(value · coefficient) >> 16`, lane-wise.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn multiply_high_lanes(value: __m128i, coefficient: i32) -> __m128i {
        _mm_srai_epi32(_mm_mullo_epi32(value, _mm_set1_epi32(coefficient)), 16)
    }

    let width = luma.len();
    let zero = _mm_setzero_si128();
    let max = _mm_set1_epi32(255);
    let mut x = 0;
    while x + 4 <= width {
        unsafe {
            let y_scaled = _mm_sub_epi32(
                _mm_slli_epi32(_mm_loadu_si128(luma.as_ptr().add(x).cast()), 3),
                _mm_set1_epi32(128),
            );
            let u_scaled = _mm_sub_epi32(
                _mm_slli_epi32(duplicate(cb.as_ptr().add(x / 2)), 3),
                _mm_set1_epi32(1024),
            );
            let v_scaled = _mm_sub_epi32(
                _mm_slli_epi32(duplicate(cr.as_ptr().add(x / 2)), 3),
                _mm_set1_epi32(1024),
            );
            let y_term = multiply_high_lanes(y_scaled, Y_COEFF);
            let clip = |value| _mm_min_epi32(_mm_max_epi32(value, zero), max);
            let red = clip(_mm_add_epi32(y_term, multiply_high_lanes(v_scaled, V_TO_R)));
            let green = clip(_mm_add_epi32(
                _mm_add_epi32(y_term, multiply_high_lanes(u_scaled, U_TO_G)),
                multiply_high_lanes(v_scaled, V_TO_G),
            ));
            let blue = clip(_mm_add_epi32(y_term, multiply_high_lanes(u_scaled, U_TO_B)));
            let packed = _mm_or_si128(
                _mm_or_si128(red, _mm_slli_epi32(green, 8)),
                _mm_or_si128(_mm_slli_epi32(blue, 16), _mm_set1_epi32(OPAQUE)),
            );
            _mm_storeu_si128(out.as_mut_ptr().add(x * 4).cast(), packed);
        }
        x += 4;
    }
    convert_row_scalar(luma, cb, cr, out, x);
}

/// Eight pixels per step in `i32` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convert_row_avx2(luma: &[i32], cb: &[i32], cr: &[i32], out: &mut [u8]) {
    /// `[a, b, c, d] -> [a, a, b, b, c, c, d, d]`: the horizontal upsample.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn duplicate(quad: *const i32, pattern: __m256i) -> __m256i {
        unsafe {
            _mm256_permutevar8x32_epi32(
                _mm256_broadcastsi128_si256(_mm_loadu_si128(quad.cast())),
                pattern,
            )
        }
    }
    /// `(value · coefficient) >> 16`, lane-wise.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn multiply_high_lanes(value: __m256i, coefficient: i32) -> __m256i {
        _mm256_srai_epi32(
            _mm256_mullo_epi32(value, _mm256_set1_epi32(coefficient)),
            16,
        )
    }

    let width = luma.len();
    let pattern = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    let zero = _mm256_setzero_si256();
    let max = _mm256_set1_epi32(255);
    let mut x = 0;
    while x + 8 <= width {
        unsafe {
            let y_scaled = _mm256_sub_epi32(
                _mm256_slli_epi32(_mm256_loadu_si256(luma.as_ptr().add(x).cast()), 3),
                _mm256_set1_epi32(128),
            );
            let u_scaled = _mm256_sub_epi32(
                _mm256_slli_epi32(duplicate(cb.as_ptr().add(x / 2), pattern), 3),
                _mm256_set1_epi32(1024),
            );
            let v_scaled = _mm256_sub_epi32(
                _mm256_slli_epi32(duplicate(cr.as_ptr().add(x / 2), pattern), 3),
                _mm256_set1_epi32(1024),
            );
            let y_term = multiply_high_lanes(y_scaled, Y_COEFF);
            let clip = |value| _mm256_min_epi32(_mm256_max_epi32(value, zero), max);
            let red = clip(_mm256_add_epi32(
                y_term,
                multiply_high_lanes(v_scaled, V_TO_R),
            ));
            let green = clip(_mm256_add_epi32(
                _mm256_add_epi32(y_term, multiply_high_lanes(u_scaled, U_TO_G)),
                multiply_high_lanes(v_scaled, V_TO_G),
            ));
            let blue = clip(_mm256_add_epi32(
                y_term,
                multiply_high_lanes(u_scaled, U_TO_B),
            ));
            let packed = _mm256_or_si256(
                _mm256_or_si256(red, _mm256_slli_epi32(green, 8)),
                _mm256_or_si256(_mm256_slli_epi32(blue, 16), _mm256_set1_epi32(OPAQUE)),
            );
            _mm256_storeu_si256(out.as_mut_ptr().add(x * 4).cast(), packed);
        }
        x += 8;
    }
    convert_row_scalar(luma, cb, cr, out, x);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{self, SimdIsa};

    /// Every available backend against the scalar reference, over content that
    /// exercises the clip on both ends and both chroma parities.
    fn planes(width: usize, height: usize) -> (Vec<i32>, Vec<i32>, Vec<i32>, usize) {
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let mut luma = vec![0; width * height];
        for (index, sample) in luma.iter_mut().enumerate() {
            // A ramp that reaches both 0 and 255, so the clip is live.
            *sample = ((index * 7) % 256) as i32;
        }
        let mut cb = vec![0; chroma_width * chroma_height];
        let mut cr = vec![0; chroma_width * chroma_height];
        for (index, sample) in cb.iter_mut().enumerate() {
            *sample = ((index * 13) % 256) as i32;
        }
        for (index, sample) in cr.iter_mut().enumerate() {
            *sample = (255 - (index * 11) % 256) as i32;
        }
        (luma, cb, cr, chroma_width)
    }

    fn convert(isa: Isa, width: usize, height: usize) -> Vec<u8> {
        let (luma, cb, cr, chroma_width) = planes(width, height);
        let mut rgba = vec![0_u8; width * height * 4];
        convert_yuv420_to_rgba_with(
            isa,
            &luma,
            width,
            &cb,
            &cr,
            chroma_width,
            width,
            height,
            &mut rgba,
            width * 4,
        );
        rgba
    }

    #[test]
    fn every_backend_is_bit_exact_with_the_scalar_reference() {
        // Widths that leave a tail for the 4-lane and the 8-lane kernels, and
        // odd widths where the last pixel's chroma column is a half sample.
        for width in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 64] {
            for height in [1, 2, 3, 8] {
                let reference = convert(Isa::Scalar, width, height);
                for isa in backends() {
                    assert_eq!(
                        convert(isa, width, height),
                        reference,
                        "{isa:?} diverged at {width}x{height}"
                    );
                }
            }
        }
    }

    fn backends() -> Vec<Isa> {
        let mut isas = vec![Isa::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.1") {
                isas.push(Isa::Sse41);
            }
            if is_x86_feature_detected!("avx2") {
                isas.push(Isa::Avx2);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            isas.push(Isa::Neon);
        }
        isas
    }

    #[test]
    fn the_alpha_channel_is_opaque_and_the_clip_saturates() {
        // Y = 255 with neutral chroma is the top of the range; Y = 0 is the
        // bottom, and both have to land on the clip rather than wrap.
        for (luma_sample, expected) in [(0_i32, 0_u8), (255, 255)] {
            let mut rgba = [0_u8; 4 * 4];
            convert_yuv420_to_rgba_with(
                detected_isa(),
                &[luma_sample; 4],
                4,
                &[128; 2],
                &[128; 2],
                2,
                4,
                1,
                &mut rgba,
                16,
            );
            for pixel in rgba.chunks_exact(4) {
                assert_eq!(pixel[0], expected);
                assert_eq!(pixel[3], 255, "alpha is always opaque");
            }
        }
    }

    #[test]
    fn the_override_reaches_this_kernel() {
        let _guard = simd::test_lock();
        for isa in simd::available() {
            simd::set_override(Some(isa));
            let expected = match isa {
                SimdIsa::Scalar => Isa::Scalar,
                #[cfg(target_arch = "x86_64")]
                SimdIsa::Sse41 => Isa::Sse41,
                #[cfg(target_arch = "x86_64")]
                SimdIsa::Avx2 => Isa::Avx2,
                #[cfg(target_arch = "aarch64")]
                SimdIsa::Neon => Isa::Neon,
                #[allow(unreachable_patterns)]
                _ => Isa::Scalar,
            };
            assert_eq!(detected_isa(), expected, "{}", isa.name());
        }
        simd::set_override(None);
    }
}
