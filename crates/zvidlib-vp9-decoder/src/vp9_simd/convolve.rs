//! The 8-tap sub-pixel convolution of inter prediction
//! (`vp9_dec::recon::convolve`, libvpx's `vpx_convolve8_c` family), eight
//! output pixels at a time.
//!
//! The regular, smooth, sharp and bilinear filters are all tables of the
//! same 8-tap shape, so one kernel covers them, and compound prediction's
//! rounded average is folded into its store. Every pixel of an output row
//! shares one vertical phase, so the vertical pass is vectorized for scaled
//! references too; a scaled horizontal step gives each pixel of a row its
//! own phase, and that pass alone stays scalar.
//!
//! The arithmetic is the scalar code's: each tap product and their sum in
//! 32 bits, `ROUND_POWER_OF_TWO(sum, 7)`, then a saturating narrow to bytes,
//! which is the scalar clamp to `0..=255` because the rounded sum always fits
//! 16 bits. The control flow is written once, over [`Convolver`], whose few
//! primitives each instruction set implements with its own intrinsics, the
//! way `av1_mc` writes its passes: one 8-byte load feeds eight outputs, which
//! the 32-bit lane abstraction the other VP9 kernels use cannot express.

// Loops over vectors index rather than iterate: an iterator adapter or an
// `array::from_fn` closure over vector values is a separate function the
// inliner can leave outside the `#[target_feature]` wrapper, compiled at the
// baseline instruction set (#341).
#![allow(clippy::needless_range_loop)]

pub(crate) use crate::vp9_dec::recon::Kernel;

/// The per-instruction-set primitives of [`convolve`]. Every method reads or
/// writes through raw pointers, so the caller must have checked that the
/// whole window it filters lies inside its buffers.
pub(super) trait Convolver {
    /// The eight taps of one phase, prepared for [`Convolver::filter8`].
    type Taps: Copy;
    /// Up to eight output pixels, as bytes.
    type Pixels: Copy;

    unsafe fn taps(taps: &[i16; 8]) -> Self::Taps;
    /// `clamp(ROUND_POWER_OF_TWO(sum_k src[k * pitch + i] * taps[k], 7))`
    /// for the eight pixels `i` in `0..8`.
    unsafe fn filter8(src: *const u8, pitch: usize, taps: &Self::Taps) -> Self::Pixels;
    /// [`Convolver::filter8`] for the four pixels `i` in `0..4`, reading no
    /// further than they need.
    unsafe fn filter4(src: *const u8, pitch: usize, taps: &Self::Taps) -> Self::Pixels;
    unsafe fn load8(src: *const u8) -> Self::Pixels;
    unsafe fn load4(src: *const u8) -> Self::Pixels;
    /// Stores eight pixels, or with `average` their rounded average with
    /// the eight already at `dst`.
    unsafe fn store8(dst: *mut u8, pixels: Self::Pixels, average: bool);
    /// [`Convolver::store8`] for four pixels.
    unsafe fn store4(dst: *mut u8, pixels: Self::Pixels, average: bool);
}

/// [`Convolver::filter4`] or [`Convolver::filter8`], by block width.
#[inline(always)]
unsafe fn filter<C: Convolver>(
    narrow: bool,
    src: *const u8,
    pitch: usize,
    taps: &C::Taps,
) -> C::Pixels {
    unsafe {
        if narrow {
            C::filter4(src, pitch, taps)
        } else {
            C::filter8(src, pitch, taps)
        }
    }
}

/// [`Convolver::store4`] or [`Convolver::store8`], by block width.
#[inline(always)]
unsafe fn store<C: Convolver>(narrow: bool, dst: *mut u8, pixels: C::Pixels, average: bool) {
    unsafe {
        if narrow {
            C::store4(dst, pixels, average);
        } else {
            C::store8(dst, pixels, average);
        }
    }
}

/// `lo..=hi` lies inside a buffer of `len` bytes.
fn inside(lo: isize, hi: isize, len: usize) -> bool {
    lo >= 0 && hi >= lo && (hi as usize) < len
}

/// The vector form of `vp9_dec::recon::convolve`. Returns `false`, having
/// written nothing, when `w` is not a multiple of four, or when the window
/// the scalar code would index reaches outside `src`, `dest` or `temp` (the
/// scalar code then reports it).
#[inline(always)]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) unsafe fn convolve<C: Convolver>(
    src: &[u8],
    origin: usize,
    src_stride: usize,
    dest: &mut [u8],
    stride: usize,
    w: usize,
    h: usize,
    kernel: &Kernel,
    x_frac: i32,
    x_step: i32,
    y_frac: i32,
    y_step: i32,
    average: bool,
    temp: &mut [u8; 64 * 135],
) -> bool {
    unsafe {
        if w == 0 || h == 0 || w % 4 != 0 || w > 64 || x_frac < 0 || y_frac < 0 {
            return false;
        }
        if (h - 1) * stride + w > dest.len() {
            return false;
        }
        let narrow = w == 4;
        let chunk = if narrow { 4 } else { 8 };
        let (origin_i, ss, wi, hi) = (origin as isize, src_stride as isize, w as isize, h as isize);
        let whole_x = x_frac == 0 && x_step == 16;
        let whole_y = y_frac == 0 && y_step == 16;
        let source = src.as_ptr();
        let out = dest.as_mut_ptr();

        if whole_x && whole_y {
            let last = origin_i + (hi - 1) * ss + wi - 1;
            if !inside(origin_i, last, src.len()) {
                return false;
            }
            for y in 0..h {
                let line = source.add(origin + y * src_stride);
                let row = out.add(y * stride);
                if average {
                    for x in (0..w).step_by(chunk) {
                        let pixels = if narrow {
                            C::load4(line.add(x))
                        } else {
                            C::load8(line.add(x))
                        };
                        store::<C>(narrow, row.add(x), pixels, true);
                    }
                } else {
                    core::ptr::copy_nonoverlapping(line, row, w);
                }
            }
            return true;
        }
        if whole_y {
            // A scaled step gives every pixel its own phase.
            if x_step != 16 {
                return false;
            }
            let offset = (x_frac >> 4) as isize;
            let first = origin_i - 3 + offset;
            if !inside(first, first + (hi - 1) * ss + wi - 1 + 7, src.len()) {
                return false;
            }
            let taps = C::taps(&kernel[(x_frac & 15) as usize]);
            for y in 0..h {
                let line = source.offset(first + y as isize * ss);
                let row = out.add(y * stride);
                for x in (0..w).step_by(chunk) {
                    let pixels = filter::<C>(narrow, line.add(x), 1, &taps);
                    store::<C>(narrow, row.add(x), pixels, average);
                }
            }
            return true;
        }
        if whole_x {
            let top = origin_i - 3 * ss;
            let last_row = (((h as i32 - 1) * y_step + y_frac) >> 4) as isize;
            if !inside(top, top + (last_row + 7) * ss + wi - 1, src.len()) {
                return false;
            }
            let mut y_q4 = y_frac;
            let mut taps = C::taps(&kernel[(y_q4 & 15) as usize]);
            for y in 0..h {
                if y_step != 16 {
                    taps = C::taps(&kernel[(y_q4 & 15) as usize]);
                }
                let base = source.offset(top + (y_q4 >> 4) as isize * ss);
                let row = out.add(y * stride);
                for x in (0..w).step_by(chunk) {
                    let pixels = filter::<C>(narrow, base.add(x), src_stride, &taps);
                    store::<C>(narrow, row.add(x), pixels, average);
                }
                y_q4 += y_step;
            }
            return true;
        }

        let intermediate_height = ((((h as i32 - 1) * y_step + y_frac) >> 4) + 8) as usize;
        if intermediate_height > 135 {
            return false;
        }
        let top = origin_i - 3 * ss - 3;
        let temp_rows = temp.as_mut_ptr();
        if x_step == 16 {
            let offset = (x_frac >> 4) as isize;
            let first = top + offset;
            let last = first + (intermediate_height as isize - 1) * ss + wi - 1 + 7;
            if !inside(first, last, src.len()) {
                return false;
            }
            let taps = C::taps(&kernel[(x_frac & 15) as usize]);
            for row in 0..intermediate_height {
                let line = source.offset(first + row as isize * ss);
                for x in (0..w).step_by(chunk) {
                    let pixels = filter::<C>(narrow, line.add(x), 1, &taps);
                    store::<C>(narrow, temp_rows.add(row * 64 + x), pixels, false);
                }
            }
        } else {
            // The scalar reference's horizontal pass, unchanged.
            let top = top as usize;
            for row in 0..intermediate_height {
                let line = &src[top + row * src_stride..];
                let mut x_q4 = x_frac;
                for x in 0..w {
                    let base = (x_q4 >> 4) as usize;
                    let taps = &kernel[(x_q4 & 15) as usize];
                    let mut sum = 0i32;
                    for k in 0..8 {
                        sum += i32::from(line[base + k]) * i32::from(taps[k]);
                    }
                    temp[row * 64 + x] = ((sum + 64) >> 7).clamp(0, 255) as u8;
                    x_q4 += x_step;
                }
            }
        }
        let temp_rows = temp.as_ptr();
        let mut y_q4 = y_frac;
        let mut taps = C::taps(&kernel[(y_q4 & 15) as usize]);
        for y in 0..h {
            if y_step != 16 {
                taps = C::taps(&kernel[(y_q4 & 15) as usize]);
            }
            let base = temp_rows.add((y_q4 >> 4) as usize * 64);
            let row = out.add(y * stride);
            for x in (0..w).step_by(chunk) {
                let pixels = filter::<C>(narrow, base.add(x), 64, &taps);
                store::<C>(narrow, row.add(x), pixels, average);
            }
            y_q4 += y_step;
        }
        true
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;

    use super::Convolver;
    use zvidlib_core::simd::vector::{Avx2, Sse4};

    #[inline(always)]
    unsafe fn load4(src: *const u8) -> __m128i {
        unsafe { _mm_cvtsi32_si128(src.cast::<i32>().read_unaligned()) }
    }

    #[inline(always)]
    unsafe fn store8(dst: *mut u8, pixels: __m128i, average: bool) {
        unsafe {
            let pixels = if average {
                _mm_avg_epu8(_mm_loadl_epi64(dst.cast()), pixels)
            } else {
                pixels
            };
            _mm_storel_epi64(dst.cast(), pixels);
        }
    }

    #[inline(always)]
    unsafe fn store4(dst: *mut u8, pixels: __m128i, average: bool) {
        unsafe {
            let pixels = if average {
                _mm_avg_epu8(load4(dst), pixels)
            } else {
                pixels
            };
            dst.cast::<i32>().write_unaligned(_mm_cvtsi128_si32(pixels));
        }
    }

    /// Four outputs from 128-bit lanes, shared by both instruction sets.
    #[inline(always)]
    unsafe fn filter4(src: *const u8, pitch: usize, taps: &[__m128i; 8]) -> __m128i {
        unsafe {
            let mut sum = _mm_set1_epi32(64);
            for k in 0..8 {
                let pixels = _mm_cvtepu8_epi32(load4(src.add(k * pitch)));
                sum = _mm_add_epi32(sum, _mm_mullo_epi32(pixels, taps[k]));
            }
            let words = _mm_packs_epi32(_mm_srai_epi32::<7>(sum), _mm_srai_epi32::<7>(sum));
            _mm_packus_epi16(words, words)
        }
    }

    impl Convolver for Sse4 {
        type Taps = [__m128i; 8];
        type Pixels = __m128i;

        #[inline(always)]
        unsafe fn taps(taps: &[i16; 8]) -> [__m128i; 8] {
            unsafe {
                let mut splatted = [_mm_setzero_si128(); 8];
                for k in 0..8 {
                    splatted[k] = _mm_set1_epi32(i32::from(taps[k]));
                }
                splatted
            }
        }

        #[inline(always)]
        unsafe fn filter8(src: *const u8, pitch: usize, taps: &[__m128i; 8]) -> __m128i {
            unsafe {
                let mut low = _mm_set1_epi32(64);
                let mut high = low;
                for k in 0..8 {
                    let bytes = _mm_loadl_epi64(src.add(k * pitch).cast());
                    let tap = taps[k];
                    low = _mm_add_epi32(low, _mm_mullo_epi32(_mm_cvtepu8_epi32(bytes), tap));
                    high = _mm_add_epi32(
                        high,
                        _mm_mullo_epi32(_mm_cvtepu8_epi32(_mm_srli_si128::<4>(bytes)), tap),
                    );
                }
                let words = _mm_packs_epi32(_mm_srai_epi32::<7>(low), _mm_srai_epi32::<7>(high));
                _mm_packus_epi16(words, words)
            }
        }

        #[inline(always)]
        unsafe fn filter4(src: *const u8, pitch: usize, taps: &[__m128i; 8]) -> __m128i {
            unsafe { filter4(src, pitch, taps) }
        }

        #[inline(always)]
        unsafe fn load8(src: *const u8) -> __m128i {
            unsafe { _mm_loadl_epi64(src.cast()) }
        }

        #[inline(always)]
        unsafe fn load4(src: *const u8) -> __m128i {
            unsafe { load4(src) }
        }

        #[inline(always)]
        unsafe fn store8(dst: *mut u8, pixels: __m128i, average: bool) {
            unsafe { store8(dst, pixels, average) }
        }

        #[inline(always)]
        unsafe fn store4(dst: *mut u8, pixels: __m128i, average: bool) {
            unsafe { store4(dst, pixels, average) }
        }
    }

    impl Convolver for Avx2 {
        type Taps = [__m256i; 8];
        type Pixels = __m128i;

        #[inline(always)]
        unsafe fn taps(taps: &[i16; 8]) -> [__m256i; 8] {
            unsafe {
                let mut splatted = [_mm256_setzero_si256(); 8];
                for k in 0..8 {
                    splatted[k] = _mm256_set1_epi32(i32::from(taps[k]));
                }
                splatted
            }
        }

        #[inline(always)]
        unsafe fn filter8(src: *const u8, pitch: usize, taps: &[__m256i; 8]) -> __m128i {
            unsafe {
                let mut sum = _mm256_set1_epi32(64);
                for k in 0..8 {
                    let bytes = _mm_loadl_epi64(src.add(k * pitch).cast());
                    sum = _mm256_add_epi32(
                        sum,
                        _mm256_mullo_epi32(_mm256_cvtepu8_epi32(bytes), taps[k]),
                    );
                }
                let shifted = _mm256_srai_epi32::<7>(sum);
                let words = _mm_packs_epi32(
                    _mm256_castsi256_si128(shifted),
                    _mm256_extracti128_si256::<1>(shifted),
                );
                _mm_packus_epi16(words, words)
            }
        }

        #[inline(always)]
        unsafe fn filter4(src: *const u8, pitch: usize, taps: &[__m256i; 8]) -> __m128i {
            unsafe {
                // The low half of a 256-bit broadcast is the 128-bit one.
                let mut narrow = [_mm_setzero_si128(); 8];
                for k in 0..8 {
                    narrow[k] = _mm256_castsi256_si128(taps[k]);
                }
                filter4(src, pitch, &narrow)
            }
        }

        #[inline(always)]
        unsafe fn load8(src: *const u8) -> __m128i {
            unsafe { _mm_loadl_epi64(src.cast()) }
        }

        #[inline(always)]
        unsafe fn load4(src: *const u8) -> __m128i {
            unsafe { load4(src) }
        }

        #[inline(always)]
        unsafe fn store8(dst: *mut u8, pixels: __m128i, average: bool) {
            unsafe { store8(dst, pixels, average) }
        }

        #[inline(always)]
        unsafe fn store4(dst: *mut u8, pixels: __m128i, average: bool) {
            unsafe { store4(dst, pixels, average) }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use core::arch::aarch64::*;

    use super::Convolver;
    use zvidlib_core::simd::vector::Neon;

    #[inline(always)]
    unsafe fn load4(src: *const u8) -> uint8x8_t {
        unsafe { vcreate_u8(u64::from(src.cast::<u32>().read_unaligned())) }
    }

    impl Convolver for Neon {
        type Taps = [i16; 8];
        type Pixels = uint8x8_t;

        #[inline(always)]
        unsafe fn taps(taps: &[i16; 8]) -> [i16; 8] {
            *taps
        }

        #[inline(always)]
        unsafe fn filter8(src: *const u8, pitch: usize, taps: &[i16; 8]) -> uint8x8_t {
            unsafe {
                let mut low = vdupq_n_s32(64);
                let mut high = low;
                for k in 0..8 {
                    let wide = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src.add(k * pitch))));
                    low = vmlal_n_s16(low, vget_low_s16(wide), taps[k]);
                    high = vmlal_n_s16(high, vget_high_s16(wide), taps[k]);
                }
                // A saturating shift to unsigned 16 bits, then to bytes: the
                // scalar clamp to `0..=255` of the rounded sum.
                vqmovn_u16(vcombine_u16(
                    vqshrun_n_s32::<7>(low),
                    vqshrun_n_s32::<7>(high),
                ))
            }
        }

        #[inline(always)]
        unsafe fn filter4(src: *const u8, pitch: usize, taps: &[i16; 8]) -> uint8x8_t {
            unsafe {
                let mut sum = vdupq_n_s32(64);
                for k in 0..8 {
                    let wide = vreinterpretq_s16_u16(vmovl_u8(load4(src.add(k * pitch))));
                    sum = vmlal_n_s16(sum, vget_low_s16(wide), taps[k]);
                }
                let narrowed = vqshrun_n_s32::<7>(sum);
                vqmovn_u16(vcombine_u16(narrowed, narrowed))
            }
        }

        #[inline(always)]
        unsafe fn load8(src: *const u8) -> uint8x8_t {
            unsafe { vld1_u8(src) }
        }

        #[inline(always)]
        unsafe fn load4(src: *const u8) -> uint8x8_t {
            unsafe { load4(src) }
        }

        #[inline(always)]
        unsafe fn store8(dst: *mut u8, pixels: uint8x8_t, average: bool) {
            unsafe {
                let pixels = if average {
                    vrhadd_u8(vld1_u8(dst), pixels)
                } else {
                    pixels
                };
                vst1_u8(dst, pixels);
            }
        }

        #[inline(always)]
        unsafe fn store4(dst: *mut u8, pixels: uint8x8_t, average: bool) {
            unsafe {
                let pixels = if average {
                    vrhadd_u8(load4(dst), pixels)
                } else {
                    pixels
                };
                dst.cast::<u32>()
                    .write_unaligned(vget_lane_u32::<0>(vreinterpret_u32_u8(pixels)));
            }
        }
    }
}
