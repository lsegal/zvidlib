//! SIMD-accelerated RGBA8 to YUV420 input conversion for the encoder.
//!
//! Every encoded frame arrives as interleaved 8-bit RGBA and has to be converted to the
//! planar 8-bit YUV 4:2:0 the encoder codes, which is straight-line integer arithmetic over
//! a contiguous plane — the most directly vectorizable stage in the encoder. Two kernels
//! cover it:
//!
//! - [`luma_row`], the BT.601 studio-swing luma of one row of pixels.
//! - [`chroma_row_pair`], the Cb and Cr of one row of chroma samples, each averaged over the
//!   2x2 block of source pixels it subsamples.
//!
//! Both dispatch once per call through cached runtime CPU feature detection ([`isa`]) to an
//! SSE4.1 or AVX2 implementation on `x86_64`, a NEON implementation on `aarch64`, or the
//! portable scalar implementation everywhere else. Every vectorized path is bit-identical to
//! the scalar one, so enabling SIMD changes only how fast a frame is converted, never what
//! the encoder codes.
//!
//! # Coefficients, and why the clamps never fire
//!
//! The coefficients are the usual BT.601 full-range-RGB to studio-swing-YUV integer forms:
//! `Y = ((66R + 129G + 25B + 128) >> 8) + 16` per pixel, and, over the sum of a 2x2 block,
//! `Cb = (-38R - 74G + 112B + 131584) >> 10` and `Cr = (112R - 94G - 18B + 131584) >> 10`.
//! Their reachable ranges are exactly the studio-swing ranges the scalar code clamps to —
//! luma spans 16 through 235 and chroma 16 through 240 at the extremes of the RGB cube — so
//! the clamps are saturation the kernels can rely on rather than a case they must reproduce.
//! That is what lets the luma path stay in 16-bit lanes (its widest intermediate is
//! 66*255 + 129*255 + 25*255 + 128 = 56228) and the chroma path narrow with a saturating
//! pack. The scalar reference keeps its clamps written out, and the tests below drive both
//! paths over the corners of the RGB cube, so a coefficient change that made them reachable
//! would be caught rather than silently diverging.

use std::sync::atomic::{AtomicU8, Ordering};

/// The instruction set the conversion kernels in this module are running on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Isa {
    /// Portable fallback used on targets without a vectorized implementation.
    Scalar,
    /// x86_64 SSE4.1 (128-bit).
    #[cfg(target_arch = "x86_64")]
    Sse41,
    /// x86_64 AVX2 (256-bit).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// aarch64 Advanced SIMD (128-bit).
    #[cfg(target_arch = "aarch64")]
    Neon,
}

const ISA_UNDETECTED: u8 = 0;
const ISA_SCALAR: u8 = 1;
#[cfg(target_arch = "x86_64")]
const ISA_SSE41: u8 = 2;
#[cfg(target_arch = "x86_64")]
const ISA_AVX2: u8 = 3;
#[cfg(target_arch = "aarch64")]
const ISA_NEON: u8 = 4;

static DETECTED_ISA: AtomicU8 = AtomicU8::new(ISA_UNDETECTED);

fn detect_isa() -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return ISA_AVX2;
        }
        // The 128-bit path deinterleaves RGBA with `pshufb`, which is SSSE3; every CPU with
        // SSE4.1 has it, but the probe says so rather than assuming it.
        if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("ssse3") {
            return ISA_SSE41;
        }
    }
    // NEON is part of the aarch64 baseline, so no runtime probe is needed there.
    #[cfg(target_arch = "aarch64")]
    {
        return ISA_NEON;
    }
    #[allow(unreachable_code)]
    ISA_SCALAR
}

/// Maps the crate-wide SIMD override, if any, onto this module's ISA codes.
#[inline]
fn overridden_isa_code() -> Option<u8> {
    use crate::simd::SimdIsa;
    Some(match crate::simd::override_isa()? {
        SimdIsa::Scalar => ISA_SCALAR,
        #[cfg(target_arch = "x86_64")]
        SimdIsa::Sse41 => ISA_SSE41,
        #[cfg(target_arch = "x86_64")]
        SimdIsa::Avx2 => ISA_AVX2,
        #[cfg(target_arch = "aarch64")]
        SimdIsa::Neon => ISA_NEON,
        #[allow(unreachable_patterns)]
        _ => ISA_SCALAR,
    })
}

/// The cached ISA code, with any [`crate::simd::set_override`] override taking precedence so
/// it applies even after detection has resolved.
fn isa_code() -> u8 {
    if let Some(code) = overridden_isa_code() {
        return code;
    }
    let cached = DETECTED_ISA.load(Ordering::Relaxed);
    if cached != ISA_UNDETECTED {
        return cached;
    }
    let detected = detect_isa();
    DETECTED_ISA.store(detected, Ordering::Relaxed);
    detected
}

/// Returns the instruction set the conversion kernels will use on this machine.
pub(crate) fn isa() -> Isa {
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 => Isa::Avx2,
        #[cfg(target_arch = "x86_64")]
        ISA_SSE41 => Isa::Sse41,
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => Isa::Neon,
        _ => Isa::Scalar,
    }
}

/// Panics unless `rgba` holds at least `pixels` whole RGBA quads, so the vectorized paths can
/// read whole vectors of pixels through raw pointers without re-checking bounds per pixel.
fn check_row(rgba: &[u8], pixels: usize) {
    let needed = pixels * 4;
    assert!(
        rgba.len() >= needed,
        "row holds {} bytes, {pixels} RGBA pixels need {needed}",
        rgba.len()
    );
}

/// Converts one row of interleaved RGBA8 pixels to `out.len()` luma samples.
///
/// Panics if `rgba` is shorter than `out.len()` whole pixels.
pub(crate) fn luma_row(rgba: &[u8], out: &mut [u8]) {
    check_row(rgba, out.len());
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 => {
            // SAFETY: `check_row` proved the row is fully in bounds, and this arm was only
            // selected after `is_x86_feature_detected!("avx2")`.
            unsafe { x86::luma_row_avx2(rgba, out) }
        }
        #[cfg(target_arch = "x86_64")]
        ISA_SSE41 => {
            // SAFETY: as above, with SSE4.1 and SSSE3 detected.
            unsafe { x86::luma_row_sse41(rgba, out) }
        }
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => {
            // SAFETY: `check_row` proved the row is fully in bounds, and NEON is part of the
            // aarch64 baseline this code is compiled for.
            unsafe { neon::luma_row(rgba, out) }
        }
        _ => luma_row_scalar(rgba, out),
    }
}

/// Converts two rows of interleaved RGBA8 pixels to `cb.len()` Cb and Cr samples, each
/// averaged over the 2x2 block of source pixels it subsamples.
///
/// Panics if `cb` and `cr` differ in length or either row is shorter than `2 * cb.len()`
/// whole pixels.
pub(crate) fn chroma_row_pair(top: &[u8], bottom: &[u8], cb: &mut [u8], cr: &mut [u8]) {
    assert_eq!(
        cb.len(),
        cr.len(),
        "Cb and Cr rows differ in length: {} vs {}",
        cb.len(),
        cr.len()
    );
    let pixels = cb.len() * 2;
    check_row(top, pixels);
    check_row(bottom, pixels);
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 => {
            // SAFETY: `check_row` proved both rows hold the pixels the samples subsample, and
            // this arm was only selected after `is_x86_feature_detected!("avx2")`.
            unsafe { x86::chroma_row_pair_avx2(top, bottom, cb, cr) }
        }
        #[cfg(target_arch = "x86_64")]
        ISA_SSE41 => {
            // SAFETY: as above, with SSE4.1 and SSSE3 detected.
            unsafe { x86::chroma_row_pair_sse41(top, bottom, cb, cr) }
        }
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => {
            // SAFETY: `check_row` proved both rows hold the pixels the samples subsample, and
            // NEON is part of the aarch64 baseline this code is compiled for.
            unsafe { neon::chroma_row_pair(top, bottom, cb, cr) }
        }
        _ => chroma_row_pair_scalar(top, bottom, cb, cr),
    }
}

/// Portable luma conversion, and the reference every vectorized luma path is checked against.
pub(crate) fn luma_row_scalar(rgba: &[u8], out: &mut [u8]) {
    for (x, y) in out.iter_mut().enumerate() {
        let p = &rgba[x * 4..];
        let (r, g, b) = (i32::from(p[0]), i32::from(p[1]), i32::from(p[2]));
        *y = (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16).clamp(16, 235) as u8;
    }
}

/// Portable chroma conversion, and the reference every vectorized chroma path is checked
/// against.
pub(crate) fn chroma_row_pair_scalar(top: &[u8], bottom: &[u8], cb: &mut [u8], cr: &mut [u8]) {
    for (x, (u, v)) in cb.iter_mut().zip(cr.iter_mut()).enumerate() {
        let (mut r, mut g, mut b) = (0i32, 0i32, 0i32);
        for row in [top, bottom] {
            for dx in 0..2 {
                let at = (x * 2 + dx) * 4;
                r += i32::from(row[at]);
                g += i32::from(row[at + 1]);
                b += i32::from(row[at + 2]);
            }
        }
        *u = ((-38 * r - 74 * g + 112 * b + 131_584) >> 10).clamp(16, 240) as u8;
        *v = ((112 * r - 94 * g - 18 * b + 131_584) >> 10).clamp(16, 240) as u8;
    }
}

/// Converts the `pixels` trailing pixels a vector kernel could not fill a whole vector with.
fn luma_tail(rgba: &[u8], out: &mut [u8], done: usize) {
    if done < out.len() {
        luma_row_scalar(&rgba[done * 4..], &mut out[done..]);
    }
}

/// Converts the trailing chroma samples a vector kernel could not fill a whole vector with.
fn chroma_tail(top: &[u8], bottom: &[u8], cb: &mut [u8], cr: &mut [u8], done: usize) {
    if done < cb.len() {
        chroma_row_pair_scalar(
            &top[done * 8..],
            &bottom[done * 8..],
            &mut cb[done..],
            &mut cr[done..],
        );
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// `pshufb` selectors that scatter one colour channel of four consecutive RGBA pixels into
    /// the low byte of four `i32` lanes, zeroing the rest. Applied to a 256-bit vector they
    /// act per 128-bit lane, which is exactly one group of four pixels each.
    const SHUFFLE_R: [i8; 16] = [0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12, -1, -1, -1];
    const SHUFFLE_G: [i8; 16] = [1, -1, -1, -1, 5, -1, -1, -1, 9, -1, -1, -1, 13, -1, -1, -1];
    const SHUFFLE_B: [i8; 16] = [2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1];
    /// Gathers the low byte of four `i32` lanes into the low four bytes of the vector.
    const NARROW: [i8; 16] = [0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];

    #[target_feature(enable = "ssse3", enable = "sse4.1")]
    unsafe fn selector(bytes: &[i8; 16]) -> __m128i {
        unsafe { _mm_loadu_si128(bytes.as_ptr() as *const __m128i) }
    }

    /// The R, G and B of four consecutive pixels, one channel per vector of `i32` lanes.
    #[target_feature(enable = "ssse3", enable = "sse4.1")]
    unsafe fn split_rgb(px: __m128i) -> (__m128i, __m128i, __m128i) {
        unsafe {
            (
                _mm_shuffle_epi8(px, selector(&SHUFFLE_R)),
                _mm_shuffle_epi8(px, selector(&SHUFFLE_G)),
                _mm_shuffle_epi8(px, selector(&SHUFFLE_B)),
            )
        }
    }

    /// `((66R + 129G + 25B + 128) >> 8) + 16` over four pixels' worth of `i32` lanes.
    #[target_feature(enable = "ssse3", enable = "sse4.1")]
    unsafe fn luma_lanes(r: __m128i, g: __m128i, b: __m128i) -> __m128i {
        let acc = _mm_mullo_epi32(r, _mm_set1_epi32(66));
        let acc = _mm_add_epi32(acc, _mm_mullo_epi32(g, _mm_set1_epi32(129)));
        let acc = _mm_add_epi32(acc, _mm_mullo_epi32(b, _mm_set1_epi32(25)));
        let acc = _mm_add_epi32(acc, _mm_set1_epi32(128));
        _mm_add_epi32(_mm_srai_epi32(acc, 8), _mm_set1_epi32(16))
    }

    /// `(-38R - 74G + 112B + 131584) >> 10` over four chroma samples' worth of 2x2 channel
    /// sums, and the same for Cr's `(112R - 94G - 18B + 131584) >> 10`.
    #[target_feature(enable = "ssse3", enable = "sse4.1")]
    unsafe fn chroma_lanes(r: __m128i, g: __m128i, b: __m128i) -> (__m128i, __m128i) {
        let bias = _mm_set1_epi32(131_584);
        let cb = _mm_add_epi32(bias, _mm_mullo_epi32(b, _mm_set1_epi32(112)));
        let cb = _mm_sub_epi32(cb, _mm_mullo_epi32(r, _mm_set1_epi32(38)));
        let cb = _mm_sub_epi32(cb, _mm_mullo_epi32(g, _mm_set1_epi32(74)));
        let cr = _mm_add_epi32(bias, _mm_mullo_epi32(r, _mm_set1_epi32(112)));
        let cr = _mm_sub_epi32(cr, _mm_mullo_epi32(g, _mm_set1_epi32(94)));
        let cr = _mm_sub_epi32(cr, _mm_mullo_epi32(b, _mm_set1_epi32(18)));
        (_mm_srai_epi32(cb, 10), _mm_srai_epi32(cr, 10))
    }

    /// Writes four `i32` lanes, each already inside `0..=255`, as four bytes at `out`.
    #[target_feature(enable = "ssse3", enable = "sse4.1")]
    unsafe fn store4(v: __m128i, out: *mut u8) {
        unsafe {
            let packed = _mm_shuffle_epi8(v, selector(&NARROW));
            (out as *mut u32).write_unaligned(_mm_cvtsi128_si32(packed) as u32);
        }
    }

    /// SSE4.1 luma: four pixels per iteration, with a scalar tail.
    #[target_feature(enable = "ssse3", enable = "sse4.1")]
    pub(super) unsafe fn luma_row_sse41(rgba: &[u8], out: &mut [u8]) {
        unsafe {
            let w = out.len();
            let src = rgba.as_ptr();
            let dst = out.as_mut_ptr();
            let mut x = 0;
            while x + 4 <= w {
                let px = _mm_loadu_si128(src.add(x * 4) as *const __m128i);
                let (r, g, b) = split_rgb(px);
                store4(luma_lanes(r, g, b), dst.add(x));
                x += 4;
            }
            super::luma_tail(rgba, out, x);
        }
    }

    /// AVX2 luma: eight pixels per iteration, with the 128-bit path finishing a four-pixel
    /// remainder and the scalar reference the rest.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn luma_row_avx2(rgba: &[u8], out: &mut [u8]) {
        unsafe {
            let w = out.len();
            let src = rgba.as_ptr();
            let dst = out.as_mut_ptr();
            let r_sel =
                _mm256_broadcastsi128_si256(_mm_loadu_si128(SHUFFLE_R.as_ptr() as *const __m128i));
            let g_sel =
                _mm256_broadcastsi128_si256(_mm_loadu_si128(SHUFFLE_G.as_ptr() as *const __m128i));
            let b_sel =
                _mm256_broadcastsi128_si256(_mm_loadu_si128(SHUFFLE_B.as_ptr() as *const __m128i));
            let mut x = 0;
            while x + 8 <= w {
                let px = _mm256_loadu_si256(src.add(x * 4) as *const __m256i);
                let r = _mm256_shuffle_epi8(px, r_sel);
                let g = _mm256_shuffle_epi8(px, g_sel);
                let b = _mm256_shuffle_epi8(px, b_sel);
                let acc = _mm256_mullo_epi32(r, _mm256_set1_epi32(66));
                let acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(g, _mm256_set1_epi32(129)));
                let acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(b, _mm256_set1_epi32(25)));
                let acc = _mm256_add_epi32(acc, _mm256_set1_epi32(128));
                let y = _mm256_add_epi32(_mm256_srai_epi32(acc, 8), _mm256_set1_epi32(16));
                store4(_mm256_castsi256_si128(y), dst.add(x));
                store4(_mm256_extracti128_si256(y, 1), dst.add(x + 4));
                x += 8;
            }
            if x + 4 <= w {
                let px = _mm_loadu_si128(src.add(x * 4) as *const __m128i);
                let (r, g, b) = split_rgb(px);
                store4(luma_lanes(r, g, b), dst.add(x));
                x += 4;
            }
            super::luma_tail(rgba, out, x);
        }
    }

    /// The 2x2 channel sums for two chroma samples, from four pixels of each row.
    ///
    /// `_mm_hadd_epi32` adds the adjacent pixel pairs the subsampling averages over, so the
    /// two sums land in the low two lanes of each returned vector.
    #[target_feature(enable = "ssse3", enable = "sse4.1")]
    unsafe fn block_sums_sse41(top: *const u8, bottom: *const u8) -> (__m128i, __m128i, __m128i) {
        unsafe {
            let (tr, tg, tb) = split_rgb(_mm_loadu_si128(top as *const __m128i));
            let (br, bg, bb) = split_rgb(_mm_loadu_si128(bottom as *const __m128i));
            let r = _mm_add_epi32(tr, br);
            let g = _mm_add_epi32(tg, bg);
            let b = _mm_add_epi32(tb, bb);
            (
                _mm_hadd_epi32(r, r),
                _mm_hadd_epi32(g, g),
                _mm_hadd_epi32(b, b),
            )
        }
    }

    /// SSE4.1 chroma: four chroma samples (eight source pixels) per iteration, with a scalar
    /// tail.
    #[target_feature(enable = "ssse3", enable = "sse4.1")]
    pub(super) unsafe fn chroma_row_pair_sse41(
        top: &[u8],
        bottom: &[u8],
        cb: &mut [u8],
        cr: &mut [u8],
    ) {
        unsafe {
            let n = cb.len();
            let (tp, bp) = (top.as_ptr(), bottom.as_ptr());
            let (cbp, crp) = (cb.as_mut_ptr(), cr.as_mut_ptr());
            let mut x = 0;
            while x + 4 <= n {
                let lo = block_sums_sse41(tp.add(x * 8), bp.add(x * 8));
                let hi = block_sums_sse41(tp.add(x * 8 + 16), bp.add(x * 8 + 16));
                // Each half carries its two sums in its low two lanes; interleaving the low
                // halves puts all four samples in order.
                let r = _mm_unpacklo_epi64(lo.0, hi.0);
                let g = _mm_unpacklo_epi64(lo.1, hi.1);
                let b = _mm_unpacklo_epi64(lo.2, hi.2);
                let (u, v) = chroma_lanes(r, g, b);
                store4(u, cbp.add(x));
                store4(v, crp.add(x));
                x += 4;
            }
            super::chroma_tail(top, bottom, cb, cr, x);
        }
    }

    /// AVX2 chroma: eight chroma samples (sixteen source pixels) per iteration, with the
    /// 128-bit path finishing a four-sample remainder and the scalar reference the rest.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn chroma_row_pair_avx2(
        top: &[u8],
        bottom: &[u8],
        cb: &mut [u8],
        cr: &mut [u8],
    ) {
        unsafe {
            let n = cb.len();
            let (tp, bp) = (top.as_ptr(), bottom.as_ptr());
            let (cbp, crp) = (cb.as_mut_ptr(), cr.as_mut_ptr());
            let r_sel =
                _mm256_broadcastsi128_si256(_mm_loadu_si128(SHUFFLE_R.as_ptr() as *const __m128i));
            let g_sel =
                _mm256_broadcastsi128_si256(_mm_loadu_si128(SHUFFLE_G.as_ptr() as *const __m128i));
            let b_sel =
                _mm256_broadcastsi128_si256(_mm_loadu_si128(SHUFFLE_B.as_ptr() as *const __m128i));
            // `_mm256_hadd_epi32(a, b)` interleaves its operands per 128-bit lane, so the
            // eight pairwise sums come out as [a01 a23 b01 b23 | a45 a67 b45 b67]; this
            // permutation restores sample order.
            let order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
            let mut x = 0;
            while x + 8 <= n {
                // Channel sums for pixels 0..8 (`a`) and 8..16 (`c`) of this iteration.
                let load = |base: usize| {
                    let t = _mm256_loadu_si256(tp.add(base) as *const __m256i);
                    let b = _mm256_loadu_si256(bp.add(base) as *const __m256i);
                    [
                        _mm256_add_epi32(
                            _mm256_shuffle_epi8(t, r_sel),
                            _mm256_shuffle_epi8(b, r_sel),
                        ),
                        _mm256_add_epi32(
                            _mm256_shuffle_epi8(t, g_sel),
                            _mm256_shuffle_epi8(b, g_sel),
                        ),
                        _mm256_add_epi32(
                            _mm256_shuffle_epi8(t, b_sel),
                            _mm256_shuffle_epi8(b, b_sel),
                        ),
                    ]
                };
                let a = load(x * 8);
                let c = load(x * 8 + 32);
                let mut sums = [_mm256_setzero_si256(); 3];
                for (channel, sum) in sums.iter_mut().enumerate() {
                    *sum = _mm256_permutevar8x32_epi32(
                        _mm256_hadd_epi32(a[channel], c[channel]),
                        order,
                    );
                }
                let bias = _mm256_set1_epi32(131_584);
                let (r, g, b) = (sums[0], sums[1], sums[2]);
                let u = _mm256_add_epi32(bias, _mm256_mullo_epi32(b, _mm256_set1_epi32(112)));
                let u = _mm256_sub_epi32(u, _mm256_mullo_epi32(r, _mm256_set1_epi32(38)));
                let u = _mm256_srai_epi32(
                    _mm256_sub_epi32(u, _mm256_mullo_epi32(g, _mm256_set1_epi32(74))),
                    10,
                );
                let v = _mm256_add_epi32(bias, _mm256_mullo_epi32(r, _mm256_set1_epi32(112)));
                let v = _mm256_sub_epi32(v, _mm256_mullo_epi32(g, _mm256_set1_epi32(94)));
                let v = _mm256_srai_epi32(
                    _mm256_sub_epi32(v, _mm256_mullo_epi32(b, _mm256_set1_epi32(18))),
                    10,
                );
                store4(_mm256_castsi256_si128(u), cbp.add(x));
                store4(_mm256_extracti128_si256(u, 1), cbp.add(x + 4));
                store4(_mm256_castsi256_si128(v), crp.add(x));
                store4(_mm256_extracti128_si256(v, 1), crp.add(x + 4));
                x += 8;
            }
            if x + 4 <= n {
                let lo = block_sums_sse41(tp.add(x * 8), bp.add(x * 8));
                let hi = block_sums_sse41(tp.add(x * 8 + 16), bp.add(x * 8 + 16));
                let (u, v) = chroma_lanes(
                    _mm_unpacklo_epi64(lo.0, hi.0),
                    _mm_unpacklo_epi64(lo.1, hi.1),
                    _mm_unpacklo_epi64(lo.2, hi.2),
                );
                store4(u, cbp.add(x));
                store4(v, crp.add(x));
                x += 4;
            }
            super::chroma_tail(top, bottom, cb, cr, x);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// `((66R + 129G + 25B + 128) >> 8) + 16` over eight pixels, in 16-bit lanes.
    ///
    /// The widest intermediate a full-scale pixel reaches is 56228, so the accumulation stays
    /// inside a `u16` lane and the shift can narrow straight to bytes.
    unsafe fn luma8(r: uint8x8_t, g: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
        unsafe {
            let acc = vmull_u8(r, vdup_n_u8(66));
            let acc = vmlal_u8(acc, g, vdup_n_u8(129));
            let acc = vmlal_u8(acc, b, vdup_n_u8(25));
            let acc = vaddq_u16(acc, vdupq_n_u16(128));
            vadd_u8(vshrn_n_u16::<8>(acc), vdup_n_u8(16))
        }
    }

    /// NEON luma: sixteen pixels per iteration through `vld4q_u8`'s four-way deinterleave,
    /// with a scalar tail.
    pub(super) unsafe fn luma_row(rgba: &[u8], out: &mut [u8]) {
        unsafe {
            let w = out.len();
            let src = rgba.as_ptr();
            let dst = out.as_mut_ptr();
            let mut x = 0;
            while x + 16 <= w {
                let px = vld4q_u8(src.add(x * 4));
                let lo = luma8(vget_low_u8(px.0), vget_low_u8(px.1), vget_low_u8(px.2));
                let hi = luma8(vget_high_u8(px.0), vget_high_u8(px.1), vget_high_u8(px.2));
                vst1q_u8(dst.add(x), vcombine_u8(lo, hi));
                x += 16;
            }
            super::luma_tail(rgba, out, x);
        }
    }

    /// The 2x2 channel sums of one channel over eight chroma samples.
    ///
    /// `vld4q_u8` leaves adjacent pixels in adjacent lanes, so `vpaddlq_u8` is exactly the
    /// horizontal half of the 2x2 sum and a plain add is the vertical half.
    unsafe fn block_sums(top: uint8x16_t, bottom: uint8x16_t) -> uint16x8_t {
        unsafe { vaddq_u16(vpaddlq_u8(top), vpaddlq_u8(bottom)) }
    }

    /// Cb and Cr over four chroma samples' 2x2 channel sums, in 32-bit lanes.
    unsafe fn chroma4(r: int32x4_t, g: int32x4_t, b: int32x4_t) -> (int32x4_t, int32x4_t) {
        unsafe {
            let bias = vdupq_n_s32(131_584);
            let cb = vmlaq_n_s32(bias, b, 112);
            let cb = vmlsq_n_s32(cb, r, 38);
            let cb = vmlsq_n_s32(cb, g, 74);
            let cr = vmlaq_n_s32(bias, r, 112);
            let cr = vmlsq_n_s32(cr, g, 94);
            let cr = vmlsq_n_s32(cr, b, 18);
            (vshrq_n_s32::<10>(cb), vshrq_n_s32::<10>(cr))
        }
    }

    /// Widens the low or high half of eight 2x2 channel sums to signed 32-bit lanes.
    unsafe fn widen_low(v: uint16x8_t) -> int32x4_t {
        unsafe { vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v))) }
    }

    unsafe fn widen_high(v: uint16x8_t) -> int32x4_t {
        unsafe { vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v))) }
    }

    /// Narrows eight chroma samples, each already inside `0..=255`, to bytes.
    unsafe fn narrow(lo: int32x4_t, hi: int32x4_t) -> uint8x8_t {
        unsafe { vqmovn_u16(vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi))) }
    }

    /// NEON chroma: eight chroma samples (sixteen source pixels) per iteration, with a scalar
    /// tail.
    pub(super) unsafe fn chroma_row_pair(top: &[u8], bottom: &[u8], cb: &mut [u8], cr: &mut [u8]) {
        unsafe {
            let n = cb.len();
            let (tp, bp) = (top.as_ptr(), bottom.as_ptr());
            let (cbp, crp) = (cb.as_mut_ptr(), cr.as_mut_ptr());
            let mut x = 0;
            while x + 8 <= n {
                let t = vld4q_u8(tp.add(x * 8));
                let b = vld4q_u8(bp.add(x * 8));
                let rs = block_sums(t.0, b.0);
                let gs = block_sums(t.1, b.1);
                let bs = block_sums(t.2, b.2);
                let (u_lo, v_lo) = chroma4(widen_low(rs), widen_low(gs), widen_low(bs));
                let (u_hi, v_hi) = chroma4(widen_high(rs), widen_high(gs), widen_high(bs));
                vst1_u8(cbp.add(x), narrow(u_lo, u_hi));
                vst1_u8(crp.add(x), narrow(v_lo, v_hi));
                x += 8;
            }
            super::chroma_tail(top, bottom, cb, cr, x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{self, SimdIsa};

    /// Deterministic xorshift so every run compares the SIMD and scalar paths on identical
    /// data.
    struct Rng(u64);

    impl Rng {
        fn next_u8(&mut self) -> u8 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 33) as u8
        }
    }

    fn row(seed: u64, pixels: usize) -> Vec<u8> {
        let mut rng = Rng(seed);
        (0..pixels * 4).map(|_| rng.next_u8()).collect()
    }

    /// Widths that exercise every vector width and every tail: below one vector, exactly one,
    /// one plus a partial, and several vectors plus a partial. Chroma consumes pixels in
    /// pairs, so every width here is even.
    const WIDTHS: &[usize] = &[
        2, 4, 6, 8, 10, 14, 16, 18, 24, 30, 32, 34, 48, 62, 64, 66, 80, 128, 176, 640,
    ];

    /// Runs `check` once per instruction set this host can execute, with the crate-wide
    /// override pinned to it.
    fn for_each_isa(check: impl Fn(SimdIsa)) {
        let _guard = simd::test_lock();
        for isa in simd::available() {
            simd::set_override(Some(isa));
            assert_eq!(
                super::isa() == Isa::Scalar,
                isa == SimdIsa::Scalar,
                "pinning {} did not reach the colour conversion kernels",
                isa.name()
            );
            check(isa);
        }
        simd::set_override(None);
    }

    #[test]
    fn luma_matches_the_scalar_reference_at_every_width() {
        for_each_isa(|isa| {
            for &w in WIDTHS {
                let src = row(0x1234_5678_9abc_def1, w);
                let mut expected = vec![0u8; w];
                luma_row_scalar(&src, &mut expected);
                let mut got = vec![0u8; w];
                luma_row(&src, &mut got);
                assert_eq!(got, expected, "{w}-pixel luma row on {}", isa.name());
            }
        });
    }

    #[test]
    fn chroma_matches_the_scalar_reference_at_every_width() {
        for_each_isa(|isa| {
            for &w in WIDTHS {
                let top = row(0x0fed_cba9_8765_4321, w);
                let bottom = row(0x5deece66_d0000001, w);
                let (mut want_cb, mut want_cr) = (vec![0u8; w / 2], vec![0u8; w / 2]);
                chroma_row_pair_scalar(&top, &bottom, &mut want_cb, &mut want_cr);
                let (mut cb, mut cr) = (vec![0u8; w / 2], vec![0u8; w / 2]);
                chroma_row_pair(&top, &bottom, &mut cb, &mut cr);
                assert_eq!(cb, want_cb, "{w}-pixel Cb row on {}", isa.name());
                assert_eq!(cr, want_cr, "{w}-pixel Cr row on {}", isa.name());
            }
        });
    }

    /// The corners of the RGB cube are where the studio-swing ranges are reached, so this is
    /// what would catch a vector path that saturated differently from the scalar clamps.
    fn rgb_cube_corners(pixels: usize) -> Vec<u8> {
        const CORNERS: [[u8; 3]; 8] = [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 255, 255],
        ];
        (0..pixels)
            .flat_map(|x| {
                let c = CORNERS[x % CORNERS.len()];
                [c[0], c[1], c[2], 255]
            })
            .collect()
    }

    #[test]
    fn the_rgb_cube_corners_stay_inside_studio_swing_on_every_path() {
        let pixels = 64;
        let src = rgb_cube_corners(pixels);
        // Offsetting the second row by one corner puts eight different 2x2 blocks under the
        // chroma kernel rather than eight copies of one.
        let bottom = rgb_cube_corners(pixels + 1)[4..].to_vec();
        for_each_isa(|isa| {
            let mut luma = vec![0u8; pixels];
            luma_row(&src, &mut luma);
            let mut expected = vec![0u8; pixels];
            luma_row_scalar(&src, &mut expected);
            assert_eq!(luma, expected, "corner luma on {}", isa.name());
            assert!(
                luma.iter().all(|&y| (16..=235).contains(&y)),
                "corner luma left studio swing on {}: {luma:?}",
                isa.name()
            );
            assert!(luma.contains(&16) && luma.contains(&235));

            let (mut cb, mut cr) = (vec![0u8; pixels / 2], vec![0u8; pixels / 2]);
            chroma_row_pair(&src, &bottom, &mut cb, &mut cr);
            let (mut want_cb, mut want_cr) = (vec![0u8; pixels / 2], vec![0u8; pixels / 2]);
            chroma_row_pair_scalar(&src, &bottom, &mut want_cb, &mut want_cr);
            assert_eq!(cb, want_cb, "corner Cb on {}", isa.name());
            assert_eq!(cr, want_cr, "corner Cr on {}", isa.name());
            assert!(
                cb.iter().chain(cr.iter()).all(|&c| (16..=240).contains(&c)),
                "corner chroma left studio swing on {}",
                isa.name()
            );
        });
    }

    #[test]
    fn the_conversion_is_a_dispatch_site_the_override_reaches() {
        let _guard = simd::test_lock();
        for isa in simd::available() {
            simd::set_override(Some(isa));
            let sites = simd::active_by_site();
            let (_, site_isa) = sites
                .iter()
                .find(|(name, _)| *name == "hevc_colorconv")
                .expect("hevc_colorconv is a reported dispatch site");
            assert_eq!(
                *site_isa,
                isa,
                "pinning {} left the site behind",
                isa.name()
            );
        }
        simd::set_override(None);
    }
}
