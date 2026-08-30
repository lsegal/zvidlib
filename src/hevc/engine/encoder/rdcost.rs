//! SIMD-accelerated distortion metrics for encoder-side rate-distortion cost estimation.
//!
//! Mode and motion decisions evaluate the same two block distortion metrics over every
//! candidate partition, which is where an encoder spends most of its CPU time:
//!
//! - [`sad`], the sum of absolute differences between a source block and a prediction.
//! - [`satd`], the sum of absolute Hadamard-transformed differences, a closer proxy for the
//!   number of bits the transform stage will spend on the residual.
//!
//! Both dispatch once per call through cached runtime CPU feature detection ([`isa`]) to an
//! SSE4.1 or AVX2 implementation on `x86_64`, a NEON implementation on `aarch64`, or the
//! portable scalar implementation everywhere else. Every vectorized path is bit-identical to
//! the scalar one, so enabling SIMD changes only how fast a cost is computed, never which mode
//! the encoder picks or what it writes to the bitstream.
//!
//! `satd` follows x264's normalization so the two metrics stay on comparable scales across
//! block sizes: a 4x4 Hadamard sum is rounded down by one bit and an 8x8 sum by two.

use std::sync::atomic::{AtomicU8, Ordering};

/// The instruction set the distortion metrics in this module are running on.
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
        if is_x86_feature_detected!("sse4.1") {
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

fn isa_code() -> u8 {
    let cached = DETECTED_ISA.load(Ordering::Relaxed);
    if cached != ISA_UNDETECTED {
        return cached;
    }
    let detected = detect_isa();
    DETECTED_ISA.store(detected, Ordering::Relaxed);
    detected
}

/// Returns the instruction set the distortion metrics will use on this machine.
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

/// Largest block edge the metrics accept, matching HEVC's largest coding block.
const MAX_BLOCK: usize = 64;

/// Panics unless `plane` holds a `w` x `h` block at `stride`, so the vectorized paths below
/// can read whole rows through raw pointers without re-checking bounds per row.
fn check_block(plane: &[u8], stride: usize, w: usize, h: usize) {
    assert!(w >= 4 && h >= 4, "block is smaller than 4x4: {w}x{h}");
    assert!(
        w <= MAX_BLOCK && h <= MAX_BLOCK,
        "block is larger than {MAX_BLOCK}x{MAX_BLOCK}: {w}x{h}"
    );
    assert!(
        stride >= w,
        "stride {stride} is narrower than block width {w}"
    );
    let needed = (h - 1) * stride + w;
    assert!(
        plane.len() >= needed,
        "plane holds {} bytes, {w}x{h} block at stride {stride} needs {needed}",
        plane.len()
    );
}

/// Sum of absolute differences between a `w` x `h` source block and a prediction block.
///
/// Both planes are addressed by their own stride so callers can point at a picture buffer
/// directly. Panics if either plane is too small for the requested block.
pub(crate) fn sad(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    check_block(src, src_stride, w, h);
    check_block(pred, pred_stride, w, h);
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 => {
            // SAFETY: `check_block` proved both blocks are fully in bounds, and the dispatch
            // code was only produced by `detect_isa` after `is_x86_feature_detected!("avx2")`.
            unsafe { x86::sad_avx2(src, src_stride, pred, pred_stride, w, h) }
        }
        #[cfg(target_arch = "x86_64")]
        ISA_SSE41 => {
            // SAFETY: as above, with SSE4.1 detected.
            unsafe { x86::sad_sse41(src, src_stride, pred, pred_stride, w, h) }
        }
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => {
            // SAFETY: `check_block` proved both blocks are fully in bounds, and NEON is part
            // of the aarch64 baseline this code is compiled for.
            unsafe { neon::sad(src, src_stride, pred, pred_stride, w, h) }
        }
        _ => sad_scalar(src, src_stride, pred, pred_stride, w, h),
    }
}

/// Sum of absolute Hadamard-transformed differences between a source and prediction block.
///
/// The block is tiled into 8x8 Hadamard transforms when both dimensions are multiples of 8 and
/// into 4x4 transforms otherwise; `w` and `h` must be multiples of 4. Panics if either plane is
/// too small for the requested block or a dimension is not a multiple of 4.
pub(crate) fn satd(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    check_block(src, src_stride, w, h);
    check_block(pred, pred_stride, w, h);
    assert!(
        w % 4 == 0 && h % 4 == 0,
        "SATD block dimensions must be multiples of 4: {w}x{h}"
    );
    if w % 8 != 0 || h % 8 != 0 {
        // 4xN, Nx4 and other non-multiple-of-8 shapes only support the 4x4 transform, whose
        // scalar form is already short enough that vectorizing it buys nothing measurable.
        return satd_4x4_tiles_scalar(src, src_stride, pred, pred_stride, w, h);
    }
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        ISA_AVX2 => {
            // SAFETY: `check_block` proved both blocks are fully in bounds, and the dispatch
            // code was only produced by `detect_isa` after `is_x86_feature_detected!("avx2")`.
            unsafe { x86::satd_avx2(src, src_stride, pred, pred_stride, w, h) }
        }
        #[cfg(target_arch = "x86_64")]
        ISA_SSE41 => {
            // SAFETY: as above, with SSE4.1 detected.
            unsafe { x86::satd_sse41(src, src_stride, pred, pred_stride, w, h) }
        }
        #[cfg(target_arch = "aarch64")]
        ISA_NEON => {
            // SAFETY: `check_block` proved both blocks are fully in bounds, and NEON is part
            // of the aarch64 baseline this code is compiled for.
            unsafe { neon::satd_8x8_tiles(src, src_stride, pred, pred_stride, w, h) }
        }
        _ => satd_8x8_tiles_scalar(src, src_stride, pred, pred_stride, w, h),
    }
}

/// Portable SAD, and the reference every vectorized SAD path is checked against.
pub(crate) fn sad_scalar(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut sum = 0u32;
    for y in 0..h {
        let s = &src[y * src_stride..][..w];
        let p = &pred[y * pred_stride..][..w];
        for x in 0..w {
            sum += u32::from(s[x].abs_diff(p[x]));
        }
    }
    sum
}

/// Portable SATD, and the reference every vectorized SATD path is checked against.
pub(crate) fn satd_scalar(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    if w % 8 == 0 && h % 8 == 0 {
        satd_8x8_tiles_scalar(src, src_stride, pred, pred_stride, w, h)
    } else {
        satd_4x4_tiles_scalar(src, src_stride, pred, pred_stride, w, h)
    }
}

fn satd_8x8_tiles_scalar(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut sum = 0u32;
    for y in (0..h).step_by(8) {
        for x in (0..w).step_by(8) {
            sum += satd_8x8_scalar(
                &src[y * src_stride + x..],
                src_stride,
                &pred[y * pred_stride + x..],
                pred_stride,
            );
        }
    }
    sum
}

fn satd_4x4_tiles_scalar(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> u32 {
    let mut sum = 0u32;
    for y in (0..h).step_by(4) {
        for x in (0..w).step_by(4) {
            sum += satd_4x4_scalar(
                &src[y * src_stride + x..],
                src_stride,
                &pred[y * pred_stride + x..],
                pred_stride,
            );
        }
    }
    sum
}

/// In-place 4-point Hadamard butterfly, the same pairing the vectorized paths use.
fn hadamard4(v: &mut [i32; 4]) {
    let a0 = v[0] + v[1];
    let a1 = v[2] + v[3];
    let a2 = v[0] - v[1];
    let a3 = v[2] - v[3];
    v[0] = a0 + a1;
    v[1] = a2 + a3;
    v[2] = a0 - a1;
    v[3] = a2 - a3;
}

/// In-place 8-point Hadamard butterfly, the same pairing the vectorized paths use.
fn hadamard8(v: &mut [i32; 8]) {
    let a0 = v[0] + v[1];
    let a1 = v[2] + v[3];
    let a2 = v[4] + v[5];
    let a3 = v[6] + v[7];
    let a4 = v[0] - v[1];
    let a5 = v[2] - v[3];
    let a6 = v[4] - v[5];
    let a7 = v[6] - v[7];
    let b0 = a0 + a1;
    let b1 = a2 + a3;
    let b2 = a0 - a1;
    let b3 = a2 - a3;
    let b4 = a4 + a5;
    let b5 = a6 + a7;
    let b6 = a4 - a5;
    let b7 = a6 - a7;
    v[0] = b0 + b1;
    v[1] = b2 + b3;
    v[2] = b4 + b5;
    v[3] = b6 + b7;
    v[4] = b0 - b1;
    v[5] = b2 - b3;
    v[6] = b4 - b5;
    v[7] = b6 - b7;
}

fn satd_4x4_scalar(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize) -> u32 {
    let mut d = [[0i32; 4]; 4];
    for (y, row) in d.iter_mut().enumerate() {
        for (x, cell) in row.iter_mut().enumerate() {
            *cell = i32::from(src[y * src_stride + x]) - i32::from(pred[y * pred_stride + x]);
        }
    }
    for row in d.iter_mut() {
        hadamard4(row);
    }
    let mut cols = [[0i32; 4]; 4];
    for (y, row) in d.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            cols[x][y] = *cell;
        }
    }
    let mut sum = 0u32;
    for col in cols.iter_mut() {
        hadamard4(col);
        sum += col.iter().map(|v| v.unsigned_abs()).sum::<u32>();
    }
    (sum + 1) >> 1
}

fn satd_8x8_scalar(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize) -> u32 {
    let mut d = [[0i32; 8]; 8];
    for (y, row) in d.iter_mut().enumerate() {
        for (x, cell) in row.iter_mut().enumerate() {
            *cell = i32::from(src[y * src_stride + x]) - i32::from(pred[y * pred_stride + x]);
        }
    }
    for row in d.iter_mut() {
        hadamard8(row);
    }
    let mut cols = [[0i32; 8]; 8];
    for (y, row) in d.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            cols[x][y] = *cell;
        }
    }
    let mut sum = 0u32;
    for col in cols.iter_mut() {
        hadamard8(col);
        sum += col.iter().map(|v| v.unsigned_abs()).sum::<u32>();
    }
    (sum + 2) >> 2
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// Horizontal sum of the four `i32` lanes.
    #[target_feature(enable = "sse4.1")]
    unsafe fn hsum_epi32(v: __m128i) -> u32 {
        unsafe {
            let hi = _mm_unpackhi_epi64(v, v);
            let sum = _mm_add_epi32(v, hi);
            let sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0b01_01_01_01));
            _mm_cvtsi128_si32(sum) as u32
        }
    }

    /// SSE4.1 SAD: 16 bytes per `_mm_sad_epu8`, plus a scalar tail for 4- and 8-wide blocks.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn sad_sse41(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        w: usize,
        h: usize,
    ) -> u32 {
        unsafe {
            let mut acc = _mm_setzero_si128();
            let mut tail = 0u32;
            for y in 0..h {
                let s = src.as_ptr().add(y * src_stride);
                let p = pred.as_ptr().add(y * pred_stride);
                let mut x = 0;
                while x + 16 <= w {
                    let a = _mm_loadu_si128(s.add(x) as *const __m128i);
                    let b = _mm_loadu_si128(p.add(x) as *const __m128i);
                    acc = _mm_add_epi64(acc, _mm_sad_epu8(a, b));
                    x += 16;
                }
                if x + 8 <= w {
                    let a = _mm_loadl_epi64(s.add(x) as *const __m128i);
                    let b = _mm_loadl_epi64(p.add(x) as *const __m128i);
                    acc = _mm_add_epi64(acc, _mm_sad_epu8(a, b));
                    x += 8;
                }
                while x < w {
                    tail += u32::from((*s.add(x)).abs_diff(*p.add(x)));
                    x += 1;
                }
            }
            let lo = _mm_cvtsi128_si32(acc) as u32;
            let hi = _mm_cvtsi128_si32(_mm_unpackhi_epi64(acc, acc)) as u32;
            lo + hi + tail
        }
    }

    /// AVX2 SAD: 32 bytes per `_mm256_sad_epu8`, falling back to the 128-bit path for the
    /// 4-, 8- and 16-wide remainder of a row.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sad_avx2(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        w: usize,
        h: usize,
    ) -> u32 {
        unsafe {
            let mut acc = _mm256_setzero_si256();
            let mut narrow = _mm_setzero_si128();
            let mut tail = 0u32;
            for y in 0..h {
                let s = src.as_ptr().add(y * src_stride);
                let p = pred.as_ptr().add(y * pred_stride);
                let mut x = 0;
                while x + 32 <= w {
                    let a = _mm256_loadu_si256(s.add(x) as *const __m256i);
                    let b = _mm256_loadu_si256(p.add(x) as *const __m256i);
                    acc = _mm256_add_epi64(acc, _mm256_sad_epu8(a, b));
                    x += 32;
                }
                if x + 16 <= w {
                    let a = _mm_loadu_si128(s.add(x) as *const __m128i);
                    let b = _mm_loadu_si128(p.add(x) as *const __m128i);
                    narrow = _mm_add_epi64(narrow, _mm_sad_epu8(a, b));
                    x += 16;
                }
                if x + 8 <= w {
                    let a = _mm_loadl_epi64(s.add(x) as *const __m128i);
                    let b = _mm_loadl_epi64(p.add(x) as *const __m128i);
                    narrow = _mm_add_epi64(narrow, _mm_sad_epu8(a, b));
                    x += 8;
                }
                while x < w {
                    tail += u32::from((*s.add(x)).abs_diff(*p.add(x)));
                    x += 1;
                }
            }
            let folded = _mm_add_epi64(
                _mm256_castsi256_si128(acc),
                _mm256_extracti128_si256(acc, 1),
            );
            let folded = _mm_add_epi64(folded, narrow);
            let lo = _mm_cvtsi128_si32(folded) as u32;
            let hi = _mm_cvtsi128_si32(_mm_unpackhi_epi64(folded, folded)) as u32;
            lo + hi + tail
        }
    }

    /// One stage of the 8-point Hadamard butterfly over eight `i16` vectors.
    macro_rules! hadamard8_vectors {
        ($add:ident, $sub:ident, $v:expr) => {{
            let v = $v;
            let a0 = $add(v[0], v[1]);
            let a1 = $add(v[2], v[3]);
            let a2 = $add(v[4], v[5]);
            let a3 = $add(v[6], v[7]);
            let a4 = $sub(v[0], v[1]);
            let a5 = $sub(v[2], v[3]);
            let a6 = $sub(v[4], v[5]);
            let a7 = $sub(v[6], v[7]);
            let b0 = $add(a0, a1);
            let b1 = $add(a2, a3);
            let b2 = $sub(a0, a1);
            let b3 = $sub(a2, a3);
            let b4 = $add(a4, a5);
            let b5 = $add(a6, a7);
            let b6 = $sub(a4, a5);
            let b7 = $sub(a6, a7);
            [
                $add(b0, b1),
                $add(b2, b3),
                $add(b4, b5),
                $add(b6, b7),
                $sub(b0, b1),
                $sub(b2, b3),
                $sub(b4, b5),
                $sub(b6, b7),
            ]
        }};
    }

    /// Transposes an 8x8 `i16` matrix held one row per 128-bit vector.
    #[target_feature(enable = "sse4.1")]
    unsafe fn transpose8_epi16(r: [__m128i; 8]) -> [__m128i; 8] {
        unsafe {
            let a0 = _mm_unpacklo_epi16(r[0], r[1]);
            let a1 = _mm_unpacklo_epi16(r[2], r[3]);
            let a2 = _mm_unpacklo_epi16(r[4], r[5]);
            let a3 = _mm_unpacklo_epi16(r[6], r[7]);
            let a4 = _mm_unpackhi_epi16(r[0], r[1]);
            let a5 = _mm_unpackhi_epi16(r[2], r[3]);
            let a6 = _mm_unpackhi_epi16(r[4], r[5]);
            let a7 = _mm_unpackhi_epi16(r[6], r[7]);
            let b0 = _mm_unpacklo_epi32(a0, a1);
            let b1 = _mm_unpackhi_epi32(a0, a1);
            let b2 = _mm_unpacklo_epi32(a2, a3);
            let b3 = _mm_unpackhi_epi32(a2, a3);
            let b4 = _mm_unpacklo_epi32(a4, a5);
            let b5 = _mm_unpackhi_epi32(a4, a5);
            let b6 = _mm_unpacklo_epi32(a6, a7);
            let b7 = _mm_unpackhi_epi32(a6, a7);
            [
                _mm_unpacklo_epi64(b0, b2),
                _mm_unpackhi_epi64(b0, b2),
                _mm_unpacklo_epi64(b1, b3),
                _mm_unpackhi_epi64(b1, b3),
                _mm_unpacklo_epi64(b4, b6),
                _mm_unpackhi_epi64(b4, b6),
                _mm_unpacklo_epi64(b5, b7),
                _mm_unpackhi_epi64(b5, b7),
            ]
        }
    }

    /// SSE4.1 SATD over one 8x8 tile. Every intermediate fits in `i16`: a 2-D 8x8 Hadamard of
    /// 8-bit differences is bounded by 255 * 64 = 16320.
    #[target_feature(enable = "sse4.1")]
    unsafe fn satd_8x8_sse41(
        src: *const u8,
        src_stride: usize,
        pred: *const u8,
        pred_stride: usize,
    ) -> u32 {
        unsafe {
            let mut rows = [_mm_setzero_si128(); 8];
            for (y, row) in rows.iter_mut().enumerate() {
                let s =
                    _mm_cvtepu8_epi16(_mm_loadl_epi64(src.add(y * src_stride) as *const __m128i));
                let p =
                    _mm_cvtepu8_epi16(_mm_loadl_epi64(pred.add(y * pred_stride) as *const __m128i));
                *row = _mm_sub_epi16(s, p);
            }
            let rows = hadamard8_vectors!(_mm_add_epi16, _mm_sub_epi16, rows);
            let rows = transpose8_epi16(rows);
            let rows = hadamard8_vectors!(_mm_add_epi16, _mm_sub_epi16, rows);
            let ones = _mm_set1_epi16(1);
            let mut acc = _mm_setzero_si128();
            for row in rows {
                // Widen through `madd` before accumulating: eight lanes of up to 16320 would
                // overflow an `i16` accumulator.
                acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_abs_epi16(row), ones));
            }
            (hsum_epi32(acc) + 2) >> 2
        }
    }

    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn satd_sse41(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        w: usize,
        h: usize,
    ) -> u32 {
        unsafe {
            let mut sum = 0u32;
            for y in (0..h).step_by(8) {
                for x in (0..w).step_by(8) {
                    sum += satd_8x8_sse41(
                        src.as_ptr().add(y * src_stride + x),
                        src_stride,
                        pred.as_ptr().add(y * pred_stride + x),
                        pred_stride,
                    );
                }
            }
            sum
        }
    }

    /// AVX2 SATD over two horizontally adjacent 8x8 tiles at once. Every shuffle below is
    /// lane-local, so the 128-bit halves of each vector carry the two tiles independently and
    /// run the exact same butterfly and transpose as `satd_8x8_sse41`.
    #[target_feature(enable = "avx2")]
    unsafe fn satd_8x8_pair_avx2(
        src: *const u8,
        src_stride: usize,
        pred: *const u8,
        pred_stride: usize,
    ) -> (u32, u32) {
        unsafe {
            let mut rows = [_mm256_setzero_si256(); 8];
            for (y, row) in rows.iter_mut().enumerate() {
                let s = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                    src.add(y * src_stride) as *const __m128i
                ));
                let p = _mm256_cvtepu8_epi16(_mm_loadu_si128(
                    pred.add(y * pred_stride) as *const __m128i
                ));
                *row = _mm256_sub_epi16(s, p);
            }
            let rows = hadamard8_vectors!(_mm256_add_epi16, _mm256_sub_epi16, rows);
            let a0 = _mm256_unpacklo_epi16(rows[0], rows[1]);
            let a1 = _mm256_unpacklo_epi16(rows[2], rows[3]);
            let a2 = _mm256_unpacklo_epi16(rows[4], rows[5]);
            let a3 = _mm256_unpacklo_epi16(rows[6], rows[7]);
            let a4 = _mm256_unpackhi_epi16(rows[0], rows[1]);
            let a5 = _mm256_unpackhi_epi16(rows[2], rows[3]);
            let a6 = _mm256_unpackhi_epi16(rows[4], rows[5]);
            let a7 = _mm256_unpackhi_epi16(rows[6], rows[7]);
            let b0 = _mm256_unpacklo_epi32(a0, a1);
            let b1 = _mm256_unpackhi_epi32(a0, a1);
            let b2 = _mm256_unpacklo_epi32(a2, a3);
            let b3 = _mm256_unpackhi_epi32(a2, a3);
            let b4 = _mm256_unpacklo_epi32(a4, a5);
            let b5 = _mm256_unpackhi_epi32(a4, a5);
            let b6 = _mm256_unpacklo_epi32(a6, a7);
            let b7 = _mm256_unpackhi_epi32(a6, a7);
            let rows = [
                _mm256_unpacklo_epi64(b0, b2),
                _mm256_unpackhi_epi64(b0, b2),
                _mm256_unpacklo_epi64(b1, b3),
                _mm256_unpackhi_epi64(b1, b3),
                _mm256_unpacklo_epi64(b4, b6),
                _mm256_unpackhi_epi64(b4, b6),
                _mm256_unpacklo_epi64(b5, b7),
                _mm256_unpackhi_epi64(b5, b7),
            ];
            let rows = hadamard8_vectors!(_mm256_add_epi16, _mm256_sub_epi16, rows);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_si256();
            for row in rows {
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_abs_epi16(row), ones));
            }
            let left = hsum_epi32(_mm256_castsi256_si128(acc));
            let right = hsum_epi32(_mm256_extracti128_si256(acc, 1));
            ((left + 2) >> 2, (right + 2) >> 2)
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn satd_avx2(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        w: usize,
        h: usize,
    ) -> u32 {
        unsafe {
            let mut sum = 0u32;
            for y in (0..h).step_by(8) {
                let mut x = 0;
                while x + 16 <= w {
                    let (left, right) = satd_8x8_pair_avx2(
                        src.as_ptr().add(y * src_stride + x),
                        src_stride,
                        pred.as_ptr().add(y * pred_stride + x),
                        pred_stride,
                    );
                    sum += left + right;
                    x += 16;
                }
                if x < w {
                    sum += satd_8x8_sse41(
                        src.as_ptr().add(y * src_stride + x),
                        src_stride,
                        pred.as_ptr().add(y * pred_stride + x),
                        pred_stride,
                    );
                }
            }
            sum
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// NEON SAD: 16 bytes per `vabdq_u8`, plus 8-wide and scalar remainders. Row sums are
    /// folded into 32-bit lanes every row so the 16-bit pairwise accumulator cannot overflow.
    pub(super) unsafe fn sad(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        w: usize,
        h: usize,
    ) -> u32 {
        unsafe {
            let mut acc = vdupq_n_u32(0);
            let mut tail = 0u32;
            for y in 0..h {
                let s = src.as_ptr().add(y * src_stride);
                let p = pred.as_ptr().add(y * pred_stride);
                let mut row = vdupq_n_u16(0);
                let mut x = 0;
                while x + 16 <= w {
                    let d = vabdq_u8(vld1q_u8(s.add(x)), vld1q_u8(p.add(x)));
                    row = vpadalq_u8(row, d);
                    x += 16;
                }
                if x + 8 <= w {
                    let d = vabd_u8(vld1_u8(s.add(x)), vld1_u8(p.add(x)));
                    row = vpadalq_u8(row, vcombine_u8(d, vdup_n_u8(0)));
                    x += 8;
                }
                while x < w {
                    tail += u32::from((*s.add(x)).abs_diff(*p.add(x)));
                    x += 1;
                }
                acc = vpadalq_u16(acc, row);
            }
            vaddvq_u32(acc) + tail
        }
    }

    /// The 8-point Hadamard butterfly over eight `int16x8_t` vectors, matching `hadamard8`.
    unsafe fn hadamard8_vectors(v: [int16x8_t; 8]) -> [int16x8_t; 8] {
        unsafe {
            let a0 = vaddq_s16(v[0], v[1]);
            let a1 = vaddq_s16(v[2], v[3]);
            let a2 = vaddq_s16(v[4], v[5]);
            let a3 = vaddq_s16(v[6], v[7]);
            let a4 = vsubq_s16(v[0], v[1]);
            let a5 = vsubq_s16(v[2], v[3]);
            let a6 = vsubq_s16(v[4], v[5]);
            let a7 = vsubq_s16(v[6], v[7]);
            let b0 = vaddq_s16(a0, a1);
            let b1 = vaddq_s16(a2, a3);
            let b2 = vsubq_s16(a0, a1);
            let b3 = vsubq_s16(a2, a3);
            let b4 = vaddq_s16(a4, a5);
            let b5 = vaddq_s16(a6, a7);
            let b6 = vsubq_s16(a4, a5);
            let b7 = vsubq_s16(a6, a7);
            [
                vaddq_s16(b0, b1),
                vaddq_s16(b2, b3),
                vaddq_s16(b4, b5),
                vaddq_s16(b6, b7),
                vsubq_s16(b0, b1),
                vsubq_s16(b2, b3),
                vsubq_s16(b4, b5),
                vsubq_s16(b6, b7),
            ]
        }
    }

    /// Transposes an 8x8 `i16` matrix held one row per vector.
    unsafe fn transpose8_s16(r: [int16x8_t; 8]) -> [int16x8_t; 8] {
        unsafe {
            let a01 = vtrnq_s16(r[0], r[1]);
            let a23 = vtrnq_s16(r[2], r[3]);
            let a45 = vtrnq_s16(r[4], r[5]);
            let a67 = vtrnq_s16(r[6], r[7]);
            let b02 = vtrnq_s32(vreinterpretq_s32_s16(a01.0), vreinterpretq_s32_s16(a23.0));
            let b13 = vtrnq_s32(vreinterpretq_s32_s16(a01.1), vreinterpretq_s32_s16(a23.1));
            let b46 = vtrnq_s32(vreinterpretq_s32_s16(a45.0), vreinterpretq_s32_s16(a67.0));
            let b57 = vtrnq_s32(vreinterpretq_s32_s16(a45.1), vreinterpretq_s32_s16(a67.1));
            let (b02_0, b02_1) = (vreinterpretq_s16_s32(b02.0), vreinterpretq_s16_s32(b02.1));
            let (b13_0, b13_1) = (vreinterpretq_s16_s32(b13.0), vreinterpretq_s16_s32(b13.1));
            let (b46_0, b46_1) = (vreinterpretq_s16_s32(b46.0), vreinterpretq_s16_s32(b46.1));
            let (b57_0, b57_1) = (vreinterpretq_s16_s32(b57.0), vreinterpretq_s16_s32(b57.1));
            [
                vcombine_s16(vget_low_s16(b02_0), vget_low_s16(b46_0)),
                vcombine_s16(vget_low_s16(b13_0), vget_low_s16(b57_0)),
                vcombine_s16(vget_low_s16(b02_1), vget_low_s16(b46_1)),
                vcombine_s16(vget_low_s16(b13_1), vget_low_s16(b57_1)),
                vcombine_s16(vget_high_s16(b02_0), vget_high_s16(b46_0)),
                vcombine_s16(vget_high_s16(b13_0), vget_high_s16(b57_0)),
                vcombine_s16(vget_high_s16(b02_1), vget_high_s16(b46_1)),
                vcombine_s16(vget_high_s16(b13_1), vget_high_s16(b57_1)),
            ]
        }
    }

    /// NEON SATD over one 8x8 tile; every intermediate fits in `i16` for 8-bit input.
    unsafe fn satd_8x8(
        src: *const u8,
        src_stride: usize,
        pred: *const u8,
        pred_stride: usize,
    ) -> u32 {
        unsafe {
            let mut rows = [vdupq_n_s16(0); 8];
            for (y, row) in rows.iter_mut().enumerate() {
                let s = vld1_u8(src.add(y * src_stride));
                let p = vld1_u8(pred.add(y * pred_stride));
                // The widening unsigned subtract wraps modulo 2^16, which reinterpreted as
                // `i16` is exactly the signed difference of the two samples.
                *row = vreinterpretq_s16_u16(vsubl_u8(s, p));
            }
            let rows = hadamard8_vectors(rows);
            let rows = transpose8_s16(rows);
            let rows = hadamard8_vectors(rows);
            let mut acc = vdupq_n_u32(0);
            for row in rows {
                // Pairwise-widen before accumulating: eight lanes of up to 16320 would
                // overflow 16-bit accumulation.
                acc = vaddq_u32(acc, vpaddlq_u16(vreinterpretq_u16_s16(vabsq_s16(row))));
            }
            (vaddvq_u32(acc) + 2) >> 2
        }
    }

    pub(super) unsafe fn satd_8x8_tiles(
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        w: usize,
        h: usize,
    ) -> u32 {
        unsafe {
            let mut sum = 0u32;
            for y in (0..h).step_by(8) {
                for x in (0..w).step_by(8) {
                    sum += satd_8x8(
                        src.as_ptr().add(y * src_stride + x),
                        src_stride,
                        pred.as_ptr().add(y * pred_stride + x),
                        pred_stride,
                    );
                }
            }
            sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A distortion metric's signature, shared by the dispatching and scalar entry points.
    type Metric = fn(&[u8], usize, &[u8], usize, usize, usize) -> u32;

    /// Deterministic xorshift so every run compares the SIMD and scalar paths on identical data.
    struct Rng(u64);

    impl Rng {
        fn next_u8(&mut self) -> u8 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 33) as u8
        }
    }

    fn plane(seed: u64, stride: usize, h: usize) -> Vec<u8> {
        let mut rng = Rng(seed);
        (0..stride * h).map(|_| rng.next_u8()).collect()
    }

    /// The block shapes an HEVC mode/motion search evaluates: square coding blocks plus the
    /// symmetric and asymmetric prediction partitions derived from them.
    const SHAPES: &[(usize, usize)] = &[
        (4, 4),
        (4, 8),
        (8, 4),
        (8, 8),
        (8, 16),
        (16, 8),
        (12, 16),
        (16, 12),
        (16, 16),
        (16, 4),
        (4, 16),
        (16, 32),
        (32, 16),
        (24, 32),
        (32, 8),
        (32, 32),
        (32, 64),
        (64, 32),
        (48, 64),
        (64, 16),
        (64, 64),
    ];

    #[test]
    fn sad_matches_the_scalar_reference_for_every_block_shape() {
        for &(w, h) in SHAPES {
            for (src_stride, pred_stride) in [(w, w), (w + 7, w + 3), (MAX_BLOCK + 5, w + 1)] {
                let src = plane(0x1234_5678_9abc_def1, src_stride, h);
                let pred = plane(0x0fed_cba9_8765_4321, pred_stride, h);
                let expected = sad_scalar(&src, src_stride, &pred, pred_stride, w, h);
                assert_eq!(
                    sad(&src, src_stride, &pred, pred_stride, w, h),
                    expected,
                    "{w}x{h} SAD on {:?} at strides {src_stride}/{pred_stride}",
                    isa()
                );
            }
        }
    }

    #[test]
    fn satd_matches_the_scalar_reference_for_every_block_shape() {
        for &(w, h) in SHAPES {
            if w % 4 != 0 || h % 4 != 0 {
                continue;
            }
            for (src_stride, pred_stride) in [(w, w), (w + 7, w + 3), (MAX_BLOCK + 5, w + 1)] {
                let src = plane(0x2468_ace0_1357_9bdf, src_stride, h);
                let pred = plane(0x7777_3333_1111_9999, pred_stride, h);
                let expected = satd_scalar(&src, src_stride, &pred, pred_stride, w, h);
                assert_eq!(
                    satd(&src, src_stride, &pred, pred_stride, w, h),
                    expected,
                    "{w}x{h} SATD on {:?} at strides {src_stride}/{pred_stride}",
                    isa()
                );
            }
        }
    }

    #[test]
    fn identical_blocks_cost_nothing_and_extreme_blocks_saturate() {
        for &(w, h) in SHAPES {
            let flat = vec![0x5au8; w * h];
            assert_eq!(sad(&flat, w, &flat, w, w, h), 0, "{w}x{h} SAD");
            if w % 4 == 0 && h % 4 == 0 {
                assert_eq!(satd(&flat, w, &flat, w, w, h), 0, "{w}x{h} SATD");
            }

            let black = vec![0u8; w * h];
            let white = vec![255u8; w * h];
            // A constant difference is entirely DC: SAD counts every sample, while SATD keeps
            // only the (normalized) DC coefficient of each transform tile.
            assert_eq!(sad(&black, w, &white, w, w, h), (255 * w * h) as u32);
            if w % 4 == 0 && h % 4 == 0 {
                let tile = if w % 8 == 0 && h % 8 == 0 { 8 } else { 4 };
                let tiles = ((w / tile) * (h / tile)) as u32;
                let per_tile = 255 * (tile * tile) as u32;
                let expected = tiles * ((per_tile + tile as u32 / 4) >> (tile as u32 / 4));
                assert_eq!(satd(&black, w, &white, w, w, h), expected, "{w}x{h} SATD");
            }
        }
    }

    #[test]
    fn a_single_sample_difference_is_visible_in_both_metrics() {
        let mut src = vec![100u8; 16 * 16];
        let pred = vec![100u8; 16 * 16];
        src[5 * 16 + 9] = 140;
        assert_eq!(sad(&src, 16, &pred, 16, 16, 16), 40);
        // One impulse spreads over its whole 8x8 transform: 64 coefficients of 40, >> 2.
        assert_eq!(satd(&src, 16, &pred, 16, 16, 16), (40 * 64 + 2) >> 2);
    }

    /// Each vectorized implementation is checked directly rather than only through `sad`/`satd`
    /// dispatch, so a machine that selects AVX2 still validates the SSE4.1 code as well.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_supported_x86_implementation_matches_the_scalar_reference() {
        for &(w, h) in SHAPES {
            let (src_stride, pred_stride) = (w + 9, w + 2);
            let src = plane(0xdead_beef_cafe_f00d, src_stride, h);
            let pred = plane(0x0123_4567_89ab_cdef, pred_stride, h);
            let sad_expected = sad_scalar(&src, src_stride, &pred, pred_stride, w, h);
            let satd_expected = satd_scalar(&src, src_stride, &pred, pred_stride, w, h);
            let square = w % 8 == 0 && h % 8 == 0;
            if is_x86_feature_detected!("sse4.1") {
                // SAFETY: the feature was just detected and both blocks fit their planes.
                unsafe {
                    assert_eq!(
                        x86::sad_sse41(&src, src_stride, &pred, pred_stride, w, h),
                        sad_expected,
                        "{w}x{h} SSE4.1 SAD"
                    );
                    if square {
                        assert_eq!(
                            x86::satd_sse41(&src, src_stride, &pred, pred_stride, w, h),
                            satd_expected,
                            "{w}x{h} SSE4.1 SATD"
                        );
                    }
                }
            }
            if is_x86_feature_detected!("avx2") {
                // SAFETY: the feature was just detected and both blocks fit their planes.
                unsafe {
                    assert_eq!(
                        x86::sad_avx2(&src, src_stride, &pred, pred_stride, w, h),
                        sad_expected,
                        "{w}x{h} AVX2 SAD"
                    );
                    if square {
                        assert_eq!(
                            x86::satd_avx2(&src, src_stride, &pred, pred_stride, w, h),
                            satd_expected,
                            "{w}x{h} AVX2 SATD"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "needs")]
    fn a_block_that_overruns_its_plane_is_rejected() {
        let src = vec![0u8; 16 * 16];
        let pred = vec![0u8; 16 * 15];
        sad(&src, 16, &pred, 16, 16, 16);
    }

    #[test]
    fn the_detected_instruction_set_is_the_best_one_this_machine_supports() {
        let detected = isa();
        #[cfg(target_arch = "x86_64")]
        {
            let expected = if is_x86_feature_detected!("avx2") {
                Isa::Avx2
            } else if is_x86_feature_detected!("sse4.1") {
                Isa::Sse41
            } else {
                Isa::Scalar
            };
            assert_eq!(detected, expected);
        }
        #[cfg(target_arch = "aarch64")]
        assert_eq!(detected, Isa::Neon);
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert_eq!(detected, Isa::Scalar);
    }

    /// Encoder cost-estimation throughput, scalar versus the detected SIMD path.
    ///
    /// Ignored by default because it measures rather than asserts; run it with
    /// `cargo test --features native --release --lib rdcost::tests::bench -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn bench_distortion_metrics() {
        const STRIDE: usize = 1920;
        const ROWS: usize = 1088;
        let src = plane(0xa5a5_5a5a_c3c3_3c3c, STRIDE, ROWS);
        let pred = plane(0x1122_3344_5566_7788, STRIDE, ROWS);

        println!("detected ISA: {:?}", isa());
        println!(
            "{:<12}{:>14}{:>14}{:>10}",
            "metric", "scalar Mpix/s", "simd Mpix/s", "speedup"
        );
        for &(w, h) in &[(8usize, 8usize), (16, 16), (32, 32), (64, 64)] {
            // Sweep a 1080p-sized search area so the measurement is dominated by the kernel
            // rather than by cache misses on a tiny block.
            let steps: Vec<(usize, usize)> = (0..ROWS - h)
                .step_by(h)
                .flat_map(|y| (0..STRIDE - w).step_by(w).map(move |x| (x, y)))
                .collect();
            let pixels = (steps.len() * w * h) as f64;

            for (name, simd, scalar) in [
                ("sad", sad as Metric, sad_scalar as Metric),
                ("satd", satd, satd_scalar),
            ] {
                let run = |f: Metric| {
                    let start = std::time::Instant::now();
                    let mut acc = 0u64;
                    for &(x, y) in &steps {
                        acc += u64::from(f(
                            &src[y * STRIDE + x..],
                            STRIDE,
                            &pred[y * STRIDE + x..],
                            STRIDE,
                            w,
                            h,
                        ));
                    }
                    (start.elapsed().as_secs_f64(), acc)
                };
                let (scalar_secs, scalar_acc) = run(scalar);
                let (simd_secs, simd_acc) = run(simd);
                assert_eq!(scalar_acc, simd_acc, "{name} {w}x{h} diverged");
                println!(
                    "{:<12}{:>14.1}{:>14.1}{:>9.2}x",
                    format!("{name} {w}x{h}"),
                    pixels / scalar_secs / 1e6,
                    pixels / simd_secs / 1e6,
                    scalar_secs / simd_secs
                );
            }
        }
    }
}
