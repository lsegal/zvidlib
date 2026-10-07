//! Runtime-dispatched kernels for the AV1, VP8 and VP9 software decoders'
//! YUV-to-RGBA output conversion (issue #574).
//!
//! Every picture those three decoders return passes through
//! [`crate::av1_filters::convert_to_rgba8`], which applies the signalled colour
//! range and matrix coefficients to each pixel. Its definition is a
//! floating-point one: normalize each sample, apply the matrix in `f64`, clamp
//! to `0..=1`, scale by 255 and round half away from zero. That definition is
//! kept, unchanged, as [`Conversion::reference`]; this module computes the
//! same bytes faster.
//!
//! # The kernel
//!
//! For each colour range and matrix the conversion is an affine map of the
//! three 8-bit samples, so each output component is
//!
//! ```text
//! X = K + Cy·Y + Cu·Cb + Cv·Cr          (Q20 fixed point, i32)
//! C = Clip3(0, 255, X >> 20)
//! ```
//!
//! with the coefficients and offset rounded from the `f64` matrix and the `+0.5`
//! of the final rounding folded into `K`. [`Conversion::new`] builds the 3x4
//! table for any range and [`MatrixCoefficients`], `Identity` included (it is
//! the same shape with one nonzero coefficient per row). The vector backends
//! compute four (SSE4.1, NEON) or eight (AVX2) pixels at a time in `i32` lanes
//! and store each pixel as one little-endian `u32` `R | G<<8 | B<<16 |
//! 0xFF00_0000`, exactly as `crate::hevc::color_convert` does.
//!
//! # Bit-exactness with the `f64` reference
//!
//! Fixed point alone cannot reproduce the reference: where the exact value of
//! a component lies on or next to a `.5` rounding boundary, the coefficient
//! rounding and the reference's own `f64` rounding can each land on either
//! side of it (`1.772 · 125 = 221.5` under BT.601 full range is one), and the
//! two disagree by one. So the kernel flags every component whose Q20 value is
//! within [`TIE_MARGIN`] of a boundary and recomputes that pixel with
//! [`Conversion::reference`].
//!
//! The margin is provably wide enough. Each of the four Q20 terms is off by at
//! most half a unit from rounding, and the samples are at most 255, so `X` is
//! within `0.5 · (3 · 255 + 1) = 383` units of the exact value, below
//! `TIE_MARGIN = 512`. A component outside the margin is therefore at least 129
//! units, about `1.2e-4` of an output step, from a boundary, which is far
//! beyond the `f64` reference's own error, and both round it the same way.
//! `every_input_matches_the_f64_reference` checks this for all 2^24 sample
//! triples under every range and matrix, and around 0.1-0.9% of uniformly
//! random pixels take the fallback.
//!
//! The vector backends are bit-exact with [`convert_row_scalar`]: the same
//! `i32` arithmetic in the same order, an arithmetic `>> 20`, and the same
//! tie test, after which the scalar code redoes any vector holding a flagged
//! pixel. The largest `|X|` any range and matrix can form is below `2^30.2`,
//! so the wrapping vector arithmetic never overflows and agrees with the scalar
//! code's.
//!
//! # Dispatch
//!
//! [`detected_isa`] resolves the backend once per process, and consults
//! [`crate::simd::override_isa`] ahead of that cache on every call, so
//! `simd::set_override` reaches this kernel the way it reaches every other. The
//! site is reported as `yuv_to_rgba` by [`crate::simd::active_by_site`], on
//! every target: `wasm32` and any other architecture without a vector backend
//! report and run [`crate::simd::SimdIsa::Scalar`].

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use std::sync::OnceLock;

use crate::ColorRange;
use crate::frame::MatrixCoefficients;

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

/// Fractional bits of the fixed-point matrix.
const SHIFT: i32 = 20;

/// Mask of the fractional bits of a Q20 value.
const FRACTION: i32 = (1 << SHIFT) - 1;

/// Half-width, in Q20 units, of the band around each rounding boundary inside
/// which a component is recomputed in `f64`. See the module docs for why 512
/// covers the fixed-point error of at most 383 units.
const TIE_MARGIN: i32 = 1 << 9;

/// The `A = 255` byte, pre-placed in the high lane of the packed pixel.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const OPAQUE: i32 = 0xFF00_0000_u32 as i32;

/// One colour range and matrix, as both the `f64` reference and the Q20 table
/// the kernels evaluate.
#[derive(Clone, Copy, Debug)]
pub struct Conversion {
    range: ColorRange,
    matrix: MatrixCoefficients,
    /// Per output component (R, G, B): the `Y`, `Cb` and `Cr` coefficients and
    /// the offset, rounding term included.
    rows: [[i32; 4]; 3],
}

impl Conversion {
    /// The conversion for `range` and `matrix`.
    #[must_use]
    pub fn new(range: ColorRange, matrix: MatrixCoefficients) -> Self {
        let (y_low, y_scale, uv_scale) = range_scales(range);
        // Every row below is in units of one output step, so a component is
        // `Cy·Y + Cu·Cb + Cv·Cr + K` before the rounding the offset absorbs.
        let luma = 255.0 / y_scale;
        let rows: [[f64; 4]; 3] = if matches!(matrix, MatrixCoefficients::Identity) {
            // Identity: the planes are already (G, B, R), each normalized on its
            // own with the luma range.
            let offset = -y_low * luma;
            [
                [0.0, 0.0, luma, offset],
                [luma, 0.0, 0.0, offset],
                [0.0, luma, 0.0, offset],
            ]
        } else {
            let (kr, kb) = matrix.kr_kb();
            let v_to_r = 255.0 * 2.0 * (1.0 - kr) / uv_scale;
            let u_to_b = 255.0 * 2.0 * (1.0 - kb) / uv_scale;
            let kg = 1.0 - kr - kb;
            let u_to_g = -kb * u_to_b / kg;
            let v_to_g = -kr * v_to_r / kg;
            let y_offset = -y_low * luma;
            [
                [luma, 0.0, v_to_r, y_offset - 128.0 * v_to_r],
                [luma, u_to_g, v_to_g, y_offset - 128.0 * (u_to_g + v_to_g)],
                [luma, u_to_b, 0.0, y_offset - 128.0 * u_to_b],
            ]
        };
        let one = f64::from(1 << SHIFT);
        let rows = rows.map(|[y, u, v, k]| {
            [
                (y * one).round() as i32,
                (u * one).round() as i32,
                (v * one).round() as i32,
                (k * one).round() as i32 + (1 << (SHIFT - 1)),
            ]
        });
        Self {
            range,
            matrix,
            rows,
        }
    }

    /// The defining `f64` conversion of one pixel, as
    /// [`crate::av1_filters::convert_to_rgba8`] has always computed it.
    #[must_use]
    pub fn reference(&self, y: u8, cb: u8, cr: u8) -> [u8; 3] {
        // Chroma is always centered at 128 regardless of range (spec §7.11); only
        // the scale differs between full and studio (limited) swing.
        let uv_lo: f64 = 128.0;
        let (y_lo, y_scale, uv_scale) = range_scales(self.range);
        let y_sample = f64::from(y);
        if matches!(self.matrix, MatrixCoefficients::Identity) {
            // Identity: planes are already (G, B, R) per spec §7.11.
            return [
                to_u8(normalize(f64::from(cr), y_lo, y_scale)),
                to_u8(normalize(y_sample, y_lo, y_scale)),
                to_u8(normalize(f64::from(cb), y_lo, y_scale)),
            ];
        }
        let (kr, kb) = self.matrix.kr_kb();
        let yn = (y_sample - y_lo) / y_scale.max(1.0);
        let un = (f64::from(cb) - uv_lo) / uv_scale;
        let vn = (f64::from(cr) - uv_lo) / uv_scale;
        let r = yn + 2.0 * (1.0 - kr) * vn;
        let b = yn + 2.0 * (1.0 - kb) * un;
        let g = (yn - kr * r - kb * b) / (1.0 - kr - kb).max(1e-9);
        [to_u8(r), to_u8(g), to_u8(b)]
    }

    /// One pixel through the fixed-point table, falling back to
    /// [`Self::reference`] when any component is within [`TIE_MARGIN`] of a
    /// rounding boundary.
    #[inline]
    fn pixel(&self, y: u8, cb: u8, cr: u8) -> [u8; 3] {
        let (y32, cb32, cr32) = (i32::from(y), i32::from(cb), i32::from(cr));
        let mut out = [0_u8; 3];
        for (component, [cy, cu, cv, k]) in out.iter_mut().zip(self.rows) {
            let value = k + cy * y32 + cu * cb32 + cv * cr32;
            if near_tie(value) {
                return self.reference(y, cb, cr);
            }
            *component = (value >> SHIFT).clamp(0, 255) as u8;
        }
        out
    }
}

/// `(y_low, y_scale, uv_scale)` for a colour range.
fn range_scales(range: ColorRange) -> (f64, f64, f64) {
    match range {
        ColorRange::Limited => (16.0, 235.0 - 16.0, 224.0),
        // `Full`, and any range a later zvidlib-core adds, which this
        // conversion can only treat as full range.
        _ => (0.0, 255.0, 255.0),
    }
}

fn normalize(v: f64, low: f64, scale: f64) -> f64 {
    ((v - low) / scale.max(1.0)).clamp(0.0, 1.0)
}

#[inline]
fn to_u8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Whether a Q20 component is within [`TIE_MARGIN`] of a rounding boundary.
///
/// The rounding half is already in the value, so a boundary is where the
/// fractional bits wrap through zero.
#[inline]
fn near_tie(value: i32) -> bool {
    (value.wrapping_add(TIE_MARGIN) & FRACTION) < 2 * TIE_MARGIN
}

/// Converts one output row on the detected backend.
///
/// `luma` holds the row's samples, one per output pixel. `cb` and `cr` hold
/// the co-sited chroma row: pixel `x` reads chroma sample `x / 2` when
/// `subsampled_x` is set and `x` otherwise, clamped to the last sample so a
/// chroma row narrower than that still converts. `out` receives
/// `luma.len() * 4` bytes of RGBA.
///
/// # Panics
/// Panics if `cb` or `cr` is empty or of different lengths, or `out` is too
/// small.
pub fn convert_row(
    conversion: &Conversion,
    luma: &[u8],
    cb: &[u8],
    cr: &[u8],
    subsampled_x: bool,
    out: &mut [u8],
) {
    convert_row_with(detected_isa(), conversion, luma, cb, cr, subsampled_x, out);
}

/// [`convert_row`] on an explicitly chosen backend.
///
/// An [`Isa`] the running CPU cannot execute is not reachable — the variants
/// are compiled per architecture and [`crate::simd::set_override`] refuses to
/// pin an unavailable one — so every arm here is safe to call.
///
/// # Panics
/// As [`convert_row`].
pub fn convert_row_with(
    isa: Isa,
    conversion: &Conversion,
    luma: &[u8],
    cb: &[u8],
    cr: &[u8],
    subsampled_x: bool,
    out: &mut [u8],
) {
    assert!(!cb.is_empty() && cb.len() == cr.len());
    let out = &mut out[..luma.len() * 4];
    // The vector loops only run where every lane's chroma sample exists, so
    // they never need the clamp; the scalar tail applies it.
    #[cfg_attr(
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        allow(unused_variables)
    )]
    let vector_end = if subsampled_x {
        luma.len().min(cb.len() * 2)
    } else {
        luma.len().min(cb.len())
    };
    match isa {
        Isa::Scalar => {
            convert_row_scalar(conversion, luma, cb, cr, subsampled_x, out, 0);
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `Isa::Sse41` is only reachable on a CPU `detect` or
        // `crate::simd::available` confirmed supports SSE4.1, and every vector
        // load stays below `vector_end`.
        Isa::Sse41 => unsafe {
            convert_row_sse41(conversion, luma, cb, cr, subsampled_x, out, vector_end);
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above, for AVX2.
        Isa::Avx2 => unsafe {
            convert_row_avx2(conversion, luma, cb, cr, subsampled_x, out, vector_end);
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is architecturally mandatory on aarch64, and every
        // vector load stays below `vector_end`.
        Isa::Neon => unsafe {
            convert_row_neon(conversion, luma, cb, cr, subsampled_x, out, vector_end);
        },
    }
}

/// The scalar reference, converting pixels `start..end` of the row.
///
/// The vector backends call it for their tail and for any vector that holds a
/// pixel near a rounding boundary, so both run the arithmetic that defines the
/// result.
#[allow(clippy::too_many_arguments)]
fn convert_range_scalar(
    conversion: &Conversion,
    luma: &[u8],
    cb: &[u8],
    cr: &[u8],
    subsampled_x: bool,
    out: &mut [u8],
    start: usize,
    end: usize,
) {
    let last = cb.len() - 1;
    for x in start..end {
        let c = if subsampled_x { x / 2 } else { x }.min(last);
        let [r, g, b] = conversion.pixel(luma[x], cb[c], cr[c]);
        out[x * 4..x * 4 + 4].copy_from_slice(&[r, g, b, 255]);
    }
}

/// The scalar reference from `start` to the end of the row.
fn convert_row_scalar(
    conversion: &Conversion,
    luma: &[u8],
    cb: &[u8],
    cr: &[u8],
    subsampled_x: bool,
    out: &mut [u8],
    start: usize,
) {
    convert_range_scalar(
        conversion,
        luma,
        cb,
        cr,
        subsampled_x,
        out,
        start,
        luma.len(),
    );
}

/// Four pixels per step in `i32` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn convert_row_sse41(
    conversion: &Conversion,
    luma: &[u8],
    cb: &[u8],
    cr: &[u8],
    subsampled_x: bool,
    out: &mut [u8],
    vector_end: usize,
) {
    /// Four `u8` samples at `at`, widened to `i32` lanes.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn load4(samples: &[u8], at: usize) -> __m128i {
        let bytes = unsafe { samples.as_ptr().add(at).cast::<i32>().read_unaligned() };
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128(bytes))
    }
    /// The four lanes' chroma: `[a, a, b, b]` from two samples at `at / 2`
    /// when subsampled, four samples at `at` otherwise.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn chroma4(samples: &[u8], at: usize, subsampled_x: bool) -> __m128i {
        if subsampled_x {
            let pair = unsafe { samples.as_ptr().add(at / 2).cast::<u16>().read_unaligned() };
            let widened = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(i32::from(pair)));
            _mm_shuffle_epi32(widened, 0b01_01_00_00)
        } else {
            unsafe { load4(samples, at) }
        }
    }

    /// One component of four pixels, clipped, with the lanes near a rounding
    /// boundary or'd into `ties`.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn component(
        [cy, cu, cv, k]: [i32; 4],
        y: __m128i,
        u: __m128i,
        v: __m128i,
        ties: &mut __m128i,
    ) -> __m128i {
        let value = _mm_add_epi32(
            _mm_add_epi32(
                _mm_add_epi32(_mm_set1_epi32(k), _mm_mullo_epi32(_mm_set1_epi32(cy), y)),
                _mm_mullo_epi32(_mm_set1_epi32(cu), u),
            ),
            _mm_mullo_epi32(_mm_set1_epi32(cv), v),
        );
        let wrapped = _mm_and_si128(
            _mm_add_epi32(value, _mm_set1_epi32(TIE_MARGIN)),
            _mm_set1_epi32(FRACTION),
        );
        *ties = _mm_or_si128(
            *ties,
            _mm_cmplt_epi32(wrapped, _mm_set1_epi32(2 * TIE_MARGIN)),
        );
        _mm_min_epi32(
            _mm_max_epi32(_mm_srai_epi32(value, SHIFT), _mm_setzero_si128()),
            _mm_set1_epi32(255),
        )
    }

    let [r_row, g_row, b_row] = conversion.rows;
    let mut x = 0;
    while x + 4 <= vector_end {
        unsafe {
            let y = load4(luma, x);
            let u = chroma4(cb, x, subsampled_x);
            let v = chroma4(cr, x, subsampled_x);
            let mut ties = _mm_setzero_si128();
            let red = component(r_row, y, u, v, &mut ties);
            let green = component(g_row, y, u, v, &mut ties);
            let blue = component(b_row, y, u, v, &mut ties);
            let packed = _mm_or_si128(
                _mm_or_si128(red, _mm_slli_epi32(green, 8)),
                _mm_or_si128(_mm_slli_epi32(blue, 16), _mm_set1_epi32(OPAQUE)),
            );
            _mm_storeu_si128(out.as_mut_ptr().add(x * 4).cast(), packed);
            if _mm_movemask_epi8(ties) != 0 {
                convert_range_scalar(conversion, luma, cb, cr, subsampled_x, out, x, x + 4);
            }
        }
        x += 4;
    }
    convert_row_scalar(conversion, luma, cb, cr, subsampled_x, out, x);
}

/// Eight pixels per step in `i32` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convert_row_avx2(
    conversion: &Conversion,
    luma: &[u8],
    cb: &[u8],
    cr: &[u8],
    subsampled_x: bool,
    out: &mut [u8],
    vector_end: usize,
) {
    /// Eight `u8` samples at `at`, widened to `i32` lanes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load8(samples: &[u8], at: usize) -> __m256i {
        unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(samples.as_ptr().add(at).cast())) }
    }
    /// The eight lanes' chroma: `[a, a, b, b, c, c, d, d]` from four samples
    /// at `at / 2` when subsampled, eight samples at `at` otherwise.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn chroma8(samples: &[u8], at: usize, subsampled_x: bool, pattern: __m256i) -> __m256i {
        if subsampled_x {
            let quad = unsafe { samples.as_ptr().add(at / 2).cast::<i32>().read_unaligned() };
            let widened = _mm256_cvtepu8_epi32(_mm_cvtsi32_si128(quad));
            _mm256_permutevar8x32_epi32(widened, pattern)
        } else {
            unsafe { load8(samples, at) }
        }
    }

    /// One component of eight pixels, clipped, with the lanes near a rounding
    /// boundary or'd into `ties`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn component(
        [cy, cu, cv, k]: [i32; 4],
        y: __m256i,
        u: __m256i,
        v: __m256i,
        ties: &mut __m256i,
    ) -> __m256i {
        let value = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_set1_epi32(k),
                    _mm256_mullo_epi32(_mm256_set1_epi32(cy), y),
                ),
                _mm256_mullo_epi32(_mm256_set1_epi32(cu), u),
            ),
            _mm256_mullo_epi32(_mm256_set1_epi32(cv), v),
        );
        let wrapped = _mm256_and_si256(
            _mm256_add_epi32(value, _mm256_set1_epi32(TIE_MARGIN)),
            _mm256_set1_epi32(FRACTION),
        );
        *ties = _mm256_or_si256(
            *ties,
            _mm256_cmpgt_epi32(_mm256_set1_epi32(2 * TIE_MARGIN), wrapped),
        );
        _mm256_min_epi32(
            _mm256_max_epi32(_mm256_srai_epi32(value, SHIFT), _mm256_setzero_si256()),
            _mm256_set1_epi32(255),
        )
    }

    let [r_row, g_row, b_row] = conversion.rows;
    let pattern = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    let mut x = 0;
    while x + 8 <= vector_end {
        unsafe {
            let y = load8(luma, x);
            let u = chroma8(cb, x, subsampled_x, pattern);
            let v = chroma8(cr, x, subsampled_x, pattern);
            let mut ties = _mm256_setzero_si256();
            let red = component(r_row, y, u, v, &mut ties);
            let green = component(g_row, y, u, v, &mut ties);
            let blue = component(b_row, y, u, v, &mut ties);
            let packed = _mm256_or_si256(
                _mm256_or_si256(red, _mm256_slli_epi32(green, 8)),
                _mm256_or_si256(_mm256_slli_epi32(blue, 16), _mm256_set1_epi32(OPAQUE)),
            );
            _mm256_storeu_si256(out.as_mut_ptr().add(x * 4).cast(), packed);
            if _mm256_movemask_epi8(ties) != 0 {
                convert_range_scalar(conversion, luma, cb, cr, subsampled_x, out, x, x + 8);
            }
        }
        x += 8;
    }
    convert_row_scalar(conversion, luma, cb, cr, subsampled_x, out, x);
}

/// Four pixels per step in `i32` lanes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_row_neon(
    conversion: &Conversion,
    luma: &[u8],
    cb: &[u8],
    cr: &[u8],
    subsampled_x: bool,
    out: &mut [u8],
    vector_end: usize,
) {
    /// The low four bytes of `bytes`, widened to `i32` lanes.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn widen(bytes: u32) -> int32x4_t {
        let wide = vmovl_u8(vcreate_u8(u64::from(bytes)));
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(wide)))
    }
    /// The four lanes' chroma: `[a, a, b, b]` from two samples at `at / 2`
    /// when subsampled, four samples at `at` otherwise.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn chroma4(samples: &[u8], at: usize, subsampled_x: bool) -> int32x4_t {
        if subsampled_x {
            let pair = unsafe { samples.as_ptr().add(at / 2).cast::<u16>().read_unaligned() };
            let widened = unsafe { widen(u32::from(pair)) };
            vzip1q_s32(widened, widened)
        } else {
            unsafe { widen(samples.as_ptr().add(at).cast::<u32>().read_unaligned()) }
        }
    }

    /// One component of four pixels, clipped, with the lanes near a rounding
    /// boundary or'd into `ties`.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn component(
        [cy, cu, cv, k]: [i32; 4],
        y: int32x4_t,
        u: int32x4_t,
        v: int32x4_t,
        ties: &mut uint32x4_t,
    ) -> int32x4_t {
        let value = vaddq_s32(
            vaddq_s32(
                vaddq_s32(vdupq_n_s32(k), vmulq_n_s32(y, cy)),
                vmulq_n_s32(u, cu),
            ),
            vmulq_n_s32(v, cv),
        );
        let wrapped = vandq_s32(
            vaddq_s32(value, vdupq_n_s32(TIE_MARGIN)),
            vdupq_n_s32(FRACTION),
        );
        *ties = vorrq_u32(*ties, vcltq_s32(wrapped, vdupq_n_s32(2 * TIE_MARGIN)));
        vminq_s32(
            vmaxq_s32(vshrq_n_s32::<SHIFT>(value), vdupq_n_s32(0)),
            vdupq_n_s32(255),
        )
    }

    let [r_row, g_row, b_row] = conversion.rows;
    let mut x = 0;
    while x + 4 <= vector_end {
        unsafe {
            let y = widen(luma.as_ptr().add(x).cast::<u32>().read_unaligned());
            let u = chroma4(cb, x, subsampled_x);
            let v = chroma4(cr, x, subsampled_x);
            let mut ties = vdupq_n_u32(0);
            let red = component(r_row, y, u, v, &mut ties);
            let green = component(g_row, y, u, v, &mut ties);
            let blue = component(b_row, y, u, v, &mut ties);
            let packed = vorrq_s32(
                vorrq_s32(red, vshlq_n_s32::<8>(green)),
                vorrq_s32(vshlq_n_s32::<16>(blue), vdupq_n_s32(OPAQUE)),
            );
            vst1q_u8(out.as_mut_ptr().add(x * 4), vreinterpretq_u8_s32(packed));
            if vmaxvq_u32(ties) != 0 {
                convert_range_scalar(conversion, luma, cb, cr, subsampled_x, out, x, x + 4);
            }
        }
        x += 4;
    }
    convert_row_scalar(conversion, luma, cb, cr, subsampled_x, out, x);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{self, SimdIsa};

    const RANGES: [ColorRange; 2] = [ColorRange::Full, ColorRange::Limited];
    const MATRICES: [MatrixCoefficients; 4] = [
        MatrixCoefficients::Identity,
        MatrixCoefficients::Bt709,
        MatrixCoefficients::Bt601,
        MatrixCoefficients::Bt2020Ncl,
    ];

    fn conversions() -> impl Iterator<Item = Conversion> {
        RANGES.into_iter().flat_map(|range| {
            MATRICES
                .into_iter()
                .map(move |matrix| Conversion::new(range, matrix))
        })
    }

    fn backends() -> Vec<Isa> {
        #[cfg_attr(
            not(any(target_arch = "x86_64", target_arch = "aarch64")),
            allow(unused_mut)
        )]
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

    /// The fixed-point path, tie fallback included, against the `f64`
    /// definition for every one of the 2^24 sample triples, under every range
    /// and matrix. This is the whole input space, so it is the proof the module
    /// docs' error bound argues for rather than a sample of it.
    #[test]
    fn every_input_matches_the_f64_reference() {
        for conversion in conversions() {
            for y in 0..=255 {
                for cb in 0..=255 {
                    for cr in 0..=255 {
                        let fixed = conversion.pixel(y, cb, cr);
                        let reference = conversion.reference(y, cb, cr);
                        assert_eq!(fixed, reference, "{conversion:?} at Y={y} Cb={cb} Cr={cr}");
                    }
                }
            }
        }
    }

    /// The largest `|X|` a table can form stays clear of `i32`, so the vector
    /// backends' wrapping arithmetic agrees with the scalar code's.
    #[test]
    fn no_partial_sum_can_overflow() {
        for conversion in conversions() {
            for row in conversion.rows {
                let bound: i64 = row[..3]
                    .iter()
                    .map(|c| i64::from(*c).abs() * 255)
                    .sum::<i64>()
                    + i64::from(row[3]).abs()
                    + i64::from(TIE_MARGIN);
                assert!(bound < i64::from(i32::MAX), "{conversion:?}: {bound}");
            }
        }
    }

    /// The vectors that exercise every chroma pairing and every rounding
    /// boundary: an exhaustive (Y, Cb) sweep under a cycling Cr, laid out as
    /// rows of `width` pixels.
    fn sweep_row(seed: usize, width: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let luma = (0..width)
            .map(|x| ((x * 7 + seed * 13) % 256) as u8)
            .collect();
        let cb = (0..width).map(|x| ((x * 11 + seed) % 256) as u8).collect();
        let cr = (0..width)
            .map(|x| (255 - (x * 5 + seed * 3) % 256) as u8)
            .collect();
        (luma, cb, cr)
    }

    #[test]
    fn every_backend_is_bit_exact_with_the_scalar_reference() {
        // Widths that leave a tail for the 4-lane and the 8-lane kernels, and
        // odd widths where the last pixel's chroma column is a half sample.
        for conversion in conversions() {
            for width in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 64, 1921] {
                for subsampled_x in [true, false] {
                    for seed in 0..64 {
                        let (luma, cb, cr) = sweep_row(seed, width);
                        let chroma_width = if subsampled_x {
                            width.div_ceil(2)
                        } else {
                            width
                        };
                        let (cb, cr) = (&cb[..chroma_width], &cr[..chroma_width]);
                        let mut reference = vec![0; width * 4];
                        convert_row_with(
                            Isa::Scalar,
                            &conversion,
                            &luma,
                            cb,
                            cr,
                            subsampled_x,
                            &mut reference,
                        );
                        for isa in backends() {
                            let mut out = vec![0; width * 4];
                            convert_row_with(
                                isa,
                                &conversion,
                                &luma,
                                cb,
                                cr,
                                subsampled_x,
                                &mut out,
                            );
                            assert_eq!(out, reference, "{isa:?} {conversion:?} width {width}");
                        }
                    }
                }
            }
        }
    }

    /// Randomized rows, including a chroma row narrower than the luma row
    /// needs, where the last chroma sample is reused past its end.
    #[test]
    fn randomized_rows_and_short_chroma_are_bit_exact() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for conversion in conversions() {
            for _ in 0..64 {
                let width = (next() % 200) as usize + 1;
                let subsampled_x = next() % 2 == 0;
                let full = if subsampled_x {
                    width.div_ceil(2)
                } else {
                    width
                };
                let chroma_width = (next() as usize % full) + 1;
                let luma: Vec<u8> = (0..width).map(|_| next() as u8).collect();
                let cb: Vec<u8> = (0..chroma_width).map(|_| next() as u8).collect();
                let cr: Vec<u8> = (0..chroma_width).map(|_| next() as u8).collect();
                let mut reference = vec![0; width * 4];
                for (x, pixel) in reference.chunks_exact_mut(4).enumerate() {
                    let c = if subsampled_x { x / 2 } else { x }.min(chroma_width - 1);
                    let [r, g, b] = conversion.reference(luma[x], cb[c], cr[c]);
                    pixel.copy_from_slice(&[r, g, b, 255]);
                }
                for isa in backends() {
                    let mut out = vec![0; width * 4];
                    convert_row_with(isa, &conversion, &luma, &cb, &cr, subsampled_x, &mut out);
                    assert_eq!(out, reference, "{isa:?} {conversion:?} width {width}");
                }
            }
        }
    }

    /// A row whose every pixel sits on an exact rounding tie — BT.601 full
    /// range `B = Y + 1.772·125`, which is `x.5` for every `Y` — so every
    /// vector takes the fallback, and the result is still the reference's.
    #[test]
    fn a_row_of_exact_ties_takes_the_fallback_everywhere() {
        let conversion = Conversion::new(ColorRange::Full, MatrixCoefficients::Bt601);
        let luma: Vec<u8> = (0..=33).collect();
        let cb = vec![253; luma.len()];
        let cr = vec![128; luma.len()];
        assert!(luma.iter().all(|&y| {
            let blue = conversion.rows[2];
            near_tie(blue[3] + blue[0] * i32::from(y) + blue[1] * 253 + blue[2] * 128)
        }));
        for isa in backends() {
            let mut out = vec![0; luma.len() * 4];
            convert_row_with(isa, &conversion, &luma, &cb, &cr, false, &mut out);
            for (y, pixel) in luma.iter().zip(out.chunks_exact(4)) {
                let [r, g, b] = conversion.reference(*y, 253, 128);
                assert_eq!(pixel, [r, g, b, 255], "{isa:?} Y={y}");
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
