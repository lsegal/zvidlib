//! §8.5.3.3.3 — fractional sample interpolation, plus the §8.5.3.3.4
//! weighted sample prediction combines (default and explicit).
//!
//! This module turns a reference-picture sample plane and a motion vector
//! into an `(nPbW)x(nPbH)` array of inter-predicted samples. Four ITU-T
//! H.265 subclauses are implemented, in the order §8.5.3.3.3.1 invokes
//! them:
//!
//! * §8.5.3.3.3.2 **luma sample interpolation** ([`interp_luma_block`]) —
//!   the separable 8-tap quarter-pel filter of equations 8-224..8-238,
//!   with the Table 8-8 phase selection. `shift1 = Min(4, BitDepthY − 8)`,
//!   `shift2 = 6`, `shift3 = Max(2, 14 − BitDepthY)`; the full-pel case is
//!   `A << shift3`.
//! * §8.5.3.3.3.3 **chroma sample interpolation** ([`interp_chroma_block`])
//!   — the separable 4-tap eighth-pel filter of equations 8-241..8-261,
//!   with the Table 8-9 phase selection. `shift1 = Min(4, BitDepthC − 8)`,
//!   `shift2 = 6`, `shift3 = Max(2, 14 − BitDepthC)`.
//! * §8.5.3.3.4.2 **default weighted sample prediction**
//!   ([`default_weighted_pred`]) — the uni- / bi-predictive combine of
//!   equations 8-262..8-264 (`weightedPredFlag == 0`), with
//!   `shift1 = Max(2, 14 − bitDepth)`, `shift2 = Max(3, 15 − bitDepth)`.
//! * §8.5.3.3.4.3 **explicit weighted sample prediction**
//!   ([`explicit_weighted_pred`]) — the per-reference weight / offset
//!   combine of equations 8-265..8-277 (`weightedPredFlag == 1`, i.e.
//!   `weighted_pred_flag` for P slices / `weighted_bipred_flag` for B
//!   slices), with `log2Wd = log2WeightDenom + shift1`.
//!
//! The interpolation processes emit *intermediate* sample values at the
//! `14 − BitDepth`-bit internal precision the spec carries between
//! §8.5.3.3.3 and §8.5.3.3.4 (i.e. the `>> shift1` / `>> shift2` outputs,
//! `A << shift3` for full-pel — they are **not** yet clipped to the
//! sample range). The weighted combines consume those intermediate
//! arrays and produce the final `[0, (1 << bitDepth) − 1]` prediction
//! samples.
//!
//! ## Scope
//!
//! The numerics are self-contained. The §8.5.3.2 merge / §8.5.3.1 MV
//! derivation that produces `mvLX`, the §8.5.3.3.1 driver that splits a
//! motion vector into its integer / fractional parts and walks the
//! prediction block, and the §8.6.5 picture-construction step that adds
//! the residual are the caller's responsibility — this module starts at
//! a `(xInt, yInt, xFrac, yFrac)` location and a reference plane, and
//! stops at the prediction sample arrays.

use crate::hevc::engine::simd::{self, Isa};

/// A reference-picture luma / chroma sample plane with the §8.5.3.3.3
/// `Clip3( 0, dim − 1, … )` edge-extension border (equations 8-222 /
/// 8-223 for luma, 8-239 / 8-240 for chroma).
///
/// The interpolation filters read samples at negative and past-the-edge
/// coordinates; this type clamps every access into the valid plane so the
/// callers can index with the raw `xInt + i` / `yInt + j` offsets the
/// equations use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefPlane<'a> {
    /// Row-major samples, `width * height` of them. `sample[ y * width + x ]`
    /// is the plane sample at full-sample location `( x, y )`.
    samples: &'a [i32],
    /// Plane width in samples (`pic_width_in_luma_samples` for luma, or
    /// `pic_width_in_luma_samples / SubWidthC` for chroma).
    width: usize,
    /// Plane height in samples.
    height: usize,
}

impl<'a> RefPlane<'a> {
    /// Wraps a row-major `width * height` sample plane.
    ///
    /// # Errors
    ///
    /// [`InterPredError::PlaneLengthMismatch`] if `samples.len()` is not
    /// exactly `width * height`, or [`InterPredError::EmptyPlane`] if
    /// either dimension is zero.
    pub fn new(samples: &'a [i32], width: usize, height: usize) -> Result<Self, InterPredError> {
        if width == 0 || height == 0 {
            return Err(InterPredError::EmptyPlane);
        }
        let expected = width
            .checked_mul(height)
            .ok_or(InterPredError::EmptyPlane)?;
        if samples.len() != expected {
            return Err(InterPredError::PlaneLengthMismatch {
                expected,
                got: samples.len(),
            });
        }
        Ok(Self {
            samples,
            width,
            height,
        })
    }

    /// The plane width in samples.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The plane height in samples.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Sample at full-sample location `( x, y )` with the §8.5.3.3.3
    /// `Clip3( 0, dim − 1, … )` edge extension (equations 8-222 / 8-223 /
    /// 8-239 / 8-240). `x` and `y` may be negative or past the edge.
    #[inline]
    #[must_use]
    pub fn at(&self, x: i32, y: i32) -> i32 {
        let xc = x.clamp(0, self.width as i32 - 1) as usize;
        let yc = y.clamp(0, self.height as i32 - 1) as usize;
        self.samples[yc * self.width + xc]
    }

    /// Copies `dst.len()` samples of row `y` starting at full-sample
    /// column `x_start` into `dst`, with the same §8.5.3.3.3 `Clip3` edge
    /// extension [`RefPlane::at`] applies. `x_start` may be negative and
    /// the window may run past either edge.
    ///
    /// The wholly-inside case is a straight row copy; only a window that
    /// actually crosses an edge pays for the per-sample clamp.
    #[inline]
    fn copy_row(&self, x_start: i32, y: i32, dst: &mut [i32]) {
        let yc = y.clamp(0, self.height as i32 - 1) as usize;
        let base = yc * self.width;
        if x_start >= 0 && (x_start as usize).saturating_add(dst.len()) <= self.width {
            let start = base + x_start as usize;
            dst.copy_from_slice(&self.samples[start..start + dst.len()]);
            return;
        }
        let row = &self.samples[base..base + self.width];
        let last = self.width as i32 - 1;
        for (i, d) in dst.iter_mut().enumerate() {
            *d = row[x_start.saturating_add(i as i32).clamp(0, last) as usize];
        }
    }

    /// A `len`-sample window of row `y` starting at column `x_start`.
    ///
    /// Borrowed straight out of the plane when the window lies wholly
    /// inside it — the common case for the interpolation filters, which
    /// then read the reference samples with no copy at all — and
    /// materialized into `scratch` with edge extension when it does not.
    /// `scratch` must be at least `len` long.
    #[inline]
    fn row_window<'s>(
        &'s self,
        x_start: i32,
        len: usize,
        y: i32,
        scratch: &'s mut [i32],
    ) -> &'s [i32] {
        let yc = y.clamp(0, self.height as i32 - 1) as usize;
        let base = yc * self.width;
        if x_start >= 0 && (x_start as usize).saturating_add(len) <= self.width {
            let start = base + x_start as usize;
            return &self.samples[start..start + len];
        }
        self.copy_row(x_start, y, &mut scratch[..len]);
        &scratch[..len]
    }

    /// The [`RefPlane::row_window`] window of row `y`, narrowed into
    /// `dst` for the 16-bit interpolation path.
    ///
    /// Converts straight out of the plane rather than staging through
    /// [`RefPlane::row_window`]'s `i32` scratch: the narrow path already
    /// has to write a buffer, so routing it through the wide one would
    /// add a whole `i32` pass over the source that the wide path does
    /// not pay — `row_window` borrows the plane outright when the window
    /// lies inside it. `dst` must be at least `len` long.
    #[inline]
    fn row_window_narrow(&self, x_start: i32, len: usize, y: i32, dst: &mut [i16]) {
        let yc = y.clamp(0, self.height as i32 - 1) as usize;
        let base = yc * self.width;
        if x_start >= 0 && (x_start as usize).saturating_add(len) <= self.width {
            let start = base + x_start as usize;
            narrow_samples(&self.samples[start..start + len], &mut dst[..len]);
            return;
        }
        let row = &self.samples[base..base + self.width];
        let last = self.width as i32 - 1;
        for (i, d) in dst[..len].iter_mut().enumerate() {
            let s = row[x_start.saturating_add(i as i32).clamp(0, last) as usize];
            debug_assert!(
                (0..=i32::from(NARROW_MAX_SAMPLE)).contains(&s),
                "sample {s} is outside the eight-bit range the narrow interpolation path requires"
            );
            *d = s as i16;
        }
    }

    /// [`RefPlane::gather`] narrowed to `i16` for the 16-bit
    /// interpolation path.
    fn gather_narrow(&self, x0: i32, y0: i32, w: usize, h: usize) -> Vec<i16> {
        let mut buf = vec![0i16; w * h];
        for (r, row) in buf.chunks_exact_mut(w).enumerate() {
            self.row_window_narrow(x0, w, y0 + r as i32, row);
        }
        buf
    }

    /// Gathers the edge-extended `w` x `h` region whose top-left
    /// full-sample location is `( x0, y0 )`, row-major.
    fn gather(&self, x0: i32, y0: i32, w: usize, h: usize) -> Vec<i32> {
        let mut buf = vec![0i32; w * h];
        for (r, row) in buf.chunks_exact_mut(w).enumerate() {
            self.copy_row(x0, y0 + r as i32, row);
        }
        buf
    }
}

/// The largest tap magnitude the `i16` accumulator of
/// [`simd::filter_taps_narrow`] stays in range for, given the
/// §8.5.3.3.3.2 and §8.5.3.3.3.3 coefficient sets.
///
/// The widest kernel is `[ −1, 4, −11, 40, 40, −11, 4, −1 ]`, whose
/// positive coefficients sum to 88; a partial sum taken in tap order is
/// a subset sum, so `88 · 255 = 22440` bounds it and `−24 · 255 = −6120`
/// bounds it below. Both sit inside `i16`, and neither does at nine bits
/// (`88 · 511 = 44968`).
const NARROW_MAX_SAMPLE: u8 = 255;

/// Copies eight-bit plane samples into a 16-bit buffer for
/// [`simd::filter_taps_narrow`].
///
/// The `i16` accumulator that kernel uses is only in range for samples
/// the caller has already established are eight-bit, which is why
/// [`interp_block`] takes the narrow path only at `bit_depth == 8`; the
/// `debug_assert` is what holds a caller to that.
#[inline]
fn narrow_samples(src: &[i32], dst: &mut [i16]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        debug_assert!(
            (0..=i32::from(NARROW_MAX_SAMPLE)).contains(&s),
            "sample {s} is outside the eight-bit range the narrow interpolation path requires"
        );
        *d = s as i16;
    }
}

/// Errors from the §8.5.3.3 inter-prediction processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterPredError {
    /// A reference plane had a zero width or height.
    EmptyPlane,
    /// A reference plane's sample count did not equal `width * height`.
    PlaneLengthMismatch {
        /// The `width * height` count the plane requires.
        expected: usize,
        /// The element count actually supplied.
        got: usize,
    },
    /// A prediction-block dimension (`nPbW` or `nPbH`) was zero.
    EmptyBlock,
    /// `xFracL` / `yFracL` was outside the `0..=3` quarter-pel range, or
    /// `xFracC` / `yFracC` was outside the `0..=7` eighth-pel range.
    InvalidFraction(i32),
    /// `bitDepth` was outside the 8..=16 range the equations are
    /// dimensioned for.
    InvalidBitDepth(u8),
    /// The two `predSamplesLX` arrays handed to the weighted combine did
    /// not have matching `nPbW * nPbH` lengths.
    ArrayLengthMismatch {
        /// The `nPbW * nPbH` count both arrays require.
        expected: usize,
        /// The element count actually supplied.
        got: usize,
    },
}

impl core::fmt::Display for InterPredError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyPlane => write!(f, "reference plane has zero width or height"),
            Self::PlaneLengthMismatch { expected, got } => {
                write!(
                    f,
                    "reference plane length {got} != width*height = {expected}"
                )
            }
            Self::EmptyBlock => write!(f, "prediction block dimension nPbW/nPbH is zero"),
            Self::InvalidFraction(v) => {
                write!(
                    f,
                    "invalid fractional offset {v} (luma 0..=3, chroma 0..=7)"
                )
            }
            Self::InvalidBitDepth(b) => write!(f, "invalid bitDepth {b} (expected 8..=16)"),
            Self::ArrayLengthMismatch { expected, got } => {
                write!(f, "prediction array length {got} != nPbW*nPbH = {expected}")
            }
        }
    }
}

impl std::error::Error for InterPredError {}

/// `shift1` for luma / chroma interpolation: `Min( 4, BitDepth − 8 )`.
#[inline]
fn interp_shift1(bit_depth: u8) -> i32 {
    core::cmp::min(4, bit_depth as i32 - 8)
}

/// `shift3` for luma / chroma interpolation: `Max( 2, 14 − BitDepth )`.
#[inline]
fn interp_shift3(bit_depth: u8) -> i32 {
    core::cmp::max(2, 14 - bit_depth as i32)
}

// ---------------------------------------------------------------------------
// §8.5.3.3.3.2 — luma sample interpolation
// ---------------------------------------------------------------------------

/// The §8.5.3.3.3.2 horizontal 8-tap luma filters, indexed by `xFracL`.
///
/// Row 0 (`xFracL == 0`) is the identity (the spec leaves the integer
/// sample untouched on the horizontal pass); rows 1/2/3 are the `a`/`b`/`c`
/// kernels of equations 8-224 / 8-225 / 8-226, each over the eight taps
/// `A[−3..4]`.
const LUMA_FILTER: [[i32; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// One separable 8-tap luma sample at fractional offset `( x_frac, y_frac )`,
/// centred on integer location `( x_int, y_int )` (§8.5.3.3.3.2).
///
/// Returns the intermediate sample value (`>> shift1` / `>> shift2`, or
/// `A << shift3` for the full-pel `( 0, 0 )` corner) at the
/// `14 − BitDepthY`-bit internal precision — not yet clipped.
#[inline]
fn interp_luma_sample(
    plane: &RefPlane<'_>,
    x_int: i32,
    y_int: i32,
    x_frac: i32,
    y_frac: i32,
    bit_depth: u8,
) -> i32 {
    let shift1 = interp_shift1(bit_depth);
    let shift3 = interp_shift3(bit_depth);

    // Full-pel: A << shift3 (Table 8-8, xFracL == yFracL == 0).
    if x_frac == 0 && y_frac == 0 {
        return plane.at(x_int, y_int) << shift3;
    }

    let hk = &LUMA_FILTER[x_frac as usize];
    let vk = &LUMA_FILTER[y_frac as usize];

    if y_frac == 0 {
        // Horizontal-only (a / b / c): >> shift1.
        let mut acc = 0i32;
        for (t, &c) in hk.iter().enumerate() {
            acc += c * plane.at(x_int - 3 + t as i32, y_int);
        }
        return acc >> shift1;
    }

    if x_frac == 0 {
        // Vertical-only (d / h / n): >> shift1.
        let mut acc = 0i32;
        for (t, &c) in vk.iter().enumerate() {
            acc += c * plane.at(x_int, y_int - 3 + t as i32);
        }
        return acc >> shift1;
    }

    // Two-dimensional (e/i/p, f/j/q, g/k/r): horizontal pass at >> shift1
    // over rows j = −3..4, then a vertical pass at >> shift2 = 6.
    let mut acc = 0i32;
    for (vt, &cv) in vk.iter().enumerate() {
        let row = y_int - 3 + vt as i32;
        let mut h = 0i32;
        for (ht, &ch) in hk.iter().enumerate() {
            h += ch * plane.at(x_int - 3 + ht as i32, row);
        }
        acc += cv * (h >> shift1);
    }
    acc >> 6
}

/// §8.5.3.3.3.2 — fill an `(nPbW)x(nPbH)` luma prediction block.
///
/// `( x_int, y_int )` is the integer part of the motion-compensated
/// top-left location (`xPb + ( mvLX[0] >> 2 )`, `yPb + ( mvLX[1] >> 2 )`
/// per equations 8-214 / 8-215) and `( x_frac, y_frac )` the
/// quarter-pel remainder (`mvLX[..] & 3`, equations 8-216 / 8-217). The
/// output is row-major, `predSamples[ y * nPbW + x ]`, holding the
/// intermediate-precision values §8.5.3.3.4 consumes.
///
/// # Errors
///
/// [`InterPredError::EmptyBlock`] for a zero block dimension,
/// [`InterPredError::InvalidFraction`] for a fraction outside `0..=3`, and
/// [`InterPredError::InvalidBitDepth`] for a bit depth outside `8..=16`.
// The §8.5.3.3.3.2 location / fraction / dimension / bit-depth inputs are
// each distinct spec quantities; bundling them would obscure the mapping.
#[allow(clippy::too_many_arguments)]
pub fn interp_luma_block(
    plane: &RefPlane<'_>,
    x_int: i32,
    y_int: i32,
    x_frac: i32,
    y_frac: i32,
    n_pb_w: usize,
    n_pb_h: usize,
    bit_depth: u8,
) -> Result<Vec<i32>, InterPredError> {
    interp_luma_block_with(
        simd::detected_isa(),
        plane,
        x_int,
        y_int,
        x_frac,
        y_frac,
        n_pb_w,
        n_pb_h,
        bit_depth,
    )
}

/// [`interp_luma_block`] on an explicitly chosen SIMD backend.
///
/// [`interp_luma_block`] picks [`simd::detected_isa`]; this entry point
/// exists so tests and benchmarks can drive every backend the host
/// supports and compare it against [`Isa::Scalar`]. An `isa` the running
/// CPU does not support degrades to the scalar kernels, and every
/// backend produces bit-identical output.
///
/// # Errors
/// Same contract as [`interp_luma_block`].
#[allow(clippy::too_many_arguments)]
pub fn interp_luma_block_with(
    isa: Isa,
    plane: &RefPlane<'_>,
    x_int: i32,
    y_int: i32,
    x_frac: i32,
    y_frac: i32,
    n_pb_w: usize,
    n_pb_h: usize,
    bit_depth: u8,
) -> Result<Vec<i32>, InterPredError> {
    if n_pb_w == 0 || n_pb_h == 0 {
        return Err(InterPredError::EmptyBlock);
    }
    if !(0..=3).contains(&x_frac) {
        return Err(InterPredError::InvalidFraction(x_frac));
    }
    if !(0..=3).contains(&y_frac) {
        return Err(InterPredError::InvalidFraction(y_frac));
    }
    if !(8..=16).contains(&bit_depth) {
        return Err(InterPredError::InvalidBitDepth(bit_depth));
    }
    Ok(interp_block::<8>(
        isa,
        plane,
        x_int,
        y_int,
        (x_frac != 0).then(|| &LUMA_FILTER[x_frac as usize]),
        (y_frac != 0).then(|| &LUMA_FILTER[y_frac as usize]),
        n_pb_w,
        n_pb_h,
        bit_depth,
    ))
}

/// The separable filter block walk shared by §8.5.3.3.3.2 luma and
/// §8.5.3.3.3.3 chroma interpolation.
///
/// `N` is the tap count — 8 for luma, 4 for chroma — and the leading
/// halo is `N / 2 − 1` samples, matching the `A[−3..4]` / `B[−1..2]` tap
/// windows of the two subclauses. `hk` / `vk` are `None` when that
/// dimension's fractional phase is zero, where the spec leaves the
/// dimension unfiltered; both `None` is the full-pel `A << shift3`
/// corner of Table 8-8 / Table 8-9.
///
/// All four cases hand their inner loop to [`simd::filter_taps`], which
/// evaluates the tap accumulation eight (AVX2) or four (SSE4.1 / NEON)
/// output samples at a time. The two-dimensional case still caches the
/// horizontal pass across the vertical one, so a `w * h` block costs
/// `w * ( h + N − 1 )` horizontal and `w * h` vertical accumulations
/// rather than `N` horizontal dot products per output sample.
#[allow(clippy::too_many_arguments)]
fn interp_block<const N: usize>(
    isa: Isa,
    plane: &RefPlane<'_>,
    x_int: i32,
    y_int: i32,
    hk: Option<&[i32; N]>,
    vk: Option<&[i32; N]>,
    w: usize,
    h: usize,
    bit_depth: u8,
) -> Vec<i32> {
    interp_block_with_width::<N>(
        isa,
        plane,
        x_int,
        y_int,
        hk,
        vk,
        w,
        h,
        bit_depth,
        narrows(isa, bit_depth, w, hk.is_none() && vk.is_some()),
    )
}

/// Whether [`interp_block`] takes the 16-bit accumulation for this
/// backend, `bit_depth`, block width and phase.
///
/// Four conditions, and the narrow path is only faster when all of them
/// hold:
///
/// * **Eight-bit content.** The 16-bit accumulator is only in range
///   there: `shift1` is zero, so the tap accumulation is bounded by
///   [`NARROW_MAX_SAMPLE`] times the coefficient sums. At nine bits and
///   above the pre-shift accumulator overflows `i16`.
/// * **A vector backend.** The whole point is that a vector unit
///   multiplies twice as many `i16` lanes per instruction as `i32`
///   lanes. [`Isa::Scalar`] multiplies one either way, so for it the
///   narrowing pass over the source is cost with nothing behind it.
/// * **A row of at least eight samples.** A shorter row never reaches
///   the 16-bit vector loop at all: `simd::measure_narrow_filter_taps`
///   reads 0.96x at a row of four, where the whole call is the widening
///   remainder plus the cost of having narrowed the source for it.
/// * **The vertical-only phase**, and this is the one that is about the
///   caller rather than the kernel. `measure_narrow_vs_wide_block` reads
///   the vertical-only phase at 1.08x to 1.77x and the horizontal-only
///   and two-dimensional phases at 0.82x to 1.11x, and the difference is
///   not the kernel — it is the same kernel — but who pays to narrow the
///   source. The vertical-only case reaches its source through
///   [`RefPlane::gather`], which materializes a `w x ( h + N − 1 )`
///   buffer either way, so writing it as `i16` instead of `i32` is free
///   and the kernel's 1.85x lands intact. The other two reach theirs
///   through [`RefPlane::row_window`], which **borrows the plane with no
///   copy at all** whenever the window lies inside it — the common case
///   — so narrowing has to add a whole materialized pass the wide path
///   never performs, and that pass costs about what the wider lanes
///   save. The two-dimensional case is worse again, because only its
///   horizontal pass can use the narrowing: its vertical pass multiplies
///   a 16-bit intermediate by a coefficient of up to 58, needs a 32-bit
///   accumulator, and on every vector unit here that is the same lane
///   count the `i32` kernel already issues.
#[inline]
fn narrows(isa: Isa, bit_depth: u8, w: usize, vertical_only: bool) -> bool {
    bit_depth == 8 && isa != Isa::Scalar && w >= 8 && vertical_only
}

/// [`interp_block`] with the 16-bit accumulation forced on or off.
///
/// Separate from [`interp_block`] only so `measure_narrow_vs_wide_block`
/// can A/B the two arms in one process; the decoder always reaches this
/// through [`interp_block`], which decides with [`narrows`].
#[allow(clippy::too_many_arguments)]
fn interp_block_with_width<const N: usize>(
    isa: Isa,
    plane: &RefPlane<'_>,
    x_int: i32,
    y_int: i32,
    hk: Option<&[i32; N]>,
    vk: Option<&[i32; N]>,
    w: usize,
    h: usize,
    bit_depth: u8,
    narrow: bool,
) -> Vec<i32> {
    debug_assert!(!narrow || (bit_depth == 8 && w >= 8));
    let shift1 = interp_shift1(bit_depth);
    let halo = N as i32 / 2 - 1;
    let span = w + N - 1;
    let mut out = vec![0i32; w * h];
    match (hk, vk) {
        // Full-pel (Table 8-8 / 8-9 phase 0, 0): A << shift3.
        (None, None) => {
            let shift3 = interp_shift3(bit_depth);
            let mut scratch = vec![0i32; w];
            for (y, row) in out.chunks_exact_mut(w).enumerate() {
                plane.copy_row(x_int, y_int + y as i32, &mut scratch);
                for (o, &s) in row.iter_mut().zip(scratch.iter()) {
                    *o = s << shift3;
                }
            }
        }
        // Horizontal-only (a/b/c, aX): one source window per output row.
        (Some(hk), None) => {
            if narrow {
                let hk16: [i16; N] = std::array::from_fn(|t| hk[t] as i16);
                let mut win = vec![0i16; span];
                for y in 0..h {
                    plane.row_window_narrow(x_int - halo, span, y_int + y as i32, &mut win);
                    let taps: [&[i16]; N] = std::array::from_fn(|t| &win[t..t + w]);
                    simd::filter_taps_narrow(
                        isa,
                        &taps,
                        &hk16,
                        shift1,
                        &mut out[y * w..(y + 1) * w],
                    );
                }
                return out;
            }
            let mut scratch = vec![0i32; span];
            for y in 0..h {
                let src = plane.row_window(x_int - halo, span, y_int + y as i32, &mut scratch);
                let taps: [&[i32]; N] = std::array::from_fn(|t| &src[t..t + w]);
                simd::filter_taps(isa, &taps, hk, shift1, &mut out[y * w..(y + 1) * w]);
            }
        }
        // Vertical-only (d/h/n, Xa): gather the `h + N − 1` source rows
        // once, then accumulate down the columns — the tap slices are
        // consecutive rows, so the loads stay contiguous.
        (None, Some(vk)) => {
            if narrow {
                let vk16: [i16; N] = std::array::from_fn(|t| vk[t] as i16);
                let src = plane.gather_narrow(x_int, y_int - halo, w, h + N - 1);
                for y in 0..h {
                    let taps: [&[i16]; N] =
                        std::array::from_fn(|t| &src[(y + t) * w..(y + t + 1) * w]);
                    simd::filter_taps_narrow(
                        isa,
                        &taps,
                        &vk16,
                        shift1,
                        &mut out[y * w..(y + 1) * w],
                    );
                }
                return out;
            }
            let src = plane.gather(x_int, y_int - halo, w, h + N - 1);
            for y in 0..h {
                let taps: [&[i32]; N] = std::array::from_fn(|t| &src[(y + t) * w..(y + t + 1) * w]);
                simd::filter_taps(isa, &taps, vk, shift1, &mut out[y * w..(y + 1) * w]);
            }
        }
        // Two-dimensional (e/i/p, f/j/q, g/k/r, XY): horizontal pass over
        // the block plus its vertical halo rows at >> shift1, then the
        // vertical pass at >> shift2 = 6.
        (Some(hk), Some(vk)) => {
            // Issue #309 measured a wrap-around ring that keeps only the
            // `N` horizontal rows the vertical pass has live, instead of
            // all `h + N − 1` of them, and it lost at every block size
            // (0.56x at 8x8 to 0.98x at 64x64): the modular slot index
            // costs more than the intermediate does. At the largest luma
            // block the intermediate is 64 x 71 x 4 = 18 KiB, which sits
            // inside a 128 KiB L1D, so there was never a spill to
            // recover. See `measure_2d_ring_vs_flat`.
            let rows = h + N - 1;
            let mut horizontal = vec![0i32; w * rows];
            if narrow {
                // Only the horizontal pass narrows. Its own output is
                // `i16`-ranged, but the vertical pass multiplies it by a
                // coefficient of up to 58 and needs an `i32`
                // accumulator, which on every vector unit here is the
                // same four (NEON / SSE4.1) or eight (AVX2) lanes per
                // multiply the `i32` kernel already issues — see
                // `measure_narrow_filter_taps`, where the widening
                // formulation reads 1.00x at long rows. So the
                // intermediate stays `i32` and the vertical pass stays
                // on [`simd::filter_taps`] unchanged.
                let hk16: [i16; N] = std::array::from_fn(|t| hk[t] as i16);
                let mut win = vec![0i16; span];
                for row in 0..rows {
                    plane.row_window_narrow(
                        x_int - halo,
                        span,
                        y_int - halo + row as i32,
                        &mut win,
                    );
                    let taps: [&[i16]; N] = std::array::from_fn(|t| &win[t..t + w]);
                    simd::filter_taps_narrow(
                        isa,
                        &taps,
                        &hk16,
                        shift1,
                        &mut horizontal[row * w..(row + 1) * w],
                    );
                }
            } else {
                let mut scratch = vec![0i32; span];
                for row in 0..rows {
                    let src = plane.row_window(
                        x_int - halo,
                        span,
                        y_int - halo + row as i32,
                        &mut scratch,
                    );
                    let taps: [&[i32]; N] = std::array::from_fn(|t| &src[t..t + w]);
                    simd::filter_taps(
                        isa,
                        &taps,
                        hk,
                        shift1,
                        &mut horizontal[row * w..(row + 1) * w],
                    );
                }
            }
            for y in 0..h {
                let taps: [&[i32]; N] =
                    std::array::from_fn(|t| &horizontal[(y + t) * w..(y + t + 1) * w]);
                simd::filter_taps(isa, &taps, vk, 6, &mut out[y * w..(y + 1) * w]);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// §8.5.3.3.3.3 — chroma sample interpolation
// ---------------------------------------------------------------------------

/// The §8.5.3.3.3.3 4-tap chroma filters, indexed by the eighth-pel phase.
///
/// Row 0 (phase 0) is the identity; rows 1..7 are the `ab`/`ac`/`ad`/`ae`/
/// `af`/`ag`/`ah` kernels of equations 8-241..8-247, each over the four
/// taps `B[−1..2]`.
const CHROMA_FILTER: [[i32; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// One separable 4-tap chroma sample at eighth-pel offset
/// `( x_frac, y_frac )`, centred on integer location `( x_int, y_int )`
/// (§8.5.3.3.3.3). Returns the intermediate-precision value (not clipped).
#[inline]
fn interp_chroma_sample(
    plane: &RefPlane<'_>,
    x_int: i32,
    y_int: i32,
    x_frac: i32,
    y_frac: i32,
    bit_depth: u8,
) -> i32 {
    let shift1 = interp_shift1(bit_depth);
    let shift3 = interp_shift3(bit_depth);

    if x_frac == 0 && y_frac == 0 {
        return plane.at(x_int, y_int) << shift3;
    }

    let hk = &CHROMA_FILTER[x_frac as usize];
    let vk = &CHROMA_FILTER[y_frac as usize];

    if y_frac == 0 {
        // Horizontal-only (aX, equations 8-241..8-247): >> shift1.
        let mut acc = 0i32;
        for (t, &c) in hk.iter().enumerate() {
            acc += c * plane.at(x_int - 1 + t as i32, y_int);
        }
        return acc >> shift1;
    }

    if x_frac == 0 {
        // Vertical-only (Xa, equations 8-248..8-254): >> shift1.
        let mut acc = 0i32;
        for (t, &c) in vk.iter().enumerate() {
            acc += c * plane.at(x_int, y_int - 1 + t as i32);
        }
        return acc >> shift1;
    }

    // Two-dimensional (XY, equations 8-255..8-261): horizontal pass at
    // >> shift1 over rows i = −1..2, then a vertical pass at >> shift2 = 6.
    let mut acc = 0i32;
    for (vt, &cv) in vk.iter().enumerate() {
        let row = y_int - 1 + vt as i32;
        let mut h = 0i32;
        for (ht, &ch) in hk.iter().enumerate() {
            h += ch * plane.at(x_int - 1 + ht as i32, row);
        }
        acc += cv * (h >> shift1);
    }
    acc >> 6
}

/// §8.5.3.3.3.3 — fill an `(nPbW / SubWidthC)x(nPbH / SubHeightC)` chroma
/// prediction block.
///
/// `( x_int, y_int )` is the integer chroma location and
/// `( x_frac, y_frac )` the eighth-pel remainder (`mvCLX[..] & 7`,
/// equations 8-220 / 8-221). `block_w` / `block_h` are the chroma block
/// dimensions. Output is row-major intermediate-precision values.
///
/// # Errors
///
/// [`InterPredError::EmptyBlock`] for a zero block dimension,
/// [`InterPredError::InvalidFraction`] for a fraction outside `0..=7`, and
/// [`InterPredError::InvalidBitDepth`] for a bit depth outside `8..=16`.
// The §8.5.3.3.3.3 location / fraction / dimension / bit-depth inputs are
// each distinct spec quantities; bundling them would obscure the mapping.
#[allow(clippy::too_many_arguments)]
pub fn interp_chroma_block(
    plane: &RefPlane<'_>,
    x_int: i32,
    y_int: i32,
    x_frac: i32,
    y_frac: i32,
    block_w: usize,
    block_h: usize,
    bit_depth: u8,
) -> Result<Vec<i32>, InterPredError> {
    interp_chroma_block_with(
        simd::detected_isa(),
        plane,
        x_int,
        y_int,
        x_frac,
        y_frac,
        block_w,
        block_h,
        bit_depth,
    )
}

/// [`interp_chroma_block`] on an explicitly chosen SIMD backend; see
/// [`interp_luma_block_with`].
///
/// # Errors
/// Same contract as [`interp_chroma_block`].
#[allow(clippy::too_many_arguments)]
pub fn interp_chroma_block_with(
    isa: Isa,
    plane: &RefPlane<'_>,
    x_int: i32,
    y_int: i32,
    x_frac: i32,
    y_frac: i32,
    block_w: usize,
    block_h: usize,
    bit_depth: u8,
) -> Result<Vec<i32>, InterPredError> {
    if block_w == 0 || block_h == 0 {
        return Err(InterPredError::EmptyBlock);
    }
    if !(0..=7).contains(&x_frac) {
        return Err(InterPredError::InvalidFraction(x_frac));
    }
    if !(0..=7).contains(&y_frac) {
        return Err(InterPredError::InvalidFraction(y_frac));
    }
    if !(8..=16).contains(&bit_depth) {
        return Err(InterPredError::InvalidBitDepth(bit_depth));
    }
    Ok(interp_block::<4>(
        isa,
        plane,
        x_int,
        y_int,
        (x_frac != 0).then(|| &CHROMA_FILTER[x_frac as usize]),
        (y_frac != 0).then(|| &CHROMA_FILTER[y_frac as usize]),
        block_w,
        block_h,
        bit_depth,
    ))
}

// ---------------------------------------------------------------------------
// §8.5.3.3.4.2 — default weighted sample prediction
// ---------------------------------------------------------------------------

/// §8.5.3.3.4.2 — combine the L0 / L1 intermediate prediction arrays into
/// the final `(nPbW)x(nPbH)` prediction samples (the
/// `weighted_pred_flag == 0` path, equations 8-262..8-264).
///
/// `pred_l0` / `pred_l1` are the intermediate-precision arrays produced by
/// [`interp_luma_block`] / [`interp_chroma_block`]; `pred_flag_l0` /
/// `pred_flag_l1` are the §8.5.3.2.1 prediction-list utilisation flags. At
/// least one flag must be set (the spec only invokes this process for a
/// predicted block). Output is the clipped `[0, (1 << bitDepth) − 1]`
/// sample array.
///
/// Unused arrays may be empty when their `pred_flag` is `false`; only the
/// array(s) whose flag is set are read and length-checked.
///
/// # Errors
///
/// [`InterPredError::EmptyBlock`] for a zero block dimension,
/// [`InterPredError::InvalidBitDepth`] for a bit depth outside `8..=16`,
/// [`InterPredError::EmptyPlane`] (re-used as "no list selected") when
/// both flags are `false`, and [`InterPredError::ArrayLengthMismatch`]
/// when a selected array is not `nPbW * nPbH` long.
pub fn default_weighted_pred(
    pred_l0: &[i32],
    pred_l1: &[i32],
    pred_flag_l0: bool,
    pred_flag_l1: bool,
    n_pb_w: usize,
    n_pb_h: usize,
    bit_depth: u8,
) -> Result<Vec<i32>, InterPredError> {
    default_weighted_pred_with(
        simd::detected_isa(),
        pred_l0,
        pred_l1,
        pred_flag_l0,
        pred_flag_l1,
        n_pb_w,
        n_pb_h,
        bit_depth,
    )
}

/// [`default_weighted_pred`] on an explicitly chosen SIMD backend; see
/// [`interp_luma_block_with`].
///
/// # Errors
/// Same contract as [`default_weighted_pred`].
#[allow(clippy::too_many_arguments)]
pub fn default_weighted_pred_with(
    isa: Isa,
    pred_l0: &[i32],
    pred_l1: &[i32],
    pred_flag_l0: bool,
    pred_flag_l1: bool,
    n_pb_w: usize,
    n_pb_h: usize,
    bit_depth: u8,
) -> Result<Vec<i32>, InterPredError> {
    if n_pb_w == 0 || n_pb_h == 0 {
        return Err(InterPredError::EmptyBlock);
    }
    if !(8..=16).contains(&bit_depth) {
        return Err(InterPredError::InvalidBitDepth(bit_depth));
    }
    let count = n_pb_w * n_pb_h;
    if pred_flag_l0 && pred_l0.len() != count {
        return Err(InterPredError::ArrayLengthMismatch {
            expected: count,
            got: pred_l0.len(),
        });
    }
    if pred_flag_l1 && pred_l1.len() != count {
        return Err(InterPredError::ArrayLengthMismatch {
            expected: count,
            got: pred_l1.len(),
        });
    }

    let shift1 = core::cmp::max(2, 14 - bit_depth as i32);
    let shift2 = core::cmp::max(3, 15 - bit_depth as i32);
    let offset1 = 1i32 << (shift1 - 1);
    let offset2 = 1i32 << (shift2 - 1);
    let max_val = (1i32 << bit_depth) - 1;

    // Every case is `Clip3( 0, max, ( Σ 1 · predLX + offset ) >> shift )`,
    // which is exactly the shape `simd::combine_weighted` vectorizes. The
    // intermediate samples and the offsets both stay well inside `i32`
    // for every supported bit depth, so no widening is needed.
    let mut out = vec![0i32; count];
    match (pred_flag_l0, pred_flag_l1) {
        // Uni-predictive from L0 (equation 8-262).
        (true, false) => {
            simd::combine_weighted(isa, &[pred_l0], &[1], offset1, shift1, 0, max_val, &mut out)
        }
        // Uni-predictive from L1 (equation 8-263).
        (false, true) => {
            simd::combine_weighted(isa, &[pred_l1], &[1], offset1, shift1, 0, max_val, &mut out)
        }
        // Bi-predictive (equation 8-264).
        (true, true) => simd::combine_weighted(
            isa,
            &[pred_l0, pred_l1],
            &[1, 1],
            offset2,
            shift2,
            0,
            max_val,
            &mut out,
        ),
        (false, false) => return Err(InterPredError::EmptyPlane),
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// §8.5.3.3.4.3 — explicit weighted sample prediction
// ---------------------------------------------------------------------------

/// §8.5.3.3.4.3 — combine the L0 / L1 intermediate prediction arrays with
/// per-reference explicit weights and offsets (the `weightedPredFlag == 1`
/// path, equations 8-265..8-277).
///
/// `log2_weight_denom` is the component's raw denominator
/// (`luma_log2_weight_denom` for luma, `ChromaLog2WeightDenom` for
/// chroma); the process adds `shift1 = Max(2, 14 − bitDepth)` internally
/// (equations 8-265 / 8-270). `w0` / `w1` are `LumaWeightLX[refIdxLX]` or
/// `ChromaWeightLX[refIdxLX][cIdx−1]`; `o0` / `o1` are the offsets
/// **already scaled** by `WpOffsetBdShiftY` / `WpOffsetBdShiftC`
/// (equations 8-268 / 8-269 / 8-273 / 8-274). An unused list's weight /
/// offset values are ignored.
///
/// # Errors
/// Same contract as [`default_weighted_pred`].
#[allow(clippy::too_many_arguments)]
pub fn explicit_weighted_pred(
    pred_l0: &[i32],
    pred_l1: &[i32],
    pred_flag_l0: bool,
    pred_flag_l1: bool,
    n_pb_w: usize,
    n_pb_h: usize,
    log2_weight_denom: u8,
    w0: i32,
    o0: i32,
    w1: i32,
    o1: i32,
    bit_depth: u8,
) -> Result<Vec<i32>, InterPredError> {
    explicit_weighted_pred_with(
        simd::detected_isa(),
        pred_l0,
        pred_l1,
        pred_flag_l0,
        pred_flag_l1,
        n_pb_w,
        n_pb_h,
        log2_weight_denom,
        w0,
        o0,
        w1,
        o1,
        bit_depth,
    )
}

/// [`explicit_weighted_pred`] on an explicitly chosen SIMD backend; see
/// [`interp_luma_block_with`].
///
/// # Errors
/// Same contract as [`explicit_weighted_pred`].
#[allow(clippy::too_many_arguments)]
pub fn explicit_weighted_pred_with(
    isa: Isa,
    pred_l0: &[i32],
    pred_l1: &[i32],
    pred_flag_l0: bool,
    pred_flag_l1: bool,
    n_pb_w: usize,
    n_pb_h: usize,
    log2_weight_denom: u8,
    w0: i32,
    o0: i32,
    w1: i32,
    o1: i32,
    bit_depth: u8,
) -> Result<Vec<i32>, InterPredError> {
    if n_pb_w == 0 || n_pb_h == 0 {
        return Err(InterPredError::EmptyBlock);
    }
    if !(8..=16).contains(&bit_depth) {
        return Err(InterPredError::InvalidBitDepth(bit_depth));
    }
    let count = n_pb_w * n_pb_h;
    if pred_flag_l0 && pred_l0.len() != count {
        return Err(InterPredError::ArrayLengthMismatch {
            expected: count,
            got: pred_l0.len(),
        });
    }
    if pred_flag_l1 && pred_l1.len() != count {
        return Err(InterPredError::ArrayLengthMismatch {
            expected: count,
            got: pred_l1.len(),
        });
    }

    // Equations 8-265 / 8-270: log2Wd = log2WeightDenom + shift1.
    let shift1 = core::cmp::max(2, 14 - i64::from(bit_depth));
    let log2_wd = i64::from(log2_weight_denom) + shift1;
    let max_val = (1i64 << bit_depth) - 1;
    let (w0, o0, w1, o1) = (i64::from(w0), i64::from(o0), i64::from(w1), i64::from(o1));

    let mut out = vec![0i32; count];
    match (pred_flag_l0, pred_flag_l1) {
        // Uni-predictive from L0 (equation 8-275).
        (true, false) => explicit_uni(isa, pred_l0, w0, o0, log2_wd, max_val, &mut out),
        // Uni-predictive from L1 (equation 8-276).
        (false, true) => explicit_uni(isa, pred_l1, w1, o1, log2_wd, max_val, &mut out),
        // Bi-predictive (equation 8-277).
        (true, true) => explicit_bi(
            isa,
            (pred_l0, w0, o0),
            (pred_l1, w1, o1),
            log2_wd,
            max_val,
            &mut out,
        ),
        (false, false) => return Err(InterPredError::EmptyPlane),
    }
    Ok(out)
}

/// The largest `| v |` in `values`, widened to `i64`.
fn max_abs(values: &[i32]) -> i64 {
    values
        .iter()
        .fold(0i64, |max, &v| max.max(i64::from(v).abs()))
}

/// §8.5.3.3.4.3 uni-predictive combine (equations 8-275 / 8-276).
///
/// The spec arithmetic is `i64`, but for every legal HEVC weight, offset
/// and intermediate sample the whole expression fits in `i32` — so when
/// a bound derived from the block's actual peak magnitude proves it,
/// the combine runs on the vectorized `i32` kernel, and otherwise it
/// falls back to the `i64` scalar loop. Both produce the same values.
fn explicit_uni(
    isa: Isa,
    pred: &[i32],
    w: i64,
    o: i64,
    log2_wd: i64,
    max_val: i64,
    out: &mut [i32],
) {
    let round = 1i64 << (log2_wd - 1);
    let peak = max_abs(pred) * w.abs() + round;
    let limit = i64::from(i32::MAX);
    if log2_wd < 31 && peak <= limit && (peak >> log2_wd) + o.abs() < limit {
        simd::combine_weighted(
            isa,
            &[pred],
            &[w as i32],
            round as i32,
            log2_wd as i32,
            o as i32,
            max_val as i32,
            out,
        );
        return;
    }
    for (o_out, &p) in out.iter_mut().zip(pred) {
        *o_out = ((((i64::from(p) * w + round) >> log2_wd) + o).clamp(0, max_val)) as i32;
    }
}

/// §8.5.3.3.4.3 bi-predictive combine (equation 8-277), with the same
/// `i32`-when-provable / `i64`-otherwise split as [`explicit_uni`].
/// Each list is passed as its `( predSamplesLX, wX, oX )` triple.
fn explicit_bi(
    isa: Isa,
    l0: (&[i32], i64, i64),
    l1: (&[i32], i64, i64),
    log2_wd: i64,
    max_val: i64,
    out: &mut [i32],
) {
    let (p0, w0, o0) = l0;
    let (p1, w1, o1) = l1;
    let round = (o0 + o1 + 1) << log2_wd;
    let shift = log2_wd + 1;
    let peak = max_abs(p0) * w0.abs() + max_abs(p1) * w1.abs() + round.abs();
    if shift < 31 && peak <= i64::from(i32::MAX) {
        simd::combine_weighted(
            isa,
            &[p0, p1],
            &[w0 as i32, w1 as i32],
            round as i32,
            shift as i32,
            0,
            max_val as i32,
            out,
        );
        return;
    }
    for ((o_out, &a), &b) in out.iter_mut().zip(p0).zip(p1) {
        *o_out =
            (((i64::from(a) * w0 + i64::from(b) * w1 + round) >> shift).clamp(0, max_val)) as i32;
    }
}

/// One reference list's §8.5.3.3.4.3 weights / offsets for one PU,
/// resolved for its `refIdxLX`: `w` is `LumaWeightLX[refIdx]` /
/// `ChromaWeightLX[refIdx][j]`; `o` is the corresponding offset already
/// scaled by `WpOffsetBdShiftY` / `WpOffsetBdShiftC` (equations
/// 8-268 / 8-269 / 8-273 / 8-274).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WpListWeights {
    /// `LumaWeightLX[refIdxLX]` (equations 8-266 / 8-267).
    pub w_luma: i32,
    /// `luma_offset_lX[refIdxLX] << WpOffsetBdShiftY`.
    pub o_luma: i32,
    /// `ChromaWeightLX[refIdxLX][0]` (Cb).
    pub w_cb: i32,
    /// `ChromaOffsetLX[refIdxLX][0] << WpOffsetBdShiftC` (Cb).
    pub o_cb: i32,
    /// `ChromaWeightLX[refIdxLX][1]` (Cr).
    pub w_cr: i32,
    /// `ChromaOffsetLX[refIdxLX][1] << WpOffsetBdShiftC` (Cr).
    pub o_cr: i32,
}

/// The complete §8.5.3.3.4.3 inputs for one PU's weighted combine: the
/// two log2 denominators plus each list's per-component weights, already
/// resolved for the PU's reference indices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PuWeights {
    /// `luma_log2_weight_denom` (§7.4.7.3).
    pub luma_log2_weight_denom: u8,
    /// `ChromaLog2WeightDenom` (§7.4.7.3).
    pub chroma_log2_weight_denom: u8,
    /// L0 weights (ignored when `predFlagL0 == 0`).
    pub l0: WpListWeights,
    /// L1 weights (ignored when `predFlagL1 == 0`).
    pub l1: WpListWeights,
}

// ---------------------------------------------------------------------------
// §8.5.3.3.1 — inter prediction sample block-walk driver
// ---------------------------------------------------------------------------

/// A `[mvLX[0], mvLX[1]]` motion vector in quarter-luma-sample units
/// (the §8.5.3 luma MV) — the integer / fractional split of equations
/// 8-214..8-217 is performed by the driver.
pub type MotionVector = [i32; 2];

/// One reference list's prediction inputs for a prediction unit: the
/// reference picture planes selected by §8.5.3.3.2 plus the luma motion
/// vector mvLX (quarter-pel) and chroma motion vector mvCLX (eighth-pel,
/// already derived per §8.5.3.2.10). `pred_flag == false` means the list
/// is not used and the planes / vectors are ignored.
#[derive(Debug, Clone, Copy)]
pub struct ListPrediction<'a> {
    /// `predFlagLX` — whether reference list X contributes to this PU.
    pub pred_flag: bool,
    /// `refPicLXL` — the §8.5.3.3.2 luma reference plane.
    pub luma: RefPlane<'a>,
    /// `refPicLXCb` — the §8.5.3.3.2 Cb reference plane (ignored when
    /// `chroma_array_type == 0`).
    pub cb: Option<RefPlane<'a>>,
    /// `refPicLXCr` — the §8.5.3.3.2 Cr reference plane.
    pub cr: Option<RefPlane<'a>>,
    /// `mvLX` in quarter-luma-sample units (equations 8-214..8-217).
    pub mv_l: MotionVector,
    /// `mvCLX` in eighth-chroma-sample units (equations 8-218..8-221,
    /// derived from `mvLX` by §8.5.3.2.10).
    pub mv_c: MotionVector,
}

/// The reconstructed prediction-sample planes for one inter prediction
/// block, produced by [`predict_inter_pu`]. Each plane is row-major and
/// holds the final clipped `[0, (1 << bitDepth) − 1]` prediction samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterPrediction {
    /// `predSamplesL` — `nPbW * nPbH` luma prediction samples.
    pub luma: Vec<i32>,
    /// `predSamplesCb` — `(nPbW / SubWidthC) * (nPbH / SubHeightC)` Cb
    /// prediction samples, empty when `chroma_array_type == 0`.
    pub cb: Vec<i32>,
    /// `predSamplesCr` — Cr prediction samples, empty when monochrome.
    pub cr: Vec<i32>,
}

/// §8.5.3.3.1 — geometry / format inputs constant for one PU prediction.
#[derive(Debug, Clone, Copy)]
pub struct InterPredGeometry {
    /// `xPb = xCb + xBl` (equation 8-212) — the PU's luma top-left x.
    pub x_pb: i32,
    /// `yPb = yCb + yBl` (equation 8-213) — the PU's luma top-left y.
    pub y_pb: i32,
    /// `nPbW` — luma prediction-block width.
    pub n_pb_w: usize,
    /// `nPbH` — luma prediction-block height.
    pub n_pb_h: usize,
    /// `ChromaArrayType` (0 = monochrome, 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4).
    pub chroma_array_type: u8,
    /// `BitDepthY`.
    pub bit_depth_luma: u8,
    /// `BitDepthC`.
    pub bit_depth_chroma: u8,
}

/// `(SubWidthC, SubHeightC)` from Table 6-1 (mirrors
/// [`crate::hevc::engine::picture::sub_wh_c`] without the cross-module dependency).
#[inline]
fn sub_wh_c_local(chroma_array_type: u8) -> (i32, i32) {
    match chroma_array_type {
        1 => (2, 2),
        2 => (2, 1),
        3 => (1, 1),
        _ => (2, 2),
    }
}

/// Fill one list's intermediate luma prediction array for a PU
/// (§8.5.3.3.3.1 equations 8-214..8-217 + the §8.5.3.3.3.2 per-sample
/// interpolation). `xPb`/`yPb` are added inside the integer split.
fn list_luma_pred(
    list: &ListPrediction<'_>,
    geom: &InterPredGeometry,
) -> Result<Vec<i32>, InterPredError> {
    // §8.5.3.3.3.1: xIntL = xPb + (mvLX[0] >> 2), xFracL = mvLX[0] & 3.
    let x_int = geom.x_pb + (list.mv_l[0] >> 2);
    let y_int = geom.y_pb + (list.mv_l[1] >> 2);
    let x_frac = list.mv_l[0] & 3;
    let y_frac = list.mv_l[1] & 3;
    interp_luma_block(
        &list.luma,
        x_int,
        y_int,
        x_frac,
        y_frac,
        geom.n_pb_w,
        geom.n_pb_h,
        geom.bit_depth_luma,
    )
}

/// Fill one list's intermediate chroma prediction array for a PU
/// (§8.5.3.3.3.1 equations 8-218..8-221 + §8.5.3.3.3.3 interpolation).
fn list_chroma_pred(
    plane: &RefPlane<'_>,
    list: &ListPrediction<'_>,
    geom: &InterPredGeometry,
    sub_w: i32,
    sub_h: i32,
) -> Result<Vec<i32>, InterPredError> {
    // §8.5.3.3.3.1: xIntC = (xPb / SubWidthC) + (mvCLX[0] >> 3),
    //               xFracC = mvCLX[0] & 7.
    let x_int = geom.x_pb / sub_w + (list.mv_c[0] >> 3);
    let y_int = geom.y_pb / sub_h + (list.mv_c[1] >> 3);
    let x_frac = list.mv_c[0] & 7;
    let y_frac = list.mv_c[1] & 7;
    interp_chroma_block(
        plane,
        x_int,
        y_int,
        x_frac,
        y_frac,
        geom.n_pb_w / sub_w as usize,
        geom.n_pb_h / sub_h as usize,
        geom.bit_depth_chroma,
    )
}

/// §8.5.3.3.1 — drive the inter-prediction sample process for one
/// prediction block: split each used list's motion vector into its
/// integer / fractional parts, run the §8.5.3.3.3 fractional-sample
/// interpolation over the whole block for luma and (when chroma is
/// present) Cb / Cr, then combine the L0 / L1 intermediate arrays with
/// the §8.5.3.3.4.2 default weighted sample prediction.
///
/// This is the `weightedPredFlag == 0` path of §8.5.3.3.4.1; see
/// [`predict_inter_pu_weighted`] for the full dispatch including the
/// §8.5.3.3.4.3 explicit-weighting path.
///
/// # Errors
///
/// [`InterPredError::EmptyBlock`] for a zero PU dimension,
/// [`InterPredError::EmptyPlane`] when neither list is used, and the
/// interpolation / combine errors propagated from the primitives.
pub fn predict_inter_pu(
    l0: &ListPrediction<'_>,
    l1: &ListPrediction<'_>,
    geom: &InterPredGeometry,
) -> Result<InterPrediction, InterPredError> {
    predict_inter_pu_weighted(l0, l1, geom, None)
}

/// §8.5.3.3.1 + §8.5.3.3.4.1 — as [`predict_inter_pu`], with the
/// weighted-sample-prediction dispatch: `weights == None` is the
/// `weightedPredFlag == 0` default combine (§8.5.3.3.4.2);
/// `weights == Some(..)` is the explicit per-reference combine
/// (§8.5.3.3.4.3), with the PU's `refIdxLX`-resolved weights carried in
/// the [`PuWeights`].
///
/// # Errors
/// Same contract as [`predict_inter_pu`].
pub fn predict_inter_pu_weighted(
    l0: &ListPrediction<'_>,
    l1: &ListPrediction<'_>,
    geom: &InterPredGeometry,
    weights: Option<&PuWeights>,
) -> Result<InterPrediction, InterPredError> {
    if geom.n_pb_w == 0 || geom.n_pb_h == 0 {
        return Err(InterPredError::EmptyBlock);
    }
    if !l0.pred_flag && !l1.pred_flag {
        return Err(InterPredError::EmptyPlane);
    }

    // Luma: interpolate each used list, then combine per §8.5.3.3.4.1.
    let pred_l0_luma = if l0.pred_flag {
        list_luma_pred(l0, geom)?
    } else {
        Vec::new()
    };
    let pred_l1_luma = if l1.pred_flag {
        list_luma_pred(l1, geom)?
    } else {
        Vec::new()
    };
    let luma = match weights {
        None => default_weighted_pred(
            &pred_l0_luma,
            &pred_l1_luma,
            l0.pred_flag,
            l1.pred_flag,
            geom.n_pb_w,
            geom.n_pb_h,
            geom.bit_depth_luma,
        )?,
        Some(wp) => explicit_weighted_pred(
            &pred_l0_luma,
            &pred_l1_luma,
            l0.pred_flag,
            l1.pred_flag,
            geom.n_pb_w,
            geom.n_pb_h,
            wp.luma_log2_weight_denom,
            wp.l0.w_luma,
            wp.l0.o_luma,
            wp.l1.w_luma,
            wp.l1.o_luma,
            geom.bit_depth_luma,
        )?,
    };

    let (mut cb, mut cr) = (Vec::new(), Vec::new());
    if geom.chroma_array_type != 0 {
        let (sub_w, sub_h) = sub_wh_c_local(geom.chroma_array_type);
        let wp_cb = weights.map(|wp| {
            (
                wp.chroma_log2_weight_denom,
                wp.l0.w_cb,
                wp.l0.o_cb,
                wp.l1.w_cb,
                wp.l1.o_cb,
            )
        });
        let wp_cr = weights.map(|wp| {
            (
                wp.chroma_log2_weight_denom,
                wp.l0.w_cr,
                wp.l0.o_cr,
                wp.l1.w_cr,
                wp.l1.o_cr,
            )
        });
        cb = combine_chroma(l0, l1, geom, sub_w, sub_h, wp_cb, |lp| lp.cb)?;
        cr = combine_chroma(l0, l1, geom, sub_w, sub_h, wp_cr, |lp| lp.cr)?;
    }

    Ok(InterPrediction { luma, cb, cr })
}

/// Interpolate and §8.5.3.3.4-combine one chroma component for a PU.
/// `select` picks the Cb or Cr reference plane from a [`ListPrediction`];
/// `wp` is `Some((ChromaLog2WeightDenom, w0, o0, w1, o1))` for the
/// §8.5.3.3.4.3 explicit path, `None` for the §8.5.3.3.4.2 default.
fn combine_chroma<'a>(
    l0: &ListPrediction<'a>,
    l1: &ListPrediction<'a>,
    geom: &InterPredGeometry,
    sub_w: i32,
    sub_h: i32,
    wp: Option<(u8, i32, i32, i32, i32)>,
    select: impl Fn(&ListPrediction<'a>) -> Option<RefPlane<'a>>,
) -> Result<Vec<i32>, InterPredError> {
    let cw = geom.n_pb_w / sub_w as usize;
    let ch = geom.n_pb_h / sub_h as usize;
    let p0 = if l0.pred_flag {
        let plane = select(l0).ok_or(InterPredError::EmptyPlane)?;
        list_chroma_pred(&plane, l0, geom, sub_w, sub_h)?
    } else {
        Vec::new()
    };
    let p1 = if l1.pred_flag {
        let plane = select(l1).ok_or(InterPredError::EmptyPlane)?;
        list_chroma_pred(&plane, l1, geom, sub_w, sub_h)?
    } else {
        Vec::new()
    };
    match wp {
        None => default_weighted_pred(
            &p0,
            &p1,
            l0.pred_flag,
            l1.pred_flag,
            cw,
            ch,
            geom.bit_depth_chroma,
        ),
        Some((denom, w0, o0, w1, o1)) => explicit_weighted_pred(
            &p0,
            &p1,
            l0.pred_flag,
            l1.pred_flag,
            cw,
            ch,
            denom,
            w0,
            o0,
            w1,
            o1,
            geom.bit_depth_chroma,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat plane interpolates to the constant sample value scaled to
    /// the internal precision: every tap kernel sums to 64, so
    /// `64 * v >> shift1` at 8-bit (`shift1 == 0`) is `64 * v`, matching
    /// the `v << shift3` (`shift3 == 6`) full-pel value.
    #[test]
    fn luma_flat_plane_constant() {
        let plane_samples = vec![100i32; 16 * 16];
        let plane = RefPlane::new(&plane_samples, 16, 16).unwrap();
        for xf in 0..=3 {
            for yf in 0..=3 {
                let blk = interp_luma_block(&plane, 5, 5, xf, yf, 4, 4, 8).unwrap();
                for &s in &blk {
                    assert_eq!(s, 100 << 6, "xf={xf} yf={yf}");
                }
            }
        }
    }

    #[test]
    fn separable_luma_block_matches_per_sample_reference() {
        let samples = (0..24 * 20)
            .map(|index| (index * 37 + index / 11) & 0xff)
            .collect::<Vec<_>>();
        let plane = RefPlane::new(&samples, 24, 20).unwrap();
        for x_frac in 1..=3 {
            for y_frac in 1..=3 {
                let actual = interp_luma_block(&plane, -1, 2, x_frac, y_frac, 9, 7, 8).unwrap();
                let expected = (0..7)
                    .flat_map(|y| {
                        (0..9).map(move |x| {
                            interp_luma_sample(&plane, -1 + x, 2 + y, x_frac, y_frac, 8)
                        })
                    })
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "x_frac={x_frac} y_frac={y_frac}");
            }
        }
    }

    #[test]
    fn separable_chroma_block_matches_per_sample_reference() {
        let samples = (0..13 * 11)
            .map(|index| (index * 53 + index / 7) & 0xff)
            .collect::<Vec<_>>();
        let plane = RefPlane::new(&samples, 13, 11).unwrap();
        for x_frac in 1..=7 {
            for y_frac in 1..=7 {
                let actual = interp_chroma_block(&plane, 3, -1, x_frac, y_frac, 6, 5, 8).unwrap();
                let expected = (0..5)
                    .flat_map(|y| {
                        (0..6).map(move |x| {
                            interp_chroma_sample(&plane, 3 + x, -1 + y, x_frac, y_frac, 8)
                        })
                    })
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "x_frac={x_frac} y_frac={y_frac}");
            }
        }
    }

    /// Full-pel luma is `A << shift3`; at 8-bit `shift3 == 6`.
    #[test]
    fn luma_full_pel_shift3() {
        let mut s = vec![0i32; 8 * 8];
        for (i, v) in s.iter_mut().enumerate() {
            *v = i as i32;
        }
        let plane = RefPlane::new(&s, 8, 8).unwrap();
        let blk = interp_luma_block(&plane, 2, 3, 0, 0, 2, 2, 8).unwrap();
        // predSamples[0][0] = A(2,3) << 6 = (3*8 + 2) << 6 = 26 << 6.
        assert_eq!(blk[0], (3 * 8 + 2) << 6);
        // predSamples[1][1] = A(3,4) << 6 = (4*8 + 3) << 6 = 35 << 6.
        assert_eq!(blk[3], (4 * 8 + 3) << 6);
    }

    /// The luma `a` kernel (xFracL == 1) on a known column reproduces
    /// equation 8-224 hand-computed.
    #[test]
    fn luma_a_kernel_hand_value() {
        // A row of samples; pick a center so the 8 taps land inside.
        // Coords x = −3..4 around x_int = 5 -> indices 2..9.
        let mut s = vec![0i32; 16];
        let vals = [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        s.copy_from_slice(&vals);
        let plane = RefPlane::new(&s, 16, 1).unwrap();
        let blk = interp_luma_block(&plane, 5, 0, 1, 0, 1, 1, 8).unwrap();
        // a = −A−3 + 4A−2 − 10A−1 + 58A0 + 17A1 − 5A2 + A3, >> shift1(=0).
        // A−3..A3 = x=2..8 = 30,40,50,60,70,80,90.
        let expected = -30 + 4 * 40 - 10 * 50 + 58 * 60 + 17 * 70 - 5 * 80 + 90;
        assert_eq!(blk[0], expected);
    }

    /// Chroma flat plane interpolates to the constant value (each 4-tap
    /// kernel sums to 64) for all 8x8 eighth-pel phases.
    #[test]
    fn chroma_flat_plane_constant() {
        let plane_samples = vec![77i32; 12 * 12];
        let plane = RefPlane::new(&plane_samples, 12, 12).unwrap();
        for xf in 0..=7 {
            for yf in 0..=7 {
                let blk = interp_chroma_block(&plane, 4, 4, xf, yf, 3, 3, 8).unwrap();
                for &s in &blk {
                    assert_eq!(s, 77 << 6, "xf={xf} yf={yf}");
                }
            }
        }
    }

    /// Chroma `ab` kernel (xFracC == 1) reproduces equation 8-241.
    #[test]
    fn chroma_ab_kernel_hand_value() {
        let s = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let plane = RefPlane::new(&s, 8, 1).unwrap();
        // x_int = 3, taps x = −1..2 -> x=2,3,4,5 = 30,40,50,60.
        let blk = interp_chroma_block(&plane, 3, 0, 1, 0, 1, 1, 8).unwrap();
        // ab = −2B−1 + 58B0 + 10B1 − 2B2, >> shift1(=0).
        let expected = -2 * 30 + 58 * 40 + 10 * 50 - 2 * 60;
        assert_eq!(blk[0], expected);
    }

    /// Edge extension clamps negative / past-edge coordinates.
    #[test]
    fn ref_plane_edge_extension() {
        let s = vec![1, 2, 3, 4, 5, 6]; // 3x2
        let plane = RefPlane::new(&s, 3, 2).unwrap();
        assert_eq!(plane.at(-5, -5), 1); // top-left corner
        assert_eq!(plane.at(99, 99), 6); // bottom-right corner
        assert_eq!(plane.at(1, -1), 2); // clamp y to row 0
        assert_eq!(plane.at(99, 1), 6); // clamp x to col 2, row 1
    }

    /// Uni-predictive L0 default weight: (p + offset1) >> shift1, clipped.
    #[test]
    fn weighted_uni_l0() {
        // 8-bit: shift1 = Max(2, 6) = 6, offset1 = 32.
        let p0 = vec![100 << 6, 0, 200 << 6, 255 << 6];
        let out = default_weighted_pred(&p0, &[], true, false, 2, 2, 8).unwrap();
        assert_eq!(out[0], ((100 << 6) + 32) >> 6); // = 100
        assert_eq!(out[1], 32 >> 6); // = 0
        assert_eq!(out[2], ((200 << 6) + 32) >> 6); // = 200
        assert_eq!(out[3], ((255 << 6) + 32) >> 6); // = 255
    }

    /// Bi-predictive default weight: (p0 + p1 + offset2) >> shift2.
    #[test]
    fn weighted_bi() {
        // 8-bit: shift2 = Max(3, 7) = 7, offset2 = 64.
        let p0 = vec![100 << 6];
        let p1 = vec![140 << 6];
        let out = default_weighted_pred(&p0, &p1, true, true, 1, 1, 8).unwrap();
        // ((100<<6) + (140<<6) + 64) >> 7 = (6400 + 8960 + 64) >> 7 = 15424>>7 = 120.
        assert_eq!(out[0], ((100 << 6) + (140 << 6) + 64) >> 7);
        assert_eq!(out[0], 120);
    }

    /// Default weight clips to the sample range.
    #[test]
    fn weighted_clips() {
        let p0 = vec![-50 << 6, 1000 << 6];
        let out = default_weighted_pred(&p0, &[], true, false, 2, 1, 8).unwrap();
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 255);
    }

    /// Explicit uni-predictive L0 weight (equation 8-275): 8-bit,
    /// denom 3 → log2Wd = 9, w0 = 7, o0 = 2:
    /// `((100·64·7 + 256) >> 9) + 2 = 88 + 2`.
    #[test]
    fn explicit_weighted_uni_l0() {
        let p0 = vec![100 << 6];
        let out = explicit_weighted_pred(&p0, &[], true, false, 1, 1, 3, 7, 2, 0, 0, 8).unwrap();
        assert_eq!(out[0], 90);
    }

    /// Explicit uni-predictive L1 weight (equation 8-276) mirrors L0.
    #[test]
    fn explicit_weighted_uni_l1() {
        let p1 = vec![100 << 6];
        let out = explicit_weighted_pred(&[], &p1, false, true, 1, 1, 3, 0, 0, 7, 2, 8).unwrap();
        assert_eq!(out[0], 90);
    }

    /// Explicit bi-predictive combine (equation 8-277): denom 0 →
    /// log2Wd = 6, w0 = 1, w1 = 3, zero offsets:
    /// `(50·64 + 100·64·3 + 64) >> 7 = 175`.
    #[test]
    fn explicit_weighted_bi() {
        let p0 = vec![50 << 6];
        let p1 = vec![100 << 6];
        let out = explicit_weighted_pred(&p0, &p1, true, true, 1, 1, 0, 1, 0, 3, 0, 8).unwrap();
        assert_eq!(out[0], 175);
        // Bi offsets enter as ((o0 + o1 + 1) << log2Wd) >> (log2Wd + 1)
        // ≈ (o0 + o1 + 1) / 2 added to the weighted mean.
        let out = explicit_weighted_pred(&p0, &p1, true, true, 1, 1, 0, 1, 10, 3, 9, 8).unwrap();
        assert_eq!(out[0], 175 + 10);
    }

    /// The explicit combine clips to the sample range on both sides.
    #[test]
    fn explicit_weighted_clips() {
        let p0 = vec![100 << 6, 200 << 6];
        // Large negative offset floors at 0; weight 127 saturates at 255.
        let out = explicit_weighted_pred(&p0, &[], true, false, 2, 1, 0, 1, -128, 0, 0, 8).unwrap();
        assert_eq!(out[0], 0);
        let out = explicit_weighted_pred(&p0, &[], true, false, 2, 1, 0, 127, 0, 0, 0, 8).unwrap();
        assert_eq!(out[1], 255);
    }

    /// With weight `1 << denom` and zero offset, the explicit combine
    /// degenerates to the default uni combine (§7.4.7.3 inferred values).
    #[test]
    fn explicit_default_weights_match_default_combine() {
        let p0: Vec<i32> = (0..16).map(|v| v << 6).collect();
        for denom in 0..=7u8 {
            let explicit =
                explicit_weighted_pred(&p0, &[], true, false, 4, 4, denom, 1 << denom, 0, 0, 0, 8)
                    .unwrap();
            let default = default_weighted_pred(&p0, &[], true, false, 4, 4, 8).unwrap();
            assert_eq!(explicit, default, "denom={denom}");
        }
    }

    /// Explicit-combine argument validation matches the default combine.
    #[test]
    fn explicit_weighted_errors() {
        assert_eq!(
            explicit_weighted_pred(&[], &[], false, false, 1, 1, 0, 1, 0, 1, 0, 8),
            Err(InterPredError::EmptyPlane)
        );
        assert_eq!(
            explicit_weighted_pred(&[1, 2], &[], true, false, 1, 1, 0, 1, 0, 1, 0, 8),
            Err(InterPredError::ArrayLengthMismatch {
                expected: 1,
                got: 2
            })
        );
        assert_eq!(
            explicit_weighted_pred(&[1], &[], true, false, 0, 1, 0, 1, 0, 1, 0, 8),
            Err(InterPredError::EmptyBlock)
        );
        assert_eq!(
            explicit_weighted_pred(&[1], &[], true, false, 1, 1, 0, 1, 0, 1, 0, 7),
            Err(InterPredError::InvalidBitDepth(7))
        );
    }

    /// 10-bit full-pel luma uses shift3 = Max(2, 4) = 4.
    #[test]
    fn luma_full_pel_10bit() {
        let s = vec![500i32; 8 * 8];
        let plane = RefPlane::new(&s, 8, 8).unwrap();
        let blk = interp_luma_block(&plane, 2, 2, 0, 0, 1, 1, 10).unwrap();
        assert_eq!(blk[0], 500 << 4);
    }

    /// Error surface: zero block, bad fraction, bad bit depth, bad plane.
    #[test]
    fn errors() {
        let s = vec![0i32; 4];
        let plane = RefPlane::new(&s, 2, 2).unwrap();
        assert_eq!(
            interp_luma_block(&plane, 0, 0, 0, 0, 0, 1, 8),
            Err(InterPredError::EmptyBlock)
        );
        assert_eq!(
            interp_luma_block(&plane, 0, 0, 4, 0, 1, 1, 8),
            Err(InterPredError::InvalidFraction(4))
        );
        assert_eq!(
            interp_chroma_block(&plane, 0, 0, 8, 0, 1, 1, 8),
            Err(InterPredError::InvalidFraction(8))
        );
        assert_eq!(
            interp_luma_block(&plane, 0, 0, 0, 0, 1, 1, 7),
            Err(InterPredError::InvalidBitDepth(7))
        );
        assert!(matches!(
            RefPlane::new(&[0, 1, 2], 2, 2),
            Err(InterPredError::PlaneLengthMismatch { .. })
        ));
        assert_eq!(RefPlane::new(&[], 0, 2), Err(InterPredError::EmptyPlane));
        assert_eq!(
            default_weighted_pred(&[], &[], false, false, 1, 1, 8),
            Err(InterPredError::EmptyPlane)
        );
        assert!(matches!(
            default_weighted_pred(&[1, 2], &[], true, false, 1, 1, 8),
            Err(InterPredError::ArrayLengthMismatch { .. })
        ));
    }

    /// End-to-end: interpolate two reference blocks and bi-combine.
    #[test]
    fn pipeline_luma_bi() {
        let a = vec![80i32; 16 * 16];
        let b = vec![120i32; 16 * 16];
        let pa = RefPlane::new(&a, 16, 16).unwrap();
        let pb = RefPlane::new(&b, 16, 16).unwrap();
        let l0 = interp_luma_block(&pa, 4, 4, 2, 2, 4, 4, 8).unwrap();
        let l1 = interp_luma_block(&pb, 4, 4, 1, 3, 4, 4, 8).unwrap();
        let out = default_weighted_pred(&l0, &l1, true, true, 4, 4, 8).unwrap();
        // Flat planes: l0 == 80<<6 everywhere, l1 == 120<<6; bi-combine
        // = ((80+120)<<6 + 64) >> 7 = (12800 + 64) >> 7 = 100.
        for &s in &out {
            assert_eq!(s, 100);
        }
    }

    // -- §8.5.3.3.1 driver tests -------------------------------------------

    /// A full-pel uni-L0 PU on a flat luma plane reproduces the reference
    /// sample value (full-pel: `A << shift3`, then default-weight
    /// `(p + offset1) >> shift1` recovers `A`).
    #[test]
    fn driver_uni_l0_full_pel_flat() {
        let luma = vec![130i32; 32 * 32];
        let cb = vec![70i32; 16 * 16];
        let cr = vec![200i32; 16 * 16];
        let lp = RefPlane::new(&luma, 32, 32).unwrap();
        let cbp = RefPlane::new(&cb, 16, 16).unwrap();
        let crp = RefPlane::new(&cr, 16, 16).unwrap();
        let l0 = ListPrediction {
            pred_flag: true,
            luma: lp,
            cb: Some(cbp),
            cr: Some(crp),
            mv_l: [0, 0],
            mv_c: [0, 0],
        };
        // Unused L1: a dummy (1x1) plane that is never read.
        let dummy = vec![0i32; 1];
        let dp = RefPlane::new(&dummy, 1, 1).unwrap();
        let l1 = ListPrediction {
            pred_flag: false,
            luma: dp,
            cb: None,
            cr: None,
            mv_l: [0, 0],
            mv_c: [0, 0],
        };
        let geom = InterPredGeometry {
            x_pb: 4,
            y_pb: 4,
            n_pb_w: 8,
            n_pb_h: 8,
            chroma_array_type: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
        };
        let pred = predict_inter_pu(&l0, &l1, &geom).unwrap();
        assert_eq!(pred.luma.len(), 64);
        assert_eq!(pred.cb.len(), 16);
        assert_eq!(pred.cr.len(), 16);
        assert!(pred.luma.iter().all(|&v| v == 130));
        assert!(pred.cb.iter().all(|&v| v == 70));
        assert!(pred.cr.iter().all(|&v| v == 200));
    }

    /// A full-pel motion vector shifts the reference window: a ramp plane
    /// predicted with `mvL = [4, 0]` (one full luma sample right) reads
    /// the column one to the right of `xPb`.
    #[test]
    fn driver_full_pel_mv_shifts_window() {
        // 16-wide luma ramp where sample(x,y) == x.
        let mut luma = vec![0i32; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                luma[y * 16 + x] = x as i32;
            }
        }
        let lp = RefPlane::new(&luma, 16, 16).unwrap();
        let dummy = vec![0i32; 1];
        let dp = RefPlane::new(&dummy, 1, 1).unwrap();
        let l0 = ListPrediction {
            pred_flag: true,
            luma: lp,
            cb: None,
            cr: None,
            mv_l: [4, 0], // +1 full luma sample horizontally.
            mv_c: [0, 0],
        };
        let l1 = ListPrediction {
            pred_flag: false,
            luma: dp,
            cb: None,
            cr: None,
            mv_l: [0, 0],
            mv_c: [0, 0],
        };
        let geom = InterPredGeometry {
            x_pb: 2,
            y_pb: 2,
            n_pb_w: 4,
            n_pb_h: 4,
            chroma_array_type: 0,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
        };
        let pred = predict_inter_pu(&l0, &l1, &geom).unwrap();
        // predSamples[xL] reads ref column xPb + 1 + xL = 3 + xL.
        for yl in 0..4 {
            for xl in 0..4 {
                assert_eq!(pred.luma[yl * 4 + xl], 3 + xl as i32, "xl={xl}");
            }
        }
        assert!(pred.cb.is_empty(), "monochrome PU has no chroma");
    }

    /// Bi-prediction on two flat planes averages the two reference values.
    #[test]
    fn driver_bi_averages() {
        let a = vec![60i32; 16 * 16];
        let b = vec![100i32; 16 * 16];
        let pa = RefPlane::new(&a, 16, 16).unwrap();
        let pb = RefPlane::new(&b, 16, 16).unwrap();
        let l0 = ListPrediction {
            pred_flag: true,
            luma: pa,
            cb: None,
            cr: None,
            mv_l: [0, 0],
            mv_c: [0, 0],
        };
        let l1 = ListPrediction {
            pred_flag: true,
            luma: pb,
            cb: None,
            cr: None,
            mv_l: [2, 1], // quarter-pel; flat plane is unaffected.
            mv_c: [0, 0],
        };
        let geom = InterPredGeometry {
            x_pb: 4,
            y_pb: 4,
            n_pb_w: 4,
            n_pb_h: 4,
            chroma_array_type: 0,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
        };
        let pred = predict_inter_pu(&l0, &l1, &geom).unwrap();
        // ((60 + 100) >> 1) == 80.
        assert!(pred.luma.iter().all(|&v| v == 80));
    }

    /// The driver rejects a PU with no list selected and a zero block.
    #[test]
    fn driver_errors() {
        let dummy = vec![0i32; 1];
        let dp = RefPlane::new(&dummy, 1, 1).unwrap();
        let none = ListPrediction {
            pred_flag: false,
            luma: dp,
            cb: None,
            cr: None,
            mv_l: [0, 0],
            mv_c: [0, 0],
        };
        let geom = InterPredGeometry {
            x_pb: 0,
            y_pb: 0,
            n_pb_w: 4,
            n_pb_h: 4,
            chroma_array_type: 0,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
        };
        assert_eq!(
            predict_inter_pu(&none, &none, &geom),
            Err(InterPredError::EmptyPlane)
        );
        let l0 = ListPrediction {
            pred_flag: true,
            ..none
        };
        let zero = InterPredGeometry { n_pb_w: 0, ..geom };
        assert_eq!(
            predict_inter_pu(&l0, &none, &zero),
            Err(InterPredError::EmptyBlock)
        );
    }

    // -----------------------------------------------------------------
    // SIMD backend bit-exactness (§8.5.3.3 vectorized kernels)
    // -----------------------------------------------------------------

    /// Deterministic pseudo-random samples in `[0, max]`, so the
    /// cross-backend comparisons run on realistic data rather than on a
    /// constant plane whose filter output is degenerate.
    fn pseudo_random(seed: u64, len: usize, max: i32) -> Vec<i32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 33) % (max as u64 + 1)) as i32
            })
            .collect()
    }

    /// Luma / chroma prediction-block shapes HEVC can signal (the PART_*
    /// splits of a 64x64 through 8x8 CU, plus their 4:2:0 chroma halves).
    const BLOCK_SHAPES: [(usize, usize); 12] = [
        (4, 4),
        (4, 8),
        (8, 4),
        (8, 8),
        (4, 16),
        (16, 4),
        (12, 16),
        (16, 12),
        (16, 16),
        (24, 32),
        (32, 32),
        (64, 64),
    ];

    /// Motion-compensated block origins: interior, straddling the left /
    /// top edge, and straddling the right / bottom edge — so the
    /// borrowed-window fast path and the `Clip3` edge-extension path are
    /// both exercised on every backend.
    const ORIGINS: [(i32, i32); 5] = [(9, 7), (-5, 3), (2, -6), (70, 40), (-3, -3)];

    /// The eight-bit block path accumulates at 16-bit width, and this
    /// is what holds that formulation against the normative per-sample
    /// §8.5.3.3.3.2 / §8.5.3.3.3.3 equations rather than against
    /// another 16-bit accumulation.
    ///
    /// `every_backend_matches_scalar_luma_block` compares the vector
    /// kernels to the scalar one, but at eight bits both sides are the
    /// narrow kernel, so it cannot see a range error the narrowing
    /// itself introduces. This runs every block shape, origin and phase
    /// through the block path and through `interp_luma_sample` /
    /// `interp_chroma_sample`, which are `i32` throughout.
    #[test]
    fn the_eight_bit_block_path_matches_the_per_sample_equations() {
        let (pw, ph) = (96usize, 72usize);
        let plane_samples = pseudo_random(8, pw * ph, 255);
        let plane = RefPlane::new(&plane_samples, pw, ph).unwrap();
        for &(w, h) in &BLOCK_SHAPES {
            for &(x_int, y_int) in &ORIGINS {
                for x_frac in 0..4 {
                    for y_frac in 0..4 {
                        let got = interp_luma_block(&plane, x_int, y_int, x_frac, y_frac, w, h, 8)
                            .unwrap();
                        let want = (0..h as i32)
                            .flat_map(|y| {
                                (0..w as i32).map(move |x| {
                                    interp_luma_sample(
                                        &plane,
                                        x_int + x,
                                        y_int + y,
                                        x_frac,
                                        y_frac,
                                        8,
                                    )
                                })
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(
                            got, want,
                            "luma {w}x{h} @({x_int},{y_int}) frac=({x_frac},{y_frac})"
                        );
                    }
                }
                for x_frac in 0..8 {
                    for y_frac in 0..8 {
                        let got =
                            interp_chroma_block(&plane, x_int, y_int, x_frac, y_frac, w, h, 8)
                                .unwrap();
                        let want = (0..h as i32)
                            .flat_map(|y| {
                                (0..w as i32).map(move |x| {
                                    interp_chroma_sample(
                                        &plane,
                                        x_int + x,
                                        y_int + y,
                                        x_frac,
                                        y_frac,
                                        8,
                                    )
                                })
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(
                            got, want,
                            "chroma {w}x{h} @({x_int},{y_int}) frac=({x_frac},{y_frac})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_backend_matches_scalar_luma_block() {
        let (pw, ph) = (96usize, 72usize);
        for bit_depth in [8u8, 10, 16] {
            let plane_samples = pseudo_random(bit_depth as u64, pw * ph, (1 << bit_depth) - 1);
            let plane = RefPlane::new(&plane_samples, pw, ph).unwrap();
            for &(w, h) in &BLOCK_SHAPES {
                for &(x_int, y_int) in &ORIGINS {
                    for x_frac in 0..4 {
                        for y_frac in 0..4 {
                            let reference = interp_luma_block_with(
                                Isa::Scalar,
                                &plane,
                                x_int,
                                y_int,
                                x_frac,
                                y_frac,
                                w,
                                h,
                                bit_depth,
                            )
                            .unwrap();
                            for isa in simd::available_isas() {
                                let got = interp_luma_block_with(
                                    isa, &plane, x_int, y_int, x_frac, y_frac, w, h, bit_depth,
                                )
                                .unwrap();
                                assert_eq!(
                                    got, reference,
                                    "{isa:?} luma {w}x{h} @({x_int},{y_int}) \
                                     frac=({x_frac},{y_frac}) bd={bit_depth}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_backend_matches_scalar_chroma_block() {
        let (pw, ph) = (64usize, 48usize);
        for bit_depth in [8u8, 10, 16] {
            let plane_samples = pseudo_random(bit_depth as u64 + 99, pw * ph, (1 << bit_depth) - 1);
            let plane = RefPlane::new(&plane_samples, pw, ph).unwrap();
            for &(w, h) in &BLOCK_SHAPES[..9] {
                for &(x_int, y_int) in &ORIGINS {
                    for x_frac in 0..8 {
                        for y_frac in 0..8 {
                            let reference = interp_chroma_block_with(
                                Isa::Scalar,
                                &plane,
                                x_int,
                                y_int,
                                x_frac,
                                y_frac,
                                w,
                                h,
                                bit_depth,
                            )
                            .unwrap();
                            for isa in simd::available_isas() {
                                let got = interp_chroma_block_with(
                                    isa, &plane, x_int, y_int, x_frac, y_frac, w, h, bit_depth,
                                )
                                .unwrap();
                                assert_eq!(
                                    got, reference,
                                    "{isa:?} chroma {w}x{h} @({x_int},{y_int}) \
                                     frac=({x_frac},{y_frac}) bd={bit_depth}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The interpolation output range the combines actually see: the
    /// §8.5.3.3.3 intermediates are signed and span roughly
    /// `± 2^( bitDepth − 2 ) · 64`.
    fn intermediates(seed: u64, len: usize, bit_depth: u8) -> Vec<i32> {
        let span = ((1i64 << bit_depth) - 1) * 80;
        pseudo_random(seed, len, span as i32)
            .into_iter()
            .map(|v| v - (span / 2) as i32)
            .collect()
    }

    #[test]
    fn every_backend_matches_scalar_default_weighted_pred() {
        for bit_depth in [8u8, 10, 16] {
            for &(w, h) in &BLOCK_SHAPES {
                let count = w * h;
                let p0 = intermediates(1 + bit_depth as u64, count, bit_depth);
                let p1 = intermediates(2 + bit_depth as u64, count, bit_depth);
                for &(f0, f1) in &[(true, false), (false, true), (true, true)] {
                    let reference =
                        default_weighted_pred_with(Isa::Scalar, &p0, &p1, f0, f1, w, h, bit_depth)
                            .unwrap();
                    for isa in simd::available_isas() {
                        let got =
                            default_weighted_pred_with(isa, &p0, &p1, f0, f1, w, h, bit_depth)
                                .unwrap();
                        assert_eq!(got, reference, "{isa:?} default {w}x{h} bd={bit_depth}");
                    }
                }
            }
        }
    }

    #[test]
    fn every_backend_matches_scalar_explicit_weighted_pred() {
        // Legal HEVC weights/offsets, plus a deliberately out-of-range
        // weight that forces the `i64` scalar fallback, so both the
        // vectorized and the widened path are compared against scalar.
        let cases: [(u8, i32, i32, i32, i32); 5] = [
            (0, 1, 0, 1, 0),
            (6, 64, 0, 64, 0),
            (7, 127, 40, -128, -40),
            (5, -3, -12, 41, 7),
            (0, i32::MAX / 2, i32::MIN / 4, i32::MAX / 2, 0),
        ];
        for bit_depth in [8u8, 10, 16] {
            for &(w, h) in &BLOCK_SHAPES {
                let count = w * h;
                let p0 = intermediates(21 + bit_depth as u64, count, bit_depth);
                let p1 = intermediates(22 + bit_depth as u64, count, bit_depth);
                for &(denom, w0, o0, w1, o1) in &cases {
                    for &(f0, f1) in &[(true, false), (false, true), (true, true)] {
                        let reference = explicit_weighted_pred_with(
                            Isa::Scalar,
                            &p0,
                            &p1,
                            f0,
                            f1,
                            w,
                            h,
                            denom,
                            w0,
                            o0,
                            w1,
                            o1,
                            bit_depth,
                        )
                        .unwrap();
                        for isa in simd::available_isas() {
                            let got = explicit_weighted_pred_with(
                                isa, &p0, &p1, f0, f1, w, h, denom, w0, o0, w1, o1, bit_depth,
                            )
                            .unwrap();
                            assert_eq!(
                                got, reference,
                                "{isa:?} explicit {w}x{h} denom={denom} bd={bit_depth}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Micro-benchmark for the §8.5.3.3 inter-prediction kernels: the
    /// two-dimensional 8-tap luma filter, the two-dimensional 4-tap
    /// chroma filter, and the bi-predictive default combine, each timed
    /// on the scalar reference and on every SIMD backend the host CPU
    /// offers.
    ///
    /// A/Bs the 16-bit accumulation issue #378 landed against the 32-bit
    /// one it replaces, over the §8.5.3.3.3.2 two-dimensional luma path
    /// at every prediction-unit size, with the block walk, the source
    /// narrowing and the allocations all included.
    ///
    /// `simd::measure_narrow_filter_taps` times the kernel alone and is
    /// the ceiling; this is what a block actually gets, because the
    /// narrow arm has to convert its source window to `i16` first, and
    /// only the horizontal pass of the two-dimensional case narrows —
    /// the vertical pass multiplies a 16-bit intermediate by a
    /// coefficient of up to 58 and needs a 32-bit accumulator either way.
    ///
    /// Both arms run in one process, interleaved, best of many rounds:
    /// separate benchmark processes on this host disagree with each
    /// other by more than the effect being measured. The two arms are
    /// asserted to agree sample-for-sample before anything is timed.
    ///
    /// Ignored by default because it is a timing measurement, not an
    /// assertion. Run it with
    /// `cargo test --release --features native --lib
    /// measure_narrow_vs_wide_block -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn measure_narrow_vs_wide_block() {
        use std::time::Instant;

        let (pw, ph) = (256usize, 256usize);
        let plane_samples = pseudo_random(8, pw * ph, 255);
        let plane = RefPlane::new(&plane_samples, pw, ph).unwrap();
        let isa = simd::detected_isa();
        let rounds = 15;
        // Every phase case of Table 8-8 that filters at all, because
        // they narrow very differently: the one-dimensional cases have a
        // single pass and narrow all of it, while the two-dimensional
        // case narrows only its horizontal pass and pays the conversion
        // for a vertical pass that cannot use it.
        type Phase<'k> = (&'static str, Option<&'k [i32; 8]>, Option<&'k [i32; 8]>);
        let cases: [Phase<'_>; 3] = [
            ("h-only", Some(&LUMA_FILTER[2]), None),
            ("v-only", None, Some(&LUMA_FILTER[2])),
            ("2-D", Some(&LUMA_FILTER[2]), Some(&LUMA_FILTER[2])),
        ];

        println!("\n8-tap luma interp_block, i32 vs i16 accumulation, {isa:?}, best of {rounds}");
        println!("  (equal total sample count at every block size)");
        println!("  phase   block     i32 ms   i16 ms   narrow");
        for (name, hk, vk) in cases {
            for &(w, h) in &[(8usize, 8usize), (16, 16), (32, 32), (64, 64)] {
                let calls = (1 << 22) / (w * h);
                let run = |narrow: bool| {
                    interp_block_with_width::<8>(isa, &plane, 4, 4, hk, vk, w, h, 8, narrow)
                };
                assert_eq!(run(false), run(true), "{name} arms disagree at {w}x{h}");

                let (mut bw, mut bn) = (f64::INFINITY, f64::INFINITY);
                for _ in 0..rounds {
                    let start = Instant::now();
                    for _ in 0..calls {
                        std::hint::black_box(run(false));
                    }
                    bw = bw.min(start.elapsed().as_secs_f64());
                    let start = Instant::now();
                    for _ in 0..calls {
                        std::hint::black_box(run(true));
                    }
                    bn = bn.min(start.elapsed().as_secs_f64());
                }
                println!(
                    "  {name:>6}  {:>5}  {:9.2} {:8.2}  {:5.2}x",
                    format!("{w}x{h}"),
                    bw * 1e3,
                    bn * 1e3,
                    bw / bn
                );
            }
        }
    }

    /// A/Bs the full-height `w x ( h + 7 )` intermediate the
    /// two-dimensional 8-tap luma path uses against the wrap-around ring
    /// issue #309 proposed in its place, which keeps only the eight
    /// horizontal rows the vertical pass has live.
    ///
    /// Both arms are spelled out here rather than one of them being the
    /// production path, because the ring lost and was not kept: the
    /// comparison has to stay runnable after the revert for the table in
    /// `benches/README.md` to be reproducible.
    ///
    /// Both arms run in one process, interleaved, best of many rounds:
    /// separate benchmark processes on this host disagree with each
    /// other by more than the effect being measured.
    ///
    /// Ignored by default because it is a timing measurement, not an
    /// assertion. Run it with
    /// `cargo test --release --features native --lib
    /// measure_2d_ring_vs_flat -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn measure_2d_ring_vs_flat() {
        use std::time::Instant;

        /// The full-height intermediate `interp_block` uses today.
        // The arms have to be spelled the same way to be comparable, and
        // both mirror `interp_block`'s own signature.
        #[allow(clippy::too_many_arguments)]
        fn flat_2d(
            isa: Isa,
            plane: &RefPlane<'_>,
            x_int: i32,
            y_int: i32,
            hk: &[i32; 8],
            vk: &[i32; 8],
            w: usize,
            h: usize,
            shift1: i32,
        ) -> Vec<i32> {
            let (halo, span, rows) = (3i32, w + 7, h + 7);
            let mut out = vec![0i32; w * h];
            let mut horizontal = vec![0i32; w * rows];
            let mut scratch = vec![0i32; span];
            for row in 0..rows {
                let src =
                    plane.row_window(x_int - halo, span, y_int - halo + row as i32, &mut scratch);
                let taps: [&[i32]; 8] = std::array::from_fn(|t| &src[t..t + w]);
                simd::filter_taps(
                    isa,
                    &taps,
                    hk,
                    shift1,
                    &mut horizontal[row * w..(row + 1) * w],
                );
            }
            for y in 0..h {
                let taps: [&[i32]; 8] =
                    std::array::from_fn(|t| &horizontal[(y + t) * w..(y + t + 1) * w]);
                simd::filter_taps(isa, &taps, vk, 6, &mut out[y * w..(y + 1) * w]);
            }
            out
        }

        /// The wrap-around ring issue #309 proposed: only the eight
        /// horizontal rows the vertical pass has live are kept, so the
        /// intermediate is `8 x w` instead of `( h + 7 ) x w`. Same total
        /// horizontal work, no row filtered twice.
        #[allow(clippy::too_many_arguments)]
        fn ring_2d(
            isa: Isa,
            plane: &RefPlane<'_>,
            x_int: i32,
            y_int: i32,
            hk: &[i32; 8],
            vk: &[i32; 8],
            w: usize,
            h: usize,
            shift1: i32,
        ) -> Vec<i32> {
            let (halo, span) = (3i32, w + 7);
            let mut out = vec![0i32; w * h];
            let mut ring = vec![0i32; w * 8];
            let mut scratch = vec![0i32; span];
            let horizontal_row = |ring: &mut [i32], scratch: &mut [i32], row: usize| {
                let src = plane.row_window(x_int - halo, span, y_int - halo + row as i32, scratch);
                let taps: [&[i32]; 8] = std::array::from_fn(|t| &src[t..t + w]);
                let slot = row % 8;
                simd::filter_taps(isa, &taps, hk, shift1, &mut ring[slot * w..(slot + 1) * w]);
            };
            for row in 0..8 {
                horizontal_row(&mut ring, &mut scratch, row);
            }
            for y in 0..h {
                {
                    let ring = &ring;
                    let taps: [&[i32]; 8] = std::array::from_fn(|t| {
                        let slot = (y + t) % 8;
                        &ring[slot * w..(slot + 1) * w]
                    });
                    simd::filter_taps(isa, &taps, vk, 6, &mut out[y * w..(y + 1) * w]);
                }
                if y + 8 < h + 7 {
                    horizontal_row(&mut ring, &mut scratch, y + 8);
                }
            }
            out
        }

        let (pw, ph) = (1920usize, 1088usize);
        let plane_samples = pseudo_random(7, pw * ph, 255);
        let plane = RefPlane::new(&plane_samples, pw, ph).unwrap();
        let isas = simd::available_isas();
        let rounds = 15;
        let shift1 = interp_shift1(8);
        let (hk, vk) = (&LUMA_FILTER[2], &LUMA_FILTER[3]);

        println!("\n2D 8-tap luma: ring vs full-height intermediate");
        println!("  (one process, best of {rounds} interleaved rounds, ms)");
        println!("  size      isa       flat       ring   speedup");
        // The A/B is only meaningful if the two arms compute the same
        // block, and if the flat arm is still what `interp_block` runs.
        for &size in &[8usize, 16, 32, 64] {
            for &isa in &isas {
                let flat = flat_2d(isa, &plane, 64, 64, hk, vk, size, size, shift1);
                let ring = ring_2d(isa, &plane, 64, 64, hk, vk, size, size, shift1);
                let prod =
                    interp_luma_block_with(isa, &plane, 64, 64, 2, 3, size, size, 8).unwrap();
                assert_eq!(flat, ring, "{isa:?} {size}x{size}: ring differs from flat");
                assert_eq!(
                    flat, prod,
                    "{isa:?} {size}x{size}: flat differs from interp_block"
                );
            }
        }

        for &size in &[8usize, 16, 32, 64] {
            let (cols, rows_of_blocks) = (pw / size, ph / size);
            let blocks: Vec<(i32, i32)> = (0..rows_of_blocks)
                .flat_map(|by| (0..cols).map(move |bx| ((bx * size) as i32, (by * size) as i32)))
                .collect();
            let mut best = vec![[f64::INFINITY; 2]; isas.len()];
            let mut sink = 0i64;
            for _ in 0..rounds {
                for (i, &isa) in isas.iter().enumerate() {
                    let start = Instant::now();
                    for &(x, y) in &blocks {
                        let b = flat_2d(isa, &plane, x, y, hk, vk, size, size, shift1);
                        sink += b[0] as i64;
                    }
                    let flat = start.elapsed().as_secs_f64();

                    let start = Instant::now();
                    for &(x, y) in &blocks {
                        let b = ring_2d(isa, &plane, x, y, hk, vk, size, size, shift1);
                        sink += b[0] as i64;
                    }
                    let ring = start.elapsed().as_secs_f64();

                    best[i][0] = best[i][0].min(flat);
                    best[i][1] = best[i][1].min(ring);
                }
            }
            for (isa, t) in isas.iter().zip(best.iter()) {
                println!(
                    "  {size:>4}  {:>7}  {:9.2}  {:9.2}  {:6.2}x{}",
                    format!("{isa:?}"),
                    t[0] * 1e3,
                    t[1] * 1e3,
                    t[0] / t[1],
                    if sink == i64::MIN { "!" } else { "" },
                );
            }
        }
    }

    /// Times the two-dimensional 8-tap luma path's horizontal and
    /// vertical passes separately, per block size and per backend.
    ///
    /// Issue #280 inferred from the shape of its block-size sweep that
    /// the `w x ( h + 7 )` intermediate the two-dimensional path
    /// materializes is what erodes the kernel's advantage at 32x32 and
    /// 64x64. This measures the two passes apart instead of inferring
    /// it, alongside the one-dimensional phases that have no
    /// intermediate at all, so the mechanism is read off the numbers.
    ///
    /// Ignored by default because it is a timing measurement, not an
    /// assertion. Run it with
    /// `cargo test --release --features native --lib
    /// measure_interp_pass_split -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn measure_interp_pass_split() {
        use std::time::Instant;

        const N: usize = 8;
        let (pw, ph) = (1920usize, 1088usize);
        let plane_samples = pseudo_random(7, pw * ph, 255);
        let plane = RefPlane::new(&plane_samples, pw, ph).unwrap();
        let isas = simd::available_isas();
        let rounds = 5;
        let shift1 = interp_shift1(8);
        let halo = N as i32 / 2 - 1;
        let hk = &LUMA_FILTER[2];
        let vk = &LUMA_FILTER[3];

        println!("\n2D 8-tap luma pass split, best of {rounds} interleaved rounds");
        println!(
            "  size      isa    horiz-pass   vert-pass    2D total   H-only   V-only  full-pel"
        );
        for &size in &[8usize, 16, 32, 64] {
            // One 1080p frame's worth of `size` x `size` luma blocks, so
            // every row of the table does the same sample count.
            let cols = pw / size;
            let rows_of_blocks = ph / size;
            let blocks: Vec<(i32, i32)> = (0..rows_of_blocks)
                .flat_map(|by| (0..cols).map(move |bx| ((bx * size) as i32, (by * size) as i32)))
                .collect();
            let (w, h) = (size, size);
            let span = w + N - 1;
            let src_rows = h + N - 1;
            let mut best = vec![[f64::INFINITY; 6]; isas.len()];
            let mut sink = 0i64;
            for _ in 0..rounds {
                for (i, &isa) in isas.iter().enumerate() {
                    // Horizontal pass alone: the `h + 7` filtered rows the
                    // vertical pass consumes, written to a fresh buffer.
                    let mut horizontal = vec![0i32; w * src_rows];
                    let mut scratch = vec![0i32; span];
                    let start = Instant::now();
                    for &(x, y) in &blocks {
                        for row in 0..src_rows {
                            let src = plane.row_window(
                                x - halo,
                                span,
                                y - halo + row as i32,
                                &mut scratch,
                            );
                            let taps: [&[i32]; N] = std::array::from_fn(|t| &src[t..t + w]);
                            simd::filter_taps(
                                isa,
                                &taps,
                                hk,
                                shift1,
                                &mut horizontal[row * w..(row + 1) * w],
                            );
                        }
                        sink += horizontal[0] as i64;
                    }
                    let horiz = start.elapsed().as_secs_f64();

                    // Vertical pass alone: the same intermediate, already
                    // populated, filtered down the columns into `out`.
                    let mut out = vec![0i32; w * h];
                    let start = Instant::now();
                    for _ in &blocks {
                        for y in 0..h {
                            let taps: [&[i32]; N] =
                                std::array::from_fn(|t| &horizontal[(y + t) * w..(y + t + 1) * w]);
                            simd::filter_taps(isa, &taps, vk, 6, &mut out[y * w..(y + 1) * w]);
                        }
                        sink += out[0] as i64;
                    }
                    let vert = start.elapsed().as_secs_f64();

                    let start = Instant::now();
                    for &(x, y) in &blocks {
                        let b = interp_luma_block_with(isa, &plane, x, y, 2, 3, w, h, 8).unwrap();
                        sink += b[0] as i64;
                    }
                    let both = start.elapsed().as_secs_f64();

                    let start = Instant::now();
                    for &(x, y) in &blocks {
                        let b = interp_luma_block_with(isa, &plane, x, y, 2, 0, w, h, 8).unwrap();
                        sink += b[0] as i64;
                    }
                    let h_only = start.elapsed().as_secs_f64();

                    let start = Instant::now();
                    for &(x, y) in &blocks {
                        let b = interp_luma_block_with(isa, &plane, x, y, 0, 3, w, h, 8).unwrap();
                        sink += b[0] as i64;
                    }
                    let v_only = start.elapsed().as_secs_f64();

                    let start = Instant::now();
                    for &(x, y) in &blocks {
                        let b = interp_luma_block_with(isa, &plane, x, y, 0, 0, w, h, 8).unwrap();
                        sink += b[0] as i64;
                    }
                    let full_pel = start.elapsed().as_secs_f64();

                    for (slot, t) in best[i]
                        .iter_mut()
                        .zip([horiz, vert, both, h_only, v_only, full_pel])
                    {
                        *slot = slot.min(t);
                    }
                }
            }
            let base = best[0];
            for (isa, times) in isas.iter().zip(best.iter().copied()) {
                let cells: String = times
                    .iter()
                    .zip(base.iter())
                    .map(|(t, b)| format!("{:7.2} ({:4.2}x)", t * 1e3, b / t))
                    .collect::<Vec<_>>()
                    .join("  ");
                println!(
                    "  {size:>4}  {:>7}  {cells}{}",
                    format!("{isa:?}"),
                    if sink == i64::MIN { "!" } else { "" },
                );
            }
        }
    }

    /// Ignored by default because it is a timing measurement, not an
    /// assertion. Run it with
    /// `cargo test --release --features native --lib
    /// simd_inter_pred_benchmark -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn simd_inter_pred_benchmark() {
        use std::time::Instant;

        let (pw, ph) = (1920usize, 1088usize);
        let plane_samples = pseudo_random(7, pw * ph, 255);
        let plane = RefPlane::new(&plane_samples, pw, ph).unwrap();
        // One 1080p frame's worth of 16x16 luma / 8x8 chroma blocks.
        let blocks: Vec<(i32, i32)> = (0..68)
            .flat_map(|by| (0..120).map(move |bx| (bx * 16, by * 16)))
            .collect();
        let combine_len = 16 * 16;
        let p0 = intermediates(31, combine_len, 8);
        let p1 = intermediates(32, combine_len, 8);

        // Every figure below is the *minimum* over `rounds` timed passes,
        // and the backends are measured round-robin inside each round. A
        // single timed pass on a loaded machine swings by more than 2x,
        // which is enough to report a real speedup as a regression; the
        // minimum is the pass that suffered least interference and is the
        // closest this can get to the kernel's own cost.
        let rounds = 5;
        let isas = simd::available_isas();
        println!(
            "\ninter-prediction kernels, {} 16x16 blocks/pass, best of {rounds} rounds",
            blocks.len()
        );
        let mut best = vec![[f64::INFINITY; 3]; isas.len()];
        let mut sink = 0i64;
        for _ in 0..rounds {
            for (i, &isa) in isas.iter().enumerate() {
                let start = Instant::now();
                for &(x, y) in &blocks {
                    let b = interp_luma_block_with(isa, &plane, x, y, 2, 3, 16, 16, 8).unwrap();
                    sink += b[0] as i64;
                }
                let luma = start.elapsed().as_secs_f64();

                let start = Instant::now();
                for &(x, y) in &blocks {
                    let b =
                        interp_chroma_block_with(isa, &plane, x / 2, y / 2, 5, 3, 8, 8, 8).unwrap();
                    sink += b[0] as i64;
                }
                let chroma = start.elapsed().as_secs_f64();

                let start = Instant::now();
                for _ in &blocks {
                    let b =
                        default_weighted_pred_with(isa, &p0, &p1, true, true, 16, 16, 8).unwrap();
                    sink += b[0] as i64;
                }
                let combine = start.elapsed().as_secs_f64();

                for (slot, t) in best[i].iter_mut().zip([luma, chroma, combine]) {
                    *slot = slot.min(t);
                }
            }
        }
        // The three stages above measure the whole §8.5.3.3 block path,
        // which allocates and zeroes a result `Vec` per block. That
        // allocation is the same work on every backend, so it compresses
        // every ratio towards 1.00x — most visibly for the combine, whose
        // kernel is only two multiply-accumulates per sample and so does
        // less work than the allocation that surrounds it. These two arms
        // time the kernels alone over one large buffer to show what the
        // backend itself is worth, with no allocator in the measurement.
        // Sized to stay resident in L1 the way a real block does: a
        // 256 KiB buffer would make both backends memory-bound and report
        // 1.00x for reasons that have nothing to do with the kernels.
        let kernel_len = 2048;
        let a = intermediates(41, kernel_len, 8);
        let b = intermediates(42, kernel_len, 8);
        let mut kernel_out = vec![0i32; kernel_len];
        let mut kernel_best = vec![[f64::INFINITY; 3]; isas.len()];
        for _ in 0..rounds {
            for (i, &isa) in isas.iter().enumerate() {
                let taps8: [&[i32]; 8] = std::array::from_fn(|t| &a[t..t + kernel_len - 8]);
                let start = Instant::now();
                for _ in 0..1024 {
                    simd::filter_taps(
                        isa,
                        &taps8,
                        &LUMA_FILTER[2],
                        6,
                        &mut kernel_out[..kernel_len - 8],
                    );
                }
                let taps = start.elapsed().as_secs_f64();
                sink += kernel_out[0] as i64;

                // The arm above passes the compile-time literal
                // `LUMA_FILTER[2]`, which is not what §8.5.3.3.3 does: the
                // block path indexes `LUMA_FILTER` by a runtime fractional
                // position, so its coefficients are opaque to the optimizer.
                // That difference is not cosmetic — it decides what the
                // *scalar* baseline compiles to, and so what the vector
                // backends are being compared against (issue #321). This arm
                // is the same call with the coefficients hidden, which is the
                // kernel-against-kernel comparison; the one above is the
                // kernel against a loop LLVM was allowed to specialize.
                let start = Instant::now();
                for _ in 0..1024 {
                    simd::filter_taps(
                        isa,
                        &taps8,
                        std::hint::black_box(&LUMA_FILTER[2]),
                        6,
                        &mut kernel_out[..kernel_len - 8],
                    );
                }
                let taps_opaque = start.elapsed().as_secs_f64();
                sink += kernel_out[0] as i64;

                let start = Instant::now();
                for _ in 0..1024 {
                    simd::combine_weighted(isa, &[&a, &b], &[1, 1], 64, 7, 0, 255, &mut kernel_out);
                }
                let combine = start.elapsed().as_secs_f64();
                sink += kernel_out[0] as i64;

                for (slot, t) in kernel_best[i].iter_mut().zip([taps, taps_opaque, combine]) {
                    *slot = slot.min(t);
                }
            }
        }

        let baselines = best[0];
        for (isa, [luma, chroma, combine]) in isas.iter().zip(best.iter().copied()) {
            println!(
                "  {:>8}  luma 8-tap 2D {:7.2} ms ({:4.2}x)  chroma 4-tap 2D {:7.2} ms ({:4.2}x)  \
                 bi combine {:7.2} ms ({:4.2}x){}",
                format!("{isa:?}"),
                luma * 1e3,
                baselines[0] / luma,
                chroma * 1e3,
                baselines[1] / chroma,
                combine * 1e3,
                baselines[2] / combine,
                if sink == i64::MIN { "!" } else { "" },
            );
        }
        println!("  kernels alone, {kernel_len} L1-resident samples x 1024 passes:");
        let kernel_baselines = kernel_best[0];
        for (isa, [taps, taps_opaque, combine]) in isas.iter().zip(kernel_best.iter().copied()) {
            println!(
                "  {:>8}  filter_taps 8-tap {:7.2} ms ({:4.2}x)  filter_taps 8-tap opaque coeffs \
                 {:7.2} ms ({:4.2}x)  combine_weighted bi {:7.2} ms ({:4.2}x)",
                format!("{isa:?}"),
                taps * 1e3,
                kernel_baselines[0] / taps,
                taps_opaque * 1e3,
                kernel_baselines[1] / taps_opaque,
                combine * 1e3,
                kernel_baselines[2] / combine,
            );
        }
    }
}
