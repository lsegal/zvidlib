//! SIMD-accelerated pixel kernels for the native VP9 encoder.
//!
//! The encoder's rate-distortion search runs the same handful of kernels over
//! every candidate partition, transform size, prediction mode and motion
//! vector, so they are where it spends its time:
//!
//! - [`forward_transform_4x4`] and [`matrix_product`], the forward transforms:
//!   libvpx's integer 4x4 DCT/ADST and the floating-point 8x8 to 32x32 ones.
//! - [`quantize`], which turns transform coefficients into levels and the
//!   dequantized values the reconstruction adds back.
//! - [`sad`], [`sse`] and [`residual`], the distortion metrics behind motion
//!   search, mode decisions and the loop filter level search.
//! - [`predict_tm`] and [`convolve8`], the TM intra predictor and the 8-tap
//!   motion compensation, which must match the decoder bit for bit.
//! - [`luma_row`] and [`chroma_row`], the RGBA/BGRA to YUV 4:2:0 input
//!   conversion.
//!
//! Each dispatches once per call through cached runtime CPU feature detection
//! ([`isa`]) to an SSE4.1 or AVX2 implementation on `x86_64`, a NEON
//! implementation on `aarch64`, or the portable scalar implementation
//! everywhere else, `wasm32` included. Every vectorized path is bit-identical
//! to its scalar reference, so enabling SIMD changes only how fast a frame is
//! encoded, never the bitstream; the tests below compare the arms on
//! randomized and edge-case inputs.
//!
//! The inverse transforms and the loop filter are the native VP9 decoder's
//! own (`crate::vp9_dec`), and the encoder calls them directly, so they are not
//! duplicated here.
//!
//! Like the HEVC kernels, every vector function writes its intrinsics directly
//! inside a `#[target_feature]` function rather than a generic body that the
//! inliner might leave behind at the baseline instruction set (#341). An AVX2
//! arm that has no wider formulation calls the SSE4.1 function, whose features
//! are a subset of its own.

use std::sync::atomic::{AtomicU8, Ordering};

/// The instruction set the kernels in this module are running on.
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
        // The 128-bit kernels use `phaddd`, which is SSSE3; every CPU with
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

/// The cached ISA code, with any [`crate::simd::set_override`] override taking
/// precedence so it applies even after detection has resolved.
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

/// Returns the instruction set the VP9 encoder kernels will use on this machine.
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

// ---------------------------------------------------------------------------
// Forward transforms
// ---------------------------------------------------------------------------

/// The 4x4 forward transform of a residual block in raster order, as
/// `super::dsp` defines it; `vertical_adst` and `horizontal_adst` pick the ADST
/// over the DCT for each direction.
///
/// The scalar reference computes in `i64`. The vector arms compute in `i32`
/// lanes, which is exact for 8-bit residuals: the widest intermediate, the
/// row-pass ADST's `x2 - x0 + x3`, stays below 2^30 (see
/// `the_4x4_transform_matches_at_the_extremes`).
pub(super) fn forward_transform_4x4(
    residual: &[i32],
    vertical_adst: bool,
    horizontal_adst: bool,
) -> [i32; 16] {
    let residual: &[i32; 16] = residual[..16].try_into().expect("a 4x4 residual block");
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: SSE4.1 and SSSE3 were detected (AVX2 implies both).
        ISA_AVX2 | ISA_SSE41 => unsafe {
            x86::forward_transform_4x4_sse41(residual, vertical_adst, horizontal_adst)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline.
        ISA_NEON => unsafe {
            neon::forward_transform_4x4(residual, vertical_adst, horizontal_adst)
        },
        _ => super::dsp::forward_transform_4x4_scalar(residual, vertical_adst, horizontal_adst),
    }
}

/// `output = weights * rows` for `n` x `n` row-major matrices, accumulating
/// each output element over `m` in increasing order from `0.0`, with a
/// separate multiply and add (never a fused one) per term.
///
/// This is both passes of the 8x8 to 32x32 forward transforms. Fixing the
/// summation order and the rounding of each term is what makes every arm
/// bit-identical: each output lane performs exactly the scalar sequence of
/// IEEE operations.
pub(super) fn matrix_product(weights: &[f64], rows: &[f64], n: usize, output: &mut [f64]) {
    assert!(weights.len() >= n * n && rows.len() >= n * n && output.len() >= n * n);
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the bounds were checked above, and AVX2 was detected.
        ISA_AVX2 if n % 4 == 0 => unsafe { x86::matrix_product_avx2(weights, rows, n, output) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above, with SSE4.1 detected.
        ISA_AVX2 | ISA_SSE41 if n % 2 == 0 => unsafe {
            x86::matrix_product_sse41(weights, rows, n, output)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the bounds were checked above, and NEON is in the baseline.
        ISA_NEON if n % 2 == 0 => unsafe { neon::matrix_product(weights, rows, n, output) },
        _ => matrix_product_scalar(weights, rows, n, output),
    }
}

pub(super) fn matrix_product_scalar(weights: &[f64], rows: &[f64], n: usize, output: &mut [f64]) {
    output[..n * n].fill(0.0);
    for (k, out) in output.chunks_exact_mut(n).take(n).enumerate() {
        for m in 0..n {
            let weight = weights[k * n + m];
            for (out, &row) in out.iter_mut().zip(&rows[m * n..][..n]) {
                *out += weight * row;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

/// The quantizer of one transform block.
#[derive(Clone, Copy, Debug)]
pub(super) struct Quantizer {
    pub(super) dc_q: i32,
    pub(super) ac_q: i32,
    /// 32x32 levels dequantize to half the step.
    pub(super) half_step: bool,
    /// Added to the magnitude over the step before it is floored.
    pub(super) rounding: f64,
}

impl Quantizer {
    /// The effective step, the level limit and the dequantization shift for a
    /// quantizer `step`.
    fn lane(&self, step: i32) -> (f64, i32, u32) {
        if self.half_step {
            (f64::from(step) / 2.0, 65535 / step, 1)
        } else {
            (f64::from(step), 32767 / step, 0)
        }
    }
}

/// Quantizes `coefficients` into `levels` and the `dequantized` values the
/// decoder will reconstruct from them; the first coefficient is DC.
///
/// Every level is clamped so that its dequantized value stays inside the
/// 16-bit range the decoder stores coefficients in.
pub(super) fn quantize(
    coefficients: &[f64],
    quantizer: &Quantizer,
    levels: &mut [i32],
    dequantized: &mut [i32],
) {
    let count = coefficients.len();
    assert!(levels.len() >= count && dequantized.len() >= count);
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the bounds were checked above, and AVX2 was detected.
        ISA_AVX2 => unsafe { x86::quantize_avx2(coefficients, quantizer, levels, dequantized) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above, with SSE4.1 detected.
        ISA_SSE41 => unsafe { x86::quantize_sse41(coefficients, quantizer, levels, dequantized) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the bounds were checked above, and NEON is in the baseline.
        ISA_NEON => unsafe { neon::quantize(coefficients, quantizer, levels, dequantized) },
        _ => quantize_scalar(coefficients, quantizer, levels, dequantized, 0),
    }
}

/// The scalar reference, starting at coefficient `from` so the vector arms can
/// finish a block's tail with it.
pub(super) fn quantize_scalar(
    coefficients: &[f64],
    quantizer: &Quantizer,
    levels: &mut [i32],
    dequantized: &mut [i32],
    from: usize,
) {
    for index in from..coefficients.len() {
        let coefficient = coefficients[index];
        let step = if index == 0 {
            quantizer.dc_q
        } else {
            quantizer.ac_q
        };
        let (effective, limit, shift) = quantizer.lane(step);
        let level =
            ((coefficient.abs() / effective + quantizer.rounding).floor() as i32).min(limit);
        let value = (level * step) >> shift;
        levels[index] = if coefficient < 0.0 { -level } else { level };
        dequantized[index] = if coefficient < 0.0 { -value } else { value };
    }
}

// ---------------------------------------------------------------------------
// Distortion metrics
// ---------------------------------------------------------------------------

/// The sum of absolute differences between a `size` x `size` source block and
/// a reference block, returning early, after the first row that brings the
/// sum to `limit` or above, with the sum so far.
#[allow(clippy::too_many_arguments)]
pub(super) fn sad(
    source: &[u8],
    source_stride: usize,
    reference: &[u8],
    reference_stride: usize,
    size: usize,
    limit: u32,
) -> u32 {
    if size == 0 {
        return 0;
    }
    assert!(source.len() >= (size - 1) * source_stride + size);
    assert!(reference.len() >= (size - 1) * reference_stride + size);
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the bounds were checked above. The kernel is SSE2 only, the
        // x86_64 baseline, so it can be inlined here: the search often exits
        // after a row or two, where a call into a `#[target_feature]` function
        // costs more than the row. There is no 256-bit arm, because the sum
        // has to reach a scalar after every row for the early exit, and at
        // the 8- to 64-wide rows the search uses, a 256-bit load, its lane
        // fold and the `vzeroupper` on return measured slower.
        ISA_AVX2 | ISA_SSE41 => unsafe {
            x86::sad_sse2(
                source,
                source_stride,
                reference,
                reference_stride,
                size,
                limit,
            )
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the bounds were checked above, and NEON is in the baseline.
        ISA_NEON => unsafe {
            neon::sad(
                source,
                source_stride,
                reference,
                reference_stride,
                size,
                limit,
            )
        },
        _ => sad_scalar(
            source,
            source_stride,
            reference,
            reference_stride,
            size,
            limit,
        ),
    }
}

pub(super) fn sad_scalar(
    source: &[u8],
    source_stride: usize,
    reference: &[u8],
    reference_stride: usize,
    size: usize,
    limit: u32,
) -> u32 {
    let mut sad = 0_u32;
    for row in 0..size {
        sad += row_sad_scalar(
            &source[row * source_stride..][..size],
            &reference[row * reference_stride..][..size],
        );
        if sad >= limit {
            return sad;
        }
    }
    sad
}

fn row_sad_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| u32::from(a.abs_diff(b)))
        .sum()
}

/// The sum of squared differences between two `width` x `height` blocks.
pub(super) fn sse(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    if width == 0 || height == 0 {
        return 0;
    }
    assert!(a.len() >= (height - 1) * a_stride + width);
    assert!(b.len() >= (height - 1) * b_stride + width);
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the bounds were checked above, and AVX2 was detected. Below
        // 32 samples a row has no whole 256-bit step to amortize its lane fold.
        ISA_AVX2 if width >= 32 => unsafe {
            x86::sse_avx2(a, a_stride, b, b_stride, width, height)
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above, with SSE4.1 detected (AVX2 implies it).
        ISA_AVX2 | ISA_SSE41 => unsafe { x86::sse_sse41(a, a_stride, b, b_stride, width, height) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the bounds were checked above, and NEON is in the baseline.
        ISA_NEON => unsafe { neon::sse(a, a_stride, b, b_stride, width, height) },
        _ => sse_scalar(a, a_stride, b, b_stride, width, height),
    }
}

pub(super) fn sse_scalar(
    a: &[u8],
    a_stride: usize,
    b: &[u8],
    b_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    (0..height)
        .map(|row| row_sse_scalar(&a[row * a_stride..][..width], &b[row * b_stride..][..width]))
        .sum()
}

fn row_sse_scalar(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| u64::from(a.abs_diff(b)).pow(2))
        .sum()
}

/// Writes the `n` x `n` residual `source - prediction` to `output` in raster
/// order and returns its sum of squares.
pub(super) fn residual(
    source: &[u8],
    source_stride: usize,
    prediction: &[u8],
    prediction_stride: usize,
    n: usize,
    output: &mut [i32],
) -> u64 {
    if n == 0 {
        return 0;
    }
    assert!(source.len() >= (n - 1) * source_stride + n);
    assert!(prediction.len() >= (n - 1) * prediction_stride + n);
    assert!(output.len() >= n * n);
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the bounds were checked above, and SSE4.1 was detected (AVX2
        // implies it). Rows are at most 64 samples, which a 128-bit loop
        // covers in four steps, so there is no wider arm.
        ISA_AVX2 | ISA_SSE41 => unsafe {
            x86::residual_sse41(
                source,
                source_stride,
                prediction,
                prediction_stride,
                n,
                output,
            )
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the bounds were checked above, and NEON is in the baseline.
        ISA_NEON => unsafe {
            neon::residual(
                source,
                source_stride,
                prediction,
                prediction_stride,
                n,
                output,
            )
        },
        _ => residual_scalar(
            source,
            source_stride,
            prediction,
            prediction_stride,
            n,
            output,
        ),
    }
}

pub(super) fn residual_scalar(
    source: &[u8],
    source_stride: usize,
    prediction: &[u8],
    prediction_stride: usize,
    n: usize,
    output: &mut [i32],
) -> u64 {
    let mut error = 0_u64;
    for row in 0..n {
        error += residual_row_scalar(
            &source[row * source_stride..][..n],
            &prediction[row * prediction_stride..][..n],
            &mut output[row * n..][..n],
        );
    }
    error
}

fn residual_row_scalar(source: &[u8], prediction: &[u8], output: &mut [i32]) -> u64 {
    let mut error = 0_u64;
    for ((&source, &predicted), output) in source.iter().zip(prediction).zip(output) {
        let difference = i32::from(source) - i32::from(predicted);
        *output = difference;
        error += (difference * difference) as u64;
    }
    error
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

/// The TM ("true motion") intra prediction of a `size` x `size` block whose
/// top-left sample is `block[0]`: `clamp(left[row] + above[column] -
/// above_left)`.
pub(super) fn predict_tm(
    block: &mut [u8],
    stride: usize,
    size: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
) {
    if size == 0 {
        return;
    }
    assert!(above.len() >= size && left.len() >= size);
    assert!(block.len() >= (size - 1) * stride + size);
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the bounds were checked above, and SSE4.1 was detected (AVX2
        // implies it). Rows are at most 32 samples, two 128-bit stores.
        ISA_AVX2 | ISA_SSE41 => unsafe {
            x86::predict_tm_sse41(block, stride, size, above, left, above_left)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the bounds were checked above, and NEON is in the baseline.
        ISA_NEON => unsafe { neon::predict_tm(block, stride, size, above, left, above_left) },
        _ => predict_tm_scalar(block, stride, size, above, left, above_left, 0),
    }
}

/// The scalar reference, starting at column `from` so the vector arms can
/// finish a narrow block with it.
pub(super) fn predict_tm_scalar(
    block: &mut [u8],
    stride: usize,
    size: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
    from: usize,
) {
    for (row, &left) in left[..size].iter().enumerate() {
        let output = &mut block[row * stride..][..size];
        for column in from..size {
            output[column] = (i32::from(left) + i32::from(above[column]) - i32::from(above_left))
                .clamp(0, 255) as u8;
        }
    }
}

/// VP9's 8-tap separable interpolation of a `w` x `h` block from `window`, the
/// `(w + 7)` x `(h + 7)` source samples starting three rows above and three
/// columns left of the block, with row stride `window_stride`. `intermediate`
/// holds the horizontal pass, `(h + 7) * w` samples.
///
/// Like libvpx's `vpx_convolve8_c`, the horizontal pass is rounded and clipped
/// to 8 bits before the vertical pass. The taps of each filter sum to 128.
///
/// A pass with the identity kernel, a whole-sample component, copies its
/// centre sample exactly, so it is skipped: the other pass reads the window
/// directly, and only `h` rows are filtered horizontally.
#[allow(clippy::too_many_arguments)]
pub(super) fn convolve8(
    window: &[u8],
    window_stride: usize,
    w: usize,
    h: usize,
    filter_x: &[i32],
    filter_y: &[i32],
    intermediate: &mut [u8],
    output: &mut [u8],
) {
    if w == 0 || h == 0 {
        return;
    }
    assert!(window_stride >= w + 7);
    assert!(window.len() >= (h + 6) * window_stride + w + 7);
    assert!(output.len() >= w * h);
    let identity = |filter: &[i32]| filter[..8] == [0, 0, 0, 128, 0, 0, 0, 0];
    if identity(filter_y) {
        filter_rows(
            &window[3 * window_stride..],
            window_stride,
            w,
            h,
            filter_x,
            output,
        );
    } else if identity(filter_x) {
        filter_columns(&window[3..], window_stride, w, h, filter_y, output);
    } else {
        let intermediate = &mut intermediate[..(h + 7) * w];
        filter_rows(window, window_stride, w, h + 7, filter_x, intermediate);
        filter_columns(intermediate, w, w, h, filter_y, output);
    }
}

/// The horizontal 8-tap pass over `rows` rows of `w` outputs into `output`
/// (row stride `w`), reading `w + 7` samples of each `source` row.
fn filter_rows(
    source: &[u8],
    source_stride: usize,
    w: usize,
    rows: usize,
    filter: &[i32],
    output: &mut [u8],
) {
    assert!(source_stride >= w + 7);
    assert!(source.len() >= (rows - 1) * source_stride + w + 7);
    assert!(output.len() >= w * rows);
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the bounds were checked above, and AVX2 was detected.
        ISA_AVX2 if w >= 16 => unsafe {
            x86::filter_rows_avx2(source, source_stride, w, rows, filter, output)
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above, with SSE4.1 detected.
        ISA_AVX2 | ISA_SSE41 => unsafe {
            x86::filter_rows_sse41(source, source_stride, w, rows, filter, output)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the bounds were checked above, and NEON is in the baseline.
        ISA_NEON => unsafe { neon::filter_rows(source, source_stride, w, rows, filter, output) },
        _ => {
            for row in 0..rows {
                filter_row_scalar(
                    &source[row * source_stride..],
                    filter,
                    &mut output[row * w..][..w],
                    0,
                );
            }
        }
    }
}

/// The vertical 8-tap pass over `h` rows of `w` outputs into `output` (row
/// stride `w`), reading `h + 7` rows of `source`.
fn filter_columns(
    source: &[u8],
    source_stride: usize,
    w: usize,
    h: usize,
    filter: &[i32],
    output: &mut [u8],
) {
    assert!(source_stride >= w);
    assert!(source.len() >= (h + 6) * source_stride + w);
    assert!(output.len() >= w * h);
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the bounds were checked above, and AVX2 was detected.
        ISA_AVX2 if w >= 16 => unsafe {
            x86::filter_columns_avx2(source, source_stride, w, h, filter, output)
        },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above, with SSE4.1 detected.
        ISA_AVX2 | ISA_SSE41 => unsafe {
            x86::filter_columns_sse41(source, source_stride, w, h, filter, output)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the bounds were checked above, and NEON is in the baseline.
        ISA_NEON => unsafe { neon::filter_columns(source, source_stride, w, h, filter, output) },
        _ => {
            for row in 0..h {
                filter_column_scalar(
                    &source[row * source_stride..],
                    source_stride,
                    filter,
                    &mut output[row * w..][..w],
                    0,
                );
            }
        }
    }
}

/// The full two-pass reference for [`convolve8`], identity passes included.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn convolve8_scalar(
    window: &[u8],
    window_stride: usize,
    w: usize,
    h: usize,
    filter_x: &[i32],
    filter_y: &[i32],
    output: &mut [u8],
) {
    let mut intermediate = vec![0_u8; (h + 7) * w];
    for row in 0..h + 7 {
        filter_row_scalar(
            &window[row * window_stride..],
            filter_x,
            &mut intermediate[row * w..][..w],
            0,
        );
    }
    for row in 0..h {
        filter_column_scalar(
            &intermediate[row * w..],
            w,
            filter_y,
            &mut output[row * w..][..w],
            0,
        );
    }
}

/// One horizontal pass over `output.len()` samples, from column `from`.
fn filter_row_scalar(source: &[u8], filter: &[i32], output: &mut [u8], from: usize) {
    for column in from..output.len() {
        let sum: i32 = (0..8)
            .map(|tap| filter[tap] * i32::from(source[column + tap]))
            .sum();
        output[column] = ((sum + 64) >> 7).clamp(0, 255) as u8;
    }
}

/// One vertical pass over `output.len()` samples of rows `stride` apart, from
/// column `from`.
fn filter_column_scalar(
    source: &[u8],
    stride: usize,
    filter: &[i32],
    output: &mut [u8],
    from: usize,
) {
    for column in from..output.len() {
        let sum: i32 = (0..8)
            .map(|tap| filter[tap] * i32::from(source[tap * stride + column]))
            .sum();
        output[column] = ((sum + 64) >> 7).clamp(0, 255) as u8;
    }
}

// ---------------------------------------------------------------------------
// Input conversion
// ---------------------------------------------------------------------------

/// Converts `out.len()` 4-byte pixels to luma:
/// `clamp(((c[0] * p[0] + c[1] * p[1] + c[2] * p[2] + 128) >> 8) + offset)`.
///
/// `coefficients` are in the pixel's byte order, so BGRA input passes the red
/// and blue coefficients swapped. They must be non-negative, fit a byte, and
/// sum to at most 256, as every BT.601 luma row does; that keeps the
/// weighted sum inside 16 bits for the NEON arm.
pub(super) fn luma_row(pixels: &[u8], coefficients: [i32; 3], offset: i32, out: &mut [u8]) {
    assert!(pixels.len() >= out.len() * 4);
    debug_assert!(coefficients.iter().all(|c| (0..=255).contains(c)));
    debug_assert!(coefficients.iter().sum::<i32>() <= 256);
    debug_assert!((0..=255).contains(&offset));
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the bounds were checked above, and AVX2 was detected.
        ISA_AVX2 => unsafe { x86::luma_row_avx2(pixels, coefficients, offset, out) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above, with SSE4.1 and SSSE3 detected.
        ISA_SSE41 => unsafe { x86::luma_row_sse41(pixels, coefficients, offset, out) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the bounds were checked above, and NEON is in the baseline.
        ISA_NEON => unsafe { neon::luma_row(pixels, coefficients, offset, out) },
        _ => luma_row_scalar(pixels, coefficients, offset, out, 0),
    }
}

/// The scalar reference, starting at pixel `from`.
pub(super) fn luma_row_scalar(
    pixels: &[u8],
    coefficients: [i32; 3],
    offset: i32,
    out: &mut [u8],
    from: usize,
) {
    for x in from..out.len() {
        let p = &pixels[x * 4..x * 4 + 3];
        let sum = coefficients[0] * i32::from(p[0])
            + coefficients[1] * i32::from(p[1])
            + coefficients[2] * i32::from(p[2]);
        out[x] = (((sum + 128) >> 8) + offset).clamp(0, 255) as u8;
    }
}

/// Converts `cb.len()` 2x2 blocks of 4-byte pixels from two rows to Cb and Cr.
///
/// Each channel is first averaged over its block, `(sum + 2) >> 2`, and the
/// average converted as `clamp(((c[0] * p[0] + c[1] * p[1] + c[2] * p[2] + 128)
/// >> 8) + 128)`, with the coefficients in byte order as for [`luma_row`].
pub(super) fn chroma_row(
    top: &[u8],
    bottom: &[u8],
    cb_coefficients: [i32; 3],
    cr_coefficients: [i32; 3],
    cb: &mut [u8],
    cr: &mut [u8],
) {
    assert_eq!(cb.len(), cr.len());
    assert!(top.len() >= cb.len() * 8 && bottom.len() >= cb.len() * 8);
    debug_assert!(
        cb_coefficients
            .iter()
            .chain(&cr_coefficients)
            .all(|c| (-256..=256).contains(c))
    );
    match isa_code() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the bounds were checked above, and SSE4.1 and SSSE3 were
        // detected (AVX2 implies both).
        ISA_AVX2 | ISA_SSE41 => unsafe {
            x86::chroma_row_sse41(top, bottom, cb_coefficients, cr_coefficients, cb, cr)
        },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: the bounds were checked above, and NEON is in the baseline.
        ISA_NEON => unsafe {
            neon::chroma_row(top, bottom, cb_coefficients, cr_coefficients, cb, cr)
        },
        _ => chroma_row_scalar(top, bottom, cb_coefficients, cr_coefficients, cb, cr, 0),
    }
}

/// The scalar reference, starting at chroma sample `from`.
pub(super) fn chroma_row_scalar(
    top: &[u8],
    bottom: &[u8],
    cb_coefficients: [i32; 3],
    cr_coefficients: [i32; 3],
    cb: &mut [u8],
    cr: &mut [u8],
    from: usize,
) {
    for x in from..cb.len() {
        let mut average = [0_i32; 3];
        for (channel, average) in average.iter_mut().enumerate() {
            let at = x * 8 + channel;
            let sum = i32::from(top[at])
                + i32::from(top[at + 4])
                + i32::from(bottom[at])
                + i32::from(bottom[at + 4]);
            *average = (sum + 2) >> 2;
        }
        let convert = |c: [i32; 3]| {
            let sum = c[0] * average[0] + c[1] * average[1] + c[2] * average[2];
            (((sum + 128) >> 8) + 128).clamp(0, 255) as u8
        };
        cb[x] = convert(cb_coefficients);
        cr[x] = convert(cr_coefficients);
    }
}

// ---------------------------------------------------------------------------
// x86_64
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::Quantizer;
    use std::arch::x86_64::*;

    // ---- forward 4x4 ----------------------------------------------------

    const COSPI_8_64: i32 = 15137;
    const COSPI_16_64: i32 = 11585;
    const COSPI_24_64: i32 = 6270;
    const SINPI_1_9: i32 = 5283;
    const SINPI_2_9: i32 = 9929;
    const SINPI_3_9: i32 = 13377;
    const SINPI_4_9: i32 = 15212;

    /// `(value + 2^13) >> 14` in every lane.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn round_shift(value: __m128i) -> __m128i {
        _mm_srai_epi32::<14>(_mm_add_epi32(value, _mm_set1_epi32(1 << 13)))
    }

    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn mul(value: __m128i, constant: i32) -> __m128i {
        _mm_mullo_epi32(value, _mm_set1_epi32(constant))
    }

    /// Four 1-D DCTs at once: lane `i` of `x[k]` is input `k` of transform `i`.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn fdct4(x: [__m128i; 4]) -> [__m128i; 4] {
        let step0 = _mm_add_epi32(x[0], x[3]);
        let step1 = _mm_add_epi32(x[1], x[2]);
        let step2 = _mm_sub_epi32(x[1], x[2]);
        let step3 = _mm_sub_epi32(x[0], x[3]);
        [
            round_shift(mul(_mm_add_epi32(step0, step1), COSPI_16_64)),
            round_shift(_mm_add_epi32(
                mul(step2, COSPI_24_64),
                mul(step3, COSPI_8_64),
            )),
            round_shift(mul(_mm_sub_epi32(step0, step1), COSPI_16_64)),
            round_shift(_mm_sub_epi32(
                mul(step3, COSPI_24_64),
                mul(step2, COSPI_8_64),
            )),
        ]
    }

    /// Four 1-D ADSTs at once. The scalar reference returns zeros early for an
    /// all-zero input, which these formulas produce anyway.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn fadst4(x: [__m128i; 4]) -> [__m128i; 4] {
        let s0 = mul(x[0], SINPI_1_9);
        let s1 = mul(x[0], SINPI_4_9);
        let s2 = mul(x[1], SINPI_2_9);
        let s3 = mul(x[1], SINPI_1_9);
        let s4 = mul(x[2], SINPI_3_9);
        let s5 = mul(x[3], SINPI_4_9);
        let s6 = mul(x[3], SINPI_2_9);
        let s7 = _mm_sub_epi32(_mm_add_epi32(x[0], x[1]), x[3]);
        let x0 = _mm_add_epi32(_mm_add_epi32(s0, s2), s5);
        let x1 = mul(s7, SINPI_3_9);
        let x2 = _mm_add_epi32(_mm_sub_epi32(s1, s3), s6);
        let x3 = s4;
        [
            round_shift(_mm_add_epi32(x0, x3)),
            round_shift(x1),
            round_shift(_mm_sub_epi32(x2, x3)),
            round_shift(_mm_add_epi32(_mm_sub_epi32(x2, x0), x3)),
        ]
    }

    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn transpose(x: [__m128i; 4]) -> [__m128i; 4] {
        let t0 = _mm_unpacklo_epi32(x[0], x[1]);
        let t1 = _mm_unpacklo_epi32(x[2], x[3]);
        let t2 = _mm_unpackhi_epi32(x[0], x[1]);
        let t3 = _mm_unpackhi_epi32(x[2], x[3]);
        [
            _mm_unpacklo_epi64(t0, t1),
            _mm_unpackhi_epi64(t0, t1),
            _mm_unpacklo_epi64(t2, t3),
            _mm_unpackhi_epi64(t2, t3),
        ]
    }

    #[target_feature(enable = "sse4.1")]
    pub(super) fn forward_transform_4x4_sse41(
        residual: &[i32; 16],
        vertical_adst: bool,
        horizontal_adst: bool,
    ) -> [i32; 16] {
        // SAFETY: `residual` holds 16 values, four rows of four.
        let rows: [__m128i; 4] = core::array::from_fn(|row| unsafe {
            _mm_loadu_si128(residual.as_ptr().add(row * 4).cast())
        });
        // Column pass: lane `c` of row vector `r` is sample (r, c), so the
        // four row vectors are the inputs of four column transforms.
        let mut input = rows.map(|row| _mm_slli_epi32::<4>(row));
        // `input[0] += 1` for the first column when it is nonzero.
        let first = _mm_and_si128(
            _mm_andnot_si128(
                _mm_cmpeq_epi32(input[0], _mm_setzero_si128()),
                _mm_set1_epi32(1),
            ),
            _mm_setr_epi32(-1, 0, 0, 0),
        );
        input[0] = _mm_add_epi32(input[0], first);
        let columns = if vertical_adst {
            fadst4(input)
        } else {
            fdct4(input)
        };
        // Row pass: transposed, lane `r` of vector `k` is coefficient (r, k).
        let input = transpose(columns);
        let output = if horizontal_adst {
            fadst4(input)
        } else {
            fdct4(input)
        };
        let output = transpose(output);
        let mut coefficients = [0_i32; 16];
        for (row, vector) in output.into_iter().enumerate() {
            let rounded = _mm_srai_epi32::<2>(_mm_add_epi32(vector, _mm_set1_epi32(1)));
            // SAFETY: `coefficients` holds 16 values.
            unsafe { _mm_storeu_si128(coefficients.as_mut_ptr().add(row * 4).cast(), rounded) };
        }
        coefficients
    }

    // ---- matrix product -------------------------------------------------

    /// # Safety
    ///
    /// The caller checks that every matrix holds `n * n` values and that `n`
    /// is even.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn matrix_product_sse41(
        weights: &[f64],
        rows: &[f64],
        n: usize,
        output: &mut [f64],
    ) {
        let (w, r, o) = (weights.as_ptr(), rows.as_ptr(), output.as_mut_ptr());
        for k in 0..n {
            let mut column = 0;
            while column < n {
                // SAFETY: `k, m, column + 1 < n` and every matrix is `n * n`.
                unsafe {
                    let mut acc = _mm_setzero_pd();
                    for m in 0..n {
                        let weight = _mm_set1_pd(*w.add(k * n + m));
                        let row = _mm_loadu_pd(r.add(m * n + column));
                        acc = _mm_add_pd(acc, _mm_mul_pd(weight, row));
                    }
                    _mm_storeu_pd(o.add(k * n + column), acc);
                }
                column += 2;
            }
        }
    }

    /// # Safety
    ///
    /// As [`matrix_product_sse41`], with `n` a multiple of four.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn matrix_product_avx2(
        weights: &[f64],
        rows: &[f64],
        n: usize,
        output: &mut [f64],
    ) {
        let (w, r, o) = (weights.as_ptr(), rows.as_ptr(), output.as_mut_ptr());
        for k in 0..n {
            let mut column = 0;
            // Two independent accumulators per step keep both FP adders busy;
            // each lane still sums its own terms in order.
            while column + 8 <= n {
                // SAFETY: `k, m, column + 7 < n` and every matrix is `n * n`.
                unsafe {
                    let mut acc0 = _mm256_setzero_pd();
                    let mut acc1 = _mm256_setzero_pd();
                    for m in 0..n {
                        let weight = _mm256_set1_pd(*w.add(k * n + m));
                        let row = r.add(m * n + column);
                        acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(weight, _mm256_loadu_pd(row)));
                        acc1 =
                            _mm256_add_pd(acc1, _mm256_mul_pd(weight, _mm256_loadu_pd(row.add(4))));
                    }
                    _mm256_storeu_pd(o.add(k * n + column), acc0);
                    _mm256_storeu_pd(o.add(k * n + column + 4), acc1);
                }
                column += 8;
            }
            while column < n {
                // SAFETY: as above, for `column + 3 < n`.
                unsafe {
                    let mut acc = _mm256_setzero_pd();
                    for m in 0..n {
                        let weight = _mm256_set1_pd(*w.add(k * n + m));
                        acc = _mm256_add_pd(
                            acc,
                            _mm256_mul_pd(weight, _mm256_loadu_pd(r.add(m * n + column))),
                        );
                    }
                    _mm256_storeu_pd(o.add(k * n + column), acc);
                }
                column += 4;
            }
        }
    }

    // ---- quantization ---------------------------------------------------

    /// The per-lane constants of one vector of coefficients: the effective
    /// step, the level limit and the step.
    struct Lanes {
        effective: [f64; 4],
        limit: [i32; 4],
        step: [i32; 4],
    }

    impl Lanes {
        fn new(quantizer: &Quantizer, first: usize) -> Self {
            let step: [i32; 4] = core::array::from_fn(|lane| {
                if first + lane == 0 {
                    quantizer.dc_q
                } else {
                    quantizer.ac_q
                }
            });
            Self {
                effective: step.map(|step| quantizer.lane(step).0),
                limit: step.map(|step| quantizer.lane(step).1),
                step,
            }
        }
    }

    /// `value` negated in the lanes `mask` is set in.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn apply_sign(value: __m128i, mask: __m128i) -> __m128i {
        _mm_sub_epi32(_mm_xor_si128(value, mask), mask)
    }

    /// # Safety
    ///
    /// The caller checks that `levels` and `dequantized` hold at least
    /// `coefficients.len()` values.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn quantize_sse41(
        coefficients: &[f64],
        quantizer: &Quantizer,
        levels: &mut [i32],
        dequantized: &mut [i32],
    ) {
        let count = coefficients.len();
        let rounding = _mm_set1_pd(quantizer.rounding);
        let shift = _mm_cvtsi32_si128(i32::from(quantizer.half_step));
        let sign = _mm_set1_pd(-0.0);
        let ac = Lanes::new(quantizer, 1);
        let mut index = 0;
        while index + 2 <= count {
            let first;
            let lanes = if index == 0 {
                first = Lanes::new(quantizer, 0);
                &first
            } else {
                &ac
            };
            // SAFETY: `index + 1 < count` and the caller checked the outputs;
            // the lane arrays hold four values and these loads read two.
            unsafe {
                let effective = _mm_loadu_pd(lanes.effective.as_ptr());
                let limit = _mm_loadl_epi64(lanes.limit.as_ptr().cast());
                let step = _mm_loadl_epi64(lanes.step.as_ptr().cast());
                let c = _mm_loadu_pd(coefficients.as_ptr().add(index));
                let magnitude = _mm_andnot_pd(sign, c);
                let floored = _mm_floor_pd(_mm_add_pd(_mm_div_pd(magnitude, effective), rounding));
                let level = _mm_min_epi32(_mm_cvttpd_epi32(floored), limit);
                let value = _mm_sra_epi32(_mm_mullo_epi32(level, step), shift);
                // The low dword of each 64-bit compare mask, in lanes 0 and 1.
                let negative = _mm_shuffle_epi32::<0b1000>(_mm_castpd_si128(_mm_cmplt_pd(
                    c,
                    _mm_setzero_pd(),
                )));
                _mm_storel_epi64(
                    levels.as_mut_ptr().add(index).cast(),
                    apply_sign(level, negative),
                );
                _mm_storel_epi64(
                    dequantized.as_mut_ptr().add(index).cast(),
                    apply_sign(value, negative),
                );
            }
            index += 2;
        }
        super::quantize_scalar(coefficients, quantizer, levels, dequantized, index);
    }

    /// # Safety
    ///
    /// As [`quantize_sse41`].
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn quantize_avx2(
        coefficients: &[f64],
        quantizer: &Quantizer,
        levels: &mut [i32],
        dequantized: &mut [i32],
    ) {
        let count = coefficients.len();
        let rounding = _mm256_set1_pd(quantizer.rounding);
        let shift = _mm_cvtsi32_si128(i32::from(quantizer.half_step));
        let sign = _mm256_set1_pd(-0.0);
        let low_dwords = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        let ac = Lanes::new(quantizer, 1);
        let mut index = 0;
        while index + 4 <= count {
            let first;
            let lanes = if index == 0 {
                first = Lanes::new(quantizer, 0);
                &first
            } else {
                &ac
            };
            // SAFETY: `index + 3 < count` and the caller checked the outputs;
            // the lane arrays hold four values each.
            unsafe {
                let effective = _mm256_loadu_pd(lanes.effective.as_ptr());
                let limit = _mm_loadu_si128(lanes.limit.as_ptr().cast());
                let step = _mm_loadu_si128(lanes.step.as_ptr().cast());
                let c = _mm256_loadu_pd(coefficients.as_ptr().add(index));
                let negative = _mm256_cmp_pd::<_CMP_LT_OQ>(c, _mm256_setzero_pd());
                let magnitude = _mm256_andnot_pd(sign, c);
                let floored =
                    _mm256_floor_pd(_mm256_add_pd(_mm256_div_pd(magnitude, effective), rounding));
                let level = _mm_min_epi32(_mm256_cvttpd_epi32(floored), limit);
                let value = _mm_sra_epi32(_mm_mullo_epi32(level, step), shift);
                let negative = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(
                    _mm256_castpd_si256(negative),
                    low_dwords,
                ));
                _mm_storeu_si128(
                    levels.as_mut_ptr().add(index).cast(),
                    apply_sign(level, negative),
                );
                _mm_storeu_si128(
                    dequantized.as_mut_ptr().add(index).cast(),
                    apply_sign(value, negative),
                );
            }
            index += 4;
        }
        super::quantize_scalar(coefficients, quantizer, levels, dequantized, index);
    }

    // ---- distortion -----------------------------------------------------

    /// The sum of the two 64-bit lanes of a `psadbw` accumulator.
    #[inline]
    fn sum_sad(acc: __m128i) -> u32 {
        // SAFETY: SSE2 is part of the x86_64 baseline.
        unsafe { _mm_cvtsi128_si32(_mm_add_epi32(acc, _mm_unpackhi_epi64(acc, acc))) as u32 }
    }

    /// The SAD of one row of `width` samples, 16 and then 8 at a time.
    ///
    /// # Safety
    ///
    /// `a` and `b` must be valid for `width` bytes.
    #[inline]
    unsafe fn row_sad_sse2(a: *const u8, b: *const u8, width: usize) -> u32 {
        // SAFETY: SSE2 is part of the x86_64 baseline.
        let mut acc = unsafe { _mm_setzero_si128() };
        let mut column = 0;
        // SAFETY: every load stays below `width`.
        unsafe {
            while column + 16 <= width {
                let x = _mm_loadu_si128(a.add(column).cast());
                let y = _mm_loadu_si128(b.add(column).cast());
                acc = _mm_add_epi64(acc, _mm_sad_epu8(x, y));
                column += 16;
            }
            if column + 8 <= width {
                let x = _mm_loadl_epi64(a.add(column).cast());
                let y = _mm_loadl_epi64(b.add(column).cast());
                acc = _mm_add_epi64(acc, _mm_sad_epu8(x, y));
                column += 8;
            }
        }
        let mut sad = sum_sad(acc);
        while column < width {
            // SAFETY: `column < width`.
            sad += u32::from(unsafe { (*a.add(column)).abs_diff(*b.add(column)) });
            column += 1;
        }
        sad
    }

    /// The SAD kernel, in SSE2 alone, which is the x86_64 baseline.
    ///
    /// # Safety
    ///
    /// The caller checks that both blocks are `size` rows of `size` samples.
    #[inline]
    pub(super) unsafe fn sad_sse2(
        source: &[u8],
        source_stride: usize,
        reference: &[u8],
        reference_stride: usize,
        size: usize,
        limit: u32,
    ) -> u32 {
        let (a, b) = (source.as_ptr(), reference.as_ptr());
        let mut sad = 0_u32;
        // The block sizes the search uses get loops without a remainder.
        if size == 8 {
            for row in 0..size {
                // SAFETY: the caller checked the row is in bounds.
                sad += unsafe {
                    let x = _mm_loadl_epi64(a.add(row * source_stride).cast());
                    let y = _mm_loadl_epi64(b.add(row * reference_stride).cast());
                    _mm_cvtsi128_si32(_mm_sad_epu8(x, y)) as u32
                };
                if sad >= limit {
                    return sad;
                }
            }
        } else if size % 16 == 0 {
            for row in 0..size {
                let (x, y) = (row * source_stride, row * reference_stride);
                // SAFETY: SSE2 is part of the x86_64 baseline.
                let mut acc = unsafe { _mm_setzero_si128() };
                for column in (0..size).step_by(16) {
                    // SAFETY: the caller checked the row is in bounds.
                    unsafe {
                        let p = _mm_loadu_si128(a.add(x + column).cast());
                        let q = _mm_loadu_si128(b.add(y + column).cast());
                        acc = _mm_add_epi64(acc, _mm_sad_epu8(p, q));
                    }
                }
                sad += sum_sad(acc);
                if sad >= limit {
                    return sad;
                }
            }
        } else {
            for row in 0..size {
                // SAFETY: the caller checked the row is in bounds.
                sad += unsafe {
                    row_sad_sse2(
                        a.add(row * source_stride),
                        b.add(row * reference_stride),
                        size,
                    )
                };
                if sad >= limit {
                    return sad;
                }
            }
        }
        sad
    }

    /// The squared differences of 16 samples, as four `i32` partial sums.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn squares16(x: __m128i, y: __m128i) -> __m128i {
        let zero = _mm_setzero_si128();
        let lo = _mm_sub_epi16(_mm_unpacklo_epi8(x, zero), _mm_unpacklo_epi8(y, zero));
        let hi = _mm_sub_epi16(_mm_unpackhi_epi8(x, zero), _mm_unpackhi_epi8(y, zero));
        _mm_add_epi32(_mm_madd_epi16(lo, lo), _mm_madd_epi16(hi, hi))
    }

    /// The sum of the four `i32` lanes, each non-negative.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn sum_epi32(acc: __m128i) -> u64 {
        let pairs = _mm_add_epi64(
            _mm_unpacklo_epi32(acc, _mm_setzero_si128()),
            _mm_unpackhi_epi32(acc, _mm_setzero_si128()),
        );
        (_mm_cvtsi128_si64(pairs) + _mm_extract_epi64::<1>(pairs)) as u64
    }

    /// The SSE of one row from column `done`, 16 and then 8 at a time.
    ///
    /// Each `i32` lane gains at most `2 * 255^2` per 16 samples, so a row of
    /// up to 65536 samples cannot overflow it.
    ///
    /// # Safety
    ///
    /// `a` and `b` must be valid for `width` bytes.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    unsafe fn row_sse_sse41(a: *const u8, b: *const u8, width: usize, done: usize) -> u64 {
        let mut acc = _mm_setzero_si128();
        let mut column = done;
        // SAFETY: every load stays below `width`.
        unsafe {
            while column + 16 <= width {
                let x = _mm_loadu_si128(a.add(column).cast());
                let y = _mm_loadu_si128(b.add(column).cast());
                acc = _mm_add_epi32(acc, squares16(x, y));
                column += 16;
            }
            if column + 8 <= width {
                let x = _mm_loadl_epi64(a.add(column).cast());
                let y = _mm_loadl_epi64(b.add(column).cast());
                acc = _mm_add_epi32(acc, squares16(x, y));
                column += 8;
            }
        }
        let mut sse = sum_epi32(acc);
        while column < width {
            // SAFETY: `column < width`.
            let difference = u64::from(unsafe { (*a.add(column)).abs_diff(*b.add(column)) });
            sse += difference * difference;
            column += 1;
        }
        sse
    }

    /// # Safety
    ///
    /// The caller checks that both blocks are `height` rows of `width` samples.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn sse_sse41(
        a: &[u8],
        a_stride: usize,
        b: &[u8],
        b_stride: usize,
        width: usize,
        height: usize,
    ) -> u64 {
        (0..height)
            // SAFETY: the caller checked every row is in bounds.
            .map(|row| unsafe {
                row_sse_sse41(
                    a.as_ptr().add(row * a_stride),
                    b.as_ptr().add(row * b_stride),
                    width,
                    0,
                )
            })
            .sum()
    }

    /// # Safety
    ///
    /// As [`sse_sse41`].
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sse_avx2(
        a: &[u8],
        a_stride: usize,
        b: &[u8],
        b_stride: usize,
        width: usize,
        height: usize,
    ) -> u64 {
        let mut total = 0_u64;
        for row in 0..height {
            let (x, y) = (a[row * a_stride..].as_ptr(), b[row * b_stride..].as_ptr());
            let mut acc = _mm256_setzero_si256();
            let mut column = 0;
            // SAFETY: the caller checked the row is in bounds.
            unsafe {
                while column + 16 <= width {
                    let p = _mm256_cvtepu8_epi16(_mm_loadu_si128(x.add(column).cast()));
                    let q = _mm256_cvtepu8_epi16(_mm_loadu_si128(y.add(column).cast()));
                    let d = _mm256_sub_epi16(p, q);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d, d));
                    column += 16;
                }
                let folded = _mm_add_epi32(
                    _mm256_castsi256_si128(acc),
                    _mm256_extracti128_si256::<1>(acc),
                );
                total += sum_epi32(folded) + row_sse_sse41(x, y, width, column);
            }
        }
        total
    }

    /// # Safety
    ///
    /// The caller checks that both blocks are `n` rows of `n` samples and
    /// that `output` holds `n * n` values.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn residual_sse41(
        source: &[u8],
        source_stride: usize,
        prediction: &[u8],
        prediction_stride: usize,
        n: usize,
        output: &mut [i32],
    ) -> u64 {
        let mut error = 0_u64;
        for row in 0..n {
            let s = source[row * source_stride..].as_ptr();
            let p = prediction[row * prediction_stride..].as_ptr();
            let o = output[row * n..].as_mut_ptr();
            let mut acc = _mm_setzero_si128();
            let mut column = 0;
            // SAFETY: the caller checked the row is in bounds; every access
            // stays below `n`.
            unsafe {
                while column + 8 <= n {
                    let x = _mm_cvtepu8_epi16(_mm_loadl_epi64(s.add(column).cast()));
                    let y = _mm_cvtepu8_epi16(_mm_loadl_epi64(p.add(column).cast()));
                    let d = _mm_sub_epi16(x, y);
                    acc = _mm_add_epi32(acc, _mm_madd_epi16(d, d));
                    _mm_storeu_si128(o.add(column).cast(), _mm_cvtepi16_epi32(d));
                    _mm_storeu_si128(
                        o.add(column + 4).cast(),
                        _mm_cvtepi16_epi32(_mm_srli_si128::<8>(d)),
                    );
                    column += 8;
                }
                if column + 4 <= n {
                    let x = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(
                        s.add(column).cast::<i32>().read_unaligned(),
                    ));
                    let y = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(
                        p.add(column).cast::<i32>().read_unaligned(),
                    ));
                    let d = _mm_sub_epi32(x, y);
                    acc = _mm_add_epi32(acc, _mm_mullo_epi32(d, d));
                    _mm_storeu_si128(o.add(column).cast(), d);
                    column += 4;
                }
            }
            error += sum_epi32(acc);
            if column < n {
                error += super::residual_row_scalar(
                    &source[row * source_stride + column..row * source_stride + n],
                    &prediction[row * prediction_stride + column..row * prediction_stride + n],
                    &mut output[row * n + column..row * n + n],
                );
            }
        }
        error
    }

    // ---- prediction -----------------------------------------------------

    /// # Safety
    ///
    /// The caller checks that `block` holds `size` rows of `size` samples at
    /// `stride` and that `above` and `left` hold `size` samples.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn predict_tm_sse41(
        block: &mut [u8],
        stride: usize,
        size: usize,
        above: &[u8],
        left: &[u8],
        above_left: u8,
    ) {
        let corner = _mm_set1_epi16(i16::from(above_left));
        let mut column = 0;
        while column + 8 <= size {
            // SAFETY: `column + 7 < size`, inside `above` and every row.
            unsafe {
                let base = _mm_sub_epi16(
                    _mm_cvtepu8_epi16(_mm_loadl_epi64(above.as_ptr().add(column).cast())),
                    corner,
                );
                for (row, &left) in left[..size].iter().enumerate() {
                    let value = _mm_add_epi16(base, _mm_set1_epi16(i16::from(left)));
                    _mm_storel_epi64(
                        block.as_mut_ptr().add(row * stride + column).cast(),
                        _mm_packus_epi16(value, value),
                    );
                }
            }
            column += 8;
        }
        super::predict_tm_scalar(block, stride, size, above, left, above_left, column);
    }

    /// One 8-tap pass over 8 outputs: `taps[t]` weighs `samples[t]`, each the
    /// 8 inputs that output lanes 0..8 read at tap `t`, widened to `i16`.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn filter8(samples: [__m128i; 8], pairs: &[__m128i; 4]) -> __m128i {
        let mut lo = _mm_set1_epi32(64);
        let mut hi = lo;
        for (pair, &taps) in pairs.iter().enumerate() {
            let (a, b) = (samples[2 * pair], samples[2 * pair + 1]);
            lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), taps));
            hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), taps));
        }
        let packed = _mm_packs_epi32(_mm_srai_epi32::<7>(lo), _mm_srai_epi32::<7>(hi));
        _mm_packus_epi16(packed, packed)
    }

    /// The four tap pairs `(taps[2p], taps[2p + 1])`, broadcast for `pmaddwd`.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn tap_pairs(taps: &[i16; 8]) -> [__m128i; 4] {
        core::array::from_fn(|pair| {
            let packed = (i32::from(taps[2 * pair + 1]) << 16) | i32::from(taps[2 * pair] as u16);
            _mm_set1_epi32(packed)
        })
    }

    /// # Safety
    ///
    /// The caller checks that `source` holds `rows` rows of `w + 7` samples at
    /// `source_stride` and that `output` holds `w * rows` samples.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn filter_rows_sse41(
        source: &[u8],
        source_stride: usize,
        w: usize,
        rows: usize,
        filter: &[i32],
        output: &mut [u8],
    ) {
        let taps: [i16; 8] = core::array::from_fn(|tap| filter[tap] as i16);
        let pairs = tap_pairs(&taps);
        for row in 0..rows {
            let source = &source[row * source_stride..];
            let out = &mut output[row * w..][..w];
            let mut column = 0;
            while column + 8 <= w {
                // SAFETY: the 8-byte loads end at `column + 15 <= w + 7`.
                unsafe {
                    let samples: [__m128i; 8] = core::array::from_fn(|tap| {
                        _mm_cvtepu8_epi16(_mm_loadl_epi64(source.as_ptr().add(column + tap).cast()))
                    });
                    _mm_storel_epi64(
                        out.as_mut_ptr().add(column).cast(),
                        filter8(samples, &pairs),
                    );
                }
                column += 8;
            }
            super::filter_row_scalar(source, filter, out, column);
        }
    }

    /// # Safety
    ///
    /// The caller checks that `source` holds `h + 7` rows of `w` samples at
    /// `source_stride` and that `output` holds `w * h` samples.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn filter_columns_sse41(
        source: &[u8],
        source_stride: usize,
        w: usize,
        h: usize,
        filter: &[i32],
        output: &mut [u8],
    ) {
        let taps: [i16; 8] = core::array::from_fn(|tap| filter[tap] as i16);
        let pairs = tap_pairs(&taps);
        for row in 0..h {
            let source = &source[row * source_stride..];
            let out = &mut output[row * w..][..w];
            let mut column = 0;
            while column + 8 <= w {
                // SAFETY: row `row + 7` of the source exists and
                // `column + 7 < w`.
                unsafe {
                    let samples: [__m128i; 8] = core::array::from_fn(|tap| {
                        _mm_cvtepu8_epi16(_mm_loadl_epi64(
                            source.as_ptr().add(tap * source_stride + column).cast(),
                        ))
                    });
                    _mm_storel_epi64(
                        out.as_mut_ptr().add(column).cast(),
                        filter8(samples, &pairs),
                    );
                }
                column += 8;
            }
            super::filter_column_scalar(source, source_stride, filter, out, column);
        }
    }

    /// One 8-tap pass over 16 outputs, as [`filter8`] in 256-bit lanes.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn filter16(samples: [__m256i; 8], pairs: &[__m256i; 4]) -> __m128i {
        let mut lo = _mm256_set1_epi32(64);
        let mut hi = lo;
        for (pair, &taps) in pairs.iter().enumerate() {
            let (a, b) = (samples[2 * pair], samples[2 * pair + 1]);
            lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), taps));
            hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), taps));
        }
        // Per 128-bit lane, `lo` holds outputs 0..4 (8..12) and `hi` 4..8
        // (12..16), so the in-lane packs leave them in order within each lane.
        let packed = _mm256_packs_epi32(_mm256_srai_epi32::<7>(lo), _mm256_srai_epi32::<7>(hi));
        let bytes = _mm256_packus_epi16(packed, packed);
        _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(bytes))
    }

    /// The four tap pairs of `filter`, broadcast to 256-bit lanes.
    #[target_feature(enable = "avx2")]
    #[inline]
    fn wide_tap_pairs(taps: &[i16; 8]) -> [__m256i; 4] {
        core::array::from_fn(|pair| {
            let packed = (i32::from(taps[2 * pair + 1]) << 16) | i32::from(taps[2 * pair] as u16);
            _mm256_set1_epi32(packed)
        })
    }

    /// # Safety
    ///
    /// As [`filter_rows_sse41`].
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn filter_rows_avx2(
        source: &[u8],
        source_stride: usize,
        w: usize,
        rows: usize,
        filter: &[i32],
        output: &mut [u8],
    ) {
        let taps: [i16; 8] = core::array::from_fn(|tap| filter[tap] as i16);
        let (pairs, narrow) = (wide_tap_pairs(&taps), tap_pairs(&taps));
        for row in 0..rows {
            let source = &source[row * source_stride..];
            let out = &mut output[row * w..][..w];
            let mut column = 0;
            // SAFETY: the 16-byte loads end at `column + 22 < w + 7`, and the
            // 8-byte ones at `column + 14 < w + 7`.
            unsafe {
                while column + 16 <= w {
                    let samples: [__m256i; 8] = core::array::from_fn(|tap| {
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(
                            source.as_ptr().add(column + tap).cast(),
                        ))
                    });
                    _mm_storeu_si128(
                        out.as_mut_ptr().add(column).cast(),
                        filter16(samples, &pairs),
                    );
                    column += 16;
                }
                while column + 8 <= w {
                    let samples: [__m128i; 8] = core::array::from_fn(|tap| {
                        _mm_cvtepu8_epi16(_mm_loadl_epi64(source.as_ptr().add(column + tap).cast()))
                    });
                    _mm_storel_epi64(
                        out.as_mut_ptr().add(column).cast(),
                        filter8(samples, &narrow),
                    );
                    column += 8;
                }
            }
            super::filter_row_scalar(source, filter, out, column);
        }
    }

    /// # Safety
    ///
    /// As [`filter_columns_sse41`].
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn filter_columns_avx2(
        source: &[u8],
        source_stride: usize,
        w: usize,
        h: usize,
        filter: &[i32],
        output: &mut [u8],
    ) {
        let taps: [i16; 8] = core::array::from_fn(|tap| filter[tap] as i16);
        let (pairs, narrow) = (wide_tap_pairs(&taps), tap_pairs(&taps));
        for row in 0..h {
            let source = &source[row * source_stride..];
            let out = &mut output[row * w..][..w];
            let mut column = 0;
            // SAFETY: row `row + 7` of the source exists and every load ends
            // below `w` within it.
            unsafe {
                while column + 16 <= w {
                    let samples: [__m256i; 8] = core::array::from_fn(|tap| {
                        _mm256_cvtepu8_epi16(_mm_loadu_si128(
                            source.as_ptr().add(tap * source_stride + column).cast(),
                        ))
                    });
                    _mm_storeu_si128(
                        out.as_mut_ptr().add(column).cast(),
                        filter16(samples, &pairs),
                    );
                    column += 16;
                }
                while column + 8 <= w {
                    let samples: [__m128i; 8] = core::array::from_fn(|tap| {
                        _mm_cvtepu8_epi16(_mm_loadl_epi64(
                            source.as_ptr().add(tap * source_stride + column).cast(),
                        ))
                    });
                    _mm_storel_epi64(
                        out.as_mut_ptr().add(column).cast(),
                        filter8(samples, &narrow),
                    );
                    column += 8;
                }
            }
            super::filter_column_scalar(source, source_stride, filter, out, column);
        }
    }

    // ---- input conversion -----------------------------------------------

    /// The three coefficients and a zero, repeated for the two pixels of a
    /// widened 8-byte load, for `pmaddwd`.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn pixel_weights(c: [i32; 3]) -> __m128i {
        let [a, b, d] = c.map(|c| c as i16);
        _mm_setr_epi16(a, b, d, 0, a, b, d, 0)
    }

    /// The weighted sums of four pixels, each widened to four `i16` lanes,
    /// two pixels per vector.
    #[target_feature(enable = "sse4.1,ssse3")]
    #[inline]
    fn weighted4(p01: __m128i, p23: __m128i, weights: __m128i) -> __m128i {
        _mm_hadd_epi32(_mm_madd_epi16(p01, weights), _mm_madd_epi16(p23, weights))
    }

    /// `((sum + 128) >> 8) + offset` in every lane.
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn descale(sum: __m128i, offset: __m128i) -> __m128i {
        _mm_add_epi32(
            _mm_srai_epi32::<8>(_mm_add_epi32(sum, _mm_set1_epi32(128))),
            offset,
        )
    }

    /// # Safety
    ///
    /// The caller checks that `pixels` holds `out.len()` whole pixels.
    #[target_feature(enable = "sse4.1,ssse3")]
    pub(super) unsafe fn luma_row_sse41(
        pixels: &[u8],
        coefficients: [i32; 3],
        offset: i32,
        out: &mut [u8],
    ) {
        let weights = pixel_weights(coefficients);
        let offset_v = _mm_set1_epi32(offset);
        let mut x = 0;
        while x + 8 <= out.len() {
            // SAFETY: pixels `x..x + 8` are in bounds.
            unsafe {
                let a = _mm_loadu_si128(pixels.as_ptr().add(x * 4).cast());
                let b = _mm_loadu_si128(pixels.as_ptr().add(x * 4 + 16).cast());
                let sums_a = weighted4(
                    _mm_cvtepu8_epi16(a),
                    _mm_cvtepu8_epi16(_mm_srli_si128::<8>(a)),
                    weights,
                );
                let sums_b = weighted4(
                    _mm_cvtepu8_epi16(b),
                    _mm_cvtepu8_epi16(_mm_srli_si128::<8>(b)),
                    weights,
                );
                let words = _mm_packs_epi32(descale(sums_a, offset_v), descale(sums_b, offset_v));
                _mm_storel_epi64(
                    out.as_mut_ptr().add(x).cast(),
                    _mm_packus_epi16(words, words),
                );
            }
            x += 8;
        }
        super::luma_row_scalar(pixels, coefficients, offset, out, x);
    }

    /// # Safety
    ///
    /// As [`luma_row_sse41`].
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn luma_row_avx2(
        pixels: &[u8],
        coefficients: [i32; 3],
        offset: i32,
        out: &mut [u8],
    ) {
        let [a, b, d] = coefficients.map(|c| c as i16);
        let weights = _mm256_setr_epi16(a, b, d, 0, a, b, d, 0, a, b, d, 0, a, b, d, 0);
        let round = _mm256_set1_epi32(128);
        let offset_v = _mm256_set1_epi32(offset);
        let mut x = 0;
        while x + 8 <= out.len() {
            // SAFETY: pixels `x..x + 8` are in bounds.
            unsafe {
                let p = _mm256_loadu_si256(pixels.as_ptr().add(x * 4).cast());
                let lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(p));
                let hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(p));
                // Per 128-bit lane: pixels 0, 1, 4, 5 and then 2, 3, 6, 7.
                let sums = _mm256_hadd_epi32(
                    _mm256_madd_epi16(lo, weights),
                    _mm256_madd_epi16(hi, weights),
                );
                let sums = _mm256_permute4x64_epi64::<0b1101_1000>(sums);
                let values = _mm256_add_epi32(
                    _mm256_srai_epi32::<8>(_mm256_add_epi32(sums, round)),
                    offset_v,
                );
                let words = _mm_packs_epi32(
                    _mm256_castsi256_si128(values),
                    _mm256_extracti128_si256::<1>(values),
                );
                _mm_storel_epi64(
                    out.as_mut_ptr().add(x).cast(),
                    _mm_packus_epi16(words, words),
                );
            }
            x += 8;
        }
        super::luma_row_scalar(pixels, coefficients, offset, out, x);
    }

    /// # Safety
    ///
    /// The caller checks that both rows hold `2 * cb.len()` whole pixels and
    /// that `cb` and `cr` have the same length.
    #[target_feature(enable = "sse4.1,ssse3")]
    pub(super) unsafe fn chroma_row_sse41(
        top: &[u8],
        bottom: &[u8],
        cb_coefficients: [i32; 3],
        cr_coefficients: [i32; 3],
        cb: &mut [u8],
        cr: &mut [u8],
    ) {
        let cb_weights = pixel_weights(cb_coefficients);
        let cr_weights = pixel_weights(cr_coefficients);
        let offset = _mm_set1_epi32(128);
        let two = _mm_set1_epi16(2);
        let mut x = 0;
        while x + 4 <= cb.len() {
            // SAFETY: chroma samples `x..x + 4` read pixels `2x..2x + 8` of
            // each row, which are in bounds.
            unsafe {
                // The column sums of the two rows, two pixels per vector.
                let pairs: [__m128i; 4] = core::array::from_fn(|pair| {
                    let at = x * 8 + pair * 8;
                    _mm_add_epi16(
                        _mm_cvtepu8_epi16(_mm_loadl_epi64(top.as_ptr().add(at).cast())),
                        _mm_cvtepu8_epi16(_mm_loadl_epi64(bottom.as_ptr().add(at).cast())),
                    )
                });
                // Each pair's two pixels are one chroma sample's 2x2 block.
                let average = |a: __m128i, b: __m128i| {
                    let sum = _mm_add_epi16(_mm_unpacklo_epi64(a, b), _mm_unpackhi_epi64(a, b));
                    _mm_srli_epi16::<2>(_mm_add_epi16(sum, two))
                };
                let a01 = average(pairs[0], pairs[1]);
                let a23 = average(pairs[2], pairs[3]);
                let u = descale(weighted4(a01, a23, cb_weights), offset);
                let v = descale(weighted4(a01, a23, cr_weights), offset);
                let words = _mm_packs_epi32(u, v);
                let bytes = _mm_packus_epi16(words, words);
                cb.as_mut_ptr()
                    .add(x)
                    .cast::<i32>()
                    .write_unaligned(_mm_cvtsi128_si32(bytes));
                cr.as_mut_ptr()
                    .add(x)
                    .cast::<i32>()
                    .write_unaligned(_mm_extract_epi32::<1>(bytes));
            }
            x += 4;
        }
        super::chroma_row_scalar(top, bottom, cb_coefficients, cr_coefficients, cb, cr, x);
    }
}

// ---------------------------------------------------------------------------
// aarch64
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Quantizer;
    use std::arch::aarch64::*;

    const COSPI_8_64: i32 = 15137;
    const COSPI_16_64: i32 = 11585;
    const COSPI_24_64: i32 = 6270;
    const SINPI_1_9: i32 = 5283;
    const SINPI_2_9: i32 = 9929;
    const SINPI_3_9: i32 = 13377;
    const SINPI_4_9: i32 = 15212;

    #[target_feature(enable = "neon")]
    #[inline]
    fn round_shift(value: int32x4_t) -> int32x4_t {
        vshrq_n_s32::<14>(vaddq_s32(value, vdupq_n_s32(1 << 13)))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn fdct4(x: [int32x4_t; 4]) -> [int32x4_t; 4] {
        let step0 = vaddq_s32(x[0], x[3]);
        let step1 = vaddq_s32(x[1], x[2]);
        let step2 = vsubq_s32(x[1], x[2]);
        let step3 = vsubq_s32(x[0], x[3]);
        [
            round_shift(vmulq_n_s32(vaddq_s32(step0, step1), COSPI_16_64)),
            round_shift(vaddq_s32(
                vmulq_n_s32(step2, COSPI_24_64),
                vmulq_n_s32(step3, COSPI_8_64),
            )),
            round_shift(vmulq_n_s32(vsubq_s32(step0, step1), COSPI_16_64)),
            round_shift(vsubq_s32(
                vmulq_n_s32(step3, COSPI_24_64),
                vmulq_n_s32(step2, COSPI_8_64),
            )),
        ]
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn fadst4(x: [int32x4_t; 4]) -> [int32x4_t; 4] {
        let s0 = vmulq_n_s32(x[0], SINPI_1_9);
        let s1 = vmulq_n_s32(x[0], SINPI_4_9);
        let s2 = vmulq_n_s32(x[1], SINPI_2_9);
        let s3 = vmulq_n_s32(x[1], SINPI_1_9);
        let s4 = vmulq_n_s32(x[2], SINPI_3_9);
        let s5 = vmulq_n_s32(x[3], SINPI_4_9);
        let s6 = vmulq_n_s32(x[3], SINPI_2_9);
        let s7 = vsubq_s32(vaddq_s32(x[0], x[1]), x[3]);
        let x0 = vaddq_s32(vaddq_s32(s0, s2), s5);
        let x1 = vmulq_n_s32(s7, SINPI_3_9);
        let x2 = vaddq_s32(vsubq_s32(s1, s3), s6);
        let x3 = s4;
        [
            round_shift(vaddq_s32(x0, x3)),
            round_shift(x1),
            round_shift(vsubq_s32(x2, x3)),
            round_shift(vaddq_s32(vsubq_s32(x2, x0), x3)),
        ]
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn transpose(x: [int32x4_t; 4]) -> [int32x4_t; 4] {
        let t0 = vtrnq_s32(x[0], x[1]);
        let t1 = vtrnq_s32(x[2], x[3]);
        let r = |a: int32x4_t, b: int32x4_t, high: bool| {
            let (a, b) = (vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b));
            vreinterpretq_s32_s64(if high {
                vzip2q_s64(a, b)
            } else {
                vzip1q_s64(a, b)
            })
        };
        [
            r(t0.0, t1.0, false),
            r(t0.1, t1.1, false),
            r(t0.0, t1.0, true),
            r(t0.1, t1.1, true),
        ]
    }

    /// # Safety
    ///
    /// NEON must be available, which it always is on aarch64.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn forward_transform_4x4(
        residual: &[i32; 16],
        vertical_adst: bool,
        horizontal_adst: bool,
    ) -> [i32; 16] {
        // SAFETY: `residual` holds four rows of four.
        let rows: [int32x4_t; 4] =
            core::array::from_fn(|row| unsafe { vld1q_s32(residual.as_ptr().add(row * 4)) });
        let mut input = rows.map(|row| vshlq_n_s32::<4>(row));
        let nonzero = vmvnq_u32(vceqzq_s32(input[0]));
        let first_lane = vsetq_lane_u32::<0>(1, vdupq_n_u32(0));
        input[0] = vaddq_s32(
            input[0],
            vreinterpretq_s32_u32(vandq_u32(nonzero, first_lane)),
        );
        let columns = if vertical_adst {
            fadst4(input)
        } else {
            fdct4(input)
        };
        let input = transpose(columns);
        let output = if horizontal_adst {
            fadst4(input)
        } else {
            fdct4(input)
        };
        let output = transpose(output);
        let mut coefficients = [0_i32; 16];
        for (row, vector) in output.into_iter().enumerate() {
            let rounded = vshrq_n_s32::<2>(vaddq_s32(vector, vdupq_n_s32(1)));
            // SAFETY: `coefficients` holds 16 values.
            unsafe { vst1q_s32(coefficients.as_mut_ptr().add(row * 4), rounded) };
        }
        coefficients
    }

    /// # Safety
    ///
    /// The caller checks that every matrix holds `n * n` values and that `n`
    /// is even.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn matrix_product(
        weights: &[f64],
        rows: &[f64],
        n: usize,
        output: &mut [f64],
    ) {
        let (w, r, o) = (weights.as_ptr(), rows.as_ptr(), output.as_mut_ptr());
        if n % 8 == 0 {
            // Two output rows by eight columns at a time: eight independent
            // accumulators hide the add latency, and each loaded row vector
            // serves both output rows. Each lane still sums its own terms in
            // order.
            for k in (0..n).step_by(2) {
                for column in (0..n).step_by(8) {
                    // SAFETY: `k + 1, m, column + 7 < n` and every matrix is
                    // `n * n`.
                    unsafe {
                        let mut acc = [vdupq_n_f64(0.0); 8];
                        for m in 0..n {
                            let upper = vdupq_n_f64(*w.add(k * n + m));
                            let lower = vdupq_n_f64(*w.add((k + 1) * n + m));
                            let row = r.add(m * n + column);
                            for lane in 0..4 {
                                let values = vld1q_f64(row.add(lane * 2));
                                acc[lane] = vaddq_f64(acc[lane], vmulq_f64(upper, values));
                                acc[lane + 4] = vaddq_f64(acc[lane + 4], vmulq_f64(lower, values));
                            }
                        }
                        for lane in 0..4 {
                            vst1q_f64(o.add(k * n + column + lane * 2), acc[lane]);
                            vst1q_f64(o.add((k + 1) * n + column + lane * 2), acc[lane + 4]);
                        }
                    }
                }
            }
            return;
        }
        for k in 0..n {
            let mut column = 0;
            while column + 4 <= n {
                // SAFETY: `k, m, column + 3 < n` and every matrix is `n * n`.
                unsafe {
                    let mut acc0 = vdupq_n_f64(0.0);
                    let mut acc1 = vdupq_n_f64(0.0);
                    for m in 0..n {
                        let weight = vdupq_n_f64(*w.add(k * n + m));
                        let row = r.add(m * n + column);
                        // `vmulq` then `vaddq`, never the fused `vfmaq`, so
                        // each lane rounds exactly as the scalar code does.
                        acc0 = vaddq_f64(acc0, vmulq_f64(weight, vld1q_f64(row)));
                        acc1 = vaddq_f64(acc1, vmulq_f64(weight, vld1q_f64(row.add(2))));
                    }
                    vst1q_f64(o.add(k * n + column), acc0);
                    vst1q_f64(o.add(k * n + column + 2), acc1);
                }
                column += 4;
            }
            while column < n {
                // SAFETY: as above, for `column + 1 < n`.
                unsafe {
                    let mut acc = vdupq_n_f64(0.0);
                    for m in 0..n {
                        let weight = vdupq_n_f64(*w.add(k * n + m));
                        acc = vaddq_f64(acc, vmulq_f64(weight, vld1q_f64(r.add(m * n + column))));
                    }
                    vst1q_f64(o.add(k * n + column), acc);
                }
                column += 2;
            }
        }
    }

    /// # Safety
    ///
    /// The caller checks that `levels` and `dequantized` hold at least
    /// `coefficients.len()` values.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn quantize(
        coefficients: &[f64],
        quantizer: &Quantizer,
        levels: &mut [i32],
        dequantized: &mut [i32],
    ) {
        let count = coefficients.len();
        let rounding = vdupq_n_f64(quantizer.rounding);
        let shift = vdup_n_s32(-i32::from(quantizer.half_step));
        let lanes = |first: usize| {
            let step: [i32; 2] = core::array::from_fn(|lane| {
                if first + lane == 0 {
                    quantizer.dc_q
                } else {
                    quantizer.ac_q
                }
            });
            (
                step.map(|step| quantizer.lane(step).0),
                step.map(|step| quantizer.lane(step).1),
                step,
            )
        };
        let ac = lanes(1);
        let mut index = 0;
        while index + 2 <= count {
            let (effective, limit, step) = if index == 0 { lanes(0) } else { ac };
            // SAFETY: `index + 1 < count` and the caller checked the outputs.
            unsafe {
                let c = vld1q_f64(coefficients.as_ptr().add(index));
                let negative = vreinterpret_s32_u32(vmovn_u64(vcltzq_f64(c)));
                let quotient = vdivq_f64(vabsq_f64(c), vld1q_f64(effective.as_ptr()));
                let floored = vrndmq_f64(vaddq_f64(quotient, rounding));
                let level = vmin_s32(vmovn_s64(vcvtq_s64_f64(floored)), vld1_s32(limit.as_ptr()));
                let value = vshl_s32(vmul_s32(level, vld1_s32(step.as_ptr())), shift);
                let sign = |v: int32x2_t| vsub_s32(veor_s32(v, negative), negative);
                vst1_s32(levels.as_mut_ptr().add(index), sign(level));
                vst1_s32(dequantized.as_mut_ptr().add(index), sign(value));
            }
            index += 2;
        }
        super::quantize_scalar(coefficients, quantizer, levels, dequantized, index);
    }

    /// The SAD of one row of `width` samples from column `done`.
    ///
    /// # Safety
    ///
    /// `a` and `b` must be valid for `width` bytes.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn row_sad(a: *const u8, b: *const u8, width: usize) -> u32 {
        let mut acc = vdupq_n_u16(0);
        let mut column = 0;
        // SAFETY: every load stays below `width`. Each `u16` lane gains at
        // most 510 per 16 samples, so rows up to 2048 samples cannot wrap.
        unsafe {
            while column + 16 <= width {
                let d = vabdq_u8(vld1q_u8(a.add(column)), vld1q_u8(b.add(column)));
                acc = vpadalq_u8(acc, d);
                column += 16;
            }
            if column + 8 <= width {
                let d = vabd_u8(vld1_u8(a.add(column)), vld1_u8(b.add(column)));
                acc = vaddq_u16(acc, vmovl_u8(d));
                column += 8;
            }
        }
        let mut sad = vaddlvq_u16(acc);
        while column < width {
            // SAFETY: `column < width`.
            sad += u32::from(unsafe { (*a.add(column)).abs_diff(*b.add(column)) });
            column += 1;
        }
        sad
    }

    /// # Safety
    ///
    /// The caller checks that both blocks are `size` rows of `size` samples.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sad(
        source: &[u8],
        source_stride: usize,
        reference: &[u8],
        reference_stride: usize,
        size: usize,
        limit: u32,
    ) -> u32 {
        let mut sad = 0_u32;
        for row in 0..size {
            // SAFETY: the caller checked the row is in bounds.
            sad += unsafe {
                row_sad(
                    source.as_ptr().add(row * source_stride),
                    reference.as_ptr().add(row * reference_stride),
                    size,
                )
            };
            if sad >= limit {
                return sad;
            }
        }
        sad
    }

    /// # Safety
    ///
    /// `a` and `b` must be valid for `width` bytes.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn row_sse(a: *const u8, b: *const u8, width: usize) -> u64 {
        let mut acc = vdupq_n_u32(0);
        let mut column = 0;
        // SAFETY: every load stays below `width`. Each `u32` lane gains at
        // most `4 * 255^2` per 16 samples.
        unsafe {
            while column + 16 <= width {
                let d = vabdq_u8(vld1q_u8(a.add(column)), vld1q_u8(b.add(column)));
                acc = vpadalq_u16(acc, vmull_u8(vget_low_u8(d), vget_low_u8(d)));
                acc = vpadalq_u16(acc, vmull_high_u8(d, d));
                column += 16;
            }
            if column + 8 <= width {
                let d = vabd_u8(vld1_u8(a.add(column)), vld1_u8(b.add(column)));
                acc = vpadalq_u16(acc, vmull_u8(d, d));
                column += 8;
            }
        }
        let mut sse = vaddlvq_u32(acc);
        while column < width {
            // SAFETY: `column < width`.
            let difference = u64::from(unsafe { (*a.add(column)).abs_diff(*b.add(column)) });
            sse += difference * difference;
            column += 1;
        }
        sse
    }

    /// # Safety
    ///
    /// The caller checks that both blocks are `height` rows of `width` samples.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sse(
        a: &[u8],
        a_stride: usize,
        b: &[u8],
        b_stride: usize,
        width: usize,
        height: usize,
    ) -> u64 {
        (0..height)
            // SAFETY: the caller checked every row is in bounds.
            .map(|row| unsafe {
                row_sse(
                    a.as_ptr().add(row * a_stride),
                    b.as_ptr().add(row * b_stride),
                    width,
                )
            })
            .sum()
    }

    /// # Safety
    ///
    /// The caller checks that both blocks are `n` rows of `n` samples and
    /// that `output` holds `n * n` values.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn residual(
        source: &[u8],
        source_stride: usize,
        prediction: &[u8],
        prediction_stride: usize,
        n: usize,
        output: &mut [i32],
    ) -> u64 {
        let mut error = 0_u64;
        for row in 0..n {
            let s = source[row * source_stride..].as_ptr();
            let p = prediction[row * prediction_stride..].as_ptr();
            let o = output[row * n..].as_mut_ptr();
            let mut acc = vdupq_n_s32(0);
            let mut column = 0;
            // SAFETY: the caller checked the row is in bounds; every access
            // stays below `n`.
            unsafe {
                while column + 8 <= n {
                    let d = vreinterpretq_s16_u16(vsubl_u8(
                        vld1_u8(s.add(column)),
                        vld1_u8(p.add(column)),
                    ));
                    acc = vmlal_s16(acc, vget_low_s16(d), vget_low_s16(d));
                    acc = vmlal_high_s16(acc, d, d);
                    vst1q_s32(o.add(column), vmovl_s16(vget_low_s16(d)));
                    vst1q_s32(o.add(column + 4), vmovl_high_s16(d));
                    column += 8;
                }
            }
            error += vaddlvq_s32(acc) as u64;
            if column < n {
                error += super::residual_row_scalar(
                    &source[row * source_stride + column..row * source_stride + n],
                    &prediction[row * prediction_stride + column..row * prediction_stride + n],
                    &mut output[row * n + column..row * n + n],
                );
            }
        }
        error
    }

    /// # Safety
    ///
    /// The caller checks that `block` holds `size` rows of `size` samples at
    /// `stride` and that `above` and `left` hold `size` samples.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn predict_tm(
        block: &mut [u8],
        stride: usize,
        size: usize,
        above: &[u8],
        left: &[u8],
        above_left: u8,
    ) {
        let mut column = 0;
        while column + 8 <= size {
            // SAFETY: `column + 7 < size`, inside `above` and every row.
            unsafe {
                let base = vreinterpretq_s16_u16(vsubl_u8(
                    vld1_u8(above.as_ptr().add(column)),
                    vdup_n_u8(above_left),
                ));
                for (row, &left) in left[..size].iter().enumerate() {
                    let value = vaddq_s16(base, vdupq_n_s16(i16::from(left)));
                    vst1_u8(
                        block.as_mut_ptr().add(row * stride + column),
                        vqmovun_s16(value),
                    );
                }
            }
            column += 8;
        }
        super::predict_tm_scalar(block, stride, size, above, left, above_left, column);
    }

    /// One 8-tap pass over eight samples in 16-bit lanes, as libvpx's NEON
    /// convolutions compute it: the outer taps sum without overflow, and the
    /// two centre taps, which are never negative, are added saturating. A
    /// saturated sum is past `255 << 7`, where the result clamps anyway, so
    /// this is `clamp((sum + 64) >> 7)` exactly.
    #[target_feature(enable = "neon")]
    #[inline]
    fn filter8(samples: [int16x8_t; 8], taps: &[i16; 8]) -> uint8x8_t {
        let mut sum = vmulq_n_s16(samples[0], taps[0]);
        for tap in [1, 2, 5, 6, 7] {
            sum = vmlaq_n_s16(sum, samples[tap], taps[tap]);
        }
        sum = vqaddq_s16(sum, vmulq_n_s16(samples[3], taps[3]));
        sum = vqaddq_s16(sum, vmulq_n_s16(samples[4], taps[4]));
        vqrshrun_n_s16::<7>(sum)
    }

    /// [`filter8`] over sixteen samples.
    #[target_feature(enable = "neon")]
    #[inline]
    fn filter16(samples: [uint8x16_t; 8], taps: &[i16; 8]) -> uint8x16_t {
        let low = samples.map(|bytes| vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bytes))));
        let high = samples.map(|bytes| vreinterpretq_s16_u16(vmovl_high_u8(bytes)));
        vcombine_u8(filter8(low, taps), filter8(high, taps))
    }

    /// The taps of a filter for [`filter8`], which needs its two centre taps
    /// non-negative and its outer ones small enough to sum in 16 bits, as
    /// every VP9 sub-sample filter's are.
    fn narrow_taps(filter: &[i32]) -> [i16; 8] {
        debug_assert!(filter[3] >= 0 && filter[4] >= 0);
        debug_assert!(
            [0, 1, 2, 5, 6, 7]
                .map(|tap| filter[tap].abs())
                .iter()
                .sum::<i32>()
                * 255
                <= i32::from(i16::MAX)
        );
        core::array::from_fn(|tap| filter[tap] as i16)
    }

    /// # Safety
    ///
    /// The caller checks that `source` holds `rows` rows of `w + 7` samples at
    /// `source_stride` and that `output` holds `w * rows` samples.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn filter_rows(
        source: &[u8],
        source_stride: usize,
        w: usize,
        rows: usize,
        filter: &[i32],
        output: &mut [u8],
    ) {
        let taps = narrow_taps(filter);
        for row in 0..rows {
            let source = &source[row * source_stride..];
            let out = &mut output[row * w..][..w];
            let mut column = 0;
            // SAFETY: the 16-byte loads end at `column + 22 < w + 7`, and the
            // 8-byte ones at `column + 14 < w + 7`.
            unsafe {
                while column + 16 <= w {
                    let samples: [uint8x16_t; 8] =
                        core::array::from_fn(|tap| vld1q_u8(source.as_ptr().add(column + tap)));
                    vst1q_u8(out.as_mut_ptr().add(column), filter16(samples, &taps));
                    column += 16;
                }
                while column + 8 <= w {
                    let samples: [int16x8_t; 8] = core::array::from_fn(|tap| {
                        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(source.as_ptr().add(column + tap))))
                    });
                    vst1_u8(out.as_mut_ptr().add(column), filter8(samples, &taps));
                    column += 8;
                }
            }
            super::filter_row_scalar(source, filter, out, column);
        }
    }

    /// # Safety
    ///
    /// The caller checks that `source` holds `h + 7` rows of `w` samples at
    /// `source_stride` and that `output` holds `w * h` samples.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn filter_columns(
        source: &[u8],
        source_stride: usize,
        w: usize,
        h: usize,
        filter: &[i32],
        output: &mut [u8],
    ) {
        let taps = narrow_taps(filter);
        for row in 0..h {
            let source = &source[row * source_stride..];
            let out = &mut output[row * w..][..w];
            let mut column = 0;
            // SAFETY: row `row + 7` of the source exists and every load ends
            // below `w` within it.
            unsafe {
                while column + 16 <= w {
                    let samples: [uint8x16_t; 8] = core::array::from_fn(|tap| {
                        vld1q_u8(source.as_ptr().add(tap * source_stride + column))
                    });
                    vst1q_u8(out.as_mut_ptr().add(column), filter16(samples, &taps));
                    column += 16;
                }
                while column + 8 <= w {
                    let samples: [int16x8_t; 8] = core::array::from_fn(|tap| {
                        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(
                            source.as_ptr().add(tap * source_stride + column),
                        )))
                    });
                    vst1_u8(out.as_mut_ptr().add(column), filter8(samples, &taps));
                    column += 8;
                }
            }
            super::filter_column_scalar(source, source_stride, filter, out, column);
        }
    }

    /// # Safety
    ///
    /// The caller checks that `pixels` holds `out.len()` whole pixels.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn luma_row(
        pixels: &[u8],
        coefficients: [i32; 3],
        offset: i32,
        out: &mut [u8],
    ) {
        let [c0, c1, c2] = coefficients.map(|c| vdup_n_u8(c as u8));
        let offset_v = vdup_n_u8(offset as u8);
        let mut x = 0;
        while x + 16 <= out.len() {
            // SAFETY: pixels `x..x + 16` are in bounds.
            unsafe {
                let p = vld4q_u8(pixels.as_ptr().add(x * 4));
                // The coefficients sum to at most 256, so the `u16` sums
                // cannot wrap; the rounding shift saturates 256 to 255.
                let lo = vmlal_u8(
                    vmlal_u8(vmull_u8(vget_low_u8(p.0), c0), vget_low_u8(p.1), c1),
                    vget_low_u8(p.2),
                    c2,
                );
                let hi = vmlal_high_u8(
                    vmlal_high_u8(
                        vmull_high_u8(p.0, vdupq_n_u8(coefficients[0] as u8)),
                        p.1,
                        vdupq_n_u8(coefficients[1] as u8),
                    ),
                    p.2,
                    vdupq_n_u8(coefficients[2] as u8),
                );
                let lo = vqadd_u8(vqrshrn_n_u16::<8>(lo), offset_v);
                let hi = vqadd_u8(vqrshrn_n_u16::<8>(hi), offset_v);
                vst1q_u8(out.as_mut_ptr().add(x), vcombine_u8(lo, hi));
            }
            x += 16;
        }
        super::luma_row_scalar(pixels, coefficients, offset, out, x);
    }

    /// One chroma channel's weighted sum, `(sum + 128) >> 8` plus 128,
    /// saturated to a byte, for eight samples.
    #[target_feature(enable = "neon")]
    #[inline]
    fn convert8(average: [int16x8_t; 3], c: [i32; 3]) -> uint8x8_t {
        let [c0, c1, c2] = c.map(|c| c as i16);
        let lo = vmlal_n_s16(
            vmlal_n_s16(
                vmull_n_s16(vget_low_s16(average[0]), c0),
                vget_low_s16(average[1]),
                c1,
            ),
            vget_low_s16(average[2]),
            c2,
        );
        let hi = vmlal_high_n_s16(
            vmlal_high_n_s16(vmull_high_n_s16(average[0], c0), average[1], c1),
            average[2],
            c2,
        );
        let offset = vdupq_n_s32(128);
        let lo = vaddq_s32(vrshrq_n_s32::<8>(lo), offset);
        let hi = vaddq_s32(vrshrq_n_s32::<8>(hi), offset);
        vqmovn_u16(vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi)))
    }

    /// # Safety
    ///
    /// The caller checks that both rows hold `2 * cb.len()` whole pixels and
    /// that `cb` and `cr` have the same length.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn chroma_row(
        top: &[u8],
        bottom: &[u8],
        cb_coefficients: [i32; 3],
        cr_coefficients: [i32; 3],
        cb: &mut [u8],
        cr: &mut [u8],
    ) {
        let mut x = 0;
        while x + 8 <= cb.len() {
            // SAFETY: chroma samples `x..x + 8` read pixels `2x..2x + 16` of
            // each row, which are in bounds.
            unsafe {
                let t = vld4q_u8(top.as_ptr().add(x * 8));
                let b = vld4q_u8(bottom.as_ptr().add(x * 8));
                let average = |t: uint8x16_t, b: uint8x16_t| {
                    vreinterpretq_s16_u16(vrshrq_n_u16::<2>(vpadalq_u8(vpaddlq_u8(t), b)))
                };
                let average = [average(t.0, b.0), average(t.1, b.1), average(t.2, b.2)];
                vst1_u8(cb.as_mut_ptr().add(x), convert8(average, cb_coefficients));
                vst1_u8(cr.as_mut_ptr().add(x), convert8(average, cr_coefficients));
            }
            x += 8;
        }
        super::chroma_row_scalar(top, bottom, cb_coefficients, cr_coefficients, cb, cr, x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{self, SimdIsa};

    /// A small deterministic generator, so failures reproduce.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }

        fn byte(&mut self) -> u8 {
            self.next() as u8
        }

        fn bytes(&mut self, count: usize) -> Vec<u8> {
            (0..count).map(|_| self.byte()).collect()
        }
    }

    /// Runs `check` under every vector instruction set this host has, with
    /// the override pinned, and then restores detection.
    fn for_each_vector_isa(mut check: impl FnMut(SimdIsa)) {
        let _guard = simd::test_lock();
        for isa in simd::available() {
            if isa == SimdIsa::Scalar {
                continue;
            }
            simd::set_override(Some(isa));
            check(isa);
        }
        simd::set_override(None);
    }

    /// Sample patterns that reach the arithmetic's extremes as well as
    /// random values.
    fn pattern(rng: &mut Lcg, kind: usize, count: usize) -> Vec<u8> {
        match kind {
            0 => vec![0; count],
            1 => vec![255; count],
            2 => (0..count)
                .map(|i| if i % 2 == 0 { 0 } else { 255 })
                .collect(),
            3 => (0..count)
                .map(|i| if (i / 3) % 2 == 0 { 255 } else { 0 })
                .collect(),
            _ => rng.bytes(count),
        }
    }

    #[test]
    fn the_4x4_transform_matches_the_scalar_reference() {
        let mut rng = Lcg(1);
        let mut cases: Vec<[i32; 16]> = (0..2000)
            .map(|_| core::array::from_fn(|_| i32::from(rng.byte()) - i32::from(rng.byte())))
            .collect();
        // The extremes of an 8-bit residual, and their checkerboards.
        for (a, b) in [
            (255, 255),
            (-255, -255),
            (255, -255),
            (-255, 255),
            (0, 0),
            (1, 0),
        ] {
            cases.push(core::array::from_fn(|i| {
                if (i + i / 4) % 2 == 0 { a } else { b }
            }));
            cases.push(core::array::from_fn(|i| if i % 4 < 2 { a } else { b }));
            cases.push(core::array::from_fn(|i| if i < 8 { a } else { b }));
        }
        for_each_vector_isa(|isa| {
            for residual in &cases {
                for (vertical, horizontal) in
                    [(false, false), (true, false), (false, true), (true, true)]
                {
                    assert_eq!(
                        forward_transform_4x4(residual, vertical, horizontal),
                        super::super::dsp::forward_transform_4x4_scalar(
                            residual, vertical, horizontal
                        ),
                        "{} {residual:?} {vertical} {horizontal}",
                        isa.name()
                    );
                }
            }
        });
    }

    #[test]
    fn the_matrix_product_matches_the_scalar_reference() {
        let mut rng = Lcg(2);
        for_each_vector_isa(|isa| {
            for n in [2, 4, 6, 8, 12, 16, 32] {
                let random = |rng: &mut Lcg| {
                    (0..n * n)
                        .map(|_| (f64::from(rng.next()) - 2_147_483_648.0) / 1_234.567)
                        .collect::<Vec<f64>>()
                };
                let (weights, rows) = (random(&mut rng), random(&mut rng));
                let mut expected = vec![0.0; n * n];
                let mut actual = vec![1.0; n * n];
                matrix_product_scalar(&weights, &rows, n, &mut expected);
                matrix_product(&weights, &rows, n, &mut actual);
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&actual), bits(&expected), "{} n={n}", isa.name());
            }
        });
    }

    #[test]
    fn quantization_matches_the_scalar_reference() {
        let mut rng = Lcg(3);
        for_each_vector_isa(|isa| {
            for count in [16, 64, 256, 1024, 1, 3, 7] {
                for (dc_q, ac_q) in [(4, 4), (8, 9), (1336, 1828), (37, 41), (1, 1)] {
                    for half_step in [false, true] {
                        for rounding in [0.25, 0.375] {
                            let quantizer = Quantizer {
                                dc_q,
                                ac_q,
                                half_step,
                                rounding,
                            };
                            let mut coefficients: Vec<f64> = (0..count)
                                .map(|_| (f64::from(rng.next() % 80_000) - 40_000.0) / 3.0)
                                .collect();
                            // Exact multiples, signed zeros and rounding ties.
                            for (index, value) in [0.0, -0.0, 4.0, -4.0, 1.5, -1.5, 65_535.0]
                                .into_iter()
                                .enumerate()
                            {
                                if index < count {
                                    coefficients[index] = value;
                                }
                            }
                            let mut expected = (vec![0; count], vec![0; count]);
                            let mut actual = (vec![7; count], vec![7; count]);
                            quantize_scalar(
                                &coefficients,
                                &quantizer,
                                &mut expected.0,
                                &mut expected.1,
                                0,
                            );
                            quantize(&coefficients, &quantizer, &mut actual.0, &mut actual.1);
                            assert_eq!(actual, expected, "{} {quantizer:?}", isa.name());
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn distortion_metrics_match_the_scalar_reference() {
        let mut rng = Lcg(4);
        for_each_vector_isa(|isa| {
            for size in [1, 3, 4, 7, 8, 12, 16, 24, 32, 48, 64, 65] {
                for kind in 0..6 {
                    let stride = size + 5;
                    let a = pattern(&mut rng, kind, stride * size);
                    let b = pattern(&mut rng, (kind + 2) % 6, stride * size + 3);
                    for limit in [u32::MAX, 0, 1, 1000] {
                        assert_eq!(
                            sad(&a, stride, &b[3..], stride, size, limit),
                            sad_scalar(&a, stride, &b[3..], stride, size, limit),
                            "{} sad {size} {kind} {limit}",
                            isa.name()
                        );
                    }
                    for width in [size, size.saturating_sub(1).max(1)] {
                        assert_eq!(
                            sse(&a, stride, &b, stride, width, size),
                            sse_scalar(&a, stride, &b, stride, width, size),
                            "{} sse {width}x{size} {kind}",
                            isa.name()
                        );
                    }
                    let mut expected = vec![0; size * size];
                    let mut actual = vec![9; size * size];
                    assert_eq!(
                        residual(&a, stride, &b[1..], stride, size, &mut actual),
                        residual_scalar(&a, stride, &b[1..], stride, size, &mut expected),
                        "{} residual error {size} {kind}",
                        isa.name()
                    );
                    assert_eq!(actual, expected, "{} residual {size} {kind}", isa.name());
                }
            }
            // A row long enough to stress the lane accumulators.
            let (a, b) = (vec![0_u8; 4096], vec![255_u8; 4096]);
            assert_eq!(sse(&a, 0, &b, 0, 4096, 1), 4096 * 255 * 255);
        });
    }

    #[test]
    fn tm_prediction_matches_the_scalar_reference() {
        let mut rng = Lcg(5);
        for_each_vector_isa(|isa| {
            for size in [4, 8, 16, 32] {
                for kind in 0..6 {
                    let above = pattern(&mut rng, kind, size);
                    let left = pattern(&mut rng, (kind + 1) % 6, size);
                    for above_left in [0, 1, 128, 254, 255, rng.byte()] {
                        let stride = size + 3;
                        let mut expected = rng.bytes(stride * size);
                        let mut actual = expected.clone();
                        predict_tm_scalar(
                            &mut expected,
                            stride,
                            size,
                            &above,
                            &left,
                            above_left,
                            0,
                        );
                        predict_tm(&mut actual, stride, size, &above, &left, above_left);
                        assert_eq!(actual, expected, "{} {size} {kind}", isa.name());
                    }
                }
            }
        });
    }

    #[test]
    fn interpolation_matches_the_scalar_reference() {
        use super::super::tables::SUBPEL_FILTERS_REGULAR;
        let mut rng = Lcg(6);
        for_each_vector_isa(|isa| {
            for size in [4, 8, 16, 32, 64] {
                for kind in 0..6 {
                    let stride = size + 7 + 2;
                    let window = pattern(&mut rng, kind, stride * (size + 7));
                    for phase_x in 0..16 {
                        let phase_y = (phase_x * 7 + kind) % 16;
                        let filter_x = &SUBPEL_FILTERS_REGULAR[phase_x * 8..][..8];
                        let filter_y = &SUBPEL_FILTERS_REGULAR[phase_y * 8..][..8];
                        let mut expected = vec![0; size * size];
                        let mut actual = vec![1; size * size];
                        convolve8_scalar(
                            &window,
                            stride,
                            size,
                            size,
                            filter_x,
                            filter_y,
                            &mut expected,
                        );
                        convolve8(
                            &window,
                            stride,
                            size,
                            size,
                            filter_x,
                            filter_y,
                            &mut [0; 71 * 64],
                            &mut actual,
                        );
                        assert_eq!(
                            actual,
                            expected,
                            "{} {size} {kind} {phase_x}/{phase_y}",
                            isa.name()
                        );
                    }
                }
            }
        });
    }

    /// Luma coefficients, luma offset, Cb coefficients and Cr coefficients.
    type Matrix = ([i32; 3], i32, [i32; 3], [i32; 3]);

    /// The BT.601 rows `super::source_picture` converts with, in RGBA byte
    /// order: limited and full range.
    const MATRICES: [Matrix; 2] = [
        ([66, 129, 25], 16, [-38, -74, 112], [112, -94, -18]),
        ([77, 150, 29], 0, [-43, -85, 128], [128, -107, -21]),
    ];

    #[test]
    fn color_conversion_matches_the_scalar_reference() {
        let mut rng = Lcg(7);
        for_each_vector_isa(|isa| {
            for width in [1, 2, 7, 8, 9, 15, 16, 17, 31, 64, 131] {
                for kind in 0..6 {
                    let top = pattern(&mut rng, kind, width * 8);
                    let bottom = pattern(&mut rng, (kind + 3) % 6, width * 8);
                    for (luma, offset, cb_c, cr_c) in MATRICES {
                        for swap in [false, true] {
                            let order = |mut c: [i32; 3]| {
                                if swap {
                                    c.swap(0, 2);
                                }
                                c
                            };
                            let mut expected = vec![0; width * 2];
                            let mut actual = vec![1; width * 2];
                            luma_row_scalar(&top, order(luma), offset, &mut expected, 0);
                            luma_row(&top, order(luma), offset, &mut actual);
                            assert_eq!(actual, expected, "{} luma {width} {kind}", isa.name());

                            let mut expected = (vec![0; width], vec![0; width]);
                            let mut actual = (vec![1; width], vec![1; width]);
                            chroma_row_scalar(
                                &top,
                                &bottom,
                                order(cb_c),
                                order(cr_c),
                                &mut expected.0,
                                &mut expected.1,
                                0,
                            );
                            chroma_row(
                                &top,
                                &bottom,
                                order(cb_c),
                                order(cr_c),
                                &mut actual.0,
                                &mut actual.1,
                            );
                            assert_eq!(actual, expected, "{} chroma {width} {kind}", isa.name());
                        }
                    }
                }
            }
        });
    }

    /// Every corner of the RGB cube, which is where the conversions reach
    /// the ends of their ranges.
    #[test]
    fn color_conversion_matches_at_the_rgb_cube_corners() {
        let corners: Vec<u8> = (0..64)
            .flat_map(|i| {
                let corner = i % 8;
                [
                    if corner & 1 != 0 { 255 } else { 0 },
                    if corner & 2 != 0 { 255 } else { 0 },
                    if corner & 4 != 0 { 255 } else { 0 },
                    255,
                ]
            })
            .collect();
        let shifted: Vec<u8> = corners[8..].iter().chain(&corners[..8]).copied().collect();
        for_each_vector_isa(|isa| {
            for (luma, offset, cb_c, cr_c) in MATRICES {
                let mut expected = vec![0; 64];
                let mut actual = vec![1; 64];
                luma_row_scalar(&corners, luma, offset, &mut expected, 0);
                luma_row(&corners, luma, offset, &mut actual);
                assert_eq!(actual, expected, "{} luma", isa.name());
                let mut expected = (vec![0; 32], vec![0; 32]);
                let mut actual = (vec![1; 32], vec![1; 32]);
                chroma_row_scalar(
                    &corners,
                    &shifted,
                    cb_c,
                    cr_c,
                    &mut expected.0,
                    &mut expected.1,
                    0,
                );
                chroma_row(&corners, &shifted, cb_c, cr_c, &mut actual.0, &mut actual.1);
                assert_eq!(actual, expected, "{} chroma", isa.name());
            }
        });
    }

    #[test]
    fn the_kernels_are_a_dispatch_site_the_override_reaches() {
        let _guard = simd::test_lock();
        simd::set_override(Some(SimdIsa::Scalar));
        assert_eq!(isa(), Isa::Scalar);
        simd::set_override(None);
        assert_eq!(
            isa() != Isa::Scalar,
            simd::detected() != SimdIsa::Scalar,
            "detection disagrees with the crate-wide probe"
        );
    }
}
#[cfg(test)]
mod tmp_timing {
    #[test]
    #[ignore]
    fn tmp_sad_timing() {
        let a: Vec<u8> = (0..640 * 360).map(|i| (i * 7 % 251) as u8).collect();
        let b: Vec<u8> = (0..640 * 360).map(|i| (i * 13 % 241) as u8).collect();
        for isa in crate::simd::available() {
            crate::simd::set_override(Some(isa));
            for size in [8, 16, 32, 64] {
                let start = std::time::Instant::now();
                let mut total = 0u64;
                for i in 0..200_000usize {
                    let off = (i * 37) % (640 * 200);
                    total += u64::from(super::sad(
                        &a[off..],
                        640,
                        &b[(off + 3)..],
                        640,
                        size,
                        u32::MAX,
                    ));
                }
                println!(
                    "{} {size}: {:?} ({total})",
                    isa.name(),
                    start.elapsed() / 200_000
                );
            }
        }
        crate::simd::set_override(None);
    }
}
